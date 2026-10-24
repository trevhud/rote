"""A compiled skill's driver: runs its plan, and asks only when it must. Stdlib only.

rote emits this file verbatim into a compiled skill as ``scripts/_rote_skill.py``,
beside ``plan.json`` (:func:`rote.plan.build_plan`, live mode), the verbatim
``_rote_schema.py`` (:mod:`rote.plan._schema`) and ``_rote_mcp.py``
(:mod:`rote.mcp._runtime_helper`), and the compile's ``extracted/`` code.
It imports nothing from rote and nothing outside the standard library at
module level; fastmcp arrives through ``_rote_mcp`` only when a step calls
MCP, so a call that needs none starts in tens of milliseconds.

Three tiers, one direction of escalation: this program does every tool call,
pure function, retry and check it can; it escalates to the agent running the
skill for model work (``DECIDE``; ``LOOP`` once the tool proxy exists); and
it reaches a person only through that agent (``HUMAN``). Every byte printed
lands in the agent's context, so each call prints one short block:

    ROTE/1 <KIND> <skill> run=<id> [step=<id>] [...]

``DECIDE`` and ``HUMAN`` exit 0 and say what to run next. ``DONE`` exits 0,
writes the result to ``result.json`` and prints a bounded summary. ``ERROR``
exits 1 (2 for a malformed command) with the cause and the fix.

Commands (``argv`` after the script)::

    start '<input json>' | --file <path>
    answer <run> <step> '<json>' | --file <path> [--said '<words>' | --said-file <path>]
    resume <run>
    status [<run>]
    cancel <run>
    runs [--prune]

The run's journal lives in its own folder (:func:`run_root`), mode 0700:
``run.json`` (input, pending step, answers, attempts, the wave in flight),
``nodes/<id>.json`` per finished node, ``result.json`` at the end. Every
write is atomic, and every read and write holds ``run.lock`` (``flock``,
waited for at most :data:`LOCK_WAIT_SECONDS`). The lock is never held across
a tool call. A call that runs steps claims the run: it records its pid in
the journal and holds an ``flock`` on ``claim.lock`` until it prints its
block. Another call that finds that lock held says the run is busy; one
that finds it free takes the run over, because the OS drops the lock when
its holder exits, so a reused pid never reads as busy. Re-entry runs from
the journal, a fan_out step's finished elements included, so a crash costs
only what was running (at-least-once tool calls, the guarantee the durable
runtimes give).

Not here yet, on purpose: ``LOOP`` (an agent loop needs the tool proxy, the
``tool`` command that is its only way to its tools; until then ``start``
refuses a plan with a loop before anything runs, and the journal already
counts ``tool_calls`` per loop for it), and the Codex sandbox check (the
values ``CODEX_SANDBOX_NETWORK_DISABLED`` takes are unverified).

A plan emitted with ``--judges helper`` (``"judges": "helper"``) never
asks the agent to decide: each judge runs in code through the verbatim
``_rote_inference.py`` (:mod:`rote.inference._runtime_helper`), one call
per element, on the IR's model, exactly as the emitted Python judge calls
it, and its answer is checked against the output schema the same way an
agent's would be.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import os
import re
import secrets
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

PROTOCOL = "ROTE/1"
PLAN_FORMAT = "rote-plan/1"
RUN_FORMAT = "rote-skill-run/1"

#: The plan step kinds this driver runs. ``loop`` joins with the tool proxy;
#: until then ``start`` refuses a plan that has one, and so does the emitter
#: (``rote.adapters.skill``), which reads this tuple.
STEP_KINDS = ("call", "mcp", "judge", "gate")
#: The skill names a driver accepts (the agentskills.io ``name`` rule). The
#: name is a directory under the run root and a word in a shell command.
SKILL_NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
SKILL_NAME_MAX = 64

#: Answers per step before the run stops with an ERROR (still resumable).
MAX_ANSWER_ATTEMPTS = 3
#: A fanned judge's items per DECIDE block, at most; and the budgets that
#: can make a batch smaller: the items' input, and their answers' output,
#: estimated from the output schema. ``rote.eval`` sweeps these.
MAX_BATCH_ITEMS = 10
BATCH_INPUT_TOKENS = 12_000
BATCH_OUTPUT_TOKENS = 8_000
#: rote's labeled heuristic (``rote.eval`` Priors.chars_per_token).
CHARS_PER_TOKEN = 3.8
#: How long a call waits for another call's journal read or write. The lock
#: covers only those (milliseconds), never a wave, so this stays short.
LOCK_WAIT_SECONDS = 1.0
#: The python runtime's default base delay between retry attempts.
RETRY_BASE_DELAY_SECONDS = 1.0
#: DONE's summary of the result, at most this many bytes.
DONE_SUMMARY_BYTES = 400
#: Concurrent steps in one wave, as the python runtime's thread pool.
MAX_PARALLEL_STEPS = 8
#: The sweep at every ``start``: finished runs after this many days, a run
#: waiting with no deadline after the longer one, a run waiting past its
#: deadline after one day more.
SWEEP_FINISHED_DAYS = 7
SWEEP_WAITING_DAYS = 30
SWEEP_GRACE_DAYS = 1

#: Failures no retry can fix, matched by name (``_rote_mcp`` loads lazily):
#: a login is needed (the run waits for it), no endpoint or a write refused
#: (configuration), or a server asked for input mid-call. Every runtime's
#: retry opt-out excludes the same three.
NEVER_RETRIED = frozenset({"RoteMcpAuthNeeded", "RoteMcpConfigError", "RoteMcpInputRequired"})

#: ``{{ dotted.path }}``, exactly as the emitted judge's ``_interpolate``
#: matches it (``rote.plan.PLACEHOLDER``; pinned by a parity test).
_PLACEHOLDER = re.compile(r"\{\{\s*([\w.]+)\s*\}\}")
_MISSING = object()

_FINISHED = ("done", "cancelled", "expired")


# ───────── blocks ─────────


class Block:
    """One printed block: a header line, body lines, an exit status."""

    def __init__(self, kind: str, header: str, lines: list[str], status: int = 0) -> None:
        self.kind = kind
        self.header = header
        self.lines = lines
        self.status = status

    def text(self) -> str:
        return "\n".join([f"{PROTOCOL} {self.kind} {self.header}".rstrip(), *self.lines]) + "\n"


class Stop(Exception):
    """End this call with ``block``."""

    def __init__(self, block: Block) -> None:
        super().__init__(block.header)
        self.block = block


def _compact(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _progress(text: str) -> None:
    print(f"rote: {text}", file=sys.stderr, flush=True)


def _now() -> float:
    return time.time()


def _iso(t: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


# ───────── where runs live ─────────


def run_root() -> Path:
    """Where runs live: ``$ROTE_SKILL_RUN_DIR``, else ``$XDG_STATE_HOME/rote/skill-runs``,
    else ``~/.local/state/rote/skill-runs``."""
    override = os.environ.get("ROTE_SKILL_RUN_DIR")
    if override:
        return Path(override)
    xdg = os.environ.get("XDG_STATE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".local" / "state"
    return base / "rote" / "skill-runs"


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ───────── values ─────────


class BindingError(Exception):
    """An ``inputs:`` reference the upstream result cannot satisfy."""


def _keys(value: Any) -> str:
    if isinstance(value, dict):
        return "keys " + (", ".join(sorted(map(str, value))) or "none")
    return f"a {type(value).__name__}, not an object"


def _resolve(binding: list[Any], run_input: dict[str, Any], results: dict[str, Any]) -> Any:
    """One payload value, or ``_MISSING`` for an omitted optional input field."""
    param, source, node_id, fld = binding
    if source == "input":
        return run_input
    if source == "input_field":
        return run_input.get(fld, _MISSING)
    upstream = results[node_id]
    if fld is None:
        return upstream
    if not isinstance(upstream, dict) or fld not in upstream:
        raise BindingError(
            f"`{param}` is bound to {node_id}.output.{fld}, but {node_id} returned "
            f"{_keys(upstream)}"
        )
    return upstream[fld]


def _payload(
    step: dict[str, Any], run_input: dict[str, Any], results: dict[str, Any]
) -> dict[str, Any]:
    payload = {}
    for binding in step["bindings"]:
        value = _resolve(binding, run_input, results)
        if value is not _MISSING:
            payload[binding[0]] = value
    return payload


def _fan_list(value: Any, node_id: str, param: str) -> list[Any]:
    """The emitted ``_fan_out_list`` guard: the message names the node and the input."""
    if not isinstance(value, list):
        raise TypeError(
            f"fan_out node {node_id!r} expected a list from {param!r}, got "
            f"{type(value).__name__}. The upstream node must return that "
            f"key as a list."
        )
    return value


def missing_inputs_error(run_input: object, required: list[str]) -> str | None:
    """The emitted ``_missing_inputs_error``, word for word."""
    if not isinstance(run_input, dict):
        return "pipeline input must be a JSON object, got " + type(run_input).__name__
    missing = [name for name in required if name not in run_input]
    if not missing:
        return None
    return "pipeline input is missing required field(s): " + ", ".join(missing)


def _serialize(obj: Any) -> Any:
    """The emitted ``_serialize``: pydantic models and tuples to plain values."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump(by_alias=True)
    if isinstance(obj, (list, tuple)):
        return [_serialize(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    return obj


def _not_json(value: Any, path: str = "result") -> str | None:
    if value is None or isinstance(value, (bool, int, float, str)):
        return None
    if isinstance(value, list):
        for i, item in enumerate(value):
            where = _not_json(item, f"{path}[{i}]")
            if where:
                return where
        return None
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                return f"{path} has a {type(key).__name__} key {key!r}"
            where = _not_json(item, f"{path}.{key}")
            if where:
                return where
        return None
    return f"{path} is a {type(value).__name__}"


def interpolate(template: str, variables: dict[str, Any]) -> str:
    """The emitted judge's ``_interpolate``: ``{{ dotted.path }}`` against the input."""

    def _sub(match: re.Match[str]) -> str:
        path = match.group(1)
        value: Any = variables
        for part in path.split("."):
            if not isinstance(value, dict) or part not in value:
                raise KeyError(
                    f"prompt template references {{{{ {path} }}}} but the "
                    f"input has no such field; available top-level keys: "
                    f"{sorted(variables)}"
                )
            value = value[part]
        if value is None:
            return ""
        return value if isinstance(value, str) else json.dumps(value, default=str)

    return _PLACEHOLDER.sub(_sub, template)


def select(value: Any, paths: list[list[str]]) -> Any:
    """``value`` cut down to the dotted ``paths`` a prompt reads (``[]`` = all of it)."""
    if any(not p for p in paths) or not isinstance(value, dict):
        return value
    heads: dict[str, list[list[str]]] = {}
    for p in paths:
        heads.setdefault(p[0], []).append(p[1:])
    return {k: select(value[k], rest) for k, rest in heads.items() if k in value}


def _schema() -> Any:
    return importlib.import_module("_rote_schema")


# ───────── async work, on one background loop ─────────

_LOOP: Any = None


def _run_async(awaitable: Any) -> Any:
    """Run ``awaitable`` on this process's one background event loop.

    The python runtime's ``_run_async``: one loop for the process, never
    ``asyncio.run`` per call, so a module-level async client keeps working
    from its second call on (each fan_out element, each retry).
    """
    global _LOOP
    import asyncio
    import threading

    if _LOOP is None:
        loop = asyncio.new_event_loop()
        threading.Thread(target=loop.run_forever, name="rote-async", daemon=True).start()
        _LOOP = loop

    async def _await() -> Any:
        return await awaitable

    return asyncio.run_coroutine_threadsafe(_await(), _LOOP).result()


def _call_impl(fn: Any, payload: dict[str, Any]) -> Any:
    import inspect

    result = fn(**payload)
    if inspect.isawaitable(result):
        result = _run_async(result)
    return result


def unrunnable_step(plan: dict[str, Any]) -> str | None:
    """Why this driver cannot run ``plan``, naming the first step it cannot run; or None.

    ``start`` refuses on it before any folder exists, and the emitter refuses
    the same plan before writing a file, from this one sentence.
    """
    for step in plan.get("steps") or []:
        if step.get("kind") in STEP_KINDS:
            continue
        if step.get("kind") == "loop":
            # The LOOP block hands the agent a bounded task whose tools it may
            # call only through this driver (`tool <run> <step> <tool>`), so
            # the cap, the declared tools and the read-only gate hold. Without
            # that proxy a LOOP block would let the agent use any tool it has.
            return (
                f"step {step['id']} is an agent loop, and this driver does not run agent "
                "loops yet: they need its tool proxy."
            )
        return (
            f"step {step.get('id')} is a {step.get('kind')!r} step, which this driver cannot run."
        )
    return None


# ───────── the driver ─────────


class Driver:
    def __init__(self, scripts: Path, command: str | None = None, skill: str | None = None) -> None:
        """``skill`` is the compiled skill's name (its SKILL.md ``name``, which
        ``run.py`` passes): what the agent types after ``rote skill run`` and
        the run root's folder. Without one, the pipeline's name stands in."""
        self.scripts = scripts
        self.plan_text = (scripts / "plan.json").read_text(encoding="utf-8")
        import hashlib

        self.plan_hash = hashlib.sha256(self.plan_text.encode("utf-8")).hexdigest()
        self.plan: dict[str, Any] = json.loads(self.plan_text)
        self.name = skill or str(self.plan["name"])
        if skill is not None and not (len(skill) <= SKILL_NAME_MAX and SKILL_NAME.fullmatch(skill)):
            self.name = str(self.plan["name"])
            raise Stop(
                self._error(
                    None,
                    f"{skill!r} is not a skill name (lowercase letters, digits and single "
                    f"hyphens, at most {SKILL_NAME_MAX} characters).",
                    "re-emit the skill with rote emit --runtime skill",
                )
            )
        #: Who answers judges: the agent (DECIDE) or the inference helper.
        self.helper_judges = self.plan.get("judges") == "helper"
        if self.plan.get("format") != PLAN_FORMAT:
            raise Stop(
                self._error(
                    None,
                    f"this skill's plan.json is {self.plan.get('format')!r}, and this driver "
                    f"reads {PLAN_FORMAT!r}.",
                    "re-emit the skill with the rote that compiled it",
                )
            )
        self.command = command or f"rote skill run {self.name}"
        self.steps: list[dict[str, Any]] = self.plan["steps"]
        self.by_id = {s["id"]: s for s in self.steps}
        self.root = run_root() / self.name
        #: The open ``claim.lock`` while this call holds its run (see _claim_free).
        self._claim_handle: Any = None

    # ── blocks ──

    def _head(self, run: str | None, *extra: str) -> str:
        parts = [self.name] + ([f"run={run}"] if run else []) + [e for e in extra if e]
        return " ".join(parts)

    def _error(
        self,
        run: str | None,
        cause: str,
        fix: str,
        *,
        step: str | None = None,
        usage: bool = False,
    ) -> Block:
        """An ERROR block. ``fix`` is required: every error says what to do next."""
        lines = [cause, f"Fix: {fix}"]
        lines.append(
            "Fix the command and run it again."
            if usage
            else "Tell the user what this says. Do not do the skill's steps yourself."
        )
        header = self._head(run, f"step={step}" if step else "")
        return Block("ERROR", header, lines, 2 if usage else 1)

    def _next(self, *args: str) -> list[str]:
        return ["--- NEXT ---", " ".join([self.command, *args])]

    # ── the journal ──

    def _dir(self, run: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{12}", run):
            raise Stop(
                self._error(
                    None,
                    f"{run!r} is not a run id.",
                    f"use the run=<id> from a block, or list them: {self.command} runs",
                    usage=True,
                )
            )
        path = self.root / run
        if not (path / "run.json").is_file():
            raise Stop(self._error(None, f"no run {run} of {self.name}.", f"{self.command} runs"))
        return path

    @contextlib.contextmanager
    def _locked(self, path: Path) -> Iterator[None]:
        import fcntl

        with open(path / "run.lock", "a+") as handle:
            deadline = _now() + LOCK_WAIT_SECONDS
            while True:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if _now() >= deadline:
                        raise Stop(
                            self._error(
                                path.name,
                                "run busy: another call holds its journal.",
                                f"{self.command} status {path.name}",
                            )
                        ) from None
                    time.sleep(0.02)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _read(self, path: Path) -> dict[str, Any]:
        state: dict[str, Any] = json.loads((path / "run.json").read_text(encoding="utf-8"))
        if state.get("format") != RUN_FORMAT:
            raise Stop(
                self._error(
                    path.name,
                    f"run.json is not a {RUN_FORMAT} journal, so this run cannot go on.",
                    f"start a new run: {self.command} start '<input json>'",
                )
            )
        if state.get("plan_hash") != self.plan_hash:
            raise Stop(
                self._error(
                    path.name,
                    "this skill was re-emitted after the run started, and two programs' "
                    "results must never mix.",
                    f"start a new run: {self.command} start ...",
                )
            )
        return state

    def _write(self, path: Path, state: dict[str, Any]) -> None:
        state["updated"] = _now()
        _atomic_write(path / "run.json", json.dumps(state, ensure_ascii=False, indent=1))

    def _write_node(self, path: Path, node_id: str, record: dict[str, Any]) -> None:
        nodes = path / "nodes"
        nodes.mkdir(mode=0o700, exist_ok=True)
        _atomic_write(nodes / f"{node_id}.json", json.dumps(record, ensure_ascii=False))

    def _results(self, path: Path, state: dict[str, Any]) -> dict[str, Any]:
        """Every finished node's result, refusing a decision nobody recorded.

        A gate's result counts only when its node file and ``run.json``'s
        answer log both say a person answered it, with the same words, so a
        single hand-written file is refused. That is a consistency check,
        not a barrier: a process running as the same user can write both
        files, and nothing in a journal can stop it. The journal is an audit
        trail of who answered what (``answered_by``, ``said``, ``at``).
        """
        results: dict[str, Any] = {}
        nodes = path / "nodes"
        for step in self.steps:
            file = nodes / f"{step['id']}.json"
            if not file.is_file():
                continue
            record = json.loads(file.read_text(encoding="utf-8"))
            if step["kind"] == "gate":
                logged = (state.get("answers") or {}).get(step["id"]) or {}
                said = record.get("said")
                if not (
                    record.get("answered_by") == "human"
                    and isinstance(said, str)
                    and said.strip()
                    and logged.get("answered_by") == "human"
                    and logged.get("said") == said
                ):
                    raise Stop(
                        self._error(
                            path.name,
                            f"nodes/{step['id']}.json holds an answer to a step only a person "
                            "may answer, and no person's words are recorded for it. The run "
                            "cannot pass this gate on it.",
                            f"delete that file, then: {self.command} resume {path.name}",
                            step=step["id"],
                        )
                    )
            elif step["kind"] in ("judge", "loop") and record.get("answered_by") != (
                "helper" if step["kind"] == "judge" and self.helper_judges else "agent"
            ):
                raise Stop(
                    self._error(
                        path.name,
                        f"nodes/{step['id']}.json was not written by this driver.",
                        f"delete that file, then: {self.command} resume {path.name}",
                        step=step["id"],
                    )
                )
            results[step["id"]] = record["result"]
        return results

    def _claim_free(self, path: Path, state: dict[str, Any]) -> None:
        """Refuse while another call holds the run; take over from one that died.

        Call under the journal lock. Whether the holder lives is not read
        from its pid, which the OS reuses: a holder keeps an ``flock`` on
        ``claim.lock`` for as long as it runs steps, and the OS drops that
        lock when the process exits, however it exits. So a free lock is a
        dead (or finished) holder, whatever process now has its pid. The pid
        in the journal is only for the message.
        """
        if self._claim_handle is not None:
            return  # this call holds it
        import fcntl

        wave = state.get("wave")
        with open(path / "claim.lock", "a+") as probe:
            try:
                fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pid = (wave or {}).get("pid", "unknown")
                raise Stop(
                    self._error(
                        path.name,
                        f"run busy: another call (pid {pid}) is running its steps.",
                        f"wait, then: {self.command} status {path.name}",
                    )
                ) from None
            fcntl.flock(probe.fileno(), fcntl.LOCK_UN)
        if wave:
            _progress(f"taking over from pid {wave.get('pid')}, which no longer holds the run")
            state["wave"] = None

    # ── commands ──

    def start(self, raw: str) -> Block:
        run_input = self._parse(None, raw, "the input")
        problem = missing_inputs_error(run_input, self.plan["required_inputs"])
        if problem:
            return self._error(None, problem + ".", "pass every required field", usage=True)
        unrunnable = unrunnable_step(self.plan)
        if unrunnable:
            return self._error(
                None,
                unrunnable,
                "compile this skill for a workflow runtime (rote compile --runtime dbos)",
            )
        self.sweep()
        self.root.mkdir(parents=True, mode=0o700, exist_ok=True)
        os.chmod(self.root, 0o700)
        run = secrets.token_hex(6)
        path = self.root / run
        path.mkdir(mode=0o700)
        state = {
            "format": RUN_FORMAT,
            "skill": self.name,
            "run": run,
            "plan_hash": self.plan_hash,
            "input": run_input,
            "state": "running",
            "created": _now(),
            "pending": None,
            "answers": {},
            "attempts": {},
            # Tool calls each agent loop has made, so its max_iterations cap
            # survives a resume (the LOOP proxy counts into it).
            "tool_calls": {},
            "wave": None,
            "error": None,
        }
        with self._locked(path):
            self._write(path, state)
        return self.advance(path)

    def resume(self, run: str) -> Block:
        path = self._dir(run)
        with self._locked(path):
            state = self._read(path)
            if state["state"] == "error":
                state["state"] = "waiting" if state.get("pending") else "running"
                state["error"] = None
                if state.get("pending"):
                    state["attempts"][state["pending"]["step"]] = 0
                self._write(path, state)
            pending = state.get("pending")
            if pending and pending["kind"] == "login":
                state["pending"] = None
                state["state"] = "running"
                self._write(path, state)
        return self.advance(path)

    def status(self, run: str | None) -> Block:
        if run is None:
            runs = sorted(self._runs(), key=lambda r: r[1].get("updated", 0), reverse=True)
            live = [r for r in runs if r[1].get("state") not in _FINISHED]
            if not live:
                return self._error(
                    None,
                    f"no run of {self.name} is waiting.",
                    f"start one: {self.command} start '<input json>'",
                )
            run = live[0][0].name
        path = self._dir(run)
        with self._locked(path):
            state = self._read(path)
            return self._current(path, state)

    def cancel(self, run: str) -> Block:
        path = self._dir(run)
        with self._locked(path):
            state = self._read(path)
            self._claim_free(path, state)
            if state["state"] not in _FINISHED:
                state["state"] = "cancelled"
                state["pending"] = None
                self._write(path, state)
        return Block("DONE", self._head(run, "cancelled"), ["The run is cancelled."])

    def runs(self, prune: bool) -> Block:
        removed = self.sweep() if prune else []
        lines = [
            f"{p.name} {s.get('state')} {_iso(float(s.get('updated', 0)))} "
            f"{(s.get('pending') or {}).get('step', '')}".rstrip()
            for p, s in sorted(self._runs(), key=lambda r: r[1].get("updated", 0))
        ]
        if prune:
            lines.append(f"pruned {len(removed)}")
        return Block("DONE", self._head(None, f"runs={len(lines) - int(prune)}"), lines)

    def answer(self, run: str, step_id: str, raw: str, said: str | None) -> Block:
        path = self._dir(run)
        with self._locked(path):
            state = self._read(path)
            self._claim_free(path, state)
            pending = state.get("pending")
            if state["state"] in _FINISHED or not pending or pending["step"] != step_id:
                answered = step_id in (state.get("answers") or {})
                note = (
                    f"Note: {step_id} is already answered; nothing changed."
                    if answered
                    else f"Note: {step_id} is not the pending step; nothing changed."
                )
                return self._current(path, state, note)
            step = self.by_id[step_id]
            if pending["kind"] == "login":
                return self._error(
                    run,
                    f"{step_id} waits for a login, not an answer.",
                    f"{self.command} resume {run} after the person logs in",
                    step=step_id,
                    usage=True,
                )
            if step["kind"] == "gate" and not (said and said.strip()):
                return self._error(
                    run,
                    f"{step_id} is answered by a person: pass their words as they gave them.",
                    f"{self.command} answer {run} {step_id} '<json>' --said '<their words>'",
                    step=step_id,
                    usage=True,
                )
            if step["kind"] != "gate" and said is not None:
                return self._error(
                    run,
                    f"{step_id} is yours to answer; --said is only for a person's words "
                    "at a HUMAN step.",
                    f"{self.command} answer {run} {step_id} '<json>'",
                    step=step_id,
                    usage=True,
                )
            if step["kind"] == "gate" and _now() > float(pending["deadline"]):
                state["state"] = "expired"
                state["pending"] = None
                self._write(path, state)
                return self._error(
                    run,
                    f"{step_id} waited for a person until {_iso(float(pending['deadline']))}, "
                    "and that deadline has passed. The run has ended.",
                    f"start a new run: {self.command} start ...",
                    step=step_id,
                )
            attempts = int(state["attempts"].get(step_id, 0)) + 1
            state["attempts"][step_id] = attempts
            try:
                value = json.loads(raw)
                errors: list[str] = []
            except json.JSONDecodeError as e:
                value = None
                errors = [
                    f"not JSON (line {e.lineno} column {e.colno}: {e.msg}). Shell quoting is "
                    "the usual cause: write the JSON to a file and pass --file <path>"
                ]
            if step["kind"] == "gate":
                block = self._take_gate(path, state, step, value, errors, said or "")
            else:
                block = self._take_decision(path, state, step, value, errors)
            if block is not None:
                return block
            # Claimed under the same lock that recorded the answer, so a second
            # call never finds the run between this answer and the next block.
            self._claim(path, state)
        return self.advance(path)

    # ── answers ──

    def _too_many(self, path: Path, state: dict[str, Any], step: str, errors: list[str]) -> Block:
        state["state"] = "error"
        state["error"] = "; ".join(errors)
        self._write(path, state)
        return self._error(
            path.name,
            f"{step}: {MAX_ANSWER_ATTEMPTS} answers did not validate. Last: "
            + "; ".join(errors[:5]),
            f"{self.command} resume {path.name} (it asks again)",
            step=step,
        )

    def _take_gate(
        self,
        path: Path,
        state: dict[str, Any],
        step: dict[str, Any],
        value: Any,
        errors: list[str],
        said: str,
    ) -> Block | None:
        schema = _schema()
        if not errors:
            errors = schema.schema_errors(step["answer_schema"], value)
        attempts = state["attempts"][step["id"]]
        if errors:
            if attempts >= MAX_ANSWER_ATTEMPTS:
                return self._too_many(path, state, step["id"], errors)
            self._write(path, state)
            lines = [
                "Do not answer this yourself. Their reply did not fit; ask the person again "
                "for exactly this, in plain words:",
                *[f"- {e}" for e in errors],
                *self._next("answer", path.name, step["id"], "'<JSON>'", "--said", "'<words>'"),
            ]
            header = self._head(
                path.name, f"step={step['id']}", "kind=gate", f"attempt={attempts + 1}"
            )
            return Block("HUMAN", header, lines)
        at = _now()
        record = {
            "result": schema.project(step["answer_schema"], value),
            "answered_by": "human",
            "said": said,
            "at": at,
        }
        self._write_node(path, step["id"], record)
        state["answers"][step["id"]] = {"answered_by": "human", "said": said, "at": at}
        state["pending"] = None
        state["state"] = "running"
        self._write(path, state)
        return None

    def _take_decision(
        self, path: Path, state: dict[str, Any], step: dict[str, Any], value: Any, errors: list[str]
    ) -> Block | None:
        schema = _schema()
        pending = state["pending"]
        output = step["output_schema"]
        attempts = state["attempts"][step["id"]]
        if step["fan_param"] is None:
            if not errors:
                errors = schema.schema_errors(output, value)
            if errors:
                if attempts >= MAX_ANSWER_ATTEMPTS:
                    return self._too_many(path, state, step["id"], errors)
                self._write(path, state)
                return self._reask(path.name, step["id"], attempts + 1, errors, [], pending)
            self._record_decision(path, state, step, schema.project(output, value))
            return None
        wanted = list(pending["batches"][pending["batch"]])
        collected: dict[str, Any] = pending["collected"]
        if not errors and not isinstance(value, dict):
            errors = [f"answer: expected an object keyed by item id, got {schema.json_type(value)}"]
        if not errors:
            for item_id, item in value.items():
                if item_id not in wanted:
                    errors.append(f"answer: {item_id!r} is not an item of this batch")
                    continue
                found = schema.schema_errors(output, item, f"answer.{item_id}")
                if found:
                    errors.extend(found)
                else:
                    collected[item_id] = schema.project(output, item)
        missing = [i for i in wanted if i not in collected]
        if errors or missing:
            if attempts >= MAX_ANSWER_ATTEMPTS:
                return self._too_many(
                    path, state, step["id"], errors or [f"missing items: {', '.join(missing)}"]
                )
            self._write(path, state)
            return self._reask(path.name, step["id"], attempts + 1, errors, missing, pending)
        if pending["batch"] + 1 < len(pending["batches"]):
            pending["batch"] += 1
            state["attempts"][step["id"]] = 0
            self._write(path, state)
            return self._decide_block(path.name, step, pending)
        ordered = [collected[str(i)] for i in range(pending["count"])]
        self._record_decision(path, state, step, ordered)
        return None

    def _record_decision(
        self, path: Path, state: dict[str, Any], step: dict[str, Any], result: Any
    ) -> None:
        at = _now()
        self._write_node(path, step["id"], {"result": result, "answered_by": "agent", "at": at})
        state["answers"][step["id"]] = {"answered_by": "agent", "at": at}
        state["pending"] = None
        state["state"] = "running"
        self._write(path, state)

    def _reask(
        self,
        run: str,
        step: str,
        attempt: int,
        errors: list[str],
        missing: list[str],
        pending: dict[str, Any],
    ) -> Block:
        extra = []
        if pending.get("batches") and len(pending["batches"]) > 1:
            extra.append(f"batch={pending['batch'] + 1}/{len(pending['batches'])}")
        lines = ["Your answer did not validate. Answer again with only these fixed:"]
        lines += [f"- {e}" for e in errors]
        if missing:
            lines.append("missing items: " + ", ".join(missing))
        lines += self._next("answer", run, step, "'<your JSON>'")
        return Block("DECIDE", self._head(run, f"step={step}", *extra, f"attempt={attempt}"), lines)

    # ── running ──

    def advance(self, path: Path) -> Block:
        """Run forward from the journal to the next decision, or to the end.

        The call claims the run in the journal (``wave``: its pid) for as
        long as it runs steps, and releases it when it prints its block. The
        journal lock is taken only for each read and write in between.
        """
        with self._locked(path):
            state = self._read(path)
            if state["state"] in (*_FINISHED, "error") or state.get("pending"):
                self._release(path, state)
                return self._current(path, state)
            self._claim(path, state)
            results = self._results(path, state)
        try:
            return self._run_forward(path, results)
        finally:
            with self._locked(path):
                self._release(path, self._read(path))

    def _claim(self, path: Path, state: dict[str, Any], wave: int | None = None) -> None:
        """Hold the run for this call (call under the journal lock)."""
        import fcntl

        self._claim_free(path, state)
        if self._claim_handle is None:
            handle = open(path / "claim.lock", "a+")  # noqa: SIM115 (held until release)
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._claim_handle = handle
        state["wave"] = {"index": wave, "pid": os.getpid(), "since": _now()}
        self._write(path, state)

    def _release(self, path: Path, state: dict[str, Any]) -> None:
        if self._claim_handle is None:
            return
        if state.get("wave"):
            state["wave"] = None
            self._write(path, state)
        self.close()

    def close(self) -> None:
        """Drop this call's claim on its run, if it holds one."""
        if self._claim_handle is not None:
            self._claim_handle.close()  # closing the file releases its flock
            self._claim_handle = None

    def _run_forward(self, path: Path, results: dict[str, Any]) -> Block:
        in_code = ("call", "mcp", "judge") if self.helper_judges else ("call", "mcp")
        for wave in sorted({int(s["wave"]) for s in self.steps}):
            in_wave = [s for s in self.steps if int(s["wave"]) == wave]
            runnable = [s for s in in_wave if s["kind"] in in_code and s["id"] not in results]
            if runnable:
                with self._locked(path):
                    state = self._read(path)
                    self._claim(path, state, wave)
                failure = self._run_wave(path, state, runnable, results)
                if failure is not None:
                    with self._locked(path):
                        state = self._read(path)
                        block = self._failure_block(path, state, *failure)
                        self._write(path, state)
                        return block
            for step in in_wave:
                if step["kind"] in ("judge", "gate") and step["id"] not in results:
                    with self._locked(path):
                        return self._ask(path, self._read(path), step, results)
        return self._finish(path, results)

    def _run_wave(
        self,
        path: Path,
        state: dict[str, Any],
        steps: list[dict[str, Any]],
        results: dict[str, Any],
    ) -> tuple[dict[str, Any], BaseException] | None:
        """Run a wave's code and tool steps; the first failure, in step order, or None."""
        outcomes: dict[str, Any] = {}

        def one(step: dict[str, Any]) -> None:
            try:
                value = self._run_step(path, step, state["input"], results)
            except BaseException as exc:  # noqa: BLE001 (reported as the step's failure)
                outcomes[step["id"]] = exc
                return
            record: dict[str, Any] = {"result": value, "at": _now()}
            if step["kind"] == "judge":
                record["answered_by"] = "helper"
            with self._locked(path):
                self._write_node(path, step["id"], record)
                parts = path / "nodes" / f"{step['id']}.parts"
                if parts.is_dir() and not parts.is_symlink():
                    import shutil

                    shutil.rmtree(parts)
            outcomes[step["id"]] = value
            results[step["id"]] = value

        if len(steps) == 1:
            one(steps[0])
        else:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_STEPS, len(steps))) as pool:
                list(pool.map(one, steps))
        for step in steps:
            outcome = outcomes.get(step["id"])
            if isinstance(outcome, BaseException):
                return step, outcome
        return None

    def _run_step(
        self, path: Path, step: dict[str, Any], run_input: dict[str, Any], results: dict[str, Any]
    ) -> Any:
        """One step's result. A fan_out step runs each element on its own.

        Each element has its own retry budget, as in the python runtime,
        and is journaled as it finishes (``nodes/<id>.parts/<i>.json``), so a
        failing element never re-runs one that finished, now or on resume.
        """
        payload = _payload(step, run_input, results)
        if step["kind"] == "mcp":
            _progress(f"calling {step['server']}.{step['tool']}")
        elif step["kind"] == "judge":
            _progress(f"asking {step['lane']['model']} to judge {step['id']}")
        else:
            _progress(f"running {step['id']}")
        fan = step.get("fan_param")
        if fan is None:
            return self._checked(step["id"], self._attempts(step, step["id"], payload))
        elements = _fan_list(payload.get(fan), step["id"], fan)
        parts = path / "nodes" / f"{step['id']}.parts"
        done = {
            int(p.stem): json.loads(p.read_text(encoding="utf-8"))["result"]
            for p in (parts.glob("*.json") if parts.is_dir() else ())
        }
        failures: dict[int, BaseException] = {}

        def element(index: int) -> None:
            label = f"{step['id']}[{index}]"
            try:
                value = self._checked(
                    label, self._attempts(step, label, {**payload, fan: elements[index]})
                )
            except BaseException as exc:  # noqa: BLE001 (reported as the step's failure)
                failures[index] = exc
                return
            with self._locked(path):
                parts.mkdir(mode=0o700, parents=True, exist_ok=True)
                _atomic_write(parts / f"{index}.json", json.dumps({"result": value}))
            done[index] = value

        todo = [i for i in range(len(elements)) if i not in done]
        if step["kind"] == "mcp" and len(todo) > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_STEPS, len(todo))) as pool:
                list(pool.map(element, todo))
        else:
            for index in todo:
                element(index)
                if failures:
                    break
        if failures:
            raise failures[min(failures)]
        return [done[i] for i in range(len(elements))]

    def _attempts(self, step: dict[str, Any], label: str, payload: dict[str, Any]) -> Any:
        """One call, retried per the IR, never for the failures no retry fixes."""
        policy = step.get("retry") or {"attempts": 1, "backoff": "constant"}
        attempts = int(policy["attempts"])
        for attempt in range(1, attempts + 1):
            try:
                return self._once(step, payload)
            except Exception as exc:
                if type(exc).__name__ in NEVER_RETRIED or attempt == attempts:
                    raise
                delay = {
                    "exponential": RETRY_BASE_DELAY_SECONDS * 2 ** (attempt - 1),
                    "linear": RETRY_BASE_DELAY_SECONDS * attempt,
                }.get(policy["backoff"], RETRY_BASE_DELAY_SECONDS)
                _progress(f"{label} failed ({type(exc).__name__}); retry {attempt} in {delay:g}s")
                time.sleep(delay)
        raise AssertionError("unreachable: the retry loop returns or raises")

    @staticmethod
    def _checked(label: str, value: Any) -> Any:
        where = _not_json(value)
        if where:
            raise _NotJson(f"{label} returned a value that is not plain JSON: {where}")
        return value

    def _once(self, step: dict[str, Any], payload: dict[str, Any]) -> Any:
        if step["kind"] == "call":
            fn = getattr(importlib.import_module(step["module"]), step["symbol"])
            return _serialize(_call_impl(fn, {**payload, **(step.get("constants") or {})}))
        if step["kind"] == "judge":
            return self._judge_by_helper(step, payload)
        mcp = importlib.import_module("_rote_mcp")
        if step["args"] is None:
            arguments = payload
        else:
            arguments = {
                arg: value if origin in ("constant", "literal") else payload[value]
                for arg, origin, value in step["args"]
            }
        return _run_async(mcp.call_mcp_tool(step["server"], step["url"], step["tool"], arguments))

    @staticmethod
    def _judge_by_helper(step: dict[str, Any], payload: dict[str, Any]) -> Any:
        """One judge call through the inference helper, as the emitted Python judge makes it.

        The input is checked and projected through the judge's input schema
        and rendered with the same ``interpolate``; the model, endpoint and
        temperature are the IR's, each overridable per node by the
        ``ROTE_MODEL_`` / ``ROTE_BASE_URL_`` / ``ROTE_TEMPERATURE_<ID>``
        knobs every runtime reads. The answer is held to the output schema
        and projected as the judge's model would dump it.
        """
        schema = _schema()
        errors = schema.schema_errors(step["input_schema"], payload, "input")
        if errors:
            raise ValueError(
                f"{step['id']}'s input does not fit its input schema: " + "; ".join(errors[:5])
            )
        prompt = interpolate(step["prompt"], schema.project(step["input_schema"], payload))
        lane = step["lane"]
        knob = step["id"].upper()
        kwargs: dict[str, Any] = {}
        if lane.get("temperature") is not None:
            temperature = os.environ.get(f"ROTE_TEMPERATURE_{knob}", str(lane["temperature"]))
            kwargs["sampling"] = {"temperature": float(temperature)} if temperature.strip() else {}
        answer = importlib.import_module("_rote_inference").call_judge(
            node_id=step["id"],
            client=lane["client"],
            model=os.environ.get(f"ROTE_MODEL_{knob}", lane["model"]),
            base_url=os.environ.get(f"ROTE_BASE_URL_{knob}", lane.get("base_url")),
            prompt=prompt,
            output_schema=step["output_schema"],
            **kwargs,
        )
        errors = schema.schema_errors(step["output_schema"], answer)
        if errors:
            raise ValueError(
                f"{step['id']}'s judge answered outside its output schema: " + "; ".join(errors[:5])
            )
        return schema.project(step["output_schema"], answer)

    def _failure_block(
        self, path: Path, state: dict[str, Any], step: dict[str, Any], exc: BaseException
    ) -> Block:
        name = type(exc).__name__
        run = path.name
        if name == "RoteMcpAuthNeeded":
            server = str(getattr(exc, "server", "") or "")
            state["pending"] = {"step": step["id"], "kind": "login", "server": server}
            state["state"] = "waiting"
            return self._login_block(run, step["id"], server)
        state["state"] = "error"
        state["error"] = f"{name}: {exc}"
        if name == "RoteMcpConfigError":
            message = str(exc)
            server = str(step.get("server") or "")
            if "this run is read-only" in message:
                fix = f"rote skill trust {self.name} --allow-writes (after the user agrees)"
            else:
                fix = (
                    f"rote mcp add {server} <url> (or set ROTE_MCP_{server.upper()}_URL), "
                    f"then: {self.command} resume {run}"
                )
            return self._error(run, message, fix, step=step["id"])
        if name == "RoteMcpInputRequired":
            return self._error(
                run,
                str(exc),
                "the step cannot run unattended until the server takes that input as tool "
                "arguments; tell the user, and compile it for a workflow runtime meanwhile",
                step=step["id"],
            )
        if isinstance(exc, (BindingError, _NotJson)):
            return self._error(
                run,
                f"{step['id']}: {exc}. The compiled pipeline does not fit this data.",
                "recompile the skill (rote compile)",
                step=step["id"],
            )
        where = _location(exc, self.scripts)
        return self._error(
            run,
            f"{step['id']} raised {name}: {str(exc)[:500]}{where}",
            f"{self.command} resume {run} to retry",
            step=step["id"],
        )

    # ── asking ──

    def _ask(
        self, path: Path, state: dict[str, Any], step: dict[str, Any], results: dict[str, Any]
    ) -> Block:
        run = path.name
        try:
            payload = _payload(step, state["input"], results)
        except BindingError as exc:
            state["state"] = "error"
            state["error"] = str(exc)
            self._write(path, state)
            return self._error(
                run, f"{step['id']}: {exc}.", "recompile the skill (rote compile)", step=step["id"]
            )
        if step["kind"] == "gate":
            pending = {
                "step": step["id"],
                "kind": "gate",
                "deadline": _now() + float(step["timeout_s"]),
                "payload": payload,
            }
        else:
            prepared = self._judge_items(step, payload)
            if isinstance(prepared, Block):
                state["state"] = "error"
                state["error"] = prepared.lines[0]
                self._write(path, state)
                prepared.header = self._head(run, f"step={step['id']}")
                return prepared
            pending = {"step": step["id"], "kind": "decide", **prepared}
        state["pending"] = pending
        state["state"] = "waiting"
        state["attempts"][step["id"]] = 0
        self._write(path, state)
        return self._pending_block(run, step, pending)

    def _judge_items(self, step: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any] | Block:
        """The judge's input as its emitted model would see it, cut to what the prompt reads."""
        schema = _schema()
        spec_in = step["input_schema"]
        fan = step["fan_param"]
        refs = [r.split(".") for r in step["references"]]
        if fan is None:
            errors = schema.schema_errors(spec_in, payload, "input")
            if errors:
                return self._judge_input_error(step, errors)
            projected = schema.project(spec_in, payload)
            try:
                task = interpolate(step["prompt"], projected)
            except KeyError as exc:
                return self._judge_input_error(step, [str(exc.args[0])])
            return {"task": task}
        try:
            elements = _fan_list(payload.get(fan), step["id"], fan)
        except TypeError as exc:
            return self._judge_input_error(step, [str(exc)])
        shared_payload = {k: v for k, v in payload.items() if k != fan}
        items: list[Any] = []
        shared: dict[str, Any] = {}
        for index, element in enumerate(elements):
            full = {**shared_payload, fan: element}
            errors = schema.schema_errors(spec_in, full, f"input[{index}]")
            if errors:
                return self._judge_input_error(step, errors)
            projected = schema.project(spec_in, full)
            items.append(select(projected.get(fan), [r[1:] for r in refs if r[0] == fan]))
            if index == 0:
                shared = {
                    k: select(projected.get(k), [r[1:] for r in refs if r[0] == k])
                    for k in dict.fromkeys(r[0] for r in refs)
                    if k != fan and k in projected
                }
        return {
            "shared": shared,
            "items": items,
            "count": len(items),
            "batches": _batches(items, step["output_schema"]),
            "batch": 0,
            "collected": {},
        }

    def _judge_input_error(self, step: dict[str, Any], errors: list[str]) -> Block:
        return self._error(
            None,
            f"{step['id']}'s input does not fit its input schema: " + "; ".join(errors[:5]) + ".",
            "recompile the skill (rote compile)",
            step=step["id"],
        )

    def _pending_block(
        self, run: str, step: dict[str, Any], pending: dict[str, Any], note: str = ""
    ) -> Block:
        if pending["kind"] == "login":
            return self._login_block(run, step["id"], pending["server"], note)
        if pending["kind"] == "gate":
            return self._gate_block(run, step, pending, note)
        return self._decide_block(run, step, pending, note)

    def _decide_block(
        self, run: str, step: dict[str, Any], pending: dict[str, Any], note: str = ""
    ) -> Block:
        notes = [note] if note else []
        schema_line = _compact(step["output_schema"])
        if step["fan_param"] is None:
            lines = [
                *notes,
                "Decide this yourself, as TASK says, using only what is here. Do not ask the user.",
                "--- TASK ---",
                pending["task"],
                "--- ANSWER SCHEMA (json) ---",
                schema_line,
                *self._next("answer", run, step["id"], "'<your JSON>'"),
            ]
            return Block("DECIDE", self._head(run, f"step={step['id']}"), lines)
        ids = pending["batches"][pending["batch"]]
        fan = step["fan_param"]
        lines = [
            *notes,
            f"Decide each item yourself, as TASK says; {{{{ {fan} }}}} is the item. "
            "Do not ask the user.",
            "--- TASK ---",
            step["prompt"].strip(),
        ]
        if pending["shared"]:
            lines += ["--- SHARED (json) ---", _compact(pending["shared"])]
        lines += [
            "--- ITEMS (json, by id) ---",
            _compact({i: pending["items"][int(i)] for i in ids}),
            "--- ANSWER SCHEMA (json, one per item) ---",
            schema_line,
            "Answer one JSON object mapping each item id to its answer.",
            *self._next("answer", run, step["id"], "'<your JSON>'"),
        ]
        extra = []
        if len(pending["batches"]) > 1:
            extra.append(f"batch={pending['batch'] + 1}/{len(pending['batches'])}")
        return Block("DECIDE", self._head(run, f"step={step['id']}", *extra), lines)

    def _gate_block(
        self, run: str, step: dict[str, Any], pending: dict[str, Any], note: str = ""
    ) -> Block:
        payload = pending["payload"]
        lines = [note] if note else []
        lines += [
            "Do not answer this yourself. Ask the person in plain words and wait for their reply.",
            "--- QUESTION ---",
            step["description"],
        ]
        shown: set[str] = set()
        if step.get("message"):
            message, shown = _render_message(step["message"], payload)
            lines.append(message)
        unshown = {k: v for k, v in payload.items() if k not in shown}
        if unshown:
            lines += ["--- THEY SEE (json) ---", _compact(unshown)]
        lines += [
            "--- ANSWER SCHEMA (json) ---",
            _compact(step["answer_schema"]),
            "Put their reply into this shape. If it does not fit, ask again; never fill in "
            "what they did not say.",
            *self._next(
                "answer", run, step["id"], "'<JSON>'", "--said", "'<their words, verbatim>'"
            ),
            f"Deadline: {_iso(float(pending['deadline']))}",
        ]
        return Block("HUMAN", self._head(run, f"step={step['id']}", "kind=gate"), lines)

    def _login_block(self, run: str, step: str, server: str, note: str = "") -> Block:
        lines = [note] if note else []
        lines += [
            f"Do not answer this yourself. The person must log in to {server}:",
            f"at their terminal: rote mcp login {server}",
            f"or relay it: rote mcp login {server} --no-wait (give them the URL), then "
            f"rote mcp login {server} --complete",
            *self._next("resume", run),
        ]
        return Block(
            "HUMAN", self._head(run, f"step={step}", "kind=login", f"server={server}"), lines
        )

    def _current(self, path: Path, state: dict[str, Any], note: str = "") -> Block:
        """The block this run stands at: what ``status`` prints."""
        run = path.name
        if state["state"] == "done":
            return self._done_block(path, note)
        if state["state"] == "cancelled":
            return Block(
                "DONE", self._head(run, "cancelled"), [note] if note else ["The run is cancelled."]
            )
        if state["state"] == "expired":
            return self._error(
                run,
                "a gate's deadline passed and the run has ended.",
                f"start a new run: {self.command} start '<input json>'",
            )
        if state["state"] == "error":
            return self._error(
                run,
                f"{state.get('error')}" + (f" ({note})" if note else ""),
                f"{self.command} resume {run}",
            )
        pending = state.get("pending")
        if pending:
            return self._pending_block(run, self.by_id[pending["step"]], pending, note)
        self._claim_free(path, state)
        return self._error(run, "the run is between steps.", f"{self.command} resume {run}")

    # ── the end ──

    def _finish(self, path: Path, results: dict[str, Any]) -> Block:
        result = {node: results[node] for node in self.plan["exit_nodes"]}
        _atomic_write(path / "result.json", json.dumps(result, ensure_ascii=False, indent=1) + "\n")
        with self._locked(path):
            state = self._read(path)
            state["state"] = "done"
            state["pending"] = None
            self._write(path, state)
        return self._done_block(path)

    def _done_block(self, path: Path, note: str = "") -> Block:
        file = path / "result.json"
        result = json.loads(file.read_text(encoding="utf-8"))
        lines = [note] if note else []
        lines += [
            f"result: {file} ({file.stat().st_size} bytes). Read it for the full result.",
            "summary: " + summarize(result),
        ]
        return Block("DONE", self._head(path.name), lines)

    # ── housekeeping ──

    def _runs(self) -> list[tuple[Path, dict[str, Any]]]:
        """This skill's run folders: real directories (never a symlink) whose own
        ``run.json`` (never a symlink either) this driver wrote. Anything else,
        or anything unreadable, is skipped, so one bad entry breaks nothing."""
        found: list[tuple[Path, dict[str, Any]]] = []
        if self.root.is_symlink() or not self.root.is_dir():
            return found
        for child in self.root.iterdir():
            try:
                journal = child / "run.json"
                if child.is_symlink() or not child.is_dir() or journal.is_symlink():
                    continue
                state = json.loads(journal.read_text(encoding="utf-8"))
                float(state.get("updated", 0))
            except (OSError, ValueError, TypeError, AttributeError):
                continue
            if isinstance(state, dict) and state.get("format") == RUN_FORMAT:
                found.append((child, state))
        return found

    def sweep(self) -> list[str]:
        """Remove this skill's expired runs; only dirs holding a run.json this driver wrote."""
        import shutil

        now = _now()
        day = 86400.0
        removed = []
        for path, state in self._runs():
            try:
                updated = float(state.get("updated", 0))
                pending = state.get("pending") or {}
                if state.get("state") in (*_FINISHED, "error"):
                    expired = now - updated > SWEEP_FINISHED_DAYS * day
                elif pending.get("deadline"):
                    expired = now > float(pending["deadline"]) + SWEEP_GRACE_DAYS * day
                else:
                    expired = now - updated > SWEEP_WAITING_DAYS * day
                # Checked again right before removing: rmtree on a symlink swapped
                # in since the listing would remove what it points at.
                if expired and not path.is_symlink() and path.is_dir():
                    shutil.rmtree(path)
                    removed.append(path.name)
            except (OSError, ValueError, TypeError, AttributeError) as exc:
                _progress(f"sweep skipped {path.name}: {exc}")
        return removed

    # ── input ──

    def _parse(self, run: str | None, raw: str, what: str) -> Any:
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise Stop(
                self._error(
                    run,
                    f"{what} is not JSON (line {e.lineno} column {e.colno}: {e.msg}).",
                    "pass it as one single-quoted argument, or write it to a file and pass "
                    "--file <path>",
                    usage=True,
                )
            ) from None


class _NotJson(Exception):
    """A step returned a value a journal cannot hold."""


def _location(exc: BaseException, root: Path) -> str:
    import traceback

    found = ""
    for frame in traceback.extract_tb(exc.__traceback__):
        try:
            rel = Path(frame.filename).resolve().relative_to(root.resolve())
        except ValueError:
            continue
        found = f" at {rel}:{frame.lineno}"
    return found


def _render_message(template: str, payload: dict[str, Any]) -> tuple[str, set[str]]:
    """A gate's ``{name}`` placeholders filled from its declared scalar inputs,
    and the names it filled (the block does not show those values twice)."""
    used: set[str] = set()

    def _sub(match: re.Match[str]) -> str:
        value = payload.get(match.group(1), _MISSING)
        if value is _MISSING or isinstance(value, (dict, list)):
            return match.group(0)
        used.add(match.group(1))
        return "" if value is None else str(value)

    return re.sub(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", _sub, template), used


def _answer_tokens(schema: Any, depth: int = 0) -> float:
    """A rough count of the tokens one answer to ``schema`` takes to write."""
    if not isinstance(schema, dict) or depth > 6:
        return 20.0
    for key in ("anyOf", "oneOf"):
        if key in schema:
            return max(_answer_tokens(s, depth + 1) for s in schema[key])
    if "$ref" in schema:
        return 40.0
    kind = schema.get("type")
    if kind == "object":
        props = schema.get("properties") or {}
        return 2.0 + sum(3.0 + _answer_tokens(p, depth + 1) for p in props.values())
    if kind == "array":
        return 3.0 * _answer_tokens(schema.get("items"), depth + 1)
    if kind == "string" and "enum" not in schema:
        return 40.0
    return 3.0


def _batches(items: list[Any], output_schema: Any) -> list[list[str]]:
    """Item ids in DECIDE batches: at most :data:`MAX_BATCH_ITEMS`, within both budgets."""
    per_answer = _answer_tokens(output_schema)
    batches: list[list[str]] = []
    current: list[str] = []
    used = 0.0
    for index, item in enumerate(items):
        size = len(_compact(item)) / CHARS_PER_TOKEN
        full = (
            len(current) >= MAX_BATCH_ITEMS
            or used + size > BATCH_INPUT_TOKENS
            or (len(current) + 1) * per_answer > BATCH_OUTPUT_TOKENS
        )
        if current and full:
            batches.append(current)
            current, used = [], 0.0
        current.append(str(index))
        used += size
    if current:
        batches.append(current)
    return batches or [[]]


def _describe(value: Any) -> str:
    if isinstance(value, list):
        return f"<list, {len(value)} items>"
    if isinstance(value, dict):
        return f"<object, {len(value)} keys>"
    if isinstance(value, str) and len(value) > 60:
        return f"<text, {len(value)} chars>"
    return _compact(value)


def summarize(result: dict[str, Any]) -> str:
    """At most :data:`DONE_SUMMARY_BYTES` of the result: scalars inline, sizes for the rest.

    One exit node is summarized by its own keys; several, by each node's.
    """
    body: Any = next(iter(result.values())) if len(result) == 1 else result
    if not isinstance(body, dict):
        return _describe(body)[:DONE_SUMMARY_BYTES]
    parts: list[str] = []
    used = 2
    for index, (key, value) in enumerate(body.items()):
        part = f"{_compact(key)}: {_describe(value)}"
        rest = len(body) - index
        if used + len(part.encode("utf-8")) + 2 > DONE_SUMMARY_BYTES - 24:
            parts.append(f"... (+{rest} more keys)")
            break
        parts.append(part)
        used += len(part.encode("utf-8")) + 2
    return "{" + ", ".join(parts) + "}"


# ───────── the command line ─────────


def _value_arg(args: list[str], what: str) -> str:
    """One JSON argument, or ``--file <path>``."""
    if len(args) == 2 and args[0] == "--file":
        return Path(args[1]).read_text(encoding="utf-8")
    if len(args) == 1 and not args[0].startswith("--"):
        return args[0]
    raise ValueError(f"expected {what} as one single-quoted argument, or --file <path>")


def _said(args: list[str]) -> tuple[list[str], str | None]:
    for flag in ("--said", "--said-file"):
        if flag in args:
            i = args.index(flag)
            if i + 1 >= len(args):
                raise ValueError(f"{flag} needs a value")
            value = args[i + 1]
            if flag == "--said-file":
                value = Path(value).read_text(encoding="utf-8")
            return args[:i] + args[i + 2 :], value
    return args, None


USAGE = (
    "start '<input json>' | --file <path>; "
    "answer <run> <step> '<json>' | --file <path> [--said '<their words>']; "
    "resume <run>; status [<run>]; cancel <run>; runs [--prune]"
)


def dispatch(driver: Driver, argv: list[str]) -> Block:
    verb, args = (argv[0], argv[1:]) if argv else ("", [])
    try:
        if verb == "start":
            return driver.start(_value_arg(args, "the input"))
        if verb == "answer" and len(args) >= 3:
            rest, said = _said(args[2:])
            return driver.answer(args[0], args[1], _value_arg(rest, "the answer"), said)
        if verb == "resume" and len(args) == 1:
            return driver.resume(args[0])
        if verb == "status" and len(args) <= 1:
            return driver.status(args[0] if args else None)
        if verb == "cancel" and len(args) == 1:
            return driver.cancel(args[0])
        if verb == "runs" and args in ([], ["--prune"]):
            return driver.runs(prune=bool(args))
    except ValueError as e:
        return driver._error(None, str(e) + ".", f"{driver.command} {USAGE}", usage=True)
    except OSError as e:
        return driver._error(
            None,
            f"cannot read {e.filename}: {e.strerror}.",
            "pass the path of a readable file",
            usage=True,
        )
    return driver._error(
        None,
        f"{argv[0] if argv else 'no command'} is not a command here.",
        f"{driver.command} {USAGE}",
        usage=True,
    )


def _read_only_servers(scripts: Path) -> dict[str, str]:
    """``{server: url}`` for every server the user registered read-only.

    Read from the registry the MCP helper already reads for endpoints
    (``_registry_servers``), so a declaration means the same thing here as
    on every MCP runtime: the helper honors it only at that URL. A skill
    with no MCP step ships no ``_rote_mcp.py`` and reads nothing.
    """
    if not (scripts / "_rote_mcp.py").is_file():
        return {}
    servers = importlib.import_module("_rote_mcp")._registry_servers()
    return {
        str(name): str(config["url"])
        for name, config in servers.items()
        if isinstance(config, dict) and config.get("read_only") is True and config.get("url")
    }


def main(
    argv: list[str],
    scripts: Path | None = None,
    command: str | None = None,
    *,
    allow_writes: bool = False,
    skill: str | None = None,
) -> int:
    """Run one command and print its one block. The skill's scripts dir goes first on the path.

    ``skill`` is the compiled skill's name, which the emitted ``run.py``
    passes (see :class:`Driver`).

    ``allow_writes`` is the user's consent to write tools, and it is a
    parameter, never read from the environment: the agent's shell sets the
    environment. The caller that holds the consent record (``rote skill
    run``, from the trust registry) passes it; until that exists every
    write tool is refused before it is called. For the same reason the
    read-only server list the MCP helper honors is never taken from the
    environment: it is rebuilt from the user's own declarations, the
    servers ``rote mcp add --read-only`` registered (:func:`_read_only_servers`).
    """
    scripts = (scripts or Path(__file__).resolve().parent).resolve()
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    if allow_writes:
        os.environ.pop("ROTE_MCP_READ_ONLY", None)
    else:
        os.environ["ROTE_MCP_READ_ONLY"] = "1"
    os.environ.pop("ROTE_MCP_READ_ONLY_SERVERS", None)
    declared = _read_only_servers(scripts)
    if declared:
        os.environ["ROTE_MCP_READ_ONLY_SERVERS"] = json.dumps(declared)
    driver: Driver | None = None
    try:
        driver = Driver(scripts, command or os.environ.get("ROTE_SKILL_COMMAND") or None, skill)
        block = dispatch(driver, argv)
    except Stop as stop:
        block = stop.block
    finally:
        if driver is not None:
            driver.close()
    sys.stdout.write(block.text())
    sys.stdout.flush()
    return block.status


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
