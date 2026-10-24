"""Run skills and emitted pipelines locally (``rote run``).

The runners layer is detection + ergonomics over machinery that already
exists and is already tested: the skill side reuses the baseline trial
runner (streamed ``claude -p`` with the rote registry injected as MCP
config and the read-only gate applied), and the pipeline side reuses
the empirical trial runner (plain python script, or DBOS app with
cross-process gate signaling via ``DBOSClient``). Nothing in this
module spawns a subprocess of its own — see :mod:`rote.eval.baseline`
and :mod:`rote.eval.empirical` for the execution primitives.
"""

from __future__ import annotations

import contextlib
import json
import os
import shlex
import sys
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from rote.adapters._common import unimplemented_stubs
from rote.eval.baseline import (
    ObservedToolCall,
    derive_input_payload,
    resolve_mcp_wiring,
    run_baseline_trial,
)
from rote.eval.empirical import (
    GATE_INSTRUCTIONS,
    MeasuredRun,
    gate_answers_server,
    run_pipeline_trial,
)
from rote.eval.gate_server import GATE_SERVER
from rote.extras import EXTRAS, RUNTIME_EXTRAS, MissingDependencyError, require_module

__all__ = [
    "RUNNABLE_RUNTIMES",
    "PipelineRunOutcome",
    "RunTarget",
    "SkillRunOutcome",
    "TargetError",
    "detect_target",
    "native_run_hint",
    "parse_signal_args",
    "preflight_pipeline",
    "resolve_gate_signals",
    "resolve_input",
    "run_pipeline",
    "run_skill",
]


class TargetError(RuntimeError):
    """The path cannot be run as given; the message carries the fix."""


#: Adapter names ``rote run`` can execute locally — all six emitted
#: runtimes. The hint table below covers any future adapter whose
#: orchestration hasn't landed yet.
RUNNABLE_RUNTIMES = frozenset({"python", "dbos", "cloudflare", "inngest", "dbos-ts", "temporal"})

_NATIVE_RUN_HINTS: dict[str, str] = {}


def native_run_hint(runtime: str) -> str:
    """How to run an emitted runtime ``rote run`` does not orchestrate yet."""
    return _NATIVE_RUN_HINTS.get(runtime, "see the emitted README.md")


# ───────── Target detection ─────────


@dataclass(frozen=True)
class RunTarget:
    """What ``rote run <path>`` resolved to."""

    kind: Literal["skill", "pipeline"]
    path: Path
    """The skill directory, or the emitted runtime directory."""
    runtime: str | None = None
    """Adapter name for pipeline targets (``python`` / ``dbos`` / …)."""
    pipeline_yaml: Path | None = None
    """The compiled IR, when it can be located — enables gate-name
    discovery and input derivation from the recorded source skill."""


def _detect_runtime(d: Path) -> str | None:
    """Classify an emitted runtime directory by its marker files.

    Order matters: cloudflare and dbos-ts both carry a ``package.json``,
    and dbos (python) shares ``dbos-config.yaml`` with dbos-ts — the
    discriminators are ``wrangler.jsonc``, ``src/inngest/``, and
    ``main.py`` respectively.
    """
    if (d / "wrangler.jsonc").is_file():
        return "cloudflare"
    if (d / "src" / "inngest").is_dir():
        return "inngest"
    if (d / "dbos-config.yaml").is_file():
        return "dbos" if (d / "main.py").is_file() else "dbos-ts"
    if (d / "workflow.py").is_file() and (d / "activities.py").is_file():
        return "temporal"
    if (d / "main.py").is_file():
        return "python"
    return None


def _find_pipeline_yaml(runtime_dir: Path) -> Path | None:
    """Locate the compiled IR for an emitted runtime directory.

    A ``compile --out`` layout keeps it at ``../../compiled/`` relative
    to ``runtime/<target>/``; a bare ``rote emit`` output may sit next to
    the pipeline.yaml it was rendered from.
    """
    candidates = [runtime_dir / "pipeline.yaml"]
    if runtime_dir.parent.name == "runtime":
        candidates.append(runtime_dir.parent.parent / "compiled" / "pipeline.yaml")
    candidates.append(runtime_dir.parent / "pipeline.yaml")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def detect_target(path: str | Path, runtime: str | None = None) -> RunTarget:
    """Resolve ``rote run``'s path argument to something executable.

    Accepts a skill directory (``SKILL.md``), an emitted runtime
    directory (marker files), or a ``rote compile --out`` directory
    (``runtime/<target>/`` children — ``runtime`` disambiguates when
    several were emitted). Raises :class:`TargetError` with actionable
    guidance for everything else.
    """
    p = Path(path).resolve()
    if not p.is_dir():
        raise TargetError(f"{p} is not a directory")
    from rote.adapters.skill import emitted_skill

    if emitted_skill(p) is not None:
        # Not a source skill: running it as one would pay an agent to run
        # a skill whose steps are code. Its own driver runs it.
        raise TargetError(
            f"{p} is a compiled skill (--runtime skill): an agent client runs it through "
            f"its driver, not `rote run`. To run the driver yourself: "
            f"{shlex.quote(sys.executable)} -I {shlex.quote(str(p / 'scripts' / 'run.py'))} "
            "start '<input json>'"
        )
    if (p / "SKILL.md").is_file():
        return RunTarget(kind="skill", path=p)

    if (p / "runtime").is_dir():
        emitted = sorted(
            d for d in (p / "runtime").iterdir() if d.is_dir() and _detect_runtime(d) is not None
        )
        if not emitted:
            raise TargetError(
                f"{p / 'runtime'} contains no emitted runtime — run `rote emit` first"
            )
        if runtime is not None:
            matches = [d for d in emitted if d.name == runtime]
            if not matches:
                raise TargetError(
                    f"no emitted `{runtime}` runtime under {p / 'runtime'} "
                    f"(found: {', '.join(d.name for d in emitted)})"
                )
            chosen = matches[0]
        elif len(emitted) == 1:
            chosen = emitted[0]
        else:
            raise TargetError(
                f"{p / 'runtime'} contains several emitted runtimes "
                f"({', '.join(d.name for d in emitted)}) — pick one with --runtime"
            )
        return RunTarget(
            kind="pipeline",
            path=chosen,
            runtime=chosen.name,
            pipeline_yaml=_find_pipeline_yaml(chosen),
        )

    detected = _detect_runtime(p)
    if detected is None:
        raise TargetError(
            f"{p} is neither a skill (no SKILL.md) nor an emitted runtime "
            "directory — pass a skill dir, an emitted runtime dir, or a "
            "`rote compile --out` dir"
        )
    if runtime is not None and runtime != detected:
        raise TargetError(f"{p} looks like an emitted `{detected}` runtime, not `{runtime}`")
    return RunTarget(
        kind="pipeline", path=p, runtime=detected, pipeline_yaml=_find_pipeline_yaml(p)
    )


# ───────── Input resolution ─────────


def resolve_input(
    input_arg: str | None, *, skill_dir: Path | None, assume_yes: bool
) -> dict[str, Any] | None:
    """The run's task payload: inline JSON, a file, or derived + confirmed.

    ``input_arg`` may be an inline JSON object (starts with ``{``) or a
    path to a JSON file. With no ``input_arg``, the payload is derived
    from the skill's text (cheap single-shot call) and must be accepted —
    interactively on a terminal, or via ``assume_yes``. The proposal is
    persisted to a temp file so a declined run can be edited and re-run
    with ``--input``. Returns ``None`` when the user declines (callers
    exit 0); raises :class:`TargetError` for usage errors.
    """
    if input_arg is not None:
        stripped = input_arg.strip()
        if stripped.startswith("{"):
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as e:
                raise TargetError(f"--input is not valid JSON: {e}") from e
        else:
            # `@FILE`, as --signal takes it, or a bare path.
            input_path = Path(input_arg[1:] if input_arg.startswith("@") else input_arg)
            if not input_path.is_file():
                raise TargetError(
                    f"--input is neither an inline JSON object nor a file: {input_path}"
                )
            try:
                payload = json.loads(input_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                raise TargetError(f"--input file is not valid JSON: {e}") from e
        if not isinstance(payload, dict):
            raise TargetError("--input must be a JSON object")
        return payload

    if skill_dir is None:
        raise TargetError(
            "no --input given and no source skill to derive one from — pass "
            "--input with an inline JSON object or a file path"
        )
    print(
        "no --input given — deriving a representative task from SKILL.md (cheap single-shot call)…",
        file=sys.stderr,
    )
    derived = derive_input_payload(skill_dir)
    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", prefix="rote-run-input-", delete=False, encoding="utf-8"
    ) as f:
        f.write(json.dumps(derived, indent=2) + "\n")
        derived_path = Path(f.name)

    print("derived input proposal:", file=sys.stderr)
    print(json.dumps(derived, indent=2), file=sys.stderr)
    if assume_yes:
        print("--yes: accepting derived input", file=sys.stderr)
        return derived
    if not sys.stdin.isatty():
        raise TargetError(
            "not a terminal and no --input given — pass --input, or --yes to "
            f"accept the derived proposal (saved to {derived_path})"
        )
    answer = input(f"Run with this input? (saved to {derived_path}) [y/N] ")
    if answer.strip().lower() not in {"y", "yes"}:
        print(
            f"declined — edit {derived_path} and re-run with --input {derived_path}",
            file=sys.stderr,
        )
        return None
    return derived


# ───────── HITL gate signals ─────────


def parse_signal_args(raw: list[str]) -> dict[str, Any]:
    """``--signal NAME=JSON`` / ``--signal NAME=@FILE`` arguments → ``{name: payload}``.

    ``@FILE`` reads the payload from a JSON file, the curl convention, so a
    gate answer too long for a shell argument does not need quoting.
    """
    signals: dict[str, Any] = {}
    for item in raw:
        name, sep, value = item.partition("=")
        if not sep or not name:
            raise TargetError(f"--signal must be NAME=JSON or NAME=@FILE, got: {item!r}")
        if value.startswith("@"):
            path = Path(value[1:])
            try:
                value = path.read_text(encoding="utf-8")
            except OSError as e:
                raise TargetError(f"--signal {name}: cannot read {path}: {e}") from e
            source = f"{path} is"
        else:
            source = "payload is"
        try:
            signals[name] = json.loads(value)
        except json.JSONDecodeError as e:
            raise TargetError(f"--signal {name}: {source} not valid JSON: {e}") from e
    return signals


def resolve_gate_signals(
    gate_signals: list[str], provided: dict[str, Any], *, interactive: bool
) -> dict[str, Any]:
    """Ensure every HITL gate has a resume payload before the run starts.

    DBOS notifications persist per-topic, so payloads collected up front
    are safe to send before the workflow reaches its ``recv()`` — the
    gate picks them up when it parks. Interactively, missing gates are
    prompted for (empty answer = ``{}``); non-interactively they are a
    hard error listing the exact flags to pass, because a gated run with
    no payload coming would park until its IR timeout.
    """
    unknown = sorted(set(provided) - set(gate_signals))
    if unknown:
        raise TargetError(
            f"--signal names not in the pipeline's gates: {', '.join(unknown)} "
            f"(gates: {', '.join(gate_signals) or 'none'})"
        )
    missing = [g for g in gate_signals if g not in provided]
    if not missing:
        return provided
    if not interactive:
        flags = " ".join(f"--signal {g}='{{...}}'" for g in missing)
        raise TargetError(
            f"this pipeline has HITL gate(s) with no resume payload: "
            f"{', '.join(missing)} — pass {flags} (payloads are delivered "
            "when the workflow parks)"
        )
    resolved = dict(provided)
    for gate in missing:
        answer = input(f"HITL gate '{gate}' — resume payload JSON (empty for {{}}): ").strip()
        if not answer:
            resolved[gate] = {}
            continue
        try:
            resolved[gate] = json.loads(answer)
        except json.JSONDecodeError as e:
            raise TargetError(f"gate '{gate}': payload is not valid JSON: {e}") from e
    return resolved


# ───────── Execution ─────────


@dataclass(frozen=True)
class SkillRunOutcome:
    """One ``rote run`` of a raw skill."""

    run: MeasuredRun
    observations: tuple[ObservedToolCall, ...]
    servers_wired: tuple[str, ...]
    servers_skipped: dict[str, str]
    read_only: bool
    #: How many times the run called ``ask_person`` (None without ``--signal``).
    gates_asked: int | None = None

    @property
    def observed_servers(self) -> list[str]:
        return sorted({o.server for o in self.observations})


def run_skill(
    skill_dir: Path,
    input_payload: dict[str, Any],
    *,
    model: str,
    allow_writes: bool = False,
    signals: dict[str, Any] | None = None,
    max_turns: int = 60,
    timeout_seconds: float = 1800.0,
    executable: str = "claude",
) -> SkillRunOutcome:
    """Run the raw skill once as an agent, same rules as the baseline.

    Registry-wide MCP injection, read-only gate unless ``allow_writes``,
    subscription billing (API-key env vars scrubbed). The observed MCP
    traffic rides along — it's free and it tells the user what the run
    actually touched. ``signals`` are the answers a person gives at the
    skill's gates, in order: a skill has no gate names of its own, so a
    name is only a label, and the run is told the answers up front the way
    the compile bench's references are: through ``ask_person``
    (:mod:`rote.eval.gate_server`), one answer per ask, when the run
    reaches each gate. ``gates_asked`` says how many it asked for.
    """
    wiring = resolve_mcp_wiring(allow_writes=allow_writes)
    with tempfile.TemporaryDirectory(prefix="rote-gates-") as tmp:
        work = Path(tmp)
        wired = dict(wiring.servers)
        tool_ids = list(wiring.allowed)
        if signals:
            entry, gate_tool = gate_answers_server(list(signals.values()), work)
            wired[GATE_SERVER] = entry
            tool_ids.append(gate_tool)
        run, observed, _lines = run_baseline_trial(
            skill_dir,
            input_payload,
            model=model,
            max_turns=max_turns,
            mcp_tool_ids=tool_ids,
            denied_mcp_tool_ids=wiring.denied,
            executable=executable,
            timeout_seconds=timeout_seconds,
            mcp_servers=wired,
            servers_skipped=wiring.skipped,
            extra_instructions=GATE_INSTRUCTIONS if signals else None,
        )
        asks = work / "calls.jsonl"
        gates_asked = (
            sum(1 for line in asks.read_text(encoding="utf-8").splitlines() if line)
            if signals and asks.is_file()
            else (0 if signals else None)
        )
    return SkillRunOutcome(
        run=run,
        observations=tuple(o for o in observed if o.server != GATE_SERVER),
        servers_wired=tuple(sorted(wiring.servers)),
        servers_skipped=wiring.skipped,
        read_only=not allow_writes,
        gates_asked=gates_asked,
    )


#: Runtimes whose emitted implementations are TypeScript.
_TS_RUNTIMES = frozenset({"cloudflare", "inngest", "dbos-ts"})


def preflight_pipeline(target: RunTarget, *, allow_stubs: bool = False) -> None:
    """Refuse a run that cannot work, before anything starts or retries.

    Two checks, both cheap and both things a newcomer hits first:

    * the runtime's extra is installed (otherwise the app dies on its
      first import, behind a traceback that never says how to install it).
      ``rote run`` offers the install first
      (:mod:`rote.setup.install_extra`); this is the refusal for a caller
      that did not, or a person who declined;
    * no emitted ``extracted/`` module is still a stub. A stub raises on
      every call, and a durable runtime retries it per the IR's policy
      before failing, so the user waits through the retries only to
      learn something the emit manifest already knew. ``allow_stubs``
      skips this one, for a pipeline whose stubbed node is on a branch
      the run will not take (``rote run --allow-stubs``).

    ``rote run`` calls this before input derivation and gate prompts;
    :func:`run_pipeline` does not repeat it.
    """
    runtime = target.runtime or ""
    extra = RUNTIME_EXTRAS.get(runtime)
    if extra is not None:
        try:
            require_module(EXTRAS[extra].module, f"running a {runtime} app")
        except MissingDependencyError as e:
            raise TargetError(str(e)) from e
    stubs = [] if allow_stubs else unimplemented_stubs(target.path)
    if stubs:
        listing = "\n".join(f"  {stub}" for stub in stubs)
        if runtime in _TS_RUNTIMES:
            # A TS app's implementations are TypeScript: Python modules
            # beside the pipeline.yaml are never sourced into it.
            fix = "Implement them in TypeScript in the files above"
        else:
            fix = (
                "Implement them, or re-emit with the real modules in an "
                "extracted/ directory beside the pipeline.yaml (emit copies them in)"
            )
        raise TargetError(
            f"{len(stubs)} emitted module(s) are still stubs and would raise "
            f"on their first call:\n{listing}\n"
            f"{fix}, then re-run: rote run {target.path}\n"
            f"If the run will not reach them, pass --allow-stubs."
        )


@dataclass(frozen=True)
class PipelineRunOutcome:
    """One ``rote run`` of an emitted pipeline."""

    run: MeasuredRun
    runtime: str
    app_dir: Path
    #: MCP tools that may write were refused (``--allow-writes`` lifts it).
    read_only: bool = True


def read_only_env(allow_writes: bool) -> dict[str, str]:
    """The variables that make an emitted app's MCP calls read-only.

    The pipeline side of the gate a skill run gets from its tool
    allowlist: every emitted MCP helper refuses a tool that may write
    (``rote.mcp._runtime_helper.write_refusal``), with the servers the user
    registered read-only passed along, since a Worker cannot read the
    registry. Each declared server carries its registered URL, so the
    declaration covers that endpoint and no other.
    """
    from rote.mcp._runtime_helper import READ_ONLY_ENV, READ_ONLY_SERVERS_ENV
    from rote.mcp.registry import read_only_servers

    if allow_writes:
        return {}
    return {READ_ONLY_ENV: "1", READ_ONLY_SERVERS_ENV: json.dumps(read_only_servers())}


@contextlib.contextmanager
def _environ(values: dict[str, str]) -> Iterator[None]:
    """``values`` in ``os.environ`` for the length of one run, then restored."""
    saved = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run_pipeline(
    target: RunTarget,
    input_payload: dict[str, Any],
    *,
    allow_writes: bool = False,
    signals: dict[str, Any] | None = None,
    gate_order: list[str] | None = None,
    pipeline: Any = None,
    timeout_seconds: float = 600.0,
    interactive: bool = False,
) -> PipelineRunOutcome:
    """Run an emitted pipeline once.

    ``python``/``dbos`` execute in-process via the empirical trial
    runner (DBOS gate payloads are sent up front — notifications
    persist per-topic). ``cloudflare`` runs under ``wrangler dev`` with
    parked-then-send gate delivery, which needs ``gate_order`` (the
    pipeline's gate signal names in topological order). ``inngest``
    runs the emitted serve entrypoint against a managed dev server and
    additionally needs the loaded ``pipeline`` IR for event naming.

    On the four runtimes that park on MCP auth (dbos, dbos-ts, inngest,
    cloudflare) a watcher reports each park, naming the server, the node
    and the command that releases it. ``interactive`` (a terminal on
    both ends) additionally offers the browser login in place; without
    it nothing ever prompts or opens a browser.
    """
    from rote.mcp.parked import watch_factory

    park_watch = watch_factory(interactive)
    runtime = target.runtime
    if runtime is None or runtime not in RUNNABLE_RUNTIMES:
        raise TargetError(
            f"`rote run` cannot orchestrate the `{runtime}` runtime yet — "
            f"{native_run_hint(runtime or '')}"
        )
    gate = read_only_env(allow_writes)
    with _environ(gate):
        run = _dispatch(
            target, input_payload, signals, gate_order, pipeline, timeout_seconds, park_watch, gate
        )
    return PipelineRunOutcome(
        run=run, runtime=runtime, app_dir=target.path, read_only=not allow_writes
    )


def _dispatch(
    target: RunTarget,
    input_payload: dict[str, Any],
    signals: dict[str, Any] | None,
    gate_order: list[str] | None,
    pipeline: Any,
    timeout_seconds: float,
    park_watch: Any,
    gate: dict[str, str],
) -> MeasuredRun:
    """One run on ``target``'s runtime (see :func:`run_pipeline`)."""
    if target.runtime == "cloudflare":
        from rote.runners.cloudflare import run_cloudflare

        run = run_cloudflare(
            target.path,
            input_payload,
            signals=signals or {},
            gate_order=gate_order or sorted(signals or {}),
            timeout_seconds=timeout_seconds,
            park_watch=park_watch,
            # workerd does not see this process's environment.
            worker_vars=gate,
        )
    elif target.runtime == "inngest":
        if pipeline is None:
            raise TargetError(
                "running an inngest app needs the compiled pipeline.yaml for "
                "event naming — keep it at ../../compiled/pipeline.yaml "
                "relative to the app (the `rote compile --out` layout) or "
                "next to it"
            )
        from rote.runners.inngest import run_inngest

        run = run_inngest(
            target.path,
            input_payload,
            pipeline=pipeline,
            signals=signals or {},
            gate_order=gate_order or sorted(signals or {}),
            timeout_seconds=timeout_seconds,
            park_watch=park_watch,
        )
    elif target.runtime == "dbos-ts":
        from rote.runners.dbos_ts import run_dbos_ts

        run = run_dbos_ts(
            target.path,
            input_payload,
            signals=signals or {},
            timeout_seconds=timeout_seconds,
            park_watch=park_watch,
        )
    elif target.runtime == "temporal":
        from rote.runners.temporal import run_temporal

        run = run_temporal(
            target.path,
            input_payload,
            signals=signals or {},
            timeout_seconds=timeout_seconds,
        )
    else:
        run = run_pipeline_trial(
            target.path,
            input_payload,
            signals=signals or None,
            timeout_seconds=timeout_seconds,
            park_watch=park_watch,
        )
    return run
