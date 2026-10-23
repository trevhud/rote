"""Behavioral fidelity runs: the skill as an agent, and the compiled pipeline.

An example opts in with ``examples/<name>/fidelity/cases.yaml``: an output
contract, the fixture MCP servers the skill talks to, and two or three
fixed cases (an input, the payload a person gives at each gate, and the
fixture responses). Two operations use it:

* :func:`record_reference` runs the skill itself, through the baseline
  path (``claude -p``, the same prompt and billing as ``rote baseline``),
  several times per case, and writes the committed ``reference.json``.
  This spends tokens once per change to the cases, not per compile.
* :func:`pipeline_fidelity` emits a compiled pipeline for DBOS, runs it
  on the same cases against the same fixtures (judges hit a real model),
  reads every step's output back from DBOS, and scores the run against
  the reference with :mod:`rote.bench.fidelity_score`.

DBOS, not the plain python runtime, because it persists every step's
output: a pipeline that dies at its last node still shows what it
computed before, and the gate payloads it received.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from rote.bench.fidelity_score import fidelity_summary, score_case, stable_reference
from rote.bench.http_fixtures import HttpFixtureError, api_calls, api_side, load_http
from rote.compiler.api_docs import api_view
from rote.compiler.backend import read_backend
from rote.eval.empirical import GATE_INSTRUCTIONS, gate_answers_server
from rote.eval.gate_server import GATE_SERVER
from rote.ir import NodeKind, Pipeline, load_pipeline
from rote.vendor_specs.store import SpecStore

FIDELITY_DIRNAME = "fidelity"
CASES_FILENAME = "cases.yaml"
REFERENCE_FILENAME = "reference.json"
REFERENCE_SCHEMA_VERSION = 1

#: A pipeline run that outlives this is reported as stopped. Generous for
#: judges on a cold CLI, short enough that a gate nobody can answer (a
#: compiled gate the case has no payload for) does not stall the bench.
PIPELINE_TIMEOUT_SECONDS = 600.0


@dataclass(frozen=True)
class Case:
    id: str
    description: str
    input: dict[str, Any]
    gates: list[dict[str, Any]]
    fixtures: dict[str, dict[str, list[dict[str, Any]]]]
    #: ``--backend api`` fixtures: spec key -> FixtureSet (rote.vendor_specs.fixtures).
    http: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CaseFile:
    example_dir: Path
    output_fields: dict[str, str]
    servers: dict[str, dict[str, Any]]
    cases: list[Case] = field(default_factory=list)

    @property
    def reference_path(self) -> Path:
        return self.example_dir / FIDELITY_DIRNAME / REFERENCE_FILENAME


def load_cases(example_dir: Path) -> CaseFile:
    """Read ``fidelity/cases.yaml``; ``input_file`` paths resolve from it."""
    path = example_dir / FIDELITY_DIRNAME / CASES_FILENAME
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    cases = []
    for c in raw["cases"]:
        if "input_file" in c:
            payload = json.loads((path.parent / c["input_file"]).read_text(encoding="utf-8"))
        else:
            payload = c["input"]
        for patch in c.get("input_patches") or []:
            _apply_patch(payload, patch)
        _require_distinct_gate_payloads(path, c["id"], list(c.get("gates") or []))
        cases.append(
            Case(
                id=c["id"],
                description=c.get("description", ""),
                input=payload,
                gates=list(c.get("gates") or []),
                fixtures=c.get("fixtures") or {},
                http=load_http(c.get("http")),
            )
        )
    return CaseFile(
        example_dir=example_dir,
        output_fields=dict(raw["output_fields"]),
        servers=_fixture_servers_spec(example_dir, raw.get("servers") or {}),
        cases=cases,
    )


def _require_distinct_gate_payloads(path: Path, case_id: str, gates: list[Any]) -> None:
    """Refuse a case whose gates are sent equal payloads.

    The pipeline side finds each gate's park by the message its recv
    received (:func:`reads_before_parks`). Equal payloads cannot tell two
    gates apart, so they fall back to start order, and a compile that
    reorders the gates is then scored against the wrong reference gate.
    """
    seen: dict[str, int] = {}
    for i, payload in enumerate(gates):
        key = _message_key(payload)
        if key in seen:
            raise ValueError(
                f"{path}: case {case_id!r}: gates {seen[key]} and {i} are sent the same "
                "payload, so fidelity cannot tell their parks apart (it matches each "
                "gate by the message it received). Make a value differ in a field the "
                "skill's gate already takes (a different approver, decision or comment). "
                "Add a key only if the skill's gate really has one: gate fit compares "
                "the payload's keys with the compiled gate's, so an extra key lowers "
                "every compile's score."
            )
        seen[key] = i


def _fixture_servers_spec(
    example_dir: Path, declared: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """The fixture servers: each tool from the skill's bundled tools/list.

    ``cases.yaml`` adds only what a fixture needs beyond the snapshot
    (``aliases``, each tool's ``side_effect``), so the schemas a plain
    compile grounds in and the ones the fixtures enforce are one copy.
    The two must name the same servers and tools.
    """
    from rote.tool_schema import BUNDLED_TOOLS_FILENAME, load_bundled_tools

    if not declared:
        return {}
    snapshot = load_bundled_tools(example_dir / "skill")
    where = f"{example_dir / 'skill' / BUNDLED_TOOLS_FILENAME} and {CASES_FILENAME}"
    if sorted(snapshot) != sorted(declared):
        raise ValueError(
            f"{where} name different servers: {sorted(snapshot)} vs {sorted(declared)}"
        )
    servers: dict[str, dict[str, Any]] = {}
    for name, server in declared.items():
        tools = server.get("tools") or {}
        if sorted(tools) != sorted(snapshot[name]):
            raise ValueError(f"{where} list different {name} tools")
        servers[name] = {
            "aliases": list(server.get("aliases") or []),
            "tools": {
                tool: {
                    "description": snapshot[name][tool]["description"],
                    "parameters": snapshot[name][tool]["inputSchema"],
                    "side_effect": bool((tools[tool] or {}).get("side_effect", False)),
                }
                for tool in tools
            },
        }
    return servers


def _apply_patch(payload: dict[str, Any], patch: dict[str, Any]) -> None:
    """``{"set": "submissions.3.currency", "value": "CHF"}``, or ``append``."""
    if "append" in patch:
        target: Any = payload
        for part in patch["append"].split("."):
            target = target[int(part)] if isinstance(target, list) else target[part]
        target.append(patch["value"])
        return
    *head, last = patch["set"].split(".")
    target = payload
    for part in head:
        target = target[int(part)] if isinstance(target, list) else target[part]
    if isinstance(target, list):
        target[int(last)] = patch["value"]
    else:
        target[last] = patch["value"]


# ───────── fixture servers ─────────


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port: int = s.getsockname()[1]
        return port


class FixtureError(RuntimeError):
    """The bench could not stand up its fixture servers.

    An environment problem, not a pipeline result: :func:`pipeline_fidelity`
    lets it through rather than scoring every case 0.
    """


def _wait_for_port(
    port: int, proc: subprocess.Popen[Any], deadline: float, stderr_log: Path
) -> None:
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            tail = stderr_log.read_text(encoding="utf-8", errors="replace").strip()
            raise FixtureError(
                f"fixture MCP server exited with code {proc.returncode}: "
                f"{_last_line(tail) or 'no output'}"
            )
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)
    raise FixtureError(f"fixture MCP server did not listen on {port}")


def _require_fixture_runtime() -> None:
    """Fail before any case runs when the fixture server cannot start.

    A fixture server is a FastMCP server, and the base install's fastmcp
    is the slim client: the server half is the serve extra."""
    from rote.extras import install_hint, is_installed

    if not is_installed("serve"):
        raise FixtureError(
            "fidelity cases with fixture MCP servers need the serve extra "
            f"(fastmcp's server half): {install_hint('serve')}"
        )


@contextlib.contextmanager
def fixture_servers(
    casefile: CaseFile, case: Case, work: Path
) -> Iterator[tuple[dict[str, str], Path]]:
    """Start one fixture server per declared server; yield ``(urls, call_log)``."""
    if casefile.servers:
        _require_fixture_runtime()
    work.mkdir(parents=True, exist_ok=True)
    log = work / "calls.jsonl"
    log.write_text("", encoding="utf-8")
    procs: list[subprocess.Popen[Any]] = []
    urls: dict[str, str] = {}
    try:
        for name, server in casefile.servers.items():
            tools = {}
            for tool_name, tool in server["tools"].items():
                tools[tool_name] = {
                    **tool,
                    "responses": case.fixtures.get(name, {}).get(tool_name, []),
                }
            spec = work / f"fixture-{name}.json"
            spec.write_text(json.dumps({"server": name, "tools": tools}), encoding="utf-8")
            port = _free_port()
            stderr_log = work / f"fixture-{name}.stderr.log"
            with stderr_log.open("w") as stderr:
                proc = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "rote.bench.fixture_mcp",
                        str(spec),
                        str(port),
                        str(log),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=stderr,
                )
            procs.append(proc)
            _wait_for_port(port, proc, time.monotonic() + 30, stderr_log)
            urls[name] = f"http://127.0.0.1:{port}/mcp"
        yield urls, log
    finally:
        for proc in procs:
            proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


def read_calls(log: Path) -> list[dict[str, Any]]:
    if not log.is_file():
        return []
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line]


# ───────── the skill side: reference traces ─────────


def agent_instructions(casefile: CaseFile, case: Case) -> str:
    """The output contract, and how the run reaches the person at each gate."""
    fields = "\n".join(f"- `{k}`: {v}" for k, v in casefile.output_fields.items())
    parts = [
        "This run is recorded as a reference. Your result.json must use exactly "
        "these fields (more are fine):\n"
        f"{fields}\n"
        "- `outcome`: `completed` if you ran the skill to its end, or `stopped` if "
        "the skill told you to stop (say why in `stop_reason`).\n"
        "Write every field above, even when the skill stops first: a field the "
        "run never got to compute is `null`, never a zero or an empty list you "
        "did not derive."
    ]
    if case.gates:
        parts.append(GATE_INSTRUCTIONS)
    if casefile.servers:
        parts.append(
            "Use the connected MCP tools for every external system the skill "
            "names; there is no other way to reach them."
        )
    return "\n\n".join(parts)


def record_reference(
    casefile: CaseFile,
    *,
    trials: int,
    model: str,
    work: Path,
    log: Any = print,
    budget: float | None = None,
) -> dict[str, Any]:
    """Run the skill ``trials`` times per case and build ``reference.json``.

    Trial-major order (every case once, then every case again), so a
    ``budget`` that runs out mid-recording leaves each case with as many
    runs as the money allowed rather than some cases with none.
    """
    from rote.eval.baseline import run_baseline_trial

    skill_dir = casefile.example_dir / "skill"
    runs: dict[str, list[dict[str, Any]]] = {case.id: [] for case in casefile.cases}
    total = 0.0
    for trial in range(1, trials + 1):
        for case in casefile.cases:
            if budget is not None and total >= budget:
                log(f"rote bench: reference {case.id} #{trial}: skipped, budget spent")
                continue
            with fixture_servers(casefile, case, work / case.id / f"t{trial}") as (urls, calls_log):
                servers: dict[str, dict[str, Any]] = {
                    n: {"type": "http", "url": u} for n, u in urls.items()
                }
                tool_ids = [f"mcp__{n}__*" for n in urls]
                if case.gates:
                    # The gate answers come from ask_person, logged into the
                    # same call log, so the reads before each ask are known.
                    servers[GATE_SERVER], gate_tool = gate_answers_server(
                        case.gates, calls_log.parent
                    )
                    tool_ids.append(gate_tool)
                run, _observed, _lines = run_baseline_trial(
                    skill_dir,
                    case.input,
                    model=model,
                    mcp_servers=servers,
                    mcp_tool_ids=tool_ids,
                    output_fields=[*casefile.output_fields, "outcome"],
                    extra_instructions=agent_instructions(casefile, case),
                )
                calls = read_calls(calls_log)
            total += run.cost_usd or 0.0
            log(
                f"rote bench: reference {casefile.example_dir.name}/{case.id} #{trial}: "
                f"{'ok' if run.error is None else run.error[:120]}, {run.turns} turns, "
                f"${run.cost_usd}, {len(calls)} tool call(s) (spent ${total:.2f})"
            )
            runs[case.id].append(
                {
                    "output": run.output,
                    "error": run.error,
                    "calls": [_portable_call(c) for c in calls],
                    "turns": run.turns,
                    "cost_usd": run.cost_usd,
                    "wall_seconds": round(run.wall_seconds, 1),
                }
            )
    cases = {
        case.id: {
            "description": case.description,
            "runs": runs[case.id],
            "stable": stable_reference(
                runs[case.id], contract=[*casefile.output_fields, "outcome"]
            ),
        }
        for case in casefile.cases
    }
    reference = {
        "schema_version": REFERENCE_SCHEMA_VERSION,
        "example": casefile.example_dir.name,
        "model": model,
        "trials": trials,
        "recorded_at": datetime.now(UTC).isoformat(),
        "cost_usd": round(total, 4),
        "cases": cases,
    }
    return reference


def _portable_call(call: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "server",
        "tool",
        "arguments",
        "side_effect",
        "matched",
        "response",
        "invalid_arguments",
        # Set by a recording proxy in front of a live server (never by a
        # fixture): only a live answer can drift (fidelity_score._live).
        "live",
    )
    return {k: call.get(k) for k in keys}


# ───────── the pipeline side ─────────


def _server_key(name: str) -> str:
    return "".join(ch for ch in name.casefold() if ch.isalnum())


def map_servers(pipeline: Pipeline, casefile: CaseFile) -> tuple[dict[str, str], list[str]]:
    """``{compiled server name: fixture server}``, and the names nothing serves.

    The compiler names servers itself (``google_drive``, ``gdrive``); a
    fixture lists the aliases a reasonable compile might use. A server no
    fixture answers to is a fidelity finding, not a bench error.
    """
    by_key: dict[str, str] = {}
    for name, server in casefile.servers.items():
        for alias in [name, *(server.get("aliases") or [])]:
            by_key[_server_key(alias)] = name
    mapped: dict[str, str] = {}
    unknown: list[str] = []
    for required in pipeline.required_mcp_servers:
        target = by_key.get(_server_key(required))
        if target is None:
            unknown.append(required)
        else:
            mapped[required] = target
    return mapped, unknown


def _gate_order(pipeline: Pipeline) -> list[Any]:
    return [n for wave in pipeline.execution_waves() for n in wave if n.kind is NodeKind.HITL_GATE]


def _payload_fit(payload: dict[str, Any], gate: Any) -> float:
    """Share of the case's gate payload keys the compiled gate declares."""
    from rote.bench.score import _payload_keys

    want = set(payload)
    have = _payload_keys(gate)
    if not want and not have:
        return 1.0
    return round(len(want & have) / len(want | have), 4)


def _dbos_objects(app: Path) -> tuple[list[dict[str, Any]], list[Any]]:
    """Every step (and fanned child step) of the app's workflow, with outputs."""
    from dbos import DBOSClient

    from rote._dbos import dbos_system_database_url

    client = DBOSClient(system_database_url=dbos_system_database_url(app))
    try:
        steps: list[dict[str, Any]] = []
        objects: list[Any] = []
        for wf in client.list_workflows():
            for step in client.list_workflow_steps(wf.workflow_id):
                name = str(step["function_name"])
                output = step["output"]
                steps.append(
                    {
                        "workflow": wf.workflow_id,
                        "name": name,
                        "error": bool(step["error"]),
                        "started_at_ms": step.get("started_at_epoch_ms"),
                        # What a recv received identifies which wait it was
                        # (reads_before_parks); other steps' outputs are below.
                        "received": output if name == "DBOS.recv" else None,
                    }
                )
                # A gate's received payload is the case's own input echoed
                # back; crediting it would reward a pipeline for being told
                # the answer. Gates are scored separately (gate fit).
                if output is not None and name not in ("DBOS.sleep", "DBOS.recv"):
                    objects.append(output)
        return steps, objects
    finally:
        client.destroy()


#: What a bench pipeline inherits from the user's environment, by exact name:
#: the process essentials, and the credentials its judges call a model with.
#: Nothing else, so a vendor credential (GOOGLE_APPLICATION_CREDENTIALS, a
#: HUBSPOT_TOKEN, one read under a name built at run time) never reaches the
#: run: the fixture credential api_side sets is the only one there.
BENCH_ENV_NAMES = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "TMPDIR",
        "TEMP",
        "TMP",
        "LANG",
        "TZ",
        "TERM",
        "SYSTEMROOT",
        "VIRTUAL_ENV",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
    }
)
#: And by prefix: locale, the DBOS runtime's own settings, rote's (its
#: per-node inference knobs, isolation paths, MCP endpoints).
BENCH_ENV_PREFIXES = ("LC_", "DBOS_", "ROTE_")


def _message_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def reads_before_parks(
    steps: list[dict[str, Any]], calls: list[dict[str, Any]], gate_payloads: list[Any]
) -> list[list[dict[str, Any]] | None]:
    """For each gate, in gate order, the reads the pipeline made before it parked.

    A DBOS gate parks in ``DBOS.recv``; its step's ``started_at_ms`` is the
    moment it starts to wait, and a read counts when its call-log ``ts`` is
    earlier. Each gate is found by what its recv received: the bench sends
    gate i's payload on gate i's signal, so the recv that got that payload
    is gate i's, whatever else waited around it. DBOS records no topic on a
    step, and the message stands in for it: an auth park (a recv on
    ``rote:auth:<server>``, released with another message) or a gate the
    skill does not have cannot shift the comparison, as matching the i-th
    recv by start time would. Two gates sent equal payloads take their
    recvs in start order. A gate whose recv never ran is None.
    """
    recvs = sorted(
        (s for s in steps if s["name"] == "DBOS.recv" and s.get("started_at_ms") is not None),
        key=lambda s: s["started_at_ms"],
    )
    reads = [c for c in calls if not c.get("side_effect") and c.get("server") != GATE_SERVER]
    used: set[int] = set()
    out: list[list[dict[str, Any]] | None] = []
    for payload in gate_payloads:
        want = _message_key(payload)
        park = next(
            (
                (k, s["started_at_ms"])
                for k, s in enumerate(recvs)
                if k not in used and _message_key(s.get("received")) == want
            ),
            None,
        )
        if park is None:
            out.append(None)
            continue
        used.add(park[0])
        out.append(
            [
                _portable_call(c)
                for c in reads
                if c.get("ts") is not None and c["ts"] * 1000 < park[1]
            ]
        )
    return out


def bench_environment() -> dict[str, str]:
    """The allowlisted part of ``os.environ`` a bench pipeline runs with."""
    return {
        k: v
        for k, v in os.environ.items()
        if k in BENCH_ENV_NAMES or k.startswith(BENCH_ENV_PREFIXES)
    }


def run_pipeline_case(
    compiled_dir: Path,
    casefile: CaseFile,
    case: Case,
    work: Path,
    spec_store: SpecStore | None = None,
) -> dict[str, Any]:
    """Emit ``compiled_dir`` for DBOS, run one case, return its trace."""
    from rote.adapters import get_adapter
    from rote.eval.empirical import run_pipeline_trial

    pipeline = load_pipeline(compiled_dir / "pipeline.yaml")
    backend = read_backend(compiled_dir) or "mcp"
    app = work / "app"
    if app.exists():
        shutil.rmtree(app)
    get_adapter("dbos", external_backend=backend, extracted_source_dir=compiled_dir).emit(
        pipeline, app
    )

    gates = _gate_order(pipeline)
    signals: dict[str, Any] = {}
    gate_fit = []
    for i, gate in enumerate(gates):
        payload = case.gates[i] if i < len(case.gates) else {}
        assert gate.signal is not None
        signals[gate.signal] = payload
        gate_fit.append(
            {"gate": gate.id, "signal": gate.signal, "fit": _payload_fit(payload, gate)}
        )
    for extra in case.gates[len(gates) :]:
        gate_fit.append({"gate": None, "signal": None, "fit": 0.0, "unused_payload": extra})

    # On the api backend no binding is called: the vendor requests go to the
    # spec-checked fake, and only an agent loop's servers still need MCP.
    called = api_view(pipeline) if backend == "api" else pipeline
    mapped, unknown = map_servers(called, casefile)
    if backend == "api" and not case.http:
        raise HttpFixtureError(
            f"case {case.id!r} has no http: fixtures, so an --backend api pipeline has "
            f"nothing to call (fidelity/cases.yaml)"
        )
    with contextlib.ExitStack() as stack:
        urls, calls_log = stack.enter_context(fixture_servers(casefile, case, work / "fixtures"))
        http_env: dict[str, str] = {}
        python: str | None = None
        http_log: Path | None = None
        if backend == "api":
            http_env, python, http_log = stack.enter_context(
                api_side(app, case.http, work / "http", spec_store, compiled_dir)
            )
        env = {f"ROTE_MCP_{server.upper()}_URL": urls[target] for server, target in mapped.items()}
        run = run_pipeline_trial(
            app,
            case.input,
            signals=signals or None,
            timeout_seconds=PIPELINE_TIMEOUT_SECONDS,
            extra_env={**env, **http_env},
            base_env=bench_environment(),
            python_executable=python,
        )
        calls = read_calls(calls_log) + (api_calls(http_log) if http_log else [])
    steps, objects = _dbos_objects(app)
    if run.output is not None:
        objects.append(run.output)
    return {
        "completed": run.error is None,
        "error": _last_line(run.error),
        "failure": None if run.error is None else failure_site(app, pipeline, case.input),
        "objects": objects,
        "steps": steps,
        "calls": [_portable_call(c) for c in calls],
        "reads_before_gate": reads_before_parks(steps, calls, [signals[g.signal] for g in gates]),
        "gate_fit": gate_fit,
        "unknown_servers": unknown,
        "backend": backend,
        "judge_usage": list(run.judge_usage),
        "wall_seconds": round(run.wall_seconds, 1),
    }


#: Replays one node's implementation in a fresh interpreter (so the
#: emitted app's ``extracted`` package never collides with another app's)
#: and prints where it raised. Reads ``[impl, kwargs]`` as JSON on stdin.
_REPLAY = """
import importlib, json, sys, traceback
impl, kwargs = json.load(sys.stdin)
module, symbol = impl.split(":")
fn = getattr(importlib.import_module(module.removesuffix(".py").replace("/", ".")), symbol)
try:
    fn(**kwargs)
except Exception as e:
    frames = [f for f in traceback.extract_tb(e.__traceback__) if "/extracted/" in f.filename]
    site = frames[-1] if frames else None
    print(json.dumps({
        "exception": f"{type(e).__name__}: {e}"[:300],
        "at": None if site is None else (
            "extracted/" + site.filename.split("/extracted/")[1] + f":{site.lineno}"
        ),
        "code": None if site is None else site.line,
    }))
else:
    print(json.dumps({"exception": None}))
"""


def failure_site(
    app: Path, pipeline: Pipeline, case_input: dict[str, Any]
) -> dict[str, Any] | None:
    """Which node a failed run died in, and the line of its code that raised.

    The run's own error is a DBOS re-raise with the traceback gone (an
    exception does not keep its frames across a durable step). The failing
    node is the step DBOS recorded as an error; its inputs are rebuilt from
    the outputs DBOS recorded for its upstream nodes, the gate payloads it
    received, and the case input, and its implementation is called once
    more to see where it raises. ``None`` when nothing can be replayed
    (the run never started, or the failing node has no ``impl``).
    """
    from dbos import DBOSClient

    from rote._dbos import dbos_system_database_url
    from rote.ir import fan_out_element_param, parse_input_ref

    try:
        client = DBOSClient(system_database_url=dbos_system_database_url(app))
    except Exception:  # noqa: BLE001 (no database: the run never started)
        return None
    by_id = {n.id: n for n in pipeline.nodes}
    outputs: dict[str, Any] = {}
    failed: str | None = None
    failed_error: Any = None
    gate_ids = iter(g.id for g in _gate_order(pipeline))
    try:
        for wf in client.list_workflows():
            for step in client.list_workflow_steps(wf.workflow_id):
                name = str(step["function_name"])
                if name == "DBOS.recv":
                    gate = next(gate_ids, None)
                    if gate is not None:
                        outputs[gate] = step["output"]
                elif name in by_id:
                    if step["error"] is not None:
                        if failed is None:
                            failed, failed_error = name, step["error"]
                    elif by_id[name].fan_out:
                        outputs.setdefault(name, []).append(step["output"])
                    else:
                        outputs[name] = step["output"]
    except Exception:  # noqa: BLE001
        return None
    finally:
        client.destroy()
    if failed is None:
        return None
    node = by_id[failed]
    recorded = (
        None
        if failed_error is None
        else _last_line(f"{type(failed_error).__name__}: {failed_error}")
    )
    if node.mcp is not None:
        # The default backend runs an mcp-bound node as the tool call, not
        # its impl (often a stub): replaying the impl would blame code that
        # never ran. The recorded error is the call's own.
        return {
            "node": failed,
            "exception": recorded,
            "at": f"mcp {node.mcp.server}.{node.mcp.tool}",
            "code": None,
        }
    if not node.impl:
        return {"node": failed, "exception": recorded, "at": None, "code": None}
    kwargs: dict[str, Any] = dict(node.constants or {})
    for param, ref in (node.inputs or {}).items():
        parsed = parse_input_ref(ref)
        source = case_input if parsed.node_id is None else outputs.get(parsed.node_id)
        if parsed.field is not None and isinstance(source, dict):
            source = source.get(parsed.field)
        kwargs[param] = source
    if node.fan_out:
        param = fan_out_element_param(node, pipeline)
        items = kwargs.get(param) or []
        kwargs[param] = items[0] if items else None
    proc = subprocess.run(
        [sys.executable, "-c", _REPLAY],
        input=json.dumps([node.impl, kwargs], default=str),
        cwd=app,
        capture_output=True,
        text=True,
        timeout=60,
    )
    try:
        site: dict[str, Any] = json.loads(proc.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        site = {"exception": _last_line(proc.stderr)}
    return {"node": failed, **site}


def _last_line(error: str | None) -> str | None:
    """The line that says what failed: a traceback's final line, trimmed.

    The rest of a traceback is frames inside the emitted app and DBOS, the
    same on every run and full of local paths.
    """
    if not error:
        return None
    lines = [ln.strip() for ln in error.strip().splitlines() if ln.strip()]
    return lines[-1][:500] if lines else None


def load_reference(casefile: CaseFile) -> dict[str, Any]:
    path = casefile.reference_path
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist: record it once with "
            f"`rote bench --record-reference --skills {casefile.example_dir.name}`"
        )
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def has_cases(example_dir: Path) -> bool:
    return (example_dir / FIDELITY_DIRNAME / CASES_FILENAME).is_file()


def pipeline_fidelity(
    compiled_dir: Path,
    example_dir: Path,
    work: Path | None = None,
    spec_store: SpecStore | None = None,
) -> dict[str, Any]:
    """Score one compiled pipeline on every case of its example.

    Never raises on a bad pipeline: an emit that fails or a run that dies
    is a case scored on what it did produce.
    """
    from rote.eval.empirical import judge_usage_cost_usd
    from rote.eval.pricing import PricingError, fetch_catalog

    casefile = load_cases(example_dir)
    reference = load_reference(casefile)
    owned = work is None
    root = Path(tempfile.mkdtemp(prefix="rote-fidelity-")) if work is None else work
    cases: dict[str, Any] = {}
    usage: list[dict[str, Any]] = []
    try:
        for case in casefile.cases:
            # Re-derived from the recorded runs rather than read from the
            # stored summary, so a change to the stability rule applies to
            # existing references without paying to record them again.
            ref = stable_reference(
                reference["cases"][case.id]["runs"],
                contract=[*casefile.output_fields, "outcome"],
            )
            try:
                trace = run_pipeline_case(compiled_dir, casefile, case, root / case.id, spec_store)
            except (FixtureError, HttpFixtureError):
                raise  # the bench's environment, not the pipeline: never a score
            except Exception as e:  # noqa: BLE001 (an unrunnable pipeline is a result)
                trace = {
                    "completed": False,
                    "error": f"{type(e).__name__}: {e}"[:1500],
                    "objects": [],
                    "calls": [],
                }
            usage += trace.get("judge_usage", [])
            scored = score_case(ref, trace)
            scored["run_error"] = trace.get("error")
            scored["failure"] = trace.get("failure")
            # Calls the fixture refused, with why: a retried tool error is
            # otherwise only "exceeded its maximum of N retries".
            scored["refused_calls"] = [
                {
                    "call": f"{c['server']}.{c['tool']}",
                    "arguments": c["arguments"],
                    "why": c.get("invalid_arguments") or "no fixture response for these arguments",
                }
                for c in trace.get("calls", [])
                if not c.get("matched")
            ]
            scored["unknown_servers"] = trace.get("unknown_servers", [])
            scored["steps"] = len(trace.get("steps", []))
            scored["gate_fit"] = trace.get("gate_fit", [])
            # How much of the skill's behavior this case can check at all: a
            # high score over a thin reference is a weak claim.
            scored["reference"] = {
                "successful_runs": ref["successful_runs"],
                "insufficient_runs": ref["insufficient_runs"],
                **ref["coverage"],
                "write_fields": ref["write_fields"],
                "write_fields_dropped": ref["write_fields_dropped"],
            }
            cases[case.id] = scored
    finally:
        if owned:
            shutil.rmtree(root, ignore_errors=True)
    try:
        catalog = fetch_catalog(provider="anthropic")
    except PricingError:
        catalog = None
    cost, unpriced = judge_usage_cost_usd(usage, catalog)
    return {
        **fidelity_summary(cases),
        "cost_usd": round(cost, 4),
        "unpriced_models": sorted(unpriced),
        "judge_calls": len(usage),
        "reference_recorded_at": reference.get("recorded_at"),
        "reference_coverage": _mean_rate(cases),
        "reference_low_cases": sorted(
            cid
            for cid, c in cases.items()
            if c["reference"]["low"] or c["reference"]["insufficient_runs"]
        ),
    }


def _mean_rate(cases: dict[str, Any]) -> float | None:
    rates = [c["reference"]["rate"] for c in cases.values() if c["reference"]["rate"] is not None]
    return round(sum(rates) / len(rates), 4) if rates else None
