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
from rote.ir import NodeKind, Pipeline, load_pipeline

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
    gate_instructions: str | None
    fixtures: dict[str, dict[str, list[dict[str, Any]]]]


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
        cases.append(
            Case(
                id=c["id"],
                description=c.get("description", ""),
                input=payload,
                gates=list(c.get("gates") or []),
                gate_instructions=c.get("gate_instructions"),
                fixtures=c.get("fixtures") or {},
            )
        )
    return CaseFile(
        example_dir=example_dir,
        output_fields=dict(raw["output_fields"]),
        servers=dict(raw.get("servers") or {}),
        cases=cases,
    )


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


def _wait_for_port(port: int, proc: subprocess.Popen[Any], deadline: float) -> None:
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"fixture MCP server exited with code {proc.returncode}")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)
    raise RuntimeError(f"fixture MCP server did not listen on {port}")


@contextlib.contextmanager
def fixture_servers(
    casefile: CaseFile, case: Case, work: Path
) -> Iterator[tuple[dict[str, str], Path]]:
    """Start one fixture server per declared server; yield ``(urls, call_log)``."""
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
            proc = subprocess.Popen(
                [sys.executable, "-m", "rote.bench.fixture_mcp", str(spec), str(port), str(log)],
                stdout=subprocess.DEVNULL,
                stderr=(work / f"fixture-{name}.stderr.log").open("w"),
            )
            procs.append(proc)
            _wait_for_port(port, proc, time.monotonic() + 30)
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
    """The output contract, and the answers a person gives at each gate."""
    fields = "\n".join(f"- `{k}`: {v}" for k, v in casefile.output_fields.items())
    parts = [
        "This run is recorded as a reference. Your result.json must use exactly "
        "these fields (more are fine):\n"
        f"{fields}\n"
        "- `outcome`: `completed` if you ran the skill to its end, or `stopped` if "
        "the skill told you to stop (say why in `stop_reason`)."
    ]
    if case.gates:
        answers = case.gate_instructions or "\n".join(
            f"{i + 1}. {json.dumps(g)}" for i, g in enumerate(case.gates)
        )
        parts.append(
            "When the skill needs a person's input or approval, do not wait and do "
            "not ask: these are the answers you get, in the order the skill asks:\n"
            f"{answers}"
        )
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
                run, _observed, _lines = run_baseline_trial(
                    skill_dir,
                    case.input,
                    model=model,
                    mcp_servers={n: {"type": "http", "url": u} for n, u in urls.items()},
                    mcp_tool_ids=[f"mcp__{n}__*" for n in urls],
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
            "stable": stable_reference(runs[case.id]),
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
    keys = ("server", "tool", "arguments", "side_effect", "matched", "response")
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
                    {"workflow": wf.workflow_id, "name": name, "error": bool(step["error"])}
                )
                # A gate's received payload is the case's own input echoed
                # back; crediting it would reward a pipeline for being told
                # the answer. Gates are scored separately (gate fit).
                if output is not None and name not in ("DBOS.sleep", "DBOS.recv"):
                    objects.append(output)
        return steps, objects
    finally:
        client.destroy()


def run_pipeline_case(
    compiled_dir: Path, casefile: CaseFile, case: Case, work: Path
) -> dict[str, Any]:
    """Emit ``compiled_dir`` for DBOS, run one case, return its trace."""
    from rote.adapters import get_adapter
    from rote.eval.empirical import run_pipeline_trial

    pipeline = load_pipeline(compiled_dir / "pipeline.yaml")
    app = work / "app"
    if app.exists():
        shutil.rmtree(app)
    get_adapter("dbos", extracted_source_dir=compiled_dir).emit(pipeline, app)

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

    mapped, unknown = map_servers(pipeline, casefile)
    with fixture_servers(casefile, case, work / "fixtures") as (urls, calls_log):
        env = {f"ROTE_MCP_{server.upper()}_URL": urls[target] for server, target in mapped.items()}
        run = run_pipeline_trial(
            app,
            case.input,
            signals=signals or None,
            timeout_seconds=PIPELINE_TIMEOUT_SECONDS,
            extra_env=env,
        )
        calls = read_calls(calls_log)
    steps, objects = _dbos_objects(app)
    if run.output is not None:
        objects.append(run.output)
    return {
        "completed": run.error is None,
        "error": (run.error or "")[:1500] or None,
        "objects": objects,
        "steps": steps,
        "calls": [_portable_call(c) for c in calls],
        "gate_fit": gate_fit,
        "unknown_servers": unknown,
        "judge_usage": list(run.judge_usage),
        "wall_seconds": round(run.wall_seconds, 1),
    }


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
    compiled_dir: Path, example_dir: Path, work: Path | None = None
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
            ref = reference["cases"][case.id]["stable"]
            try:
                trace = run_pipeline_case(compiled_dir, casefile, case, root / case.id)
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
            scored["unknown_servers"] = trace.get("unknown_servers", [])
            scored["steps"] = len(trace.get("steps", []))
            scored["gate_fit"] = trace.get("gate_fit", [])
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
    }
