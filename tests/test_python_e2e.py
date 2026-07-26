"""End-to-end test: emit → overlay → run the emitted script as a subprocess.

The python adapter has the easiest e2e bar in the repo — its "runtime"
is just the interpreter — so this test is *not* marked slow: no engine
to launch, no npm install, just one ``python main.py`` subprocess.

What it proves, empirically:

1. The emitted script exits 0 with a JSON result on stdout.
2. Wave order is respected (the fan-in loop starts only after both
   parallel wave-1 nodes finished).
3. Payloads thread through the DAG: the entry input reaches the entry
   nodes, node outputs chain downstream (whole-output and field
   references), and the fan-in exit node receives values from three
   different sources.
4. The emitted retry loop actually retries: a stub seeded to fail twice
   is called exactly three times and the run still succeeds.
5. Every import in the emitted+overlaid directory resolves from the
   stdlib or the emitted files themselves — the script is genuinely
   dependency-free once the judge is mocked.

Judge handling (decision, documented): the generated
``signatures/grade.py`` calls the real Anthropic SDK, so the overlay
replaces it with a recorder mock exposing the same
class/``forward``/``model_dump`` surface the emitted ``main.py``
imports (same technique as the DBOS e2e). The generated module itself
is still validated by the adapter tests (AST parse + MCP scan); what
this test exercises is orchestration, not vendor I/O.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from rote.adapters.python import PythonAdapter, PythonAdapterConfig
from tests._python_fixture import build_gateless_pipeline

BRIEF = {"topic": "  Rare Disease  ", "depth": 2}

# The recorder appends one JSON line per node call so the test can
# assert ordering and the exact payload each node received.
_RECORDER_PRELUDE = """\
import json
import time

_RECORD_PATH = {record_path!r}


def _record(name, payload):
    with open(_RECORD_PATH, "a") as f:
        f.write(
            json.dumps({{"node": name, "t": time.monotonic(), "payload": payload}})
            + "\\n"
        )
"""

_OVERLAYS = {
    # external_call with retry: fail the first two calls, succeed on the
    # third — exactly what retry max=2 must absorb.
    "extracted/profile.py": """\
_FAILURES_BEFORE_SUCCESS = 2
_calls = {"count": 0}


def fetch_profile(**payload):
    _calls["count"] += 1
    _record("fetch_profile", payload)
    if _calls["count"] <= _FAILURES_BEFORE_SUCCESS:
        raise RuntimeError("transient upstream error (seeded by the e2e test)")
    return {"profile": {"id": "p1"}}
""",
    "extracted/brief.py": """\
def normalize_brief(**payload):
    _record("normalize_brief", payload)
    return {"topic": payload["topic"].strip().lower()}


def score_item(**payload):
    _record("score_item", payload)
    return {"score": 1}
""",
    # The agent loop is no longer a stub in extracted/ — it calls
    # run_agent_loop in the shared inference helper, which would spawn
    # `claude -p` or hit a vendor endpoint. Overlay the helper for the
    # same reason the judge is overlaid: this test is about
    # orchestration, not vendor I/O.
    "signatures/_rote_inference.py": """\
def run_agent_loop(**kwargs):
    # The node payload reaches the agent as the JSON `task`; record it
    # decoded so the data-flow assertions read the same as every other
    # node. Also record the loop_body sub-nodes actually bound as tools.
    _record("research_loop", json.loads(kwargs["task"]))
    _record("research_loop:tools", sorted(kwargs.get("local_tools") or {}))
    return {"findings": ["f1", "f2"]}
""",
    "extracted/report.py": """\
def build_report(**payload):
    _record("final_report", payload)
    return {"report": payload}
""",
    # Recorder mock with the same surface main.py imports from the
    # generated signature module (see module docstring).
    "signatures/grade.py": """\
class GradeInput:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _Result:
    # Mirrors pydantic's model_dump signature: the emitted call sites pass
    # by_alias=True so aliased schema keys survive into the data flow.
    def model_dump(self, *, by_alias=False):
        return {"grade": 9, "rationale": "solid"}


class Grade:
    def forward(self, inputs):
        _record("grade", inputs.kwargs)
        return _Result()
""",
}


@pytest.fixture(scope="module")
def script_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, str]:
    """Emit + overlay + run once for the whole module.

    Returns (out_dir, record_path, stdout).
    """
    out = tmp_path_factory.mktemp("python-e2e")
    record_path = out / "record.jsonl"

    pipeline = build_gateless_pipeline()
    # Tiny backoff base so the two seeded failures cost ~30ms, not ~3s.
    PythonAdapter(PythonAdapterConfig(retry_base_delay_seconds=0.01)).emit(pipeline, out)

    prelude = _RECORDER_PRELUDE.format(record_path=str(record_path))
    for rel, body in _OVERLAYS.items():
        (out / rel).write_text(prelude + "\n\n" + body, encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(out / "main.py"), json.dumps(BRIEF)],
        cwd=out,
        capture_output=True,
        text=True,
        timeout=60,
        # No PYTHONPATH leakage: imports must resolve from the stdlib and
        # the script's own directory alone.
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
    )
    assert proc.returncode == 0, f"emitted script failed:\n{proc.stderr}"
    return out, record_path, proc.stdout


def _records(record_path: Path) -> list[dict]:
    return [json.loads(line) for line in record_path.read_text().splitlines()]


# ───────── The end-to-end assertions ─────────


def test_script_exits_zero_with_threaded_result(script_run: tuple[Path, Path, str]) -> None:
    """Exit 0 (asserted in the fixture) and the exit node's result on
    stdout carries values that fan in from three different sources."""
    _, _, stdout = script_run
    result = json.loads(stdout)
    assert result == {
        "final_report": {
            "report": {
                "grade": 9,  # from the mocked judge, via grade.output.grade
                "profile": {"id": "p1"},  # from fetch_profile.output.profile
                "topic": BRIEF["topic"],  # from pipeline.input.topic
            }
        }
    }


def test_wave_order_respected(script_run: tuple[Path, Path, str]) -> None:
    """The fan-in loop starts strictly after both parallel wave-1 nodes
    completed, and the judge/report chain follows in order."""
    _, record_path, _ = script_run
    records = _records(record_path)
    by_node: dict[str, float] = {}
    for r in records:
        by_node[r["node"]] = r["t"]  # keep the *last* call per node

    assert by_node["research_loop"] > by_node["fetch_profile"]
    assert by_node["research_loop"] > by_node["normalize_brief"]
    assert by_node["grade"] > by_node["research_loop"]
    assert by_node["final_report"] > by_node["grade"]


def test_retry_loop_actually_retries(script_run: tuple[Path, Path, str]) -> None:
    """fetch_profile was seeded to fail twice: the emitted retry loop
    must call it exactly 3 times (1 initial + 2 retries) while every
    other node runs exactly once."""
    _, record_path, _ = script_run
    calls: dict[str, int] = {}
    for r in _records(record_path):
        calls[r["node"]] = calls.get(r["node"], 0) + 1
    assert calls["fetch_profile"] == 3
    assert calls["normalize_brief"] == 1
    assert calls["research_loop"] == 1
    assert calls["grade"] == 1
    assert calls["final_report"] == 1
    # Loop-body sub-node: never dispatched by run_pipeline.
    assert "score_item" not in calls


def test_payloads_threaded_through_dag(script_run: tuple[Path, Path, str]) -> None:
    """Every node received exactly the payload its ``inputs:`` declare."""
    _, record_path, _ = script_run
    payloads = {r["node"]: r["payload"] for r in _records(record_path)}

    # Entry nodes: pipeline input field vs. whole pipeline input.
    assert payloads["normalize_brief"] == {"topic": BRIEF["topic"]}
    assert payloads["fetch_profile"] == {"brief": BRIEF}

    # Fan-in loop: whole upstream output + upstream field + input field.
    assert payloads["research_loop"] == {
        "profile": {"profile": {"id": "p1"}},
        "topic": "rare disease",
        "depth": 2,
    }

    # Judge: upstream field reference.
    assert payloads["grade"] == {"findings": ["f1", "f2"]}

    # Exit node: three-source fan-in.
    assert payloads["final_report"] == {
        "grade": 9,
        "profile": {"id": "p1"},
        "topic": BRIEF["topic"],
    }


def test_all_imports_resolve_from_stdlib_and_emitted_files(
    script_run: tuple[Path, Path, str],
) -> None:
    """Dependency-light means it: after the judge overlay, every import
    anywhere in the directory (module-level or lazy) is stdlib or one of
    the emitted local packages.

    The two verbatim shared helpers are exempt: ``_rote_inference.py``
    *is* the vendor call, so its lazy ``import anthropic`` / ``import
    openai`` are the dependency the overlay exists to replace. Nothing
    imports it once the judges are stubbed out, so it never executes —
    but every other file must stay clean whether stubbed or not.
    """
    out, _, _ = script_run
    allowed = set(sys.stdlib_module_names) | {"extracted", "signatures"}
    for path in out.rglob("*.py"):
        if path.name in {"_rote_inference.py", "_rote_mcp.py"}:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            roots: list[str] = []
            if isinstance(node, ast.Import):
                roots = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                roots = [node.module.split(".")[0]]
            for root in roots:
                assert root in allowed, f"{path.name} imports non-stdlib module {root!r}"


def test_loop_body_sub_nodes_are_bound_as_agent_tools(
    script_run: tuple[Path, Path, str],
) -> None:
    """The agent_loop drives its loop_body sub-nodes, as the IR says.

    This is the half of "agent loops are no longer stubs" that the
    emission tests can't prove: the sub-node arrived at the runtime as a
    live callable the agent can invoke, not as prose in a docstring.
    """
    _, record_path, _ = script_run
    bound = [r["payload"] for r in _records(record_path) if r["node"] == "research_loop:tools"]
    assert bound == [["score_item"]]
