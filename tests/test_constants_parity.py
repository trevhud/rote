"""Cross-runtime contract: each node's step receives its OWN constants.

Several nodes may share one ``impl`` and differ only in their IR
``constants``: ops-report's three fetch nodes all call
``google_drive.fetch_file_content`` with a different ``file_id``. The
Python runtimes used to pass none of them, so every fetch got ``{}`` and
no implementation could tell which file to fetch. The TS runtimes emit one
stub per node, so an implementer can hard-code each node's values there.
A shared Python function has no such escape: the constants have to travel
in the payload.

The fixture is deliberately non-degenerate. Two nodes share one function
with different values for the same key, one of them carries an extra key
the other lacks, and a fan_out node needs its constant on every element.
A runtime that passed the first node's constants to both, or dropped the
extra key, or dropped constants on the fanned path, fails here. Each
runtime runs the emitted app for real; nothing is asserted on source text.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from rote.adapters import get_adapter
from rote.eval.empirical import run_pipeline_trial
from rote.ir import Pipeline

_PIPELINE = Pipeline.model_validate(
    {
        "name": "shared-impl",
        "version": "0.1.0",
        "input": {"type": "In", "required": ["region"], "optional": []},
        "nodes": [
            {
                "id": "fetch_containers",
                "kind": "external_call",
                "description": "Fetch the containers sheet.",
                "impl": "extracted/drive.py:fetch_file_content",
                "constants": {"file_id": "FILE-CONTAINERS"},
            },
            {
                "id": "fetch_dwell",
                "kind": "external_call",
                "description": "Fetch one tab of the dwell sheet.",
                "impl": "extracted/drive.py:fetch_file_content",
                "constants": {"file_id": "FILE-DWELL", "tab_name": "Dwell Log"},
                "inputs": {"region": "pipeline.input.region"},
            },
            {
                "id": "list_docks",
                "kind": "pure_function",
                "description": "List the docks.",
                "impl": "extracted/docks.py:list_docks",
            },
            {
                "id": "label_dock",
                "kind": "pure_function",
                "description": "Label one dock.",
                "impl": "extracted/docks.py:label_dock",
                "fan_out": True,
                "constants": {"prefix": "DOCK"},
                "inputs": {"dock": "list_docks.output.docks"},
            },
        ],
        "edges": [{"from": "list_docks", "to": "label_dock", "fan_out": True}],
        "entry_nodes": ["fetch_containers", "fetch_dwell", "list_docks"],
        "exit_nodes": ["fetch_containers", "fetch_dwell", "label_dock"],
    }
)

#: The agent-written modules: they echo what they were called with.
_DRIVE = "def fetch_file_content(**payload):\n    return {'called_with': payload}\n"
_DOCKS = (
    "def list_docks():\n"
    "    return {'docks': ['d1', 'd2', 'd3']}\n"
    "\n"
    "\n"
    "def label_dock(dock, prefix):\n"
    "    return {'label': f'{prefix}-{dock}'}\n"
)

_EXPECTED = {
    "fetch_containers": {"called_with": {"file_id": "FILE-CONTAINERS"}},
    "fetch_dwell": {
        "called_with": {"file_id": "FILE-DWELL", "tab_name": "Dwell Log", "region": "west"}
    },
    "label_dock": [{"label": "DOCK-d1"}, {"label": "DOCK-d2"}, {"label": "DOCK-d3"}],
}


def _emit(runtime: str, tmp_path: Path) -> Path:
    source = tmp_path / "compiled"
    (source / "extracted").mkdir(parents=True)
    (source / "extracted" / "drive.py").write_text(_DRIVE, encoding="utf-8")
    (source / "extracted" / "docks.py").write_text(_DOCKS, encoding="utf-8")
    out = tmp_path / runtime
    get_adapter(runtime, extracted_source_dir=source).emit(_PIPELINE, out)
    return out


@pytest.mark.parametrize("runtime", ["dbos", "python", "temporal"])
def test_shared_stub_documents_every_callers_constants(runtime: str, tmp_path: Path) -> None:
    """The stub for a shared impl listed only the first node's constants,
    so whoever filled it in never saw the second file id or the extra key."""
    out = tmp_path / runtime
    get_adapter(runtime).emit(_PIPELINE, out)
    stub = (out / "extracted" / "drive.py").read_text(encoding="utf-8")

    assert "from node fetch_containers:" in stub
    assert "from node fetch_dwell:" in stub
    assert "'FILE-CONTAINERS'" in stub
    assert "'FILE-DWELL'" in stub
    assert "tab_name = 'Dwell Log'" in stub


def test_a_constant_cannot_share_a_name_with_an_input() -> None:
    """One payload carries both, so a shared name would silently drop one."""
    data = _PIPELINE.model_dump(mode="json", by_alias=True, exclude_none=True)
    data["nodes"][1]["constants"]["region"] = "east"
    with pytest.raises(ValueError, match="region is both a constant and an input"):
        Pipeline.model_validate(data)


def test_python_runtime_passes_each_nodes_own_constants(tmp_path: Path) -> None:
    run = run_pipeline_trial(_emit("python", tmp_path), {"region": "west"}, timeout_seconds=60)
    assert run.error is None, run.error
    assert run.output == _EXPECTED


def test_temporal_runtime_passes_each_nodes_own_constants(tmp_path: Path) -> None:
    from rote.runners.temporal import run_temporal

    app = _emit("temporal", tmp_path)
    try:
        run = run_temporal(app, {"region": "west"}, signals={}, timeout_seconds=60)
    finally:
        sys.path.remove(str(app))
        for name in [
            m
            for m in sys.modules
            if m.split(".")[0] in ("workflow", "activities", "extracted", "signatures")
        ]:
            del sys.modules[name]
    assert run.error is None, run.error
    assert run.output == _EXPECTED


@pytest.mark.slow
def test_dbos_runtime_passes_each_nodes_own_constants(tmp_path: Path) -> None:
    """Slow: launches a real DBOS runtime (system-DB migrations, queues)."""
    run = run_pipeline_trial(_emit("dbos", tmp_path), {"region": "west"}, timeout_seconds=120)
    assert run.error is None, run.error
    assert run.output == _EXPECTED
