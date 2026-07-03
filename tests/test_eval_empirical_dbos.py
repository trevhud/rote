"""Slow test: the empirical pipeline runner drives a real gated DBOS app.

The static harness tests cover the plain python-adapter path; this one
proves the DBOS branch end-to-end: spawn the emitted one-shot app,
harvest the workflow id from stderr, deliver the gate's resume payload
cross-process via ``DBOSClient`` over SQLite, and verify the payload
flowed into the exit node's result. Marked slow: launches a real DBOS
runtime (system-DB migrations, queue workers).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rote.adapters import get_adapter
from rote.eval.empirical import run_pipeline_trial
from rote.ir import Pipeline

pytest.importorskip("dbos", reason="dbos not installed (pip install rote[dbos])")

pytestmark = pytest.mark.slow

GATED_PIPELINE = Pipeline.model_validate(
    {
        "name": "toy-gated",
        "description": "Sum, wait for approval, finalize",
        "input": {
            "type": "Numbers",
            "required": ["numbers"],
            "input_schema": {
                "type": "object",
                "properties": {"numbers": {"type": "array", "items": {"type": "number"}}},
                "required": ["numbers"],
            },
        },
        "nodes": [
            {
                "id": "prep",
                "kind": "pure_function",
                "description": "Sum the numbers",
                "impl": "extracted/toyg.py:prep",
                "inputs": {"numbers": "pipeline.input.numbers"},
                "output": {"total": "int"},
            },
            {
                "id": "approve",
                "kind": "hitl_gate",
                "description": "Human approves the total",
                "signal": "approval_given",
                "timeout": "1h",
                "output": {"approved": "bool", "note": "str"},
            },
            {
                "id": "finalize",
                "kind": "pure_function",
                "description": "Combine total and approval",
                "impl": "extracted/toyg.py:finalize",
                "inputs": {
                    "total": "prep.output.total",
                    "note": "approve.output.note",
                },
                "output": {"done": "str"},
            },
        ],
        "edges": [
            {"from": "prep", "to": "approve"},
            {"from": "approve", "to": "finalize", "on_signal": "approval_given"},
        ],
        "entry_nodes": ["prep"],
        "exit_nodes": ["finalize"],
    }
)

WORKING_IMPLS = """\
def prep(**payload):
    return {"total": sum(payload["numbers"])}


def finalize(**payload):
    return {"done": f"{payload['total']}:{payload['note']}"}
"""


def test_dbos_trial_delivers_gate_signal_and_completes(tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    get_adapter("dbos").emit(GATED_PIPELINE, app_dir)
    (app_dir / "extracted" / "toyg.py").write_text(WORKING_IMPLS, encoding="utf-8")

    run = run_pipeline_trial(
        app_dir,
        {"numbers": [2, 3, 5]},
        signals={"approval_given": {"approved": True, "note": "lgtm"}},
        timeout_seconds=120.0,
    )
    assert run.error is None, run.error
    assert run.output is not None
    # The gate's resume payload must have threaded into the exit node.
    rendered = str(run.output)
    assert "10:lgtm" in rendered
