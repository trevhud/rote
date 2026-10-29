"""Cross-runtime contract: a gate shows the data it binds, not only its message.

A gate binds what the person decides on (``Node._validate_gate_shows_what_it_binds``),
but the workflow runtimes reported only the rendered message. A message that
calls the draft "below" with no placeholder showed nothing below it (#325).
Each runtime now reports its other bound inputs as ``shows``, and ``rote run``
prints them under the message: a string as written, keeping its lines, and
anything else as compact JSON. The TS runtimes live in
test_gate_shows_parity_ts_e2e.py and test_gate_shows_parity_dbos_ts_e2e.py.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from rote.adapters import get_adapter
from rote.eval.empirical import run_pipeline_trial
from rote.ir import Pipeline
from rote.mcp.parked import GateReached, gate_announcement, watch_factory

PIPELINE = Pipeline.model_validate(
    {
        "name": "gate-shows",
        "version": "0.1.0",
        "input": {"type": "In", "required": ["draft", "author"]},
        "nodes": [
            {
                "id": "outline",
                "kind": "pure_function",
                "description": "Outline the draft.",
                "impl": "extracted/steps.py:outline",
                "inputs": {"draft": "pipeline.input.draft"},
                "output": {"sections": "list"},
            },
            {
                "id": "approve_draft",
                "kind": "hitl_gate",
                "description": "Review the draft before approval.",
                "signal": "draft_approved",
                "inputs": {
                    "draft": "pipeline.input.draft",
                    "author": "pipeline.input.author",
                    "sections": "outline.output.sections",
                },
                "notify": {"message_template": "Approve {author}'s draft below?"},
                "output": {"approved": "bool"},
            },
            {
                "id": "publish",
                "kind": "pure_function",
                "description": "Publish it.",
                "impl": "extracted/steps.py:publish",
                "inputs": {"decision": "approve_draft.output"},
            },
        ],
        "edges": [
            {"from": "outline", "to": "approve_draft"},
            {"from": "approve_draft", "to": "publish"},
        ],
        "entry_nodes": ["outline"],
        "exit_nodes": ["publish"],
    }
)

STEPS = (
    "def outline(draft):\n"
    "    return {'sections': ['intro', 'body']}\n"
    "\n"
    "\n"
    "def publish(decision):\n"
    "    return {'published': decision['approved']}\n"
)

INPUT: dict[str, Any] = {"draft": "NEUTRAL_DRAFT_CONTENT_29\nsecond line", "author": "Ana"}
SIGNALS: dict[str, Any] = {"draft_approved": {"approved": True}}
EXPECTED_OUTPUT = {"publish": {"published": True}}

#: What ``rote run`` prints at the gate: the message, then what it binds and
#: does not name (``author`` is in the message, so it is not repeated).
EXPECTED_ANNOUNCEMENT = (
    "rote run: waiting at gate 'approve_draft' for signal 'draft_approved':\n"
    "  Approve Ana's draft below?\n"
    "  draft:\n"
    "    NEUTRAL_DRAFT_CONTENT_29\n"
    "    second line\n"
    '  sections: ["intro","body"]\n'
)


def check(run: Any, stderr: str) -> None:
    assert run.error is None, run.error
    assert run.output == EXPECTED_OUTPUT
    assert EXPECTED_ANNOUNCEMENT in stderr, stderr


def test_the_announcement_shows_what_the_message_does_not_name() -> None:
    gate = GateReached(
        gate="g",
        signal="s",
        message=None,
        key="k",
        shows={"draft": "line one‮", "items": [1, {"a": "é"}]},
    )
    assert gate_announcement(gate) == (
        "rote run: waiting at gate 'g' for signal 's':\n"
        "  draft: line one\\u202e\n"
        '  items: [1,{"a":"é"}]'
    )


def _emit(runtime: str, tmp_path: Path) -> Path:
    source = tmp_path / "compiled"
    (source / "extracted").mkdir(parents=True)
    (source / "extracted" / "steps.py").write_text(STEPS, encoding="utf-8")
    out = tmp_path / runtime
    get_adapter(runtime, extracted_source_dir=source).emit(PIPELINE, out)
    return out


def test_temporal_shows_the_gates_data(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from rote.runners.temporal import run_temporal

    app = _emit("temporal", tmp_path)
    try:
        run = run_temporal(
            app, INPUT, signals=SIGNALS, timeout_seconds=60, park_watch=watch_factory(False)
        )
    finally:
        sys.path.remove(str(app))
        for name in [
            m
            for m in sys.modules
            if m.split(".")[0] in ("workflow", "activities", "extracted", "signatures")
        ]:
            del sys.modules[name]
    check(run, capsys.readouterr().err)


@pytest.mark.slow
def test_dbos_shows_the_gates_data(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run = run_pipeline_trial(
        _emit("dbos", tmp_path),
        INPUT,
        signals=SIGNALS,
        timeout_seconds=120,
        park_watch=watch_factory(False),
    )
    check(run, capsys.readouterr().err)
