"""Cross-runtime contract: ``rote run`` shows a gate's rendered message.

A ``hitl_gate``'s ``notify.message_template`` names the gate's bound
inputs (``{name}``). Every workflow runtime used to ignore it, so a parked
gate showed the person nothing. Now the emitted gate renders it with the
gate's actual input values and puts it in the park signal ``rote run``
already watches (a DBOS workflow event, a Temporal query, a log marker on
Inngest and Cloudflare), and ``rote run`` prints it to stderr when it says
the run waits at that gate.

The rendering rule is one rule (:func:`rote.adapters._py_common.render_gate_message`
and its TS twin), so the fixture is built to tell the runtimes apart where
they could differ: a float that is a whole number (Python writes ``2.0``,
JavaScript ``2``), numbers JavaScript writes in exponent form, an object
whose integer-like key JavaScript moves to the front, null, a list, an
omitted optional input and a key the upstream result lacks (both kept as
written, never a KeyError at the gate), a bidi override inside a value
and a control character in the template (both escaped, never raw on a
terminal). Each runtime runs the emitted app for real and the exact stderr
text is asserted. The TS runtimes live in test_gate_message_parity_ts_e2e.py
and test_gate_message_parity_dbos_ts_e2e.py.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from rote.adapters import get_adapter
from rote.eval.empirical import run_pipeline_trial
from rote.ir import Pipeline
from rote.mcp.parked import watch_factory

_TEMPLATE = (
    "Approve {title}: {count} items at {ratio}x\n"
    "tags {tags} meta {meta} none {nothing} big {big} tiny {tiny}\n"
    "by {who}, note {note}, absent {absent}\u0007"
)

_PIPELINE = Pipeline.model_validate(
    {
        "name": "gate-message",
        "version": "0.1.0",
        "input": {"type": "In", "required": ["title"], "optional": ["note"]},
        "nodes": [
            {
                "id": "summarize",
                "kind": "pure_function",
                "description": "Summarize the plan.",
                "impl": "extracted/summary.py:summarize",
            },
            {
                "id": "approve",
                "kind": "hitl_gate",
                "description": "A person approves the plan.",
                "signal": "approval",
                "inputs": {
                    "title": "pipeline.input.title",
                    "count": "summarize.output.count",
                    "ratio": "summarize.output.ratio",
                    "tags": "summarize.output.tags",
                    "meta": "summarize.output.meta",
                    "nothing": "summarize.output.nothing",
                    "big": "summarize.output.big",
                    "tiny": "summarize.output.tiny",
                    "who": "summarize.output.who",
                    "note": "pipeline.input.note",
                    "absent": "summarize.output.absent",
                },
                "notify": {"channel": "slack", "target": "#plans", "message_template": _TEMPLATE},
            },
            {
                "id": "publish",
                "kind": "pure_function",
                "description": "Publish the decision.",
                "impl": "extracted/summary.py:publish",
                "inputs": {"decision": "approve.output"},
            },
        ],
        "edges": [
            {"from": "summarize", "to": "approve"},
            {"from": "approve", "to": "publish"},
        ],
        "entry_nodes": ["summarize"],
        "exit_nodes": ["publish"],
    }
)

_SUMMARY = (
    "def summarize():\n"
    "    return {\n"
    "        'count': 3,\n"
    "        'ratio': 2.0,\n"
    "        'tags': ['a', 'b'],\n"
    "        'meta': {'b': 1, '2': True},\n"
    "        'nothing': None,\n"
    "        'big': 1e21,\n"
    "        'tiny': 1e-7,\n"
    "        'who': 'Ana \\u202e reversed',\n"
    "    }\n"
    "\n"
    "\n"
    "def publish(decision):\n"
    "    return {'decision': decision}\n"
)

INPUT: dict[str, Any] = {"title": 'Q3 "plan" \U0001f680'}
SIGNALS: dict[str, Any] = {"approval": {"ok": True}}
EXPECTED_OUTPUT = {"publish": {"decision": {"ok": True}}}

#: What ``rote run`` prints when the run reaches the gate, exactly.
EXPECTED_ANNOUNCEMENT = (
    "rote run: waiting at gate 'approve' for signal 'approval':\n"
    '  Approve Q3 "plan" \U0001f680: 3 items at 2x\n'
    '  tags ["a","b"] meta {"2":true,"b":1} none null big 1e+21 tiny 1e-7\n'
    "  by Ana \\u202e reversed, note {note}, absent {absent}\\u0007\n"
)


def check_announcement(stderr: str) -> None:
    """The gate is announced once, with exactly the rendered message."""
    assert EXPECTED_ANNOUNCEMENT in stderr, stderr
    assert stderr.count("waiting at gate 'approve'") == 1, stderr


def _emit(runtime: str, tmp_path: Path) -> Path:
    source = tmp_path / "compiled"
    (source / "extracted").mkdir(parents=True)
    (source / "extracted" / "summary.py").write_text(_SUMMARY, encoding="utf-8")
    out = tmp_path / runtime
    get_adapter(runtime, extracted_source_dir=source).emit(_PIPELINE, out)
    return out


def test_temporal_run_shows_the_gate_message(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
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
    assert run.error is None, run.error
    assert run.output == EXPECTED_OUTPUT
    check_announcement(capsys.readouterr().err)


def test_a_temporal_run_that_fails_after_its_gate_reports_its_own_error(tmp_path: Path) -> None:
    """Watching the gates must not replace a failure with a query error."""
    from rote.runners.temporal import run_temporal

    source = tmp_path / "compiled"
    (source / "extracted").mkdir(parents=True)
    (source / "extracted" / "summary.py").write_text(
        _SUMMARY.replace(
            "    return {'decision': decision}\n", "    raise RuntimeError('publish refused')\n"
        ),
        encoding="utf-8",
    )
    app = tmp_path / "temporal"
    get_adapter("temporal", extracted_source_dir=source).emit(_PIPELINE, app)
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
    assert run.error is not None and "publish refused" in run.error, run.error


@pytest.mark.slow
def test_dbos_run_shows_the_gate_message(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Slow: launches a real DBOS runtime (system-DB migrations, queues)."""
    run = run_pipeline_trial(
        _emit("dbos", tmp_path),
        INPUT,
        signals=SIGNALS,
        timeout_seconds=120,
        park_watch=watch_factory(False),
    )
    assert run.error is None, run.error
    assert run.output == EXPECTED_OUTPUT
    check_announcement(capsys.readouterr().err)
