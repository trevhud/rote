"""Cross-runtime contract: a gate with ``when:`` asks only when its condition holds.

A skill that asks the person something only when a condition holds ("ask
who owns the meeting if that is ambiguous") used to compile to a gate that
waited on every run (#309). ``when: <node>.output.<field>`` names a boolean
an earlier node computes: true, the gate waits as before; false, it is
skipped and its result is null, which the node after it reads whole. Any
other value fails the run naming the gate, on every runtime, rather than
being truthy in one language and an error in another.

Each runtime runs the emitted app for real, three times: the condition
true (the person's answer arrives), false (no wait, no announcement, a null
result), and a string (the named failure). The TS runtimes live in
test_conditional_gate_parity_ts_e2e.py and
test_conditional_gate_parity_dbos_ts_e2e.py.
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

PIPELINE = Pipeline.model_validate(
    {
        "name": "conditional-gate",
        "version": "0.1.0",
        "input": {"type": "In", "required": ["notes", "flag"]},
        "nodes": [
            {
                "id": "detect_owner",
                "kind": "pure_function",
                "description": "Decide whether the meeting's owner is ambiguous.",
                "impl": "extracted/steps.py:detect_owner",
                "inputs": {"notes": "pipeline.input.notes", "flag": "pipeline.input.flag"},
                "output": {"ambiguous": "bool", "candidates": "list"},
            },
            {
                "id": "ask_owner",
                "kind": "hitl_gate",
                "description": "Ask who owns the meeting.",
                "signal": "owner",
                "when": "detect_owner.output.ambiguous",
                "inputs": {"candidates": "detect_owner.output.candidates"},
                "notify": {"message_template": "Who owns it? {candidates}"},
                "output": {"owner": "str"},
            },
            {
                "id": "recap",
                "kind": "pure_function",
                "description": "Write the recap.",
                "impl": "extracted/steps.py:recap",
                "inputs": {"notes": "pipeline.input.notes", "answer": "ask_owner.output"},
            },
        ],
        "edges": [
            {"from": "detect_owner", "to": "ask_owner"},
            {"from": "ask_owner", "to": "recap"},
        ],
        "entry_nodes": ["detect_owner"],
        "exit_nodes": ["recap"],
    }
)

STEPS = (
    "def detect_owner(notes, flag):\n"
    "    return {'ambiguous': flag, 'candidates': ['ana', 'bo']}\n"
    "\n"
    "\n"
    "def recap(notes, answer):\n"
    "    if answer is None:\n"
    "        return {'recap': notes, 'owner': 'me', 'asked': False}\n"
    "    return {'recap': notes, 'owner': answer['owner'], 'asked': True}\n"
)

SIGNALS: dict[str, Any] = {"owner": {"owner": "ana"}}

#: (pipeline input, the run's output or None, a fragment of its error or None, asked)
CASES = {
    "asks": ({"notes": "kickoff", "flag": True}, {"owner": "ana", "asked": True}, None),
    "skips": ({"notes": "kickoff", "flag": False}, {"owner": "me", "asked": False}, None),
    "refuses": (
        {"notes": "kickoff", "flag": "yes"},
        None,
        "waits only when detect_owner.output.ambiguous is true, and it is",
    ),
}


def check(case: str, run: Any, stderr: str) -> None:
    """The run's output (or its named failure) and whether it announced the gate."""
    _, expected, error = CASES[case]
    if error is not None:
        assert run.error is not None and error in run.error, run.error
        assert "waiting at gate 'ask_owner'" not in stderr, stderr
        return
    assert run.error is None, run.error
    assert run.output == {"recap": {"recap": "kickoff", **expected}}
    announced = stderr.count("waiting at gate 'ask_owner'")
    assert announced == (1 if expected["asked"] else 0), stderr


def _emit(runtime: str, tmp_path: Path) -> Path:
    source = tmp_path / "compiled"
    (source / "extracted").mkdir(parents=True)
    (source / "extracted" / "steps.py").write_text(STEPS, encoding="utf-8")
    out = tmp_path / runtime
    get_adapter(runtime, extracted_source_dir=source).emit(PIPELINE, out)
    return out


@pytest.mark.parametrize("case", sorted(CASES))
def test_temporal_asks_only_when_the_condition_holds(
    case: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from rote.runners.temporal import run_temporal

    app = _emit("temporal", tmp_path)
    try:
        run = run_temporal(
            app,
            CASES[case][0],
            signals=SIGNALS,
            timeout_seconds=60,
            park_watch=watch_factory(False),
        )
    finally:
        sys.path.remove(str(app))
        for name in [
            m
            for m in sys.modules
            if m.split(".")[0] in ("workflow", "activities", "extracted", "signatures")
        ]:
            del sys.modules[name]
    check(case, run, capsys.readouterr().err)


@pytest.mark.slow
@pytest.mark.parametrize("case", sorted(CASES))
def test_dbos_asks_only_when_the_condition_holds(
    case: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Slow: launches a real DBOS runtime (system-DB migrations, queues)."""
    run = run_pipeline_trial(
        _emit("dbos", tmp_path),
        CASES[case][0],
        signals=SIGNALS,
        timeout_seconds=120,
        park_watch=watch_factory(False),
    )
    check(case, run, capsys.readouterr().err)
