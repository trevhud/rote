"""A run of decisions is asked in one DECIDE block (#344).

Every block a compiled skill prints is a round trip that re-reads the agent's
whole conversation. A measured compiled skill read twice the tokens of its
source, because its four chained decisions were four stops. The driver now
asks a run of decisions together, found from the plan's graph alone: no code
step, tool call, gate or DO step between them. A decision that reads another
shows that value as a marker; each answer is then checked as if asked alone.
No inference: the test is the agent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests._skill_driver import drive, header, journal, make_skill, result_of


def _obj(**props: str) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {k: {"type": t} for k, t in props.items()},
        "required": list(props),
        "additionalProperties": False,
    }


def _judge(
    id_: str, prompt: str, inputs: dict[str, str], ins: dict[str, Any], out: dict[str, Any]
) -> dict[str, Any]:
    return {
        "id": id_,
        "kind": "llm_judge",
        "description": f"Decide {id_}.",
        "inputs": inputs,
        "signature_spec": {
            "prompt": prompt,
            "input_schema": {"type": "object", "properties": ins, "required": list(ins)},
            "output_schema": out,
        },
    }


PREP = """\
def extract(text):
    return {"text": text.strip()}


def report(closing, verdict):
    return {"question": closing["question"], "ok": verdict["ok"]}
"""

#: extract -> {structure, tone} -> verify(structure) -> closing(verify, tone) -> report
CHAIN = {
    "name": "addr",
    "version": "0.1.0",
    "input": {"type": "In", "required": ["text"]},
    "nodes": [
        {
            "id": "extract",
            "kind": "pure_function",
            "description": "Trim the text.",
            "impl": "extracted/prep.py:extract",
            "inputs": {"text": "pipeline.input.text"},
        },
        _judge(
            "structure",
            "Split this address into a street: {{ src.text }}",
            {"src": "extract.output"},
            {"src": _obj(text="string")},
            _obj(street="string"),
        ),
        _judge(
            "tone",
            "How formal is this text? {{ src.text }}",
            {"src": "extract.output"},
            {"src": _obj(text="string")},
            _obj(tone="string"),
        ),
        _judge(
            "verify",
            "Is {{ addr.street }} a real street?",
            {"addr": "structure.output"},
            {"addr": _obj(street="string")},
            _obj(ok="boolean"),
        ),
        _judge(
            "closing",
            "Write one closing question. Valid: {{ ok }}. Tone: {{ tone.tone }}.",
            {"ok": "verify.output.ok", "tone": "tone.output"},
            {"ok": {"type": "boolean"}, "tone": _obj(tone="string")},
            _obj(question="string"),
        ),
        {
            "id": "report",
            "kind": "pure_function",
            "description": "Report it.",
            "impl": "extracted/prep.py:report",
            "inputs": {"closing": "closing.output", "verdict": "verify.output"},
        },
    ],
    "edges": [
        {"from": "extract", "to": "structure"},
        {"from": "extract", "to": "tone"},
        {"from": "structure", "to": "verify"},
        {"from": "verify", "to": "closing"},
        {"from": "tone", "to": "closing"},
        {"from": "closing", "to": "report"},
    ],
    "entry_nodes": ["extract"],
    "exit_nodes": ["report"],
}

ANSWERS = {
    "structure": {"street": "1 Main St"},
    "tone": {"tone": "casual"},
    "verify": {"ok": True},
    "closing": {"question": "Anything else?"},
}


@pytest.fixture
def chain(tmp_path: Path) -> Path:
    return make_skill(tmp_path, CHAIN, {"prep.py": PREP})


def _start(skill: Path) -> str:
    status, out = drive(skill, "start", json.dumps({"text": "  1 Main St, hi there "}))
    assert status == 0, out
    return out


def test_a_run_of_decisions_is_one_block_with_markers_for_what_it_waits_on(chain: Path) -> None:
    block = _start(chain)
    fields = header(block)
    # One stop for four decisions, in plan order: same-wave ones side by side,
    # chained ones after what they read.
    assert fields["kind"] == "DECIDE"
    assert fields["decisions"] == "structure,tone,verify,closing"
    assert "Split this address into a street: 1 Main St, hi there" in block
    # A value another decision supplies is a marker, never a guess.
    assert "Is «`street` of your answer to structure» a real street?" in block
    assert "Valid: «`ok` of your answer to verify». Tone: «`tone` of your answer to tone»." in block
    for step in ANSWERS:
        assert f"--- DECISION {step} ---" in block
        assert f"--- {step} ANSWER SCHEMA (json) ---" in block


def test_one_answer_records_every_decision_and_the_run_goes_on(chain: Path) -> None:
    run = header(_start(chain))["run"]
    status, done = drive(chain, "answer", run, "structure", json.dumps(ANSWERS))
    assert status == 0, done
    assert header(done)["kind"] == "DONE"
    assert result_of(done) == {"report": {"question": "Anything else?", "ok": True}}
    answers = journal(done)["answers"]
    assert [answers[s]["answered_by"] for s in ANSWERS] == ["agent"] * 4


def test_a_bad_answer_is_asked_again_alone_and_the_ones_before_it_are_kept(chain: Path) -> None:
    run = header(_start(chain))["run"]
    bad = {**ANSWERS, "verify": {"ok": "yes"}}
    _, again = drive(chain, "answer", run, "structure", json.dumps(bad))
    fields = header(again)
    assert (fields["kind"], fields["step"]) == ("DECIDE", "verify")
    assert "answer.ok: expected boolean" in again and "decisions" not in fields
    kept = journal(again)["answers"]
    assert set(kept) == {"structure", "tone"}
    # verify answered: what came after it is asked again, as its own run.
    _, after = drive(chain, "answer", run, "verify", json.dumps({"ok": False}))
    assert header(after)["step"] == "closing" and "decisions" not in header(after)
    _, done = drive(chain, "answer", run, "closing", json.dumps({"question": "Why?"}))
    assert result_of(done) == {"report": {"question": "Why?", "ok": False}}


@pytest.mark.parametrize(
    ("answer", "message"),
    [
        ({"structure": ANSWERS["structure"]}, "answer: missing tone, verify, closing"),
        ({**ANSWERS, "extra": {}}, "'extra' is not a decision of this block"),
        (["not", "an", "object"], "expected an object keyed by decision id"),
    ],
)
def test_a_malformed_batch_answer_is_asked_again_whole(
    chain: Path, answer: Any, message: str
) -> None:
    run = header(_start(chain))["run"]
    _, again = drive(chain, "answer", run, "structure", json.dumps(answer))
    assert message in again
    assert journal(again)["answers"] == {}
    # status prints the batch block again.
    _, shown = drive(chain, "status", run)
    assert header(shown)["decisions"] == "structure,tone,verify,closing"


def test_a_code_step_between_decisions_ends_the_run(tmp_path: Path) -> None:
    """closing waits on a code step here, so it is never asked before it runs."""
    nodes = [dict(n) for n in CHAIN["nodes"]]
    nodes.insert(
        4,
        {
            "id": "normalize",
            "kind": "pure_function",
            "description": "Normalize the verdict.",
            "impl": "extracted/prep.py:normalize",
            "inputs": {"v": "verify.output"},
        },
    )
    closing = next(n for n in nodes if n["id"] == "closing")
    closing["inputs"] = {"ok": "normalize.output.ok", "tone": "tone.output"}
    gated = {
        **CHAIN,
        "nodes": nodes,
        "edges": [
            {"from": "extract", "to": "structure"},
            {"from": "extract", "to": "tone"},
            {"from": "structure", "to": "verify"},
            {"from": "verify", "to": "normalize"},
            {"from": "normalize", "to": "closing"},
            {"from": "tone", "to": "closing"},
            {"from": "closing", "to": "report"},
        ],
    }
    skill = make_skill(
        tmp_path,
        gated,
        {"prep.py": PREP + "\n\ndef normalize(v):\n    return {'ok': bool(v['ok'])}\n"},
    )
    first = _start(skill)
    assert header(first)["decisions"] == "structure,tone,verify"
    run = header(first)["run"]
    trio = {k: ANSWERS[k] for k in ("structure", "tone", "verify")}
    _, then = drive(skill, "answer", run, "structure", json.dumps(trio))
    assert (header(then)["step"], header(then).get("decisions")) == ("closing", None)
