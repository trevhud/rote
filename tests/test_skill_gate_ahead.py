"""A question for the person joins the block before it.

Each block a compiled skill prints is a round trip that re-reads the agent's
conversation. A measured skill stopped for a decision, then stopped again only
to learn the next step was a question for the person, whose text was known all
along. The driver now asks that question in the decision's block, and the
person's reply comes back with the agent's answer (``--with``). The gate keeps
every rule it had: a person's words, the answer schema, the journal record.
No inference: the test is the agent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests._skill_driver import drive, header, journal, make_skill, result_of

CODE = """\
def prep(text):
    return {"text": text.strip(), "words": len(text.split())}


def finish(label, manual):
    return {"label": label["label"], "count": manual["count"]}
"""


def _pipeline(gate_inputs: dict[str, str], when: str | None = None) -> dict[str, Any]:
    """prep -> classify (judge) -> ask (gate) -> finish."""
    gate: dict[str, Any] = {
        "id": "ask",
        "kind": "hitl_gate",
        "description": "Ask the duty manager for today's count.",
        "signal": "manual",
        "timeout": "1h",
        "output": {"count": "int"},
        "inputs": gate_inputs,
        "notify": {"channel": "slack", "target": "#x", "message_template": "Give the count."},
    }
    if when:
        gate["when"] = when
    return {
        "name": "ahead",
        "version": "0.1.0",
        "input": {"type": "In", "required": ["text"]},
        "nodes": [
            {
                "id": "prep",
                "kind": "pure_function",
                "description": "Trim the text.",
                "impl": "extracted/m.py:prep",
                "inputs": {"text": "pipeline.input.text"},
                "output": {"text": "str", "words": "int", "long": "bool"},
            },
            {
                "id": "classify",
                "kind": "llm_judge",
                "description": "Label the text.",
                "inputs": {"src": "prep.output"},
                "signature_spec": {
                    "prompt": "Label this: {{ src.text }}",
                    "input_schema": {
                        "type": "object",
                        "properties": {"src": {"type": "object"}},
                        "required": ["src"],
                    },
                    "output_schema": {
                        "type": "object",
                        "properties": {"label": {"type": "string"}},
                        "required": ["label"],
                        "additionalProperties": False,
                    },
                },
            },
            gate,
            {
                "id": "finish",
                "kind": "pure_function",
                "description": "Finish.",
                "impl": "extracted/m.py:finish",
                "inputs": {"label": "classify.output", "manual": "ask.output"},
            },
        ],
        "edges": [
            {"from": "prep", "to": "classify"},
            {"from": "classify", "to": "ask"},
            {"from": "ask", "to": "finish"},
            {"from": "classify", "to": "finish"},
        ],
        "entry_nodes": ["prep"],
        "exit_nodes": ["finish"],
    }


@pytest.fixture
def skill(tmp_path: Path) -> Path:
    return make_skill(tmp_path, _pipeline({"words": "prep.output.words"}), {"m.py": CODE})


def _start(skill: Path) -> tuple[str, str]:
    status, out = drive(skill, "start", json.dumps({"text": " two words "}))
    assert status == 0, out
    return header(out)["run"], out


LABEL = json.dumps({"label": "short"})
REPLY = json.dumps({"count": 7})


def _stops(scripts: Path) -> list[dict[str, Any]]:
    from rote.skill_target._driver import stops

    return stops(json.loads((scripts / "plan.json").read_text(encoding="utf-8"))["steps"])


def test_the_decisions_block_also_asks_the_person(skill: Path) -> None:
    run, block = _start(skill)
    assert header(block)["kind"] == "DECIDE"
    assert header(block)["step"] == "classify"
    # Said first: an agent told last decided before asking, then decided again.
    assert block.splitlines()[1].startswith("First, before you work on anything here: ask")
    at = block.index("--- ALSO ASK THE PERSON (ask) ---")
    assert block.index("--- TASK ---") < at < block.index("--- NEXT ---")
    asked = block[at:]
    # The gate's own question, message and bound value, as its HUMAN block shows them.
    assert "it is not yours to answer" in asked
    assert "Ask the duty manager for today's count.\nGive the count." in asked
    assert '--- THEY SEE (json) ---\n{"words":2}' in asked
    assert "--- ask ANSWER SCHEMA (json) ---" in asked
    assert asked.rstrip().endswith(
        f"answer {run} classify '<your JSON>' --with ask '<their reply as JSON>' "
        "--said '<their words, verbatim>'"
    )
    assert _stops(skill) == [{"block": "DECIDE", "steps": ["classify"], "ahead": "ask"}]


def test_one_command_records_both_and_the_run_never_stops_at_the_gate(skill: Path) -> None:
    run, _ = _start(skill)
    status, done = drive(
        skill, "answer", run, "classify", LABEL, "--with", "ask", REPLY, "--said", "seven today"
    )
    assert status == 0, done
    assert header(done)["kind"] == "DONE"
    assert result_of(done) == {"finish": {"label": "short", "count": 7}}
    answers = journal(done)["answers"]
    assert answers["classify"]["answered_by"] == "agent"
    assert answers["ask"]["answered_by"] == "human"
    assert answers["ask"]["said"] == "seven today"
    assert not journal(done).get("early")


def test_an_answer_without_the_reply_stops_at_the_gate_as_before(skill: Path) -> None:
    run, _ = _start(skill)
    status, human = drive(skill, "answer", run, "classify", LABEL)
    assert status == 0, human
    assert header(human)["kind"] == "HUMAN"
    assert header(human)["step"] == "ask"
    status, done = drive(skill, "answer", run, "ask", REPLY, "--said", "seven")
    assert header(done)["kind"] == "DONE", done


def test_a_reply_that_does_not_fit_is_asked_again_and_the_decision_is_kept(skill: Path) -> None:
    run, _ = _start(skill)
    bad = json.dumps({"count": "many"})
    status, human = drive(
        skill, "answer", run, "classify", LABEL, "--with", "ask", bad, "--said", "many"
    )
    assert header(human)["kind"] == "HUMAN"
    assert header(human)["step"] == "ask"
    assert "ask the person again" in human
    assert "count" in human
    assert journal(human)["answers"]["classify"]["answered_by"] == "agent"
    assert "ask" not in journal(human)["answers"]
    status, done = drive(skill, "answer", run, "ask", REPLY, "--said", "seven")
    assert result_of(done) == {"finish": {"label": "short", "count": 7}}


def test_a_bad_decision_keeps_the_persons_reply(skill: Path) -> None:
    """The person replied once. A decision asked again must not ask them again."""
    run, _ = _start(skill)
    status, again = drive(
        skill, "answer", run, "classify", '{"label": 3}', "--with", "ask", REPLY, "--said", "seven"
    )
    assert header(again)["kind"] == "DECIDE"
    assert header(again)["attempt"] == "2"
    assert "ask" not in journal(again)["answers"]
    # Printed again, the block no longer asks for a reply it already holds.
    status, shown = drive(skill, "status", run)
    assert header(shown)["kind"] == "DECIDE"
    assert "ALSO ASK THE PERSON" not in shown
    assert "--with" not in shown
    status, done = drive(skill, "answer", run, "classify", LABEL)
    assert header(done)["kind"] == "DONE", done
    assert journal(done)["answers"]["ask"]["said"] == "seven"
    assert result_of(done) == {"finish": {"label": "short", "count": 7}}


def test_a_reply_needs_the_persons_words_and_the_gate_this_block_asked(skill: Path) -> None:
    run, _ = _start(skill)
    status, out = drive(skill, "answer", run, "classify", LABEL, "--with", "ask", REPLY)
    assert status == 2
    assert "answered by a person" in out
    status, out = drive(
        skill, "answer", run, "classify", LABEL, "--with", "finish", REPLY, "--said", "seven"
    )
    assert status == 2
    assert "did not ask the person finish" in out
    # Neither refused command recorded anything.
    status, shown = drive(skill, "status", run)
    assert header(shown)["kind"] == "DECIDE"
    assert not journal(shown)["answers"]
    # A person's words alone, with no reply to carry them, are still refused.
    status, out = drive(skill, "answer", run, "classify", LABEL, "--said", "seven")
    assert status == 2


def test_a_gate_that_shows_the_decision_keeps_its_own_stop(tmp_path: Path) -> None:
    skill = make_skill(tmp_path, _pipeline({"label": "classify.output.label"}), {"m.py": CODE})
    assert _stops(skill) == [
        {"block": "DECIDE", "steps": ["classify"]},
        {"block": "HUMAN", "steps": ["ask"]},
    ]
    _, block = _start(skill)
    assert "ALSO ASK THE PERSON" not in block
    assert "--with" not in block


def test_a_conditional_gate_keeps_its_own_stop(tmp_path: Path) -> None:
    code = CODE.replace('"words": len(text.split())}', '"words": 2, "long": True}')
    skill = make_skill(tmp_path, _pipeline({}, when="prep.output.long"), {"m.py": code})
    assert _stops(skill) == [
        {"block": "DECIDE", "steps": ["classify"]},
        {"block": "HUMAN", "steps": ["ask"], "conditional": True},
    ]
    _, block = _start(skill)
    assert "ALSO ASK THE PERSON" not in block
