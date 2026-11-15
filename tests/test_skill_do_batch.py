"""A wave's DO steps are handed to the agent in one DO block.

Every block a compiled skill prints is a round trip that re-reads the agent's
whole conversation. Compiled from Anthropic's connector-style skills, a
cash-flow skill pulled five settlement sources in one wave and stopped five
times for them, and a business-pulse skill stopped 22 times. Steps in one
wave read none of each other's results, so the driver hands them over
together and checks each answer as if asked alone. No inference: the test is
the agent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rote.plan import build_plan
from rote.skill_target._driver import MAX_BATCHED_DECISIONS, stops
from tests._skill_driver import drive, header, journal, make_skill, result_of

REPORT = """\
def report(sales, refunds):
    return {"net": sales["total"] - refunds["total"]}
"""


def _pull(id_: str, inputs: dict[str, str] | None = None) -> dict[str, Any]:
    """An external_call with no code: the agent does it with its own tools."""
    return {
        "id": id_,
        "kind": "external_call",
        "description": f"Pull {id_} from the store.",
        "impl": f"extracted/store.py:{id_}",
        "inputs": inputs or {"month": "pipeline.input.month"},
        "output": {"total": "int"},
    }


def _pipeline(*pulls: dict[str, Any], edges: list[dict[str, str]] | None = None) -> dict[str, Any]:
    return {
        "name": "pulse",
        "version": "0.1.0",
        "input": {"type": "In", "required": ["month"]},
        "nodes": [
            *pulls,
            {
                "id": "report",
                "kind": "pure_function",
                "description": "Net it.",
                "impl": "extracted/calc.py:report",
                "inputs": {"sales": "sales.output", "refunds": "refunds.output"},
            },
        ],
        "edges": [
            *(edges or []),
            {"from": "sales", "to": "report"},
            {"from": "refunds", "to": "report"},
        ],
        "entry_nodes": [p["id"] for p in pulls if not p.get("after")],
        "exit_nodes": ["report"],
    }


SIDE_BY_SIDE = _pipeline(_pull("sales"), _pull("refunds"))


def _skill(root: Path, pipeline: dict[str, Any]) -> Path:
    """The compiled skill, every pull's impl a stub (so a DO step)."""
    stubs = {n["id"]: "stub" for n in pipeline["nodes"] if n["kind"] == "external_call"}
    return make_skill(root, pipeline, {"calc.py": REPORT}, implementations=stubs)


@pytest.fixture
def pulse(tmp_path: Path) -> Path:
    return _skill(tmp_path, SIDE_BY_SIDE)


def _start(skill: Path) -> str:
    status, out = drive(skill, "start", json.dumps({"month": "2026-10"}))
    assert status == 0, out
    return out


def _answer(skill: Path, block: str, value: Any) -> str:
    fields = header(block)
    _, out = drive(skill, "answer", fields["run"], fields["step"], json.dumps(value))
    return out


def test_a_waves_do_steps_are_one_block_and_each_answer_is_recorded(pulse: Path) -> None:
    block = _start(pulse)
    fields = header(block)
    assert (fields["kind"], fields["step"], fields["steps"]) == ("DO", "refunds", "refunds,sales")
    assert "Do these 2 steps yourself with your own tools" in block
    assert block.count("follow your client's approval for it") == 1
    for step in ("refunds", "sales"):
        assert f"--- STEP {step} ---\nPull {step} from the store." in block
        assert f'--- {step} INPUTS (json) ---\n{{"month":"2026-10"}}' in block
        assert f"--- {step} ANSWER SHAPE (key? = optional) ---\n{{total: integer}}" in block
    done = _answer(pulse, block, {"sales": {"total": 900}, "refunds": {"total": 150}})
    assert header(done)["kind"] == "DONE", done
    assert result_of(done) == {"report": {"net": 750}}
    answers = journal(done)["answers"]
    assert {k: v["answered_by"] for k, v in answers.items()} == {
        "sales": "agent",
        "refunds": "agent",
    }


def test_stops_lists_the_wave_as_one_do_stop() -> None:
    from rote.ir import Pipeline

    stubs = {"sales": "stub", "refunds": "stub"}
    ir = Pipeline.model_validate(SIDE_BY_SIDE)
    steps = build_plan(ir, mcp="live", implementations=stubs)["steps"]
    # In plan order: a wave's steps by id.
    assert stops(steps) == [{"block": "DO", "steps": ["refunds", "sales"]}]


def test_a_do_step_that_reads_another_is_its_own_block(tmp_path: Path) -> None:
    """The negative half: a step in a later wave can read the earlier answer,
    so it waits for it instead of joining the block."""
    chained = _pipeline(
        _pull("sales"),
        _pull("refunds", {"sales": "sales.output"}),
        edges=[{"from": "sales", "to": "refunds"}],
    )
    skill = _skill(tmp_path, chained)
    block = _start(skill)
    assert (header(block)["step"], "steps" in header(block)) == ("sales", False)
    nxt = _answer(skill, block, {"total": 900})
    assert (header(nxt)["kind"], header(nxt)["step"]) == ("DO", "refunds")
    assert '{"sales":{"total":900}}' in nxt


def test_a_bad_answer_is_asked_again_alone_with_the_ones_before_it_kept(pulse: Path) -> None:
    block = _start(pulse)
    again = _answer(pulse, block, {"refunds": {"total": 150}, "sales": {"total": "lots"}})
    fields = header(again)
    assert (fields["kind"], fields["step"], "steps" in fields) == ("DO", "sales", False)
    assert "Your answer to sales did not validate. The steps before it are recorded." in again
    assert "--- STEP ---\nPull sales from the store." in again
    assert set(journal(again)["answers"]) == {"refunds"}
    done = _answer(pulse, again, {"total": 900})
    assert result_of(done) == {"report": {"net": 750}}


def test_an_answer_not_keyed_by_step_is_asked_again_whole(pulse: Path) -> None:
    block = _start(pulse)
    again = _answer(pulse, block, {"refunds": {"total": 150}})
    fields = header(again)
    assert (fields["kind"], fields["step"], fields["attempt"]) == ("DO", "refunds", "2")
    assert "- answer: missing sales" in again
    assert journal(again)["answers"] == {}
    stray = _answer(pulse, block, [1, 2])
    assert "- answer: expected an object keyed by step id (refunds, sales)" in stray


def test_resume_shows_the_same_block(pulse: Path) -> None:
    block = _start(pulse)
    _, again = drive(pulse, "resume", header(block)["run"])
    assert again == block


def test_the_block_holds_at_most_the_cap(tmp_path: Path) -> None:
    extra = [_pull(f"source_{i}") for i in range(MAX_BATCHED_DECISIONS)]
    wide = _pipeline(_pull("sales"), _pull("refunds"), *extra)
    skill = _skill(tmp_path, wide)
    first = header(_start(skill))["steps"].split(",")
    assert len(first) == MAX_BATCHED_DECISIONS


def test_a_fanned_do_step_is_asked_alone() -> None:
    """A fanned DO step hands over its items in batches of its own, so it
    never joins a block of plain DO steps in its wave."""
    from rote.ir import Pipeline

    fanned = {
        **_pull("sales"),
        "fan_out": True,
        "inputs": {"store": "pipeline.input.stores"},
    }
    doc = _pipeline(fanned, _pull("refunds"))
    doc["input"] = {
        "type": "In",
        "required": ["month", "stores"],
        "input_schema": {
            "type": "object",
            "properties": {"month": {"type": "string"}, "stores": {"type": "array"}},
        },
    }
    stubs = {"sales": "stub", "refunds": "stub"}
    steps = build_plan(Pipeline.model_validate(doc), mcp="live", implementations=stubs)["steps"]
    assert stops(steps) == [
        {"block": "DO", "steps": ["refunds"]},
        {"block": "DO", "steps": ["sales"], "fanned": True},
    ]
