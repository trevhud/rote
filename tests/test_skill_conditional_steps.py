"""A compiled skill's step with ``when:`` runs only when its condition holds.

A step a skill does only when something holds ("post it if chat is
connected") used to be a stop on every run. The driver decides each skip
when the run reaches the step's wave, before the wave's code runs and
before any block is printed, so a skipped DO step is never handed to the
agent and a skipped code step never runs. A skip writes nothing: a resumed
run derives it again from the result the condition names. No inference: the
test is the agent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rote.ir import Pipeline
from rote.plan import build_plan
from rote.skill_target._driver import decision_run, stops
from tests._skill_driver import drive, header, make_skill, result_of, run_dir

CALC = """\
def check(chat, crm):
    return {"chat": chat, "crm": crm}


def tidy(month):
    return {"tidied": month}


def report(post, log, tidy):
    return {"post": post, "log": log, "tidy": tidy}
"""


def _do(id_: str, when: str) -> dict[str, Any]:
    """An external_call with no code (a DO step), done only when ``when``."""
    return {
        "id": id_,
        "kind": "external_call",
        "description": f"Do {id_}.",
        "impl": f"extracted/tools.py:{id_}",
        "when": when,
        "inputs": {"month": "pipeline.input.month"},
        "output": {"ref": "str"},
    }


PIPELINE: dict[str, Any] = {
    "name": "connectors",
    "version": "0.1.0",
    "input": {"type": "In", "required": ["month", "chat", "crm"]},
    "nodes": [
        {
            "id": "check",
            "kind": "pure_function",
            "description": "See which tools answer.",
            "impl": "extracted/calc.py:check",
            "inputs": {"chat": "pipeline.input.chat", "crm": "pipeline.input.crm"},
            "output": {"chat": "bool", "crm": "bool"},
        },
        _do("post", "check.output.chat"),
        _do("log", "check.output.crm"),
        {
            "id": "tidy",
            "kind": "pure_function",
            "description": "Tidy the CRM month.",
            "impl": "extracted/calc.py:tidy",
            "when": "check.output.crm",
            "inputs": {"month": "pipeline.input.month"},
        },
        {
            "id": "report",
            "kind": "pure_function",
            "description": "Say what was done.",
            "impl": "extracted/calc.py:report",
            "inputs": {"post": "post.output", "log": "log.output", "tidy": "tidy.output"},
        },
    ],
    "edges": [
        {"from": "check", "to": "post"},
        {"from": "check", "to": "log"},
        {"from": "check", "to": "tidy"},
        {"from": "post", "to": "report"},
        {"from": "log", "to": "report"},
        {"from": "tidy", "to": "report"},
    ],
    "entry_nodes": ["check"],
    "exit_nodes": ["report"],
}

STUBS = {"post": "stub", "log": "stub"}


def _steps() -> list[dict[str, Any]]:
    plan = build_plan(Pipeline.model_validate(PIPELINE), mcp="live", implementations=STUBS)
    steps: list[dict[str, Any]] = plan["steps"]
    return steps


@pytest.fixture
def skill(tmp_path: Path) -> Path:
    return make_skill(tmp_path, PIPELINE, {"calc.py": CALC}, implementations=STUBS)


def _start(skill: Path, chat: Any, crm: Any) -> str:
    _, out = drive(skill, "start", json.dumps({"month": "2026-10", "chat": chat, "crm": crm}))
    return out


def test_the_plan_carries_the_condition_on_every_kind_of_step() -> None:
    steps = _steps()
    whens = {s["id"]: (s["kind"], s.get("when")) for s in steps}
    assert whens == {
        "check": ("call", None),
        "log": ("do", {"node": "check", "field": "crm"}),
        "post": ("do", {"node": "check", "field": "chat"}),
        "tidy": ("call", {"node": "check", "field": "crm"}),
        "report": ("call", None),
    }


def test_every_condition_false_runs_to_the_end_with_no_stop(skill: Path) -> None:
    done = _start(skill, chat=False, crm=False)
    assert header(done)["kind"] == "DONE", done
    assert result_of(done) == {"report": {"post": None, "log": None, "tidy": None}}
    # Nothing stands in for a skipped step on disk: a resume derives it again.
    written = sorted(p.name for p in (run_dir(done) / "nodes").iterdir())
    assert written == ["check.json", "report.json"]


def test_only_the_step_whose_condition_holds_is_handed_to_the_agent(skill: Path) -> None:
    block = _start(skill, chat=True, crm=False)
    fields = header(block)
    assert (fields["kind"], fields["step"], "steps" in fields) == ("DO", "post", False)
    assert "Do log." not in block
    # A resume derives the same skips and prints the same block.
    _, again = drive(skill, "resume", fields["run"])
    assert (header(again)["kind"], header(again)["step"]) == ("DO", "post")
    assert "Do log." not in again
    _, done = drive(skill, "answer", fields["run"], "post", '{"ref": "m-1"}')
    assert header(done)["kind"] == "DONE", done
    assert result_of(done) == {"report": {"post": {"ref": "m-1"}, "log": None, "tidy": None}}


def test_steps_whose_conditions_hold_share_one_block_and_the_code_step_runs(skill: Path) -> None:
    block = _start(skill, chat=True, crm=True)
    fields = header(block)
    assert (fields["kind"], fields["steps"]) == ("DO", "log,post")
    answer = json.dumps({"log": {"ref": "c-9"}, "post": {"ref": "m-1"}})
    _, done = drive(skill, "answer", fields["run"], fields["step"], answer)
    assert result_of(done) == {
        "report": {"post": {"ref": "m-1"}, "log": {"ref": "c-9"}, "tidy": {"tidied": "2026-10"}}
    }


def test_a_condition_that_is_not_a_boolean_ends_the_run_naming_the_step(skill: Path) -> None:
    out = _start(skill, chat="yes", crm=False)
    assert header(out)["kind"] == "ERROR", out
    assert "external_call 'post' runs only when check.output.chat is true, and it is str" in out
    # The wave's other steps did not run past the broken condition.
    assert not (run_dir(out) / "nodes" / "report.json").exists()


def test_stops_marks_a_block_whose_steps_may_be_skipped() -> None:
    steps = _steps()
    assert stops(steps) == [{"block": "DO", "steps": ["log", "post"], "conditional": True}]
    # The negative half: one step that always runs means the stop always happens.
    for step in steps:
        if step["id"] == "log":
            del step["when"]
    assert stops(steps) == [{"block": "DO", "steps": ["log", "post"]}]


def _judge(id_: str, wave: int, when: dict[str, str] | None = None) -> dict[str, Any]:
    step: dict[str, Any] = {"id": id_, "kind": "judge", "fan_param": None, "wave": wave}
    if when:
        step["when"] = when
    return step


def test_a_decision_whose_condition_is_not_known_yet_waits_for_its_wave() -> None:
    """A later decision may read the first one's answer as its condition, so
    it cannot be asked in the same block. Both halves: with no condition it
    joins, and one in the first decision's own wave (decided already) joins."""
    first = _judge("triage", 0)
    later = _judge("escalate", 1, {"node": "triage", "field": "urgent"})
    assert decision_run([first, later], first, set()) == [first]
    assert decision_run([first, _judge("escalate", 1)], first, set()) == [
        first,
        _judge("escalate", 1),
    ]
    beside = _judge("label", 0, {"node": "check", "field": "ok"})
    assert decision_run([first, beside], first, set()) == [first, beside]
