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
from rote.skill_target._driver import (
    MAX_ARGUMENT_CHARS,
    MAX_BATCHED_DECISIONS,
    MIN_SHARED_INPUT_CHARS,
    stops,
)
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


def test_a_section_several_steps_share_is_printed_once(tmp_path: Path) -> None:
    """Steps cut from one skill section each carried all of it: a measured
    block of four reads was 7.5 KB, 5.6 KB of it the same text three more
    times, and every later request re-read it."""
    shared = "## Sources\nPull each from the store, month by month."
    own = "## Tax\nPull tax from the ledger."
    pipeline = _pipeline(_pull("sales"), _pull("refunds"), _pull("tax"))
    stubs = {"sales": "stub", "refunds": "stub", "tax": "stub"}
    context = {
        "sales": {"instructions": shared},
        "refunds": {"instructions": shared},
        "tax": {"instructions": own},
    }
    skill = make_skill(
        tmp_path, pipeline, {"calc.py": REPORT}, implementations=stubs, do_context=context
    )
    block = _start(skill)
    assert header(block)["steps"] == "refunds,sales,tax"
    assert block.count(shared) == 1
    # Before the first step it is for, naming every step it is for.
    assert (
        f"--- WHAT THE SKILL SAYS FOR refunds, sales ---\n{shared}\n--- STEP refunds ---" in block
    )
    # A section of one step's own still reaches it, under its own name.
    assert f"--- WHAT THE SKILL SAYS FOR tax ---\n{own}\n--- STEP tax ---" in block


def test_a_do_block_says_how_to_pass_a_long_answer(pulse: Path) -> None:
    """Agents staged a long answer in a file or a script first, which their
    client refused: two wasted requests in two of three measured runs. And a
    client stops approving the command near 10,000 characters, so a block of
    several steps says to answer in parts and never names a file."""
    block = _start(pulse)
    how, command = block.split("--- NEXT ---\n")
    assert "Inside the JSON, write an apostrophe as \\u0027." in how
    assert "no heredoc, no other command" in how
    assert f"Past about {MAX_ARGUMENT_CHARS:,} characters the command is refused: " in how
    assert "answer one step per command, with that step's id in the command" in how
    # The file form comes last, for one step's answer that is still too long.
    assert how.index("answer one step per command") < how.index("pass --file <path> instead.")
    assert "If one step's answer is still that long, write the JSON to a file" in how
    assert command.startswith("rote skill run pulse answer ")
    assert command.rstrip().endswith("'<your JSON>'")


def test_a_block_answered_in_parts_is_not_a_failed_attempt(tmp_path: Path) -> None:
    """Two of three measured runs lost a 10,000-character answer for three
    steps to the client's command length limit. Answered a step at a time, each
    part is recorded, the rest is asked for, and no attempt is spent."""
    three = _pipeline(_pull("sales"), _pull("refunds"), _pull("fees"))
    skill = _skill(tmp_path, three)
    block = _start(skill)
    rest = _answer(skill, block, {"fees": {"total": 5}})
    fields = header(rest)
    assert (fields["kind"], fields["step"], fields["steps"]) == ("DO", "refunds", "refunds,sales")
    assert "attempt" not in fields
    assert "Recorded: fees. Do not do those again. Answer only refunds, sales now." in rest
    assert "with these fixed" not in rest and "missing" not in rest
    assert "--- STEP fees ---" not in rest
    last = _answer(skill, rest, {"sales": {"total": 900}})
    assert (header(last)["step"], "attempt" in header(last)) == ("refunds", False)
    assert set(journal(last)["answers"]) == {"fees", "sales"}
    # Two parts spent no attempt: the first answer that fails is the first failure.
    bad = _answer(skill, last, {"total": "lots"})
    assert header(bad)["attempt"] == "2"
    done = _answer(skill, bad, {"total": 150})
    assert result_of(done) == {"report": {"net": 750}}


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


def test_a_bad_answer_is_asked_again_alone_and_the_valid_ones_are_kept(pulse: Path) -> None:
    block = _start(pulse)
    again = _answer(pulse, block, {"refunds": {"total": 150}, "sales": {"total": "lots"}})
    fields = header(again)
    assert (fields["kind"], fields["step"], "steps" in fields) == ("DO", "sales", False)
    assert fields["attempt"] == "2"
    assert "Recorded: refunds. Do not do those again. Answer only sales now" in again
    assert "- answer.sales.total:" in again
    # Asked again in short: the agent has the step's text and inputs already,
    # and a block of several large inputs printed again was 53 KB a time.
    assert "Pull sales from the store." not in again and "INPUTS" not in again
    assert "--- sales ANSWER SHAPE (key? = optional) ---\n{total: integer}" in again
    _, full = drive(pulse, "resume", fields["run"])
    assert "--- STEP ---\nPull sales from the store." in full
    assert f"rote skill run pulse resume {fields['run']} prints them again" in again
    assert set(journal(again)["answers"]) == {"refunds"}
    done = _answer(pulse, again, {"total": 900})
    assert result_of(done) == {"report": {"net": 750}}


def test_a_failing_step_never_drops_a_valid_answer_after_it(tmp_path: Path) -> None:
    """A DO step may write: asking again for one already answered would have
    the agent do it twice. The steps share a wave, so the order is not why."""
    three = _pipeline(_pull("sales"), _pull("refunds"), _pull("fees"))
    skill = _skill(tmp_path, three)
    block = _start(skill)
    assert header(block)["steps"] == "fees,refunds,sales"
    value = {"fees": {"total": 5}, "refunds": {"total": "x"}, "sales": {"total": 900}}
    again = _answer(skill, block, value)
    assert set(journal(again)["answers"]) == {"fees", "sales"}
    assert (header(again)["step"], "steps" in header(again)) == ("refunds", False)
    assert "Recorded: fees, sales." in again
    done = _answer(skill, again, {"total": 150})
    assert result_of(done) == {"report": {"net": 750}}


def test_several_failures_are_asked_again_together(tmp_path: Path) -> None:
    three = _pipeline(_pull("sales"), _pull("refunds"), _pull("fees"))
    skill = _skill(tmp_path, three)
    block = _start(skill)
    again = _answer(skill, block, {"fees": {"total": 5}, "refunds": {"total": "x"}})
    fields = header(again)
    assert (fields["kind"], fields["step"], fields["steps"]) == ("DO", "refunds", "refunds,sales")
    assert "Answer only refunds, sales now, with these fixed:" in again
    assert "- answer.refunds.total:" in again
    assert "--- STEP fees ---" not in again
    done = _answer(skill, again, {"refunds": {"total": 150}, "sales": {"total": 900}})
    assert result_of(done) == {"report": {"net": 750}}


def test_an_answer_that_is_no_step_of_the_block_is_asked_again_whole(pulse: Path) -> None:
    block = _start(pulse)
    again = _answer(pulse, block, [1, 2])
    fields = header(again)
    assert (fields["kind"], fields["step"], fields["attempt"]) == ("DO", "refunds", "2")
    assert fields["steps"] == "refunds,sales"
    assert "- answer.refunds: expected object" in again
    assert journal(again)["answers"] == {}
    assert 'mapping each step\'s id to its answer: {"refunds":"<its answer>"' in again


def test_one_step_is_answered_by_itself_under_its_own_id(tmp_path: Path) -> None:
    """In all three measured runs the agent's first move for a long answer
    was a command per step, naming the step. The driver took only the block's
    first id and an object keyed by step, so each try printed the block again."""
    three = _pipeline(_pull("sales"), _pull("refunds"), _pull("fees"))
    skill = _skill(tmp_path, three)
    block = _start(skill)
    run = header(block)["run"]
    assert header(block)["step"] == "fees"
    # Not the block's first step, and its answer alone.
    _, rest = drive(skill, "answer", run, "sales", json.dumps({"total": 900}))
    assert set(journal(rest)["answers"]) == {"sales"}
    assert (header(rest)["steps"], "attempt" in header(rest)) == ("fees,refunds", False)
    # The first step's own answer, not keyed: `total` is no step of the block.
    _, last = drive(skill, "answer", run, "fees", json.dumps({"total": 5}))
    assert set(journal(last)["answers"]) == {"sales", "fees"}
    assert (header(last)["step"], "steps" in header(last)) == ("refunds", False)
    # A step already recorded is not taken twice.
    _, twice = drive(skill, "answer", run, "sales", json.dumps({"total": 1}))
    assert "sales is already answered; nothing changed." in twice
    done = _answer(skill, last, {"total": 150})
    assert result_of(done) == {"report": {"net": 750}}


def test_a_value_several_steps_read_is_printed_once(tmp_path: Path) -> None:
    """Three DO steps of one wave each printed the same 13 KB result."""
    rows = "pipeline.input.rows"
    two = _pipeline(
        _pull("sales", {"rows": rows, "month": "pipeline.input.month"}),
        _pull("refunds", {"rows": rows}),
    )
    two["input"] = {"type": "In", "required": ["month", "rows"]}
    skill = _skill(tmp_path, two)
    big = [{"sku": f"S-{i:04d}", "note": "x" * 40} for i in range(MIN_SHARED_INPUT_CHARS // 40)]
    status, block = drive(skill, "start", json.dumps({"month": "2026-10", "rows": big}))
    assert status == 0, block
    assert block.count("S-0001") == 1
    shared = block.split("--- INPUTS SEVERAL STEPS READ (json) ---\n")[1].split("\n")[0]
    assert json.loads(shared) == {"rows": big}
    assert '--- refunds INPUTS (json) ---\n{"rows":"\u00abrows, above\u00bb"}' in block
    # A short value stays with its step: a marker would save nothing.
    assert '"month":"2026-10"' in block.split("--- sales INPUTS (json) ---\n")[1].split("\n")[0]


def test_a_mistyped_step_id_is_named_and_the_valid_part_is_kept(pulse: Path) -> None:
    block = _start(pulse)
    again = _answer(pulse, block, {"refunds": {"total": 150}, "slaes": {"total": 900}})
    assert "- answer: 'slaes' is not a step of this block" in again
    assert "Recorded: refunds." in again and header(again)["attempt"] == "2"
    assert set(journal(again)["answers"]) == {"refunds"}


def test_an_answer_that_does_not_parse_after_a_part_names_what_is_left(tmp_path: Path) -> None:
    three = _pipeline(_pull("sales"), _pull("refunds"), _pull("fees"))
    skill = _skill(tmp_path, three)
    block = _start(skill)
    run = header(block)["run"]
    drive(skill, "answer", run, "fees", json.dumps({"total": 5}))
    _, again = drive(skill, "answer", run, "refunds", "{not json")
    fields = header(again)
    assert (fields["step"], fields["steps"], fields["attempt"]) == ("refunds", "refunds,sales", "2")
    assert "- not JSON" in again
    assert "--- sales ANSWER SHAPE (key? = optional) ---" in again
    assert "answer one step per command" in again


def test_two_inputs_of_one_step_are_not_several_steps(tmp_path: Path) -> None:
    rows = "pipeline.input.rows"
    two = _pipeline(_pull("sales", {"rows": rows, "again": rows}), _pull("refunds"))
    two["input"] = {"type": "In", "required": ["month", "rows"]}
    big = ["x" * MIN_SHARED_INPUT_CHARS]
    status, block = drive(
        _skill(tmp_path, two), "start", json.dumps({"month": "2026-10", "rows": big})
    )
    assert status == 0, block
    assert "INPUTS SEVERAL STEPS READ" not in block and "above\u00bb" not in block


def _with_gate() -> dict[str, Any]:
    """sales, refunds (one DO block) -> ask (a gate showing no result of theirs) -> report."""
    pipeline = _pipeline(_pull("sales"), _pull("refunds"))
    pipeline["nodes"].insert(
        2,
        {
            "id": "ask",
            "kind": "hitl_gate",
            "description": "Ask the manager to confirm the month.",
            "signal": "confirm",
            "timeout": "1h",
            "output": {"ok": "bool"},
            "inputs": {"month": "pipeline.input.month"},
            "notify": {"channel": "x", "target": "y", "message_template": "Close {month}?"},
        },
    )
    pipeline["edges"] += [
        {"from": "sales", "to": "ask"},
        {"from": "refunds", "to": "ask"},
        {"from": "ask", "to": "report"},
    ]
    return pipeline


def test_the_first_print_of_a_block_of_several_asks_the_person_too(tmp_path: Path) -> None:
    """Only `resume` printed the question: the block a run stops at did not,
    so the agent never asked and the run stopped again at the gate."""
    skill = _skill(tmp_path, _with_gate())
    block = _start(skill)
    assert header(block)["steps"] == "refunds,sales"
    assert "--- ALSO ASK THE PERSON (ask) ---" in block
    assert "--with ask '<their reply as JSON>'" in block
    _, again = drive(skill, "resume", header(block)["run"])
    assert again == block


def test_answering_in_parts_keeps_the_question_for_the_person(tmp_path: Path) -> None:
    """The agent asks while it works, and sends the reply with its last part."""
    skill = _skill(tmp_path, _with_gate())
    run = header(_start(skill))["run"]
    _, rest = drive(skill, "answer", run, "sales", json.dumps({"total": 900}))
    assert header(rest)["step"] == "refunds"
    assert "--- ALSO ASK THE PERSON (ask) ---" in rest
    status, done = drive(
        skill, "answer", run, "refunds", json.dumps({"total": 150}),
        "--with", "ask", json.dumps({"ok": True}), "--said", "yes, close it",
    )  # fmt: skip
    assert status == 0 and header(done)["kind"] == "DONE", done
    assert journal(done)["answers"]["ask"]["said"] == "yes, close it"


def test_a_reply_sent_with_the_first_part_is_not_asked_for_again(tmp_path: Path) -> None:
    skill = _skill(tmp_path, _with_gate())
    run = header(_start(skill))["run"]
    _, rest = drive(
        skill, "answer", run, "sales", json.dumps({"total": 900}),
        "--with", "ask", json.dumps({"ok": True}), "--said", "yes",
    )  # fmt: skip
    assert header(rest)["step"] == "refunds"
    assert "ALSO ASK THE PERSON" not in rest and "--with" not in rest
    _, done = drive(skill, "answer", run, "refunds", json.dumps({"total": 150}))
    assert header(done)["kind"] == "DONE", done


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


def test_the_attempt_cap_holds_across_the_steps_asked_again(pulse: Path) -> None:
    """The steps asked again carry the block's attempt count, so an answer
    that keeps failing ends the run instead of being asked forever."""
    block = _start(pulse)
    bad = {"refunds": {"total": 150}, "sales": {"total": "lots"}}
    again = _answer(pulse, block, bad)
    again = _answer(pulse, again, {"total": "still lots"})
    assert header(again)["attempt"] == "3"
    end = _answer(pulse, again, {"total": "no"})
    assert header(end)["kind"] == "ERROR", end
    assert "3 answers did not validate" in end
