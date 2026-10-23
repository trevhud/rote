"""The compiled-skill driver (``rote.skill_target._driver``), run for real.

No inference and no network beyond localhost: decisions are answered by
the test, as an agent would, and MCP steps call ``tests/_skill_mcp_server.py``
(real fastmcp, real HTTP), whose hit log is the evidence of what was called.
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import stat
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from rote.plan import judge_references, plan_hash
from rote.skill_target import HELPER_FILES, write_scripts
from tests._skill_driver import (
    FixtureServer,
    drive,
    fixture_server,
    header,
    journal,
    load_driver,
    make_skill,
    result_of,
    run_dir,
)

pytest.importorskip("fastmcp")

FAST = {"RETRY_BASE_DELAY_SECONDS": 0.0}

MATH = """\
def double(n, factor):
    return n * factor


def total(values, offset, label=None):
    return {"label": label, "sum": sum(values) + offset, "values": values}
"""


def _pipeline(nodes: list[dict[str, Any]], edges: list[tuple[str, str]], **kw: Any) -> dict:
    return {
        "name": kw.pop("name", "drv"),
        "version": "0.1.0",
        "input": kw.pop("input", {"type": "In", "required": ["items"], "optional": ["label"]}),
        "nodes": nodes,
        "edges": [{"from": a, "to": b} for a, b in edges],
        "entry_nodes": [nodes[0]["id"]],
        "exit_nodes": [nodes[-1]["id"]],
        **kw,
    }


CODE = _pipeline(
    [
        {
            "id": "doubled",
            "kind": "pure_function",
            "description": "Double each item.",
            "impl": "extracted/math.py:double",
            "fan_out": True,
            "inputs": {"n": "pipeline.input.items"},
            "constants": {"factor": 2},
        },
        {
            "id": "summed",
            "kind": "pure_function",
            "description": "Add them up.",
            "impl": "extracted/math.py:total",
            "inputs": {"values": "doubled.output", "label": "pipeline.input.label"},
            "constants": {"offset": 100},
        },
    ],
    [("doubled", "summed")],
)


@pytest.fixture
def code_skill(tmp_path: Path) -> Path:
    return make_skill(tmp_path, CODE, {"math.py": MATH})


# ───────── a run that needs nobody ─────────


def test_a_code_only_run_ends_in_done_with_the_result_on_disk(code_skill: Path) -> None:
    status, out = drive(code_skill, "start", '{"items": [1, 2, 3], "label": "x"}')
    assert status == 0, out
    assert header(out)["kind"] == "DONE"
    # Constants merged last, one call per element, results in order.
    assert result_of(out) == {"summed": {"label": "x", "sum": 112, "values": [2, 4, 6]}}
    assert 'summary: {"label": "x", "sum": 112, "values": <list, 3 items>}' in out
    state = journal(out)
    assert state["state"] == "done" and state["pending"] is None


def test_an_omitted_optional_input_is_left_out_not_none(code_skill: Path) -> None:
    """The impl's own default applies (``label=None`` here, but absent, not passed)."""
    (code_skill / "extracted" / "math.py").write_text(
        MATH.replace("label=None", 'label="default"'), encoding="utf-8"
    )
    _, out = drive(code_skill, "start", '{"items": [1]}')
    assert result_of(out)["summed"]["label"] == "default"


def test_the_run_folder_is_private_and_under_the_run_dir(code_skill: Path) -> None:
    _, out = drive(code_skill, "start", '{"items": [1]}')
    folder = run_dir(out)
    assert folder.parent.parent == Path(os.environ["ROTE_SKILL_RUN_DIR"])
    assert stat.S_IMODE(folder.stat().st_mode) == 0o700
    assert stat.S_IMODE(folder.parent.stat().st_mode) == 0o700
    assert sorted(p.name for p in (folder / "nodes").iterdir()) == ["doubled.json", "summed.json"]


@pytest.mark.parametrize(
    ("argument", "message"),
    [
        ('["not", "an", "object"]', "pipeline input must be a JSON object, got list"),
        ('{"label": "x"}', "pipeline input is missing required field(s): items"),
        ("{not json", "the input is not JSON (line 1 column 2"),
    ],
)
def test_a_bad_input_is_refused_before_any_folder_exists(
    code_skill: Path, argument: str, message: str
) -> None:
    status, out = drive(code_skill, "start", argument)
    assert status == 2
    assert header(out)["kind"] == "ERROR"
    assert message in out
    assert not Path(os.environ["ROTE_SKILL_RUN_DIR"]).exists()


def test_start_reads_the_input_from_a_file(code_skill: Path, tmp_path: Path) -> None:
    (tmp_path / "in.json").write_text('{"items": [5]}', encoding="utf-8")
    _, out = drive(code_skill, "start", "--file", str(tmp_path / "in.json"))
    assert result_of(out)["summed"]["sum"] == 110


def test_a_raising_function_is_an_error_naming_its_line_and_resume_retries(
    code_skill: Path,
) -> None:
    broken = MATH.replace("return n * factor", "raise ValueError('bad item')")
    (code_skill / "extracted" / "math.py").write_text(broken, encoding="utf-8")
    status, out = drive(code_skill, "start", '{"items": [1]}')
    assert status == 1
    assert "doubled raised ValueError: bad item at extracted/math.py:2" in out
    assert f"resume {header(out)['run']}" in out
    (code_skill / "extracted" / "math.py").write_text(MATH, encoding="utf-8")
    status, again = drive(code_skill, "resume", header(out)["run"])
    assert status == 0 and header(again)["kind"] == "DONE"


FLAKY_ELEMENT = """\
import pathlib

LOG = pathlib.Path({log!r})


def check(n):
    calls = LOG.read_text().split() if LOG.exists() else []
    LOG.write_text(" ".join(calls + [str(n)]))
    if n == 2 and calls.count("2") < 1:
        raise RuntimeError("element 2 failed once")
    return n * 10
"""


def _fanned(tmp_path: Path, retry: int) -> tuple[Path, Path]:
    log = tmp_path / "calls.txt"
    node: dict[str, Any] = {
        "id": "checked",
        "kind": "pure_function",
        "description": "Check each item.",
        "impl": "extracted/fan.py:check",
        "fan_out": True,
        "inputs": {"n": "pipeline.input.items"},
    }
    if retry:
        node["retry"] = {"max": retry, "backoff": "constant"}
    scripts = make_skill(
        tmp_path, _pipeline([node], []), {"fan.py": FLAKY_ELEMENT.format(log=str(log))}
    )
    return scripts, log


def test_a_fan_element_is_retried_alone(tmp_path: Path) -> None:
    """As in the python runtime: element 2's retry never re-runs 1 or 3."""
    scripts, log = _fanned(tmp_path, retry=1)
    status, out = drive(scripts, "start", '{"items": [1, 2, 3]}', patch=FAST)
    assert status == 0 and result_of(out) == {"checked": [10, 20, 30]}
    assert log.read_text().split() == ["1", "2", "2", "3"]
    assert not (run_dir(out) / "nodes" / "checked.parts").exists()  # folded into the node


def test_a_resume_after_a_mid_fan_failure_runs_only_the_unfinished_elements(
    tmp_path: Path,
) -> None:
    scripts, log = _fanned(tmp_path, retry=0)
    status, out = drive(scripts, "start", '{"items": [1, 2, 3]}', patch=FAST)
    assert status == 1 and "element 2 failed once" in out
    assert log.read_text().split() == ["1", "2"]
    parts = run_dir(out) / "nodes" / "checked.parts"
    assert sorted(p.name for p in parts.iterdir()) == ["0.json"]
    status, done = drive(scripts, "resume", header(out)["run"], patch=FAST)
    assert status == 0 and result_of(done) == {"checked": [10, 20, 30]}
    assert log.read_text().split() == ["1", "2", "2", "3"]  # element 1 ran once


def test_an_agent_loop_is_refused_at_start_not_half_run(tmp_path: Path) -> None:
    pipeline = _pipeline(
        [
            *CODE["nodes"],
            {
                "id": "explore",
                "kind": "agent_loop",
                "description": "Look around.",
                "tools": ["search"],
                "inputs": {"summary": "summed.output"},
            },
        ],
        [("doubled", "summed"), ("summed", "explore")],
    )
    scripts = make_skill(tmp_path, pipeline, {"math.py": MATH})
    status, out = drive(scripts, "start", '{"items": [1]}')
    assert status == 1
    assert "step explore is an agent loop" in out
    assert not Path(os.environ["ROTE_SKILL_RUN_DIR"]).exists()


# ───────── DECIDE: the agent's own model work ─────────

PREP = """\
def prep(notes):
    return {
        "items": [{"text": n, "secret": "UNREFERENCED-" + n} for n in notes],
        "context": "CONTEXT-NOT-IN-ANY-PROMPT",
    }


def tally(labels, headline):
    return {"kept": sum(1 for x in labels if x["label"] == "keep"), "headline": headline}
"""

LABEL = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["keep", "drop"]},
        "reason": {"type": "string"},
    },
    "required": ["label"],
    "additionalProperties": False,
}

DECIDE = _pipeline(
    [
        {
            "id": "prepped",
            "kind": "pure_function",
            "description": "Wrap the notes.",
            "impl": "extracted/prep.py:prep",
            "inputs": {"notes": "pipeline.input.notes"},
        },
        {
            "id": "classify",
            "kind": "llm_judge",
            "description": "Keep or drop each note.",
            "fan_out": True,
            "inputs": {"note": "prepped.output.items", "topic": "pipeline.input.topic"},
            "signature_spec": {
                "prompt": "Topic: {{ topic }}. Keep or drop this note: {{ note.text }}",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "note": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "secret": {"type": "string"},
                            },
                            "required": ["text"],
                        },
                        "topic": {"type": "string"},
                    },
                    "required": ["note", "topic"],
                },
                "output_schema": LABEL,
            },
        },
        {
            "id": "headline",
            "kind": "llm_judge",
            "description": "Write a headline.",
            "inputs": {"topic": "pipeline.input.topic", "context": "prepped.output.context"},
            "signature_spec": {
                "prompt": "Write a five-word headline about {{ topic }}.",
                "input_schema": {
                    "type": "object",
                    "properties": {"topic": {"type": "string"}, "context": {"type": "string"}},
                    "required": ["topic"],
                },
                "output_schema": {
                    "type": "object",
                    "properties": {"headline": {"type": "string"}},
                    "required": ["headline"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "id": "tallied",
            "kind": "pure_function",
            "description": "Count the keeps.",
            "impl": "extracted/prep.py:tally",
            "inputs": {"labels": "classify.output", "headline": "headline.output.headline"},
        },
    ],
    [
        ("prepped", "classify"),
        ("prepped", "headline"),
        ("classify", "tallied"),
        ("headline", "tallied"),
    ],
    input={"type": "In", "required": ["notes", "topic"]},
)


@pytest.fixture
def decide_skill(tmp_path: Path) -> Path:
    return make_skill(tmp_path, DECIDE, {"prep.py": PREP})


def _notes(n: int) -> str:
    return json.dumps({"notes": [f"note{i}" for i in range(n)], "topic": "cats"})


def _labels(ids: list[str], bad: set[str] = frozenset()) -> str:  # type: ignore[assignment]
    return json.dumps({i: {"label": "maybe" if i in bad else "keep"} for i in ids})


def test_a_fanned_judge_asks_in_batches_showing_only_what_its_prompt_reads(
    decide_skill: Path,
) -> None:
    status, out = drive(decide_skill, "start", _notes(12))
    assert status == 0
    head = header(out)
    assert (head["kind"], head["step"], head["batch"]) == ("DECIDE", "classify", "1/2")
    assert "Do not ask the user." in out.splitlines()[1]
    items = json.loads(out.split("--- ITEMS (json, by id) ---\n")[1].splitlines()[0])
    assert items == {str(i): {"text": f"note{i}"} for i in range(10)}
    shared = json.loads(out.split("--- SHARED (json) ---\n")[1].splitlines()[0])
    assert shared == {"topic": "cats"}
    # Negative half: a field no prompt reads never reaches the agent.
    assert "UNREFERENCED" not in out
    assert "CONTEXT-NOT-IN-ANY-PROMPT" not in out
    # Pinned size: the template, ten short items, the schema and the command.
    assert len(out.encode()) < 900, len(out.encode())


def test_a_bad_item_is_asked_again_alone_and_the_good_ones_are_kept(decide_skill: Path) -> None:
    _, out = drive(decide_skill, "start", _notes(12))
    run = header(out)["run"]
    first = [str(i) for i in range(10)]
    status, again = drive(
        decide_skill, "answer", run, "classify", _labels(first[:8] + ["9"], bad={"3"})
    )
    assert status == 0
    assert header(again)["attempt"] == "2"
    assert "- answer.3.label: 'maybe' is not one of ['keep', 'drop']" in again
    assert "missing items: 3, 8" in again
    # Nothing repeats: no items, no task, no schema.
    assert "--- ITEMS" not in again and "--- TASK" not in again and "SCHEMA" not in again
    assert len(again.encode()) < 300
    status, second = drive(decide_skill, "answer", run, "classify", _labels(["3", "8"]))
    assert header(second)["batch"] == "2/2"
    assert json.loads(second.split("--- ITEMS (json, by id) ---\n")[1].splitlines()[0]) == {
        "10": {"text": "note10"},
        "11": {"text": "note11"},
    }
    _, third = drive(decide_skill, "answer", run, "classify", _labels(["10", "11"]))
    assert header(third)["step"] == "headline"
    # The unfanned judge's TASK is its prompt rendered, nothing else.
    assert "--- TASK ---\nWrite a five-word headline about cats.\n" in third
    assert "CONTEXT-NOT-IN-ANY-PROMPT" not in third
    _, done = drive(decide_skill, "answer", run, "headline", '{"headline": "Cats rule"}')
    assert result_of(done) == {"tallied": {"kept": 12, "headline": "Cats rule"}}
    node = json.loads((run_dir(done) / "nodes" / "classify.json").read_text())
    # Projected like the emitted model: an omitted optional key is None.
    assert node["result"][0] == {"label": "keep", "reason": None}
    assert node["answered_by"] == "agent"
    assert journal(done)["answers"]["classify"]["answered_by"] == "agent"


def test_three_bad_answers_end_in_an_error_that_resume_reopens(decide_skill: Path) -> None:
    _, out = drive(decide_skill, "start", _notes(2))
    run = header(out)["run"]
    for attempt in (2, 3):
        _, again = drive(decide_skill, "answer", run, "classify", "{broken")
        assert header(again)["attempt"] == str(attempt)
        assert "not JSON (line 1 column 2" in again and "--file <path>" in again
    status, error = drive(decide_skill, "answer", run, "classify", '{"0": {}}')
    assert status == 1 and header(error)["kind"] == "ERROR"
    assert "3 answers did not validate" in error
    _, reopened = drive(decide_skill, "resume", run)
    assert header(reopened)["kind"] == "DECIDE"
    assert journal(reopened)["attempts"]["classify"] == 0


def test_an_answer_for_a_step_already_done_changes_nothing(decide_skill: Path) -> None:
    _, out = drive(decide_skill, "start", _notes(1))
    run = header(out)["run"]
    _, nxt = drive(decide_skill, "answer", run, "classify", _labels(["0"]))
    before = (run_dir(nxt) / "nodes" / "classify.json").read_text()
    _, again = drive(decide_skill, "answer", run, "classify", _labels(["0"], bad={"0"}))
    assert again.splitlines()[1] == "Note: classify is already answered; nothing changed."
    assert header(again)["step"] == "headline"  # the pending block, again
    assert (run_dir(nxt) / "nodes" / "classify.json").read_text() == before
    _, wrong = drive(decide_skill, "answer", run, "tallied", "{}")
    assert wrong.splitlines()[1] == "Note: tallied is not the pending step; nothing changed."


def test_said_is_refused_on_an_agent_answer(decide_skill: Path) -> None:
    _, out = drive(decide_skill, "start", _notes(1))
    run = header(out)["run"]
    status, refused = drive(decide_skill, "answer", run, "classify", _labels(["0"]), "--said", "x")
    assert status == 2
    assert "classify is yours to answer; --said is only for a person's words" in refused
    assert journal(out)["attempts"]["classify"] == 0  # a refused command is not an attempt


def test_a_judge_input_its_schema_refuses_is_an_error_not_a_question(tmp_path: Path) -> None:
    """The emitted judge's input model would reject it, so the agent is never asked."""
    closed = json.loads(json.dumps(DECIDE))
    note = closed["nodes"][1]["signature_spec"]["input_schema"]["properties"]["note"]
    note["properties"].pop("secret")
    note["additionalProperties"] = False
    scripts = make_skill(tmp_path, closed, {"prep.py": PREP})
    status, out = drive(scripts, "start", _notes(2))
    assert status == 1 and header(out)["kind"] == "ERROR"
    assert header(out)["step"] == "classify" and header(out)["run"]
    assert "classify's input does not fit its input schema: input[0].note: unexpected key" in out
    assert journal(out)["state"] == "error"


def test_status_reprints_the_pending_block_and_nothing_else(decide_skill: Path) -> None:
    _, out = drive(decide_skill, "start", _notes(3))
    status, again = drive(decide_skill, "status", header(out)["run"])
    assert status == 0 and again == out
    _, latest = drive(decide_skill, "status")
    assert latest == out


# ───────── HUMAN: a person, through the agent ─────────

GATED = """\
def table(notes):
    return {"table": "| " + " | ".join(notes) + " |", "rows": len(notes), "big": "X" * 5000}


def finish(approval):
    return {"approved": approval["approved"], "by": approval["approver"]}
"""


def _gated(inputs: dict[str, str] | None) -> dict:
    gate: dict[str, Any] = {
        "id": "approve",
        "kind": "hitl_gate",
        "description": "Approve the list before anything is sent.",
        "signal": "approved",
        "timeout": "1h",
        "output": {"approved": "bool", "approver": "str"},
        "notify": {
            "channel": "slack",
            "target": "#ops",
            "message_template": "Review: {table}",
        },
    }
    if inputs is not None:
        gate["inputs"] = inputs
    return _pipeline(
        [
            {
                "id": "tabled",
                "kind": "pure_function",
                "description": "Tabulate.",
                "impl": "extracted/gate.py:table",
                "inputs": {"notes": "pipeline.input.notes"},
            },
            gate,
            {
                "id": "finished",
                "kind": "pure_function",
                "description": "Finish.",
                "impl": "extracted/gate.py:finish",
                "inputs": {"approval": "approve.output"},
            },
        ],
        [("tabled", "approve"), ("approve", "finished")],
        input={"type": "In", "required": ["notes"]},
    )


@pytest.fixture
def gate_skill(tmp_path: Path) -> Path:
    inputs = {"table": "tabled.output.table", "rows": "tabled.output.rows"}
    return make_skill(tmp_path, _gated(inputs), {"gate.py": GATED})


def test_a_gate_asks_a_person_through_the_agent_with_only_its_inputs(gate_skill: Path) -> None:
    status, out = drive(gate_skill, "start", '{"notes": ["a", "b"]}')
    assert status == 0
    head = header(out)
    assert (head["kind"], head["step"], head["of"]) == ("HUMAN", "approve", "gate")
    assert out.splitlines()[1].startswith("Do not answer this yourself. Ask the person")
    assert "Review: | a | b |" in out  # the message, its declared input filled in
    # What the message already shows is not shown again; the rest is.
    assert out.count("| a | b |") == 1
    assert '--- THEY SEE (json) ---\n{"rows":2}' in out
    assert "XXXX" not in out  # an undeclared upstream field never shows
    assert "--said '<their words, verbatim>'" in out
    assert len(out.encode()) < 900, len(out.encode())


def test_a_gate_with_no_inputs_shows_only_its_question(tmp_path: Path) -> None:
    scripts = make_skill(tmp_path, _gated(None), {"gate.py": GATED})
    _, out = drive(scripts, "start", '{"notes": ["a"]}')
    assert "THEY SEE" not in out and "| a |" not in out
    assert "Review: {table}" in out  # nothing declared to fill it with


def test_a_gate_answer_needs_the_persons_words_and_records_them(gate_skill: Path) -> None:
    _, out = drive(gate_skill, "start", '{"notes": ["a"]}')
    run = header(out)["run"]
    answer = '{"approved": true, "approver": "Dana"}'
    status, refused = drive(gate_skill, "answer", run, "approve", answer)
    assert status == 2
    assert "approve is answered by a person: pass their words" in refused
    assert journal(out)["attempts"]["approve"] == 0
    status, done = drive(gate_skill, "answer", run, "approve", answer, "--said", "yes, ship it")
    assert status == 0 and result_of(done) == {"finished": {"approved": True, "by": "Dana"}}
    logged = journal(done)["answers"]["approve"]
    assert logged["answered_by"] == "human" and logged["said"] == "yes, ship it"
    assert logged["at"] > 0
    node = json.loads((run_dir(done) / "nodes" / "approve.json").read_text())
    assert (node["answered_by"], node["said"]) == ("human", "yes, ship it")


def test_a_gate_answer_that_does_not_fit_asks_the_person_again(gate_skill: Path) -> None:
    _, out = drive(gate_skill, "start", '{"notes": ["a"]}')
    run = header(out)["run"]
    _, again = drive(gate_skill, "answer", run, "approve", '{"approved": "yes"}', "--said", "yes")
    assert header(again)["kind"] == "HUMAN" and header(again)["attempt"] == "2"
    assert "- answer.approved: expected boolean, got string" in again
    assert "- answer: missing key 'approver'" in again
    assert "THEY SEE" not in again and "QUESTION" not in again


def test_a_single_hand_written_gate_file_is_refused(gate_skill: Path) -> None:
    """A gate result in nodes/ that run.json's answer log does not match is
    refused. This is a consistency check, not a barrier: a process running as
    the user can write both files, and the journal is an audit trail."""
    _, out = drive(gate_skill, "start", '{"notes": ["a"]}')
    run = header(out)["run"]
    forged = {"result": {"approved": True, "approver": "me"}, "answered_by": "human", "said": "ok"}
    (run_dir(out) / "nodes" / "approve.json").write_text(json.dumps(forged), encoding="utf-8")
    state = journal(out)
    state["pending"] = None
    (run_dir(out) / "run.json").write_text(json.dumps(state), encoding="utf-8")
    status, refused = drive(gate_skill, "resume", run)
    assert status == 1
    assert "only a person may answer, and no person's words are recorded" in refused
    assert not (run_dir(out) / "result.json").exists()


def test_a_late_gate_answer_ends_the_run(gate_skill: Path) -> None:
    _, out = drive(gate_skill, "start", '{"notes": ["a"]}')
    run = header(out)["run"]
    later = time.time() + 3601
    status, late = drive(
        gate_skill,
        "answer",
        run,
        "approve",
        '{"approved": true, "approver": "D"}',
        "--said",
        "yes",
        patch={"_now": lambda: later},
    )
    assert status == 1 and "that deadline has passed" in late
    assert journal(out)["state"] == "expired"
    status, before = drive(gate_skill, "status", run, patch={"_now": lambda: later})
    assert "deadline passed" in before


# ───────── the journal: one writer, one program ─────────


def test_a_reemitted_plan_ends_the_run(gate_skill: Path) -> None:
    _, out = drive(gate_skill, "start", '{"notes": ["a"]}')
    plan = gate_skill / "plan.json"
    plan.write_text(plan.read_text().replace("Approve the list", "Approve it"), encoding="utf-8")
    status, refused = drive(gate_skill, "status", header(out)["run"])
    assert status == 1 and "re-emitted after the run started" in refused
    assert journal(out)["plan_hash"] != plan_hash(plan.read_text())


def _holding(lock: Path) -> subprocess.Popen[str]:
    """A process that holds ``lock``'s flock, as a call running steps does."""
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import fcntl, sys, time\n"
            "f = open(sys.argv[1], 'a+')\n"
            "fcntl.flock(f.fileno(), fcntl.LOCK_EX)\n"
            "print('held', flush=True)\n"
            "time.sleep(60)\n",
            str(lock),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout is not None and proc.stdout.readline().strip() == "held"
    return proc


def _between_steps(skill: Path) -> tuple[str, Path, dict[str, Any]]:
    """A run whose first step failed and was fixed, ready to resume."""
    broken = MATH.replace("return n * factor", "raise ValueError('x')")
    (skill / "extracted" / "math.py").write_text(broken, encoding="utf-8")
    _, out = drive(skill, "start", '{"items": [1]}')
    (skill / "extracted" / "math.py").write_text(MATH, encoding="utf-8")
    state = journal(out)
    state.update(state="running", error=None)
    return header(out)["run"], run_dir(out), state


def test_a_run_claimed_by_a_live_call_is_busy_and_a_dead_ones_is_taken_over(
    code_skill: Path,
) -> None:
    run, folder, state = _between_steps(code_skill)
    holder = _holding(folder / "claim.lock")
    try:
        state["wave"] = {"index": 0, "pid": holder.pid, "since": time.time()}
        (folder / "run.json").write_text(json.dumps(state), encoding="utf-8")
        status, busy = drive(code_skill, "resume", run)
        assert status == 1 and f"run busy: another call (pid {holder.pid})" in busy
        assert not (folder / "result.json").exists()
    finally:
        holder.kill()
        holder.wait()
    # The holder exited, so the OS dropped its lock: the run is taken over.
    status, done = drive(code_skill, "resume", run)
    assert status == 0 and result_of(done)["summed"]["sum"] == 102
    assert journal(done)["wave"] is None


def test_a_reused_pid_does_not_read_as_busy(code_skill: Path) -> None:
    """The journal names a pid that is alive but is not the holder (the OS gave
    a dead holder's pid to another process): the free lock says the holder
    is gone, so the run goes on."""
    run, folder, state = _between_steps(code_skill)
    stranger = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        state["wave"] = {"index": 0, "pid": stranger.pid, "since": time.time()}
        (folder / "run.json").write_text(json.dumps(state), encoding="utf-8")
        status, done = drive(code_skill, "resume", run)
    finally:
        stranger.kill()
        stranger.wait()
    assert status == 0 and result_of(done)["summed"]["sum"] == 102


def test_the_journal_lock_waits_briefly_then_says_busy(code_skill: Path) -> None:
    import fcntl

    _, out = drive(code_skill, "start", '{"items": [1]}')
    with open(run_dir(out) / "run.lock", "a+") as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX)
        started = time.monotonic()
        status, busy = drive(code_skill, "status", header(out)["run"])
        waited = time.monotonic() - started
    assert status == 1 and "run busy: another call holds its journal" in busy
    assert 0.9 < waited < 3.0


def test_two_concurrent_answers_leave_one_journal_entry(decide_skill: Path) -> None:
    _, out = drive(decide_skill, "start", _notes(1))
    run = header(out)["run"]
    driver = decide_skill / "_rote_skill.py"
    procs = [
        subprocess.Popen(
            [sys.executable, "-I", str(driver), "answer", run, "classify", _labels(["0"])],
            stdout=subprocess.PIPE,
            text=True,
            env={**os.environ},
        )
        for _ in range(2)
    ]
    outs = [p.communicate(timeout=60)[0] for p in procs]
    kinds = sorted(header(o)["kind"] for o in outs)
    assert kinds in (["DECIDE", "DECIDE"], ["DECIDE", "ERROR"]), outs
    assert any("already answered" in o or "run busy" in o for o in outs), outs
    state = journal(out)
    assert list(state["answers"]) == ["classify"]
    assert state["attempts"]["classify"] == 1


def test_an_answer_holds_the_run_until_its_next_block(decide_skill: Path) -> None:
    """Deterministic half of the race above: the answer that is recorded and the
    claim on the run land under one lock, so no second call can find the run
    between this answer and the block it leads to."""
    _, out = drive(decide_skill, "start", _notes(1))
    run = header(out)["run"]
    seen: list[Any] = []
    saved_path, saved_env = list(sys.path), dict(os.environ)
    try:
        module = load_driver(decide_skill)

        class Watching(module.Driver):  # type: ignore[name-defined,misc]
            def advance(self, path: Path) -> Any:
                seen.append(json.loads((path / "run.json").read_text())["wave"])
                return super().advance(path)

        module.Driver = Watching
        import contextlib
        import io

        with contextlib.redirect_stdout(io.StringIO()):
            module.main(["answer", run, "classify", _labels(["0"])], decide_skill)
    finally:
        sys.path[:] = saved_path
        os.environ.clear()
        os.environ.update(saved_env)
    assert seen and seen[0]["pid"] == os.getpid()
    assert journal(out)["wave"] is None  # released once the block was printed


def test_cancel_ends_the_run_and_later_calls_see_it(code_skill: Path, gate_skill: Path) -> None:
    _, out = drive(gate_skill, "start", '{"notes": ["a"]}')
    run = header(out)["run"]
    status, cancelled = drive(gate_skill, "cancel", run)
    assert status == 0 and "cancelled" in cancelled.splitlines()[0]
    _, after = drive(gate_skill, "answer", run, "approve", "{}", "--said", "x")
    assert "not the pending step" in after and journal(out)["state"] == "cancelled"


def test_the_sweep_removes_only_expired_runs_this_driver_wrote(code_skill: Path) -> None:
    _, old = drive(code_skill, "start", '{"items": [1]}')
    _, recent = drive(code_skill, "start", '{"items": [2]}')
    state = journal(old)
    state["updated"] = time.time() - 8 * 86400
    (run_dir(old) / "run.json").write_text(json.dumps(state), encoding="utf-8")
    skill_dir = run_dir(old).parent
    foreign = skill_dir / "notes"
    foreign.mkdir()
    (foreign / "keep.txt").write_text("mine", encoding="utf-8")
    other = skill_dir / "aaaaaaaaaaaa"
    other.mkdir()
    (other / "run.json").write_text('{"format": "something-else", "updated": 0}')
    status, listing = drive(code_skill, "runs", "--prune")
    assert status == 0 and "pruned 1" in listing
    assert not run_dir(old).exists()
    assert run_dir(recent).exists() and foreign.exists() and other.exists()


def test_a_waiting_run_is_swept_only_after_its_deadline(gate_skill: Path) -> None:
    _, out = drive(gate_skill, "start", '{"notes": ["a"]}')
    drive(gate_skill, "runs", "--prune")
    assert run_dir(out).exists()
    later = time.time() + 3600 + 86400 + 5
    drive(gate_skill, "runs", "--prune", patch={"_now": lambda: later})
    assert not run_dir(out).exists()


def _expired_run(folder: Path) -> None:
    folder.mkdir(parents=True)
    state = {"format": "rote-skill-run/1", "state": "done", "updated": 0}
    (folder / "run.json").write_text(json.dumps(state), encoding="utf-8")
    (folder / "precious.txt").write_text("keep me", encoding="utf-8")


def test_the_sweep_never_follows_a_symlink_and_survives_bad_entries(
    code_skill: Path, tmp_path: Path
) -> None:
    _, first = drive(code_skill, "start", '{"items": [1]}')
    skill_dir = run_dir(first).parent
    # An expired-looking run outside the run dir, reached two ways by symlink.
    outside = tmp_path / "elsewhere" / "bbbbbbbbbbbb"
    _expired_run(outside)
    (skill_dir / "cccccccccccc").symlink_to(outside, target_is_directory=True)
    linked_journal = skill_dir / "dddddddddddd"
    linked_journal.mkdir()
    (linked_journal / "run.json").symlink_to(outside / "run.json")
    # Entries that cannot be read as a journal.
    (skill_dir / "eeeeeeeeeeee").mkdir()
    (skill_dir / "eeeeeeeeeeee" / "run.json").write_text('{"updated": "soon"}')
    (skill_dir / "ffffffffffff").mkdir()
    (skill_dir / "ffffffffffff" / "run.json").write_text("[1, 2]")
    (skill_dir / "a-file").write_text("not a dir")
    _expired_run(skill_dir / "aaaaaaaaaaaa")  # a real expired run, for contrast
    status, out = drive(code_skill, "start", '{"items": [2]}')  # start sweeps
    assert status == 0 and header(out)["kind"] == "DONE"
    assert (outside / "precious.txt").read_text() == "keep me"
    assert (skill_dir / "cccccccccccc").is_symlink() and linked_journal.exists()
    assert not (skill_dir / "aaaaaaaaaaaa").exists()


# ───────── MCP, live, against a real fastmcp server ─────────


@pytest.fixture(scope="module")
def server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[FixtureServer]:
    with fixture_server(tmp_path_factory.mktemp("mcp") / "hits.jsonl") as s:
        yield s


@pytest.fixture
def fx(server: FixtureServer, monkeypatch: pytest.MonkeyPatch) -> FixtureServer:
    server.reset()
    monkeypatch.setenv("ROTE_MCP_FX_URL", server.url)
    return server


PICK = """\
def pick(found):
    return {"ids": [row["title"] for row in found]}


def finish(scored):
    return {"best": max(scored, key=lambda s: s["score"])["id"], "n": len(scored)}
"""


def _mcp_pipeline(tool: str = "search", retry: int = 0) -> dict:
    found: dict[str, Any] = {
        "id": "found",
        "kind": "external_call",
        "description": "Search.",
        "mcp": {
            "server": "fx",
            "tool": tool,
            "args": {"query": "q"} if tool != "flaky" else {"key": "q"},
        },
        "inputs": {"q": "pipeline.input.q"},
    }
    if retry:
        found["retry"] = {"max": retry, "backoff": "exponential"}
    return _pipeline(
        [
            found,
            {
                "id": "picked",
                "kind": "pure_function",
                "description": "Pick ids.",
                "impl": "extracted/pick.py:pick",
                "inputs": {"found": "found.output"},
            },
            {
                "id": "scored",
                "kind": "external_call",
                "description": "Score each.",
                "fan_out": True,
                "mcp": {"server": "fx", "tool": "lookup", "args": {"id": "id"}},
                "inputs": {"id": "picked.output.ids"},
            },
            {
                "id": "finished",
                "kind": "pure_function",
                "description": "Best.",
                "impl": "extracted/pick.py:finish",
                "inputs": {"scored": "scored.output"},
            },
        ],
        [("found", "picked"), ("picked", "scored"), ("scored", "finished")],
        input={"type": "In", "required": ["q"]},
    )


def test_tool_steps_run_live_and_a_resume_never_repeats_a_finished_call(
    tmp_path: Path, fx: FixtureServer
) -> None:
    broken = PICK.replace(
        'return {"best"', "raise RuntimeError('later step broke')\n    return {\"best\""
    )
    scripts = make_skill(tmp_path, _mcp_pipeline(), {"pick.py": broken})
    status, out = drive(scripts, "start", '{"q": "owl"}')
    assert status == 1 and "later step broke" in out
    assert [c["arguments"] for c in fx.calls("search")] == [{"query": "owl"}]
    # One lookup per element, the fan's arguments built from mcp.args.
    assert sorted(c["arguments"]["id"] for c in fx.calls("lookup")) == ["owl 0", "owl 1", "owl 2"]
    (scripts / "extracted" / "pick.py").write_text(PICK, encoding="utf-8")
    status, done = drive(scripts, "resume", header(out)["run"])
    assert status == 0 and result_of(done) == {"finished": {"best": "owl 0", "n": 3}}
    assert len(fx.calls("search")) == 1 and len(fx.calls("lookup")) == 3  # nothing repeated


def test_a_failing_tool_is_retried_per_the_ir(tmp_path: Path, fx: FixtureServer) -> None:
    scripts = make_skill(tmp_path, _mcp_pipeline("flaky", retry=1), {"pick.py": PICK})
    _, out = drive(scripts, "start", '{"q": "k1"}', patch=FAST)
    assert len(fx.calls("flaky")) == 2  # failed once, answered on the retry
    assert (run_dir(out) / "nodes" / "found.json").exists()
    fx.reset()
    scripts2 = make_skill(tmp_path / "b", _mcp_pipeline("flaky", retry=0), {"pick.py": PICK})
    status, failed = drive(scripts2, "start", '{"q": "k2"}', patch=FAST)
    assert status == 1 and len(fx.calls("flaky")) == 1  # negative half: no retry declared


def test_a_missing_endpoint_is_never_retried(
    tmp_path: Path, fx: FixtureServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ROTE_MCP_FX_URL")
    scripts = make_skill(tmp_path, _mcp_pipeline(retry=3), {"pick.py": PICK})
    progress: list[str] = []
    status, out = drive(scripts, "start", '{"q": "x"}', patch=_recording(progress))
    assert status == 1
    assert "no endpoint for MCP server 'fx': set ROTE_MCP_FX_URL" in out
    assert progress == ["calling fx.search"]  # one attempt of four, no retry line


def _recording(progress: list[str]) -> dict[str, Any]:
    return {"RETRY_BASE_DELAY_SECONDS": 0.0, "_progress": progress.append}


def test_a_write_tool_is_refused_before_it_is_called(tmp_path: Path, fx: FixtureServer) -> None:
    pipeline = _pipeline(
        [
            {
                "id": "noted",
                "kind": "external_call",
                "description": "Write.",
                "mcp": {"server": "fx", "tool": "write_note", "args": {"text": "q"}},
                "inputs": {"q": "pipeline.input.q"},
            }
        ],
        [],
        input={"type": "In", "required": ["q"]},
    )
    scripts = make_skill(tmp_path, pipeline)
    status, out = drive(scripts, "start", '{"q": "hi"}')
    assert status == 1 and "fx.write_note may write" in out
    assert "Fix: rote skill trust drv --allow-writes" in out
    assert fx.calls("write_note") == []
    # The environment cannot grant consent: the agent's shell sets it.
    for key, value in (
        ("ROTE_SKILL_ALLOW_WRITES", "1"),
        ("ROTE_MCP_READ_ONLY", "0"),
        ("ROTE_MCP_READ_ONLY_SERVERS", json.dumps({"fx": fx.url})),
    ):
        os.environ[key] = value
    try:
        status, again = drive(scripts, "resume", header(out)["run"])
    finally:
        for key in ("ROTE_SKILL_ALLOW_WRITES", "ROTE_MCP_READ_ONLY", "ROTE_MCP_READ_ONLY_SERVERS"):
            del os.environ[key]
    assert status == 1 and "may write" in again and fx.calls("write_note") == []
    # Consent arrives only as main()'s parameter (rote skill run, from the trust record).
    status, done = drive(scripts, "resume", header(out)["run"], allow_writes=True)
    assert status == 0 and result_of(done) == {"noted": {"written": "hi"}}


def test_a_401_waits_for_a_login_and_resume_runs_the_step(
    tmp_path: Path, fx: FixtureServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.test_mcp_client import _StatusServer

    denied = _StatusServer(401)
    try:
        monkeypatch.setenv("ROTE_MCP_FX_URL", denied.url)
        scripts = make_skill(tmp_path, _mcp_pipeline(retry=2), {"pick.py": PICK})
        progress: list[str] = []
        status, out = drive(scripts, "start", '{"q": "owl"}', patch=_recording(progress))
    finally:
        denied.close()
    assert status == 0
    assert progress == ["calling fx.search"]  # a login is never retried
    assert out.splitlines()[0].endswith("step=found kind=login server=fx")
    assert out.splitlines()[1] == "Do not answer this yourself. The person must log in to fx:"
    assert "rote mcp login fx --no-wait" in out
    run = header(out)["run"]
    status, refused = drive(scripts, "answer", run, "found", "{}")
    assert status == 2 and "waits for a login, not an answer" in refused
    monkeypatch.setenv("ROTE_MCP_FX_URL", fx.url)  # the login landed
    status, done = drive(scripts, "resume", run)
    assert status == 0 and result_of(done)["finished"]["n"] == 3


def test_the_driver_and_the_prefix_runner_agree_on_docs_digest(
    tmp_path: Path, fx: FixtureServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plan parity: one plan builder, two interpreters, the same node results.

    The prefix runner replays the recorded calls; the driver makes them live
    against a server answering with those recordings."""
    from rote import demo
    from rote.ir import load_pipeline
    from tests.test_docs_digest_demo import _replay

    compiled = tmp_path / "compiled"
    shutil.copytree(demo.docs_digest_dir() / "compiled", compiled)
    ref = demo.load_reference()
    replayed = _replay(compiled, ref)
    monkeypatch.setenv("ROTE_MCP_MSLEARN_URL", fx.url)
    monkeypatch.setenv("ROTE_MCP_DEEPWIKI_URL", fx.url)
    pipeline = load_pipeline(compiled / "pipeline.yaml")
    modules = {p.name: p.read_text() for p in (compiled / "extracted").glob("*.py")}
    scripts = make_skill(tmp_path / "skill", pipeline, modules)
    status, out = drive(scripts, "start", json.dumps(ref.input))
    assert status == 0, out
    nodes = run_dir(out) / "nodes"
    for node in pipeline.nodes:
        live = json.loads((nodes / f"{node.id}.json").read_text())["result"]
        assert replayed[node.id]["status"] == "ok"
        if "output" in replayed[node.id]:
            assert live == replayed[node.id]["output"], node.id
    assert result_of(out) == ref.pipeline_output
    # DONE is a path and a bounded summary, never the 4 KB digest itself.
    assert 'summary: {"digest": <text, ' in out
    assert len(out.encode()) < 400 + len(str(run_dir(out))) + 100, len(out.encode())
    # The driver sent exactly the recorded arguments.
    assert [c["arguments"] for c in fx.calls("microsoft_docs_search")] == [
        c.arguments for c in ref.tool_calls if c.tool == "microsoft_docs_search"
    ]


# ───────── what the driver is: verbatim, stdlib, fast ─────────


def test_the_scripts_are_verbatim_copies(code_skill: Path) -> None:
    for name, source in HELPER_FILES.items():
        assert (code_skill / name).read_bytes() == source.read_bytes(), name


STDLIB = set(sys.stdlib_module_names)


@pytest.mark.parametrize("name", ["_rote_skill.py", "_rote_schema.py"])
def test_module_level_imports_are_stdlib_only(name: str) -> None:
    tree = ast.parse(HELPER_FILES[name].read_text(encoding="utf-8"))
    imported = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported and imported <= STDLIB | {"__future__"}, imported - STDLIB


#: Wall clock for one driver call on a code-only plan, python start-up included.
#: Measured at 0.05 s on an M-series laptop (bare `python -I -c pass`: 0.02 s); the
#: budget is six times that, for CI. An eager fastmcp import (0.16 s warm here) is caught
#: by the module check in the same test, not by the clock.
STARTUP_BUDGET_SECONDS = 0.3


def test_a_call_starts_fast_and_loads_no_heavy_module(code_skill: Path) -> None:
    probe = (
        "import sys, runpy\n"
        "sys.argv = [sys.argv[1], *sys.argv[2:]]\n"
        "try:\n"
        "    runpy.run_path(sys.argv[0], run_name='__main__')\n"
        "except SystemExit:\n"
        "    pass\n"
        "heavy = sorted(m for m in sys.modules if m.split('.')[0] in "
        "{'fastmcp', 'mcp', 'httpx', 'pydantic', 'yaml', 'rote', 'asyncio', 'concurrent'})\n"
        "print('HEAVY', heavy, file=sys.stderr)\n"
    )
    driver = str(code_skill / "_rote_skill.py")
    timings = []
    for _ in range(3):
        started = time.perf_counter()
        done = subprocess.run(
            [sys.executable, "-I", driver, "start", '{"items": [1, 2]}'],
            capture_output=True,
            text=True,
            env={**os.environ},
            timeout=30,
        )
        timings.append(time.perf_counter() - started)
        assert done.returncode == 0, done.stdout + done.stderr
    assert min(timings) < STARTUP_BUDGET_SECONDS, timings
    done = subprocess.run(
        [sys.executable, "-I", "-c", probe, driver, "start", '{"items": [1]}'],
        capture_output=True,
        text=True,
        env={**os.environ},
        timeout=30,
    )
    assert "HEAVY []" in done.stderr, done.stderr


def test_the_done_summary_is_bounded() -> None:
    from rote.skill_target._driver import DONE_SUMMARY_BYTES, summarize

    big = {"r": {f"key_{i}": "v" * 50 for i in range(40)} | {"rows": list(range(500))}}
    text = summarize(big)
    assert len(text.encode()) <= DONE_SUMMARY_BYTES
    assert text.startswith('{"key_0": "') and "more keys)" in text
    assert summarize({"r": {"digest": "x" * 5000}}) == '{"digest": <text, 5000 chars>}'
    assert summarize({"a": 1, "b": [1, 2]}) == '{"a": 1, "b": <list, 2 items>}'


def test_interpolate_matches_the_emitted_judge_on_every_example_prompt() -> None:
    import types

    import yaml

    from rote.adapters._py_common import emit_signature_module
    from rote.ir import NodeKind, Pipeline
    from rote.skill_target._driver import interpolate

    checked = 0
    for path in sorted(
        (Path(__file__).resolve().parent.parent / "examples").rglob("pipeline.yaml")
    ):
        if "runtime" in path.parts:
            continue
        pipeline = Pipeline.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
        for node in pipeline.nodes:
            if node.kind is not NodeKind.LLM_JUDGE or node.signature_spec is None:
                continue
            source = emit_signature_module(
                node,
                anthropic_default_model="m",
                openai_default_model="m",
                generated_by="t",
                regen_command="t",
                context_note="t",
            )
            module = types.ModuleType("_interp")
            exec(compile(source, "_interp", "exec"), module.__dict__)
            prompt = node.signature_spec.prompt
            values = _values_for(judge_references(prompt))
            assert interpolate(prompt, values) == module._interpolate(prompt, values)
            if values:
                # A missing field raises the same KeyError on both.
                gone = dict(list(values.items())[1:])
                with pytest.raises(KeyError) as theirs:
                    module._interpolate(prompt, gone)
                with pytest.raises(KeyError) as ours:
                    interpolate(prompt, gone)
                assert ours.value.args == theirs.value.args
            checked += 1
    assert checked >= 20


#: Leaf values of every kind the renderer treats differently.
_LEAVES = ["text", None, 3, [1, "two"], {"k": "v"}, True]


def _values_for(references: list[str]) -> dict[str, Any]:
    """Input values under which every dotted reference resolves."""
    values: dict[str, Any] = {}
    for n, ref in enumerate(references):
        *parents, leaf = ref.split(".")
        here = values
        for part in parents:
            if not isinstance(here.get(part), dict):
                here[part] = {}
            here = here[part]
        here.setdefault(leaf, _LEAVES[n % len(_LEAVES)])
    return values


def test_the_input_and_fan_messages_match_the_emitted_helpers() -> None:
    from rote.adapters._py_common import _FAN_OUT_LIST_HELPER_PY, _MISSING_INPUTS_HELPER_PY
    from rote.skill_target._driver import _fan_list, missing_inputs_error

    emitted: dict[str, Any] = {}
    exec(_FAN_OUT_LIST_HELPER_PY + _MISSING_INPUTS_HELPER_PY, emitted)
    for value in ([], {"a": 1}, None, 3):
        assert missing_inputs_error(value, ["a", "b"]) == emitted["_missing_inputs_error"](
            value, ("a", "b")
        )
    with pytest.raises(TypeError) as ours:
        _fan_list({"x": 1}, "node", "items")
    with pytest.raises(TypeError) as theirs:
        emitted["_fan_out_list"]({"x": 1}, "node", "items")
    assert str(ours.value) == str(theirs.value)


def test_write_scripts_writes_the_plan_the_driver_hashes(tmp_path: Path) -> None:
    from rote.ir import Pipeline
    from rote.plan import build_plan

    plan = build_plan(Pipeline.model_validate(CODE), mcp="live")
    driver = write_scripts(tmp_path / "s", plan)
    module = load_driver(tmp_path / "s")
    loaded = module.Driver(tmp_path / "s")
    assert loaded.plan == json.loads(json.dumps(plan))
    assert loaded.plan_hash == plan_hash((tmp_path / "s" / "plan.json").read_text())
    assert driver.name == "_rote_skill.py"
