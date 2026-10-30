"""A compiled skill's DO steps: an external call with no code, done by the agent.

The compile leaves a vendor call it could not write as a stub. On every other
runtime that stub stops the run; a compiled skill hands it to the agent that
loaded it instead, which does the step with its own tools, as the source
skill had it do, and answers. No inference: the test is the agent.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from rote.adapters import get_adapter
from rote.cli import main
from rote.ir import load_pipeline
from tests._skill_driver import drive, header, journal, make_skill, result_of, run_dir

TEXT = """\
def normalize(account):
    return {"name": account.strip().title(), "raw": account}


def summarize(name, note_id):
    return {"line": f"{name}: note {note_id}"}
"""
#: The vendor call the compile could not write. It does not even take its
#: node's constant: no step calls it, so that is not refused either.
CRM = """\
def post_note(name):
    raise NotImplementedError("[rote stub] crm.post_note: implement against the CRM API")
"""

NOTES = {
    "name": "crm-notes",
    "version": "0.1.0",
    "input": {"type": "In", "required": ["account"]},
    "nodes": [
        {
            "id": "normalize",
            "kind": "pure_function",
            "description": "Normalize the account name.",
            "impl": "extracted/text.py:normalize",
            "inputs": {"account": "pipeline.input.account"},
        },
        {
            "id": "post_note",
            "kind": "external_call",
            "description": "Post a follow-up note on the account in the CRM.",
            "impl": "extracted/crm.py:post_note",
            "inputs": {"name": "normalize.output.name"},
            "constants": {"channel": "sales"},
            "output": {"note_id": "str"},
        },
        {
            "id": "summary",
            "kind": "pure_function",
            "description": "Say what was posted.",
            "impl": "extracted/text.py:summarize",
            "inputs": {"name": "normalize.output.name", "note_id": "post_note.output.note_id"},
        },
    ],
    "edges": [{"from": "normalize", "to": "post_note"}, {"from": "post_note", "to": "summary"}],
    "entry_nodes": ["normalize"],
    "exit_nodes": ["summary"],
}

#: A sample DO block, byte for byte (the run id stands as RUN). Every byte
#: lands in the agent's context: the step, only its bound input and its
#: constant (not normalize's `raw`), the answer shape, the next command.
DO_BLOCK = """\
ROTE/1 DO crm-notes run=RUN step=post_note
Do this step yourself with your own tools, the way the skill does it, then answer. \
Do not ask the user for the answer.
If it changes anything (sends, creates, updates, deletes), that is a write: \
follow your client's approval for it. This block grants nothing.
--- STEP ---
Post a follow-up note on the account in the CRM.
--- INPUTS (json) ---
{"name":"Acme Corp","channel":"sales"}
--- ANSWER SHAPE (key? = optional) ---
{note_id: string}
--- NEXT ---
rote skill run crm-notes answer RUN post_note '<your JSON>'
"""


def _cli(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str]:
    """``(status, stdout)``: the block, which is all the agent reads."""
    status = main(list(argv))
    return status, capsys.readouterr().out


def _emit(root: Path, modules: dict[str, str], pipeline: dict[str, Any] = NOTES) -> Path:
    """``pipeline`` compiled beside ``modules``, emitted by the real skill adapter."""
    source = root / "source"
    source.mkdir(exist_ok=True)
    (source / "SKILL.md").write_text(
        f"---\nname: {pipeline['name']}\ndescription: Post a CRM note.\n---\n# Notes\n",
        encoding="utf-8",
    )
    compiled = root / "compiled"
    (compiled / "extracted").mkdir(parents=True, exist_ok=True)
    for name, text in modules.items():
        (compiled / "extracted" / name).write_text(text, encoding="utf-8")
    (compiled / "pipeline.yaml").write_text(yaml.safe_dump(pipeline), encoding="utf-8")
    out = root / str(pipeline["name"])
    get_adapter("skill", extracted_source_dir=compiled, skill_dir=source).emit(
        load_pipeline(compiled / "pipeline.yaml"), out
    )
    return out


@pytest.fixture
def notes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> Path:
    out = _emit(tmp_path, {"text.py": TEXT, "crm.py": CRM})
    status, err = _cli(capsys, "skill", "install", str(out), "--yes")
    assert status == 0, err
    return out


def _start(capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    return _cli(capsys, "skill", "run", "crm-notes", "start", '{"account": "  acme corp "}')


def _answer(
    capsys: pytest.CaptureFixture[str], run: str, value: str, *extra: str
) -> tuple[int, str]:
    return _cli(capsys, "skill", "run", "crm-notes", "answer", run, "post_note", value, *extra)


def test_a_do_step_runs_through_rote_skill_run_to_done(
    notes: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status, block = _start(capsys)
    assert status == 0, block
    fields = header(block)
    assert (fields["kind"], fields["step"]) == ("DO", "post_note")
    status, done = _answer(capsys, fields["run"], '{"note_id": "N-17"}')
    assert status == 0, done
    assert header(done)["kind"] == "DONE"
    # The step after it ran on the agent's answer, in code.
    assert result_of(done) == {"summary": {"line": "Acme Corp: note N-17"}}
    record = json.loads((run_dir(done) / "nodes" / "post_note.json").read_text("utf-8"))
    assert record["result"] == {"note_id": "N-17"}
    assert record["answered_by"] == "agent" and "said" not in record
    assert journal(done)["answers"]["post_note"]["answered_by"] == "agent"
    # The skill tells the agent what a DO block is, in its one tier line.
    body = (notes / "SKILL.md").read_text(encoding="utf-8")
    assert "- DO: yours to do. Do the step yourself with your own tools" in body
    readme = (notes / "README.md").read_text(encoding="utf-8")
    assert "| `post_note` | external_call | the agent, with its own tools (DO) |" in readme
    assert "a DO step needs no consent from rote" in readme


def test_the_do_block_is_pinned_byte_for_byte(
    notes: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, block = _start(capsys)
    run = header(block)["run"]
    assert block.replace(run, "RUN") == DO_BLOCK
    assert len(block.encode("utf-8")) == 574
    # status prints the same block again, for an agent that lost track.
    status, again = _cli(capsys, "skill", "run", "crm-notes", "status", run)
    assert status == 0 and again == block


def test_a_bad_answer_is_asked_again_then_stops_and_resume_asks_afresh(
    notes: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, block = _start(capsys)
    run = header(block)["run"]
    status, again = _answer(capsys, run, '{"id": "N-17"}')
    assert status == 0, again
    fields = header(again)
    assert (fields["kind"], fields["step"], fields["attempt"]) == ("DO", "post_note", "2")
    assert "- answer: unexpected key 'id'" in again
    assert "note_id" in again.split("Answer again with only these fixed:\n")[1]
    assert f"rote skill run crm-notes answer {run} post_note '<your JSON>'" in again
    _answer(capsys, run, '{"note_id": 17}')
    status, stopped = _answer(capsys, run, "not json")
    assert status == 1 and header(stopped)["kind"] == "ERROR", stopped
    assert "post_note: 3 answers did not validate" in stopped
    assert not (run_dir(stopped) / "nodes" / "post_note.json").exists()
    status, resumed = _cli(capsys, "skill", "run", "crm-notes", "resume", run)
    assert status == 0 and header(resumed)["kind"] == "DO" and "attempt" not in header(resumed)


def test_said_is_refused_at_a_do_step(notes: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _, block = _start(capsys)
    run = header(block)["run"]
    status, refused = _answer(capsys, run, '{"note_id": "N-1"}', "--said", "the user said N-1")
    assert status == 2 and header(refused)["kind"] == "ERROR", refused
    assert "post_note is yours to answer; --said is only for a person's words" in refused
    assert not (run_dir(refused) / "nodes" / "post_note.json").exists()
    assert journal(refused)["pending"]["step"] == "post_note"  # still waiting


def test_a_pure_function_stub_is_still_refused_at_emit(tmp_path: Path) -> None:
    """A missing pure function is a compile defect, not agent work."""
    stubbed = TEXT.replace(
        '    return {"line": f"{name}: note {note_id}"}', "    raise NotImplementedError('todo')"
    )
    with pytest.raises(ValueError, match=r"1 extracted/ module\(s\) would still be stubs") as e:
        _emit(tmp_path, {"text.py": stubbed, "crm.py": CRM})
    assert "scripts/extracted/text.py" in str(e.value)
    assert "crm.py" not in str(e.value)  # the external_call's stub is a DO step


def test_a_written_external_call_still_runs_in_code(tmp_path: Path) -> None:
    """The negative half: only an unwritten external_call is the agent's."""
    written = "def post_note(name, channel):\n    return {'note_id': f'{channel}-1'}\n"
    out = _emit(tmp_path, {"text.py": TEXT, "crm.py": written})
    plan = json.loads((out / "scripts" / "plan.json").read_text(encoding="utf-8"))
    assert {s["id"]: s["kind"] for s in plan["steps"]} == {
        "normalize": "call",
        "post_note": "call",
        "summary": "call",
    }
    assert "- DO:" not in (out / "SKILL.md").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "modules",
    [
        {
            "crm.py": "from ._crm_impl import post_note  # noqa: F401\n",
            "_crm_impl.py": "def post_note(name, channel):\n    return {'note_id': channel}\n",
        },
        {
            "crm.py": "if True:\n\n    def post_note(name, channel):\n"
            "        return {'note_id': channel}\n"
        },
    ],
    ids=["re-export", "conditional-def"],
)
def test_shipped_code_rote_cannot_read_runs_as_code_not_as_a_do_step(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], modules: dict[str, str]
) -> None:
    """The AST sees no top-level def (status ``absent``), but the module ships
    and works: handing it to the agent would replace working vendor code."""
    out = _emit(tmp_path, {"text.py": TEXT, **modules})
    plan = json.loads((out / "scripts" / "plan.json").read_text(encoding="utf-8"))
    assert {s["id"]: s["kind"] for s in plan["steps"]}["post_note"] == "call"
    assert _cli(capsys, "skill", "install", str(out), "--yes")[0] == 0
    status, done = _start(capsys)
    assert status == 0 and header(done)["kind"] == "DONE", done
    assert result_of(done) == {"summary": {"line": "Acme Corp: note sales"}}


def test_a_stub_beside_shipped_code_is_still_a_do_step(tmp_path: Path) -> None:
    out = _emit(tmp_path, {"text.py": TEXT, "crm.py": CRM})
    plan = json.loads((out / "scripts" / "plan.json").read_text(encoding="utf-8"))
    assert {s["id"]: s["kind"] for s in plan["steps"]}["post_note"] == "do"


# ───────── fan_out, and inside an agent loop ─────────


FANNED = {
    "name": "tagger",
    "version": "0.1.0",
    "input": {
        "type": "In",
        "required": ["accounts", "tag"],
        "input_schema": {
            "type": "object",
            "properties": {"accounts": {"type": "array"}, "tag": {"type": "string"}},
        },
    },
    "nodes": [
        {
            "id": "tag",
            "kind": "external_call",
            "description": "Tag the account in the CRM.",
            "fan_out": True,
            "impl": "extracted/crm.py:tag",
            "inputs": {"account": "pipeline.input.accounts", "tag": "pipeline.input.tag"},
            "constants": {"source": "rote"},
            "output": {"tagged": "bool"},
        }
    ],
    "edges": [],
    "entry_nodes": ["tag"],
    "exit_nodes": ["tag"],
}


def test_a_fanned_do_step_asks_per_item_and_keeps_their_order(tmp_path: Path) -> None:
    scripts = make_skill(tmp_path, FANNED, implementations={"tag": "stub"})
    payload = {"accounts": ["acme", "globex", "initech"], "tag": "renewal"}
    status, block = drive(scripts, "start", json.dumps(payload), skill="tagger")
    assert status == 0, block
    assert header(block)["kind"] == "DO"
    assert "for each item (its `account` input)" in block
    assert '{"tag":"renewal","source":"rote"}' in block  # shared, constants last
    assert '{"0":"acme","1":"globex","2":"initech"}' in block
    run = header(block)["run"]
    answer = {"2": {"tagged": False}, "0": {"tagged": True}}
    status, again = drive(scripts, "answer", run, "tag", json.dumps(answer), skill="tagger")
    assert status == 0 and header(again)["kind"] == "DO", again
    assert "missing items: 1" in again
    answer["1"] = {"tagged": True}
    status, done = drive(scripts, "answer", run, "tag", json.dumps(answer), skill="tagger")
    assert status == 0 and header(done)["kind"] == "DONE", done
    assert result_of(done) == {"tag": [{"tagged": True}, {"tagged": True}, {"tagged": False}]}


LOOPED = {
    "name": "looper",
    "version": "0.1.0",
    "input": {"type": "In", "required": ["q"]},
    "nodes": [
        {
            "id": "research",
            "kind": "agent_loop",
            "description": "Research the question.",
            "inputs": {"q": "pipeline.input.q"},
            "tools": ["search"],
            "tool_servers": {"search": "fx"},
            "loop_body": ["lookup"],
            "output": {"answer": "str"},
            "termination": {"condition": "the question is answered", "max_iterations": 3},
        },
        {
            "id": "lookup",
            "kind": "external_call",
            "description": "Look the company up in the CRM.",
            "impl": "extracted/crm.py:lookup",
            "output": {"found": "bool"},
        },
    ],
    "edges": [],
    "entry_nodes": ["research"],
    "exit_nodes": ["research"],
}


def test_a_loop_lists_its_do_sub_node_as_a_step_never_a_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests._skill_driver import fixture_server

    scripts = make_skill(tmp_path, LOOPED, implementations={"lookup": "stub"})
    server = fixture_server(tmp_path / "hits.jsonl")
    with server as fx:
        monkeypatch.setenv("ROTE_MCP_FX_URL", fx.url)
        _loop_through(scripts)


def _loop_through(scripts: Path) -> None:
    status, block = drive(scripts, "start", '{"q": "who"}', skill="looper")
    assert status == 0 and header(block)["kind"] == "LOOP", block
    steps = block.split(
        "--- STEPS (do each yourself with your own tools where the task needs it; not tools) ---\n"
    )[1]
    assert steps.startswith(
        "lookup: Look the company up in the CRM.\nlookup gives: {found: boolean}\n"
    )
    assert "search {query: string}" in block.split("--- TOOLS")[1]  # the one tool
    assert "lookup {" not in block
    run = header(block)["run"]
    status, refused = drive(scripts, "tool", run, "research", "lookup", "{}", skill="looper")
    assert status == 2 and re.search(r"lookup is not one of this loop's tools", refused), refused
    status, done = drive(scripts, "answer", run, "research", '{"answer": "Acme"}', skill="looper")
    assert status == 0 and result_of(done) == {"research": {"answer": "Acme"}}


# ───────── the source's own instructions and the answer's contract (#336) ─────────

BOOKINGS_SKILL = """\
---
name: session-roster
description: Reconcile session bookings against the roster.
---
# Session roster

## Step 1: Read the bookings

Run this read-only query against the bookings database:

```sql
SELECT s.date, s.slot, p.email, p.name, b.cohort_id, b.created_at
FROM bookings b
JOIN sessions s ON s.id = b.session_id
JOIN people p ON p.id = b.person_id
WHERE s.date >= :since
```

## Step 2: Check the rows

Every row needs all six columns.
"""

BOOKINGS = """\
def read_bookings(since):
    \"\"\"Returns {"bookings": [row, ...]}: one row per booking, each with date, slot,
    email, name, cohort_id and created_at, exactly as the query selects them.\"\"\"
    raise NotImplementedError("[rote stub] db.read_bookings: no connector grounded")
"""

CHECK = """\
REQUIRED = ("date", "slot", "email", "name", "cohort_id", "created_at")


def check_rows(bookings):
    rows = bookings["bookings"]
    return {"complete": all(all(k in r for k in REQUIRED) for r in rows), "n": len(rows)}
"""

ROSTER = {
    "name": "session-roster",
    "version": "0.1.0",
    "input": {"type": "In", "required": ["since"]},
    "nodes": [
        {
            "id": "read_bookings",
            "kind": "external_call",
            "description": "Read bookings and their session and person fields.",
            "impl": "extracted/db.py:read_bookings",
            "inputs": {"since": "pipeline.input.since"},
            "output": {"bookings": "list"},
            "source": {"section": "Step 1: Read the bookings"},
        },
        {
            "id": "check_rows",
            "kind": "pure_function",
            "description": "Check every row has the six columns.",
            "impl": "extracted/check.py:check_rows",
            "inputs": {"bookings": "read_bookings.output"},
        },
    ],
    "edges": [{"from": "read_bookings", "to": "check_rows"}],
    "entry_nodes": ["read_bookings"],
    "exit_nodes": ["check_rows"],
}


def test_a_do_block_carries_the_skills_query_and_the_answers_contract(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The handoff used to be a one-line description and `{bookings: Any}`: the
    agent had to reverse-engineer the generated files to redo the read."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "SKILL.md").write_text(BOOKINGS_SKILL, encoding="utf-8")
    compiled = tmp_path / "compiled"
    (compiled / "extracted").mkdir(parents=True)
    (compiled / "extracted" / "db.py").write_text(BOOKINGS, encoding="utf-8")
    (compiled / "extracted" / "check.py").write_text(CHECK, encoding="utf-8")
    (compiled / "pipeline.yaml").write_text(yaml.safe_dump(ROSTER), encoding="utf-8")
    out = tmp_path / "session-roster"
    get_adapter("skill", extracted_source_dir=compiled, skill_dir=source).emit(
        load_pipeline(compiled / "pipeline.yaml"), out
    )
    plan = json.loads((out / "scripts" / "plan.json").read_text(encoding="utf-8"))
    step = next(s for s in plan["steps"] if s["id"] == "read_bookings")
    assert step["kind"] == "do"
    assert "JOIN people p ON p.id = b.person_id" in step["instructions"]
    assert step["instructions"].startswith("## Step 1: Read the bookings")
    assert "Step 2" not in step["instructions"]
    assert "cohort_id and created_at" in step["contract"]
    status, err = _cli(capsys, "skill", "install", str(out), "--yes")
    assert status == 0, err
    status, block = _cli(
        capsys, "skill", "run", "session-roster", "start", '{"since": "2026-10-01"}'
    )
    assert status == 0, block
    assert header(block)["kind"] == "DO"
    query = block.split("--- WHAT THE SKILL SAYS FOR THIS STEP ---\n", 1)[1]
    assert query.index("SELECT s.date, s.slot, p.email, p.name") < query.index(
        "--- WHAT THE ANSWER"
    )
    contract = block.split("--- WHAT THE ANSWER MUST HOLD (later steps read it) ---\n", 1)[1]
    assert contract.startswith('Returns {"bookings": [row, ...]}')
    assert contract.index("created_at") < contract.index("--- INPUTS (json) ---")


def test_a_long_section_is_cut_says_so_and_closes_its_fence(tmp_path: Path) -> None:
    from rote.adapters.skill import MAX_DO_INSTRUCTIONS, do_context
    from rote.ir import Pipeline

    (tmp_path / "SKILL.md").write_text(
        BOOKINGS_SKILL.replace("WHERE s.date", "x" * (MAX_DO_INSTRUCTIONS + 10) + "\nWHERE s.date")
    )
    context = do_context(Pipeline.model_validate(ROSTER), tmp_path, None)
    text = context["read_bookings"]["instructions"]
    assert len(text) < MAX_DO_INSTRUCTIONS + 80
    # The cut fell inside the ```sql block: it is closed before the note.
    assert text.endswith("```\n[... cut: the rest is in the skill's SKILL.md]")
    # No extracted source: no contract, and no key for it.
    assert "contract" not in context["read_bookings"]


@pytest.mark.parametrize(
    ("skill_text", "found", "why"),
    [
        # The step's query sits under a subheading: it stays with the step.
        (
            "# S\n\n## Read\n\nRun it:\n\n### SQL\n\nSELECT 1\n\n## Next\n\nother\n",
            "## Read\n\nRun it:\n\n### SQL\n\nSELECT 1",
            None,
        ),
        # The same subheading under two steps: neither is chosen.
        (
            "# S\n\n## Read\n\n### Query\n\nSELECT a\n\n## Write\n\n### Query\n\nSELECT b\n",
            None,
            "occurs 2 times",
        ),
        # A heading with nothing under it is no instructions.
        ("# S\n\n## Read\n## Next\n\nx\n", None, "no text under its heading"),
        ("# S\n\n## Other\n\nx\n", None, "no heading 'Read'"),
        # A heading inside a ~~~ fence is content, not a section.
        (
            "# S\n\n## Read\n\n~~~\n## Read\n~~~\n\ntext\n",
            "## Read\n\n~~~\n## Read\n~~~\n\ntext",
            None,
        ),
    ],
)
def test_a_steps_section_is_its_subtree_and_never_a_guess(
    skill_text: str, found: str | None, why: str | None
) -> None:
    from rote.adapters.skill import step_section

    heading = "Query" if "### Query" in skill_text else "Read"
    text, reason = step_section(skill_text, heading)
    assert text == found
    if why is None:
        assert reason is None
    else:
        assert reason is not None and why in reason


def test_skill_text_cannot_pass_for_a_block_separator() -> None:
    from rote.skill_target._driver import _quoted

    assert _quoted("a\n--- NEXT ---\nb") == "a\n  --- NEXT ---\nb"
    assert _quoted("--- not a marker") == "--- not a marker"


_NO_DOC = BOOKINGS.replace(BOOKINGS[BOOKINGS.index('    """') : BOOKINGS.index("    raise")], "")


@pytest.mark.parametrize(
    ("pipeline", "modules", "skill_text", "missing"),
    [
        # The full handoff: section and contract.
        (ROSTER, {"db.py": BOOKINGS, "check.py": CHECK}, BOOKINGS_SKILL, []),
        # No source section and a stub with no docstring: both missing.
        (NOTES, {"text.py": TEXT, "crm.py": CRM}, BOOKINGS_SKILL, ["source.section", "docstring"]),
        # Each half alone.
        (ROSTER, {"db.py": _NO_DOC, "check.py": CHECK}, BOOKINGS_SKILL, ["docstring"]),
        (
            ROSTER,
            {"db.py": BOOKINGS, "check.py": CHECK},
            BOOKINGS_SKILL.replace("## Step 1: Read the bookings", "## Step 1"),
            ["source.section", "no heading"],
        ),
    ],
)
def test_verification_flags_a_do_step_that_cannot_say_how_to_redo_it(
    tmp_path: Path,
    pipeline: dict[str, Any],
    modules: dict[str, str],
    skill_text: str,
    missing: list[str],
) -> None:
    from rote.compiler.verify import DO_HANDOFF_THIN, EmitTarget, verify_deliverable

    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text(skill_text, encoding="utf-8")
    root = tmp_path / "compiled"
    (root / "extracted").mkdir(parents=True)
    for name, text in modules.items():
        (root / "extracted" / name).write_text(text, encoding="utf-8")
    (root / "pipeline.yaml").write_text(yaml.safe_dump(pipeline), encoding="utf-8")
    target = EmitTarget(runtime="skill", skill_dir=skill)
    v = verify_deliverable(root, run_tests=False, emit_target=target)
    thin = [f for f in v.review_gaps if f.kind == DO_HANDOFF_THIN]
    if not missing:
        assert thin == []
        return
    [finding] = thin
    assert finding.severity == "warning"
    for word in missing:
        assert word in finding.message
    if "source.section" not in missing:
        assert "source.section" not in finding.message
    if "docstring" not in missing:
        assert "docstring" not in finding.message
    # A workflow runtime has no DO steps: nothing to flag.
    other = verify_deliverable(root, run_tests=False, emit_target=EmitTarget(runtime="dbos"))
    assert [f for f in other.review_gaps if f.kind == DO_HANDOFF_THIN] == []


def test_on_a_system1_compile_a_missing_section_is_a_question_not_a_repair(
    tmp_path: Path,
) -> None:
    """System-1 decides and restores `source`: the agent cannot change it."""
    from rote.compiler.verify import DO_HANDOFF_THIN, EmitTarget, verify_deliverable

    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        BOOKINGS_SKILL.replace("## Step 1: Read the bookings", "## Step 1"), encoding="utf-8"
    )
    root = tmp_path / "compiled"
    (root / "extracted").mkdir(parents=True)
    (root / "extracted" / "db.py").write_text(BOOKINGS, encoding="utf-8")
    (root / "extracted" / "check.py").write_text(CHECK, encoding="utf-8")
    (root / "pipeline.yaml").write_text(yaml.safe_dump(ROSTER), encoding="utf-8")
    (root / "system1").mkdir()
    (root / "system1" / "skeleton.yaml").write_text(yaml.safe_dump(ROSTER), encoding="utf-8")
    v = verify_deliverable(
        root, run_tests=False, emit_target=EmitTarget(runtime="skill", skill_dir=skill)
    )
    assert [f for f in v.review_gaps if f.kind == DO_HANDOFF_THIN] == []
    [question] = [q for q in v.questions if q.node_id == "read_bookings"]
    assert "cannot show the skill's text" in question.text and "no heading" in question.text


def test_an_unreadable_module_is_no_contract_not_a_crash(tmp_path: Path) -> None:
    from rote.adapters.skill import do_context
    from rote.ir import Pipeline

    (tmp_path / "SKILL.md").write_text(BOOKINGS_SKILL)
    (tmp_path / "extracted").mkdir()
    (tmp_path / "extracted" / "db.py").write_bytes(b"def read_bookings(since):\n    '\xe9'\n")
    context = do_context(Pipeline.model_validate(ROSTER), tmp_path, tmp_path)
    assert "contract" not in context["read_bookings"]
    assert "instructions" in context["read_bookings"]
