"""A compile's agent adds a declared pipeline input field (rote.compiler.system1.added_inputs).

Issue #291: a generated module read ``request['omni_dashboard_url']`` and
raised without it, while System-1's input schema never declared it, and the
agent's proposal to add it was refused as "not a decision field". These
tests pin the one narrow way in, on both System-1 paths. No inference: the
single-session path writes the agent's file by hand, and the parallel path
answers from a fake generator.
"""

from __future__ import annotations

import copy
import functools
from pathlib import Path
from typing import Any

import pytest
import yaml

from rote.compiler import Compiler, system1
from rote.compiler.drivers import Generation, GenerationRequest
from rote.compiler.grounding import Grounding
from rote.compiler.system1 import added_inputs
from rote.compiler.system1.generate import Context
from rote.contracts import ContractFinding
from rote.ir import load_pipeline

REPO = Path(__file__).resolve().parent.parent
SKILL = REPO / "examples" / "expense-audit" / "skill"
#: The skill's own words: "The input is the month's export: one row per
#: expense line, plus the reporting period (`YYYY-MM`)."
QUOTE = "one row per expense line, plus the reporting period"


def _entry(**over: Any) -> dict[str, Any]:
    entry = {
        "name": "reporting_period",
        "type": "string",
        "required": True,
        "description": "The reporting period as YYYY-MM.",
        "quote": QUOTE,
    }
    entry.update(over)
    return entry


def _prepare(tmp_path: Path) -> tuple[Path, system1.Plan]:
    work = tmp_path / "work"
    work.mkdir()
    return work, system1.prepare(SKILL, work)


def _write_entries(work: Path, entries: list[Any]) -> None:
    (work / "system1" / added_inputs.FILENAME).write_text(yaml.safe_dump(entries))


def _doc(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text())
    assert isinstance(loaded, dict)
    return loaded


# ───────── the single-session path: system1/input_fields.yaml ─────────


def test_an_added_input_field_is_declared_and_survives_every_settle(tmp_path: Path) -> None:
    work, plan = _prepare(tmp_path)
    skeleton = _doc(work / "system1" / "skeleton.yaml")
    _write_entries(work, [_entry()])

    system1.settle(work, plan)
    schema = _doc(work / "pipeline.yaml")["input"]["input_schema"]
    assert schema["properties"]["reporting_period"] == {
        "type": "string",
        "description": "The reporting period as YYYY-MM.",
    }
    assert schema["required"] == ["reporting_period"]
    # The skeleton's own field is untouched.
    assert schema["properties"]["month"] == skeleton["input"]["input_schema"]["properties"]["month"]

    # The next stop restores the decisions again: the addition is laid back
    # over the skeleton, and reading it back is not an edit to restore.
    changes = system1.settle(work, plan, final=True)
    assert not [c for c in changes if c.startswith("pipeline input")]
    pipeline = load_pipeline(work / "pipeline.yaml")
    assert "reporting_period" in pipeline.required_input_fields
    # Nothing else moved: every node, edge and exit is the skeleton's.
    after = _doc(work / "pipeline.yaml")
    for key in ("nodes", "edges", "entry_nodes", "exit_nodes", "name"):
        assert after[key] == skeleton[key], key


def test_an_agent_edit_to_the_input_is_still_restored_around_the_addition(
    tmp_path: Path,
) -> None:
    """The file adds; pipeline.yaml edits to the input stay undone."""
    work, plan = _prepare(tmp_path)
    _write_entries(work, [_entry()])
    doc = _doc(work / "pipeline.yaml")
    doc["input"]["input_schema"]["properties"] = {"renamed": {"type": "string"}}
    (work / "pipeline.yaml").write_text(yaml.safe_dump(doc, sort_keys=False))
    changes = system1.settle(work, plan)
    assert "pipeline input: restored the skeleton's" in changes
    props = _doc(work / "pipeline.yaml")["input"]["input_schema"]["properties"]
    assert set(props) == {"month", "reporting_period"}


def test_a_declared_untyped_field_can_be_made_required_and_keeps_its_schema(
    tmp_path: Path,
) -> None:
    work, plan = _prepare(tmp_path)
    before = _doc(work / "system1" / "skeleton.yaml")["input"]["input_schema"]["properties"]
    _write_entries(work, [_entry(name="month", quote=QUOTE)])
    system1.settle(work, plan)
    schema = _doc(work / "pipeline.yaml")["input"]["input_schema"]
    assert schema["required"] == ["month"]
    assert schema["properties"] == before


@pytest.mark.parametrize(
    ("entry", "why"),
    [
        (_entry(quote="the caller always sends a reporting period"), "not the skill's text"),
        (_entry(name="reporting-period"), "must be an identifier"),
        (_entry(type="date"), "must be one of"),
        (_entry(description="two\nlines"), "one line"),
        ({**_entry(), "nodes": []}, "exactly the keys"),
        (_entry(name="month", required=False), "already declared"),
    ],
)
def test_a_bad_entry_is_refused_and_changes_nothing(
    tmp_path: Path, entry: dict[str, Any], why: str
) -> None:
    work, plan = _prepare(tmp_path)
    decided = _doc(work / "system1" / "skeleton.yaml")["input"]
    _write_entries(work, [entry])
    system1.settle(work, plan)
    assert _doc(work / "pipeline.yaml")["input"] == decided
    (refused,) = added_inputs.recorded(work).refused
    assert why in refused[1]


def test_a_field_the_sample_lacks_cannot_be_required(tmp_path: Path) -> None:
    """The sample is a real input: requiring a field it lacks fails it on entry."""
    work, plan = _prepare(tmp_path)
    skeleton = _doc(work / "system1" / "skeleton.yaml")
    out = added_inputs.apply(
        copy.deepcopy(skeleton),
        [_entry()],
        added_inputs.SkillText.read(SKILL),
        sample_payload={"month": "2026-09"},
    )
    assert not out.applied
    assert "sample input has no `reporting_period`" in out.refused[0][1]
    optional = added_inputs.apply(
        copy.deepcopy(skeleton),
        [_entry(required=False)],
        added_inputs.SkillText.read(SKILL),
        sample_payload={"month": "2026-09"},
    )
    assert [a["name"] for a in optional.applied] == ["reporting_period"]


def test_every_added_and_refused_field_is_an_open_question(tmp_path: Path) -> None:
    work, plan = _prepare(tmp_path)
    _write_entries(work, [_entry(), _entry(name="bad-name")])
    system1.settle(work, plan, final=True)
    questions = dict(system1.report_extras(work, plan).questions)
    added = questions["The agent added a pipeline input field: reporting_period"]
    assert "`reporting_period` (string, required)" in added and QUOTE in added
    assert (
        "`bad-name`: name 'bad-name' must be an identifier"
        in (questions["An input field the agent asked for was not added"])
    )


def test_a_refused_entry_reaches_the_agents_next_repair_turn(tmp_path: Path) -> None:
    work, plan = _prepare(tmp_path)
    _write_entries(work, [_entry(quote="made up words that are not there")])
    compiler = Compiler(
        system1=True, system1_tiebreak="none", system1_structure="rules", run_generated_tests=False
    )
    repair = compiler._deliverable_repairer(
        None,
        SKILL,
        before_verify=functools.partial(system1.settle, p=plan),
        portable_judges=True,
    )
    instruction = repair(work / "pipeline.yaml")
    assert instruction is not None
    assert "did not add these system1/input_fields.yaml entries" in instruction
    assert "`reporting_period`: its quote is not the skill's text" in instruction


# ───────── the parallel path: a module piece's input_fields ─────────


class _AddsField:
    """A fake generator whose answer for one module also declares input fields."""

    def __init__(self, base: Any, label: str, answers: list[list[dict[str, Any]]]) -> None:
        self.base = base
        self.label = label
        self.answers = answers
        self.name = base.name
        self.requests = base.requests

    def is_available(self) -> tuple[bool, str]:
        return (True, "")

    async def generate(self, request: GenerationRequest) -> Generation:
        g = await self.base.generate(request)
        if request.label == self.label and self.answers:
            g.payload["input_fields"] = self.answers.pop(0)
        else:
            g.payload.setdefault("input_fields", [])
        return g


@pytest.mark.asyncio
async def test_a_module_piece_adds_an_input_field_on_the_parallel_path(tmp_path: Path) -> None:
    from tests.test_system1_generate import FakeGenerator, _write

    work, plan = _prepare(tmp_path)
    ctx = Context.load(work, plan, Grounding())
    first, other = ctx.pipeline.nodes[0].id, ctx.pipeline.nodes[1].id
    base = FakeGenerator({n.id: n for n in ctx.pipeline.nodes})
    fake = _AddsField(base, first, [[_entry(quote="words the skill never says")], [_entry()]])
    # A finding on another node forces a repair round after the addition.
    nudge = ContractFinding(node_id=other, kind="x", severity="error", message="fix it")
    await _write(ctx, plan, fake, inject=[[nudge]])  # type: ignore[arg-type]

    # The bad quote was sent back to that piece alone, by code, before any merge.
    asked = [r for r in base.requests if r.label == first]
    assert len(asked) >= 2
    assert "input_fields entry `reporting_period` was not added" in asked[1].prompt
    assert "its quote is not the skill's text" in asked[1].prompt
    # The good one is in the shared file and in the settled pipeline.
    assert yaml.safe_load((work / "system1" / added_inputs.FILENAME).read_text()) == [_entry()]
    schema = _doc(work / "pipeline.yaml")["input"]["input_schema"]
    assert schema["required"] == ["reporting_period"]
    assert schema["properties"]["reporting_period"]["type"] == "string"
    # A piece asked after the addition sees the declared field in its card;
    # one asked before it did not.
    asked_other = [r.prompt for r in base.requests if r.label == other]
    assert len(asked_other) == 2
    assert "reporting_period" not in asked_other[0]
    assert "reporting_period" in asked_other[1]


@pytest.mark.asyncio
async def test_a_judge_cannot_add_input_fields(tmp_path: Path) -> None:
    from tests.test_system1_generate import FakeGenerator, _write

    work = tmp_path / "work"
    work.mkdir()
    plan = system1.prepare(REPO / "examples" / "deal-monitor" / "skill", work)
    ctx = Context.load(work, plan, Grounding())
    judge = next(n.id for n in ctx.pipeline.nodes if n.kind.value == "llm_judge")
    base = FakeGenerator({n.id: n for n in ctx.pipeline.nodes})
    fake = _AddsField(base, judge, [[_entry()]])
    await _write(ctx, plan, fake)  # type: ignore[arg-type]
    asked = [r for r in base.requests if r.label == judge]
    assert len(asked) >= 2
    assert "a judge cannot add pipeline input fields" in asked[1].prompt
    assert not (work / "system1" / added_inputs.FILENAME).exists()
