"""A decision model double-checks the structure pass: a second opinion, never an override."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from rote.compiler.system1 import checker, structure
from rote.compiler.system1.decision_model import CLEF, Verdict, Verdicts
from rote.compiler.system1.parse import parse_skill
from rote.compiler.system1.tiebreak import Lane
from tests.test_system1_structure import SKILL, TOOLS, FakeCall, good_answer, node

# SKILL's headings, by section index: 0 the title, 1 fetch, 2 summarize, 3 deliver.


@pytest.fixture
def skill(tmp_path: Path) -> Any:
    d = tmp_path / "skill"
    d.mkdir()
    (d / "SKILL.md").write_text(SKILL)
    return replace(parse_skill(d), units=structure.section_units(d))


def _verdicts(**by_section: tuple[str, float, float]) -> Verdicts:
    """``section_<i>=(kind, kind confidence, P(a person must act))``."""
    return Verdicts(
        CLEF,
        {
            int(k.split("_")[1]): Verdict(kind, conf, human, 0.001)
            for k, (kind, conf, human) in by_section.items()
        },
    )


def test_a_confident_kind_the_skeleton_lacks_is_a_disagreement(skill: Any) -> None:
    lines = checker.disagreements(
        _verdicts(section_2=("pure_function", 0.95, 0.0)), skill.units, good_answer()
    )
    assert lines == [
        "section 2 'Step 2: Summarize' (`summarize_email`): Clef reads this step's main work "
        "as pure_function (confidence 0.95); the skeleton gives it llm_judge"
    ]


def test_below_threshold_or_among_a_split_steps_kinds_is_not(skill: Any) -> None:
    answer = good_answer()
    # Below Clef's frozen 0.85: no opinion worth sending.
    assert (
        checker.disagreements(
            _verdicts(section_2=("pure_function", 0.84, 0.0)), skill.units, answer
        )
        == []
    )
    # A step split into a call and the work on its result: the model's kind is one of them.
    split = copy.deepcopy(answer)
    split["nodes"].append(node("tidy", 2, "pure_function", reads=["summarize_email"]))
    assert (
        checker.disagreements(_verdicts(section_2=("pure_function", 0.95, 0.0)), skill.units, split)
        == []
    )


def test_a_missing_or_extra_approval_point_is_a_disagreement(skill: Any) -> None:
    missing = checker.disagreements(
        _verdicts(section_3=("pure_function", 0.9, 0.7)), skill.units, good_answer()
    )
    assert missing == [
        "section 3 'Step 3: Deliver' (`render_digest`): Clef says a person must act or answer "
        "here before the steps after it run (probability 0.70); the skeleton has no hitl_gate "
        "for it"
    ]
    gated = good_answer()
    gated["nodes"].append(node("approve", 3, "hitl_gate", reads=["render_digest"]))
    extra = checker.disagreements(
        _verdicts(section_3=("pure_function", 0.9, 0.2)), skill.units, gated
    )
    assert extra == [
        "section 3 'Step 3: Deliver' (`render_digest`, `approve`): Clef says no person has to "
        "act here (probability a person must: 0.20); the skeleton pauses for one"
    ]
    # A heading the pass made no node for can still need a person.
    title = checker.disagreements(
        _verdicts(section_0=("pure_function", 0.5, 0.9)), skill.units, good_answer()
    )
    assert title and "section 0" in title[0] and "no hitl_gate" in title[0]


def _decide(skill: Any, fake: FakeCall, opinion: Any) -> tuple[Any, Any]:
    return structure.decide(
        skill, SKILL, "rubric", Lane("claude-cli"), tools=TOOLS, call=fake, checker=opinion
    )


def test_a_disagreement_goes_back_to_the_pass_and_its_change_is_taken(skill: Any) -> None:
    changed = good_answer()
    changed["nodes"][2]["why"] = "the model was right"
    seen: list[dict[str, Any]] = []

    def opinion(answer: dict[str, Any]) -> list[str]:
        seen.append(answer)
        return ["X disagrees"] if len(seen) == 1 else []

    fake = FakeCall(good_answer(), changed)
    plan, rec = _decide(skill, fake, opinion)
    assert len(fake.prompts) == 2
    assert "=== SECOND OPINION ===\n- X disagrees" in fake.prompts[1]
    assert rec.checked == ["X disagrees"] and rec.kept == [] and rec.rounds == 2
    assert not any(r.startswith(checker.CHECKER_NOTE_PREFIX) for r in plan.review)
    assert rec.answer["nodes"][2]["why"] == "the model was right"


def test_a_disagreement_the_pass_keeps_is_an_open_question(skill: Any, tmp_path: Path) -> None:
    from rote.compiler.system1 import report_extras

    fake = FakeCall(good_answer(), good_answer())
    plan, rec = _decide(skill, fake, lambda answer: ["section 2 'Step 2: Summarize': X"])
    assert rec.kept == ["section 2 'Step 2: Summarize': X"]
    assert f"{checker.CHECKER_NOTE_PREFIX}section 2 'Step 2: Summarize': X" in plan.review
    plan.structure = rec
    work = tmp_path / "work"
    work.mkdir()
    titles = [t for t, _ in report_extras(work, plan).questions]
    assert "A decision model disagrees: section 2 'Step 2: Summarize'" in titles


def test_a_reply_that_breaks_a_code_check_is_not_taken(skill: Any) -> None:
    broken = good_answer()
    broken["nodes"][0]["quote"] = "words nowhere in the skill"
    fake = FakeCall(good_answer(), broken)
    _, rec = _decide(skill, fake, lambda answer: ["X"])
    assert rec.answer["nodes"][0]["quote"] == good_answer()["nodes"][0]["quote"]
    assert rec.kept == ["X"] and rec.contradictions[-1]


def test_no_disagreement_costs_no_round(skill: Any) -> None:
    fake = FakeCall(good_answer())
    _, rec = _decide(skill, fake, lambda answer: [])
    assert len(fake.prompts) == 1 and rec.rounds == 1 and rec.checked == []


def test_prepare_runs_the_decider_as_the_checker_and_records_its_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rote.compiler import system1
    from rote.compiler.system1 import calls_cost_usd, decision_model

    d = tmp_path / "skill"
    d.mkdir()
    (d / "SKILL.md").write_text(SKILL)
    rubric = tmp_path / "rubric.md"
    rubric.write_text("rubric")
    asked: list[list[str]] = []

    def decide(model: Any, headings: Any) -> Verdicts:
        asked.append([u.title for u in headings.units])
        return _verdicts(section_2=("pure_function", 0.95, 0.0))

    monkeypatch.setattr(decision_model, "decide", decide)
    monkeypatch.setattr(
        structure, "_structured_call", lambda: FakeCall(good_answer(), good_answer())
    )
    work = tmp_path / "w"
    work.mkdir()
    plan = system1.prepare(
        d, work, "clef", tools=TOOLS, structure_lane=Lane("claude-cli"), rubric=rubric
    )
    # Every SKILL.md heading, as the structure pass numbers them.
    assert asked == [[u.title for u in structure.section_units(d)]]
    assert plan.structure is not None and plan.structure.checked and plan.structure.kept
    doc = json.loads((work / "system1" / "decisions.json").read_text())
    assert doc["decider"]["name"] == "clef" and doc["structure"]["kept"]
    # FakeCall's two $0.01 rounds and the one heading's $0.001 verdict.
    assert calls_cost_usd(work) == pytest.approx(0.02 + 0.001)
