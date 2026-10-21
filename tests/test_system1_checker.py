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
    assert [(d.section, d.about) for d in lines] == [(2, "kind")]
    assert [d.text for d in lines] == [
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
    assert [d.about for d in missing] == ["no_gate"]
    assert [d.text for d in missing] == [
        "section 3 'Step 3: Deliver' (`render_digest`): Clef says a person must act or answer "
        "here before the steps after it run (probability 0.70); the skeleton has no hitl_gate "
        "for it"
    ]
    gated = good_answer()
    gated["nodes"].append(node("approve", 3, "hitl_gate", reads=["render_digest"]))
    extra = checker.disagreements(
        _verdicts(section_3=("pure_function", 0.9, 0.2)), skill.units, gated
    )
    assert [d.about for d in extra] == ["extra_gate"]
    assert [d.text for d in extra] == [
        "section 3 'Step 3: Deliver' (`render_digest`, `approve`): Clef says no person has to "
        "act here (probability a person must: 0.20); the skeleton pauses for one"
    ]
    # A heading the pass made no node for can still need a person.
    title = checker.disagreements(
        _verdicts(section_0=("pure_function", 0.5, 0.9)), skill.units, good_answer()
    )
    assert [(d.section, d.about) for d in title] == [(0, "no_gate")]


def _decide(skill: Any, fake: FakeCall, opinion: Any) -> tuple[Any, Any]:
    return structure.decide(
        skill, SKILL, "rubric", Lane("claude-cli"), tools=TOOLS, call=fake, checker=opinion
    )


D2 = checker.Disagreement(
    2, "kind", "section 2 'Step 2: Summarize': Clef reads it as pure_function"
)
#: A passage under heading 2 of SKILL, verbatim.
UNDER_2 = "Write a three-line summary of each email"


def _reply(keeps: list[dict[str, Any]] | None = None, **change: Any) -> dict[str, Any]:
    answer = good_answer()
    answer["nodes"][1].update(change)
    return {**answer, "keeps": keeps or []}


def test_a_changed_node_settles_the_disagreement(skill: Any) -> None:
    def opinion(answer: dict[str, Any]) -> list[Any]:
        return [D2] if answer["nodes"][1]["kind"] == "llm_judge" else []

    fake = FakeCall(good_answer(), _reply(kind="pure_function", fan_out_over=""))
    _, rec = _decide(skill, fake, opinion)
    assert len(fake.prompts) == 2 and "section 2: " in fake.prompts[1]
    assert rec.changed == [D2.text] and rec.kept == []


def test_a_keep_with_a_verbatim_quote_from_its_own_heading_is_accepted(skill: Any) -> None:
    fake = FakeCall(good_answer(), _reply([{"section": 2, "quote": UNDER_2}]))
    plan, rec = _decide(skill, fake, lambda answer: [D2])
    assert len(fake.prompts) == 2
    assert rec.kept == [f'{D2.text}; the pass kept its answer, citing: "{UNDER_2}"']
    assert f"{checker.CHECKER_NOTE_PREFIX}{rec.kept[0]}" in plan.review


@pytest.mark.parametrize(
    "bad",
    [
        [],  # no reason at all
        [{"section": 2, "quote": "a sentence the skill never says"}],
        # Verbatim, but from another heading: it does not speak to this step.
        [{"section": 2, "quote": "Send the digest to the finance team"}],
    ],
)
def test_a_keep_without_a_verbatim_quote_goes_back_once(skill: Any, bad: Any) -> None:
    fake = FakeCall(good_answer(), _reply(bad), _reply([{"section": 2, "quote": UNDER_2}]))
    _, rec = _decide(skill, fake, lambda answer: [D2])
    assert len(fake.prompts) == 3
    assert "=== WHAT FAILED IN THAT ANSWER ===" in fake.prompts[2]
    assert "no passage copied exactly from under that heading" in fake.prompts[2]
    assert rec.kept == [f'{D2.text}; the pass kept its answer, citing: "{UNDER_2}"']


def test_a_keep_still_unsupported_after_the_resend_says_so(skill: Any) -> None:
    fake = FakeCall(good_answer(), _reply(), _reply())
    _, rec = _decide(skill, fake, lambda answer: [D2])
    assert len(fake.prompts) == 3
    assert rec.kept == [f"{D2.text}; the pass kept its answer without a passage that supports it"]


def test_a_reply_that_breaks_a_code_check_is_not_taken(skill: Any) -> None:
    broken = _reply()
    broken["nodes"][0]["quote"] = "words nowhere in the skill"
    fake = FakeCall(good_answer(), broken, copy.deepcopy(broken))
    _, rec = _decide(skill, fake, lambda answer: [D2])
    assert rec.answer["nodes"][0]["quote"] == good_answer()["nodes"][0]["quote"]
    assert rec.kept == [f"{D2.text}; the pass's reply broke a code check, so it was not taken"]


def test_no_disagreement_costs_no_round(skill: Any) -> None:
    fake = FakeCall(good_answer())
    _, rec = _decide(skill, fake, lambda answer: [])
    assert len(fake.prompts) == 1 and rec.rounds == 1 and rec.checked == []


def test_a_kept_disagreement_is_an_open_question(skill: Any, tmp_path: Path) -> None:
    from rote.compiler.system1 import report_extras

    fake = FakeCall(good_answer(), _reply([{"section": 2, "quote": UNDER_2}]))
    plan, rec = _decide(skill, fake, lambda answer: [D2])
    plan.structure = rec
    work = tmp_path / "work"
    work.mkdir()
    titles = [t for t, _ in report_extras(work, plan).questions]
    assert "A decision model disagrees: section 2 'Step 2: Summarize'" in titles


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
    fake = FakeCall(good_answer(), _reply([{"section": 2, "quote": UNDER_2}]))
    monkeypatch.setattr(structure, "_structured_call", lambda: fake)
    work = tmp_path / "w"
    work.mkdir()
    plan = system1.prepare(
        d, work, "clef", tools=TOOLS, structure_lane=Lane("claude-cli"), rubric=rubric
    )
    assert asked == [[u.title for u in structure.section_units(d)]]
    assert plan.structure is not None and plan.structure.checked and plan.structure.kept
    doc = json.loads((work / "system1" / "decisions.json").read_text())
    assert doc["decider"]["name"] == "clef" and doc["structure"]["kept"]
    # FakeCall's two $0.01 rounds and the one heading's $0.001 verdict.
    assert calls_cost_usd(work) == pytest.approx(0.02 + 0.001)


@pytest.mark.asyncio
async def test_a_checker_with_no_credentials_fails_the_compile_before_any_spend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--decider jev with no Cloudflare account: loud, and before the pass runs."""
    import shutil

    from rote.compiler import Compiler, CompilerError

    def spent(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the structure pass ran although the checker cannot")

    class _Agent:
        name = "claude"

        async def run(self, **kwargs: Any) -> Any:
            raise AssertionError("the agent ran although the checker cannot")

    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)
    monkeypatch.setattr(structure, "decide", spent)
    monkeypatch.setattr(structure, "lane_for", lambda driver, choice: Lane("claude-cli"))
    monkeypatch.setattr(Compiler, "select_driver", lambda self: _Agent())
    skill = tmp_path / "skill"
    shutil.copytree(Path(__file__).parent.parent / "examples" / "ops-report" / "skill", skill)
    with pytest.raises(
        CompilerError,
        match=r"^--decider jev needs CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN",
    ):
        await Compiler(
            system1_decider="jev", run_generated_tests=False, tool_discovery="none"
        ).compile(skill, tmp_path / "out")
