"""A constant is a typed value with a unit, checked against its own quote (#247).

Measured on the compiles on disk: the structure pass copied constants as
prose (ops-report '75%', expense-audit '$2,500.00', docs-digest "first
five"), so every module parsed text, or hard-coded the number beside a dead
constant. No inference: the model's answers are fixtures.
"""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from rote.compiler import system1
from rote.compiler.report import render_compile_report
from rote.compiler.system1 import generate, structure
from rote.compiler.system1.decide import REVIEW_BELOW
from rote.compiler.system1.enforce import enforce
from rote.compiler.system1.parse import constants_in, parse_skill
from rote.compiler.system1.skeleton import pipeline_doc
from rote.compiler.system1.tiebreak import Lane
from rote.inference._runtime_helper import structured_output_schema
from rote.ir import Node, Pipeline
from tests.test_system1_structure import FakeCall, node

#: The measured shapes, each in the words its skill used.
SKILL = """\
---
name: measured-constants
description: Constants written the way people write them
---

# Measured constants

## Step 1: Site completion

Parse the facility summary. Flag any site below 75% completion.

## Step 2: Sorter metrics

Flag if fail rate > 5% (action needed) or if volume dropped >15% day-over-day.
Also flag a margin below -5%.

## Step 3: Expense policy

The policy window is 60 days. A single employee's report over **$2,500.00**
for the month always goes to review. A receipt is required on every line of
**$25.00 or more**.

## Step 4: Reading list

Search the docs, then keep the first five results.
"""


def _c(name: str, value: Any, unit: str, quote: str) -> dict[str, Any]:
    return {"name": name, "value": value, "unit": unit, "quote": quote}


def answer() -> dict[str, Any]:
    pure = {"kind": "pure_function"}
    return {
        "input_fields": [{"name": "rows", "description": "the rows to check"}],
        "nodes": [
            node(
                "check_completion",
                1,
                **pure,
                quote="Flag any site below 75% completion",
                constants=[_c("completion_floor", 75, "percent", "below 75% completion")],
            ),
            node(
                "check_sorters",
                2,
                **pure,
                quote="Flag if fail rate > 5%",
                constants=[
                    _c("max_fail_rate", 5, "percent", "fail rate > 5%"),
                    _c("max_volume_drop", 15, "percent", "volume dropped >15%"),
                ],
            ),
            node(
                "audit_reports",
                3,
                **pure,
                quote="The policy window is 60 days",
                constants=[
                    _c("policy_window", 60, "days", "The policy window is 60 days"),
                    _c("report_limit", 2500, "currency:USD", "report over $2,500.00"),
                    _c("receipt_floor", 25.0, "currency:USD", "$25.00 or more"),
                ],
            ),
            node(
                "pick_reading_list",
                4,
                **pure,
                quote="keep the first five results",
                constants=[_c("result_limit", 5, "count", "keep the first five results")],
            ),
        ],
    }


@pytest.fixture
def skill(tmp_path: Path) -> Any:
    d = tmp_path / "skill"
    d.mkdir()
    (d / "SKILL.md").write_text(SKILL)
    return replace(parse_skill(d), units=structure.section_units(d))


def _check(skill: Any, a: dict[str, Any]) -> list[str]:
    return structure.check(a, skill, SKILL, structure.declared_tools(skill, {}))


def _decide(skill: Any, *answers: dict[str, Any]) -> tuple[Any, Any]:
    fake = FakeCall(*answers)
    return structure.decide(skill, SKILL, "rubric", Lane("claude-cli"), tools={}, call=fake)


def test_measured_prose_constants_compile_to_numbers_with_units(skill: Any) -> None:
    assert _check(skill, answer()) == []
    plan, rec = _decide(skill, answer())
    assert rec.rounds == 1
    by_id = {n.id: n for n in plan.nodes}
    assert by_id["check_completion"].constants == {"completion_floor": 75}
    assert by_id["check_completion"].constant_units == {"completion_floor": "percent"}
    assert by_id["check_sorters"].constants == {"max_fail_rate": 5, "max_volume_drop": 15}
    assert by_id["audit_reports"].constants == {
        "policy_window": 60,
        "report_limit": 2500,
        "receipt_floor": 25.0,
    }
    assert by_id["audit_reports"].constant_units == {
        "policy_window": "days",
        "report_limit": "currency:USD",
        "receipt_floor": "currency:USD",
    }
    # The reviewer's record of where each came from is the skill's own words.
    assert [c.phrase for c in by_id["audit_reports"].unit.constants] == [
        "The policy window is 60 days",
        "report over $2,500.00",
        "$25.00 or more",
    ]
    pipeline = Pipeline.model_validate(pipeline_doc(plan))
    audit = next(n for n in pipeline.nodes if n.id == "audit_reports")
    # The impl receives exactly the numbers: units ride beside, never inside.
    assert audit.constants == {"policy_window": 60, "report_limit": 2500, "receipt_floor": 25.0}
    assert audit.constant_units == {
        "policy_window": "days",
        "report_limit": "currency:USD",
        "receipt_floor": "currency:USD",
    }


def test_a_number_read_from_words_is_kept_and_listed_for_review(skill: Any) -> None:
    plan, rec = _decide(skill, answer())
    assert rec.contradictions == [[]]
    pick = next(n for n in plan.nodes if n.id == "pick_reading_list")
    assert pick.constants == {"result_limit": 5}
    low = [d for d in pick.decisions if d.field == "constants.result_limit"]
    assert len(low) == 1 and low[0].confidence < REVIEW_BELOW
    assert [r for r in plan.review if "result_limit" in r] == [
        f"`pick_reading_list` constants.result_limit = 5 (confidence "
        f"{low[0].confidence:.2f}): read from words, not digits: "
        "'keep the first five results'"
    ]
    # A number the quote writes in digits is not listed: only the reading is.
    assert not any("completion_floor" in r or "report_limit" in r for r in plan.review)
    # A boolean is always a reading of words; a string is copied, never read.
    assert structure._from_words(_c("include_drafts", True, "", "include drafts"))
    assert not structure._from_words(_c("label", "BDR", "", "label BDR"))


@pytest.mark.parametrize(
    ("node_index", "constant", "expected"),
    [
        (
            0,
            _c("completion_floor", 0.75, "percent", "below 75% completion"),
            "node check_completion: constant 'completion_floor' = 0.75: its quote "
            "'below 75% completion' writes 75, not 0.75; give the number the skill writes, "
            "with its sign (75% is 75 with unit percent)",
        ),
        (
            2,
            _c("report_limit", 2000, "currency:USD", "report over $2,500.00"),
            "node audit_reports: constant 'report_limit' = 2000: its quote "
            "'report over $2,500.00' writes 2500.00, not 2000; give the number the skill "
            "writes, with its sign (75% is 75 with unit percent)",
        ),
        (
            2,
            _c("report_limit", 2500, "dollars", "report over $2,500.00"),
            "node audit_reports: constant 'report_limit': unit 'dollars' is not one of "
            "percent, count, seconds, minutes, hours, days, weeks, months, "
            "currency:<ISO code>, or ''",
        ),
        (
            2,
            _c("policy_window", "60 days", "days", "The policy window is 60 days"),
            "node audit_reports: constant 'policy_window' = '60 days' has unit 'days', "
            "but only a number has a unit",
        ),
        (
            2,
            _c("policy_window", "90 days", "", "The policy window is 60 days"),
            "node audit_reports: constant 'policy_window' = '90 days' does not appear in "
            "its quote 'The policy window is 60 days'",
        ),
        (
            3,
            _c("result_limit", 5, "count", "keep the first five"),
            None,
        ),
        (
            # The skill writes the sign, so dropping it is a different rule.
            1,
            _c("max_fail_rate", 5, "percent", "a margin below -5%"),
            "node check_sorters: constant 'max_fail_rate' = 5: its quote "
            "'a margin below -5%' writes -5, not 5; give the number the skill writes, "
            "with its sign (75% is 75 with unit percent)",
        ),
        (
            1,
            _c("max_fail_rate", -5, "percent", "a margin below -5%"),
            None,
        ),
        (
            # A negative reading of an unsigned number is kept (and listed).
            1,
            _c("max_volume_drop", -15, "percent", "volume dropped >15%"),
            None,
        ),
    ],
)
def test_a_constant_is_checked_against_its_own_quote(
    skill: Any, node_index: int, constant: dict[str, Any], expected: str | None
) -> None:
    a = answer()
    consts = a["nodes"][node_index]["constants"]
    a["nodes"][node_index]["constants"] = [
        constant if c["name"] == constant["name"] else c for c in consts
    ]
    assert _check(skill, a) == ([] if expected is None else [expected])


def test_a_sign_the_skill_does_not_write_is_listed_for_review(skill: Any) -> None:
    a = answer()
    a["nodes"][1]["constants"][1] = _c("max_volume_drop", -15, "percent", "volume dropped >15%")
    plan, rec = _decide(skill, a)
    assert rec.contradictions == [[]]
    sorters = next(n for n in plan.nodes if n.id == "check_sorters")
    assert sorters.constants["max_volume_drop"] == -15
    assert [r for r in plan.review if "max_volume_drop" in r] == [
        next(
            f"`check_sorters` constants.max_volume_drop = -15 (confidence "
            f"{d.confidence:.2f}): negative, though the skill writes no sign: "
            "'volume dropped >15%'"
            for d in sorters.decisions
            if d.field == "constants.max_volume_drop"
        )
    ]
    # A positive value of the same quote is not a reading of a sign.
    assert not any("max_fail_rate" in r for r in plan.review)


@pytest.mark.parametrize(
    ("quote", "numbers"),
    [
        ("a margin below -5%", {-5}),
        ("below \u22125.5 degrees", {Decimal("-5.5")}),
        ("between -5 and 5", {-5, 5}),
        ("wait 5-10 days", {5, 10}),
        ("after COVID-19", {19}),
        ("over $-2,500.00", {Decimal("-2500")}),
    ],
)
def test_a_written_number_keeps_only_a_sign_that_stands_alone(
    quote: str, numbers: set[Any]
) -> None:
    assert structure._written_numbers(quote) == {Decimal(n) for n in numbers}


def test_a_constant_that_survives_the_repairs_wrong_is_dropped(skill: Any) -> None:
    bad = answer()
    bad["nodes"][0]["constants"] = [_c("completion_floor", 0.75, "percent", "below 75%")]
    plan, rec = _decide(skill, *[bad] * (structure.REPAIR_ROUNDS + 1))
    check = next(n for n in plan.nodes if n.id == "check_completion")
    assert check.constants == {} and check.constant_units == {}
    assert any("check_completion" in r and "dropped constants" in r for r in rec.resolved)


def test_the_answer_schema_fits_structured_outputs() -> None:
    item = structure.SCHEMA["properties"]["nodes"]["items"]["properties"]["constants"]["items"]
    assert item["required"] == ["name", "value", "unit", "quote"]
    assert [b["type"] for b in item["properties"]["value"]["anyOf"]] == [
        "number",
        "boolean",
        "string",
    ]
    structured_output_schema(structure.SCHEMA, node_id=structure.NODE_ID)  # raises if not


def _node(**over: Any) -> Node:
    base: dict[str, Any] = {
        "id": "audit",
        "kind": "pure_function",
        "description": "audit",
        "impl": "extracted/audit.py:audit",
        "constants": {"report_limit": 2500, "label": "BDR"},
    }
    return Node.model_validate({**base, **over})


@pytest.mark.parametrize(
    ("units", "message"),
    [
        ({"missing": "count"}, "constant_units names 'missing', not a constant"),
        ({"label": "count"}, "constant 'label' has unit 'count' but is not a number"),
        ({"report_limit": "usd"}, "unit 'usd' must be one of"),
        ({"report_limit": "currency:usd"}, "unit 'currency:usd' must be one of"),
        ({"report_limit": "percent\nimport os"}, "must be one of"),
    ],
)
def test_the_ir_holds_units_to_the_closed_set(units: dict[str, str], message: str) -> None:
    assert _node(constant_units={"report_limit": "currency:USD"}).constant_units == {
        "report_limit": "currency:USD"
    }
    with pytest.raises(ValidationError, match=message):
        _node(constant_units=units)


def test_the_module_card_states_each_constants_type_and_unit() -> None:
    lines = generate._constant_lines(
        _node(
            constants={"report_limit": 2500, "floor": 75, "label": "BDR"},
            constant_units={"report_limit": "currency:USD", "floor": "percent"},
        )
    )
    assert lines == [
        "- `report_limit`: int = 2500 (currency:USD)",
        "- `floor`: int = 75 (percent)",
        "- `label`: str = 'BDR'",
        generate.CONSTANT_USE,
    ]
    assert "75 means 75 percent" in generate.CONSTANT_USE
    assert "never parse it from text" in generate.CONSTANT_USE


def test_the_compile_report_lists_each_constant_with_its_unit(skill: Any, tmp_path: Path) -> None:
    plan, _ = _decide(skill, answer())
    pipeline = Pipeline.model_validate(pipeline_doc(plan))
    report = render_compile_report(pipeline, tmp_path)
    line = next(x for x in report.splitlines() if x.startswith("- Constants: "))
    assert "`check_completion.completion_floor` = 75 (percent)" in line
    assert "`audit_reports.report_limit` = 2500 (currency:USD)" in line
    assert "`pick_reading_list.result_limit` = 5 (count)" in line


def test_the_rules_path_reads_units_from_the_symbols_it_matched() -> None:
    found = constants_in(
        "Flag if fail rate > 5% or volume dropped >15%.\n"
        "Only rows from the last 7 days.\n"
        "Batches of 25 contacts per call.\n"
        "Flag totals over $2,500 or more than 60 records in the past 3 hours.",
        1,
    )
    assert [(c.value, c.unit) for c in found] == [
        (5, "percent"),
        (15, "percent"),
        (7, "days"),
        (25, ""),
        (2500, ""),  # a `$` names no single currency
        (60, ""),  # "more than" is not a time unit
        (3, "hours"),
    ]


def test_the_rules_path_carries_units_into_the_skeleton(tmp_path: Path) -> None:
    examples = Path(__file__).resolve().parent.parent / "examples"
    work = tmp_path / "work"
    work.mkdir()
    plan = system1.prepare(examples / "expense-audit" / "skill", work)
    late = next(n for n in plan.nodes if n.id == "check_late_submissions")
    # "The policy window is 60 days" and "more than 60 days" are one constant.
    assert (late.constants, late.constant_units) == ({"threshold": 60}, {"threshold": "days"})
    doc = pipeline_doc(plan)
    assert next(n for n in doc["nodes"] if n["id"] == late.id)["constant_units"] == {
        "threshold": "days"
    }
    # A unit is a decision: the text agent cannot turn 60 days into 60 hours.
    written = yaml.safe_load((work / "pipeline.yaml").read_text())
    entry = next(n for n in written["nodes"] if n["id"] == late.id)
    entry["constant_units"] = {"threshold": "hours"}
    (work / "pipeline.yaml").write_text(yaml.safe_dump(written, sort_keys=False))
    changes = enforce(work)
    restored = yaml.safe_load((work / "pipeline.yaml").read_text())
    assert next(n for n in restored["nodes"] if n["id"] == late.id)["constant_units"] == {
        "threshold": "days"
    }
    assert any(f"`{late.id}` constant_units:" in c for c in changes)
