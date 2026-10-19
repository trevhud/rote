"""The citation and outcome-marker checks cannot be satisfied for free.

Each case here is one a reviewer used to get a guessed outcome past the
first version of the check, against a toy skill. A check that accepts any
of them tells a reviewer nothing, so each must fail, and the honest
version of the same code must pass.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rote.compiler.provenance import (
    citation_problem,
    citations,
    open_questions,
    provenance_findings,
    skill_paragraphs,
    unmarked_outcomes,
)
from rote.ir import Node, NodeKind, Pipeline

SKILL = """\
# Returns triage

## Review

Every return is either refunded at once or sent to manual review.

A return over $50.00 without a receipt goes to manual review. Returns
marked "Damaged in Transit" are refunded at once and the carrier is billed.

Staff must log each decision in the returns sheet.
"""


@pytest.fixture
def paragraphs(tmp_path: Path) -> list[str]:
    (tmp_path / "SKILL.md").write_text(SKILL)
    return skill_paragraphs(tmp_path)


# ───────── Citations ─────────


@pytest.mark.parametrize(
    "quote",
    [
        "Review",  # a heading, and one word
        "must",  # one word that appears in the prose
    ],
)
def test_one_word_and_heading_quotes_are_rejected(quote: str, paragraphs: list[str]) -> None:
    assert citation_problem(quote, paragraphs) is not None


def test_a_heading_is_not_prose_even_when_long_enough(tmp_path: Path) -> None:
    (tmp_path / "SKILL.md").write_text("# Returns triage\n\n## How every return is routed\n\nx\n")
    problem = citation_problem("How every return is routed", skill_paragraphs(tmp_path))
    assert problem is not None and "headings do not count" in problem


def test_fragments_must_appear_in_order(paragraphs: list[str]) -> None:
    in_order = "A return over $50.00 ... goes to manual review"
    reversed_ = "goes to manual review ... A return over $50.00"
    assert citation_problem(in_order, paragraphs) is None
    problem = citation_problem(reversed_, paragraphs)
    assert problem is not None and "in this order" in problem


def test_fragments_must_come_from_one_paragraph(paragraphs: list[str]) -> None:
    across = "refunded at once or sent ... must log each decision"
    assert citation_problem(across, paragraphs) is not None


def test_a_short_fragment_inside_an_elision_is_rejected(paragraphs: list[str]) -> None:
    problem = citation_problem("A return over $50.00 ... review", paragraphs)
    assert problem is not None and "under 4 words" in problem


def test_a_quote_may_contain_double_quotes(paragraphs: list[str]) -> None:
    """Skill prose quotes labels. The old `[^"]+` pattern could not cite them."""
    source = (
        "def route(ret):\n"
        '    # skill: "Returns marked "Damaged in Transit" are refunded at once"\n'
        '    return "refund"\n'
    )
    [cite] = citations(source)
    assert cite.quote == 'Returns marked "Damaged in Transit" are refunded at once'
    assert citation_problem(cite.quote, paragraphs) is None


def test_a_wrapped_quote_continues_on_comment_lines() -> None:
    source = '# skill: "A return over $50.00 without a receipt\n#   goes to manual review"\nx = 1\n'
    [cite] = citations(source)
    assert cite.quote == "A return over $50.00 without a receipt goes to manual review"


def test_an_unclosed_quote_is_reported_not_guessed() -> None:
    source = '# skill: "A return over $50.00 without a receipt\nroute = "manual_review"\n'
    [cite] = citations(source)
    assert cite.quote is None


# ───────── Outcome literals need a marker ─────────


@pytest.mark.parametrize(
    "source",
    [
        'OUTCOME = "block"\n',
        'def route(ret):\n    return "manual_review"\n',
        'def f(r):\n    return {"severity": "block", "id": r}\n',
        'def f(r):\n    findings.append(Finding(rule="cap", severity="block"))\n',
        'def f(row):\n    row["alert_level"] = "missort_alert" if row["n"] > 40 else "none"\n',
    ],
)
def test_an_outcome_with_no_marker_is_reported(source: str) -> None:
    assert unmarked_outcomes(source), source


@pytest.mark.parametrize(
    "source",
    [
        '# rote: assumption: over-cap is block\nOUTCOME = "block"\n',
        "def route(ret):\n"
        '    # skill: "A return over $50.00 without a receipt goes to manual review"\n'
        '    return "manual_review"\n',
        # A comment above an if/elif chain covers the whole chain.
        "def f(n):\n"
        '    # skill: "packages >= 25 (Missort Warning), >= 41 (Missort Alert)"\n'
        "    if n >= 41:\n"
        '        return "Missort Alert"\n'
        "    if n >= 25:\n"
        '        return "Missort Warning"\n'
        '    return ""\n',
        # Strings that are not outcomes need nothing.
        'def f(raw):\n    rows = raw.split(",")\n    return rows\n',
        'def f(p):\n    return open(p, encoding="utf-8").read()\n',
    ],
)
def test_a_marked_outcome_or_a_non_outcome_passes(source: str) -> None:
    assert unmarked_outcomes(source) == []


def test_a_citation_for_one_check_does_not_cover_the_next() -> None:
    """The real case: an expense-audit compile's over-cap severity sat in a
    second, independent `if`, with an unrelated lodging-nights citation a
    few lines up. Proximity "covered" it; the decision went unexamined."""
    source = (
        "def audit(line, findings):\n"
        '    # skill: "Entertainment always needs a pre-approval id"\n'
        '    if line["category"] == "entertainment":\n'
        '        findings.append({"severity": "block"})\n'
        '    if line["category"] in CAPS:\n'
        '        if line["category"] == "lodging":\n'
        '            # skill: "the cap is $325.00 times nights"\n'
        '            cap = CAPS["lodging"] * line["nights"]\n'
        '        if line["amount"] > cap:\n'
        '            findings.append({"severity": "review"})\n'
    )
    assert [lit.value for lit in unmarked_outcomes(source)] == ["review"]


def test_a_marker_does_not_reach_past_a_blank_line() -> None:
    source = (
        "def f(n):\n"
        '    # skill: "A return over $50.00 without a receipt goes to manual review"\n'
        '    first = "manual_review"\n'
        "\n"
        '    status = "refund"\n'
        "    return first, status\n"
    )
    [literal] = unmarked_outcomes(source)
    assert literal.value == "refund"


def test_a_marker_does_not_vouch_for_a_whole_long_function() -> None:
    body = "".join(f"    x{i} = {i}\n" for i in range(12))
    source = (
        "def f(n):\n    # rote: assumption: one marker at the top\n"
        + body
        + '    status = "refund"\n'
    )
    assert [lit.value for lit in unmarked_outcomes(source)] == ["refund"]


# ───────── Assumptions need their own Open questions ─────────


def _pipeline() -> Pipeline:
    node = Node(
        id="route",
        kind=NodeKind.PURE_FUNCTION,
        description="d",
        impl="extracted/route.py:route",
        inputs={"ret": "pipeline.input"},
    )
    return Pipeline(
        name="p",
        input={"type": "In", "required": []},
        nodes=[node],
        edges=[],
        entry_nodes=["route"],
        exit_nodes=["route"],
    )


_TWO_ASSUMPTIONS = (
    "def route(ret):\n"
    "    # rote: assumption: an unlisted reason is manual review\n"
    '    if ret["reason"] not in REASONS:\n'
    '        return "manual_review"\n'
    "    # rote: assumption: a missing amount is treated as zero\n"
    '    return "refund"\n'
)


@pytest.mark.parametrize(
    ("report", "ok"),
    [
        ("## Open questions\n\n- Unlisted reasons go to manual review.\n", False),
        (
            "## Open questions\n\n- Unlisted reasons go to manual review.\n"
            "- A missing amount counts as zero.\n",
            True,
        ),
        # Sub-headings are the entries; the bullets under one are its detail.
        (
            "## Open questions\n\n### OQ-1: unlisted reasons\n- we chose review\n"
            "- because nothing says otherwise\n",
            False,
        ),
    ],
)
def test_each_assumption_needs_its_own_open_question(tmp_path: Path, report: str, ok: bool) -> None:
    (tmp_path / "extracted").mkdir()
    (tmp_path / "extracted" / "route.py").write_text(_TWO_ASSUMPTIONS)
    (tmp_path / "compile-report.md").write_text(report)
    kinds = [f.kind for f in provenance_findings(_pipeline(), tmp_path, None)]
    assert (kinds == []) is ok, kinds
    if not ok:
        assert kinds == ["unlisted_assumption"]


def test_open_questions_counts_subheadings_not_their_detail() -> None:
    report = "## Open questions\n\n### OQ-1\n- a\n- b\n\n### OQ-2\n- c\n\n## Next\n- d\n"
    assert len(open_questions(report)) == 2
