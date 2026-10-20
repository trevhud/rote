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


def test_a_heading_inside_a_fenced_template_is_citable(tmp_path: Path) -> None:
    """A skill's report template in a markdown fence is content, not a
    heading. A real compile was bounced for citing it."""
    (tmp_path / "SKILL.md").write_text(
        "# Returns triage\n\nThe report looks like this:\n\n"
        "```markdown\n# Returns report: <week> by store\n\nStatus: final\n```\n"
    )
    assert citation_problem("# Returns report: <week> by store", skill_paragraphs(tmp_path)) is None
    # The real heading outside the fence is still not citable prose.
    assert citation_problem("Returns triage", skill_paragraphs(tmp_path)) is not None


def test_a_whole_table_row_is_citable(tmp_path: Path) -> None:
    """The rubric's worked example: cite a row whole, pipes included."""
    (tmp_path / "SKILL.md").write_text(
        "| Reason | Code | Window | Notes |\n| --- | --- | --- | --- |\n"
        "| Wrong Item | `wrong_item` | 30 days | Refund once the item is back in stock. |\n"
    )
    row = "| Wrong Item | wrong_item | 30 days | Refund once the item is back in stock. |"
    assert citation_problem(row, skill_paragraphs(tmp_path)) is None


def test_a_markdown_link_is_cited_by_its_text(tmp_path: Path) -> None:
    (tmp_path / "SKILL.md").write_text(
        "Look up the period's month-end rates in\n"
        "[references/fx-rates.md](references/fx-rates.md) and convert each line.\n"
    )
    quote = "month-end rates in references/fx-rates.md and convert each line"
    assert citation_problem(quote, skill_paragraphs(tmp_path)) is None


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


@pytest.mark.parametrize(
    ("source", "unmarked"),
    [
        # A policy table, then a lookup: the natural expense-audit style.
        (
            'RULES = {"over_cap": "block", "late": "review"}\n\n\n'
            "def audit(rule):\n"
            '    return {"severity": RULES[rule]}\n',
            ["block", "review"],
        ),
        (
            'RULES = {"over_cap": "block"}\n\n\n'
            "def audit(rule):\n"
            '    return {"severity": RULES.get(rule, "review")}\n',
            ["block", "review"],
        ),
        # A table assigned to an outcome-named variable is itself outcomes.
        ('severity_by_rule = {"over_cap": "block"}\n', ["block"]),
        # A module constant used by name.
        ('BLOCK = "block"\n\n\ndef audit(r):\n    return {"severity": BLOCK}\n', ["block"]),
        # An Enum member.
        (
            "from enum import StrEnum\n\n\n"
            "class Severity(StrEnum):\n"
            '    BLOCK = "block"\n'
            '    REVIEW = "review"\n\n\n'
            "def audit(r):\n"
            "    return Finding(severity=Severity.BLOCK)\n",
            ["block"],
        ),
        # A tuple return.
        (
            'def route(r):\n    return ("manual_review", "no receipt")\n',
            ["manual_review", "no receipt"],
        ),
    ],
)
def test_labels_reached_by_reference_or_container_need_markers(
    source: str, unmarked: list[str]
) -> None:
    assert sorted(lit.value for lit in unmarked_outcomes(source)) == sorted(unmarked)


def test_a_marker_at_the_definition_covers_a_referenced_label() -> None:
    source = (
        "# rote: assumption: an over-cap line is a block; the skill names the rule only\n"
        'SEVERITY_BY_RULE = {"over_cap": "block"}\n\n\n'
        "def audit(rule):\n"
        '    return {"severity": SEVERITY_BY_RULE[rule]}\n'
    )
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


def test_each_assumption_gets_its_own_open_question(tmp_path: Path) -> None:
    """rote renders the Open questions, one entry per distinct assumption,
    so the agent can no longer under-count them (a gap that used to cost a
    repair round), and no provenance finding asks it to."""
    from rote.compiler.report import render_compile_report

    (tmp_path / "extracted").mkdir()
    (tmp_path / "extracted" / "route.py").write_text(_TWO_ASSUMPTIONS)
    assert provenance_findings(_pipeline(), tmp_path, None) == []
    entries = open_questions(render_compile_report(_pipeline(), tmp_path))
    assert entries == ["### OQ-1: route", "### OQ-2: route"]


def test_one_assumption_marked_twice_is_one_question(tmp_path: Path) -> None:
    """A real report listed the "-" placeholder assumption once although the
    renderer marked it at two call sites, and was bounced for it. The
    rendered report lists it once, at both locations."""
    from rote.compiler.report import render_compile_report

    twice = (
        "def route(ret):\n"
        "    # rote: assumption: an unlisted reason is manual review\n"
        '    first = "manual_review"\n'
        "    # rote: assumption: an unlisted  reason is Manual Review\n"
        '    second = "manual_review"\n'
        "    return first, second\n"
    )
    (tmp_path / "extracted").mkdir()
    (tmp_path / "extracted" / "route.py").write_text(twice)
    report = render_compile_report(_pipeline(), tmp_path)
    assert open_questions(report) == ["### OQ-1: route"]
    assert "`extracted/route.py:2`, `extracted/route.py:4`" in report


def test_a_punctuation_placeholder_is_not_an_outcome() -> None:
    source = 'def f(alert):\n    flag = "—" if alert is None else alert\n    return flag\n'
    assert unmarked_outcomes(source) == []


def test_a_fenced_comment_does_not_end_the_open_questions() -> None:
    """A real report quoted its `# rote: assumption:` line in a code fence;
    read as a heading, it cut the section off after three of nine entries."""
    report = (
        "## Open questions\n\n### OQ-1: one\n**Assumption marked in code:**\n"
        "```python\n# rote: assumption: something\n```\n\n"
        "### OQ-2: two\n\n### OQ-3: three\n\n## Next steps\n"
    )
    assert [q.split(":")[0] for q in open_questions(report)] == [
        "### OQ-1",
        "### OQ-2",
        "### OQ-3",
    ]


def test_a_citation_that_only_changes_the_quote_style_is_verbatim(tmp_path: Path) -> None:
    """The real BDR case: inside its own double-quoted citation the agent
    wrote the skill's "BDR do not contact" with single quotes, and the
    compile failed on a citation that was verbatim but for the quote mark."""
    (tmp_path / "SKILL.md").write_text(
        '1. **Do-not-contact list** — search for "BDR do not contact" list, check each '
        "contact's memberships\n"
    )
    paragraphs = skill_paragraphs(tmp_path)
    quote = (
        "Do-not-contact list — search for 'BDR do not contact' list, "
        "check each contact's memberships"
    )
    assert citation_problem(quote, paragraphs) is None
    curly = "search for “BDR do not contact” list, check each contact’s memberships"
    assert citation_problem(curly, paragraphs) is None
    # The negative half: a changed word is still not the skill.
    assert citation_problem(quote.replace("each", "every"), paragraphs) is not None


def test_open_questions_counts_subheadings_not_their_detail() -> None:
    report = "## Open questions\n\n### OQ-1\n- a\n- b\n\n### OQ-2\n- c\n\n## Next\n- d\n"
    assert len(open_questions(report)) == 2
