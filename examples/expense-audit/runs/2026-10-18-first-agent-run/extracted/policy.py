"""
Policy rule application for the expense audit pipeline.

Applies every rule from references/expense-policy.md to the converted
(USD) expense lines and produces structured findings plus auto-approval
decisions.

All rules in this module are deterministic boolean/numeric comparisons
against fixed policy thresholds. No LLM reasoning is involved.

Dependencies: none (stdlib only).
"""

from __future__ import annotations

from typing import TypedDict

from extracted.checks import Finding
from extracted.fx import ConvertedLine

# ---------------------------------------------------------------------------
# Policy constants — sourced from references/expense-policy.md
# ---------------------------------------------------------------------------

# Minimum line amount (USD) requiring a receipt attachment.
RECEIPT_REQUIRED_THRESHOLD: float = 25.00

# Maximum single-employee report total (USD) before controller review is
# required, even when every individual line is clean.
REPORT_TOTAL_LIMIT: float = 2500.00

# Category caps in USD.
# Key: expense tool category code. Value: cap per line (or per night for
# lodging). None means "no cap" (entertainment); lines NOT in this dict
# are unknown categories (BLOCK).
CATEGORY_CAPS: dict[str, float | None] = {
    "airfare": 900.00,
    "lodging": 325.00,    # cap is PER NIGHT: 325.00 × line.nights
    "meals": 75.00,
    "ground_transport": 150.00,
    "software": 100.00,
    "office_supplies": 250.00,
    "entertainment": None,  # no cap, but always needs preapproval_id
}

KNOWN_CATEGORIES: frozenset[str] = frozenset(CATEGORY_CAPS.keys())


def apply_policy_rules(
    converted_lines: list[ConvertedLine],
    duplicate_findings: list[Finding],
    late_findings: list[Finding],
) -> dict[str, object]:
    """
    Group converted lines by employee and apply every policy rule.

    Rules applied (all sourced from expense-policy.md):
    1. Category cap exceeded:
       - Most categories: line.usd_amount > CATEGORY_CAPS[category]
       - Lodging: line.usd_amount > CATEGORY_CAPS["lodging"] × line.nights
       Severity: review (needs written exception per policy; controller decides)
    2. Receipt missing: line.usd_amount >= RECEIPT_REQUIRED_THRESHOLD
       and line.receipt is False. Severity: block.
    3. Entertainment without pre-approval: category == "entertainment"
       and preapproval_id is None or empty. Severity: block.
    4. Unknown category: category not in KNOWN_CATEGORIES. Severity: block.
    5. Duplicate findings from check_duplicates (passed in). Severity: block.
    6. Late findings from check_late_submissions (passed in). Severity: review.
    7. Report total limit: employee's sum of usd_amount > REPORT_TOTAL_LIMIT,
       even with no other findings. Severity: review.

    A report is AUTO-APPROVED when it has NO findings at all (including no
    report-total finding). Do not ask the controller about auto-approved reports.

    Returns a dict with keys:
    - "findings_by_employee": dict[str, list[Finding]]
        Maps employee name → list of findings (all severities).
    - "auto_approved_employees": list[str]
        Employees whose reports have zero findings.
    - "employee_totals_usd": dict[str, float]
        Sum of usd_amount per employee, across all their lines.
    - "employee_line_counts": dict[str, int]
        Number of expense lines per employee.
    - "employee_departments": dict[str, str]
        Department per employee (from the first line seen for that employee).
    """
    raise NotImplementedError(
        "implement: group converted_lines by employee; for each employee's "
        "lines apply rules 1-4; merge in duplicate_findings and late_findings "
        "filtered to that employee; compute usd total; flag report_total "
        "when total > REPORT_TOTAL_LIMIT; auto-approve if no findings."
    )
