"""
Review queue assembly and audit report rendering for the expense audit.

All output formats are fixed (sourced from references/report-format.md).
No LLM reasoning is involved.

Dependencies: none (stdlib only).
"""

from __future__ import annotations

from typing import TypedDict

from extracted.checks import Finding


class QueueRow(TypedDict):
    expense_id: str
    employee: str
    rule: str
    severity: str   # "block" | "review"
    detail: str
    decision: str   # "pending" | "approved" | "rejected" | "returned"


class BatchSummary(TypedDict):
    # All USD amounts at this stage are pre-decision totals.
    total_lines: int
    total_usd: float
    auto_approved_count: int
    review_count: int   # number of REVIEW findings (controller decides)
    block_count: int    # number of BLOCK findings (returned regardless)


def build_review_queue(
    findings_by_employee: dict[str, list[Finding]],
    auto_approved_employees: list[str],
    employee_totals_usd: dict[str, float],
) -> dict[str, object]:
    """
    Aggregate all findings into a sorted review queue and compute initial
    batch totals.

    Sort order: by employee name (ascending), then by expense_id (ascending)
    within each employee.

    Auto-approved employees contribute to the batch totals but have no rows
    in the review queue.

    Returns a dict with keys:
    - "review_queue": list[QueueRow]
        All findings, sorted as above, with decision="pending" for REVIEW
        items and decision="returned" for BLOCK items.
    - "batch_summary": BatchSummary
        Aggregate counts and USD totals across the entire batch. USD totals
        at this stage do NOT yet reflect controller decisions — they are used
        for informational display in the HITL notification.
    """
    raise NotImplementedError(
        "implement: flatten findings_by_employee into QueueRow items; "
        "set decision='returned' for block and 'pending' for review; "
        "sort by (employee, expense_id); compute BatchSummary from "
        "employee_totals_usd and auto_approved_employees."
    )


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

# Fixed report layout from references/report-format.md.
# Placeholders use Python .format() syntax.
_REPORT_TEMPLATE = """\
# Expense audit: {period}

Status: **{status}** (reviewed by {controller})

| | USD |
| --- | --- |
| Reimbursable | ${reimbursable:.2f} |
| Pending review | ${pending_review:.2f} |
| Returned to employee | ${returned:.2f} |
| Batch total | ${batch_total:.2f} |

## Employees

| Employee | Department | Lines | Total | Outcome |
| --- | --- | --- | --- | --- |
{employee_rows}

## Review queue

| Expense | Employee | Rule | Severity | Detail | Decision |
| --- | --- | --- | --- | --- | --- |
{queue_rows}
"""


def render_audit_report(
    period: str,
    controller: str,
    auto_approved_employees: list[str],
    employee_totals_usd: dict[str, float],
    employee_line_counts: dict[str, int],
    employee_departments: dict[str, str],
    review_queue: list[QueueRow],
    decisions: dict[str, str],
) -> dict[str, str]:
    """
    Render the final audit report in the fixed layout from report-format.md.

    Reimbursability rules (per expense-policy.md and SKILL.md Step 7):
    - A line is reimbursable when it has no findings, OR when all of its
      findings are REVIEW items that the controller approved.
    - Blocked and rejected lines are returned to the employee.
    - An auto-approved employee's entire report is reimbursable.

    Employee outcome values (per report-format.md):
    - "auto-approved": no findings at all.
    - "reviewed": controller decided every REVIEW finding on their report.
    - "pending review": at least one REVIEW finding still has no decision.

    Report status:
    - "final" when all REVIEW findings have a controller decision.
    - "draft" when any REVIEW finding is still pending.

    Decision column values (per report-format.md):
    - BLOCK rows always read "returned".
    - REVIEW rows read the controller's decision ("approved" / "rejected"),
      or "pending" when the report is still a draft.

    Args:
        decisions: maps expense_id → "approved" | "rejected". BLOCK items
                   are not included — their decision is always "returned".

    Returns:
        {"report_markdown": str}
        The complete report as a markdown string, ready to write to disk.
    """
    raise NotImplementedError(
        "implement: compute per-employee outcomes from review_queue + decisions; "
        "compute reimbursable/returned/pending totals from employee_totals_usd "
        "and decisions; fill _REPORT_TEMPLATE; mark status 'final' when no "
        "REVIEW rows remain pending."
    )
