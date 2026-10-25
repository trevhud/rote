"""Step 7: apply the controller's decisions and file the audit report.

The layout is references/report-format.md. The report is written to the
finance folder, ``$EXPENSE_AUDIT_REPORT_DIR`` (default ``reports/`` under
the working directory), as ``<period>-expense-audit.md``. Overwriting is
deliberate: a durable runtime may retry this step, and the same inputs
always render the same bytes.
"""

from __future__ import annotations

import os
from decimal import Decimal
from pathlib import Path
from typing import Any

from extracted.audit import REPORT_FINDING_PREFIX
from extracted.fx import to_cents

REPORT_DIR_ENV = "EXPENSE_AUDIT_REPORT_DIR"


def _usd(value: Decimal) -> str:
    return f"${value:,.2f}"


def _decisions(
    summary: dict[str, Any], review: dict[str, Any] | None
) -> tuple[dict[int, str], str, str | None, list[str]]:
    """Decision per queue row, the report status, the reviewer, and the
    block ids the decision tried to approve or reject (it cannot)."""
    queue = summary["review_queue"]
    if review is None:
        return (
            {i: "returned" if f["severity"] == "block" else "pending" for i, f in enumerate(queue)},
            "draft",
            None,
            [],
        )

    approved = set(review.get("approved_ids") or [])
    rejected = set(review.get("rejected_ids") or [])
    queued = {f["expense_id"] for f in queue}
    # The skill says to ask again rather than guess when an id "isn't in
    # the queue"; in a workflow that means failing loudly with the ids the
    # controller has to correct. A decision on a block item is not that:
    # block items are "for information only" and go back to the employee
    # "no matter what", so the decision is ignored below and the id is
    # reported back as overridden, as the skill run as an agent tells the
    # controller.
    unknown = sorted((approved | rejected) - queued)
    if unknown:
        raise ValueError(
            f"controller decision names id(s) that are not in the review queue: "
            f"{', '.join(unknown)} (queue: {', '.join(sorted(queued)) or 'empty'})."
        )
    both = sorted(approved & rejected)
    if both:
        raise ValueError(f"controller both approved and rejected: {', '.join(both)}")

    decisions: dict[int, str] = {}
    for i, f in enumerate(queue):
        if f["severity"] == "block":
            decisions[i] = "returned"
        elif f["expense_id"] in approved:
            decisions[i] = "approved"
        elif f["expense_id"] in rejected:
            decisions[i] = "rejected"
        else:
            decisions[i] = "pending"
    status = "draft" if "pending" in decisions.values() else "final"
    blocked = {f["expense_id"] for f in queue if f["severity"] == "block"}
    overridden = sorted((approved | rejected) & blocked)
    return decisions, status, review.get("reviewer"), overridden


def _line_outcomes(summary: dict[str, Any], decisions: dict[int, str]) -> dict[str, str]:
    """Each expense line's outcome: reimbursable, pending, or returned."""
    per_line: dict[str, list[str]] = {}
    per_report: dict[str, list[str]] = {}
    for i, f in enumerate(summary["review_queue"]):
        report_level = f["expense_id"].startswith(REPORT_FINDING_PREFIX)
        bucket = per_report if report_level else per_line
        key = f["employee"] if report_level else f["expense_id"]
        bucket.setdefault(key, []).append(decisions[i])

    outcomes: dict[str, str] = {}
    for expense in summary["expenses"]:
        ds = per_line.get(expense["id"], []) + per_report.get(expense["employee"], [])
        if any(d in ("returned", "rejected") for d in ds):
            outcomes[expense["id"]] = "returned"
        elif "pending" in ds:
            outcomes[expense["id"]] = "pending"
        else:
            outcomes[expense["id"]] = "reimbursable"
    return outcomes


def render_report(
    period: str,
    summary: dict[str, Any],
    decisions: dict[int, str],
    status: str,
    reviewer: str | None,
    totals: dict[str, Decimal],
) -> str:
    status_line = (
        f"Status: **final** (reviewed by {reviewer})"
        if status == "final"
        else "Status: **draft** (awaiting controller review)"
    )
    lines = [
        f"# Expense audit: {period}",
        "",
        status_line,
        "",
        "| | USD |",
        "| --- | --- |",
        f"| Reimbursable | {_usd(totals['reimbursable'])} |",
        f"| Pending review | {_usd(totals['pending'])} |",
        f"| Returned to employee | {_usd(totals['returned'])} |",
        f"| Batch total | {_usd(totals['batch'])} |",
        "",
        "## Employees",
        "",
        "| Employee | Department | Lines | Total | Outcome |",
        "| --- | --- | --- | --- | --- |",
    ]
    pending_employees = {
        f["employee"] for i, f in enumerate(summary["review_queue"]) if decisions[i] == "pending"
    }
    for e in summary["employees"]:
        if e["auto_approved"]:
            outcome = "auto-approved"
        elif e["employee"] in pending_employees:
            outcome = "pending review"
        else:
            outcome = "reviewed"
        total = Decimal(str(e["total_usd"]))
        lines.append(
            f"| {e['employee']} | {e['department']} | {e['lines']} | {_usd(total)} | {outcome} |"
        )
    lines += [
        "",
        "## Review queue",
        "",
        "| Expense | Employee | Rule | Severity | Detail | Decision |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for i, f in enumerate(summary["review_queue"]):
        lines.append(
            f"| {f['expense_id']} | {f['employee']} | {f['rule']} | {f['severity']} "
            f"| {f['detail']} | {decisions[i]} |"
        )
    return "\n".join(lines) + "\n"


def file_report(
    period: str, summary: dict[str, Any], review: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Apply decisions, write ``<period>-expense-audit.md``, return the totals.

    ``review`` is the controller's resume payload
    (``{"reviewer", "approved_ids", "rejected_ids"}``). Without one the
    report is filed as a draft with every review item pending.
    ``overridden_ids`` lists the block items the decision approved or
    rejected: they are returned anyway, and the caller should say so.
    """
    decisions, status, reviewer, overridden = _decisions(summary, review)
    outcomes = _line_outcomes(summary, decisions)

    totals = {"reimbursable": Decimal(0), "pending": Decimal(0), "returned": Decimal(0)}
    for expense in summary["expenses"]:
        totals[outcomes[expense["id"]]] += Decimal(str(expense["amount_usd"]))
    totals = {k: to_cents(v) for k, v in totals.items()}
    totals["batch"] = to_cents(Decimal(str(summary["batch_total_usd"])))

    report_dir = Path(os.environ.get(REPORT_DIR_ENV) or "reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{period}-expense-audit.md"
    report_path.write_text(
        render_report(period, summary, decisions, status, reviewer, totals), encoding="utf-8"
    )

    return {
        "report_path": str(report_path.resolve()),
        "status": status,
        "reviewer": reviewer,
        "reimbursable_usd": float(totals["reimbursable"]),
        "pending_usd": float(totals["pending"]),
        "returned_usd": float(totals["returned"]),
        "batch_total_usd": float(totals["batch"]),
        "auto_approved": summary["auto_approved"],
        "review_items": sum(1 for f in summary["review_queue"] if f["severity"] == "review"),
        "blocked_items": sum(1 for f in summary["review_queue"] if f["severity"] == "block"),
        "overridden_ids": overridden,
    }
