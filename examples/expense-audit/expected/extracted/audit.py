"""Steps 4 and 5: per-employee policy audit and the batch summary.

Every cap and threshold below is a rule from references/expense-policy.md.
They are module constants, not prompt text, so a rule can't be skipped or
re-interpreted from one month to the next.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from extracted.checks import expense_sort_key
from extracted.fx import to_cents

#: Per-line caps in USD. Lodging is per night (the line carries ``nights``).
CATEGORY_CAPS_USD: dict[str, Decimal] = {
    "airfare": Decimal("900.00"),
    "lodging": Decimal("325.00"),
    "meals": Decimal("75.00"),
    "ground_transport": Decimal("150.00"),
    "software": Decimal("100.00"),
    "office_supplies": Decimal("250.00"),
}
#: No cap, but always needs a pre-approval id.
PREAPPROVAL_CATEGORIES = frozenset({"entertainment"})
KNOWN_CATEGORIES = frozenset(CATEGORY_CAPS_USD) | PREAPPROVAL_CATEGORIES
RECEIPT_THRESHOLD_USD = Decimal("25.00")
REPORT_LIMIT_USD = Decimal("2500.00")


def _usd(value: Decimal) -> str:
    return f"${value:,.2f}"


def group_by_employee(expenses: list[dict[str, Any]]) -> dict[str, Any]:
    """One report per employee, sorted by name; lines sorted by expense id."""
    by_employee: dict[str, list[dict[str, Any]]] = {}
    for expense in expenses:
        by_employee.setdefault(expense["employee"], []).append(expense)
    reports = [
        {
            "employee": employee,
            "department": lines[0]["department"],
            "expenses": sorted(lines, key=lambda e: expense_sort_key(e["id"])),
        }
        for employee, lines in sorted(by_employee.items())
    ]
    return {"reports": reports}


def _line_findings(
    expense: dict[str, Any], duplicate_ids: set[str], late_ids: set[str]
) -> list[dict[str, str]]:
    amount = Decimal(str(expense["amount_usd"]))
    category = expense["category"]
    found: list[tuple[str, str, str]] = []

    if category not in KNOWN_CATEGORIES:
        found.append(("unknown_category", "block", f"category {category!r} is not in the policy"))
    elif category in PREAPPROVAL_CATEGORIES:
        if not expense.get("preapproval_id"):
            found.append(("preapproval_missing", "review", f"{category} without a pre-approval id"))
    else:
        cap = CATEGORY_CAPS_USD[category]
        if category == "lodging":
            cap *= int(expense.get("nights") or 1)
        if amount > cap:
            found.append(("over_cap", "review", f"{_usd(amount)} over the {_usd(cap)} {category} cap"))

    if amount >= RECEIPT_THRESHOLD_USD and not expense["receipt"]:
        found.append(("receipt_missing", "block", f"no receipt on a {_usd(amount)} line"))
    if expense["id"] in duplicate_ids:
        found.append(("duplicate", "block", "same employee, merchant, amount, and date as an earlier line"))
    if expense["id"] in late_ids:
        found.append(("late", "review", f"dated {expense['date']}, outside the 60-day window"))

    return [
        {
            "expense_id": expense["id"],
            "employee": expense["employee"],
            "rule": rule,
            "severity": severity,
            "detail": detail,
        }
        for rule, severity, detail in found
    ]


def audit_report(
    report: dict[str, Any], duplicate_ids: list[str], late_ids: list[str]
) -> dict[str, Any]:
    """Apply every policy rule to one employee's report."""
    duplicates, late = set(duplicate_ids), set(late_ids)
    findings: list[dict[str, str]] = []
    for expense in report["expenses"]:
        findings.extend(_line_findings(expense, duplicates, late))

    total = to_cents(sum((Decimal(str(e["amount_usd"])) for e in report["expenses"]), Decimal(0)))
    if total > REPORT_LIMIT_USD:
        findings.append(
            {
                "expense_id": "(report)",
                "employee": report["employee"],
                "rule": "report_limit",
                "severity": "review",
                "detail": f"report total {_usd(total)} over the {_usd(REPORT_LIMIT_USD)} limit",
            }
        )
    return {
        "employee": report["employee"],
        "department": report["department"],
        "lines": len(report["expenses"]),
        "total_usd": float(total),
        "findings": findings,
        "auto_approved": not findings,
    }


def summarize_audit(audits: list[dict[str, Any]], expenses: list[dict[str, Any]]) -> dict[str, Any]:
    """Batch total, auto-approved employees, and the sorted review queue."""
    queue = [f for audit in audits for f in audit["findings"]]
    queue.sort(key=lambda f: (f["employee"], expense_sort_key(f["expense_id"]), f["rule"]))
    batch_total = to_cents(sum((Decimal(str(e["amount_usd"])) for e in expenses), Decimal(0)))
    return {
        "batch_total_usd": float(batch_total),
        "employees": [{k: v for k, v in a.items() if k != "findings"} for a in audits],
        "auto_approved": [a["employee"] for a in audits if a["auto_approved"]],
        "review_queue": queue,
        "expenses": expenses,
    }
