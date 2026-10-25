"""Steps 4 and 5: per-employee policy audit and the batch summary.

Every cap and threshold is a rule from references/expense-policy.md. The
pipeline's IR carries them as the audit node's constants, and the step
passes them on every call, so they live in code and config, never in
prompt text. They are required parameters with no default here: a copy of
the IR value as a default would let a constant that stopped arriving fall
back silently instead of failing.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from extracted.checks import expense_sort_key
from extracted.fx import to_cents

#: No cap, but always needs a pre-approval id.
PREAPPROVAL_CATEGORIES = frozenset({"entertainment"})


#: Prefix of a report-level finding's id. Each employee's gets its own id
#: so the controller can approve one over-limit report and reject another.
REPORT_FINDING_PREFIX = "REPORT:"


def report_finding_id(employee: str) -> str:
    return f"{REPORT_FINDING_PREFIX}{employee}"


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
    expense: dict[str, Any],
    duplicate_ids: set[str],
    late_ids: set[str],
    caps: dict[str, Decimal],
    receipt_threshold: Decimal,
) -> list[dict[str, str]]:
    amount = Decimal(str(expense["amount_usd"]))
    category = expense["category"]
    found: list[tuple[str, str, str]] = []

    if category not in frozenset(caps) | PREAPPROVAL_CATEGORIES:
        found.append(("unknown_category", "block", f"category {category!r} is not in the policy"))
    elif category in PREAPPROVAL_CATEGORIES:
        if not expense.get("preapproval_id"):
            # rote: assumption: entertainment without a pre-approval id is a block. The policy says it "Always needs a pre-approval id" and names no severity; the original skill blocked it in 11 of 12 recorded runs (fidelity/reference.json and the 2026-10-24 A/B runs).
            found.append(("preapproval_missing", "block", f"{category} without a pre-approval id"))
    else:
        cap = caps[category]
        if category == "lodging":
            cap *= int(expense.get("nights") or 1)
        if amount > cap:
            # rote: assumption: a line over its category cap is a block. The policy names no severity for caps ("Anything above the cap needs a written exception.", "Above the cap goes through IT procurement instead."); the original skill blocked cap breaches in 10 of 12 recorded runs (fidelity/reference.json and the 2026-10-24 A/B runs).
            found.append(
                (
                    "over_cap",
                    "block",
                    (
                        f"{_usd(amount)} exceeds the {_usd(cap)} {category} cap "
                        f"by {_usd(amount - cap)}"
                    ),
                )
            )

    if amount >= receipt_threshold and not expense["receipt"]:
        found.append(("receipt_missing", "block", f"no receipt on a {_usd(amount)} line"))
    if expense["id"] in duplicate_ids:
        found.append(
            ("duplicate", "block", "same employee, merchant, amount, and date as an earlier line")
        )
    if expense["id"] in late_ids:
        found.append(("late", "review", f"dated {expense['date']}, outside the late-submission window"))

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


def _caps(category_caps_usd: dict[str, float]) -> dict[str, Decimal]:
    """Per-line caps in USD keyed by category. The IR names lodging's cap
    ``lodging_per_night``, since a lodging line's cap scales with its nights."""
    return {
        ("lodging" if k == "lodging_per_night" else k): Decimal(str(v))
        for k, v in category_caps_usd.items()
    }


def audit_report(
    report: dict[str, Any],
    duplicate_ids: list[str],
    late_ids: list[str],
    *,
    receipt_threshold_usd: float,
    report_limit_usd: float,
    category_caps_usd: dict[str, float],
) -> dict[str, Any]:
    """Apply every policy rule to one employee's report.

    The keyword arguments are the audit node's IR constants, which the step
    passes on every call, so editing one in pipeline.yaml changes the rule.
    """
    caps = _caps(category_caps_usd)
    receipt_threshold = Decimal(str(receipt_threshold_usd))
    report_limit = Decimal(str(report_limit_usd))
    duplicates, late = set(duplicate_ids), set(late_ids)
    findings: list[dict[str, str]] = []
    for expense in report["expenses"]:
        findings.extend(_line_findings(expense, duplicates, late, caps, receipt_threshold))

    total = to_cents(sum((Decimal(str(e["amount_usd"])) for e in report["expenses"]), Decimal(0)))
    if total > report_limit:
        findings.append(
            {
                "expense_id": report_finding_id(report["employee"]),
                "employee": report["employee"],
                "rule": "report_limit",
                "severity": "review",
                "detail": (
                    f"report total {_usd(total)} exceeds the {_usd(report_limit)} "
                    f"limit by {_usd(total - report_limit)}"
                ),
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
