"""Steps 2 and 3: the two MANDATORY batch-wide checks.

Both run over the raw export, independent of currency conversion, which
is why they sit in the same parallel wave as ``convert_to_usd``.
"""

from __future__ import annotations

import calendar
from datetime import date, timedelta
from decimal import Decimal
from typing import Any

#: references/expense-policy.md, "Duplicates and late submissions".
LATE_WINDOW_DAYS = 60


def expense_sort_key(expense_id: str) -> tuple[int, str]:
    """Order ids like EXP-2412 numerically, so "first submitted" is stable."""
    prefix, _, number = expense_id.rpartition("-")
    return (int(number), prefix) if number.isdigit() else (0, expense_id)


def find_duplicates(submissions: list[dict[str, Any]]) -> dict[str, Any]:
    """Ids of every copy after the first of an identical line.

    Identical means same employee, merchant (ignoring case), amount,
    currency, and date. The first one submitted (lowest id) is kept.
    """
    seen: set[tuple[str, str, Decimal, str, str]] = set()
    duplicates: list[str] = []
    for line in sorted(submissions, key=lambda s: expense_sort_key(s["id"])):
        key = (
            line["employee"],
            line["merchant"].strip().lower(),
            Decimal(str(line["amount"])),
            line["currency"].upper(),
            line["date"],
        )
        if key in seen:
            duplicates.append(line["id"])
        else:
            seen.add(key)
    return {"duplicate_ids": duplicates}


def period_bounds(period: str) -> tuple[date, date]:
    """First and last day of a ``YYYY-MM`` period."""
    year_str, _, month_str = period.partition("-")
    year, month = int(year_str), int(month_str)
    last_day = calendar.monthrange(year, month)[1]
    return date(year, month, 1), date(year, month, last_day)


def find_late_submissions(submissions: list[dict[str, Any]], period: str) -> dict[str, Any]:
    """Ids dated more than 60 days before period close, or after it."""
    _, period_end = period_bounds(period)
    earliest = period_end - timedelta(days=LATE_WINDOW_DAYS)
    late = [
        line["id"]
        for line in submissions
        if not earliest <= date.fromisoformat(line["date"]) <= period_end
    ]
    return {"late_ids": sorted(late, key=expense_sort_key)}
