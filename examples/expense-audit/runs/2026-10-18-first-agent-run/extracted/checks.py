"""
Mandatory pre-audit checks for the expense audit pipeline.

Both functions in this module are MANDATORY: they run on every expense
batch every month, regardless of batch size. Moving these from prose to
code makes skipping them structurally impossible.

Dependencies: none (stdlib only — datetime for period parsing).
"""

from __future__ import annotations

import datetime
from typing import TypedDict

from extracted.fx import ExpenseLine

# ---------------------------------------------------------------------------
# Constants (sourced from references/expense-policy.md and SKILL.md)
# ---------------------------------------------------------------------------

# Policy window for late submissions (days before last day of period).
LATE_SUBMISSION_DAYS: int = 60


class Finding(TypedDict):
    expense_id: str
    employee: str
    rule: str        # short rule name, e.g. "duplicate", "late_submission"
    severity: str    # "block" | "review"
    detail: str      # one-line human-readable explanation


def check_duplicates(
    lines: list[ExpenseLine],
) -> dict[str, list[Finding]]:
    """
    Identify duplicate expense submissions.

    Two lines are duplicates when they share the same (employee,
    merchant-ignoring-case, amount, currency, date). The first submission
    by expense_id (lowest ID) is kept; every later copy is flagged as a
    BLOCK finding.

    This check is MANDATORY (per SKILL.md Step 2: "Run it on every line,
    every month, even when the batch looks small enough to eyeball.").
    Duplicates are a BLOCK: they go back to the employee; the controller
    cannot approve them.

    Returns:
        {"duplicate_findings": list[Finding]}
        Each finding has severity="block" and rule="duplicate".
    """
    raise NotImplementedError(
        "implement: group lines by (employee, merchant.lower(), amount, "
        "currency, date); within each group sort by expense_id and flag "
        "all but the lowest as block findings."
    )


def check_late_submissions(
    lines: list[ExpenseLine],
    period: str,
) -> dict[str, list[Finding]]:
    """
    Flag expense lines outside the policy submission window.

    A line is LATE when:
    - Its date is more than LATE_SUBMISSION_DAYS (60) days before the last
      day of the reporting period, OR
    - Its date is after the last day of the reporting period (period closed).

    Late lines are REVIEW severity — the controller can approve them.

    This check is MANDATORY (per SKILL.md Step 3).

    Args:
        lines: raw expense lines (pre-FX conversion — only date is needed).
        period: YYYY-MM string, e.g. "2026-09".

    Returns:
        {"late_findings": list[Finding]}
        Each finding has severity="review" and rule="late_submission".

    Implementation notes:
    - "Last day of the reporting period" is the last calendar day of the
      month in ``period`` (e.g. 2026-09-30 for period="2026-09").
    - The 60-day lookback is from that last day, inclusive.
    """
    raise NotImplementedError(
        "implement: parse period into last_day = date(year, month, last_dom); "
        "cutoff = last_day - timedelta(days=LATE_SUBMISSION_DAYS); "
        "for each line flag when line_date < cutoff or line_date > last_day."
    )
