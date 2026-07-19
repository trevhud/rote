"""
MANDATORY row eligibility checks for the invoice-push pipeline.

These rules must execute before any ⋮ menu interaction — they are
enforced in code so they cannot be skipped by prompt drift or model
changes.

Rules (from SKILL.md §7b and §7c):
  1. Status must be exactly "IMPORTED". Never click ⋮ on FAILED or
     IN PROGRESS rows.
  2. Date Imported must be at least RECENT_HOURS before run_time
     (invoices imported within 24 hours of the run are not eligible).
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

ELIGIBLE_STATUS: str = "IMPORTED"
RECENT_HOURS: int = 24  # hours — the 24-hour rule


class EligibilityOutcome(str, Enum):
    ELIGIBLE = "eligible"
    SKIP_STATUS = "skip_status"   # status is not IMPORTED
    SKIP_RECENT = "skip_recent"   # imported within 24 hours of run


@dataclass
class EligibilityResult:
    eligible: bool
    outcome: EligibilityOutcome
    skip_reason: str | None = None  # goes in Failure Detail column on skip


def check_row_eligibility(
    status: str,
    date_imported_dt: datetime,
    run_time: datetime,
) -> EligibilityResult:
    """
    MANDATORY: Determine whether an invoice row is eligible for push.

    Returns EligibilityResult.eligible=True only when both rules pass.
    Callers MUST NOT call push_invoice when eligible is False.
    """
    if status != ELIGIBLE_STATUS:
        return EligibilityResult(
            eligible=False,
            outcome=EligibilityOutcome.SKIP_STATUS,
            skip_reason=f"Status is {status} — not eligible for push.",
        )

    age = run_time - date_imported_dt
    if age < timedelta(hours=RECENT_HOURS):
        return EligibilityResult(
            eligible=False,
            outcome=EligibilityOutcome.SKIP_RECENT,
            skip_reason="Imported within 24 hours of run — not eligible.",
        )

    return EligibilityResult(eligible=True, outcome=EligibilityOutcome.ELIGIBLE)
