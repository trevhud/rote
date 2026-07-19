"""
Date window calculation for the invoice-push pipeline.

The window is always: yesterday − WINDOW_DAYS through yesterday, formatted
M/D/YYYY as expected by the Cloud Imports date filter.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

WINDOW_DAYS: int = 7


def _format_date_mdy(d: datetime) -> str:
    """Return M/D/YYYY with no leading zeros, as expected by the platform filter."""
    return f"{d.month}/{d.day}/{d.year}"


@dataclass
class DateWindow:
    start_date: str  # M/D/YYYY — beginning of the 7-day window
    end_date: str    # M/D/YYYY — yesterday (invoices imported today are never pushed)


def calculate_date_window(override_date: str | None = None) -> DateWindow:
    """
    Calculate the 7-day rolling import window ending yesterday.

    end_date  = today − 1 day
    start_date = end_date − WINDOW_DAYS days

    override_date (ISO-8601 YYYY-MM-DD) lets callers fix 'today' for
    testing without touching wall-clock time.
    """
    if override_date is not None:
        today = datetime.fromisoformat(override_date).replace(tzinfo=timezone.utc)
    else:
        today = datetime.now(tz=timezone.utc)

    end = today - timedelta(days=1)
    start = end - timedelta(days=WINDOW_DAYS)

    return DateWindow(
        start_date=_format_date_mdy(start),
        end_date=_format_date_mdy(end),
    )


@dataclass
class RunStart:
    timestamp: datetime  # UTC timestamp for the Run Summary display
    iso_date: str        # YYYY-MM-DD for the Google Sheet file name


def record_run_start() -> RunStart:
    """
    Capture the current timestamp immediately before row processing begins.
    This is Step 5 of the skill — recorded after navigation and filter setup,
    so it reflects actual processing start time, not workflow enqueue time.
    """
    now = datetime.now(tz=timezone.utc)
    return RunStart(
        timestamp=now,
        iso_date=now.strftime("%Y-%m-%d"),
    )
