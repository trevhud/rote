"""The Dock Activity Gmail query, derived from the report date.

The source skill searches ``label:dock-activity newer_than:1d``. Gmail
resolves ``newer_than`` against the moment of the search, so a rerun of
the same report date (a retry, a backfill, an afternoon rerun) saw a
different set of threads. Gmail's ``after:`` / ``before:`` take calendar
dates, so a window built from ``report_date`` returns the same threads
whenever it runs.
"""

from __future__ import annotations

import datetime as dt


def build_dock_email_query(*, label: str, window_days: int, report_date: str | None = None) -> str:
    """``label:<label> after:<start> before:<report_date>``.

    The window is the ``window_days`` whole days before ``report_date``:
    for the 8 AM daily run that is the prior day, which is what
    ``newer_than:1d`` mostly covered. ``report_date`` is an optional
    pipeline input, so an omitted one is left out of the call and the
    report is for today, the only clock read here. ``label`` and
    ``window_days`` are this node's constants and every call sends them,
    so neither has a default.
    """
    if window_days < 1:
        raise ValueError(f"build_dock_email_query: window_days must be >= 1, got {window_days}")
    if not label.strip():
        raise ValueError("build_dock_email_query: label is empty")
    day = dt.date.today() if report_date is None else dt.date.fromisoformat(report_date)
    start = day - dt.timedelta(days=window_days)
    return f"label:{label} after:{start:%Y/%m/%d} before:{day:%Y/%m/%d}"
