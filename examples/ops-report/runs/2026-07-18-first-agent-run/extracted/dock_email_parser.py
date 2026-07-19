"""Parse Gmail dock-activity thread summaries into appointment counts.

Pure function. Extracted from SKILL.md data-source section:
'Count Approved vs. Requested appointments. Flag any outstanding requests
not yet approved.'
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DockEmailSummary:
    approved_count: int
    requested_count: int
    outstanding_requests: list[str] = field(default_factory=list)


def parse_dock_activity_emails(threads: list[dict]) -> DockEmailSummary:
    """Classify Gmail threads as Approved or Requested dock appointments.

    Classification is keyword-based on each thread's subject and/or snippet:
      'approved'            → Approved
      'request*', 'pending' (without 'approved') → Requested/outstanding

    Adjust the keywords below to match your team's email conventions if they
    use different subject-line patterns.

    Args:
        threads: list of Gmail thread summary dicts, each with 'snippet'
                 and/or 'subject' fields from gmail.search_threads.
    """
    approved: list[dict] = []
    requested: list[dict] = []

    for thread in threads:
        text = (
            str(thread.get("snippet") or "")
            + " "
            + str(thread.get("subject") or "")
        ).lower()

        if "approved" in text:
            approved.append(thread)
        elif any(k in text for k in ("requested", "request", "pending")):
            requested.append(thread)

    outstanding = [
        str(t.get("subject") or t.get("snippet") or "")[:80]
        for t in requested
    ]

    return DockEmailSummary(
        approved_count=len(approved),
        requested_count=len(requested),
        outstanding_requests=outstanding,
    )
