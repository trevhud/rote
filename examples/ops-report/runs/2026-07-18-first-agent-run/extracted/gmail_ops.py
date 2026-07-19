"""Gmail MCP binding for ops-report dock-activity email fetch.

External-call stub: the ``mcp`` backend emits a Streamable-HTTP call to the
gmail MCP server; the ``api`` backend uses the Gmail API directly.

Underlying API: GET https://gmail.googleapis.com/gmail/v1/users/me/threads
  with q=GMAIL_DOCK_ACTIVITY_QUERY
"""
from __future__ import annotations

GMAIL_DOCK_ACTIVITY_QUERY = "label:dock-activity newer_than:1d"


def fetch_dock_activity_emails() -> list[dict]:
    """Search Gmail for dock-activity threads from the past 24 hours.

    MCP call: gmail.search_threads(query=GMAIL_DOCK_ACTIVITY_QUERY)
    Returns a list of Gmail thread summary dicts. Each dict should contain at
    least 'snippet' and/or 'subject' for downstream classification. Adjust
    the parse_dock_emails classifier if your team uses different subject-line
    conventions for approved/requested appointments.
    """
    raise NotImplementedError(
        "Implement via gmail MCP search_threads "
        f"with query={GMAIL_DOCK_ACTIVITY_QUERY!r}"
    )
