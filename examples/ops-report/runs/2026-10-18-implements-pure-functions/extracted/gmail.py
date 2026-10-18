"""
Gmail MCP external_call stub for ops-report.

With --backend mcp (default), the adapter emits a direct call to the
gmail MCP server's search_threads tool instead of this function.
This implementation is the --backend api alternative.

Underlying vendor API: Gmail API
  GET https://gmail.googleapis.com/gmail/v1/users/me/threads?q={query}

Credentials: GOOGLE_APPLICATION_CREDENTIALS (service-account JSON path)
             or GOOGLE_OAUTH_TOKEN (bearer token)

Regime 1 — no probe-context, no web research tools.
"""

# ── constants lifted from skill prose ────────────────────────────────────────

DOCK_ACTIVITY_QUERY: str = "label:dock-activity newer_than:1d"


def fetch_dock_emails() -> dict:
    """
    Search Gmail for dock-activity threads from the last 24 hours.

    MCP tool: gmail / search_threads
    Query:    DOCK_ACTIVITY_QUERY  ("label:dock-activity newer_than:1d")

    Returns: {"threads": list}  — list of thread objects from search_threads.
    """
    raise NotImplementedError(
        "gmail MCP tool: search_threads with query '"
        + DOCK_ACTIVITY_QUERY
        + "' — use --backend mcp or implement via Gmail API"
    )
