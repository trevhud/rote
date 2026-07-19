"""Gmail data fetcher for the ops-report pipeline.

Wraps a single Gmail MCP search_threads call with the fixed dock-activity
label query. The query string is a constant — it never varies run-to-run.

Underlying vendor API: GET /gmail/v1/users/me/threads?q={query}
"""

DOCK_ACTIVITY_QUERY = "label:dock-activity newer_than:1d"


def fetch_dock_activity_emails() -> list:
    """Return Gmail thread list matching dock-activity label from the past 24 hours.

    MCP call: gmail / search_threads
    Query:    DOCK_ACTIVITY_QUERY
    """
    raise NotImplementedError(
        "Call the Gmail MCP search_threads tool with "
        f"query={DOCK_ACTIVITY_QUERY!r}. "
        "Return the list of thread objects from the response."
    )
