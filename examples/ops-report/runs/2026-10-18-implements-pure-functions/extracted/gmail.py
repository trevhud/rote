"""
Gmail MCP thread search — direct-API stub (--backend api).

MCP binding: server=gmail, tool=search_threads
Vendor API: Gmail API v1
  GET https://gmail.googleapis.com/gmail/v1/users/me/threads?q={query}

Dependencies (not imported here — stub only): google-api-python-client, google-auth
"""


def search_gmail_threads(*, query: str) -> dict:
    """Search Gmail threads matching a query.

    Args:
        query: Gmail search query string (e.g. 'label:dock-activity newer_than:1d').

    Returns:
        {"threads": list, "resultSizeEstimate": int} — Gmail threads list response.
        Each thread contains at minimum: id (str), snippet (str), and optionally
        messages (list of message dicts with payload.headers).

    Raises:
        NotImplementedError: Contract-documented stub; MCP backend is the default path.
    """
    raise NotImplementedError(
        "Gmail API stub: authenticate with google-auth and call "
        "GET https://gmail.googleapis.com/gmail/v1/users/me/threads?q={query}. "
        "The default --backend mcp uses search_threads on the gmail server "
        "and does not reach this function."
    )
