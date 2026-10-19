"""
Extracted stubs for external MCP data-pull calls.

No probe-context directory exists for this skill, so these are Regime-1
contract-documented stubs. The mcp: bindings on the pipeline nodes are the
load-bearing wiring; these stubs are the --backend api alternative.

Dependencies: none (stdlib only).
"""


def fetch_drive_file(*, file_id: str, tab: str | None = None) -> str:
    """
    Fetch a file's content from Google Drive.

    Vendor: Google Drive (via MCP)
    MCP tool: read_file_content
    Arguments:
      file_id (str): the Drive file ID (e.g. '1ExampleShipmentContainersFileId00000000000')
      tab (str | None): optional spreadsheet tab/sheet name to return
    Returns: file content as a UTF-8 string (CSV or TSV for spreadsheets).

    Auth: GOOGLE_APPLICATION_CREDENTIALS or OAuth token — see
    https://developers.google.com/drive/api/guides/manage-downloads

    Stub: no probe-context available; use --backend mcp to call
    google_drive.read_file_content directly.
    """
    raise NotImplementedError(
        "google_drive.read_file_content MCP tool — implement via the Drive API "
        "(files.export for Sheets → text/csv) or run with --backend mcp. "
        "file_id and tab are the required parameters."
    )


def fetch_gmail_threads(*, search_query: str) -> dict:
    """
    Search Gmail threads matching a query.

    Vendor: Gmail (via MCP)
    MCP tool: search_threads
    Arguments:
      search_query (str): Gmail search syntax, e.g. 'label:dock-activity newer_than:1d'
    Returns: dict with at minimum a 'threads' key containing a list of thread
             objects, each with 'id', 'snippet', and 'subject' (or equivalent).

    Auth: GOOGLE_APPLICATION_CREDENTIALS or OAuth token — see
    https://developers.google.com/gmail/api/reference/rest/v1/users.threads/list

    Stub: no probe-context available; use --backend mcp to call
    gmail.search_threads directly.
    """
    raise NotImplementedError(
        "gmail.search_threads MCP tool — implement via the Gmail API "
        "(users.threads.list with q=search_query) or run with --backend mcp."
    )
