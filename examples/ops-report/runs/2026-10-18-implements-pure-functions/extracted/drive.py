"""
Google Drive MCP file fetch — direct-API stub (--backend api).

MCP binding: server=google_drive, tool=read_file_content
Vendor API: Google Drive API v3
  GET https://www.googleapis.com/drive/v3/files/{fileId}/export?mimeType=text/csv

The default MCP backend calls read_file_content on the google_drive server and
needs no impl. This stub is the --backend api fallback.

Dependencies (not imported here — stub only): google-api-python-client, google-auth
"""


def fetch_drive_file(*, file_id: str) -> dict:
    """Fetch text content of a Google Drive file.

    Args:
        file_id: Google Drive file ID (declared as a node constant in the IR).

    Returns:
        {"content": str} — the file exported as CSV/TSV text.

    Raises:
        NotImplementedError: Contract-documented stub; MCP backend is the default path.
    """
    raise NotImplementedError(
        "Google Drive API stub: authenticate with google-auth and call "
        "GET https://www.googleapis.com/drive/v3/files/{fileId}/export"
        "?mimeType=text/csv. "
        "The default --backend mcp uses read_file_content on the google_drive "
        "server and does not reach this function."
    )
