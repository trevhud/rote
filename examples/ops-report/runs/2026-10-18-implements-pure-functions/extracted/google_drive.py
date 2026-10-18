"""
Google Drive MCP external_call stubs for ops-report.

With --backend mcp (default), the adapter emits a direct call to the
google_drive MCP server's read_file_content tool instead of these functions.
These implementations are the --backend api alternative.

Underlying vendor API: Google Drive Files API
  GET https://www.googleapis.com/drive/v3/files/{fileId}/export  (for Sheets)
  GET https://www.googleapis.com/drive/v3/files/{fileId}?alt=media  (for raw files)

Credentials: GOOGLE_APPLICATION_CREDENTIALS (service-account JSON path)
             or GOOGLE_OAUTH_TOKEN (bearer token)

Regime 1 — no probe-context, no web research tools.
"""

# ── per-file constants (lifted from skill prose) ──────────────────────────────

SHIPMENT_CONTAINERS_FILE_ID: str = "1ExampleShipmentContainersFileId00000000000"
DOCK_PENDING_LOG_FILE_ID: str = "1ExampleDockPendingLogFileId0000000000000000"
DWELL_TICKET_LOG_FILE_ID: str = "1ExampleDwellTicketLogFileId0000000000000000"
DWELL_TICKET_TAB: str = "Dwell Log - Form Responses"


def fetch_shipment_containers() -> dict:
    """
    Fetch the Shipment Containers spreadsheet from Google Drive.

    MCP tool: google_drive / read_file_content
    File ID:  SHIPMENT_CONTAINERS_FILE_ID

    Returns: {"raw_content": str}  — CSV/TSV text of the sheet.
    """
    raise NotImplementedError(
        "google_drive MCP tool: read_file_content for file "
        + SHIPMENT_CONTAINERS_FILE_ID
        + " — use --backend mcp or implement via Google Drive Files API"
    )


def fetch_dock_pending_log() -> dict:
    """
    Fetch the Dock Pending Log spreadsheet from Google Drive.

    MCP tool: google_drive / read_file_content
    File ID:  DOCK_PENDING_LOG_FILE_ID

    Returns: {"raw_content": str}  — CSV/TSV text of the sheet.
    """
    raise NotImplementedError(
        "google_drive MCP tool: read_file_content for file "
        + DOCK_PENDING_LOG_FILE_ID
        + " — use --backend mcp or implement via Google Drive Files API"
    )


def fetch_dwell_tickets() -> dict:
    """
    Fetch the Dwell Ticket Log spreadsheet from Google Drive.

    MCP tool: google_drive / read_file_content
    File ID:  DWELL_TICKET_LOG_FILE_ID
    Tab:      DWELL_TICKET_TAB

    Returns: {"raw_content": str}  — CSV/TSV text of the named tab.
    """
    raise NotImplementedError(
        "google_drive MCP tool: read_file_content for file "
        + DWELL_TICKET_LOG_FILE_ID
        + " tab '"
        + DWELL_TICKET_TAB
        + "' — use --backend mcp or implement via Google Drive Files API"
    )
