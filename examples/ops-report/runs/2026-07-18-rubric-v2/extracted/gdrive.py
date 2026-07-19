"""Google Drive file fetchers for the ops-report pipeline.

Each function wraps a single call to the Google Drive MCP read_file_content
tool. File IDs are fixed constants extracted from the source skill's prose.

Underlying vendor API: GET /drive/v3/files/{fileId}/export (Google Sheets
export as text/csv or text/html, depending on MCP implementation).
"""

SHIPMENT_CONTAINERS_FILE_ID = "1ExampleShipmentContainersFileId00000000000"
DOCK_PENDING_LOG_FILE_ID = "1ExampleDockPendingLogFileId0000000000000000"
DWELL_TICKET_LOG_FILE_ID = "1ExampleDwellTicketLogFileId0000000000000000"
DWELL_TICKET_TAB = "Dwell Log - Form Responses"


def fetch_shipment_containers() -> str:
    """Return raw spreadsheet content for the Shipment Containers file.

    MCP call: google_drive / read_file_content
    File ID: SHIPMENT_CONTAINERS_FILE_ID
    """
    raise NotImplementedError(
        "Call the Google Drive MCP read_file_content tool with "
        f"file_id={SHIPMENT_CONTAINERS_FILE_ID!r}. "
        "Return the raw text content of the spreadsheet."
    )


def fetch_dock_pending_log() -> str:
    """Return raw spreadsheet content for the Dock Pending Log file.

    MCP call: google_drive / read_file_content
    File ID: DOCK_PENDING_LOG_FILE_ID
    """
    raise NotImplementedError(
        "Call the Google Drive MCP read_file_content tool with "
        f"file_id={DOCK_PENDING_LOG_FILE_ID!r}. "
        "Return the raw text content of the spreadsheet."
    )


def fetch_dwell_tickets() -> str:
    """Return raw spreadsheet content for the Dwell Ticket Log (Dwell Log - Form Responses tab).

    MCP call: google_drive / read_file_content
    File ID: DWELL_TICKET_LOG_FILE_ID
    Tab:     DWELL_TICKET_TAB
    """
    raise NotImplementedError(
        "Call the Google Drive MCP read_file_content tool with "
        f"file_id={DWELL_TICKET_LOG_FILE_ID!r} and "
        f"tab={DWELL_TICKET_TAB!r}. "
        "Return the raw text content of the specified tab."
    )
