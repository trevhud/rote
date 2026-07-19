"""Google Drive MCP bindings for ops-report.

Each function is an external_call stub: the ``mcp`` backend emits a working
Streamable-HTTP call to the google_drive MCP server; the ``api`` backend uses
the Drive REST API directly. Fill in the api backend body (or implement a
FastMCP client) before running without the MCP runtime.

Underlying API: Google Sheets export endpoint
  GET https://sheets.googleapis.com/v4/spreadsheets/{fileId}/values/{range}
or the Files.export for XLSX. The MCP tool read_file_content wraps this and
returns rows as a list of dicts (header row becomes keys).
"""
from __future__ import annotations

SHIPMENT_CONTAINERS_FILE_ID = "1ExampleShipmentContainersFileId00000000000"
DOCK_PENDING_LOG_FILE_ID = "1ExampleDockPendingLogFileId0000000000000000"
DWELL_TICKET_LOG_FILE_ID = "1ExampleDwellTicketLogFileId0000000000000000"
DWELL_TICKET_TAB = "Dwell Log - Form Responses"


def fetch_shipment_containers() -> list[dict]:
    """Fetch the Shipment Containers spreadsheet from Google Drive.

    MCP call: google_drive.read_file_content(file_id=SHIPMENT_CONTAINERS_FILE_ID)
    Returns a list of row dicts from the facility summary table.
    Expected columns: site, active, loaded, staged, shipped, completion_pct
                      (optionally prior_completion_pct for week-over-week trend)
    """
    raise NotImplementedError(
        "Implement via google_drive MCP read_file_content "
        f"with file_id={SHIPMENT_CONTAINERS_FILE_ID!r}"
    )


def fetch_dock_pending_log() -> list[dict]:
    """Fetch the Dock Pending Log spreadsheet from Google Drive.

    MCP call: google_drive.read_file_content(file_id=DOCK_PENDING_LOG_FILE_ID)
    Returns a list of row dicts from the dock pending log.
    Expected columns: Resolved, Site, Issue Type
    """
    raise NotImplementedError(
        "Implement via google_drive MCP read_file_content "
        f"with file_id={DOCK_PENDING_LOG_FILE_ID!r}"
    )


def fetch_dwell_tickets() -> list[dict]:
    """Fetch the Open Dwell Ticket Log from Google Drive (specific tab).

    MCP call: google_drive.read_file_content(
        file_id=DWELL_TICKET_LOG_FILE_ID, tab=DWELL_TICKET_TAB
    )
    Returns rows from the 'Dwell Log - Form Responses' tab.
    Expected columns: Open/Closed, Date, Facility, # packages,
                      Client Impact, Carrier Impact, Issue Description
    """
    raise NotImplementedError(
        "Implement via google_drive MCP read_file_content "
        f"with file_id={DWELL_TICKET_LOG_FILE_ID!r}, tab={DWELL_TICKET_TAB!r}"
    )
