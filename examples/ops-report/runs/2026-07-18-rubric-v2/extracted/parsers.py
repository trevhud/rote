"""Deterministic parsers for ops-report data sources.

Each function takes raw content returned by an MCP tool call and returns a
structured dict. All thresholds that appear as numeric constants in the source
skill are defined here as module-level constants so they cannot drift through
prompt edits.
"""
from __future__ import annotations

# ── Thresholds lifted from SKILL.md ──────────────────────────────────────────

COMPLETION_WARNING_THRESHOLD = 0.75       # flag sites below 75% completion

DWELL_MISSORT_WARNING_PACKAGES = 25       # brand pkg count: light red
DWELL_MISSORT_ALERT_PACKAGES = 41         # brand pkg count: bright red


# ── Return-type shapes (documented as dicts for adapter compatibility) ────────
#
# ShipmentContainerData:
#   sites: list of {
#     site: str, active: int, loaded: int, staged: int, shipped: int,
#     completion_pct: float, below_threshold: bool, wow_trend: str | None
#   }
#   avg_completion_pct: float
#   flagged_sites: list[str]
#
# DockPendingData:
#   total_open: int
#   by_site: dict[str, int]       e.g. {"EAST1": 3, "WEST2": 1}
#   by_issue_type: dict[str, int] e.g. {"PO Number": 2, "Pallet/Piece Count Discrepancy": 2}
#
# DockActivityData:
#   approved: int
#   requested: int
#   outstanding_unapproved: int
#
# DwellTicketData:
#   open_tickets: list of {
#     date: str, facility: str, packages: int,
#     client_impact: str, carrier_impact: str, issue_description: str,
#     alert_level: str  # "missort_warning" | "missort_alert" | "normal"
#   }
#   total_open: int


def parse_shipment_containers(raw: str) -> dict:
    """Parse the facility summary table from Shipment Containers spreadsheet content.

    Expects tabular data with columns: Site, Active, Loaded, Staged, Shipped,
    Completion %. Flags any site where completion_pct < COMPLETION_WARNING_THRESHOLD.
    Records week-over-week trend string if a prior-week column is present.

    Returns: ShipmentContainerData dict (see shape above).
    """
    raise NotImplementedError(
        "Parse 'raw' (Google Drive export of the Shipment Containers spreadsheet) "
        "into a ShipmentContainerData dict. Extract the facility summary table. "
        "Columns: Site, Active, Loaded, Staged, Shipped, Completion %. "
        f"Flag sites where completion_pct < {COMPLETION_WARNING_THRESHOLD} "
        "(COMPLETION_WARNING_THRESHOLD). Include week-over-week trend if a "
        "prior-week column exists. Compute avg_completion_pct across all sites."
    )


def parse_dock_pending_log(raw: str) -> dict:
    """Parse the Dock Pending Log spreadsheet content.

    Counts all rows where the Resolved column = FALSE (or equivalent falsy
    value). Groups by Site (EAST1, WEST2, etc.) and Issue Type
    (PO Number; Pallet/Piece Count Discrepancy).

    Returns: DockPendingData dict (see shape above).
    """
    raise NotImplementedError(
        "Parse 'raw' (Google Drive export of the Dock Pending Log spreadsheet) "
        "into a DockPendingData dict. Filter rows where Resolved=FALSE. "
        "Group by Site column and by Issue Type column. Return total_open, "
        "by_site dict, and by_issue_type dict."
    )


def parse_dock_activity(threads: list) -> dict:
    """Parse Gmail dock-activity threads into appointment counts.

    Counts Approved vs. Requested appointments from thread subjects or
    body content. Any thread that records a request with no corresponding
    approval is flagged as outstanding_unapproved.

    Returns: DockActivityData dict (see shape above).
    """
    raise NotImplementedError(
        "Parse 'threads' (Gmail search_threads result for "
        "label:dock-activity newer_than:1d) into a DockActivityData dict. "
        "Count threads/messages classified as Approved vs. Requested. "
        "Compute outstanding_unapproved = requested - approved (floor 0). "
        "Classification may use thread subject lines or a status field if present."
    )


def parse_dwell_tickets(raw: str) -> dict:
    """Parse the Dwell Ticket Log (Dwell Log - Form Responses tab).

    Filters rows where the Open/Closed column = 'Open'. Applies alert thresholds
    to the # packages field:
      packages >= DWELL_MISSORT_WARNING_PACKAGES (25) → alert_level = 'missort_warning'
      packages >= DWELL_MISSORT_ALERT_PACKAGES   (41) → alert_level = 'missort_alert'
      otherwise                                       → alert_level = 'normal'

    Note: carrier >= 80 threshold applies to BI dashboard data supplied by the
    duty manager, not to this spreadsheet — it is handled in thresholds.py.

    Returns: DwellTicketData dict (see shape above).
    """
    raise NotImplementedError(
        "Parse 'raw' (Google Drive export of the Dwell Ticket Log, "
        "'Dwell Log - Form Responses' tab) into a DwellTicketData dict. "
        "Filter rows where Open/Closed='Open'. For each open ticket extract: "
        "date, facility, # packages, client_impact, carrier_impact, issue_description. "
        f"Apply thresholds: packages >= {DWELL_MISSORT_WARNING_PACKAGES} → "
        f"'missort_warning'; >= {DWELL_MISSORT_ALERT_PACKAGES} → 'missort_alert'."
    )
