"""
Browser automation stubs for the invoice-push pipeline.

Each function wraps one deterministic interaction with the Cloud Imports
screen via the browser-automation MCP tool. Implement each stub by calling
the appropriate MCP browser action (navigate, click, read, wait, refresh).

The underlying platform URL is: https://ops.example.com/app/audit/import
"""

from dataclasses import dataclass
from datetime import datetime

# ── Constants (from SKILL.md) ────────────────────────────────────────────────

IMPORTS_URL: str = "https://ops.example.com/app/audit/import"
PAGE_SIZE: int = 90  # SKILL.md §Step 4
TOAST_TIMEOUT_SECONDS: int = 5  # SKILL.md §7f

# Fixed mapping from toast text to error code (SKILL.md §7f).
# A green toast with any message → success (not listed here; detect by color).
TOAST_TO_ERROR_CODE: dict[str, str] = {
    "Procurement is not enabled for this invoice type": "ERR-INVOICE-TYPE",
    "Invoice already exists in procurement": "ERR-DUPLICATE",
    "Procurement request timed out": "ERR-TIMEOUT",
    "Authorization denied": "ERR-AUTH",
    "Cannot connect to procurement": "ERR-CONN",
    # No toast after TOAST_TIMEOUT_SECONDS → "ERR-NO-CONFIRM" (handled in push_invoice)
}

# These errors require stopping the run immediately and alerting the user.
FATAL_ERROR_CODES: frozenset[str] = frozenset({"ERR-AUTH", "ERR-CONN"})


# ── Exceptions ────────────────────────────────────────────────────────────────

class FatalPushError(Exception):
    """
    Raised by push_invoice when ERR-AUTH or ERR-CONN is received.
    The durable executor surfaces this as a workflow failure; the run
    does not continue. The batch_id identifies the last attempted row.
    """

    def __init__(self, error_code: str, batch_id: str) -> None:
        self.error_code = error_code
        self.batch_id = batch_id
        super().__init__(
            f"{error_code} on batch {batch_id!r} — run stopped; alert operator"
        )


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class InvoiceRow:
    """All 15 readable columns for one row on the Imports screen (SKILL.md §7a)."""

    batch_id: str
    status: str
    sent_date: str              # "-" if not yet pushed
    invoice_number: str
    carrier: str
    invoice_type: str
    import_file: str
    date_imported: str          # raw string exactly as displayed on screen
    date_imported_dt: datetime  # parsed datetime for the 24-hour eligibility check
    imported_by: str
    record_count: str
    charges: str
    packages: str
    ship_packages: str
    earliest_ship_date: str
    latest_ship_date: str


@dataclass
class PushResult:
    batch_id: str
    is_success: bool
    error_code: str | None   # None on success; one of TOAST_TO_ERROR_CODE keys or ERR-NO-CONFIRM
    toast_message: str | None


# ── Browser action stubs ──────────────────────────────────────────────────────

def check_prerequisites() -> None:
    """
    MANDATORY: Verify the browser-automation plugin is active and the user
    is logged in to the Cloud Imports screen.

    Raises RuntimeError with an operator-facing message if either check fails.
    The workflow will not proceed past this node.

    Wraps: plugin status query (Settings → Plugins) + active tab URL check.
    """
    raise NotImplementedError(
        "Implement: (1) query plugin status — raise RuntimeError if not enabled; "
        "(2) check active tab URL matches IMPORTS_URL — raise RuntimeError if not logged in."
    )


def navigate_to_imports() -> None:
    """
    Navigate the browser to IMPORTS_URL and wait for the Imports table to load.

    Wait condition: tab title == 'Imports' and at least one table row is visible.

    Wraps: browser.navigate(IMPORTS_URL) + wait for table readiness.
    """
    raise NotImplementedError(
        f"Implement: browser.navigate({IMPORTS_URL!r}), wait for table to load."
    )


def clear_filters() -> None:
    """
    MANDATORY: Remove all active filter pills before applying the run's filters.

    Stale filters silently exclude qualifying invoices — this node must run
    even if no pills appear to be active.

    Wraps: detect filter pills → click × on each; or Filters → Clear all →
    close panel. Confirm table resets to unfiltered state.
    """
    raise NotImplementedError(
        "Implement: check for active filter pills; click × on each, or "
        "Filters → Clear all → close panel. Confirm unfiltered state."
    )


def apply_filters(start_date: str, end_date: str) -> None:
    """
    Apply the qualifying filters:
      - Date Imported: start_date to end_date (M/D/YYYY format)
      - Sent To Procurement: No

    Confirms filter pills show:
      'Date Imported: [start] - [end] ×'
      'Sent To Procurement: No ×'

    Wraps: Filters panel → set Date Imported range → set Sent To Procurement = No
    → close panel → verify pills.
    """
    raise NotImplementedError(
        f"Implement: open Filters panel, set Date Imported to "
        f"{start_date!r} – {end_date!r}, set Sent To Procurement = No, "
        "close panel, verify pills."
    )


def set_page_size() -> None:
    """
    Set the page size to PAGE_SIZE (90) via the bottom-left dropdown.

    Waits for the table to reload after changing the selection.

    Wraps: click '25 per page' dropdown → select 90 → wait for table reload.
    """
    raise NotImplementedError(
        f"Implement: click page-size dropdown, select {PAGE_SIZE}, wait for table reload."
    )


def read_total_count() -> int:
    """
    Read the total qualifying invoice count from the pagination label.

    Parses a label like '1-22 of 22 imports' and returns 22.

    Wraps: browser.read(pagination label) → regex match 'of (\\d+) imports'.
    """
    raise NotImplementedError(
        "Implement: read pagination label text, parse integer after 'of', return it."
    )


def read_invoice_row(row_index: int) -> InvoiceRow:
    """
    Read all 15 column values for the row at row_index on the current page.

    Do not infer or assume any column value — read exactly as displayed.
    Parse date_imported into date_imported_dt using the platform's timestamp
    format (include timezone from the 'Date Imported' cell).

    Wraps: browser.read_table_row(row_index) → map cells to InvoiceRow fields.
    """
    raise NotImplementedError(
        f"Implement: read table row {row_index}, parse all columns into InvoiceRow."
    )


def push_invoice(batch_id: str) -> PushResult:
    """
    Open the ⋮ menu for the row identified by batch_id, click 'Push to
    Procurement', and wait up to TOAST_TIMEOUT_SECONDS for a toast notification.

    Row identification MUST use batch_id, not row position (positions shift).

    Toast → result mapping (SKILL.md §7f):
      Green toast (any text)   → success
      Red toast (known text)   → ERR-INVOICE-TYPE / ERR-DUPLICATE / ERR-TIMEOUT
      "Authorization denied"   → ERR-AUTH  → raises FatalPushError
      "Cannot connect..."      → ERR-CONN  → raises FatalPushError
      No toast after 5s        → ERR-NO-CONFIRM

    Raises FatalPushError for ERR-AUTH or ERR-CONN — the run must stop.

    Wraps: find row by batch_id, click ⋮ (3rd column), click Push to Procurement,
    wait for toast, classify toast using TOAST_TO_ERROR_CODE.
    """
    raise NotImplementedError(
        f"Implement: locate row by batch_id={batch_id!r}, click ⋮ menu, "
        f"click Push to Procurement, wait {TOAST_TIMEOUT_SECONDS}s for toast, "
        "classify toast using TOAST_TO_ERROR_CODE; raise FatalPushError for fatal codes."
    )


def capture_sent_date(batch_id: str) -> str:
    """
    Refresh the table and re-locate the row by batch_id to read the Sent Date.

    MANDATORY: always re-identify the row by Batch ID after refresh —
    row order may shift after the push action.

    Returns the Sent Date string exactly as displayed (e.g. '5/4/2026 9:15 AM').

    Wraps: click ↻ refresh icon (top right of table toolbar) → wait for reload
    → find row by batch_id → read Sent Date cell.
    """
    raise NotImplementedError(
        f"Implement: click refresh, wait for reload, find row by batch_id={batch_id!r}, "
        "read and return the Sent Date cell value."
    )
