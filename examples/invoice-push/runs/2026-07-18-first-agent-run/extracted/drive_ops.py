"""
Google Drive / Sheets operations for the invoice-push report.

Creates or navigates to the YYYY/MM subfolder structure under the
archive folder, creates a Google Sheet with three populated tabs,
and returns the sheet URL.
"""

from dataclasses import dataclass
from typing import Any

# ── Constants ─────────────────────────────────────────────────────────────────

# Google Drive folder ID of the root "Invoice Push Reports Archive".
# (SKILL.md §Step 9 — replace with the real production folder ID.)
ARCHIVE_FOLDER_ID: str = "1ExampleInvoicePushArchiveFolderId0000000000"

REPORT_FILE_PREFIX: str = "Invoice_Push_Report"  # file name: {PREFIX}_{YYYY-MM-DD}

TAB_INVOICE_DETAIL: str = "Invoice Detail"
TAB_RUN_SUMMARY: str = "Run Summary"
TAB_FAILURE_CODES: str = "Failure Code Reference"


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class ReportLocation:
    sheet_id: str
    sheet_url: str
    folder_path: str  # human-readable path, e.g. "Archive/2026/05"


# ── Drive/Sheets stub ─────────────────────────────────────────────────────────

def save_report_to_drive(
    run_date: str,                               # YYYY-MM-DD — used in file name and folder path
    invoice_detail_rows: list[dict[str, Any]],  # Tab 1 rows from build_report
    run_summary_rows: list[dict[str, str]],     # Tab 2 metric/value rows from build_report
    failure_code_reference: list[dict[str, str]],  # Tab 3 static rows from build_report
) -> ReportLocation:
    """
    Ensure the YYYY → MM subfolder structure exists under ARCHIVE_FOLDER_ID,
    create a new Google Sheet named '{REPORT_FILE_PREFIX}_{run_date}' in the
    MM folder, populate three tabs, and return the sheet link.

    Folder structure (SKILL.md §Step 9):
        ARCHIVE_FOLDER_ID / YYYY / MM / Invoice_Push_Report_YYYY-MM-DD

    Create YYYY and MM subfolders if missing.

    Tab order and column layout must match SKILL.md §Step 9 exactly:
      Tab 1 — Invoice Detail: rows from invoice_detail_rows, 18 columns
      Tab 2 — Run Summary: rows from run_summary_rows (Metric / Value)
      Tab 3 — Failure Code Reference: rows from failure_code_reference
        (Code / Toast Message / Meaning / Operator Action)

    Wraps: Google Drive API (files.list, files.create) + Sheets API
    (spreadsheets.create, spreadsheets.values.update).
    """
    raise NotImplementedError(
        "Implement: (1) derive YYYY/MM from run_date; "
        "(2) look up or create YYYY and MM subfolders under ARCHIVE_FOLDER_ID; "
        "(3) create Google Sheet in MM folder with name "
        f"'{REPORT_FILE_PREFIX}_{{run_date}}'; "
        "(4) populate Tab 1 (Invoice Detail), Tab 2 (Run Summary), "
        "Tab 3 (Failure Code Reference); "
        "(5) return ReportLocation with sheet_id, sheet_url, folder_path."
    )
