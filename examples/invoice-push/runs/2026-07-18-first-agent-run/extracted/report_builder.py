"""
Report assembly for the invoice-push pipeline.

Builds the 3-tab structure (Invoice Detail, Run Summary, Failure Code
Reference) from the structured results collected during the run.
No LLM, no browser — pure string formatting from typed inputs.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

# ── Enums & constants ─────────────────────────────────────────────────────────

class PushStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


# Static reference table for Tab 3 (SKILL.md §7f + §Step 9 Tab 3).
# The order matches the toast table in the skill.
FAILURE_CODE_REFERENCE: list[dict[str, str]] = [
    {
        "Code": "ERR-INVOICE-TYPE",
        "Toast Message": "Procurement is not enabled for this invoice type",
        "Meaning": "Invoice type is not configured for procurement",
        "Operator Action": "No action needed — this invoice type is excluded from procurement.",
    },
    {
        "Code": "ERR-DUPLICATE",
        "Toast Message": "Invoice already exists in procurement",
        "Meaning": "Invoice was previously pushed to procurement",
        "Operator Action": "No action needed — invoice is already in procurement.",
    },
    {
        "Code": "ERR-TIMEOUT",
        "Toast Message": "Procurement request timed out",
        "Meaning": "Procurement system did not respond in time",
        "Operator Action": "Check the procurement portal manually; retry if invoice is absent.",
    },
    {
        "Code": "ERR-AUTH",
        "Toast Message": "Authorization denied",
        "Meaning": "User session expired or permission revoked",
        "Operator Action": "Run stopped. Re-authenticate and restart the push.",
    },
    {
        "Code": "ERR-CONN",
        "Toast Message": "Cannot connect to procurement",
        "Meaning": "Procurement service is unreachable",
        "Operator Action": "Run stopped. Check connectivity; escalate to administrator if persistent.",
    },
    {
        "Code": "ERR-NO-CONFIRM",
        "Toast Message": "(no toast after 5 seconds)",
        "Meaning": "Push did not confirm success or failure within the timeout",
        "Operator Action": "Check the procurement portal manually; retry if invoice is absent.",
    },
]

# Column order for Tab 1 (SKILL.md §Step 9 Tab 1).
INVOICE_DETAIL_COLUMNS: list[str] = [
    "Batch ID",
    "Status",
    "Invoice Number",
    "Carrier",
    "Invoice Type",
    "Import File",
    "Date Imported",
    "Imported By",
    "Record Count",
    "Charges",
    "Packages",
    "Ship Packages",
    "Earliest Ship Date",
    "Latest Ship Date",
    "Sent Date",
    "Push Status",
    "Failure Code",
    "Failure Detail",
]

# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class InvoiceResult:
    """One processed invoice row — source for a single Tab 1 detail row."""

    batch_id: str
    status: str
    invoice_number: str
    carrier: str
    invoice_type: str
    import_file: str
    date_imported: str       # raw string as read from screen
    imported_by: str
    record_count: str
    charges: str
    packages: str
    ship_packages: str
    earliest_ship_date: str
    latest_ship_date: str
    sent_date: str = "-"
    push_status: PushStatus = PushStatus.SKIPPED
    failure_code: str = ""
    failure_detail: str = ""


@dataclass
class RunReport:
    """Three-tab report structure ready for Google Sheets population."""

    invoice_detail_rows: list[dict[str, Any]]    # Tab 1 rows, columns ordered
    run_summary_rows: list[dict[str, str]]        # Tab 2 metric/value pairs
    failure_code_reference: list[dict[str, str]]  # Tab 3 static table


# ── Pure builder function ─────────────────────────────────────────────────────

def build_report(
    results: list[InvoiceResult],
    run_start_time: datetime,
    date_window_start: str,
    date_window_end: str,
    total_qualifying: int,
    stopped_early: bool,
    stop_reason: str = "",
) -> RunReport:
    """
    Build the three-tab report from the structured run results.

    All formatting is deterministic — no LLM calls.
    """
    # ── Tab 1: Invoice Detail ──────────────────────────────────────────────
    invoice_detail_rows: list[dict[str, Any]] = []
    for r in results:
        if r.push_status == PushStatus.SUCCESS:
            status_display = "✅ Success"
        elif r.push_status == PushStatus.FAILED:
            status_display = "❌ Failed"
        else:
            status_display = "⚠️ Skipped"

        invoice_detail_rows.append({
            "Batch ID": r.batch_id,
            "Status": r.status,
            "Invoice Number": r.invoice_number,
            "Carrier": r.carrier,
            "Invoice Type": r.invoice_type,
            "Import File": r.import_file,
            "Date Imported": r.date_imported,
            "Imported By": r.imported_by,
            "Record Count": r.record_count,
            "Charges": r.charges,
            "Packages": r.packages,
            "Ship Packages": r.ship_packages,
            "Earliest Ship Date": r.earliest_ship_date,
            "Latest Ship Date": r.latest_ship_date,
            "Sent Date": r.sent_date,
            "Push Status": status_display,
            "Failure Code": r.failure_code,
            "Failure Detail": r.failure_detail,
        })

    # ── Tab 2: Run Summary ─────────────────────────────────────────────────
    success_count = sum(1 for r in results if r.push_status == PushStatus.SUCCESS)
    failed_count = sum(1 for r in results if r.push_status == PushStatus.FAILED)
    skipped_count = sum(1 for r in results if r.push_status == PushStatus.SKIPPED)

    error_breakdown: dict[str, int] = {}
    for r in results:
        if r.failure_code:
            error_breakdown[r.failure_code] = error_breakdown.get(r.failure_code, 0) + 1

    failed_str = str(failed_count)
    if error_breakdown:
        breakdown = ", ".join(f"{k}: {v}" for k, v in sorted(error_breakdown.items()))
        failed_str = f"{failed_count} ({breakdown})"

    ts_display = run_start_time.strftime("%B %-d, %Y %-I:%M %p %Z").strip()

    run_summary_rows: list[dict[str, str]] = [
        {"Metric": "Run start time",             "Value": ts_display},
        {"Metric": "Date window",                "Value": f"{date_window_start} - {date_window_end}"},
        {"Metric": "Total qualifying invoices",  "Value": str(total_qualifying)},
        {"Metric": "Pushed successfully",        "Value": str(success_count)},
        {"Metric": "Failed",                     "Value": failed_str},
        {"Metric": "Skipped",                    "Value": str(skipped_count)},
        {"Metric": "Run stopped early",          "Value": f"Yes — {stop_reason}" if stopped_early else "No"},
    ]

    return RunReport(
        invoice_detail_rows=invoice_detail_rows,
        run_summary_rows=run_summary_rows,
        failure_code_reference=FAILURE_CODE_REFERENCE,
    )
