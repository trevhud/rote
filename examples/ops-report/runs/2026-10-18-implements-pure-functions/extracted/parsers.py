"""
Deterministic parsers for ops-report data sources.

All functions accept CSV/TSV text returned by Google Drive MCP read_file_content
and return structured dicts. Columns are located case- and whitespace-insensitively.
Missing required columns raise ValueError naming the absent column.

No third-party imports — stdlib only.
"""

import csv
import io
from typing import Any

# ── constants lifted from skill prose ────────────────────────────────────────

COMPLETION_THRESHOLD_PCT: float = 75.0   # flag sites below this
MISSORT_WARNING_PACKAGES: int = 25       # Brand 25-40 → Missort Warning
MISSORT_ALERT_PACKAGES: int = 41         # Brand 41+  → Missort Alert
MISLOAD_ALERT_CARRIER: int = 80          # Carrier 80+ missing → Possible Misload Alert


# ── helpers ───────────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    """Lowercase + strip for case-insensitive column matching."""
    return s.strip().lower()


def _find_col(headers: list[str], *candidates: str) -> int:
    """Return index of first matching column name; raise ValueError if none found."""
    norm_headers = [_norm(h) for h in headers]
    for cand in candidates:
        cand_norm = _norm(cand)
        if cand_norm in norm_headers:
            return norm_headers.index(cand_norm)
    raise ValueError(
        "Required column not found. Tried: "
        + str(list(candidates))
        + ". Found columns: "
        + str(headers)
    )


def _parse_csv(raw: str) -> tuple[list[str], list[list[str]]]:
    """Parse CSV or TSV text; return (headers, data_rows)."""
    dialect = "excel-tab" if raw.count("\t") > raw.count(",") else "excel"
    reader = csv.reader(io.StringIO(raw.strip()), dialect=dialect)
    rows = list(reader)
    if not rows:
        return [], []
    return rows[0], rows[1:]


# ── parse_shipment_containers ─────────────────────────────────────────────────

def parse_shipment_containers(raw_content: str) -> dict[str, Any]:
    """
    Parse the Shipment Containers Google Drive spreadsheet.

    Assumed input shape: CSV/TSV with header row containing columns for
    site/facility, active, loaded, staged, shipped, completion % (and
    optionally a prior-week completion % for trend).

    Returns:
        {
            "sites": [{"site": str, "active": int, "loaded": int,
                       "staged": int, "shipped": int,
                       "completion_pct": float,
                       "wow_trend_pct": float | None,
                       "flagged": bool}, ...],
            "flagged_sites": [str, ...],
            "avg_completion_pct": float,
            "total_active": int,
            "active_site_count": int,
        }
    """
    headers, rows = _parse_csv(raw_content)
    if not headers:
        return {
            "sites": [],
            "flagged_sites": [],
            "avg_completion_pct": 0.0,
            "total_active": 0,
            "active_site_count": 0,
        }

    site_col = _find_col(headers, "site", "facility", "location", "dc")
    active_col = _find_col(headers, "active")
    loaded_col = _find_col(headers, "loaded")
    staged_col = _find_col(headers, "staged")
    shipped_col = _find_col(headers, "shipped")
    completion_col = _find_col(headers, "completion %", "completion%", "completion", "complete %", "pct complete")

    # Optional week-over-week column — absent is fine
    wow_col: int | None = None
    for cand in ("prior week completion %", "prior completion %", "wow %", "last week %", "prev week %"):
        try:
            wow_col = _find_col(headers, cand)
            break
        except ValueError:
            pass

    def _int(val: str) -> int:
        try:
            return int(val.strip().replace(",", ""))
        except (ValueError, AttributeError):
            return 0

    def _pct(val: str) -> float:
        try:
            cleaned = val.strip().rstrip("%").replace(",", "")
            return float(cleaned)
        except (ValueError, AttributeError):
            return 0.0

    sites = []
    for row in rows:
        if not any(cell.strip() for cell in row):
            continue  # skip blank rows
        if len(row) <= max(site_col, active_col, loaded_col, staged_col, shipped_col, completion_col):
            continue

        site = row[site_col].strip()
        if not site:
            continue

        completion_pct = _pct(row[completion_col])
        wow_trend: float | None = None
        if wow_col is not None and len(row) > wow_col and row[wow_col].strip():
            wow_trend = _pct(row[wow_col])

        flagged = completion_pct < COMPLETION_THRESHOLD_PCT

        sites.append({
            "site": site,
            "active": _int(row[active_col]),
            "loaded": _int(row[loaded_col]),
            "staged": _int(row[staged_col]),
            "shipped": _int(row[shipped_col]),
            "completion_pct": completion_pct,
            "wow_trend_pct": wow_trend,
            "flagged": flagged,
        })

    flagged_sites = [s["site"] for s in sites if s["flagged"]]
    avg_pct = sum(s["completion_pct"] for s in sites) / len(sites) if sites else 0.0
    total_active = sum(s["active"] for s in sites)

    return {
        "sites": sites,
        "flagged_sites": flagged_sites,
        "avg_completion_pct": round(avg_pct, 1),
        "total_active": total_active,
        "active_site_count": len(sites),
    }


# ── parse_dock_pending_log ────────────────────────────────────────────────────

def parse_dock_pending_log(raw_content: str) -> dict[str, Any]:
    """
    Parse the Dock Pending Log Google Drive spreadsheet.

    Assumed input shape: CSV/TSV with columns for resolved flag, site, and
    issue type. Counts rows where Resolved = FALSE (case-insensitive).

    Returns:
        {
            "total_open": int,
            "by_site": {"EAST1": int, ...},
            "by_issue_type": {"PO Number": int, "Pallet/Piece Count Discrepancy": int, ...},
        }
    """
    headers, rows = _parse_csv(raw_content)
    if not headers:
        return {"total_open": 0, "by_site": {}, "by_issue_type": {}}

    resolved_col = _find_col(headers, "resolved", "resolved?", "is resolved", "status")
    site_col = _find_col(headers, "site", "facility", "location", "dc")
    issue_col = _find_col(headers, "issue type", "issue", "type", "problem type")

    by_site: dict[str, int] = {}
    by_issue: dict[str, int] = {}
    total_open = 0

    for row in rows:
        if not any(cell.strip() for cell in row):
            continue
        if len(row) <= max(resolved_col, site_col, issue_col):
            continue

        resolved_val = row[resolved_col].strip().upper()
        if resolved_val in ("FALSE", "NO", "0", "N", ""):
            total_open += 1
            site = row[site_col].strip()
            issue = row[issue_col].strip()
            by_site[site] = by_site.get(site, 0) + 1
            by_issue[issue] = by_issue.get(issue, 0) + 1

    return {
        "total_open": total_open,
        "by_site": by_site,
        "by_issue_type": by_issue,
    }


# ── parse_dwell_tickets ───────────────────────────────────────────────────────

def parse_dwell_tickets(raw_content: str) -> dict[str, Any]:
    """
    Parse the Dwell Ticket Log Google Drive spreadsheet (tab: Dwell Log - Form Responses).

    Assumed input shape: CSV/TSV with columns for open/closed status, date, facility,
    number of packages, client impact, carrier impact, and issue description.

    Applies thresholds from skill prose:
      - packages >= MISSORT_WARNING_PACKAGES (25): Missort Warning
      - packages >= MISSORT_ALERT_PACKAGES (41): Missort Alert (supersedes warning)
      - carrier packages >= MISLOAD_ALERT_CARRIER (80): Possible Misload Alert

    Returns:
        {
            "open_tickets": [{date, facility, packages, client_impact,
                              carrier_impact, issue_description,
                              alert_level: "none"|"missort_warning"|"missort_alert"|"misload_alert"}, ...],
            "missort_warnings": [ticket, ...],
            "missort_alerts": [ticket, ...],
            "misload_alerts": [ticket, ...],
            "total_open": int,
        }
    """
    headers, rows = _parse_csv(raw_content)
    if not headers:
        return {
            "open_tickets": [],
            "missort_warnings": [],
            "missort_alerts": [],
            "misload_alerts": [],
            "total_open": 0,
        }

    status_col = _find_col(headers, "open/closed", "status", "open closed", "open?", "closed?")
    date_col = _find_col(headers, "date", "timestamp", "submitted", "created date", "date submitted")
    facility_col = _find_col(headers, "facility", "site", "location", "dc")
    packages_col = _find_col(headers, "# packages", "packages", "package count", "qty", "quantity", "# pkgs")
    client_col = _find_col(headers, "client impact", "client", "brand impact", "shipper impact")
    carrier_col = _find_col(headers, "carrier impact", "carrier")
    issue_col = _find_col(headers, "issue description", "issue", "description", "notes")

    def _int(val: str) -> int:
        try:
            return int(val.strip().replace(",", ""))
        except (ValueError, AttributeError):
            return 0

    open_tickets = []
    missort_warnings = []
    missort_alerts = []
    misload_alerts = []

    for row in rows:
        if not any(cell.strip() for cell in row):
            continue
        max_col = max(status_col, date_col, facility_col, packages_col, client_col, carrier_col, issue_col)
        if len(row) <= max_col:
            continue

        status_val = row[status_col].strip().upper()
        if status_val != "OPEN":
            continue

        packages = _int(row[packages_col])
        carrier_str = row[carrier_col].strip()
        carrier_packages = _int(carrier_str)

        # Determine alert level (misload_alert can co-exist with missort levels)
        if packages >= MISSORT_ALERT_PACKAGES:
            alert_level = "missort_alert"
        elif packages >= MISSORT_WARNING_PACKAGES:
            alert_level = "missort_warning"
        else:
            alert_level = "none"

        ticket = {
            "date": row[date_col].strip(),
            "facility": row[facility_col].strip(),
            "packages": packages,
            "client_impact": row[client_col].strip(),
            "carrier_impact": carrier_str,
            "issue_description": row[issue_col].strip(),
            "alert_level": alert_level,
            "misload_alert": packages >= MISLOAD_ALERT_CARRIER,
        }
        open_tickets.append(ticket)

        if alert_level == "missort_alert":
            missort_alerts.append(ticket)
        elif alert_level == "missort_warning":
            missort_warnings.append(ticket)
        if ticket["misload_alert"]:
            misload_alerts.append(ticket)

    return {
        "open_tickets": open_tickets,
        "missort_warnings": missort_warnings,
        "missort_alerts": missort_alerts,
        "misload_alerts": misload_alerts,
        "total_open": len(open_tickets),
    }
