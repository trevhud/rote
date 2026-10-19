"""
Pure-function parsers for the ops-report data sources.

All four Google Drive / Gmail pulls flow through these functions before
any downstream processing. Stdlib only; no third-party imports.
"""

import csv
import io
import json


# ---------------------------------------------------------------------------
# Constants (lifted from skill prose — Pattern 2)
# ---------------------------------------------------------------------------

# skill: "Flag any site below 75% completion"
COMPLETION_THRESHOLD: float = 75.0

# skill: "packages >= 25 (Missort Warning), >= 41 (Missort Alert),
#         carrier >= 80 (Possible Misload Alert)"
DWELL_PACKAGES_WARNING: int = 25
DWELL_PACKAGES_ALERT: int = 41
# rote: assumption: "carrier >= 80 (Possible Misload Alert)" is applied to the
# raw # packages column of each open ticket row, treating the highest threshold
# as Possible Misload Alert (80+), middle as Missort Alert (41-79), and lowest
# as Missort Warning (25-40). The skill does not state whether this is per-row
# or aggregated by carrier name; per-row is chosen as the simpler/more
# actionable interpretation. See compile-report.md Open questions.
DWELL_CARRIER_ALERT: int = 80


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_col(headers: list, *names: str) -> int:
    """Return index of first header matching any of names (case+whitespace insensitive)."""
    name_set = {n.strip().lower() for n in names}
    for i, h in enumerate(headers):
        if h.strip().lower() in name_set:
            return i
    raise ValueError(
        "Table has no column matching any of " + repr(names) + "; found " + repr(headers)
    )


def _int_or_zero(val: str) -> int:
    val = val.strip()
    if not val:
        return 0
    return int(val)


def _float_from_pct(val: str) -> float:
    return float(val.strip().rstrip("%"))


# ---------------------------------------------------------------------------
# parse_shipment_containers
# ---------------------------------------------------------------------------

def parse_shipment_containers(
    *,
    raw: str,
    completion_threshold: float = COMPLETION_THRESHOLD,
) -> dict:
    """
    Parse the shipment containers CSV from Google Drive.

    Input shape assumed: CSV text with a header row containing SITE (or
    Facility), ACTIVE, LOADED, STAGED, SHIPPED, COMPLETION % columns.
    An optional column whose header contains both 'prev' and 'completion'
    supplies week-over-week trend data.

    Returns:
      {
        "sites": list[dict],          # per-site rows
        "avg_completion_pct": float,
        "active_site_count": int,
        "flagged_sites": list[str]    # sites below completion_threshold
      }

    Each site dict: {site, active, loaded, staged, shipped,
                     completion_pct, flagged, wow_trend (float|None)}
    """
    reader = csv.reader(io.StringIO(raw.strip()))
    rows = list(reader)
    if not rows:
        raise ValueError("Shipment containers table is empty")

    headers = rows[0]

    # skill: "Parse the facility summary table with COMPLETION % column"
    site_col = _find_col(headers, "SITE", "Facility", "Site")
    active_col = _find_col(headers, "ACTIVE", "Active")
    loaded_col = _find_col(headers, "LOADED", "Loaded")
    staged_col = _find_col(headers, "STAGED", "Staged")
    shipped_col = _find_col(headers, "SHIPPED", "Shipped")
    completion_col = _find_col(headers, "COMPLETION %", "Completion %", "COMPLETION%", "Completion%")

    # Optional week-over-week column
    prev_completion_col = None
    for i, h in enumerate(headers):
        hl = h.strip().lower()
        if "prev" in hl and "completion" in hl:
            prev_completion_col = i
            break

    sites = []
    total_completion = 0.0
    flagged_sites = []

    for row in rows[1:]:
        if not row or not any(cell.strip() for cell in row):
            continue

        site = row[site_col].strip()
        active = _int_or_zero(row[active_col])
        loaded = _int_or_zero(row[loaded_col])
        staged = _int_or_zero(row[staged_col])
        shipped = _int_or_zero(row[shipped_col])
        completion_pct = _float_from_pct(row[completion_col]) if row[completion_col].strip() else 0.0

        wow_trend = None
        if prev_completion_col is not None:
            prev_raw = row[prev_completion_col].strip().rstrip("%")
            if prev_raw:
                wow_trend = round(completion_pct - float(prev_raw), 2)

        # skill: "Flag any site below 75% completion"
        flagged = completion_pct < completion_threshold
        if flagged:
            flagged_sites.append(site)

        sites.append({
            "site": site,
            "active": active,
            "loaded": loaded,
            "staged": staged,
            "shipped": shipped,
            "completion_pct": completion_pct,
            "flagged": flagged,
            "wow_trend": wow_trend,
        })
        total_completion += completion_pct

    avg_completion_pct = round(total_completion / len(sites), 2) if sites else 0.0

    return {
        "sites": sites,
        "avg_completion_pct": avg_completion_pct,
        "active_site_count": len(sites),
        "flagged_sites": flagged_sites,
    }


# ---------------------------------------------------------------------------
# parse_dock_pending_log
# ---------------------------------------------------------------------------

def parse_dock_pending_log(*, raw: str) -> dict:
    """
    Parse the dock pending log CSV from Google Drive.

    Input shape assumed: CSV text with header row containing RESOLVED (or
    Resolved), SITE (or Site), and ISSUE TYPE (or Issue Type) columns.

    Returns:
      {
        "total_open": int,
        "by_site": {site: count},
        "by_issue_type": {issue_type: count}
      }
    """
    reader = csv.reader(io.StringIO(raw.strip()))
    rows = list(reader)
    if not rows:
        raise ValueError("Dock pending log table is empty")

    headers = rows[0]

    resolved_col = _find_col(headers, "RESOLVED", "Resolved")
    site_col = _find_col(headers, "SITE", "Site")
    issue_col = _find_col(
        headers,
        "ISSUE TYPE", "Issue Type", "ISSUE_TYPE", "IssueType",
        "PO Number", "PO/Issue Type",
    )

    total_open = 0
    by_site: dict = {}
    by_issue_type: dict = {}

    for row in rows[1:]:
        if not row or not any(cell.strip() for cell in row):
            continue

        # skill: "Count all rows where Resolved = FALSE"
        resolved_val = row[resolved_col].strip().upper()
        if resolved_val in ("FALSE", "0", "NO", "N"):
            total_open += 1
            site = row[site_col].strip()
            issue = row[issue_col].strip()
            by_site[site] = by_site.get(site, 0) + 1
            by_issue_type[issue] = by_issue_type.get(issue, 0) + 1

    return {
        "total_open": total_open,
        "by_site": by_site,
        "by_issue_type": by_issue_type,
    }


# ---------------------------------------------------------------------------
# parse_dock_activity_emails
# ---------------------------------------------------------------------------

def parse_dock_activity_emails(*, raw) -> dict:
    """
    Parse the Gmail search_threads result for dock-activity emails.

    Input shape assumed: dict (or JSON string) with a 'threads' key containing
    a list of thread objects. Each thread object is expected to have at least
    one of 'subject' or 'snippet' as a string.

    # rote: assumption: The Gmail MCP search_threads tool returns a JSON-
    # serializable dict with a 'threads' key. Each thread has a 'subject' or
    # 'snippet' field. Appointment status is inferred from the word 'approved'
    # (→ Approved) or 'request'/'requested' (→ Requested) appearing in the
    # thread subject/snippet. The skill does not define the email format; this
    # keyword heuristic is the simplest defensible interpretation.

    Returns:
      {
        "approved_count": int,
        "requested_count": int,
        "unapproved_requests": list[str],   # subjects of unapproved request threads
        "total_threads": int
      }
    """
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(
                "parse_dock_activity_emails: could not parse raw as JSON: "
                + repr(raw[:200])
            ) from exc
    elif isinstance(raw, dict):
        data = raw
    else:
        raise ValueError(
            "parse_dock_activity_emails: expected str or dict, got "
            + type(raw).__name__
        )

    threads = data.get("threads", [])
    if not isinstance(threads, list):
        raise ValueError(
            "parse_dock_activity_emails: 'threads' field must be a list, got "
            + type(threads).__name__
        )

    approved_count = 0
    requested_count = 0
    unapproved_requests: list = []

    for thread in threads:
        subject = thread.get("subject") or thread.get("snippet") or ""
        subject_lower = subject.lower()

        # skill: "Count Approved vs. Requested appointments"
        if "approved" in subject_lower:
            approved_count += 1
        elif "request" in subject_lower:
            # skill: "Flag any outstanding requests not yet approved"
            requested_count += 1
            unapproved_requests.append(subject)

    return {
        "approved_count": approved_count,
        "requested_count": requested_count,
        "unapproved_requests": unapproved_requests,
        "total_threads": len(threads),
    }


# ---------------------------------------------------------------------------
# parse_dwell_tickets
# ---------------------------------------------------------------------------

def _classify_dwell_packages(
    packages: int,
    *,
    packages_warning_threshold: int,
    packages_alert_threshold: int,
    carrier_alert_threshold: int,
) -> str:
    """Return alert level string for a package count, or empty string if no alert."""
    # skill: "packages >= 25 (Missort Warning), >= 41 (Missort Alert),
    #         carrier >= 80 (Possible Misload Alert)"
    # rote: assumption: thresholds are applied as ranges (not cumulative), with
    # the highest threshold taking precedence: 80+ = Possible Misload Alert,
    # 41-79 = Missort Alert, 25-40 = Missort Warning.
    if packages >= carrier_alert_threshold:
        return "Possible Misload Alert"
    if packages >= packages_alert_threshold:
        return "Missort Alert"
    if packages >= packages_warning_threshold:
        return "Missort Warning"
    return ""


def parse_dwell_tickets(
    *,
    raw: str,
    packages_warning_threshold: int = DWELL_PACKAGES_WARNING,
    packages_alert_threshold: int = DWELL_PACKAGES_ALERT,
    carrier_alert_threshold: int = DWELL_CARRIER_ALERT,
) -> dict:
    """
    Parse the dwell ticket log CSV from Google Drive, tab "Dwell Log - Form Responses".

    Input shape assumed: CSV text with header row. Required columns (case- and
    whitespace-insensitive): Open/Closed (or Status), Date (or Timestamp),
    Facility (or Site), # Packages (or Packages), Client Impact, Carrier Impact,
    Issue Description.

    Returns:
      {
        "open_tickets": list[dict],
        "total_open": int
      }

    Each ticket dict: {date, facility, packages (int), client_impact,
                       carrier_impact, issue_description, alert_level}
    """
    reader = csv.reader(io.StringIO(raw.strip()))
    rows = list(reader)
    if not rows:
        raise ValueError("Dwell ticket log table is empty")

    headers = rows[0]

    # skill: 'Tab: Dwell Log - Form Responses. Filter rows where Open/Closed column = "Open".'
    status_col = _find_col(headers, "Open/Closed", "open/closed", "Status", "status")
    date_col = _find_col(headers, "Date", "date", "Timestamp", "timestamp")
    facility_col = _find_col(headers, "Facility", "facility", "Site", "site")
    packages_col = _find_col(headers, "# Packages", "# packages", "Packages", "packages", "# Orders/Packages")
    client_col = _find_col(headers, "Client Impact", "client impact", "ClientImpact")
    carrier_col = _find_col(headers, "Carrier Impact", "carrier impact", "CarrierImpact")
    issue_col = _find_col(headers, "Issue Description", "issue description", "Issue", "issue")

    open_tickets: list = []

    for row in rows[1:]:
        if not row or not any(cell.strip() for cell in row):
            continue

        # skill: 'Filter rows where Open/Closed column = "Open"'
        status_val = row[status_col].strip().lower()
        if status_val != "open":
            continue

        packages = _int_or_zero(row[packages_col].strip().replace(",", ""))
        alert_level = _classify_dwell_packages(
            packages,
            packages_warning_threshold=packages_warning_threshold,
            packages_alert_threshold=packages_alert_threshold,
            carrier_alert_threshold=carrier_alert_threshold,
        )

        open_tickets.append({
            "date": row[date_col].strip(),
            "facility": row[facility_col].strip(),
            "packages": packages,
            "client_impact": row[client_col].strip(),
            "carrier_impact": row[carrier_col].strip(),
            "issue_description": row[issue_col].strip(),
            "alert_level": alert_level,
        })

    return {
        "open_tickets": open_tickets,
        "total_open": len(open_tickets),
    }
