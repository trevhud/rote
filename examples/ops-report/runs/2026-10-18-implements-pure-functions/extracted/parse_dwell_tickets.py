"""
Parse the Dwell Ticket Log spreadsheet from Google Drive.

Assumed input shape: CSV or TSV text with a header row, representing the
'Dwell Log - Form Responses' tab.
Required columns (case- and whitespace-insensitive):
  Open/Closed, Date, Facility, # Packages
Optional columns: Client Impact, Carrier Impact, Issue Description
"""

import csv
import io


BRAND_WARNING_THRESHOLD = 25
BRAND_ALERT_THRESHOLD = 41
CARRIER_ALERT_THRESHOLD = 80


def _normalize(h: str) -> str:
    return (
        h.strip().lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("#", "num")
        .replace(".", "")
    )


def _find_col(headers: list, candidates: list):
    norm = {_normalize(h): h for h in headers}
    for c in candidates:
        if _normalize(c) in norm:
            return norm[_normalize(c)]
    return None


def _classify_alert(
    packages: int,
    carrier_packages: int,
    *,
    brand_warning_threshold: int,
    brand_alert_threshold: int,
    carrier_alert_threshold: int,
) -> str | None:
    # skill: "Flag packages >= 25 (Missort Warning), >= 41 (Missort Alert), carrier >= 80 (Possible Misload Alert)"
    # skill: "Carrier 80+ missing = Possible Misload Alert (darkest red)"
    # rote: assumption: when both carrier >= 80 and packages >= 41, Possible Misload Alert takes
    # priority because the skill's color scheme assigns it "darkest red" (most severe).
    if carrier_packages >= carrier_alert_threshold:
        return "Possible Misload Alert"
    if packages >= brand_alert_threshold:
        return "Missort Alert"
    if packages >= brand_warning_threshold:
        return "Missort Warning"
    return None


def parse_dwell_tickets(
    raw_content: str,
    *,
    brand_warning_threshold: int = BRAND_WARNING_THRESHOLD,
    brand_alert_threshold: int = BRAND_ALERT_THRESHOLD,
    carrier_alert_threshold: int = CARRIER_ALERT_THRESHOLD,
) -> dict:
    """Parse open dwell tickets and apply alert-level thresholds.

    Args:
        raw_content: CSV/TSV text from the 'Dwell Log - Form Responses' tab.
        brand_warning_threshold: Brand packages >= this → Missort Warning.
        brand_alert_threshold: Brand packages >= this → Missort Alert.
        carrier_alert_threshold: Carrier packages >= this → Possible Misload Alert.

    Returns:
        {
            "open_tickets": list[dict],
            "total_open": int,
            "alerts": {"missort_warnings": int, "missort_alerts": int,
                       "possible_misload_alerts": int},
        }

    Raises:
        ValueError: If required columns are absent.
    """
    if not raw_content or not raw_content.strip():
        return {
            "open_tickets": [],
            "total_open": 0,
            "alerts": {"missort_warnings": 0, "missort_alerts": 0, "possible_misload_alerts": 0},
        }

    first_line = raw_content.split("\n")[0]
    dialect = "excel-tab" if "\t" in first_line else "excel"
    reader = csv.DictReader(io.StringIO(raw_content), dialect=dialect)
    headers = list(reader.fieldnames or [])

    status_col = _find_col(headers, ["Open/Closed", "Status", "Open_Closed"])
    date_col = _find_col(headers, ["Date", "Timestamp", "Submitted"])
    facility_col = _find_col(headers, ["Facility", "Site", "Location"])
    packages_col = _find_col(headers, [
        "# Packages", "Packages", "Num Packages", "Package Count", "# of Packages",
    ])
    client_col = _find_col(headers, ["Client Impact", "Client", "Brand Impact"])
    carrier_col = _find_col(headers, ["Carrier Impact", "Carrier", "Carrier Packages"])
    issue_col = _find_col(headers, [
        "Issue Description", "Issue", "Description", "Issue_Description",
    ])

    missing = [
        name for name, col in [
            ("Open/Closed", status_col), ("Date", date_col),
            ("Facility", facility_col), ("# Packages", packages_col),
        ] if col is None
    ]
    if missing:
        raise ValueError(
            "Dwell Ticket Log missing required columns: "
            + ", ".join(missing) + f"; found: {headers}"
        )

    open_tickets = []
    missort_warnings = 0
    missort_alerts = 0
    possible_misload_alerts = 0

    for row in reader:
        # skill: "Filter rows where Open/Closed column = "Open"."
        if row[status_col].strip().lower() != "open":
            continue

        pkg_raw = (row.get(packages_col) or "").strip()
        packages = int(pkg_raw) if pkg_raw.isdigit() else 0

        carrier_raw = (row.get(carrier_col) or "").strip() if carrier_col else ""
        # rote: assumption: carrier_packages is parsed from a numeric Carrier Impact
        # column. If the column holds free text (e.g. "FedEx"), carrier_packages = 0
        # and the carrier alert will not fire from dwell-log data alone.
        carrier_packages = int(carrier_raw) if carrier_raw.isdigit() else 0

        alert_level = _classify_alert(
            packages, carrier_packages,
            brand_warning_threshold=brand_warning_threshold,
            brand_alert_threshold=brand_alert_threshold,
            carrier_alert_threshold=carrier_alert_threshold,
        )

        open_tickets.append({
            "date": row[date_col].strip(),
            "facility": row[facility_col].strip(),
            "packages": packages,
            "client_impact": (row.get(client_col) or "").strip() if client_col else "",
            "carrier_impact": carrier_raw,
            "issue_description": (row.get(issue_col) or "").strip() if issue_col else "",
            "alert_level": alert_level,
        })

        if alert_level == "Possible Misload Alert":
            possible_misload_alerts += 1
        elif alert_level == "Missort Alert":
            missort_alerts += 1
        elif alert_level == "Missort Warning":
            missort_warnings += 1

    return {
        "open_tickets": open_tickets,
        "total_open": len(open_tickets),
        "alerts": {
            "missort_warnings": missort_warnings,
            "missort_alerts": missort_alerts,
            "possible_misload_alerts": possible_misload_alerts,
        },
    }
