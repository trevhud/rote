"""
Parse the Dock Pending Log spreadsheet from Google Drive.

Assumed input shape: CSV or TSV text with a header row.
Required columns (case- and whitespace-insensitive):
  Resolved, Site, Issue Type
"""

import csv
import io


def _normalize(h: str) -> str:
    return h.strip().lower().replace(" ", "_")


def _find_col(headers: list, candidates: list):
    norm = {_normalize(h): h for h in headers}
    for c in candidates:
        if _normalize(c) in norm:
            return norm[_normalize(c)]
    return None


def parse_dock_pending_log(raw_content: str) -> dict:
    """Count unresolved dock pending entries, broken down by site and issue type.

    Args:
        raw_content: CSV/TSV text content from Google Drive read_file_content.

    Returns:
        {
            "total_open": int,
            "by_site": dict[str, int],
            "by_issue_type": dict[str, int],
        }

    Raises:
        ValueError: If any required column is absent.
    """
    if not raw_content or not raw_content.strip():
        return {"total_open": 0, "by_site": {}, "by_issue_type": {}}

    first_line = raw_content.split("\n")[0]
    dialect = "excel-tab" if "\t" in first_line else "excel"
    reader = csv.DictReader(io.StringIO(raw_content), dialect=dialect)
    headers = list(reader.fieldnames or [])

    resolved_col = _find_col(headers, ["Resolved", "Is Resolved", "Resolved?"])
    site_col = _find_col(headers, ["Site", "Facility", "Location"])
    issue_col = _find_col(headers, ["Issue Type", "Issue", "Issue_Type", "Type", "Category"])

    missing = [
        name for name, col in [
            ("Resolved", resolved_col), ("Site", site_col), ("Issue Type", issue_col),
        ] if col is None
    ]
    if missing:
        raise ValueError(
            "Dock Pending Log missing required columns: "
            + ", ".join(missing) + f"; found: {headers}"
        )

    total_open = 0
    by_site: dict = {}
    by_issue_type: dict = {}

    for row in reader:
        # skill: "Count all rows where Resolved = FALSE"
        resolved_val = row[resolved_col].strip().upper()
        if resolved_val in ("FALSE", "NO", "0", ""):
            total_open += 1
            site = row[site_col].strip()
            issue = row[issue_col].strip()
            by_site[site] = by_site.get(site, 0) + 1
            by_issue_type[issue] = by_issue_type.get(issue, 0) + 1

    return {"total_open": total_open, "by_site": by_site, "by_issue_type": by_issue_type}
