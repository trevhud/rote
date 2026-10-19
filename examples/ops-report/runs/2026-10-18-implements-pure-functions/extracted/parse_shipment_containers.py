"""
Parse the Shipment Containers spreadsheet from Google Drive.

Assumed input shape: CSV or TSV text with a header row.
Required columns (case- and whitespace-insensitive):
  Site, Active, Loaded, Staged, Shipped, COMPLETION %
Optional column for week-over-week trend:
  Prev Completion %, Last Week %, Prior %, Previous %, Prior Completion %
"""

import csv
import io


COMPLETION_THRESHOLD = 75.0


def _normalize(h: str) -> str:
    return h.strip().lower().replace(" ", "_").replace("%", "pct").replace("/", "_")


def _find_col(headers: list, candidates: list):
    norm = {_normalize(h): h for h in headers}
    for c in candidates:
        if _normalize(c) in norm:
            return norm[_normalize(c)]
    return None


def parse_shipment_containers(raw_content: str, *, completion_threshold: float = COMPLETION_THRESHOLD) -> dict:
    """Parse facility summary table from the Shipment Containers spreadsheet.

    Args:
        raw_content: CSV/TSV text content from Google Drive read_file_content.
        completion_threshold: Flag sites with completion % strictly below this value.

    Returns:
        {
            "sites": list[dict],     # per-site data including flagged bool and wow_trend
            "flagged_sites": list[str],
            "total_active": int,
            "avg_completion_pct": float,
        }

    Raises:
        ValueError: If any required column is absent from the header row.
    """
    if not raw_content or not raw_content.strip():
        return {"sites": [], "flagged_sites": [], "total_active": 0, "avg_completion_pct": 0.0}

    first_line = raw_content.split("\n")[0]
    dialect = "excel-tab" if "\t" in first_line else "excel"
    reader = csv.DictReader(io.StringIO(raw_content), dialect=dialect)
    headers = list(reader.fieldnames or [])

    site_col = _find_col(headers, ["Site", "Facility"])
    active_col = _find_col(headers, ["Active"])
    loaded_col = _find_col(headers, ["Loaded"])
    staged_col = _find_col(headers, ["Staged"])
    shipped_col = _find_col(headers, ["Shipped"])
    completion_col = _find_col(headers, [
        "COMPLETION %", "Completion %", "Completion_Pct", "COMPLETION_PCT", "completion_pct",
    ])

    missing = [
        name for name, col in [
            ("Site", site_col), ("Active", active_col), ("Loaded", loaded_col),
            ("Staged", staged_col), ("Shipped", shipped_col), ("COMPLETION %", completion_col),
        ] if col is None
    ]
    if missing:
        raise ValueError(
            "Shipment Containers table missing required columns: "
            + ", ".join(missing) + f"; found: {headers}"
        )

    prev_col = _find_col(headers, [
        "Prev Completion %", "Last Week %", "Prior %", "Previous %",
        "Prev %", "Prior Completion %",
    ])

    sites = []
    flagged_sites = []
    total_active = 0

    for row in reader:
        site = row[site_col].strip()
        if not site:
            continue

        active = int(row[active_col].strip() or 0)
        loaded = int(row[loaded_col].strip() or 0)
        staged = int(row[staged_col].strip() or 0)
        shipped = int(row[shipped_col].strip() or 0)

        pct_raw = row[completion_col].strip().rstrip("%")
        completion_pct = float(pct_raw) if pct_raw else 0.0

        wow_trend = None
        if prev_col:
            prev_raw = (row.get(prev_col) or "").strip().rstrip("%")
            if prev_raw:
                try:
                    wow_trend = round(completion_pct - float(prev_raw), 2)
                except ValueError:
                    wow_trend = None

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
        total_active += active

    avg_completion_pct = (
        round(sum(s["completion_pct"] for s in sites) / len(sites), 2) if sites else 0.0
    )
    return {
        "sites": sites,
        "flagged_sites": flagged_sites,
        "total_active": total_active,
        "avg_completion_pct": avg_completion_pct,
    }
