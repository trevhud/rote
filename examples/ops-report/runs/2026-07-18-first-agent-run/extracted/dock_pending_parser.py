"""Parse the Dock Pending Log into a structured open-issue summary.

Pure function. Extracted from SKILL.md data-source section:
'Count all rows where Resolved = FALSE. Break down by site and issue type.'
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SitePendingBreakdown:
    site: str
    total: int
    by_issue_type: dict[str, int] = field(default_factory=dict)


@dataclass
class DockPendingSummary:
    total_open: int
    by_site: list[SitePendingBreakdown] = field(default_factory=list)


def parse_dock_pending_log(rows: list[dict]) -> DockPendingSummary:
    """Count open dock pending rows (Resolved = FALSE), grouped by site and issue type.

    Expects each row dict to have:
      Resolved (or resolved): boolean-ish — "FALSE", False, "No", 0 = unresolved
      Site (or site): site name, e.g. EAST1, WEST2
      Issue Type (or issue_type): e.g. "PO Number", "Pallet/Piece Count Discrepancy"
    """
    open_rows = [r for r in rows if _is_unresolved(r)]
    site_map: dict[str, dict[str, int]] = {}

    for row in open_rows:
        site = str(row.get("Site") or row.get("site") or "UNKNOWN")
        issue_type = str(row.get("Issue Type") or row.get("issue_type") or "Other")
        site_map.setdefault(site, {})
        site_map[site][issue_type] = site_map[site].get(issue_type, 0) + 1

    breakdowns = [
        SitePendingBreakdown(
            site=site,
            total=sum(counts.values()),
            by_issue_type=dict(sorted(counts.items())),
        )
        for site, counts in sorted(site_map.items())
    ]

    return DockPendingSummary(total_open=len(open_rows), by_site=breakdowns)


def _is_unresolved(row: dict) -> bool:
    val = row.get("Resolved") or row.get("resolved") or ""
    if isinstance(val, bool):
        return not val
    return str(val).strip().upper() in ("FALSE", "NO", "0", "")
