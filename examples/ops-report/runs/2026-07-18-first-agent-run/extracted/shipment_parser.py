"""Parse the Shipment Containers spreadsheet into a structured facility summary.

Pure function: same rows in, same summary out. No LLM, no API calls.
Extracted from SKILL.md data-source section: 'Parse the facility summary table
with COMPLETION % column. Flag any site below 75% completion.'
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Extracted from SKILL.md: 'Flag any site below 75% completion'
COMPLETION_THRESHOLD_PCT: float = 75.0


@dataclass
class FacilitySummary:
    site: str
    active: int
    loaded: int
    staged: int
    shipped: int
    completion_pct: float
    below_threshold: bool
    week_over_week_delta: float | None = None  # None when prior-week data unavailable


@dataclass
class ShipmentContainersSummary:
    facilities: list[FacilitySummary] = field(default_factory=list)
    avg_completion_pct: float = 0.0
    active_site_count: int = 0
    flagged_sites: list[str] = field(default_factory=list)


def parse_shipment_containers(rows: list[dict]) -> ShipmentContainersSummary:
    """Parse facility summary rows from the Shipment Containers spreadsheet.

    Expects each row dict to have columns (case-insensitive, underscore aliases):
      site / Site, active / Active, loaded / Loaded, staged / Staged,
      shipped / Shipped, completion_pct / COMPLETION % / completion
      (optional) prior_completion_pct / prior_completion  — for week-over-week delta

    Flags any facility with completion_pct < COMPLETION_THRESHOLD_PCT (75%).
    """
    facilities: list[FacilitySummary] = []

    for row in rows:
        site = str(row.get("site") or row.get("Site") or "")
        active = int(row.get("active") or row.get("Active") or 0)
        loaded = int(row.get("loaded") or row.get("Loaded") or 0)
        staged = int(row.get("staged") or row.get("Staged") or 0)
        shipped = int(row.get("shipped") or row.get("Shipped") or 0)

        raw_pct = (
            row.get("completion_pct")
            or row.get("COMPLETION %")
            or row.get("completion")
            or 0
        )
        completion_pct = float(str(raw_pct).strip("%").strip())

        prior_raw = row.get("prior_completion_pct") or row.get("prior_completion")
        wow_delta: float | None = None
        if prior_raw is not None:
            prior_pct = float(str(prior_raw).strip("%").strip())
            wow_delta = completion_pct - prior_pct

        facilities.append(
            FacilitySummary(
                site=site,
                active=active,
                loaded=loaded,
                staged=staged,
                shipped=shipped,
                completion_pct=completion_pct,
                below_threshold=completion_pct < COMPLETION_THRESHOLD_PCT,
                week_over_week_delta=wow_delta,
            )
        )

    flagged = [f.site for f in facilities if f.below_threshold]
    avg_pct = (
        sum(f.completion_pct for f in facilities) / len(facilities)
        if facilities
        else 0.0
    )

    return ShipmentContainersSummary(
        facilities=facilities,
        avg_completion_pct=round(avg_pct, 1),
        active_site_count=sum(1 for f in facilities if f.active > 0),
        flagged_sites=flagged,
    )
