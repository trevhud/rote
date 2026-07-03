"""ZoomInfo taxonomy lookups — resolved once per campaign, cached 30 days.

Wraps the zoominfo_lookup MCP tool from the source skill into a single
deterministic function. The skill made four parallel lookups in Phase 2 setup;
this function runs them and returns a typed result.

The taxonomy IDs (management level, industry, department) are stable —
they do not change between campaigns or even between months. The pipeline
caches this node's output for 30 days (see pipeline.yaml cache: config).

Source: references/lead-generation.md lines 19–26.

MCP tool originally used: zoominfo_lookup (management-levels, industries,
departments with fuzzyMatch).

Underlying vendor API:
- GET https://api.zoominfo.com/lookup/managementLevel
- GET https://api.zoominfo.com/lookup/industry?fuzzyMatch=pharmaceutical
- GET https://api.zoominfo.com/lookup/industry?fuzzyMatch=biotechnology
- GET https://api.zoominfo.com/lookup/department?fuzzyMatch=medical
"""

from __future__ import annotations

from typing import Any


def resolve_taxonomy_ids(brief: dict[str, Any]) -> dict[str, Any]:
    """Resolve ZoomInfo taxonomy IDs needed for lead generation searches.

    Runs four lookups in parallel:
    1. management-levels → IDs for "VP Level Exec" and "Director"
    2. industries (fuzzyMatch=pharmaceutical) → pharma industry ID
    3. industries (fuzzyMatch=biotechnology) → biotech industry ID
    4. departments (fuzzyMatch=medical) → Medical & Health department ID

    Args:
        brief: Campaign brief (currently unused — taxonomy IDs are global,
            not per-campaign — retained for future TA-specific filtering).

    Returns:
        Dict with keys:
          vp_level_exec_id: str
          director_id:      str
          pharma_industry_id: str
          biotech_industry_id: str
          medical_dept_id: str

    Raises:
        NotImplementedError: stub. Production calls the ZoomInfo Lookup API
            in four parallel requests, one per taxonomy dimension.
    """
    raise NotImplementedError(
        "Replace with real ZoomInfo Lookup API calls.\n"
        "Four parallel GETs (management-levels, pharma industry, biotech industry,\n"
        "medical department). Cache result for 30 days."
    )
