"""ZoomInfo taxonomy lookups — resolved once per campaign and cached.

# MCP origin

Originally invoked from inside the bdr-outreach skill via the
``zoominfo_lookup`` MCP tool. The skill made three lookups in parallel
during Phase 2 setup:

* ``management-levels`` — VP Level Exec, Director
* ``industries`` (fuzzyMatch=pharmaceutical) — pharma ID
* ``industries`` (fuzzyMatch=biotechnology) — biotech ID
* ``departments`` (fuzzyMatch=medical) — Medical & Health ID

# Compiled form

The taxonomy IDs are stable — they do not change run-over-run. After
compilation, this function calls the ZoomInfo Lookup REST endpoint
directly and caches the result for 30 days (per the IR's ``cache:``
config). No agent loop, no LLM, no MCP server in the path.

Real implementation would call::

    GET https://api.zoominfo.com/lookup/managementLevel
    GET https://api.zoominfo.com/lookup/industry?fuzzyMatch=pharmaceutical
    GET https://api.zoominfo.com/lookup/industry?fuzzyMatch=biotechnology
    GET https://api.zoominfo.com/lookup/department?fuzzyMatch=medical
"""

from __future__ import annotations

from ..types import CampaignBrief, TaxonomyIds


async def resolve_taxonomy_ids(brief: CampaignBrief) -> TaxonomyIds:
    """Resolve the ZoomInfo IDs needed for lead generation searches.

    Args:
        brief: The campaign brief (currently unused — taxonomy IDs are
            global, not per-campaign — but kept in the signature for
            future extensibility, e.g., TA-specific industry filtering).

    Returns:
        TaxonomyIds with all five IDs resolved.

    Raises:
        NotImplementedError: stub for v0. Production implementation calls
            the ZoomInfo Lookup endpoints via httpx.
    """
    raise NotImplementedError(
        "taxonomy.resolve_taxonomy_ids: implement against ZoomInfo Lookup API"
    )
