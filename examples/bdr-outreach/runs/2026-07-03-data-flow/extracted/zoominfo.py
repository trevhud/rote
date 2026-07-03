"""ZoomInfo API client functions for the BDR outreach pipeline.

Wraps ZoomInfo MCP tool calls from the source skill into deterministic
extracted functions. Every constant that appeared in the source skill's
prose is defined here as a module-level constant so it cannot drift.

Underlying vendor APIs:
- GET /lookup/* (ZoomInfo Lookup API) — taxonomy resolution
- POST /enrich/contact (ZoomInfo Enrich API v2) — contact enrichment
"""

from __future__ import annotations

from typing import Any


# ── Constants (from references/lead-generation.md and conference-enrichment.md)

ENRICH_BATCH_SIZE: int = 10  # ZoomInfo API hard limit per enrich_contacts call

# Output fields for standard lead-gen enrichment (includes employmentHistory
# required by vet_contact to apply the red-flags rubric).
STANDARD_OUTPUT_FIELDS: tuple[str, ...] = (
    "firstName",
    "lastName",
    "jobTitle",
    "email",
    "phone",
    "mobilePhone",
    "contactAccuracyScore",
    "companyName",
    "externalUrls",
    "employmentHistory",
    "directPhoneDoNotCall",
    "mobilePhoneDoNotCall",
    "validDate",
)

# Output fields for conference enrichment (no employmentHistory — vetting is
# not performed on conference contacts; just need email + LinkedIn).
CONFERENCE_OUTPUT_FIELDS: tuple[str, ...] = (
    "firstName",
    "lastName",
    "email",
    "externalUrls",
    "companyName",
    "jobTitle",
)

# Taxonomy lookups to run before any lead generation search. IDs are stable
# across runs — cache result for 30 days (configured in pipeline.yaml cache:).
TAXONOMY_LOOKUPS: tuple[dict[str, Any], ...] = (
    {"field": "management-levels", "values": ["VP Level Exec", "Director"]},
    {"field": "industries", "fuzzyMatch": "pharmaceutical"},
    {"field": "industries", "fuzzyMatch": "biotechnology"},
    {"field": "departments", "fuzzyMatch": "medical"},
)


def resolve_taxonomy_ids(brief: dict[str, Any]) -> dict[str, Any]:
    """Resolve ZoomInfo taxonomy IDs for management levels, industries, and departments.

    Runs four parallel lookups (management-levels, pharma industry, biotech
    industry, Medical & Health department). Results are stable across campaigns
    — the pipeline caches them for 30 days (see pipeline.yaml cache: config).

    Args:
        brief: The campaign brief (currently unused — taxonomy IDs are global,
            not per-campaign — retained in signature for future TA-specific
            industry filtering).

    Returns:
        Dict with keys: vp_director_level_ids, pharma_industry_id,
        biotech_industry_id, medical_department_id.

    Raises:
        NotImplementedError: stub. Production calls
            GET /lookup/outputfields once per entry in TAXONOMY_LOOKUPS.
    """
    raise NotImplementedError(
        "Replace with real ZoomInfo taxonomy lookups.\n"
        "Run four parallel GET /lookup/* calls, one per entry in TAXONOMY_LOOKUPS.\n"
        "Cache result for 30 days."
    )


def enrich_contacts_batch(
    contacts: list[dict[str, Any]],
    output_fields: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Enrich up to ENRICH_BATCH_SIZE contacts via ZoomInfo.

    Enforces the batch size at the function boundary so the pipeline cannot
    accidentally send oversized batches. The caller (workflow or loop harness)
    is responsible for pre-chunking.

    For the standard lead-gen path, uses STANDARD_OUTPUT_FIELDS (includes
    employmentHistory required by vet_contact). For the conference path, uses
    CONFERENCE_OUTPUT_FIELDS instead.

    Args:
        contacts: List of {"firstName", "lastName", "companyName"} dicts.
            Maximum ENRICH_BATCH_SIZE entries; raises ValueError if exceeded.
        output_fields: Override the field list. None → STANDARD_OUTPUT_FIELDS.

    Returns:
        Enriched contact dicts from the ZoomInfo API response.

    Raises:
        ValueError: if len(contacts) > ENRICH_BATCH_SIZE.
        NotImplementedError: stub. Production calls POST /enrich/contact.
    """
    if len(contacts) > ENRICH_BATCH_SIZE:
        raise ValueError(
            f"Batch size {len(contacts)} exceeds ZoomInfo limit {ENRICH_BATCH_SIZE}"
        )
    fields = output_fields or list(STANDARD_OUTPUT_FIELDS)
    raise NotImplementedError(
        f"Replace with real ZoomInfo call: POST /enrich/contact\n"
        f"matchPersonInput={contacts}\n"
        f"outputFields={fields}"
    )


def extract_linkedin_url(external_urls: list[dict[str, Any]]) -> str:
    """Extract LinkedIn URL from ZoomInfo's externalUrls field.

    LinkedIn URLs are nested inside the externalUrls array returned by
    enrich_contacts. This function parses them out so callers don't need
    to know the nesting structure.

    Source: references/conference-enrichment.md lines 107-114.
    """
    if not external_urls:
        return ""
    for url_obj in external_urls:
        url = url_obj.get("url", "")
        if "linkedin.com" in url.lower():
            return url
    return ""
