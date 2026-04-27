"""ZoomInfo API client functions for the BDR outreach pipeline.

Wraps the ZoomInfo MCP tools from the source skill into deterministic
extracted functions. The underlying vendor API is the ZoomInfo Enrich API v2
and the ZoomInfo Search API. Every constant that appeared in the source skill's
prose is defined here as a module-level constant so it cannot drift silently.
"""

from __future__ import annotations

from typing import Any


# ── Constants (from references/lead-generation.md and conference-enrichment.md) ──

ENRICH_BATCH_SIZE = 10  # ZoomInfo API hard limit per enrich_contacts call

STANDARD_OUTPUT_FIELDS = [
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
]

CONFERENCE_OUTPUT_FIELDS = [
    "firstName",
    "lastName",
    "email",
    "externalUrls",
    "companyName",
    "jobTitle",
]

TAXONOMY_LOOKUPS = [
    {"field": "management-levels", "values": ["VP Level Exec", "Director"]},
    {"field": "industries", "fuzzyMatch": "pharmaceutical"},
    {"field": "industries", "fuzzyMatch": "biotechnology"},
    {"field": "departments", "fuzzyMatch": "medical"},
]


def lookup_taxonomy_ids(brief: dict[str, Any]) -> dict[str, Any]:
    """Resolve ZoomInfo taxonomy IDs for management levels, industries, and departments.

    Runs four parallel lookups (management-levels, pharma industry,
    biotech industry, Medical & Health department). Results are stable across
    runs — cache with TTL=30d to avoid repeated setup API calls.

    Underlying API:
    - GET /lookup/outputfields (ZoomInfo Lookup API)
    """
    raise NotImplementedError(
        "Replace with real ZoomInfo taxonomy lookups.\n"
        "Four parallel GET /lookup calls, one per entry in TAXONOMY_LOOKUPS.\n"
        "Cache result for 30 days."
    )


def enrich_contacts_batch(
    contacts: list[dict[str, Any]],
    output_fields: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Enrich up to ENRICH_BATCH_SIZE contacts via ZoomInfo.

    Always includes employmentHistory for the standard path — required
    by vet_contact to apply the red-flags rubric. The conference path
    uses CONFERENCE_OUTPUT_FIELDS instead (no employmentHistory needed).

    Raises ValueError if the batch exceeds ENRICH_BATCH_SIZE.

    Underlying API: POST /enrich/contact (ZoomInfo Enrich API v2)
    """
    if len(contacts) > ENRICH_BATCH_SIZE:
        raise ValueError(
            f"Batch size {len(contacts)} exceeds ZoomInfo limit {ENRICH_BATCH_SIZE}"
        )
    fields = output_fields or STANDARD_OUTPUT_FIELDS
    raise NotImplementedError(
        f"Replace with real ZoomInfo call: POST /enrich/contact\n"
        f"contacts={contacts}\n"
        f"outputFields={fields}"
    )
