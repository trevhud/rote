"""
Extracted from: examples/bdr-outreach/skill/references/lead-generation.md
                examples/bdr-outreach/skill/references/conference-enrichment.md

ZoomInfo enrichment helper — wraps the enrich_contacts MCP tool with the
fixed batch size and fixed output field set that appear throughout the skill.

The underlying vendor API:  POST /enrich/contacts  (ZoomInfo Enterprise API)
The MCP tool:               zoominfo_enrich_contacts
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: ZoomInfo API limit — documented in skill limits table and lead-generation.md
ENRICH_BATCH_SIZE: int = 10

#: Fixed output field set — source: lead-generation.md lines 96–100
ENRICH_OUTPUT_FIELDS: list[str] = [
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

#: Minimal field set for conference enrichment (no employment history needed)
#: Source: conference-enrichment.md lines 101–103
CONFERENCE_ENRICH_OUTPUT_FIELDS: list[str] = [
    "firstName",
    "lastName",
    "email",
    "externalUrls",
    "companyName",
    "jobTitle",
]

#: Minimum accuracy score to accept a contact — source: lead-generation.md, quality-and-vetting.md
MIN_ACCURACY_SCORE: int = 85


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Contact = dict[str, Any]
EnrichedContact = dict[str, Any]


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def chunk(items: list, size: int) -> list[list]:
    """Split items into chunks of at most `size` elements."""
    return [items[i : i + size] for i in range(0, len(items), size)]


def enrich_contacts_batch(
    contacts: list[Contact],
    tool_fn: Any,  # callable: (contacts, outputFields) -> list[EnrichedContact]
    output_fields: list[str] = ENRICH_OUTPUT_FIELDS,
    batch_size: int = ENRICH_BATCH_SIZE,
) -> list[EnrichedContact]:
    """Enrich a list of contacts via ZoomInfo in batches of `batch_size`.

    Source: lead-generation.md lines 92–110, conference-enrichment.md lines 126–135.

    Args:
        contacts:     List of contacts with at least firstName, lastName, companyName.
        tool_fn:      Callable wrapping the zoominfo_enrich_contacts MCP tool.
        output_fields: Fields to request from ZoomInfo.
        batch_size:   Max contacts per API call (ZoomInfo limit = 10).

    Returns:
        Flat list of enriched contact dicts in the same order as input.
    """
    if batch_size > ENRICH_BATCH_SIZE:
        raise ValueError(
            f"batch_size {batch_size} exceeds ZoomInfo limit of {ENRICH_BATCH_SIZE}"
        )

    results: list[EnrichedContact] = []
    for batch in chunk(contacts, batch_size):
        batch_results = tool_fn(contacts=batch, outputFields=output_fields)
        results.extend(batch_results)
    return results


def meets_accuracy_threshold(contact: EnrichedContact) -> bool:
    """Return True if the contact's accuracy score is at or above the minimum.

    Source: quality-and-vetting.md line 42, lead-generation.md line 108.
    This is a hard pre-filter — contacts below this threshold are discarded
    without calling the LLM vetting judge.
    """
    score = contact.get("contactAccuracyScore")
    if score is None:
        return False
    return int(score) >= MIN_ACCURACY_SCORE


def has_valid_email(contact: EnrichedContact) -> bool:
    """Return True if the contact has a non-empty email address.

    Source: lead-generation.md line 107 ("Discard contacts without valid email addresses").
    """
    email = contact.get("email", "")
    return bool(email and email.strip())
