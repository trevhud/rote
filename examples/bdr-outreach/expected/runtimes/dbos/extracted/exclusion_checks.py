"""Exclusion checks — MANDATORY filtering before sequence enrollment.

# MCP origin

The bdr-outreach skill ran three exclusion checks in Phase 5, each as
prose-driven loops over the upserted contacts. The skill marked them
**MANDATORY** in English, which is the kind of constraint that's easy
for a fuzzy LLM agent to skip if the prompt drift is bad.

The MCP tools used were:

* ``hubspot_search_lists`` + ``hubspot_get_contact_list_memberships``
* ``hubspot_get_contact_emails`` (with ``daysBack=30``, ``direction=OUTBOUND``)
* ``hubspot_get_contact_enrollment``

# Compiled form

Compilation is the highest-leverage move for these checks because it
**makes them impossible to skip**. Each function is a deterministic
loop. The IR marks the corresponding nodes ``mandatory: true``, the
Temporal adapter emits them as activities the workflow always calls in
order, and the prose enforcement disappears entirely.

Endpoints used:

* ``GET /crm/v3/objects/contacts/{id}/list-memberships``
* ``GET /crm/v3/objects/contacts/{id}/emails?direction=OUTBOUND``
* ``GET /sales/v3/sequences/contacts/{id}/enrollments``
"""

from __future__ import annotations

from ..types import ExclusionRecord, HubSpotContact

# Constant lifted from the IR (constants.days_back). The "30 days"
# window comes from the source skill's prose; once codified here, it
# cannot drift.
RECENT_EMAIL_DAYS: int = 30


async def check_do_not_contact(
    contacts: list[HubSpotContact],
    dnc_list_id: str,
) -> tuple[list[HubSpotContact], list[ExclusionRecord]]:
    """Filter out contacts that appear on the do-not-contact list.

    Args:
        contacts: Contacts to check (after Phase 4 upsert).
        dnc_list_id: HubSpot list ID for "BDR do not contact" — looked
            up once per campaign by the workflow's setup activity.

    Returns:
        Tuple of (passed contacts, exclusion records). Order of passed
        contacts matches the input order; excluded contacts are not
        included in the passed list.

    Raises:
        NotImplementedError: stub for v0.
    """
    raise NotImplementedError(
        "exclusion_checks.check_do_not_contact: implement against HubSpot Lists API"
    )


async def check_recently_emailed(
    contacts: list[HubSpotContact],
) -> tuple[list[HubSpotContact], list[ExclusionRecord]]:
    """Filter out contacts emailed in the last 30 days (outbound).

    Uses the ``RECENT_EMAIL_DAYS`` constant — same value as the source
    skill's prose, but enforced in code so it cannot drift.

    Returns:
        Tuple of (passed contacts, exclusion records).

    Raises:
        NotImplementedError: stub for v0.
    """
    raise NotImplementedError(
        "exclusion_checks.check_recently_emailed: "
        "implement against HubSpot Engagement API"
    )


async def check_active_sequence(
    contacts: list[HubSpotContact],
) -> tuple[list[HubSpotContact], list[ExclusionRecord]]:
    """Filter out contacts already enrolled in any active HubSpot sequence.

    A contact can only be in one HubSpot sequence at a time. This is the
    most error-prone check to skip because the failure mode is silent —
    the second sequence would simply fail at enrollment time without
    surfacing why.

    Raises:
        NotImplementedError: stub for v0.
    """
    raise NotImplementedError(
        "exclusion_checks.check_active_sequence: "
        "implement against HubSpot Sequences API"
    )
