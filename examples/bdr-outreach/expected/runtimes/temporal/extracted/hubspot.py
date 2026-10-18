"""HubSpot CRM operations — contacts, lists, sales templates.

# MCP origin

The bdr-outreach skill used these HubSpot MCP tools inside its agent loop:

* ``hubspot_batch_upsert_contacts`` (batch=100)
* ``hubspot_create_list``
* ``hubspot_add_contacts_to_list`` (batch=250)
* ``hubspot_create_sales_template`` / ``hubspot_update_sales_template``

# Compiled form

After compilation these become direct calls to the public HubSpot CRM API
v3 (for contacts/lists) and the internal HubSpot UI API (for sales
templates — which is why a cookie rotation is required).

REST endpoints, in order:

* ``POST /crm/v3/objects/contacts/batch/upsert`` — batch=100 enforced
* ``POST /crm/v3/lists`` — create static list
* ``POST /crm/v3/lists/{listId}/memberships/add`` — batch=250 enforced
* ``POST /sales/v3/templates``  — internal API, requires cookies

The Temporal adapter wraps each function as an ``@activity.defn`` so
the workflow gets free retry/durability and the underlying API never
sees a workflow crash mid-batch.
"""

from __future__ import annotations

from ..types import (
    HubSpotContact,
    PersonalizationOutput,
    VettedContact,
)

# Constants lifted from the IR — these are HubSpot API limits, not
# heuristics, so they live in code rather than config.
UPSERT_BATCH_SIZE: int = 100
ADD_TO_LIST_BATCH_SIZE: int = 250


async def batch_upsert_contacts(contacts: list[VettedContact]) -> list[HubSpotContact]:
    """Upsert contacts to HubSpot in batches of 100.

    Creates new contacts where the email is not yet in HubSpot, updates
    existing contacts otherwise. Preserves order across batches.

    Args:
        contacts: Vetted contacts (any size — internal batching applies).

    Returns:
        HubSpotContact objects with HubSpot IDs assigned.

    Raises:
        NotImplementedError: stub for v0. Production calls
            ``POST /crm/v3/objects/contacts/batch/upsert`` repeatedly,
            chunking by ``UPSERT_BATCH_SIZE``.
    """
    raise NotImplementedError(
        "hubspot.batch_upsert_contacts: implement against HubSpot CRM API v3"
    )


async def create_campaign_list(
    campaign_name: str,
    contacts: list[HubSpotContact],
) -> str:
    """Create a static HubSpot list and add the contacts to it.

    Args:
        campaign_name: Display name (e.g., "Rare Disease Q2 2026 - Orladeyo").
        contacts: HubSpot-resident contacts to add (any size — internal
            batching applies up to ``ADD_TO_LIST_BATCH_SIZE`` per call).

    Returns:
        The HubSpot list ID (used by exclusion checks downstream).

    Raises:
        NotImplementedError: stub for v0.
    """
    raise NotImplementedError(
        "hubspot.create_campaign_list: implement against HubSpot Lists API"
    )


async def upsert_sales_template(
    campaign_name: str,
    personalizations: list[PersonalizationOutput],
) -> list[str]:
    """Create or update HubSpot sales templates for the campaign sequence.

    NOTE: This uses the *internal* HubSpot UI API rather than the public
    REST API. It requires a rotated set of session cookies (see the
    source skill's ``email-templates.md`` for the cookie list). The
    Temporal activity should fail loudly if cookies are missing or
    expired rather than retry — a 401 here means human intervention.

    Args:
        campaign_name: Display name for the template set.
        personalizations: One personalization per contact, used to fill
            in the per-recipient sections of each template.

    Returns:
        List of HubSpot sales template IDs (one per sequence step).

    Raises:
        NotImplementedError: stub for v0.
    """
    raise NotImplementedError(
        "hubspot.upsert_sales_template: implement against internal HubSpot UI API"
    )
