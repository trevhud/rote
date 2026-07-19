"""HubSpot CRM client functions for the BDR outreach pipeline.

Wraps HubSpot MCP tool calls from the source skill into deterministic
extracted functions. Every constant from the source skill's prose is
defined here as a module-level constant so it cannot drift.

The three MANDATORY exclusion checks live in extracted/exclusion_checks.py
(separate file — they have their own constants and test surface).

Underlying vendor APIs:
- POST /crm/v3/objects/contacts/batch/upsert  — batch contact upsert
- POST /contacts/v1/lists                     — create static list
- POST /contacts/v1/lists/{listId}/add        — add contacts to list
- Internal HubSpot UI API (undocumented)      — sales template CRUD
"""

from __future__ import annotations

from typing import Any


# ── Constants (from references/hubspot-operations.md and SKILL.md limits table)

UPSERT_BATCH_SIZE: int = 100       # HubSpot batch upsert hard limit per call
ADD_TO_LIST_BATCH_SIZE: int = 250  # HubSpot add-to-list hard limit per call


def batch_upsert_contacts(contacts: list[dict[str, Any]]) -> dict[str, Any]:
    """Upsert contacts to HubSpot in chunks of UPSERT_BATCH_SIZE.

    Creates contacts that don't exist (matched by email); updates existing
    ones. Returns HubSpot contact IDs needed by downstream exclusion checks
    and list management.

    Args:
        contacts: Vetted contacts (any size — internal chunking applies).

    Returns:
        Dict with keys:
          upserted: list[dict] — contacts with hubspot_id assigned
          created_count: int
          updated_count: int

    Raises:
        NotImplementedError: stub. Production calls
            POST /crm/v3/objects/contacts/batch/upsert
            in chunks of UPSERT_BATCH_SIZE.
    """
    if not contacts:
        return {"upserted": [], "created_count": 0, "updated_count": 0}
    raise NotImplementedError(
        "Replace with real HubSpot call: POST /crm/v3/objects/contacts/batch/upsert\n"
        f"Chunk input list into batches of {UPSERT_BATCH_SIZE}."
    )


def create_campaign_list(campaign_name: str) -> dict[str, Any]:
    """Create a static (MANUAL) HubSpot list for the campaign.

    Naming convention (from hubspot-operations.md):
      "Rare Disease Campaign - Q1 2026"
      "Denver Conference 2026 - Pharma Speakers"

    Args:
        campaign_name: Display name for the list.

    Returns:
        Dict with keys: list_id: str, list_name: str.

    Raises:
        NotImplementedError: stub. Production calls POST /contacts/v1/lists.
    """
    raise NotImplementedError(
        f"Replace with real HubSpot call: POST /contacts/v1/lists\n"
        f"body: {{name: {campaign_name!r}, dynamic: false}}"
    )


def add_contacts_to_list(
    list_id: str,
    contacts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Add upserted contacts to the campaign list in chunks of ADD_TO_LIST_BATCH_SIZE.

    Requires HubSpot contact IDs (from batch_upsert_contacts) and the list_id
    (from create_campaign_list).

    Args:
        list_id: HubSpot list ID from create_campaign_list.
        contacts: HubSpot-resident contacts with hubspot_id (any size —
            internal chunking applies).

    Returns:
        Dict with keys: added_count: int, list_url: str.

    Raises:
        NotImplementedError: stub. Production calls
            POST /contacts/v1/lists/{listId}/add
            in chunks of ADD_TO_LIST_BATCH_SIZE.
    """
    if not contacts:
        return {"added_count": 0, "list_url": f"https://app.hubspot.com/contacts/lists/{list_id}"}
    raise NotImplementedError(
        f"Replace with real HubSpot call: POST /contacts/v1/lists/{list_id}/add\n"
        f"Chunk contact IDs into batches of {ADD_TO_LIST_BATCH_SIZE}."
    )


def upsert_sales_template(
    campaign_name: str,
    personalizations: list[dict[str, Any]],
) -> list[str]:
    """Create or update HubSpot sales email templates for the campaign sequence.

    Uses the INTERNAL HubSpot UI API (not the CMS Design Manager). Requires
    browser session cookies: csrf.app, hubspotapi, hubspotapi-csrf,
    hubspotapi-strict, hubspotapi-lax. Cookies expire periodically — rotate
    via your rotate-hubspot-cookie script.

    Template changes do NOT affect contacts already enrolled in a sequence
    (they use a snapshot captured at enrollment time).

    NOTE: A 401 response means cookies have expired and require human
    intervention. This function should raise loudly on auth failure rather
    than retrying — the workflow surface area will make the root cause clear.

    Args:
        campaign_name: Display name for the template set.
        personalizations: One PersonalizationOutput dict per contact
            (keys: opening_line, ta_callout).

    Returns:
        List of HubSpot sales template IDs (one per sequence step).

    Raises:
        NotImplementedError: stub. Production calls
            hubspot_create_sales_template or hubspot_update_sales_template.
    """
    raise NotImplementedError(
        "Replace with real HubSpot internal API call.\n"
        "Use hubspot_create_sales_template (new) or hubspot_update_sales_template (edit).\n"
        "Requires 5 browser session cookies — see references/email-templates.md."
    )
