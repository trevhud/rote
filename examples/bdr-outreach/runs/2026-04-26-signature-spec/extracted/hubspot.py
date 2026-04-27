"""HubSpot CRM client functions for the BDR outreach pipeline.

Every constant that appeared in the source skill's prose is defined here
as a module-level constant. The three exclusion-check helpers live in
extracted/exclusion_checks.py (they have their own constants and test surface).
"""

from __future__ import annotations

from typing import Any


# ── Constants (from references/hubspot-operations.md) ──

UPSERT_BATCH_SIZE = 100   # HubSpot batch upsert hard limit per call
ADD_TO_LIST_BATCH_SIZE = 250  # HubSpot add-to-list hard limit per call


def batch_upsert_contacts(contacts: list[dict[str, Any]]) -> dict[str, Any]:
    """Upsert contacts to HubSpot in chunks of UPSERT_BATCH_SIZE.

    Creates contacts that don't exist; updates existing ones matched by email.
    Returns counts of created vs updated records and the list of HubSpot contacts
    (including their assigned hubspot_id, required by later nodes).

    Underlying API: POST /crm/v3/objects/contacts/batch/upsert
    """
    if not contacts:
        return {"upserted": [], "created_count": 0, "updated_count": 0}
    # Internally chunks into batches of UPSERT_BATCH_SIZE
    raise NotImplementedError(
        "Replace with real HubSpot call: POST /crm/v3/objects/contacts/batch/upsert\n"
        f"Chunk input list into batches of {UPSERT_BATCH_SIZE}."
    )


def create_campaign_list(campaign_name: str) -> dict[str, Any]:
    """Create a static (MANUAL) HubSpot list for the campaign.

    List naming convention from hubspot-operations.md:
      e.g. "Rare Disease Campaign - Q1 2026", "Denver Conference 2026 - Pharma Speakers"

    Returns the new list_id and list_name.

    Underlying API: POST /contacts/v1/lists
    """
    raise NotImplementedError(
        "Replace with real HubSpot call: POST /contacts/v1/lists\n"
        f"body: {{name: {campaign_name!r}, dynamic: false}}"
    )


def add_contacts_to_list(list_id: str, contacts: list[dict[str, Any]]) -> dict[str, Any]:
    """Add upserted contacts to the campaign list in chunks of ADD_TO_LIST_BATCH_SIZE.

    Requires HubSpot contact IDs (from batch_upsert_contacts) and the list_id
    (from create_campaign_list).

    Underlying API: POST /contacts/v1/lists/{listId}/add
    """
    if not contacts:
        return {"added_count": 0}
    raise NotImplementedError(
        "Replace with real HubSpot call: POST /contacts/v1/lists/{listId}/add\n"
        f"Chunk contact IDs into batches of {ADD_TO_LIST_BATCH_SIZE}."
    )


def save_sales_template(
    name: str,
    subject: str,
    body_html: str,
    template_id: str | None = None,
) -> dict[str, Any]:
    """Create a new or update an existing HubSpot sales email template.

    Uses the internal HubSpot sales templates API (not the CMS Design Manager).
    Requires browser session cookies: csrf.app, hubspotapi, hubspotapi-csrf,
    hubspotapi-strict, hubspotapi-lax. Cookies rotate periodically.

    If template_id is provided, updates the existing template.
    Otherwise creates a new one.

    Template changes do NOT retroactively affect contacts already enrolled in
    a sequence — they use a snapshot from enrollment time.

    Underlying tools: hubspot_create_sales_template / hubspot_update_sales_template
    """
    raise NotImplementedError(
        "Replace with real HubSpot internal API call.\n"
        "Use hubspot_create_sales_template (new) or hubspot_update_sales_template (edit)."
    )
