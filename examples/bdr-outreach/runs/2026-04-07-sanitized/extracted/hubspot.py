"""
Extracted from: examples/bdr-outreach/skill/references/hubspot-operations.md
                examples/bdr-outreach/skill/SKILL.md (Limits table)

HubSpot CRM operations — wraps batch upsert, list creation, and list membership
calls with the fixed batch sizes documented in the skill.

Underlying vendor APIs:
  POST /crm/v3/objects/contacts/batch/upsert   (hubspot_batch_upsert_contacts)
  POST /crm/v3/lists                           (hubspot_create_list)
  PUT  /crm/v3/lists/{listId}/memberships/add  (hubspot_add_contacts_to_list)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: HubSpot batch upsert limit — source: SKILL.md limits table, hubspot-operations.md line 7
UPSERT_BATCH_SIZE: int = 100

#: HubSpot add-to-list limit — source: SKILL.md limits table, hubspot-operations.md line 21
LIST_ADD_BATCH_SIZE: int = 250

#: List naming convention — source: hubspot-operations.md lines 25–29
LIST_NAME_TEMPLATE: str = "{campaign_name}"

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

HubSpotContact = dict[str, Any]


@dataclass
class UpsertResult:
    upserted: list[HubSpotContact]
    created_count: int
    updated_count: int


@dataclass
class ListCreateResult:
    list_id: str
    list_name: str


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def _chunk(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def batch_upsert_contacts(
    contacts: list[HubSpotContact],
    tool_fn: Any,  # callable: (contacts) -> UpsertResult
    batch_size: int = UPSERT_BATCH_SIZE,
) -> UpsertResult:
    """Batch upsert contacts to HubSpot, chunking at the API limit.

    Source: hubspot-operations.md lines 7, SKILL.md limits table.
    Underlying API: POST /crm/v3/objects/contacts/batch/upsert

    Args:
        contacts:   Full list of contacts to upsert (create or update by email).
        tool_fn:    Callable wrapping hubspot_batch_upsert_contacts MCP tool.
        batch_size: Max contacts per call (HubSpot limit = 100).

    Returns:
        Aggregated UpsertResult across all batches.
    """
    if batch_size > UPSERT_BATCH_SIZE:
        raise ValueError(
            f"batch_size {batch_size} exceeds HubSpot upsert limit of {UPSERT_BATCH_SIZE}"
        )

    all_upserted: list[HubSpotContact] = []
    total_created = 0
    total_updated = 0

    for batch in _chunk(contacts, batch_size):
        result = tool_fn(contacts=batch)
        all_upserted.extend(result.upserted)
        total_created += result.created_count
        total_updated += result.updated_count

    return UpsertResult(
        upserted=all_upserted,
        created_count=total_created,
        updated_count=total_updated,
    )


def create_campaign_list(
    campaign_name: str,
    tool_fn: Any,  # callable: (name, listType) -> ListCreateResult
) -> ListCreateResult:
    """Create a static HubSpot list for the campaign.

    Source: hubspot-operations.md lines 25–29.
    Underlying API: POST /crm/v3/lists

    The list name follows the documented naming convention:
        "[Campaign] - [Qualifier]"   e.g. "Rare Disease Campaign - Q1 2026"
    """
    list_name = LIST_NAME_TEMPLATE.format(campaign_name=campaign_name)
    return tool_fn(name=list_name, listType="STATIC")


def add_contacts_to_list(
    list_id: str,
    contacts: list[HubSpotContact],
    tool_fn: Any,  # callable: (listId, contactIds) -> None
    batch_size: int = LIST_ADD_BATCH_SIZE,
) -> None:
    """Add contacts to a static HubSpot list, chunking at the API limit.

    Source: SKILL.md limits table ("Add to list: 250 contacts per call").
    Underlying API: PUT /crm/v3/lists/{listId}/memberships/add

    Args:
        list_id:    HubSpot list ID returned by create_campaign_list.
        contacts:   Upserted contacts with HubSpot contact IDs.
        tool_fn:    Callable wrapping hubspot_add_contacts_to_list MCP tool.
        batch_size: Max contacts per call (HubSpot limit = 250).
    """
    if batch_size > LIST_ADD_BATCH_SIZE:
        raise ValueError(
            f"batch_size {batch_size} exceeds HubSpot add-to-list limit of {LIST_ADD_BATCH_SIZE}"
        )

    contact_ids = [c["id"] for c in contacts if "id" in c]
    for batch in _chunk(contact_ids, batch_size):
        tool_fn(listId=list_id, contactIds=batch)
