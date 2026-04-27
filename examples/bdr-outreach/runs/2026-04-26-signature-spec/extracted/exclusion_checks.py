"""MANDATORY exclusion checks for the BDR outreach pipeline.

All three functions in this module are marked mandatory in the IR. Moving
these checks from prose enforcement into code makes it impossible to skip
them accidentally — the workflow always calls them in order before the
pre-enrollment report is generated.

Source: references/hubspot-operations.md, "Exclusion Checks (MANDATORY)" section.
"""

from __future__ import annotations

from typing import Any


# ── Constants (from references/hubspot-operations.md) ──

RECENT_EMAIL_DAYS = 30       # lookback window: "recently emailed" means within 30 days
DNC_LIST_QUERY = "BDR do not contact"  # search term to locate the DNC list


def check_do_not_contact(
    contacts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """MANDATORY: Check each contact against the "BDR do not contact" HubSpot list.

    Procedure (from hubspot-operations.md lines 51–60):
    1. Call hubspot_search_lists(query=DNC_LIST_QUERY) — once per invocation.
    2. For each contact: call hubspot_get_contact_list_memberships.
    3. If the DNC list_id appears in their memberships → excluded.

    Returns a list of ExclusionResult dicts, one per contact:
      {contact_id, excluded: bool, reason: "dnc" | None}

    Underlying API:
    - GET /contacts/v1/lists/search?query=BDR+do+not+contact
    - GET /contacts/v1/contact/id/{contactId}/lists
    """
    raise NotImplementedError(
        "Replace with real HubSpot calls.\n"
        "Step 1: hubspot_search_lists(query='BDR do not contact') → dnc_list_id\n"
        "Step 2: For each contact, hubspot_get_contact_list_memberships → check dnc_list_id"
    )


def check_recently_emailed(
    contacts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """MANDATORY: Check each contact for outbound emails in the last RECENT_EMAIL_DAYS days.

    Procedure (from hubspot-operations.md lines 62–70):
    1. For each contact: call hubspot_get_contact_emails(daysBack=RECENT_EMAIL_DAYS, direction=OUTBOUND).
    2. If wasEmailedInPeriod is true → excluded.

    Returns a list of ExclusionResult dicts, one per contact:
      {contact_id, excluded: bool, reason: "recently_emailed" | None}

    Underlying API: GET /engagements/v1/engagements/associated/contact/{contactId}/paged
    """
    raise NotImplementedError(
        f"Replace with real HubSpot call.\n"
        f"For each contact: hubspot_get_contact_emails(daysBack={RECENT_EMAIL_DAYS}, direction=OUTBOUND)"
    )


def check_active_sequence(
    contacts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """MANDATORY: Check each contact for active HubSpot sequence enrollment.

    A contact can only be enrolled in one sequence at a time.

    Procedure (from hubspot-operations.md lines 72–79):
    1. For each contact: call hubspot_get_contact_enrollment.
    2. If isEnrolled is true → excluded.

    Returns a list of ExclusionResult dicts, one per contact:
      {contact_id, excluded: bool, reason: "active_sequence" | None}

    Underlying API: GET /automation/v4/sequences/enrollments/contacts/{contactId}
    """
    raise NotImplementedError(
        "Replace with real HubSpot call.\n"
        "For each contact: hubspot_get_contact_enrollment → check isEnrolled"
    )
