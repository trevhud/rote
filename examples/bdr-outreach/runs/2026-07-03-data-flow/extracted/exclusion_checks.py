"""MANDATORY exclusion checks for the BDR outreach pipeline.

All three functions in this module correspond to nodes marked mandatory: true
in the IR. Moving these checks from prose enforcement into code makes it
impossible to skip them accidentally — the Temporal adapter emits them as
unconditional activities the workflow always calls before the pre-enrollment
report is generated.

Source: references/hubspot-operations.md, "Exclusion Checks (MANDATORY)" section.

MCP tools originally used:
- hubspot_search_lists + hubspot_get_contact_list_memberships  (DNC check)
- hubspot_get_contact_emails with daysBack=30, direction=OUTBOUND (recent check)
- hubspot_get_contact_enrollment                               (sequence check)

Underlying vendor APIs:
- GET /contacts/v1/lists/search?query=BDR+do+not+contact
- GET /contacts/v1/contact/id/{contactId}/lists
- GET /engagements/v1/engagements/associated/contact/{contactId}/paged
- GET /automation/v4/sequences/enrollments/contacts/{contactId}
"""

from __future__ import annotations

from typing import Any


# ── Constants (from references/hubspot-operations.md) ─────────────────────────

RECENT_EMAIL_DAYS: int = 30         # "recently emailed" window; source prose says "30 days"
DNC_LIST_QUERY: str = "BDR do not contact"  # search term to locate the DNC list in HubSpot


def check_do_not_contact(
    contacts: list[dict[str, Any]],
    dnc_list_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """MANDATORY: Check each contact against the "BDR do not contact" HubSpot list.

    Procedure (hubspot-operations.md lines 51–60):
    1. (Setup, once per campaign): hubspot_search_lists(query=DNC_LIST_QUERY) → dnc_list_id.
       Pass this as the dnc_list_id argument; the workflow obtains it during setup.
    2. Per contact: hubspot_get_contact_list_memberships → check for dnc_list_id.
    3. Any match → excluded.

    Args:
        contacts: HubSpot-resident contacts (with hubspot_id) to check.
        dnc_list_id: The "BDR do not contact" list ID (deployment configuration —
            not pipeline data flow, so left unbound in pipeline.yaml inputs:).

    Returns:
        Tuple of (passed_contacts, excluded_records).
        excluded_records dicts: {contact_id, reason: "do_not_contact"}.

    Raises:
        NotImplementedError: stub.
    """
    raise NotImplementedError(
        "Replace with real HubSpot calls.\n"
        f"Setup: hubspot_search_lists(query={DNC_LIST_QUERY!r}) → dnc_list_id\n"
        "Per contact: hubspot_get_contact_list_memberships → check for dnc_list_id"
    )


def check_recently_emailed(
    contacts: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """MANDATORY: Check each contact for outbound emails in the last RECENT_EMAIL_DAYS days.

    Procedure (hubspot-operations.md lines 62–70):
    1. Per contact: hubspot_get_contact_emails(daysBack=RECENT_EMAIL_DAYS, direction=OUTBOUND).
    2. If wasEmailedInPeriod is true → excluded.
    3. Log which contacts were skipped and why.

    The RECENT_EMAIL_DAYS constant (30) comes from the source skill's prose and
    is enforced here in code so it cannot drift with prompt edits.

    Returns:
        Tuple of (passed_contacts, excluded_records).
        excluded_records dicts: {contact_id, reason: "recently_emailed"}.

    Raises:
        NotImplementedError: stub.
    """
    raise NotImplementedError(
        f"Replace with real HubSpot call.\n"
        f"Per contact: hubspot_get_contact_emails("
        f"daysBack={RECENT_EMAIL_DAYS}, direction='OUTBOUND')"
    )


def check_active_sequence(
    contacts: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """MANDATORY: Check each contact for active HubSpot sequence enrollment.

    A contact can only be enrolled in one HubSpot sequence at a time. This is
    the easiest exclusion to miss accidentally — the failure mode is silent
    (the second sequence enrollment fails at enrollment time with no trace to
    the root cause).

    Procedure (hubspot-operations.md lines 72–79):
    1. Per contact: hubspot_get_contact_enrollment.
    2. If isEnrolled is true → excluded.

    Returns:
        Tuple of (passed_contacts, excluded_records).
        excluded_records dicts: {contact_id, reason: "active_sequence"}.

    Raises:
        NotImplementedError: stub.
    """
    raise NotImplementedError(
        "Replace with real HubSpot call.\n"
        "Per contact: hubspot_get_contact_enrollment → check isEnrolled"
    )
