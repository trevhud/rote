"""
Extracted from: examples/bdr-outreach/skill/references/hubspot-operations.md
Lines 48–112 — the three MANDATORY exclusion checks before enrollment.

These were prose-only MANDATORY checks in the source skill. Moving them to
code makes them impossible to skip: the workflow calls each function as an
unconditional activity regardless of LLM trajectory or prompt drift.

Underlying vendor APIs:
  hubspot_search_lists              → GET  /crm/v3/lists/search
  hubspot_get_contact_list_memberships → GET /crm/v3/lists/contacts/{contactId}/memberships
  hubspot_get_contact_emails        → GET  /engagements/v1/engagements/associated/CONTACT/{id}/paged
  hubspot_get_contact_enrollment    → GET  /automation/v4/enrollment/{contactId}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Days to look back for recent outbound email — source: hubspot-operations.md line 65
RECENT_EMAIL_DAYS: int = 30

#: Name of the do-not-contact list — source: hubspot-operations.md lines 53–54
DNC_LIST_QUERY: str = "BDR do not contact"

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

HubSpotContact = dict[str, Any]


@dataclass
class ExclusionResult:
    contact_id: str
    contact_email: str
    excluded: bool
    reason: str | None = None  # "dnc" | "recently_emailed" | "active_sequence" | None


@dataclass
class ExclusionReport:
    passed: list[ExclusionResult] = field(default_factory=list)
    excluded: list[ExclusionResult] = field(default_factory=list)

    @property
    def dnc_count(self) -> int:
        return sum(1 for r in self.excluded if r.reason == "dnc")

    @property
    def recently_emailed_count(self) -> int:
        return sum(1 for r in self.excluded if r.reason == "recently_emailed")

    @property
    def active_sequence_count(self) -> int:
        return sum(1 for r in self.excluded if r.reason == "active_sequence")


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


def resolve_dnc_list_id(search_lists_fn: Any) -> str:
    """Look up the HubSpot 'BDR do not contact' list ID.

    Source: hubspot-operations.md lines 53–57.
    Called once per campaign run and cached for the duration of Phase 5.
    """
    results = search_lists_fn(query=DNC_LIST_QUERY)
    if not results:
        raise ValueError(
            f"Could not find HubSpot list matching '{DNC_LIST_QUERY}'. "
            "Verify the list exists before running exclusion checks."
        )
    # Return the first matching list ID
    return str(results[0]["listId"])


# ---------------------------------------------------------------------------
# Check 1 — Do-Not-Contact list (MANDATORY)
# ---------------------------------------------------------------------------


def check_do_not_contact(
    contact: HubSpotContact,
    dnc_list_id: str,
    get_memberships_fn: Any,
) -> ExclusionResult:
    """Check whether a contact is on the BDR do-not-contact list.

    Source: hubspot-operations.md lines 51–60.
    MANDATORY — must run before any enrollment recommendation.

    Underlying API:
        hubspot_get_contact_list_memberships
        → GET /crm/v3/lists/contacts/{contactId}/memberships
    """
    contact_id = str(contact["id"])
    memberships = get_memberships_fn(contactId=contact_id)
    list_ids = [str(m.get("listId", "")) for m in (memberships or [])]

    if dnc_list_id in list_ids:
        return ExclusionResult(
            contact_id=contact_id,
            contact_email=contact.get("email", ""),
            excluded=True,
            reason="dnc",
        )
    return ExclusionResult(
        contact_id=contact_id,
        contact_email=contact.get("email", ""),
        excluded=False,
    )


# ---------------------------------------------------------------------------
# Check 2 — Recently emailed (MANDATORY, 30-day window)
# ---------------------------------------------------------------------------


def check_recently_emailed(
    contact: HubSpotContact,
    get_emails_fn: Any,
    days_back: int = RECENT_EMAIL_DAYS,
) -> ExclusionResult:
    """Check whether a contact received an outbound email in the last 30 days.

    Source: hubspot-operations.md lines 62–70.
    MANDATORY — must run before any enrollment recommendation.

    Underlying API:
        hubspot_get_contact_emails (daysBack=30, direction=OUTBOUND)
        → GET /engagements/v1/engagements/associated/CONTACT/{id}/paged
    """
    contact_id = str(contact["id"])
    result = get_emails_fn(
        contactId=contact_id,
        daysBack=days_back,
        direction="OUTBOUND",
    )
    was_emailed = result.get("wasEmailedInPeriod", False)

    return ExclusionResult(
        contact_id=contact_id,
        contact_email=contact.get("email", ""),
        excluded=bool(was_emailed),
        reason="recently_emailed" if was_emailed else None,
    )


# ---------------------------------------------------------------------------
# Check 3 — Active sequence enrollment (MANDATORY)
# ---------------------------------------------------------------------------


def check_active_sequence(
    contact: HubSpotContact,
    get_enrollment_fn: Any,
) -> ExclusionResult:
    """Check whether a contact is already enrolled in a HubSpot sequence.

    Source: hubspot-operations.md lines 72–79.
    MANDATORY — a contact can only be in one sequence at a time.

    Underlying API:
        hubspot_get_contact_enrollment
        → GET /automation/v4/enrollment/{contactId}
    """
    contact_id = str(contact["id"])
    result = get_enrollment_fn(contactId=contact_id)
    is_enrolled = result.get("isEnrolled", False)

    return ExclusionResult(
        contact_id=contact_id,
        contact_email=contact.get("email", ""),
        excluded=bool(is_enrolled),
        reason="active_sequence" if is_enrolled else None,
    )


# ---------------------------------------------------------------------------
# Aggregator — run all three checks for a contact list
# ---------------------------------------------------------------------------


def run_all_exclusion_checks(
    contacts: list[HubSpotContact],
    dnc_list_id: str,
    get_memberships_fn: Any,
    get_emails_fn: Any,
    get_enrollment_fn: Any,
) -> ExclusionReport:
    """Run all three MANDATORY exclusion checks for every contact.

    Stops at the first failing check per contact (no need to check sequence
    if already on DNC list). Returns an ExclusionReport with passed/excluded
    partitions for use in generate_pre_enrollment_report().
    """
    report = ExclusionReport()

    for contact in contacts:
        dnc_result = check_do_not_contact(contact, dnc_list_id, get_memberships_fn)
        if dnc_result.excluded:
            report.excluded.append(dnc_result)
            continue

        email_result = check_recently_emailed(contact, get_emails_fn)
        if email_result.excluded:
            report.excluded.append(email_result)
            continue

        seq_result = check_active_sequence(contact, get_enrollment_fn)
        if seq_result.excluded:
            report.excluded.append(seq_result)
            continue

        report.passed.append(dnc_result)  # passed all checks

    return report
