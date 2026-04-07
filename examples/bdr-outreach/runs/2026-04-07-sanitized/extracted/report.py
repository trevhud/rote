"""
Extracted from: examples/bdr-outreach/skill/references/hubspot-operations.md (lines 86–103)
                examples/bdr-outreach/skill/references/lead-generation.md (lines 124–133)

Fixed report templates — the LLM was generating these markdown strings on every
run by reading the template in the reference files. Extracting them here makes
the output format stable and eliminates the token cost of re-deriving it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

VettedContact = dict[str, Any]
ExclusionRecord = dict[str, Any]


@dataclass
class DiscardSummary:
    """Aggregate counts by discard reason for the narrative section."""

    indication_mismatch: int = 0
    msl_role: int = 0
    biomarker_discovery: int = 0
    translational_research: int = 0
    sales_commercial: int = 0
    ops_strategy: int = 0
    program_management: int = 0
    low_accuracy: int = 0
    no_valid_email: int = 0
    other: int = 0

    @property
    def total(self) -> int:
        return (
            self.indication_mismatch
            + self.msl_role
            + self.biomarker_discovery
            + self.translational_research
            + self.sales_commercial
            + self.ops_strategy
            + self.program_management
            + self.low_accuracy
            + self.no_valid_email
            + self.other
        )

    def as_prose(self) -> str:
        """Return a compact discard summary string for the narrative section."""
        parts = []
        if self.indication_mismatch:
            parts.append(f"{self.indication_mismatch} indication mismatch")
        if self.msl_role:
            parts.append(f"{self.msl_role} MSL roles")
        if self.biomarker_discovery:
            parts.append(f"{self.biomarker_discovery} biomarker/translational")
        if self.translational_research:
            parts.append(f"{self.translational_research} translational research")
        if self.sales_commercial:
            parts.append(f"{self.sales_commercial} sales/commercial")
        if self.ops_strategy:
            parts.append(f"{self.ops_strategy} ops/strategy")
        if self.program_management:
            parts.append(f"{self.program_management} program management")
        if self.low_accuracy:
            parts.append(f"{self.low_accuracy} accuracy below 85")
        if self.no_valid_email:
            parts.append(f"{self.no_valid_email} no valid email")
        if self.other:
            parts.append(f"{self.other} other")
        return "; ".join(parts) if parts else "none"


# ---------------------------------------------------------------------------
# Contact table (Phase 2 output)
# ---------------------------------------------------------------------------


def build_contact_table(
    vetted_contacts: list[VettedContact],
    drug_name: str,
    condition: str,
    discard_summary: DiscardSummary,
    total_searched: int,
) -> str:
    """Render the ranked contact table with narrative and discard summary.

    Source: lead-generation.md lines 124–133.
    Output format is fixed — no LLM needed once the inputs are structured.
    """
    # Table header
    lines = [
        f"### Priority Contacts — {drug_name} / {condition} Campaign",
        "",
        "| # | Name | Title | Company | Email | LinkedIn | Accuracy | Priority | Relevance Evidence |",
        "| - | ---- | ----- | ------- | ----- | -------- | -------- | -------- | ------------------ |",
    ]

    tier_counts: dict[str, int] = {"ideal": 0, "strong": 0, "good": 0}

    for i, contact in enumerate(vetted_contacts, start=1):
        tier = contact.get("tier", "good")
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        linkedin = contact.get("linkedin", "")
        linkedin_cell = f"[link]({linkedin})" if linkedin else "—"
        lines.append(
            f"| {i} "
            f"| {contact.get('firstName', '')} {contact.get('lastName', '')} "
            f"| {contact.get('jobTitle', '')} "
            f"| {contact.get('companyName', '')} "
            f"| {contact.get('email', '')} "
            f"| {linkedin_cell} "
            f"| {contact.get('contactAccuracyScore', '')} "
            f"| {tier.capitalize()} "
            f"| {contact.get('relevance_evidence', '')} |"
        )

    # Tier breakdown for narrative
    tier_breakdown = ", ".join(
        f"{v} {k}" for k, v in tier_counts.items() if v > 0
    )

    lines += [
        "",
        f"**Total searched:** {total_searched}  "
        f"**Discarded:** {discard_summary.total} ({discard_summary.as_prose()})  "
        f"**Tiers:** {tier_breakdown}",
        "",
        "#### Discarded Contacts Summary",
        "",
        f"{discard_summary.total} discarded: {discard_summary.as_prose()}",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pre-enrollment report (Phase 5/7 output)
# ---------------------------------------------------------------------------


def generate_pre_enrollment_report(
    campaign_name: str,
    total_enriched: int,
    already_in_hubspot: int,
    new_contacts: int,
    dnc_excluded: int,
    recently_emailed_excluded: int,
    active_sequence_excluded: int,
    ready_to_enroll: int,
    template_ids: list[str] | None = None,
) -> str:
    """Render the pre-enrollment report for the BDR.

    Source: hubspot-operations.md lines 86–103 (verbatim template extraction).
    The BDR reads this report before manually enrolling contacts in HubSpot UI.
    """
    template_line = ""
    if template_ids:
        ids_str = ", ".join(template_ids)
        template_line = f"\nSequence templates: {ids_str}"

    return f"""Campaign: {campaign_name}
Total contacts enriched: {total_enriched}
Already in HubSpot: {already_in_hubspot} (will update)
New contacts: {new_contacts}

Exclusions:
  - On "do not contact" list: {dnc_excluded}
  - Emailed in last 30 days: {recently_emailed_excluded}
  - Already in active sequence: {active_sequence_excluded}
{template_line}
Ready to enroll: {ready_to_enroll}
-> BDR should enroll these contacts manually in HubSpot UI
"""
