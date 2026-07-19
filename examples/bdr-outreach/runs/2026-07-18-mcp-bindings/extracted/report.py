"""Report generation — pure string formatting functions.

All functions produce deterministic Markdown from structured inputs.
No LLM involved. These are fixed templates lifted directly from the
source skill's reference files.

Sources:
- build_contact_table:              references/lead-generation.md lines 124–133
- generate_pre_enrollment_report:   references/hubspot-operations.md lines 86–103
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


@dataclass
class DiscardSummary:
    total_discarded: int
    by_reason: dict[str, int] = field(default_factory=dict)


def build_contact_table(
    vetted_contacts: list[dict],
    drug_name: str,
    condition: str,
    discard_summary: DiscardSummary,
    total_searched: int,
) -> str:
    """Render the ranked contact table, campaign narrative, and discard summary.

    Format sourced from references/lead-generation.md lines 124–133.
    Contacts are sorted by tier: ideal first, then strong, then good.
    Every contact in this table has already passed the full vetting criteria.
    """
    tier_order = {"ideal": 0, "strong": 1, "good": 2}
    sorted_contacts = sorted(
        vetted_contacts,
        key=lambda c: tier_order.get(c.get("tier", "good"), 2),
    )

    header = (
        f"### Priority Contacts — {drug_name} / {condition} Campaign\n\n"
        "| # | Name | Title | Company | Email | LinkedIn | Accuracy | Priority | Relevance Evidence |\n"
        "| - | ---- | ----- | ------- | ----- | -------- | -------- | -------- | ------------------ |\n"
    )
    rows = []
    for i, c in enumerate(sorted_contacts, start=1):
        contact = c.get("contact", c)
        rows.append(
            f"| {i} "
            f"| {contact.get('first_name', '')} {contact.get('last_name', '')} "
            f"| {contact.get('job_title', '')} "
            f"| {contact.get('company_name', '')} "
            f"| {contact.get('email', '')} "
            f"| {contact.get('linkedin_url', '')} "
            f"| {contact.get('accuracy_score', '')} "
            f"| {c.get('tier', '').capitalize()} "
            f"| {c.get('relevance_evidence', '')} |"
        )

    discard_reasons_str = ", ".join(
        f"{count} {reason.replace('_', ' ')}"
        for reason, count in sorted(discard_summary.by_reason.items())
    )
    discard_line = (
        f"\n**Discarded Contacts Summary**: {discard_summary.total_discarded} discarded"
        + (f": {discard_reasons_str}" if discard_reasons_str else "")
        + "\n"
    )

    return header + "\n".join(rows) + discard_line


def generate_pre_enrollment_report(
    campaign_name: str,
    vetted_count: int,
    passed_contacts: list[dict],
    exclusions: list[dict],
    template_ids: list[str],
) -> str:
    """Render the pre-enrollment Markdown report for the BDR.

    Template format sourced verbatim from references/hubspot-operations.md
    lines 86–103. Aggregates exclusion results from all three MANDATORY checks
    into counts (DNC, recently emailed, active sequence).

    Args:
        campaign_name: Display name (e.g., "Orladeyo HAE Campaign").
        vetted_count: Total contacts that passed the LLM vetting loop.
        passed_contacts: Contacts that cleared all three exclusion checks.
        exclusions: ExclusionRecord dicts with keys: contact_id, reason.
        template_ids: HubSpot template IDs created by create_sales_template.

    Returns:
        Markdown report string ready to display to the BDR.
    """
    exclusion_counts = Counter(rec.get("reason", "unknown") for rec in exclusions)
    dnc = exclusion_counts.get("do_not_contact", 0)
    recent = exclusion_counts.get("recently_emailed", 0)
    active = exclusion_counts.get("active_sequence", 0)

    lines = [
        f"# Pre-Enrollment Report — {campaign_name}",
        "",
        f"**Total contacts vetted:** {vetted_count}",
        f"**Ready to enroll:** {len(passed_contacts)}",
        "",
        "## Exclusions",
        "",
        f"- On do-not-contact list: {dnc}",
        f"- Emailed in last 30 days: {recent}",
        f"- Already in active sequence: {active}",
        f"- **Total excluded:** {len(exclusions)}",
        "",
        "## Tier breakdown (ready to enroll)",
        "",
    ]
    tier_counts = Counter(c.get("tier", "unknown") for c in passed_contacts)
    for tier in ("ideal", "strong", "good"):
        lines.append(f"- {tier.capitalize()}: {tier_counts.get(tier, 0)}")

    if template_ids:
        lines.extend([
            "",
            "## Sales templates created",
            "",
            *(f"- {tid}" for tid in template_ids),
        ])

    lines.extend([
        "",
        "## Next step",
        "",
        "BDR should enroll the contacts above **manually in the HubSpot UI**. "
        "API enrollment sends emails immediately with no verification step "
        "and cannot be safely automated.",
    ])
    return "\n".join(lines)
