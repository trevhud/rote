"""Report generation functions for the BDR outreach pipeline.

Both functions produce deterministic Markdown from structured inputs.
No LLM is involved — these are pure string templates lifted verbatim from
the reference files.

Sources:
- generate_pre_enrollment_report: references/hubspot-operations.md lines 86–103
- build_contact_table: references/lead-generation.md lines 124–133
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DiscardSummary:
    total_discarded: int
    by_reason: dict[str, int] = field(default_factory=dict)


@dataclass
class ExclusionCounts:
    dnc: int = 0
    recently_emailed: int = 0
    active_sequence: int = 0


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
            f"| {i} | {contact.get('first_name', '')} {contact.get('last_name', '')} "
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
    total_enriched: int,
    already_in_hubspot: int,
    new_contacts: int,
    exclusion_results: list[dict],
    template_ids: list[str],
) -> tuple[str, int]:
    """Render the pre-enrollment Markdown report and return the ready-to-enroll count.

    Template sourced verbatim from references/hubspot-operations.md lines 86–103.
    Returns (report_markdown, ready_to_enroll_count).
    """
    counts = ExclusionCounts()
    for result in exclusion_results:
        reason = result.get("reason")
        if reason == "dnc":
            counts.dnc += 1
        elif reason == "recently_emailed":
            counts.recently_emailed += 1
        elif reason == "active_sequence":
            counts.active_sequence += 1

    total_excluded = counts.dnc + counts.recently_emailed + counts.active_sequence
    ready = total_enriched - total_excluded

    report = f"""Campaign: {campaign_name}
Total contacts enriched: {total_enriched}
Already in HubSpot: {already_in_hubspot} (will update)
New contacts: {new_contacts}

Exclusions:
  - On "do not contact" list: {counts.dnc}
  - Emailed in last 30 days: {counts.recently_emailed}
  - Already in active sequence: {counts.active_sequence}

Ready to enroll: {ready}
-> BDR should enroll these contacts manually in HubSpot UI
"""
    if template_ids:
        report += f"\nEmail templates created: {', '.join(template_ids)}\n"

    return report, ready
