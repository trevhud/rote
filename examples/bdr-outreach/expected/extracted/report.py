"""Pre-enrollment report generation — pure string formatting.

# MCP origin

The bdr-outreach skill generated this report inside the LLM loop in
Phase 7, formatting a fixed template with run-time counts. The skill's
prose included the exact format string already, which is the canonical
"why are we paying tokens to re-derive a string template" smell.

# Graduated form

Pure Python. No LLM, no MCP, no API calls. The function takes typed
inputs from upstream nodes and produces a markdown string. Cheap,
deterministic, and trivially testable.
"""

from __future__ import annotations

from collections import Counter

from ..types import ExclusionRecord, HubSpotContact


def generate_pre_enrollment_report(
    campaign_name: str,
    vetted_count: int,
    passed_contacts: list[HubSpotContact],
    exclusions: list[ExclusionRecord],
    template_ids: list[str],
) -> str:
    """Render the pre-enrollment report as Markdown.

    The format is fixed by the source skill's prose; this function
    encodes it once and replaces a per-run LLM call.
    """
    exclusion_counts = Counter(rec.reason for rec in exclusions)
    dnc = exclusion_counts.get("do_not_contact", 0)
    recent = exclusion_counts.get("recently_emailed", 0)
    active = exclusion_counts.get("active_sequence", 0)
    tier_counts = Counter(c.tier.value for c in passed_contacts)

    lines = [
        f"# Pre-Enrollment Report — {campaign_name}",
        "",
        f"**Vetted contacts:** {vetted_count}",
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
    for tier in ("ideal", "strong", "good"):
        lines.append(f"- {tier.capitalize()}: {tier_counts.get(tier, 0)}")
    lines.extend(
        [
            "",
            "## Sales templates created",
            "",
            *(f"- {tid}" for tid in template_ids),
            "",
            "## Next step",
            "",
            "BDR should enroll the contacts above manually in the HubSpot UI. "
            "API enrollment sends emails immediately with no verification step "
            "and is not safe to automate.",
        ]
    )
    return "\n".join(lines)
