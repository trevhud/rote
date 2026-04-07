"""
LLM Judge Signature: PersonalizeEmail

Source rubric: examples/bdr-outreach/skill/references/email-templates.md
Node: personalize_email (llm_judge, fan_out=true)

This signature drafts a personalized outreach email for a single vetted contact
using the contact's enrichment data, the campaign brief, and the intel brief.

The output is bounded:
  - subject: one-line string
  - body_html: HubSpot-formatted HTML, < 200 words
  - campaign_type_used: which template structure was applied

The base template structure (email-templates.md lines 63–77) is injected into
the LLM's system prompt — the model only fills in the personalization variables,
not the structural boilerplate.

Usage:
    judge = PersonalizeEmail()
    result = await judge.forward(PersonalizeEmailInput(
        contact=vetted_contact,
        brief=campaign_brief,
        intel=intel_brief,
        acme_experience=validated_experience,
    ))
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class CampaignType(str, Enum):
    """Determines which template variation to use.

    Source: email-templates.md lines 58–60, lead-generation.md lines 137–143.
    """

    DRUG_SPECIFIC = "drug-specific"
    CONDITION_SPECIFIC = "condition-specific"
    GENERAL_CAPABILITIES = "general-capabilities"


# ---------------------------------------------------------------------------
# Input model
# ---------------------------------------------------------------------------


class VettedContactSummary(BaseModel):
    """Subset of contact data needed for email personalization."""

    model_config = ConfigDict(extra="forbid")

    first_name: str
    last_name: str
    job_title: str
    company_name: str
    email: str
    tier: str  # ideal | strong | good
    relevance_evidence: str
    employment_history_summary: str  # prose summary from vet_contact output


class PersonalizeEmailInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contact: VettedContactSummary
    brief: "CampaignBrief"  # forward ref — same type as in vet_contact.py
    intel: "IntelBrief"     # working context from target_research
    acme_experience: str    # validated from internal sources only (NOT fabricated)
    sequence_step: int = 1  # which step in the sequence (1 = initial outreach)


class CampaignBrief(BaseModel):
    model_config = ConfigDict(extra="forbid")

    drug_brand: str
    drug_generic: str
    condition_full: str
    condition_acronym: str
    therapeutic_area: list[str]
    manufacturer: str
    campaign_type: CampaignType
    job_focus: str | None = None


class IntelBrief(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pipeline_summary: str
    rwe_signals: str
    key_programs: list[str] = []
    messaging_angles: list[str] = []


# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------


class PersonalizeEmailOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject: str
    body_html: str  # HubSpot-formatted HTML, < 200 words
    campaign_type_used: CampaignType
    personalization_hook: str  # 1-sentence summary of the leading observation used


# ---------------------------------------------------------------------------
# Signature class
# ---------------------------------------------------------------------------

#: Base template injected as system context — source: email-templates.md lines 63–77
BASE_TEMPLATE_PROSE = """
Hi [firstname],

I am reaching out to support [company]'s work in [condition] with real-world
evidence generation.

Acme conducts research without the inefficiencies inherent in traditional
research methods. Utilizing world-class AI, we recruit patients directly into
studies, curate medical records from everywhere they receive care, and enable
patients to participate in research virtually. This approach enables us to
rapidly deploy research studies, achieve high patient retention year-over-year,
and deliver high quality data compared to traditional approaches.

Would you like to learn more about our capabilities? If so, do you have
availability in the next two weeks?

Best wishes,
[sender name]
"""

#: Key principles from email-templates.md lines 79–83
EMAIL_PRINCIPLES = """
- Lead with relevance to their specific work, not a generic intro about Acme
- Keep it under 200 words total
- End with a clear, low-friction CTA (e.g. "Would a 20-minute call make sense?")
- Personalize using enrichment data (title, company pipeline, career history)
- For drug-specific: reference the drug directly
- For condition-specific: reference the condition and Acme's TA experience
- For general-capabilities: broader RWE/registry intro, calling out their TA specifically
- Do NOT fabricate or assume Acme experience — use only validated acme_experience input
"""


class PersonalizeEmail:
    """LLM judge for drafting personalized BDR outreach emails.

    The template structure and writing principles are injected as system context.
    The model's job is to fill in the personalization variables (opening hook,
    company-specific references, CTA framing) — not to invent the structure.
    """

    async def forward(self, inputs: PersonalizeEmailInput) -> PersonalizeEmailOutput:
        # No pre-filters needed here — all contacts reaching this node have
        # already passed vet_contact and exclusion_checks.
        raise NotImplementedError(
            "LLM dispatch not yet implemented. "
            "Wire up DSPy Predict or BAML function call here. "
            "System prompt should include BASE_TEMPLATE_PROSE, EMAIL_PRINCIPLES, "
            "and the validated acme_experience from the input. "
            "IMPORTANT: the model must use inputs.acme_experience verbatim for "
            "Acme capability claims — never extrapolate beyond what is provided."
        )
