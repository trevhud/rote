"""Typed signature for the personalize_email llm_judge node.

Generates a personalized opening line and TA-relevant callout for each
vetted contact's outreach email. The structural boilerplate comes from the
base template in references/email-templates.md; only the personalized
portions are LLM-generated.

IMPORTANT: Acme experience claims in the output MUST be grounded in the
acme_experience input field only — never fabricated. The prompt enforces this.

Source: references/email-templates.md
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict


class CampaignType(str, Enum):
    DRUG_SPECIFIC = "drug-specific"
    CONDITION_SPECIFIC = "condition-specific"
    GENERAL_CAPABILITIES = "general-capabilities"


class ContactTier(str, Enum):
    IDEAL = "ideal"
    STRONG = "strong"
    GOOD = "good"


class EmploymentEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    company: str
    title: str
    start_year: int | None = None
    end_year: int | None = None


class EnrichedContact(BaseModel):
    model_config = ConfigDict(extra="forbid")
    zoominfo_id: str
    first_name: str
    last_name: str
    job_title: str
    company_name: str
    email: str | None = None
    accuracy_score: int = 0
    linkedin_url: str | None = None
    employment_history: list[EmploymentEntry] = []


class VettedContact(BaseModel):
    model_config = ConfigDict(extra="forbid")
    contact: EnrichedContact
    tier: ContactTier
    relevance_evidence: str


class IntelBrief(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pipeline_summary: str
    rwe_signals: list[str] = []
    key_programs: list[str] = []
    prior_interactions: str = ""
    relevant_experience: str = ""
    messaging_angles: list[str] = []


class PersonalizeEmailInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    contact: VettedContact
    brief_campaign_type: CampaignType
    brief_drug_brand: str
    brief_condition: str
    brief_therapeutic_area: str
    intel: IntelBrief
    acme_experience: str  # validated TA experience; all claims must come from here


class PersonalizeEmailOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    opening_line: str     # ≤ 2 sentences; specific observation about their work
    ta_callout: str       # 1 sentence connecting Acme experience to their TA


class PersonalizeEmail:
    """Generate personalized opening line and TA callout for a BDR outreach email.

    Base template structure (from references/email-templates.md):
    - Opening: specific observation about their work (never generic)
    - Body: boilerplate about Acme's RWE/HEOR capabilities (not generated)
    - Callout: grounded connection to their TA using acme_experience only
    - CTA: standard low-friction ask (not generated)

    Target total email length: 150-200 words.
    """

    async def forward(self, inputs: PersonalizeEmailInput) -> PersonalizeEmailOutput:
        raise NotImplementedError(
            "Replace with DSPy or BAML call.\n"
            "Use the prompt and schemas from pipeline.yaml node personalize_email.signature_spec.\n"
            "Return a PersonalizeEmailOutput with opening_line and ta_callout.\n"
            "CRITICAL: all Acme experience claims must come from inputs.acme_experience only."
        )
