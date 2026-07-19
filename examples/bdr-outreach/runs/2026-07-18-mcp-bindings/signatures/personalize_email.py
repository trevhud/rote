"""Typed signature for the personalize_email llm_judge node.

Generates a personalized opening line and TA-relevant callout for each
vetted contact's outreach email. The structural boilerplate comes from the
base template in references/email-templates.md; only the two personalized
slots are LLM-generated, keeping per-contact token cost low.

IMPORTANT: Acme experience claims in the output MUST be grounded in the
acme_experience input field only — never fabricated. The prompt enforces
this constraint and the eval set tests for violations.

Source: references/email-templates.md
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict


# ── Enums ─────────────────────────────────────────────────────────────────────

class CampaignType(str, Enum):
    DRUG_SPECIFIC = "drug-specific"
    CONDITION_SPECIFIC = "condition-specific"
    GENERAL_CAPABILITIES = "general-capabilities"


class ContactTier(str, Enum):
    IDEAL = "ideal"
    STRONG = "strong"
    GOOD = "good"


# ── Input / Output models ─────────────────────────────────────────────────────

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
    intel: IntelBrief
    campaign_type: CampaignType
    drug_brand: str
    condition_full: str
    therapeutic_area: str
    acme_experience: str  # validated TA experience; all claims must come only from here


class PersonalizeEmailOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    opening_line: str    # ≤ 2 sentences; specific observation about their work
    ta_callout: str      # 1 sentence grounding Acme's TA experience; no fabrication


# ── Signature class ───────────────────────────────────────────────────────────

class PersonalizeEmail:
    """Generate a personalized opening line and TA callout for BDR outreach.

    Base template structure (from references/email-templates.md):
      - Opening:  specific observation about their work (this node generates it)
      - Body:     boilerplate about Acme's RWE/HEOR capabilities (NOT generated)
      - TA callout: grounded connection to their TA using acme_experience only
      - CTA:      standard low-friction ask (NOT generated)

    Target total email length: 150–200 words.
    """

    async def forward(self, inputs: PersonalizeEmailInput) -> PersonalizeEmailOutput:
        raise NotImplementedError(
            "Replace with DSPy or BAML call.\n"
            "Use the prompt and schemas from pipeline.yaml node personalize_email.signature_spec.\n"
            "Return a PersonalizeEmailOutput with opening_line and ta_callout.\n"
            "CRITICAL: all Acme experience claims must come from inputs.acme_experience only."
        )
