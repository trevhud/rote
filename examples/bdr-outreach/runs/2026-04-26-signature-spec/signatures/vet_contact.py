"""Typed signature for the vet_contact llm_judge node.

Applies the BDR red-flags rubric to a single enriched contact and returns
a keep/discard decision with tier and discard reason.

Pre-filter in forward() short-circuits on two hard thresholds before calling
the LLM — saving tokens on ~20–40% of contacts and making hard rules
impossible to drift:
  - accuracy_score < MIN_ACCURACY_SCORE → discard:low_accuracy (no LLM call)
  - no valid email → discard:no_valid_email (no LLM call)

Source rubric: references/quality-and-vetting.md
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict

MIN_ACCURACY_SCORE: int = 85  # from references/quality-and-vetting.md, line 42


# ── Enums ──

class VetDecision(str, Enum):
    KEEP = "keep"
    DISCARD = "discard"


class ContactTier(str, Enum):
    IDEAL = "ideal"
    STRONG = "strong"
    GOOD = "good"


class DiscardReason(str, Enum):
    INDICATION_MISMATCH = "indication_mismatch"
    MSL_ROLE = "msl_role"
    BIOMARKER_DISCOVERY = "biomarker_discovery"
    TRANSLATIONAL = "translational"
    SALES_COMMERCIAL = "sales_commercial"
    OPS_STRATEGY = "ops_strategy"
    PROGRAM_MANAGEMENT = "program_management"
    LOW_ACCURACY = "low_accuracy"
    NO_VALID_EMAIL = "no_valid_email"
    OTHER = "other"


# ── Input / Output models ──

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
    phone: str | None = None
    mobile_phone: str | None = None
    accuracy_score: int = 0
    linkedin_url: str | None = None
    employment_history: list[EmploymentEntry] = []
    direct_phone_dnc: bool = False
    mobile_phone_dnc: bool = False


class CampaignBrief(BaseModel):
    model_config = ConfigDict(extra="forbid")
    drug_brand: str
    drug_generic: str
    condition_full: str
    condition_acronym: str
    therapeutic_area: str
    manufacturer: str
    campaign_type: str  # drug-specific | condition-specific | general-capabilities
    target_quota: int
    job_focus: str | None = None


class IntelBrief(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pipeline_summary: str
    rwe_signals: list[str] = []
    key_programs: list[str] = []
    prior_interactions: str = ""
    relevant_experience: str = ""
    messaging_angles: list[str] = []


class VetContactInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    contact: EnrichedContact
    brief: CampaignBrief
    intel: IntelBrief


class VetContactOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: VetDecision
    tier: ContactTier | None = None
    discard_reason: DiscardReason | None = None
    relevance_evidence: str


# ── Signature class ──

class VetContact:
    """Apply the BDR red-flags rubric to a single enriched contact.

    Rubric summary (full rubric in references/quality-and-vetting.md):
    Discard if: MSL/Field Medical, Biomarker/Discovery, Translational Research,
    Sales/Commercial, Operations/Strategy MA, Program Management, indication mismatch,
    accuracy < 85, no valid email.

    Core test: "Would this person commission, design, or approve a real-world
    evidence study?"

    Tier surviving contacts: ideal (drug/condition in title or full RWE career) /
    strong (RWE/HEOR/Evidence Director+) / good (TA-relevant Director+ with
    research commissioning path).
    """

    async def forward(self, inputs: VetContactInput) -> VetContactOutput:
        # Pre-filter 1: accuracy below threshold → auto-discard (no LLM)
        if inputs.contact.accuracy_score < MIN_ACCURACY_SCORE:
            return VetContactOutput(
                decision=VetDecision.DISCARD,
                discard_reason=DiscardReason.LOW_ACCURACY,
                relevance_evidence=(
                    f"Accuracy score {inputs.contact.accuracy_score} "
                    f"is below minimum threshold {MIN_ACCURACY_SCORE}."
                ),
            )
        # Pre-filter 2: no valid email → auto-discard (no LLM)
        if not inputs.contact.email:
            return VetContactOutput(
                decision=VetDecision.DISCARD,
                discard_reason=DiscardReason.NO_VALID_EMAIL,
                relevance_evidence="No valid email address found in ZoomInfo enrichment.",
            )

        # Dispatch to LLM for fuzzy rubric evaluation
        raise NotImplementedError(
            "Replace with DSPy or BAML call.\n"
            "Use the prompt and schemas from pipeline.yaml node vet_contact.signature_spec.\n"
            "Return a VetContactOutput with decision, tier (if keep), "
            "discard_reason (if discard), and relevance_evidence."
        )
