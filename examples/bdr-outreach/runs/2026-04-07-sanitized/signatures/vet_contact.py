"""
LLM Judge Signature: VetContact

Source rubric: examples/bdr-outreach/skill/references/quality-and-vetting.md
Node: vet_contact (llm_judge, fan_out=true)

This signature applies the BDR red-flags rubric to a single enriched contact
and returns a bounded decision (keep/discard), optional tier, discard reason,
and relevance evidence string.

Hard thresholds (accuracy < 85, no valid email) are enforced as pre-filters
BEFORE the LLM is called — these contacts never consume tokens.

Usage:
    judge = VetContact()
    result = await judge.forward(VetContactInput(
        contact=enriched_contact,
        brief=campaign_brief,
        intel=intel_brief,
    ))
"""

from __future__ import annotations

from enum import Enum
from typing import NotRequired, TypedDict

from pydantic import BaseModel, ConfigDict

# ---------------------------------------------------------------------------
# Constants (from quality-and-vetting.md and lead-generation.md)
# ---------------------------------------------------------------------------

#: Source: quality-and-vetting.md line 42 / lead-generation.md line 108
MIN_ACCURACY_SCORE: int = 85


# ---------------------------------------------------------------------------
# Enums — every discrete output value from the rubric is enumerated
# ---------------------------------------------------------------------------


class VetDecision(str, Enum):
    """Top-level keep/discard decision."""

    KEEP = "keep"
    DISCARD = "discard"


class ContactTier(str, Enum):
    """Priority tier, only set when decision == KEEP.

    Source: lead-generation.md lines 116–119, quality-and-vetting.md.
      ideal  — drug/condition in title OR career entirely in RWE/HEOR for this TA
      strong — RWE/HEOR/Evidence Generation/Medical Affairs Director+ with domain history
      good   — TA-relevant Director/VP with credible path to research commissioning
    """

    IDEAL = "ideal"
    STRONG = "strong"
    GOOD = "good"


class DiscardReason(str, Enum):
    """Reason for discard, only set when decision == DISCARD.

    Source: quality-and-vetting.md lines 16–28 (red flags) + numeric thresholds.
    One member per named discard category so downstream consumers can count by reason.
    """

    INDICATION_MISMATCH = "indication_mismatch"
    MSL_ROLE = "msl_role"
    BIOMARKER_DISCOVERY = "biomarker_discovery"
    TRANSLATIONAL_RESEARCH = "translational_research"
    SALES_COMMERCIAL = "sales_commercial"
    OPS_STRATEGY = "ops_strategy"
    PROGRAM_MANAGEMENT = "program_management"
    US_COMMERCIAL_GLOBAL_FLAG = "us_commercial_global_flag"
    LOW_ACCURACY = "low_accuracy"
    NO_VALID_EMAIL = "no_valid_email"
    OTHER = "other"


# ---------------------------------------------------------------------------
# Input model
# ---------------------------------------------------------------------------


class EmploymentRecord(TypedDict):
    title: str
    company: str
    start_date: NotRequired[str]
    end_date: NotRequired[str | None]


class EnrichedContact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    first_name: str
    last_name: str
    job_title: str
    company_name: str
    email: str | None
    contact_accuracy_score: int
    employment_history: list[EmploymentRecord] = []
    external_urls: list[dict] = []
    direct_phone_do_not_call: bool = False
    mobile_phone_do_not_call: bool = False


class CampaignBrief(BaseModel):
    model_config = ConfigDict(extra="forbid")

    drug_brand: str
    drug_generic: str
    condition_full: str
    condition_acronym: str
    therapeutic_area: list[str]
    manufacturer: str
    campaign_type: str  # "drug-specific" | "condition-specific" | "general-capabilities"
    job_focus: str | None = None


class IntelBrief(BaseModel):
    """Working context from target_research — not a user-facing deliverable."""

    model_config = ConfigDict(extra="forbid")

    pipeline_summary: str
    rwe_signals: str
    key_programs: list[str] = []
    messaging_angles: list[str] = []
    acme_history: str = ""


class VetContactInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contact: EnrichedContact
    brief: CampaignBrief
    intel: IntelBrief


# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------


class VetContactOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: VetDecision
    tier: ContactTier | None = None              # only when decision == KEEP
    discard_reason: DiscardReason | None = None  # only when decision == DISCARD
    relevance_evidence: str                       # 1-2 sentences regardless of decision


# ---------------------------------------------------------------------------
# Signature class
# ---------------------------------------------------------------------------


class VetContact:
    """LLM judge for the BDR vetting rubric.

    Pre-filters handle hard thresholds (accuracy, email) without calling the LLM.
    The LLM handles all fuzzy classifications (employment history reading, core test,
    indication alignment, franchise-level matching).
    """

    async def forward(self, inputs: VetContactInput) -> VetContactOutput:
        # ------------------------------------------------------------------
        # Pre-filter 1: accuracy threshold (hard rule, no LLM needed)
        # Source: quality-and-vetting.md line 42, lead-generation.md line 108
        # ------------------------------------------------------------------
        if inputs.contact.contact_accuracy_score < MIN_ACCURACY_SCORE:
            return VetContactOutput(
                decision=VetDecision.DISCARD,
                discard_reason=DiscardReason.LOW_ACCURACY,
                relevance_evidence=(
                    f"Accuracy score {inputs.contact.contact_accuracy_score} "
                    f"below threshold {MIN_ACCURACY_SCORE}."
                ),
            )

        # ------------------------------------------------------------------
        # Pre-filter 2: valid email required (hard rule, no LLM needed)
        # Source: lead-generation.md line 107
        # ------------------------------------------------------------------
        if not inputs.contact.email or not inputs.contact.email.strip():
            return VetContactOutput(
                decision=VetDecision.DISCARD,
                discard_reason=DiscardReason.NO_VALID_EMAIL,
                relevance_evidence="No valid email address in ZoomInfo enrichment.",
            )

        # ------------------------------------------------------------------
        # Fuzzy classification — dispatch to LLM
        # The LLM applies:
        #   - Red flags rubric (quality-and-vetting.md lines 16–28)
        #   - Core test: "Would this person commission, design, or approve an RWE study?"
        #   - Tier assignment (ideal / strong / good)
        #   - Franchise-level indication alignment check
        # ------------------------------------------------------------------
        raise NotImplementedError(
            "LLM dispatch not yet implemented. "
            "Wire up DSPy Predict or BAML function call here. "
            "The system prompt should include the full rubric from "
            "references/quality-and-vetting.md."
        )
