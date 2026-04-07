"""Shared types for the graduated BDR pipeline.

These models are referenced from the IR's input/output type names
(``CampaignBrief``, ``EnrichedContact``, etc.) and are imported by both
the ``extracted/`` modules and the ``signatures/`` modules.

Keeping them in one place lets the Temporal adapter generate
``from .types import ...`` consistently and lets the test suite construct
realistic fixtures.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


# ───────── Enums ─────────


class CampaignType(str, Enum):
    DRUG_SPECIFIC = "drug-specific"
    CONDITION_SPECIFIC = "condition-specific"
    GENERAL_CAPABILITIES = "general-capabilities"


class JobFocus(str, Enum):
    MEDICAL = "medical"
    RWE = "rwe"
    HEOR = "heor"
    CLINICAL_DEVELOPMENT = "clinical-development"


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


# ───────── Inputs ─────────


class CampaignBrief(BaseModel):
    """Top-level pipeline input — one campaign, one drug, one indication."""

    model_config = ConfigDict(extra="forbid")

    drug_brand: str
    drug_generic: str
    condition_full: str
    condition_acronym: str
    therapeutic_area: str
    manufacturer: str
    campaign_type: CampaignType
    target_quota: int = Field(ge=1, description="Number of vetted contacts to deliver")
    job_focus: JobFocus | None = None


# ───────── Intermediate types ─────────


class TaxonomyIds(BaseModel):
    """ZoomInfo taxonomy IDs resolved once per campaign and cached."""

    model_config = ConfigDict(extra="forbid")

    vp_level_id: int
    director_level_id: int
    pharma_industry_id: int
    biotech_industry_id: int
    medical_dept_id: int


class IntelBrief(BaseModel):
    """Working context produced by Phase 1.5 target research."""

    model_config = ConfigDict(extra="forbid")

    pipeline_summary: str
    rwe_signals: list[str] = Field(default_factory=list)
    key_programs: list[str] = Field(default_factory=list)
    prior_interactions: str = ""
    relevant_experience: str = ""
    messaging_angles: list[str] = Field(default_factory=list)


class RawContact(BaseModel):
    """A contact returned by ZoomInfo search before enrichment."""

    model_config = ConfigDict(extra="forbid")

    zoominfo_id: str
    first_name: str
    last_name: str
    job_title: str
    company_name: str


class EmploymentEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    company: str
    title: str
    start_year: int | None = None
    end_year: int | None = None


class EnrichedContact(BaseModel):
    """A contact after ZoomInfo enrichment with full employment history."""

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
    employment_history: list[EmploymentEntry] = Field(default_factory=list)
    direct_phone_dnc: bool = False
    mobile_phone_dnc: bool = False


class VettedContact(BaseModel):
    """A contact that survived the LLM vetting loop."""

    model_config = ConfigDict(extra="forbid")

    contact: EnrichedContact
    tier: ContactTier
    relevance_evidence: str


class VettingSummary(BaseModel):
    """Aggregate stats for the vetting loop, surfaced at the review gate."""

    model_config = ConfigDict(extra="forbid")

    total_searched: int
    total_kept: int
    total_discarded: int
    discard_counts: dict[str, int] = Field(default_factory=dict)
    tier_breakdown: dict[str, int] = Field(default_factory=dict)


class HubSpotContact(BaseModel):
    """A contact after upsert into HubSpot — has a HubSpot ID."""

    model_config = ConfigDict(extra="forbid")

    hubspot_id: str
    email: str
    first_name: str
    last_name: str
    company_name: str
    job_title: str
    tier: ContactTier
    relevance_evidence: str


class ExclusionRecord(BaseModel):
    """A contact that failed an exclusion check."""

    model_config = ConfigDict(extra="forbid")

    contact_id: str
    email: str
    reason: str  # "do_not_contact" | "recently_emailed" | "active_sequence"
    detail: str = ""


# ───────── Phase 6 outputs ─────────


class PersonalizationOutput(BaseModel):
    """Per-contact email personalization produced by the LLM judge."""

    model_config = ConfigDict(extra="forbid")

    contact_id: str
    opening_line: str
    ta_callout: str
