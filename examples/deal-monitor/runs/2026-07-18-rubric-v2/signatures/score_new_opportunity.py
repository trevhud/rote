"""
Typed signature for score_new_opportunity.

Scores a new (unquoted) deal-intake opportunity against the deal-scoring
rubric. The rubric lives at ~/.claude/skills/deal-scoring/SKILL.md and
is injected into the prompt at runtime — the scoring criteria are not
inlined here because they are maintained in the separate deal-scoring skill.

Scored dimensions (inferred from deal-monitor context):
  - Fit:  does the location, volume, and profile match warehouse capabilities?
  - Size: ARR / monthly order volume signal
  - Risk: unknown SKU count, go-live urgency, special requirements
  - Urgency: go-live date proximity

Output:
  score: 1-10 numeric (10 = highest priority)
  tier: hot | warm | cold
  key_factors: list of 1-3 positive signals
  risks: list of 0-3 risk flags

Compiled from: SKILL.md > Step 4 (Score New Opportunities)
"""
from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


# ── Enums ─────────────────────────────────────────────────────────────────────

class OpportunityTier(str, Enum):
    HOT  = "hot"
    WARM = "warm"
    COLD = "cold"


# ── Input / Output models ─────────────────────────────────────────────────────

class NewOpportunityInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    account_name: str
    submitter: str
    arr: Optional[str]
    monthly_dtc_orders: Optional[int]
    sku_count: Optional[int]
    sku_count_unknown: bool
    requested_location: Optional[str]
    sales_channels: Optional[list[str]]
    current_situation: Optional[str]
    pallets: Optional[int]
    sq_ft: Optional[int]
    international_shipping: Optional[bool]
    go_live_date: Optional[str]
    product_item_context: Optional[str]
    deal_scoring_rubric: str   # contents of deal-scoring/SKILL.md injected at runtime


class ScoreNewOpportunityOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: int = Field(ge=1, le=10)
    tier: OpportunityTier
    key_factors: list[str]     # 1-3 positive signals
    risks: list[str]           # 0-3 risk flags; empty list = no notable risks


# ── Pre-filter ────────────────────────────────────────────────────────────────
# No hard numeric pre-filter exists in the deal-monitor skill for scoring.
# (The NA_SKU_THRESHOLD pre-filter already ran in filter_and_extract_opps —
# every opp reaching this signature passed the inclusion gate.)


# ── Signature class ───────────────────────────────────────────────────────────

class ScoreNewOpportunity:
    """
    LLM judge: score an unquoted opportunity against the deal-scoring rubric.

    The deal_scoring_rubric field carries the external skill's content at
    runtime, loaded from ~/.claude/skills/deal-scoring/SKILL.md.
    """

    async def forward(
        self, inputs: NewOpportunityInput
    ) -> ScoreNewOpportunityOutput:
        raise NotImplementedError(
            "Wire to LLM: pass inputs (with deal_scoring_rubric injected) and "
            "return score/tier/key_factors/risks. "
            "See signature_spec in pipeline.yaml for the prompt template."
        )
