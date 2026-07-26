# Phase 4 — LLM-Judge signature for: score_new_opportunity
# Source: SKILL.md Step 4 ("Read the deal-scoring skill... and apply its scoring rules")
#
# IMPORTANT — OPEN QUESTION:
# The source skill references an external scoring rubric at
# ~/.claude/skills/deal-scoring/SKILL.md which does not exist in this
# workspace. The SCORE_OPP_PROMPT below contains a PLACEHOLDER where that
# rubric must be inlined before deploying this pipeline.
# See compile-report.md §Open Questions.
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict

from signatures.extract_deal_fields import DealRecord


class OpportunityScore(str, Enum):
    HOT = "hot"
    WARM = "warm"
    COLD = "cold"
    DISQUALIFIED = "disqualified"


class ScoreOpportunityInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    deal: DealRecord


class ScoreOpportunityOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: OpportunityScore
    rationale: str            # 2-3 sentence explanation
    risk_flags: list[str]     # e.g. ["unknown SKU count", "tight go-live date"]


class ScoreNewOpportunity:
    """Score a new (unquoted) opportunity against the deal-scoring rubric.

    Pre-filter: DealRecord marked as excluded by apply_inclusion_filter
    should never reach this node — the upstream pure_function handles that.
    """

    async def forward(self, inputs: ScoreOpportunityInput) -> ScoreOpportunityOutput:
        raise NotImplementedError(
            "Dispatch to LLM with SCORE_OPP_PROMPT. "
            "Inline the rubric from ~/.claude/skills/deal-scoring/SKILL.md "
            "into SCORE_OPP_PROMPT before deploying."
        )


SCORE_OPP_PROMPT = """\
Score this fulfillment brokerage deal opportunity.

=== SCORING RUBRIC ===
[PLACEHOLDER — Copy the full scoring rubric from ~/.claude/skills/deal-scoring/SKILL.md here.
Until this is filled in, this node will NOT produce reliable scores.]
=== END RUBRIC ===

Score the opportunity as one of:
- hot: high-probability, good fit, should prioritize immediately
- warm: moderate fit, worth pursuing with standard effort
- cold: weak fit or incomplete information, low priority
- disqualified: does not meet minimum criteria (e.g., confirmed ≥2,000 SKUs passed
  through by mistake, location outside service area)

Also list any risk_flags (short strings) that the account owner should know.

Deal:
{{ deal }}

Return your score, rationale, and risk_flags via the structured output tool.
"""
