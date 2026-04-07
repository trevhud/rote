"""LLM signature: vet a single enriched contact against the BDR rubric.

# Source rubric

The bdr-outreach skill's ``references/quality-and-vetting.md`` defines
the red-flags list and the core test:

    "Would this person commission, design, or approve a real-world
     evidence study?"

Plus the explicit discard categories:

    MSL / Medical Science Liaison
    Biomarker / Discovery Science
    Translational Research
    Sales / Commercial background
    Operations / Strategy MA
    Program / Project Management
    Indication mismatch (different TA franchise)
    Low accuracy (< 85)

# Graduated form

This is the canonical "fuzzy classification with structured output" case.
The LLM is still in the loop because employment-history reasoning is
genuinely fuzzy, but its inputs and outputs are typed and its decision
space is bounded by an enum. Regression evals live in
``../evals/vet_contact.jsonl``.

A production implementation would use DSPy::

    class VetContact(dspy.Signature):
        contact: EnrichedContact
        brief: CampaignBrief
        intel: IntelBrief
        decision: VetDecision
        tier: ContactTier | None
        discard_reason: DiscardReason | None
        relevance_evidence: str

    program = dspy.Predict(VetContact)
    program = dspy.MIPROv2(metric=...).compile(program, trainset=...)

For v0 this is a stub returning a canned ``keep`` decision.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from ..types import (
    CampaignBrief,
    ContactTier,
    DiscardReason,
    EnrichedContact,
    IntelBrief,
    VetDecision,
)

# Constant lifted from the IR (constants.min_accuracy_score). Discard
# logic for low-accuracy contacts is enforced *here* rather than in the
# LLM prompt so the rule cannot drift.
MIN_ACCURACY_SCORE: int = 85


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


class VetContact:
    """Typed LLM judge for BDR contact vetting."""

    async def forward(self, inputs: VetContactInput) -> VetContactOutput:
        """Apply the BDR vetting rubric to a single contact.

        Pre-LLM filter: low-accuracy contacts are discarded in code,
        not by the LLM, because the threshold is a hard rule and
        running it through the model wastes tokens.
        """
        if inputs.contact.accuracy_score < MIN_ACCURACY_SCORE:
            return VetContactOutput(
                decision=VetDecision.DISCARD,
                discard_reason=DiscardReason.LOW_ACCURACY,
                relevance_evidence=(
                    f"Accuracy score {inputs.contact.accuracy_score} below "
                    f"threshold {MIN_ACCURACY_SCORE}."
                ),
            )

        # Real implementation would dispatch to DSPy/BAML here.
        raise NotImplementedError(
            "VetContact.forward: implement against an LLM signature framework"
        )
