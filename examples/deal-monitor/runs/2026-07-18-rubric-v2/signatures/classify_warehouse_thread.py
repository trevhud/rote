"""
Typed signature for classify_warehouse_thread.

Classifies a single Gmail warehouse-quoting thread into one of six
pipeline steps based on reading the full email thread contents.

Decision space (SKILL.md Step 3):
  1 = RFP Sent        — we sent the RFP, no warehouse reply yet
  2 = Response Needed — warehouse replied, we need to act
  3 = Follow-Up Needed — we sent follow-up, no warehouse reply
  4 = Pricing Returned — warehouse returned pricing, we have it
  5 = Pricing Sent    — we sent pricing to the client
  6 = Declined        — warehouse declined or opp is dead

Compiled from: SKILL.md > Step 3
"""
from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, ConfigDict


# ── Enums ─────────────────────────────────────────────────────────────────────

class PipelineStep(int, Enum):
    RFP_SENT          = 1
    RESPONSE_NEEDED   = 2
    FOLLOW_UP_NEEDED  = 3
    PRICING_RETURNED  = 4
    PRICING_SENT      = 5
    DECLINED          = 6


# ── Input / Output models ─────────────────────────────────────────────────────

class EmailMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    from_address: str
    to_addresses: list[str]
    date: str         # ISO 8601
    subject: str
    body_text: str


class ClassifyWarehouseThreadInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread_id: str
    account_name: str
    warehouse: str
    messages: list[EmailMessage]   # full thread, chronological


class ClassifyWarehouseThreadOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pipeline_step: PipelineStep
    step_name: str                 # human-readable label (e.g. "Response Needed")
    classification_reason: str     # 1-2 sentence explanation citing the thread evidence


STEP_NAMES: dict[int, str] = {
    1: "RFP Sent",
    2: "Response Needed",
    3: "Follow-Up Needed",
    4: "Pricing Returned",
    5: "Pricing Sent",
    6: "Declined",
}


# ── Signature class ───────────────────────────────────────────────────────────

class ClassifyWarehouseThread:
    """
    LLM judge: classify a warehouse quoting thread into the 6-step pipeline.

    No pre-filter applies here — the classification depends entirely on
    reading the thread's message history, which is free-text prose.
    """

    async def forward(
        self, inputs: ClassifyWarehouseThreadInput
    ) -> ClassifyWarehouseThreadOutput:
        # Dispatch to LLM — implement with DSPy or direct Anthropic SDK call.
        raise NotImplementedError(
            "Wire to LLM: pass inputs.messages as context, return pipeline_step enum. "
            "See signature_spec in pipeline.yaml for the prompt template."
        )
