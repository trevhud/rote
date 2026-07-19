# Phase 4 — LLM-Judge signature for: classify_thread_status
# Source: SKILL.md Step 3 ("Classify each warehouse thread into a pipeline step")
from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict


class PipelineStep(IntEnum):
    """6-step warehouse quoting pipeline. SKILL.md Step 3."""

    RFP_SENT = 1
    RESPONSE_NEEDED = 2
    FOLLOW_UP_NEEDED = 3
    PRICING_RETURNED = 4
    PRICING_SENT = 5
    DECLINED = 6


# Steps that need attention — surfaces as color-coded rows in the dashboard
NEEDS_ATTENTION_STEPS: frozenset[int] = frozenset({
    PipelineStep.RESPONSE_NEEDED,
    PipelineStep.FOLLOW_UP_NEEDED,
})


class ClassifyThreadInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread_id: str
    messages: list[str]           # email bodies in order
    account_name: str | None = None
    warehouse_name: str | None = None


class ClassifyThreadOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: int                     # PipelineStep int value (1-6)
    step_label: str               # human-readable label
    needs_attention: bool         # True when step in NEEDS_ATTENTION_STEPS
    rationale: str                # 1-2 sentence explanation


class ClassifyThreadStatus:
    """Classify a warehouse quoting thread into one of 6 pipeline steps."""

    async def forward(self, inputs: ClassifyThreadInput) -> ClassifyThreadOutput:
        raise NotImplementedError(
            "Dispatch to LLM with CLASSIFY_THREAD_PROMPT. "
            "Set needs_attention=True when step is 2 or 3."
        )


CLASSIFY_THREAD_PROMPT = """\
Classify this warehouse quoting email thread into one of the 6 pipeline steps below.
Read the ENTIRE thread before deciding.

Pipeline steps:
  1 = RFP Sent — Initial quote request sent to this warehouse, no reply yet
  2 = Response Needed — Warehouse replied; we need to respond (e.g. clarify, acknowledge)
  3 = Follow-Up Needed — We sent a message >3 days ago with no reply
  4 = Pricing Returned — Warehouse has sent back their pricing sheet
  5 = Pricing Sent — We have sent final pricing to the prospective client
  6 = Declined — Either the warehouse declined to quote, or we declined to proceed

Return:
- step: integer (1-6)
- step_label: the name above (e.g. "RFP Sent")
- needs_attention: true if step is 2 or 3
- rationale: 1-2 sentence explanation of why you chose this step

Account: {{ account_name }}
Warehouse: {{ warehouse_name }}

Thread messages (oldest first):
{{ messages }}

Return your classification via the structured output tool.
"""
