# Phase 4 — LLM-Judge signature for: extract_deal_fields
# Source: SKILL.md Step 1 ("Extract for each: Account Name, Submitter, ARR, ...")
from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ExtractDealFieldsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message_text: str


class DealRecord(BaseModel):
    """Structured representation of a single deal intake submission."""

    model_config = ConfigDict(extra="forbid")

    account_name: str
    submitter: str
    arr: str | None = None               # e.g. "$1.2M", "unknown"
    monthly_dtc_orders: str | None = None
    sku_count: int | None = None
    sku_count_unknown: bool = False      # True when SKU count not stated
    requested_location: str | None = None
    sales_channels: list[str] = []
    current_situation: str | None = None
    pallets: int | None = None
    sq_ft: int | None = None
    international_shipping: bool | None = None
    go_live_date: str | None = None      # ISO date string or free text
    product_context: str | None = None


class ExtractDealFields:
    """Parse a #deal-intake Slack message into a typed DealRecord.

    Pre-filter: if the message_text is fewer than 20 characters, skip
    (likely a reaction or a non-intake message) and raise ValueError.
    """

    MIN_MESSAGE_LENGTH = 20

    async def forward(self, inputs: ExtractDealFieldsInput) -> DealRecord:
        if len(inputs.message_text.strip()) < self.MIN_MESSAGE_LENGTH:
            raise ValueError(
                f"Message too short ({len(inputs.message_text)} chars) "
                "to contain a deal intake."
            )
        raise NotImplementedError(
            "Dispatch to LLM with EXTRACT_DEAL_FIELDS_PROMPT. "
            "Return a DealRecord. Set sku_count_unknown=True when SKU count "
            "is absent or described as 'unknown'/'TBD'."
        )


EXTRACT_DEAL_FIELDS_PROMPT = """\
Extract the following fields from this #deal-intake Slack message.
If a field is not mentioned, return null (or empty list for arrays).
For sku_count: if the count is explicitly stated as unknown, TBD, or not mentioned,
set sku_count=null and sku_count_unknown=true.

Fields to extract:
- account_name: company/brand name
- submitter: Slack user who posted
- arr: annual recurring revenue (keep original format, e.g. "$1.2M")
- monthly_dtc_orders: monthly direct-to-consumer order volume
- sku_count: integer count of SKUs (null if unknown)
- sku_count_unknown: true if SKU count was not stated
- requested_location: warehouse/fulfillment location preference
- sales_channels: list of channels (e.g. ["DTC", "Amazon", "Wholesale"])
- current_situation: description of their current fulfillment state
- pallets: pallet count (integer or null)
- sq_ft: square footage needed (integer or null)
- international_shipping: true/false/null
- go_live_date: target go-live date
- product_context: product category and any notable characteristics

Message:
{{ message_text }}

Return your extraction via the structured output tool.
"""
