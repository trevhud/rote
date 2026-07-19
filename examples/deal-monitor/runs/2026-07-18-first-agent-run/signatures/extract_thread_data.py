# Phase 4 — LLM-Judge signature for: extract_thread_data
# Source: SKILL.md Step 2 ("extract: account name, every warehouse emailed,
# per warehouse sent date/replies/state, and the Gmail thread URL")
from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ExtractThreadDataInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread_id: str
    messages: list[str]          # raw email message bodies, in order
    known_account_names: list[str]  # from #deal-intake, for fuzzy matching


class WarehouseContact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    warehouse_name: str
    sent_date: str | None = None      # ISO date or free text
    reply_received: bool = False
    state_description: str | None = None


class ThreadRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread_id: str
    gmail_thread_url: str              # built by extracted/gmail.py:build_thread_url
    account_name: str | None = None   # matched from known_account_names or extracted
    warehouses: list[WarehouseContact] = []


class ExtractThreadData:
    """Extract structured data from a Gmail quoting thread."""

    async def forward(self, inputs: ExtractThreadDataInput) -> ThreadRecord:
        raise NotImplementedError(
            "Dispatch to LLM with EXTRACT_THREAD_DATA_PROMPT. "
            "Build gmail_thread_url using build_thread_url(thread_id). "
            "Fuzzy-match account_name from known_account_names."
        )


EXTRACT_THREAD_DATA_PROMPT = """\
Read this Gmail email thread and extract structured quoting data.

For each warehouse/3PL contact in the thread:
- warehouse_name: name of the warehouse or 3PL company
- sent_date: the date the RFP or quote request was sent to this warehouse (ISO YYYY-MM-DD if possible)
- reply_received: true if the warehouse sent any reply
- state_description: brief (1 sentence) description of where this warehouse interaction stands

Also:
- account_name: the prospective client/brand this quoting thread is about.
  Use the known_account_names list for fuzzy matching — pick the closest name
  if one matches. If none match, extract from the email content.
- gmail_thread_url: set to https://mail.google.com/mail/u/0/#inbox/{{ thread_id }}

Thread ID: {{ thread_id }}
Known account names: {{ known_account_names }}

Email messages (oldest first):
{{ messages }}

Return your extraction via the structured output tool.
"""
