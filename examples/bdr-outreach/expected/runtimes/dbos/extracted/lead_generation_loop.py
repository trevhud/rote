"""Extracted module: lead_generation_loop

Auto-generated stubs by rote.adapters.dbos. Replace each body
with the real implementation (direct vendor API calls — the MCP
tool calls from the source skill were graduated away at emit
time). Keep the signatures: the DBOS steps in main.py call these
with the step payload as keyword arguments.
"""

from __future__ import annotations

from typing import Any


def lead_generation_loop(**payload: Any) -> Any:
    """
    Iterative search-enrich-vet loop. Starts with three parallel ZoomInfo

    STUB — agent loops require an LLM agent runtime. Implement
    against the project's preferred agent harness (Anthropic
    Agent SDK, OpenAI Agents SDK, LangGraph, etc.).

    Tools the agent should be allowed to call:
      - zoominfo_search_contacts
      - zoominfo_search_companies

    Loop body sub-nodes (call once per iteration):
      - enrich_contact_batch
      - vet_contact
    """
    raise NotImplementedError(
        "lead_generation_loop.lead_generation_loop: implement against the vendor API"
    )
