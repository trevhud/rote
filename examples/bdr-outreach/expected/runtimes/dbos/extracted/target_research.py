"""Extracted module: target_research

Auto-generated stubs by rote.adapters.dbos. Replace each body
with the real implementation (direct vendor API calls — the MCP
tool calls from the source skill were graduated away at emit
time). Keep the signatures: the DBOS steps in main.py call these
with the step payload as keyword arguments.
"""

from __future__ import annotations

from typing import Any


def target_research(**payload: Any) -> Any:
    """
    Run external research (Bright Data web search, ClinicalTrials.gov)

    STUB — agent loops require an LLM agent runtime. Implement
    against the project's preferred agent harness (Anthropic
    Agent SDK, OpenAI Agents SDK, LangGraph, etc.).

    Tools the agent should be allowed to call:
      - bright_data_search
      - bright_data_scrape
      - clinical_trials_search
      - airweave_search
      - salesforce_query
    """
    raise NotImplementedError(
        "target_research.target_research: implement against the vendor API"
    )
