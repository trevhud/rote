"""Extracted module: hubspot

Auto-generated stubs by rote.adapters.dbos. Replace each body
with the real implementation (direct vendor API calls — the MCP
tool calls from the source skill were graduated away at emit
time). Keep the signatures: the DBOS steps in main.py call these
with the step payload as keyword arguments.
"""

from __future__ import annotations

from typing import Any


def batch_upsert_contacts(**payload: Any) -> Any:
    """
    Batch upsert contacts to HubSpot (create or update by email). Hard

    STUB — replace with the deterministic API call.

    Constants from the IR (lifted from the source skill):
      batch_size = 100
    """
    raise NotImplementedError(
        "hubspot.batch_upsert_contacts: implement against the vendor API"
    )


def create_campaign_list(**payload: Any) -> Any:
    """
    Create a static list named after the campaign and add the upserted

    STUB — replace with the deterministic API call.

    Constants from the IR (lifted from the source skill):
      add_batch_size = 250
    """
    raise NotImplementedError(
        "hubspot.create_campaign_list: implement against the vendor API"
    )


def upsert_sales_template(**payload: Any) -> Any:
    """
    Create or update HubSpot sales templates for the campaign sequence

    STUB — replace with the deterministic API call.
    """
    raise NotImplementedError(
        "hubspot.upsert_sales_template: implement against the vendor API"
    )
