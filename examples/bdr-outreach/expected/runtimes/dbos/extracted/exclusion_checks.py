"""Extracted module: exclusion_checks

Auto-generated stubs by rote.adapters.dbos. Replace each body
with the real implementation (direct vendor API calls — the MCP
tool calls from the source skill were graduated away at emit
time). Keep the signatures: the DBOS steps in main.py call these
with the step payload as keyword arguments.
"""

from __future__ import annotations

from typing import Any


def check_do_not_contact(**payload: Any) -> Any:
    """
    For each contact, look up \"BDR do not contact\" list memberships.

    STUB — replace with the deterministic API call.

    MANDATORY: this node was marked mandatory in the source
    skill. The workflow always calls it; do not make it
    conditional.
    """
    raise NotImplementedError(
        "exclusion_checks.check_do_not_contact: implement against the vendor API"
    )


def check_recently_emailed(**payload: Any) -> Any:
    """
    For each contact, check if they were emailed (outbound) in the last

    STUB — replace with the deterministic API call.

    MANDATORY: this node was marked mandatory in the source
    skill. The workflow always calls it; do not make it
    conditional.

    Constants from the IR (lifted from the source skill):
      days_back = 30
      direction = 'OUTBOUND'
    """
    raise NotImplementedError(
        "exclusion_checks.check_recently_emailed: implement against the vendor API"
    )


def check_active_sequence(**payload: Any) -> Any:
    """
    For each contact, check if they are already enrolled in an active

    STUB — replace with the deterministic API call.

    MANDATORY: this node was marked mandatory in the source
    skill. The workflow always calls it; do not make it
    conditional.
    """
    raise NotImplementedError(
        "exclusion_checks.check_active_sequence: implement against the vendor API"
    )
