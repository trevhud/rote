"""
Gmail search helpers for deal-monitor.

MCP tool: gmail_search_threads (gmail MCP server)
Underlying API: GET /gmail/v1/users/me/threads?q=... (Gmail REST API v1)
"""
from __future__ import annotations
from datetime import date, timedelta

GMAIL_LOOKBACK_DAYS: int = 60
GMAIL_SUBJECT_FILTERS: list[str] = ["Quote Request", "New Business", "RFP"]


def _lookback_date(days: int = GMAIL_LOOKBACK_DAYS) -> str:
    """Return Gmail 'after:YYYY/MM/DD' filter for the lookback window."""
    cutoff = date.today() - timedelta(days=days)
    return f"after:{cutoff.strftime('%Y/%m/%d')}"


def gmail_thread_url(thread_id: str) -> str:
    """Canonical Gmail deep-link for a thread."""
    return f"https://mail.google.com/mail/u/0/#inbox/{thread_id}"


def search_gmail_fixed(lookback_days: int = GMAIL_LOOKBACK_DAYS) -> list[dict]:
    """
    Run fixed searches 1-4 (subject keywords + known warehouse domains).

    Searches:
    1. subject:"Quote Request"
    2. subject:"New Business"
    3. subject:"RFP"
    4. from/to known warehouse contact domains

    Returns merged list of thread dicts with keys:
      thread_id, account_name_raw, warehouse, sent_date, replies, state, thread_url

    Compiled from: SKILL.md > Step 2 searches 1-4
    """
    after = _lookback_date(lookback_days)
    queries = [f'subject:"{s}" {after}' for s in GMAIL_SUBJECT_FILTERS]
    # Search 4: warehouse domain contacts — operator must supply domain list
    # at runtime via WAREHOUSE_CONTACT_DOMAINS env var or rote config.
    queries.append(f"(from:WAREHOUSE_DOMAINS OR to:WAREHOUSE_DOMAINS) {after}")

    raise NotImplementedError(
        "Wire each query to gmail MCP server: tool=gmail_search_threads. "
        "Merge results and deduplicate by thread_id before returning."
    )


def search_gmail_by_account(account_names: list[str], lookback_days: int = GMAIL_LOOKBACK_DAYS) -> list[dict]:
    """
    Search Gmail for threads matching known account names from #deal-intake.

    This is search 5 from the skill — depends on filter_and_extract_opps output.
    Returns same schema as search_gmail_fixed.

    Compiled from: SKILL.md > Step 2 search 5
    """
    after = _lookback_date(lookback_days)
    raise NotImplementedError(
        "For each account_name, run: gmail_search_threads(query=f'{account_name} {after}'). "
        "Merge and deduplicate by thread_id."
    )
