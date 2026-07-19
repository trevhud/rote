# Crystallized from SKILL.md Step 2.
# Vendor API: Gmail threads.list + threads.get (via Gmail MCP tool)
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

GMAIL_SEARCH_DAYS = 60  # SKILL.md:38 — "last 60 days"

# SKILL.md:34-37 — fixed subject-pattern searches (searches 1-3)
GMAIL_FIXED_SUBJECT_PATTERNS = [
    "Quote Request",
    "New Business",
    "RFP",
]

GMAIL_THREAD_URL_TEMPLATE = "https://mail.google.com/mail/u/0/#inbox/{thread_id}"


@dataclass
class ThreadContent:
    thread_id: str
    gmail_thread_url: str
    messages: list[str]  # each message as raw text


def build_gmail_queries(account_names: list[str]) -> list[str]:
    """Build the full list of Gmail search queries.

    Produces queries for searches 1-3 (fixed subjects), search 4
    (warehouse domain — implementor should extend with real domains),
    and search 5 (per account name from #deal-intake).

    Returns deduplicated, ordered query list. Pure function.
    """
    cutoff = (date.today() - timedelta(days=GMAIL_SEARCH_DAYS)).strftime("%Y/%m/%d")
    queries: list[str] = []

    # Searches 1-3: fixed subject patterns
    for subject in GMAIL_FIXED_SUBJECT_PATTERNS:
        queries.append(f'subject:"{subject}" after:{cutoff}')

    # Search 4: known warehouse contact domains
    # TODO: replace with real warehouse domains for this deployment
    warehouse_domains = ["@warehouse1.com", "@3plprovider.com"]
    domain_clause = " OR ".join(f"(from:{d} OR to:{d})" for d in warehouse_domains)
    queries.append(f"({domain_clause}) after:{cutoff}")

    # Search 5: by known account names from #deal-intake
    for name in account_names:
        if name:
            queries.append(f'subject:"{name}" after:{cutoff}')

    return queries


def search_gmail_for_quoting_threads(account_names: list[str]) -> list[str]:
    """Run all Gmail searches and return a deduplicated list of thread IDs.

    Calls Gmail threads.list for each query from build_gmail_queries().
    Deduplicates by thread ID. Raises on API errors.

    NOTE: Implementor must wire the Gmail MCP client here.
    GET /gmail/v1/users/me/threads?q={query} for each query in build_gmail_queries().
    """
    raise NotImplementedError(
        "Wire the Gmail MCP client: for each query in build_gmail_queries(account_names), "
        "call threads.list and collect unique thread IDs."
    )


def fetch_thread_content(thread_id: str) -> ThreadContent:
    """Fetch the full content of a Gmail thread.

    Calls Gmail threads.get, extracts message bodies as plain text,
    and builds the gmail_thread_url.

    NOTE: Implementor must wire the Gmail MCP client here.
    GET /gmail/v1/users/me/threads/{thread_id}?format=full
    """
    raise NotImplementedError(
        "Wire the Gmail MCP client: call threads.get on thread_id, "
        "extract all message body parts as text, return ThreadContent."
    )


def build_thread_url(thread_id: str) -> str:
    """Return the Gmail deep-link URL for a thread."""
    return GMAIL_THREAD_URL_TEMPLATE.format(thread_id=thread_id)
