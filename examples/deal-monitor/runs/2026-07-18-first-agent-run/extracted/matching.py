# Crystallized from SKILL.md Step 2: "Match threads to #deal-intake opps
# by account name (fuzzy match fine)."
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

FUZZY_MATCH_THRESHOLD = 80  # minimum rapidfuzz score (0-100) to count as a match


@dataclass
class QuotedDeal:
    deal: Any          # DealRecord
    threads: list[Any]  # list of ThreadRecord


@dataclass
class MatchResult:
    quoted: list[QuotedDeal]
    unquoted: list[Any]  # DealRecord list with no matched threads


def _normalize(name: str) -> str:
    return name.lower().strip()


def match_threads_to_deals(deals: list[Any], threads: list[Any]) -> MatchResult:
    """Fuzzy-match email threads to deal records by account name.

    Uses rapidfuzz for string similarity (falls back to difflib if unavailable).
    A thread matches a deal when the similarity score >= FUZZY_MATCH_THRESHOLD.

    Each deal that matches at least one thread goes to MatchResult.quoted.
    Deals with no matching threads go to MatchResult.unquoted.

    SKILL.md Step 2: "Any opp with at least one matched email thread goes
    to the Quoting tab."
    """
    try:
        from rapidfuzz import fuzz
        score_fn = lambda a, b: fuzz.partial_ratio(_normalize(a), _normalize(b))
    except ImportError:
        import difflib
        def score_fn(a: str, b: str) -> float:
            return difflib.SequenceMatcher(None, _normalize(a), _normalize(b)).ratio() * 100

    # Build: thread -> best matching deal
    deal_to_threads: dict[int, list[Any]] = {i: [] for i in range(len(deals))}

    for thread in threads:
        thread_account = thread.account_name or ""
        best_idx: int | None = None
        best_score = 0.0
        for i, deal in enumerate(deals):
            s = score_fn(thread_account, deal.account_name or "")
            if s > best_score:
                best_score = s
                best_idx = i
        if best_idx is not None and best_score >= FUZZY_MATCH_THRESHOLD:
            deal_to_threads[best_idx].append(thread)

    quoted = []
    unquoted = []
    for i, deal in enumerate(deals):
        if deal_to_threads[i]:
            quoted.append(QuotedDeal(deal=deal, threads=deal_to_threads[i]))
        else:
            unquoted.append(deal)

    return MatchResult(quoted=quoted, unquoted=unquoted)
