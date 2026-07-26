"""
Fuzzy-match Gmail threads to deal-intake opportunities.

Deterministic — uses rapidfuzz for normalized string similarity.
No LLM needed: the matching criterion is a name similarity threshold,
not a judgment call.

Compiled from: SKILL.md > Step 2 ("Match threads to #deal-intake opps by
account name (fuzzy match fine)") and Step 3 (partition for tabs).
"""
from __future__ import annotations
from dataclasses import dataclass

try:
    from rapidfuzz import fuzz, process
except ImportError as exc:
    raise ImportError("rapidfuzz is required: pip install rapidfuzz") from exc

from extracted.opps import Opportunity

FUZZY_MATCH_THRESHOLD: int = 80  # token_sort_ratio score 0-100


@dataclass
class MatchedOpp:
    opp: Opportunity
    threads: list[dict]  # one or more Gmail thread dicts matched to this opp


@dataclass
class MatchResult:
    matched: list[MatchedOpp]    # opps with at least one matched thread → Quoting tab
    unmatched: list[Opportunity] # opps with no matched threads → New Opportunities tab
    all_threads: list[dict]      # flat list of all threads (for classify_warehouse_thread)


def match_opps_to_threads(
    opps: list[Opportunity],
    fixed_threads: list[dict],
    account_threads: list[dict],
) -> MatchResult:
    """
    Fuzzy-match Gmail threads to opportunities by account name.

    Deduplicates threads by thread_id across both sources, then groups
    each thread under the best-matching opportunity (score ≥ FUZZY_MATCH_THRESHOLD).

    Unmatched threads (score below threshold) are not surfaced in the dashboard.
    """
    # Deduplicate threads across both search result sets
    seen_ids: set[str] = set()
    all_threads: list[dict] = []
    for thread in (fixed_threads or []) + (account_threads or []):
        tid = thread.get("thread_id") or thread.get("id")
        if tid and tid not in seen_ids:
            seen_ids.add(tid)
            all_threads.append(thread)

    if not opps:
        return MatchResult(matched=[], unmatched=[], all_threads=all_threads)

    opp_names = [o.account_name for o in opps]
    opp_buckets: dict[int, list[dict]] = {i: [] for i in range(len(opps))}

    for thread in all_threads:
        raw_name = (thread.get("account_name_raw") or "").strip()
        if not raw_name:
            continue
        result = process.extractOne(
            raw_name, opp_names, scorer=fuzz.token_sort_ratio
        )
        if result and result[1] >= FUZZY_MATCH_THRESHOLD:
            best_idx = opp_names.index(result[0])
            opp_buckets[best_idx].append(thread)

    matched: list[MatchedOpp] = []
    unmatched: list[Opportunity] = []
    for i, opp in enumerate(opps):
        if opp_buckets[i]:
            matched.append(MatchedOpp(opp=opp, threads=opp_buckets[i]))
        else:
            unmatched.append(opp)

    return MatchResult(matched=matched, unmatched=unmatched, all_threads=all_threads)
