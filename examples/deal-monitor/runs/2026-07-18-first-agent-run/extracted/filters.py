# Crystallized from SKILL.md Step 1 inclusion rules (lines 26-29).
# These are MANDATORY business rules — moving them from prose to code
# makes them impossible to skip through prompt drift or model upgrade.
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

MAX_NA_SKU_COUNT = 2000  # SKILL.md:29 — "Only if SKU count < 2,000"

# SKILL.md:28 — "submitted by Alex Rivers or Sam Patel (always UK)"
ALWAYS_UK_SUBMITTERS = frozenset({"Alex Rivers", "Sam Patel"})

# Region labels used for flagging
REGION_EU_UK = "EU/UK"
REGION_NA = "North America"


@dataclass
class FilterResult:
    included: list[Any]  # DealRecord list (typed by caller)
    excluded: list[Any]
    sku_risk_flags: list[str]  # account names where SKU count is unknown


def _is_eu_uk_location(requested_location: str | None) -> bool:
    """Return True for any UK, EU, or non-US location string."""
    if requested_location is None:
        return False
    loc = requested_location.lower()
    us_keywords = ("us", "usa", "united states", "north america", "na", "canada")
    for kw in us_keywords:
        if kw in loc:
            return False
    return True


def apply_inclusion_filter(deals: list[Any]) -> FilterResult:
    """Apply the MANDATORY inclusion/exclusion rules from SKILL.md Step 1.

    Rules (from SKILL.md lines 26-29):
    - EU/UK: include ALL opps — any UK, EU, or non-US location.
    - EU/UK: include if submitter is in ALWAYS_UK_SUBMITTERS.
    - North America: include ONLY if SKU count < MAX_NA_SKU_COUNT.
    - Unknown SKU on a NA opp: include, but flag as risk.
    - Confirmed >= MAX_NA_SKU_COUNT on a NA opp: exclude.

    Args:
        deals: list of DealRecord objects with .account_name, .submitter,
               .requested_location, .sku_count, .sku_count_unknown fields.

    Returns:
        FilterResult with included/excluded deal lists and sku_risk_flags.
    """
    included = []
    excluded = []
    sku_risk_flags = []

    for deal in deals:
        submitter_always_uk = deal.submitter in ALWAYS_UK_SUBMITTERS
        is_eu_uk = submitter_always_uk or _is_eu_uk_location(deal.requested_location)

        if is_eu_uk:
            included.append(deal)
            continue

        # North America opp
        if deal.sku_count_unknown or deal.sku_count is None:
            # Unknown SKU: include, flag as risk
            included.append(deal)
            sku_risk_flags.append(deal.account_name)
        elif deal.sku_count < MAX_NA_SKU_COUNT:
            included.append(deal)
        else:
            # Confirmed >= 2000: exclude
            excluded.append(deal)

    return FilterResult(
        included=included,
        excluded=excluded,
        sku_risk_flags=sku_risk_flags,
    )
