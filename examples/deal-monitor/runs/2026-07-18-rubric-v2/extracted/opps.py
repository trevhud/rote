"""
Opportunity extraction and inclusion filtering for deal-monitor.

All logic here is deterministic — no LLM needed.
Graduated from: SKILL.md > Step 1 (filter and extract)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

# ── Inclusion rules ────────────────────────────────────────────────────────────
NA_SKU_THRESHOLD: int = 2_000
EU_UK_ALWAYS_INCLUDE_REPS: list[str] = ["Alex Rivers", "Sam Patel"]

# Fields the skill asks to extract per opportunity
OPP_FIELDS: list[str] = [
    "account_name",
    "submitter",
    "arr",
    "monthly_dtc_orders",
    "sku_count",
    "requested_location",
    "sales_channels",
    "current_situation",
    "pallets",
    "sq_ft",
    "international_shipping",
    "go_live_date",
    "product_item_context",
]


@dataclass
class Opportunity:
    account_name: str
    submitter: str
    arr: Optional[str]
    monthly_dtc_orders: Optional[int]
    sku_count: Optional[int]               # None = unknown
    requested_location: Optional[str]
    sales_channels: Optional[list[str]]
    current_situation: Optional[str]
    pallets: Optional[int]
    sq_ft: Optional[int]
    international_shipping: Optional[bool]
    go_live_date: Optional[str]
    product_item_context: Optional[str]
    sku_count_unknown: bool = False        # True when SKU count not stated
    include_flag_risk: bool = False        # True when included despite unknown SKU


@dataclass
class FilteredOpps:
    included: list[Opportunity]
    excluded: list[Opportunity]
    account_names: list[str]              # for downstream Gmail search 5


def _is_eu_uk(opp_raw: dict) -> bool:
    """Return True for EU/UK/non-US location opps, or always-include reps."""
    submitter = (opp_raw.get("submitter") or "").strip()
    location = (opp_raw.get("requested_location") or "").lower()
    if submitter in EU_UK_ALWAYS_INCLUDE_REPS:
        return True
    non_us_keywords = ["uk", "europe", "eu", "germany", "france", "netherlands", "spain", "italy"]
    return any(kw in location for kw in non_us_keywords)


def _parse_sku_count(raw: str | int | None) -> tuple[Optional[int], bool]:
    """
    Return (count, is_unknown).
    is_unknown=True when the field is absent or non-numeric.
    """
    if raw is None or str(raw).strip() == "":
        return None, True
    try:
        return int(str(raw).replace(",", "")), False
    except ValueError:
        return None, True


def filter_and_extract_opps(messages: list[dict]) -> FilteredOpps:
    """
    Apply the geographic / SKU inclusion rules from the skill and extract
    structured Opportunity objects from raw Slack message dicts.

    Inclusion rules (SKILL.md Step 1):
    - EU/UK: ALL opps — any non-US location OR submitted by always-include rep
    - North America: only if sku_count < NA_SKU_THRESHOLD (2000)
      - Unknown SKU: include but set include_flag_risk=True
      - Confirmed >= 2000: exclude

    Returns FilteredOpps with included/excluded lists and a flat account_names
    list for downstream Gmail search 5.
    """
    included: list[Opportunity] = []
    excluded: list[Opportunity] = []

    for msg in messages:
        sku_raw = msg.get("sku_count")
        sku_count, sku_unknown = _parse_sku_count(sku_raw)
        eu_uk = _is_eu_uk(msg)

        flag_risk = False
        if eu_uk:
            should_include = True
        elif sku_unknown:
            should_include = True
            flag_risk = True
        elif sku_count is not None and sku_count >= NA_SKU_THRESHOLD:
            should_include = False
        else:
            should_include = True

        opp = Opportunity(
            account_name=msg.get("account_name", ""),
            submitter=msg.get("submitter", ""),
            arr=msg.get("arr"),
            monthly_dtc_orders=msg.get("monthly_dtc_orders"),
            sku_count=sku_count,
            requested_location=msg.get("requested_location"),
            sales_channels=msg.get("sales_channels"),
            current_situation=msg.get("current_situation"),
            pallets=msg.get("pallets"),
            sq_ft=msg.get("sq_ft"),
            international_shipping=msg.get("international_shipping"),
            go_live_date=msg.get("go_live_date"),
            product_item_context=msg.get("product_item_context"),
            sku_count_unknown=sku_unknown,
            include_flag_risk=flag_risk,
        )
        if should_include:
            included.append(opp)
        else:
            excluded.append(opp)

    account_names = [o.account_name for o in included if o.account_name]
    return FilteredOpps(included=included, excluded=excluded, account_names=account_names)
