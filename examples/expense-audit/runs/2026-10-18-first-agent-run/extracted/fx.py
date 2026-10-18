"""
FX conversion utilities for the expense audit pipeline.

No external API calls. Uses the finance-published month-end rate table
from references/fx-rates.md, embedded here as constants. Finance publishes
rates on the last business day of each month; the general ledger uses the
same values, so the audit must match them exactly.

Rate format: USD per one unit of the foreign currency.
  e.g. EUR 1.105000 means 1 EUR = 1.105000 USD

Dependencies: none (stdlib only).
"""

from __future__ import annotations

from typing import TypedDict

# ---------------------------------------------------------------------------
# Rate table — sourced from references/fx-rates.md.
# Key: YYYY-MM period. Value: {ISO-4217 code → USD per unit}.
# ---------------------------------------------------------------------------
MONTH_END_RATES: dict[str, dict[str, float]] = {
    "2026-09": {
        "USD": 1.000000,
        "EUR": 1.105000,
        "GBP": 1.312500,
        "CAD": 0.724000,
        "JPY": 0.006780,
    },
    "2026-08": {
        "USD": 1.000000,
        "EUR": 1.098000,
        "GBP": 1.301000,
        "CAD": 0.731000,
        "JPY": 0.006850,
    },
}


class ExpenseLine(TypedDict):
    expense_id: str
    employee: str
    department: str
    merchant: str
    amount: float
    currency: str       # ISO-4217 code
    date: str           # YYYY-MM-DD
    category: str       # expense tool code (e.g. "airfare", "lodging")
    receipt: bool       # True when an image is attached in the expense tool
    nights: int | None  # only present for "lodging" lines
    preapproval_id: str | None  # only present for "entertainment" lines


class ConvertedLine(TypedDict):
    expense_id: str
    employee: str
    department: str
    merchant: str
    amount: float       # original amount
    currency: str       # original currency
    date: str
    category: str
    receipt: bool
    nights: int | None
    preapproval_id: str | None
    usd_amount: float   # amount × month-end rate, rounded to cent


def convert_to_usd(
    lines: list[ExpenseLine],
    period: str,
    fx_rates: dict[str, float] | None = None,
) -> dict[str, list[ConvertedLine]]:
    """
    Convert every expense line's ``amount`` to USD at the period's
    finance-published month-end rate.

    Contract:
    - ``period``: YYYY-MM string matching a key in MONTH_END_RATES or fx_rates.
    - The month-end rate of the *reporting period* applies to EVERY line,
      even lines dated in an earlier month (per expense-policy.md).
    - Rounding to the nearest cent is applied after converting each line
      (not before — per the skill: "Round to the cent only after converting").
    - ``fx_rates`` allows finance to supply rates for a period not yet in
      MONTH_END_RATES; values from fx_rates take precedence over the table.
    - Raises RuntimeError when:
        - ``period`` is absent from both MONTH_END_RATES and fx_rates
        - any line's ``currency`` has no rate for the period
      (Per skill: "If a line uses a currency that isn't in the table, stop
      and ask finance for the rate. Do not look one up online.")

    Returns:
        {"converted_lines": list[ConvertedLine]}
    """
    raise NotImplementedError(
        "implement: merge MONTH_END_RATES.get(period, {}) with fx_rates, "
        "then for each line compute round(line['amount'] * rate, 2) and copy "
        "remaining fields verbatim into a ConvertedLine."
    )
