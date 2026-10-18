"""Step 1: convert every expense line to USD at the month-end rate.

The rate table is references/fx-rates.md, lifted out of prose into code.
Finance publishes one table per month and payroll must match the general
ledger, so an unlisted month or currency is an error, never a live lookup.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from typing import Any

#: USD per one unit of the currency, by reporting period.
MONTH_END_RATES: dict[str, dict[str, Decimal]] = {
    "2026-09": {
        "USD": Decimal("1.000000"),
        "EUR": Decimal("1.105000"),
        "GBP": Decimal("1.312500"),
        "CAD": Decimal("0.724000"),
        "JPY": Decimal("0.006780"),
    },
    "2026-08": {
        "USD": Decimal("1.000000"),
        "EUR": Decimal("1.098000"),
        "GBP": Decimal("1.301000"),
        "CAD": Decimal("0.731000"),
        "JPY": Decimal("0.006850"),
    },
}

CENT = Decimal("0.01")


def to_cents(value: Decimal) -> Decimal:
    """Round to the cent, half up (the ledger's rounding rule)."""
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def convert_to_usd(submissions: list[dict[str, Any]], period: str) -> dict[str, Any]:
    """Every submission plus ``rate`` and ``amount_usd``, in input order."""
    rates = MONTH_END_RATES.get(period)
    if rates is None:
        raise ValueError(
            f"no month-end FX table for period {period!r} "
            f"(have: {', '.join(sorted(MONTH_END_RATES))}); ask finance for the rates"
        )
    expenses = []
    for line in submissions:
        currency = line["currency"].upper()
        rate = rates.get(currency)
        if rate is None:
            raise ValueError(
                f"expense {line['id']}: no {period} month-end rate for {currency}; "
                "ask finance for the rate"
            )
        # Convert first, round once: rounding the source amount first
        # would compound error on high-denomination currencies like JPY.
        amount_usd = to_cents(Decimal(str(line["amount"])) * rate)
        expenses.append(
            {**line, "currency": currency, "rate": float(rate), "amount_usd": float(amount_usd)}
        )
    return {"expenses": expenses}
