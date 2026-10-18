"""Live Cloudflare Workflows prices, and what one pipeline run costs in them.

Cloudflare began billing Workflows steps and storage on 2026-08-10, so a
pipeline that runs on the ``cloudflare`` runtime (and on rote cloud, which
runs Cloudflare Dynamic Workflows) pays infrastructure per run on top of
its judge tokens.

Cloudflare publishes no price API. The prices live in one markdown table
in the docs repo, ``src/content/partials/workflows/workflows-pricing.mdx``,
which renders on both the Workflows and the Workers pricing pages. That
file is fetched live, cached for the same 24h as the model catalog, and
parsed strictly: a missing row or a cell whose wording no longer parses
is a :class:`~rote.eval.pricing.PricingError`, never a guessed number.
Its wording has changed before (2026-09-09, "Update unit for included
GB-month"), so the parser is the canary, not a formality.
"""

from __future__ import annotations

import contextlib
import json
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from rote.eval.estimate import FanOutWidth, Range, fan_out_width
from rote.eval.pricing import (
    DEFAULT_CACHE_TTL_SECONDS,
    PricingError,
    default_cache_dir,
    get_text,
)
from rote.eval.priors import Priors
from rote.ir import NodeKind, Pipeline

WORKFLOWS_PRICING_REPO = "cloudflare/cloudflare-docs"
WORKFLOWS_PRICING_REF = "production"
WORKFLOWS_PRICING_PATH = "src/content/partials/workflows/workflows-pricing.mdx"
WORKFLOWS_PRICING_URL = (
    f"https://raw.githubusercontent.com/{WORKFLOWS_PRICING_REPO}/"
    f"{WORKFLOWS_PRICING_REF}/{WORKFLOWS_PRICING_PATH}"
)

#: The paid-plan column. Workflows on the free plan are never billed
#: beyond their allowances, so only this column has a marginal price.
_PAID_COLUMN = "Workers Paid"

#: Row key -> the prefix its first cell starts with in the table.
_ROWS = {
    "requests": "Requests",
    "cpu_ms": "CPU time",
    "storage_gb_month": "Storage",
    "steps": "Steps",
}

_NUMBER = r"\d[\d,]*(?:\.\d+)?"
#: Unit words a cell may name, before "included" and after the price. A
#: closed set on purpose: anything else ("100K steps", "1M") is wording
#: this parser has not seen, and guessing at it is how "100K" became 100.
_UNIT = r"(?:CPU milliseconds|GB-month)"

# The whole cell, anchored at both ends. Matches all four paid cells as
# published:
#   "500,000 included per month + $0.80/ additional 100,000 per month"
#   "10 million included per month + $0.30 per additional million"
#   "30 million CPU milliseconds included per month + $0.02 per additional
#    million CPU milliseconds"
#   "1 GB-month included + $0.20/ GB-month"
_PAID_CELL = re.compile(
    rf"(?P<included>{_NUMBER})(?:\s+(?P<included_scale>million))?(?:\s+{_UNIT})?"
    r"\s+included(?:\s+per\s+month)?\s*\+\s*"
    rf"\$(?P<usd>{_NUMBER})\s*(?:/|per)\s*(?:additional\s+)?"
    rf"(?:(?P<per>{_NUMBER})\s*)?(?P<per_scale>million)?"
    rf"(?:\s*{_UNIT})?(?:\s*per\s+month)?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class MeteredRate:
    """One paid-plan line: a monthly allowance, then a price per quantity."""

    included_per_month: float
    usd: float
    per: float
    cell: str
    """The table cell verbatim, so the scorecard can show what was parsed."""

    @property
    def usd_per_unit(self) -> float:
        return self.usd / self.per


@dataclass(frozen=True)
class WorkflowsPricing:
    requests: MeteredRate
    cpu_ms: MeteredRate
    storage_gb_month: MeteredRate
    steps: MeteredRate
    source: str
    fetched_at: str


def _number(raw: str, scale: str | None) -> float:
    value = float(raw.replace(",", ""))
    return value * 1_000_000 if scale else value


def _parse_paid_cell(row: str, cell: str) -> MeteredRate:
    match = _PAID_CELL.fullmatch(cell.strip())
    if match is None:
        raise PricingError(
            f"{WORKFLOWS_PRICING_PATH}: the {row!r} paid-plan cell no longer parses: {cell!r}"
        )
    per_raw = match.group("per")
    per_scale = match.group("per_scale")
    per = _number(per_raw, per_scale) if per_raw else (1_000_000.0 if per_scale else 1.0)
    return MeteredRate(
        included_per_month=_number(match.group("included"), match.group("included_scale")),
        usd=float(match.group("usd").replace(",", "")),
        per=per,
        cell=cell,
    )


def _table_rows(mdx: str) -> list[list[str]]:
    rows = []
    for line in mdx.splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            rows.append([cell.strip() for cell in stripped[1:-1].split("|")])
    return rows


def parse_workflows_pricing(mdx: str, *, source: str, fetched_at: str) -> WorkflowsPricing:
    """Parse the Workflows pricing table. Every row must be there and parse."""
    rows = _table_rows(mdx)
    if not rows:
        raise PricingError(f"{WORKFLOWS_PRICING_PATH}: no markdown table found")
    header = rows[0]
    if _PAID_COLUMN not in header:
        raise PricingError(
            f"{WORKFLOWS_PRICING_PATH}: no {_PAID_COLUMN!r} column (header was {header!r})"
        )
    paid = header.index(_PAID_COLUMN)
    rates: dict[str, MeteredRate] = {}
    for key, label in _ROWS.items():
        found = [r for r in rows[1:] if r and r[0].startswith(label)]
        if len(found) != 1:
            raise PricingError(
                f"{WORKFLOWS_PRICING_PATH}: expected one {label!r} row, found {len(found)}"
            )
        if len(found[0]) <= paid:
            raise PricingError(f"{WORKFLOWS_PRICING_PATH}: the {label!r} row has no paid cell")
        rates[key] = _parse_paid_cell(label, found[0][paid])
    return WorkflowsPricing(**rates, source=source, fetched_at=fetched_at)


def _default_cache_path() -> Path:
    return default_cache_dir() / "cloudflare-workflows-pricing.json"


def fetch_workflows_pricing(
    *,
    timeout: float = 15.0,
    cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
    cache_path: Path | None = None,
) -> WorkflowsPricing:
    """Fetch and parse the live table, with the model catalog's 24h cache.

    The cache holds the raw file, not parsed numbers, so a cached copy
    goes through the same strict parser as a fresh one.
    """
    path = cache_path or _default_cache_path()
    with contextlib.suppress(Exception):
        cached = json.loads(path.read_text(encoding="utf-8"))
        if time.time() - float(cached["cached_at_epoch"]) <= cache_ttl_seconds:
            return parse_workflows_pricing(
                str(cached["mdx"]), source=_source(), fetched_at=str(cached["fetched_at"])
            )
    mdx = get_text(WORKFLOWS_PRICING_URL, timeout)
    fetched_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    pricing = parse_workflows_pricing(mdx, source=_source(), fetched_at=fetched_at)
    with contextlib.suppress(OSError):  # a read-only cache dir shouldn't break eval
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"cached_at_epoch": time.time(), "fetched_at": fetched_at, "mdx": mdx}),
            encoding="utf-8",
        )
    return pricing


def _source() -> str:
    return f"{WORKFLOWS_PRICING_REPO}@{WORKFLOWS_PRICING_REF}:{WORKFLOWS_PRICING_PATH}"


# ───────── One run, in Cloudflare's billing units ─────────


def cloudflare_steps_per_run(
    pipeline: Pipeline, widths: Mapping[str, FanOutWidth], priors: Priors
) -> Range:
    """Billed steps for one run of the emitted Cloudflare workflow.

    One ``step.do`` per top-level node, one per element of a fan_out node,
    and one ``step.waitForEvent`` per HITL gate. ``widths`` are the fan
    widths the token estimate already used (the compiler's ``eval.yaml``
    when it records one, else the stated prior), so the step count and the
    LLM dollars assume the same number of elements. Counting a gate's wait
    as a step is a conservative assumption; Cloudflare's docs do not say.
    ``loop_body`` sub-nodes run inside their agent loop's step, retries
    are not billed as steps, and the extra wait a parked MCP step adds
    only happens when it parks.
    """
    nested = {sub for node in pipeline.nodes for sub in node.loop_body or []}
    steps = Range.exact(0)
    for node in pipeline.nodes:
        if node.id in nested:
            continue
        if node.fan_out and node.kind is not NodeKind.HITL_GATE:
            width = widths.get(node.id) or fan_out_width(node, priors, None)
            steps = steps + width.elements
        else:
            steps = steps + Range.exact(1)
    return steps


@dataclass(frozen=True)
class InfraLine:
    label: str
    quantity: Range
    unit: str
    rate: MeteredRate

    @property
    def usd(self) -> Range:
        return self.quantity.scale(self.rate.usd_per_unit)


@dataclass(frozen=True)
class InfraCost:
    """Per-run Cloudflare Workflows cost at the paid plan's marginal rates."""

    runtime_label: str
    steps_per_run: Range
    lines: tuple[InfraLine, ...]
    pricing: WorkflowsPricing
    priors: Priors
    fan_out_widths: tuple[FanOutWidth, ...] = ()

    @property
    def usd(self) -> Range:
        total = Range.exact(0)
        for line in self.lines:
            total = total + line.usd
        return total

    @property
    def runs_within_included_steps(self) -> Range:
        """Monthly runs inside the included steps: most at the low step count."""
        included = self.pricing.steps.included_per_month
        return Range(
            float(int(included // max(self.steps_per_run.high, 1))),
            float(int(included // max(self.steps_per_run.low, 1))),
        )


def cloudflare_infra_cost(
    pipeline: Pipeline,
    pricing: WorkflowsPricing,
    priors: Priors,
    *,
    runtime_label: str,
    fan_out_widths: tuple[FanOutWidth, ...] = (),
) -> InfraCost:
    widths = {w.node_id: w for w in fan_out_widths}
    steps = cloudflare_steps_per_run(pipeline, widths, priors)
    gb_months = steps.scale(
        priors.cloudflare_state_bytes_per_step / 1e9 * priors.cloudflare_retention_days / 30
    )
    lines = (
        InfraLine("Steps", steps, "steps", pricing.steps),
        InfraLine("Requests", Range.exact(1), "request", pricing.requests),
        InfraLine(
            "CPU time", steps.scale(priors.cloudflare_cpu_ms_per_step), "CPU ms", pricing.cpu_ms
        ),
        InfraLine("Storage", gb_months, "GB-month", pricing.storage_gb_month),
    )
    return InfraCost(
        runtime_label=runtime_label,
        steps_per_run=steps,
        lines=lines,
        pricing=pricing,
        priors=priors,
        fan_out_widths=tuple(w for w in fan_out_widths if w.node_id in widths),
    )
