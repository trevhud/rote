"""Live model + price discovery. Nothing here is hardcoded.

The catalog is fetched at eval time from public machine-readable
sources, cached briefly on disk, and errors loudly when unreachable —
a stale-or-wrong price silently baked into a wheel is worse than a
clear failure asking the user to retry online.

Sources (both verified live against each other and against provider
announcements before this module was written):

* **models.dev** (``https://models.dev/api.json``) — primary. Open
  dataset maintained against providers' official pricing docs; no
  auth; per-model ``cost`` in USD per million tokens including cache
  read/write, plus ``release_date`` (which drives recency detection).
* **OpenRouter** (``https://openrouter.ai/api/v1/models``) — live
  cross-check. No auth; prices in USD per token. When the two sources
  disagree on a sampled model beyond 2%, the scorecard's price source
  is annotated so the reader knows to double-check. (Empirically
  necessary: at Sonnet 5's launch a third catalog still carried the
  previous generation's price.)

Model *tiers* (flagship/mid/small) are detected from the data alone —
price ordering within the provider's recent lineup — never from model
name patterns, which churn every release.
"""

from __future__ import annotations

import contextlib
import json
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

MODELS_DEV_URL = "https://models.dev/api.json"
OPENROUTER_URL = "https://openrouter.ai/api/v1/models"

DEFAULT_CACHE_TTL_SECONDS = 24 * 3600
CROSS_CHECK_TOLERANCE = 0.02

# Both windows anchor on the provider's own newest release date (never
# the wall clock), so tier detection is a pure function of the fetched
# data. The wide window bounds the whole considered lineup; the narrow
# one defines the "current generation" a flagship must belong to —
# without it, a lingering previous-generation flagship (e.g. a $15/Mtok
# Opus 4.1 next to a $10/Mtok Fable 5) wins on price alone.
_RECENCY_WINDOW = timedelta(days=450)
_CURRENT_GENERATION_WINDOW = timedelta(days=150)


class PricingError(RuntimeError):
    """The live price source could not be fetched or parsed."""


class ModelTier(StrEnum):
    """Relative capability tier within one provider's current lineup.

    Detected from the pricing data itself (relative price ordering),
    never from hardcoded model-name lists — names churn every release;
    the price ordering is the durable signal.
    """

    FLAGSHIP = "flagship"
    MID = "mid"
    SMALL = "small"


@dataclass(frozen=True)
class ModelPrice:
    """One model's official per-token prices, as fetched."""

    model_id: str
    provider: str
    display_name: str
    tier: ModelTier
    input_per_mtok: float
    """USD per million input tokens."""
    output_per_mtok: float
    """USD per million output tokens."""
    cache_read_per_mtok: float | None
    cache_write_per_mtok: float | None
    source: str
    """Where this price was fetched from (URL), for the scorecard's
    assumptions section."""
    fetched_at: str
    """ISO-8601 UTC timestamp of the fetch."""


@dataclass(frozen=True)
class PricingCatalog:
    """A fetched set of current models with prices.

    ``prices`` holds the tier representatives the scorecard samples;
    ``reference_prices`` holds every priced model from the fetch
    (normalized id → (input, output) USD per Mtok) so measured usage
    from arbitrary judge models can be priced too.
    """

    prices: tuple[ModelPrice, ...]
    reference_prices: dict[str, tuple[float, float]] = field(default_factory=dict)

    def by_provider(self, provider: str) -> list[ModelPrice]:
        return [p for p in self.prices if p.provider == provider]

    def price_for(self, model_id: str) -> tuple[float, float] | None:
        """(input, output) USD per Mtok for any fetched model, else None."""
        return self.reference_prices.get(_normalize_model_id(model_id))

    def sample(self, provider: str = "anthropic") -> list[ModelPrice]:
        """The scorecard's model sampling: one model per tier, newest
        lineup, for the given provider. Ordered flagship → small."""
        chosen: list[ModelPrice] = []
        for tier in (ModelTier.FLAGSHIP, ModelTier.MID, ModelTier.SMALL):
            candidates = [p for p in self.by_provider(provider) if p.tier is tier]
            if candidates:
                # Most expensive input price within the tier = the
                # current-generation representative (older generations
                # are discounted or removed upstream).
                chosen.append(max(candidates, key=lambda p: p.input_per_mtok))
        if not chosen:
            raise PricingError(
                f"No priced models found for provider {provider!r} in the fetched catalog"
            )
        return chosen


# ───────── Fetching ─────────


def _get_json(url: str, timeout: float) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "rote-eval"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8"))
    except Exception as e:
        raise PricingError(f"fetch of {url} failed: {e}") from e


def _parse_release_date(raw: object) -> date | None:
    if not isinstance(raw, str):
        return None
    try:
        return date.fromisoformat(raw)
    except ValueError:
        return None


@dataclass(frozen=True)
class _RawModel:
    model_id: str
    display_name: str
    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float | None
    cache_write_per_mtok: float | None
    release_date: date | None


def _parse_models_dev(payload: Any, provider: str) -> list[_RawModel]:
    """Extract one provider's priced models from the models.dev dataset.

    models.dev ``cost`` values are already USD per million tokens.
    Models without both input and output prices (embeddings, free
    tiers, unpriced previews) are skipped.
    """
    if not isinstance(payload, dict):
        raise PricingError("models.dev payload is not a JSON object")
    provider_entry = payload.get(provider)
    if not isinstance(provider_entry, dict) or "models" not in provider_entry:
        raise PricingError(f"models.dev has no provider section {provider!r}")
    models = provider_entry["models"]
    if not isinstance(models, dict):
        raise PricingError(f"models.dev provider {provider!r} has a malformed models map")

    raw: list[_RawModel] = []
    for model_id, entry in models.items():
        if not isinstance(entry, dict):
            continue
        cost = entry.get("cost")
        if not isinstance(cost, dict):
            continue
        input_cost = cost.get("input")
        output_cost = cost.get("output")
        if not isinstance(input_cost, int | float) or not isinstance(output_cost, int | float):
            continue
        cache_read = cost.get("cache_read")
        cache_write = cost.get("cache_write")
        raw.append(
            _RawModel(
                model_id=str(model_id),
                display_name=str(entry.get("name", model_id)),
                input_per_mtok=float(input_cost),
                output_per_mtok=float(output_cost),
                cache_read_per_mtok=(
                    float(cache_read) if isinstance(cache_read, int | float) else None
                ),
                cache_write_per_mtok=(
                    float(cache_write) if isinstance(cache_write, int | float) else None
                ),
                release_date=_parse_release_date(entry.get("release_date")),
            )
        )
    return raw


def _normalize_model_id(model_id: str) -> str:
    """Comparable form across catalogs: no provider prefix, dots → dashes.

    OpenRouter writes ``anthropic/claude-opus-4.8``; models.dev (and
    Anthropic's own API) write ``claude-opus-4-8``.
    """
    bare = model_id.rsplit("/", 1)[-1].lower()
    return bare.replace(".", "-")


def _parse_openrouter(payload: Any) -> dict[str, tuple[float, float]]:
    """Normalized id → (input, output) USD per million tokens."""
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise PricingError("OpenRouter payload has no data list")
    out: dict[str, tuple[float, float]] = {}
    for entry in payload["data"]:
        if not isinstance(entry, dict):
            continue
        pricing = entry.get("pricing")
        model_id = entry.get("id")
        if not isinstance(pricing, dict) or not isinstance(model_id, str):
            continue
        try:
            prompt = float(pricing.get("prompt", ""))
            completion = float(pricing.get("completion", ""))
        except (TypeError, ValueError):
            continue
        # OpenRouter prices are USD per single token.
        out[_normalize_model_id(model_id)] = (prompt * 1_000_000, completion * 1_000_000)
    return out


# ───────── Tier detection ─────────


def _assign_tiers(models: list[_RawModel]) -> dict[str, ModelTier]:
    """Detect flagship/mid/small representatives from the data alone.

    1. Keep the provider's *recent* lineup: models released within
       ``_RECENCY_WINDOW`` of the provider's own newest release. If
       recency data is too thin, use everything.
    2. Deduplicate by (input, output) price pair, keeping the newest —
       serving aliases and dated snapshots share their price, and old
       generations often share the new generation's price point.
    3. **Flagship** = most expensive model of the *current generation*
       (released within ``_CURRENT_GENERATION_WINDOW`` of the anchor).
       Price alone would crown a lingering previous-gen flagship.
    4. **Small** = cheapest model in the recent lineup.
    5. **Mid** = the *newest* model priced strictly between the two —
       the provider's most recent mid-tier release — falling back to
       the median price point when nothing sits strictly between.
    """
    priced = [m for m in models if m.input_per_mtok > 0]
    dated = [m for m in priced if m.release_date is not None]
    anchor: date | None = None
    if len(dated) >= 3:
        anchor = max(m.release_date for m in dated if m.release_date is not None)
        recent = [
            m
            for m in dated
            if m.release_date is not None and anchor - m.release_date <= _RECENCY_WINDOW
        ]
    else:
        recent = priced

    by_price: dict[tuple[float, float], _RawModel] = {}
    for m in recent:
        key = (m.input_per_mtok, m.output_per_mtok)
        existing = by_price.get(key)
        if existing is None or _newer(m, existing):
            by_price[key] = m
    if len(by_price) < 3:
        # Too few distinct price points in the recent window — widen.
        for m in priced:
            key = (m.input_per_mtok, m.output_per_mtok)
            existing = by_price.get(key)
            if existing is None:
                by_price[key] = m

    candidates = list(by_price.values())
    tiers: dict[str, ModelTier] = {}
    if not candidates:
        return tiers

    if anchor is not None:
        current_gen = [
            m
            for m in candidates
            if m.release_date is not None and anchor - m.release_date <= _CURRENT_GENERATION_WINDOW
        ]
    else:
        current_gen = []
    flagship = max(current_gen or candidates, key=lambda m: m.input_per_mtok)
    tiers[flagship.model_id] = ModelTier.FLAGSHIP

    small = min(candidates, key=lambda m: m.input_per_mtok)
    if small.model_id != flagship.model_id:
        tiers[small.model_id] = ModelTier.SMALL

    between = [
        m for m in candidates if small.input_per_mtok < m.input_per_mtok < flagship.input_per_mtok
    ]
    if between:
        floor = date.min
        mid = max(between, key=lambda m: (m.release_date or floor, -len(m.model_id)))
        tiers[mid.model_id] = ModelTier.MID
    return tiers


def _newer(a: _RawModel, b: _RawModel) -> bool:
    if a.release_date and b.release_date and a.release_date != b.release_date:
        return a.release_date > b.release_date
    # Tie-break: the shorter id is the alias (`claude-haiku-4-5` vs its
    # dated snapshot) and the better display choice.
    return len(a.model_id) < len(b.model_id)


# ───────── The catalog builder ─────────


def build_catalog(
    models_dev_payload: Any,
    openrouter_payload: Any | None,
    *,
    provider: str,
    fetched_at: str,
) -> PricingCatalog:
    """Pure construction from already-fetched payloads (testable offline)."""
    raw_models = _parse_models_dev(models_dev_payload, provider)
    if not raw_models:
        raise PricingError(f"models.dev lists no priced models for provider {provider!r}")
    tiers = _assign_tiers(raw_models)

    cross: dict[str, tuple[float, float]] = {}
    if openrouter_payload is not None:
        cross = _parse_openrouter(openrouter_payload)

    prices: list[ModelPrice] = []
    for m in raw_models:
        tier = tiers.get(m.model_id)
        if tier is None:
            continue
        source = MODELS_DEV_URL
        checked = cross.get(_normalize_model_id(m.model_id))
        if checked is not None:
            in_ok = abs(checked[0] - m.input_per_mtok) <= CROSS_CHECK_TOLERANCE * m.input_per_mtok
            out_ok = (
                abs(checked[1] - m.output_per_mtok) <= CROSS_CHECK_TOLERANCE * m.output_per_mtok
            )
            if in_ok and out_ok:
                source = f"{MODELS_DEV_URL} (cross-checked against {OPENROUTER_URL})"
            else:
                source = (
                    f"{MODELS_DEV_URL} (WARNING: {OPENROUTER_URL} disagrees — "
                    f"${checked[0]:g}/${checked[1]:g} per Mtok there; verify manually)"
                )
        prices.append(
            ModelPrice(
                model_id=m.model_id,
                provider=provider,
                display_name=m.display_name,
                tier=tier,
                input_per_mtok=m.input_per_mtok,
                output_per_mtok=m.output_per_mtok,
                cache_read_per_mtok=m.cache_read_per_mtok,
                cache_write_per_mtok=m.cache_write_per_mtok,
                source=source,
                fetched_at=fetched_at,
            )
        )
    reference = {
        _normalize_model_id(m.model_id): (m.input_per_mtok, m.output_per_mtok) for m in raw_models
    }
    return PricingCatalog(prices=tuple(prices), reference_prices=reference)


# ───────── Disk cache + entry point ─────────


def _default_cache_path() -> Path:
    return Path.home() / ".cache" / "rote" / "pricing.json"


def _load_cache(path: Path, ttl_seconds: int, provider: str) -> PricingCatalog | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if time.time() - float(raw["cached_at_epoch"]) > ttl_seconds:
            return None
        if raw.get("provider") != provider:
            return None
        prices = tuple(
            ModelPrice(**{**entry, "tier": ModelTier(entry["tier"])}) for entry in raw["prices"]
        )
        reference = {
            str(k): (float(v[0]), float(v[1])) for k, v in raw.get("reference_prices", {}).items()
        }
        if not prices:
            return None
        return PricingCatalog(prices=prices, reference_prices=reference)
    except Exception:
        return None  # any unreadable/stale/foreign cache means refetch


def _store_cache(path: Path, catalog: PricingCatalog, provider: str) -> None:
    payload = {
        "cached_at_epoch": time.time(),
        "provider": provider,
        "prices": [asdict(p) | {"tier": p.tier.value} for p in catalog.prices],
        "reference_prices": {k: list(v) for k, v in catalog.reference_prices.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def fetch_catalog(
    *,
    provider: str = "anthropic",
    timeout: float = 15.0,
    cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
    cache_path: Path | None = None,
) -> PricingCatalog:
    """Fetch current models + official prices, with a short disk cache.

    Raises :class:`PricingError` when the primary source is
    unreachable — never falls back to a value baked into the package.
    The OpenRouter cross-check is best-effort: its absence downgrades
    the ``source`` annotation, not the catalog.
    """
    path = cache_path or _default_cache_path()
    cached = _load_cache(path, cache_ttl_seconds, provider)
    if cached is not None:
        return cached

    models_dev = _get_json(MODELS_DEV_URL, timeout)
    try:
        openrouter: Any | None = _get_json(OPENROUTER_URL, timeout)
    except PricingError:
        openrouter = None  # cross-check only; noted via the source string

    fetched_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    catalog = build_catalog(models_dev, openrouter, provider=provider, fetched_at=fetched_at)
    with contextlib.suppress(OSError):  # a read-only cache dir shouldn't break eval
        _store_cache(path, catalog, provider)
    return catalog
