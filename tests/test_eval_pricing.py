"""Pricing catalog tests — all offline, using fixture payloads shaped
exactly like the live sources (models.dev api.json, OpenRouter
/api/v1/models) as verified on 2026-07-03."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from rote.eval.pricing import (
    ModelTier,
    PricingError,
    build_catalog,
    fetch_catalog,
)


def _models_dev(models: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {"anthropic": {"models": models}}


def _model(
    input_cost: float,
    output_cost: float,
    release_date: str | None,
    name: str = "",
    cache_read: float | None = None,
    cache_write: float | None = None,
) -> dict[str, Any]:
    cost: dict[str, float] = {"input": input_cost, "output": output_cost}
    if cache_read is not None:
        cost["cache_read"] = cache_read
    if cache_write is not None:
        cost["cache_write"] = cache_write
    entry: dict[str, Any] = {"name": name, "cost": cost}
    if release_date is not None:
        entry["release_date"] = release_date
    return entry


# The 2026-07 Anthropic lineup shape, including the trap this module
# exists to dodge: an old $15 flagship outliving a cheaper new one.
LINEUP = _models_dev(
    {
        "claude-fable-5": _model(10, 50, "2026-06-09", cache_read=1, cache_write=12.5),
        "claude-sonnet-5": _model(2, 10, "2026-06-30", cache_read=0.2, cache_write=2.5),
        "claude-opus-4-8": _model(5, 25, "2026-05-28"),
        "claude-opus-4-7": _model(5, 25, "2026-04-14"),
        "claude-sonnet-4-6": _model(3, 15, "2026-02-17"),
        "claude-haiku-4-5": _model(1, 5, "2025-10-15", cache_read=0.1, cache_write=1.25),
        "claude-haiku-4-5-20251001": _model(1, 5, "2025-10-15"),
        "claude-opus-4-1": _model(15, 75, "2025-08-05"),
        "claude-3-haiku-20240307": _model(0.25, 1.25, "2024-03-07"),
        "claude-embeddings-x": {"name": "no cost block"},
    }
)


def test_tier_detection_matches_current_lineup() -> None:
    catalog = build_catalog(LINEUP, None, provider="anthropic", fetched_at="t")
    sample = {p.tier: p.model_id for p in catalog.sample("anthropic")}
    # The old $15 Opus 4.1 must NOT beat the current-generation Fable 5,
    # despite being more expensive and inside the recency window.
    assert sample[ModelTier.FLAGSHIP] == "claude-fable-5"
    # Mid = the newest price point strictly between flagship and small.
    assert sample[ModelTier.MID] == "claude-sonnet-5"
    # Small = cheapest recent model; the 2024 model is out of window,
    # and the dated snapshot alias loses to the shorter id.
    assert sample[ModelTier.SMALL] == "claude-haiku-4-5"


def test_cache_prices_carried_through() -> None:
    catalog = build_catalog(LINEUP, None, provider="anthropic", fetched_at="t")
    flagship = catalog.sample("anthropic")[0]
    assert flagship.cache_read_per_mtok == 1
    assert flagship.cache_write_per_mtok == 12.5


def test_missing_provider_raises() -> None:
    with pytest.raises(PricingError):
        build_catalog(LINEUP, None, provider="acme-llm", fetched_at="t")


def test_cross_check_agreement_annotates_source() -> None:
    openrouter = {
        "data": [
            {
                "id": "anthropic/claude-fable-5",
                "pricing": {"prompt": "0.00001", "completion": "0.00005"},
            }
        ]
    }
    catalog = build_catalog(LINEUP, openrouter, provider="anthropic", fetched_at="t")
    flagship = catalog.sample("anthropic")[0]
    assert "cross-checked" in flagship.source
    assert "WARNING" not in flagship.source


def test_cross_check_disagreement_warns_in_source() -> None:
    openrouter = {
        "data": [
            {
                # 20% cheaper than models.dev says — beyond tolerance.
                "id": "anthropic/claude-fable-5",
                "pricing": {"prompt": "0.000008", "completion": "0.00005"},
            }
        ]
    }
    catalog = build_catalog(LINEUP, openrouter, provider="anthropic", fetched_at="t")
    flagship = catalog.sample("anthropic")[0]
    assert "WARNING" in flagship.source and "disagrees" in flagship.source


def test_openrouter_id_normalization() -> None:
    """OpenRouter's dotted ids must match models.dev's dashed ids.

    Uses a lineup where the dotted-id model IS the flagship, so the
    assertion always runs (a tier-less model never reaches the catalog).
    """
    lineup = _models_dev(
        {
            "claude-opus-4-8": _model(5, 25, "2026-05-28"),
            "claude-sonnet-4-6": _model(3, 15, "2026-02-17"),
            "claude-haiku-4-5": _model(1, 5, "2025-10-15"),
        }
    )
    openrouter = {
        "data": [
            {
                "id": "anthropic/claude-opus-4.8",
                "pricing": {"prompt": "0.000005", "completion": "0.000025"},
            }
        ]
    }
    catalog = build_catalog(lineup, openrouter, provider="anthropic", fetched_at="t")
    flagship = catalog.sample("anthropic")[0]
    assert flagship.model_id == "claude-opus-4-8"
    assert "cross-checked" in flagship.source


def test_malformed_openrouter_payload_is_nonfatal() -> None:
    catalog = build_catalog(LINEUP, {"unexpected": "shape"}, provider="anthropic", fetched_at="t")
    flagship = catalog.sample("anthropic")[0]
    # Catalog still built; no false cross-check claim.
    assert "cross-checked" not in flagship.source


def test_fetch_catalog_uses_fresh_cache_and_skips_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(url: str, timeout: float) -> Any:
        raise AssertionError("network must not be touched when cache is fresh")

    monkeypatch.setattr("rote.eval.pricing._get_json", _boom)

    cache = tmp_path / "pricing.json"
    catalog = build_catalog(LINEUP, None, provider="anthropic", fetched_at="t")
    from rote.eval.pricing import _store_cache

    _store_cache(cache, catalog, "anthropic")
    loaded = fetch_catalog(provider="anthropic", cache_path=cache)
    assert {p.model_id for p in loaded.prices} == {p.model_id for p in catalog.prices}


def test_fetch_catalog_refetches_when_cache_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def _fake_get(url: str, timeout: float) -> Any:
        calls.append(url)
        if "models.dev" in url:
            return LINEUP
        return {"data": []}

    monkeypatch.setattr("rote.eval.pricing._get_json", _fake_get)

    cache = tmp_path / "pricing.json"
    catalog = build_catalog(LINEUP, None, provider="anthropic", fetched_at="t")
    from rote.eval.pricing import _store_cache

    _store_cache(cache, catalog, "anthropic")
    # Age the cache beyond the TTL.
    payload = json.loads(cache.read_text())
    payload["cached_at_epoch"] = time.time() - 100_000
    cache.write_text(json.dumps(payload))

    fetch_catalog(provider="anthropic", cache_path=cache)
    assert any("models.dev" in c for c in calls)


def test_fetch_catalog_dies_loudly_when_primary_source_down(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _down(url: str, timeout: float) -> Any:
        raise PricingError(f"fetch of {url} failed: connection refused")

    monkeypatch.setattr("rote.eval.pricing._get_json", _down)
    with pytest.raises(PricingError):
        fetch_catalog(provider="anthropic", cache_path=tmp_path / "pricing.json")


def test_openrouter_outage_is_not_fatal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_get(url: str, timeout: float) -> Any:
        if "models.dev" in url:
            return LINEUP
        raise PricingError(f"fetch of {url} failed: connection refused")

    monkeypatch.setattr("rote.eval.pricing._get_json", _fake_get)
    catalog = fetch_catalog(provider="anthropic", cache_path=tmp_path / "pricing.json")
    flagship = catalog.sample("anthropic")[0]
    assert "cross-checked" not in flagship.source  # no false claim of verification
