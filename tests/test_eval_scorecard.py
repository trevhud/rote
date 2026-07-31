"""Scorecard price arithmetic and rendering — all offline.

Prices here are synthetic fixtures exercising the math; the live
fetch path is covered separately with mocked HTTP.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rote.eval.estimate import estimate_pipeline, estimate_skill
from rote.eval.pricing import ModelPrice, ModelTier, PricingCatalog, PricingError
from rote.eval.priors import Priors
from rote.eval.scorecard import agent_run_cost_usd, build_scorecard, pipeline_cost_usd
from rote.eval.sidecar import EvalEstimates, TurnRange
from rote.eval.tokens import HeuristicTokenCounter
from rote.ir import load_pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
BDR_PIPELINE = REPO_ROOT / "examples" / "bdr-outreach" / "expected" / "pipeline.yaml"
BDR_SKILL = REPO_ROOT / "examples" / "bdr-outreach" / "skill"


def _price(
    *,
    model_id: str = "test-model-1",
    tier: ModelTier = ModelTier.FLAGSHIP,
    input_per_mtok: float = 10.0,
    output_per_mtok: float = 50.0,
    cache_read_per_mtok: float | None = 1.0,
    cache_write_per_mtok: float | None = 12.5,
) -> ModelPrice:
    return ModelPrice(
        model_id=model_id,
        provider="anthropic",
        display_name=model_id,
        tier=tier,
        input_per_mtok=input_per_mtok,
        output_per_mtok=output_per_mtok,
        cache_read_per_mtok=cache_read_per_mtok,
        cache_write_per_mtok=cache_write_per_mtok,
        source="https://example.test/prices",
        fetched_at="2026-07-03T00:00:00Z",
    )


@pytest.fixture(scope="module")
def estimates():  # type: ignore[no-untyped-def]
    counter = HeuristicTokenCounter()
    pipeline = load_pipeline(BDR_PIPELINE)
    pe = estimate_pipeline(pipeline, counter)
    se = estimate_skill(
        BDR_SKILL, counter, sidecar=EvalEstimates(totals=TurnRange(low=30, high=50))
    )
    return pipeline, pe, se


def test_agent_cost_is_the_cache_aware_sum(estimates) -> None:  # type: ignore[no-untyped-def]
    """Pin the dollars, not just the ordering.

    The comparison test below measures this function against *itself*,
    so it stays green when every rate is wrong by the same factor — a
    1000x unit slip, or the cache rates dropped for the plain input
    price. Both survived a mutation sweep. The four rates in `_price()`
    are distinct on purpose (12.5 write / 1.0 read / 50.0 output / 10.0
    input); only the correct pairing reproduces these numbers.
    """
    _, _, se = estimates
    cost = agent_run_cost_usd(se, _price())
    for bound in ("low", "high"):
        expected = (
            getattr(se.fresh_input_tokens, bound) * 12.5
            + getattr(se.cached_read_tokens, bound) * 1.0
            + getattr(se.output_tokens, bound) * 50.0
        ) / 1_000_000
        assert getattr(cost, bound) == pytest.approx(expected)
    # A model with no cache pricing bills both halves at plain input.
    plain = agent_run_cost_usd(se, _price(cache_read_per_mtok=None, cache_write_per_mtok=None))
    expected_plain = (
        (se.fresh_input_tokens.high + se.cached_read_tokens.high) * 10.0
        + se.output_tokens.high * 50.0
    ) / 1_000_000
    assert plain.high == pytest.approx(expected_plain)


def test_agent_cost_is_cache_aware(estimates) -> None:  # type: ignore[no-untyped-def]
    _, _, se = estimates
    with_cache = agent_run_cost_usd(se, _price())
    without_cache = agent_run_cost_usd(
        se, _price(cache_read_per_mtok=None, cache_write_per_mtok=None)
    )
    # Re-read context at 1/10th input price must be much cheaper than
    # billing every re-read token at full input price.
    assert with_cache.mid < without_cache.mid


def test_pipeline_cost_only_bills_llm_tokens(estimates) -> None:  # type: ignore[no-untyped-def]
    _, pe, _ = estimates
    cost = pipeline_cost_usd(pe, _price())
    expected_high = (pe.llm_input_tokens.high * 10.0 + pe.llm_output_tokens.high * 50.0) / 1_000_000
    assert cost.high == pytest.approx(expected_high)


def test_compilation_saves_money_on_bdr(estimates) -> None:  # type: ignore[no-untyped-def]
    """The product claim, as a regression test: for the canonical BDR
    skill, the compiled pipeline must estimate cheaper than the raw
    agent run at every point of the range, for the same model."""
    _, pe, se = estimates
    price = _price()
    assert pipeline_cost_usd(pe, price).high < agent_run_cost_usd(se, price).low


def test_scorecard_markdown_and_dict(estimates) -> None:  # type: ignore[no-untyped-def]
    pipeline, pe, se = estimates
    card = build_scorecard(
        pipeline_name=pipeline.name,
        pipeline_estimate=pe,
        skill_estimate=se,
        prices=[_price(), _price(model_id="test-model-2", tier=ModelTier.SMALL)],
        priors=Priors(),
        generated_at="2026-07-03T00:00:00Z",
    )

    md = card.to_markdown()
    assert "## Before → after" in md
    assert "## Cost per run (live official prices)" in md
    assert "## Assumptions (audit me)" in md
    assert "test-model-1" in md and "test-model-2" in md
    assert "https://example.test/prices" in md
    # The turn-estimate method must be auditable from the rendered card.
    assert "compiler sidecar (eval.yaml)" in md

    # Roteness is a first-class, purely-structural scorecard field.
    assert "| Roteness (deterministic steps) |" in md

    payload = card.to_dict()
    json.dumps(payload)  # must be JSON-serializable
    assert payload["pipeline"] == pipeline.name
    assert payload["before"] is not None
    assert payload["roteness"] == pe.sampling.roteness
    assert 0.0 <= payload["roteness"] <= 1.0  # type: ignore[operator]
    costs = payload["costs"]
    assert isinstance(costs, list) and len(costs) == 2
    assert costs[0]["before_usd"]["low"] > 0  # type: ignore[index]


def test_scorecard_without_skill_baseline(estimates) -> None:  # type: ignore[no-untyped-def]
    pipeline, pe, _ = estimates
    card = build_scorecard(
        pipeline_name=pipeline.name,
        pipeline_estimate=pe,
        skill_estimate=None,
        prices=[_price()],
        priors=Priors(),
        generated_at="2026-07-03T00:00:00Z",
    )
    md = card.to_markdown()
    assert "_no skill baseline_" in md
    assert card.to_dict()["before"] is None


def test_catalog_sample_picks_one_per_tier() -> None:
    catalog = PricingCatalog(
        prices=(
            _price(model_id="flag-old", input_per_mtok=8.0),
            _price(model_id="flag-new", input_per_mtok=10.0),
            _price(model_id="mid", tier=ModelTier.MID, input_per_mtok=3.0),
            _price(model_id="small", tier=ModelTier.SMALL, input_per_mtok=0.8),
        )
    )
    sample = catalog.sample("anthropic")
    assert [p.model_id for p in sample] == ["flag-new", "mid", "small"]


def test_catalog_sample_unknown_provider_raises() -> None:
    catalog = PricingCatalog(prices=(_price(),))
    with pytest.raises(PricingError):
        catalog.sample("acme-llm")
