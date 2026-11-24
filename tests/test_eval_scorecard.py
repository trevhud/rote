"""Scorecard price arithmetic and rendering — all offline.

Prices here are synthetic fixtures exercising the math; the live
fetch path is covered separately with mocked HTTP.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from rote.eval.empirical import MeasuredRun, measured_run_record
from rote.eval.estimate import (
    FanOutWidth,
    NodeEstimate,
    Range,
    estimate_compiled_skill,
    estimate_pipeline,
    estimate_skill,
    measured_agent_run,
)
from rote.eval.pricing import ModelPrice, ModelTier, PricingCatalog, PricingError
from rote.eval.priors import Priors
from rote.eval.scorecard import agent_run_cost_usd, build_scorecard, pipeline_cost_usd
from rote.eval.sidecar import EvalEstimates, TurnRange
from rote.eval.tokens import HeuristicTokenCounter
from rote.ir import NodeKind, load_pipeline

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

    # The step counts on each side are different units; the row says so.
    (row,) = [ln for ln in md.splitlines() if ln.startswith("| Steps decided by a sampled LLM")]
    before, after, change = (cell.strip() for cell in row.split("|")[2:5])
    assert before.endswith(" agent steps")
    assert " pipeline nodes (" in after
    assert change.startswith("different units")

    payload = card.to_dict()
    json.dumps(payload)  # must be JSON-serializable
    assert payload["pipeline"] == pipeline.name
    assert payload["before"] is not None
    assert payload["roteness"] == pe.sampling.roteness
    assert 0.0 <= payload["roteness"] <= 1.0  # type: ignore[operator]
    costs = payload["costs"]
    assert isinstance(costs, list) and len(costs) == 2
    assert costs[0]["before_usd"]["low"] > 0  # type: ignore[index]


def _run(
    turns: int | None, read: int, wrote: int, out: int, seconds: float, flags: tuple[str, ...] = ()
) -> dict[str, object]:
    return measured_run_record(
        MeasuredRun(
            wall_seconds=seconds,
            output={},
            turns=turns,
            input_tokens=4,
            cache_read_tokens=read,
            cache_creation_tokens=wrote,
            output_tokens=out,
            model="claude-test-1",
            flags=flags,
        )
    )


def test_a_measured_baseline_replaces_the_agent_side_estimate(estimates) -> None:  # type: ignore[no-untyped-def]
    pipeline, pe, se = estimates
    runs = [
        _run(3, 14_000, 13_000, 5_000, 40.0),
        _run(4, 20_000, 14_000, 5_500, 45.0),
        _run(5, 30_000, 15_000, 6_000, 55.0),
        # A truncated run is not the skill's cost, however cheap it looks.
        _run(1, 10, 10, 10, 2.0, flags=("hit_max_turns",)),
    ]
    measured = measured_agent_run(se, runs)
    assert measured is not None and measured.measured
    # The estimate said 30 to 50 turns; the runs took 3 to 5.
    assert (measured.turns.low, measured.turns.high) == (3.0, 5.0)
    assert (measured.wall_seconds.low, measured.wall_seconds.high) == (40.0, 55.0)
    # Tokens are the median run's own, so no figure mixes two runs.
    assert (measured.fresh_input_tokens.low, measured.fresh_input_tokens.high) == (14_004.0,) * 2
    assert (measured.cached_read_tokens.low, measured.cached_read_tokens.high) == (20_000.0,) * 2
    assert (measured.output_tokens.low, measured.output_tokens.high) == (5_500.0,) * 2
    # Dollars follow the measurement at this price: write 12.5, read 1, output 50 per Mtok.
    cost = agent_run_cost_usd(measured, _price())
    assert cost.low == pytest.approx((14_004 * 12.5 + 20_000 * 1.0 + 5_500 * 50.0) / 1e6)
    assert cost.low < agent_run_cost_usd(se, _price()).low / 2

    card = build_scorecard(
        pipeline_name=pipeline.name,
        pipeline_estimate=pe,
        skill_estimate=measured,
        prices=[_price()],
        priors=Priors(),
        generated_at="2026-07-03T00:00:00Z",
    )
    md = card.to_markdown()
    assert "the agent side is a measured baseline run" in md
    assert "nothing was executed" not in md
    assert "measured, 3 baseline run(s) of the skill on claude-test-1" in md
    # The static model's own wording describes none of these numbers.
    assert "Turn estimate" not in md and "transcript growth" not in md
    assert card.to_dict()["before"]["measured"] is True  # type: ignore[index]
    # A run that reported no usage measures nothing, and the estimate stands.
    assert measured_agent_run(se, [_run(None, 0, 0, 0, 1.0)]) is None
    static = build_scorecard(
        pipeline_name=pipeline.name,
        pipeline_estimate=pe,
        skill_estimate=se,
        prices=[_price()],
        priors=Priors(),
        generated_at="2026-07-03T00:00:00Z",
    ).to_markdown()
    assert "nothing was executed" in static and "Turn estimate" in static


def test_the_scorecard_builder_and_rote_eval_use_a_saved_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from rote.eval import build_scorecard_for, pricing
    from rote.eval.baseline import recorded_runs

    monkeypatch.setattr(
        pricing,
        "fetch_catalog",
        lambda provider: SimpleNamespace(sample=lambda provider: [_price()]),
    )
    good, flagged = (
        _run(3, 14_000, 13_000, 5_000, 40.0),
        _run(1, 10, 10, 10, 2.0, flags=("errored",)),
    )
    (tmp_path / "baseline").mkdir()
    (tmp_path / "baseline" / "metrics.json").write_text(json.dumps({"runs": [flagged, good]}))
    assert recorded_runs(tmp_path) == [flagged, good]
    assert recorded_runs(tmp_path / "nowhere") == []

    def card(runs: list[dict[str, object]]):  # type: ignore[no-untyped-def]
        return build_scorecard_for(
            load_pipeline(BDR_PIPELINE), BDR_PIPELINE, BDR_SKILL, "test", measured_runs=runs
        )

    measured = card(recorded_runs(tmp_path)).skill
    assert measured is not None and measured.measured
    assert (measured.turns.low, measured.turns.high) == (3.0, 3.0)
    # The skill runtime prices the compiled side as a conversation, from
    # the plan's own stops: BDR has 2 loops, 2 gates, 3 DO blocks and 1 DECIDE.
    as_skill = build_scorecard_for(
        load_pipeline(BDR_PIPELINE), BDR_PIPELINE, BDR_SKILL, "test", runtime="skill"
    )
    assert as_skill.compiled_skill is not None
    assert as_skill.compiled_skill.turns.low == 2 + 2 * 3 + 2 * 2 + 3 * 2 + 1
    # With --judges helper the DECIDE block is gone, as in compile-report.md.
    helped = build_scorecard_for(
        load_pipeline(BDR_PIPELINE),
        BDR_PIPELINE,
        BDR_SKILL,
        "test",
        runtime="skill",
        helper_judges=True,
    )
    assert helped.compiled_skill is not None
    assert helped.compiled_skill.turns.low == as_skill.compiled_skill.turns.low - 1
    assert card([]).compiled_skill is None
    # Only a flagged run, or none: the static estimate, as before.
    for runs in ([flagged], []):
        static = card(runs).skill
        assert static is not None and not static.measured and static.turns.low > 3


def _node(node_id: str, kind: NodeKind, calls: float, tin: float, tout: float) -> NodeEstimate:
    return NodeEstimate(
        node_id=node_id,
        kind=kind,
        calls=Range.exact(calls),
        llm_input_tokens_per_call=Range.exact(tin),
        llm_output_tokens_per_call=Range.exact(tout),
        wall_seconds_per_call=1.0,
    )


def test_a_compiled_skill_is_priced_as_a_conversation_not_as_api_calls(estimates) -> None:  # type: ignore[no-untyped-def]
    pipeline, pe, se = estimates
    priors = Priors()
    base = float(priors.system_overhead_tokens)
    tool = priors.tokens_per_external_call_result
    judge, fanned, do, loop, helper = (
        _node("judge", NodeKind.LLM_JUDGE, 1, 1000, 100),
        _node("fanned", NodeKind.LLM_JUDGE, 25, 200, 20),
        _node("do", NodeKind.EXTERNAL_CALL, 1, 0, 0),
        _node("loop", NodeKind.AGENT_LOOP, 4, 5000, 250),
        _node("helper", NodeKind.LLM_JUDGE, 1, 700, 70),
    )

    def run(stops: list[dict[str, object]], *nodes: NodeEstimate):  # type: ignore[no-untyped-def]
        shaped = replace(
            pe,
            nodes=list(nodes),
            fan_out_widths=(FanOutWidth("fanned", Range(25.0, 25.0), "test"),),
        )
        return estimate_compiled_skill(stops, shaped, priors)

    # No stops: a start and a finish, each reading the fixed context.
    bare = run([])
    assert (bare.turns.low, bare.turns.high) == (2.0, 2.0)
    assert bare.fresh_input_tokens.low == base and bare.cached_read_tokens.low == base

    # One DECIDE block: one more request; what it showed is re-read by half of the rest.
    one = run([{"block": "DECIDE", "steps": ["judge"]}], judge)
    assert one.turns.low == 3.0
    assert one.fresh_input_tokens.low == base + 1100
    assert one.cached_read_tokens.low == pytest.approx(2 * (base + 550))
    assert one.output_tokens.low == 100 + 3 * priors.output_tokens_per_turn

    def requests(stop: dict[str, object], *nodes: NodeEstimate) -> tuple[float, float]:
        turns = run([stop], *nodes).turns
        return turns.low - 2, turns.high - 2

    # A question asked ahead is one more request; a HUMAN block is two.
    assert requests({"block": "DECIDE", "steps": ["judge"], "ahead": "g"}, judge) == (2.0, 2.0)
    assert requests({"block": "HUMAN", "steps": ["g"]}) == (2.0, 2.0)
    # A conditional gate may never ask.
    assert requests({"block": "HUMAN", "steps": ["g"], "conditional": True}) == (0.0, 2.0)
    # A DO block is the agent's tool call and then its answer, and the tool's
    # result enters the conversation.
    assert requests({"block": "DO", "steps": ["do"]}, do) == (2.0, 2.0)
    assert run([{"block": "DO", "steps": ["do"]}], do).fresh_input_tokens.low == base + tool
    # 25 fanned items are three blocks of at most ten.
    assert requests({"block": "DECIDE", "steps": ["fanned"], "fanned": True}, fanned) == (3.0, 3.0)
    # A loop costs its own turns, and its tokens, which already hold their
    # re-reads, are added once and never re-read again.
    looped = run([{"block": "LOOP", "steps": ["loop"]}], loop)
    loop_turns = 4 * priors.agent_loop_turns_per_iteration
    assert looped.turns.low == 2 + loop_turns
    assert looped.fresh_input_tokens.low == base
    assert looped.cached_read_tokens.low == pytest.approx((1 + loop_turns) * base + 4 * 5000)
    assert looped.output_tokens.low == 4 * 250 + 2 * priors.output_tokens_per_turn
    # A judge no block asks (--judges helper) is an API call: read once.
    api = run([], helper)
    assert (api.turns.low, api.fresh_input_tokens.low) == (2.0, base + 700)
    assert api.cached_read_tokens.low == base and api.output_tokens.low == 70 + 2 * 250

    run = one  # type: ignore[assignment]

    def card(compiled):  # type: ignore[no-untyped-def]
        return build_scorecard(
            pipeline_name=pipeline.name,
            pipeline_estimate=pe,
            skill_estimate=se,
            prices=[_price()],
            priors=priors,
            generated_at="2026-07-03T00:00:00Z",
            compiled_skill=compiled,
        )

    skill, workflow = card(run), card(None)
    assert skill.after_llm_tokens.low == 3 * base + 1100 + 2 * 550 + one.output_tokens.low
    assert skill.costs[0].after_usd.low == agent_run_cost_usd(run, _price()).low
    assert workflow.costs[0].after_usd.low == pipeline_cost_usd(pe, _price()).low
    md = skill.to_markdown()
    assert "| Metric | As agent instructions | As compiled skill | Change |" in md
    assert "- Compiled skill: 3 requests" in md
    assert "asked in the conversation" in md and "durable" not in md
    assert "durable" in workflow.to_markdown()
    assert "As compiled pipeline" in workflow.to_markdown()
    assert "Compiled skill:" not in workflow.to_markdown()
    assert skill.to_dict()["after"]["compiled_skill"]["requests"] == {"low": 3.0, "high": 3.0}  # type: ignore[index]
    assert workflow.to_dict()["after"]["compiled_skill"] is None  # type: ignore[index]


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
