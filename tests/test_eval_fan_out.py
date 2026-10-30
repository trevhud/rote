"""A fan_out node is priced per element, not as one call.

The estimator used to give every non-loop node ``calls = 1``, so a fanned
judge (BDR's ``personalize_email``, deal-monitor's ``classify_thread_step``)
understated the after-side dollars by the fan width. Dollars are asserted
outright, never as an ordering (CLAUDE.md, third mutation sweep).
"""

from __future__ import annotations

import pytest

from rote.eval.estimate import estimate_pipeline
from rote.eval.pricing import ModelPrice, ModelTier
from rote.eval.priors import Priors
from rote.eval.scorecard import build_scorecard, pipeline_cost_usd
from rote.eval.sidecar import EvalEstimates, StepEstimate, TurnRange
from rote.eval.tokens import HeuristicTokenCounter
from rote.ir import Pipeline

_PRICE = ModelPrice(
    model_id="test-model",
    provider="anthropic",
    display_name="test-model",
    tier=ModelTier.MID,
    input_per_mtok=3.0,
    output_per_mtok=15.0,
    cache_read_per_mtok=None,
    cache_write_per_mtok=None,
    source="https://example.test/prices",
    fetched_at="2026-10-18T00:00:00Z",
)


def _pipeline(*, fan_out: bool) -> Pipeline:
    return Pipeline.model_validate(
        {
            "name": "fan",
            "input": {"type": "In", "required": [], "optional": []},
            "nodes": [
                {
                    "id": "list_posts",
                    "kind": "pure_function",
                    "description": "list",
                    "impl": "posts.py:list_posts",
                },
                {
                    "id": "score_post",
                    "kind": "llm_judge",
                    "description": "score one post",
                    "fan_out": fan_out,
                    "inputs": {"post": "list_posts.output.posts"},
                    "signature_spec": {
                        "input_schema": {
                            "type": "object",
                            "properties": {"post": {"type": "string"}},
                        },
                        "output_schema": {
                            "type": "object",
                            "properties": {"score": {"type": "number"}},
                        },
                        "prompt": "Score this post from 1 to 10: {{ post }}",
                    },
                },
            ],
            "edges": [{"from": "list_posts", "to": "score_post", "fan_out": fan_out}],
            "entry_nodes": ["list_posts"],
            "exit_nodes": ["score_post"],
        }
    )


def _single_call_usd() -> float:
    """One call of the judge, from the unfanned twin of the same node."""
    single = estimate_pipeline(_pipeline(fan_out=False), HeuristicTokenCounter())
    (judge,) = [n for n in single.nodes if n.node_id == "score_post"]
    assert judge.calls.low == judge.calls.high == 1
    return (judge.llm_input_tokens.high * 3.0 + judge.llm_output_tokens.high * 15.0) / 1_000_000


def test_the_sidecar_iterations_set_the_fan_width() -> None:
    sidecar = EvalEstimates(
        steps=[
            StepEstimate(
                description="score each post",
                node_id="score_post",
                estimated_turns=TurnRange(low=1, high=2),
                iterations=TurnRange(low=4, high=6),
            )
        ]
    )
    est = estimate_pipeline(_pipeline(fan_out=True), HeuristicTokenCounter(), sidecar=sidecar)

    cost = pipeline_cost_usd(est, _PRICE)
    assert cost.low == pytest.approx(4 * _single_call_usd())
    assert cost.high == pytest.approx(6 * _single_call_usd())
    (width,) = est.fan_out_widths
    assert (width.node_id, width.elements.low, width.elements.high) == ("score_post", 4, 6)
    assert width.source == "eval.yaml iterations"


def test_without_a_sidecar_the_stated_prior_sets_it_never_one() -> None:
    est = estimate_pipeline(
        _pipeline(fan_out=True), HeuristicTokenCounter(), Priors(fan_out_elements_per_run=7)
    )

    cost = pipeline_cost_usd(est, _PRICE)
    assert cost.low == cost.high == pytest.approx(7 * _single_call_usd())


def test_elements_run_concurrently_so_wall_time_does_not_multiply() -> None:
    counter = HeuristicTokenCounter()
    fanned = estimate_pipeline(_pipeline(fan_out=True), counter, Priors(fan_out_elements_per_run=7))
    single = estimate_pipeline(_pipeline(fan_out=False), counter)
    assert fanned.critical_path_seconds == single.critical_path_seconds


def test_the_scorecard_states_the_width_and_its_source() -> None:
    est = estimate_pipeline(
        _pipeline(fan_out=True), HeuristicTokenCounter(), Priors(fan_out_elements_per_run=7)
    )
    card = build_scorecard(
        pipeline_name="fan",
        pipeline_estimate=est,
        skill_estimate=None,
        prices=[_PRICE],
        priors=Priors(fan_out_elements_per_run=7),
        generated_at="2026-10-18T00:00:00Z",
    )
    assert (
        "- fan_out `score_post`: 7 elements per run (prior fan_out_elements_per_run)"
        in card.to_markdown()
    )
    assert card.to_dict()["after"]["fan_out_widths"] == [  # type: ignore[index]
        {
            "node": "score_post",
            "elements": {"low": 7.0, "high": 7.0},
            "source": "prior fan_out_elements_per_run",
        }
    ]
