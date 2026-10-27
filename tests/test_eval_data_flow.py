"""A compiled model node pays for the data its ``inputs:`` bind (#294).

The static evaluator charged the original skill for every external result
(``tokens_per_external_call_result`` or its per-tool override) but sized a
compiled judge's input as one ``tokens_per_input_field`` per schema field,
so a judge that receives a whole external result looked tiny, and raising
the payload prior raised only the projected savings. These tests pin the
token values (never an ordering of a quantity against itself) along the
IR's data flow: whole results, projections, unknown transforms, fan-out
elements, lists, and the scorecard's statement of each size.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from rote.cli import main
from rote.eval.estimate import (
    NodeEstimate,
    PipelineEstimate,
    Range,
    estimate_pipeline,
    external_call_payload_tokens,
)
from rote.eval.pricing import ModelPrice, ModelTier, PricingCatalog
from rote.eval.priors import Priors
from rote.eval.scorecard import build_scorecard
from rote.eval.tokens import HeuristicTokenCounter
from rote.ir import Pipeline

COUNTER = HeuristicTokenCounter()
PROMPT = "Summarize the report for the request: {{ report }} {{ request }}"
FIELD = Priors().tokens_per_input_field
OUT_FIELD = Priors().tokens_per_output_field


def _fetch(**extra: Any) -> dict[str, Any]:
    return {
        "id": "fetch",
        "kind": "external_call",
        "description": "fetch the report",
        "mcp": {"server": "reports", "tool": "get_report"},
        **extra,
    }


def _judge(node_id: str, inputs: dict[str, str], **extra: Any) -> dict[str, Any]:
    return {
        "id": node_id,
        "kind": "llm_judge",
        "description": "judge",
        "inputs": inputs,
        "signature_spec": {
            "input_schema": {
                "type": "object",
                "properties": {name: {} for name in inputs},
            },
            "output_schema": {"type": "object", "properties": {"verdict": {"type": "string"}}},
            "prompt": PROMPT,
        },
        **extra,
    }


def _pipeline(nodes: list[dict[str, Any]], edges: list[tuple[str, str]]) -> Pipeline:
    return Pipeline.model_validate(
        {
            "name": "flow",
            "input": {"type": "Req", "required": ["topic"]},
            "nodes": nodes,
            "edges": [{"from": a, "to": b} for a, b in edges],
        }
    )


def _whole_result_pipeline() -> Pipeline:
    """The issue's shape: an external result passed whole into a judge that
    also receives the whole request."""
    return _pipeline(
        [_fetch(), _judge("summarize", {"report": "fetch.output", "request": "pipeline.input"})],
        [("fetch", "summarize")],
    )


def _node(est: PipelineEstimate, node_id: str) -> NodeEstimate:
    (found,) = [n for n in est.nodes if n.node_id == node_id]
    return found


PROMPT_TOKENS = COUNTER.count(PROMPT)


# ───────── whole results ─────────


@pytest.mark.parametrize("payload", [6_000.0, 60_000.0])
def test_a_judge_receiving_a_whole_external_result_pays_the_prior(payload: float) -> None:
    priors = Priors(tokens_per_external_call_result=payload)
    judge = _node(estimate_pipeline(_whole_result_pipeline(), COUNTER, priors), "summarize")
    # prompt + the whole result + the request (one declared input field).
    assert judge.llm_input_tokens == Range.exact(PROMPT_TOKENS + payload + FIELD)
    (report, request) = judge.input_sizes
    assert report.tokens == Range.exact(payload)
    assert "tokens_per_external_call_result" in report.basis
    assert request.tokens == Range.exact(FIELD)


def test_raising_the_upstream_prior_raises_the_judge_by_the_same_amount() -> None:
    def judge_input(payload: float) -> Range:
        priors = Priors(tokens_per_external_call_result=payload)
        est = estimate_pipeline(_whole_result_pipeline(), COUNTER, priors)
        return est.llm_input_tokens

    # The issue's reproduction: 6,000 vs 60,000 left the compiled side at
    # exactly the same number. It must move by the 54,000 the prior moved.
    assert judge_input(60_000.0).low - judge_input(6_000.0).low == 54_000.0
    assert judge_input(60_000.0).high - judge_input(6_000.0).high == 54_000.0


def test_a_per_tool_override_reaches_the_compiled_consumer_and_the_agent_side() -> None:
    priors = Priors(payload_tokens_per_tool={"get_report": 12_000.0})
    pipeline = _whole_result_pipeline()
    judge = _node(estimate_pipeline(pipeline, COUNTER, priors), "summarize")
    assert judge.input_sizes[0].tokens == Range.exact(12_000.0)
    assert "payload_tokens_per_tool.get_report" in judge.input_sizes[0].basis
    # The agent side is charged the same number for the same result.
    assert external_call_payload_tokens(pipeline, priors) == 12_000.0


def test_each_judge_pays_for_its_own_input_and_unrelated_judges_pay_nothing_extra() -> None:
    pipeline = _pipeline(
        [
            _fetch(),
            _judge("first", {"report": "fetch.output"}),
            _judge("second", {"report": "fetch.output"}),
            _judge("unrelated", {"request": "pipeline.input"}),
        ],
        [("fetch", "first"), ("fetch", "second")],
    )
    est = estimate_pipeline(pipeline, COUNTER, Priors(tokens_per_external_call_result=8_000.0))
    assert _node(est, "first").llm_input_tokens == Range.exact(PROMPT_TOKENS + 8_000.0)
    assert _node(est, "second").llm_input_tokens == Range.exact(PROMPT_TOKENS + 8_000.0)
    assert _node(est, "unrelated").llm_input_tokens == Range.exact(PROMPT_TOKENS + FIELD)
    assert est.llm_input_tokens == Range.exact(3 * PROMPT_TOKENS + 16_000.0 + FIELD)


def test_an_agent_loop_rereads_a_whole_result_on_every_turn() -> None:
    pipeline = _pipeline(
        [
            _fetch(),
            {
                "id": "investigate",
                "kind": "agent_loop",
                "description": "dig into the report",
                "tools": ["search"],
                "inputs": {"report": "fetch.output"},
                "termination": {"max_iterations": 2, "condition": "done"},
            },
        ],
        [("fetch", "investigate")],
    )
    priors = Priors(tokens_per_external_call_result=6_000.0)
    turns = priors.agent_loop_turns_per_iteration
    assert turns > 1, "a degenerate 1.0 would make this test unable to tell turns apart"
    loop = _node(estimate_pipeline(pipeline, COUNTER, priors), "investigate")
    per_call = turns * (6_000.0 + priors.transcript_growth_per_turn)
    assert loop.llm_input_tokens_per_call == Range.exact(per_call)
    assert loop.llm_input_tokens == Range(per_call, 2 * per_call)


def test_a_fanned_judge_over_a_whole_result_pays_it_once_in_total() -> None:
    pipeline = _pipeline(
        [_fetch(), _judge("score", {"report": "fetch.output"}, fan_out=True)],
        [("fetch", "score")],
    )
    priors = Priors(tokens_per_external_call_result=6_000.0, fan_out_elements_per_run=10)
    judge = _node(estimate_pipeline(pipeline, COUNTER, priors), "score")
    # Ten calls, each one element: a tenth of the result.
    assert judge.input_sizes[0].tokens == Range.exact(600.0)
    assert judge.llm_input_tokens == Range.exact(10 * (PROMPT_TOKENS + 600.0))


def test_a_fanned_external_call_is_charged_per_call_on_both_sides() -> None:
    pipeline = _pipeline(
        [
            {"id": "ids", "kind": "pure_function", "description": "ids", "impl": "a.py:ids"},
            _fetch(fan_out=True, inputs={"report_id": "ids.output"}),
            _judge("summarize", {"report": "fetch.output"}),
        ],
        [("ids", "fetch"), ("fetch", "summarize")],
    )
    priors = Priors(tokens_per_external_call_result=1_000.0, fan_out_elements_per_run=4)
    judge = _node(estimate_pipeline(pipeline, COUNTER, priors), "summarize")
    assert judge.input_sizes[0].tokens == Range.exact(4_000.0)
    assert external_call_payload_tokens(pipeline, priors) == 4_000.0


def test_a_passthrough_field_carries_the_whole_result_to_the_next_judge() -> None:
    report_schema: dict[str, Any] = {"type": "object"}
    classify = {
        "id": "classify",
        "kind": "llm_judge",
        "description": "classify the report",
        "inputs": {"report": "fetch.output"},
        "signature_spec": {
            "input_schema": {"type": "object", "properties": {"report": report_schema}},
            "output_schema": {
                "type": "object",
                "properties": {"report": report_schema, "label": {"type": "string"}},
            },
            "prompt": "Label {{ report }}",
            "passthrough": {"report": "report"},
        },
    }
    pipeline = _pipeline(
        [_fetch(), classify, _judge("summarize", {"report": "classify.output"})],
        [("fetch", "classify"), ("classify", "summarize")],
    )
    est = estimate_pipeline(pipeline, COUNTER, Priors(tokens_per_external_call_result=6_000.0))
    # The model writes only `label`; code copies the 6,000-token report on.
    assert _node(est, "classify").llm_output_tokens == Range.exact(OUT_FIELD)
    assert _node(est, "summarize").input_sizes[0].tokens == Range.exact(6_000.0 + OUT_FIELD)


# ───────── projections and unknown transforms ─────────


def test_a_projection_is_a_range_not_the_whole_result_or_one_field() -> None:
    pipeline = _pipeline(
        [
            _fetch(output_schema={"type": "object", "properties": {"count": {"type": "integer"}}}),
            _judge("body", {"report": "fetch.output.body"}),
            _judge("count", {"report": "fetch.output.count"}),
        ],
        [("fetch", "body"), ("fetch", "count")],
    )
    est = estimate_pipeline(pipeline, COUNTER, Priors(tokens_per_external_call_result=6_000.0))
    body = _node(est, "body").input_sizes[0]
    assert body.tokens == Range(FIELD, 6_000.0)
    assert "size unknown" in body.basis
    # A field the source declares an integer is one value, not the result.
    assert _node(est, "count").input_sizes[0].tokens == Range.exact(FIELD)


def test_a_consumer_that_declares_a_scalar_receives_one_value() -> None:
    judge = _judge("check", {"report": "fetch.output"})
    judge["signature_spec"]["input_schema"]["properties"]["report"] = {"type": "boolean"}
    pipeline = _pipeline([_fetch(), judge], [("fetch", "check")])
    est = estimate_pipeline(pipeline, COUNTER, Priors(tokens_per_external_call_result=6_000.0))
    (size,) = _node(est, "check").input_sizes
    # The judge's own schema refuses anything but a boolean, so a whole
    # 6,000-token result cannot reach the model.
    assert size.tokens == Range.exact(FIELD)
    assert "declared a scalar" in size.basis


def test_a_pure_function_between_does_not_pass_the_whole_size_silently() -> None:
    def through(output_schema: dict[str, Any] | None) -> Range:
        shrink: dict[str, Any] = {
            "id": "shrink",
            "kind": "pure_function",
            "description": "keep the headline",
            "impl": "a.py:shrink",
            "inputs": {"report": "fetch.output"},
        }
        if output_schema is not None:
            shrink["output_schema"] = output_schema
        pipeline = _pipeline(
            [_fetch(), shrink, _judge("summarize", {"report": "shrink.output"})],
            [("fetch", "shrink"), ("shrink", "summarize")],
        )
        est = estimate_pipeline(pipeline, COUNTER, Priors(tokens_per_external_call_result=6_000.0))
        return _node(est, "summarize").input_sizes[0].tokens

    # Undeclared output: anything from one value to all it read.
    assert through(None) == Range(FIELD, 6_000.0)
    # A declared, bounded output is sized from its schema.
    assert through({"type": "object", "properties": {"n": {"type": "integer"}}}) == Range.exact(
        FIELD
    )


def test_a_pipeline_with_no_model_nodes_reports_zero_model_tokens() -> None:
    pipeline = _pipeline(
        [
            _fetch(),
            {
                "id": "tally",
                "kind": "pure_function",
                "description": "tally",
                "impl": "a.py:tally",
                "inputs": {"report": "fetch.output"},
            },
        ],
        [("fetch", "tally")],
    )
    est = estimate_pipeline(pipeline, COUNTER, Priors(tokens_per_external_call_result=60_000.0))
    assert est.llm_input_tokens == Range.exact(0)
    assert est.llm_output_tokens == Range.exact(0)
    assert all(not n.input_sizes for n in est.nodes)


# ───────── lists and containers ─────────


def test_lists_and_free_form_objects_are_ranges_on_both_ends_of_a_judge() -> None:
    pipeline = Pipeline.model_validate(
        {
            "name": "shapes",
            "input": {
                "type": "Req",
                "input_schema": {
                    "type": "object",
                    "properties": {"options": {"type": "object"}},
                },
            },
            "nodes": [
                {
                    "id": "rank",
                    "kind": "llm_judge",
                    "description": "rank",
                    "inputs": {"options": "pipeline.input.options"},
                    "signature_spec": {
                        "input_schema": {
                            "type": "object",
                            "properties": {"options": {"type": "object"}},
                        },
                        "output_schema": {
                            "type": "object",
                            "properties": {
                                "ranked": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "name": {"type": "string"},
                                            "score": {"type": "number"},
                                        },
                                    },
                                }
                            },
                        },
                        "prompt": "Rank {{ options }}",
                    },
                }
            ],
            "edges": [],
        }
    )
    priors = Priors(elements_per_list=5)
    rank = _node(estimate_pipeline(pipeline, COUNTER, priors), "rank")
    # A free-form object: one value up to elements_per_list of them.
    assert rank.input_sizes[0].tokens == Range(FIELD, 5 * FIELD)
    # A list of two-field objects: one element up to five, never one scalar.
    assert rank.llm_output_tokens == Range(2 * OUT_FIELD, 10 * OUT_FIELD)


# ───────── the scorecard states the sizes ─────────

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
    fetched_at="2026-10-27T00:00:00Z",
)


def test_the_scorecard_states_each_input_size_and_its_basis() -> None:
    priors = Priors(tokens_per_external_call_result=60_000.0)
    est = estimate_pipeline(_whole_result_pipeline(), COUNTER, priors)
    card = build_scorecard(
        pipeline_name="flow",
        pipeline_estimate=est,
        skill_estimate=None,
        prices=[_PRICE],
        priors=priors,
        generated_at="2026-10-27T00:00:00Z",
    )
    (judge,) = [n for n in card.to_dict()["after"]["nodes"] if n["id"] == "summarize"]  # type: ignore[index]
    assert judge["llm_input_tokens_per_call"] == {
        "low": PROMPT_TOKENS + 60_000.0 + FIELD,
        "high": PROMPT_TOKENS + 60_000.0 + FIELD,
    }
    assert judge["input_sizes"][0] == {
        "param": "report",
        "ref": "fetch.output",
        "tokens": {"low": 60_000.0, "high": 60_000.0},
        "basis": "the whole output of `fetch` (prior tokens_per_external_call_result per call)",
    }
    markdown = card.to_markdown()
    assert (
        "- `summarize` input `report` (`fetch.output`): 60k tokens per call, the whole "
        "output of `fetch` (prior tokens_per_external_call_result per call)."
    ) in markdown
    assert "Static sizes, not measured usage." in markdown


# ───────── through the CLI ─────────


def test_rote_eval_json_moves_the_compiled_side_with_the_prior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "rote.eval.pricing.fetch_catalog", lambda **kwargs: PricingCatalog(prices=(_PRICE,))
    )
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("# Report\n\n1. Fetch the report.\n2. Summarize it.\n")
    compiled = tmp_path / "compiled"
    compiled.mkdir()
    (compiled / "pipeline.yaml").write_text(
        yaml.safe_dump(_whole_result_pipeline().model_dump(mode="json", by_alias=True))
    )

    def after(payload: int) -> dict[str, Any]:
        prior = f"tokens_per_external_call_result={payload}"
        rc = main(["eval", str(compiled), "--skill", str(skill), "--json", "--prior", prior])
        assert rc == 0
        return json.loads(capsys.readouterr().out)  # type: ignore[no-any-return]

    small, large = after(6_000), after(60_000)
    assert large["before"]["context_tokens"] - small["before"]["context_tokens"] == 54_000
    after_small, after_large = (
        small["after"]["llm_input_tokens"],
        large["after"]["llm_input_tokens"],
    )
    assert after_large["low"] - after_small["low"] == 54_000
    assert after_large["high"] - after_small["high"] == 54_000
