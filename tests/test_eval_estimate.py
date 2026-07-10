"""Unit tests for the static eval estimators.

Everything here is offline: heuristic token counting, no pricing
fetches. The BDR expected/ pipeline and skill are the real-world
fixtures; small inline pipelines cover the structural edge cases.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rote.eval.estimate import (
    Range,
    estimate_pipeline,
    estimate_skill,
)
from rote.eval.priors import Priors
from rote.eval.sidecar import EvalEstimates, StepEstimate, TurnRange
from rote.eval.tokens import HeuristicTokenCounter
from rote.ir import NodeKind, Pipeline, load_pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
BDR_PIPELINE = REPO_ROOT / "examples" / "bdr-outreach" / "expected" / "pipeline.yaml"
BDR_SKILL = REPO_ROOT / "examples" / "bdr-outreach" / "skill"


@pytest.fixture(scope="module")
def bdr_pipeline() -> Pipeline:
    return load_pipeline(BDR_PIPELINE)


@pytest.fixture(scope="module")
def counter() -> HeuristicTokenCounter:
    return HeuristicTokenCounter()


# ───────── Range ─────────


def test_sampling_surface_roteness_is_pure_structural_math() -> None:
    """Roteness = deterministic steps / total steps — a function of node
    kinds only, never a model's estimate. 0 = pure agent, 1 = pure code."""
    from rote.eval.estimate import Range, SamplingSurface

    def surface(total: int, sampled: int) -> SamplingSurface:
        return SamplingSurface(
            total_steps=total,
            sampled_steps=sampled,
            schema_constrained_steps=sampled,
            sampled_output_tokens=Range(0.0, 0.0),
        )

    assert surface(4, 2).roteness == 0.5  # 2 code + 2 inference
    assert surface(3, 0).roteness == 1.0  # all deterministic
    assert surface(2, 2).roteness == 0.0  # all inference
    assert surface(0, 0).roteness == 0.0  # no divide-by-zero on empty


def test_range_rejects_inverted_bounds() -> None:
    with pytest.raises(ValueError):
        Range(2.0, 1.0)


def test_range_arithmetic() -> None:
    r = Range(1.0, 3.0) + Range(2.0, 4.0)
    assert (r.low, r.high) == (3.0, 7.0)
    assert Range(1.0, 3.0).scale(2).high == 6.0
    assert Range.exact(5.0).mid == 5.0


# ───────── Pipeline (after) side ─────────


def test_bdr_pipeline_estimate_shape(
    bdr_pipeline: Pipeline, counter: HeuristicTokenCounter
) -> None:
    est = estimate_pipeline(bdr_pipeline, counter)

    # Only sampled kinds contribute LLM tokens.
    for node_est in est.nodes:
        if node_est.kind in (NodeKind.PURE_FUNCTION, NodeKind.EXTERNAL_CALL, NodeKind.HITL_GATE):
            assert node_est.llm_input_tokens_per_call == 0
            assert node_est.llm_output_tokens_per_call == 0

    assert est.llm_input_tokens.high > est.llm_input_tokens.low >= 0
    assert est.critical_path_seconds.low > 0

    # Determinism surface: sampled = top-level judges + loops (loop_body
    # judges are costed inside their parent loop), all judges constrained.
    nested = {nid for n in bdr_pipeline.nodes if n.loop_body for nid in n.loop_body}
    judges = len([n for n in bdr_pipeline.nodes_by_kind(NodeKind.LLM_JUDGE) if n.id not in nested])
    gates = [n.id for n in bdr_pipeline.nodes_by_kind(NodeKind.HITL_GATE)]
    assert est.sampling.schema_constrained_steps == judges
    assert est.sampling.sampled_steps >= judges
    assert est.sampling.sampled_steps < est.sampling.total_steps
    assert set(est.hitl_gates) == set(gates)


def test_loop_body_subnodes_are_not_double_counted(
    bdr_pipeline: Pipeline, counter: HeuristicTokenCounter
) -> None:
    nested = {nid for n in bdr_pipeline.nodes if n.loop_body for nid in n.loop_body}
    if not nested:
        pytest.skip("BDR baseline has no loop_body nodes")
    est = estimate_pipeline(bdr_pipeline, counter)
    estimated_ids = {n.node_id for n in est.nodes}
    assert nested.isdisjoint(estimated_ids)


def test_unbounded_agent_loop_gets_prior_default_and_note(
    counter: HeuristicTokenCounter,
) -> None:
    pipeline = Pipeline.model_validate(
        {
            "name": "loop-only",
            "input": {"type": "In"},
            "nodes": [
                {
                    "id": "explore",
                    "kind": "agent_loop",
                    "description": "unbounded exploration",
                    "tools": ["search"],
                }
            ],
            "edges": [],
        }
    )
    priors = Priors()
    est = estimate_pipeline(pipeline, counter, priors)
    (node_est,) = est.nodes
    assert node_est.calls.high == priors.agent_loop_default_max_iterations
    assert node_est.note is not None and "no termination config" in node_est.note


def test_judge_tokens_scale_with_prompt_and_fields(counter: HeuristicTokenCounter) -> None:
    def judge_pipeline(prompt: str, out_fields: dict[str, object]) -> Pipeline:
        return Pipeline.model_validate(
            {
                "name": "judge-only",
                "input": {"type": "In"},
                "nodes": [
                    {
                        "id": "vet",
                        "kind": "llm_judge",
                        "description": "judge",
                        "signature_spec": {
                            "input_schema": {
                                "type": "object",
                                "properties": {"a": {"type": "string"}},
                            },
                            "output_schema": {"type": "object", "properties": out_fields},
                            "prompt": prompt,
                        },
                    }
                ],
                "edges": [],
            }
        )

    small = estimate_pipeline(judge_pipeline("Judge {{ a }}.", {"ok": {}}), counter)
    big = estimate_pipeline(
        judge_pipeline(
            "Judge {{ a }}. " + "Consider the rubric carefully. " * 50,
            {"ok": {}, "why": {}, "tier": {}},
        ),
        counter,
    )
    assert big.llm_input_tokens.high > small.llm_input_tokens.high
    assert big.llm_output_tokens.high > small.llm_output_tokens.high


# ───────── Skill (before) side ─────────


def test_bdr_skill_estimate_structural(counter: HeuristicTokenCounter) -> None:
    est = estimate_skill(BDR_SKILL, counter)
    assert est.turn_method.startswith("structural heuristic")
    # C0 must include the skill corpus on top of the system overhead.
    assert est.context_tokens > Priors().system_overhead_tokens
    assert est.turns.low >= 3
    # Cache-aware split: later turns re-read context, so cached >> fresh.
    assert est.cached_read_tokens.high > est.fresh_input_tokens.high
    # Before graduation every step is sampled and nothing is constrained.
    assert est.sampling.sampled_steps == est.sampling.total_steps
    assert est.sampling.schema_constrained_steps == 0


def test_sidecar_overrides_structural_heuristic(counter: HeuristicTokenCounter) -> None:
    sidecar = EvalEstimates(
        steps=[
            StepEstimate(description="a", estimated_turns=TurnRange(low=10, high=20)),
            StepEstimate(description="b", estimated_turns=TurnRange(low=5, high=10)),
        ]
    )
    est = estimate_skill(BDR_SKILL, counter, sidecar=sidecar)
    assert est.turn_method == "graduator sidecar (eval.yaml)"
    assert (est.turns.low, est.turns.high) == (15.0, 30.0)


def test_sidecar_totals_take_precedence(counter: HeuristicTokenCounter) -> None:
    sidecar = EvalEstimates(
        steps=[StepEstimate(description="a", estimated_turns=TurnRange(low=10, high=20))],
        totals=TurnRange(low=8, high=12),
    )
    est = estimate_skill(BDR_SKILL, counter, sidecar=sidecar)
    assert (est.turns.low, est.turns.high) == (8.0, 12.0)


def test_missing_skill_md_raises(tmp_path: Path, counter: HeuristicTokenCounter) -> None:
    with pytest.raises(FileNotFoundError):
        estimate_skill(tmp_path, counter)


def test_more_turns_cost_more(counter: HeuristicTokenCounter) -> None:
    short = EvalEstimates(totals=TurnRange(low=5, high=5))
    long = EvalEstimates(totals=TurnRange(low=50, high=50))
    est_short = estimate_skill(BDR_SKILL, counter, sidecar=short)
    est_long = estimate_skill(BDR_SKILL, counter, sidecar=long)
    assert est_long.fresh_input_tokens.low > est_short.fresh_input_tokens.low
    assert est_long.cached_read_tokens.low > est_short.cached_read_tokens.low
    assert est_long.wall_seconds.low > est_short.wall_seconds.low


# ───────── Payload-aware before side ─────────


def test_external_call_payload_sums_footprint(bdr_pipeline: Pipeline) -> None:
    """Payload = one per-tool prior per external_call node, counted over the
    same wave decomposition as estimate_pipeline (loop-body sub-nodes are not
    double-counted), each at the default constant unless pinned."""
    from rote.adapters._common import _execution_waves
    from rote.eval.estimate import external_call_payload_tokens

    priors = Priors()
    n_external = sum(
        1
        for wave in _execution_waves(bdr_pipeline)
        for n in wave
        if n.kind is NodeKind.EXTERNAL_CALL
    )
    # sanity: the BDR fixture has external_call nodes to sum over
    assert n_external >= 1
    payload = external_call_payload_tokens(bdr_pipeline, priors)
    assert payload == n_external * priors.tokens_per_external_call_result


def test_per_tool_override_changes_payload(bdr_pipeline: Pipeline) -> None:
    """A pinned per-tool payload beats the default constant for that tool."""
    from rote.eval.estimate import external_call_payload_tokens

    external = [n for n in bdr_pipeline.nodes if n.kind is NodeKind.EXTERNAL_CALL and n.mcp]
    if not external:
        pytest.skip("BDR fixture has no MCP-bound external_call to override")
    tool = external[0].mcp.tool
    base = external_call_payload_tokens(bdr_pipeline, Priors())
    bumped = external_call_payload_tokens(
        bdr_pipeline, Priors(payload_tokens_per_tool={tool: 99_000.0})
    )
    assert bumped > base


def test_data_payload_raises_cached_read(counter: HeuristicTokenCounter) -> None:
    """Folding a data payload into C₀ inflates the dominant cache-read term —
    the fix for the data-heavy-skill underestimate."""
    sidecar = EvalEstimates(totals=TurnRange(low=20, high=20))
    baseline = estimate_skill(BDR_SKILL, counter, sidecar=sidecar, data_payload_tokens=0.0)
    heavy = estimate_skill(BDR_SKILL, counter, sidecar=sidecar, data_payload_tokens=400_000.0)
    assert heavy.context_tokens == baseline.context_tokens + 400_000.0
    assert heavy.cached_read_tokens.high > baseline.cached_read_tokens.high
    # Default is backward-compatible: no payload ⇒ identical to before.
    assert estimate_skill(BDR_SKILL, counter, sidecar=sidecar).cached_read_tokens.high == (
        baseline.cached_read_tokens.high
    )
