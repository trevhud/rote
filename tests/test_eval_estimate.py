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
from rote.ir import Node, NodeKind, Pipeline, load_pipeline

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


def test_agent_loop_per_call_cost_scales_with_turns_per_iteration(
    counter: HeuristicTokenCounter,
) -> None:
    """One loop iteration is several agent turns, and every per-call
    figure must carry that factor.

    This is the turn-dominated cost regime the loop-aware model exists
    for: `agent_loop_turns_per_iteration` defaults to 3, so dropping it
    understates a loop's tokens and wall time threefold. Asserting
    `calls` alone leaves all three derived numbers unpinned — a mutation
    sweep dropped the factor from both the token and the seconds line
    with the suite green.
    """
    pipeline = Pipeline.model_validate(
        {
            "name": "loop-only",
            "input": {"type": "In"},
            "nodes": [
                {
                    "id": "explore",
                    "kind": "agent_loop",
                    "description": "bounded exploration",
                    "tools": ["search"],
                    "termination": {"max_iterations": 4, "condition": "nothing left to search"},
                }
            ],
            "edges": [],
        }
    )
    priors = Priors()
    turns = priors.agent_loop_turns_per_iteration
    assert turns > 1, "a degenerate 1.0 would make this test unable to fail"

    (node_est,) = estimate_pipeline(pipeline, counter, priors).nodes
    assert node_est.calls.high == 4  # the declared bound, not the prior default
    assert node_est.note is None
    assert node_est.llm_input_tokens_per_call == round(turns * priors.transcript_growth_per_turn)
    assert node_est.llm_output_tokens_per_call == round(turns * priors.output_tokens_per_turn)
    assert node_est.wall_seconds_per_call == pytest.approx(turns * priors.seconds_per_turn)
    # …and the per-call figures compound over the iteration bound.
    assert node_est.wall_seconds.high == pytest.approx(4 * turns * priors.seconds_per_turn)


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
    # Before compilation every step is sampled and nothing is constrained.
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
    assert est.turn_method == "compiler sidecar (eval.yaml)"
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


# ───────── Transcript cap (long-run saturation) ─────────


def test_short_runs_are_unaffected_by_the_transcript_cap(
    counter: HeuristicTokenCounter,
) -> None:
    """Below the cap the model is the original uncapped quadratic:
    cached ≈ (N−1)·C₀ + Δ·(N−2)(N−1)/2."""
    sidecar = EvalEstimates(totals=TurnRange(low=30, high=30))
    est = estimate_skill(BDR_SKILL, counter, sidecar=sidecar)
    p = Priors()
    c0 = est.context_tokens
    assert c0 + 29 * p.transcript_growth_per_turn < p.transcript_cap_tokens
    expected = 29 * c0 + p.transcript_growth_per_turn * (28 * 29 / 2)
    assert est.cached_read_tokens.low == pytest.approx(expected)


def test_long_runs_saturate_at_the_transcript_cap(counter: HeuristicTokenCounter) -> None:
    """The push-to-coupa regime: ~700 turns. Uncapped, Δ·N²/2 alone
    predicts ~220M cached tokens; both measured production runs
    plateaued at ~165k/turn, bounding the total near N·cap."""
    n = 700.0
    sidecar = EvalEstimates(totals=TurnRange(low=n, high=n))
    est = estimate_skill(BDR_SKILL, counter, sidecar=sidecar)
    p = Priors()
    uncapped = (n - 1) * est.context_tokens + p.transcript_growth_per_turn * ((n - 2) * (n - 1) / 2)
    hard_ceiling = (n - 1) * p.transcript_cap_tokens
    assert est.cached_read_tokens.low < uncapped
    assert est.cached_read_tokens.low <= hard_ceiling
    # Saturation must not undercut the pre-cap ramp: the total still
    # exceeds what a run half as long reads.
    half = estimate_skill(
        BDR_SKILL, counter, sidecar=EvalEstimates(totals=TurnRange(low=n / 2, high=n / 2))
    )
    assert est.cached_read_tokens.low > half.cached_read_tokens.low


def test_payload_pushing_c0_past_the_cap_clamps_per_turn_reads(
    counter: HeuristicTokenCounter,
) -> None:
    """When fetched-once data alone exceeds the compaction ceiling, every
    re-read bills the cap, not the raw C₀."""
    sidecar = EvalEstimates(totals=TurnRange(low=10, high=10))
    p = Priors()
    est = estimate_skill(
        BDR_SKILL,
        counter,
        sidecar=sidecar,
        data_payload_tokens=p.transcript_cap_tokens * 2,
    )
    assert est.context_tokens > p.transcript_cap_tokens
    assert est.cached_read_tokens.low == pytest.approx(9 * p.transcript_cap_tokens)


def test_fresh_tokens_ignore_the_cap(counter: HeuristicTokenCounter) -> None:
    """First-sight content is written to cache once regardless of
    compaction; only re-reads saturate."""
    n = 700.0
    sidecar = EvalEstimates(totals=TurnRange(low=n, high=n))
    est = estimate_skill(BDR_SKILL, counter, sidecar=sidecar)
    p = Priors()
    assert est.fresh_input_tokens.low == pytest.approx(
        est.context_tokens + (n - 1) * p.transcript_growth_per_turn
    )


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


def test_per_tool_override_changes_payload() -> None:
    """A pinned per-tool payload beats the default constant for that tool.

    Built on a purpose-made fixture rather than the BDR example. This
    test previously took its mcp-bound node from BDR and skipped when
    there wasn't one — and there never was: BDR's five required MCP
    servers all come from its two agent loops' `tool_servers`, while its
    four external_call nodes carry no `mcp:` binding at all. So it
    silently never ran. A test whose subject is an example's incidental
    shape is a test that can quietly stop testing.
    """
    from rote.eval.estimate import external_call_payload_tokens

    pinned = Node(
        id="fetch_pinned",
        kind=NodeKind.EXTERNAL_CALL,
        description="d",
        impl="m.py:fetch",
        mcp={"server": "slack", "tool": "slack_get_messages"},
    )
    unpinned = Node(
        id="fetch_other",
        kind=NodeKind.EXTERNAL_CALL,
        description="d",
        impl="m.py:other",
        mcp={"server": "gmail", "tool": "gmail_search"},
    )
    pipeline = Pipeline(
        name="payloads",
        input={"type": "In", "required": [], "optional": []},
        nodes=[pinned, unpinned],
        edges=[{"from": "fetch_pinned", "to": "fetch_other"}],
        entry_nodes=["fetch_pinned"],
        exit_nodes=["fetch_other"],
    )

    priors = Priors()
    default_each = priors.tokens_per_external_call_result
    assert external_call_payload_tokens(pipeline, priors) == 2 * default_each

    bumped = external_call_payload_tokens(
        pipeline, Priors(payload_tokens_per_tool={"slack_get_messages": 99_000.0})
    )
    # Exactly one node is pinned: the other keeps the default. Asserting
    # `bumped > base` alone would also pass if the override leaked onto
    # every external_call.
    assert bumped == 99_000.0 + default_each


def test_priors_from_overrides_scalars_and_per_tool() -> None:
    """--prior KEY=VALUE closes the flywheel: --run's re-fits feed back in
    without editing source. Ints stay ints; the per-tool table uses dot
    syntax; untouched fields keep their defaults."""
    from rote.eval.priors import priors_from_overrides

    p = priors_from_overrides(
        [
            "transcript_growth_per_turn=5962",
            "system_overhead_tokens=21000",
            "payload_tokens_per_tool.slack_get_messages=12000",
        ]
    )
    assert p.transcript_growth_per_turn == 5962.0
    assert p.system_overhead_tokens == 21000
    assert isinstance(p.system_overhead_tokens, int)
    assert p.payload_tokens_per_tool == {"slack_get_messages": 12000.0}
    assert p.seconds_per_turn == Priors().seconds_per_turn  # untouched default


def test_priors_from_overrides_rejects_bad_input() -> None:
    from rote.eval.priors import priors_from_overrides

    with pytest.raises(ValueError, match="valid names"):
        priors_from_overrides(["not_a_prior=1"])
    with pytest.raises(ValueError, match="valid names"):
        priors_from_overrides(["payload_tokens_per_tool=1"])  # dict needs dot syntax
    with pytest.raises(ValueError, match="KEY=VALUE"):
        priors_from_overrides(["transcript_growth_per_turn"])
    with pytest.raises(ValueError, match="non-numeric"):
        priors_from_overrides(["seconds_per_turn=fast"])
    with pytest.raises(ValueError, match="names no tool"):
        priors_from_overrides(["payload_tokens_per_tool.=5"])


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
