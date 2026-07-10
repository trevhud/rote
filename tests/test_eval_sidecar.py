"""Validation tests for the graduator-emitted eval.yaml sidecar."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from rote.eval.sidecar import EvalEstimates, StepEstimate, TurnRange, load_eval_estimates


def test_turn_range_rejects_inverted_bounds() -> None:
    with pytest.raises(ValidationError):
        TurnRange(low=5, high=2)


def test_turn_range_sums_across_steps() -> None:
    est = EvalEstimates(
        steps=[
            StepEstimate(description="a", estimated_turns=TurnRange(low=1, high=2)),
            StepEstimate(description="b", estimated_turns=TurnRange(low=3, high=5)),
        ]
    )
    tr = est.turn_range()
    assert (tr.low, tr.high) == (4, 7)


def test_totals_override_step_sum() -> None:
    est = EvalEstimates(
        steps=[StepEstimate(description="a", estimated_turns=TurnRange(low=1, high=2))],
        totals=TurnRange(low=10, high=20),
    )
    assert est.turn_range().low == 10


def test_iterations_multiply_into_the_step_sum() -> None:
    # The push-to-coupa shape: a flat setup step plus a per-row loop
    # step. The loop's whole-run contribution is turns × iterations —
    # the schema gap that made the coupa estimate miss 10× low.
    est = EvalEstimates(
        steps=[
            StepEstimate(description="setup", estimated_turns=TurnRange(low=4, high=8)),
            StepEstimate(
                description="push each invoice row",
                estimated_turns=TurnRange(low=3, high=6),
                iterations=TurnRange(low=20, high=90),
            ),
        ]
    )
    tr = est.turn_range()
    assert (tr.low, tr.high) == (4 + 3 * 20, 8 + 6 * 90)


def test_iterations_absent_means_step_runs_once() -> None:
    step = StepEstimate(description="a", estimated_turns=TurnRange(low=2, high=5))
    contribution = step.turn_contribution()
    assert (contribution.low, contribution.high) == (2, 5)


def test_totals_still_override_an_iterated_step_sum() -> None:
    est = EvalEstimates(
        steps=[
            StepEstimate(
                description="loop",
                estimated_turns=TurnRange(low=3, high=6),
                iterations=TurnRange(low=20, high=90),
            )
        ],
        totals=TurnRange(low=100, high=400),
    )
    assert (est.turn_range().low, est.turn_range().high) == (100, 400)


def test_iterations_load_from_yaml(tmp_path: Path) -> None:
    p = tmp_path / "eval.yaml"
    p.write_text(
        """
version: 1
steps:
  - description: Process each invoice row
    estimated_turns: {low: 3, high: 6}
    iterations: {low: 20, high: 90}
""",
        encoding="utf-8",
    )
    est = load_eval_estimates(p)
    assert est.steps[0].iterations is not None
    assert est.turn_range().high == 540


def test_phase_coerced_to_string() -> None:
    step = StepEstimate.model_validate(
        {"description": "x", "phase": 1.5, "estimated_turns": {"low": 1, "high": 2}}
    )
    assert step.phase == "1.5"


def test_unknown_fields_rejected() -> None:
    with pytest.raises(ValidationError):
        EvalEstimates.model_validate({"version": 1, "bogus": True})


def test_load_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "eval.yaml"
    p.write_text(
        """
version: 1
source_skill: examples/foo/skill
steps:
  - description: Research the target
    node_id: target_research
    phase: "1"
    estimated_turns: {low: 4, high: 10}
    estimated_tool_calls: {low: 5, high: 14}
totals: {low: 28, high: 45}
notes: Lead generation dominates.
""",
        encoding="utf-8",
    )
    est = load_eval_estimates(p)
    assert est.source_skill == "examples/foo/skill"
    assert est.steps[0].node_id == "target_research"
    assert est.turn_range().high == 45
