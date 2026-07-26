"""The compiler-emitted eval sidecar (``eval.yaml``).

During compilation the agent reads every step of the source skill
anyway — the sidecar captures, at near-zero marginal cost, its
judgment of how many agent turns each step would consume if the skill
were run as raw instructions. That per-step estimate is the single
biggest input to the "before" cost model, and an agent that has just
deep-read the skill produces a far better one than any structural
heuristic.

The sidecar is deliberately *not* part of the IR: it describes the
source skill's behavior as an agent, not the compiled pipeline, so it
travels next to ``pipeline.yaml`` rather than inside it (the IR stays
runtime-agnostic and lean — invariant #1).
"""

from __future__ import annotations

from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

EVAL_SIDECAR_FILENAME = "eval.yaml"


class TurnRange(BaseModel):
    """An inclusive low/high estimate. ``low <= high`` is enforced."""

    model_config = ConfigDict(extra="forbid")

    low: float = Field(ge=0)
    high: float = Field(ge=0)

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.low > self.high:
            raise ValueError(f"low ({self.low}) must be <= high ({self.high})")
        return self


class StepEstimate(BaseModel):
    """One source-skill step, as the compiler judged it."""

    model_config = ConfigDict(extra="forbid")

    description: str
    node_id: str | None = Field(
        default=None,
        description="IR node this step became, when the mapping is 1:1",
    )
    phase: str | None = None
    estimated_turns: TurnRange
    estimated_tool_calls: TurnRange | None = None
    iterations: TurnRange | None = Field(
        default=None,
        description=(
            "How many times this step repeats per run (per row, per page, "
            "per item). When present, estimated_turns/estimated_tool_calls "
            "are PER-ITERATION and the whole-run contribution is their "
            "product. Absent means the step runs once. This is where loop "
            "cost lives: a 3-turn step over 200 rows is 600 turns, and a "
            "sidecar that cannot say so underestimates loop-dominated "
            "skills by an order of magnitude."
        ),
    )

    def turn_contribution(self) -> TurnRange:
        """This step's whole-run turns: per-iteration turns × iterations."""
        if self.iterations is None:
            return self.estimated_turns
        return TurnRange(
            low=self.estimated_turns.low * self.iterations.low,
            high=self.estimated_turns.high * self.iterations.high,
        )

    @model_validator(mode="before")
    @classmethod
    def _coerce_phase(cls, data: object) -> object:
        if isinstance(data, dict) and data.get("phase") is not None:
            return {**data, "phase": str(data["phase"])}
        return data


class EvalEstimates(BaseModel):
    """Contents of an ``eval.yaml`` sidecar."""

    model_config = ConfigDict(extra="forbid")

    version: int = 1
    source_skill: str | None = None
    steps: list[StepEstimate] = Field(default_factory=list)
    totals: TurnRange | None = Field(
        default=None,
        description=(
            "Whole-run turn estimate. Optional: absent means 'sum the steps'. "
            "Present when the compiler judges the whole differs from the sum "
            "(steps that interleave, shared exploration, etc.)."
        ),
    )
    notes: str | None = None

    def turn_range(self) -> TurnRange:
        """The whole-run turn estimate: explicit totals, else the step sum.

        The step sum multiplies per-iteration estimates by their
        ``iterations`` range, so a per-row loop step contributes its
        whole-run cost. An explicit ``totals`` must therefore also be a
        whole-run (post-multiplication) figure.
        """
        if self.totals is not None:
            return self.totals
        low = sum(s.turn_contribution().low for s in self.steps)
        high = sum(s.turn_contribution().high for s in self.steps)
        return TurnRange(low=low, high=high)


def load_eval_estimates(path: str | Path) -> EvalEstimates:
    """Load and validate an ``eval.yaml`` sidecar file."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return EvalEstimates.model_validate(raw)
