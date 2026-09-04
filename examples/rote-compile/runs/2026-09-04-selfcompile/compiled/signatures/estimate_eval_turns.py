"""
EstimateEvalTurns — llm_judge signature for Phase 5 eval sidecar estimation.

Given a classified step and the full skill text for context, estimate how many
agent turns the step consumes when the skill runs as raw instructions. Provides
low/high bounds and, when the step repeats per item, an iteration count range.

Fanned out over classify_step.output — one invocation per classified step.

Source rubric: skills/rote-compile/references/eval-estimates.md
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator


# Calibration anchors from eval-estimates.md (measured production runs).
# Used in the pre-filter to sanity-check estimates before dispatch.
_CALIBRATION = {
    "pure_function":  (1, 2),   # deterministic; near-zero agent cost
    "external_call":  (1, 3),   # deterministic call; minor overhead
    "llm_judge":      (1, 3),   # one structured call per invocation
    "agent_loop":     (3, 15),  # genuinely exploratory; wide range
    "hitl_gate":      (0, 0),   # human signal; zero agent turns
}


class StepDescription(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: str
    name: str
    description: str
    source_section: str
    phase: str | None = None


class ClassifyStepOutput(BaseModel):
    """Minimal view of classify_step.output needed for estimation."""

    model_config = ConfigDict(extra="forbid")

    step: StepDescription
    kind: str
    justification: str
    suggest_mandatory: bool


class EstimateEvalTurnsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: ClassifyStepOutput
    """The classified step whose turns to estimate."""

    skill_md: str
    """Full SKILL.md text, for context on batch sizes, iteration counts, etc."""


class EstimateEvalTurnsOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    turns_low: int
    """Minimum agent turns per iteration of this step."""

    turns_high: int
    """Maximum agent turns per iteration of this step."""

    has_iterations: bool
    """True when the step repeats per item (row, contact, page, step-in-skill)."""

    iterations_low: int | None = None
    """Minimum item count per run; only meaningful when has_iterations is True."""

    iterations_high: int | None = None
    """Maximum item count per run; only meaningful when has_iterations is True."""

    notes: str | None = None
    """Optional explanation of the estimate or calibration uncertainty."""

    @field_validator("turns_high")
    @classmethod
    def high_ge_low(cls, v: int, info) -> int:
        low = info.data.get("turns_low", 0)
        if v < low:
            raise ValueError(
                f"turns_high ({v}) must be >= turns_low ({low})."
            )
        return v


class EstimateEvalTurns:
    """Estimate agent turns per step for the source skill's raw-instruction run."""

    async def forward(self, inputs: EstimateEvalTurnsInput) -> EstimateEvalTurnsOutput:
        kind = inputs.step.kind

        # Pre-filter: hitl_gate steps consume zero agent turns.
        if kind == "hitl_gate":
            return EstimateEvalTurnsOutput(
                turns_low=0,
                turns_high=0,
                has_iterations=False,
                notes="HITL gates are human signals; the agent waits, not acts.",
            )

        # Pre-filter: use calibration floor/ceiling as sanity bounds.
        # The LLM can override these if the skill prose provides evidence.
        cal_low, cal_high = _CALIBRATION.get(kind, (1, 5))

        # Pure functions and external calls at the top level are nearly free.
        if kind in ("pure_function", "external_call"):
            return EstimateEvalTurnsOutput(
                turns_low=cal_low,
                turns_high=cal_high,
                has_iterations=False,
                notes=(
                    f"Deterministic step ({kind}); agent overhead is reading "
                    "the step description and issuing the tool call."
                ),
            )

        # LLM dispatch for agent_loop and ambiguous llm_judge steps.
        raise NotImplementedError(
            "LLM dispatch not implemented. Use rote's inference runtime helper "
            "(`signatures/_rote_inference.py:call_judge`) to call this signature."
        )
