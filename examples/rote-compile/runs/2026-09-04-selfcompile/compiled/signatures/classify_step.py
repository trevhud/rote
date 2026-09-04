"""
ClassifyStep — llm_judge signature for Phase 2 node kind classification.

Given a single identified step and the full skill bundle for context, classify
the step into exactly one of the five IR node kinds. Prefers the more
deterministic kind when the decision is ambiguous.

Fanned out over identify_steps.output.steps — one invocation per step.

Source rubric: skills/rote-compile/references/node-kinds.md
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict


class NodeKind(str, Enum):
    PURE_FUNCTION = "pure_function"
    EXTERNAL_CALL = "external_call"
    LLM_JUDGE = "llm_judge"
    AGENT_LOOP = "agent_loop"
    HITL_GATE = "hitl_gate"


class StepDescription(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: str
    name: str
    description: str
    source_section: str
    phase: str | None = None


class ClassifyStepInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: StepDescription
    """The step to classify."""

    skill_md: str
    """Full SKILL.md text, for context on data flow between steps."""

    reference_files: dict[str, str]
    """All reference files, so the classifier can re-read rubric sections."""


class ClassifyStepOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: StepDescription
    """The step being classified, passed through for downstream fan-out consumers."""

    kind: NodeKind
    """The assigned node kind."""

    justification: str
    """One-sentence explanation of why this kind was chosen over alternatives."""

    suggest_mandatory: bool
    """True when skipping this step would produce wrong or incomplete output."""


# Determinism preference ordering (most → least):
DETERMINISM_ORDER = [
    NodeKind.PURE_FUNCTION,
    NodeKind.EXTERNAL_CALL,
    NodeKind.LLM_JUDGE,
    NodeKind.AGENT_LOOP,
    NodeKind.HITL_GATE,
]

# A hitl_gate is always mandatory by the IR schema — flag it automatically.
_ALWAYS_MANDATORY_KINDS = {NodeKind.HITL_GATE}

# Steps whose name contains these phrases are almost certainly pure_function.
_PURE_FUNCTION_HINTS = {
    "render", "format", "template", "report", "write_", "assemble_report",
    "progress_marker",
}

# Steps whose name contains these phrases are almost certainly external_call.
_EXTERNAL_CALL_HINTS = {
    "read_", "fetch_", "invoke_", "validate_ir", "load_",
}


class ClassifyStep:
    """Classify a single skill step into one of five IR node kinds."""

    async def forward(self, inputs: ClassifyStepInput) -> ClassifyStepOutput:
        step_id = inputs.step.step_id.lower()

        # Pre-filter 1: pure_function by naming convention.
        for hint in _PURE_FUNCTION_HINTS:
            if hint in step_id:
                return ClassifyStepOutput(
                    step=inputs.step,
                    kind=NodeKind.PURE_FUNCTION,
                    justification=(
                        f"Step name contains '{hint}', which signals fixed-template "
                        "string rendering with no LLM or external service."
                    ),
                    suggest_mandatory=True,
                )

        # Pre-filter 2: external_call by naming convention.
        for hint in _EXTERNAL_CALL_HINTS:
            if hint in step_id:
                return ClassifyStepOutput(
                    step=inputs.step,
                    kind=NodeKind.EXTERNAL_CALL,
                    justification=(
                        f"Step name contains '{hint}', which signals a deterministic "
                        "call to an external service (filesystem or subprocess)."
                    ),
                    suggest_mandatory=True,
                )

        # LLM dispatch for all other cases.
        raise NotImplementedError(
            "LLM dispatch not implemented. Use rote's inference runtime helper "
            "(`signatures/_rote_inference.py:call_judge`) to call this signature."
        )
