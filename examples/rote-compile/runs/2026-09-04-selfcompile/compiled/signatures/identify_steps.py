"""
IdentifySteps — llm_judge signature for Phase 2 step identification.

Given the full text of SKILL.md and every reference file, identify all distinct
pipeline steps. Each step maps to one future IR node. The output list is the
fan-out seed for classify_step.

Source rubric: SKILL.md §Phase 2: Node Classification
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict


class StepDescription(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: str
    """snake_case identifier; will become the IR node id."""

    name: str
    """Human-readable step name."""

    description: str
    """What the step does, in 1-2 sentences."""

    source_section: str
    """Exact SKILL.md heading text the step was derived from (without # markers)."""

    phase: str | None = None
    """Skill phase number or name, when the skill makes phases explicit."""


class IdentifyStepsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skill_md: str
    """Full text of SKILL.md."""

    reference_files: dict[str, str]
    """Map of filename → full text for each references/*.md file."""


class IdentifyStepsOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    steps: list[StepDescription]
    """All distinct steps in the skill, in execution order."""


class IdentifySteps:
    """Identify all distinct steps in a skill bundle.

    A 'step' is a unit of work granular enough to become one IR node.
    Steps must have unambiguous input/output types and map to one SKILL.md section.
    """

    async def forward(self, inputs: IdentifyStepsInput) -> IdentifyStepsOutput:
        # Pre-filter: if SKILL.md is empty or missing a body, fast-path.
        body = inputs.skill_md.strip()
        if not body or body.count("\n") < 5:
            raise ValueError(
                f"SKILL.md appears too short to contain phases ({len(body)} chars). "
                "Check that skill_dir points to a valid skill bundle."
            )

        # LLM dispatch — identify steps from prose.
        raise NotImplementedError(
            "LLM dispatch not implemented. Use rote's inference runtime helper "
            "(`signatures/_rote_inference.py:call_judge`) to call this signature."
        )
