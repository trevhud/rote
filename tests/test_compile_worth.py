"""A skill-runtime compile stops before generation when it would not pay off."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from rote.compiler import Compiler, CompilerError
from rote.compiler.verify import EmitTarget
from rote.compiler.worth import MAX_MODEL_STOPS, worth


@dataclass
class _Node:
    id: str
    kind: str
    in_loop: str | None = None


@dataclass
class _Structure:
    cost_usd: float | None = 0.0123


@dataclass
class _Plan:
    nodes: list[_Node] = field(default_factory=list)
    structure: _Structure | None = field(default_factory=_Structure)


def _judges(n: int) -> list[_Node]:
    return [_Node(f"judge_{i}", "llm_judge") for i in range(n)]


def test_judges_and_loops_stop_for_the_model_and_nothing_else_does() -> None:
    nodes = [
        _Node("fetch", "external_call"),
        _Node("total", "pure_function"),
        _Node("approve", "hitl_gate"),
        _Node("pick", "llm_judge"),
        _Node("research", "agent_loop"),
        _Node("vet", "llm_judge", in_loop="research"),
    ]
    assert worth(nodes, "session").stops == (("pick", "llm_judge"), ("research", "agent_loop"))
    # Judges the inference helper answers are API calls, not round trips.
    assert worth(nodes, "helper").stops == (("research", "agent_loop"),)


def test_the_line_is_at_two_model_stops() -> None:
    assert MAX_MODEL_STOPS == 2
    assert worth(_judges(2), "session").pays_off
    assert not worth(_judges(3), "session").pays_off


def _compiler(runtime: str, *, anyway: bool = False, judges: str = "session") -> Compiler:
    target = EmitTarget(
        runtime=runtime,
        skill_dir=Path("skill") if runtime == "skill" else None,
        judges=judges if runtime == "skill" else None,
    )
    return Compiler(emit_target=target, compile_anyway=anyway)


def test_a_skill_compile_that_would_not_pay_off_stops_naming_the_steps() -> None:
    with pytest.raises(CompilerError) as e:
        _compiler("skill")._refuse_if_not_worth_it(_Plan(_judges(4)))  # type: ignore[arg-type]
    message = str(e.value)
    assert "stop for the model 4 times (judge_0 (llm_judge)" in message
    assert "--compile-anyway" in message
    assert message.endswith("(spent so far: $0.0123)")


@pytest.mark.parametrize(
    "compiler",
    [
        _compiler("skill", anyway=True),
        _compiler("skill", judges="helper"),
        _compiler("dbos"),
    ],
    ids=["anyway", "helper-judges", "workflow-runtime"],
)
def test_the_override_helper_judges_and_workflow_runtimes_compile(compiler: Any) -> None:
    compiler._refuse_if_not_worth_it(_Plan(_judges(4)))
