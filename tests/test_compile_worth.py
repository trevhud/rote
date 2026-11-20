"""A skill-runtime compile stops before generation when it would not pay off."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from rote.compiler import Compiler, CompilerError
from rote.compiler.verify import EmitTarget
from rote.compiler.worth import MAX_MODEL_STOPS, UNBOUND_CALL, worth


@dataclass
class _Node:
    id: str
    kind: str
    in_loop: str | None = None
    mcp: dict[str, Any] | None = None


@dataclass
class _Structure:
    cost_usd: float | None = 0.0123


@dataclass
class _Plan:
    nodes: list[_Node] = field(default_factory=list)
    structure: _Structure | None = field(default_factory=_Structure)


def _judges(n: int) -> list[_Node]:
    return [_Node(f"judge_{i}", "llm_judge") for i in range(n)]


BOUND = {"server": "crm", "tool": "search"}


def test_judges_and_loops_stop_for_the_model_and_nothing_else_does() -> None:
    nodes = [
        _Node("fetch", "external_call", mcp=BOUND),
        _Node("total", "pure_function"),
        _Node("approve", "hitl_gate"),
        _Node("pick", "llm_judge"),
        _Node("research", "agent_loop"),
        _Node("vet", "llm_judge", in_loop="research"),
    ]
    assert worth(nodes, "session", "mcp").stops == (
        ("pick", "llm_judge"),
        ("research", "agent_loop"),
    )
    # Judges the inference helper answers are API calls, not round trips.
    assert worth(nodes, "helper", "mcp").stops == (("research", "agent_loop"),)


def test_the_line_is_at_two_model_stops() -> None:
    assert MAX_MODEL_STOPS == 2
    assert worth(_judges(2), "session", "mcp").pays_off
    assert not worth(_judges(3), "session", "mcp").pays_off


def _reads(n: int, mcp: dict[str, Any] | None = None) -> list[_Node]:
    return [_Node(f"read_{i}", "external_call", mcp=mcp) for i in range(n)]


def test_a_call_no_tool_is_bound_to_stops_for_the_model_like_a_judge() -> None:
    """The agent makes an unbound call itself (a DO block). Measured: four of
    them in one block ran 3.1x costlier than the same four bound."""
    found = worth([*_reads(2), *_judges(1)], "session", "mcp")
    assert found.stops == (
        ("read_0", UNBOUND_CALL),
        ("read_1", UNBOUND_CALL),
        ("judge_0", "llm_judge"),
    )
    assert not found.pays_off
    assert worth(_reads(2), "session", "mcp").pays_off
    assert worth(_reads(2), "session", None).stops == worth(_reads(2), "session", "mcp").stops
    # Helper judges are API calls; an unbound call is still the agent's.
    assert len(worth([*_reads(3), *_judges(3)], "helper", "mcp").stops) == 3
    # Bound, the driver makes the call; on the api backend, the compile writes it.
    assert worth(_reads(4, BOUND), "session", "mcp").stops == ()
    assert worth(_reads(4), "session", "api").stops == ()
    # A loop's body runs inside the loop's stop.
    assert worth([_Node("push", "external_call", in_loop="sync")], "session", "mcp").stops == ()


def test_the_refusal_says_how_to_make_unbound_calls_run_in_code() -> None:
    message = worth([*_reads(3), *_judges(1)], "session", "mcp").explain()
    assert "stop for the model 4 times" in message
    assert "3 of them (read_0, read_1, read_2) are calls no bound MCP tool makes" in message
    assert "rote mcp import" in message and "--compile-anyway" in message
    assert "no bound MCP tool" not in worth(_judges(4), "session", "mcp").explain()


def _compiler(
    runtime: str, *, anyway: bool = False, judges: str = "session", backend: str | None = None
) -> Compiler:
    target = EmitTarget(
        runtime=runtime,
        skill_dir=Path("skill") if runtime == "skill" else None,
        judges=judges if runtime == "skill" else None,
        external_backend=backend,
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


def test_a_skill_compile_with_unbound_calls_stops_unless_the_backend_writes_them() -> None:
    with pytest.raises(CompilerError, match="read_0, read_1, read_2"):
        _compiler("skill")._refuse_if_not_worth_it(_Plan(_reads(3)))  # type: ignore[arg-type]
    _compiler("skill", backend="api")._refuse_if_not_worth_it(_Plan(_reads(3)))  # type: ignore[arg-type]
    _compiler("dbos")._refuse_if_not_worth_it(_Plan(_reads(3)))  # type: ignore[arg-type]
