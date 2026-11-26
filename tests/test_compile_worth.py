"""A skill-runtime compile stops before generation when it would not pay off."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from rote.compiler import Compiler, CompilerError
from rote.compiler.verify import EmitTarget
from rote.compiler.worth import MAX_MODEL_STOPS, MAX_TOKEN_RATIO, predict, worth


@dataclass
class _Node:
    id: str
    kind: str
    in_loop: str | None = None
    mcp: dict[str, Any] | None = None
    decisions: list[Any] = field(default_factory=list)


@dataclass
class _Decision:
    field: str
    value: Any


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


# ── with the skill's own measured run ──

EXPENSE_AUDIT = Path(__file__).parent.parent / "examples/expense-audit/expected/pipeline.yaml"
#: What the plan above is predicted to use compiled: its one unbound call
#: (saving the report) written as code, and left to the agent as a DO block.
AS_CODE, AS_AGENT_WORK = 65_000, 118_000


def _run(total: int, **changes: Any) -> dict[str, Any]:
    run = {
        "succeeded": True,
        "flags": [],
        "turns": 5,
        "input_tokens": 10,
        "cache_creation_tokens": 4_000,
        "cache_read_tokens": total - 8_000,
        "output_tokens": 3_990,
    }
    return {**run, **changes}


def _predict(total: int, code: tuple[str, ...] = ("file_report",)) -> Any:
    return predict(EXPENSE_AUDIT, [_run(total)], judges="session", backend="mcp", code=code)


def test_the_plan_is_priced_as_a_conversation_against_the_measured_run() -> None:
    found = _predict(114_000)
    assert found.measured == 114_000 and found.runs == 1
    assert found.compiled[0] == pytest.approx(AS_CODE, rel=0.02)
    assert found.pays_off
    assert "predicted at 65k to 65k tokens (4 to 4 requests)" in found.summary()
    assert "measured run used 114k in 5 turns" in found.summary()


def test_a_call_generation_does_not_write_as_code_is_a_do_block() -> None:
    """The same plan costs nearly twice as much when the agent saves the file."""
    found = _predict(114_000, code=())
    assert found.compiled[0] == pytest.approx(AS_AGENT_WORK, rel=0.02)
    assert found.requests == (6.0, 6.0)


def test_the_line_is_a_multiple_of_the_measured_run() -> None:
    assert MAX_TOKEN_RATIO == 1.5
    low = _predict(114_000).compiled[0]
    assert _predict(int(low / MAX_TOKEN_RATIO) + 1_000).pays_off
    assert not _predict(int(low / MAX_TOKEN_RATIO) - 1_000).pays_off


@pytest.mark.parametrize(
    "runs",
    [[], [_run(30_000, succeeded=False)], [_run(30_000, flags=["truncated"])]],
    ids=["none", "failed", "flagged"],
)
def test_a_run_that_is_not_the_skills_cost_predicts_nothing(runs: list[dict[str, Any]]) -> None:
    assert predict(EXPENSE_AUDIT, runs, judges="session", backend="mcp", code=()) is None


def _skeleton(tmp_path: Path) -> Path:
    (tmp_path / "pipeline.yaml").write_text(EXPENSE_AUDIT.read_text("utf-8"), "utf-8")
    return tmp_path


def _saves_a_file() -> _Node:
    return _Node(
        "file_report", "external_call", decisions=[_Decision("system", {"how": "local_file"})]
    )


def test_a_compile_predicted_to_cost_more_than_the_skill_stops(tmp_path: Path) -> None:
    plan = _Plan([_saves_a_file()])
    with pytest.raises(CompilerError) as e:
        _compiler("skill")._refuse_if_not_worth_it(plan, _skeleton(tmp_path), [_run(30_000)])  # type: ignore[arg-type]
    message = str(e.value)
    assert "predicted at 65k to 65k tokens" in message
    assert "measured run used 30k" in message
    assert "--compile-anyway" in message
    assert message.endswith("(spent so far: $0.0123)")
    _compiler("skill", anyway=True)._refuse_if_not_worth_it(plan, tmp_path, [_run(30_000)])  # type: ignore[arg-type]


def test_a_measured_run_replaces_the_count_of_stops(tmp_path: Path) -> None:
    """Four judges would stop the compile on the count; the skill's own run
    says it pays off, and how much work the original does is what the count
    cannot see."""
    plan = _Plan([*_judges(4), _saves_a_file()])
    logged: list[str] = []
    compiler = _compiler("skill")
    compiler.on_event = lambda event: logged.append(str(event))
    compiler._refuse_if_not_worth_it(plan, _skeleton(tmp_path), [_run(114_000)])  # type: ignore[arg-type]
    assert any("worth compiling: a compiled run is predicted at 65k" in line for line in logged)
    # A step the plan does not mark as a local file is the agent's: 118k
    # against a 70k run is past the line.
    with pytest.raises(CompilerError, match="predicted at 118k"):
        compiler._refuse_if_not_worth_it(_Plan(_judges(4)), tmp_path, [_run(70_000)])  # type: ignore[arg-type]


def test_without_a_usable_run_the_stops_are_counted(tmp_path: Path) -> None:
    with pytest.raises(CompilerError, match="stop for the model 4 times"):
        _compiler("skill")._refuse_if_not_worth_it(
            _Plan(_judges(4)),  # type: ignore[arg-type]
            _skeleton(tmp_path),
            [_run(900_000, flags=["missing_mcp_servers"])],
        )
