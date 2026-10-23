"""The step runner ``rote setup`` will use (rote.setup), with fake steps."""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import pytest

from rote.interact.mode import Mode
from rote.interact.output import Output
from rote.interact.plan import Action, ConfirmationRequired, Launch, Plan
from rote.interact.prompter import Answer, Prompter, ScriptedPrompter
from rote.setup import SetupContext, StepStatus, run_steps


@dataclass
class FakeStep:
    id: str
    done: bool = False
    fail: bool = False
    ran: list[str] = field(default_factory=list)
    title: str = "fake"
    command: str = "rote fake"

    def status(self, ctx: SetupContext) -> StepStatus:
        return StepStatus("done" if self.done else "pending", "state")

    def plan(self, ctx: SetupContext, prompter: Prompter) -> Plan:
        def run() -> str:
            self.ran.append(self.id)
            if self.fail:
                raise RuntimeError("nope")
            return f"{self.id}-result"

        return Plan(self.id, (Action(f"{self.id}-a", "a", (Launch("x", "y"),), run),))


def _ctx(yes: bool = False) -> SetupContext:
    err = io.StringIO()
    mode = Mode.resolve(yes=yes, env={}, stdin=io.StringIO(), stdout=io.StringIO(), stderr=err)
    return SetupContext(mode=mode, out=Output(mode, stderr=err))


def test_done_steps_are_skipped_and_the_rest_apply_in_order() -> None:
    steps = [FakeStep("one", done=True), FakeStep("two"), FakeStep("three")]
    ctx = _ctx()
    prompter = ScriptedPrompter(
        [Answer("confirm", "--yes", True), Answer("confirm", "--yes", True)]
    )
    reports = run_steps(steps, ctx, prompter)
    assert [(r.step_id, r.state) for r in reports] == [
        ("one", "skipped"),
        ("two", "applied"),
        ("three", "applied"),
    ]
    assert steps[0].ran == [] and steps[1].ran == ["two"]
    # The runner records what each applied action returned, after it ran.
    assert ctx.results == {"two": {"two-a": "two-result"}, "three": {"three-a": "three-result"}}


def test_a_failure_or_a_no_stops_the_later_steps() -> None:
    steps = [FakeStep("one", fail=True), FakeStep("two")]
    reports = run_steps(steps, _ctx(), ScriptedPrompter([Answer("confirm", "--yes", True)]))
    assert [(r.step_id, r.state, r.detail) for r in reports] == [
        ("one", "failed", "RuntimeError: nope")
    ]
    assert steps[1].ran == []

    declined = [FakeStep("one"), FakeStep("two")]
    reports = run_steps(declined, _ctx(), ScriptedPrompter([Answer("confirm", "--yes", False)]))
    assert [(r.step_id, r.state) for r in reports] == [("one", "declined")]
    assert declined[0].ran == []


def test_nobody_at_the_terminal_needs_yes() -> None:
    from rote.interact.prompter import NonInteractivePrompter

    step = FakeStep("one")
    with pytest.raises(ConfirmationRequired):
        run_steps([step], _ctx(), NonInteractivePrompter(why="x"))
    assert step.ran == []
    reports = run_steps(
        [step], _ctx(yes=True), NonInteractivePrompter(why="x", accept_defaults=True)
    )
    assert reports[0].state == "applied"
