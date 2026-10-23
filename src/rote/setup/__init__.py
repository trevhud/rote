"""The seam ``rote setup`` will be built on: steps that plan, then apply.

``rote setup`` is not built yet (docs/design/first-run.md has the plan).
This module fixes the contract its steps follow, so the step PRs (MCP
import, skill pick, proof) can land independently:

- A :class:`Step` reports its :meth:`~Step.status` from the world, read
  only (is a server registered, is it logged in, is a skill chosen). That
  is what makes setup resumable without a state file of its own: a step
  that is done says so, and a re-run skips it.
- :meth:`~Step.plan` asks its questions through the prompter and returns a
  :class:`~rote.interact.plan.Plan`. It has no side effects; everything it
  would do is an Action in the plan.
- :func:`run_steps` walks the steps in order: skip what is done, plan,
  show, confirm, apply, and stop at the first step that did not finish,
  since later steps build on earlier ones (a skill is ranked by the servers
  step 1 registered).

Each step is also its own command (``rote mcp import``, ``rote try``),
which builds the same Step and calls :func:`run_steps` with just it.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from rote.interact.mode import Mode
from rote.interact.output import Output
from rote.interact.plan import Outcome, Plan, apply_plan, confirm_plan
from rote.interact.prompter import Prompter


@dataclass(frozen=True)
class StepStatus:
    state: Literal["done", "pending"]
    #: One line for the person ("3 servers registered, 1 not logged in").
    detail: str


@dataclass
class SetupContext:
    mode: Mode
    out: Output
    #: What earlier steps decided, by step id, for later steps to read.
    results: dict[str, Any] = field(default_factory=dict)


class Step(Protocol):
    id: str
    title: str
    #: The standalone command for this step, for agents and for "next:" hints.
    command: str

    def status(self, ctx: SetupContext) -> StepStatus: ...

    def plan(self, ctx: SetupContext, prompter: Prompter) -> Plan: ...


@dataclass(frozen=True)
class StepReport:
    step_id: str
    state: Literal["skipped", "declined", "applied", "failed"]
    detail: str
    plan: Plan | None = None
    outcomes: tuple[Outcome, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.step_id,
            "state": self.state,
            "detail": self.detail,
            "plan": None if self.plan is None else self.plan.to_dict(),
            "outcomes": [o.to_dict() for o in self.outcomes],
        }


def run_steps(
    steps: Sequence[Step],
    ctx: SetupContext,
    prompter: Prompter,
    *,
    granted: Collection[str] = (),
) -> list[StepReport]:
    """Run ``steps`` in order. Raises what the prompter or the confirmation
    raises when nobody is at the terminal (``PromptRequired``,
    ``ConfirmationRequired``), with the reports so far lost to the caller
    only in the sense that every finished step is already applied and its
    status will read ``done`` on the next run."""
    reports: list[StepReport] = []
    total = len(steps)
    for index, step in enumerate(steps, start=1):
        ctx.out.say(f"Step {index}/{total}  {step.title}", style="bold")
        status = step.status(ctx)
        if status.state == "done":
            ctx.out.say(f"  done already: {status.detail}")
            reports.append(StepReport(step.id, "skipped", status.detail))
            continue
        plan = step.plan(ctx, prompter)
        ctx.out.plan(plan)
        if not confirm_plan(plan, prompter, yes=ctx.mode.yes, granted=granted):
            reports.append(StepReport(step.id, "declined", "not applied", plan))
            break
        outcomes = tuple(apply_plan(plan))
        failed = [o for o in outcomes if o.status != "done"]
        state: Literal["applied", "failed"] = "failed" if failed else "applied"
        detail = failed[0].error or "failed" if failed else f"{len(outcomes)} action(s) applied"
        reports.append(StepReport(step.id, state, detail, plan, outcomes))
        if failed:
            break
    return reports
