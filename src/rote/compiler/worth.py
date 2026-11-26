"""Whether compiling a skill to the skill runtime will pay off.

A compiled skill runs its code itself and stops for the agent only at a
model step: a judge (a DECIDE block) or an agent loop (a LOOP block). Each
stop is a round trip in the agent's conversation, which re-reads everything
said so far, so a skill whose work is mostly judgment gets slower and
costlier compiled. Measured on Sonnet 4.6 (2026-10-26):

==========================  ===========  ===========================
skill                       model stops  compiled vs the original
==========================  ===========  ===========================
expense-audit               0            3.6x faster, 62% cheaper
docs-digest                 1            as fast, 47% cheaper
ops-report                  1-2          a little faster, even cost
search-company-knowledge    4            4x slower, 3x costlier
==========================  ===========  ===========================

A run of decisions with nothing between them is asked in one DECIDE block
(#344), but each judge still counts here. Measured 2026-10-31 (Sonnet 4.6,
2 trials a side) on contact-note-triage, three judges and one small pure
function: batching cut the compiled side from 6 requests and 72.5k input
tokens to 5 and 60.6k, and the original still did the work in 2 requests and
22.5k, so the compiled skill cost 2.7x the tokens. Counting a batch as one
stop would have let that skill compile: the line moves only on a measured
eval that says so.

Stops after batching (``_driver.stops``, listed in compile-report.md) do
not predict cost better. Broken down per request (2026-11-03), each request
re-read about 11k tokens of fixed context and the driver's blocks added
under 1k, so the compiled side cost about one 11k read per stop. Whether a
stop pays off depends on how much work the original did between stops,
which the plan does not show. And in the committed examples a code or tool
step sits between every two judges (deal-monitor, bdr-outreach), so stops
equal judges there; the one skill where they differ is the one above.

A human gate is not counted: the original skill stops for the person too.
Judges answered by the inference helper (``--judges helper``) are API calls,
not round trips, so they are not counted either.

When the compile measured the skill itself (``--baseline``), the count is
not used: :func:`predict` prices the plan as a conversation
(``estimate_compiled_skill``) and compares that with the measured run. The
count cannot see a skill that is already cheap. Predicted low against the
measured run, with what a paired eval then measured (2026-11-24):

==================  =========  ========  =====  ==========================
skill               predicted  measured  ratio  compiled, in the eval
==================  =========  ========  =====  ==========================
expense-audit       65k        114k      0.6    82k, cheaper
ops-report          82k        62k       1.3    70k to 76k, even cost
route-lead          93k        34k       2.7    138k to 179k, 2.5x cost
inventory-planner   116k       44k       2.6    not run
==================  =========  ========  =====  ==========================

The compile stops above ``MAX_TOKEN_RATIO``. The line sits between the
skill that came out even (and gained fixed code for its rules) and the two
that lost outright. Each eval figure fell inside the predicted range or within
about 20% of it, so its error does not confuse 1.3 with 2.6.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

#: The most model stops a compiled skill may have and still be compiled
#: by default. The table above puts the line between 2 and 4.
MAX_MODEL_STOPS = 2

#: The most a compiled run may be predicted to cost, as a multiple of the
#: skill's measured run, and still be compiled by default (table above).
MAX_TOKEN_RATIO = 1.5


@dataclass(frozen=True)
class Worth:
    #: ``(node id, kind)`` of each step that stops for the agent.
    stops: tuple[tuple[str, str], ...]

    @property
    def pays_off(self) -> bool:
        return len(self.stops) <= MAX_MODEL_STOPS

    def explain(self) -> str:
        names = ", ".join(f"{node} ({kind})" for node, kind in self.stops)
        return (
            f"compiling this skill would not pay off: each run would stop for the model "
            f"{len(self.stops)} times ({names}), and each stop is a round trip that "
            f"re-reads the conversation. Compiling helps a skill whose work is mostly code "
            f"(at most {MAX_MODEL_STOPS} model steps); one that is mostly judgment runs "
            "slower and costs more compiled, so keep it as a skill. Pass --compile-anyway "
            "to compile it regardless"
        )


def worth(nodes: Iterable[Any], judges: str | None) -> Worth:
    """The model stops in a plan's ``nodes`` (System-1's NodePlans, each with
    an ``id`` and a ``kind``), with judges counted unless ``judges`` is
    ``"helper"``. Loop-body sub-nodes run inside their loop's stop."""
    counted = {"agent_loop"} | (set() if judges == "helper" else {"llm_judge"})
    stops = tuple(
        (n.id, _kind(n)) for n in nodes if _kind(n) in counted and not getattr(n, "in_loop", False)
    )
    return Worth(stops)


def _kind(node: Any) -> str:
    kind = node.kind
    return str(getattr(kind, "value", kind))


@dataclass(frozen=True)
class Prediction:
    """A compiled run's predicted tokens beside the skill's own measured run."""

    #: Every token one compiled run would use, low to high.
    compiled: tuple[float, float]
    #: Every token the median measured run of the skill used.
    measured: float
    runs: int
    requests: tuple[float, float]
    measured_turns: float

    @property
    def pays_off(self) -> bool:
        return self.compiled[0] <= self.measured * MAX_TOKEN_RATIO

    def summary(self) -> str:
        low, high = self.compiled
        return (
            f"a compiled run is predicted at {_k(low)} to {_k(high)} tokens "
            f"({self.requests[0]:.0f} to {self.requests[1]:.0f} requests); the skill's own "
            f"measured run used {_k(self.measured)} in {self.measured_turns:.0f} turns "
            f"({self.runs} baseline run(s))"
        )

    def explain(self) -> str:
        return (
            f"compiling this skill would not pay off: {self.summary()}. A compiled skill "
            "re-reads the conversation at every stop, so a skill that is already this cheap "
            "costs more compiled; keep it as a skill. Pass --compile-anyway to compile it "
            "regardless"
        )


def _k(tokens: float) -> str:
    return f"{tokens / 1000:.0f}k"


class NoPrediction(Exception):
    """Why a plan could not be priced against the measured runs."""


def predict(
    pipeline_yaml: Path,
    runs: Sequence[Mapping[str, Any]],
    *,
    judges: str | None,
    backend: Literal["mcp", "api"],
    code: Collection[str],
) -> Prediction:
    """Price the skeleton at ``pipeline_yaml`` as a compiled skill against the
    skill's measured ``runs`` (``measured_run_record`` dicts).

    Raises :class:`NoPrediction`, saying why, when no run counts, the
    skeleton does not load, or the skill runtime cannot plan it yet (an agent
    loop's tools are not pinned to a server before generation): the caller
    counts stops instead and says so. ``code`` names the external calls
    generation writes as code (:func:`rote.adapters.skill.planned_stops`).
    Nothing here makes a call: tokens are counted by the character heuristic.

    The figure leans low. The skeleton's judge prompts are placeholders, and
    a compiled run in the person's own client carries more fixed context than
    the isolated baseline did. Both sides are raw token totals of one model
    family, not dollars.
    """
    from pydantic import ValidationError

    from rote.adapters.skill import planned_stops
    from rote.eval.estimate import (
        estimate_compiled_skill,
        estimate_pipeline,
        median_measured_run,
        run_total_tokens,
        total_tokens,
    )
    from rote.eval.priors import Priors
    from rote.eval.sidecar import EVAL_SIDECAR_FILENAME, load_eval_estimates
    from rote.eval.tokens import HeuristicTokenCounter
    from rote.ir import load_pipeline
    from rote.plan import PlanError

    picked = median_measured_run(runs)
    if picked is None:
        raise NoPrediction("no baseline run succeeded with its usage and nothing flagged")
    run, usable = picked
    try:
        pipeline = load_pipeline(pipeline_yaml)
    except (ValidationError, ValueError) as e:
        raise NoPrediction(f"the skeleton is not valid IR yet ({type(e).__name__})") from e
    try:
        stops = planned_stops(
            pipeline,
            judges="helper" if judges == "helper" else "session",
            backend=backend,
            code=code,
        )
    except PlanError as e:
        raise NoPrediction(f"the skill runtime cannot plan the skeleton yet ({e})") from e
    priors = Priors()
    sidecar_path = pipeline_yaml.parent / EVAL_SIDECAR_FILENAME
    sidecar = load_eval_estimates(sidecar_path) if sidecar_path.is_file() else None
    estimate = estimate_compiled_skill(
        stops,
        estimate_pipeline(pipeline, HeuristicTokenCounter(priors), priors, sidecar=sidecar),
        priors,
    )
    total = total_tokens(estimate)
    return Prediction(
        compiled=(total.low, total.high),
        measured=run_total_tokens(run),
        runs=len(usable),
        requests=(estimate.turns.low, estimate.turns.high),
        measured_turns=float(run["turns"]),
    )
