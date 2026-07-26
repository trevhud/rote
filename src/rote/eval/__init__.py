"""Eval harness: speed / cost / determinism, before and after compilation.

The package answers one question in two modes:

*How much does this skill cost to run as agent instructions, and what
does compiling it buy you?* — expressed on three axes: wall-clock
speed, dollar cost (across a sampling of current models at live
official prices), and determinism (how much of the work product passes
through a sampled LLM).

Static mode (this package, Phase 1) predicts without executing
anything: the "after" side is derived almost exactly from the
:class:`rote.ir.Pipeline` IR; the "before" side is modeled from the
source skill's structure plus calibratable priors, with every uncertain
quantity carried as an explicit :class:`~rote.eval.estimate.Range` and
every method named in the output. Prices and model lists are fetched
live (see :mod:`rote.eval.pricing`) — nothing is hardcoded.
"""

from rote.eval.estimate import (
    AgentRunEstimate,
    NodeEstimate,
    PipelineEstimate,
    Range,
    SamplingSurface,
    estimate_pipeline,
    estimate_skill,
    external_call_payload_tokens,
)
from rote.eval.priors import Priors, priors_from_overrides
from rote.eval.scorecard import Scorecard, build_scorecard, build_scorecard_for
from rote.eval.sidecar import EvalEstimates, load_eval_estimates

__all__ = [
    "AgentRunEstimate",
    "EvalEstimates",
    "NodeEstimate",
    "PipelineEstimate",
    "Priors",
    "Range",
    "SamplingSurface",
    "Scorecard",
    "build_scorecard",
    "build_scorecard_for",
    "estimate_pipeline",
    "estimate_skill",
    "external_call_payload_tokens",
    "load_eval_estimates",
    "priors_from_overrides",
]
