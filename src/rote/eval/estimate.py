"""Static estimators: raw skill (before) and compiled pipeline (after).

The two sides are deliberately asymmetric in confidence:

* :func:`estimate_pipeline` works on a closed world — the IR describes
  exactly what will execute. Waves give the critical path; only
  ``llm_judge`` and ``agent_loop`` nodes consume tokens; a judge's
  prompt template is countable text. The residual uncertainty (runtime
  payload sizes, loop iteration counts) is carried as explicit ranges.
* :func:`estimate_skill` models an open-world agent loop. The dominant
  unknown is the turn count; it comes from the compiler-emitted
  ``eval.yaml`` sidecar when present (an agent that read the skill
  estimated each step) or a labeled structural heuristic otherwise.
  Everything downstream of the turn count is arithmetic on the
  transcript-growth model, cache-aware because real agent runs use
  prompt caching.

No prices here — token counts and seconds only. Dollars are applied by
the scorecard using live-fetched prices.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

# _execution_waves is IR-only (no runtime semantics) and already the
# reference implementation every adapter uses; the estimator must agree
# with the emitted code's parallelism, so it reuses it rather than
# reimplementing.
from rote.adapters._common import _execution_waves
from rote.eval.priors import Priors
from rote.eval.sidecar import EvalEstimates
from rote.eval.tokens import TokenCounter
from rote.ir import Node, NodeKind, Pipeline, fan_out_element_param, parse_input_ref

# ───────── Ranges ─────────


@dataclass(frozen=True)
class Range:
    """An inclusive [low, high] estimate. Exact values have low == high."""

    low: float
    high: float

    def __post_init__(self) -> None:
        if self.low > self.high:
            raise ValueError(f"Range low ({self.low}) must be <= high ({self.high})")

    @staticmethod
    def exact(value: float) -> Range:
        return Range(value, value)

    @property
    def mid(self) -> float:
        return (self.low + self.high) / 2

    def __add__(self, other: Range) -> Range:
        return Range(self.low + other.low, self.high + other.high)

    def __mul__(self, other: Range) -> Range:
        """Product of two non-negative ranges (loop iterations x fan width)."""
        return Range(self.low * other.low, self.high * other.high)

    def scale(self, factor: float) -> Range:
        return Range(self.low * factor, self.high * factor)


ZERO = Range.exact(0.0)


# ───────── Shared result shapes ─────────


@dataclass(frozen=True)
class SamplingSurface:
    """Determinism, quantified as exposure to LLM sampling.

    ``sampled_steps`` counts steps whose outcome a sampled LLM decides;
    ``schema_constrained_steps`` counts the subset whose output is
    validated against a typed schema (retried on mismatch) — still
    sampled, but with a fenced blast radius. ``sampled_output_tokens``
    is the amount of sampled text per run: the rawest single measure of
    non-determinism surface.
    """

    total_steps: int
    sampled_steps: int
    schema_constrained_steps: int
    sampled_output_tokens: Range

    @property
    def sampled_fraction(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return self.sampled_steps / self.total_steps

    @property
    def roteness(self) -> float:
        """Share of steps that run as deterministic routine (code) rather
        than LLM inference — 0.0 = pure agent, 1.0 = pure code. A purely
        structural function of the pipeline's node kinds: no model, no
        estimate, same pipeline → same number."""
        if self.total_steps == 0:
            return 0.0
        return 1.0 - self.sampled_fraction


@dataclass(frozen=True)
class AgentRunEstimate:
    """Predicted footprint of running the skill as raw agent instructions."""

    turns: Range
    context_tokens: int
    """C₀: skill text + references + system/tool overhead."""
    fresh_input_tokens: Range
    """Tokens billed at input (cache-write) price: first-sight content."""
    cached_read_tokens: Range
    """Tokens re-read from prompt cache on later turns."""
    output_tokens: Range
    wall_seconds: Range
    sampling: SamplingSurface
    turn_method: str
    token_count_method: str
    measured: bool = False
    """True when the figures are a baseline run's own, not this model's."""


def estimate_compiled_skill(
    stops: Sequence[Mapping[str, Any]], pipeline: PipelineEstimate, priors: Priors | None = None
) -> AgentRunEstimate:
    """Predict one run of the pipeline as a compiled skill, in an agent's conversation.

    A workflow runtime pays for a judge as one API call. A compiled skill
    pays for the conversation: every request re-reads the agent's fixed
    context. Measured with paired evals on Haiku 5.5: a routing skill with
    one judge and one gate made 6 requests and used 138k input tokens where
    pricing its judge as an API call said 3.7k to 43k; expense-audit (one
    gate) made 5 and used 82k. Both evals add one request to write a result
    file, which this does not count.

    Requests, from ``stops`` (:func:`rote.skill_target._driver.stops`, built
    for the compile's own judge mode and backend):

    * one to start the run and one to finish;
    * a DECIDE block: one, to answer it;
    * a DO block: two, the agent's own tool call and then the answer;
    * a fanned DECIDE or DO block: that per batch of
      ``MAX_BATCH_ITEMS`` items, from the pipeline's fan width;
    * a question to the person, as its own HUMAN block or asked ahead by
      another block: one more to ask (the answer rides with the block's own
      for a question asked ahead, and is one more for a HUMAN block);
    * a LOOP block: the loop's own turns, which the agent runs itself;
    * a gate with a condition: in the high figure only.

    What the blocks show and the agent answers (each judge's input and
    output, a DO step's tool result) enters the conversation once and is
    re-read by the requests after it, half of them on average. A loop's
    figures from ``pipeline`` already hold its own re-reads, so they are
    added as they are. A judge no block asks (``--judges helper``) is the
    API call it is: its tokens are read once and never re-read.
    """
    from rote.skill_target._driver import MAX_BATCH_ITEMS

    priors = priors or Priors()
    nodes = {n.node_id: n for n in pipeline.nodes}
    widths = {w.node_id: w.elements for w in pipeline.fan_out_widths}
    zero = Range(0.0, 0.0)
    low = high = 2.0
    loop_turns = zero
    shown = zero  # enters the conversation once, re-read afterwards
    loop_in = zero  # a loop's own input total, re-reads included
    asked: set[str] = set()
    for stop in stops:
        steps = [nodes[s] for s in stop["steps"] if s in nodes]
        asked.update(n.node_id for n in steps)
        if stop["block"] == "LOOP":
            for node in steps:
                turns = node.calls.scale(priors.agent_loop_turns_per_iteration)
                loop_turns += turns
                loop_in += node.llm_input_tokens
            continue
        if stop["block"] == "HUMAN":
            cost = Range(2.0, 2.0)
        else:
            per_batch = 2.0 if stop["block"] == "DO" else 1.0
            width = widths.get(stop["steps"][0]) if stop.get("fanned") else None
            batches = (
                Range(1.0, 1.0)
                if width is None
                else Range(
                    max(1.0, math.ceil(width.low / MAX_BATCH_ITEMS)),
                    max(1.0, math.ceil(width.high / MAX_BATCH_ITEMS)),
                )
            )
            ahead = 1.0 if stop.get("ahead") else 0.0
            cost = Range(batches.low * per_batch + ahead, batches.high * per_batch + ahead)
            for node in steps:
                shown += node.llm_input_tokens + node.llm_output_tokens
                if stop["block"] == "DO":
                    shown += node.calls.scale(priors.tokens_per_external_call_result)
        high += cost.high
        if not stop.get("conditional"):
            low += cost.low
    # A judge or loop no block asks runs as an API call from the driver.
    api = [
        n
        for n in pipeline.nodes
        if n.node_id not in asked and n.kind in (NodeKind.LLM_JUDGE, NodeKind.AGENT_LOOP)
    ]
    api_in = sum((n.llm_input_tokens for n in api), zero)
    api_out = sum((n.llm_output_tokens for n in api), zero)
    own = Range(low, high)  # requests outside any loop
    requests = own + loop_turns
    base = float(priors.system_overhead_tokens)
    judged_out = sum((n.llm_output_tokens for n in pipeline.nodes if n.node_id in asked), zero)
    return AgentRunEstimate(
        turns=requests,
        context_tokens=priors.system_overhead_tokens,
        fresh_input_tokens=Range(base + shown.low, base + shown.high) + api_in,
        cached_read_tokens=Range(
            (requests.low - 1) * (base + shown.low / 2),
            (requests.high - 1) * (base + shown.high / 2),
        )
        + loop_in,
        # A loop's output already counts its turns; ``judged_out`` holds it.
        output_tokens=judged_out + api_out + own.scale(priors.output_tokens_per_turn),
        wall_seconds=requests.scale(priors.seconds_per_turn),
        sampling=pipeline.sampling,
        turn_method=(
            "one request to start, one to finish, one per DECIDE block, two per DO block, "
            "one more per question to the person, a loop's own turns"
        ),
        token_count_method=(
            f"each request re-reads {priors.system_overhead_tokens} tokens of fixed context "
            "(`system_overhead_tokens`) plus what the blocks before it showed"
        ),
    )


_RUN_BUCKETS = ("input_tokens", "cache_read_tokens", "cache_creation_tokens", "output_tokens")


def median_measured_run(
    runs: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]] | None:
    """The median run by total tokens, and every run that counts, or None.

    A run counts when it succeeded with nothing flagged and reported its
    usage: a truncated run, or one missing its tools, is not the skill's cost.
    """
    usable = [
        r
        for r in runs
        if r.get("succeeded")
        and not r.get("flags")
        and r.get("turns")
        and all(r.get(b) is not None for b in _RUN_BUCKETS)
    ]
    if not usable:
        return None
    by_total = sorted(usable, key=run_total_tokens)
    return by_total[(len(by_total) - 1) // 2], usable


def run_total_tokens(run: Mapping[str, Any]) -> float:
    """Every token a measured run reported, cache reads and writes included."""
    return float(sum(run[b] for b in _RUN_BUCKETS))


def total_tokens(estimate: AgentRunEstimate) -> Range:
    """Every token of an agent run, the figure :func:`run_total_tokens` measures."""
    return estimate.fresh_input_tokens + estimate.cached_read_tokens + estimate.output_tokens


def measured_agent_run(
    estimate: AgentRunEstimate, runs: Sequence[Mapping[str, Any]]
) -> AgentRunEstimate | None:
    """``estimate`` with every figure a baseline measured put in its place.

    ``runs`` are the skill's own measured runs as ``measured_run_record``
    writes them (a compile's baseline, or its saved ``metrics.json``). A
    static turn count can be far off: a skill that reads one uploaded file
    and asks once was sized at 9 steps and 287k to 479k tokens while its
    baseline, run in the same compile, took 3 turns and 28k. So a run that
    succeeded with nothing flagged and reported its usage replaces the
    model; a truncated run, or one missing its tools, is not the skill's
    cost. What a run does not report (the step counts) stays the
    estimate's. ``None`` when no run qualifies.

    Turns and wall time span the runs. The token buckets are one run's, the
    median by total tokens: a low built from one run's cache reads and
    another's cache writes would be below anything a run cost.
    """
    picked = median_measured_run(runs)
    if picked is None:
        return None
    run, usable = picked

    def span(key: str) -> Range:
        values = [float(r[key]) for r in usable]
        return Range(min(values), max(values))

    def one(value: float) -> Range:
        return Range(float(value), float(value))

    models = sorted({str(r["model"]) for r in usable if r.get("model")})
    where = f" on {', '.join(models)}" if models else ""
    how = f"measured, {len(usable)} baseline run(s) of the skill{where}"
    return replace(
        estimate,
        turns=span("turns"),
        fresh_input_tokens=one(run["input_tokens"] + run["cache_creation_tokens"]),
        cached_read_tokens=one(run["cache_read_tokens"]),
        output_tokens=one(run["output_tokens"]),
        wall_seconds=span("wall_seconds"),
        sampling=replace(estimate.sampling, sampled_output_tokens=one(run["output_tokens"])),
        turn_method=how,
        token_count_method=(
            f"{how}: the usage the run reported"
            + (", from the median run by total tokens" if len(usable) > 1 else "")
        ),
        measured=True,
    )


@dataclass(frozen=True)
class InputSize:
    """How many tokens one bound input brings into a model call, and why.

    ``basis`` names where the number came from (a prior, a declared
    schema, the one-field allowance), so the scorecard can state every
    size it assumed instead of printing a precise-looking total.
    """

    param: str
    ref: str
    tokens: Range
    basis: str


@dataclass(frozen=True)
class NodeEstimate:
    """Per-run footprint of one pipeline node."""

    node_id: str
    kind: NodeKind
    calls: Range
    """How many times the node body runs (loops make this a range)."""
    llm_input_tokens_per_call: Range
    llm_output_tokens_per_call: Range
    wall_seconds_per_call: float
    note: str | None = None
    """A per-node modeling assumption worth surfacing (e.g., an unbounded
    loop estimated at the prior default)."""
    sequential_calls: Range | None = None
    """Calls that run one after another, when fewer than ``calls``. A
    fan_out node's elements run concurrently on every runtime, so they
    multiply its tokens but not its wall time. None means ``calls``."""
    input_sizes: tuple[InputSize, ...] = ()
    """For a judge or agent loop: the size of each bound input, per call."""

    @property
    def llm_input_tokens(self) -> Range:
        return self.calls * self.llm_input_tokens_per_call

    @property
    def llm_output_tokens(self) -> Range:
        return self.calls * self.llm_output_tokens_per_call

    @property
    def wall_seconds(self) -> Range:
        sequential = self.sequential_calls if self.sequential_calls is not None else self.calls
        return sequential.scale(self.wall_seconds_per_call)


@dataclass(frozen=True)
class FanOutWidth:
    """Elements a ``fan_out`` node runs over per run, and where that came from."""

    node_id: str
    elements: Range
    source: str


@dataclass(frozen=True)
class PipelineEstimate:
    """Predicted footprint of one run of the compiled pipeline."""

    nodes: list[NodeEstimate]
    critical_path_seconds: Range
    """Compute time along the wave critical path — HITL waits excluded."""
    hitl_gates: list[str]
    """Gate node ids: each adds a human-approval wait the estimate
    deliberately does not guess at."""
    llm_input_tokens: Range
    llm_output_tokens: Range
    sampling: SamplingSurface
    fan_out_widths: tuple[FanOutWidth, ...] = ()
    """The width assumed for every fan_out node, for the Assumptions list."""


# ───────── Data sizes along the IR's data flow ─────────
#
# A model node's input is whatever its ``inputs:`` bind, so its size comes
# from the node that produced the value, not from how many fields the
# judge's schema lists. The rules, in one place:
#
# * an external_call's whole result is the same payload the before side
#   charges the agent for (``external_result_tokens``);
# * a field of a result is somewhere between one value and the whole
#   result, unless its declared type bounds it (a number, a boolean);
# * a pure function's output, when no schema declares it, is somewhere
#   between one value and everything it reads (a filter or a formatter);
# * a list is one element up to ``elements_per_list`` of them;
# * a value nothing describes gets the stated one-field allowance.

_BOUNDED_TYPES = frozenset({"number", "integer", "boolean", "null"})
_BOUNDED_TYPE_NAMES = frozenset({"int", "integer", "float", "number", "bool", "boolean"})
_MAX_SCHEMA_DEPTH = 8


@dataclass(frozen=True)
class _Size:
    tokens: Range
    basis: str


def _schema_type(schema: dict[str, Any]) -> str | None:
    kind = schema.get("type")
    if isinstance(kind, list):
        kinds = [k for k in kind if k != "null"]
        return kinds[0] if len(kinds) == 1 and isinstance(kinds[0], str) else None
    return kind if isinstance(kind, str) else None


def _is_bounded(schema: Any) -> bool:
    """A value whose type caps its size at one short scalar."""
    if not isinstance(schema, dict):
        return False
    if "enum" in schema or "const" in schema:
        return True
    return _schema_type(schema) in _BOUNDED_TYPES


def _defs(schema: Any) -> dict[str, Any]:
    """A schema's local definitions (``$defs``, or the older ``definitions``)."""
    if not isinstance(schema, dict):
        return {}
    found = schema.get("$defs", schema.get("definitions"))
    return found if isinstance(found, dict) else {}


def schema_tokens(
    schema: Any,
    per_value: float,
    priors: Priors,
    defs: dict[str, Any] | None = None,
    _depth: int = 0,
) -> Range:
    """Tokens a value of ``schema`` carries, with ``per_value`` per scalar.

    An object sums its declared properties; a free-form object and a list
    are ranges (one value or element up to ``priors.elements_per_list``),
    never one short scalar. A local ``$ref`` is followed through ``defs``
    (the root schema's own definitions by default), and an ``anyOf`` /
    ``oneOf`` spans its smallest to its largest branch. A part rote cannot
    read (a remote ``$ref``, past the depth limit) is one value.
    """
    defs = _defs(schema) if defs is None else defs
    one = Range.exact(per_value)
    if not isinstance(schema, dict) or _depth > _MAX_SCHEMA_DEPTH:
        return one

    def child(sub: Any) -> Range:
        return schema_tokens(sub, per_value, priors, defs, _depth + 1)

    ref = schema.get("$ref")
    if isinstance(ref, str):
        name = ref.rpartition("/")[2]
        local = ref.startswith(("#/$defs/", "#/definitions/"))
        return child(defs[name]) if local and name in defs else one
    branches = schema.get("anyOf", schema.get("oneOf"))
    if isinstance(branches, list) and branches:
        sizes = [child(branch) for branch in branches]
        return Range(min(s.low for s in sizes), max(s.high for s in sizes))
    kind = _schema_type(schema)
    if kind == "array":
        return child(schema.get("items")) * Range(1.0, float(priors.elements_per_list))
    props = schema.get("properties")
    if isinstance(props, dict) and props:
        total = ZERO
        for prop in props.values():
            total = total + child(prop)
        return total
    if kind == "object":
        return Range(per_value, per_value * priors.elements_per_list)
    return one


def external_result_tokens(node: Node, priors: Priors) -> tuple[float, str]:
    """Tokens one call of an external_call returns, and which prior said so.

    The one source of the payload size for both sides: the before side's
    transcript and every compiled model node that receives the result.
    """
    tool = node.mcp.tool if node.mcp is not None else None
    if tool is not None and tool in priors.payload_tokens_per_tool:
        return priors.payload_tokens_per_tool[tool], f"payload_tokens_per_tool.{tool}"
    return priors.tokens_per_external_call_result, "tokens_per_external_call_result"


def _output_schema(node: Node) -> dict[str, Any] | None:
    if node.signature_spec is not None:
        return node.signature_spec.output_schema
    return node.output_schema


def _input_schema_property(node: Node, param: str) -> Any:
    schema = node.signature_spec.input_schema if node.signature_spec else node.input_schema
    props = schema.get("properties") if isinstance(schema, dict) else None
    return props.get(param) if isinstance(props, dict) else None


def _field_is_bounded(node: Node, field_name: str) -> bool:
    schema = _output_schema(node)
    props = schema.get("properties") if isinstance(schema, dict) else None
    if isinstance(props, dict) and field_name in props:
        return _is_bounded(props[field_name])
    if isinstance(node.output, dict) and field_name in node.output:
        return node.output[field_name].strip().lower() in _BOUNDED_TYPE_NAMES
    return False


def _between_one_and(whole: Range, per_value: float) -> Range:
    return Range(min(per_value, whole.low), max(per_value, whole.high))


def _fmt_range(r: Range) -> str:
    return f"{r.low:g}" if r.low == r.high else f"{r.low:g}-{r.high:g}"


def _ref_size(ref: str, pipeline: Pipeline, outputs: dict[str, _Size], priors: Priors) -> _Size:
    """The size of the value an ``inputs:`` reference binds."""
    one = priors.tokens_per_input_field
    parsed = parse_input_ref(ref)
    if parsed.node_id is None:
        schema = pipeline.input.input_schema
        if parsed.field is None:
            if schema is not None:
                return _Size(schema_tokens(schema, one, priors), "the pipeline input schema")
            fields_declared = len(pipeline.input.required) + len(pipeline.input.optional)
            return _Size(
                Range.exact(one * max(fields_declared, 1)),
                "no input schema: tokens_per_input_field per declared input field",
            )
        props = schema.get("properties") if isinstance(schema, dict) else None
        if isinstance(props, dict) and parsed.field in props:
            return _Size(
                schema_tokens(props[parsed.field], one, priors, _defs(schema)),
                "the pipeline input schema",
            )
        return _Size(Range.exact(one), "an undeclared pipeline input field: tokens_per_input_field")

    whole = outputs[parsed.node_id]
    if parsed.field is None:
        return _Size(whole.tokens, f"the whole output of `{parsed.node_id}` ({whole.basis})")
    if _field_is_bounded(pipeline.node_by_id(parsed.node_id), parsed.field):
        return _Size(
            Range.exact(one),
            f"`{parsed.node_id}.{parsed.field}` is declared a scalar: tokens_per_input_field",
        )
    return _Size(
        _between_one_and(whole.tokens, one),
        f"a field of `{parsed.node_id}`'s output, size unknown: between one value "
        f"(tokens_per_input_field) and the whole output ({whole.basis})",
    )


def _bound_inputs(
    node: Node,
    pipeline: Pipeline,
    outputs: dict[str, _Size],
    priors: Priors,
    width: FanOutWidth | None,
) -> tuple[InputSize, ...]:
    """The per-call size of every input ``node`` binds."""
    if not node.inputs or node.kind not in _MODEL_KINDS | {NodeKind.PURE_FUNCTION}:
        return ()
    one = priors.tokens_per_input_field
    fan_param = fan_out_element_param(node, pipeline) if width is not None else None
    sizes: list[InputSize] = []
    for param, ref in node.inputs.items():
        size = _ref_size(ref, pipeline, outputs, priors)
        tokens, basis = size.tokens, size.basis
        if param == fan_param and width is not None:
            # Each call gets one element: the list's size spread over the
            # elements, never less than one value.
            tokens = Range(
                max(one, tokens.low / max(width.elements.high, 1.0)),
                max(one, tokens.high / max(width.elements.low, 1.0)),
            )
            basis = f"one element of {basis}, spread over {_fmt_range(width.elements)} elements"
        declared = _input_schema_property(node, param)
        if _is_bounded(declared) and tokens.high > one:
            tokens = Range.exact(one)
            basis = f"declared a scalar by `{node.id}`'s input schema: tokens_per_input_field"
        sizes.append(InputSize(param=param, ref=ref, tokens=tokens, basis=basis))
    return tuple(sizes)


def _inputs_total(sizes: tuple[InputSize, ...]) -> Range:
    total = ZERO
    for size in sizes:
        total = total + size.tokens
    return total


def _model_written_tokens(node: Node, priors: Priors) -> Range:
    """Tokens of a judge's answer, or of the output an agent loop submits.

    Passthrough fields are copied from the input in code, so the model
    writes none of them.
    """
    per_field = priors.tokens_per_output_field
    schema = _output_schema(node)
    copied = (node.signature_spec.passthrough or {}) if node.signature_spec else {}
    if isinstance(schema, dict):
        props = schema.get("properties")
        if isinstance(props, dict) and props:
            written = ZERO
            for name, prop in props.items():
                if name not in copied:
                    written = written + schema_tokens(prop, per_field, priors, _defs(schema))
            tokens = written if written.high > 0 else Range.exact(per_field)
        else:
            tokens = schema_tokens(schema, per_field, priors)
    else:
        # Legacy path-based signature: the IR's documentary output map.
        n_out = len(node.output) if isinstance(node.output, dict) and node.output else 1
        tokens = Range.exact(n_out * per_field)
    return Range(max(16.0, tokens.low), max(16.0, tokens.high))


def _passthrough_tokens(node: Node, inputs: tuple[InputSize, ...], priors: Priors) -> Range:
    """Tokens a judge's passthrough fields carry downstream (copied, not written)."""
    if node.signature_spec is None or not node.signature_spec.passthrough:
        return ZERO
    one = priors.tokens_per_input_field
    by_param = {size.param: size.tokens for size in inputs}
    total = ZERO
    for path in node.signature_spec.passthrough.values():
        # A per-item source (``prs[number].body``) is read from its input list.
        key, _, rest = path.split("[", 1)[0].partition(".")
        source = by_param.get(key, Range.exact(one))
        total = total + (_between_one_and(source, one) if rest else source)
    return total


def _output_size(
    node: Node,
    inputs: tuple[InputSize, ...],
    priors: Priors,
    width: FanOutWidth | None,
) -> _Size:
    """The size of one run's worth of ``node``'s output, for its consumers."""
    one = priors.tokens_per_input_field
    if node.kind is NodeKind.EXTERNAL_CALL:
        per_call, prior = external_result_tokens(node, priors)
        size = _Size(Range.exact(per_call), f"prior {prior} per call")
    elif node.kind is NodeKind.PURE_FUNCTION:
        read = _inputs_total(inputs)
        if node.output_schema is not None:
            size = _Size(schema_tokens(node.output_schema, one, priors), "its output schema")
        elif read.high > 0:
            size = _Size(
                _between_one_and(read, one),
                "an undeclared pure_function output: between one value and everything it reads",
            )
        else:
            size = _Size(Range.exact(one), "an undeclared pure_function output: one value")
    elif node.kind in _MODEL_KINDS:
        size = _Size(
            _model_written_tokens(node, priors) + _passthrough_tokens(node, inputs, priors),
            "its declared output, tokens_per_output_field per value, plus passthrough copies",
        )
    else:
        n_fields = len(node.output) if isinstance(node.output, dict) and node.output else 1
        size = _Size(Range.exact(one * n_fields), "a person's resume payload")
    if width is None:
        return size
    return _Size(
        size.tokens * width.elements,
        f"{size.basis}, times {_fmt_range(width.elements)} fan_out elements",
    )


_MODEL_KINDS = frozenset({NodeKind.LLM_JUDGE, NodeKind.AGENT_LOOP})


# ───────── After: pipeline estimator ─────────


def fan_out_width(node: Node, priors: Priors, sidecar: EvalEstimates | None) -> FanOutWidth:
    """How many elements a ``fan_out`` node runs over in one run.

    The list is data, so the IR cannot say. The compiler's ``eval.yaml``
    can: it records ``iterations`` (per item, per row) for the source step
    a node came from. Without that, the stated, overridable
    ``fan_out_elements_per_run`` prior applies. Never 1 by default: that
    would price a fanned judge as a single call.
    """
    if sidecar is not None:
        for step in sidecar.steps:
            if step.node_id == node.id and step.iterations is not None:
                return FanOutWidth(
                    node_id=node.id,
                    elements=Range(step.iterations.low, step.iterations.high),
                    source="eval.yaml iterations",
                )
    return FanOutWidth(
        node_id=node.id,
        elements=Range.exact(float(priors.fan_out_elements_per_run)),
        source="prior fan_out_elements_per_run",
    )


def _unbound_field_count(node: Node) -> int:
    if node.signature_spec is not None:
        props = node.signature_spec.input_schema.get("properties")
        return max(len(props), 1) if isinstance(props, dict) else 1
    return max(len(node.input), 1)


def _estimate_node(
    node: Node,
    counter: TokenCounter,
    priors: Priors,
    inputs: tuple[InputSize, ...],
) -> NodeEstimate:
    if node.kind in (NodeKind.PURE_FUNCTION, NodeKind.EXTERNAL_CALL):
        seconds = (
            priors.pure_function_seconds
            if node.kind is NodeKind.PURE_FUNCTION
            else priors.external_call_seconds
        )
        return NodeEstimate(
            node_id=node.id,
            kind=node.kind,
            calls=Range.exact(1),
            llm_input_tokens_per_call=ZERO,
            llm_output_tokens_per_call=ZERO,
            wall_seconds_per_call=seconds,
        )

    if node.kind is NodeKind.LLM_JUDGE:
        prompt_tokens = (
            counter.count(node.signature_spec.prompt) if node.signature_spec is not None else 0
        )
        if node.inputs:
            payload = _inputs_total(inputs)
        else:
            # No data-flow bindings (a hand-written or older IR): one
            # allowance per field the input schema lists.
            payload = Range.exact(_unbound_field_count(node) * priors.tokens_per_input_field)
        output_tokens = _model_written_tokens(node, priors)
        return NodeEstimate(
            node_id=node.id,
            kind=node.kind,
            calls=Range.exact(1),
            llm_input_tokens_per_call=payload + Range.exact(prompt_tokens),
            llm_output_tokens_per_call=output_tokens,
            wall_seconds_per_call=(
                priors.llm_ttft_seconds + output_tokens.high / priors.llm_output_tokens_per_second
            ),
            input_sizes=inputs,
        )

    if node.kind is NodeKind.AGENT_LOOP:
        note: str | None = None
        if node.termination is not None:
            max_iterations = node.termination.max_iterations
        else:
            max_iterations = priors.agent_loop_default_max_iterations
            note = (
                f"no termination config; assumed <= {max_iterations} iterations "
                f"(priors.agent_loop_default_max_iterations)"
            )
        turns = priors.agent_loop_turns_per_iteration
        # Every turn re-reads the loop's context, and the bound inputs are
        # in it from the first turn on.
        per_turn = _inputs_total(inputs) + Range.exact(priors.transcript_growth_per_turn)
        return NodeEstimate(
            node_id=node.id,
            kind=node.kind,
            calls=Range(1, float(max_iterations)),
            llm_input_tokens_per_call=Range(
                round(turns * per_turn.low), round(turns * per_turn.high)
            ),
            llm_output_tokens_per_call=Range.exact(round(turns * priors.output_tokens_per_turn)),
            wall_seconds_per_call=turns * priors.seconds_per_turn,
            note=note,
            input_sizes=inputs,
        )

    # hitl_gate: zero compute; the wait is human time, reported separately.
    return NodeEstimate(
        node_id=node.id,
        kind=node.kind,
        calls=Range.exact(1),
        llm_input_tokens_per_call=ZERO,
        llm_output_tokens_per_call=ZERO,
        wall_seconds_per_call=0.0,
    )


def estimate_pipeline(
    pipeline: Pipeline,
    counter: TokenCounter,
    priors: Priors | None = None,
    sidecar: EvalEstimates | None = None,
) -> PipelineEstimate:
    """Predict one run of the compiled pipeline, without executing it.

    Uses the same wave decomposition the adapters emit, so the critical
    path matches the parallelism of the generated code. Loop-body
    sub-nodes are costed inside their parent loop's per-iteration model,
    exactly as they execute. Waves run in data-flow order, so every model
    node's input is sized from the output its source produced.
    """
    priors = priors or Priors()
    waves = _execution_waves(pipeline)

    estimates: dict[str, NodeEstimate] = {}
    outputs: dict[str, _Size] = {}
    widths: list[FanOutWidth] = []
    for wave in waves:
        for node in wave:
            width = (
                fan_out_width(node, priors, sidecar)
                if node.fan_out and node.kind is not NodeKind.HITL_GATE
                else None
            )
            inputs = _bound_inputs(node, pipeline, outputs, priors, width)
            estimate = _estimate_node(node, counter, priors, inputs)
            outputs[node.id] = _output_size(node, inputs, priors, width)
            if width is not None:
                widths.append(width)
                estimate = replace(
                    estimate,
                    calls=estimate.calls * width.elements,
                    sequential_calls=estimate.calls,
                )
            estimates[node.id] = estimate

    critical_path = ZERO
    for wave in waves:
        wave_estimates = [estimates[n.id] for n in wave]
        low = max(e.wall_seconds.low for e in wave_estimates)
        high = max(e.wall_seconds.high for e in wave_estimates)
        critical_path = critical_path + Range(low, high)

    all_estimates = list(estimates.values())
    llm_input = ZERO
    llm_output = ZERO
    for e in all_estimates:
        llm_input = llm_input + e.llm_input_tokens
        llm_output = llm_output + e.llm_output_tokens

    sampled = [e for e in all_estimates if e.kind in (NodeKind.LLM_JUDGE, NodeKind.AGENT_LOOP)]
    sampled_tokens = ZERO
    for e in sampled:
        sampled_tokens = sampled_tokens + e.llm_output_tokens

    return PipelineEstimate(
        nodes=all_estimates,
        critical_path_seconds=critical_path,
        hitl_gates=[e.node_id for e in all_estimates if e.kind is NodeKind.HITL_GATE],
        llm_input_tokens=llm_input,
        llm_output_tokens=llm_output,
        sampling=SamplingSurface(
            total_steps=len(all_estimates),
            sampled_steps=len(sampled),
            schema_constrained_steps=sum(1 for e in sampled if e.kind is NodeKind.LLM_JUDGE),
            sampled_output_tokens=sampled_tokens,
        ),
        fan_out_widths=tuple(widths),
    )


# ───────── Before: raw-skill estimator ─────────


def external_call_payload_tokens(
    pipeline: Pipeline,
    priors: Priors | None = None,
    sidecar: EvalEstimates | None = None,
) -> float:
    """Estimate the data the agent pulls into context, from the pipeline's
    ``external_call`` footprint.

    The compiled pipeline's ``external_call`` nodes name the sources the
    skill fetches (Slack, Gmail, a Drive file …); the *before*-side agent
    pulls the same data, so its transcript carries the same payload. Each
    call contributes :func:`external_result_tokens`, the same size the
    after side carries to every judge and agent loop that receives the
    result. A fan_out external_call is one call per element, on both
    sides, so it contributes its width (the sidecar's ``iterations`` or the
    ``fan_out_elements_per_run`` prior) times that. The midpoint of the
    width, because this side's context is a single number.

    Uses the same wave decomposition as :func:`estimate_pipeline`, so
    loop-body sub-nodes are not double-counted.
    """
    priors = priors or Priors()
    total = 0.0
    for wave in _execution_waves(pipeline):
        for node in wave:
            if node.kind is not NodeKind.EXTERNAL_CALL:
                continue
            per_call, _ = external_result_tokens(node, priors)
            calls = fan_out_width(node, priors, sidecar).elements.mid if node.fan_out else 1.0
            total += per_call * calls
    return total


_STEP_LINE_RE = re.compile(r"^\s{0,3}(?:\d+[.)]\s|#{2,3}\s|[-*]\s+\*\*)", re.MULTILINE)


def _skill_corpus(skill_dir: Path) -> str:
    """All text the agent would load: SKILL.md plus references/."""
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.is_file():
        raise FileNotFoundError(f"{skill_dir} does not contain a SKILL.md")
    parts = [skill_md.read_text(encoding="utf-8")]
    references = skill_dir / "references"
    if references.is_dir():
        for path in sorted(references.rglob("*")):
            if path.is_file():
                try:
                    parts.append(path.read_text(encoding="utf-8"))
                except UnicodeDecodeError:
                    continue  # binary asset; the agent wouldn't read it as text
    return "\n".join(parts)


def _structural_step_count(skill_md_text: str) -> int:
    """Identifiable steps in a SKILL.md: numbered items, section headings,
    and bold-led bullets — the shapes skills use for procedure steps."""
    return max(3, len(_STEP_LINE_RE.findall(skill_md_text)))


def estimate_skill(
    skill_dir: str | Path,
    counter: TokenCounter,
    priors: Priors | None = None,
    sidecar: EvalEstimates | None = None,
    data_payload_tokens: float = 0.0,
) -> AgentRunEstimate:
    """Predict running the skill as raw agent instructions.

    Cache-aware transcript model: turn *i* re-reads
    min(C₀ + (i−1)·Δ, cap) tokens of context, of which only ~Δ is
    first-sight (billed at cache-write price); the rest is a cache read.
    The cap (``priors.transcript_cap_tokens``) models harness compaction:
    real transcripts saturate near the context window instead of growing
    without bound, so on long runs the cached-read total transitions from
    quadratic to linear. Uncapped totals over N turns:

    * fresh  ≈ C₀ + (N−1)·Δ
    * cached ≈ (N−1)·C₀ + Δ·(N−2)(N−1)/2

    ``data_payload_tokens`` (the sources the skill pulls, from
    :func:`external_call_payload_tokens`) folds into C₀: fetched-once data
    is billed as one cache-write and re-read on every later turn — exactly
    C₀'s billing shape, and the dominant cost on data-heavy skills.
    """
    priors = priors or Priors()
    skill_path = Path(skill_dir)
    corpus = _skill_corpus(skill_path)
    context_tokens = (
        counter.count(corpus) + priors.system_overhead_tokens + round(data_payload_tokens)
    )

    if sidecar is not None and (sidecar.steps or sidecar.totals is not None):
        tr = sidecar.turn_range()
        turns = Range(tr.low, tr.high)
        turn_method = "compiler sidecar (eval.yaml)"
        step_count = len(sidecar.steps) or _structural_step_count(
            (skill_path / "SKILL.md").read_text(encoding="utf-8")
        )
    else:
        step_count = _structural_step_count((skill_path / "SKILL.md").read_text(encoding="utf-8"))
        turns = Range(
            step_count * priors.turns_per_step_low,
            step_count * priors.turns_per_step_high,
        )
        turn_method = f"structural heuristic ({step_count} steps in SKILL.md)"

    def _fresh(n: float) -> float:
        # Fresh (first-sight) content is unaffected by the cap: each
        # turn's new tool results are written to cache once even when
        # compaction evicts older content to make room.
        return context_tokens + max(0.0, n - 1) * priors.transcript_growth_per_turn

    def _cached(n: float) -> float:
        # Turn i (i >= 2) re-reads min(C₀ + (i−2)·Δ, cap): quadratic
        # growth until the transcript saturates at the compaction cap,
        # linear after. m counts the pre-saturation re-reads.
        if n <= 1:
            return 0.0
        delta = priors.transcript_growth_per_turn
        cap = priors.transcript_cap_tokens
        reads = n - 1
        if context_tokens >= cap or delta <= 0:
            return reads * min(float(context_tokens), cap)
        m = min(reads, (cap - context_tokens) / delta + 1)
        return m * context_tokens + delta * (m - 1) * m / 2 + (reads - m) * cap

    output_tokens = turns.scale(priors.output_tokens_per_turn)

    return AgentRunEstimate(
        turns=turns,
        context_tokens=context_tokens,
        fresh_input_tokens=Range(_fresh(turns.low), _fresh(turns.high)),
        cached_read_tokens=Range(_cached(turns.low), _cached(turns.high)),
        output_tokens=output_tokens,
        wall_seconds=turns.scale(priors.seconds_per_turn),
        sampling=SamplingSurface(
            total_steps=step_count,
            sampled_steps=step_count,
            schema_constrained_steps=0,
            sampled_output_tokens=output_tokens,
        ),
        turn_method=turn_method,
        token_count_method=counter.method,
    )
