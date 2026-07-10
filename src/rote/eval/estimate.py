"""Static estimators: raw skill (before) and graduated pipeline (after).

The two sides are deliberately asymmetric in confidence:

* :func:`estimate_pipeline` works on a closed world — the IR describes
  exactly what will execute. Waves give the critical path; only
  ``llm_judge`` and ``agent_loop`` nodes consume tokens; a judge's
  prompt template is countable text. The residual uncertainty (runtime
  payload sizes, loop iteration counts) is carried as explicit ranges.
* :func:`estimate_skill` models an open-world agent loop. The dominant
  unknown is the turn count; it comes from the graduator-emitted
  ``eval.yaml`` sidecar when present (an agent that read the skill
  estimated each step) or a labeled structural heuristic otherwise.
  Everything downstream of the turn count is arithmetic on the
  transcript-growth model, cache-aware because real agent runs use
  prompt caching.

No prices here — token counts and seconds only. Dollars are applied by
the scorecard using live-fetched prices.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# _execution_waves is IR-only (no runtime semantics) and already the
# reference implementation every adapter uses; the estimator must agree
# with the emitted code's parallelism, so it reuses it rather than
# reimplementing.
from rote.adapters._common import _execution_waves
from rote.eval.priors import Priors
from rote.eval.sidecar import EvalEstimates
from rote.eval.tokens import TokenCounter
from rote.ir import Node, NodeKind, Pipeline

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


@dataclass(frozen=True)
class NodeEstimate:
    """Per-run footprint of one pipeline node."""

    node_id: str
    kind: NodeKind
    calls: Range
    """How many times the node body runs (loops make this a range)."""
    llm_input_tokens_per_call: int
    llm_output_tokens_per_call: int
    wall_seconds_per_call: float
    note: str | None = None
    """A per-node modeling assumption worth surfacing (e.g., an unbounded
    loop estimated at the prior default)."""

    @property
    def llm_input_tokens(self) -> Range:
        return self.calls.scale(self.llm_input_tokens_per_call)

    @property
    def llm_output_tokens(self) -> Range:
        return self.calls.scale(self.llm_output_tokens_per_call)

    @property
    def wall_seconds(self) -> Range:
        return self.calls.scale(self.wall_seconds_per_call)


@dataclass(frozen=True)
class PipelineEstimate:
    """Predicted footprint of one run of the graduated pipeline."""

    nodes: list[NodeEstimate]
    critical_path_seconds: Range
    """Compute time along the wave critical path — HITL waits excluded."""
    hitl_gates: list[str]
    """Gate node ids: each adds a human-approval wait the estimate
    deliberately does not guess at."""
    llm_input_tokens: Range
    llm_output_tokens: Range
    sampling: SamplingSurface


# ───────── After: pipeline estimator ─────────


def _judge_field_counts(node: Node) -> tuple[int, int]:
    """(input fields, output fields) for an llm_judge node."""
    if node.signature_spec is not None:
        in_props = node.signature_spec.input_schema.get("properties")
        out_props = node.signature_spec.output_schema.get("properties")
        n_in = len(in_props) if isinstance(in_props, dict) else 1
        n_out = len(out_props) if isinstance(out_props, dict) else 1
        return max(n_in, 1), max(n_out, 1)
    # Legacy path-based signature: fall back to the IR's documentary
    # input/output field maps.
    n_in = len(node.input) if node.input else 1
    n_out = len(node.output) if isinstance(node.output, dict) and node.output else 1
    return max(n_in, 1), max(n_out, 1)


def _estimate_node(node: Node, counter: TokenCounter, priors: Priors) -> NodeEstimate:
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
            llm_input_tokens_per_call=0,
            llm_output_tokens_per_call=0,
            wall_seconds_per_call=seconds,
        )

    if node.kind is NodeKind.LLM_JUDGE:
        n_in, n_out = _judge_field_counts(node)
        prompt_tokens = (
            counter.count(node.signature_spec.prompt) if node.signature_spec is not None else 0
        )
        input_tokens = prompt_tokens + round(n_in * priors.tokens_per_input_field)
        output_tokens = max(16, round(n_out * priors.tokens_per_output_field))
        seconds = priors.llm_ttft_seconds + output_tokens / priors.llm_output_tokens_per_second
        return NodeEstimate(
            node_id=node.id,
            kind=node.kind,
            calls=Range.exact(1),
            llm_input_tokens_per_call=input_tokens,
            llm_output_tokens_per_call=output_tokens,
            wall_seconds_per_call=seconds,
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
        iterations = Range(1, float(max_iterations))
        turns = priors.agent_loop_turns_per_iteration
        input_tokens = round(turns * priors.transcript_growth_per_turn)
        output_tokens = round(turns * priors.output_tokens_per_turn)
        seconds = turns * priors.seconds_per_turn
        return NodeEstimate(
            node_id=node.id,
            kind=node.kind,
            calls=iterations,
            llm_input_tokens_per_call=input_tokens,
            llm_output_tokens_per_call=output_tokens,
            wall_seconds_per_call=seconds,
            note=note,
        )

    # hitl_gate: zero compute; the wait is human time, reported separately.
    return NodeEstimate(
        node_id=node.id,
        kind=node.kind,
        calls=Range.exact(1),
        llm_input_tokens_per_call=0,
        llm_output_tokens_per_call=0,
        wall_seconds_per_call=0.0,
    )


def estimate_pipeline(
    pipeline: Pipeline,
    counter: TokenCounter,
    priors: Priors | None = None,
) -> PipelineEstimate:
    """Predict one run of the graduated pipeline, without executing it.

    Uses the same wave decomposition the adapters emit, so the critical
    path matches the parallelism of the generated code. Loop-body
    sub-nodes are costed inside their parent loop's per-iteration model,
    exactly as they execute.
    """
    priors = priors or Priors()
    waves = _execution_waves(pipeline)

    estimates: dict[str, NodeEstimate] = {}
    for wave in waves:
        for node in wave:
            estimates[node.id] = _estimate_node(node, counter, priors)

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
    )


# ───────── Before: raw-skill estimator ─────────


def external_call_payload_tokens(pipeline: Pipeline, priors: Priors | None = None) -> float:
    """Estimate the data the agent pulls into context, from the pipeline's
    ``external_call`` footprint.

    The graduated pipeline's ``external_call`` nodes name the sources the
    skill fetches (Slack, Gmail, a Drive file …); the *before*-side agent
    pulls the same data, so its transcript carries the same payload. Each
    external_call contributes ``payload_tokens_per_tool[tool]`` when its MCP
    tool is known and pinned, otherwise ``tokens_per_external_call_result``.

    Uses the same wave decomposition as :func:`estimate_pipeline`, so
    loop-body sub-nodes are not double-counted.
    """
    priors = priors or Priors()
    total = 0.0
    for wave in _execution_waves(pipeline):
        for node in wave:
            if node.kind is not NodeKind.EXTERNAL_CALL:
                continue
            tool = node.mcp.tool if node.mcp is not None else None
            per_tool = priors.payload_tokens_per_tool.get(tool) if tool else None
            total += per_tool if per_tool is not None else priors.tokens_per_external_call_result
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

    Cache-aware transcript model: turn *i* re-reads C₀ + (i−1)·Δ tokens
    of context, of which only ~Δ is first-sight (billed at cache-write
    price); the rest is a cache read. Totals over N turns:

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
        turn_method = "graduator sidecar (eval.yaml)"
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
        return context_tokens + max(0.0, n - 1) * priors.transcript_growth_per_turn

    def _cached(n: float) -> float:
        if n <= 1:
            return 0.0
        return (n - 1) * context_tokens + priors.transcript_growth_per_turn * (
            (n - 2) * (n - 1) / 2
        )

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
