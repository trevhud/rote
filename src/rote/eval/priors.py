"""Calibratable priors for the static estimator.

These are the only invented numbers in the eval harness, so they live
in one place, with provenance, and every one is overridable. They are
*models of agent behavior* (how fast a turn goes, how much a transcript
grows), not prices — prices are always fetched live and never appear
here.

Provenance: calibrated against the BDR graduation runs in
``examples/bdr-outreach/runs/`` (real ``claude -p`` agent runs over a
tool-heavy skill: 30–57 turns, ~13 minutes wall clock, Sonnet-class
models) and Anthropic's published latency characteristics. As the
empirical eval mode (Phase 2) accumulates measured runs, these priors
should be re-fitted from that corpus rather than hand-tuned.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Priors:
    """Behavioral constants the static estimator needs.

    Every field is a modeling assumption, surfaced verbatim in the
    scorecard's assumptions section so a reader can judge (and override)
    them. None of them is a price.
    """

    # ── Raw-skill agent loop (the "before" side) ──
    seconds_per_turn: float = 13.0
    """Wall-clock seconds per agent turn, including tool execution.

    BDR graduation runs: ~13 min for 57 turns ≈ 13.7 s/turn; earlier
    runs were similar. Tool-light skills run faster.
    """

    output_tokens_per_turn: float = 250.0
    """Assistant-generated tokens per turn (text + tool-call arguments)."""

    transcript_growth_per_turn: float = 900.0
    """Tokens added to the conversation per turn (assistant output plus
    tool results). This is the Δ in the quadratic input-token model:
    turn *i* re-reads roughly C₀ + (i−1)·Δ tokens of context.
    """

    system_overhead_tokens: int = 16_000
    """Context the agent carries beyond the skill itself (system prompt,
    tool definitions, environment preamble) — part of C₀.

    Measured 2026-07-03: a bare 1-turn ``claude -p`` reported ~21k
    context tokens (5.4k cache-write + 15.3k cache-read) on a
    plugin-heavy install; leaner installs sit lower. 16k splits the
    difference until the Phase 2 corpus fits it properly.
    """

    tokens_per_external_call_result: float = 6_000.0
    """Tokens a single data-pull injects into the agent's transcript — the
    payload of one tool/MCP result (a page of Slack messages, a Gmail
    thread, a spreadsheet dump). Inferred for the *before* side from the
    graduated pipeline's ``external_call`` footprint: the agent pulls the
    same sources the pipeline binds to, and that payload then re-reads on
    every subsequent turn (the dominant cache-read cost on data-heavy
    skills).

    First-shot constant, deliberately moderate. ``transcript_growth_per_turn``
    alone (calibrated on the text-light BDR runs) undercounts a skill that
    fetches large documents by 5–15×; this term restores the missing
    payload. Re-fit per source from the Phase-2 corpus — the measured
    effective transcript growth from a real ``eval --run`` is the anchor.
    """

    payload_tokens_per_tool: dict[str, float] = field(default_factory=dict)
    """Per-MCP-tool overrides for ``tokens_per_external_call_result``, keyed
    by ``Node.mcp.tool`` (e.g. ``{"slack_get_messages": 12000,
    "gmail_get_thread": 8000}``). Empty by default — every external_call
    uses the single constant until a source is measured and pinned here.
    """

    turns_per_step_low: float = 1.0
    turns_per_step_high: float = 2.5
    """Structural fallback when no graduator-emitted eval sidecar exists:
    each identifiable step in the skill costs this many agent turns.
    The sidecar's per-step estimates, produced by an agent that actually
    read the skill, always take precedence.
    """

    # ── Graduated pipeline (the "after" side) ──
    llm_ttft_seconds: float = 1.2
    """Time to first token for a single bounded LLM call."""

    llm_output_tokens_per_second: float = 60.0
    """Streaming throughput for judge output generation."""

    tokens_per_input_field: float = 60.0
    """Expected payload tokens per input-schema property of a judge call
    (we can count the prompt template exactly, but the runtime payload
    filling its holes is only known at run time).
    """

    tokens_per_output_field: float = 40.0
    """Expected generated tokens per output-schema property of a judge —
    schema-constrained output is short; this is deliberately not
    ``output_tokens_per_turn``.
    """

    external_call_seconds: float = 1.5
    """One retryable vendor API round trip."""

    pure_function_seconds: float = 0.01
    """Extracted deterministic code; effectively free."""

    agent_loop_turns_per_iteration: float = 3.0
    """A bounded in-pipeline agent loop still runs an agent; each
    iteration is a few turns of the same dynamics as the raw skill.
    """

    agent_loop_default_max_iterations: int = 10
    """Iteration bound assumed for an ``agent_loop`` node that declares
    no ``termination`` config. Recorded as a per-node assumption in the
    output; the real fix is bounding the node in the IR.
    """

    chars_per_token: float = 3.8
    """Fallback tokenizer approximation used only when no API key is
    available for the exact ``count_tokens`` endpoint; the scorecard
    labels which method produced its numbers.
    """
