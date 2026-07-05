"""The before/after scorecard: speed, cost, determinism.

Pure presentation + price arithmetic. Token counts and seconds come
from :mod:`rote.eval.estimate`; dollars come from applying a live
:class:`~rote.eval.pricing.ModelPrice` to them; nothing here invents a
number. Every modeling assumption that produced the estimates is
restated in the rendered output's assumptions section — a scorecard a
reader can't audit is marketing, not measurement.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

from rote.eval.estimate import AgentRunEstimate, PipelineEstimate, Range
from rote.eval.pricing import ModelPrice
from rote.eval.priors import Priors

# ───────── Price arithmetic ─────────


def agent_run_cost_usd(estimate: AgentRunEstimate, price: ModelPrice) -> Range:
    """Dollar cost of running the skill as agent instructions.

    Cache-aware: fresh tokens bill at the cache-write price when the
    model has one (real agent harnesses cache their context), else at
    the plain input price; re-read context bills at the cache-read
    price, else at the input price.
    """
    fresh_rate = (
        price.cache_write_per_mtok
        if price.cache_write_per_mtok is not None
        else price.input_per_mtok
    )
    cached_rate = (
        price.cache_read_per_mtok if price.cache_read_per_mtok is not None else price.input_per_mtok
    )
    per_mtok = (
        estimate.fresh_input_tokens.scale(fresh_rate)
        + estimate.cached_read_tokens.scale(cached_rate)
        + estimate.output_tokens.scale(price.output_per_mtok)
    )
    return per_mtok.scale(1 / 1_000_000)


def pipeline_cost_usd(estimate: PipelineEstimate, price: ModelPrice) -> Range:
    """Dollar cost of one pipeline run: only the LLM nodes bill.

    Judge calls are small independent prompts, so no caching benefit is
    assumed — inputs bill at the plain input price.
    """
    per_mtok = estimate.llm_input_tokens.scale(
        price.input_per_mtok
    ) + estimate.llm_output_tokens.scale(price.output_per_mtok)
    return per_mtok.scale(1 / 1_000_000)


@dataclass(frozen=True)
class CostRow:
    price: ModelPrice
    before_usd: Range | None
    after_usd: Range


# ───────── The scorecard ─────────


@dataclass(frozen=True)
class Scorecard:
    pipeline_name: str
    skill: AgentRunEstimate | None
    pipeline: PipelineEstimate
    costs: list[CostRow]
    priors: Priors
    generated_at: str

    def to_dict(self) -> dict[str, object]:
        """JSON-ready form, for programmatic consumers."""

        def _range(r: Range) -> dict[str, float]:
            return {"low": r.low, "high": r.high}

        skill_part: dict[str, object] | None = None
        if self.skill is not None:
            skill_part = {
                "turns": _range(self.skill.turns),
                "context_tokens": self.skill.context_tokens,
                "fresh_input_tokens": _range(self.skill.fresh_input_tokens),
                "cached_read_tokens": _range(self.skill.cached_read_tokens),
                "output_tokens": _range(self.skill.output_tokens),
                "wall_seconds": _range(self.skill.wall_seconds),
                "sampled_steps": self.skill.sampling.sampled_steps,
                "total_steps": self.skill.sampling.total_steps,
                "sampled_output_tokens": _range(self.skill.sampling.sampled_output_tokens),
                "turn_method": self.skill.turn_method,
                "token_count_method": self.skill.token_count_method,
            }
        return {
            "pipeline": self.pipeline_name,
            "generated_at": self.generated_at,
            # Roteness: share of pipeline steps that run as deterministic
            # routine (code) vs. LLM inference. Purely structural — a function
            # of node kinds, not any model's estimate.
            "roteness": self.pipeline.sampling.roteness,
            "before": skill_part,
            "after": {
                "critical_path_seconds": _range(self.pipeline.critical_path_seconds),
                "hitl_gates": list(self.pipeline.hitl_gates),
                "llm_input_tokens": _range(self.pipeline.llm_input_tokens),
                "llm_output_tokens": _range(self.pipeline.llm_output_tokens),
                "sampled_steps": self.pipeline.sampling.sampled_steps,
                "total_steps": self.pipeline.sampling.total_steps,
                "schema_constrained_steps": self.pipeline.sampling.schema_constrained_steps,
                "sampled_output_tokens": _range(self.pipeline.sampling.sampled_output_tokens),
                "nodes": [
                    {
                        "id": n.node_id,
                        "kind": n.kind.value,
                        "calls": _range(n.calls),
                        "llm_input_tokens_per_call": n.llm_input_tokens_per_call,
                        "llm_output_tokens_per_call": n.llm_output_tokens_per_call,
                        "wall_seconds_per_call": n.wall_seconds_per_call,
                        **({"note": n.note} if n.note else {}),
                    }
                    for n in self.pipeline.nodes
                ],
            },
            "costs": [
                {
                    "model": row.price.model_id,
                    "provider": row.price.provider,
                    "tier": row.price.tier.value,
                    "price_source": row.price.source,
                    "fetched_at": row.price.fetched_at,
                    "input_per_mtok": row.price.input_per_mtok,
                    "output_per_mtok": row.price.output_per_mtok,
                    "before_usd": _range(row.before_usd) if row.before_usd else None,
                    "after_usd": _range(row.after_usd),
                }
                for row in self.costs
            ],
            "priors": asdict(self.priors),
        }

    def to_markdown(self) -> str:
        lines: list[str] = [
            f"# Eval scorecard — {self.pipeline_name}",
            "",
            f"Generated: {self.generated_at} · static estimate (nothing was executed)",
            "",
        ]

        # ── Headline ──
        lines += ["## Before → after", ""]
        lines.append("| Metric | As agent instructions | As graduated pipeline | Change |")
        lines.append("|---|---|---|---|")
        if self.skill is not None:
            lines.append(
                "| Wall clock (compute) | "
                f"{_seconds(self.skill.wall_seconds)} | "
                f"{_seconds(self.pipeline.critical_path_seconds)} | "
                f"{_improvement(self.skill.wall_seconds, self.pipeline.critical_path_seconds)} |"
            )
            before_llm_tokens = (
                self.skill.fresh_input_tokens
                + self.skill.cached_read_tokens
                + self.skill.output_tokens
            )
            after_llm_tokens = self.pipeline.llm_input_tokens + self.pipeline.llm_output_tokens
            lines.append(
                "| LLM tokens touched | "
                f"{_tokens(before_llm_tokens)} | "
                f"{_tokens(after_llm_tokens)} | "
                f"{_improvement(before_llm_tokens, after_llm_tokens)} |"
            )
            sampled_change = _improvement(
                self.skill.sampling.sampled_output_tokens,
                self.pipeline.sampling.sampled_output_tokens,
            )
            lines.append(
                "| Sampled output tokens | "
                f"{_tokens(self.skill.sampling.sampled_output_tokens)} | "
                f"{_tokens(self.pipeline.sampling.sampled_output_tokens)} | "
                f"{sampled_change} |"
            )
            lines.append(
                "| Steps decided by a sampled LLM | "
                f"{self.skill.sampling.sampled_steps} of {self.skill.sampling.total_steps} | "
                f"{self.pipeline.sampling.sampled_steps} of {self.pipeline.sampling.total_steps} "
                f"({self.pipeline.sampling.schema_constrained_steps} schema-constrained) | — |"
            )
            lines.append(
                "| Roteness (deterministic steps) | "
                f"{1 - self.skill.sampling.sampled_fraction:.0%} | "
                f"{self.pipeline.sampling.roteness:.0%} | — |"
            )
        else:
            lines.append(
                "| Wall clock (compute) | _no skill baseline_ | "
                f"{_seconds(self.pipeline.critical_path_seconds)} | — |"
            )
        if self.pipeline.hitl_gates:
            gates = ", ".join(f"`{g}`" for g in self.pipeline.hitl_gates)
            lines.append(
                f"| Human approval gates | implicit (chat back-and-forth) | "
                f"{len(self.pipeline.hitl_gates)} durable ({gates}) | — |"
            )
        lines.append("")

        # ── Cost ──
        lines += ["## Cost per run (live official prices)", ""]
        lines.append("| Model | Tier | Before | After | Change |")
        lines.append("|---|---|---|---|---|")
        for row in self.costs:
            before = _usd(row.before_usd) if row.before_usd is not None else "—"
            change = (
                _improvement(row.before_usd, row.after_usd) if row.before_usd is not None else "—"
            )
            lines.append(
                f"| {row.price.model_id} | {row.price.tier.value} | "
                f"{before} | {_usd(row.after_usd)} | {change} |"
            )
        lines.append("")
        sources = sorted({row.price.source for row in self.costs})
        fetched = sorted({row.price.fetched_at for row in self.costs})
        lines.append(
            f"Prices fetched {fetched[0]} from: " + ", ".join(sources) + ". Never hardcoded."
        )
        lines.append("")

        # ── Node detail ──
        lines += ["## Pipeline node detail", ""]
        lines.append("| Node | Kind | Calls | LLM in/call | LLM out/call | Wall s/call |")
        lines.append("|---|---|---|---|---|---|")
        for n in self.pipeline.nodes:
            lines.append(
                f"| `{n.node_id}` | {n.kind.value} | {_count(n.calls)} | "
                f"{n.llm_input_tokens_per_call or '—'} | "
                f"{n.llm_output_tokens_per_call or '—'} | "
                f"{n.wall_seconds_per_call:g} |"
            )
        notes = [n for n in self.pipeline.nodes if n.note]
        if notes:
            lines.append("")
            for n in notes:
                lines.append(f"- `{n.node_id}`: {n.note}")
        lines.append("")

        # ── Assumptions ──
        lines += ["## Assumptions (audit me)", ""]
        if self.skill is not None:
            lines.append(f"- Turn estimate: {self.skill.turn_method}.")
            lines.append(f"- Token counting: {self.skill.token_count_method}.")
            lines.append(
                "- Agent cost model: cache-aware transcript growth — turn *i* "
                "re-reads C₀ + (i−1)·Δ context; fresh tokens bill at cache-write "
                "price, re-reads at cache-read price."
            )
        lines.append("- HITL gate waits are human time and excluded from wall-clock on both sides.")
        lines.append(
            "- Priors (calibrated on the BDR runs in `examples/bdr-outreach/runs/`; "
            "every value overridable):"
        )
        for f in fields(self.priors):
            lines.append(f"  - `{f.name}` = {getattr(self.priors, f.name)}")
        lines.append("")
        return "\n".join(lines)


def build_scorecard(
    *,
    pipeline_name: str,
    pipeline_estimate: PipelineEstimate,
    skill_estimate: AgentRunEstimate | None,
    prices: list[ModelPrice],
    priors: Priors,
    generated_at: str,
) -> Scorecard:
    """Assemble a scorecard: apply each live price to both estimates."""
    costs = [
        CostRow(
            price=p,
            before_usd=(
                agent_run_cost_usd(skill_estimate, p) if skill_estimate is not None else None
            ),
            after_usd=pipeline_cost_usd(pipeline_estimate, p),
        )
        for p in prices
    ]
    return Scorecard(
        pipeline_name=pipeline_name,
        skill=skill_estimate,
        pipeline=pipeline_estimate,
        costs=costs,
        priors=priors,
        generated_at=generated_at,
    )


# ───────── Formatting helpers ─────────


def _fmt_num(v: float) -> str:
    if v >= 1_000_000:
        return f"{v / 1_000_000:.1f}M"
    if v >= 10_000:
        return f"{v / 1000:.0f}k"
    if v >= 1_000:
        return f"{v / 1000:.1f}k"
    return f"{v:.0f}"


def _range_str(r: Range) -> str:
    if r.low == r.high:
        return _fmt_num(r.low)
    return f"{_fmt_num(r.low)}–{_fmt_num(r.high)}"


def _tokens(r: Range) -> str:
    return _range_str(r)


def _count(r: Range) -> str:
    return _range_str(r)


def _seconds(r: Range) -> str:
    def one(v: float) -> str:
        if v >= 90:
            return f"{v / 60:.1f} min"
        return f"{v:.0f} s"

    if r.low == r.high:
        return one(r.low)
    return f"{one(r.low)} – {one(r.high)}"


def _usd(r: Range) -> str:
    def one(v: float) -> str:
        return f"${v:,.2f}" if v >= 0.10 else f"${v:.3f}"

    if r.low == r.high:
        return one(r.low)
    return f"{one(r.low)} – {one(r.high)}"


def _improvement(before: Range, after: Range) -> str:
    """Mid-to-mid change, phrased as a reduction/increase multiple."""
    if before.mid <= 0:
        return "—"
    ratio = after.mid / before.mid
    if ratio <= 0:
        return "eliminated"
    if ratio < 1:
        pct = (1 - ratio) * 100
        return f"−{pct:.0f}% ({1 / ratio:.1f}× less)"
    return f"+{(ratio - 1) * 100:.0f}%"
