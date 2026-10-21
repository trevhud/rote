"""A small LLM decides the steps System-1 is unsure of (``--tiebreak haiku``, opt-in).

After the rules (and a decision model, with ``--decider clef|jev``) have
decided every step, some steps are still unsure: the rules had no clear
signal, or the model disagreed below its threshold. Those steps' kind
decisions sit below ``REVIEW_BELOW``. Their text goes to Haiku in one
schema-locked call, with the node-kind rubric, and Haiku's node list
replaces the conservative default for each. Every other step keeps its
decision. ``--tiebreak none`` (the default, :data:`DEFAULT_TIEBREAK`) skips
the call, and the unsure steps keep the conservative default and are listed
for review.

The call bills the way the compile does. Under the ``claude`` driver it is
``claude -p`` on the user's subscription. Under the ``api`` driver it uses
that driver's own client, so a gateway base URL and headers apply to it
too. Under any other driver it uses an Anthropic key if one is set, then
the ``claude`` CLI. With none of those the compile fails, naming the fix:
asking for a tie-break and silently getting none would report decisions
that never happened. Both transports are :mod:`rote.inference`'s
``structured_call``, the same code emitted judges run: a forced tool call,
no ``temperature``, all four usage buckets.

The measured gain is in ``docs/compile-bench.md``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rote.compiler.system1.decide import REVIEW_BELOW, Segment, infeasible_kinds, role_of_kind
from rote.compiler.system1.decision_model import Verdicts
from rote.compiler.system1.parse import SkillParse, Unit

#: The choices of ``--tiebreak`` and the ``tiebreak`` setting.
TIEBREAKS = ("haiku", "none")
#: Off by default. Measured on the cached API driver, 3 compiles per cell,
#: fidelity: expense-audit 0.909 with it against 0.915 without, ops-report
#: 0.855 against 0.858. That is within run-to-run spread, but not at least
#: the rules' level, which was the bar for default-on. Its kind-accuracy gain
#: is on held-out skeletons, where fidelity is not measured.
DEFAULT_TIEBREAK = "none"
#: The model behind ``haiku``: the current Haiku (models.dev, 2026-10-07).
MODEL = "claude-haiku-5-5"
KINDS = ("pure_function", "external_call", "llm_judge", "agent_loop", "hitl_gate")
#: A decision the tie-break made is settled: it is not listed for review.
CONFIDENCE = REVIEW_BELOW
#: The node id the call reports under (usage rows, error messages).
NODE_ID = "system1_tiebreak"
#: One more try when the answer does not cover exactly the steps asked about.
ATTEMPTS = 2

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "step": {"type": "integer", "description": "the step's number, as given"},
                    "nodes": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 4,
                        "items": {
                            "type": "object",
                            "properties": {
                                "kind": {"type": "string", "enum": list(KINDS)},
                                "reason": {"type": "string", "description": "one short clause"},
                            },
                            "required": ["kind", "reason"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["step", "nodes"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["steps"],
    "additionalProperties": False,
}

_TASK = """\
You decide workflow node kinds for the rote compiler, which turns an AI
skill into a deterministic pipeline. Below are the node-kind rubric, the
skill's step list, and the steps a rule-based classifier was unsure of,
each with its own guess (which may be wrong).

For each unsure step, give the nodes it becomes, in data-flow order. A step
may become several nodes: for example a bare tool call (external_call),
then the pure_function or llm_judge that processes its result, as the
rubric's split rule says. Prefer the more deterministic kind when a step
could go either way. Answer every unsure step, and only those.

=== RUBRIC (node-kinds.md) ===
{rubric}

=== SKILL ===
{skill}
"""


class TieBreakUnavailable(RuntimeError):
    """A tie-break was asked for and cannot run (no lane, or no usable answer)."""


@dataclass(frozen=True)
class Lane:
    """How the call is billed: ``claude-cli`` or ``api``, and the compile driver's client."""

    provider: str
    sdk_client: Any = None
    #: The model id on this lane: ``MODEL``, or its gateway form when the
    #: compile model is provider-prefixed (:func:`gateway_model_id`). On an
    #: OpenAI-compatible lane, the compile's own model.
    model: str = MODEL
    #: The wire format ``structured_call`` speaks: ``anthropic`` or ``openai``.
    client: str = "anthropic"

    @property
    def label(self) -> str:
        if self.provider == "claude-cli":
            return "claude -p (subscription)"
        if self.provider == "codex-cli":
            return "codex exec"
        return "OpenAI-compatible API" if self.client == "openai" else "Anthropic API"


@dataclass
class TieBreak:
    """The tie-break's answers, by unit index, and what the call cost."""

    model: str
    lane: str
    asked: list[int]
    segments: dict[int, list[Segment]] = field(default_factory=dict)
    usage: list[dict[str, Any]] = field(default_factory=list)
    #: Nodes the model proposed that a rule forbids, kept out (by unit index).
    overruled: dict[int, list[str]] = field(default_factory=dict)
    #: ``None`` when a usage row could not be priced (no catalog, an alias).
    cost_usd: float | None = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "lane": self.lane,
            "steps_asked": [str(i) for i in self.asked],
            "overruled": {str(i): v for i, v in self.overruled.items()},
            "cost_usd": None if self.cost_usd is None else round(self.cost_usd, 6),
            "usage": self.usage,
        }


def gateway_model_id(prefix: str, model: str) -> str:
    """``model`` as Cloudflare's ``/ai`` endpoint names it: the provider prefix,
    and the version with a dot. Verified live on a rote gateway:
    ``anthropic/claude-haiku-5.5`` answers, ``anthropic/claude-haiku-5-5`` is
    "Model not found", and a bare id is unknown there."""
    return f"{prefix}/" + re.sub(r"-(\d+)-(\d+)$", r"-\1.\2", model)


def lane_for(
    driver: Any,
    *,
    caller: str = "the System-1 tie-break asks a model about the steps the rules are unsure of",
    opt_out: str = "--tiebreak none",
    error: type[RuntimeError] = TieBreakUnavailable,
) -> Lane:
    """The compile driver's own client, as a lane: same endpoint, headers and auth.

    ``claude`` calls ``claude -p``, ``api`` the AnthropicApiDriver's client
    with a Haiku id in the compile model's own form, ``openai-api`` the
    OpenAIApiDriver's client with the compile's model, and ``codex`` the
    CodexDriver's binary through ``codex exec --output-schema`` with its
    model. Nothing falls back to another provider's key or binary: a compile
    that chose GPT-5.5 must not bill the user's Claude subscription, and
    rote-cloud's container has neither. Any other driver fails here.
    ``caller``, ``opt_out`` and ``error`` say who is asking, so a failure
    names the flag that turns that call off."""
    from rote.inference._runtime_helper import provider_availability

    name = getattr(driver, "name", None)
    if name == "claude":
        ok, reason = provider_availability("claude-cli", "anthropic")
        if not ok:
            raise error(f"{caller}, and cannot use claude -p: {reason} (or pass {opt_out})")
        return Lane("claude-cli")
    if name == "api":
        import anthropic

        from rote.compiler.drivers.anthropic_api import AnthropicApiDriver

        if isinstance(driver, AnthropicApiDriver):
            prefix, _, _ = str(driver.model).rpartition("/")
            return Lane(
                "api",
                anthropic.Anthropic(**driver._client_kwargs()),
                gateway_model_id(prefix, MODEL) if prefix else MODEL,
            )
    if name == "openai-api":
        import openai

        from rote.compiler.drivers.openai_api import OpenAIApiDriver

        if isinstance(driver, OpenAIApiDriver):
            return Lane(
                "api", openai.OpenAI(**driver._client_kwargs()), driver.model, client="openai"
            )
    if name == "codex":
        from rote.compiler.drivers.codex import CodexDriver
        from rote.compiler.system1.lane_call import PROVIDER, CodexCli

        if isinstance(driver, CodexDriver):
            ok, reason = driver.is_available()
            if not ok:
                raise error(f"{caller}, and cannot use codex exec: {reason} (or pass {opt_out})")
            return Lane(
                PROVIDER, CodexCli(driver.codex_executable), driver.model or "", client="openai"
            )
    raise error(
        f"{caller}, and the {name!r} driver has no client rote can send that call "
        f"through: compile with --agent claude, api, openai-api or codex, or pass {opt_out}"
    )


def prompt_for(
    skill: SkillParse,
    unsure: list[Unit],
    segments: Mapping[int, list[Segment]],
    verdicts: Verdicts | None,
    rubric: str,
) -> str:
    steps: list[dict[str, Any]] = []
    for u in unsure:
        row: dict[str, Any] = {
            "step": u.index,
            "section": u.section,
            "title": u.title,
            "marked_mandatory": u.mandatory,
            "mcp_mentions": [list(m) for m in u.mcp],
            "classifier_guess": [kind for _, kind, _, _ in segments[u.index]],
            "text": u.text,
        }
        if verdicts is not None and u.index in verdicts.by_unit:
            v = verdicts.by_unit[u.index]
            row["decision_model_guess"] = {
                "model": verdicts.model.label,
                "kind": v.kind,
                "kind_confidence": v.kind_confidence,
                "person_must_act": v.human,
            }
        steps.append(row)
    body = {
        "skill": skill.name,
        "description": skill.description,
        "all_steps": [x.title for x in skill.units],
        "unsure_steps": steps,
    }
    return _TASK.format(rubric=rubric, skill=json.dumps(body, indent=1))


Call = Callable[..., tuple[dict[str, Any], list[tuple[str, dict[str, Any]]]]]


def _structured_call() -> Call:
    from rote.compiler.system1.lane_call import structured_call

    return structured_call


def _one_line(text: str, limit: int = 160) -> str:
    """Model-written text becomes a decision's rule, which lands in the brief
    and the report: one line, bounded."""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _parse(
    payload: dict[str, Any], asked: list[int], model: str = MODEL
) -> dict[int, list[Segment]] | None:
    """The answer as segments, or None unless it covers exactly the steps asked.

    Rule constraints are applied after (:func:`_constrain`)."""
    by_step: dict[int, list[Segment]] = {}
    for entry in payload.get("steps") or []:
        step, nodes = entry.get("step"), entry.get("nodes") or []
        if not isinstance(step, int) or step in by_step or not 1 <= len(nodes) <= 4:
            return None
        if any(n.get("kind") not in KINDS for n in nodes):
            return None
        by_step[step] = [
            (
                role_of_kind(n["kind"]),
                n["kind"],
                CONFIDENCE,
                f"tie-break by {model}: {_one_line(n.get('reason', ''))}",
            )
            for n in nodes
        ]
    return by_step if sorted(by_step) == sorted(asked) else None


def _constrain(
    skill: SkillParse,
    answer: dict[int, list[Segment]],
    result: TieBreak,
    tools: Collection[tuple[str, str]] = (),
) -> dict[int, list[Segment]]:
    """Drop every proposed node of a kind a rule forbids for its step
    (:func:`rote.compiler.system1.decide.infeasible_kinds`). A step left with
    no node keeps the rules' decision."""
    units = {u.index: u for u in skill.units}
    kept: dict[int, list[Segment]] = {}
    for step, segments in answer.items():
        forbidden = infeasible_kinds(units[step], skill.tool_registry, tools)
        allowed = [seg for seg in segments if seg[1] not in forbidden]
        dropped = [f"{seg[1]}: {forbidden[seg[1]]}" for seg in segments if seg[1] in forbidden]
        if dropped:
            result.overruled[step] = dropped
        if allowed:
            kept[step] = allowed
    return kept


def decide(
    skill: SkillParse,
    segments: Mapping[int, list[Segment]],
    unsure: list[Unit],
    verdicts: Verdicts | None,
    lane: Lane,
    rubric_path: Path,
    call: Call | None = None,
    tools: Collection[tuple[str, str]] = (),
) -> TieBreak:
    """Ask the model about ``unsure`` and return its node lists. One call."""
    asked = [u.index for u in unsure]
    result = TieBreak(lane.model, lane.label, asked)
    if not unsure:
        return result
    if lane.provider != "claude-cli" and lane.sdk_client is None:
        # structured_call would build a client from whatever Anthropic key the
        # environment holds, whichever driver the compile chose.
        raise TieBreakUnavailable(
            "the tie-break has no client for its lane; build lanes with lane_for"
        )
    prompt = prompt_for(skill, unsure, segments, verdicts, rubric_path.read_text("utf-8"))
    call = call or _structured_call()
    for _ in range(ATTEMPTS):
        try:
            payload, rows = call(
                node_id=NODE_ID,
                provider=lane.provider,
                client=lane.client,
                model=lane.model,
                prompt=prompt,
                output_schema=SCHEMA,
                sdk_client=lane.sdk_client,
            )
        except RuntimeError as e:
            raise TieBreakUnavailable(
                f"the System-1 tie-break ({lane.model}, {lane.label}) failed: {e}. "
                "Pass --tiebreak none to compile without it"
            ) from e
        result.usage += [{"model": m, **u} for m, u in rows]
        parsed = _parse(payload, asked, lane.model)
        if parsed is not None:
            result.segments = _constrain(skill, parsed, result, tools)
            result.cost_usd = _cost(result.usage)
            return result
    raise TieBreakUnavailable(
        f"the System-1 tie-break ({lane.model}) did not answer exactly the {len(asked)} steps it "
        f"was asked about in {ATTEMPTS} tries. Pass --tiebreak none to compile without it"
    )


def _cost(rows: list[dict[str, Any]]) -> float | None:
    """Dollars for the call: the CLI's own estimate, else catalog rates.

    An unpriceable row (no catalog offline, a model it does not list) makes
    the whole cost unknown rather than understated."""
    from rote.eval.empirical import judge_usage_cost_usd
    from rote.eval.pricing import PricingError, fetch_catalog

    catalog = None
    if any(row.get("cost_usd") is None for row in rows):
        try:
            catalog = fetch_catalog()
        except PricingError:
            catalog = None
    total, unpriced = judge_usage_cost_usd(rows, catalog)
    return None if unpriced else total
