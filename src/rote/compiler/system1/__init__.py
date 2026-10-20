"""System-1 compile, the default ``rote compile`` path (``--no-system1`` opts out).

Every decision a compile makes (which nodes a step becomes, their kinds,
data flow, fan_out, mandatory flags, gates and signals, constants, rule
tables as code) is made here, by rules over the skill's text. The LLM
agent is then given a much smaller rubric and writes only text: module
implementations and tests, judge signatures and prompts, the report.

The research behind it, with measured numbers, is
``docs/design/system1-compile.md``. ``--no-system1`` and ``--update``
run the full-rubric agent instead, which decides the structure itself.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rote.compiler.report import ReportExtras
from rote.compiler.system1 import decision_model, tiebreak
from rote.compiler.system1.decide import REVIEW_BELOW, Plan, plan, unit_segments, unsure_units
from rote.compiler.system1.decision_model import DeciderUnavailable, Verdicts
from rote.compiler.system1.enforce import all_changes, check_proposals, enforce
from rote.compiler.system1.parse import UnsafeSkillText, parse_skill
from rote.compiler.system1.skeleton import BRIEF_DIRNAME, DECISIONS_FILENAME, write
from rote.compiler.system1.tiebreak import TIEBREAKS, Lane, TieBreakUnavailable

__all__ = [
    "DECIDERS",
    "REVIEW_BELOW",
    "DeciderUnavailable",
    "Lane",
    "TIEBREAKS",
    "TieBreakUnavailable",
    "Plan",
    "all_changes",
    "TEXT_RUBRIC_FILE",
    "TEXT_RUBRIC_REFERENCES",
    "UnsafeSkillText",
    "check_proposals",
    "enforce",
    "materialize_text_rubric",
    "prepare",
    "report_extras",
    "settle",
]

#: The text agent's rubric, inside the rote-compile bundle.
TEXT_RUBRIC_FILE = "SYSTEM1.md"
#: The only references the text agent needs. Copied from the one
#: rote-compile bundle at run time, so the two rubrics cannot drift.
TEXT_RUBRIC_REFERENCES = ("implementation.md", "llm-judge-extraction.md")


#: Who decides step kinds and gates. Mandatory and fan_out are always rules.
#: ``rules`` makes no call. The others are decision models on Workers AI.
DECIDERS = ("rules", *decision_model.MODELS)


def prepare(
    skill_dir: Path,
    work_dir: Path,
    decider: str = "rules",
    sample: Any = None,
    tools: Mapping[tuple[str, str], Any] | None = None,
    tiebreak_lane: Lane | None = None,
    rubric: Path | None = None,
) -> Plan:
    """Decide everything and write the skeleton into ``work_dir``.

    ``sample`` (a :class:`rote.compiler.grounding.Sample`) decides the input
    schema and gate payload keys, ahead of the skill's wording. ``tools``
    (grounding's tool schemas, keyed ``(server, tool)``) names MCP arguments.

    ``rules`` makes no call at all. ``clef`` and ``jev`` ask that decision
    model (no generated text) about each step's kind and gate, and use its
    answers where it is confident (:mod:`rote.compiler.system1.decision_model`).

    With ``tiebreak_lane``, the steps still unsure after that go to Haiku in
    one call, with ``rubric`` (node-kinds.md) as its guide, and its node
    lists replace their conservative defaults
    (:mod:`rote.compiler.system1.tiebreak`). Without it nothing is asked."""
    if decider not in DECIDERS:
        raise ValueError(f"unknown --decider {decider!r}; choose from {DECIDERS}")
    skill = parse_skill(skill_dir)
    model = decision_model.MODELS.get(decider)
    verdicts = decision_model.decide(model, skill) if model is not None else None
    segments = unit_segments(skill, verdicts)
    tb = None
    if tiebreak_lane is not None:
        if rubric is None:
            raise ValueError("a tie-break needs the node-kinds rubric")
        unsure = unsure_units(skill, segments)
        tb = tiebreak.decide(
            skill, segments, unsure, verdicts, tiebreak_lane, rubric, tools=list(tools or {})
        )
        segments = {**segments, **tb.segments}
    p = plan(skill, verdicts, tools, segments)
    p.sample = sample
    p.tiebreak = tb
    write(p, work_dir)
    if verdicts is not None or tb is not None:
        _record_calls(work_dir, verdicts, tb)
    return p


def _record_calls(work_dir: Path, verdicts: Verdicts | None, tb: tiebreak.TieBreak | None) -> None:
    """Record every model call the decisions came from in decisions.json."""
    path = work_dir / BRIEF_DIRNAME / DECISIONS_FILENAME
    doc = json.loads(path.read_text(encoding="utf-8"))
    if verdicts is not None:
        doc["decider"] = {
            "name": verdicts.model.name,
            "model": verdicts.model.route,
            "cost_usd": round(verdicts.cost_usd, 6),
            "units": {
                str(i): {"kind": v.kind, "kind_confidence": v.kind_confidence, "human": v.human}
                for i, v in verdicts.by_unit.items()
            },
        }
    if tb is not None:
        doc["tiebreak"] = tb.as_dict()
    path.write_text(json.dumps(doc, indent=1), encoding="utf-8")


def materialize_text_rubric(compiler_skill_dir: Path, dest: Path) -> Path:
    """A skill bundle for the text agent: SYSTEM1.md as its SKILL.md, two references."""
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(compiler_skill_dir / TEXT_RUBRIC_FILE, dest / "SKILL.md")
    (dest / "references").mkdir(exist_ok=True)
    for name in TEXT_RUBRIC_REFERENCES:
        shutil.copyfile(compiler_skill_dir / "references" / name, dest / "references" / name)
    return dest


def settle(work_dir: Path, p: Plan, *, final: bool = False) -> list[str]:
    """After an agent stop: restore the decisions.

    Runs before every verification (repair rounds included) and once more
    after the agent's last stop, so what is verified is what ships."""
    return enforce(work_dir, final=final)


def report_extras(work_dir: Path, p: Plan) -> ReportExtras:
    """What System-1 adds to the shared compile report (rote.compiler.report).

    Read from ``work_dir`` (the proposals and the enforcement record), so it
    is built before the work dir moves to the output."""
    questions: list[tuple[str, str]] = []
    for line in p.review:
        node = line.split("`", 2)[1] if line.startswith("`") else "pipeline"
        questions.append((f"System-1 decision below {REVIEW_BELOW} confidence: {node}", line))
    questions += [("Proposed by the agent, not applied", x) for x in check_proposals(work_dir)]
    questions += [
        ("The agent edited a decision; rote restored it", c) for c in all_changes(work_dir)
    ]
    questions += [
        (
            f"{tiebreak.MODEL} proposed a kind a rule forbids",
            f"step `{p.skill.units[int(i)].title}`: {why}",
        )
        for i, reasons in _overruled(work_dir).items()
        for why in reasons
    ]
    # Details carry agent-written text (a proposal's node, a restored value):
    # one line each, so a newline and "## ..." cannot reshape the report.
    questions = [(" ".join(t.split()), " ".join(d.split())) for t, d in questions]
    model, asked, costs = _calls_made(work_dir)
    deciders = ["rules"]
    if model:
        deciders.append(f"the {model.label} decision model")
    if asked:
        deciders.append(f"{tiebreak.MODEL} for the {asked} step(s) they were unsure of")
    who = deciders[0] if len(deciders) == 1 else ", ".join(deciders[:-1]) + " and " + deciders[-1]
    sources = [x for x in (model.label if model else None, tiebreak.MODEL if asked else None) if x]
    return ReportExtras(
        preamble=(
            f"Compiled with System-1: rote decided the structure with {who}, and the "
            "agent wrote the implementations and judge prompts. Every decision, with its "
            f"confidence and rule, is in `{BRIEF_DIRNAME}/{DECISIONS_FILENAME}`."
            + _cost_sentence(costs)
        ),
        questions=questions,
        next_steps=(
            "Review every Open question above; each assumption is a rule the skill does "
            "not state, and each low-confidence decision took a conservative default.",
            "Run the pipeline once (`rote run`) and compare it with a run of the skill.",
            (
                "Recompile with `rote compile` after the skill changes. Some "
                f"decisions are model answers ({' and '.join(sources)}), and rules "
                "elsewhere: compare decisions.json between runs before trusting a diff."
                if sources
                else "Recompile with `rote compile` after the skill changes: the "
                "rules are deterministic, so the same skill text gives the same decisions."
            ),
        ),
    )


def _calls_made(
    work_dir: Path,
) -> tuple[decision_model.DecisionModel | None, int, dict[str, float | None]]:
    """From decisions.json: the decision model asked (or None), how many steps
    the tie-break was asked about (0 when it did not run or had none), and
    each model call's cost by who made it (``None`` when it could not be priced)."""
    path = work_dir / BRIEF_DIRNAME / DECISIONS_FILENAME
    if not path.is_file():
        return None, 0, {}
    doc = json.loads(path.read_text(encoding="utf-8"))
    decider = doc.get("decider")
    model = None if decider is None else decision_model.MODELS[decider["name"]]
    tb = doc.get("tiebreak") or {}
    asked = len(tb.get("steps_asked") or [])
    costs: dict[str, float | None] = {}
    if model is not None:
        costs[model.label] = decider.get("cost_usd")
    if asked:
        costs[tiebreak.MODEL] = tb.get("cost_usd")
    return model, asked, costs


def _overruled(work_dir: Path) -> dict[str, list[str]]:
    """The tie-break's proposals a rule forbade, by unit index, from decisions.json."""
    path = work_dir / BRIEF_DIRNAME / DECISIONS_FILENAME
    if not path.is_file():
        return {}
    tb = json.loads(path.read_text(encoding="utf-8")).get("tiebreak") or {}
    return dict(tb.get("overruled") or {})


def _cost_sentence(costs: dict[str, float | None]) -> str:
    """`` Its model calls cost $0.0049 (Clef) and $0.0004 (claude-haiku-5-5).``"""
    if not costs:
        return ""
    parts = [
        f"{'an unpriced amount' if c is None else f'${c:.4f}'} ({who})" for who, c in costs.items()
    ]
    return " Its model calls cost " + " and ".join(parts) + ", on top of the agent's run."
