"""System-1 compile (EXPERIMENTAL, ``rote compile --system1``).

Every decision a compile makes (which nodes a step becomes, their kinds,
data flow, fan_out, mandatory flags, gates and signals, constants, rule
tables as code) is made here, by rules over the skill's text. The LLM
agent is then given a much smaller rubric and writes only text: module
implementations and tests, judge signatures and prompts, the report.

The research behind it, with measured numbers, is
``docs/design/system1-compile.md``. Nothing on the default
compile path calls it.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from rote.compiler.system1 import clef, report
from rote.compiler.system1.clef import ClefUnavailable
from rote.compiler.system1.decide import REVIEW_BELOW, Plan, plan
from rote.compiler.system1.enforce import all_changes, check_proposals, enforce
from rote.compiler.system1.parse import UnsafeSkillText, parse_skill
from rote.compiler.system1.skeleton import BRIEF_DIRNAME, DECISIONS_FILENAME, write
from rote.ir import load_pipeline

__all__ = [
    "DECIDERS",
    "REVIEW_BELOW",
    "ClefUnavailable",
    "Plan",
    "all_changes",
    "TEXT_RUBRIC_FILE",
    "TEXT_RUBRIC_REFERENCES",
    "UnsafeSkillText",
    "check_proposals",
    "enforce",
    "materialize_text_rubric",
    "prepare",
    "settle",
]

#: The text agent's rubric, inside the rote-compile bundle.
TEXT_RUBRIC_FILE = "SYSTEM1.md"
#: The only references the text agent needs. Copied from the one
#: rote-compile bundle at run time, so the two rubrics cannot drift.
TEXT_RUBRIC_REFERENCES = ("implementation.md", "llm-judge-extraction.md")


#: Who decides step kinds and gates. Mandatory and fan_out are always rules.
DECIDERS = ("rules", "clef")


def prepare(skill_dir: Path, work_dir: Path, decider: str = "rules") -> Plan:
    """Decide everything and write the skeleton into ``work_dir``.

    ``rules`` makes no call at all. ``clef`` asks Cloudflare's Clef decision
    model (no generated text) about each step's kind and gate, and uses its
    answers where it is confident (:mod:`rote.compiler.system1.clef`)."""
    if decider not in DECIDERS:
        raise ValueError(f"unknown --system1-decider {decider!r}; choose from {DECIDERS}")
    skill = parse_skill(skill_dir)
    verdicts = clef.decide(skill) if decider == "clef" else None
    p = plan(skill, verdicts)
    write(p, work_dir)
    if verdicts is not None:
        _record_decider(work_dir, verdicts)
    return p


def _record_decider(work_dir: Path, verdicts: dict[int, clef.Verdict]) -> None:
    path = work_dir / BRIEF_DIRNAME / DECISIONS_FILENAME
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["decider"] = {
        "model": clef.MODEL_ROUTE,
        "cost_usd": round(sum(v.cost_usd for v in verdicts.values()), 6),
        "units": {
            str(i): {"kind": v.kind, "kind_confidence": v.kind_confidence, "human": v.human}
            for i, v in verdicts.items()
        },
    }
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
    """After an agent stop: restore the decisions, then write the report from data.

    Runs before every verification (repair rounds included) and once more
    after the agent's last stop, so what is verified is what ships."""
    changes = enforce(work_dir, final=final)
    try:
        pipeline = load_pipeline(work_dir / "pipeline.yaml")
    except Exception:  # noqa: BLE001 (verification reports it to the agent)
        return changes
    report.write(work_dir, p, pipeline, all_changes(work_dir))
    return changes
