"""The structure pass on the outside-system eval set (research/system1-io/eval.yaml).

Every case is a one-step skill after a neutral first step, as in
research/system1-io/score.py. Two answers are scored per case: the model's
own (``intent``: what it says the step reads or writes) and the compiled
plan's (``final``: after the evidence checks, where a write to a system the
skill never identifies becomes output).

usage: python eval_io.py haiku|sonnet [--low] [--json out.json]
(Anthropic key in the environment)
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from rote.compiler.system1 import structure
from rote.compiler.system1.parse import parse_skill
from rote.compiler.system1.tiebreak import Lane
from rote.eval.empirical import judge_usage_cost_usd
from rote.eval.pricing import fetch_catalog

REPO = Path(__file__).resolve().parents[2]
RUBRIC = (REPO / "skills" / "rote-compile" / "references" / "node-kinds.md").read_text()
CONTEXT = (
    "## Step 1: Receive the batch\n\n"
    "The pipeline's input is this run's batch: its records, lines and totals. "
    "Each step below works on it and on the results of the steps before.\n\n"
)


class _Schema:
    """A declared tool with no known parameters (the eval set names tools only)."""

    input_schema: dict[str, Any] = {}


def one(case: dict[str, Any], lane: Lane) -> dict[str, Any]:
    tools = {tuple(t): _Schema() for t in case.get("tools") or []}
    with tempfile.TemporaryDirectory() as t:
        skill_dir = Path(t) / "skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: t\ndescription: part of a skill\n---\n\n# T\n\n"
            + CONTEXT
            + f"## Step 2: Do it\n\n{case['text']}\n"
        )
        skill = replace(parse_skill(skill_dir), units=structure.section_units(skill_dir))
        unit = skill.units[-1].index
        row: dict[str, Any] = {"text": case["text"], "label": case["label"]}
        try:
            plan, rec = structure.decide(
                skill, structure.skill_text(skill_dir), RUBRIC, lane, tools=tools
            )
        except structure.StructureUnavailable as e:
            return {**row, "error": str(e)[:200], "usage": []}
    raw = [n for n in rec.first_answer.get("nodes") or [] if n.get("section") == unit]
    ios = {(n.get("system") or {}).get("io") for n in raw if n.get("kind") == "external_call"}
    row["intent"] = "write" if "write" in ios else "read" if "read" in ios else "none"
    roles = {n.role for n in plan.nodes if n.unit.index == unit and n.kind == "external_call"}
    row["final"] = "write" if "write" in roles else "read" if "fetch" in roles else "none"
    row["rounds"] = rec.rounds
    row["usage"] = rec.usage
    return row


def score(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    got = [r for r in rows if key in r]
    right = sum(1 for r in got if r[key] == r["label"])
    return {
        "n": len(got),
        "accuracy": round(right / len(got), 3) if got else None,
        "spurious_calls": sum(1 for r in got if r["label"] == "none" and r[key] != "none"),
        "missed_calls": sum(1 for r in got if r["label"] != "none" and r[key] == "none"),
        "wrong_direction": sum(
            1 for r in got if r["label"] != "none" and r[key] not in ("none", r["label"])
        ),
    }


def main() -> None:
    args = sys.argv[1:]
    model = args[0]
    os.environ["ROTE_EFFORT_SYSTEM1_STRUCTURE"] = "low" if "--low" in args else ""
    lane = structure.lane_with(Lane("api"), model)
    cases = yaml.safe_load((REPO / "research" / "system1-io" / "eval.yaml").read_text())["cases"]
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(lambda c: one(c, lane), cases))
    usage = [u for r in rows for u in r.pop("usage")]
    cost, _ = judge_usage_cost_usd(usage, fetch_catalog())
    print(json.dumps({k: score(rows, k) for k in ("intent", "final")}, indent=1))
    print(f"errors {sum(1 for r in rows if 'error' in r)}; cost ${cost:.4f} for {len(rows)} cases")
    for k in ("intent", "final"):
        wrong = [(r["text"][:70], r["label"], r[k]) for r in rows if k in r and r[k] != r["label"]]
        print(f"\n{k} wrong ({len(wrong)}):")
        for w in wrong:
            print("  ", w)
    if "--json" in args:
        Path(args[args.index("--json") + 1]).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
