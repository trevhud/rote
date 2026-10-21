"""With no sample input: does the skeleton's input contract name the skill's real inputs?

Truth is each example's expected pipeline input (its top-level properties).
Rules are System-1's wording rules (skeleton.input_fields); the structure
pass is the model's ``input_fields``.

usage: python input_contract.py haiku|sonnet   (Anthropic key in the environment)
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

from rote.compiler import system1
from rote.compiler.system1 import structure
from rote.compiler.system1.skeleton import input_fields
from rote.compiler.system1.tiebreak import Lane
from rote.eval.empirical import judge_usage_cost_usd
from rote.eval.pricing import fetch_catalog

REPO = Path(__file__).resolve().parents[2]
RUBRIC = REPO / "skills" / "rote-compile" / "references" / "node-kinds.md"
SKILLS = [
    REPO / "examples" / s
    for s in ("bdr-outreach", "deal-monitor", "expense-audit", "ops-report", "invoice-push")
] + [
    REPO / "examples" / "holdout" / s
    for s in ("customer-onboarding", "incident-triage", "saas-spend-audit")
]


def truth(example: Path) -> list[str]:
    doc = yaml.safe_load((example / "expected" / "pipeline.yaml").read_text())
    schema = (doc.get("input") or {}).get("input_schema") or {}
    return sorted(schema.get("properties") or (doc.get("input") or {}).get("required") or [])


def main() -> None:
    lane = structure.lane_with(Lane("api"), sys.argv[1])
    usage: list[dict[str, Any]] = []
    totals = {"rules": [0, 0, 0], "model": [0, 0, 0]}  # hit, expected, predicted
    for ex in SKILLS:
        want = truth(ex)
        with tempfile.TemporaryDirectory() as t:
            rules = sorted(input_fields(system1.prepare(ex / "skill", Path(t))))
        with tempfile.TemporaryDirectory() as t:
            p = system1.prepare(ex / "skill", Path(t), structure_lane=lane, rubric=RUBRIC)
        model = sorted(p.input_fields or {})
        usage += p.structure.usage
        for name, got in (("rules", rules), ("model", model)):
            tot = totals[name]
            tot[0] += len(set(got) & set(want))
            tot[1] += len(want)
            tot[2] += len(got)
        print(json.dumps({"skill": ex.name, "expected": want, "rules": rules, "model": model}))
    for name, (hit, exp, pred) in totals.items():
        print(f"{name}: recall {hit}/{exp} = {hit / exp:.2f}, precision {hit}/{pred}")
    cost, _ = judge_usage_cost_usd(usage, fetch_catalog())
    print(f"cost ${cost:.4f}")


if __name__ == "__main__":
    main()
