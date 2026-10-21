"""Skeleton scores: the rules against the whole-skill structure pass.

usage: python skeletons.py [haiku|sonnet|openai:<model> ...] [--trials N] [--json out.json]
(Anthropic key in the environment; for openai:<model>, the lane an openai-api
compile gets: OPENAI_BASE_URL and CF_AIG_TOKEN, a Cloudflare AI endpoint)
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

import anthropic

from rote.bench.score import score_pipeline
from rote.compiler import system1
from rote.compiler.system1 import structure
from rote.compiler.system1.tiebreak import Lane
from rote.eval.empirical import judge_usage_cost_usd
from rote.eval.pricing import fetch_catalog
from rote.ir import load_pipeline

REPO = Path(__file__).resolve().parents[2]
RUBRIC = REPO / "skills" / "rote-compile" / "references" / "node-kinds.md"
SPLITS = {
    "dev": (
        REPO / "examples",
        ["bdr-outreach", "deal-monitor", "expense-audit", "ops-report", "invoice-push"],
    ),
    "holdout": (
        REPO / "examples" / "holdout",
        ["customer-onboarding", "incident-triage", "saas-spend-audit"],
    ),
}


#: ``--`` choices: a model, optionally at low effort ("haiku-low").
CHOICES = [*structure.MODELS, *(f"{m}-low" for m in structure.MODELS)]


def openai_lane(model: str) -> Lane:
    """What an openai-api compile on rote-cloud gets: the driver's own client."""
    from rote.compiler.drivers.openai_api import OpenAIApiDriver
    from rote.compiler.system1 import tiebreak

    token = os.environ["CF_AIG_TOKEN"]
    os.environ.setdefault("OPENAI_API_KEY", token)
    driver = OpenAIApiDriver(
        model=model,
        base_url=os.environ["OPENAI_BASE_URL"],
        default_headers={
            "Authorization": f"Bearer {token}",
            "cf-aig-authorization": f"Bearer {token}",
        },
    )
    return tiebreak.lane_for(driver)


def row(examples: Path, skill: str, *, choice: str) -> dict[str, Any]:
    model = choice.removesuffix("-low")
    if choice == "rules":
        lane = None
    elif choice.startswith("openai:"):
        lane = openai_lane(choice.removeprefix("openai:"))
    else:
        lane = structure.lane_with(Lane("api", anthropic.Anthropic()), model)
    start = time.monotonic()
    with tempfile.TemporaryDirectory() as t:
        try:
            plan = system1.prepare(
                examples / skill / "skill", Path(t), structure_lane=lane, rubric=RUBRIC
            )
        except structure.StructureUnavailable as e:
            return {"skill": skill, "error": str(e)[:300], "usage": []}
        compiled = load_pipeline(Path(t) / "pipeline.yaml")
    elapsed = time.monotonic() - start
    s = score_pipeline(load_pipeline(examples / skill / "expected" / "pipeline.yaml"), compiled)
    st = plan.structure
    calls = [n for n in compiled.nodes if n.kind.value == "external_call"]
    return {
        "skill": skill,
        "kind": s["structure"]["kind_accuracy"],
        "kmix": s["structure"].get("kind_mix_overlap"),
        "edgeF1": s["graph"]["edge_f1"],
        "nodes": len(compiled.nodes),
        "calls": len(calls),
        "unbound_calls": sum(1 for n in calls if n.mcp is None),
        "rounds": st.rounds if st else 0,
        "first_round_contradictions": len(st.contradictions[0]) if st else 0,
        "unresolved": len(st.resolved) if st else 0,
        "seconds": round(elapsed, 1),
        "usage": st.usage if st else [],
    }


def main() -> None:
    args = sys.argv[1:]
    trials = int(args[args.index("--trials") + 1]) if "--trials" in args else 1
    choices = ["rules", *[a for a in args if a in CHOICES or a.startswith("openai:")]]
    catalog = fetch_catalog()
    out: list[dict[str, Any]] = []
    for split, (examples, skills) in SPLITS.items():
        for choice in choices:
            # The structure call's effort knob (rote.inference node_effort).
            os.environ["ROTE_EFFORT_SYSTEM1_STRUCTURE"] = "low" if choice.endswith("-low") else ""
            n = 1 if choice == "rules" else trials
            jobs = [(s, k) for k in range(n) for s in skills]
            with ThreadPoolExecutor(max_workers=8) as pool:
                rows = list(pool.map(partial(row, examples, choice=choice), [j[0] for j in jobs]))
            usage = [u for r in rows for u in r.pop("usage")]
            errors = [r for r in rows if "error" in r]
            rows = [r for r in rows if "error" not in r]
            for e in errors:
                print("    ERROR", json.dumps(e), flush=True)
            cost, _ = judge_usage_cost_usd(usage, catalog)
            summary = {
                "split": split,
                "by": choice,
                "errors": len(errors),
                **{
                    k: round(statistics.mean(r[k] for r in rows), 3)
                    for k in ("kind", "kmix", "edgeF1", "seconds")
                },
                **{
                    k: round(sum(r[k] for r in rows) / n, 1)
                    for k in ("nodes", "calls", "unbound_calls", "unresolved")
                },
                "rounds_mean": round(statistics.mean(r["rounds"] for r in rows), 2),
                "cost_per_skill": round(cost / len(rows), 4) if rows else 0,
                "tokens_per_skill": {
                    k: round(sum(u.get(k) or 0 for u in usage) / max(len(rows), 1))
                    for k in ("input_tokens", "output_tokens", "cache_read_input_tokens")
                },
            }
            print(json.dumps(summary), flush=True)
            for r in rows:
                print("   ", json.dumps(r), flush=True)
            out.append({"summary": summary, "rows": rows})
    if "--json" in args:
        Path(args[args.index("--json") + 1]).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
