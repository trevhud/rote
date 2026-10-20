"""Score every System-1 predictor with the compile benchmark's own scorer.

usage: python bench_eval.py <runs-dir> [<agent-compile-root>]

Predictors, per example skill:
  extractor        deterministic extractor -> template IR (zero inference)
  hinted/<model>   small model over the extractor's units (each trial)
  raw/<model>      small model over SKILL.md alone (each trial)
  agent            today's full Sonnet compile (`<root>/<skill>/compiled/pipeline.yaml`)

Every predictor goes through `rote.bench.score.score_pipeline`, so the
numbers are directly comparable with `rote bench` results. Also reports
confidence calibration: kind accuracy of matched nodes at confidence >= 0.8
versus below, which is what a router would escalate on.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from rote.bench.score import score_pipeline
from rote.ir import load_pipeline

sys.path.insert(0, str(Path(__file__).parent))
from assemble import assemble, from_classifier, from_extraction  # noqa: E402
from extract import extract  # noqa: E402
from score import Pred, expected_nodes, preds_from_pipeline  # noqa: E402
from score import score as section_score  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
SKILLS = ["bdr-outreach", "deal-monitor", "expense-audit", "ops-report", "invoice-push"]
KEYS = [
    ("structure", "node_recall"),
    ("structure", "node_precision"),
    ("structure", "kind_accuracy"),
    ("structure", "kind_accuracy_matched"),
    ("graph", "edge_f1"),
    ("mandatory", "recall"),
    ("mandatory", "precision"),
    ("section", "kind_recall"),
    ("section", "kind_precision"),
]


def with_sections(score: dict, preds: list[Pred], skill: str) -> dict:
    """Add the naming-independent per-section kind match (score.py) to a bench score."""
    sec = section_score(
        preds, expected_nodes(REPO / "examples" / skill / "expected" / "pipeline.yaml")
    )
    return {
        **score,
        "section": {"kind_recall": sec["kind_recall"], "kind_precision": sec["kind_precision"]},
    }


def as_preds(props) -> list[Pred]:
    return [Pred(p.section, p.kind) for p in props]


def flat(score: dict) -> dict:
    out = {f"{a}.{b}": score[a].get(b) for a, b in KEYS}
    out["gates_found"] = f"{score['gates']['found']}/{score['gates']['expected']}"
    out["nodes"] = f"{score['structure']['compiled_nodes']}/{score['structure']['expected_nodes']}"
    return out


def calibration(score: dict, props) -> dict[str, list[bool]]:
    conf = {}
    used: dict[str, int] = {}
    # assemble() renames on collision; mirror its id order to map confidences
    from assemble import _ident

    seen: set[str] = set()
    for p in props:
        conf[_ident(p.id, seen)] = p.confidence
    buckets: dict[str, list[bool]] = defaultdict(list)
    for m in score["structure"]["matches"]:
        c = conf.get(m["compiled"])
        if c is None:
            continue
        buckets["high" if c >= 0.8 else "low"].append(m["expected_kind"] == m["compiled_kind"])
    del used
    return buckets


def main() -> None:
    runs = Path(sys.argv[1])
    agent_root = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    rows = []
    calib: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for skill in SKILLS:
        expected = load_pipeline(REPO / "examples" / skill / "expected" / "pipeline.yaml")
        ex = extract(REPO / "examples" / skill / "skill")
        props = from_extraction(ex)
        s = with_sections(
            score_pipeline(expected, assemble(ex.name, ex.description, props)),
            as_preds(props),
            skill,
        )
        rows.append({"skill": skill, "predictor": "extractor", "cost_usd": 0.0, **flat(s)})
        for k, v in calibration(s, props).items():
            calib["extractor"][k] += v
        for f in sorted(runs.glob(f"*-{skill}.*.json")):
            result = json.loads(f.read_text())
            props = from_classifier(ex, result)
            s = with_sections(
                score_pipeline(expected, assemble(ex.name, ex.description, props)),
                as_preds(props),
                skill,
            )
            pred = f"{result['mode']}/{result['model']}"
            rows.append(
                {
                    "skill": skill,
                    "predictor": pred,
                    "trial": f.name,
                    "cost_usd": result["cost_usd"],
                    "wall_s": result["wall_s"],
                    **flat(s),
                }
            )
            for k, v in calibration(s, props).items():
                calib[pred][k] += v
        if agent_root is not None:
            compiled = agent_root / skill / "compiled" / "pipeline.yaml"
            if compiled.exists():
                agent_preds = preds_from_pipeline(compiled)
                try:
                    agent = load_pipeline(compiled)
                except (
                    ValueError
                ):  # an older artifact the current IR now rejects: section score only
                    sec = section_score(
                        agent_preds,
                        expected_nodes(REPO / "examples" / skill / "expected" / "pipeline.yaml"),
                    )
                    rows.append(
                        {
                            "skill": skill,
                            "predictor": "agent/sonnet-4-6",
                            "invalid_under_current_ir": True,
                            "section.kind_recall": sec["kind_recall"],
                            "section.kind_precision": sec["kind_precision"],
                        }
                    )
                    continue
                s = with_sections(score_pipeline(expected, agent), agent_preds, skill)
                rows.append({"skill": skill, "predictor": "agent/sonnet-4-6", **flat(s)})
    for r in rows:
        print(json.dumps(r))
    print("--- mean over skills (trials averaged first) ---")
    by_pred: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for r in rows:
        for k in [f"{a}.{b}" for a, b in KEYS] + ["cost_usd"]:
            if r.get(k) is not None:
                by_pred[r["predictor"]][r["skill"]][k].append(r[k])
    for pred, skills in by_pred.items():
        summary = {}
        for k in [f"{a}.{b}" for a, b in KEYS] + ["cost_usd"]:
            per_skill = [statistics.mean(v[k]) for v in skills.values() if v.get(k)]
            if per_skill:
                summary[k] = round(statistics.mean(per_skill), 3)
        holdout = skills.get("invoice-push", {})
        summary["holdout_kind_accuracy"] = (
            round(statistics.mean(holdout["structure.kind_accuracy"]), 3)
            if holdout.get("structure.kind_accuracy")
            else None
        )
        print(json.dumps({"predictor": pred, "skills": len(skills), **summary}))
    print("--- confidence calibration (matched nodes, kind correct) ---")
    for pred, b in calib.items():
        print(json.dumps({p: f"{sum(v)}/{len(v)}" for p, v in b.items()} | {"predictor": pred}))


if __name__ == "__main__":
    main()
