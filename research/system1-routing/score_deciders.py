"""Score each System-1 decider's skeletons against the curated IRs.

The structure is decided before the agent runs, so a skeleton IS the compiled
structure: this scores it with ``rote.bench.score.score_pipeline``, the
function ``rote bench`` uses, with no agent run. A model decider makes real
Workers AI calls (Clef about $0.005 a skill, Jev about $0.001). Every response
is cached on disk by a hash of the request, so re-scoring and threshold
sweeps never spend twice.

Credentials come from ``--cf-env`` (a dotenv file of the PERSONAL Cloudflare
account) and are set for this process only. Never the ambient login.

Modes:

- ``score``: one row per (split, decider) with the frozen thresholds.
- ``fit``: fit one model's two thresholds on the dev split only. The rule is
  pre-registered here, before any held-out run, and is the one Clef's were
  chosen by (unit-level, against the expected IR's nodes for each step):

  * kind threshold: the lowest grid value at which the model's kind choice
    is right on at least 95% of the dev steps at or above it, with at least
    5 such steps. "Right" means the choice is one of the step's expected kinds.
  * human threshold: the grid value with the best gate F1 on dev steps (a
    step is a gate when an expected hitl_gate maps to it). A tie keeps the
    higher value, so a tie keeps the rules in charge.

  It also prints the end-to-end dev skeleton scores at the fitted pair and at
  "never override" (both thresholds above 1), as a check that the fit does
  not make dev worse than the rules.

usage:
  python score_deciders.py score --cf-env F --cache D [--splits dev,holdout]
      [--deciders rules,clef,jev] [--rows]
  python score_deciders.py fit --cf-env F --cache D --model jev
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import statistics
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from rote.bench.score import score_pipeline
from rote.compiler.system1 import decision_model as dm
from rote.compiler.system1.decide import plan
from rote.compiler.system1.parse import parse_skill
from rote.compiler.system1.skeleton import write
from rote.ir import Pipeline, load_pipeline

REPO = Path(__file__).resolve().parents[2]
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
KIND_GRID = (0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99)
HUMAN_GRID = (0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9)
#: bdr-outreach's expected IR predates ``source``: its nodes map to the
#: skill's sections by hand, by the phase each one implements.
BDR_SECTIONS = {
    "target_research": "Phase 1.5: Target Company Research",
    "taxonomy_lookup": "Phase 1.5: Target Company Research",
    "lead_generation_loop": "Phase 2: Lead Generation (Iterative)",
    "enrich_contact_batch": "Phase 2: Lead Generation (Iterative)",
    "vet_contact": "Phase 2: Lead Generation (Iterative)",
    "contact_review_gate": "Phase 3: Contact Review Gate",
    "hubspot_upsert": "Phase 4: CRM Upload + List Creation",
    "hubspot_create_list": "Phase 4: CRM Upload + List Creation",
    "exclusion_check_dnc": "Phase 5: Exclusion Checks (MANDATORY)",
    "exclusion_check_recent": "Phase 5: Exclusion Checks (MANDATORY)",
    "exclusion_check_sequence": "Phase 5: Exclusion Checks (MANDATORY)",
    "personalize_email": "Phase 6: Email Templates",
    "create_sales_template": "Phase 6: Email Templates",
    "pre_enrollment_report": "Phase 7: Handoff",
    "manual_enrollment_handoff": "Phase 7: Handoff",
}
BDR_SUBUNIT = {
    "exclusion_check_dnc": "Do-not-contact list",
    "exclusion_check_recent": "Recently emailed",
    "exclusion_check_sequence": "Active sequence",
}


def _norm(s: str) -> str:
    s = re.sub(r"^(step|phase)\s+", "", s.strip().lower())
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


def unit_labels(examples: Path, skill: str) -> dict[int, set[str]]:
    """The expected node kinds of each step unit, by unit index (mapped units only)."""
    units = parse_skill(examples / skill / "skill").units
    labels: dict[int, set[str]] = {}
    for n in load_pipeline(examples / skill / "expected" / "pipeline.yaml").nodes:
        if skill == "bdr-outreach":
            section, sub = BDR_SECTIONS[n.id], BDR_SUBUNIT.get(n.id)
            hits = [u for u in units if u.section == section and (sub is None or u.title == sub)]
        else:
            if n.source is None or not n.source.section:
                continue
            ns = _norm(n.source.section)
            hits = [
                u
                for u in units
                if _norm(u.section) == ns
                or _norm(u.section).endswith(ns)
                or ns.endswith(_norm(u.section))
            ]
        for u in hits:
            labels.setdefault(u.index, set()).add(n.kind.value)
    return labels


def load_env(path: Path) -> None:
    for line in path.read_text().splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            if k in ("CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN"):
                os.environ[k] = v.strip().strip('"')


@contextmanager
def cached_posts(cache: Path) -> Iterator[dict[str, float]]:
    """Serve repeated requests from disk; count what the live ones cost."""
    cache.mkdir(parents=True, exist_ok=True)
    live = dm._post
    spent = {"calls": 0.0}

    def post(url: str, token: str, body: dict[str, Any]) -> dict[str, Any]:
        key = hashlib.sha256(json.dumps([url, body], sort_keys=True).encode()).hexdigest()[:32]
        path = cache / f"{key}.json"
        if path.is_file():
            loaded: dict[str, Any] = json.loads(path.read_text())
            return loaded
        resp = live(url, token, body)
        path.write_text(json.dumps(resp))
        spent["calls"] += 1
        return resp

    dm._post = post
    try:
        yield spent
    finally:
        dm._post = live


def skeleton(skill_dir: Path, verdicts: dm.Verdicts | None) -> Pipeline:
    p = plan(parse_skill(skill_dir), verdicts)
    with tempfile.TemporaryDirectory() as tmp:
        write(p, Path(tmp))
        return load_pipeline(Path(tmp) / "pipeline.yaml")


def row(examples: Path, skill: str, verdicts: dm.Verdicts | None) -> dict[str, Any]:
    expected = load_pipeline(examples / skill / "expected" / "pipeline.yaml")
    s = score_pipeline(expected, skeleton(examples / skill / "skill", verdicts))
    g = s["gates"]
    return {
        "skill": skill,
        "kind": s["structure"]["kind_accuracy"],
        "kmix": s["structure"].get("kind_mix_overlap"),
        "nodeR": s["structure"]["node_recall"],
        "gates_found": g["found"],
        "gates_expected": g["expected"],
        "false_gates": g["compiled"] - g["found"],
        "mandP": s["mandatory"]["precision"],
        "mandR": s["mandatory"]["recall"],
        "cost_usd": 0.0 if verdicts is None else verdicts.cost_usd,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(key: str) -> float | None:
        vals = [r[key] for r in rows if isinstance(r[key], (int, float))]
        return round(statistics.mean(vals), 3) if vals else None

    return {
        "kind": mean("kind"),
        "kmix": mean("kmix"),
        "nodeR": mean("nodeR"),
        "gates": f"{sum(r['gates_found'] for r in rows)}/{sum(r['gates_expected'] for r in rows)}",
        "false_gates": sum(r["false_gates"] for r in rows),
        "mandP": mean("mandP"),
        "mandR": mean("mandR"),
        "cost_usd_per_skill": round(sum(r["cost_usd"] for r in rows) / len(rows), 5),
    }


def verdicts_for(model: dm.DecisionModel, examples: Path, skills: list[str]) -> dict[str, Any]:
    return {s: dm.decide(model, parse_skill(examples / s / "skill")) for s in skills}


def cmd_score(args: argparse.Namespace) -> None:
    for split in args.splits.split(","):
        examples, skills = SPLITS[split]
        for decider in args.deciders.split(","):
            model = dm.MODELS.get(decider)
            verdicts = verdicts_for(model, examples, skills) if model else {}
            rows = [row(examples, s, verdicts.get(s)) for s in skills]
            out = {"split": split, "decider": decider, **summarize(rows)}
            if args.rows:
                out["rows"] = rows
            print(json.dumps(out))


def cmd_fit(args: argparse.Namespace) -> None:
    examples, skills = SPLITS["dev"]
    base = dm.MODELS[args.model]
    raw = verdicts_for(base, examples, skills)
    steps = [
        (raw[s].by_unit[i], kinds)
        for s in skills
        for i, kinds in unit_labels(examples, s).items()
        if i in raw[s].by_unit
    ]
    print(json.dumps({"dev_steps_labeled": len(steps)}))
    kind_t = None
    for t in KIND_GRID:
        sure = [v.kind in kinds for v, kinds in steps if v.kind_confidence >= t]
        acc = sum(sure) / len(sure) if sure else None
        print(json.dumps({"kind_threshold": t, "steps": len(sure), "accuracy": acc}))
        if kind_t is None and acc is not None and len(sure) >= 5 and acc >= 0.95:
            kind_t = t
    best: tuple[float, float] | None = None
    for t in HUMAN_GRID:
        tp = sum(1 for v, k in steps if v.human >= t and "hitl_gate" in k)
        fp = sum(1 for v, k in steps if v.human >= t and "hitl_gate" not in k)
        fn = sum(1 for v, k in steps if v.human < t and "hitl_gate" in k)
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        print(json.dumps({"human_threshold": t, "tp": tp, "fp": fp, "fn": fn, "f1": round(f1, 3)}))
        if best is None or f1 >= best[0]:
            best = (f1, t)
    assert best is not None
    human_t = best[1]
    fitted = {"kind_threshold": kind_t, "human_threshold": human_t}
    print(json.dumps({"fitted": fitted}))
    for label, (k, h) in (("fitted", (kind_t or 1.01, human_t)), ("never", (1.01, 1.01))):
        model = dataclasses.replace(base, kind_threshold=k, human_threshold=h)
        rows = [row(examples, s, dm.Verdicts(model, raw[s].by_unit)) for s in skills]
        print(json.dumps({"dev_end_to_end": label, **summarize(rows)}))


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("score", "fit"):
        sp = sub.add_parser(name)
        sp.add_argument("--cf-env", type=Path, required=True)
        sp.add_argument("--cache", type=Path, required=True)
    sub.choices["score"].add_argument("--splits", default="dev,holdout")
    sub.choices["score"].add_argument("--deciders", default="rules,clef,jev")
    sub.choices["score"].add_argument("--rows", action="store_true")
    sub.choices["fit"].add_argument("--model", required=True, choices=sorted(dm.MODELS))
    args = p.parse_args()
    load_env(args.cf_env)
    with cached_posts(args.cache) as spent:
        (cmd_score if args.cmd == "score" else cmd_fit)(args)
    print(json.dumps({"live_calls": spent["calls"]}))


if __name__ == "__main__":
    main()
