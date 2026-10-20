"""Score node proposals against a curated examples/*/expected/pipeline.yaml.

Expected nodes are finer-grained than SKILL.md steps (one step often becomes
fetch + parse + filter), so scoring is per source section:

* section coverage: expected sections where the prediction has >= 1 node
* kind match: per section, the overlap of the predicted and expected kind
  multisets (sum over kinds of min(pred, exp)); precision = matched / predicted
  nodes, recall = matched / expected nodes
* exact-count sections: sections whose predicted multiset equals the expected one
* hitl gates, MANDATORY sections, MCP servers, and constant values recalled

BDR's expected pipeline carries no `source.section`, so its node-to-section
map below was written by hand from the SKILL.md (labels, not predictions).
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml

BDR_SECTIONS = {
    "target_research": "Phase 1.5: Target Company Research",
    "taxonomy_lookup": "Phase 2: Lead Generation (Iterative)",
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


def norm(section: str | None) -> str:
    s = (section or "").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    # "Step 7a: Read" (curated label) and "7a: Read" (the heading) are one section.
    return re.sub(r"^(step|phase) ", "", s)


@dataclass
class Pred:
    section: str
    kind: str
    mandatory: bool = False
    mcp_server: str | None = None
    confidence: float | None = None


def expected_nodes(pipeline: Path) -> list[Pred]:
    d = yaml.safe_load(pipeline.read_text())
    out = []
    for n in d["nodes"]:
        sec = (n.get("source") or {}).get("section") or BDR_SECTIONS.get(n["id"])
        if sec is None:
            raise SystemExit(f"{pipeline}: node {n['id']} has no section label")
        server = (n.get("mcp") or {}).get("server")
        out.append(Pred(sec, n["kind"], bool(n.get("mandatory")), server))
    return out


def expected_constants(pipeline: Path) -> list[object]:
    d = yaml.safe_load(pipeline.read_text())
    vals: list[object] = []

    def collect(v: object) -> None:
        if isinstance(v, bool):
            return
        if isinstance(v, (int, float)):
            vals.append(float(v))
        elif isinstance(v, str) and re.fullmatch(r"[A-Za-z0-9_-]{8,}", v):
            vals.append(v)
        elif isinstance(v, dict):
            for x in v.values():
                collect(x)

    for n in d["nodes"]:
        for v in (n.get("constants") or {}).values():
            collect(v)
    return vals


def score(
    preds: list[Pred], exp: list[Pred], found_constants: set[object] | None = None, exp_consts=None
) -> dict:
    by_exp: dict[str, Counter] = {}
    for e in exp:
        by_exp.setdefault(norm(e.section), Counter())[e.kind] += 1
    by_pred: dict[str, Counter] = {}
    for p in preds:
        by_pred.setdefault(norm(p.section), Counter())[p.kind] += 1
    matched = 0
    exact = 0
    covered = 0
    per_section = {}
    for sec, ek in by_exp.items():
        pk = by_pred.get(sec, Counter())
        m = sum(min(pk[k], ek[k]) for k in ek)
        matched += m
        covered += 1 if pk else 0
        exact += 1 if pk == ek else 0
        per_section[sec] = {"expected": dict(ek), "predicted": dict(pk), "matched": m}
    spurious_sections = [s for s in by_pred if s not in by_exp]
    exp_gates = sum(1 for e in exp if e.kind == "hitl_gate")
    pred_gates = sum(1 for p in preds if p.kind == "hitl_gate")
    gate_hits = sum(
        min(by_pred.get(s, Counter())["hitl_gate"], c["hitl_gate"]) for s, c in by_exp.items()
    )
    exp_servers = {e.mcp_server for e in exp if e.mcp_server}
    pred_servers = {p.mcp_server for p in preds if p.mcp_server}
    out = {
        "expected_nodes": len(exp),
        "predicted_nodes": len(preds),
        "sections_expected": len(by_exp),
        "sections_covered": covered,
        "sections_exact": exact,
        "kind_matched": matched,
        "kind_precision": round(matched / len(preds), 2) if preds else 0.0,
        "kind_recall": round(matched / len(exp), 2) if exp else 0.0,
        "hitl": f"{gate_hits}/{exp_gates} found, {pred_gates} predicted",
        "mcp_servers": f"{len(exp_servers & pred_servers)}/{len(exp_servers)}"
        if exp_servers
        else "n/a",
        "spurious_sections": spurious_sections,
        "per_section": per_section,
    }
    if exp_consts is not None and found_constants is not None:
        hit = [c for c in exp_consts if c in found_constants]
        out["constants_recalled"] = f"{len(hit)}/{len(exp_consts)}"
    return out


def preds_from_extraction(ex) -> tuple[list[Pred], set[object]]:
    preds = []
    for u in ex.units:
        server = u.mcp[0]["server"] if u.mcp else None
        for p in u.proposals:
            preds.append(
                Pred(
                    u.section,
                    p.kind,
                    u.mandatory,
                    server if p.kind == "external_call" else None,
                    p.confidence,
                )
            )
    consts: set[object] = set()
    for c in ex.constants:
        consts.add(c.value)
    return preds, consts


def preds_from_pipeline(pipeline: Path) -> list[Pred]:
    d = yaml.safe_load(pipeline.read_text())
    out = []
    for n in d["nodes"]:
        sec = (n.get("source") or {}).get("section") or ""
        out.append(
            Pred(sec, n["kind"], bool(n.get("mandatory")), (n.get("mcp") or {}).get("server"))
        )
    return out


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    from extract import extract

    repo = Path(__file__).resolve().parents[3]
    names = sys.argv[1:] or [
        "bdr-outreach",
        "deal-monitor",
        "expense-audit",
        "ops-report",
        "invoice-push",
    ]
    for name in names:
        exp_path = repo / "examples" / name / "expected" / "pipeline.yaml"
        ex = extract(repo / "examples" / name / "skill")
        preds, consts = preds_from_extraction(ex)
        r = score(preds, expected_nodes(exp_path), consts, expected_constants(exp_path))
        r.pop("per_section")
        print(json.dumps({"skill": name, **r}))
