"""What would Haiku add as a ROUTED classifier, used only where the rules are unsure?

EXPERIMENTAL, offline: replays the Haiku 5.5 "hinted" classifications from the
System-1 research (``research/compile-system1/system1/classify.py`` on the
research branch, saved as ``hinted-claude-haiku-5-5-<skill>.<n>.json``) over
this build's rule decisions. No new inference.

For each step unit whose lowest kind confidence is below ``tau``, the unit's
nodes are replaced by Haiku's node list for it; every other unit keeps the
rules. Both versions are rendered through the real skeleton writer and scored
with ``rote.bench.score.score_pipeline``.

usage: python route_haiku.py <haiku runs dir> [tau ...]
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

from rote.bench.score import score_pipeline
from rote.compiler.system1.decide import NodePlan, Plan, plan
from rote.compiler.system1.parse import parse_skill
from rote.compiler.system1.skeleton import pipeline_doc
from rote.ir import Pipeline, load_pipeline

REPO = Path(__file__).resolve().parents[2]
SKILLS = ["bdr-outreach", "deal-monitor", "expense-audit", "ops-report", "invoice-push"]
ROLE = {
    "external_call": "fetch",
    "llm_judge": "judge",
    "pure_function": "process",
    "agent_loop": "loop",
    "hitl_gate": "gate",
}


def routed(p: Plan, haiku: dict, tau: float) -> tuple[Plan, int]:
    by_unit = {row["unit"]: row["nodes"] for row in haiku["output"]["units"]}
    nodes: list[NodePlan] = []
    used: set[str] = set()
    n_routed = 0
    for u in p.skill.units:
        mine = [n for n in p.nodes if n.unit.index == u.index]
        low = min(d.confidence for n in mine for d in n.decisions if d.field == "kind")
        if low >= tau or u.index not in by_unit or not by_unit[u.index]:
            nodes += mine
            used |= {n.id for n in mine}
            continue
        n_routed += 1
        inputs = dict(mine[0].inputs)
        replacement: list[NodePlan] = []
        for h in by_unit[u.index]:
            nid = "".join(c if c.isalnum() else "_" for c in h["id"].lower()) or "node"
            if nid[0].isdigit():
                nid = f"n_{nid}"
            while nid in used:
                nid += "_2"
            used.add(nid)
            node = NodePlan(nid, h["kind"], ROLE[h["kind"]], u)
            node.inputs = (
                inputs
                if not replacement
                else {
                    "request": "pipeline.input",
                    replacement[-1].id: f"{replacement[-1].id}.output",
                }
            )
            if node.kind == "agent_loop":
                node.tools = [f"{nid}_tool"]
            if node.kind == "hitl_gate":
                node.signal = nid
            replacement.append(node)
        # Downstream nodes read the unit's last node: rewire them to the new last.
        old_last, new_last = mine[-1].id, replacement[-1].id
        for later in p.nodes:
            for k, v in list(later.inputs.items()):
                if v.startswith(f"{old_last}.output"):
                    del later.inputs[k]
                    later.inputs[new_last] = v.replace(old_last, new_last, 1)
            if later.element_source == old_last:
                later.element_source = new_last
                replacement[-1].lists_output = True
        nodes += replacement
    return Plan(skill=p.skill, nodes=nodes, review=p.review), n_routed


def main() -> None:
    runs = Path(sys.argv[1])
    taus = [float(t) for t in sys.argv[2:]] or [0.7, 0.9]
    for tau in taus:
        rule_scores, route_scores, routed_units, total_units, costs = [], [], 0, 0, []
        for skill in SKILLS:
            expected = load_pipeline(REPO / "examples" / skill / "expected" / "pipeline.yaml")
            base = plan(parse_skill(REPO / "examples" / skill / "skill"))
            rule = score_pipeline(expected, Pipeline.model_validate(pipeline_doc(base)))
            rule_scores.append(rule["structure"]["kind_accuracy"])
            trials = []
            for f in sorted(runs.glob(f"hinted-claude-haiku-5-5-{skill}.*.json")):
                h = json.loads(f.read_text())
                fresh = plan(parse_skill(REPO / "examples" / skill / "skill"))
                rp, n = routed(fresh, h, tau)
                s = score_pipeline(expected, Pipeline.model_validate(pipeline_doc(rp)))
                trials.append(s["structure"]["kind_accuracy"])
                routed_units += n
                total_units += len(fresh.skill.units)
                costs.append(h["cost_usd"])
            route_scores.append(statistics.mean(trials))
            print(
                json.dumps(
                    {
                        "tau": tau,
                        "skill": skill,
                        "rules": rule["structure"]["kind_accuracy"],
                        "routed": round(statistics.mean(trials), 3),
                    }
                )
            )
        print(
            json.dumps(
                {
                    "tau": tau,
                    "mean_kind_accuracy_rules": round(statistics.mean(rule_scores), 3),
                    "mean_kind_accuracy_routed": round(statistics.mean(route_scores), 3),
                    "units_routed": f"{routed_units}/{total_units}",
                    "haiku_cost_per_compile_usd": round(statistics.mean(costs), 4),
                }
            )
        )


if __name__ == "__main__":
    main()
