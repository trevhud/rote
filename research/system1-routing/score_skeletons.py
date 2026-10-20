"""Score System-1 skeletons against the curated IRs: the $0 hill-climbing loop.

No inference: ``rote.compiler.system1.prepare`` is pure code, and the score is
``rote.bench.score.score_pipeline``, the same function ``rote bench`` uses on
real compiles. Prints one row per skill and the mean.

usage: python score_skeletons.py [<examples dir>] [--skills a,b,c] [--json]
"""

from __future__ import annotations

import json
import statistics
import sys
import tempfile
from pathlib import Path

from rote.bench.score import score_pipeline
from rote.compiler import system1
from rote.ir import load_pipeline

REPO = Path(__file__).resolve().parents[2]
DEFAULT = ["bdr-outreach", "deal-monitor", "expense-audit", "ops-report", "invoice-push"]


def row(examples: Path, skill: str) -> dict[str, object]:
    with tempfile.TemporaryDirectory() as tmp:
        system1.prepare(examples / skill / "skill", Path(tmp))
        compiled = load_pipeline(Path(tmp) / "pipeline.yaml")
    s = score_pipeline(load_pipeline(examples / skill / "expected" / "pipeline.yaml"), compiled)
    st, g = s["structure"], s["graph"]
    return {
        "skill": skill,
        "nodes": f"{st['compiled_nodes']}/{st['expected_nodes']}",
        "nodeR": st["node_recall"],
        "nodeP": st["node_precision"],
        "kind": st["kind_accuracy"],
        "kmix": st.get("kind_mix_overlap"),
        "edgeF1": g["edge_f1"],
        "waves": g["wave_order_agreement"],
        "fanP": s["fan_out"]["precision"],
        "fanR": s["fan_out"]["recall"],
        "mandP": s["mandatory"]["precision"],
        "mandR": s["mandatory"]["recall"],
        "gates": f"{s['gates']['found']}/{s['gates']['expected']}",
        "sig": s["gates"]["signal_matches"],
        "gateJ": s["gates"]["payload_jaccard_mean"],
        "inR": s["inputs"]["recall"],
        "inP": s["inputs"]["precision"],
    }


def main() -> None:
    args = sys.argv[1:]
    examples = Path(args[0]) if args and not args[0].startswith("--") else REPO / "examples"
    skills = DEFAULT
    if "--skills" in args:
        skills = args[args.index("--skills") + 1].split(",")
    rows = [row(examples, s) for s in skills]
    keys = [k for k in rows[0] if k not in ("skill", "nodes", "gates")]
    mean: dict[str, object] = {"skill": "MEAN"}
    for k in keys:
        vals = [r[k] for r in rows if isinstance(r[k], (int, float))]
        mean[k] = round(statistics.mean(vals), 3) if vals else None
    if "--json" in args:
        print(json.dumps({"rows": rows, "mean": mean}))
        return
    cols = list(rows[0])
    print("  ".join(f"{c:>8}" for c in cols))
    for r in [*rows, mean]:
        print("  ".join(f"{str(r.get(c, '')):>8}"[:14] for c in cols))


if __name__ == "__main__":
    main()
