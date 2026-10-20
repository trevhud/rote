"""Non-generative step classifier: nearest neighbors over labeled step embeddings.

EXPERIMENTAL. Leave-one-skill-out over the five examples: each held-out
skill's steps are labeled by their most similar steps in the other four
skills (cosine similarity of local embeddings, no generative model). A
step's label is the node-kind multiset its section becomes in the curated
IR, split evenly across the section's steps.

Predictors compared, all through the same section metric (score.py) and the
bench scorer (via assemble.py):
  rules     the deterministic extractor alone
  knn       1-nearest-neighbor label
  hybrid    rules, except the KIND of each proposal comes from the neighbor
            vote when the neighbor is similar enough (>= --tau)

usage: python embed_knn.py <fastembed|model2vec> [--tau 0.8]
Needs fastembed or model2vec installed (not a rote dependency).
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from assemble import Proposed, assemble, from_extraction  # noqa: E402
from extract import extract  # noqa: E402
from score import BDR_SECTIONS, Pred, norm  # noqa: E402
from score import score as section_score  # noqa: E402

from rote.bench.score import score_pipeline  # noqa: E402
from rote.ir import load_pipeline  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
SKILLS = ["bdr-outreach", "deal-monitor", "expense-audit", "ops-report", "invoice-push"]


def embedder(name: str):
    if name == "fastembed":
        from fastembed import TextEmbedding

        model = TextEmbedding("BAAI/bge-small-en-v1.5")
        return lambda texts: np.array(list(model.embed(texts)))
    from model2vec import StaticModel

    model = StaticModel.from_pretrained("minishlab/potion-base-8M")
    return lambda texts: np.asarray(model.encode(texts))


def section_kinds(skill: str) -> dict[str, Counter]:
    d = yaml.safe_load((REPO / "examples" / skill / "expected" / "pipeline.yaml").read_text())
    out: dict[str, Counter] = {}
    for n in d["nodes"]:
        sec = (n.get("source") or {}).get("section") or BDR_SECTIONS[n["id"]]
        out.setdefault(norm(sec), Counter())[n["kind"]] += 1
    return out


def unit_labels(skill: str, ex) -> list[Counter]:
    """Split each section's kind multiset evenly across its units (remainder to the first)."""
    kinds = section_kinds(skill)
    by_sec: dict[str, list[int]] = {}
    for i, u in enumerate(ex.units):
        by_sec.setdefault(norm(u.section), []).append(i)
    labels = [Counter() for _ in ex.units]
    for sec, idxs in by_sec.items():
        total = kinds.get(sec, Counter())
        for kind, count in total.items():
            share, rem = divmod(count, len(idxs))
            for j, i in enumerate(idxs):
                labels[i][kind] += share + (1 if j < rem else 0)
    return labels


def text_of(u) -> str:
    return f"{u.title}\n{u.text[:1200]}"


def main() -> None:
    which = sys.argv[1]
    tau = float(sys.argv[sys.argv.index("--tau") + 1]) if "--tau" in sys.argv else 0.8
    embed = embedder(which)
    data = {}
    for s in SKILLS:
        ex = extract(REPO / "examples" / s / "skill")
        vecs = embed([text_of(u) for u in ex.units])
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        data[s] = (ex, vecs, unit_labels(s, ex))
    rows = []
    for held in SKILLS:
        ex, vecs, _ = data[held]
        pool_v = np.vstack([data[s][1] for s in SKILLS if s != held])
        pool_l = [lab for s in SKILLS if s != held for lab in data[s][2]]
        sims = vecs @ pool_v.T
        rule_props = from_extraction(ex)
        knn_props: list[Proposed] = []
        hyb_props: list[Proposed] = []
        by_unit = {}
        k = 0
        for i, u in enumerate(ex.units):
            n = len(u.proposals)
            by_unit[i] = rule_props[k : k + n]
            k += n
        for i, u in enumerate(ex.units):
            j = int(np.argmax(sims[i]))
            sim = float(sims[i, j])
            label = pool_l[j]
            for kind, c in label.items():
                for m in range(c):
                    knn_props.append(
                        Proposed(
                            f"{u.title}_{kind}_{m}",
                            kind,
                            u.section,
                            u.title,
                            u.mandatory,
                            u.tools,
                            None,
                            sim,
                        )
                    )
            for p in by_unit[i]:
                kind = p.kind
                if sim >= tau and label and p.kind not in label:
                    kind = label.most_common(1)[0][0]
                hyb_props.append(
                    Proposed(p.id, kind, p.section, p.description, p.mandatory, p.tools, p.mcp, sim)
                )
        expected = load_pipeline(REPO / "examples" / held / "expected" / "pipeline.yaml")
        from score import expected_nodes

        exp_nodes = expected_nodes(REPO / "examples" / held / "expected" / "pipeline.yaml")
        for name, props in (("rules", rule_props), ("knn", knn_props), ("hybrid", hyb_props)):
            sec = section_score([Pred(p.section, p.kind) for p in props], exp_nodes)
            bench = score_pipeline(expected, assemble(ex.name, ex.description, props))["structure"]
            rows.append(
                {
                    "held_out": held,
                    "predictor": name,
                    "section_recall": sec["kind_recall"],
                    "section_precision": sec["kind_precision"],
                    "bench_kind_accuracy": bench["kind_accuracy"],
                    "nodes": f"{len(props)}/{len(exp_nodes)}",
                }
            )
    for r in rows:
        print(json.dumps(r))
    for name in ("rules", "knn", "hybrid"):
        rs = [r for r in rows if r["predictor"] == name]
        mean = {
            k: round(sum(r[k] for r in rs) / len(rs), 3)
            for k in ("section_recall", "section_precision", "bench_kind_accuracy")
        }
        print(json.dumps({"embedder": which, "tau": tau, "predictor": name, **mean}))


if __name__ == "__main__":
    main()
