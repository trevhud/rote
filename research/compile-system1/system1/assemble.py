"""System-1 prototype: template IR assembly from classified steps (EXPERIMENTAL).

Turns a list of node proposals (from the deterministic extractor or the
small-model classifier) into a `rote.ir.Pipeline` that passes the real IR
validator, so it can be scored by `rote.bench.score.score_pipeline`, the
same scorer the compile benchmark uses for the agent's output.

What it fills deterministically: ids, kinds, `source.section`,
`mandatory`, `constants` (literal thresholds the prose states, per step),
kind-required references (`impl:`, `signature:`, `signal:`, `tools:`), and a
dependency chain in document order. What it does NOT know, and the scores
show: pipeline input schema, real data flow (`inputs:`), parallel waves,
fan_out, payload shapes. Those stay with the big model.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from rote.ir import Pipeline

sys.path.insert(0, str(Path(__file__).parent))
from extract import Extraction, extract, slug  # noqa: E402


@dataclass
class Proposed:
    id: str
    kind: str
    section: str
    description: str
    mandatory: bool = False
    tools: list[str] = field(default_factory=list)
    mcp: dict | None = None
    confidence: float | None = None


def _ident(raw: str, used: set[str]) -> str:
    base = slug(raw)
    if not re.match(r"[a-z_]", base):
        base = f"n_{base}"
    out, n = base, 2
    while out in used:
        out, n = f"{base}_{n}", n + 1
    used.add(out)
    return out


def from_extraction(ex: Extraction) -> list[Proposed]:
    out = []
    for u in ex.units:
        words = " ".join(u.text.split()[:30])
        for p in u.proposals:
            out.append(
                Proposed(
                    id=p.id,
                    kind=p.kind,
                    section=u.section,
                    description=f"{u.title}. {words}",
                    mandatory=u.mandatory,
                    tools=u.tools,
                    mcp=u.mcp[0] if u.mcp and p.kind == "external_call" else None,
                    confidence=p.confidence,
                )
            )
    return out


def from_classifier(ex: Extraction, result: dict) -> list[Proposed]:
    out = []
    if result["mode"] == "hinted":
        for row in result["output"]["units"]:
            u = ex.units[row["unit"]]
            for n in row["nodes"]:
                out.append(
                    Proposed(
                        id=n["id"],
                        kind=n["kind"],
                        section=u.section,
                        description=f"{n['reason']}. {u.title}",
                        mandatory=u.mandatory,
                        tools=u.tools,
                        mcp=u.mcp[0] if u.mcp and n["kind"] == "external_call" else None,
                        confidence=n["confidence"],
                    )
                )
    else:
        for n in result["output"]["nodes"]:
            out.append(
                Proposed(
                    id=n["id"],
                    kind=n["kind"],
                    section=n["section"],
                    description=n["reason"],
                    mandatory="mandatory" in n["section"].lower(),
                    confidence=n["confidence"],
                )
            )
    return out


def assemble(name: str, description: str, nodes: list[Proposed]) -> Pipeline:
    used: set[str] = set()
    rendered = []
    ids = []
    for p in nodes:
        nid = _ident(p.id, used)
        ids.append(nid)
        node: dict = {
            "id": nid,
            "kind": p.kind,
            "description": p.description,
            "source": {"section": p.section},
            "mandatory": p.mandatory and p.kind != "agent_loop",
            "output": {"result": "dict"},
        }
        if p.kind in ("pure_function", "external_call"):
            node["impl"] = f"extracted/{nid}.py:{nid}"
        elif p.kind == "llm_judge":
            node["signature"] = f"signatures/{nid}.py:{''.join(w.title() for w in nid.split('_'))}"
        elif p.kind == "agent_loop":
            node["tools"] = [t for t in p.tools if re.fullmatch(r"[A-Za-z0-9_.-]+", t)] or [
                f"{nid}_tool"
            ]
        elif p.kind == "hitl_gate":
            node["signal"] = nid
        rendered.append(node)
    edges = [{"from": a, "to": b} for a, b in zip(ids, ids[1:], strict=False)]
    doc = {
        "name": slug(name),
        "description": description[:400],
        "input": {"type": "SkillInput"},
        "nodes": rendered,
        "edges": edges,
        "entry_nodes": ids[:1],
        "exit_nodes": ids[-1:],
    }
    return Pipeline.model_validate(doc)


def dump(p: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(p.model_dump(mode="json", by_alias=True, exclude_none=True), sort_keys=False)
    )


if __name__ == "__main__":
    skill = Path(sys.argv[1])
    ex = extract(skill)
    props = (
        from_classifier(ex, json.loads(Path(sys.argv[2]).read_text()))
        if len(sys.argv) > 2
        else from_extraction(ex)
    )
    print(
        yaml.safe_dump(
            assemble(ex.name, ex.description, props).model_dump(
                mode="json", by_alias=True, exclude_none=True
            ),
            sort_keys=False,
        )
    )
