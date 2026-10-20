"""Parallel decomposition prototype: compile a skill as N concurrent sub-skills.

EXPERIMENTAL. Not wired into `rote compile`. Pipeline:

1. split   the deterministic extractor gives the step units; `--parts`
           groups them into contiguous parts (unit index ranges).
2. plan    ONE cheap schema-locked call (default Sonnet 5.5) writes the
           cross-part contract: the global pipeline input, and per part the
           fields it imports from earlier parts and must export.
3. slice   each part becomes a sub-skill: the original frontmatter + H1
           preamble, a contract block, and that part's sections VERBATIM
           (same headings, so source.section stays valid), plus every
           reference file.
4. compile N real `rote compile` runs from an installed wheel, concurrently,
           each with rote state isolated.
5. merge   nodes concatenated; a part's `pipeline.input.<f>` for an imported
           field is rewired to the earlier node that outputs `f`; edges,
           entry/exit and the input schema recomputed; files unioned
           (collisions reported, not resolved); reports concatenated.
6. check   `rote.ir.load_pipeline`, `verify_deliverable` against the ORIGINAL
           skill (citations, tests), and the benchmark scorer against
           examples/<skill>/expected/pipeline.yaml.

usage: python parallel_compile.py --rote <venv>/bin/rote --skill <dir> --parts 0-2,3-4,5-6
           --work <dir> --state <dir> [--planner-model claude-sonnet-5-5]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "system1"))
from extract import extract, parse_frontmatter, parse_sections, walk  # noqa: E402

from rote.inference._runtime_helper import (  # noqa: E402
    CLI_CONTEXT_ISOLATION_FLAGS,
    build_subscription_env,
)

LETTERS = "abcdefgh"


def isolated_env(state: Path) -> dict[str, str]:
    env = dict(os.environ)
    for var, rel in {
        "ROTE_MCP_CONFIG": "mcp-config/mcp.json",
        "ROTE_MCP_TOKEN_DIR": "mcp-tokens",
        "ROTE_APPS_PATH": "rote-apps/apps.json",
        "ROTE_CLOUD_CRED_PATH": "cloud-cred/cloud.json",
        "ROTE_CONFIG_PATH": "rote-config/config.yaml",
        "ROTE_PROJECT_CONFIG_PATH": "rote-config/rote.yaml",
    }.items():
        env[var] = str(state / rel)
    for var in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "OPENAI_API_KEY",
        "ROTE_CLOUD_TOKEN",
        "ROTE_CLOUD_URL",
    ):
        env.pop(var, None)
    return env


# ── 1-2. split + plan ─────────────────────────────────────────────────────

_FIELD = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "snake_case field name"},
        "type": {"type": "string", "description": "short type, e.g. list[Expense], str, number"},
        "description": {"type": "string"},
    },
    "required": ["name", "type", "description"],
    "additionalProperties": False,
}
CONTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "pipeline_input": {"type": "array", "items": _FIELD},
        "parts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "part": {"type": "string"},
                    "imports": {
                        "type": "array",
                        "items": {
                            **_FIELD,
                            "properties": {**_FIELD["properties"], "from_part": {"type": "string"}},
                            "required": [*_FIELD["required"], "from_part"],
                        },
                    },
                    "exports": {"type": "array", "items": _FIELD},
                },
                "required": ["part", "imports", "exports"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["pipeline_input", "parts"],
    "additionalProperties": False,
}


def plan(skill: Path, parts: list[list[str]], model: str) -> tuple[dict, dict]:
    text = (skill / "SKILL.md").read_text()
    listing = "\n".join(f"Part {LETTERS[i]}: " + " | ".join(secs) for i, secs in enumerate(parts))
    prompt = (
        "A skill will be compiled into a workflow in parallel, one part per worker. "
        "Write the data contract between the parts so their outputs can be wired together.\n\n"
        "For the whole skill, list the pipeline input fields (what the caller supplies). For each "
        "part, list the fields it IMPORTS from an earlier part (name, type, from_part) and the "
        "fields it must EXPORT for later parts. Use one snake_case name per field and reuse the "
        "exact same name on both sides. A part may also read pipeline input directly; do not "
        "list pipeline input as an import. Only list cross-part data the steps actually need.\n\n"
        f"Parts (by SKILL.md heading):\n{listing}\n\n=== SKILL.md ===\n{text}"
    )
    command = [
        shutil.which("claude") or "claude",
        "-p",
        "--output-format",
        "json",
        "--json-schema",
        json.dumps(CONTRACT_SCHEMA),
        "--model",
        model,
        "--system-prompt",
        "You design data contracts between workflow parts. Answer only through the schema.",
        *CLI_CONTEXT_ISOLATION_FLAGS,
        "--tools",
        "",
    ]
    t0 = time.monotonic()
    done = subprocess.run(
        command,
        input=prompt,
        capture_output=True,
        text=True,
        env=build_subscription_env(),
        timeout=600,
    )
    env = json.loads(done.stdout)
    if env.get("is_error") or env.get("structured_output") is None:
        raise SystemExit(f"planner failed: {str(env.get('result'))[:400]}")
    usage = {"wall_s": round(time.monotonic() - t0, 1), "cost_usd": env.get("total_cost_usd")}
    return env["structured_output"], usage


# ── 3. slice ──────────────────────────────────────────────────────────────


def slice_skill(
    skill: Path,
    sections: list[str],
    letter: str,
    n_parts: int,
    contract_part: dict,
    pipeline_input: list[dict],
    out: Path,
) -> None:
    text = (skill / "SKILL.md").read_text()
    fm, body, offset = parse_frontmatter(text)
    root = parse_sections(body, offset)
    lines = text.splitlines()
    h1 = next((s for s in walk(root) if s.level == 1), None)
    keep: list[str] = []
    if h1 is not None:
        keep.append(lines[h1.start - 1])
        keep.append(h1.body)
    for sec in walk(root):
        if sec.title in sections:
            # the heading line and its own body, verbatim
            keep.append("")
            keep.append(lines[sec.start - 1])
            keep.append(sec.body)
    imports = contract_part["imports"]
    exports = contract_part["exports"]
    block = [
        "",
        f"<!-- rote parallel compile: part {letter} of {n_parts} -->",
        f"This bundle is part {letter} of {n_parts} of a larger skill, split for parallel "
        "compilation. Compile ONLY the steps in this file; other parts are compiled separately "
        "and merged afterwards.",
        "",
        "The caller supplies these pipeline input fields; use exactly these names "
        "where your steps need them:",
        *[f"- `{f['name']}` ({f['type']}): {f['description']}" for f in pipeline_input],
        "",
        "Data from earlier parts arrives as pipeline input fields with exactly these names:",
        *([f"- `{f['name']}` ({f['type']}): {f['description']}" for f in imports] or ["- (none)"]),
        "",
        "Later parts need these fields as top-level output keys of your nodes, "
        "with exactly these names:",
        *([f"- `{f['name']}` ({f['type']}): {f['description']}" for f in exports] or ["- (none)"]),
        "",
        "Prefix every file you write under extracted/, tests/, signatures/ and evals/ "
        f"with `{letter}_` "
        f"(e.g. extracted/{letter}_checks.py) so the parts' files cannot collide.",
        "",
    ]
    fm = dict(fm)
    fm["name"] = f"{fm.get('name', skill.name)}-part-{letter}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "SKILL.md").write_text(
        "---\n"
        + yaml.safe_dump(fm, sort_keys=False)
        + "---\n\n"
        + "\n".join(keep[:2] + block + keep[2:])
        + "\n"
    )
    if (skill / "references").is_dir():
        shutil.copytree(skill / "references", out / "references", dirs_exist_ok=True)


# ── 4. compile ────────────────────────────────────────────────────────────


async def compile_part(
    rote: Path, sub_skill: Path, out: Path, cwd: Path, env: dict[str, str]
) -> dict:
    cwd.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    proc = await asyncio.create_subprocess_exec(
        str(rote),
        "compile",
        str(sub_skill),
        "--local",
        "--runtime",
        "dbos",
        "--agent",
        "claude",
        "--yes",
        "--out",
        str(out),
        cwd=str(cwd),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    wall = time.monotonic() - t0
    (cwd / "stdout.txt").write_bytes(stdout)
    (cwd / "stderr.txt").write_bytes(stderr)
    metrics_path = out / "compiled" / "compile-metrics.json"
    metrics = json.loads(metrics_path.read_text())["metadata"] if metrics_path.exists() else {}
    return {
        "part": sub_skill.name,
        "rc": proc.returncode,
        "wall_s": round(wall),
        **{
            k: metrics.get(k)
            for k in (
                "cost_usd",
                "num_turns",
                "output_tokens",
                "cache_read_tokens",
                "repair_rounds",
            )
        },
    }


# ── 5. merge ──────────────────────────────────────────────────────────────


def merge(fragments: list[tuple[str, Path]], contract: dict, skill: Path, out: Path) -> dict:
    problems: list[str] = []
    out.mkdir(parents=True, exist_ok=True)
    nodes: list[dict] = []
    edges: list[dict] = []
    producers: dict[str, list[tuple[str, str]]] = {}  # field -> [(part, node_id)]
    input_props: dict = {}
    input_defs: dict = {}  # `$defs` the properties `$ref`; dropping them leaves dangling refs
    input_required: list[str] = []
    imports_by_part = {
        p["part"].lower().removeprefix("part ").strip(): {f["name"]: f for f in p["imports"]}
        for p in contract["parts"]
    }
    seen_ids: set[str] = set()
    config: dict = {}
    for letter, compiled in fragments:
        doc = yaml.safe_load((compiled / "pipeline.yaml").read_text())
        config = config or doc.get("config", {})
        imports = imports_by_part.get(letter, {})
        rename: dict[str, str] = {}
        for n in doc["nodes"]:
            if n["id"] in seen_ids:
                rename[n["id"]] = f"{letter}_{n['id']}"
                problems.append(
                    f"id collision: {n['id']} (part {letter}) renamed {rename[n['id']]}"
                )
            seen_ids.add(rename.get(n["id"], n["id"]))
        rewired = 0
        for n in doc["nodes"]:
            n["id"] = rename.get(n["id"], n["id"])
            for param, ref in list((n.get("inputs") or {}).items()):
                for old, new in rename.items():
                    ref = re.sub(rf"^{re.escape(old)}\.output", f"{new}.output", ref)
                m = re.match(r"^pipeline\.input\.([A-Za-z_][A-Za-z0-9_]*)(.*)$", ref)
                if m and m.group(1) in imports:
                    field_name = m.group(1)
                    cands = producers.get(field_name, [])
                    want = imports[field_name]["from_part"].lower().removeprefix("part ").strip()
                    pick = [c for c in cands if c[0] == want] or cands
                    if not pick:
                        problems.append(
                            f"unresolved import: part {letter} node {n['id']} needs "
                            f"`{field_name}`; no earlier node outputs it"
                        )
                        continue
                    if len(pick) > 1:
                        problems.append(
                            f"ambiguous producer for `{field_name}`: "
                            f"{[c[1] for c in pick]}; took the last"
                        )
                    ref = f"{pick[-1][1]}.output.{field_name}{m.group(2)}"
                    rewired += 1
                n["inputs"][param] = ref
            for key in ("loop_body",):
                if n.get(key):
                    n[key] = [rename.get(x, x) for x in n[key]]
            nodes.append(n)
            if isinstance(n.get("output"), dict):
                for k in n["output"]:
                    producers.setdefault(k, []).append((letter, n["id"]))
        for e in doc.get("edges", []):
            edges.append(
                {**e, "from": rename.get(e["from"], e["from"]), "to": rename.get(e["to"], e["to"])}
            )
        schema = (doc.get("input") or {}).get("input_schema") or {}
        for k, v in (schema.get("properties") or {}).items():
            if k in imports:
                continue
            if k in input_props and input_props[k] != v:
                problems.append(
                    f"pipeline input `{k}` declared differently by two parts; kept the first"
                )
                continue
            input_props.setdefault(k, v)
        for k, v in (schema.get("$defs") or {}).items():
            if k in input_defs and input_defs[k] != v:
                problems.append(f"input schema $defs `{k}` declared differently by two parts")
                continue
            input_defs.setdefault(k, v)
        input_required += [
            r for r in schema.get("required", []) if r not in imports and r not in input_required
        ]
        problems.append(f"part {letter}: rewired {rewired} input(s) to earlier parts")
        for sub in ("extracted", "tests", "signatures", "evals"):
            src = compiled / sub
            if not src.is_dir():
                continue
            for f in src.rglob("*"):
                if f.is_file() and "__pycache__" not in f.parts:
                    dest = out / sub / f.relative_to(src)
                    if dest.exists() and dest.read_bytes() != f.read_bytes():
                        problems.append(
                            f"file collision: {sub}/{f.relative_to(src)} (part {letter} kept out)"
                        )
                        continue
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, dest)
    # every data dependency is also an edge, so waves and the DAG agree
    have = {(e["from"], e["to"]) for e in edges}
    ids = {n["id"] for n in nodes}
    for n in nodes:
        for ref in (n.get("inputs") or {}).values():
            src = ref.split(".", 1)[0]
            if src in ids and src != n["id"] and (src, n["id"]) not in have:
                edges.append({"from": src, "to": n["id"]})
                have.add((src, n["id"]))
    body = {n for n in nodes for n in (n.get("loop_body") or [])}
    has_in = {e["to"] for e in edges}
    has_out = {e["from"] for e in edges}
    top = [n["id"] for n in nodes if n["id"] not in body]
    fm, _, _ = parse_frontmatter((skill / "SKILL.md").read_text())
    merged = {
        "name": fm.get("name", skill.parent.name),
        "version": "0.1.0",
        "source_skill": str(skill),
        "description": str(fm.get("description", "")).strip(),
        "config": config,
        "input": {
            "type": "PipelineInput",
            "required": input_required,
            "input_schema": {
                "type": "object",
                "properties": input_props,
                "required": input_required,
                **({"$defs": input_defs} if input_defs else {}),
            },
        },
        "nodes": nodes,
        "edges": edges,
        "entry_nodes": [i for i in top if i not in has_in],
        "exit_nodes": [i for i in top if i not in has_out],
    }
    (out / "pipeline.yaml").write_text(yaml.safe_dump(merged, sort_keys=False, allow_unicode=True))
    reports = [f"# Compilation report (merged from {len(fragments)} parallel parts)\n"]
    evals: list[dict] = []
    for letter, compiled in fragments:
        rep = (
            (compiled / "compile-report.md").read_text()
            if (compiled / "compile-report.md").exists()
            else ""
        )
        reports.append(f"\n## Part {letter}\n\n" + re.sub(r"^(#+)", r"\1#", rep, flags=re.M))
        if (compiled / "eval.yaml").exists():
            evals += (yaml.safe_load((compiled / "eval.yaml").read_text()) or {}).get("steps", [])
    (out / "compile-report.md").write_text("".join(reports))
    (out / "eval.yaml").write_text(
        yaml.safe_dump({"version": 1, "source_skill": str(skill), "steps": evals}, sort_keys=False)
    )
    return {"problems": problems, "nodes": len(nodes), "edges": len(edges)}


# ── 6. check ──────────────────────────────────────────────────────────────


def check(merged: Path, skill: Path, expected: Path) -> dict:
    from rote.bench.score import score_pipeline
    from rote.compiler.verify import EmitTarget, verify_deliverable
    from rote.ir import load_pipeline

    result: dict = {}
    try:
        p = load_pipeline(merged / "pipeline.yaml")
        result["ir_valid"] = True
    except Exception as e:  # noqa: BLE001 (the report needs the reason, not a crash)
        result["ir_valid"] = False
        result["ir_error"] = str(e)[:1500]
        return result
    v = verify_deliverable(merged, skill_dir=skill, emit_target=EmitTarget("dbos"))
    result["verify_ok"] = v.ok
    result["verify_errors"] = [f"{f.node_id}:{f.kind}: {f.message[:200]}" for f in v.errors]
    result["tests"] = v.tests.as_dict() if v.tests else None
    s = score_pipeline(load_pipeline(expected), p)
    result["score"] = {
        "structure": {
            k: s["structure"][k]
            for k in (
                "expected_nodes",
                "compiled_nodes",
                "node_recall",
                "node_precision",
                "kind_accuracy",
            )
        },
        "graph": s["graph"],
        "mandatory": s["mandatory"],
        "fan_out": s["fan_out"],
        "gates": {k: s["gates"][k] for k in ("expected", "found", "signal_matches")},
        "inputs": {k: s["inputs"][k] for k in ("precision", "recall")},
    }
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rote", type=Path, required=True)
    ap.add_argument("--skill", type=Path, required=True)
    ap.add_argument("--parts", required=True, help="unit index ranges, e.g. 0-2,3-4,5-6")
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--planner-model", default="claude-sonnet-5-5")
    ap.add_argument(
        "--merge-only", action="store_true", help="re-merge and re-check existing part outputs"
    )
    ap.add_argument(
        "--plan-only", action="store_true", help="write the contract and slices, compile nothing"
    )
    args = ap.parse_args()
    skill = args.skill.resolve()
    units = extract(skill).units
    parts: list[list[str]] = []
    for rng in args.parts.split(","):
        a, b = (int(x) for x in rng.split("-"))
        secs: list[str] = []
        for u in units[a : b + 1]:
            if u.section not in secs:
                secs.append(u.section)
        parts.append(secs)
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    os.environ.update(isolated_env(args.state.resolve()))
    t0 = time.monotonic()
    if not args.merge_only:
        contract, plan_usage = plan(skill, parts, args.planner_model)
        (work / "contract.json").write_text(
            json.dumps({"parts": parts, "contract": contract, "usage": plan_usage}, indent=1)
        )
        for i, secs in enumerate(parts):
            cpart = (
                contract["parts"][i]
                if i < len(contract["parts"])
                else {"imports": [], "exports": []}
            )
            slice_skill(
                skill,
                secs,
                LETTERS[i],
                len(parts),
                cpart,
                contract["pipeline_input"],
                work / "slices" / f"part-{LETTERS[i]}" / "skill",
            )
        if args.plan_only:
            print(json.dumps({"parts": parts, "contract": contract, "usage": plan_usage}, indent=1))
            return
        env = isolated_env(args.state.resolve())

        async def run_all() -> list[dict]:
            return await asyncio.gather(
                *[
                    compile_part(
                        args.rote,
                        work / "slices" / f"part-{LETTERS[i]}" / "skill",
                        work / "out" / f"part-{LETTERS[i]}",
                        work / "cwd" / f"part-{LETTERS[i]}",
                        env,
                    )
                    for i in range(len(parts))
                ]
            )

        t_c = time.monotonic()
        runs = asyncio.run(run_all())
        compile_wall = round(time.monotonic() - t_c)
        (work / "runs.json").write_text(
            json.dumps({"runs": runs, "compile_wall_s": compile_wall}, indent=1)
        )
    saved = json.loads((work / "contract.json").read_text())
    contract, plan_usage = saved["contract"], saved["usage"]
    runs_doc = json.loads((work / "runs.json").read_text())
    fragments = [
        (LETTERS[i], work / "out" / f"part-{LETTERS[i]}" / "compiled") for i in range(len(parts))
    ]
    merged_dir = work / "merged" / "compiled"
    if merged_dir.exists():
        shutil.rmtree(merged_dir)
    merge_info = merge(fragments, contract, skill, merged_dir)
    checked = check(merged_dir, skill, skill.parent / "expected" / "pipeline.yaml")
    total_cost = (plan_usage.get("cost_usd") or 0) + sum(
        r.get("cost_usd") or 0 for r in runs_doc["runs"]
    )
    (merged_dir / "compile-metrics.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "driver": "parallel-prototype",
                "cost_basis": "driver_estimate",
                "metadata": {
                    "cost_usd": total_cost,
                    "num_turns": sum(r.get("num_turns") or 0 for r in runs_doc["runs"]),
                    "output_tokens": sum(r.get("output_tokens") or 0 for r in runs_doc["runs"]),
                },
            },
            indent=1,
        )
    )
    summary = {
        "parts": parts,
        "planner": plan_usage,
        "runs": runs_doc["runs"],
        "compile_wall_s": runs_doc["compile_wall_s"],
        "end_to_end_wall_s": round(time.monotonic() - t0) if not args.merge_only else None,
        "total_cost_usd": round(total_cost, 3),
        "merge": merge_info,
        "check": checked,
    }
    (work / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
