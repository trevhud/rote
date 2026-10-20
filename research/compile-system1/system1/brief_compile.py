"""Option 1 experiment: today's compile, plus a System-1 brief (EXPERIMENTAL).

Builds the brief (deterministic extractor + rule tables + one Haiku
classification), drops it into a COPY of the skill as
`references/zz-rote-system1-brief.md`, and runs an unmodified `rote compile`
from an installed wheel on that copy. No rote source changes: the agent finds
the brief because the rubric tells it to read every reference file.

This is a measurement harness, not the proposed integration (that would hand
the brief to the driver as a work-dir file, outside the skill). Citations
the agent takes from the brief would pass provenance here and must not in
the real thing; the report counts them.

usage: python brief_compile.py --rote <venv>/bin/rote --skill <dir> --work <dir> --state <dir>
                               [--classifier-model claude-haiku-5-5]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from classify import run as classify  # noqa: E402
from extract import extract  # noqa: E402
from rule_tables import render as render_tables  # noqa: E402

HEADER = """\
# rote System-1 brief (machine-generated; NOT part of the skill)

rote produced this file deterministically (plus one small-model call) before
your run. It is a pre-analysis of the skill, not skill text: never cite it in
a `# skill:` comment, and never compile it as a step. Use it to skip
re-deriving what it already settles; where it is wrong, the skill wins.

Confidence below 0.8 means the classifier was unsure: decide those yourself.
"""


def brief(skill: Path, model: str) -> tuple[str, dict]:
    ex = extract(skill)
    result = classify("hinted", model, skill)
    lines = [HEADER, "## Steps and proposed nodes", ""]
    lines.append(
        "| # | SKILL.md section | step | proposed nodes (kind, confidence) | MANDATORY | MCP |"
    )
    lines.append("|---|---|---|---|---|---|")
    by_unit = {row["unit"]: row["nodes"] for row in result["output"]["units"]}
    for i, u in enumerate(ex.units):
        nodes = "; ".join(
            f"`{n['id']}` {n['kind']} ({n['confidence']:.2f})" for n in by_unit.get(i, [])
        )
        mcp = ", ".join(f"{m['server']}.{m['tool'] or '?'}" for m in u.mcp)
        lines.append(
            f"| {i} | {u.section} | {u.title} | {nodes or '(none)'} | "
            f"{'yes' if u.mandatory else ''} | {mcp} |"
        )
    lines += ["", "## Literal thresholds and constants the prose states", ""]
    seen = set()
    for c in ex.constants:
        if c.kind == "table_cell":
            continue
        key = (c.file, c.line, c.raw)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- {c.file}:{c.line}: `{c.raw}`")
    tables_py, _ = render_tables(skill)
    lines += [
        "",
        "## Rule tables, already lifted to Python (verify, then reuse)",
        "",
        "```python",
        tables_py,
        "```",
        "",
    ]
    return "\n".join(lines), {"cost_usd": result["cost_usd"], "wall_s": result["wall_s"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rote", type=Path, required=True)
    ap.add_argument("--skill", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--classifier-model", default="claude-haiku-5-5")
    args = ap.parse_args()
    for var, rel in {
        "ROTE_MCP_CONFIG": "mcp-config/mcp.json",
        "ROTE_MCP_TOKEN_DIR": "mcp-tokens",
        "ROTE_APPS_PATH": "rote-apps/apps.json",
        "ROTE_CLOUD_CRED_PATH": "cloud-cred/cloud.json",
        "ROTE_CONFIG_PATH": "rote-config/config.yaml",
        "ROTE_PROJECT_CONFIG_PATH": "rote-config/rote.yaml",
    }.items():
        os.environ[var] = str(args.state.resolve() / rel)
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY"):
        os.environ.pop(var, None)
    work = args.work.resolve()
    skill_copy = work / "skill"
    if skill_copy.exists():
        shutil.rmtree(skill_copy)
    shutil.copytree(args.skill.resolve(), skill_copy)
    t0 = time.monotonic()
    text, usage = brief(args.skill.resolve(), args.classifier_model)
    (skill_copy / "references").mkdir(exist_ok=True)
    (skill_copy / "references" / "zz-rote-system1-brief.md").write_text(text)
    brief_s = round(time.monotonic() - t0, 1)
    cwd = work / "cwd"
    cwd.mkdir(parents=True, exist_ok=True)
    t1 = time.monotonic()
    done = subprocess.run(
        [str(args.rote), "compile", str(skill_copy), "--local", "--runtime", "dbos",
         "--agent", "claude", "--yes", "--out", str(work / "out")],
        cwd=cwd, capture_output=True, text=True,
    )  # fmt: skip
    (work / "stderr.txt").write_text(done.stderr)
    compile_s = round(time.monotonic() - t1)
    metrics = json.loads((work / "out" / "compiled" / "compile-metrics.json").read_text())
    summary = {
        "rc": done.returncode,
        "brief": {**usage, "build_s": brief_s},
        "compile_wall_s": compile_s,
        "metadata": metrics["metadata"],
    }
    (work / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
