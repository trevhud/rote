"""
Render the Phase 7 compile-report.md from structured compilation results.

The report covers:
  1. Summary metrics — node count, breakdown by kind, HITL gates, codifiable %.
  2. Crystallization log — every prose-to-code extraction from Phase 3.
  3. Open questions — judgment calls the reviewer should verify.
  4. Suggested next steps — what to dogfood, which evals to run.

Contract:
  Input:  classified_steps   — list of ClassifyStepOutput dicts
          codification_plan  — CodificationPlan dict from codification_scan
          signature_designs  — list of SignatureDesign dicts from design_signatures
          validation_result  — ValidationResult dict from validate_ir
          emitted_files      — list of file paths from invoke_adapter (empty if report_only)
          out_dir            — directory to write compile-report.md into
          skill_dir          — path to the source skill
  Output: dict with keys:
            report_path     — path of the written compile-report.md
            report_markdown — the full markdown string

Raises:
  RuntimeError — if out_dir does not exist or cannot be written to
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path


def write_compile_report(
    classified_steps: list[dict],
    codification_plan: dict,
    signature_designs: list[dict],
    validation_result: dict,
    emitted_files: list[str],
    out_dir: str | None,
    skill_dir: str,
) -> dict:
    """Render compile-report.md from compilation artifacts."""
    out_path = Path(out_dir) if out_dir else Path.cwd()
    if not out_path.is_dir():
        raise RuntimeError(
            f"out_dir does not exist or is not a directory: {out_dir}"
        )

    kind_counts = Counter(s.get("kind", "unknown") for s in classified_steps)
    total_nodes = len(classified_steps)
    codifiable = kind_counts.get("pure_function", 0) + kind_counts.get("external_call", 0)
    llm_nodes = total_nodes - kind_counts.get("hitl_gate", 0)
    codifiable_pct = round(codifiable / llm_nodes * 100) if llm_nodes > 0 else 0

    hitl_steps = [s for s in classified_steps if s.get("kind") == "hitl_gate"]
    extractions = codification_plan.get("extractions", [])
    warnings = validation_result.get("warnings", [])

    lines: list[str] = []

    lines += [
        f"# Compile Report — rote-compile",
        "",
        f"**Source skill:** `{skill_dir}`  ",
        f"**Nodes:** {total_nodes} total  ",
        f"**Codifiable:** {codifiable}/{llm_nodes} non-gate nodes ({codifiable_pct}%)  ",
        "",
        "## Summary metrics",
        "",
        "| Kind | Count |",
        "|------|-------|",
    ]
    for kind in ["pure_function", "external_call", "llm_judge", "agent_loop", "hitl_gate"]:
        lines.append(f"| `{kind}` | {kind_counts.get(kind, 0)} |")

    lines += ["", "### HITL gates"]
    if hitl_steps:
        for s in hitl_steps:
            step = s.get("step", {})
            lines.append(f"- **{step.get('name', s.get('step_id', '?'))}** — {step.get('description', '')}")
    else:
        lines.append("None. This skill has no human-in-the-loop gates.")

    lines += [
        "",
        "### Adapter invocation",
    ]
    if emitted_files:
        lines.append(f"Emitted {len(emitted_files)} files.")
        for f in emitted_files[:10]:
            lines.append(f"- `{f}`")
        if len(emitted_files) > 10:
            lines.append(f"- … and {len(emitted_files) - 10} more")
    else:
        lines.append("Adapter not invoked (report_only mode or adapter not configured).")

    if warnings:
        lines += ["", "### Validation warnings"]
        for w in warnings:
            lines.append(f"- {w}")

    lines += ["", "## Crystallization log", ""]
    if extractions:
        for ex in extractions:
            source = ex.get("source_file", "?")
            prose = ex.get("prose_excerpt", "")
            fn_name = ex.get("function_name", "?")
            lines += [
                f"### `{fn_name}`",
                "",
                f"**Source:** `{source}`  ",
                f"**Before (prose):** {prose[:200]}{'…' if len(prose) > 200 else ''}  ",
                f"**After (code):** `extracted/` — `{fn_name}()`  ",
                "",
            ]
    else:
        lines.append(
            "No extractions recorded. Codification scan found no crystallizable patterns, "
            "or the codification_plan was not populated by the agent loop."
        )

    lines += [
        "",
        "## Signature designs",
        "",
    ]
    if signature_designs:
        for sd in signature_designs:
            node_id = sd.get("node_id", "?")
            lines += [
                f"- **`{node_id}`** — `signatures/{node_id}.py` + `evals/{node_id}.jsonl`  ",
                f"  Output schema: {sd.get('output_summary', '(see file)')}  ",
            ]
    else:
        lines.append(
            "No signature designs recorded (design_signatures agent loop output not available)."
        )

    lines += [
        "",
        "## Open questions",
        "",
        "1. **identify_steps granularity.** The `identify_steps` llm_judge produces the "
        "step list that everything else fans out over. If it over-merges steps, downstream "
        "classification will be coarse. Reviewers should check whether each classified step "
        "maps to exactly one IR node, or whether any steps bundle two distinct operations.",
        "",
        "2. **codification_scan coverage.** The codification_scan agent loop has no "
        "automatic ground-truth check. The Phase 6 checklist is the backstop (validate_ir "
        "will fail if any pure_function/external_call node is missing an impl:), but the "
        "QUALITY of the extracted stubs (completeness of docstring contracts, correctness "
        "of constants) requires human review. Check extracted/ against the SKILL.md prose.",
        "",
        "3. **agent_loop tool_servers completeness.** The codification_scan, design_signatures, "
        "and assemble_ir agent loops all declare filesystem tool_servers. If any of these loops "
        "reach additional MCP servers during a real run, the pipeline's required_mcp_servers "
        "manifest will under-report. Add those servers to the node's tool_servers map after "
        "the first production run.",
        "",
        "4. **estimate_eval_turns calibration.** The llm_judge for turn estimation has no "
        "ground-truth baseline for the rote-compile skill itself. The estimates should be "
        "treated as rough order-of-magnitude until `rote eval --run` provides measured data.",
        "",
        "## Suggested next steps",
        "",
        "1. **Dogfood first on a simple skill.** Run the compiled pipeline on a small, "
        "well-understood skill (e.g. a 3-phase skill with no fan_out). Compare its output "
        "pipeline.yaml to a hand-crafted reference. This is the fastest way to verify that "
        "identify_steps and classify_step are calibrated correctly.",
        "",
        "2. **Run `rote eval` on the BDR skill.** BDR has a committed expected/pipeline.yaml "
        "snapshot. Compiling it with the new pipeline and diffing against the snapshot is the "
        "regression test for classification drift.",
        "",
        "3. **Expand evals/classify_step.jsonl.** The seed set covers the five kinds with "
        "one example each. After classifying a few real skills, add the cases where you "
        "disagreed with the model's first choice — those are the highest-value eval entries.",
        "",
        "4. **First node to recompile after production data:** `identify_steps`. The step "
        "identification quality is the load-bearing precondition for everything else. If a "
        "production run reveals that the model consistently over-splits or under-splits, "
        "update the prompt and recompile that node alone with `rote compile --update`.",
        "",
    ]

    report_markdown = "\n".join(lines)
    report_path = out_path / "compile-report.md"
    report_path.write_text(report_markdown, encoding="utf-8")

    return {
        "report_path": str(report_path),
        "report_markdown": report_markdown,
    }
