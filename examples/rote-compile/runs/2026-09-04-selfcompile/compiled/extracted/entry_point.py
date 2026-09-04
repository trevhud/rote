"""
Determine the compilation entry point from CLI flags.

The rote-compile skill has four execution modes:
  full       — run all phases, emit runtime code (default)
  report_only — run phases 1-5 only, skip adapter invocation
  update     — run phases 1-7 with incremental diff against existing IR
  re_emit    — skip to phase 6 (use existing pipeline.yaml)

Contract:
  Input:  report_only: bool, update: bool
  Output: dict with keys:
            mode   — one of 'full', 'report_only', 'update', 're_emit'
            phases — ordered list of phase names to execute

Raises:
  ValueError — if incompatible flags are combined (e.g. both report_only and update)
"""

from __future__ import annotations

PHASE_NAMES = [
    "intake",
    "node_classification",
    "codification_scan",
    "llm_judge_extraction",
    "ir_assembly",
    "adapter_invocation",
    "compilation_report",
]

REPORT_ONLY_PHASES = [
    "intake",
    "node_classification",
    "codification_scan",
    "llm_judge_extraction",
    "ir_assembly",
    "compilation_report",
]


def detect_entry_point(report_only: bool, update: bool) -> dict:
    """Determine which phases to run and the overall compilation mode."""
    if report_only and update:
        raise ValueError(
            "report_only and update are mutually exclusive: "
            "report_only skips the adapter; update re-derives changed nodes and always emits."
        )

    if report_only:
        mode = "report_only"
        phases = REPORT_ONLY_PHASES
    elif update:
        mode = "update"
        phases = PHASE_NAMES
    else:
        mode = "full"
        phases = PHASE_NAMES

    return {"mode": mode, "phases": phases}
