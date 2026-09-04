"""
Render the eval.yaml sidecar from per-step turn estimates.

eval.yaml describes the SOURCE SKILL's behavior as a raw agent (the "before"
side of the eval scorecard). It is NOT part of the pipeline IR.

Schema (rote.eval.sidecar.EvalEstimates):
  version: 1
  source_skill: <path>
  steps:
    - description: <str>          # step name in skill vocabulary
      node_id: <str>              # IR node id (1:1 mapping, if clean)
      phase: <str>                # phase from skill
      estimated_turns: {low, high}
      estimated_tool_calls: {low, high}  # optional
      iterations: {low, high}    # when step repeats per item
  totals: {low, high}            # optional; omit if sum of steps is right

Contract:
  Input:  turn_estimates       — list of EstimateEvalTurnsOutput dicts
          classified_steps     — list of ClassifyStepOutput dicts (for step metadata)
          out_dir              — directory to write eval.yaml into
          source_skill_dir     — path to the source skill (written into the sidecar)
  Output: dict with key:
            eval_yaml_path — path of the written eval.yaml file

Raises:
  RuntimeError  — if out_dir does not exist or cannot be written to
  ValueError    — if turn_estimates and classified_steps have different lengths
"""

from __future__ import annotations

from pathlib import Path

import yaml


def write_eval_sidecar(
    turn_estimates: list[dict],
    classified_steps: list[dict],
    out_dir: str | None,
    source_skill_dir: str,
) -> dict:
    """Write eval.yaml from per-step turn estimates and step metadata."""
    if len(turn_estimates) != len(classified_steps):
        raise ValueError(
            f"turn_estimates length ({len(turn_estimates)}) must equal "
            f"classified_steps length ({len(classified_steps)}): "
            "lists must be aligned (same fan-out order)."
        )

    out_path = Path(out_dir) if out_dir else Path.cwd()
    if not out_path.is_dir():
        raise RuntimeError(
            f"out_dir does not exist or is not a directory: {out_dir}"
        )

    steps: list[dict] = []
    has_any_iterations = False

    for estimate, classified in zip(turn_estimates, classified_steps):
        step = classified.get("step", {})
        has_iters = estimate.get("has_iterations", False)
        if has_iters:
            has_any_iterations = True

        entry: dict = {
            "description": step.get("name", step.get("step_id", "unknown")),
            "node_id": step.get("step_id"),
            "estimated_turns": {
                "low": estimate["turns_low"],
                "high": estimate["turns_high"],
            },
        }
        if step.get("phase"):
            entry["phase"] = step["phase"]
        if has_iters and estimate.get("iterations_low") is not None:
            entry["iterations"] = {
                "low": estimate["iterations_low"],
                "high": estimate["iterations_high"],
            }
        if estimate.get("notes"):
            entry["notes"] = estimate["notes"]
        steps.append(entry)

    doc: dict = {
        "version": 1,
        "source_skill": str(source_skill_dir),
        "steps": steps,
    }

    if has_any_iterations:
        # Caller responsible for totals if any step has iterations —
        # auto-summing would double-count the per-iteration cost.
        # Leave totals absent; rote eval will compute it from steps × iterations.
        pass

    eval_yaml_path = out_path / "eval.yaml"
    with open(eval_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(doc, f, allow_unicode=True, sort_keys=False)

    return {"eval_yaml_path": str(eval_yaml_path)}
