"""Regression guards for the committed example pipelines.

Every example's ``expected/pipeline.yaml`` is a hand-adapted regression
baseline (see each example's README); these tests keep them loading and
keep the archetype each one exists to demonstrate from silently
regressing — ops-report is the 100%-roteness + HITL archetype,
deal-monitor the data-heavy fan-out archetype.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rote.ir import NodeKind, load_pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_PIPELINES = sorted((REPO_ROOT / "examples").glob("*/expected/pipeline.yaml"))


@pytest.mark.parametrize("pipeline_yaml", EXAMPLE_PIPELINES, ids=lambda p: p.parent.parent.name)
def test_every_example_expected_ir_validates(pipeline_yaml: Path) -> None:
    pipeline = load_pipeline(pipeline_yaml)
    assert pipeline.nodes
    # Each example's source skill ships alongside its baseline and the
    # recorded pointer must resolve from the pipeline.yaml's location.
    assert pipeline.source_skill is not None
    skill_dir = (pipeline_yaml.parent / pipeline.source_skill).resolve()
    assert (skill_dir / "SKILL.md").is_file()


def test_ops_report_is_the_full_roteness_hitl_archetype() -> None:
    pipeline = load_pipeline(REPO_ROOT / "examples" / "ops-report" / "expected" / "pipeline.yaml")
    kinds = {n.kind for n in pipeline.nodes}
    # The whole point: zero inference nodes, one human gate.
    assert NodeKind.LLM_JUDGE not in kinds
    assert NodeKind.AGENT_LOOP not in kinds
    gates = [n for n in pipeline.nodes if n.kind is NodeKind.HITL_GATE]
    assert [g.id for g in gates] == ["manual_data_gate"]
    # …which is exactly what makes the python adapter refuse it.
    assert pipeline.requires_durable_execution


def test_deal_monitor_is_the_data_heavy_fan_out_archetype() -> None:
    pipeline = load_pipeline(REPO_ROOT / "examples" / "deal-monitor" / "expected" / "pipeline.yaml")
    judges = [n for n in pipeline.nodes if n.kind is NodeKind.LLM_JUDGE]
    assert len(judges) == 3
    # Cross-language judges: every one carries a structured signature_spec.
    assert all(n.signature_spec is not None for n in judges)
    # Two per-item judges fan out; the batch parser does not.
    assert sorted(n.id for n in pipeline.nodes if n.fan_out) == [
        "classify_thread_step",
        "score_new_opportunities",
    ]
    # Parallel entry: Slack pull and fixed Gmail searches start together.
    assert len(pipeline.entry_nodes) == 2
    # The external_call nodes carry MCP bindings — the estimator's
    # payload model keys off these (payload_tokens_per_tool).
    bound = [n for n in pipeline.nodes if n.kind is NodeKind.EXTERNAL_CALL and n.mcp]
    assert len(bound) >= 3
