"""Test that the IR can load and validate the BDR pipeline.

This is the first concrete validation that the IR is expressive enough to
represent a real complex skill. The BDR pipeline.yaml is hand-written from
the analysis of the bdr-outreach skill and exercises every node kind and
most optional fields.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rote.ir import (
    Edge,
    NodeKind,
    Pipeline,
    PipelineInput,
    load_pipeline,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
BDR_PIPELINE = REPO_ROOT / "examples" / "bdr-outreach" / "expected" / "pipeline.yaml"


# ───────── Loading the BDR pipeline ─────────


@pytest.fixture(scope="module")
def bdr_pipeline() -> Pipeline:
    assert BDR_PIPELINE.exists(), f"Missing fixture: {BDR_PIPELINE}"
    return load_pipeline(BDR_PIPELINE)


def test_bdr_pipeline_loads(bdr_pipeline: Pipeline) -> None:
    assert bdr_pipeline.name == "bdr-campaign"
    assert bdr_pipeline.version == "0.1.0"
    assert "End-to-end BDR" in bdr_pipeline.description


def test_bdr_pipeline_input_contract(bdr_pipeline: Pipeline) -> None:
    pi = bdr_pipeline.input
    assert isinstance(pi, PipelineInput)
    assert pi.type == "CampaignBrief"
    # Spot-check a few required fields
    assert "drug_brand" in pi.required
    assert "campaign_type" in pi.required
    assert "target_quota" in pi.required
    assert "job_focus" in pi.optional


def test_bdr_pipeline_uses_all_five_node_kinds(bdr_pipeline: Pipeline) -> None:
    """Every NodeKind should appear at least once in the BDR pipeline.

    This is the load-bearing assertion: if the BDR pipeline doesn't exercise
    every kind, the IR isn't being stress-tested by a real skill.
    """
    present_kinds = {n.kind for n in bdr_pipeline.nodes}
    assert present_kinds == set(NodeKind), (
        f"BDR pipeline missing node kinds: {set(NodeKind) - present_kinds}"
    )


def test_bdr_mandatory_exclusion_checks(bdr_pipeline: Pipeline) -> None:
    """All three exclusion checks must be marked mandatory."""
    expected_mandatory = {
        "exclusion_check_dnc",
        "exclusion_check_recent",
        "exclusion_check_sequence",
    }
    actual_mandatory = {n.id for n in bdr_pipeline.nodes if n.mandatory}
    assert actual_mandatory == expected_mandatory


def test_bdr_hitl_gates(bdr_pipeline: Pipeline) -> None:
    """Both HITL gates exist and have the right signal names."""
    gates = bdr_pipeline.nodes_by_kind(NodeKind.HITL_GATE)
    by_id = {g.id: g for g in gates}
    assert "contact_review_gate" in by_id
    assert "manual_enrollment_handoff" in by_id
    assert by_id["contact_review_gate"].signal == "contact_review_approved"
    assert by_id["manual_enrollment_handoff"].signal == "bdr_enrollment_complete"
    # Both gates should have a notify config
    assert by_id["contact_review_gate"].notify is not None
    assert by_id["manual_enrollment_handoff"].notify is not None


def test_bdr_loop_structure(bdr_pipeline: Pipeline) -> None:
    """The lead generation loop has a loop_body referencing real nodes."""
    loop = bdr_pipeline.node_by_id("lead_generation_loop")
    assert loop.kind is NodeKind.AGENT_LOOP
    assert loop.loop_body == ["enrich_contact_batch", "vet_contact"]
    assert loop.termination is not None
    assert loop.termination.max_iterations == 10
    # Loop body nodes exist as top-level nodes
    bdr_pipeline.node_by_id("enrich_contact_batch")
    bdr_pipeline.node_by_id("vet_contact")


def test_bdr_constants_extracted(bdr_pipeline: Pipeline) -> None:
    """Hard-coded constants from the source skill survive into the IR."""
    enrich = bdr_pipeline.node_by_id("enrich_contact_batch")
    assert enrich.constants is not None
    assert enrich.constants["batch_size"] == 10

    upsert = bdr_pipeline.node_by_id("hubspot_upsert")
    assert upsert.constants is not None
    assert upsert.constants["batch_size"] == 100

    create_list = bdr_pipeline.node_by_id("hubspot_create_list")
    assert create_list.constants is not None
    assert create_list.constants["add_batch_size"] == 250

    recent = bdr_pipeline.node_by_id("exclusion_check_recent")
    assert recent.constants is not None
    assert recent.constants["days_back"] == 30


def test_bdr_edges_well_formed(bdr_pipeline: Pipeline) -> None:
    """Every edge points at a real node (validation should have caught this)."""
    node_ids = {n.id for n in bdr_pipeline.nodes}
    for edge in bdr_pipeline.edges:
        assert edge.from_ in node_ids
        assert edge.to in node_ids
    # The contact_review_gate exit edge has on_signal set
    review_exit = next(e for e in bdr_pipeline.edges if e.from_ == "contact_review_gate")
    assert review_exit.on_signal == "approved"


def test_bdr_entry_and_exit_nodes(bdr_pipeline: Pipeline) -> None:
    assert set(bdr_pipeline.entry_nodes) == {"target_research", "taxonomy_lookup"}
    assert bdr_pipeline.exit_nodes == ["manual_enrollment_handoff"]


# ───────── Validation tests (negative paths) ─────────


def test_pure_function_node_requires_impl() -> None:
    from pydantic import ValidationError

    from rote.ir import Node

    with pytest.raises(ValidationError, match="missing required field"):
        Node(
            id="bad",
            kind=NodeKind.PURE_FUNCTION,
            description="missing impl",
        )


def test_llm_judge_node_requires_signature() -> None:
    from pydantic import ValidationError

    from rote.ir import Node

    with pytest.raises(ValidationError, match="missing required field"):
        Node(
            id="bad",
            kind=NodeKind.LLM_JUDGE,
            description="missing signature",
        )


def test_hitl_gate_node_requires_signal() -> None:
    from pydantic import ValidationError

    from rote.ir import Node

    with pytest.raises(ValidationError, match="missing required field"):
        Node(
            id="bad",
            kind=NodeKind.HITL_GATE,
            description="missing signal",
        )


def test_agent_loop_node_requires_tools() -> None:
    from pydantic import ValidationError

    from rote.ir import Node

    with pytest.raises(ValidationError, match="missing required field"):
        Node(
            id="bad",
            kind=NodeKind.AGENT_LOOP,
            description="missing tools",
        )


def test_agent_loop_cannot_be_mandatory() -> None:
    from pydantic import ValidationError

    from rote.ir import Node

    with pytest.raises(ValidationError, match="mandatory=true is not allowed"):
        Node(
            id="bad",
            kind=NodeKind.AGENT_LOOP,
            description="mandatory loop",
            tools=["foo"],
            mandatory=True,
        )


def test_pipeline_rejects_unknown_edge_target() -> None:
    from pydantic import ValidationError

    from rote.ir import Node

    with pytest.raises(ValidationError, match="Edge to unknown node"):
        Pipeline(
            name="bad",
            input=PipelineInput(type="X", required=[]),
            nodes=[
                Node(
                    id="a",
                    kind=NodeKind.PURE_FUNCTION,
                    description="x",
                    impl="x.py:y",
                )
            ],
            edges=[Edge(**{"from": "a", "to": "nonexistent"})],
        )
