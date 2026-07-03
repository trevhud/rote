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
)

REPO_ROOT = Path(__file__).resolve().parent.parent
BDR_PIPELINE = REPO_ROOT / "examples" / "bdr-outreach" / "expected" / "pipeline.yaml"


# ───────── Loading the BDR pipeline ─────────


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


# ───────── Code-injection / traversal hardening on emitted-verbatim fields ─────────


@pytest.mark.parametrize(
    "bad_id",
    [
        "/tmp/rote_abs_pwned",  # absolute path → escapes output dir via pathlib join
        "../../../../etc/pwned",  # ../ traversal out of the output dir
        "ok\n    import os",  # newline → breaks out of `async def {id}` line
        "has-hyphen",  # not a valid Python/TS identifier
        "has space",
        "1leading_digit",
        "",  # empty
    ],
)
def test_node_id_must_be_a_safe_identifier(bad_id: str) -> None:
    """``id`` is emitted verbatim as a code identifier and a filename, so a
    non-identifier must be rejected at the IR boundary (closes the code-
    injection and path-traversal findings for every adapter at once)."""
    from pydantic import ValidationError

    from rote.ir import Node

    with pytest.raises(ValidationError, match="must be a valid identifier"):
        Node(id=bad_id, kind=NodeKind.PURE_FUNCTION, description="x", impl="extracted/foo.py:bar")


def test_impl_symbol_injection_is_rejected() -> None:
    """The trailing symbol of ``impl`` reaches ``from … import {symbol}`` and a
    call site; a newline-laden payload (the confirmed RCE PoC) must fail."""
    from pydantic import ValidationError

    from rote.ir import Node

    payload = "extracted/x.py:run\n    import os\n    os.system('id')\n    _x = run"
    with pytest.raises(ValidationError, match="must be a valid identifier"):
        Node(id="n", kind=NodeKind.EXTERNAL_CALL, description="x", impl=payload)


@pytest.mark.parametrize(
    "bad_impl",
    [
        "/abs/foo.py:bar",  # absolute path half
        "../foo.py:bar",  # traversal in path half
        "extracted/foo.py",  # missing :symbol
        "extracted/foo-bar.py:baz",  # module name not an identifier
    ],
)
def test_impl_path_half_is_validated(bad_impl: str) -> None:
    from pydantic import ValidationError

    from rote.ir import Node

    with pytest.raises(ValidationError):
        Node(id="n", kind=NodeKind.EXTERNAL_CALL, description="x", impl=bad_impl)


def test_node_id_accepts_normal_snake_case() -> None:
    from rote.ir import Node

    node = Node(
        id="enrich_contact_batch",
        kind=NodeKind.EXTERNAL_CALL,
        description="normal",
        impl="extracted/zoominfo.py:enrich_batch",
    )
    assert node.id == "enrich_contact_batch"


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


def test_llm_judge_accepts_signature_spec_without_path() -> None:
    """The structured signature_spec is a valid alternative to the legacy path."""
    from rote.ir import LLMSignature, Node

    node = Node(
        id="vet",
        kind=NodeKind.LLM_JUDGE,
        description="vet a contact",
        signature_spec=LLMSignature(
            input_schema={"type": "object", "properties": {"x": {"type": "string"}}},
            output_schema={"type": "object", "properties": {"y": {"type": "string"}}},
            prompt="Classify {{ x }}.",
        ),
    )
    assert node.signature is None
    assert node.signature_spec is not None
    assert node.signature_spec.client == "anthropic"  # default


def test_llm_judge_rejects_when_both_signature_forms_missing() -> None:
    from pydantic import ValidationError

    from rote.ir import Node

    with pytest.raises(ValidationError, match="missing required field"):
        Node(
            id="bad",
            kind=NodeKind.LLM_JUDGE,
            description="no signature in any form",
        )


def test_llm_signature_rejects_unknown_client() -> None:
    from pydantic import ValidationError

    from rote.ir import LLMSignature

    with pytest.raises(ValidationError, match="client must be one of"):
        LLMSignature(
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            prompt="x",
            client="cohere",
        )


def test_llm_signature_base_url_accepts_http_urls() -> None:
    from rote.ir import LLMSignature

    sig = LLMSignature(
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        prompt="x",
        client="openai",
        base_url="http://localhost:11434/v1",
    )
    assert sig.base_url == "http://localhost:11434/v1"


@pytest.mark.parametrize(
    "bad_url",
    [
        "ftp://example.com",  # non-http scheme
        "localhost:11434/v1",  # no scheme
        'https://example.com/"); import os; ("',  # quote injection
        "https://example.com/a b",  # whitespace
        "https://example.com\\path",  # backslash
    ],
)
def test_llm_signature_base_url_rejects_unsafe_values(bad_url: str) -> None:
    from pydantic import ValidationError

    from rote.ir import LLMSignature

    with pytest.raises(ValidationError, match="base_url"):
        LLMSignature(
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            prompt="x",
            base_url=bad_url,
        )


def test_llm_signature_temperature_bounded() -> None:
    from pydantic import ValidationError

    from rote.ir import LLMSignature

    with pytest.raises(ValidationError):
        LLMSignature(
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            prompt="x",
            temperature=3.0,
        )


# ───────── Pipeline input schema ─────────


def test_bdr_pipeline_input_schema(bdr_pipeline: Pipeline) -> None:
    """The typed input contract is promoted to input.input_schema."""
    schema = bdr_pipeline.input.input_schema
    assert schema is not None
    assert schema["title"] == "CampaignBrief"
    assert schema["type"] == "object"
    # Schema properties must cover every declared required/optional field.
    declared = set(bdr_pipeline.input.required) | set(bdr_pipeline.input.optional)
    assert declared <= set(schema["properties"].keys())
    assert set(schema["required"]) == set(bdr_pipeline.input.required)


def test_input_schema_is_optional_for_back_compat() -> None:
    """Pipelines without input_schema (pre-v0.3 yaml) still validate."""
    pi = PipelineInput(type="X", required=["a"])
    assert pi.input_schema is None


# ───────── Data-flow input references ─────────


def test_parse_input_ref_accepts_all_four_forms() -> None:
    from rote.ir import parse_input_ref

    r = parse_input_ref("pipeline.input")
    assert r.node_id is None and r.field is None

    r = parse_input_ref("pipeline.input.drug_brand")
    assert r.node_id is None and r.field == "drug_brand"

    r = parse_input_ref("target_research.output")
    assert r.node_id == "target_research" and r.field is None

    r = parse_input_ref("hubspot_upsert.output.upserted")
    assert r.node_id == "hubspot_upsert" and r.field == "upserted"


@pytest.mark.parametrize(
    "bad_ref",
    [
        "",
        "drug_brand",  # bare field name
        "pipeline.inputs",  # wrong keyword
        "pipeline.output",  # node-output ref for a node named 'pipeline' — see below
        "foo.result",  # wrong selector
        "foo.output.a.b",  # deep paths not allowed
        "pipeline.input.a.b",  # deep paths not allowed
        "foo.output[0]",  # no subscripts / expressions
        "len(foo.output)",  # no expressions
    ],
)
def test_parse_input_ref_rejects_bad_syntax(bad_ref: str) -> None:
    from rote.ir import parse_input_ref

    if bad_ref == "pipeline.output":
        # Syntactically a node-output ref for a node named 'pipeline';
        # rejection happens at pipeline validation (unknown node), not parse.
        r = parse_input_ref(bad_ref)
        assert r.node_id == "pipeline"
        return
    with pytest.raises(ValueError, match="Invalid input reference"):
        parse_input_ref(bad_ref)


def test_bdr_pipeline_inputs_bindings(bdr_pipeline: Pipeline) -> None:
    """Spot-check the committed data-flow bindings on the BDR baseline."""
    assert bdr_pipeline.node_by_id("target_research").inputs == {"brief": "pipeline.input"}
    assert bdr_pipeline.node_by_id("hubspot_upsert").inputs == {
        "contacts": "contact_review_gate.output.approved_contacts"
    }
    loop = bdr_pipeline.node_by_id("lead_generation_loop")
    assert loop.inputs is not None
    assert loop.inputs["target_quota"] == "pipeline.input.target_quota"
    assert loop.inputs["intel"] == "target_research.output"


def _one_node_pipeline(inputs: dict[str, str] | None, **input_kwargs) -> Pipeline:  # noqa: ANN003
    from rote.ir import Node

    return Pipeline(
        name="p",
        input=PipelineInput(type="X", **input_kwargs),
        nodes=[
            Node(
                id="a",
                kind=NodeKind.PURE_FUNCTION,
                description="x",
                impl="x.py:y",
            ),
            Node(
                id="b",
                kind=NodeKind.PURE_FUNCTION,
                description="x",
                impl="x.py:y",
                inputs=inputs,
            ),
        ],
        edges=[Edge(**{"from": "a", "to": "b"})],
        entry_nodes=["a"],
        exit_nodes=["b"],
    )


def test_pipeline_rejects_inputs_with_bad_syntax() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="Invalid input reference"):
        _one_node_pipeline({"x": "not a ref"})


def test_pipeline_rejects_inputs_referencing_unknown_node() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="references unknown node"):
        _one_node_pipeline({"x": "nonexistent.output"})


def test_pipeline_rejects_inputs_self_reference() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="references its own output"):
        _one_node_pipeline({"x": "b.output"})


def test_pipeline_rejects_undeclared_pipeline_input_field() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="not declared in the pipeline's input contract"):
        _one_node_pipeline({"x": "pipeline.input.typo_field"}, required=["real_field"])


def test_pipeline_accepts_declared_pipeline_input_field() -> None:
    p = _one_node_pipeline({"x": "pipeline.input.real_field"}, required=["real_field"])
    assert p.node_by_id("b").inputs == {"x": "pipeline.input.real_field"}


def test_pipeline_accepts_schema_declared_pipeline_input_field() -> None:
    """Fields declared only in input_schema.properties also count."""
    p = _one_node_pipeline(
        {"x": "pipeline.input.schema_field"},
        required=["real_field"],
        input_schema={"type": "object", "properties": {"schema_field": {"type": "string"}}},
    )
    assert p.node_by_id("b").inputs is not None


def test_pipeline_skips_field_check_when_contract_empty() -> None:
    """Empty input contract → no field list to check against."""
    p = _one_node_pipeline({"x": "pipeline.input.anything"})
    assert p.node_by_id("b").inputs is not None


def test_pipeline_accepts_valid_upstream_reference() -> None:
    p = _one_node_pipeline({"x": "a.output", "y": "a.output.some_field"})
    assert p.node_by_id("b").inputs == {"x": "a.output", "y": "a.output.some_field"}


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
