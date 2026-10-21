"""Test that the IR can load and validate the BDR pipeline.

This is the first concrete validation that the IR is expressive enough to
represent a real complex skill. The BDR pipeline.yaml is hand-written from
the analysis of the bdr-outreach skill and exercises every node kind and
most optional fields.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from rote.ir import (
    Edge,
    Node,
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


def _pipeline_with(**overrides: object) -> Pipeline:
    """A minimal valid pipeline, with named fields overridden per test."""
    from rote.ir import Node

    node_kwargs: dict[str, object] = dict(
        id="n1", kind=NodeKind.PURE_FUNCTION, description="x", impl="extracted/a.py:go"
    )
    node_kwargs.update({k: v for k, v in overrides.items() if k in {"constants", "tools", "kind"}})
    base: dict[str, object] = dict(
        name="demo",
        version="0.1.0",
        input=PipelineInput(type="In"),
        nodes=[Node(**node_kwargs)],  # type: ignore[arg-type]
        edges=[],
        entry_nodes=["n1"],
        exit_nodes=["n1"],
    )
    base.update({k: v for k, v in overrides.items() if k in {"name", "version"}})
    return Pipeline(**base)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad_name",
    [
        'evil"""\nimport os\nos.system("id")\n"""',  # closes a Python docstring
        "has space",  # space → breaks identifier / class name
        "1leading_digit",  # digit start → invalid emitted class name
        "has.dot",  # '.' survives pascal-case → invalid class name
        "has/slash",  # path separator
        "",  # empty
    ],
)
def test_pipeline_name_must_be_a_safe_identifier(bad_name: str) -> None:
    """``name`` reaches emitted docstrings/literals AND ``_to_pascal_case``
    (class/workflow/queue names), so non-identifier names must be rejected at
    the IR boundary — the confirmed docstring-breakout PoC among them."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="Pipeline name"):
        _pipeline_with(name=bad_name)


def test_pipeline_name_accepts_kebab_case() -> None:
    assert _pipeline_with(name="bdr-outreach").name == "bdr-outreach"


@pytest.mark.parametrize(
    "bad_version",
    [
        '0.1.0"""\nimport os\n"""',  # closes a Python docstring
        "1.0 */ evil() /*",  # closes a TS block comment
        "1.0\nimport os",  # newline injection
        "",  # empty
    ],
)
def test_pipeline_version_charset_is_constrained(bad_version: str) -> None:
    """``version`` reaches emitted docstrings/JSDoc verbatim; a value that can
    close a docstring or block comment must be rejected at the IR boundary."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="Pipeline version"):
        _pipeline_with(version=bad_version)


def test_pipeline_version_accepts_semver() -> None:
    assert _pipeline_with(version="1.2.3-rc.1+build.5").version == "1.2.3-rc.1+build.5"


def test_agent_loop_tool_names_are_charset_constrained() -> None:
    """Tool names splice verbatim into emitted docstrings/JSDoc; the exact
    confirmed TS-injection payload must be rejected at the IR boundary."""
    from pydantic import ValidationError

    from rote.ir import Node, TerminationConfig

    payload = "good_tool */ ; globalThis.__ROTE_INJECTED = true; /*"
    with pytest.raises(ValidationError, match="tool name"):
        Node(
            id="loop1",
            kind=NodeKind.AGENT_LOOP,
            description="x",
            tools=[payload],
            termination=TerminationConfig(condition="done", max_iterations=3),
        )


def test_agent_loop_accepts_normal_tool_names() -> None:
    from rote.ir import Node, TerminationConfig

    node = Node(
        id="loop1",
        kind=NodeKind.AGENT_LOOP,
        description="x",
        tools=["bright_data_search", "salesforce_query", "chrome_cdp_click"],
        termination=TerminationConfig(condition="done", max_iterations=3),
    )
    assert node.tools == ["bright_data_search", "salesforce_query", "chrome_cdp_click"]


@pytest.mark.parametrize(
    "bad_key",
    [
        "x */ evil() /*",  # closes a TS block comment
        'x"""\nimport os\n"""',  # closes a Python docstring
        "has space",
        "1leading",
    ],
)
def test_constant_keys_must_be_identifiers(bad_key: str) -> None:
    """Constant *keys* are emitted verbatim into docstrings/comments; only
    identifier keys are allowed. (Values stay arbitrary — escaped at emit.)"""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="constant key"):
        _pipeline_with(constants={bad_key: 1})


def test_constant_values_stay_arbitrary() -> None:
    """Values are not charset-constrained (they're escaped at emission), so a
    value carrying comment-breakout characters still validates."""
    p = _pipeline_with(constants={"evil": 'a */ x """ b', "batch_size": 10})
    assert p.nodes[0].constants == {"evil": 'a */ x """ b', "batch_size": 10}


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


# ───────── required_mcp_servers (the MCP requirements manifest) ─────────


def test_required_mcp_servers_derived_from_bindings() -> None:
    """The manifest is derived per logical server, node ids sorted."""
    from rote.ir import load_pipeline

    pipeline = load_pipeline(REPO_ROOT / "examples" / "deal-monitor" / "expected" / "pipeline.yaml")
    assert pipeline.required_mcp_servers == {
        "gmail": [
            "fetch_email_thread_content",
            "search_gmail_by_accounts",
            "search_gmail_standard",
        ],
        "slack": ["fetch_intake_messages"],
    }


def test_required_mcp_servers_counts_agent_loop_tool_servers() -> None:
    """BDR has no ``mcp:`` binding anywhere — its MCP usage is entirely in
    two agent loops. Deriving requirements from ``mcp:`` alone therefore
    reported ZERO servers for it, so nothing ever told the user to
    authenticate. ``tool_servers`` is what closes that.
    """
    from rote.ir import load_pipeline

    required = load_pipeline(BDR_PIPELINE).required_mcp_servers
    assert not any(n.mcp for n in load_pipeline(BDR_PIPELINE).nodes)
    assert required == {
        "airweave": ["target_research"],
        "brightdata": ["target_research"],
        "clinicaltrials": ["target_research"],
        "salesforce": ["target_research"],
        "zoominfo": ["lead_generation_loop"],
    }


def test_required_mcp_servers_empty_without_any_mcp_declaration() -> None:
    """Neither a binding nor a resolved loop tool → nothing required.

    An unresolved loop tool contributes nothing on purpose: there is no
    server name to report, and inventing one would be worse than the gap.
    """
    from rote.ir import Pipeline

    pipeline = Pipeline.model_validate(
        {
            "name": "bare",
            "input": {"type": "In", "required": [], "optional": []},
            "nodes": [
                {
                    "id": "loop",
                    "kind": "agent_loop",
                    "description": "Bare tools, no resolution.",
                    "tools": ["web_search"],
                }
            ],
            "edges": [],
        }
    )
    assert pipeline.required_mcp_servers == {}


# ───────── Judge inputs vs signature_spec.input_schema ─────────
#
# Every adapter builds a judge's payload from its `inputs:` keys and
# nothing else, then validates it against the input schema. A required
# property nothing binds (or a bound param a closed schema rejects) fails
# every call at run time; these tests pin that it fails at load instead.

_THREAD_SCHEMA = {
    "type": "object",
    "properties": {"thread_id": {"type": "string"}, "body": {"type": "string"}},
    "required": ["thread_id"],
}


def _judge_pipeline(
    input_schema: dict,
    inputs: dict[str, str] | None,
    *,
    fan_out: bool = False,
    loop_body: bool = False,
) -> Pipeline:
    """``source`` (produces the list) → ``judge``; optional parent loop."""
    nodes: list[dict] = [
        {"id": "source", "kind": "pure_function", "description": "x", "impl": "x.py:y"},
        {
            "id": "judge",
            "kind": "llm_judge",
            "description": "Classify a thread.",
            "fan_out": fan_out,
            "signature_spec": {
                "input_schema": input_schema,
                "output_schema": {"type": "object", "properties": {"step": {"type": "integer"}}},
                "prompt": "Classify.",
            },
            **({"inputs": inputs} if inputs is not None else {}),
        },
    ]
    if loop_body:
        nodes.append(
            {
                "id": "loop",
                "kind": "agent_loop",
                "description": "Drives the judge.",
                "tools": ["search"],
                "loop_body": ["judge"],
            }
        )
    return Pipeline.model_validate(
        {
            "name": "judge-bindings",
            "input": {"type": "In", "required": [], "optional": ["account_name", "rubric"]},
            "nodes": nodes,
            "edges": [{"from": "source", "to": "judge", "fan_out": fan_out}],
        }
    )


def test_judge_rejects_required_schema_property_that_inputs_do_not_bind() -> None:
    from pydantic import ValidationError

    schema = {
        "type": "object",
        "title": "ClassifyInput",
        "properties": {"thread": _THREAD_SCHEMA, "rubric": {"type": "string"}},
        "required": ["thread", "rubric"],
    }
    with pytest.raises(ValidationError) as exc:
        _judge_pipeline(schema, {"thread": "source.output.threads"})
    message = str(exc.value)
    # Names the node, the param, and the schema; the bound param is not missing.
    assert "Node 'judge' (llm_judge)" in message
    assert "requires ['rubric'], which" in message
    assert "signature_spec.input_schema (ClassifyInput)" in message


def test_judge_with_no_inputs_at_all_is_rejected_when_schema_requires_fields() -> None:
    from pydantic import ValidationError

    schema = {"type": "object", "properties": {"thread": _THREAD_SCHEMA}, "required": ["thread"]}
    with pytest.raises(ValidationError, match=r"nothing bound.*requires \['thread'\]"):
        _judge_pipeline(schema, None)


def test_judge_rejects_bound_param_a_closed_schema_does_not_declare() -> None:
    from pydantic import ValidationError

    schema = {
        "type": "object",
        "properties": {"thread": _THREAD_SCHEMA},
        "required": ["thread"],
        "additionalProperties": False,
    }
    with pytest.raises(ValidationError) as exc:
        _judge_pipeline(
            schema, {"thread": "source.output.threads", "extra": "pipeline.input.rubric"}
        )
    message = str(exc.value)
    assert "binds ['extra'], which" in message
    assert "which it does not declare (properties: ['thread'])" in message
    assert "requires" not in message  # nothing required is missing


def test_judge_rejects_undeclared_bound_param_even_when_schema_is_open() -> None:
    """An open schema doesn't make an extra binding harmless: Python drops
    the value silently, and before the Zod emitter honored
    ``additionalProperties`` the TS runtimes rejected every call. Either
    way it is a binding the judge's contract cannot name."""
    from pydantic import ValidationError

    schema = {"type": "object", "properties": {"thread": _THREAD_SCHEMA}, "required": ["thread"]}
    with pytest.raises(ValidationError, match=r"binds \['extra'\], which it does not declare"):
        _judge_pipeline(
            schema, {"thread": "source.output.threads", "extra": "pipeline.input.rubric"}
        )
    # The negative half: binding exactly the declared properties passes.
    ok = _judge_pipeline(schema, {"thread": "source.output.threads"})
    assert ok.node_by_id("judge").inputs == {"thread": "source.output.threads"}


def test_fanned_judge_with_a_sibling_per_element_field_is_rejected_naming_the_element() -> None:
    """The deal-monitor defect: the fan binds only `thread`, the schema also
    requires `account_name` beside it, and the emitter builds
    ``{"thread": _item}``, so every fanned call failed validation."""
    from pydantic import ValidationError

    schema = {
        "type": "object",
        "properties": {"thread": _THREAD_SCHEMA, "account_name": {"type": "string"}},
        "required": ["thread", "account_name"],
        "additionalProperties": False,
    }
    with pytest.raises(ValidationError) as exc:
        _judge_pipeline(schema, {"thread": "source.output.matched_threads"}, fan_out=True)
    message = str(exc.value)
    assert "requires ['account_name'], which" in message
    # The element is identified by the same rule the emitters use.
    assert "'thread' is the element (one item of source.output.matched_threads per call)" in (
        message
    )


def test_fanned_judge_taking_the_whole_element_is_accepted() -> None:
    """The fix: the element property describes one whole element (through a
    ``$defs`` reference), and a shared value binds beside it."""
    schema = {
        "type": "object",
        "properties": {
            "thread": {"$ref": "#/$defs/MatchedThread"},
            "rubric": {"type": "string"},
        },
        "required": ["thread", "rubric"],
        "additionalProperties": False,
        "$defs": {
            "MatchedThread": {
                **_THREAD_SCHEMA,
                "properties": {
                    **_THREAD_SCHEMA["properties"],
                    "account_name": {"type": "string"},
                },
            }
        },
    }
    pipeline = _judge_pipeline(
        schema,
        {"thread": "source.output.matched_threads", "rubric": "pipeline.input.rubric"},
        fan_out=True,
    )
    assert pipeline.node_by_id("judge").fan_out


def test_judge_schema_with_a_root_ref_is_checked_through_the_ref() -> None:
    """A root ``$ref`` to ``$defs`` is a valid input schema; the contract is
    the definition it names, not the property-less wrapper."""
    from pydantic import ValidationError

    schema = {
        "$ref": "#/$defs/ClassifyInput",
        "$defs": {
            "ClassifyInput": {
                "type": "object",
                "properties": {"thread": _THREAD_SCHEMA},
                "required": ["thread"],
                "additionalProperties": False,
            }
        },
    }
    with pytest.raises(ValidationError, match=r"requires \['thread'\]"):
        _judge_pipeline(schema, {})
    with pytest.raises(ValidationError, match=r"binds \['other'\]"):
        _judge_pipeline(schema, {"thread": "source.output", "other": "source.output"})
    assert _judge_pipeline(schema, {"thread": "source.output"}).node_by_id("judge")


def test_judge_schema_with_an_unresolvable_root_ref_is_rejected() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="names no definition"):
        _judge_pipeline({"$ref": "#/$defs/Missing"}, {"thread": "source.output"})


def test_loop_body_judge_is_exempt_because_the_loop_supplies_its_arguments() -> None:
    """A loop-body judge is called by its parent agent loop with model-chosen
    arguments; its ``inputs:`` describe rather than bind."""
    from pydantic import ValidationError

    schema = {"type": "object", "properties": {"thread": _THREAD_SCHEMA}, "required": ["thread"]}
    pipeline = _judge_pipeline(schema, None, loop_body=True)
    assert pipeline.node_by_id("loop").loop_body == ["judge"]
    # The same judge outside a loop is rejected: the exemption is what lets
    # it through, not a lenient schema.
    with pytest.raises(ValidationError, match="nothing bound"):
        _judge_pipeline(schema, None)


def test_legacy_signature_only_judge_is_not_checked() -> None:
    """The legacy ``signature:`` path points at a Python class the IR cannot
    see, so there is no schema to check bindings against."""
    pipeline = Pipeline.model_validate(
        {
            "name": "legacy",
            "input": {"type": "In", "required": []},
            "nodes": [
                {
                    "id": "judge",
                    "kind": "llm_judge",
                    "description": "x",
                    "signature": "signatures/judge.py:Judge",
                }
            ],
            "edges": [],
        }
    )
    assert pipeline.node_by_id("judge").signature_spec is None


_COMMITTED_PIPELINES = sorted(
    str(p.relative_to(REPO_ROOT))
    for p in (REPO_ROOT / "examples").rglob("pipeline.yaml")
    if "runtimes" not in p.parts
)


def test_committed_pipeline_sweep_is_not_degenerate() -> None:
    """Every example has an expected IR plus historical runs; a glob that
    found only a handful would quietly stop guarding the snapshots."""
    assert len(_COMMITTED_PIPELINES) >= 14


@pytest.mark.parametrize("path", _COMMITTED_PIPELINES)
def test_every_committed_pipeline_validates(path: str) -> None:
    """Every IR the repo ships (expected baselines and historical compile
    snapshots) must load: a snapshot the validator rejects is one whose
    judges would fail on their first real call."""
    from rote.ir import load_pipeline

    assert load_pipeline(REPO_ROOT / path).nodes


@pytest.mark.parametrize("path", [p for p in _COMMITTED_PIPELINES if "/expected/" in p])
def test_expected_irs_never_set_judge_temperature(path: str) -> None:
    """Current Claude models reject `temperature` outright, so a baseline IR
    that sets it ships judges that 400 on every call (see the CLAUDE.md
    gotcha). Historical runs/ snapshots are left as compiled."""
    from rote.ir import load_pipeline

    offenders = [
        n.id
        for n in load_pipeline(REPO_ROOT / path).nodes
        if n.signature_spec is not None and n.signature_spec.temperature is not None
    ]
    assert offenders == []


# ───────── Input reachability: valid IR is emittable IR ─────────


def _bdr_raw() -> dict[str, Any]:
    import yaml

    raw: dict[str, Any] = yaml.safe_load(BDR_PIPELINE.read_text(encoding="utf-8"))
    return raw


def _node(raw: dict[str, Any], node_id: str) -> dict[str, Any]:
    return next(n for n in raw["nodes"] if n["id"] == node_id)


def _bdr_with_email_loop(*, loop_takes_its_body: bool) -> dict[str, Any]:
    """The shape a real bdr-outreach compile produced ($4.46, then exit 1 at
    emit): ``personalize_email`` moved into an ``email_template_loop``'s
    loop_body, and the loop ALSO bound it as an input."""
    raw = _bdr_raw()
    loop: dict[str, Any] = {
        "id": "email_template_loop",
        "kind": "agent_loop",
        "description": "Draft one personalized email per contact.",
        "tools": ["hubspot_create_template"],
        "loop_body": ["personalize_email"],
        "termination": {"condition": "every contact has a draft", "max_iterations": 5},
        "inputs": {"contacts": "exclusion_check_sequence.output.passed"},
    }
    if loop_takes_its_body:
        loop["inputs"]["personalized_emails"] = "personalize_email.output"
    raw["nodes"].append(loop)
    _node(raw, "create_sales_template")["inputs"]["personalizations"] = "email_template_loop.output"
    raw["edges"] += [
        {"from": "exclusion_check_sequence", "to": "email_template_loop"},
        {"from": "email_template_loop", "to": "create_sales_template"},
    ]
    return raw


def test_a_loop_cannot_take_its_own_body_sub_node_as_an_input() -> None:
    with pytest.raises(ValidationError) as exc:
        Pipeline.model_validate(_bdr_with_email_loop(loop_takes_its_body=True))
    message = str(exc.value)
    assert "'email_template_loop' input 'personalized_emails'" in message
    assert "in 'email_template_loop''s own loop_body" in message
    assert "Remove this input" in message


def test_the_same_loop_without_that_input_is_valid_and_emits(tmp_path: Path) -> None:
    """The positive control: the repair the error asks for is enough."""
    from rote.adapters import get_adapter

    pipeline = Pipeline.model_validate(_bdr_with_email_loop(loop_takes_its_body=False))
    get_adapter("dbos").emit(pipeline, tmp_path / "out")
    assert (tmp_path / "out" / "main.py").is_file()


def test_another_loops_body_sub_node_points_at_the_loop() -> None:
    raw = _bdr_raw()
    _node(raw, "hubspot_upsert")["inputs"] = {"contacts": "vet_contact.output"}
    with pytest.raises(ValidationError, match="loop_body sub-node of 'lead_generation_loop'"):
        Pipeline.model_validate(raw)


def test_a_forward_reference_names_the_missing_ordering() -> None:
    raw = _bdr_raw()
    _node(raw, "target_research")["inputs"]["late"] = "pre_enrollment_report.output"
    with pytest.raises(ValidationError) as exc:
        Pipeline.model_validate(raw)
    assert "'pre_enrollment_report' has no result available" in str(exc.value)
    assert "does not run before 'target_research'" in str(exc.value)


def test_a_same_wave_reference_is_a_forward_reference() -> None:
    """target_research and taxonomy_lookup share wave 1: neither result exists
    when the other starts."""
    raw = _bdr_raw()
    _node(raw, "taxonomy_lookup").setdefault("inputs", {})["intel"] = "target_research.output"
    with pytest.raises(ValidationError, match="does not run before 'taxonomy_lookup'"):
        Pipeline.model_validate(raw)


def test_every_problem_is_reported_at_once() -> None:
    raw = _bdr_with_email_loop(loop_takes_its_body=True)
    _node(raw, "hubspot_upsert")["inputs"] = {"contacts": "vet_contact.output"}
    message = str(pytest.raises(ValidationError, Pipeline.model_validate, raw).value)
    assert "personalized_emails" in message and "vet_contact" in message


def test_a_loop_body_sub_nodes_own_inputs_describe_and_are_not_checked() -> None:
    raw = _bdr_raw()
    _node(raw, "vet_contact").setdefault("inputs", {})["later"] = "pre_enrollment_report.output"
    Pipeline.model_validate(raw)


def test_a_cycle_is_a_validation_error() -> None:
    raw = _bdr_raw()
    raw["edges"].append({"from": "pre_enrollment_report", "to": "target_research"})
    with pytest.raises(ValidationError, match="Cycle detected"):
        Pipeline.model_validate(raw)


# ───────── mcp.args literals ─────────


def _mcp_node(args: dict[str, Any]) -> Node:
    return Node.model_validate(
        {
            "id": "search",
            "kind": "external_call",
            "description": "Search.",
            "mcp": {"server": "zoominfo", "tool": "search_contacts", "args": args},
        }
    )


@pytest.mark.parametrize(
    "value", ["management-levels", "BDR do not contact", 3, 2.5, True, False, "", "3"]
)
def test_mcp_args_accept_a_typed_literal(value: Any) -> None:
    from rote.ir import MCPLiteral

    node = _mcp_node({"level": {"literal": value}})
    assert node.mcp is not None and node.mcp.args is not None
    literal = node.mcp.args["level"]
    assert isinstance(literal, MCPLiteral)
    # No coercion: "3" stays a string, True stays a boolean, 3 an int.
    assert literal.literal == value
    assert type(literal.literal) is type(value)


@pytest.mark.parametrize(
    "value",
    [None, ["a", "b"], {"k": "v"}, float("nan"), float("inf")],
    ids=["null", "list", "object", "nan", "inf"],
)
def test_mcp_args_reject_a_literal_that_is_not_a_scalar(value: Any) -> None:
    with pytest.raises(ValueError):
        _mcp_node({"level": {"literal": value}})


def test_a_literal_is_only_the_literal_key() -> None:
    with pytest.raises(ValueError):
        _mcp_node({"level": {"literal": "x", "also": 1}})
    with pytest.raises(ValueError):
        _mcp_node({"level": {"value": "x"}})


def test_a_bare_string_is_always_a_reference_and_says_how_to_write_a_literal() -> None:
    """Literal text where a reference goes was the real compile failure;
    the error now names the fix."""
    with pytest.raises(ValueError, match=r"write \{literal: 'BDR do not contact'\}"):
        _mcp_node({"list_name": "BDR do not contact"})
    # A reference still validates as one.
    node = _mcp_node({"q": "query"})
    assert node.mcp is not None and node.mcp.args == {"q": "query"}


def _loop_with(sub: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": "loop",
        "input": {"type": "In", "required": []},
        "nodes": [
            {
                "id": "loop",
                "kind": "agent_loop",
                "description": "x",
                "tools": ["search"],
                "loop_body": ["sub"],
            },
            {"id": "sub", "description": "x", **sub},
        ],
        "edges": [],
    }


@pytest.mark.parametrize(
    ("sub", "message"),
    [
        (
            {"kind": "hitl_gate", "signal": "ok"},
            "Node 'loop' loop_body has 'sub', a hitl_gate: a loop calls only pure_function, "
            "external_call and llm_judge nodes per iteration",
        ),
        (
            {"kind": "pure_function", "impl": "extracted/sub.py:sub", "mandatory": True},
            "Node 'loop' loop_body has 'sub', marked mandatory: the loop's agent decides "
            "whether to call it, so it cannot be mandatory",
        ),
    ],
)
def test_a_loop_body_holds_only_optional_callable_steps(sub: dict[str, Any], message: str) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match=re.escape(message)):
        Pipeline.model_validate(_loop_with(sub))
    Pipeline.model_validate(_loop_with({"kind": "pure_function", "impl": "extracted/sub.py:sub"}))
