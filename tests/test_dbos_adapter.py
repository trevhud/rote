"""Tests for the DBOS adapter.

Three levels of validation:

1. **Emission** — adapter produces files that parse as valid Python and
   satisfy the architectural invariants (MCP-free, retry mapping, durable
   HITL receives).
2. **Registry** — the adapter is dispatchable by name from the CLI.
3. **Execution** — the emitted app actually runs on a real DBOS runtime
   (SQLite system database). See ``test_dbos_e2e.py`` for that.

No ``dbos`` import happens here: emission is pure template substitution,
so the fast suite stays runnable in environments without the extra.
"""

from __future__ import annotations

import ast
import copy
from pathlib import Path

import pytest
import yaml

from rote.adapters import get_adapter
from rote.adapters._py_common import _pascal_ident
from rote.adapters.dbos import (
    DbosAdapter,
    DbosAdapterConfig,
    _duration_to_seconds,
    emit_main,
    emit_signature_module,
)
from rote.ir import Edge, Node, NodeKind, Pipeline, load_pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
BDR_PIPELINE_YAML = REPO_ROOT / "examples" / "bdr-outreach" / "expected" / "pipeline.yaml"
BDR_EMIT_DIR = REPO_ROOT / "examples" / "bdr-outreach" / "expected" / "runtimes" / "dbos"


# ───────── Fixtures ─────────


@pytest.fixture(scope="module")
def bdr_pipeline() -> Pipeline:
    return load_pipeline(BDR_PIPELINE_YAML)


@pytest.fixture(scope="module")
def adapter() -> DbosAdapter:
    return DbosAdapter()


@pytest.fixture(scope="module")
def emit_result(adapter: DbosAdapter, bdr_pipeline: Pipeline) -> dict[str, Path]:
    """Emit into the expected runtimes dir once per module (committed snapshot)."""
    return adapter.emit(bdr_pipeline, BDR_EMIT_DIR)


@pytest.fixture(scope="module")
def main_src(emit_result: dict[str, Path]) -> str:
    return emit_result["main"].read_text(encoding="utf-8")


# ───────── Helper / internal tests ─────────


def test_duration_to_seconds() -> None:
    assert _duration_to_seconds("30s") == 30
    assert _duration_to_seconds("5m") == 300
    assert _duration_to_seconds("7d") == 604800
    assert _duration_to_seconds("500ms") == 0.5
    assert _duration_to_seconds("2h") == 7200
    with pytest.raises(ValueError, match="cannot parse IR duration"):
        _duration_to_seconds("soon")


def test_pascal_ident_preserves_interior_capitalization() -> None:
    """'EmploymentEntry' must not be mangled to 'Employmententry'."""
    assert _pascal_ident("EmploymentEntry") == "EmploymentEntry"
    assert _pascal_ident("snake_case_name") == "SnakeCaseName"
    assert _pascal_ident("with space") == "WithSpace"
    assert _pascal_ident("3d_thing") == "Model3dThing"


def test_registry_dispatches_dbos() -> None:
    adapter = get_adapter("dbos")
    assert isinstance(adapter, DbosAdapter)


# ───────── Emission tests ─────────


def test_emit_produces_expected_files(emit_result: dict[str, Path]) -> None:
    assert emit_result["main"].exists()
    assert emit_result["dbos-config"].exists()
    assert emit_result["README"].exists()
    # Impl-bearing nodes grouped by module; agent loops get their own module.
    for key in (
        "extracted/taxonomy",
        "extracted/zoominfo",
        "extracted/hubspot",
        "extracted/exclusion_checks",
        "extracted/report",
        "extracted/target_research",
        "extracted/lead_generation_loop",
    ):
        assert emit_result[key].exists(), f"missing {key}"
    # Both BDR judges carry signature_spec → generated modules.
    assert emit_result["signatures/vet_contact"].exists()
    assert emit_result["signatures/personalize_email"].exists()


def test_emitted_files_are_valid_python(emit_result: dict[str, Path]) -> None:
    for label, path in emit_result.items():
        if path.suffix != ".py":
            continue
        src = path.read_text(encoding="utf-8")
        try:
            ast.parse(src)
        except SyntaxError as e:
            pytest.fail(f"{label} ({path.name}) is not valid Python: {e}")


def test_every_non_hitl_node_has_a_step(main_src: str, bdr_pipeline: Pipeline) -> None:
    """Every non-HITL node (including loop_body sub-nodes) gets a @DBOS.step."""
    for node in bdr_pipeline.nodes:
        if node.kind is NodeKind.HITL_GATE:
            continue
        assert f'@DBOS.step(name="{node.id}"' in main_src, f"missing step for {node.id}"
        assert f"\ndef {node.id}(payload: dict) -> dict:" in main_src


def test_workflow_name_is_versioned_by_hash(main_src: str) -> None:
    """Regenerated pipelines must register as a new workflow name so DBOS
    recovery never replays new code against old checkpoints."""
    import re as _re

    m = _re.search(r'@DBOS\.workflow\(name="(BdrCampaign_[0-9a-f]{8})"\)', main_src)
    assert m, "workflow name should be BdrCampaign_<hash8>"


def test_hitl_gates_emit_durable_recv(main_src: str, bdr_pipeline: Pipeline) -> None:
    """Each hitl_gate becomes a DBOS.recv on the IR signal topic with the
    IR timeout converted to seconds, and a hard failure on timeout."""
    # contact_review_gate: timeout 7d
    assert 'topic="contact_review_approved"' in main_src
    assert "timeout_seconds=604800" in main_src
    # manual_enrollment_handoff: timeout 14d
    assert 'topic="bdr_enrollment_complete"' in main_src
    assert "timeout_seconds=1209600" in main_src
    # Timeouts fail loudly — silence is not approval.
    for gate in bdr_pipeline.nodes_by_kind(NodeKind.HITL_GATE):
        assert f"if {gate.id}_result is None:" in main_src
    assert "raise TimeoutError(" in main_src


def test_retry_policy_mapping(main_src: str) -> None:
    """IR RetryPolicy maps onto DBOS step retry parameters.

    enrich_contact_batch: max=5 exponential → max_attempts=6 (DBOS counts
    the initial attempt), backoff_rate=2.0. target_research: max=2 →
    max_attempts=3. Nodes without retry get a bare name-only decorator.
    """
    assert (
        '@DBOS.step(name="enrich_contact_batch", retries_allowed=True, '
        "max_attempts=6, backoff_rate=2.0)" in main_src
    )
    assert (
        '@DBOS.step(name="target_research", retries_allowed=True, '
        "max_attempts=3, backoff_rate=2.0)" in main_src
    )
    assert '@DBOS.step(name="taxonomy_lookup")' in main_src


def test_retry_on_categories_surface_as_comment(main_src: str) -> None:
    """DBOS retries any exception; the IR's retry_on categories can't map
    onto a config knob, so they must at least survive as guidance."""
    assert "retry_on categories from the IR: rate_limit, network" in main_src
    assert "should_retry" in main_src


def test_parallel_wave_uses_queue_fan_out(main_src: str) -> None:
    """Wave 1 (target_research ∥ taxonomy_lookup) must enqueue both nodes
    before joining either handle — otherwise the wave serializes."""
    enq_a = main_src.index("target_research_handle = queue.enqueue(")
    enq_b = main_src.index("taxonomy_lookup_handle = queue.enqueue(")
    join_a = main_src.index("target_research_result = target_research_handle.get_result()")
    join_b = main_src.index("taxonomy_lookup_result = taxonomy_lookup_handle.get_result()")
    assert max(enq_a, enq_b) < min(join_a, join_b)


def test_single_node_waves_call_step_directly(main_src: str) -> None:
    assert "lead_generation_loop_result = lead_generation_loop(" in main_src
    assert "queue.enqueue(\n        lead_generation_loop" not in main_src
    assert "queue.enqueue(lead_generation_loop" not in main_src


def test_loop_body_nodes_excluded_from_workflow(main_src: str) -> None:
    """Loop-body sub-nodes exist as steps (testable in isolation) but the
    workflow never dispatches them — the parent loop does."""
    workflow_body = main_src.split("def run_pipeline(", 1)[1]
    assert "enrich_contact_batch(" not in workflow_body
    assert "vet_contact(" not in workflow_body


def test_workflow_returns_exit_nodes(main_src: str, bdr_pipeline: Pipeline) -> None:
    for exit_id in bdr_pipeline.exit_nodes:
        assert f'"{exit_id}": {exit_id}_result,' in main_src


def test_mandatory_nodes_marked_unconditional(main_src: str) -> None:
    assert main_src.count("MANDATORY: this node was marked mandatory") == 3


# ───────── Data-flow threading ─────────


def test_workflow_binds_pipeline_input(main_src: str) -> None:
    """The workflow entrypoint takes the pipeline input as its argument."""
    assert "def run_pipeline(pipeline_input: dict) -> dict:" in main_src


def test_entry_nodes_receive_pipeline_input(main_src: str) -> None:
    """Entry nodes bound to `pipeline.input` get the run argument, not {}."""
    # Both entry nodes run in the queue fan-out wave and take the whole brief.
    assert '"brief": pipeline_input,' in main_src


def test_downstream_nodes_receive_upstream_results(main_src: str) -> None:
    """The committed BDR bindings appear as real payload expressions."""
    # HITL gate resume payload (DBOS.recv result) → downstream step payload.
    assert '"contacts": contact_review_gate_result["approved_contacts"],' in main_src
    # Node output field chains through the exclusion checks.
    assert '"contacts": hubspot_upsert_result["upserted"],' in main_src
    assert '"contacts": exclusion_check_dnc_result["passed"],' in main_src
    # Pipeline input field selection.
    assert '"target_quota": pipeline_input["target_quota"],' in main_src
    # Fan-in: the report node pulls from several sources.
    assert '"template_ids": create_sales_template_result["template_ids"],' in main_src
    # The placeholder TODO is gone for good.
    assert "TODO: pass real payload" not in main_src


def test_emitted_workflow_payloads_parse_as_dict_literals(
    main_src: str, bdr_pipeline: Pipeline
) -> None:
    """AST-level check: every step dispatch inside run_pipeline passes a
    dict literal whose keys match the node's declared ``inputs`` bindings."""
    tree = ast.parse(main_src)
    run_fn = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "run_pipeline"
    )

    node_ids = {n.id for n in bdr_pipeline.nodes}
    payload_keys_by_node: dict[str, set[str]] = {}
    for call in ast.walk(run_fn):
        if not isinstance(call, ast.Call):
            continue
        func = call.func
        if isinstance(func, ast.Attribute) and func.attr == "enqueue":
            # queue.enqueue(<step>, <payload>)
            assert len(call.args) == 2, "enqueue must receive (step, payload)"
            name_arg, payload_arg = call.args
            assert isinstance(name_arg, ast.Name)
            step_name = name_arg.id
        elif isinstance(func, ast.Name) and func.id in node_ids:
            # <step>(<payload>)
            assert len(call.args) == 1, f"step call {func.id} must receive (payload)"
            step_name = func.id
            payload_arg = call.args[0]
        else:
            continue
        assert isinstance(payload_arg, ast.Dict), (
            f"payload for {step_name!r} must be a dict literal"
        )
        payload_keys_by_node[step_name] = {
            k.value for k in payload_arg.keys if isinstance(k, ast.Constant)
        }

    nested_ids = {sub for n in bdr_pipeline.nodes if n.loop_body for sub in n.loop_body}
    for node in bdr_pipeline.nodes:
        if node.kind is NodeKind.HITL_GATE or node.id in nested_ids:
            continue
        expected_keys = set(node.inputs.keys()) if node.inputs else set()
        assert payload_keys_by_node[node.id] == expected_keys, (
            f"payload keys for {node.id!r} don't match its inputs bindings"
        )


def test_node_without_inputs_gets_empty_payload() -> None:
    """Back-compat: nodes with no ``inputs`` still receive {}."""
    node = Node(id="only", kind=NodeKind.PURE_FUNCTION, description="x", impl="x.py:y")
    src = emit_main(_mini_pipeline(node), DbosAdapterConfig())
    assert "only_result = only({})" in src


def test_emit_rejects_forward_reference() -> None:
    """A node whose inputs reference a later wave must fail at emit time,
    not as a NameError inside a running workflow."""
    pipeline = Pipeline(
        name="forward-ref",
        input={"type": "X", "required": [], "optional": []},
        nodes=[
            Node(
                id="first",
                kind=NodeKind.PURE_FUNCTION,
                description="x",
                impl="x.py:y",
                inputs={"data": "second.output"},  # runs before `second`
            ),
            Node(id="second", kind=NodeKind.PURE_FUNCTION, description="x", impl="x.py:y"),
        ],
        edges=[Edge(**{"from": "first", "to": "second"})],
        entry_nodes=["first"],
        exit_nodes=["second"],
    )
    with pytest.raises(ValueError, match="no result available"):
        emit_main(pipeline, DbosAdapterConfig())


def test_emit_rejects_reference_to_loop_body_node(bdr_pipeline: Pipeline) -> None:
    """Loop-body sub-nodes never bind a top-level result — referencing one
    from a top-level node must fail at emit time."""
    pipeline = copy.deepcopy(bdr_pipeline)
    node = pipeline.node_by_id("hubspot_upsert")
    node.inputs = {"contacts": "vet_contact.output"}  # loop-body sub-node
    with pytest.raises(ValueError, match="no result available"):
        emit_main(pipeline, DbosAdapterConfig())


def test_dbos_config_yaml_is_valid(emit_result: dict[str, Path]) -> None:
    cfg = yaml.safe_load(emit_result["dbos-config"].read_text(encoding="utf-8"))
    assert cfg["name"] == "bdr-campaign"
    # `dbos start` (and DBOS Cloud) must run the long-lived worker mode so
    # externally enqueued runs (rote serve / DBOSClient) get executed.
    assert cfg["runtimeConfig"]["start"] == ["python3 main.py --serve"]


def test_sqlite_default_with_env_override(main_src: str) -> None:
    """Local dev defaults to SQLite; production overrides via env var."""
    assert "DBOS_SYSTEM_DATABASE_URL" in main_src
    assert "sqlite:///" in main_src


# ───────── The MCP invariant ─────────


def _assert_mcp_free(path: Path) -> None:
    """Walk executable statements; comments/docstrings may mention MCP to
    explain the graduation history, but no identifier can reference it."""
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)

    def _check(value: str, context: str) -> None:
        assert "mcp" not in value.lower(), (
            f"{path.name} {context}: {value!r} references forbidden substring 'mcp'"
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                _check(alias.name, f"import at line {node.lineno}")
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                _check(node.module, f"from-import at line {node.lineno}")
            for alias in node.names:
                _check(alias.name, f"from-import name at line {node.lineno}")
        elif isinstance(node, ast.Name):
            _check(node.id, f"name at line {node.lineno}")
        elif isinstance(node, ast.Attribute):
            _check(node.attr, f"attribute at line {node.lineno}")


def test_emitted_files_never_reference_mcp(emit_result: dict[str, Path]) -> None:
    """Architectural invariant: no MCP runtime references anywhere in the
    emitted app — main.py, extracted stubs, or generated signatures."""
    checked = 0
    for path in emit_result.values():
        if path.suffix == ".py":
            _assert_mcp_free(path)
            checked += 1
    assert checked >= 10  # main + __init__s + extracted + signatures


# ───────── Extracted stub conventions ─────────


def test_extracted_stubs_raise_not_implemented(emit_result: dict[str, Path]) -> None:
    for label, path in emit_result.items():
        if not label.startswith("extracted/") or label.endswith("__init__"):
            continue
        src = path.read_text(encoding="utf-8")
        assert "raise NotImplementedError(" in src, f"{label} lost its stub"


def test_agent_loop_stub_documents_tools(emit_result: dict[str, Path]) -> None:
    src = emit_result["extracted/lead_generation_loop"].read_text(encoding="utf-8")
    assert "zoominfo_search_contacts" in src
    assert "enrich_contact_batch" in src  # loop_body sub-node documented


# ───────── Generated signature modules ─────────


def test_signature_module_from_spec(emit_result: dict[str, Path]) -> None:
    src = emit_result["signatures/vet_contact"].read_text(encoding="utf-8")
    # Pydantic models generated from the JSON Schemas
    assert "class VetContactInput(BaseModel):" in src
    assert "class VetContactOutput(BaseModel):" in src
    assert "class EmploymentEntry(BaseModel):" in src  # interior caps preserved
    assert 'VetDecision = Literal["keep", "discard"]' in src
    # Prompt + structured output call (Anthropic tool-use)
    assert "PROMPT = " in src
    assert "OUTPUT_JSON_SCHEMA" in src
    assert 'tool_choice={"type": "tool", "name": "vet_contact"}' in src
    assert "VetContactOutput.model_validate(block.input)" in src
    # Judge class exposes the Temporal-convention forward()
    assert "class VetContact:" in src
    assert "def forward(self, inputs: VetContactInput) -> VetContactOutput:" in src


def test_main_prefers_signature_spec(main_src: str) -> None:
    """BDR judges carry both legacy signature and signature_spec; the DBOS
    step must import the *generated* module, not the legacy path."""
    assert "from signatures.vet_contact import VetContact, VetContactInput" in main_src
    assert "asyncio.run" not in main_src


# ───────── Legacy signature path + openai client (synthetic pipelines) ─────────


def _mini_pipeline(judge: Node) -> Pipeline:
    return Pipeline(
        name="mini",
        input={"type": "In", "required": [], "optional": []},
        nodes=[judge],
        edges=[],
        entry_nodes=[judge.id],
        exit_nodes=[judge.id],
    )


def test_legacy_signature_path_fallback() -> None:
    """A judge with only the legacy path form imports the user-maintained
    module and bridges its async forward with asyncio.run."""
    judge = Node(
        id="grade_essay",
        kind=NodeKind.LLM_JUDGE,
        description="Grade an essay.",
        signature="signatures/grade_essay.py:GradeEssay",
    )
    src = emit_main(_mini_pipeline(judge), DbosAdapterConfig())
    assert "from signatures.grade_essay import (" in src
    assert "asyncio.run(judge.forward(GradeEssayInput(**payload)))" in src


def test_openai_client_signature_module() -> None:
    judge = Node(
        id="grade_essay",
        kind=NodeKind.LLM_JUDGE,
        description="Grade an essay.",
        signature_spec={
            "input_schema": {
                "type": "object",
                "properties": {"essay": {"type": "string"}},
                "required": ["essay"],
            },
            "output_schema": {
                "type": "object",
                "properties": {"grade": {"type": "integer"}},
                "required": ["grade"],
            },
            "prompt": "Grade: {{ essay }}",
            "client": "openai",
            "temperature": 0.2,
        },
    )
    src = emit_signature_module(judge, DbosAdapterConfig())
    assert "import openai" in src
    assert '"type": "json_schema"' in src
    assert "temperature=0.2" in src
    assert "GradeEssayOutput.model_validate(json.loads(content))" in src
    ast.parse(src)


def test_signature_module_model_and_endpoint_are_env_overridable() -> None:
    """The model and endpoint bake in IR defaults but read per-node env
    overrides first, so operators can swap either without a re-emit."""
    judge = Node(
        id="grade_essay",
        kind=NodeKind.LLM_JUDGE,
        description="Grade an essay.",
        signature_spec={
            "input_schema": {
                "type": "object",
                "properties": {"essay": {"type": "string"}},
                "required": ["essay"],
            },
            "output_schema": {
                "type": "object",
                "properties": {"grade": {"type": "integer"}},
                "required": ["grade"],
            },
            "prompt": "Grade: {{ essay }}",
            "client": "openai",
            "model": "gpt-4.1",
            "base_url": "http://localhost:11434/v1",
        },
    )
    src = emit_signature_module(judge, DbosAdapterConfig())
    assert 'MODEL = os.environ.get("ROTE_MODEL_GRADE_ESSAY", "gpt-4.1")' in src
    assert (
        'BASE_URL = os.environ.get("ROTE_BASE_URL_GRADE_ESSAY", "http://localhost:11434/v1")'
    ) in src
    assert "client = openai.OpenAI(base_url=BASE_URL)" in src
    assert "model=MODEL," in src
    ast.parse(src)


def test_signature_module_base_url_defaults_to_vendor_endpoint(
    emit_result: dict[str, Path],
) -> None:
    """Without an IR base_url the emitted constant reads only the env
    override; None falls through to the vendor SDK's default endpoint."""
    src = emit_result["signatures/vet_contact"].read_text(encoding="utf-8")
    assert 'BASE_URL = os.environ.get("ROTE_BASE_URL_VET_CONTACT")' in src
    assert "client = anthropic.Anthropic(base_url=BASE_URL)" in src
    assert 'MODEL = os.environ.get("ROTE_MODEL_VET_CONTACT", ' in src


def test_signature_spec_rejects_conflicting_defs() -> None:
    judge = Node(
        id="j",
        kind=NodeKind.LLM_JUDGE,
        description="x",
        signature_spec={
            "input_schema": {
                "$defs": {"Thing": {"type": "object", "properties": {"a": {"type": "string"}}}},
                "type": "object",
                "properties": {"t": {"$ref": "#/$defs/Thing"}},
            },
            "output_schema": {
                "$defs": {"Thing": {"type": "object", "properties": {"b": {"type": "integer"}}}},
                "type": "object",
                "properties": {"t": {"$ref": "#/$defs/Thing"}},
            },
            "prompt": "p",
        },
    )
    with pytest.raises(ValueError, match="different definitions"):
        emit_signature_module(judge, DbosAdapterConfig())
