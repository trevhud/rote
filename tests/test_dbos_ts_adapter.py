"""Tests for the DBOS TypeScript adapter.

Three levels of validation, mirroring ``test_dbos_adapter.py`` (the
semantic reference) and ``test_cloudflare_adapter.py`` (the TS-emission
reference):

1. **Emission** — adapter produces a directory of files at the expected
   layout, with each non-HITL node registered as a ``DBOS.registerStep``
   and each HITL gate represented by a durable ``DBOS.recv`` call site.
2. **Structural invariants** — text assertions over the emitted
   TypeScript: retry/timeout mapping, wave fan-out via
   ``Promise.allSettled``, data-flow threading, the MCP-free invariant.
3. **Execution** — the emitted app actually runs on a real DBOS TS
   runtime (Docker Postgres). See ``test_dbos_ts_e2e.py`` for that.
"""

from __future__ import annotations

import copy
import json
import re
import shutil
from pathlib import Path

import pytest
import yaml

from rote.adapters import get_adapter
from rote.adapters._common import _to_camel_case
from rote.adapters.dbos_ts import (
    DbosTsAdapter,
    DbosTsAdapterConfig,
    _duration_to_ms,
    emit_main,
    emit_signature_module,
)
from rote.ir import Edge, Node, NodeKind, Pipeline, PipelineInput
from tests._helpers import assert_no_mcp_in_ts, mini_pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
BDR_PIPELINE_YAML = REPO_ROOT / "examples" / "bdr-outreach" / "expected" / "pipeline.yaml"
BDR_EMIT_DIR = REPO_ROOT / "examples" / "bdr-outreach" / "expected" / "runtimes" / "dbos-ts"


# ───────── Fixtures ─────────


@pytest.fixture(scope="module")
def adapter() -> DbosTsAdapter:
    return DbosTsAdapter()


@pytest.fixture(scope="module")
def emit_result(adapter: DbosTsAdapter, bdr_pipeline: Pipeline) -> dict[str, Path]:
    """Emit the BDR pipeline once per module into the committed snapshot dir."""
    if BDR_EMIT_DIR.exists():
        # Remove stale files so renames produce a clean diff.
        shutil.rmtree(BDR_EMIT_DIR)
    return adapter.emit(bdr_pipeline, BDR_EMIT_DIR)


@pytest.fixture(scope="module")
def main_src(emit_result: dict[str, Path]) -> str:
    return emit_result["main"].read_text(encoding="utf-8")


# ───────── Helper / internal tests ─────────


def test_duration_to_ms() -> None:
    assert _duration_to_ms("30s") == 30_000
    assert _duration_to_ms("5m") == 300_000
    assert _duration_to_ms("2h") == 7_200_000
    assert _duration_to_ms("500ms") == 500
    with pytest.raises(ValueError, match="cannot parse IR duration"):
        _duration_to_ms("soon")


def test_registry_dispatches_dbos_ts() -> None:
    adapter = get_adapter("dbos-ts")
    assert isinstance(adapter, DbosTsAdapter)


def test_unknown_runtime_lists_dbos_ts() -> None:
    with pytest.raises(KeyError, match="dbos-ts"):
        get_adapter("nonexistent-runtime")


# ───────── Emission tests ─────────


def test_emit_produces_expected_files(emit_result: dict[str, Path], bdr_pipeline: Pipeline) -> None:
    assert emit_result["main"].exists()
    assert emit_result["package.json"].exists()
    assert emit_result["tsconfig.json"].exists()
    assert emit_result["dbos-config"].exists()
    assert emit_result["README"].exists()

    # One module per non-HITL node, in either signatures/ or extracted/.
    for node in bdr_pipeline.nodes:
        if node.kind is NodeKind.HITL_GATE:
            continue
        if node.kind is NodeKind.LLM_JUDGE:
            assert emit_result[f"signatures/{node.id}"].exists()
        else:
            assert emit_result[f"extracted/{node.id}"].exists()


def test_every_non_hitl_node_registered_as_step(main_src: str, bdr_pipeline: Pipeline) -> None:
    """Every non-HITL node (including loop_body sub-nodes) gets a
    DBOS.registerStep with the node id as the durable step name."""
    for node in bdr_pipeline.nodes:
        if node.kind is NodeKind.HITL_GATE:
            continue
        camel = _to_camel_case(node.id)
        assert f"export const {camel}Step = DBOS.registerStep(" in main_src, (
            f"missing step registration for {node.id}"
        )
        assert f"name: {json.dumps(node.id)}" in main_src, f"missing step name for {node.id}"


def test_workflow_name_is_versioned_by_hash(main_src: str) -> None:
    """Regenerated pipelines must register as a new workflow name so DBOS
    recovery never replays new code against old checkpoints."""
    assert re.search(r'\{ name: "BdrCampaign_[0-9a-f]{8}" \},\n\);', main_src)


def test_hitl_gates_emit_durable_recv(main_src: str, bdr_pipeline: Pipeline) -> None:
    """Each hitl_gate becomes a DBOS.recv on the IR signal topic with the
    IR timeout converted to seconds, and a hard failure on timeout. The
    timeout must be explicit — DBOS.recv defaults to 60s when omitted."""
    # contact_review_gate: timeout 7d
    assert '"contact_review_approved",\n            604800' in main_src
    # manual_enrollment_handoff: timeout 14d
    assert '"bdr_enrollment_complete",\n            1209600' in main_src
    # Timeouts fail loudly — silence is not approval.
    for gate in bdr_pipeline.nodes_by_kind(NodeKind.HITL_GATE):
        assert f"if ({gate.id}_result === null) {{" in main_src
    assert "throw new Error(" in main_src


def test_retry_policy_mapping(main_src: str) -> None:
    """IR RetryPolicy maps onto DBOS TS StepConfig.

    enrich_contact_batch: max=5 exponential + timeout 30s →
    maxAttempts=6 (DBOS counts the initial attempt), backoffRate=2,
    timeoutMS=30000. target_research: max=2 → maxAttempts=3. Nodes
    without retry get a name-only (plus timeout, if any) config.
    """
    assert (
        '{ name: "enrich_contact_batch", retriesAllowed: true, '
        "maxAttempts: 6, backoffRate: 2, timeoutMS: 30000 }" in main_src
    )
    assert (
        '{ name: "target_research", retriesAllowed: true, '
        "maxAttempts: 3, backoffRate: 2, timeoutMS: 300000 }" in main_src
    )
    assert '{ name: "taxonomy_lookup" }' in main_src


def test_ir_timeout_maps_to_timeout_ms(main_src: str) -> None:
    """The TS SDK has a real per-step timeout (timeoutMS) — unlike DBOS
    Python where the IR timeout only survives as a comment."""
    # lead_generation_loop: timeout 15m, no retry.
    assert '{ name: "lead_generation_loop", timeoutMS: 900000 }' in main_src


def test_retry_on_categories_surface_as_comment(main_src: str) -> None:
    """DBOS retries any exception; the IR's retry_on categories can't map
    onto declarative config (the TS knob is a shouldRetry predicate), so
    they must at least survive as guidance."""
    assert "retry_on categories from the IR: rate_limit, network" in main_src
    assert "shouldRetry" in main_src


def test_parallel_wave_uses_promise_all_settled(main_src: str) -> None:
    """Wave 1 (target_research ∥ taxonomy_lookup) must settle both steps
    concurrently — Promise.allSettled per the DBOS docs, never a bare
    Promise.all (unhandled rejections can crash the Node process)."""
    assert (
        "const [target_research_settled, taxonomy_lookup_settled] = "
        "await Promise.allSettled([" in main_src
    )
    assert "const target_research_result = unwrap(target_research_settled);" in main_src
    assert "const taxonomy_lookup_result = unwrap(taxonomy_lookup_settled);" in main_src
    assert "await Promise.all([" not in main_src


def test_single_node_waves_await_step_directly(main_src: str) -> None:
    assert "const lead_generation_loop_result = await leadGenerationLoopStep(" in main_src
    assert "leadGenerationLoopStep({})" not in main_src


def test_loop_body_nodes_excluded_from_workflow(main_src: str) -> None:
    """Loop-body sub-nodes exist as steps (testable in isolation) but the
    workflow never dispatches them — the parent loop does."""
    workflow_body = main_src.split("DBOS.registerWorkflow(", 1)[1]
    assert "enrichContactBatchStep(" not in workflow_body
    assert "vetContactStep(" not in workflow_body


def test_workflow_returns_exit_nodes(main_src: str, bdr_pipeline: Pipeline) -> None:
    for exit_id in bdr_pipeline.exit_nodes:
        assert f'"{exit_id}": {exit_id}_result as Record<string, unknown>,' in main_src


def test_mandatory_nodes_marked_unconditional(main_src: str) -> None:
    assert main_src.count("MANDATORY: this node was marked mandatory") == 3


def test_entrypoint_launches_and_starts_workflow(main_src: str) -> None:
    """The emitted entrypoint follows the v4 lifecycle: setConfig →
    launch → startWorkflow (curried, function-registration form) →
    shutdown."""
    assert "DBOS.setConfig({" in main_src
    assert "await DBOS.launch();" in main_src
    assert "await DBOS.startWorkflow(runPipeline)(pipelineInput);" in main_src
    assert "await DBOS.shutdown();" in main_src
    assert "require.main === module" in main_src


# ───────── Data-flow threading ─────────


def test_entry_nodes_receive_pipeline_input(main_src: str) -> None:
    """Entry nodes bound to `pipeline.input` get the run argument, not {}."""
    assert "brief: pipelineInput," in main_src


def test_downstream_nodes_receive_upstream_results(main_src: str) -> None:
    """The committed BDR bindings appear as real payload expressions."""
    rec = "as Record<string, unknown>"
    # HITL gate resume payload (DBOS.recv result) → downstream step payload.
    assert f'contacts: (contact_review_gate_result {rec})["approved_contacts"],' in main_src
    # Node output field chains through the exclusion checks.
    assert f'contacts: (hubspot_upsert_result {rec})["upserted"],' in main_src
    assert f'contacts: (exclusion_check_dnc_result {rec})["passed"],' in main_src
    # Pipeline input field selection.
    assert 'target_quota: pipelineInput["target_quota"],' in main_src
    # Fan-in: the report node pulls from several sources.
    assert f'template_ids: (create_sales_template_result {rec})["template_ids"],' in main_src


def test_node_without_inputs_gets_empty_payload() -> None:
    """Back-compat: nodes with no ``inputs`` still receive {}."""
    node = Node(id="only", kind=NodeKind.PURE_FUNCTION, description="x", impl="x.py:y")
    src = emit_main(mini_pipeline(node), DbosTsAdapterConfig())
    assert "const only_result = await onlyStep({});" in src


def test_emit_rejects_forward_reference() -> None:
    """A node whose inputs reference a later wave must fail at emit time,
    not as an undefined-variable error inside a running workflow."""
    pipeline = Pipeline(
        name="forward-ref",
        input=PipelineInput(type="X"),
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
        emit_main(pipeline, DbosTsAdapterConfig())


def test_emit_rejects_reference_to_loop_body_node(bdr_pipeline: Pipeline) -> None:
    """Loop-body sub-nodes never bind a top-level result — referencing one
    from a top-level node must fail at emit time."""
    pipeline = copy.deepcopy(bdr_pipeline)
    node = pipeline.node_by_id("hubspot_upsert")
    node.inputs = {"contacts": "vet_contact.output"}  # loop-body sub-node
    with pytest.raises(ValueError, match="no result available"):
        emit_main(pipeline, DbosTsAdapterConfig())


# ───────── The MCP invariant ─────────


def test_emitted_files_never_reference_mcp(emit_result: dict[str, Path]) -> None:
    """Architectural invariant: no MCP runtime in emitted DBOS TS code.

    Comments, JSDoc, and string literals may mention MCP to explain the
    compilation history; executable code may not. Scan logic is shared —
    see tests/_helpers.py.
    """
    # BDR's two agent loops declare MCP tools, so their modules and the
    # helper that binds them are the pipeline's legitimate MCP sites.
    assert_no_mcp_in_ts(
        emit_result,
        min_files=13,
        expected={
            "extracted/_roteMcp",
            "extracted/target_research",
            "extracted/lead_generation_loop",
        },
    )


# ───────── Extracted stub conventions ─────────


def test_extracted_stubs_throw_not_implemented(emit_result: dict[str, Path]) -> None:
    src = emit_result["extracted/taxonomy_lookup"].read_text(encoding="utf-8")
    assert "throw new Error" in src
    assert "stub not implemented" in src
    assert "Promise<never>" in src


def test_agent_loop_module_runs_a_real_bounded_loop(emit_result: dict[str, Path]) -> None:
    """Not a stub: the emitted module binds the IR's tools and runs the loop.

    A pipeline that hands its one genuinely agentic step back to the user
    has not graduated that step, so this asserts the whole contract —
    bounded iterations, the declared allowlist, the loop_body sub-nodes
    bound as callables, and the provider left to run time.
    """
    src = emit_result["extracted/lead_generation_loop"].read_text(encoding="utf-8")
    assert "runAgentLoop({" in src
    assert "requires an agent runtime" not in src
    # The allowlist IS the boundary — the IR's tool names reach the call.
    assert "zoominfo_search_contacts" in src
    assert "bindAgentTools(" in src
    # Bounded, always: an unbounded loop in a durable workflow burns
    # budget until a human notices.
    assert "maxIterations:" in src
    # loop_body sub-nodes are bound as callables, not re-derived.
    assert "enrich_contact_batch" in src
    # The provider is resolved at run time and never baked into emitted code.
    assert "_roteInference" in src


# ───────── Generated signature modules ─────────


def test_signature_module_imports_zod_and_anthropic(emit_result: dict[str, Path]) -> None:
    src = emit_result["signatures/vet_contact"].read_text(encoding="utf-8")
    assert 'from "zod"' in src
    assert 'from "@anthropic-ai/sdk"' in src
    assert "VetContactInput.parse" in src
    assert "VetContactOutput.parse" in src
    # Regeneration instructions point at this adapter, not Cloudflare.
    assert "rote.adapters.dbos_ts" in src


def test_interpolate_throws_on_unresolvable_placeholder(
    emit_result: dict[str, Path],
) -> None:
    """The shared interpolate helper must throw on a missing prompt
    variable — a hole in a judge prompt produces confident garbage."""
    src = emit_result["signatures/vet_contact"].read_text(encoding="utf-8")
    assert "prompt template references" in src
    assert "throw new Error(" in src


def test_main_calls_judges_with_env(main_src: str) -> None:
    """Judge steps thread API keys explicitly (Node process.env), so the
    signature modules stay runtime-agnostic between Workers and Node."""
    assert 'ANTHROPIC_API_KEY: requireEnv("ANTHROPIC_API_KEY"),' in main_src


def test_main_threads_operator_overrides_to_judges(main_src: str) -> None:
    """The narrow env literal each judge receives must carry the per-node
    override vars — otherwise ROTE_MODEL_* / ROTE_BASE_URL_* would be
    silently ignored on this runtime."""
    assert "ROTE_MODEL_VET_CONTACT: process.env.ROTE_MODEL_VET_CONTACT," in main_src
    assert "ROTE_BASE_URL_VET_CONTACT: process.env.ROTE_BASE_URL_VET_CONTACT," in main_src


def test_emit_rejects_llm_judge_without_signature_spec(tmp_path: Path) -> None:
    """The DBOS TS adapter requires the structured signature_spec — the
    legacy Python-path form cannot be transpiled to TypeScript."""
    judge = Node(
        id="judge",
        kind=NodeKind.LLM_JUDGE,
        description="legacy path only",
        signature="signatures/x.py:X",
    )
    with pytest.raises(ValueError, match="requires\\s+signature_spec"):
        DbosTsAdapter().emit(mini_pipeline(judge), tmp_path / "y")


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
    src = emit_signature_module(judge, DbosTsAdapterConfig())
    assert 'from "openai"' in src
    assert '"json_schema"' in src
    # temperature is a runtime knob, never a baked constant — current
    # Anthropic models 400 on the parameter, so an operator retargeting
    # ROTE_MODEL_<ID> needs a way to drop it without re-emitting.
    assert 'const TEMPERATURE = env.ROTE_TEMPERATURE_GRADE_ESSAY ?? "0.2";' in src
    assert "...(TEMPERATURE.trim() ? { temperature: Number(TEMPERATURE) } : {})," in src
    assert "temperature: 0.2" not in src
    src_main = emit_main(mini_pipeline(judge), DbosTsAdapterConfig())
    assert 'OPENAI_API_KEY: requireEnv("OPENAI_API_KEY")' in src_main


def test_openai_judge_adds_openai_dependency() -> None:
    judge = Node(
        id="grade_essay",
        kind=NodeKind.LLM_JUDGE,
        description="Grade an essay.",
        signature_spec={
            "input_schema": {"type": "object", "properties": {}},
            "output_schema": {"type": "object", "properties": {}},
            "prompt": "p",
            "client": "openai",
        },
    )
    from rote.adapters.dbos_ts import emit_package_json

    pkg = json.loads(emit_package_json(mini_pipeline(judge)))
    assert "openai" in pkg["dependencies"]
    assert "@anthropic-ai/sdk" not in pkg["dependencies"]


# ───────── Project config files ─────────


def test_package_json_has_required_deps(
    emit_result: dict[str, Path], bdr_pipeline: Pipeline
) -> None:
    pkg = json.loads(emit_result["package.json"].read_text(encoding="utf-8"))
    assert pkg["name"] == bdr_pipeline.name
    assert "@dbos-inc/dbos-sdk" in pkg["dependencies"]
    assert "zod" in pkg["dependencies"]
    # BDR judges are all Anthropic — no OpenAI SDK.
    assert "@anthropic-ai/sdk" in pkg["dependencies"]
    assert "openai" not in pkg["dependencies"]
    assert "typescript" in pkg["devDependencies"]
    assert "@types/node" in pkg["devDependencies"]
    # CommonJS: the SDK ships CJS and `require.main === module` needs it.
    assert "type" not in pkg


def test_tsconfig_targets_node(emit_result: dict[str, Path]) -> None:
    cfg = json.loads(emit_result["tsconfig.json"].read_text(encoding="utf-8"))
    assert cfg["compilerOptions"]["module"] == "NodeNext"
    assert cfg["compilerOptions"]["strict"] is True
    assert cfg["compilerOptions"]["outDir"] == "dist"


def test_dbos_config_yaml_is_valid(emit_result: dict[str, Path]) -> None:
    cfg = yaml.safe_load(emit_result["dbos-config"].read_text(encoding="utf-8"))
    assert cfg["name"] == "bdr-campaign"
    assert cfg["language"] == "node"
    assert cfg["runtimeConfig"]["start"] == ["node dist/main.js"]


# ───────── README ─────────


def test_readme_flush_left_with_gates(emit_result: dict[str, Path], bdr_pipeline: Pipeline) -> None:
    """README renders flush-left (the dedent-before-interpolate bug class:
    a multi-line gate table would defeat textwrap.dedent and ship the
    whole file as one Markdown code block) and documents every gate."""
    src = emit_result["README"].read_text(encoding="utf-8")
    assert src.startswith("# bdr-campaign — DBOS (TypeScript) runtime")
    # No line of the README body is uniformly indented boilerplate.
    assert not src.splitlines()[0].startswith(" ")
    gates = bdr_pipeline.nodes_by_kind(NodeKind.HITL_GATE)
    assert len(gates) >= 2
    for gate in gates:
        assert f"| `{gate.id}` | `{gate.signal}` |" in src
    # Resume instructions + timeout semantics.
    assert "DBOSClient" in src
    assert re.search(r"silence\s+is not approval", src)
    # The TS SDK's Postgres-only story is documented.
    assert "npx dbos postgres start" in src
