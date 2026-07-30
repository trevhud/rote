"""Tests for the Cloudflare Workflows adapter.

Three levels of validation, mirroring ``test_temporal_adapter.py``:

1. **Emission** — adapter produces a directory of files at the expected
   layout, with each non-HITL node represented by an extracted/ or
   signatures/ module and each HITL gate represented by a
   ``step.waitForEvent`` call site in the workflow.
2. **Structural invariants** — text/regex assertions over the emitted
   TypeScript: every node has a ``step.do`` (or ``step.waitForEvent``);
   the workflow class extends ``WorkflowEntrypoint``; the pipeline hash
   appears in the file header; the MCP-free architectural invariant
   holds.
3. **TypeScript compilation** — slow, gated on ``tsc`` being available
   via ``npm install``. Marked ``@pytest.mark.slow`` — see
   ``test_cloudflare_e2e.py`` for the deeper miniflare-based integration.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from rote.adapters import get_adapter
from rote.adapters._common import _to_pascal_case
from rote.adapters._ts_common import json_schema_to_zod
from rote.adapters.cloudflare import (
    CloudflareAdapter,
    CloudflareAdapterConfig,
    _ir_duration_to_cf,
    _pipeline_hash,
    _to_camel_case,
    _validate_signal_name,
    emit_workflow,
)
from rote.ir import LLMSignature, Node, NodeKind, Pipeline, RetryPolicy
from tests._helpers import assert_no_mcp_in_ts

REPO_ROOT = Path(__file__).resolve().parent.parent
BDR_PIPELINE_YAML = REPO_ROOT / "examples" / "bdr-outreach" / "expected" / "pipeline.yaml"
BDR_EMIT_DIR = REPO_ROOT / "examples" / "bdr-outreach" / "expected" / "runtimes" / "cloudflare"


# ───────── Fixtures ─────────


@pytest.fixture(scope="module")
def adapter() -> CloudflareAdapter:
    return CloudflareAdapter()


@pytest.fixture(scope="module")
def emit_result(adapter: CloudflareAdapter, bdr_pipeline: Pipeline) -> dict[str, Path]:
    """Emit the BDR pipeline once per module into the committed snapshot dir."""
    if BDR_EMIT_DIR.exists():
        # Remove stale files so renames produce a clean diff.
        shutil.rmtree(BDR_EMIT_DIR)
    return adapter.emit(bdr_pipeline, BDR_EMIT_DIR)


# ───────── Helper / internal tests ─────────


def test_pascal_and_camel_case() -> None:
    assert _to_pascal_case("bdr-campaign") == "BdrCampaign"
    assert _to_pascal_case("vet_contact") == "VetContact"
    assert _to_camel_case("bdr-campaign") == "bdrCampaign"
    assert _to_camel_case("vet_contact") == "vetContact"
    assert _to_camel_case("") == ""


def test_pipeline_hash_is_stable(bdr_pipeline: Pipeline) -> None:
    h1 = _pipeline_hash(bdr_pipeline)
    h2 = _pipeline_hash(bdr_pipeline)
    assert h1 == h2
    assert len(h1) == 8


def test_ir_duration_to_cf_conversion() -> None:
    assert _ir_duration_to_cf("5m") == "5 minutes"
    assert _ir_duration_to_cf("30s") == "30 seconds"
    assert _ir_duration_to_cf("7d") == "7 days"
    assert _ir_duration_to_cf("2h") == "2 hours"
    assert _ir_duration_to_cf("250ms") == "250 milliseconds"
    # Exactly 1 singularizes — emitted code is read by humans, and
    # "1 hours" in a reviewed artifact reads as a bug.
    assert _ir_duration_to_cf("1h") == "1 hour"
    assert _ir_duration_to_cf("1d") == "1 day"
    assert _ir_duration_to_cf("1s") == "1 second"
    # Already human-readable: pass-through.
    assert _ir_duration_to_cf("10 minutes") == "10 minutes"


def test_validate_signal_name_accepts_clean_names() -> None:
    _validate_signal_name("contact_review_approved", "node-id")
    _validate_signal_name("foo-bar_baz", "node-id")
    _validate_signal_name("ABC123", "node-id")


def test_validate_signal_name_rejects_dots_and_spaces() -> None:
    with pytest.raises(ValueError, match="invalid characters"):
        _validate_signal_name("foo.bar", "node-id")
    with pytest.raises(ValueError, match="invalid characters"):
        _validate_signal_name("has space", "node-id")
    with pytest.raises(ValueError, match="invalid characters"):
        _validate_signal_name("has/slash", "node-id")


# ───────── JSON Schema → Zod ─────────


def test_zod_converts_primitives() -> None:
    assert json_schema_to_zod({"type": "string"}) == "z.string()"
    assert json_schema_to_zod({"type": "integer"}) == "z.number().int()"
    assert json_schema_to_zod({"type": "number"}) == "z.number()"
    assert json_schema_to_zod({"type": "boolean"}) == "z.boolean()"


def test_zod_converts_enum() -> None:
    src = json_schema_to_zod({"enum": ["keep", "discard"]})
    assert src == 'z.enum(["keep", "discard"])'


def test_zod_converts_object_with_required_and_optional() -> None:
    src = json_schema_to_zod(
        {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name"],
        }
    )
    assert "z.object({" in src
    assert '"name": z.string()' in src
    assert '"age": z.number().int().optional()' in src
    assert "}).strict()" in src


def test_zod_converts_array() -> None:
    src = json_schema_to_zod({"type": "array", "items": {"type": "string"}})
    assert src == "z.array(z.string())"


def test_zod_converts_nullable_anyof() -> None:
    src = json_schema_to_zod({"anyOf": [{"type": "string"}, {"type": "null"}]})
    assert src == "z.string().nullable()"


def test_zod_resolves_refs() -> None:
    src = json_schema_to_zod(
        {
            "type": "object",
            "properties": {"x": {"$ref": "#/$defs/X"}},
            "required": ["x"],
            "$defs": {
                "X": {"enum": ["a", "b"]},
            },
        }
    )
    assert '"x": z.enum(["a", "b"])' in src


def test_zod_rejects_unknown_ref() -> None:
    with pytest.raises(ValueError, match="Unknown \\$ref target"):
        json_schema_to_zod(
            {
                "type": "object",
                "properties": {"x": {"$ref": "#/$defs/Missing"}},
            }
        )


# ───────── Adapter dispatch ─────────


def test_adapter_registered_in_dispatch_dict() -> None:
    adapter = get_adapter("cloudflare")
    assert isinstance(adapter, CloudflareAdapter)


def test_unknown_runtime_lists_cloudflare() -> None:
    with pytest.raises(KeyError, match="cloudflare"):
        get_adapter("nonexistent-runtime")


# ───────── Emission tests ─────────


def test_emitted_index_is_a_driver_router(emit_result: dict[str, Path]) -> None:
    """The emitted index.ts must expose the routes `rote run` drives.

    /start (create), /healthz (readiness), /status/<id> (poll), and
    /event/<id>/<type> (HITL gate delivery) — the e2e test and the
    cloudflare runner both depend on this surface.
    """
    content = emit_result["index"].read_text(encoding="utf-8")
    assert '"/healthz"' in content
    assert '"/start"' in content
    assert "/status\\/" in content.replace("\\\\", "\\")
    assert "sendEvent" in content
    assert "env.PIPELINE.create" in content
    assert 'req.method === "POST"' in content


def test_emit_produces_expected_files(emit_result: dict[str, Path], bdr_pipeline: Pipeline) -> None:
    assert emit_result["workflow"].exists()
    assert emit_result["index"].exists()
    assert emit_result["wrangler"].exists()
    assert emit_result["package.json"].exists()
    assert emit_result["tsconfig.json"].exists()

    # One module per non-HITL node, in either signatures/ or extracted/.
    for node in bdr_pipeline.nodes:
        if node.kind is NodeKind.HITL_GATE:
            continue
        if node.kind is NodeKind.LLM_JUDGE:
            assert emit_result[f"signatures/{node.id}"].exists()
        else:
            assert emit_result[f"extracted/{node.id}"].exists()


def test_workflow_class_extends_workflow_entrypoint(emit_result: dict[str, Path]) -> None:
    src = emit_result["workflow"].read_text()
    assert "extends WorkflowEntrypoint<Env, Params>" in src


def test_workflow_imports_cloudflare_workers(emit_result: dict[str, Path]) -> None:
    src = emit_result["workflow"].read_text()
    assert 'from "cloudflare:workers"' in src
    assert "WorkflowEntrypoint" in src
    assert "WorkflowEvent" in src
    assert "WorkflowStep" in src


def test_workflow_has_step_do_for_every_non_hitl_node(
    emit_result: dict[str, Path], bdr_pipeline: Pipeline
) -> None:
    """Every non-HITL top-level node should have a ``step.do("<id>", ...)`` call.

    Loop body sub-nodes are excluded — they're orchestrated inside the
    parent agent_loop and don't appear in the top-level workflow body.
    """
    src = emit_result["workflow"].read_text()
    nested_ids: set[str] = set()
    for node in bdr_pipeline.nodes:
        if node.loop_body:
            nested_ids.update(node.loop_body)
    for node in bdr_pipeline.nodes:
        if node.kind is NodeKind.HITL_GATE:
            continue
        if node.id in nested_ids:
            continue
        if node.fan_out:
            # One step per element, so the name is an indexed template
            # literal rather than a constant — asserting the constant
            # form here would silently accept a batch dispatch.
            assert f"`{node.id}[${{_index}}]`" in src, (
                f"Missing per-element step.do call for fan_out node {node.id!r}"
            )
            continue
        # Indentation differs by dispatch form: sequential (12), inside a
        # Promise.all wave (16), and inside a park-on-auth retry loop
        # (20, where the step name is a ternary on the attempt counter).
        assert (
            f'step.do(\n{" " * 12}"{node.id}"' in src
            or f'step.do(\n{" " * 16}"{node.id}"' in src
            or f'? "{node.id}"' in src
        ), f"Missing step.do call for node {node.id!r} in workflow.ts"


def test_workflow_has_wait_for_event_for_every_hitl_gate(
    emit_result: dict[str, Path], bdr_pipeline: Pipeline
) -> None:
    src = emit_result["workflow"].read_text()
    for node in bdr_pipeline.nodes:
        if node.kind is not NodeKind.HITL_GATE:
            continue
        assert node.signal is not None
        assert "step.waitForEvent" in src
        assert f'"{node.id}"' in src
        assert f'type: "{node.signal}"' in src


def test_workflow_header_includes_pipeline_hash(
    emit_result: dict[str, Path], bdr_pipeline: Pipeline
) -> None:
    src = emit_result["workflow"].read_text()
    expected_hash = _pipeline_hash(bdr_pipeline)
    assert f"Pipeline hash: {expected_hash}" in src


def test_emitted_files_never_reference_mcp(emit_result: dict[str, Path]) -> None:
    """Architectural invariant: no MCP runtime in emitted Cloudflare code.

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
            # The workflow imports isRoteMcpAuthNeeded to decide whether a
            # failed step should park — Cloudflare has no should-retry
            # predicate, so the check lives at the call site.
            "workflow",
        },
    )


def test_signature_module_imports_zod_and_anthropic(
    emit_result: dict[str, Path],
) -> None:
    """LLM-judge signature modules use Zod + Anthropic SDK (not BAML)."""
    src = emit_result["signatures/vet_contact"].read_text()
    assert 'from "zod"' in src
    assert 'from "@anthropic-ai/sdk"' in src
    assert "VetContactInput.parse" in src
    assert "VetContactOutput.parse" in src
    # No BAML — that's the whole point of this adapter's signature shape.
    assert "@boundaryml/baml" not in src


def test_signature_module_is_operator_overridable(emit_result: dict[str, Path]) -> None:
    """Model and endpoint stay swappable at runtime — per-node env vars
    beat the emitted defaults, so retargeting a judge (different model,
    OpenAI-compatible server, gateway) never requires a re-emit."""
    src = emit_result["signatures/vet_contact"].read_text()
    assert "ROTE_MODEL_VET_CONTACT?: string;" in src
    assert "ROTE_BASE_URL_VET_CONTACT?: string;" in src
    assert "model: env.ROTE_MODEL_VET_CONTACT ?? " in src
    assert "baseURL: env.ROTE_BASE_URL_VET_CONTACT" in src


def test_dev_vars_example_documents_judge_overrides(emit_result: dict[str, Path]) -> None:
    src = emit_result[".dev.vars.example"].read_text()
    assert "# ROTE_MODEL_VET_CONTACT=" in src
    assert "# ROTE_BASE_URL_VET_CONTACT=" in src


def _workers_ai_pipeline() -> Pipeline:
    """Minimal pipeline with a single Workers AI llm_judge node."""
    from rote.ir import Node, PipelineInput

    signature = LLMSignature(
        input_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {"ticket": {"type": "string"}},
            "required": ["ticket"],
        },
        output_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {"category": {"type": "string", "enum": ["billing", "other"]}},
            "required": ["category"],
        },
        prompt="Classify {{ ticket }}.",
        client="workers-ai",
        model="@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        temperature=0,
    )
    return Pipeline(
        name="wai-demo",
        input=PipelineInput(type="TicketInput", required=["ticket"]),
        nodes=[
            Node(
                id="classify",
                kind=NodeKind.LLM_JUDGE,
                description="classify a ticket",
                inputs={"ticket": "pipeline.input.ticket"},
                signature_spec=signature,
            )
        ],
        edges=[],
        entry_nodes=["classify"],
        exit_nodes=["classify"],
    )


def test_workers_ai_signature_uses_ai_binding_not_api_key(tmp_path: Path) -> None:
    """A ``workers-ai`` judge runs on the ``env.AI`` binding with schema-locked
    output — no vendor SDK, no API key, and the AI binding wired in wrangler."""
    result = CloudflareAdapter().emit(_workers_ai_pipeline(), tmp_path)

    sig = result["signatures/classify"].read_text()
    assert "env.AI.run" in sig
    assert "response_format" in sig
    assert '"json_schema"' in sig
    assert "json_schema: OUTPUT_JSON_SCHEMA" in sig
    assert 'from "zod"' in sig
    # No SDK, no key — the binding is the auth.
    assert "@anthropic-ai/sdk" not in sig
    assert 'from "openai"' not in sig
    assert "ANTHROPIC_API_KEY" not in sig
    assert "OPENAI_API_KEY" not in sig

    # Env carries the AI binding and no API-key secret.
    workflow = result["workflow"].read_text()
    assert "AI: Ai;" in workflow
    assert "ANTHROPIC_API_KEY" not in workflow
    assert "OPENAI_API_KEY" not in workflow

    # wrangler registers the Workers AI binding.
    wrangler = result["wrangler"].read_text()
    assert '"ai"' in wrangler
    assert '"AI"' in wrangler

    # No key is prompted for at deploy time.
    assert "ANTHROPIC_API_KEY" not in result[".dev.vars.example"].read_text()

    # The MCP-free invariant holds on the Workers AI path too.
    assert_no_mcp_in_ts(result, min_files=3)


def test_constants_value_cannot_break_out_of_ts_block_comment(tmp_path: Path) -> None:
    """A constant *value* (arbitrary — not charset-constrained at the IR
    boundary) that carries a ``*/`` block-comment breakout must be neutralized
    at emission. Before the fix, ``json.dumps(value)`` left ``*/`` intact,
    closing the JSDoc and turning the rest of the value into live TypeScript
    (confirmed: ``node`` executed the injected statement)."""
    from rote.ir import Node, PipelineInput

    breakout = "a */ globalThis.__ROTE_INJECTED = true; /* b"
    pipeline = Pipeline(
        name="demo",
        input=PipelineInput(type="In"),
        nodes=[
            Node(
                id="n1",
                kind=NodeKind.EXTERNAL_CALL,
                description="x",
                impl="extracted/a.py:go",
                constants={"evil": breakout},
            )
        ],
        edges=[],
        entry_nodes=["n1"],
        exit_nodes=["n1"],
    )
    result = CloudflareAdapter().emit(pipeline, tmp_path)
    stub = result["extracted/n1"].read_text()
    # The live breakout sequence is gone; only the neutralized "* /" remains,
    # so the injected statement stays inside the JSDoc comment.
    assert "*/ globalThis.__ROTE_INJECTED" not in stub
    assert "* / globalThis.__ROTE_INJECTED" in stub


def test_extracted_stub_throws_not_implemented(
    emit_result: dict[str, Path],
) -> None:
    src = emit_result["extracted/taxonomy_lookup"].read_text()
    assert "throw new Error" in src
    assert "stub not implemented" in src


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


def test_wrangler_config_registers_workflow(
    emit_result: dict[str, Path], bdr_pipeline: Pipeline
) -> None:
    """wrangler.jsonc must declare the workflow class so deploy works."""
    import json
    import re

    raw = emit_result["wrangler"].read_text()
    # Strip the leading // comment lines so json.loads works.
    body = re.sub(r"^\s*//[^\n]*\n", "", raw, flags=re.MULTILINE)
    cfg = json.loads(body)
    assert cfg["name"] == bdr_pipeline.name
    assert cfg["compatibility_date"] == "2026-04-25"
    [wf] = cfg["workflows"]
    assert wf["binding"] == "PIPELINE"
    assert wf["class_name"] == "BdrCampaignWorkflow"


def test_package_json_has_required_deps(emit_result: dict[str, Path]) -> None:
    import json

    pkg = json.loads(emit_result["package.json"].read_text())
    assert "@anthropic-ai/sdk" in pkg["dependencies"]
    assert "zod" in pkg["dependencies"]
    assert "wrangler" in pkg["devDependencies"]
    assert "@cloudflare/workers-types" in pkg["devDependencies"]


def test_readme_emitted_with_deploy_button(
    emit_result: dict[str, Path], bdr_pipeline: Pipeline
) -> None:
    """README.md carries the Deploy to Cloudflare button per the spec at
    https://developers.cloudflare.com/workers/platform/deploy-buttons/:
    a markdown image link targeting the deploy service with the repo URL
    in the ``url`` query parameter (placeholder — the repo URL isn't
    known at emission time).
    """
    readme = emit_result["README"]
    assert readme.name == "README.md"
    src = readme.read_text()
    assert src.startswith(f"# {bdr_pipeline.name} — Cloudflare Workflows runtime")
    assert (
        "[![Deploy to Cloudflare](https://deploy.workers.cloudflare.com/button)]"
        "(https://deploy.workers.cloudflare.com/?url=REPLACE-WITH-YOUR-REPO-URL)"
    ) in src
    # Quickstart references the real trigger surfaces of the emitted code.
    assert f"npx wrangler workflows trigger {bdr_pipeline.name}" in src
    assert f"npx wrangler workflows instances send-event {bdr_pipeline.name}" in src
    # Every HITL gate is documented with its event type.
    for node in bdr_pipeline.nodes:
        if node.kind is NodeKind.HITL_GATE:
            assert f"`{node.signal}`" in src
    # MCP triggering pointer. Mentioning MCP here is fine — the MCP-free
    # invariant test scans only .ts files, and README.md is documentation,
    # not executable code.
    assert "rote register" in src
    assert "rote serve" in src


def test_dev_vars_example_declares_secrets(emit_result: dict[str, Path]) -> None:
    """.dev.vars.example drives the deploy button's secret prompts.

    Per the deploy-buttons docs, secrets belong in ``.dev.vars.example``
    (dotenv format) so the one-click flow asks the deployer for values.
    The BDR pipeline's judges are all Anthropic, so only
    ``ANTHROPIC_API_KEY`` is declared.
    """
    src = emit_result[".dev.vars.example"].read_text()
    assert "ANTHROPIC_API_KEY=" in src
    assert "OPENAI_API_KEY" not in src


# ───────── Data-flow threading ─────────


def test_workflow_binds_pipeline_input(emit_result: dict[str, Path]) -> None:
    """The workflow binds the instance params once, up front."""
    src = emit_result["workflow"].read_text()
    assert "const pipelineInput = event.payload;" in src


def test_entry_nodes_receive_pipeline_input(emit_result: dict[str, Path]) -> None:
    src = emit_result["workflow"].read_text()
    # BDR's two entry nodes share wave 1 but dispatch differently.
    # target_research is an agent loop with MCP tools, so it parks on auth
    # — and a parkable step stays OUT of the Promise.all wave (waitForEvent
    # inside a promise combinator is undocumented, and its timeout throws,
    # which would reject the whole wave). That leaves taxonomy_lookup alone
    # in the wave, so it emits in the sequential form.
    sequential = "{\n                brief: pipelineInput,\n            }"
    parkable = "{\n                            brief: pipelineInput,\n                        }"
    assert f"targetResearch({parkable}, this.env)" in src
    assert f"taxonomyLookup({sequential})" in src


def test_downstream_nodes_receive_upstream_results(emit_result: dict[str, Path]) -> None:
    """The committed BDR bindings appear as real payload expressions."""
    src = emit_result["workflow"].read_text()
    # HITL gate output field → downstream step payload. Field access on
    # node results goes through a Record cast (step.do's Rpc.Serializable
    # constraint widens inferred result types — see _ref_to_ts_expr).
    rec = "as Record<string, unknown>"
    assert f'contacts: (contact_review_gate_result {rec})["approved_contacts"],' in src
    # Node output fields chain through the exclusion checks.
    assert f'contacts: (hubspot_upsert_result {rec})["upserted"],' in src
    assert f'contacts: (exclusion_check_dnc_result {rec})["passed"],' in src
    # Pipeline input field selection (Params is already a Record — no cast).
    assert 'target_quota: pipelineInput["target_quota"],' in src
    # Fan-in on the report node.
    assert f'template_ids: (create_sales_template_result {rec})["template_ids"],' in src
    # No node is left passing the placeholder empty payload in BDR.
    assert "async () => hubspotUpsert({})" not in src


def test_node_without_inputs_gets_empty_payload(tmp_path: Path) -> None:
    """Back-compat: nodes with no ``inputs`` still receive {} and the
    workflow skips the pipelineInput binding entirely."""
    from rote.ir import Node, NodeKind, Pipeline, PipelineInput

    pipeline = Pipeline(
        name="no-bindings",
        input=PipelineInput(type="X"),
        nodes=[
            Node(id="only", kind=NodeKind.PURE_FUNCTION, description="x", impl="x.py:y"),
        ],
        edges=[],
        entry_nodes=["only"],
        exit_nodes=["only"],
    )
    src = CloudflareAdapter().emit_workflow(pipeline)
    assert "async () => only({})" in src
    assert "pipelineInput" not in src


def test_emit_rejects_forward_reference() -> None:
    """Inputs referencing a later wave must fail at emit time, not as an
    undefined-variable error inside the deployed worker."""
    from rote.ir import Edge, Node, NodeKind, Pipeline, PipelineInput

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
        CloudflareAdapter().emit_workflow(pipeline)


def test_extracted_stub_return_type_is_never(
    emit_result: dict[str, Path],
) -> None:
    """Stubs declare Promise<never>: honest for an always-throwing stub,
    satisfies step.do's Rpc.Serializable constraint, and keeps the
    workflow's `(<id>_result as Record<...>)["field"]` casts compiling.
    (Promise<Record<string, unknown>> breaks step.do overload resolution —
    `unknown` isn't structurally serializable.)"""
    src = emit_result["extracted/hubspot_upsert"].read_text()
    assert "Promise<never> {" in src
    # An agent_loop module is NOT a stub — it returns the loop's real
    # result. Its return type is an anonymous object type rather than an
    # interface, which is what keeps the workflow's
    # `as Record<string, unknown>` casts compiling (TypeScript grants
    # implicit index signatures to the former only).
    loop_src = emit_result["extracted/lead_generation_loop"].read_text()
    assert "Promise<never> {" not in loop_src
    assert "iterations: number | null }> {" in loop_src


# ───────── Custom config ─────────


def test_custom_config_overrides_workflow_binding(bdr_pipeline: Pipeline, tmp_path: Path) -> None:
    cfg = CloudflareAdapterConfig(
        workflow_binding="MY_PIPE",
        compatibility_date="2026-01-01",
        anthropic_default_model="claude-opus-4-7",
    )
    adapter = CloudflareAdapter(cfg)
    out = tmp_path / "custom"
    adapter.emit(bdr_pipeline, out)

    workflow_src = (out / "src" / "workflow.ts").read_text()
    assert "MY_PIPE: Workflow<Params>" in workflow_src

    sig_src = (out / "src" / "signatures" / "vet_contact.ts").read_text()
    assert 'model: env.ROTE_MODEL_VET_CONTACT ?? "claude-opus-4-7"' in sig_src

    import json
    import re

    raw = (out / "wrangler.jsonc").read_text()
    body = re.sub(r"^\s*//[^\n]*\n", "", raw, flags=re.MULTILINE)
    wrangler = json.loads(body)
    assert wrangler["compatibility_date"] == "2026-01-01"
    assert wrangler["workflows"][0]["binding"] == "MY_PIPE"


# ───────── Signal validation at emit time ─────────


def test_ir_rejects_invalid_hitl_signal_name() -> None:
    """A hitl_gate.signal containing a dot must be rejected at IR validation
    (before any adapter sees it, and long before a Cloudflare runtime error).

    The constraint moved from the Cloudflare adapter to the IR so every
    adapter inherits it and a crafted pipeline.yaml can't inject a
    non-identifier signal into emitted signal-handler names. The adapter's
    ``_validate_signal_name`` remains as defense-in-depth."""
    from pydantic import ValidationError

    from rote.ir import Node, NodeKind

    with pytest.raises(ValidationError, match="must be a valid identifier"):
        Node(
            id="gate",
            kind=NodeKind.HITL_GATE,
            description="bad signal",
            signal="invalid.signal",  # dot is not a valid identifier
        )


# ───────── Signature-spec requirement for Cloudflare ─────────


def test_emit_rejects_llm_judge_without_signature_spec(tmp_path: Path) -> None:
    """The Cloudflare adapter requires the structured signature_spec.

    The legacy ``signature: 'path/to/file.py:Class'`` form is Python-specific
    and cannot be transpiled to TypeScript — so we error explicitly rather
    than silently emit broken code.
    """
    from rote.ir import Edge, Node, NodeKind, Pipeline, PipelineInput

    pipeline = Pipeline(
        name="legacy-only",
        input=PipelineInput(type="X", required=[]),
        nodes=[
            Node(
                id="entry",
                kind=NodeKind.PURE_FUNCTION,
                description="x",
                impl="extracted/foo.py:bar",
            ),
            Node(
                id="judge",
                kind=NodeKind.LLM_JUDGE,
                description="legacy path only",
                signature="signatures/x.py:X",  # legacy form, no signature_spec
            ),
        ],
        edges=[Edge(**{"from": "entry", "to": "judge"})],
        entry_nodes=["entry"],
        exit_nodes=["judge"],
    )
    adapter = CloudflareAdapter()
    with pytest.raises(ValueError, match="requires signature_spec"):
        adapter.emit(pipeline, tmp_path / "y")


def test_emit_accepts_llm_judge_with_signature_spec(tmp_path: Path) -> None:
    from rote.ir import Node, NodeKind, Pipeline, PipelineInput

    spec = LLMSignature(
        input_schema={
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
        },
        output_schema={
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "required": ["a"],
        },
        prompt="Answer: {{ q }}",
    )
    pipeline = Pipeline(
        name="spec-only",
        input=PipelineInput(type="X", required=[]),
        nodes=[
            Node(
                id="judge",
                kind=NodeKind.LLM_JUDGE,
                description="structured signature",
                signature_spec=spec,
            ),
        ],
        edges=[],
        entry_nodes=["judge"],
        exit_nodes=["judge"],
    )
    adapter = CloudflareAdapter()
    out = tmp_path / "ok"
    written = adapter.emit(pipeline, out)
    assert written["signatures/judge"].exists()
    sig_src = written["signatures/judge"].read_text()
    # Prompt and schemas survive into the emitted module.
    assert "Answer: " in sig_src
    assert "{{ q }}" in sig_src
    assert "z.object(" in sig_src


# ───────── manifest.json ─────────


def test_manifest_matches_pipeline_identity(bdr_pipeline: Pipeline, tmp_path: Path) -> None:
    from rote.adapters._common import pipeline_identity
    from rote.adapters.cloudflare import emit_manifest

    manifest = json.loads(emit_manifest(bdr_pipeline))
    identity = pipeline_identity(bdr_pipeline)

    assert manifest["schema"] == 1
    assert manifest["adapter"] == "cloudflare"
    assert manifest["name"] == identity["name"]
    assert manifest["version"] == identity["version"]
    assert manifest["pipeline_hash"] == identity["pipeline_hash"]
    assert manifest["class_name"] == identity["class_name"]
    assert manifest["entry"] == "src/workflow.ts"


def test_manifest_node_ids_match_emitted_modules(
    adapter: CloudflareAdapter, bdr_pipeline: Pipeline, tmp_path: Path
) -> None:
    """node_ids equal the signatures + extracted basenames actually written,
    exactly as rote-cloud/upload.mjs derives them from the src/ tree."""
    out = tmp_path / "emit"
    adapter.emit(bdr_pipeline, out)

    on_disk = set()
    for sub in ("signatures", "extracted"):
        d = out / "src" / sub
        if d.exists():
            # `_`-prefixed modules are shared runtime helpers (_roteMcp,
            # _roteInference), not nodes — they carry no node id and must
            # not reach the manifest.
            on_disk.update(p.stem for p in d.glob("*.ts") if not p.stem.startswith("_"))

    manifest = json.loads((out / "manifest.json").read_text())
    assert set(manifest["node_ids"]) == on_disk
    # HITL gates emit no module and contribute no id.
    gate_ids = {n.id for n in bdr_pipeline.nodes if n.kind is NodeKind.HITL_GATE}
    assert gate_ids.isdisjoint(manifest["node_ids"])


def test_manifest_input_schema_is_pipeline_input_schema(
    bdr_pipeline: Pipeline,
) -> None:
    from rote.adapters.cloudflare import emit_manifest

    manifest = json.loads(emit_manifest(bdr_pipeline))
    assert manifest["input_schema"] == (bdr_pipeline.input.input_schema or {})
    # BDR declares a real JSON Schema, so it's non-empty.
    assert manifest["input_schema"].get("properties")


def test_manifest_written_at_output_root(
    adapter: CloudflareAdapter, bdr_pipeline: Pipeline, tmp_path: Path
) -> None:
    out = tmp_path / "emit"
    written = adapter.emit(bdr_pipeline, out)
    assert written["manifest.json"] == (out / "manifest.json").resolve()
    assert (out / "manifest.json").is_file()


# ───────── emit preserved-paths exposure ─────────


def test_emit_result_exposes_preserved_paths(
    adapter: CloudflareAdapter, bdr_pipeline: Pipeline, tmp_path: Path
) -> None:
    out = tmp_path / "emit"
    first = adapter.emit(bdr_pipeline, out)
    assert first.preserved == {}

    # A user edits an emitted stub, then re-emits: the edit is preserved
    # and the conflict surfaces on the result's .preserved map.
    label, target = next((lbl, p) for lbl, p in first.items() if p.suffix == ".ts")
    target.write_text(target.read_text() + "\n// user impl\n", encoding="utf-8")

    second = adapter.emit(bdr_pipeline, out)
    rel = target.relative_to(out).as_posix()
    assert rel in second.preserved
    assert second.preserved[rel] == target.with_name(target.name + ".new")
    # Backward-compat: it still behaves as the written mapping.
    assert second[label].name.endswith(".new")


# ───────── Parallel waves ─────────
#
# The adapter computed execution waves and then emitted every step
# sequentially, so a wave's concurrency existed only as a comment. These
# pin the Promise.all form and — just as importantly — the two shapes
# that must stay sequential.


def _wave_pipeline(*nodes: Node) -> Pipeline:
    """Every node an entry node with no edges — one single wave."""
    return Pipeline(
        name="wave",
        input={"type": "In", "required": []},
        nodes=list(nodes),
        edges=[],
        entry_nodes=[n.id for n in nodes],
        exit_nodes=[n.id for n in nodes],
    )


def _plain(node_id: str) -> Node:
    return Node(
        id=node_id,
        kind=NodeKind.EXTERNAL_CALL,
        description=f"Fetch {node_id}.",
        impl=f"extracted/{node_id}.py:{node_id}",
    )


def test_multi_node_wave_dispatches_concurrently() -> None:
    src = CloudflareAdapter().emit_workflow(_wave_pipeline(_plain("a"), _plain("b"), _plain("c")))

    assert "await Promise.all([" in src
    # All three results destructured from the one combinator.
    for nid in ("a", "b", "c"):
        assert f"{nid}_result," in src
        assert f'step.do(\n                "{nid}",' in src
    # Exactly one combinator for the one wave.
    assert src.count("Promise.all([") == 1
    # Never wrapped in an outer step.do — that would burn a step, cap the
    # wave at the 1 MiB step-result limit, and collapse per-node retries.
    assert "step.do(\n            async () => Promise.all" not in src


def test_single_node_wave_stays_sequential() -> None:
    """One node is not a wave worth a combinator."""
    src = CloudflareAdapter().emit_workflow(_wave_pipeline(_plain("only")))
    assert "Promise.all" not in src
    assert "const only_result = await step.do(" in src


def test_hitl_gate_never_joins_a_parallel_wave() -> None:
    """`waitForEvent` inside a promise combinator is undocumented, and its
    timeout throw would reject the whole wave. Gates stay sequential even
    when they share a wave with dispatchable nodes."""
    gate = Node(
        id="approve",
        kind=NodeKind.HITL_GATE,
        description="Human approval.",
        signal="approved",
    )
    src = CloudflareAdapter().emit_workflow(_wave_pipeline(_plain("a"), _plain("b"), gate))

    assert "await Promise.all([" in src
    combinator = src[src.index("Promise.all([") : src.index("]);")]
    assert "waitForEvent" not in combinator
    assert "const approve_event = await step.waitForEvent" in src


def test_mcp_parkable_node_never_joins_a_parallel_wave() -> None:
    """The parkable retry loop suspends on waitForEvent too — same rule."""
    parkable = Node(
        id="search_docs",
        kind=NodeKind.EXTERNAL_CALL,
        description="Search docs over MCP.",
        mcp={"server": "docs", "tool": "search"},
    )
    src = CloudflareAdapter().emit_workflow(_wave_pipeline(_plain("a"), _plain("b"), parkable))

    assert "await Promise.all([" in src
    combinator = src[src.index("Promise.all([") : src.index("]);")]
    assert "search_docs" not in combinator
    assert "for (let search_docs_attempt = 0; ; search_docs_attempt++)" in src


def test_parallel_wave_keeps_per_node_step_config() -> None:
    """Concurrency must not flatten each node's own retry/timeout config."""
    slow = _plain("slow")
    slow.timeout = "9m"
    flaky = _plain("flaky")
    flaky.retry = RetryPolicy(max=4, backoff="exponential")
    src = CloudflareAdapter().emit_workflow(_wave_pipeline(slow, flaky))

    assert 'timeout: "9 minutes"' in src
    assert "retries: { limit: 4," in src


def test_hitl_gate_uses_its_declared_timeout_not_the_default() -> None:
    """A gate's `timeout:` must reach its `waitForEvent` config.

    Found by mutation testing: hardcoding the adapter default left the
    suite green. This one is worse than it looks on Cloudflare, because
    `waitForEvent` THROWS on timeout rather than returning — a gate that
    silently inherits a 7-day default instead of its declared 1-hour
    budget turns a fast-fail approval window into a week-long hang, and
    the reverse turns a legitimate week-long wait into a failed run.

    Asserting only that the pinned value appears would still pass if the
    adapter emitted it for every gate, so both gates are checked.
    """
    pinned = Node(
        id="quick_gate",
        kind=NodeKind.HITL_GATE,
        description="d",
        signal="quick_approved",
        timeout="1h",
    )
    defaulted = Node(
        id="slow_gate",
        kind=NodeKind.HITL_GATE,
        description="d",
        signal="slow_approved",
    )
    pipeline = Pipeline(
        name="gates",
        input={"type": "In", "required": [], "optional": []},
        nodes=[pinned, defaulted],
        edges=[{"from": "quick_gate", "to": "slow_gate"}],
        entry_nodes=["quick_gate"],
        exit_nodes=["slow_gate"],
    )
    src = emit_workflow(pipeline, CloudflareAdapterConfig(default_hitl_timeout="7d"))

    quick = src[src.index('"quick_gate"') : src.index('"slow_gate"')]
    slow = src[src.index('"slow_gate"') :]
    assert '"1 hour"' in quick, "a gate's declared timeout must reach waitForEvent"
    assert '"7 days"' in slow, "a gate without a timeout falls back to the default"
    assert '"7 days"' not in quick
