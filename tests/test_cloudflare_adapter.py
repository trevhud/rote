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

import shutil
from pathlib import Path

import pytest

from rote.adapters import get_adapter
from rote.adapters._ts_common import json_schema_to_zod
from rote.adapters.cloudflare import (
    CloudflareAdapter,
    CloudflareAdapterConfig,
    _ir_duration_to_cf,
    _pipeline_hash,
    _to_camel_case,
    _to_pascal_case,
    _validate_signal_name,
)
from rote.ir import LLMSignature, NodeKind, Pipeline
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
        assert f'step.do(\n            "{node.id}"' in src, (
            f"Missing step.do call for node {node.id!r} in workflow.ts"
        )


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
    graduation history; executable code may not. Scan logic is shared —
    see tests/_helpers.py.
    """
    assert_no_mcp_in_ts(emit_result, min_files=13)


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


def test_extracted_stub_throws_not_implemented(
    emit_result: dict[str, Path],
) -> None:
    src = emit_result["extracted/taxonomy_lookup"].read_text()
    assert "throw new Error" in src
    assert "stub not implemented" in src


def test_agent_loop_stub_documents_tools(emit_result: dict[str, Path]) -> None:
    src = emit_result["extracted/lead_generation_loop"].read_text()
    assert "agent_loop" in src
    assert "Tools the agent should be allowed to call" in src
    assert "zoominfo_search_contacts" in src
    assert "Loop body sub-nodes" in src
    assert "enrich_contact_batch" in src


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
    payload = "{\n                brief: pipelineInput,\n            }"
    assert f"targetResearch({payload}, this.env)" in src
    assert f"taxonomyLookup({payload})" in src


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
    loop_src = emit_result["extracted/lead_generation_loop"].read_text()
    assert "Promise<never> {" in loop_src


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
