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
from rote.adapters.cloudflare import (
    CloudflareAdapter,
    CloudflareAdapterConfig,
    _ir_duration_to_cf,
    _pipeline_hash,
    _to_camel_case,
    _to_pascal_case,
    _validate_signal_name,
    json_schema_to_zod,
)
from rote.ir import LLMSignature, NodeKind, Pipeline, load_pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
BDR_PIPELINE_YAML = REPO_ROOT / "examples" / "bdr-outreach" / "expected" / "pipeline.yaml"
BDR_EMIT_DIR = REPO_ROOT / "examples" / "bdr-outreach" / "expected" / "runtimes" / "cloudflare"


# ───────── Fixtures ─────────


@pytest.fixture(scope="module")
def bdr_pipeline() -> Pipeline:
    return load_pipeline(BDR_PIPELINE_YAML)


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

    Mirrors the Temporal adapter's
    ``test_emitted_activities_never_reference_mcp``. We don't have a TS
    AST in pure Python, so we use a regex over import statements + call
    expressions — any ``mcp`` substring outside comments/strings fails.

    Comments and JSDoc may *mention* MCP to explain the graduation
    history (it's part of the architecture story), so we strip
    ``/* ... */`` blocks, ``// ...`` line comments, and string literals
    before scanning.
    """
    import re

    js_string = re.compile(r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'')
    block_comment = re.compile(r"/\*[\s\S]*?\*/")
    line_comment = re.compile(r"//[^\n]*")

    forbidden = ("mcp",)
    for label, path in emit_result.items():
        if not str(path).endswith(".ts"):
            continue
        src = path.read_text()
        # Strip strings + comments before scanning — those are allowed
        # to reference MCP for documentation purposes.
        cleaned = block_comment.sub(" ", src)
        cleaned = line_comment.sub(" ", cleaned)
        cleaned = js_string.sub('""', cleaned)
        for needle in forbidden:
            assert needle not in cleaned.lower(), (
                f"{label} ({path.name}) contains forbidden substring {needle!r} in executable code"
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
    assert 'model: "claude-opus-4-7"' in sig_src

    import json
    import re

    raw = (out / "wrangler.jsonc").read_text()
    body = re.sub(r"^\s*//[^\n]*\n", "", raw, flags=re.MULTILINE)
    wrangler = json.loads(body)
    assert wrangler["compatibility_date"] == "2026-01-01"
    assert wrangler["workflows"][0]["binding"] == "MY_PIPE"


# ───────── Signal validation at emit time ─────────


def test_emit_rejects_invalid_hitl_signal_name(tmp_path: Path) -> None:
    """A pipeline whose hitl_gate.signal contains a dot must fail at emit time
    (before the user discovers it as a Cloudflare runtime error)."""
    from rote.ir import Edge, Node, NodeKind, Pipeline, PipelineInput

    pipeline = Pipeline(
        name="bad-signals",
        input=PipelineInput(type="X", required=[]),
        nodes=[
            Node(
                id="entry",
                kind=NodeKind.PURE_FUNCTION,
                description="x",
                impl="extracted/foo.py:bar",
            ),
            Node(
                id="gate",
                kind=NodeKind.HITL_GATE,
                description="bad signal",
                signal="invalid.signal",  # dot is not allowed by Cloudflare
            ),
        ],
        edges=[Edge(**{"from": "entry", "to": "gate"})],
        entry_nodes=["entry"],
        exit_nodes=["gate"],
    )
    adapter = CloudflareAdapter()
    with pytest.raises(ValueError, match="invalid characters"):
        adapter.emit(pipeline, tmp_path / "x")


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
