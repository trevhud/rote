"""Tests for the Inngest adapter.

Mirrors ``test_cloudflare_adapter.py`` / ``test_dbos_ts_adapter.py``:

1. **Emission** — adapter produces the expected file layout, with each
   non-HITL node represented by an extracted/ or signatures/ module and
   each HITL gate represented by a ``step.waitForEvent`` call site.
2. **Structural invariants** — text assertions over the emitted
   TypeScript: v4's single-options ``createFunction`` form, one
   ``step.run`` per non-HITL top-level node, ``Promise.all`` for
   parallel waves, function-level retry mapping, data-flow threading,
   and the MCP-free architectural invariant.
3. **Compilation / live execution** — see ``test_inngest_e2e.py``
   (``@pytest.mark.slow``): real ``npm install`` + ``tsc --noEmit``,
   plus a live run against the Inngest dev server.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest

from rote.adapters import get_adapter
from rote.adapters.inngest import (
    InngestAdapter,
    InngestAdapterConfig,
    _function_retries,
    _validate_signal_name,
    gate_event_name,
    trigger_event_name,
)
from rote.ir import (
    Edge,
    LLMSignature,
    Node,
    NodeKind,
    Pipeline,
    PipelineInput,
    RetryPolicy,
)
from tests._helpers import assert_no_mcp_in_ts

REPO_ROOT = Path(__file__).resolve().parent.parent
BDR_PIPELINE_YAML = REPO_ROOT / "examples" / "bdr-outreach" / "expected" / "pipeline.yaml"
BDR_EMIT_DIR = REPO_ROOT / "examples" / "bdr-outreach" / "expected" / "runtimes" / "inngest"


# ───────── Fixtures ─────────


@pytest.fixture(scope="module")
def adapter() -> InngestAdapter:
    return InngestAdapter()


@pytest.fixture(scope="module")
def emit_result(adapter: InngestAdapter, bdr_pipeline: Pipeline) -> dict[str, Path]:
    """Emit the BDR pipeline once per module into the committed snapshot dir."""
    if BDR_EMIT_DIR.exists():
        # Remove stale files so renames produce a clean diff.
        shutil.rmtree(BDR_EMIT_DIR)
    return adapter.emit(bdr_pipeline, BDR_EMIT_DIR)


@pytest.fixture(scope="module")
def pipeline_src(emit_result: dict[str, Path]) -> str:
    return emit_result["pipeline"].read_text(encoding="utf-8")


def _minimal_pipeline(**overrides) -> Pipeline:  # noqa: ANN003
    defaults = dict(
        name="mini",
        input=PipelineInput(type="X"),
        nodes=[
            Node(id="only", kind=NodeKind.PURE_FUNCTION, description="x", impl="x.py:y"),
        ],
        edges=[],
        entry_nodes=["only"],
        exit_nodes=["only"],
    )
    defaults.update(overrides)
    return Pipeline(**defaults)


# ───────── Event-name mapping ─────────


def test_trigger_event_name_is_namespaced_and_unversioned(bdr_pipeline: Pipeline) -> None:
    """The trigger event carries the pipeline namespace but NOT the
    pipeline hash — a regenerated pipeline keeps the same trigger, so
    senders never need updating."""
    assert trigger_event_name(bdr_pipeline) == "bdr-campaign/run.requested"


def test_gate_event_name_namespaces_the_ir_signal(bdr_pipeline: Pipeline) -> None:
    gate = bdr_pipeline.node_by_id("contact_review_gate")
    assert gate_event_name(bdr_pipeline, gate) == "bdr-campaign/contact_review_approved"


def test_validate_signal_name_accepts_clean_names() -> None:
    _validate_signal_name("contact_review_approved", "node-id")
    _validate_signal_name("foo-bar_baz", "node-id")
    _validate_signal_name("ABC123", "node-id")


def test_validate_signal_name_rejects_dots_spaces_slashes() -> None:
    with pytest.raises(ValueError, match="invalid characters"):
        _validate_signal_name("foo.bar", "node-id")
    with pytest.raises(ValueError, match="invalid characters"):
        _validate_signal_name("has space", "node-id")
    with pytest.raises(ValueError, match="invalid characters"):
        _validate_signal_name("has/slash", "node-id")


def test_event_prefix_rejects_unsafe_pipeline_name() -> None:
    pipeline = _minimal_pipeline(name="bad name!")
    with pytest.raises(ValueError, match="unsafe for an Inngest event-name prefix"):
        trigger_event_name(pipeline)


# ───────── Retry mapping ─────────


def test_function_retries_is_max_across_nodes(bdr_pipeline: Pipeline) -> None:
    """BDR's most retry-hungry nodes declare max 5 — the function budget."""
    declared = [n.retry.max for n in bdr_pipeline.nodes if n.retry is not None]
    assert max(declared) == 5
    assert _function_retries(bdr_pipeline) == 5


def test_function_retries_defaults_to_sdk_default_when_undeclared() -> None:
    """No node declares a retry policy → keep Inngest's default (3),
    not 0: platform retries are the baseline for transient HTTP
    failures, not an opt-in."""
    assert _function_retries(_minimal_pipeline()) == 3


def test_function_retries_clamps_to_inngest_maximum() -> None:
    pipeline = _minimal_pipeline(
        nodes=[
            Node(
                id="only",
                kind=NodeKind.PURE_FUNCTION,
                description="x",
                impl="x.py:y",
                retry=RetryPolicy(max=25),
            ),
        ],
    )
    assert _function_retries(pipeline) == 20


# ───────── Adapter dispatch ─────────


def test_adapter_registered_in_dispatch_dict() -> None:
    adapter = get_adapter("inngest")
    assert isinstance(adapter, InngestAdapter)


def test_unknown_runtime_lists_inngest() -> None:
    with pytest.raises(KeyError, match="inngest"):
        get_adapter("nonexistent-runtime")


# ───────── Emission tests ─────────


def test_emit_produces_expected_files(emit_result: dict[str, Path], bdr_pipeline: Pipeline) -> None:
    assert emit_result["client"].exists()
    assert emit_result["pipeline"].exists()
    assert emit_result["index"].exists()
    assert emit_result["package.json"].exists()
    assert emit_result["tsconfig.json"].exists()
    assert emit_result["README"].exists()

    # One module per non-HITL node, in either signatures/ or extracted/.
    for node in bdr_pipeline.nodes:
        if node.kind is NodeKind.HITL_GATE:
            continue
        if node.kind is NodeKind.LLM_JUDGE:
            assert emit_result[f"signatures/{node.id}"].exists()
        else:
            assert emit_result[f"extracted/{node.id}"].exists()


def test_create_function_uses_v4_single_options_form(pipeline_src: str) -> None:
    """v4.11.0 rejects the three-argument ``createFunction(opts, trigger,
    handler)`` form at runtime — triggers must live in the options
    object. Verified against the installed SDK with a live probe."""
    assert "inngest.createFunction(" in pipeline_src
    assert 'triggers: [{ event: "bdr-campaign/run.requested" }],' in pipeline_src
    # The legacy second-argument trigger form must NOT be emitted.
    assert re.search(r"createFunction\(\s*\{[^}]*\},\s*\{\s*event:", pipeline_src) is None


def test_function_id_is_versioned_with_pipeline_hash(
    pipeline_src: str, bdr_pipeline: Pipeline
) -> None:
    from rote.adapters._common import _pipeline_hash

    h = _pipeline_hash(bdr_pipeline)
    assert f'id: "bdr-campaign-{h}",' in pipeline_src
    assert f"Pipeline hash: {h}" in pipeline_src


def test_function_level_retries_emitted(pipeline_src: str) -> None:
    assert "retries: 5," in pipeline_src


def test_step_run_for_every_non_hitl_top_level_node(
    pipeline_src: str, bdr_pipeline: Pipeline
) -> None:
    """Every non-HITL top-level node gets a ``step.run("<id>", ...)``.
    Loop-body sub-nodes are excluded — they're orchestrated inside the
    parent agent_loop, not by the top-level function."""
    nested_ids: set[str] = set()
    for node in bdr_pipeline.nodes:
        if node.loop_body:
            nested_ids.update(node.loop_body)
    for node in bdr_pipeline.nodes:
        if node.kind is NodeKind.HITL_GATE or node.id in nested_ids:
            continue
        assert f'step.run("{node.id}"' in pipeline_src, (
            f"Missing step.run call for node {node.id!r} in pipeline.ts"
        )
    # Loop-body sub-nodes never appear as top-level steps.
    for nid in nested_ids:
        assert f'step.run("{nid}"' not in pipeline_src


def test_parallel_wave_uses_promise_all(pipeline_src: str) -> None:
    """Wave 1 (target_research ∥ taxonomy_lookup) fans out via
    ``Promise.all`` — Inngest's documented in-function parallelism
    pattern (not allSettled)."""
    assert (
        "const [target_research_result, taxonomy_lookup_result] = await Promise.all(["
    ) in pipeline_src
    assert "Promise.allSettled" not in pipeline_src


def test_wait_for_event_for_every_hitl_gate(pipeline_src: str, bdr_pipeline: Pipeline) -> None:
    """Each gate parks on ``step.waitForEvent`` with the namespaced IR
    signal as the event name and the IR timeout passed through (the IR
    duration shorthand is ms-compatible)."""
    for node in bdr_pipeline.nodes:
        if node.kind is not NodeKind.HITL_GATE:
            continue
        assert node.signal is not None
        event_name = gate_event_name(bdr_pipeline, node)
        assert f"const {node.id}_event = await step.waitForEvent(" in pipeline_src
        assert f'{{ event: "{event_name}", timeout: "{node.timeout}" }}' in pipeline_src


def test_gate_timeout_raises_non_retriable(pipeline_src: str, bdr_pipeline: Pipeline) -> None:
    """waitForEvent returns null on timeout; the emitted code throws
    NonRetriableError — silence is not approval, and the timed-out wait
    is memoized so retrying could never conjure the approval."""
    assert 'import { NonRetriableError } from "inngest";' in pipeline_src
    for node in bdr_pipeline.nodes:
        if node.kind is not NodeKind.HITL_GATE:
            continue
        assert f"if ({node.id}_event === null) {{" in pipeline_src
    assert pipeline_src.count("throw new NonRetriableError(") == 2


def test_gate_resume_payload_binds_as_result(pipeline_src: str) -> None:
    """The gate's event payload becomes ``<id>_result`` so downstream
    ``inputs:`` bindings can reference the gate like any other node."""
    assert (
        "const contact_review_gate_result = "
        "contact_review_gate_event.data as Record<string, unknown>;"
    ) in pipeline_src


def test_per_node_retry_deltas_documented_as_comments(pipeline_src: str) -> None:
    """Per-node budgets that differ from the function-level budget are
    surfaced as comments (Inngest v4 has no per-step retries), as are
    per-step timeouts (no per-step timeout primitive)."""
    assert "// IR retry policy: max 2 (exponential). Inngest v4" in pipeline_src
    assert "// IR timeout 15m: Inngest v4 has no per-step timeout —" in pipeline_src


def test_emitted_files_never_reference_mcp(emit_result: dict[str, Path]) -> None:
    """Architectural invariant: no MCP runtime in emitted Inngest code.

    Comments, JSDoc, and string literals may mention MCP to explain the
    graduation history; executable code may not. Scan logic is shared —
    see tests/_helpers.py.
    """
    assert_no_mcp_in_ts(emit_result, min_files=13)


def test_signature_module_imports_zod_and_anthropic(emit_result: dict[str, Path]) -> None:
    src = emit_result["signatures/vet_contact"].read_text(encoding="utf-8")
    assert 'from "zod"' in src
    assert 'from "@anthropic-ai/sdk"' in src
    assert "VetContactInput.parse" in src
    assert "VetContactOutput.parse" in src


def test_extracted_stub_throws_not_implemented(emit_result: dict[str, Path]) -> None:
    src = emit_result["extracted/taxonomy_lookup"].read_text(encoding="utf-8")
    assert "throw new Error" in src
    assert "stub not implemented" in src
    assert "Promise<never> {" in src


def test_agent_loop_stub_documents_tools(emit_result: dict[str, Path]) -> None:
    src = emit_result["extracted/lead_generation_loop"].read_text(encoding="utf-8")
    assert "agent_loop" in src
    assert "Tools the agent should be allowed to call" in src
    assert "zoominfo_search_contacts" in src
    assert "Loop body sub-nodes" in src
    assert "enrich_contact_batch" in src


def test_index_serves_via_inngest_node(emit_result: dict[str, Path]) -> None:
    """The serve entrypoint composes the stable ``serve`` export from
    ``inngest/node`` with ``node:http`` (the module's ``createServer``
    is marked EXPERIMENTAL in v4.11.0)."""
    src = emit_result["index"].read_text(encoding="utf-8")
    assert 'import { serve } from "inngest/node";' in src
    assert "serve({ client: inngest, functions: [runPipeline] })" in src
    assert "http.createServer(handler)" in src


def test_client_declares_app_id(emit_result: dict[str, Path]) -> None:
    src = emit_result["client"].read_text(encoding="utf-8")
    assert 'new Inngest({ id: "bdr-campaign" })' in src


def test_package_json_has_required_deps(emit_result: dict[str, Path]) -> None:
    pkg = json.loads(emit_result["package.json"].read_text(encoding="utf-8"))
    assert pkg["dependencies"]["inngest"].startswith("^4.")
    assert "zod" in pkg["dependencies"]
    assert "@anthropic-ai/sdk" in pkg["dependencies"]
    # BDR's judges are all Anthropic — no OpenAI SDK.
    assert "openai" not in pkg["dependencies"]
    assert "typescript" in pkg["devDependencies"]
    assert "@types/node" in pkg["devDependencies"]


# ───────── Data-flow threading ─────────


def test_workflow_binds_pipeline_input(pipeline_src: str) -> None:
    assert "const pipelineInput = event.data as Record<string, unknown>;" in pipeline_src


def test_entry_nodes_receive_pipeline_input(pipeline_src: str) -> None:
    assert "brief: pipelineInput," in pipeline_src


def test_downstream_nodes_receive_upstream_results(pipeline_src: str) -> None:
    """The committed BDR bindings appear as real payload expressions."""
    rec = "as Record<string, unknown>"
    # HITL gate resume payload → downstream step payload.
    assert f'contacts: (contact_review_gate_result {rec})["approved_contacts"],' in pipeline_src
    # Node output fields chain through the exclusion checks.
    assert f'contacts: (hubspot_upsert_result {rec})["upserted"],' in pipeline_src
    assert f'contacts: (exclusion_check_dnc_result {rec})["passed"],' in pipeline_src
    # Pipeline input field selection.
    assert 'target_quota: pipelineInput["target_quota"],' in pipeline_src
    # Fan-in on the report node.
    assert f'template_ids: (create_sales_template_result {rec})["template_ids"],' in pipeline_src


def test_exit_nodes_shape_the_return(pipeline_src: str, bdr_pipeline: Pipeline) -> None:
    for exit_id in bdr_pipeline.exit_nodes:
        assert f'"{exit_id}": {exit_id}_result as Record<string, unknown>,' in pipeline_src


def test_node_without_inputs_gets_empty_payload() -> None:
    """Back-compat: nodes with no ``inputs`` still receive {} and the
    handler skips the pipelineInput binding entirely."""
    src = InngestAdapter().emit_pipeline_ts(_minimal_pipeline())
    assert 'step.run("only", async () => only({}))' in src
    assert "pipelineInput" not in src


def test_emit_rejects_forward_reference() -> None:
    """Inputs referencing a later wave must fail at emit time, not as an
    undefined-variable error inside the deployed function."""
    pipeline = _minimal_pipeline(
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
        InngestAdapter().emit_pipeline_ts(pipeline)


# ───────── Signature-spec requirement ─────────


def test_emit_rejects_llm_judge_without_signature_spec(tmp_path: Path) -> None:
    pipeline = _minimal_pipeline(
        nodes=[
            Node(
                id="judge",
                kind=NodeKind.LLM_JUDGE,
                description="legacy path only",
                signature="signatures/x.py:X",  # legacy form, no signature_spec
            ),
        ],
        entry_nodes=["judge"],
        exit_nodes=["judge"],
    )
    with pytest.raises(ValueError, match="requires signature_spec"):
        InngestAdapter().emit(pipeline, tmp_path / "out")


def test_emit_accepts_llm_judge_with_signature_spec(tmp_path: Path) -> None:
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
    pipeline = _minimal_pipeline(
        nodes=[
            Node(
                id="judge",
                kind=NodeKind.LLM_JUDGE,
                description="structured signature",
                signature_spec=spec,
            ),
        ],
        entry_nodes=["judge"],
        exit_nodes=["judge"],
    )
    written = InngestAdapter().emit(pipeline, tmp_path / "ok")
    sig_src = written["signatures/judge"].read_text(encoding="utf-8")
    assert "Answer: " in sig_src
    assert "{{ q }}" in sig_src
    assert "z.object(" in sig_src


# ───────── README ─────────


def test_readme_flush_left_with_gates(emit_result: dict[str, Path], bdr_pipeline: Pipeline) -> None:
    """README renders flush-left (the dedent-before-interpolate bug class:
    a multi-line gate table would defeat textwrap.dedent and ship the
    whole file as one Markdown code block) and documents every gate."""
    src = emit_result["README"].read_text(encoding="utf-8")
    assert src.startswith("# bdr-campaign — Inngest runtime")
    assert not src.splitlines()[0].startswith(" ")
    gates = bdr_pipeline.nodes_by_kind(NodeKind.HITL_GATE)
    assert len(gates) >= 2
    for gate in gates:
        assert f"| `{gate.id}` | `{gate_event_name(bdr_pipeline, gate)}` |" in src
    # Trigger + resume instructions with the dev server's event API.
    assert "bdr-campaign/run.requested" in src
    assert "http://localhost:8288/e/dev" in src
    # The Next.js mount — this adapter's whole pitch.
    assert 'from "inngest/next"' in src
    assert "app/api/inngest/route.ts" in src
    # Timeout semantics + the function-level retry caveat.
    assert re.search(r"silence\s+is not approval", src)
    assert "function-level" in src


# ───────── Custom config ─────────


def test_custom_config_overrides_models_and_port(bdr_pipeline: Pipeline, tmp_path: Path) -> None:
    cfg = InngestAdapterConfig(anthropic_default_model="claude-opus-4-7", serve_port=4111)
    adapter = InngestAdapter(cfg)
    out = tmp_path / "custom"
    adapter.emit(bdr_pipeline, out)

    sig_src = (out / "src" / "signatures" / "vet_contact.ts").read_text(encoding="utf-8")
    assert 'model: env.ROTE_MODEL_VET_CONTACT ?? "claude-opus-4-7"' in sig_src

    index_src = (out / "src" / "index.ts").read_text(encoding="utf-8")
    assert "process.env.PORT ?? 4111" in index_src
