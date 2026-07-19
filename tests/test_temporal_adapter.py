"""Tests for the Temporal adapter.

Three levels of validation:

1. **Emission** — adapter produces files that parse as valid Python.
2. **Registration** — the emitted workflow and activities register with
   Temporal's decorators at import time.
3. **Execution** — the emitted workflow runs end-to-end inside
   ``WorkflowEnvironment.start_time_skipping()`` with mocked activities.
   See ``test_temporal_e2e.py`` for that.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

from rote.adapters.temporal import (
    TemporalAdapter,
    TemporalAdapterConfig,
    _execution_waves,
    _pipeline_hash,
    _to_pascal_case,
)
from rote.ir import Node, NodeKind, Pipeline
from tests._helpers import mini_pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
BDR_PIPELINE_YAML = REPO_ROOT / "examples" / "bdr-outreach" / "expected" / "pipeline.yaml"
BDR_EMIT_DIR = REPO_ROOT / "examples" / "bdr-outreach" / "expected" / "runtimes" / "temporal"
BDR_EXAMPLE_PKG_ROOT = REPO_ROOT / "examples" / "bdr-outreach"


# ───────── Fixtures ─────────


@pytest.fixture(scope="module")
def adapter() -> TemporalAdapter:
    return TemporalAdapter()


@pytest.fixture(scope="module")
def emit_result(adapter: TemporalAdapter, bdr_pipeline: Pipeline) -> dict[str, Path]:
    """Emit into the expected runtimes dir once per module."""
    return adapter.emit(bdr_pipeline, BDR_EMIT_DIR)


# ───────── Helper / internal tests ─────────


def test_pascal_case_conversion() -> None:
    assert _to_pascal_case("bdr-campaign") == "BdrCampaign"
    assert _to_pascal_case("bdr_campaign") == "BdrCampaign"
    assert _to_pascal_case("foo") == "Foo"
    assert _to_pascal_case("a-b_c-d") == "ABCD"


def test_pipeline_hash_is_stable(bdr_pipeline: Pipeline) -> None:
    """The pipeline hash must be stable across runs — it's used to
    version the emitted workflow type name. Temporal will error if the
    same workflow type produces different code, so the hash must be
    deterministic."""
    h1 = _pipeline_hash(bdr_pipeline)
    h2 = _pipeline_hash(bdr_pipeline)
    assert h1 == h2
    assert len(h1) == 8


def test_pipeline_hash_ignores_source_skill(bdr_pipeline: Pipeline) -> None:
    """``source_skill`` is provenance, not behavior. The graduator
    re-points it per output location, so hashing it would mint a new
    workflow type (re-versioning in-flight workflows) on every
    re-graduation to a different directory — same rule as ``Node.source``."""
    moved = bdr_pipeline.model_copy(update={"source_skill": "/somewhere/else/entirely"})
    assert _pipeline_hash(moved) == _pipeline_hash(bdr_pipeline)
    unset = bdr_pipeline.model_copy(update={"source_skill": None})
    assert _pipeline_hash(unset) == _pipeline_hash(bdr_pipeline)


def test_execution_waves_exclude_loop_body_nodes(bdr_pipeline: Pipeline) -> None:
    """Nodes inside another node's loop_body should not appear in the
    top-level execution waves — they're orchestrated inside the loop."""
    waves = _execution_waves(bdr_pipeline)
    wave_ids = {n.id for wave in waves for n in wave}
    assert "enrich_contact_batch" not in wave_ids, (
        "enrich_contact_batch is a loop-body sub-node; should not be a top-level wave"
    )
    assert "vet_contact" not in wave_ids, (
        "vet_contact is a loop-body sub-node; should not be a top-level wave"
    )


def test_execution_waves_produce_valid_dag(bdr_pipeline: Pipeline) -> None:
    """Nodes in a later wave must have at least one predecessor in an
    earlier wave (except the first wave, which is entry nodes)."""
    waves = _execution_waves(bdr_pipeline)
    assert len(waves) >= 2, "Non-trivial pipelines must produce at least 2 waves"

    # Wave 1 should contain the entry nodes
    wave1_ids = {n.id for n in waves[0]}
    assert "target_research" in wave1_ids
    assert "taxonomy_lookup" in wave1_ids

    # HITL gates appear in their own waves (no activities share their wave)
    for wave in waves:
        hitl_count = sum(1 for n in wave if n.kind is NodeKind.HITL_GATE)
        if hitl_count > 0:
            assert len(wave) == hitl_count, (
                f"HITL gates should not share a wave with activities: {[n.id for n in wave]}"
            )


# ───────── Emission tests ─────────


def test_emit_produces_three_files(emit_result: dict[str, Path]) -> None:
    assert emit_result["activities"].exists()
    assert emit_result["workflow"].exists()
    assert emit_result["__init__"].exists()


def test_emitted_files_are_valid_python(emit_result: dict[str, Path]) -> None:
    for path in (emit_result["activities"], emit_result["workflow"]):
        src = path.read_text(encoding="utf-8")
        try:
            ast.parse(src)
        except SyntaxError as e:
            pytest.fail(f"{path.name} is not valid Python: {e}")


def test_emitted_activities_contain_all_nodes(
    emit_result: dict[str, Path], bdr_pipeline: Pipeline
) -> None:
    """Every non-HITL node should have an @activity.defn in activities.py."""
    src = emit_result["activities"].read_text()
    for node in bdr_pipeline.nodes:
        if node.kind is NodeKind.HITL_GATE:
            continue
        assert f'@activity.defn(name="{node.id}")' in src, f"Missing activity for node {node.id}"


def test_emitted_workflow_has_signal_handlers(
    emit_result: dict[str, Path], bdr_pipeline: Pipeline
) -> None:
    """Every HITL gate should have a corresponding @workflow.signal handler."""
    src = emit_result["workflow"].read_text()
    for node in bdr_pipeline.nodes:
        if node.kind is NodeKind.HITL_GATE:
            assert node.signal is not None
            assert f'@workflow.signal(name="{node.signal}")' in src
            assert f"def {node.signal}(self" in src


def test_emitted_workflow_is_versioned_by_hash(
    emit_result: dict[str, Path], bdr_pipeline: Pipeline
) -> None:
    """The workflow type name should include the pipeline hash so that
    regenerated workflows become a new type (avoiding Temporal
    determinism errors on in-flight runs)."""
    src = emit_result["workflow"].read_text()
    expected = f"BdrCampaign_{_pipeline_hash(bdr_pipeline)}"
    assert f'@workflow.defn(name="{expected}")' in src


def test_emitted_activities_never_reference_mcp(
    emit_result: dict[str, Path],
) -> None:
    """Architectural invariant: no MCP runtime references in emitted code.

    MCP tool calls from the source skill must be graduated into direct
    API calls at emission time. If this assertion ever fails, it means
    the adapter is leaking the MCP abstraction into the runtime hot path.

    We parse the AST and walk only executable statements (imports,
    calls, attribute accesses) — comments and docstrings are allowed to
    *mention* MCP to explain the graduation history.
    """
    src = emit_result["activities"].read_text()
    tree = ast.parse(src)

    forbidden_substrings = ("mcp",)

    def _check(value: str, context: str) -> None:
        lower = value.lower()
        for needle in forbidden_substrings:
            assert needle not in lower, (
                f"{context}: {value!r} references forbidden substring {needle!r}"
            )

    for node in ast.walk(tree):
        # Imports: from X import Y
        if isinstance(node, ast.Import):
            for alias in node.names:
                _check(alias.name, f"import at line {node.lineno}")
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                _check(node.module, f"from-import at line {node.lineno}")
            for alias in node.names:
                _check(alias.name, f"from-import name at line {node.lineno}")
        # Name references and attribute accesses in call sites
        elif isinstance(node, ast.Name):
            _check(node.id, f"name at line {node.lineno}")
        elif isinstance(node, ast.Attribute):
            _check(node.attr, f"attribute at line {node.lineno}")


# ───────── Temporal decorator registration ─────────


@pytest.fixture(scope="module")
def imported_modules(emit_result: dict[str, Path]):  # noqa: ANN201
    """Import the emitted workflow and activities modules.

    We mutate sys.path so the emitted files can resolve their
    ``expected.extracted.*`` / ``expected.signatures.*`` imports against
    the BDR example package.
    """
    sys.path.insert(0, str(BDR_EXAMPLE_PKG_ROOT))
    try:
        # Ensure clean imports even if previous tests already loaded them
        for mod in list(sys.modules):
            if mod.startswith("expected."):
                del sys.modules[mod]
        from expected.runtimes.temporal import (  # type: ignore[import-not-found]
            activities,
            workflow,
        )

        yield workflow, activities
    finally:
        sys.path.remove(str(BDR_EXAMPLE_PKG_ROOT))


def test_emitted_workflow_registers_with_temporal(imported_modules) -> None:  # noqa: ANN001
    workflow_module, _ = imported_modules
    cls = workflow_module.BdrCampaignWorkflow
    # Temporal stores workflow metadata in a name-mangled attribute.
    assert hasattr(cls, "__temporal_workflow_definition")
    defn = cls.__temporal_workflow_definition
    assert defn.name.startswith("BdrCampaign_")


def test_emitted_activities_register_with_temporal(
    imported_modules,  # noqa: ANN001
    bdr_pipeline: Pipeline,
) -> None:
    _, activities_module = imported_modules
    registered: set[str] = set()
    for name in dir(activities_module):
        obj = getattr(activities_module, name)
        if callable(obj) and hasattr(obj, "__temporal_activity_definition"):
            registered.add(obj.__temporal_activity_definition.name)

    expected = {n.id for n in bdr_pipeline.nodes if n.kind is not NodeKind.HITL_GATE}
    missing = expected - registered
    extra = registered - expected
    assert not missing, f"Missing activity registrations: {missing}"
    assert not extra, f"Unexpected activity registrations: {extra}"


# ───────── Data-flow threading ─────────


def test_workflow_run_binds_pipeline_input(emit_result: dict[str, Path]) -> None:
    """The workflow entrypoint takes the pipeline input as its argument."""
    src = emit_result["workflow"].read_text()
    assert "async def run(self, pipeline_input: dict) -> dict:" in src


def test_entry_nodes_receive_pipeline_input(emit_result: dict[str, Path]) -> None:
    """Entry nodes bound to `pipeline.input` get the run argument, not {}."""
    src = emit_result["workflow"].read_text()
    # Both entry nodes run in a gather wave and take the whole brief.
    assert '"brief": pipeline_input,' in src


def test_downstream_nodes_receive_upstream_results(emit_result: dict[str, Path]) -> None:
    """The committed BDR bindings appear as real payload expressions."""
    src = emit_result["workflow"].read_text()
    # HITL gate output field → downstream activity payload.
    assert '"contacts": contact_review_gate_result["approved_contacts"],' in src
    # Node output field chains through the exclusion checks.
    assert '"contacts": hubspot_upsert_result["upserted"],' in src
    assert '"contacts": exclusion_check_dnc_result["passed"],' in src
    # Pipeline input field selection.
    assert '"target_quota": pipeline_input["target_quota"],' in src
    # Fan-in: the report node pulls from several sources.
    assert '"template_ids": create_sales_template_result["template_ids"],' in src
    # The placeholder TODO is gone for good.
    assert "TODO: pass real payload" not in src


def test_emitted_workflow_payloads_parse_as_dict_literals(
    emit_result: dict[str, Path], bdr_pipeline: Pipeline
) -> None:
    """AST-level check: every execute_activity call passes a dict whose
    keys match the node's declared ``inputs`` bindings."""
    src = emit_result["workflow"].read_text()
    tree = ast.parse(src)

    payload_keys_by_node: dict[str, set[str]] = {}
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call):
            continue
        func = call.func
        if not (isinstance(func, ast.Attribute) and func.attr == "execute_activity"):
            continue
        assert len(call.args) >= 2, "execute_activity must receive (name, payload)"
        name_arg, payload_arg = call.args[0], call.args[1]
        assert isinstance(name_arg, ast.Constant)
        assert isinstance(payload_arg, ast.Dict), (
            f"payload for {name_arg.value!r} must be a dict literal"
        )
        payload_keys_by_node[name_arg.value] = {
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


def test_node_without_inputs_gets_empty_payload(tmp_path: Path) -> None:
    """Back-compat: nodes with no ``inputs`` still receive {}."""
    from rote.ir import Node, Pipeline, PipelineInput

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
    src = TemporalAdapter().emit_workflow(pipeline)
    assert "            {}," in src


def test_emit_rejects_forward_reference() -> None:
    """A node whose inputs reference a later wave must fail at emit time,
    not as a NameError inside a running workflow."""
    from rote.ir import Edge, Node, Pipeline, PipelineInput

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
        TemporalAdapter().emit_workflow(pipeline)


def test_emit_rejects_reference_to_loop_body_node(bdr_pipeline: Pipeline) -> None:
    """Loop-body sub-nodes never bind a top-level result — referencing one
    from a top-level node must fail at emit time."""
    import copy

    pipeline = copy.deepcopy(bdr_pipeline)
    node = pipeline.node_by_id("hubspot_upsert")
    node.inputs = {"contacts": "vet_contact.output"}  # loop-body sub-node
    with pytest.raises(ValueError, match="no result available"):
        TemporalAdapter().emit_workflow(pipeline)


# ───────── Config variations ─────────


def test_custom_config_is_respected(bdr_pipeline: Pipeline) -> None:
    """Module paths in the config should appear in the emitted code.

    ``signatures_module`` only governs the *legacy* ``signature:`` form:
    BDR's judges carry both forms and must prefer the generated
    ``signatures/`` package, so the legacy import is asserted on a
    legacy-only judge instead.
    """
    cfg = TemporalAdapterConfig(
        types_module="myproj.types",
        extracted_module="myproj.extracted",
        signatures_module="myproj.signatures",
    )
    adapter = TemporalAdapter(cfg)
    activities_src = adapter.emit_activities(bdr_pipeline)
    assert "from myproj.extracted.taxonomy import" in activities_src
    # Both-forms judge: signature_spec wins (generated module).
    assert "from signatures.vet_contact import" in activities_src
    assert "from myproj.signatures.vet_contact import" not in activities_src

    legacy_judge = Node(
        id="grade_essay",
        kind=NodeKind.LLM_JUDGE,
        description="Grade an essay.",
        signature="signatures/grade_essay.py:GradeEssay",
    )
    legacy_src = adapter.emit_activities(mini_pipeline(legacy_judge))
    assert "from myproj.signatures.grade_essay import" in legacy_src


# ───────── Code-injection hardening ─────────


def test_malicious_description_cannot_break_out_of_docstring() -> None:
    """A node ``description`` engineered to close the emitted ``\"\"\"`` docstring
    and inject code must be neutralized: the emitted module still parses as
    Python and the payload survives only as inert escaped text in the docstring.

    Regression for the confirmed security finding — the first line of
    ``description`` was spliced raw into a docstring; ``safe_docstring_line``
    now escapes quotes and backslashes."""
    from rote.ir import Node, NodeKind, Pipeline, PipelineInput

    node = Node(
        id="n",
        kind=NodeKind.EXTERNAL_CALL,
        description='ok"""; import os; os.system("echo PWNED"); _junk = r"""',
        impl="extracted/x.py:run",
    )
    pipeline = Pipeline(
        name="t",
        input=PipelineInput(type="X", required=[]),
        nodes=[node],
        edges=[],
        entry_nodes=["n"],
        exit_nodes=["n"],
    )
    src = TemporalAdapter().emit_activities(pipeline)
    tree = ast.parse(src)  # must be valid Python

    # The injected statements must NOT exist as executable AST nodes — they
    # can only appear inside string/docstring literals.
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "system"
    ]
    assert not calls, "os.system injected as executable code"
    # And the raw (unescaped) breakout sequence must be gone.
    assert '"""; import os' not in src
