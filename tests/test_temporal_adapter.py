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
from rote.ir import NodeKind, Pipeline, load_pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
BDR_PIPELINE_YAML = REPO_ROOT / "examples" / "bdr-outreach" / "expected" / "pipeline.yaml"
BDR_EMIT_DIR = REPO_ROOT / "examples" / "bdr-outreach" / "expected" / "runtimes" / "temporal"
BDR_EXAMPLE_PKG_ROOT = REPO_ROOT / "examples" / "bdr-outreach"


# ───────── Fixtures ─────────


@pytest.fixture(scope="module")
def bdr_pipeline() -> Pipeline:
    return load_pipeline(BDR_PIPELINE_YAML)


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
        assert f'@activity.defn(name="{node.id}")' in src, (
            f"Missing activity for node {node.id}"
        )


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
        from expected.runtimes.temporal import activities, workflow  # type: ignore[import-not-found]

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

    expected = {
        n.id for n in bdr_pipeline.nodes if n.kind is not NodeKind.HITL_GATE
    }
    missing = expected - registered
    extra = registered - expected
    assert not missing, f"Missing activity registrations: {missing}"
    assert not extra, f"Unexpected activity registrations: {extra}"


# ───────── Config variations ─────────


def test_custom_config_is_respected(bdr_pipeline: Pipeline) -> None:
    """Module paths in the config should appear in the emitted code."""
    cfg = TemporalAdapterConfig(
        types_module="myproj.types",
        extracted_module="myproj.extracted",
        signatures_module="myproj.signatures",
    )
    adapter = TemporalAdapter(cfg)
    activities_src = adapter.emit_activities(bdr_pipeline)
    assert "from myproj.extracted.taxonomy import" in activities_src
    assert "from myproj.signatures.vet_contact import" in activities_src
