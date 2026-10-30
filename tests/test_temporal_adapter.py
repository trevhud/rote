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
    emit_workflow,
)
from rote.ir import Node, NodeKind, Pipeline
from tests._helpers import mini_pipeline, payload_dict_keys

REPO_ROOT = Path(__file__).resolve().parent.parent
BDR_PIPELINE_YAML = REPO_ROOT / "examples" / "bdr-outreach" / "expected" / "pipeline.yaml"
BDR_EXAMPLE_PKG_ROOT = REPO_ROOT / "examples" / "bdr-outreach"


# ───────── Fixtures ─────────


#: The BDR example keeps its hand-written implementations in the
#: ``expected.extracted`` package, so its snapshot is emitted against that
#: instead of an emitted extracted/ package.
BDR_TEMPORAL_CONFIG = TemporalAdapterConfig(extracted_module="expected.extracted")


@pytest.fixture(scope="module")
def adapter() -> TemporalAdapter:
    return TemporalAdapter(BDR_TEMPORAL_CONFIG)


@pytest.fixture(scope="module")
def emit_result(
    adapter: TemporalAdapter, bdr_pipeline: Pipeline, tmp_path_factory: pytest.TempPathFactory
) -> dict[str, Path]:
    """Emit the BDR pipeline once per module, into a temp dir.

    Never into the committed snapshot: tests/test_committed_snapshots.py
    diffs a fresh emission against it, and the conftest guard fails the
    session if anything under examples/ changes.
    """
    return adapter.emit(bdr_pipeline, tmp_path_factory.mktemp("temporal-bdr"))


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
    """``source_skill`` is provenance, not behavior. The compiler
    re-points it per output location, so hashing it would mint a new
    workflow type (re-versioning in-flight workflows) on every
    re-compilation to a different directory — same rule as ``Node.source``."""
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

    MCP tool calls from the source skill must be compiled into direct
    API calls at emission time. If this assertion ever fails, it means
    the adapter is leaking the MCP abstraction into the runtime hot path.

    We parse the AST and walk only executable statements (imports,
    calls, attribute accesses) — comments and docstrings are allowed to
    *mention* MCP to explain the compilation history.
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
    assert '**_input_field(pipeline_input, "target_quota", "target_quota"),' in src
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
        payload_keys_by_node[name_arg.value] = {k for k in payload_dict_keys(payload_arg)}

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


# ───────── Retry policies survive parallelization ─────────


def _two_node_wave_pipeline() -> Pipeline:
    """Two entry nodes with no edge between them — one parallel wave."""
    return Pipeline(
        name="parallel",
        input={"type": "In", "required": []},
        nodes=[
            Node(
                id="fetch_a",
                kind=NodeKind.EXTERNAL_CALL,
                description="Fetch A.",
                impl="extracted/a.py:fetch_a",
                retry={"max": 5, "backoff": "exponential"},
            ),
            Node(
                id="fetch_b",
                kind=NodeKind.EXTERNAL_CALL,
                description="Fetch B.",
                impl="extracted/b.py:fetch_b",
                retry={"max": 2, "backoff": "linear"},
            ),
        ],
        edges=[],
        entry_nodes=["fetch_a", "fetch_b"],
        exit_nodes=["fetch_a", "fetch_b"],
    )


def test_parallel_wave_keeps_each_node_retry_policy() -> None:
    """A node must not lose its declared retry budget just for having a
    sibling in its wave. The gather branch once emitted only the timeout,
    silently dropping retries from exactly the parallel fetch waves where
    flaky network calls live."""
    pipeline = _two_node_wave_pipeline()
    src = TemporalAdapter().emit_workflow(pipeline)

    assert "asyncio.gather(" in src, "expected the parallel branch"
    # max=5 → maximum_attempts=6 (Temporal counts initial + retries)
    assert "RetryPolicy(maximum_attempts=6, backoff_coefficient=2.0)" in src
    assert "RetryPolicy(maximum_attempts=3, backoff_coefficient=1.0)" in src
    ast.parse(src)


def test_a_node_with_no_retry_gets_one_attempt_never_temporals_unbounded_default() -> None:
    """No ``retry:`` means one attempt, as on dbos, dbos-ts and python. With
    no ``retry_policy`` Temporal retries forever, so a paid judge that kept
    missing its schema would bill without end. Every activity carries a
    policy; a declared one keeps its own budget (the negative half)."""
    pipeline = _two_node_wave_pipeline()
    pipeline.nodes[0].retry = None
    src = TemporalAdapter().emit_workflow(pipeline)

    assert "asyncio.gather(" in src
    # Every execute_activity call carries a policy.
    assert src.count("workflow.execute_activity(") == src.count("retry_policy=") == 2
    # fetch_a (no retry:) gets exactly one attempt...
    fetch_a = src[src.index('"fetch_a"') : src.index('"fetch_b"')]
    assert "retry_policy=RetryPolicy(maximum_attempts=1)," in fetch_a
    # ...and fetch_b keeps its declared max=2, not the one-attempt default.
    fetch_b = src[src.index('"fetch_b"') :]
    assert "RetryPolicy(maximum_attempts=3, backoff_coefficient=1.0)" in fetch_b
    assert "maximum_attempts=1)" not in fetch_b
    ast.parse(src)


def test_agent_loop_with_tools_does_not_retry_a_missing_mcp_endpoint() -> None:
    """Its tool servers resolve at run time; Temporal matches non-retryable
    types by exception class name, so RoteMcpConfigError fails at once.
    A loop without tools, or any other node, keeps a plain policy."""
    from tests._helpers import unconfigured_mcp_pipelines

    src = TemporalAdapter().emit_workflow(unconfigured_mcp_pipelines("t")["research"])
    assert (
        "RetryPolicy(maximum_attempts=4, backoff_coefficient=2.0, "
        'non_retryable_error_types=["RoteMcpConfigError"])'
    ) in src
    plain = TemporalAdapter().emit_workflow(_two_node_wave_pipeline())
    assert "non_retryable_error_types" not in plain


# ───────── MCP-only external_call refusal ─────────


def _mcp_only_pipeline() -> Pipeline:
    """`impl` or `mcp` satisfies external_call in the IR — this is the
    legal shape the temporal adapter cannot emit."""
    return mini_pipeline(
        Node(
            id="search_docs",
            kind=NodeKind.EXTERNAL_CALL,
            description="Search the docs over MCP.",
            mcp={"server": "docs", "tool": "search"},
        )
    )


def test_refuses_mcp_only_node_with_actionable_error(tmp_path: Path) -> None:
    """Valid IR the adapter can't serve must produce an explanatory
    refusal, not a bare AssertionError from a type-narrowing assert."""
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="no MCP backend") as excinfo:
        TemporalAdapter().emit(_mcp_only_pipeline(), out)
    message = str(excinfo.value)
    assert "search_docs" in message
    assert "--runtime dbos" in message
    # Refusal happens before any file is written — no partial output.
    assert not out.exists()


def test_emit_activities_also_refuses_mcp_only_node() -> None:
    with pytest.raises(ValueError, match="no MCP backend"):
        TemporalAdapter().emit_activities(_mcp_only_pipeline())


def test_per_node_timeout_reaches_the_activity_call() -> None:
    """A node's declared `timeout:` must beat the adapter default.

    Found by mutation testing: making `_activity_timeout` return the
    default unconditionally left the entire suite green. A pipeline that
    declares a 30-second budget for a fast step and gets the 5-minute
    default instead hangs five times too long on a wedged call, with
    nothing in the emitted code hinting why.

    Both halves matter — asserting only that "30s" appears would still
    pass if the adapter emitted it for every node.
    """
    pinned = Node(
        id="fast_step",
        kind=NodeKind.PURE_FUNCTION,
        description="d",
        impl="m.py:fast",
        timeout="30s",
    )
    defaulted = Node(
        id="slow_step",
        kind=NodeKind.PURE_FUNCTION,
        description="d",
        impl="m.py:slow",
    )
    pipeline = Pipeline(
        name="timeouts",
        input={"type": "In", "required": [], "optional": []},
        nodes=[pinned, defaulted],
        edges=[{"from": "fast_step", "to": "slow_step"}],
        entry_nodes=["fast_step"],
        exit_nodes=["slow_step"],
    )
    cfg = TemporalAdapterConfig(default_activity_timeout="5m")
    src = emit_workflow(pipeline, cfg)

    fast_call = src[src.index('"fast_step"') : src.index('"slow_step"')]
    slow_call = src[src.index('"slow_step"') :]
    assert '_parse_minutes("30s")' in fast_call, (
        "a node's declared timeout must reach its execute_activity call"
    )
    assert '_parse_minutes("5m")' in slow_call, (
        "a node without a timeout must fall back to the adapter default"
    )
    # The pinned node must NOT also carry the default.
    assert '_parse_minutes("5m")' not in fast_call


# ───────── extracted/: emitted, sourced, and reported ─────────


def _two_module_pipeline() -> Pipeline:
    """A sync step feeding an async one, each in its own extracted module."""
    return Pipeline.model_validate(
        {
            "name": "chain",
            "version": "0.1.0",
            "input": {"type": "In", "required": ["x"], "optional": []},
            "nodes": [
                {
                    "id": "double",
                    "kind": "pure_function",
                    "description": "Double x.",
                    "impl": "extracted/math_ops.py:double",
                    "inputs": {"x": "pipeline.input.x"},
                },
                {
                    "id": "add_one",
                    "kind": "external_call",
                    "description": "Add one to y.",
                    "impl": "extracted/remote_ops.py:add_one",
                    "inputs": {"y": "double.output.y"},
                },
            ],
            "edges": [{"from": "double", "to": "add_one"}],
            "entry_nodes": ["double"],
            "exit_nodes": ["add_one"],
        }
    )


def test_emit_without_real_modules_writes_and_reports_stubs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Temporal used to emit no extracted/ at all: activities imported
    ``expected.extracted.*``, which only resolves inside the BDR example,
    and ``rote emit --json`` reported zero stubs for a dir that could not
    import a single module."""
    import json

    from rote.cli import main as cli_main
    from rote.ir import save_pipeline

    (tmp_path / "compiled").mkdir()
    pipeline_yaml = save_pipeline(_two_module_pipeline(), tmp_path / "compiled" / "pipeline.yaml")
    out = tmp_path / "out"

    rc = cli_main(
        ["emit", str(pipeline_yaml), "--runtime", "temporal", "--out", str(out), "--json"]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    expected_stubs = sorted(
        str(out.resolve() / "extracted" / name) for name in ("math_ops.py", "remote_ops.py")
    )
    assert sorted(str(Path(p).resolve()) for p in payload["unimplemented_stubs"]) == (
        expected_stubs
    )
    activities = (out / "activities.py").read_text(encoding="utf-8")
    assert "from extracted.math_ops import double" in activities
    assert "from extracted.remote_ops import add_one" in activities
    assert "expected.extracted" not in activities


def test_agent_loop_with_tools_gets_the_mcp_helper(tmp_path: Path) -> None:
    """The inference helper resolves an agent loop's MCP tools through
    ``extracted/_rote_mcp.py``. Temporal emitted no extracted/ at all, so a
    loop that declared tools died with "extracted/_rote_mcp.py is missing"."""
    from rote.mcp import _runtime_helper as mcp_helper

    loop = Node(
        id="research",
        kind=NodeKind.AGENT_LOOP,
        description="Research.",
        tools=["web_search"],
    )
    written = TemporalAdapter().emit(mini_pipeline(loop), tmp_path)

    assert written["extracted/_rote_mcp"].read_text(encoding="utf-8") == Path(
        mcp_helper.__file__
    ).read_text(encoding="utf-8")


def test_emitted_app_runs_the_agent_written_modules(tmp_path: Path) -> None:
    """Real modules beside pipeline.yaml are emitted verbatim and actually
    run: a plain function (the shape agent-written modules take) and a
    coroutine function both complete, with the payload threaded through."""
    import asyncio
    from uuid import uuid4

    from temporalio.testing import WorkflowEnvironment
    from temporalio.worker import UnsandboxedWorkflowRunner, Worker

    from rote.adapters import get_adapter
    from rote.runners.temporal import _load_app_modules, start_with_retries

    source = tmp_path / "compiled"
    (source / "extracted").mkdir(parents=True)
    sync_impl = "def double(x):\n    return {'y': x * 2}\n"
    async_impl = "async def add_one(y):\n    return {'z': y + 1}\n"
    (source / "extracted" / "math_ops.py").write_text(sync_impl, encoding="utf-8")
    (source / "extracted" / "remote_ops.py").write_text(async_impl, encoding="utf-8")
    out = tmp_path / "out"

    written = get_adapter("temporal", extracted_source_dir=source).emit(_two_module_pipeline(), out)

    assert written["extracted/math_ops"].read_text(encoding="utf-8") == sync_impl
    assert written["extracted/remote_ops"].read_text(encoding="utf-8") == async_impl

    workflow_class, activities = _load_app_modules(out)

    async def _run() -> dict:
        async with await start_with_retries(WorkflowEnvironment.start_time_skipping) as env:
            queue = f"chain-{uuid4()}"
            async with Worker(
                env.client,
                task_queue=queue,
                workflows=[workflow_class],
                activities=activities,
                workflow_runner=UnsandboxedWorkflowRunner(),
            ):
                # Activities retry forever by default, so a broken call
                # would hang the test instead of failing it.
                return await asyncio.wait_for(
                    env.client.execute_workflow(
                        workflow_class.run,  # type: ignore[attr-defined]
                        {"x": 3},
                        id=f"chain-{uuid4()}",
                        task_queue=queue,
                    ),
                    timeout=30,
                )

    try:
        result = asyncio.run(_run())
    finally:
        sys.path.remove(str(out))
        for name in [
            m
            for m in sys.modules
            if m.split(".")[0] in ("workflow", "activities", "extracted", "signatures")
        ]:
            del sys.modules[name]

    assert result == {"add_one": {"z": 7}}
