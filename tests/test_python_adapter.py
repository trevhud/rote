"""Tests for the raw Python adapter.

Three levels of validation:

1. **Refusal** — pipelines that need durable execution (any
   ``hitl_gate``) are rejected at emit time with an actionable error;
   the committed BDR pipeline is the canonical gated case.
2. **Emission** — the adapter produces files that parse as valid Python
   and satisfy the architectural invariants (MCP-free, visible retry
   loops, stdlib-only parallelism, data-flow threading).
3. **Execution** — the emitted script actually runs as a subprocess.
   See ``test_python_e2e.py`` for that.
"""

from __future__ import annotations

import ast
import copy
from pathlib import Path

import pytest

from rote.adapters import get_adapter
from rote.adapters.python import (
    PythonAdapter,
    PythonAdapterConfig,
    emit_main,
    emit_requirements,
)
from rote.ir import Edge, Node, NodeKind, Pipeline
from tests._helpers import mini_pipeline
from tests._python_fixture import build_gateless_pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
BDR_PIPELINE_YAML = REPO_ROOT / "examples" / "bdr-outreach" / "expected" / "pipeline.yaml"


# ───────── Fixtures ─────────


@pytest.fixture(scope="module")
def pipeline() -> Pipeline:
    return build_gateless_pipeline()


@pytest.fixture(scope="module")
def adapter() -> PythonAdapter:
    return PythonAdapter()


@pytest.fixture(scope="module")
def emit_result(
    adapter: PythonAdapter,
    pipeline: Pipeline,
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, Path]:
    return adapter.emit(pipeline, tmp_path_factory.mktemp("python-emit"))


@pytest.fixture(scope="module")
def main_src(emit_result: dict[str, Path]) -> str:
    return emit_result["main"].read_text(encoding="utf-8")


def test_constants_value_cannot_break_out_of_python_docstring(tmp_path: Path) -> None:
    """A constant *value* carrying ``\"\"\"`` is spliced into an extracted
    stub's docstring. Before the fix, ``{v!r}`` left the triple-quote intact,
    closing the docstring and injecting module-level code. The escaped repr
    must keep every emitted ``.py`` file parseable."""
    node = Node(
        id="n1",
        kind=NodeKind.PURE_FUNCTION,
        description="x",
        impl="extracted/a.py:go",
        constants={"evil": 'a """\nimport os\nos.system("id")\n""" b'},
    )
    result = PythonAdapter().emit(mini_pipeline(node), tmp_path)
    for path in result.values():
        if path.suffix == ".py":
            # ast.parse raises SyntaxError if the docstring was broken out of.
            ast.parse(path.read_text(encoding="utf-8"))
    stub = result["extracted/a"].read_text(encoding="utf-8")
    # The raw triple-quote is gone from the value; the escaped form survives.
    assert 'os.system("id")\n"""' not in stub


# ───────── Registry + derived IR property ─────────


def test_registry_dispatches_python() -> None:
    adapter = get_adapter("python")
    assert isinstance(adapter, PythonAdapter)


def test_requires_durable_execution_property(bdr_pipeline: Pipeline, pipeline: Pipeline) -> None:
    """The derived Pipeline property, not the adapter, knows about gates."""
    assert bdr_pipeline.requires_durable_execution is True
    assert pipeline.requires_durable_execution is False


# ───────── HITL refusal ─────────


def test_refuses_hitl_pipeline_with_actionable_error(
    bdr_pipeline: Pipeline, tmp_path: Path
) -> None:
    """BDR has two gates — the python adapter must refuse it at emit time,
    name the gates, explain why, and point at a durable runtime."""
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="cannot durably park") as excinfo:
        PythonAdapter().emit(bdr_pipeline, out)
    message = str(excinfo.value)
    assert "--runtime dbos" in message
    assert "hitl_gate" in message
    assert "contact_review_gate" in message
    assert "manual_enrollment_handoff" in message
    # Refusal happens before any file is written — no partial output.
    assert not out.exists()


def test_emit_main_also_refuses(bdr_pipeline: Pipeline) -> None:
    with pytest.raises(ValueError, match="--runtime dbos"):
        emit_main(bdr_pipeline)


# ───────── Emission basics ─────────


def test_emit_produces_expected_files(emit_result: dict[str, Path]) -> None:
    for key in (
        "main",
        "requirements",
        "README",
        "extracted/brief",
        "extracted/profile",
        "extracted/report",
        "extracted/research_loop",
        "signatures/grade",
    ):
        assert emit_result[key].exists(), f"missing {key}"


def test_emitted_files_are_valid_python(emit_result: dict[str, Path]) -> None:
    for label, path in emit_result.items():
        if path.suffix != ".py":
            continue
        src = path.read_text(encoding="utf-8")
        try:
            ast.parse(src)
        except SyntaxError as e:
            pytest.fail(f"{label} ({path.name}) is not valid Python: {e}")


def test_every_node_has_a_plain_function(main_src: str, pipeline: Pipeline) -> None:
    """Every node (including loop_body sub-nodes) gets a plain function —
    no decorators, no engine registration."""
    for node in pipeline.nodes:
        assert f"\ndef {node.id}(payload: dict) -> dict:" in main_src, f"missing {node.id}"
    assert "@" not in main_src.split('"""', 2)[2], "no decorator magic in the emitted script"


def test_main_entrypoint_takes_json_argv(main_src: str) -> None:
    assert 'if __name__ == "__main__":' in main_src
    assert "json.loads(sys.argv[1])" in main_src
    assert "run_pipeline(pipeline_input)" in main_src


def test_mandatory_nodes_marked_unconditional(main_src: str) -> None:
    assert "MANDATORY: this node was marked mandatory" in main_src


# ───────── Visible retry loops ─────────


def test_retry_policy_becomes_inline_loop(main_src: str) -> None:
    """fetch_profile (max=2, exponential) gets a visible for-loop with
    time.sleep backoff — the whole point of this target."""
    fn = main_src.split("def fetch_profile(", 1)[1].split("\n\n\ndef ", 1)[0]
    assert "attempts = 3  # 1 initial call + 2 retries (IR retry policy)" in fn
    assert "for attempt in range(1, attempts + 1):" in fn
    assert "time.sleep(RETRY_BASE_DELAY_SECONDS * 2 ** (attempt - 1))" in fn
    assert "# exponential backoff, visible and in-process" in fn
    # retry_on categories survive as guidance.
    assert "retry_on categories from the IR: rate_limit, network" in fn


def test_nodes_without_retry_have_no_loop(main_src: str) -> None:
    fn = main_src.split("def normalize_brief(", 1)[1].split("\n\n\ndef ", 1)[0]
    assert "for attempt" not in fn
    assert "time.sleep" not in fn


@pytest.mark.parametrize(
    ("backoff", "sleep_expr"),
    [
        ("constant", "time.sleep(RETRY_BASE_DELAY_SECONDS)"),
        ("linear", "time.sleep(RETRY_BASE_DELAY_SECONDS * attempt)"),
    ],
)
def test_constant_and_linear_backoff(backoff: str, sleep_expr: str) -> None:
    node = Node(
        id="flaky",
        kind=NodeKind.EXTERNAL_CALL,
        description="x",
        impl="extracted/x.py:y",
        retry={"max": 1, "backoff": backoff},
    )
    src = emit_main(mini_pipeline(node))
    assert sleep_expr in src


def test_retry_base_delay_is_configurable() -> None:
    node = Node(
        id="flaky",
        kind=NodeKind.EXTERNAL_CALL,
        description="x",
        impl="extracted/x.py:y",
        retry={"max": 1, "backoff": "exponential"},
    )
    src = emit_main(mini_pipeline(node), PythonAdapterConfig(retry_base_delay_seconds=0.01))
    assert "RETRY_BASE_DELAY_SECONDS = 0.01" in src


def test_timeout_surfaces_as_comment(main_src: str) -> None:
    """No clean stdlib per-step timeout for sync functions — document it."""
    assert "IR timeout '5m'" in main_src
    assert "no\n    # per-step timeout primitive" in main_src


# ───────── Imports stay minimal ─────────


def test_stdlib_only_imports(main_src: str) -> None:
    """The script's module-level imports are stdlib-only; extracted and
    signature modules are lazily imported inside the node functions."""
    tree = ast.parse(main_src)
    top_level = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    roots = set()
    for imp in top_level:
        if isinstance(imp, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in imp.names)
        elif imp.module is not None:
            roots.add(imp.module.split(".")[0])
    assert roots == {"__future__", "json", "sys", "time", "concurrent", "typing"}


def test_conditional_imports_dropped_when_unused() -> None:
    """A single-node pipeline with no retry needs neither time nor the
    thread pool — the emitted imports say so."""
    node = Node(id="only", kind=NodeKind.PURE_FUNCTION, description="x", impl="x.py:y")
    src = emit_main(mini_pipeline(node))
    assert "import time" not in src
    assert "ThreadPoolExecutor" not in src
    assert "RETRY_BASE_DELAY_SECONDS" not in src


# ───────── Waves + parallelism ─────────


def test_parallel_wave_uses_thread_pool(main_src: str) -> None:
    """Wave 1 (fetch_profile ∥ normalize_brief) must submit both nodes
    before joining either future — otherwise the wave serializes."""
    assert "with ThreadPoolExecutor() as pool:" in main_src
    sub_a = main_src.index("fetch_profile_future = pool.submit(")
    sub_b = main_src.index("normalize_brief_future = pool.submit(")
    join_a = main_src.index("fetch_profile_result = fetch_profile_future.result()")
    join_b = main_src.index("normalize_brief_result = normalize_brief_future.result()")
    assert max(sub_a, sub_b) < min(join_a, join_b)


def test_single_node_waves_call_directly(main_src: str) -> None:
    assert "research_loop_result = research_loop(" in main_src
    assert "pool.submit(research_loop" not in main_src
    assert "pool.submit(\n            research_loop" not in main_src


def test_loop_body_nodes_excluded_from_pipeline_body(main_src: str) -> None:
    """Loop-body sub-nodes exist as functions (testable in isolation) but
    run_pipeline never dispatches them — the parent loop does."""
    body = main_src.split("def run_pipeline(", 1)[1]
    assert "score_item(" not in body


# ───────── Data-flow threading ─────────


def test_pipeline_input_reaches_entry_nodes(main_src: str) -> None:
    assert '"topic": pipeline_input["topic"],' in main_src
    assert '"brief": pipeline_input,' in main_src


def test_upstream_results_thread_downstream(main_src: str) -> None:
    # Whole-output reference, field selection, and fan-in.
    assert '"profile": fetch_profile_result,' in main_src
    assert '"topic": normalize_brief_result["topic"],' in main_src
    assert '"depth": pipeline_input["depth"],' in main_src
    assert '"findings": research_loop_result["findings"],' in main_src
    assert '"grade": grade_result["grade"],' in main_src
    assert '"profile": fetch_profile_result["profile"],' in main_src


def test_pipeline_returns_exit_nodes(main_src: str, pipeline: Pipeline) -> None:
    for exit_id in pipeline.exit_nodes:
        assert f'"{exit_id}": {exit_id}_result,' in main_src


def test_payloads_parse_as_dict_literals(main_src: str, pipeline: Pipeline) -> None:
    """AST-level check: every node dispatch inside run_pipeline passes a
    dict literal whose keys match the node's declared ``inputs`` bindings."""
    tree = ast.parse(main_src)
    run_fn = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "run_pipeline"
    )

    node_ids = {n.id for n in pipeline.nodes}
    payload_keys_by_node: dict[str, set[str]] = {}
    for call in ast.walk(run_fn):
        if not isinstance(call, ast.Call):
            continue
        func = call.func
        if isinstance(func, ast.Attribute) and func.attr == "submit":
            # pool.submit(<node>, <payload>)
            assert len(call.args) == 2, "submit must receive (node, payload)"
            name_arg, payload_arg = call.args
            assert isinstance(name_arg, ast.Name)
            node_name = name_arg.id
        elif isinstance(func, ast.Name) and func.id in node_ids:
            # <node>(<payload>)
            assert len(call.args) == 1, f"call {func.id} must receive (payload)"
            node_name = func.id
            payload_arg = call.args[0]
        else:
            continue
        assert isinstance(payload_arg, ast.Dict), (
            f"payload for {node_name!r} must be a dict literal"
        )
        payload_keys_by_node[node_name] = {
            k.value for k in payload_arg.keys if isinstance(k, ast.Constant)
        }

    nested_ids = {sub for n in pipeline.nodes if n.loop_body for sub in n.loop_body}
    for node in pipeline.nodes:
        if node.id in nested_ids:
            continue
        expected_keys = set(node.inputs.keys()) if node.inputs else set()
        assert payload_keys_by_node[node.id] == expected_keys, (
            f"payload keys for {node.id!r} don't match its inputs bindings"
        )


def test_node_without_inputs_gets_empty_payload() -> None:
    node = Node(id="only", kind=NodeKind.PURE_FUNCTION, description="x", impl="x.py:y")
    src = emit_main(mini_pipeline(node))
    assert "only_result = only({})" in src


def test_emit_rejects_forward_reference() -> None:
    """A node whose inputs reference a later wave must fail at emit time,
    not as a NameError inside the running script."""
    forward = Pipeline(
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
        emit_main(forward)


def test_emit_rejects_reference_to_loop_body_node(pipeline: Pipeline) -> None:
    """Loop-body sub-nodes never bind a top-level result — referencing one
    from a top-level node must fail at emit time."""
    broken = copy.deepcopy(pipeline)
    broken.node_by_id("grade").inputs = {"findings": "score_item.output"}
    with pytest.raises(ValueError, match="no result available"):
        emit_main(broken)


# ───────── requirements.txt ─────────


def test_requirements_lists_only_needed_sdks(emit_result: dict[str, Path]) -> None:
    """One anthropic judge → pydantic + anthropic, nothing else."""
    lines = [
        line
        for line in emit_result["requirements"].read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    assert lines == ["pydantic>=2.7", "anthropic>=0.89"]


def test_requirements_includes_openai_when_needed() -> None:
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
        },
    )
    reqs = emit_requirements(mini_pipeline(judge))
    assert "openai>=" in reqs
    assert "anthropic" not in reqs


def test_requirements_empty_with_comment_when_no_judges() -> None:
    node = Node(id="only", kind=NodeKind.PURE_FUNCTION, description="x", impl="x.py:y")
    reqs = emit_requirements(mini_pipeline(node))
    assert reqs.strip(), "the no-judge requirements file still carries an explanation"
    assert all(line.startswith("#") for line in reqs.splitlines() if line)


# ───────── README ─────────


def test_readme_is_flush_left(emit_result: dict[str, Path]) -> None:
    """Regression for the dedent-after-interpolation bug: an indented
    README renders as one giant Markdown code block."""
    readme = emit_result["README"].read_text(encoding="utf-8")
    assert readme.startswith("# research-brief — plain Python runtime\n")
    assert "\n## Run\n" in readme
    assert "\n## What you give up vs. a durable runtime\n" in readme
    assert "--runtime dbos" in readme


# ───────── Generated signatures + extracted stubs ─────────


def test_signature_module_carries_python_adapter_identity(
    emit_result: dict[str, Path],
) -> None:
    """The heavy signature-emission machinery is shared with the DBOS
    adapter (deep coverage in test_dbos_adapter.py); here we pin the
    per-adapter identity strings and the typed surface main.py imports."""
    src = emit_result["signatures/grade"].read_text(encoding="utf-8")
    assert "Auto-generated by rote.adapters.python" in src
    assert "rote emit --runtime python" in src
    assert "class GradeInput(BaseModel):" in src
    assert "class GradeOutput(BaseModel):" in src
    assert "def forward(self, inputs: GradeInput) -> GradeOutput:" in src


def test_extracted_stubs_raise_not_implemented(emit_result: dict[str, Path]) -> None:
    for label, path in emit_result.items():
        if not label.startswith("extracted/") or label.endswith("__init__"):
            continue
        src = path.read_text(encoding="utf-8")
        assert "raise NotImplementedError(" in src, f"{label} lost its stub"
        assert "Auto-generated stubs by rote.adapters.python" in src


def test_agent_loop_stub_documents_tools_and_loop_body(
    emit_result: dict[str, Path],
) -> None:
    src = emit_result["extracted/research_loop"].read_text(encoding="utf-8")
    assert "web_search" in src
    assert "score_item" in src  # loop_body sub-node documented


def test_legacy_signature_path_fallback() -> None:
    """A judge with only the legacy path form imports the user-maintained
    module and bridges its async forward with asyncio.run."""
    judge = Node(
        id="grade_essay",
        kind=NodeKind.LLM_JUDGE,
        description="Grade an essay.",
        signature="signatures/grade_essay.py:GradeEssay",
    )
    src = emit_main(mini_pipeline(judge))
    assert "from signatures.grade_essay import (" in src
    assert "asyncio.run(GradeEssay().forward(GradeEssayInput(**payload)))" in src


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
    emitted script — main.py, extracted stubs, or generated signatures."""
    checked = 0
    for path in emit_result.values():
        if path.suffix == ".py":
            _assert_mcp_free(path)
            checked += 1
    assert checked >= 7  # main + __init__s + extracted + signatures
