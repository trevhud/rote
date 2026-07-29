"""Tests for the static IR ↔ artifact contract checker.

Two things these guard. First, that the checker catches the failure it
exists for: an emitted ``func(**payload)`` that cannot possibly succeed.
Second — and the reason half of them are here — that it stays QUIET. A
checker that reports on healthy compilations gets ignored, and then it
catches nothing at all. Every "no findings" case below is a shape the
first draft flagged wrongly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rote.contracts import ContractFinding, check_contracts
from rote.ir import Node, NodeKind, Pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent


# ───────── Helpers ─────────


def _pipeline(*nodes: Node) -> Pipeline:
    return Pipeline(
        name="contracts",
        input={"type": "In", "required": []},
        nodes=list(nodes),
        edges=[],
        entry_nodes=[nodes[0].id],
        exit_nodes=[nodes[-1].id],
    )


def _write(root: Path, rel: str, src: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(src, encoding="utf-8")


def _impl_node(node_id: str = "n", **kw: object) -> Node:
    # Default to declaring an input so data-flow checking is active — a
    # pipeline with no `inputs:` anywhere takes the legacy path instead.
    kw.setdefault("inputs", {"contacts": "pipeline.input"})
    return Node(
        id=node_id,
        kind=NodeKind.PURE_FUNCTION,
        description="A step.",
        impl=f"extracted/mod.py:{node_id}",
        **kw,  # type: ignore[arg-type]
    )


def _kinds(findings: list[ContractFinding]) -> list[str]:
    return sorted(f.kind for f in findings)


# ───────── Symbol existence ─────────


def test_missing_symbol_is_an_error(tmp_path: Path) -> None:
    _write(tmp_path, "extracted/mod.py", "def something_else():\n    ...\n")
    findings = check_contracts(_pipeline(_impl_node()), tmp_path)
    assert _kinds(findings) == ["missing_symbol"]
    # The message must name what IS there — that is what makes it fixable.
    assert "something_else" in findings[0].message


def test_syntax_error_is_reported_once_and_stops_that_module(tmp_path: Path) -> None:
    _write(tmp_path, "extracted/mod.py", "def n(:\n")
    findings = check_contracts(_pipeline(_impl_node()), tmp_path)
    assert _kinds(findings) == ["syntax_error"]


def test_absent_module_is_not_a_finding(tmp_path: Path) -> None:
    """The documented stub path. Emission synthesises the symbol."""
    assert check_contracts(_pipeline(_impl_node()), tmp_path) == []


def test_module_level_assignment_counts_as_a_definition(tmp_path: Path) -> None:
    """A node may point at a partial or a callable object, not just a def."""
    _write(tmp_path, "extracted/mod.py", "from functools import partial\nn = partial(print)\n")
    assert check_contracts(_pipeline(_impl_node()), tmp_path) == []


# ───────── Call signature vs payload ─────────


def test_required_parameter_absent_from_payload(tmp_path: Path) -> None:
    _write(tmp_path, "extracted/mod.py", "def n(contacts, dnc_list_id):\n    ...\n")
    node = _impl_node(inputs={"contacts": "pipeline.input"})
    findings = check_contracts(_pipeline(node), tmp_path)
    assert _kinds(findings) == ["missing_argument"]
    assert "dnc_list_id" in findings[0].message


def test_payload_key_the_function_cannot_accept(tmp_path: Path) -> None:
    _write(tmp_path, "extracted/mod.py", "def n(campaign_name):\n    ...\n")
    node = _impl_node(inputs={"campaign_name": "pipeline.input", "contacts": "pipeline.input"})
    findings = check_contracts(_pipeline(node), tmp_path)
    assert _kinds(findings) == ["unexpected_argument"]
    assert "contacts" in findings[0].message


def test_defaulted_parameters_are_not_required(tmp_path: Path) -> None:
    _write(tmp_path, "extracted/mod.py", "def n(contacts, limit=10):\n    ...\n")
    node = _impl_node(inputs={"contacts": "pipeline.input"})
    assert check_contracts(_pipeline(node), tmp_path) == []


def test_var_keyword_accepts_anything(tmp_path: Path) -> None:
    _write(tmp_path, "extracted/mod.py", "def n(**kwargs):\n    ...\n")
    node = _impl_node(inputs={"anything": "pipeline.input"})
    assert check_contracts(_pipeline(node), tmp_path) == []


def test_required_keyword_only_parameter_is_checked(tmp_path: Path) -> None:
    _write(tmp_path, "extracted/mod.py", "def n(*, contacts, token):\n    ...\n")
    node = _impl_node(inputs={"contacts": "pipeline.input"})
    findings = check_contracts(_pipeline(node), tmp_path)
    assert _kinds(findings) == ["missing_argument"]
    assert "token" in findings[0].message


def test_defaulted_keyword_only_parameter_is_not_required(tmp_path: Path) -> None:
    _write(tmp_path, "extracted/mod.py", "def n(*, contacts, token=None):\n    ...\n")
    node = _impl_node(inputs={"contacts": "pipeline.input"})
    assert check_contracts(_pipeline(node), tmp_path) == []


def test_positional_only_parameter_can_never_be_filled(tmp_path: Path) -> None:
    """`func(**payload)` is keyword dispatch — a `/` marker makes the
    parameter unreachable no matter what the node binds."""
    _write(tmp_path, "extracted/mod.py", "def n(contacts, /):\n    ...\n")
    node = _impl_node(inputs={"contacts": "pipeline.input"})
    findings = check_contracts(_pipeline(node), tmp_path)
    assert "positional_only_param" in _kinds(findings)
    # Not ALSO reported as missing — it is one defect, not two.
    assert "missing_argument" not in _kinds(findings)


def test_async_implementations_are_checked_too(tmp_path: Path) -> None:
    _write(tmp_path, "extracted/mod.py", "async def n(contacts, token):\n    ...\n")
    node = _impl_node(inputs={"contacts": "pipeline.input"})
    assert _kinds(check_contracts(_pipeline(node), tmp_path)) == ["missing_argument"]


# ───────── Shapes that must stay quiet ─────────


def test_loop_body_subnode_skips_the_signature_check(tmp_path: Path) -> None:
    """Sub-nodes are called by hand-written loop code, not by a generated
    payload, so their `inputs` say nothing about the call site."""
    _write(tmp_path, "extracted/mod.py", "def sub(contacts, cursor):\n    ...\n")
    # The real emitted agent_loop stub shape.
    _write(tmp_path, "extracted/loop.py", "def loop(**payload):\n    ...\n")
    sub = Node(
        id="sub",
        kind=NodeKind.EXTERNAL_CALL,
        description="Per-iteration call.",
        impl="extracted/mod.py:sub",
    )
    loop = Node(
        id="loop",
        kind=NodeKind.AGENT_LOOP,
        description="Drives sub.",
        tools=["t"],
        loop_body=["sub"],
        inputs={"seed": "pipeline.input"},
    )
    # `loop` itself has inputs, so data-flow checking is active.
    findings = check_contracts(_pipeline(loop, sub), tmp_path)
    assert findings == []


def test_mcp_only_node_has_no_module_to_check(tmp_path: Path) -> None:
    node = Node(
        id="search",
        kind=NodeKind.EXTERNAL_CALL,
        description="Bare MCP tool call.",
        mcp={"server": "docs", "tool": "search"},
        inputs={"q": "pipeline.input"},
    )
    assert check_contracts(_pipeline(node), tmp_path) == []


def test_legacy_ir_without_dataflow_says_it_once(tmp_path: Path) -> None:
    """Pre-`inputs:` IRs emit `func()` everywhere. Reporting every impl
    separately would be a wall of findings all saying one thing."""
    _write(
        tmp_path,
        "extracted/mod.py",
        "def a(x, y):\n    ...\ndef b(p, q):\n    ...\ndef c(m):\n    ...\n",
    )
    nodes = [
        Node(
            id=nid,
            kind=NodeKind.PURE_FUNCTION,
            description="Step.",
            impl=f"extracted/mod.py:{nid}",
        )
        for nid in ("a", "b", "c")
    ]
    findings = check_contracts(_pipeline(*nodes), tmp_path)
    assert _kinds(findings) == ["no_dataflow"]
    assert findings[0].severity == "warning"


def test_judge_signature_checks_existence_but_not_call_shape(tmp_path: Path) -> None:
    """Judges are instantiated and driven through forward(), not called
    with the node payload."""
    _write(tmp_path, "signatures/vet.py", "class VetContact:\n    pass\n")
    node = Node(
        id="vet",
        kind=NodeKind.LLM_JUDGE,
        description="Judge.",
        signature="signatures/vet.py:VetContact",
        inputs={"contact": "pipeline.input"},
    )
    assert check_contracts(_pipeline(node), tmp_path) == []


def test_missing_judge_class_is_still_caught(tmp_path: Path) -> None:
    _write(tmp_path, "signatures/vet.py", "class SomethingElse:\n    pass\n")
    node = Node(
        id="vet",
        kind=NodeKind.LLM_JUDGE,
        description="Judge.",
        signature="signatures/vet.py:VetContact",
        inputs={"contact": "pipeline.input"},
    )
    assert _kinds(check_contracts(_pipeline(node), tmp_path)) == ["missing_symbol"]


# ───────── Judge prompt templates ─────────


def _spec_judge(prompt: str, schema: dict[str, object]) -> Node:
    return Node(
        id="judge",
        kind=NodeKind.LLM_JUDGE,
        description="Judge.",
        signature_spec={
            "input_schema": schema,
            "output_schema": {"type": "object", "properties": {"ok": {"type": "boolean"}}},
            "prompt": prompt,
        },
    )


def test_unresolvable_prompt_variable_is_an_error() -> None:
    """Emitted judges raise on this — but at judge-call time, in
    production, on a pipeline that compiled and deployed cleanly."""
    node = _spec_judge(
        "Rate {{ contact.nmae }} please.",
        {
            "type": "object",
            "properties": {
                "contact": {"type": "object", "properties": {"name": {"type": "string"}}}
            },
        },
    )
    findings = check_contracts(_pipeline(node), Path("/nonexistent"))
    assert _kinds(findings) == ["unresolved_prompt_var"]
    assert "contact.nmae" in findings[0].message


def test_prompt_variable_present_in_schema_is_fine() -> None:
    node = _spec_judge(
        "Rate {{ contact }} for {{ campaign }}.",
        {
            "type": "object",
            "properties": {"contact": {"type": "string"}, "campaign": {"type": "string"}},
        },
    )
    assert check_contracts(_pipeline(node), Path("/nonexistent")) == []


def test_prompt_variable_resolves_through_nested_properties() -> None:
    node = _spec_judge(
        "Rate {{ contact.title }}.",
        {
            "type": "object",
            "properties": {
                "contact": {"type": "object", "properties": {"title": {"type": "string"}}}
            },
        },
    )
    assert check_contracts(_pipeline(node), Path("/nonexistent")) == []


def test_path_beyond_the_described_schema_is_accepted() -> None:
    """The schema stopped making claims about the shape, so neither can we
    — flagging here would punish a legitimately free-form object."""
    node = _spec_judge(
        "Rate {{ contact.deeply.nested.thing }}.",
        {"type": "object", "properties": {"contact": {"type": "object"}}},
    )
    assert check_contracts(_pipeline(node), Path("/nonexistent")) == []


# ───────── Regression: the canonical snapshot ─────────


def test_committed_bdr_snapshot_has_the_two_known_contract_breaks() -> None:
    """The newest committed BDR compile carries two real breaks, both of
    which crash on the first run:

    * `hubspot_create_list` binds `contacts`, but `create_campaign_list`
      takes only `campaign_name`.
    * `check_do_not_contact` requires a `dnc_list_id` that no node
      produces — its own docstring says "the workflow obtains it during
      setup", and the IR has no such step.

    Pinned deliberately. When the rubric is fixed and the snapshot
    recompiled, this test should fail and be updated to assert zero —
    that is the signal the fix landed.
    """
    run = REPO_ROOT / "examples" / "bdr-outreach" / "runs" / "2026-07-18-mcp-bindings"
    from rote.ir import load_pipeline

    findings = check_contracts(load_pipeline(run / "pipeline.yaml"), run)
    by_node = {f.node_id: f.kind for f in findings}
    assert by_node == {
        "hubspot_create_list": "unexpected_argument",
        "exclusion_check_dnc": "missing_argument",
    }


def test_healthy_snapshots_report_nothing() -> None:
    """Three real compiles with no contract breaks. If a change here starts
    reporting on these, it is producing false positives."""
    from rote.ir import load_pipeline

    clean = [
        REPO_ROOT / "examples" / "deal-monitor" / "runs" / "2026-07-18-rubric-v2",
        REPO_ROOT / "examples" / "ops-report" / "runs" / "2026-07-18-rubric-v2",
        REPO_ROOT / "examples" / "invoice-push" / "runs" / "2026-07-18-first-agent-run",
    ]
    for run in clean:
        findings = check_contracts(load_pipeline(run / "pipeline.yaml"), run)
        assert findings == [], f"{run.name}: {[f.message for f in findings]}"


# ───────── Differential: the checker vs the interpreter ─────────
#
# The signature rules are fiddly enough that reasoning about them is how
# bugs get in — the first draft passed `def f(a=1, /)` with `{"a": 2}` as
# clean, where CPython raises. So don't reason: build the function, call
# it, and require the checker's verdict to match what actually happened.

#: ``(source, payload)``. Every shape the emitted `func(**payload)` call
#: site can meet, including the ones that only differ under `**kwargs`.
_SIGNATURE_CASES: list[tuple[str, dict[str, int]]] = [
    # Ordinary
    ("def f(a, b): ...", {"a": 1, "b": 2}),
    ("def f(a, b): ...", {"a": 1}),
    ("def f(a): ...", {"a": 1, "b": 2}),
    ("def f(): ...", {}),
    ("def f(): ...", {"a": 1}),
    # Defaults
    ("def f(a, b=2): ...", {"a": 1}),
    ("def f(a, b=2): ...", {"a": 1, "b": 3}),
    ("def f(a=1, b=2): ...", {}),
    # Keyword-only
    ("def f(*, a): ...", {"a": 1}),
    ("def f(*, a): ...", {}),
    ("def f(*, a=1): ...", {}),
    ("def f(a, *, b): ...", {"a": 1}),
    ("def f(a, *, b): ...", {"a": 1, "b": 2}),
    # Var-positional / var-keyword
    ("def f(a, *args): ...", {"a": 1}),
    ("def f(**kw): ...", {"a": 1, "b": 2}),
    ("def f(a, **kw): ...", {"a": 1, "z": 9}),
    ("def f(a, **kw): ...", {"z": 9}),
    ("def f(*args, **kw): ...", {"a": 1}),
    # Positional-only — the family that broke the first draft
    ("def f(a, /): ...", {"a": 1}),
    ("def f(a, /): ...", {}),
    ("def f(a=1, /): ...", {}),
    ("def f(a=1, /): ...", {"a": 2}),
    ("def f(a, /, b): ...", {"a": 1, "b": 2}),
    ("def f(a, /, b): ...", {"b": 2}),
    ("def f(a, /, **kw): ...", {"a": 1}),
    ("def f(a=1, /, **kw): ...", {"a": 2}),
    ("def f(a, /, *, b): ...", {"b": 2}),
]


def _python_accepts(source: str, payload: dict[str, int]) -> bool:
    namespace: dict[str, object] = {}
    exec(source, namespace)  # noqa: S102 - fixed literals defined above
    fn = namespace["f"]
    assert callable(fn)
    try:
        fn(**payload)
    except TypeError:
        return False
    return True


def test_checker_verdict_matches_cpython_for_every_signature_shape() -> None:
    import ast as _ast

    from rote.contracts import _check_call_signature

    disagreements: list[str] = []
    for source, payload in _SIGNATURE_CASES:
        node = _ast.parse(source).body[0]
        assert isinstance(node, _ast.FunctionDef)
        checker_says_ok = not _check_call_signature(node, set(payload))
        python_says_ok = _python_accepts(source, payload)
        if checker_says_ok != python_says_ok:
            disagreements.append(
                f"{source!r} with {payload!r}: "
                f"checker_ok={checker_says_ok} cpython_ok={python_says_ok}"
            )
    assert not disagreements, "checker disagrees with CPython:\n  " + "\n  ".join(disagreements)


def test_the_differential_cases_actually_cover_both_verdicts() -> None:
    """A table where everything passes would prove nothing."""
    accepted = [c for c in _SIGNATURE_CASES if _python_accepts(*c)]
    rejected = [c for c in _SIGNATURE_CASES if not _python_accepts(*c)]
    assert len(accepted) >= 10, "not enough passing shapes to catch over-reporting"
    assert len(rejected) >= 10, "not enough failing shapes to catch under-reporting"


# ───────── Empirical: the finding predicts a real failure ─────────


def test_bdr_finding_reproduces_as_a_real_typeerror() -> None:
    """Import the agent's actual module and make the call the adapter emits.

    A structural finding is only worth acting on if the runtime really
    fails. Both BDR breaks are checked against the real committed source
    (stdlib-only imports, so this is safe to exec), reproducing the exact
    `func(**payload)` that dbos / python / temporal all emit.
    """
    import importlib.util

    run = REPO_ROOT / "examples" / "bdr-outreach" / "runs" / "2026-07-18-mcp-bindings"

    def _load(rel: str, name: str) -> object:
        spec = importlib.util.spec_from_file_location(name, run / rel)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    hubspot = _load("extracted/hubspot.py", "_bdr_hubspot")
    exclusion = _load("extracted/exclusion_checks.py", "_bdr_exclusion")

    # hubspot_create_list binds {campaign_name, contacts}; impl takes one.
    with pytest.raises(TypeError, match="contacts"):
        hubspot.create_campaign_list(  # type: ignore[attr-defined]
            **{"campaign_name": "Orladeyo", "contacts": []}
        )

    # exclusion_check_dnc binds only {contacts}; impl also requires dnc_list_id.
    with pytest.raises(TypeError, match="dnc_list_id"):
        exclusion.check_do_not_contact(**{"contacts": []})  # type: ignore[attr-defined]


# ───────── The premise the whole checker rests on ─────────


def test_python_adapters_really_call_impls_with_double_star_payload() -> None:
    """Every signature finding assumes the emitted call is
    `func(**payload)` with the node's `inputs` as the keys. If an adapter
    ever changes that convention, this checker silently starts lying —
    so assert the convention itself."""
    from rote.adapters import get_adapter
    from rote.ir import load_pipeline

    pipeline = load_pipeline(
        REPO_ROOT / "examples" / "deal-monitor" / "runs" / "2026-07-18-rubric-v2" / "pipeline.yaml"
    )
    import tempfile

    for runtime in ("dbos", "python", "temporal"):
        with tempfile.TemporaryDirectory() as tmp:
            written = get_adapter(runtime).emit(pipeline, tmp)
            main = next(
                path for label, path in written.items() if path.name in ("main.py", "activities.py")
            )
            source = main.read_text(encoding="utf-8")
            assert "(**payload)" in source, (
                f"{runtime} no longer calls impls with **payload — "
                "rote.contracts._check_call_signature models the old convention"
            )
