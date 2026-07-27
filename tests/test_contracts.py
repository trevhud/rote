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
