"""MCP-backed external_call: IR binding + DBOS emission.

Covers the ``mcp`` backend that lets an ``external_call`` node call the
MCP tool the source skill used (over Streamable HTTP) instead of a
hand-written vendor client. The runtime end-to-end run against a live
mock MCP server lives in the slow suite (``test_mcp_e2e.py``); this file
is fast — pure IR validation + template-substitution assertions.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from pydantic import ValidationError

from rote.adapters import get_adapter
from rote.adapters.dbos import DbosAdapterConfig, emit_main
from rote.cli import main as cli_main
from rote.ir import MCPBinding, Node, NodeKind, Pipeline, PipelineInput


def _pipeline(*, impl: str | None = None, mcp: MCPBinding | None = None) -> Pipeline:
    node = Node(
        id="enrich",
        kind=NodeKind.EXTERNAL_CALL,
        description="Enrich a contact via the vendor MCP server.",
        input={"contact_id": "str"},
        inputs={"contact_id": "pipeline.input.contact_id"},
        output="dict",
        impl=impl,
        mcp=mcp,
    )
    return Pipeline(
        name="mcp_demo",
        description="Minimal MCP-backed pipeline.",
        input=PipelineInput(type="EnrichRequest", required=["contact_id"]),
        nodes=[node],
        edges=[],
        entry_nodes=["enrich"],
        exit_nodes=["enrich"],
    )


_BINDING = MCPBinding(server="vendor", tool="enrich_contact", args={"contact_id": "contact_id"})


# ───────── IR validation ─────────


def test_external_call_accepts_mcp_without_impl() -> None:
    """An MCP-backed external_call needs no direct-API impl."""
    pipeline = _pipeline(mcp=_BINDING)
    assert pipeline.nodes[0].mcp is not None
    assert pipeline.nodes[0].impl is None


def test_external_call_requires_impl_or_mcp() -> None:
    with pytest.raises(ValidationError, match="impl or mcp"):
        _pipeline()  # neither impl nor mcp


def test_mcp_rejected_on_non_external_call() -> None:
    with pytest.raises(ValidationError, match="only allowed on external_call"):
        Node(
            id="pf",
            kind=NodeKind.PURE_FUNCTION,
            description="d",
            impl="extracted/x.py:f",
            mcp=_BINDING,
        )


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"server": "bad name", "tool": "t"}, "server"),
        ({"server": "vendor-1", "tool": "t"}, "server"),  # hyphen not an identifier
        ({"server": "vendor", "tool": "bad tool"}, "tool"),
        ({"server": "vendor", "tool": "t", "url": "ftp://x"}, "url"),
        ({"server": "vendor", "tool": "t", "args": {"ok": "bad key"}}, "args"),
    ],
)
def test_binding_charset_constraints(kwargs: dict, match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        MCPBinding(**kwargs)


# ───────── DBOS emission ─────────


def test_dbos_emits_mcp_call_by_default() -> None:
    src = emit_main(_pipeline(mcp=_BINDING), DbosAdapterConfig())
    # The step body calls the tool via the emitted helper's single entry
    # point (endpoint + credentials resolve at runtime: env > registry >
    # IR; auth problems raise RoteMcpAuthNeeded).
    assert "from extracted._rote_mcp import call_mcp_tool" in src
    assert "call_mcp_tool(" in src
    assert "'vendor',\n" in src
    assert '"enrich_contact",\n' in src
    # No stub import / NotImplementedError for an MCP-backed node.
    assert "NotImplementedError" not in src
    ast.parse(src)  # emitted module is valid Python


# ───────── Park-on-auth emission ─────────


def test_mcp_step_dispatch_parks_on_auth() -> None:
    """The workflow wraps MCP-backed step calls in the auth-park loop:
    a dead credential suspends the run durably instead of failing it,
    and `rote mcp login <server>` releases it."""
    src = emit_main(_pipeline(mcp=_BINDING), DbosAdapterConfig())
    assert "from extracted._rote_mcp import RoteMcpAuthNeeded" in src
    assert "_run_with_auth_park(" in src
    # The park advertises what it waits for (CLI discovery) and blocks on
    # the rote:auth topic — the ':' makes IR-signal collisions impossible.
    assert 'DBOS.set_event("rote_auth_status", {"awaiting": server})' in src
    assert 'topic=f"rote:auth:{server}"' in src
    ast.parse(src)


def test_mcp_retry_budget_excludes_auth_failures() -> None:
    """A step with retries must not burn attempts on RoteMcpAuthNeeded —
    no retry produces a credential; the park does."""
    from rote.ir import RetryPolicy

    pipeline = _pipeline(mcp=_BINDING)
    node = pipeline.nodes[0].model_copy(update={"retry": RetryPolicy(max=3)})
    pipeline = pipeline.model_copy(update={"nodes": [node]})
    src = emit_main(pipeline, DbosAdapterConfig())
    assert "should_retry=_mcp_should_retry" in src
    ast.parse(src)


def test_parallel_mcp_wave_joins_through_auth_park() -> None:
    """Queue fan-out joins MCP handles via _join_with_auth_park with a
    re-enqueue closure over the bound payload."""
    nodes = [
        Node(
            id="seed",
            kind=NodeKind.PURE_FUNCTION,
            description="seed",
            input={"contact_id": "str"},
            inputs={"contact_id": "pipeline.input.contact_id"},
            output="dict",
            impl="extracted/seed.py:seed",
        ),
        Node(
            id="enrich_a",
            kind=NodeKind.EXTERNAL_CALL,
            description="a",
            input={"contact_id": "str"},
            inputs={"contact_id": "seed.output.contact_id"},
            output="dict",
            mcp=MCPBinding(server="vendor", tool="enrich_a"),
        ),
        Node(
            id="enrich_b",
            kind=NodeKind.EXTERNAL_CALL,
            description="b",
            input={"contact_id": "str"},
            inputs={"contact_id": "seed.output.contact_id"},
            output="dict",
            mcp=MCPBinding(server="vendor", tool="enrich_b"),
        ),
    ]
    pipeline = Pipeline(
        name="mcp_par",
        description="Parallel MCP wave.",
        input=PipelineInput(type="Req", required=["contact_id"]),
        nodes=nodes,
        edges=[{"from": "seed", "to": "enrich_a"}, {"from": "seed", "to": "enrich_b"}],
        entry_nodes=["seed"],
        exit_nodes=["enrich_a", "enrich_b"],
    )
    src = emit_main(pipeline, DbosAdapterConfig())
    assert "enrich_a_payload = {" in src
    assert "lambda: queue.enqueue(enrich_a, enrich_a_payload)" in src
    assert "_join_with_auth_park(" in src
    ast.parse(src)


def test_api_backend_emits_no_park_machinery() -> None:
    pipeline = _pipeline(impl="extracted/vendor.py:enrich_contact", mcp=_BINDING)
    src = emit_main(pipeline, DbosAdapterConfig(external_backend="api"))
    assert "_run_with_auth_park" not in src
    assert "RoteMcpAuthNeeded" not in src
    assert "rote_auth_status" not in src


def test_dbos_api_backend_uses_impl_not_mcp() -> None:
    pipeline = _pipeline(impl="extracted/vendor.py:enrich_contact", mcp=_BINDING)
    src = emit_main(pipeline, DbosAdapterConfig(external_backend="api"))
    assert "from extracted.vendor import enrich_contact" in src
    assert "from fastmcp import Client" not in src


def test_mcp_backend_ignored_when_no_binding() -> None:
    """A plain external_call (impl only) is unaffected by the mcp backend."""
    pipeline = _pipeline(impl="extracted/vendor.py:enrich_contact")
    src = emit_main(pipeline, DbosAdapterConfig(external_backend="mcp"))
    assert "from extracted.vendor import enrich_contact" in src
    assert "from fastmcp import Client" not in src


# ───────── --backend flag (registry + CLI) ─────────


def test_get_adapter_rejects_bad_backend() -> None:
    with pytest.raises(ValueError, match="'mcp' or 'api'"):
        get_adapter("dbos", external_backend="grpc")


def test_get_adapter_ignores_backend_for_other_runtimes() -> None:
    """A runtime without mcp support swallows the option instead of erroring."""
    get_adapter("python", external_backend="api")  # must not raise


_CLI_PIPELINE_YAML = """\
name: cli_backend_demo
version: "0.1.0"
description: CLI --backend flag test.
input:
  type: Req
  required: [contact_id]
nodes:
  - id: enrich
    kind: external_call
    description: Enrich via MCP.
    input:
      contact_id: str
    inputs:
      contact_id: pipeline.input.contact_id
    output: dict
    impl: extracted/vendor.py:enrich_contact
    mcp:
      server: vendor
      tool: enrich_contact
      args:
        contact_id: contact_id
edges: []
entry_nodes: [enrich]
exit_nodes: [enrich]
"""


def test_cli_emit_backend_flag(tmp_path: Path) -> None:
    yaml_path = tmp_path / "pipeline.yaml"
    yaml_path.write_text(_CLI_PIPELINE_YAML)

    out_api = tmp_path / "api"
    rc = cli_main(
        ["emit", str(yaml_path), "--runtime", "dbos", "--backend", "api", "--out", str(out_api)]
    )
    assert rc == 0
    api_main = (out_api / "main.py").read_text()
    assert "from extracted.vendor import enrich_contact" in api_main
    assert "from fastmcp import Client" not in api_main

    out_mcp = tmp_path / "mcp"  # default backend is mcp
    rc = cli_main(["emit", str(yaml_path), "--runtime", "dbos", "--out", str(out_mcp)])
    assert rc == 0
    mcp_main = (out_mcp / "main.py").read_text()
    assert "from extracted._rote_mcp import call_mcp_tool" in mcp_main
    assert '"enrich_contact",\n' in mcp_main
    # The connection helper ships with the app, verbatim from
    # rote.mcp._runtime_helper (one tested implementation).
    helper = (out_mcp / "extracted" / "_rote_mcp.py").read_text()
    assert "def mcp_client(" in helper
    assert "from fastmcp import Client" in helper
