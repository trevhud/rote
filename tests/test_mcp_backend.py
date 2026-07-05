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
    assert "from fastmcp import Client" in src
    assert 'call_tool("enrich_contact"' in src
    assert "os.environ['ROTE_MCP_VENDOR_URL']" in src
    # No stub import / NotImplementedError for an MCP-backed node.
    assert "from extracted." not in src
    ast.parse(src)  # emitted module is valid Python


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
    assert "from fastmcp import Client" in mcp_main
    assert 'call_tool("enrich_contact"' in mcp_main
