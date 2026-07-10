"""Unit tests for MCP-backed emission in the Node TS adapters.

The DBOS-TS and Inngest adapters emit working, authenticated MCP calls
for nodes with an ``mcp:`` binding (backend "mcp", the default) — the
TS twin of the DBOS Python adapter's behavior. The Cloudflare adapter
must NOT gain this: Workers have no filesystem for the token store
(that's PR3's provisioning design).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rote.adapters import get_adapter
from rote.ir import Pipeline


def _bound_pipeline() -> Pipeline:
    return Pipeline.model_validate(
        {
            "name": "ts_mcp_demo",
            "input": {"type": "In", "required": ["q"]},
            "nodes": [
                {
                    "id": "pull_data",
                    "kind": "external_call",
                    "description": "pull data over MCP",
                    "inputs": {"q": "pipeline.input.q"},
                    "mcp": {"server": "vendor", "tool": "get_data", "args": {"q": "q"}},
                },
                {
                    "id": "plain_call",
                    "kind": "external_call",
                    "description": "no binding — stays a stub",
                    "impl": "extracted/plain_call.py:plain_call",
                },
            ],
            "edges": [{"from": "pull_data", "to": "plain_call"}],
        }
    )


@pytest.mark.parametrize("runtime", ["dbos-ts", "inngest"])
def test_node_ts_adapters_emit_working_mcp_calls(runtime: str, tmp_path: Path) -> None:
    get_adapter(runtime).emit(_bound_pipeline(), tmp_path)

    bound = (tmp_path / "src" / "extracted" / "pull_data.ts").read_text(encoding="utf-8")
    assert 'import { callMcpTool } from "./_roteMcp";' in bound
    assert 'callMcpTool("vendor", null, "get_data"' in bound
    assert "throw new Error" not in bound  # a working body, not a stub

    helper = (tmp_path / "src" / "extracted" / "_roteMcp.ts").read_text(encoding="utf-8")
    assert "refreshAccessToken" in helper
    assert "grant_type" in helper
    assert "rote mcp login" in helper

    unbound = (tmp_path / "src" / "extracted" / "plain_call.ts").read_text(encoding="utf-8")
    assert "throw new Error" in unbound  # binding-less nodes still stub

    package = json.loads((tmp_path / "package.json").read_text(encoding="utf-8"))
    assert "@modelcontextprotocol/sdk" in package["dependencies"]


@pytest.mark.parametrize("runtime", ["dbos-ts", "inngest"])
def test_api_backend_keeps_stubs_and_skips_the_helper(runtime: str, tmp_path: Path) -> None:
    get_adapter(runtime, external_backend="api").emit(_bound_pipeline(), tmp_path)
    bound = (tmp_path / "src" / "extracted" / "pull_data.ts").read_text(encoding="utf-8")
    assert "throw new Error" in bound
    assert not (tmp_path / "src" / "extracted" / "_roteMcp.ts").exists()
    package = json.loads((tmp_path / "package.json").read_text(encoding="utf-8"))
    assert "@modelcontextprotocol/sdk" not in package["dependencies"]


@pytest.mark.parametrize("runtime", ["dbos-ts", "inngest"])
def test_helper_not_emitted_without_bindings(runtime: str, tmp_path: Path) -> None:
    pipeline = Pipeline.model_validate(
        {
            "name": "no_mcp",
            "input": {"type": "In"},
            "nodes": [
                {
                    "id": "plain_call",
                    "kind": "external_call",
                    "description": "plain",
                    "impl": "extracted/plain_call.py:plain_call",
                }
            ],
            "edges": [],
        }
    )
    get_adapter(runtime).emit(pipeline, tmp_path)
    assert not (tmp_path / "src" / "extracted" / "_roteMcp.ts").exists()
    package = json.loads((tmp_path / "package.json").read_text(encoding="utf-8"))
    assert "@modelcontextprotocol/sdk" not in package["dependencies"]


def test_cloudflare_never_emits_the_node_helper(tmp_path: Path) -> None:
    """Workers have no filesystem — the Node token-store helper must not
    leak into the Cloudflare output (its auth story is PR3's secret
    provisioning)."""
    get_adapter("cloudflare").emit(_bound_pipeline(), tmp_path)
    assert not (tmp_path / "src" / "extracted" / "_roteMcp.ts").exists()
    bound = (tmp_path / "src" / "extracted" / "pull_data.ts").read_text(encoding="utf-8")
    assert "callMcpTool" not in bound
