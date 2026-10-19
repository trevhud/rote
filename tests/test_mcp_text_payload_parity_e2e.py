"""Python and TS read a text-only MCP result the same way.

Many servers (the TS SDK in particular) declare no output schema and put
the result in text blocks. The Python helper's ``_result_payload`` parses
every text block, keeps non-JSON text verbatim, returns a list for
several blocks and None for none. The TS helpers read only the first
block, threw on non-JSON text and returned ``{}`` for no text, so one
pipeline's MCP step returned different values per runtime. Here both
helpers call the same real server and must agree exactly.

Slow: a real fastmcp HTTP server, npm install, tsc and node.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from rote.adapters import _ts_common, get_adapter
from rote.ir import Pipeline
from tests.test_mcp_oauth_e2e import _free_port, _wait_port

#: Each tool returns text content only (``output_schema=None``), the shape
#: the text fallback exists for.
SERVER_SCRIPT = """\
import sys

from fastmcp import FastMCP
from fastmcp.tools import ToolResult
from mcp.types import ImageContent, TextContent

mcp = FastMCP("parity")


def text(t):
    return TextContent(type="text", text=t)


@mcp.tool(output_schema=None)
def json_text():
    return ToolResult(content=[text('[{"id": "abc"}]')])


@mcp.tool(output_schema=None)
def non_json():
    return ToolResult(content=[text("plain words")])


@mcp.tool(output_schema=None)
def multi_block():
    return ToolResult(content=[text('{"a": 1}'), text("raw"), text("2")])


@mcp.tool(output_schema=None)
def image_then_text():
    image = ImageContent(type="image", data="iVBORw0KGgo=", mimeType="image/png")
    return ToolResult(content=[image, text('{"ok": true}')])


@mcp.tool(output_schema=None)
def no_text():
    image = ImageContent(type="image", data="iVBORw0KGgo=", mimeType="image/png")
    return ToolResult(content=[image])


mcp.run(transport="http", host="127.0.0.1", port=int(sys.argv[1]))
"""

TOOLS = ["json_text", "non_json", "multi_block", "image_then_text", "no_text"]

EXPECTED = {
    "json_text": [{"id": "abc"}],
    "non_json": "plain words",
    "multi_block": [{"a": 1}, "raw", 2],
    "image_then_text": {"ok": True},
    "no_text": None,
}

DRIVER_TS = f"""\
import {{ callMcpTool }} from "./extracted/_roteMcp";

async function main(): Promise<void> {{
    const url = process.argv[2];
    const out: Record<string, unknown> = {{}};
    for (const tool of {json.dumps(TOOLS)}) {{
        out[tool] = await callMcpTool("parity", url, tool, {{}});
    }}
    console.log(JSON.stringify(out));
}}

main();
"""


def test_every_ts_helper_carries_the_one_text_rule() -> None:
    """The Node, Workers and binding helpers splice the same source, and
    none keeps a first-block-only read of its own."""
    for helper in (
        _ts_common.ROTE_MCP_HELPER_TS,
        _ts_common.ROTE_MCP_WORKERS_HELPER_TS,
        _ts_common.ROTE_MCP_BINDING_HELPER_TS,
    ):
        assert _ts_common._MCP_TEXT_PAYLOAD_TS in helper
        assert '.find(\n            (block) => block.type === "text"' not in helper
        assert "JSON.parse(text) : {}" not in helper


@pytest.fixture(scope="module")
def parity_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    script = tmp_path_factory.mktemp("parity-server") / "server.py"
    script.write_text(SERVER_SCRIPT, encoding="utf-8")
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, str(script), str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_port(port)
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def _python_payloads(url: str) -> dict[str, Any]:
    from rote.mcp import _runtime_helper as helper

    async def _all() -> dict[str, Any]:
        return {tool: await helper.call_mcp_tool("parity", url, tool, {}) for tool in TOOLS}

    return asyncio.run(_all())


def _node_payloads(url: str, tmp_path: Path) -> dict[str, Any]:
    pipeline = Pipeline.model_validate(
        {
            "name": "mcp-text-parity",
            "input": {"type": "In", "required": [], "optional": []},
            "nodes": [
                {
                    "id": "read_text",
                    "kind": "external_call",
                    "description": "Read a text-only tool result.",
                    "mcp": {"server": "parity", "tool": "json_text", "url": url},
                }
            ],
            "edges": [],
            "entry_nodes": ["read_text"],
            "exit_nodes": ["read_text"],
        }
    )
    app = tmp_path / "app"
    get_adapter("dbos-ts").emit(pipeline, app)
    (app / "src" / "driver.ts").write_text(DRIVER_TS, encoding="utf-8")
    install = subprocess.run(
        ["npm", "install", "--no-audit", "--no-fund"],
        cwd=app,
        capture_output=True,
        text=True,
        timeout=600,
        env={**os.environ, "npm_config_progress": "false"},
    )
    assert install.returncode == 0, install.stderr
    build = subprocess.run(["npx", "tsc"], cwd=app, capture_output=True, text=True, timeout=300)
    assert build.returncode == 0, build.stdout + build.stderr
    run = subprocess.run(
        ["node", "dist/driver.js", url],
        cwd=app,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert run.returncode == 0, run.stderr
    payloads: dict[str, Any] = json.loads(run.stdout.strip().splitlines()[-1])
    return payloads


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("npm") is None, reason="requires node + npm")
def test_python_and_node_agree_on_text_only_results(parity_server: str, tmp_path: Path) -> None:
    assert _python_payloads(parity_server) == EXPECTED
    assert _node_payloads(parity_server, tmp_path) == EXPECTED
