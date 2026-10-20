"""Python and TS read an MCP tool result the same way.

Many servers (the TS SDK in particular) declare no output schema and put
the result in text blocks. The Python helper's ``_result_payload`` parses
every text block, keeps non-JSON text verbatim, returns a list for
several blocks and None for none. The TS helpers read only the first
block, threw on non-JSON text and returned ``{}`` for no text, so one
pipeline's MCP step returned different values per runtime. Here the
Python helper, the Node helper and the Workers helper all call the same
real server and must agree exactly, edge cases included: empty content,
an empty text block, JSON's non-finite constants, an error result,
structured content next to text, and an embedded resource.

Slow: a real fastmcp HTTP server, npm install, tsc, esbuild and node.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from rote.adapters import _ts_common, get_adapter
from rote.ir import Pipeline
from tests import _node_trees as node_trees
from tests.test_mcp_oauth_e2e import _free_port, _wait_port

#: Each tool declares no output schema (``output_schema=None``), the shape
#: the text fallback exists for, unless it returns structured content.
SERVER_SCRIPT = """\
import sys

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.tools import ToolResult
from mcp.types import EmbeddedResource, ImageContent, TextContent, TextResourceContents

mcp = FastMCP("parity")


def text(t):
    return TextContent(type="text", text=t)


def image():
    return ImageContent(type="image", data="iVBORw0KGgo=", mimeType="image/png")


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
    return ToolResult(content=[image(), text('{"ok": true}')])


@mcp.tool(output_schema=None)
def no_text():
    return ToolResult(content=[image()])


@mcp.tool(output_schema=None)
def empty_content():
    return ToolResult(content=[])


@mcp.tool(output_schema=None)
def empty_text():
    return ToolResult(content=[text("")])


@mcp.tool(output_schema=None)
def non_finite():
    return ToolResult(content=[text("NaN"), text("Infinity"), text("-Infinity")])


@mcp.tool(output_schema=None)
def embedded_resource():
    resource = EmbeddedResource(
        type="resource",
        resource=TextResourceContents(uri="file:///notes.txt", text="not a text block"),
    )
    return ToolResult(content=[resource, text('{"ok": 1}')])


@mcp.tool(output_schema={"type": "object", "properties": {"wins": {"type": "boolean"}}})
def structured_and_text():
    return ToolResult(content=[text('{"loses": true}')], structured_content={"wins": True})


@mcp.tool(output_schema=None)
def is_error():
    raise ToolError("the tool refused")


mcp.run(transport="http", host="127.0.0.1", port=int(sys.argv[1]))
"""

TOOLS = [
    "json_text",
    "non_json",
    "multi_block",
    "image_then_text",
    "no_text",
    "empty_content",
    "empty_text",
    "non_finite",
    "embedded_resource",
    "structured_and_text",
    "is_error",
]

#: What every helper must return. ``__error__`` marks a call that must
#: fail; the message text differs per language and is not compared.
EXPECTED = {
    "json_text": [{"id": "abc"}],
    "non_json": "plain words",
    "multi_block": [{"a": 1}, "raw", 2],
    "image_then_text": {"ok": True},
    "no_text": None,
    "empty_content": None,
    "empty_text": "",
    # JSON.parse rejects these, so Python keeps the raw text too.
    "non_finite": ["NaN", "Infinity", "-Infinity"],
    # A resource block carries its text on .resource, not .text: skipped.
    "embedded_resource": {"ok": 1},
    # Structured content wins over text on every runtime.
    "structured_and_text": {"wins": True},
    "is_error": "__error__",
}

_TS_LOOP = f"""\
    const out: Record<string, unknown> = {{}};
    for (const tool of {json.dumps(TOOLS)}) {{
        try {{
            out[tool] = await call(tool);
        }} catch {{
            out[tool] = "__error__";
        }}
    }}
    console.log(JSON.stringify(out));
"""

NODE_DRIVER_TS = f"""\
import {{ callMcpTool }} from "./extracted/_roteMcp";

async function main(): Promise<void> {{
    const url = process.argv[2];
    const call = (tool: string) => callMcpTool("parity", url, tool, {{}});
{_TS_LOOP}}}

main();
"""

#: The Workers helper keeps tokens in KV. The server needs none, so a
#: stand-in namespace hands back a non-expiring token.
WORKERS_DRIVER_TS = f"""\
import {{ callMcpTool }} from "./extracted/_roteMcp";

const tokens = {{
    get: async () => ({{ access_token: "unused", expires_at: null }}),
    put: async () => undefined,
}};

async function main(): Promise<void> {{
    const url = process.argv[2];
    const env = {{ ROTE_MCP_TOKENS: tokens }} as unknown as Parameters<typeof callMcpTool>[0];
    const call = (tool: string) => callMcpTool(env, "parity", url, tool, {{}});
{_TS_LOOP}}}

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


def test_python_keeps_non_finite_json_constants_as_text() -> None:
    """``json.loads`` would read these as floats; ``JSON.parse`` rejects
    them. Python refuses them too, so both runtimes keep the raw text."""
    from types import SimpleNamespace

    from rote.mcp import _runtime_helper as helper

    def _result(*texts: str) -> Any:
        return SimpleNamespace(
            structured_content=None,
            meta=None,
            content=[SimpleNamespace(text=t) for t in texts],
        )

    assert helper._result_payload(_result("NaN")) == "NaN"
    assert helper._result_payload(_result("Infinity", "-Infinity")) == ["Infinity", "-Infinity"]
    assert helper._result_payload(_result('{"x": NaN}')) == '{"x": NaN}'
    # Ordinary numbers still parse, big integers exactly.
    assert helper._result_payload(_result("12345678901234567890")) == 12345678901234567890


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
        out: dict[str, Any] = {}
        for tool in TOOLS:
            try:
                out[tool] = await helper.call_mcp_tool("parity", url, tool, {})
            except Exception:
                out[tool] = "__error__"
        return out

    return asyncio.run(_all())


def _pipeline(url: str) -> Pipeline:
    return Pipeline.model_validate(
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


def _last_json_line(app: Path, *command: str) -> dict[str, Any]:
    run = subprocess.run(list(command), cwd=app, capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr
    payloads: dict[str, Any] = json.loads(run.stdout.strip().splitlines()[-1])
    return payloads


def _node_payloads(url: str, tmp_path: Path) -> dict[str, Any]:
    app = tmp_path / "node-app"
    get_adapter("dbos-ts").emit(_pipeline(url), app)
    (app / "src" / "driver.ts").write_text(NODE_DRIVER_TS, encoding="utf-8")
    node_trees.install(app)
    build = subprocess.run(["npx", "tsc"], cwd=app, capture_output=True, text=True, timeout=300)
    assert build.returncode == 0, build.stdout + build.stderr
    return _last_json_line(app, "node", "dist/driver.js", url)


def _workers_payloads(url: str, tmp_path: Path) -> dict[str, Any]:
    """Run the Workers helper itself, bundled for Node by esbuild. It only
    needs fetch and the MCP SDK, so its result handling runs unchanged."""
    app = tmp_path / "workers-app"
    get_adapter("cloudflare").emit(_pipeline(url), app)
    (app / "src" / "driver.ts").write_text(WORKERS_DRIVER_TS, encoding="utf-8")
    node_trees.install(app, "--no-save", "esbuild")
    bundle = subprocess.run(
        [
            "npx",
            "esbuild",
            "src/driver.ts",
            "--bundle",
            "--platform=node",
            "--format=esm",
            "--outfile=dist/driver.mjs",
        ],
        cwd=app,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert bundle.returncode == 0, bundle.stdout + bundle.stderr
    return _last_json_line(app, "node", "dist/driver.mjs", url)


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("npm") is None, reason="requires node + npm")
def test_python_node_and_workers_agree_on_tool_results(parity_server: str, tmp_path: Path) -> None:
    assert _python_payloads(parity_server) == EXPECTED
    assert _node_payloads(parity_server, tmp_path) == EXPECTED
    assert _workers_payloads(parity_server, tmp_path) == EXPECTED
