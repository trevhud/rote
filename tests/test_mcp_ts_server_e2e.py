"""The emitted MCP helper against real TypeScript-SDK servers.

Every other MCP fixture in this suite is a FastMCP server, and FastMCP
servers auto-generate output schemas, which is exactly why the
"``result.data`` is None on most real-world servers" bug (CLAUDE.md
gotchas) shipped past them. This suite closes that gap with the two
TypeScript server lines a rote user actually meets:

* ``@modelcontextprotocol/sdk`` v1, handshake-era only (most deployed
  servers). fastmcp 4's client probes the sessionless protocol, gets
  refused, and falls back to the 2025-11-25 handshake.
* ``@modelcontextprotocol/server`` v2, which speaks the sessionless
  2026-07-28 protocol, so the client stays on the modern era.

Each serves the two result shapes that matter: JSON riding in a text
block with no output schema (``.data`` is None, the text fallback is the
only way to the payload) and a typed result whose keys are not Python
identifiers (``.data`` is a generated ``Root`` class that renames
``display-name`` and does not pickle; the wire value is the payload).

Slow: real npm install and real node servers on real sockets.
"""

from __future__ import annotations

import asyncio
import json
import os
import pickle
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

pytest.importorskip("fastmcp")

from fastmcp import Client  # noqa: E402
from mcp_types.version import MODERN_PROTOCOL_VERSIONS  # noqa: E402

from rote.adapters import get_adapter  # noqa: E402
from rote.ir import Pipeline  # noqa: E402
from rote.mcp import _runtime_helper as helper  # noqa: E402

# Exact pins: this suite asserts which protocol era each line negotiates,
# so a silent major bump should show up as a diff here, not as a flake.
_PACKAGES = {
    "@modelcontextprotocol/sdk": "1.32.1",
    "@modelcontextprotocol/server": "2.3.1",
    "@modelcontextprotocol/node": "2.1.1",
    "zod": "4.6.7",
}

_ORG = {"id": "abc", "display-name": "Vaib Studio"}

_V1_SERVER = """\
import { createServer } from "node:http";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { z } from "zod";

const ORG = %(org)s;
function build() {
  const server = new McpServer({ name: "ts-v1", version: "1.0.0" });
  server.registerTool("list_orgs", { description: "orgs as JSON text", inputSchema: {} },
    async () => ({ content: [{ type: "text", text: JSON.stringify([ORG]) }] }));
  server.registerTool("get_org", {
      description: "one org, typed",
      inputSchema: { id: z.string() },
      outputSchema: { id: z.string(), "display-name": z.string() },
    },
    async ({ id }) => {
      const org = { ...ORG, id };
      return { content: [{ type: "text", text: JSON.stringify(org) }], structuredContent: org };
    });
  return server;
}
createServer(async (req, res) => {
  // Stateless, the v1 docs' pattern: a fresh server and transport per request.
  let body = "";
  for await (const chunk of req) body += chunk;
  const server = build();
  const transport = new StreamableHTTPServerTransport({ sessionIdGenerator: undefined });
  res.on("close", () => { transport.close(); server.close(); });
  await server.connect(transport);
  await transport.handleRequest(req, res, body ? JSON.parse(body) : undefined);
}).listen(Number(process.argv[2]), "127.0.0.1");
"""

_V2_SERVER = """\
import { createServer } from "node:http";
import { createMcpHandler, McpServer } from "@modelcontextprotocol/server";
import { toNodeHandler } from "@modelcontextprotocol/node";
import * as z from "zod/v4";

const ORG = %(org)s;
const handler = createMcpHandler(() => {
  const server = new McpServer({ name: "ts-v2", version: "1.0.0" });
  server.registerTool("list_orgs", { description: "orgs as JSON text", inputSchema: z.object({}) },
    async () => ({ content: [{ type: "text", text: JSON.stringify([ORG]) }] }));
  server.registerTool("get_org", {
      description: "one org, typed",
      inputSchema: z.object({ id: z.string() }),
      outputSchema: z.object({ id: z.string(), "display-name": z.string() }),
    },
    async ({ id }) => {
      const org = { ...ORG, id };
      return { content: [{ type: "text", text: JSON.stringify(org) }], structuredContent: org };
    });
  return server;
});
const node = toNodeHandler(handler);
createServer((req, res) => void node(req, res)).listen(Number(process.argv[2]), "127.0.0.1");
"""


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_port(port: int, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket() as s:
            s.settimeout(0.5)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.2)
    raise RuntimeError(f"server did not come up on :{port}")


@pytest.fixture(scope="module")
def node_project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if shutil.which("node") is None or shutil.which("npm") is None:
        pytest.skip("requires node + npm")
    root = tmp_path_factory.mktemp("ts-mcp-servers")
    (root / "package.json").write_text(
        json.dumps({"name": "ts-mcp-servers", "private": True, "type": "module"}),
        encoding="utf-8",
    )
    install = subprocess.run(
        [
            "npm",
            "install",
            "--no-audit",
            "--no-fund",
            *(f"{name}@{version}" for name, version in _PACKAGES.items()),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert install.returncode == 0, f"npm install failed:\n{install.stdout}\n{install.stderr}"
    org = json.dumps(_ORG)
    (root / "v1.mjs").write_text(_V1_SERVER % {"org": org}, encoding="utf-8")
    (root / "v2.mjs").write_text(_V2_SERVER % {"org": org}, encoding="utf-8")
    return root


@pytest.fixture(params=["v1", "v2"])
def ts_server(request: pytest.FixtureRequest, node_project: Path) -> Iterator[tuple[str, str]]:
    port = _free_port()
    proc = subprocess.Popen(
        ["node", f"{request.param}.mjs", str(port)],
        cwd=node_project,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        _wait_port(port)
        yield request.param, f"http://127.0.0.1:{port}/mcp"
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_helper_reads_plain_json_from_ts_sdk_servers(ts_server: tuple[str, str]) -> None:
    line, url = ts_server

    async def _observe() -> tuple[str, object, object]:
        async with Client(url) as client:
            text_only = await client.call_tool("list_orgs", {})
            typed = await client.call_tool("get_org", {"id": "abc"})
            return client.protocol_version, text_only.data, typed.data

    version, text_data, typed_data = asyncio.run(_observe())
    # The era each line negotiates with fastmcp 4's default client.
    assert (version in MODERN_PROTOCOL_VERSIONS) is (line == "v2"), version
    # Why the helper cannot use .data: None for text-only results, and a
    # generated class with renamed attributes for typed ones.
    assert text_data is None
    assert not isinstance(typed_data, dict)
    with pytest.raises(Exception):  # noqa: B017 — the exact pickling error varies
        pickle.dumps(typed_data)

    text_payload = asyncio.run(helper.call_mcp_tool("ts", url, "list_orgs", {}))
    typed_payload = asyncio.run(helper.call_mcp_tool("ts", url, "get_org", {"id": "abc"}))
    assert text_payload == [_ORG]
    assert typed_payload == _ORG
    for payload in (text_payload, typed_payload):
        assert pickle.loads(pickle.dumps(payload)) == payload


def _rpc_log(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the JSON-RPC method of every POST the helper's transport sends.

    Hooks the same httpx2 response hook the 401 detection uses, so the
    count is of real requests on the wire, not of client method calls.
    """
    log: list[str] = []
    original = helper._HttpStatusWatch._on_response

    async def _record(self: helper._HttpStatusWatch, response: object) -> None:
        request = response.request  # type: ignore[attr-defined]
        if request.method == "POST" and request.content:
            log.append(json.loads(request.content).get("method", "?"))
        await original(self, response)

    monkeypatch.setattr(helper._HttpStatusWatch, "_on_response", _record)
    return log


def test_one_tools_list_per_step(
    ts_server: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each MCP step makes exactly one tools/list, before the call.

    The helper lists explicitly to learn the wrap flag. That is not an
    extra round trip: the MCP SDK's call_tool validates every successful
    result against the tool's output schema and lists tools itself on a
    fresh session, text-only results included. The first half pins that
    SDK behavior, so if a release stops listing, this fails and the
    helper's explicit list should be revisited; the second pins the
    helper to the same total with the listing moved ahead of the call.
    """
    _, url = ts_server
    log = _rpc_log(monkeypatch)

    async def _bare_call(tool: str, args: dict[str, str]) -> None:
        async with helper.mcp_client("ts", url, http_status=helper._HttpStatusWatch()) as client:
            await client.call_tool(tool, args)

    for tool, args in (("list_orgs", {}), ("get_org", {"id": "abc"})):
        log.clear()
        asyncio.run(_bare_call(tool, args))
        sdk = [m for m in log if m.startswith("tools/")]
        assert sdk == ["tools/call", "tools/list"], (tool, log)

        log.clear()
        asyncio.run(helper.call_mcp_tool("ts", url, tool, args))
        ours = [m for m in log if m.startswith("tools/")]
        assert ours == ["tools/list", "tools/call"], (tool, log)


# ───────── Python and TypeScript read fastmcp results the same way ─────────

_FASTMCP_SERVER = """\
import sys

from fastmcp import FastMCP
from fastmcp.tools import ToolResult
from mcp.types import TextContent

server = FastMCP("wrap-shapes")


@server.tool
def marked_list() -> list[int]:
    # Current fastmcp: wrapped, flagged in the schema AND in _meta.
    return [1, 2]


_FLAGGED = {
    "type": "object",
    "properties": {"result": {"type": "array", "items": {"type": "integer"}}},
    "required": ["result"],
    "x-fastmcp-wrap-result": True,
}


@server.tool(output_schema=_FLAGGED)
def flag_only_list():
    # fastmcp 2.x / 3.0 servers: wrapped and flagged in the schema only.
    return ToolResult(structured_content={"result": [1, 2]})


_PLAIN = {
    "type": "object",
    "properties": {"result": {"type": "array", "items": {"type": "integer"}}},
    "required": ["result"],
}


@server.tool(output_schema=_PLAIN)
def result_key_object():
    # A real object whose only key happens to be "result": not a wrapper.
    return ToolResult(structured_content={"result": [1, 2]})


@server.tool(output_schema=None)
def text_only():
    return ToolResult(content=[TextContent(type="text", text='[{"id": "abc"}]')])


server.run(transport="http", host="127.0.0.1", port=int(sys.argv[1]), show_banner=False)
"""

_SHAPES = {
    "marked_list": [1, 2],
    "flag_only_list": [1, 2],
    "result_key_object": {"result": [1, 2]},
    "text_only": [{"id": "abc"}],
}

_SHAPES_DRIVER_TS = """\
import { callMcpTool } from "./extracted/_roteMcp";

const url = process.argv[2];
const tools = JSON.parse(process.argv[3]) as string[];

(async () => {
    const out: Record<string, unknown> = {};
    for (const tool of tools) out[tool] = await callMcpTool("shapes", url, tool, {});
    console.log(JSON.stringify(out));
})();
"""


@pytest.fixture(scope="module")
def fastmcp_shapes_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    script = tmp_path_factory.mktemp("fastmcp-shapes") / "server.py"
    script.write_text(_FASTMCP_SERVER, encoding="utf-8")
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


def test_python_and_typescript_helpers_agree_on_fastmcp_results(
    fastmcp_shapes_server: str, tmp_path: Path
) -> None:
    """The same server, the same four shapes, the same values on both sides.

    The emitted TS helper used to return structuredContent as-is, so a
    list-returning fastmcp tool was a list in Python and {result: [...]}
    in TypeScript: one pipeline meaning two things. The TS unwrap is one
    source spliced into both direct-client helpers (Node and Workers);
    this runs the Node copy for real and the splice test in
    tests/test_mcp_ts.py covers the Workers copy.
    """
    url = fastmcp_shapes_server
    if shutil.which("node") is None or shutil.which("npm") is None:
        pytest.skip("requires node + npm")

    python_side = {
        tool: asyncio.run(helper.call_mcp_tool("shapes", url, tool, {})) for tool in _SHAPES
    }
    assert python_side == _SHAPES

    pipeline = Pipeline.model_validate(
        {
            "name": "wrap_shapes",
            "input": {"type": "In"},
            "nodes": [
                {
                    "id": "marked",
                    "kind": "external_call",
                    "description": "one fastmcp result shape",
                    "mcp": {"server": "shapes", "tool": "marked_list", "url": url},
                }
            ],
            "edges": [],
        }
    )
    app = tmp_path / "app"
    get_adapter("dbos-ts").emit(pipeline, app)
    (app / "src" / "driver.ts").write_text(_SHAPES_DRIVER_TS, encoding="utf-8")
    install = subprocess.run(
        ["npm", "install", "--no-audit", "--no-fund"],
        cwd=app,
        capture_output=True,
        text=True,
        timeout=600,
        env={**os.environ, "npm_config_progress": "false"},
    )
    assert install.returncode == 0, f"npm install failed:\n{install.stdout}\n{install.stderr}"
    build = subprocess.run(["npx", "tsc"], cwd=app, capture_output=True, text=True, timeout=300)
    assert build.returncode == 0, f"tsc failed:\n{build.stdout}\n{build.stderr}"
    run = subprocess.run(
        ["node", "dist/driver.js", url, json.dumps(list(_SHAPES))],
        cwd=app,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert run.returncode == 0, f"driver failed:\n{run.stdout}\n{run.stderr}"
    typescript_side = json.loads(run.stdout.strip().splitlines()[-1])
    assert typescript_side == python_side
