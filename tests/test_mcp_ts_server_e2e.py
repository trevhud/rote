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
import pickle
import shutil
import socket
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

pytest.importorskip("fastmcp")

from fastmcp import Client  # noqa: E402
from mcp_types.version import MODERN_PROTOCOL_VERSIONS  # noqa: E402

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
