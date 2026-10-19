"""Both protocol eras, from both emitted helper languages.

MCP 2026-07-28 drops the ``initialize`` handshake. A client that only
speaks the 2025 handshake gets ``-32022 Unsupported protocol version``
from a server that only speaks 2026-07-28; a client that only speaks
2026-07-28 cannot reach the many servers that have not moved. rote's
helpers have to do both: prefer the new era, fall back to the old one.

Servers, all real TypeScript SDK processes:

* ``legacy-stateless``: ``@modelcontextprotocol/sdk`` v1, no session ids.
* ``legacy-sessionful``: v1 with session ids (GitMCP's shape). A request
  without a session that is not ``initialize`` gets a 400.
* ``dual``: ``@modelcontextprotocol/server`` v2, default (both eras).
* ``modern-only``: v2 with ``legacy: "reject"``.

Clients: the Python helper (fastmcp 4) and the emitted Node helper
(``_roteMcp.ts`` from a real DBOS-TS emit, compiled with the app's own
tsconfig). Every case asserts the call result *and* the era the server
saw, so a client that silently stays on the 2025 era against ``dual``
fails. The negative half proves the fixture: the v2 client in its
default ``legacy`` mode cannot reach ``modern-only``.

The modern-only server also carries the two failures that must never
park a workflow on auth or retry it: a tool that asks for input mid-call,
and a bearer gate whose 401 must still park.

Slow: npm installs, tsc, and real node servers.
"""

from __future__ import annotations

import asyncio
import http.server
import json
import os
import shutil
import subprocess
import sys
import threading
import urllib.parse
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.slow

pytest.importorskip("fastmcp")

from mcp_types.version import MODERN_PROTOCOL_VERSIONS  # noqa: E402

from rote.adapters import get_adapter  # noqa: E402
from rote.ir import Pipeline  # noqa: E402
from rote.mcp import _runtime_helper as helper  # noqa: E402
from tests._oauth_fixture import free_port, wait_port  # noqa: E402

MODERN = "2026-07-28"

# Exact pins: this suite asserts which era each line negotiates, so a
# silent bump should show up as a diff here, not as a flake.
_SERVER_PACKAGES = {
    "@modelcontextprotocol/sdk": "1.32.1",
    "@modelcontextprotocol/server": "2.3.1",
    "@modelcontextprotocol/client": "2.3.1",
    "zod": "4.6.7",
}

# Every server logs the protocol-version header of each request it gets,
# so the test can see which era a client actually used.
_LOG_ERA = """
import { appendFileSync } from "node:fs";
function logEra(req) {
  const era = req.headers["mcp-protocol-version"] ?? "none";
  appendFileSync(process.env.ERA_LOG, `${era} ${req.headers["mcp-method"] ?? ""}\\n`);
}
"""

_V1_SERVER = (
    """\
import { createServer } from "node:http";
import { randomUUID } from "node:crypto";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { z } from "zod";
"""
    + _LOG_ERA
    + """
const sessionful = process.env.SESSIONFUL === "1";
function build() {
  const server = new McpServer({ name: "legacy", version: "1.0.0" });
  server.registerTool("echo", { description: "echo", inputSchema: { text: z.string() } },
    async ({ text }) => ({ content: [{ type: "text", text: JSON.stringify({ echoed: text }) }] }));
  return server;
}
const sessions = {};
createServer(async (req, res) => {
  logEra(req);
  let body = "";
  for await (const chunk of req) body += chunk;
  const parsed = body ? JSON.parse(body) : undefined;
  if (!sessionful) {
    const server = build();
    const transport = new StreamableHTTPServerTransport({ sessionIdGenerator: undefined });
    res.on("close", () => { transport.close(); server.close(); });
    await server.connect(transport);
    await transport.handleRequest(req, res, parsed);
    return;
  }
  const sid = req.headers["mcp-session-id"];
  let transport = sid ? sessions[sid] : undefined;
  if (!transport) {
    if (parsed?.method !== "initialize") {
      res.writeHead(400, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ jsonrpc: "2.0", id: null,
        error: { code: -32000, message: "Bad Request: Mcp-Session-Id header is required" } }));
      return;
    }
    transport = new StreamableHTTPServerTransport({
      sessionIdGenerator: () => randomUUID(),
      onsessioninitialized: (id) => { sessions[id] = transport; },
    });
    await build().connect(transport);
  }
  await transport.handleRequest(req, res, parsed);
}).listen(Number(process.argv[2]), "127.0.0.1");
"""
)

_V2_SERVER = (
    """\
import { createServer } from "node:http";
import { McpServer, createMcpHandler, inputRequired } from "@modelcontextprotocol/server";
import * as z from "zod";
"""
    + _LOG_ERA
    + """
const token = process.env.TOKEN;
const handler = createMcpHandler(() => {
  const server = new McpServer({ name: "v2", version: "1.0.0" });
  server.registerTool("echo",
    { description: "echo", inputSchema: z.object({ text: z.string() }),
      outputSchema: z.object({ echoed: z.string() }) },
    async ({ text }) => ({
      content: [{ type: "text", text: JSON.stringify({ echoed: text }) }],
      structuredContent: { echoed: text },
    }));
  server.registerTool("needs_input", { description: "asks a human", inputSchema: z.object({}) },
    async (_args, ctx) => {
      if (!ctx.mcpReq.inputResponses) {
        return inputRequired({ inputRequests: {
          account: inputRequired.elicit({
            message: "Which account?", requestedSchema: z.object({ name: z.string() }) }),
        } });
      }
      return { content: [{ type: "text", text: "{}" }] };
    });
  return server;
}, { legacy: process.env.LEGACY ?? "allow" });

createServer(async (req, res) => {
  logEra(req);
  if (token && req.headers.authorization !== `Bearer ${token}`) {
    res.writeHead(401, { "WWW-Authenticate": "Bearer" });
    res.end();
    return;
  }
  const chunks = [];
  for await (const chunk of req) chunks.push(chunk);
  const body = Buffer.concat(chunks);
  const response = await handler.fetch(new Request(`http://127.0.0.1${req.url}`, {
    method: req.method,
    headers: req.headers,
    body: ["GET", "HEAD"].includes(req.method) ? undefined : body,
  }));
  res.writeHead(response.status, Object.fromEntries(response.headers));
  if (response.body) for await (const chunk of response.body) res.write(chunk);
  res.end();
}).listen(Number(process.argv[2]), "127.0.0.1");
"""
)

# The negative half: the same v2 client the helpers use, left in its
# default `legacy` negotiation mode.
_LEGACY_MODE_PROBE = """\
import { Client, StreamableHTTPClientTransport } from "@modelcontextprotocol/client";
const client = new Client({ name: "probe", version: "0" });
try {
  await client.connect(new StreamableHTTPClientTransport(new URL(process.argv[2])));
  await client.callTool({ name: "echo", arguments: { text: "x" } });
  console.log(JSON.stringify({ ok: true }));
} catch (err) {
  console.log(JSON.stringify({ ok: false, message: String(err?.message ?? err) }));
}
"""

# Runs the emitted helper's callMcpTool and reports the result or the
# error's name, which is how emitted workflows detect it.
_TS_DRIVER = """\
import { callMcpTool } from "./extracted/_roteMcp";

const [server, url, tool, args] = process.argv.slice(2);
callMcpTool(server, url, tool, JSON.parse(args)).then(
    (result) => console.log(JSON.stringify({ result })),
    (err: Error) => console.log(JSON.stringify({ error: err.name, message: err.message })),
);
"""

SERVERS = {
    "legacy-stateless": ("v1.mjs", {}),
    "legacy-sessionful": ("v1.mjs", {"SESSIONFUL": "1"}),
    "dual": ("v2.mjs", {"LEGACY": "allow"}),
    "modern-only": ("v2.mjs", {"LEGACY": "reject"}),
}


@dataclass
class Running:
    url: str
    era_log: Path

    def eras(self) -> set[str]:
        lines = self.era_log.read_text(encoding="utf-8").splitlines()
        return {line.split(" ", 1)[0] for line in lines if line}

    def reset(self) -> None:
        self.era_log.write_text("", encoding="utf-8")


def _npm(cwd: Path, *args: str) -> None:
    proc = subprocess.run(
        ["npm", *args, "--no-audit", "--no-fund"],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=600,
        env={**os.environ, "npm_config_progress": "false"},
    )
    assert proc.returncode == 0, f"npm {args} failed:\n{proc.stdout}\n{proc.stderr}"


@pytest.fixture(scope="module")
def server_project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if shutil.which("node") is None or shutil.which("npm") is None:
        pytest.skip("requires node + npm")
    root = tmp_path_factory.mktemp("era-servers")
    (root / "package.json").write_text(
        json.dumps({"name": "era-servers", "private": True, "type": "module"}), encoding="utf-8"
    )
    _npm(root, "install", *(f"{name}@{v}" for name, v in _SERVER_PACKAGES.items()))
    (root / "v1.mjs").write_text(_V1_SERVER, encoding="utf-8")
    (root / "v2.mjs").write_text(_V2_SERVER, encoding="utf-8")
    (root / "legacy_probe.mjs").write_text(_LEGACY_MODE_PROBE, encoding="utf-8")
    return root


def _start(project: Path, script: str, env: dict[str, str]) -> Iterator[Running]:
    port = free_port()
    era_log = project / f"era-{port}.log"
    era_log.write_text("", encoding="utf-8")
    proc = subprocess.Popen(
        ["node", script, str(port)],
        cwd=project,
        env={**os.environ, **env, "ERA_LOG": str(era_log)},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_port(port)
        yield Running(url=f"http://127.0.0.1:{port}/mcp", era_log=era_log)
    finally:
        proc.terminate()
        proc.wait(timeout=10)


@pytest.fixture(scope="module", params=sorted(SERVERS))
def era_server(
    request: pytest.FixtureRequest, server_project: Path
) -> Iterator[tuple[str, Running]]:
    script, env = SERVERS[request.param]
    for running in _start(server_project, script, env):
        yield request.param, running


@pytest.fixture(scope="module")
def modern_only(server_project: Path) -> Iterator[Running]:
    yield from _start(server_project, "v2.mjs", {"LEGACY": "reject"})


@pytest.fixture(scope="module")
def modern_only_with_auth(server_project: Path) -> Iterator[Running]:
    yield from _start(server_project, "v2.mjs", {"LEGACY": "reject", "TOKEN": "good"})


def _expected_era(name: str) -> bool:
    """True when the client must end up on 2026-07-28."""
    return name in ("dual", "modern-only")


def _assert_era(name: str, eras: set[str]) -> None:
    """What the server saw: new era only, or a probe and then the 2025 era."""
    if _expected_era(name):
        assert eras == {MODERN}, eras
    else:
        assert MODERN in eras, "the probe must be tried before falling back"
        assert eras - {MODERN}, "the fallback must reach the 2025 handshake"


@pytest.fixture(scope="module")
def ts_app(tmp_path_factory: pytest.TempPathFactory, server_project: Path) -> Path:
    """A real DBOS-TS emit (for its _roteMcp.ts and toolchain), plus a driver."""
    pipeline = Pipeline.model_validate(
        {
            "name": "era_matrix",
            "input": {"type": "In", "required": ["text"]},
            "nodes": [
                {
                    "id": "echo_it",
                    "kind": "external_call",
                    "description": "echo over MCP",
                    "inputs": {"text": "pipeline.input.text"},
                    "mcp": {"server": "era", "tool": "echo", "args": {"text": "text"}},
                }
            ],
            "edges": [],
        }
    )
    app = tmp_path_factory.mktemp("era-ts-app")
    get_adapter("dbos-ts").emit(pipeline, app)
    (app / "src" / "driver.ts").write_text(_TS_DRIVER, encoding="utf-8")
    _npm(app, "install")
    build = subprocess.run(["npx", "tsc"], cwd=app, capture_output=True, text=True, timeout=300)
    assert build.returncode == 0, f"tsc failed:\n{build.stdout}\n{build.stderr}"
    return app


def _ts_call(app: Path, url: str, tool: str, args: dict[str, Any]) -> dict[str, Any]:
    proc = subprocess.run(
        ["node", "dist/driver.js", "era", url, tool, json.dumps(args)],
        cwd=app,
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ},
    )
    assert proc.returncode == 0, f"driver crashed:\n{proc.stdout}\n{proc.stderr}"
    out: dict[str, Any] = json.loads(proc.stdout.strip().splitlines()[-1])
    return out


def _py_negotiated(url: str) -> str:
    async def run() -> str:
        async with helper.mcp_client("era", url) as client:
            return str(client.protocol_version)

    return asyncio.run(run())


# ───────── the matrix ─────────


def test_python_helper_reaches_every_era(era_server: tuple[str, Running]) -> None:
    name, server = era_server
    server.reset()
    assert asyncio.run(helper.call_mcp_tool("era", server.url, "echo", {"text": "py"})) == {
        "echoed": "py"
    }
    _assert_era(name, server.eras())
    assert (_py_negotiated(server.url) in MODERN_PROTOCOL_VERSIONS) is _expected_era(name)


def test_emitted_ts_helper_reaches_every_era(era_server: tuple[str, Running], ts_app: Path) -> None:
    name, server = era_server
    server.reset()
    assert _ts_call(ts_app, server.url, "echo", {"text": "ts"}) == {"result": {"echoed": "ts"}}
    _assert_era(name, server.eras())


def test_the_modern_only_fixture_rejects_a_legacy_mode_client(
    server_project: Path, modern_only: Running
) -> None:
    proc = subprocess.run(
        ["node", "legacy_probe.mjs", modern_only.url],
        cwd=server_project,
        capture_output=True,
        text=True,
        timeout=60,
    )
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["ok"] is False
    assert "-32022" in out["message"] or "Unsupported protocol version" in out["message"]


# ───────── failures that must not park ─────────


def test_python_helper_fails_a_mid_call_input_request(modern_only: Running) -> None:
    with pytest.raises(helper.RoteMcpInputRequired) as caught:
        asyncio.run(helper.call_mcp_tool("era", modern_only.url, "needs_input", {}))
    assert caught.value.tool == "needs_input"
    assert caught.value.server == "era"
    assert not isinstance(caught.value, helper.RoteMcpAuthNeeded)


def test_emitted_ts_helper_fails_a_mid_call_input_request(
    modern_only: Running, ts_app: Path
) -> None:
    out = _ts_call(ts_app, modern_only.url, "needs_input", {})
    assert out["error"] == "RoteMcpInputRequired", out
    assert "needs_input" in out["message"]
    assert "needs (re)authentication" not in out["message"]


@pytest.fixture(scope="module")
def input_required_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """A server that sends `input_required` without checking capabilities."""
    calls = tmp_path_factory.mktemp("input-server") / "calls.log"
    calls.write_text("", encoding="utf-8")
    port = free_port()
    script = Path(__file__).resolve().parent / "_mcp_input_server.py"
    proc = subprocess.Popen(
        [sys.executable, str(script), str(port), str(calls)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_port(port)
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_emitted_ts_helper_fails_an_input_required_result(
    input_required_server: str, ts_app: Path
) -> None:
    """The other shape: the result itself is `input_required`, so the SDK's
    auto-fulfilment finds no handler (rote registers none)."""
    out = _ts_call(ts_app, input_required_server, "enrich_contact", {"contact_id": "c1"})
    assert out["error"] == "RoteMcpInputRequired", out
    assert "approval (elicitation/create)" in out["message"]


def test_a_modern_era_401_still_means_auth_needed(
    modern_only_with_auth: Running, ts_app: Path
) -> None:
    """The park predicate on the new SDKs: a 401 from the version probe
    itself must still read as auth, on both languages."""
    with pytest.raises(helper.RoteMcpAuthNeeded):
        asyncio.run(helper.call_mcp_tool("era", modern_only_with_auth.url, "echo", {"text": "x"}))
    out = _ts_call(ts_app, modern_only_with_auth.url, "echo", {"text": "x"})
    assert out["error"] == "RoteMcpAuthNeeded", out


# ───────── a stored login belongs to one endpoint ─────────


def _store_login(server_url: str, **overrides: Any) -> None:
    from rote.mcp import write_token_file

    write_token_file(
        "era",
        {
            "server_url": server_url,
            "tokens": {"access_token": "good", "token_type": "Bearer"},
            "expires_at": None,
            "client_info": {"client_id": "c"},
            "token_endpoint": None,
            **overrides,
        },
    )


def test_both_helpers_send_a_login_only_to_its_own_endpoint(
    modern_only_with_auth: Running, ts_app: Path
) -> None:
    """The token file is keyed by server name. Repoint the name (rote mcp
    add, ROTE_MCP_<SERVER>_URL) and the old endpoint's bearer token must
    not go to the new host: the run parks instead."""
    server = modern_only_with_auth
    # Same endpoint, spelled with a trailing slash and an explicit default
    # port: the login is used, and the server accepts its token.
    host_port = server.url.split("//", 1)[1].split("/", 1)[0]
    _store_login(f"http://{host_port.upper()}/mcp/")
    assert asyncio.run(helper.call_mcp_tool("era", server.url, "echo", {"text": "a"})) == {
        "echoed": "a"
    }
    assert _ts_call(ts_app, server.url, "echo", {"text": "b"}) == {"result": {"echoed": "b"}}

    _store_login("http://old.example.com/mcp")
    server.reset()
    with pytest.raises(helper.RoteMcpAuthNeeded, match="stored login is for"):
        asyncio.run(helper.call_mcp_tool("era", server.url, "echo", {"text": "c"}))
    out = _ts_call(ts_app, server.url, "echo", {"text": "d"})
    assert out["error"] == "RoteMcpAuthNeeded", out
    assert "stored login is for http://old.example.com/mcp" in out["message"]
    assert server.eras() == set(), "no request may reach the new endpoint"


def test_emitted_ts_helper_parks_on_an_unparseable_stored_url(
    modern_only_with_auth: Running, ts_app: Path
) -> None:
    """`new URL` throws a TypeError on a bad port: that crashed the run
    instead of parking it."""
    _store_login("http://old.example.com:99999/mcp")
    out = _ts_call(ts_app, modern_only_with_auth.url, "echo", {"text": "x"})
    assert out["error"] == "RoteMcpAuthNeeded", out


class _FormRecorder(http.server.BaseHTTPRequestHandler):
    bodies: list[dict[str, list[str]]] = []

    def do_POST(self) -> None:  # noqa: N802 (http.server API)
        length = int(self.headers.get("content-length", 0))
        _FormRecorder.bodies.append(urllib.parse.parse_qs(self.rfile.read(length).decode()))
        self.send_response(400)
        self.end_headers()

    def log_message(self, *args: object) -> None:
        pass


@pytest.mark.parametrize(
    ("client_info", "sent_id", "sent_secret"),
    [
        # A stored (dynamically registered) client: its own id, no secret,
        # never the registry's pre-registered secret.
        ({"client_id": "dcr-1"}, "dcr-1", None),
        # No stored client: the registry's id and secret, together.
        (None, "pre", "pre-secret"),
    ],
)
def test_emitted_ts_refresh_takes_client_id_and_secret_from_one_source(
    ts_app: Path,
    modern_only: Running,
    client_info: dict[str, str] | None,
    sent_id: str,
    sent_secret: str | None,
) -> None:
    from rote.mcp import McpRegistry, McpServerConfig, save_registry

    recorder = http.server.HTTPServer(("127.0.0.1", 0), _FormRecorder)
    threading.Thread(target=recorder.serve_forever, daemon=True).start()
    _FormRecorder.bodies = []
    try:
        save_registry(
            McpRegistry(
                servers={
                    "era": McpServerConfig(
                        url=modern_only.url, client_id="pre", client_secret="pre-secret"
                    )
                }
            )
        )
        _store_login(
            modern_only.url,
            tokens={"access_token": "old", "token_type": "Bearer", "refresh_token": "r"},
            expires_at=1.0,
            client_info=client_info,
            token_endpoint=f"http://127.0.0.1:{recorder.server_port}/token",
        )
        out = _ts_call(ts_app, modern_only.url, "echo", {"text": "x"})
    finally:
        recorder.shutdown()
    assert out["error"] == "RoteMcpAuthNeeded", out  # the recorder refuses the grant
    [form] = _FormRecorder.bodies
    assert form["client_id"] == [sent_id]
    assert form.get("client_secret") == ([sent_secret] if sent_secret else None)
