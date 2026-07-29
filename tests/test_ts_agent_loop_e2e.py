"""The emitted TypeScript agent loop, run for real.

Static checks (`tsc --noEmit`) prove the emitted loop is well-typed;
they cannot prove it *works*. Two bugs found while building this runtime
were invisible to both tsc and the SDK's own types — a tool runner that
deadlocks because `done()` observes a loop it never starts, and an
iteration count that echoed the cap instead of the turns actually run.
Only running it finds that class of defect.

So this drives the real thing end to end:

* a real MCP server (fastmcp) advertising the tools the IR declares,
* the emitted module, compiled by the emitted toolchain against the real
  npm dependencies,
* the real Anthropic SDK's tool runner, pointed by ``ROTE_BASE_URL_<ID>``
  at a local endpoint that speaks enough of the Messages API to drive a
  two-turn tool loop.

What it does NOT prove is that api.anthropic.com accepts the same
request shape — that needs a real key, and is deliberately not a thing
the test suite depends on.

Slow: real npm install + tsc + node, real MCP server.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import textwrap
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from rote.adapters import get_adapter
from rote.ir import Pipeline

pytestmark = pytest.mark.slow

pytest.importorskip("fastmcp")


def _node_available() -> bool:
    return shutil.which("node") is not None and shutil.which("npm") is not None


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


# ───────── The MCP server the loop's tools actually live on ─────────

_MOCK_MCP = textwrap.dedent(
    """
    import sys
    from fastmcp import FastMCP

    mcp = FastMCP("research-tools")

    @mcp.tool
    def web_search(query: str) -> dict:
        \"\"\"Search the public web for a query.\"\"\"
        return {"query": query, "hits": ["https://example.com/a"]}

    @mcp.tool
    def fetch_page(url: str) -> dict:
        \"\"\"Fetch one page and return its text.\"\"\"
        return {"url": url, "text": "the page said yes"}

    # Deliberately advertised but NOT in the IR's allowlist: the emitted
    # loop must never bind it.
    @mcp.tool
    def delete_everything(confirm: bool) -> dict:
        \"\"\"Destructive tool the pipeline never declared.\"\"\"
        return {"deleted": confirm}

    if __name__ == "__main__":
        mcp.run(transport="http", host="127.0.0.1", port=int(sys.argv[1]))
    """
)


@pytest.fixture(scope="module")
def mcp_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    src = tmp_path_factory.mktemp("agentmcp") / "server.py"
    src.write_text(_MOCK_MCP, encoding="utf-8")
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, str(src), str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_port(port)
        yield f"http://127.0.0.1:{port}/mcp/"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


# ───────── A local Messages endpoint that drives a two-turn loop ─────────


class _MessagesStub(BaseHTTPRequestHandler):
    """Turn 1 asks for ``web_search``; turn 2 answers from its result."""

    requests: list[dict] = []

    def do_POST(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler's API
        body = self.rfile.read(int(self.headers.get("content-length", 0)))
        payload = json.loads(body or b"{}")
        type(self).requests.append(payload)
        turn = len(type(self).requests)
        if turn == 1:
            message = {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": payload.get("model", "stub"),
                "stop_reason": "tool_use",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "web_search",
                        "input": {"query": "durable execution"},
                    }
                ],
                "usage": {"input_tokens": 11, "output_tokens": 7},
            }
        else:
            message = {
                "id": "msg_2",
                "type": "message",
                "role": "assistant",
                "model": payload.get("model", "stub"),
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "Found 1 hit for durable execution."}],
                "usage": {"input_tokens": 21, "output_tokens": 9},
            }
        raw = json.dumps(message).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args: object) -> None:
        return


@pytest.fixture()
def messages_stub() -> Iterator[str]:
    _MessagesStub.requests = []
    server = HTTPServer(("127.0.0.1", 0), _MessagesStub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


# ───────── The pipeline + the driver that runs its loop ─────────


def _pipeline() -> Pipeline:
    return Pipeline.model_validate(
        {
            "name": "agent-loop-e2e",
            "version": "0.1.0",
            "source_skill": "tests/fixtures/agent-loop-e2e",
            "description": "One bounded research loop.",
            "input": {"type": "Brief", "required": ["topic"], "optional": []},
            "nodes": [
                {
                    "id": "research_loop",
                    "kind": "agent_loop",
                    "description": "Research the topic until the brief is covered.",
                    "tools": ["web_search", "fetch_page"],
                    "termination": {"condition": "brief covered", "max_iterations": 5},
                    "inputs": {"topic": "pipeline.input.topic"},
                }
            ],
            "edges": [],
            "entry_nodes": ["research_loop"],
            "exit_nodes": ["research_loop"],
        }
    )


DRIVER_TS = """\
import { researchLoop } from "./extracted/research_loop";

async function main(): Promise<void> {
    const result = await researchLoop({ topic: "durable execution" });
    console.log(JSON.stringify(result));
}

void main();
"""


@pytest.mark.skipif(not _node_available(), reason="requires node + npm")
def test_emitted_agent_loop_binds_mcp_tools_and_runs(
    mcp_server: str, messages_stub: str, tmp_path: Path
) -> None:
    app_dir = tmp_path / "app"
    get_adapter("dbos-ts").emit(_pipeline(), app_dir)
    (app_dir / "src" / "driver.ts").write_text(DRIVER_TS, encoding="utf-8")

    install = subprocess.run(
        ["npm", "install", "--no-audit", "--no-fund"],
        cwd=app_dir,
        capture_output=True,
        text=True,
        timeout=600,
        env={**os.environ, "npm_config_progress": "false"},
    )
    assert install.returncode == 0, f"npm install failed:\n{install.stdout}\n{install.stderr}"
    build = subprocess.run(["npx", "tsc"], cwd=app_dir, capture_output=True, text=True, timeout=300)
    assert build.returncode == 0, f"tsc failed:\n{build.stdout}\n{build.stderr}"

    proc = subprocess.run(
        ["node", "dist/driver.js"],
        cwd=app_dir,
        capture_output=True,
        text=True,
        timeout=120,
        env={
            **os.environ,
            # No server is resolvable from this pipeline at emit time, so
            # the runtime escape hatch is what supplies it — the exact
            # path the emitted NOTE comment tells an operator to take.
            "ROTE_MCP_SERVERS": "research_tools",
            "ROTE_MCP_RESEARCH_TOOLS_URL": mcp_server,
            "ROTE_MCP_TOKEN_DIR": str(tmp_path / "tokens"),
            "ANTHROPIC_API_KEY": "local-stub-key",
            "ROTE_BASE_URL_RESEARCH_LOOP": messages_stub,
        },
    )
    assert proc.returncode == 0, f"driver failed:\n{proc.stdout}\n{proc.stderr}"
    result = json.loads(proc.stdout.strip().splitlines()[-1])

    # The loop ran, on the lane the operator's credentials selected.
    assert result["provider"] == "api"
    assert result["result"] == "Found 1 hit for durable execution."
    # Two turns actually ran — not the cap (5) echoed back. Reporting the
    # bound as the count made every loop look saturated.
    assert result["iterations"] == 2

    # The tools reached the model as the SERVER declares them, schema and
    # all — so the agent's contract cannot drift from the server's.
    first_turn = _MessagesStub.requests[0]
    sent = {tool["name"]: tool for tool in first_turn["tools"]}
    assert set(sent) == {"web_search", "fetch_page"}, (
        "the IR's allowlist is the boundary: the server also advertises "
        "delete_everything, which the pipeline never declared"
    )
    assert sent["web_search"]["input_schema"]["properties"]["query"]["type"] == "string"

    # The tool result came back from the real MCP server and fed turn 2.
    second_turn = _MessagesStub.requests[1]
    tool_results = [
        block
        for message in second_turn["messages"]
        if isinstance(message.get("content"), list)
        for block in message["content"]
        if isinstance(block, dict) and block.get("type") == "tool_result"
    ]
    assert tool_results, f"no tool_result reached turn 2: {second_turn['messages']}"
    assert "example.com" in json.dumps(tool_results[0]["content"])


@pytest.mark.skipif(not _node_available(), reason="requires node + npm")
def test_emitted_agent_loop_fails_loudly_when_a_tool_has_no_server(
    messages_stub: str, tmp_path: Path
) -> None:
    """A tool no reachable server provides is fatal, not silently dropped.

    Running an agent with fewer tools than the IR declared is the worst
    outcome available: it usually still produces plausible output.
    """
    app_dir = tmp_path / "app"
    get_adapter("dbos-ts").emit(_pipeline(), app_dir)
    (app_dir / "src" / "driver.ts").write_text(DRIVER_TS, encoding="utf-8")
    subprocess.run(
        ["npm", "install", "--no-audit", "--no-fund"],
        cwd=app_dir,
        capture_output=True,
        text=True,
        timeout=600,
        env={**os.environ, "npm_config_progress": "false"},
    )
    build = subprocess.run(["npx", "tsc"], cwd=app_dir, capture_output=True, text=True, timeout=300)
    assert build.returncode == 0, f"tsc failed:\n{build.stdout}\n{build.stderr}"

    proc = subprocess.run(
        ["node", "dist/driver.js"],
        cwd=app_dir,
        capture_output=True,
        text=True,
        timeout=120,
        env={
            **os.environ,
            "ROTE_MCP_SERVERS": "nowhere",
            "ROTE_MCP_NOWHERE_URL": f"http://127.0.0.1:{_free_port()}/mcp/",
            "ROTE_MCP_TOKEN_DIR": str(tmp_path / "tokens"),
            "ANTHROPIC_API_KEY": "local-stub-key",
            "ROTE_BASE_URL_RESEARCH_LOOP": messages_stub,
        },
    )
    assert proc.returncode != 0
    assert "agent tools not provided by any reachable MCP server" in proc.stderr
    assert "web_search" in proc.stderr
