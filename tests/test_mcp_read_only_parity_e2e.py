"""A read-only run refuses an MCP write the same way on every helper.

`rote run` sets ``ROTE_MCP_READ_ONLY=1`` unless ``--allow-writes``. The
Python, Node and Workers helpers then call a tool only when the server
marks it ``readOnlyHint`` or the user registered the server read-only
(``ROTE_MCP_READ_ONLY_SERVERS``), the same rule a skill run's tool
allowlist applies, and refuse anything else before it is called. Each
helper calls the same real server here, and an agent loop's tool binding
refuses to start in a read-only run.

Slow: a real fastmcp HTTP server, npm install, tsc, esbuild and node.
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

from rote.adapters import get_adapter
from rote.ir import Pipeline
from tests import _node_trees as node_trees
from tests.test_mcp_oauth_e2e import _free_port, _wait_port

pytestmark = pytest.mark.slow

SERVER_SCRIPT = """\
import sys
from pathlib import Path

from fastmcp import FastMCP
from mcp.types import ToolAnnotations

mcp = FastMCP("ro")
written = Path(sys.argv[2])


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
def lookup(q: str = "") -> dict:
    return {"found": q}


@mcp.tool
def publish(text: str = "") -> dict:
    with written.open("a") as f:
        f.write(text + "\\n")
    return {"published": text}


mcp.run(transport="http", host="127.0.0.1", port=int(sys.argv[1]))
"""

#: Each case: (environment, the outcome of lookup, of publish). An outcome
#: is the tool's value or "refused" (the read-only refusal), and the
#: agent-loop binding is "refused" whenever the run is read-only.
CASES: dict[str, dict[str, str]] = {
    "writes_allowed": {},
    "read_only": {"ROTE_MCP_READ_ONLY": "1", "ROTE_MCP_READ_ONLY_SERVERS": ""},
    "server_declared": {"ROTE_MCP_READ_ONLY": "1", "ROTE_MCP_READ_ONLY_SERVERS": "other,ro"},
}
EXPECTED = {
    "writes_allowed": {"lookup": {"found": "x"}, "publish": {"published": "x"}, "agent": "bound"},
    "read_only": {"lookup": {"found": "x"}, "publish": "refused", "agent": "refused"},
    "server_declared": {
        "lookup": {"found": "x"},
        "publish": {"published": "x"},
        "agent": "refused",
    },
}

_TS_BODY = """\
    const out: Record<string, unknown> = {};
    for (const tool of ["lookup", "publish"]) {
        try {
            out[tool] = await call(tool);
        } catch (err) {
            const message = err instanceof Error ? err.message : String(err);
            if (!isRoteMcpConfigError(err) || !message.includes("may write")) throw err;
            out[tool] = "refused";
        }
    }
    try {
        await bind();
        out.agent = "bound";
    } catch (err) {
        if (!isRoteMcpConfigError(err)) throw err;
        out.agent = "refused";
    }
    console.log(JSON.stringify(out));
"""

NODE_DRIVER_TS = (
    """\
import { bindAgentTools, callMcpTool, isRoteMcpConfigError } from "./extracted/_roteMcp";

async function main(): Promise<void> {
    const url = process.argv[2];
    const call = (tool: string) =>
        callMcpTool("ro", url, tool, tool === "lookup" ? { q: "x" } : { text: "x" });
    const bind = () => bindAgentTools(["lookup"], ["ro"], { ro: url }, { lookup: "ro" });
"""
    + _TS_BODY
    + "}\n\nmain();\n"
)

#: The Workers helper reads its variables from the Worker's env object;
#: the driver hands it this process's, as wrangler's --var would.
WORKERS_DRIVER_TS = (
    """\
import { bindAgentTools, callMcpTool, isRoteMcpConfigError } from "./extracted/_roteMcp";

const tokens = {
    get: async () => ({ access_token: "unused", expires_at: null }),
    put: async () => undefined,
};

async function main(): Promise<void> {
    const url = process.argv[2];
    const env = {
        ROTE_MCP_TOKENS: tokens,
        ROTE_MCP_READ_ONLY: process.env.ROTE_MCP_READ_ONLY,
        ROTE_MCP_READ_ONLY_SERVERS: process.env.ROTE_MCP_READ_ONLY_SERVERS,
    } as unknown as Parameters<typeof callMcpTool>[0];
    const call = (tool: string) =>
        callMcpTool(env, "ro", url, tool, tool === "lookup" ? { q: "x" } : { text: "x" });
    const bind = () =>
        bindAgentTools(env, ["lookup"], ["ro"], { ro: url }, { lookup: "ro" });
"""
    + _TS_BODY
    + "}\n\nmain();\n"
)


@pytest.fixture(scope="module")
def server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[str, Path]]:
    work = tmp_path_factory.mktemp("ro-server")
    (work / "server.py").write_text(SERVER_SCRIPT, encoding="utf-8")
    written = work / "written.txt"
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, str(work / "server.py"), str(port), str(written)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_port(port)
        yield f"http://127.0.0.1:{port}/mcp", written
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def _pipeline(url: str) -> Pipeline:
    return Pipeline.model_validate(
        {
            "name": "mcp-read-only-parity",
            "input": {"type": "In", "required": [], "optional": []},
            "nodes": [
                {
                    "id": "lookup",
                    "kind": "external_call",
                    "description": "Read.",
                    "mcp": {"server": "ro", "tool": "lookup", "url": url},
                }
            ],
            "edges": [],
            "entry_nodes": ["lookup"],
            "exit_nodes": ["lookup"],
        }
    )


def _python(url: str, env: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    from rote.inference import _runtime_helper as inference
    from rote.mcp import _runtime_helper as helper

    for key in CASES["read_only"]:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("ROTE_MCP_RO_URL", url)
    out: dict[str, Any] = {}
    for tool, args in (("lookup", {"q": "x"}), ("publish", {"text": "x"})):
        try:
            out[tool] = asyncio.run(helper.call_mcp_tool("ro", url, tool, args))
        except helper.RoteMcpConfigError as e:
            assert "may write" in str(e)
            out[tool] = "refused"
    # The Python agent loop binds its tools through the inference helper,
    # which imports the MCP helper as the emitted extracted._rote_mcp.
    sys.modules.setdefault("extracted", type(sys)("extracted"))
    sys.modules["extracted._rote_mcp"] = helper
    try:
        inference._mcp_config_for(["lookup"], {"lookup": "ro"})
        out["agent"] = "bound"
    except helper.RoteMcpConfigError:
        out["agent"] = "refused"
    return out


def _node_app(url: str, tmp_path: Path) -> Path:
    app = tmp_path / "node-app"
    get_adapter("dbos-ts").emit(_pipeline(url), app)
    (app / "src" / "driver.ts").write_text(NODE_DRIVER_TS, encoding="utf-8")
    node_trees.install(app)
    build = subprocess.run(["npx", "tsc"], cwd=app, capture_output=True, text=True, timeout=300)
    assert build.returncode == 0, build.stdout + build.stderr
    return app


def _workers_app(url: str, tmp_path: Path) -> Path:
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
    return app


def _ts(app: Path, script: str, url: str, env: dict[str, str]) -> dict[str, Any]:
    child = {k: v for k, v in os.environ.items() if k not in CASES["read_only"]}
    child.update(env)
    child["ROTE_MCP_RO_URL"] = url
    run = subprocess.run(
        ["node", script, url], cwd=app, capture_output=True, text=True, timeout=120, env=child
    )
    assert run.returncode == 0, run.stderr
    result: dict[str, Any] = json.loads(run.stdout.strip().splitlines()[-1])
    return result


@pytest.mark.parametrize("case", sorted(CASES))
def test_python_refuses_a_write_in_a_read_only_run(
    case: str, server: tuple[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    url, written = server
    before = written.read_text().count("\n") if written.exists() else 0
    assert _python(url, CASES[case], monkeypatch) == EXPECTED[case]
    after = written.read_text().count("\n") if written.exists() else 0
    # A refused write never reached the server.
    assert after - before == (0 if EXPECTED[case]["publish"] == "refused" else 1)


@pytest.mark.skipif(shutil.which("npm") is None, reason="requires node + npm")
def test_node_and_workers_refuse_the_same_writes(server: tuple[str, Path], tmp_path: Path) -> None:
    url, _written = server
    node = _node_app(url, tmp_path)
    workers = _workers_app(url, tmp_path)
    for case, env in CASES.items():
        assert _ts(node, "dist/driver.js", url, env) == EXPECTED[case], case
        assert _ts(workers, "dist/driver.mjs", url, env) == EXPECTED[case], case
