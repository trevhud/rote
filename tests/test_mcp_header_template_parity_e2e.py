"""Python and TS fill a registry header's ``${NAME}`` by one rule.

Two halves:

- **The rule**: the spliced TS source (``_HEADER_TEMPLATE_TS``) runs under
  node on the same table as the Python unit test
  (``tests/test_mcp_header_templates.py::CASES``), and every result and every
  error message must match exactly.
- **On the wire**: the Python helper and the emitted Node helper (dbos-ts)
  call one real fastmcp server whose tool returns the ``Authorization``
  header it received. Both must send the filled value, and both must fail
  with the config error, before any request, when the variable is unset.

Slow: node, plus npm install and tsc for the emitted app.
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
from rote.mcp import McpRegistry, McpServerConfig, save_registry
from tests import _node_trees as node_trees
from tests.test_mcp_header_templates import CASES
from tests.test_mcp_oauth_e2e import _free_port, _wait_port

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("node") is None, reason="requires node"),
]

RULE_DRIVER = """\
export class RoteMcpConfigError extends Error {
    constructor(message: string) {
        super(message);
        this.name = "RoteMcpConfigError";
    }
}

__SNIPPET__

const cases = JSON.parse(process.argv[2]) as [Record<string, string>, Record<string, string>][];
const out = cases.map(([headers, env]) => {
    try {
        return { ok: expandHeaders("srv", headers, env) };
    } catch (err) {
        return { error: (err as Error).message, name: (err as Error).name };
    }
});
console.log(JSON.stringify(out));
"""


def test_the_ts_rule_matches_the_python_rule(tmp_path: Path) -> None:
    driver = tmp_path / "driver.mts"
    driver.write_text(
        RULE_DRIVER.replace("__SNIPPET__", _ts_common._HEADER_TEMPLATE_TS), encoding="utf-8"
    )
    table = [[headers, env] for headers, env, _ in CASES]
    run = subprocess.run(
        ["node", "--experimental-strip-types", "--no-warnings", str(driver), json.dumps(table)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert run.returncode == 0, run.stderr
    results: list[dict[str, Any]] = json.loads(run.stdout)
    assert len(results) == len(CASES)
    for (headers, _env, expected), got in zip(CASES, results, strict=True):
        if isinstance(expected, str):
            assert got == {"error": expected, "name": "RoteMcpConfigError"}, headers
        else:
            assert got == {"ok": expected}, headers


def test_only_the_node_helper_has_static_headers_to_fill() -> None:
    """The Workers helper authenticates from provisioned OAuth secrets and the
    binding helper through the platform. If either gains a static-header
    path, it must splice the same rule; this fails first."""
    assert _ts_common._HEADER_TEMPLATE_TS in _ts_common.ROTE_MCP_HELPER_TS
    assert "return expandHeaders(server, entry.headers, process.env);" in (
        _ts_common.ROTE_MCP_HELPER_TS
    )
    assert "{ ...entry.headers }" not in _ts_common.ROTE_MCP_HELPER_TS
    for helper in (_ts_common.ROTE_MCP_WORKERS_HELPER_TS, _ts_common.ROTE_MCP_BINDING_HELPER_TS):
        assert "entry.headers" not in helper
        assert "registryServers" not in helper


SERVER_SCRIPT = """\
import sys

from fastmcp import FastMCP
from fastmcp.server.dependencies import get_http_request

mcp = FastMCP("echo-auth")


@mcp.tool
def whoami() -> str:
    return get_http_request().headers.get("authorization") or "<none>"


mcp.run(transport="http", host="127.0.0.1", port=int(sys.argv[1]))
"""

NODE_DRIVER_TS = """\
import { callMcpTool } from "./extracted/_roteMcp";

async function main(): Promise<void> {
    try {
        console.log(JSON.stringify({ ok: await callMcpTool("gh", null, "whoami", {}) }));
    } catch (err) {
        console.log(JSON.stringify({ name: (err as Error).name, error: (err as Error).message }));
    }
}

main();
"""


@pytest.fixture(scope="module")
def echo_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    script = tmp_path_factory.mktemp("echo-auth") / "server.py"
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


def _python_call() -> dict[str, Any]:
    from rote.mcp import _runtime_helper as helper

    try:
        return {"ok": asyncio.run(helper.call_mcp_tool("gh", None, "whoami", {}))}
    except helper.RoteMcpConfigError as e:
        return {"name": "RoteMcpConfigError", "error": str(e)}


def _pipeline() -> Pipeline:
    return Pipeline.model_validate(
        {
            "name": "header-parity",
            "input": {"type": "In", "required": [], "optional": []},
            "nodes": [
                {
                    "id": "who",
                    "kind": "external_call",
                    "description": "Ask the server who called it.",
                    "mcp": {"server": "gh", "tool": "whoami"},
                }
            ],
            "edges": [],
            "entry_nodes": ["who"],
            "exit_nodes": ["who"],
        }
    )


@pytest.mark.skipif(shutil.which("npm") is None, reason="requires npm")
def test_python_and_node_send_the_filled_header(
    echo_server: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    save_registry(
        McpRegistry(
            servers={
                "gh": McpServerConfig(url=echo_server, headers={"Authorization": "Bearer ${GH_T}"})
            }
        )
    )
    app = tmp_path / "node-app"
    get_adapter("dbos-ts").emit(_pipeline(), app)
    (app / "src" / "driver.ts").write_text(NODE_DRIVER_TS, encoding="utf-8")
    node_trees.install(app)
    build = subprocess.run(["npx", "tsc"], cwd=app, capture_output=True, text=True, timeout=300)
    assert build.returncode == 0, build.stdout + build.stderr

    def node_call() -> dict[str, Any]:
        run = subprocess.run(
            ["node", "dist/driver.js"],
            cwd=app,
            capture_output=True,
            text=True,
            timeout=120,
            env=dict(os.environ),
        )
        assert run.returncode == 0, run.stderr
        result: dict[str, Any] = json.loads(run.stdout.strip().splitlines()[-1])
        return result

    monkeypatch.setenv("GH_T", "filled-token")
    assert _python_call() == {"ok": "Bearer filled-token"}
    assert node_call() == {"ok": "Bearer filled-token"}

    monkeypatch.delenv("GH_T")
    unset = {
        "name": "RoteMcpConfigError",
        "error": "MCP server 'gh': header 'Authorization' needs GH_T, which is not set. Set "
        "GH_T in the environment the run starts in (rote never sends a header with an empty "
        "value). This is configuration, so the step is not retried.",
    }
    assert _python_call() == unset
    assert node_call() == unset
