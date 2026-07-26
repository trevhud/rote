"""End-to-end: an MCP-backed DBOS pipeline run against a live MCP server.

Slow (spawns a real FastMCP Streamable-HTTP server and runs the emitted
DBOS app as a subprocess over SQLite). Proves the thing unit tests can't:
a compiled pipeline with an ``mcp:`` binding actually calls the tool and
returns live data — keyless, no stubs — and that failure modes surface
loudly rather than silently. The fast-suite counterpart is
``test_mcp_backend.py``.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import textwrap
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from rote.adapters.dbos import DbosAdapter
from rote.ir import MCPBinding, Node, NodeKind, Pipeline, PipelineInput

pytestmark = pytest.mark.slow

_MOCK_SERVER = textwrap.dedent(
    """
    import sys
    from fastmcp import FastMCP

    mcp = FastMCP("mock-vendor")

    @mcp.tool
    def enrich_contact(contact_id: str) -> dict:
        return {
            "contact_id": contact_id,
            "email": f"{contact_id}@example.com",
            "accuracy_score": 92,
            "source": "mock-vendor-mcp",
        }

    @mcp.tool
    def always_fails(contact_id: str) -> dict:
        raise ValueError("simulated vendor outage")

    if __name__ == "__main__":
        mcp.run(transport="http", host="127.0.0.1", port=int(sys.argv[1]))
    """
)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_port(port: int, timeout: float = 20.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket() as s:
            s.settimeout(0.5)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.2)
    raise RuntimeError(f"mock MCP server did not come up on :{port}")


@pytest.fixture(scope="module")
def mcp_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    src = tmp_path_factory.mktemp("mcpsrv") / "server.py"
    src.write_text(_MOCK_SERVER)
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


def _enrich_pipeline(name: str, tool: str = "enrich_contact") -> Pipeline:
    node = Node(
        id="enrich",
        kind=NodeKind.EXTERNAL_CALL,
        description="Enrich a contact via the vendor MCP server.",
        input={"contact_id": "str"},
        inputs={"contact_id": "pipeline.input.contact_id"},
        output="dict",
        mcp=MCPBinding(server="vendor", tool=tool, args={"contact_id": "contact_id"}),
    )
    return Pipeline(
        name=name,
        description="MCP-backed e2e pipeline.",
        input=PipelineInput(type="Req", required=["contact_id"]),
        nodes=[node],
        edges=[],
        entry_nodes=["enrich"],
        exit_nodes=["enrich"],
    )


def _emit_and_run(
    tmp_path: Path, pipeline: Pipeline, env_extra: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    out = tmp_path / "app"
    DbosAdapter().emit(pipeline, out)  # default external_backend="mcp"
    env = {
        **os.environ,
        "DBOS_SYSTEM_DATABASE_URL": f"sqlite:///{out / 'dbos.sqlite'}",
        **env_extra,
    }
    return subprocess.run(
        [sys.executable, "main.py", json.dumps({"contact_id": "lead-42"})],
        cwd=out,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_mcp_backed_step_runs_against_live_server(mcp_server: str, tmp_path: Path) -> None:
    proc = _emit_and_run(
        tmp_path, _enrich_pipeline("mcp_e2e_ok"), {"ROTE_MCP_VENDOR_URL": mcp_server}
    )
    assert proc.returncode == 0, proc.stderr
    assert "mock-vendor-mcp" in proc.stdout  # data came from the live server, not a stub
    assert "lead-42@example.com" in proc.stdout


def test_missing_server_url_fails_loud(tmp_path: Path) -> None:
    proc = _emit_and_run(tmp_path, _enrich_pipeline("mcp_e2e_noenv"), {})  # url unset
    assert proc.returncode != 0
    assert "ROTE_MCP_VENDOR_URL" in (proc.stderr + proc.stdout)
    assert "mock-vendor-mcp" not in proc.stdout  # no silent success


def test_mcp_tool_error_propagates(mcp_server: str, tmp_path: Path) -> None:
    proc = _emit_and_run(
        tmp_path,
        _enrich_pipeline("mcp_e2e_err", tool="always_fails"),
        {"ROTE_MCP_VENDOR_URL": mcp_server},
    )
    assert proc.returncode != 0
    assert "simulated vendor outage" in (proc.stderr + proc.stdout)  # not swallowed
