"""End-to-end: park-on-auth against a real DBOS runtime, cross-process.

The full production loop, with the emitted app running as a subprocess
(exactly how users run it) and the release side driving it from this
process the way ``rote mcp login`` does:

1. Emit an MCP-backed DBOS app; give its server a *dead* stored token
   (expired, no refresh token) so the auth preflight fails.
2. Start a run. The step raises RoteMcpAuthNeeded; the workflow parks —
   verified by reading the ``rote_auth_status`` workflow event through a
   real ``DBOSClient`` from this process.
3. Prove release precision: releasing a *different* server's parked
   workflows must not wake this one.
4. Fix the credential (clear the dead token — the mock server is
   unauthenticated, so no login is needed) and run the real
   ``release_parked_workflows``. The workflow wakes, retries the step,
   calls the live MCP tool, and completes with real data.

Slow: real FastMCP HTTP server + real DBOS runtime over SQLite.
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
from typing import Any

import pytest

from rote.adapters.dbos import DbosAdapter
from rote.app_registry import record_app
from rote.ir import MCPBinding, Node, NodeKind, Pipeline, PipelineInput, RetryPolicy
from rote.mcp import clear_token_file, write_token_file
from rote.mcp.release import release_parked_workflows

dbos = pytest.importorskip("dbos", reason="dbos not installed (pip install rote[dbos])")

pytestmark = pytest.mark.slow

_MOCK_SERVER = textwrap.dedent(
    """
    import sys
    from fastmcp import FastMCP

    mcp = FastMCP("mock-vendor")

    @mcp.tool
    def enrich_contact(contact_id: str) -> dict:
        return {"contact_id": contact_id, "source": "mock-vendor-mcp"}

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


def _park_pipeline(url: str) -> Pipeline:
    node = Node(
        id="enrich",
        kind=NodeKind.EXTERNAL_CALL,
        description="Enrich a contact via the vendor MCP server.",
        input={"contact_id": "str"},
        inputs={"contact_id": "pipeline.input.contact_id"},
        output="dict",
        mcp=MCPBinding(server="vendor", tool="enrich_contact", args={"contact_id": "contact_id"}),
        # Retries on: proves the emitted should_retry keeps auth failures
        # out of the retry budget on the real runtime (a burnt budget
        # would delay the park by interval_seconds * attempts).
        retry=RetryPolicy(max=3, backoff="exponential"),
    )
    return Pipeline(
        name="park_e2e",
        description="Park-on-auth e2e pipeline.",
        input=PipelineInput(type="Req", required=["contact_id"]),
        nodes=[node],
        edges=[],
        entry_nodes=["enrich"],
        exit_nodes=["enrich"],
    )


def _dead_token_doc(url: str) -> dict[str, Any]:
    return {
        "server_url": url,
        "tokens": {"access_token": "expired-tok", "token_type": "Bearer"},  # no refresh_token
        "expires_at": time.time() - 3600,
        "client_info": None,
        "token_endpoint": None,
    }


def _wait_for_workflow_id(stderr_path: Path, proc: subprocess.Popen[Any], timeout: float) -> str:
    marker = "workflow started: "
    deadline = time.time() + timeout
    while time.time() < deadline:
        text = stderr_path.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            if line.startswith(marker):
                return line[len(marker) :].strip()
        if proc.poll() is not None:
            raise AssertionError(f"app exited (code {proc.returncode}) before starting:\n{text}")
        time.sleep(0.2)
    raise AssertionError("timed out waiting for the workflow id")


def test_workflow_parks_on_dead_token_and_login_release_completes_it(
    mcp_server: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The release path must derive the same system-DB URL the app uses —
    # keep both on the app's default SQLite file, no env override.
    monkeypatch.delenv("DBOS_SYSTEM_DATABASE_URL", raising=False)

    app_dir = tmp_path / "app"
    DbosAdapter().emit(_park_pipeline(mcp_server), app_dir)
    record_app(app_dir, "dbos", "park_e2e")
    write_token_file("vendor", _dead_token_doc(mcp_server))
    monkeypatch.setenv("ROTE_MCP_VENDOR_URL", mcp_server)

    stderr_path = tmp_path / "app-stderr.log"
    with stderr_path.open("w") as stderr_f:
        proc = subprocess.Popen(
            [sys.executable, "main.py", json.dumps({"contact_id": "lead-7"})],
            cwd=app_dir,
            env={**os.environ},
            stdout=subprocess.PIPE,
            stderr=stderr_f,
            text=True,
        )
    try:
        workflow_id = _wait_for_workflow_id(stderr_path, proc, timeout=60)

        # ── The workflow parks, advertising what it waits for ──
        from dbos import DBOSClient

        from rote._dbos import dbos_system_database_url

        client = DBOSClient(system_database_url=dbos_system_database_url(app_dir))
        try:
            deadline = time.time() + 60
            status = None
            while time.time() < deadline:
                status = client.get_event(workflow_id, "rote_auth_status", timeout_seconds=1)
                if status is not None:
                    break
                assert proc.poll() is None, (
                    f"app exited instead of parking:\n{stderr_path.read_text()}"
                )
            assert status == {"awaiting": "vendor"}

            # ── Release precision: another server's login must not wake it ──
            wrong = release_parked_workflows("other_server")
            assert wrong.released == ()

            # ── Fix the credential, then release for real ──
            # The mock server is unauthenticated: clearing the dead token
            # makes the plain client work (in production this is where
            # `rote mcp login vendor` stores a fresh token instead).
            clear_token_file("vendor")
            report = release_parked_workflows("vendor")
            assert [(r.workflow_id) for r in report.released] == [workflow_id]
            assert report.skipped == ()

            # ── The workflow wakes, retries the step, and completes ──
            stdout, _ = proc.communicate(timeout=60)
            assert proc.returncode == 0, stderr_path.read_text()
            result = json.loads(stdout)
            assert result["enrich"]["source"] == "mock-vendor-mcp"
            assert result["enrich"]["contact_id"] == "lead-7"

            # The park advertisement was cleared on release.
            assert client.get_event(workflow_id, "rote_auth_status", timeout_seconds=0) is None
        finally:
            client.destroy()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
