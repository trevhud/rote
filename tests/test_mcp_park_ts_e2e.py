"""End-to-end: park-on-auth on the real DBOS *TypeScript* runtime.

The cross-language production loop, with every boundary real:

1. Emit a dbos-ts app for an MCP-backed pipeline; compile it (npm +
   tsc); give its server a *dead* stored token so the Node helper's
   auth preflight throws RoteMcpAuthNeeded.
2. Run it against a Docker Postgres. The TS workflow parks — verified
   from PYTHON by reading the ``rote_auth_status`` workflow event
   through ``DBOSClient`` (this only works because the emitted TS code
   writes the event in DBOS's portable serialization; the TS-native
   superjson format would raise in the Python client).
3. Prove release precision (wrong server wakes nothing), fix the
   credential, and run the real ``release_parked_workflows`` — the
   Python-sent portable message crosses into the TS ``DBOS.recv``, the
   step retries against the live MCP server, and the run completes.

Slow: Docker Postgres + npm install + tsc + node + a live FastMCP
server.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import textwrap
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from rote.adapters.dbos_ts import DbosTsAdapter
from rote.app_registry import record_app
from rote.ir import MCPBinding, Node, NodeKind, Pipeline, PipelineInput, RetryPolicy
from rote.mcp import clear_token_file, write_token_file
from rote.mcp.release import release_parked_workflows

dbos = pytest.importorskip("dbos", reason="dbos not installed (pip install rote[dbos])")
pytest.importorskip("fastmcp")

pytestmark = pytest.mark.slow

_PIPELINE_NAME = "ts_park_e2e"

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


def _docker_available() -> bool:
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=15).returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _node_available() -> bool:
    try:
        return (
            subprocess.run(["node", "--version"], capture_output=True, timeout=15).returncode == 0
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


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


@pytest.fixture(scope="module")
def postgres_url() -> Iterator[str]:
    """Boot a throwaway Docker Postgres; yield the app's system DB URL."""
    if not _docker_available():
        pytest.skip("Docker daemon not available — skipping dbos-ts park e2e")
    name = f"rote-ts-park-e2e-{uuid.uuid4().hex[:8]}"
    port = _free_port()
    run = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--rm",
            "--name",
            name,
            "-p",
            f"{port}:5432",
            "-e",
            "POSTGRES_PASSWORD=dbos",
            "postgres:16-alpine",
        ],
        capture_output=True,
        text=True,
        timeout=300,  # includes a possible first-time image pull
    )
    if run.returncode != 0:
        pytest.fail(f"docker run failed: {run.stdout}\n{run.stderr}")
    try:
        deadline = time.time() + 60
        while time.time() < deadline:
            ready = subprocess.run(
                ["docker", "exec", name, "pg_isready", "-U", "postgres"],
                capture_output=True,
                timeout=10,
            )
            if ready.returncode == 0:
                break
            time.sleep(1)
        else:
            pytest.fail("Postgres container did not become ready in 60s")
        yield f"postgresql://postgres:dbos@localhost:{port}/{_PIPELINE_NAME}_dbos_sys"
    finally:
        subprocess.run(["docker", "stop", name], capture_output=True, timeout=60)


def _park_pipeline() -> Pipeline:
    node = Node(
        id="enrich",
        kind=NodeKind.EXTERNAL_CALL,
        description="Enrich a contact via the vendor MCP server.",
        input={"contact_id": "str"},
        inputs={"contact_id": "pipeline.input.contact_id"},
        output="dict",
        mcp=MCPBinding(server="vendor", tool="enrich_contact", args={"contact_id": "contact_id"}),
        # Retries on: proves the emitted shouldRetry keeps auth failures
        # out of the retry budget on the real TS runtime.
        retry=RetryPolicy(max=3, backoff="exponential"),
    )
    return Pipeline(
        name=_PIPELINE_NAME,
        description="Park-on-auth e2e pipeline (TS).",
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


def _extract_result_json(stdout: str) -> dict[str, Any]:
    """Pull the pretty-printed result object out of the app's stdout.

    DBOS TS's logger writes to stdout too, so the stream is log lines +
    one pretty-printed JSON object. The object's opening brace is the
    only bare "{" line; raw_decode from there ignores any trailing
    shutdown logs.
    """
    lines = stdout.splitlines()
    idx = lines.index("{")
    obj, _ = json.JSONDecoder().raw_decode("\n".join(lines[idx:]))
    assert isinstance(obj, dict)
    return obj


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


def test_ts_workflow_parks_and_python_release_completes_it(
    mcp_server: str,
    postgres_url: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not _node_available():
        pytest.skip("Node / npm not available")

    # Both the TS app and the Python release path resolve the system DB
    # from this env var — same rule, verified same URL.
    monkeypatch.setenv("DBOS_SYSTEM_DATABASE_URL", postgres_url)

    app_dir = tmp_path / "app"
    DbosTsAdapter().emit(_park_pipeline(), app_dir)
    record_app(app_dir, "dbos-ts", _PIPELINE_NAME)
    write_token_file("vendor", _dead_token_doc(mcp_server))
    monkeypatch.setenv("ROTE_MCP_VENDOR_URL", mcp_server)

    install = subprocess.run(
        ["npm", "install", "--no-audit", "--no-fund"],
        cwd=app_dir,
        capture_output=True,
        text=True,
        timeout=600,
        env={**os.environ, "npm_config_progress": "false"},
    )
    assert install.returncode == 0, f"npm install failed:\n{install.stdout}\n{install.stderr}"
    build = subprocess.run(
        ["npx", "--no-install", "tsc"], cwd=app_dir, capture_output=True, text=True, timeout=300
    )
    assert build.returncode == 0, f"tsc failed:\n{build.stdout}\n{build.stderr}"

    stderr_path = tmp_path / "app-stderr.log"
    with stderr_path.open("w") as stderr_f:
        proc = subprocess.Popen(
            ["node", "dist/main.js", json.dumps({"contact_id": "lead-9"})],
            cwd=app_dir,
            env={**os.environ},
            stdout=subprocess.PIPE,
            stderr=stderr_f,
            text=True,
        )
    try:
        workflow_id = _wait_for_workflow_id(stderr_path, proc, timeout=90)

        # ── The TS workflow parks; PYTHON reads its advertisement ──
        # (only possible because the emitted setEvent uses the portable
        # serialization — superjson would raise in the Python client)
        from dbos import DBOSClient

        client = DBOSClient(system_database_url=postgres_url)
        try:
            deadline = time.time() + 90
            status = None
            while time.time() < deadline:
                status = client.get_event(workflow_id, "rote_auth_status", timeout_seconds=1)
                if status is not None:
                    break
                assert proc.poll() is None, (
                    f"app exited instead of parking:\n{stderr_path.read_text()}"
                )
            assert status == {"awaiting": "vendor"}

            # ── Release precision: another server's login wakes nothing ──
            wrong = release_parked_workflows("other_server")
            assert wrong.released == ()

            # ── Fix the credential, then release from Python ──
            clear_token_file("vendor")  # mock server is unauthenticated
            report = release_parked_workflows("vendor")
            assert [r.workflow_id for r in report.released] == [workflow_id]

            # ── The TS workflow wakes, retries the step, completes ──
            stdout, _ = proc.communicate(timeout=120)
            assert proc.returncode == 0, stderr_path.read_text()
            result = _extract_result_json(stdout)
            assert result["enrich"]["source"] == "mock-vendor-mcp"
            assert result["enrich"]["contact_id"] == "lead-9"

            # The park advertisement was cleared on release.
            assert client.get_event(workflow_id, "rote_auth_status", timeout_seconds=0) is None
        finally:
            client.destroy()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
