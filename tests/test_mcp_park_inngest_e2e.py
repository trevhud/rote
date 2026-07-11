"""End-to-end: park-on-auth on the real Inngest runtime.

The broadcast-release loop, with every boundary real:

1. Emit an Inngest app for an MCP-backed pipeline (a single working MCP
   node — no mocks, no overlay); compile it; give its server a *dead*
   stored token so the Node helper throws RoteMcpAuthNeeded.
2. Trigger a run through the real dev server (`inngest-cli dev`). The
   step fails without burning retries (NonRetriableError wrap), the
   run parks on ``<pipeline>/rote.auth.vendor``.
3. Prove release precision: broadcasting a *different* server's release
   event leaves the run parked (status stays Running — Inngest has no
   "waiting" status; completion-only-after-the-right-event is the
   park's observable proof).
4. Fix the credential and run the real ``release_parked_workflows`` —
   it broadcasts the event through the dev server's event API, the
   ``waitForEvent`` wakes, the step retries against the live FastMCP
   server, and the run completes with real data.

Slow: npm install (downloads inngest-cli's platform binary) + tsc +
node + the dev server + a live FastMCP server. Reuses the proven
harness from tests/test_inngest_e2e.py.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from rote.adapters.inngest import InngestAdapter
from rote.app_registry import record_app
from rote.ir import MCPBinding, Node, NodeKind, Pipeline, PipelineInput, RetryPolicy
from rote.mcp import clear_token_file, write_token_file
from rote.mcp.release import release_parked_workflows

pytest.importorskip("fastmcp")

from tests.test_inngest_e2e import (  # noqa: E402 — reuse the proven live harness
    INNGEST_CLI_SPEC,
    _DevServer,
    _find_free_port,
    _http,
    _node_available,
    _npm_install,
    _wait_until_healthy,
)

pytestmark = pytest.mark.slow

_PIPELINE_NAME = "inngest_park_e2e"

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


@pytest.fixture(scope="module")
def mcp_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    src = tmp_path_factory.mktemp("mcpsrv") / "server.py"
    src.write_text(_MOCK_SERVER)
    port = _find_free_port()
    proc = subprocess.Popen(
        [sys.executable, str(src), str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + 20
    import socket

    while time.time() < deadline:
        with socket.socket() as s:
            s.settimeout(0.5)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                break
        time.sleep(0.2)
    else:
        proc.kill()
        raise RuntimeError(f"mock MCP server did not come up on :{port}")
    try:
        yield f"http://127.0.0.1:{port}/mcp/"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def _park_pipeline() -> Pipeline:
    node = Node(
        id="enrich",
        kind=NodeKind.EXTERNAL_CALL,
        description="Enrich a contact via the vendor MCP server.",
        input={"contact_id": "str"},
        inputs={"contact_id": "pipeline.input.contact_id"},
        output="dict",
        mcp=MCPBinding(server="vendor", tool="enrich_contact", args={"contact_id": "contact_id"}),
        # Retries on: the NonRetriableError wrap must keep auth failures
        # from burning this budget (Inngest backoff would delay the park
        # by minutes otherwise).
        retry=RetryPolicy(max=3, backoff="exponential"),
    )
    return Pipeline(
        name=_PIPELINE_NAME,
        description="Park-on-auth e2e pipeline (Inngest).",
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


def test_inngest_run_parks_and_broadcast_release_completes_it(
    mcp_server: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not _node_available():
        pytest.skip("Node / npm not available")
    import os

    out = tmp_path / "app"
    InngestAdapter().emit(_park_pipeline(), out)
    record_app(out, "inngest", _PIPELINE_NAME)
    write_token_file("vendor", _dead_token_doc(mcp_server))
    monkeypatch.setenv("ROTE_MCP_VENDOR_URL", mcp_server)

    _npm_install(out, "--ignore-scripts=false", INNGEST_CLI_SPEC)
    rebuild = subprocess.run(
        ["npm", "rebuild", "--ignore-scripts=false", "inngest-cli"],
        cwd=out,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert rebuild.returncode == 0, f"npm rebuild failed:\n{rebuild.stdout}\n{rebuild.stderr}"
    cli = out / "node_modules" / ".bin" / "inngest-cli"
    assert cli.exists(), "inngest-cli binary missing after install + rebuild"
    build = subprocess.run(
        ["npx", "--no-install", "tsc"], cwd=out, capture_output=True, text=True, timeout=120
    )
    assert build.returncode == 0, f"tsc failed:\n{build.stdout}\n{build.stderr}"

    app_port = _find_free_port()
    dev_port = _find_free_port()
    app_log_path = out / "app.log"
    dev_log_path = out / "dev-server.log"

    with app_log_path.open("w", encoding="utf-8") as app_log:
        app_proc = subprocess.Popen(
            ["node", "dist/index.js"],
            cwd=out,
            stdout=app_log,
            stderr=subprocess.STDOUT,
            env={
                **os.environ,
                "INNGEST_DEV": f"http://127.0.0.1:{dev_port}",
                "PORT": str(app_port),
            },
        )
    with dev_log_path.open("w", encoding="utf-8") as dev_log:
        dev_proc = subprocess.Popen(
            [
                str(cli),
                "dev",
                "-u",
                f"http://127.0.0.1:{app_port}",
                "--no-discovery",
                "--no-poll",
                "-p",
                str(dev_port),
            ],
            cwd=out,
            stdout=dev_log,
            stderr=subprocess.STDOUT,
        )

    try:
        _wait_until_healthy(
            f"http://127.0.0.1:{dev_port}/health", dev_proc, dev_log_path, "inngest dev server"
        )
        _wait_until_healthy(f"http://127.0.0.1:{app_port}/", app_proc, app_log_path, "emitted app")

        # Explicit registration (the --no-poll startup sync races app boot).
        deadline = time.time() + 60
        registered = False
        while time.time() < deadline:
            try:
                ack = _http("PUT", f"http://127.0.0.1:{app_port}/")
                if "registered" in str(ack.get("message", "")).lower():
                    registered = True
                    break
            except Exception:  # noqa: BLE001 — registration retry loop
                pass
            time.sleep(1)
        assert registered, (
            f"app never registered;\napp:\n{app_log_path.read_text()}\n"
            f"dev:\n{dev_log_path.read_text()}"
        )

        dev = _DevServer(dev_port)
        event_id = dev.send_event(f"{_PIPELINE_NAME}/run.requested", {"contact_id": "lead-3"})
        run_id = dev.run_id_for_event(event_id)

        # ── The run parks: after the step's auth failure + immediate
        # retry it must sit in Running (parked on waitForEvent), not
        # reach a terminal state. Give the executor time to prove it.
        parked_grace = time.time() + 15
        while time.time() < parked_grace:
            status = dev.run_status(run_id)
            assert status in ("Running", "Queued"), (
                f"run reached {status!r} instead of parking;\napp log:\n{app_log_path.read_text()}"
            )
            time.sleep(1)

        # ── Release precision: the wrong server's broadcast wakes nothing ──
        monkeypatch.setenv("ROTE_INNGEST_EVENT_URL", f"http://127.0.0.1:{dev_port}/e/dev")
        wrong = release_parked_workflows("other_server")
        assert [b.event for b in wrong.broadcasts] == [f"{_PIPELINE_NAME}/rote.auth.other_server"]
        time.sleep(3)
        assert dev.run_status(run_id) in ("Running", "Queued")  # still parked

        # ── Fix the credential, then broadcast the real release ──
        clear_token_file("vendor")  # mock server is unauthenticated
        report = release_parked_workflows("vendor")
        assert [b.event for b in report.broadcasts] == [f"{_PIPELINE_NAME}/rote.auth.vendor"]

        # ── The run wakes, retries the step, and completes with live data ──
        dev.wait_for_status(run_id, "Completed", timeout=90)
        result = dev.run_output(run_id)
        assert result["enrich"]["source"] == "mock-vendor-mcp"
        assert result["enrich"]["contact_id"] == "lead-3"
    finally:
        for proc in (app_proc, dev_proc):
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
