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
import select
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

dbos = pytest.importorskip("dbos", reason="dbos not installed (pip install 'rote-cli[dbos]')")

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
            assert status == {"awaiting": "vendor", "node": "enrich", "park": 1}

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


# ───────── rote run: the park is visible, and resolvable in place ─────────
#
# The test above drives the app directly, the way an unattended trigger
# would. These drive it through `rote run` as a user does: a real CLI
# subprocess spawns the real emitted app, and the run-side watcher
# (rote.mcp.parked) must announce its park and then resolve it.

_PROTECTED_SERVER = textwrap.dedent(
    """
    import sys
    from fastmcp import FastMCP
    from fastmcp.server.auth.providers.in_memory import InMemoryOAuthProvider
    from mcp.server.auth.settings import ClientRegistrationOptions

    port = int(sys.argv[1])
    auth = InMemoryOAuthProvider(
        base_url=f"http://127.0.0.1:{port}",
        client_registration_options=ClientRegistrationOptions(
            enabled=True, valid_scopes=["data:read"], default_scopes=["data:read"]
        ),
    )
    mcp = FastMCP("protected-vendor", auth=auth)

    @mcp.tool
    def enrich_contact(contact_id: str) -> dict:
        return {"contact_id": contact_id, "source": "protected-vendor-mcp"}

    mcp.run(transport="http", host="127.0.0.1", port=port, show_banner=False)
    """
)

#: A stand-in for the user's browser, used via $BROWSER (Python's
#: webbrowser honors it): follow the authorization URL's redirect chain.
#: The in-memory provider auto-approves, so this lands on the login's
#: local callback exactly as a human's click would. It detaches first,
#: like a real browser launcher: webbrowser runs $BROWSER synchronously,
#: and fastmcp only starts its callback server once the "browser" has
#: returned, so a follower that waited in the foreground could never
#: reach it. Retries cover the callback server binding its port.
_FOLLOW_BROWSER = textwrap.dedent(
    """
    import subprocess, sys, time

    if sys.argv[1:2] != ["--follow"]:
        subprocess.Popen([sys.executable, __file__, "--follow", sys.argv[1]],
                         start_new_session=True)
        sys.exit(0)

    import httpx
    deadline = time.monotonic() + 60
    while True:
        try:
            with httpx.Client(follow_redirects=True, timeout=30) as client:
                client.get(sys.argv[2])
            break
        except httpx.ConnectError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.5)
    """
)


@pytest.fixture(scope="module")
def protected_mcp_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    src = tmp_path_factory.mktemp("protected") / "server.py"
    src.write_text(_PROTECTED_SERVER)
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, str(src), str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_port(port)
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def _rote_run_args(app_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "rote",
        "run",
        str(app_dir),
        "--input",
        json.dumps({"contact_id": "lead-7"}),
        "--timeout",
        "120",
    ]


def test_rote_run_non_tty_announces_the_park_and_resumes_on_credential_fix(
    mcp_server: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Piped (no TTY): the park is announced naming server, node and the
    release command; nothing prompts; fixing the credential out of band
    is enough for the run to finish."""
    monkeypatch.delenv("DBOS_SYSTEM_DATABASE_URL", raising=False)
    app_dir = tmp_path / "app"
    DbosAdapter().emit(_park_pipeline(mcp_server), app_dir)
    write_token_file("vendor", _dead_token_doc(mcp_server))
    monkeypatch.setenv("ROTE_MCP_VENDOR_URL", mcp_server)

    stderr_path = tmp_path / "rote-run-stderr.log"
    with stderr_path.open("w") as stderr_f:
        proc = subprocess.Popen(
            _rote_run_args(app_dir),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=stderr_f,
            text=True,
        )
    try:
        deadline = time.time() + 90
        while "needs MCP credentials" not in stderr_path.read_text():
            assert proc.poll() is None, f"rote run exited early:\n{stderr_path.read_text()}"
            assert time.time() < deadline, f"no park message:\n{stderr_path.read_text()}"
            time.sleep(0.3)
        text = stderr_path.read_text()
        assert "node 'enrich' needs MCP credentials for server 'vendor'" in text
        assert "release it: rote mcp add vendor <url>, then rote mcp login vendor" in text
        assert "resumes on its own" in text

        # The fix: the mock server is unauthenticated, so clearing the dead
        # token is the whole repair (a login stores a fresh one instead).
        # Nobody calls release: the watcher notices and releases its run.
        clear_token_file("vendor")
        stdout, _ = proc.communicate(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
    text = stderr_path.read_text()
    assert proc.returncode == 0, text
    assert "credentials for 'vendor' changed, releasing the parked run" in text
    assert "Log in to" not in text  # never prompted
    result = json.loads(stdout)
    assert result["enrich"] == {"contact_id": "lead-7", "source": "mock-vendor-mcp"}


def test_rote_run_on_a_tty_offers_the_login_and_completes_after_it(
    protected_mcp_server: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On a terminal: the park is announced, the browser login is offered
    in place, and accepting it runs the real OAuth dance, releases the
    run, and the run completes against the protected server."""
    import pty

    from rote.mcp import McpRegistry, McpServerConfig, save_registry

    monkeypatch.delenv("DBOS_SYSTEM_DATABASE_URL", raising=False)
    save_registry(
        McpRegistry(
            servers={"vendor": McpServerConfig(url=protected_mcp_server, scopes=["data:read"])}
        )
    )
    app_dir = tmp_path / "app"
    DbosAdapter().emit(_park_pipeline(protected_mcp_server), app_dir)
    browser = tmp_path / "browser.py"
    browser.write_text(_FOLLOW_BROWSER)
    monkeypatch.setenv("BROWSER", f"{sys.executable} {browser} %s")

    leader, follower = pty.openpty()
    proc = subprocess.Popen(
        _rote_run_args(app_dir),
        stdin=follower,
        stdout=subprocess.PIPE,
        stderr=follower,
        text=True,
    )
    os.close(follower)
    transcript = ""

    def read_until(needle: str, timeout: float) -> None:
        nonlocal transcript
        deadline = time.time() + timeout
        while needle not in transcript:
            assert time.time() < deadline, f"never saw {needle!r}:\n{transcript}"
            # select first: a bare read would block past the deadline.
            readable, _, _ = select.select([leader], [], [], 0.5)
            if not readable:
                continue
            try:
                chunk = os.read(leader, 4096)
            except OSError as e:  # EIO: the child closed the terminal
                raise AssertionError(f"terminal closed before {needle!r}:\n{transcript}") from e
            transcript += chunk.decode("utf-8", errors="replace")

    try:
        read_until("[Y/n] ", timeout=90)
        assert "node 'enrich' needs MCP credentials for server 'vendor'" in transcript
        assert "release it: rote mcp login vendor" in transcript
        os.write(leader, b"y\n")
        read_until("releasing the parked run", timeout=90)
        assert "authenticated 'vendor', releasing the parked run" in transcript
        stdout, _ = proc.communicate(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
        os.close(leader)
    assert proc.returncode == 0, transcript
    result = json.loads(stdout)
    assert result["enrich"] == {"contact_id": "lead-7", "source": "protected-vendor-mcp"}
