"""OAuth end-to-end: the real dance against a real protected server.

Slow (spawns a FastMCP Streamable-HTTP server protected by the full
in-memory OAuth provider — authorization endpoint, token endpoint,
dynamic client registration, refresh grant). The client side runs
rote's REAL code path: ``rote.mcp.auth.login`` → fastmcp ``OAuth`` →
the SDK's spec flow → rote's durable token store. The only test double
is ``webbrowser.open``, replaced by a thread that GETs the
authorization URL — the in-memory provider auto-approves and redirects
to the local callback server, exactly as a human's browser would.

Why slow: real uvicorn servers on real sockets, a real multi-request
OAuth exchange. Everything else in the suite mocks network; this test
exists precisely because the auth stack is too load-bearing to trust
to mocks.
"""

from __future__ import annotations

import asyncio
import json
import socket
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Iterator

import pytest

pytestmark = pytest.mark.slow

pytest.importorskip("fastmcp")
import httpx  # noqa: E402

from rote.mcp import (  # noqa: E402
    McpRegistry,
    McpServerConfig,
    access_token_state,
    read_token_file,
    save_registry,
    token_file,
    write_token_file,
)
from rote.mcp.auth import fresh_access_token, login  # noqa: E402

SERVER_SCRIPT = """\
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
def whoami(name: str) -> dict:
    return {"hello": name, "authenticated": True}

mcp.run(transport="http", host="127.0.0.1", port=port, show_banner=False)
"""


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_port(port: int, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(0.2)
    raise RuntimeError(f"server on :{port} never came up")


@pytest.fixture(scope="module")
def protected_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    script = tmp_path_factory.mktemp("oauth-server") / "server.py"
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


@pytest.fixture()
def headless_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the browser with a thread that follows the authorization
    redirect chain — the provider auto-approves, so GET(authorize_url)
    302s straight to fastmcp's local callback server."""

    def fake_open(url: str, *args: object, **kwargs: object) -> bool:
        def follow() -> None:
            # A real browser takes human-scale time to follow the chain;
            # this thread can outrun fastmcp's local callback server
            # binding its port. Retry the whole chain (each attempt mints
            # a fresh auth code; the provider honors the same state)
            # until the callback hop stops refusing connections.
            deadline = time.monotonic() + 30
            while True:
                try:
                    with httpx.Client(follow_redirects=True, timeout=30) as client:
                        client.get(url)
                    return
                except httpx.ConnectError:
                    if time.monotonic() > deadline:
                        raise
                    time.sleep(0.5)

        threading.Thread(target=follow, daemon=True).start()
        return True

    monkeypatch.setattr("webbrowser.open", fake_open)


def test_full_oauth_dance_persists_reuses_and_refreshes(
    protected_server: str, headless_browser: None
) -> None:
    save_registry(
        McpRegistry(servers={"vendor": McpServerConfig(url=protected_server, scopes=["data:read"])})
    )
    config = McpServerConfig(url=protected_server, scopes=["data:read"])

    # 1. The dance: discovery → DCR → PKCE authorization-code → tokens
    #    land durably in rote's store.
    doc = asyncio.run(login("vendor", config))
    assert (doc["tokens"] or {}).get("access_token")
    assert doc["client_info"] and doc["client_info"].get("client_id")  # DCR happened
    path = token_file("vendor")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600

    # 2. Cross-process reuse: a fresh consumer (no dance) gets a valid
    #    token straight from the store — this is what emitted workflows
    #    and `rote mcp headers` rely on.
    first_token = asyncio.run(fresh_access_token("vendor", config))
    assert first_token == doc["tokens"]["access_token"]

    # 3. The emitted helper authenticates a real tool call end-to-end
    #    from the same store (the code path generated DBOS apps run).
    from rote.mcp import _runtime_helper

    async def call_tool() -> dict:
        async with _runtime_helper.mcp_client("vendor", None) as client:
            result = await client.call_tool("whoami", {"name": "rote"})
            return dict(result.data)

    assert asyncio.run(call_tool()) == {"hello": "rote", "authenticated": True}

    # 4. Refresh: force the access token stale; fresh_access_token must
    #    come back valid via the refresh grant, and the rotated tokens
    #    must be durable (written back through the same store).
    stale = read_token_file("vendor")
    assert stale is not None
    stale["expires_at"] = time.time() - 10
    write_token_file("vendor", stale)
    refreshed = asyncio.run(fresh_access_token("vendor", config))
    after = read_token_file("vendor")
    assert after is not None
    assert refreshed == after["tokens"]["access_token"]
    tok, fresh = access_token_state(after)
    assert tok == refreshed and fresh

    # 5. The headers command mints a usable Authorization header — the
    #    exact contract Claude Code's headersHelper invokes.
    import contextlib
    import io

    from rote.cli import main as cli_main

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = cli_main(["mcp", "headers", "vendor"])
    assert rc == 0
    headers = json.loads(buf.getvalue())
    assert headers["Authorization"].startswith("Bearer ")


def test_unauthenticated_client_is_rejected_by_the_server(protected_server: str) -> None:
    """Sanity: the server really is protected — a bare client 401s.
    (Without this, test 1 could pass against an open server.)"""
    import fastmcp

    async def bare_call() -> None:
        async with fastmcp.Client(protected_server) as client:
            await client.ping()

    with pytest.raises(Exception, match="[Uu]nauthorized|401|[Aa]uth"):
        asyncio.run(bare_call())
