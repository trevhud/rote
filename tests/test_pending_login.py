"""The pending-login record and the checks around it, without a server.

The full flow (a real listener, a real authorization server, a simulated
browser) is tests/test_mcp_login_no_wait_e2e.py.
"""

from __future__ import annotations

import io
import json
import os
import stat
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from rote.cli import main as cli_main
from rote.interact.report import authorization_pending
from rote.mcp import pending_login
from rote.mcp.pending_login import PendingLogin, PendingLoginError


def _record(server: str = "vendor", **changes: Any) -> PendingLogin:
    pending = PendingLogin.new(server, "https://vendor.example/mcp")
    for key, value in changes.items():
        setattr(pending, key, value)
    pending_login.write_pending(pending)
    return pending


def test_the_record_is_owner_only_in_an_owner_only_directory() -> None:
    pending = _record()
    path = pending_login.pending_path(pending.id)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    data = json.loads(path.read_text())
    assert set(data) == {
        "id",
        "server",
        "server_url",
        "created_at",
        "expires_at",
        "status",
        "pid",
        "port",
        "error",
        "result",
        "rejected_callbacks",
    }
    assert data["expires_at"] - data["created_at"] == pending_login.PENDING_TTL_SECONDS


@pytest.mark.parametrize("bad", ["../../etc/passwd", "ABCDEF0123456789", "short", "a" * 17])
def test_an_id_can_never_name_another_file(bad: str) -> None:
    with pytest.raises(PendingLoginError, match="not a pending login id"):
        pending_login.pending_path(bad)


def test_resolve_needs_one_unambiguous_record() -> None:
    with pytest.raises(PendingLoginError, match="no pending login"):
        pending_login.resolve_pending(None, None)
    first = _record("vendor")
    assert pending_login.resolve_pending(None, None).id == first.id
    second = _record("other")
    with pytest.raises(PendingLoginError, match="several logins are pending"):
        pending_login.resolve_pending(None, None)
    assert pending_login.resolve_pending(None, "other").id == second.id
    with pytest.raises(PendingLoginError, match="is for 'vendor', not 'other'"):
        pending_login.resolve_pending(first.id, "other")


def test_expired_records_are_swept_and_never_block_a_new_login() -> None:
    old = _record(created_at=time.time() - 1000, expires_at=time.time() - 400)
    assert pending_login.live_for_server("vendor") is None
    assert pending_login.sweep_expired() == [old.id]
    assert pending_login.read_pending(old.id) is None


def test_a_listener_that_died_is_a_failure_not_a_wait() -> None:
    pending = _record(status="waiting", pid=2**22 + 12345)  # no such process
    outcome = pending_login.wait_for_outcome(pending, wait=5)
    assert outcome.kind == "failed"
    assert "stopped without a result" in (outcome.pending.error or "")
    assert pending_login.read_pending(pending.id) is None


@pytest.mark.parametrize(
    ("url", "message"),
    [
        ("https://evil.example/callback?code=x&state=y", "must be the http://localhost"),
        ("http://localhost:33419/callback?code=x&state=y", "not this login's callback"),
        ("http://localhost:33418/other?code=x&state=y", "not this login's callback"),
        ("http://localhost:33418/callback?state=y", "carries no code"),
        ("http://localhost:abc/callback?code=x&state=y", "invalid port"),
        ("http://localhost:99999/callback?code=x&state=y", "invalid port"),
    ],
)
def test_a_pasted_callback_is_only_ever_sent_to_this_logins_listener(
    url: str, message: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[str] = []
    monkeypatch.setattr("httpx.get", lambda target, **_: sent.append(target))
    # A live listener: this test process stands in for it.
    pending = _record(status="waiting", port=33418, pid=os.getpid())
    with pytest.raises(PendingLoginError, match=message):
        pending_login.replay_callback(pending, url)
    assert sent == []
    pending_login.replay_callback(pending, "http://localhost:33418/callback?code=x&state=y")
    assert sent == ["http://127.0.0.1:33418/callback?code=x&state=y"]


def test_the_state_check_drops_a_forged_success_and_keeps_waiting() -> None:
    from fastmcp.client.oauth_callback import OAuthCallbackResult

    from rote.mcp.auth import _StateCheckedReady

    rejected: list[int] = []
    result = OAuthCallbackResult()
    ready = _StateCheckedReady(result, "expected-state", lambda: rejected.append(1))
    result.code, result.state = "forged-code", "other-state"
    ready.set()
    assert not ready.is_set()
    assert (result.code, result.state) == (None, None)
    assert rejected == [1]
    result.code, result.state = "real-code", "expected-state"
    ready.set()
    assert ready.is_set() and result.code == "real-code"


def test_the_authorization_pending_object_matches_action_required() -> None:
    payload = authorization_pending(
        server="vendor",
        pending_id="0123456789abcdef",
        authorization_url="https://as.example/authorize?state=s",
        expires_at=1.0,
    )
    assert payload["ok"] is True
    assert (payload["status"], payload["reason"]) == ("action_required", "authorization_pending")
    assert [n["command"] for n in payload["next"]] == [
        "rote mcp login --complete 0123456789abcdef",
        "rote mcp login --complete 0123456789abcdef --callback-url -",
    ]
    waiting = authorization_pending(
        server="vendor",
        pending_id="0123456789abcdef",
        authorization_url=None,
        expires_at=1.0,
        still_waiting=True,
    )
    assert waiting["ok"] is False


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["mcp", "login", "vendor", "--no-wait", "--complete"], "cannot be combined"),
        (["mcp", "login", "--complete", "--cancel"], "cannot be combined"),
        (
            ["mcp", "login", "vendor", "--callback-url", "http://localhost:1/callback"],
            "goes with --complete",
        ),
        (["mcp", "login"], "name the server"),
    ],
)
def test_login_flag_combinations_are_usage_errors(
    argv: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli_main(argv) == 2
    assert message in capsys.readouterr().err


def test_complete_without_a_pending_login_says_how_to_start_one(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli_main(["mcp", "login", "--complete", "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert "rote mcp login <server> --no-wait" in payload["error"]


class _Squatter(BaseHTTPRequestHandler):
    """Whatever bound a callback port after the listener let it go."""

    seen: list[str] = []

    def do_GET(self) -> None:  # noqa: N802 (http.server's name)
        _Squatter.seen.append(self.path)
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args: object) -> None:
        pass


@pytest.mark.parametrize(
    "state",
    [
        {"status": "done", "pid": 2**22 + 12345},
        {"status": "failed", "pid": 2**22 + 12345},
        {"status": "waiting", "pid": 2**22 + 12345},  # the listener died
        {"status": "waiting", "pid": None},
        {"status": "waiting", "pid": os.getpid(), "expires_at": time.time() - 1},
    ],
)
def test_a_pasted_callback_never_reaches_a_port_the_listener_left(state: dict[str, Any]) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Squatter)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    _Squatter.seen = []
    try:
        port = server.server_address[1]
        pending = _record(port=port, **state)
        url = f"http://localhost:{port}/callback?code=the-code&state=the-state"
        with pytest.raises(PendingLoginError, match="nothing was sent"):
            pending_login.replay_callback(pending, url)
        assert _Squatter.seen == []
        # The same URL does reach a listener that is alive and waiting.
        live = _record(server="other", port=port, status="waiting", pid=os.getpid())
        pending_login.replay_callback(live, url)
        assert _Squatter.seen == ["/callback?code=the-code&state=the-state"]
    finally:
        server.shutdown()
        server.server_close()


def _listen_with(fake_login: Any, monkeypatch: pytest.MonkeyPatch, **record: Any) -> PendingLogin:
    """Run the listener in this process with ``fake_login`` in place of the OAuth dance."""
    from rote.mcp import McpRegistry, McpServerConfig, save_registry

    save_registry(
        McpRegistry(servers={"vendor": McpServerConfig(url="https://vendor.example/mcp")})
    )
    pending = _record(**record)
    monkeypatch.setattr("rote.mcp.auth.login", fake_login)
    monkeypatch.setattr(sys, "stdout", io.StringIO())  # the handshake pipe
    assert pending_login._listen(pending.id) == 1
    return pending


def test_a_listener_error_that_quotes_the_state_is_stored_redacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "s3cret-state-value-0123456789"

    async def fake_login(server: str, config: Any, *, handoff: Any, **_: Any) -> None:
        await handoff(
            f"https://as.example/authorize?state={secret}"
            "&redirect_uri=http%3A%2F%2Flocalhost%3A33418%2Fcallback"
        )
        raise RuntimeError(f"State parameter mismatch: forged != {secret}")

    pending = _listen_with(fake_login, monkeypatch)
    stored = pending_login.read_pending(pending.id)
    assert stored is not None and stored.status == "failed"
    assert secret not in (stored.error or "")
    assert stored.error == "RuntimeError: State parameter mismatch: forged != <redacted>"


def test_a_slow_discovery_never_outlives_the_record(monkeypatch: pytest.MonkeyPatch) -> None:
    """The callback server's timeout starts after discovery; the listener's own
    deadline must hold from the start: an expired record never has a live listener."""
    import asyncio

    async def stuck_in_discovery(server: str, config: Any, **_: Any) -> None:
        await asyncio.sleep(15)  # bounded, so a regression fails instead of hanging

    now = time.time()
    began = time.monotonic()
    pending = _listen_with(stuck_in_discovery, monkeypatch, created_at=now, expires_at=now + 3.0)
    assert time.monotonic() - began < 3.0  # returned before the record expired
    stored = pending_login.read_pending(pending.id)
    assert stored is not None and stored.status == "failed"
    assert stored.error == "it expired before the browser returned"


def test_the_blocking_login_says_when_it_ignored_a_wrong_state_callback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from rote.mcp import McpRegistry, McpServerConfig, save_registry

    save_registry(
        McpRegistry(servers={"vendor": McpServerConfig(url="https://vendor.example/mcp")})
    )

    async def fake_login(server: str, config: Any, *, on_rejected_callback: Any, **_: Any) -> None:
        on_rejected_callback()
        return None

    monkeypatch.setattr("rote.mcp.auth.login", fake_login)
    assert cli_main(["mcp", "login", "vendor"]) == 0
    assert "ignored a callback with the wrong state" in capsys.readouterr().err
