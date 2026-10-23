"""The pending-login record and the checks around it, without a server.

The full flow (a real listener, a real authorization server, a simulated
browser) is tests/test_mcp_login_no_wait_e2e.py.
"""

from __future__ import annotations

import json
import stat
import time
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
    ],
)
def test_a_pasted_callback_is_only_ever_sent_to_this_logins_listener(
    url: str, message: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[str] = []
    monkeypatch.setattr("httpx.get", lambda target, **_: sent.append(target))
    pending = _record(status="waiting", port=33418)
    with pytest.raises(PendingLoginError, match=message):
        pending_login.replay_callback(pending, url)
    assert sent == []
    pending_login.replay_callback(pending, "http://localhost:33418/callback?code=x&state=y")
    assert sent == ["http://127.0.0.1:33418/callback?code=x&state=y"]


def test_the_state_check_drops_a_forged_success_and_keeps_waiting() -> None:
    pytest.importorskip("fastmcp")
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
        "rote mcp login --complete 0123456789abcdef --callback-url <url>",
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
