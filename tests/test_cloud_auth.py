"""Tests for `rote login` / `logout` / `whoami` and the credential store.

All HTTP is faked at the `_request_json` seam — the real device flow is
exercised live against the dev platform in the launch validation, not
here (suite stays fast and offline).
"""

from __future__ import annotations

import json
import stat
from typing import Any

import pytest

import rote.cloud_auth as ca
from rote.cli import main as cli_main
from rote.cloud_auth import (
    CloudCredential,
    LoginError,
    clear_credential,
    credential_path,
    load_credential,
    poll_for_token,
    save_credential,
    start_device_flow,
)

# ───────── credential store ─────────


def test_store_round_trip_and_mode() -> None:
    cred = CloudCredential(url="http://x:1/", token="rote_k", user="t@x", created_at="2026")
    path = save_credential(cred)
    assert path == credential_path()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    loaded = load_credential()
    assert loaded is not None
    assert loaded.url == "http://x:1"  # trailing slash normalized on load
    assert loaded.token == "rote_k"
    assert clear_credential() is True
    assert load_credential() is None
    assert clear_credential() is False


def test_corrupt_or_partial_store_reads_as_logged_out() -> None:
    path = credential_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not json", encoding="utf-8")
    assert load_credential() is None
    path.write_text(json.dumps({"url": "http://x"}), encoding="utf-8")  # no token
    assert load_credential() is None


# ───────── device flow ─────────


def _fake_requests(monkeypatch: pytest.MonkeyPatch, script: list[tuple[int, dict[str, Any]]]):
    """Feed `_request_json` canned (status, body) responses; record calls."""
    calls: list[tuple[str, str, dict[str, Any] | None, dict[str, str]]] = []

    def fake(
        method: str,
        url: str,
        body: dict[str, Any] | None = None,
        *,
        headers: dict[str, str] | None = None,
        timeout: float = 30.0,
    ) -> tuple[int, dict[str, Any]]:
        calls.append((method, url, body, headers or {}))
        return script.pop(0)

    monkeypatch.setattr(ca, "_request_json", fake)
    return calls


def test_start_device_flow_rejects_non_grant(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_requests(monkeypatch, [(400, {"error": "invalid_client"})])
    with pytest.raises(LoginError, match="invalid_client"):
        start_device_flow("http://p")


def test_polling_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """pending → keep going; slow_down → +5s floor; then the token."""
    _fake_requests(
        monkeypatch,
        [
            (400, {"error": "authorization_pending"}),
            (400, {"error": "slow_down"}),
            (400, {"error": "authorization_pending"}),
            (200, {"access_token": "sess_tok"}),
        ],
    )
    sleeps: list[float] = []
    token = poll_for_token("http://p", "dev_code", interval=5, expires_in=600, sleep=sleeps.append)
    assert token == "sess_tok"
    assert sleeps == [5, 5, 10, 10]


@pytest.mark.parametrize(
    ("error", "match"),
    [("access_denied", "denied"), ("expired_token", "expired"), ("invalid_grant", "expired")],
)
def test_polling_terminal_errors(monkeypatch: pytest.MonkeyPatch, error: str, match: str) -> None:
    _fake_requests(monkeypatch, [(400, {"error": error})])
    with pytest.raises(LoginError, match=match):
        poll_for_token("http://p", "d", interval=0, expires_in=600, sleep=lambda _s: None)


def test_login_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_requests(
        monkeypatch,
        [
            (
                200,
                {
                    "device_code": "dc",
                    "user_code": "ABCD-EFGH",
                    "verification_uri": "http://p/device",
                    "verification_uri_complete": "http://p/device?user_code=ABCD-EFGH",
                    "expires_in": 1800,
                    "interval": 5,
                },
            ),
            (200, {"access_token": "sess"}),
            (200, {"key": "rote_new", "user": "t@x", "tenant": "t1"}),
        ],
    )
    opened: list[str] = []
    monkeypatch.setattr(ca.webbrowser, "open", lambda u: opened.append(u) or True)
    monkeypatch.setattr(ca.time, "sleep", lambda _s: None)
    lines: list[str] = []

    cred = ca.login("http://p/", echo=lines.append)

    assert opened == ["http://p/device?user_code=ABCD-EFGH"]
    assert cred.token == "rote_new" and cred.user == "t@x" and cred.url == "http://p"
    stored = load_credential()
    assert stored is not None and stored.token == "rote_new"
    # the key mint carried the session token, not the api key
    mint = calls[-1]
    assert mint[1] == "http://p/v1/cli/keys"
    assert mint[3]["Authorization"] == "Bearer sess"
    assert any("ABCD-EFGH" in line for line in lines)


def test_login_relative_verification_uri_is_absolutized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_requests(
        monkeypatch,
        [
            (
                200,
                {
                    "device_code": "dc",
                    "user_code": "X",
                    "verification_uri_complete": "/device?user_code=X",
                    "expires_in": 60,
                    "interval": 0,
                },
            ),
            (200, {"access_token": "s"}),
            (200, {"key": "rote_k"}),
        ],
    )
    opened: list[str] = []
    monkeypatch.setattr(ca.webbrowser, "open", lambda u: opened.append(u) or True)
    monkeypatch.setattr(ca.time, "sleep", lambda _s: None)
    ca.login("http://p", echo=lambda _l: None)
    assert opened == ["http://p/device?user_code=X"]


# ───────── CLI wiring ─────────


def test_cli_login_dispatch(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: dict[str, Any] = {}

    def fake_login(url: str | None, *, open_browser: bool) -> CloudCredential:
        seen["url"] = url
        seen["open_browser"] = open_browser
        return CloudCredential(url="http://p", token="rote_k")

    monkeypatch.setattr("rote.cloud_auth.login", fake_login)
    assert cli_main(["login", "--url", "http://p", "--device"]) == 0
    assert seen == {"url": "http://p", "open_browser": False}


def test_cli_login_failure_exits_1(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fake_login(url: str | None, *, open_browser: bool) -> CloudCredential:
        raise LoginError("login denied in the browser")

    monkeypatch.setattr("rote.cloud_auth.login", fake_login)
    assert cli_main(["login"]) == 1
    assert "denied" in capsys.readouterr().err


def test_cli_whoami_not_logged_in(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli_main(["whoami"]) == 1
    assert "rote login" in capsys.readouterr().err


def test_cli_whoami_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    save_credential(CloudCredential(url="http://p", token="rote_k", user="stored@x"))
    monkeypatch.setattr("rote.cloud_auth.fetch_me", lambda cred: {"user": "live@x", "tenant": "t1"})
    assert cli_main(["whoami", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"url": "http://p", "user": "live@x", "tenant": "t1"}


def test_cli_logout_revokes_and_clears(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    save_credential(CloudCredential(url="http://p", token="rote_k"))
    revoked: list[str] = []
    monkeypatch.setattr(
        "rote.cloud_auth.revoke_key", lambda cred: revoked.append(cred.token) or True
    )
    assert cli_main(["logout"]) == 0
    assert revoked == ["rote_k"]
    assert load_credential() is None


def test_cli_logout_clears_even_when_revoke_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    save_credential(CloudCredential(url="http://p", token="rote_k"))

    def failing_revoke(cred: CloudCredential) -> bool:
        raise LoginError("could not reach http://p")

    monkeypatch.setattr("rote.cloud_auth.revoke_key", failing_revoke)
    assert cli_main(["logout"]) == 0
    assert load_credential() is None
    assert "still active server-side" in capsys.readouterr().err


def test_cli_logout_when_not_logged_in(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli_main(["logout"]) == 0
    assert "not logged in" in capsys.readouterr().err
