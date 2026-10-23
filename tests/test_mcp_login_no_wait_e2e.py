"""`rote mcp login --no-wait` / `--complete` against a real localhost OAuth server.

Each test runs the CLI the way an agent's non-interactive shell does: one
command starts the login and must return while the login is still open, a
simulated browser then follows the authorization URL (or a forged one hits
the callback port), and a second command finishes it. The listener is the
real detached child process (`python -m rote.mcp.pending_login`), bound on
rote's published callback ports.

The authorization server logs the PKCE verifier it received
(`tests/_oauth_as_server.py`), which is what lets these tests prove the
verifier never reached rote's output or its pending record.

Slow: a real AS process, real listener processes and real callbacks.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import signal
import stat
import time
from collections.abc import Iterator
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx
import pytest

pytestmark = pytest.mark.slow

pytest.importorskip("fastmcp")
pytest.importorskip("trustme")

from rote.cli import main as cli_main  # noqa: E402
from rote.mcp import (  # noqa: E402
    McpRegistry,
    McpServerConfig,
    pending_login,  # noqa: E402
    read_token_file,
    save_registry,
)
from rote.mcp.auth import (  # noqa: E402
    CALLBACK_PORTS,
    CLIENT_METADATA_URL_ENV_VAR,
    bind_callback_sockets,
)
from tests._oauth_fixture import OAuthFixture  # noqa: E402


@pytest.fixture(scope="module")
def fixture_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[OAuthFixture]:
    from tests._oauth_fixture import oauth_fixture

    with oauth_fixture(tmp_path_factory.mktemp("oauth-as")) as fx:
        yield fx


@pytest.fixture()
def fx(fixture_server: OAuthFixture, monkeypatch: pytest.MonkeyPatch) -> Iterator[OAuthFixture]:
    fixture_server.configure(
        reset=True,
        cimd=True,
        dcr=True,
        iss="correct",
        advertise_iss=True,
        active="a",
        preregistered={},
        cimd_document="served",
        prm_path=True,
        prm_origin_resource=None,
        prm_issuer=None,
    )
    for port in CALLBACK_PORTS[:2]:
        assert _port_free(port), f"callback port {port} was in use when the test started"
    monkeypatch.setenv(CLIENT_METADATA_URL_ENV_VAR, fixture_server.cimd_url)
    monkeypatch.setenv("SSL_CERT_FILE", str(fixture_server.ca_pem))
    # An agent's shell: no terminal, an agent caller in the environment.
    monkeypatch.setenv("CLAUDE_CODE_CHILD_SESSION", "1")
    save_registry(
        McpRegistry(
            servers={
                "vendor": McpServerConfig(url=fixture_server.mcp_url),
                "other": McpServerConfig(url=fixture_server.mcp_url),
            }
        )
    )
    yield fixture_server
    # Stop any listener a test left, even one a regression stopped from
    # watching its record: a stray listener would hold a port for the next.
    for pending in pending_login.list_pending():
        pending_login.delete_pending(pending.id)
        if pending.pid is not None and pending_login._pid_alive(pending.pid):
            os.kill(pending.pid, signal.SIGTERM)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not all(_port_free(p) for p in CALLBACK_PORTS[:2]):
        time.sleep(0.1)


def _port_free(port: int) -> bool:
    sockets = bind_callback_sockets(port)
    if sockets is None:
        return False
    for sock in sockets:
        sock.close()
    return True


def _rote(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict[str, Any], str]:
    code = cli_main(list(argv))
    captured = capsys.readouterr()
    payload = json.loads(captured.out) if captured.out.strip() else {}
    return code, payload, captured.err


def _start(capsys: pytest.CaptureFixture[str], server: str = "vendor") -> dict[str, Any]:
    began = time.monotonic()
    code, payload, err = _rote(
        capsys, "mcp", "login", server, "--no-wait", "--no-browser", "--yes", "--json"
    )
    assert code == 0, err
    # It returned while the login is open: the listener does the waiting.
    assert time.monotonic() - began < 30
    assert payload["status"] == "action_required"
    assert payload["reason"] == "authorization_pending"
    return payload


def _redirect_from(authorization_url: str) -> str:
    """What the AS sends the browser back to, without following it."""
    response = httpx.get(authorization_url, follow_redirects=False, timeout=30)
    assert response.status_code in (302, 303), response.text
    return response.headers["location"]


def _browser(authorization_url: str, *, state: str | None = None) -> int:
    """A browser approving the login; ``state`` forges the callback's state."""
    location = _redirect_from(authorization_url)
    if state is not None:
        parts = urlsplit(location)
        query = {k: v[0] for k, v in parse_qs(parts.query).items()}
        query["state"] = state
        location = f"{parts.scheme}://{parts.netloc}{parts.path}?{urlencode(query)}"
    return httpx.get(location, timeout=30).status_code


def _state(authorization_url: str) -> str:
    return parse_qs(urlsplit(authorization_url).query)["state"][0]


def test_start_then_browser_then_complete(
    fx: OAuthFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    started = _start(capsys)
    pending_id = started["pending_id"]
    url = started["authorization_url"]
    assert urlsplit(url).path == "/as-a/authorize"
    assert started["next"][0]["command"] == f"rote mcp login --complete {pending_id}"
    assert started["next"][1]["command"].endswith("--callback-url -")
    assert started["next"][1]["stdin"] == "<url>"
    assert [o["status"] for o in started["outcomes"]] == ["done"]

    record_path = pending_login.pending_path(pending_id)
    assert stat.S_IMODE(record_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(record_path.parent.stat().st_mode) == 0o700
    record = json.loads(record_path.read_text())
    assert record["status"] == "waiting"
    assert record["port"] == CALLBACK_PORTS[0]
    # The listener outlives the command that started it.
    assert not _port_free(CALLBACK_PORTS[0])
    assert read_token_file("vendor") is None or not (read_token_file("vendor") or {}).get("tokens")

    assert _browser(url) == 200
    code, done, err = _rote(capsys, "mcp", "login", "--complete", pending_id, "--json")
    assert code == 0, err
    assert done["ok"] is True
    assert done["registration"] == "cimd"
    assert done["issuer"] == fx.issuer("a")
    stored = read_token_file("vendor")
    assert stored is not None and stored["tokens"]["access_token"]
    assert not record_path.exists()

    # The state travels only inside the authorization URL; the verifier
    # never leaves the listener. Neither is in the record or any output.
    verifiers = [e["code_verifier"] for e in fx.events("token") if "code_verifier" in e]
    assert len(verifiers) == 1
    state = _state(url)
    elsewhere = json.dumps({k: v for k, v in started.items() if k != "authorization_url"})
    for secret in (state, verifiers[0]):
        assert secret not in elsewhere
        assert secret not in json.dumps(record)
        assert secret not in json.dumps(done) + err
    assert verifiers[0] not in url

    # A completed id is gone: the second completion finds nothing.
    code, _, err = _rote(capsys, "mcp", "login", "--complete", pending_id)
    assert code == 1
    assert f"no pending login {pending_id}" in err


def test_a_wrong_state_callback_is_ignored_and_the_real_one_completes(
    fx: OAuthFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    started = _start(capsys)
    pending_id = started["pending_id"]
    url = started["authorization_url"]

    _browser(url, state="forged-state-value")
    code, waiting, err = _rote(
        capsys, "mcp", "login", "--complete", pending_id, "--wait", "1", "--json"
    )
    assert code == 2
    assert waiting["ok"] is False and waiting["reason"] == "authorization_pending"
    assert "1 callback(s)" in err and "wrong state" in err
    assert fx.events("token") == []  # the forged code was never redeemed
    assert _state(url) not in err

    assert _browser(url) == 200
    code, done, err = _rote(capsys, "mcp", "login", "--complete", pending_id, "--json")
    assert code == 0, err
    assert done["ok"] is True
    assert len(fx.events("token")) == 1


def test_an_expired_login_is_refused_and_its_listener_is_gone(
    fx: OAuthFixture, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pending_login, "PENDING_TTL_SECONDS", 4)
    started = _start(capsys)
    pending_id = started["pending_id"]
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and not _port_free(CALLBACK_PORTS[0]):
        time.sleep(0.2)
    assert _port_free(CALLBACK_PORTS[0]), "the listener kept the port past its deadline"
    time.sleep(max(0.0, started["expires_at"] - time.time()) + 0.2)

    code, _, err = _rote(capsys, "mcp", "login", "--complete", pending_id)
    assert code == 1
    assert "expired before the browser returned" in err
    assert "rote mcp login vendor --no-wait" in err
    assert not pending_login.pending_path(pending_id).exists()
    # A browser arriving now reaches nothing, and nothing is stored.
    with pytest.raises(httpx.ConnectError):
        _browser(started["authorization_url"])
    assert (read_token_file("vendor") or {}).get("tokens") is None


def test_a_second_login_for_the_same_server_is_refused_another_server_gets_the_next_port(
    fx: OAuthFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    first = _start(capsys, "vendor")
    code, _, err = _rote(
        capsys, "mcp", "login", "vendor", "--no-wait", "--no-browser", "--yes", "--json"
    )
    assert code == 1
    assert f"already pending (id {first['pending_id']}" in err
    assert len(pending_login.list_pending()) == 1

    second = _start(capsys, "other")
    first_record = pending_login.read_pending(first["pending_id"])
    second_record = pending_login.read_pending(second["pending_id"])
    assert first_record is not None and second_record is not None
    assert (first_record.port, second_record.port) == CALLBACK_PORTS[:2]

    # Approved in the opposite order; each lands in its own server's tokens.
    assert _browser(second["authorization_url"]) == 200
    assert _browser(first["authorization_url"]) == 200
    for pending_id, server in ((first["pending_id"], "vendor"), (second["pending_id"], "other")):
        code, done, err = _rote(capsys, "mcp", "login", "--complete", pending_id, "--json")
        assert code == 0, err
        assert done["server"] == server
        assert (read_token_file(server) or {})["tokens"]["access_token"]


def test_a_pasted_callback_url_completes_a_login_whose_browser_could_not_return(
    fx: OAuthFixture, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    started = _start(capsys)
    pending_id = started["pending_id"]
    pasted = _redirect_from(started["authorization_url"])  # the page that "could not load"

    elsewhere = pasted.replace(f":{CALLBACK_PORTS[0]}/", f":{CALLBACK_PORTS[2]}/")
    code, _, err = _rote(
        capsys, "mcp", "login", "--complete", pending_id, "--callback-url", elsewhere
    )
    assert code == 1
    assert "not this login's callback" in err
    assert fx.events("token") == []

    # Pasted on stdin, so the code and state never appear in the process list.
    monkeypatch.setattr("sys.stdin", io.StringIO(pasted + "\n"))
    code, done, err = _rote(
        capsys, "mcp", "login", "--complete", pending_id, "--callback-url", "-", "--json"
    )
    assert code == 0, err
    assert done["ok"] is True
    assert (read_token_file("vendor") or {})["tokens"]["access_token"]


def test_cancel_stops_the_listener(fx: OAuthFixture, capsys: pytest.CaptureFixture[str]) -> None:
    started = _start(capsys)
    code, payload, err = _rote(capsys, "mcp", "login", "--cancel", started["pending_id"], "--json")
    assert code == 0, err
    assert payload["cancelled"] == started["pending_id"]
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not _port_free(CALLBACK_PORTS[0]):
        time.sleep(0.1)
    assert _port_free(CALLBACK_PORTS[0])
    assert fx.events("token") == []


def test_without_yes_nothing_is_contacted_started_or_written(
    fx: OAuthFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    code, payload, _ = _rote(capsys, "mcp", "login", "vendor", "--no-wait", "--json")
    assert code == 2
    assert payload["reason"] == "confirmation_required"
    assert payload["next"] == [{"command": "rote mcp login vendor --no-wait --json --yes"}]
    kinds = [e["kind"] for a in payload["plan"]["actions"] for e in a["effects"]]
    assert kinds == ["contact", "launch", "write", "launch"]
    assert fx.events() == []
    assert pending_login.list_pending() == []
    assert _port_free(CALLBACK_PORTS[0])


def test_the_listener_outlives_the_shell_tool_that_ran_the_command(fx: OAuthFixture) -> None:
    """An agent's shell tool reads the command's output through pipes and,
    when the command ends, may signal its whole process group (a timeout
    wrapper, a sandboxed exec). The command must return while the login is
    pending, and the listener must survive that cleanup: it holds no copy
    of the caller's pipes and runs in its own session."""
    import subprocess
    import sys

    began = time.monotonic()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "rote",
            "mcp",
            "login",
            "vendor",
            "--no-wait",
            "--no-browser",
            "--yes",
            "--json",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,  # the tool's process group
        env=os.environ.copy(),
    )
    # communicate() returns only once every holder of the pipes closed them.
    out, err = proc.communicate(timeout=60)
    assert proc.returncode == 0, err
    assert time.monotonic() - began < 30
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGTERM)  # the tool cleans up its group
    time.sleep(0.5)
    record = pending_login.read_pending(json.loads(out)["pending_id"])
    assert record is not None and record.status == "waiting"
    assert pending_login._pid_alive(record.pid)
    assert not _port_free(CALLBACK_PORTS[0])


def test_a_listener_stuck_in_discovery_is_gone_when_its_record_expires(
    fx: OAuthFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A server that never answers discovery: the callback server never comes
    up, so only the listener's own deadline can stop it. It must be dead by
    the record's expiry, and the start reports the expiry, not a hang."""
    import socket
    import threading

    from rote.mcp import McpAuthError

    hold: list[socket.socket] = []
    stuck = socket.socket()
    stuck.bind(("127.0.0.1", 0))
    stuck.listen()

    def accept_and_never_answer() -> None:
        while True:
            try:
                conn, _ = stuck.accept()
            except OSError:
                return
            hold.append(conn)

    threading.Thread(target=accept_and_never_answer, daemon=True).start()
    url = f"http://127.0.0.1:{stuck.getsockname()[1]}/mcp"
    save_registry(McpRegistry(servers={"stuck": McpServerConfig(url=url)}))
    monkeypatch.setattr(pending_login, "PENDING_TTL_SECONDS", 4)
    pids: list[int] = []

    def watch() -> None:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and not pids:
            for pending in pending_login.list_pending():
                if pending.pid is not None:
                    pids.append(pending.pid)
            time.sleep(0.05)

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    began = time.time()
    try:
        with pytest.raises(McpAuthError, match="expired before the browser returned"):
            pending_login.start_pending_login("stuck", McpServerConfig(url=url), timeout=30)
        watcher.join(timeout=5)
        assert pids, "the listener never recorded its pid"
        expires_at = began + 4
        while time.time() < expires_at and pending_login._pid_alive(pids[0]):
            time.sleep(0.05)
        time.sleep(0.3)  # let the exited process be reaped
        assert not _alive_or_zombie(pids[0]), "the listener outlived its record"
    finally:
        stuck.close()
        for conn in hold:
            conn.close()


def _alive_or_zombie(pid: int) -> bool:
    """A child that exited but was not reaped still answers kill(pid, 0)."""
    import subprocess

    state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    return state.returncode == 0 and not state.stdout.strip().startswith("Z")
