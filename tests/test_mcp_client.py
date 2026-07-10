"""Unit tests for the MCP client layer (registry, token store, CLI).

Everything here is offline and hermetic — the autouse conftest fixture
points ROTE_MCP_CONFIG / ROTE_MCP_TOKEN_DIR at per-test tmp dirs. The
real OAuth dance is covered by tests/test_mcp_oauth_e2e.py (slow).
"""

from __future__ import annotations

import asyncio
import json
import stat
from pathlib import Path
from typing import Any

import pytest

from rote.cli import main as cli_main
from rote.ir import Pipeline
from rote.mcp import (
    McpRegistry,
    McpServerConfig,
    access_token_state,
    clear_token_file,
    load_registry,
    read_token_file,
    resolve_server_url,
    save_registry,
    write_token_file,
)
from rote.mcp.tokens import ServerTokenKV

# ───────── Registry ─────────


def test_registry_round_trip_and_permissions() -> None:
    registry = McpRegistry(
        servers={
            "slack": McpServerConfig(url="https://mcp.example.com/slack", scopes=["read"]),
            "gh": McpServerConfig(
                url="https://mcp.example.com/gh", client_id="abc", client_secret="shh"
            ),
        }
    )
    path = save_registry(registry)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600  # may hold client secrets
    loaded = load_registry()
    assert loaded == registry


def test_resolve_server_url_order(monkeypatch: pytest.MonkeyPatch) -> None:
    save_registry(
        McpRegistry(servers={"slack": McpServerConfig(url="https://registry.example/slack")})
    )
    monkeypatch.setenv("ROTE_MCP_SLACK_URL", "https://env.example/slack")
    # Explicit binding URL wins over everything.
    assert resolve_server_url("slack", "https://ir.example/slack") == "https://ir.example/slack"
    # Registry beats env.
    assert resolve_server_url("slack") == "https://registry.example/slack"
    # Env is the fallback; unknown everywhere → None.
    assert resolve_server_url("gmail") is None
    monkeypatch.setenv("ROTE_MCP_GMAIL_URL", "https://env.example/gmail")
    assert resolve_server_url("gmail") == "https://env.example/gmail"


# ───────── Token store ─────────


def _doc(*, expires_at: float | None, refresh: bool = True) -> dict[str, Any]:
    tokens: dict[str, Any] = {"access_token": "tok-123", "token_type": "Bearer"}
    if refresh:
        tokens["refresh_token"] = "ref-456"
    return {
        "server_url": "https://mcp.example.com/slack",
        "tokens": tokens,
        "expires_at": expires_at,
        "client_info": {"client_id": "abc"},
        "token_endpoint": "https://auth.example.com/token",
    }


def test_token_file_round_trip_and_permissions() -> None:
    path = write_token_file("slack", _doc(expires_at=None))
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    doc = read_token_file("slack")
    assert doc is not None and doc["version"] == 1
    assert doc["tokens"]["access_token"] == "tok-123"
    assert clear_token_file("slack") is True
    assert read_token_file("slack") is None
    assert clear_token_file("slack") is False


def test_access_token_state() -> None:
    import time

    fresh = _doc(expires_at=time.time() + 3600)
    stale = _doc(expires_at=time.time() - 10)
    no_expiry = _doc(expires_at=None)
    assert access_token_state(fresh) == ("tok-123", True)
    assert access_token_state(stale) == ("tok-123", False)
    assert access_token_state(no_expiry) == ("tok-123", True)  # assume long-lived
    assert access_token_state({"tokens": {}}) == (None, False)


def test_server_token_kv_speaks_the_adapter_collections() -> None:
    """The AsyncKeyValue shim maps fastmcp's three OAuth collections onto
    the contract file's fields — keys (URL-derived upstream) are ignored."""
    kv = ServerTokenKV("slack", "https://mcp.example.com/slack")

    async def flow() -> None:
        assert await kv.get("ignored", collection="mcp-oauth-token") is None
        await kv.put(
            "ignored",
            {"access_token": "a1", "token_type": "Bearer", "refresh_token": "r1"},
            collection="mcp-oauth-token",
        )
        await kv.put("ignored", {"client_id": "c1"}, collection="mcp-oauth-client-info")
        await kv.put(
            "ignored", {"expires_at": 1_800_000_000.0}, collection="mcp-oauth-token-expiry"
        )
        got = await kv.get("whatever", collection="mcp-oauth-token")
        assert got is not None and got["access_token"] == "a1"
        expiry = await kv.get("x", collection="mcp-oauth-token-expiry")
        assert expiry == {"expires_at": 1_800_000_000.0}
        assert await kv.delete("x", collection="mcp-oauth-token") is True

    asyncio.run(flow())

    doc = read_token_file("slack")
    assert doc is not None
    assert doc["client_info"] == {"client_id": "c1"}
    assert doc["expires_at"] == 1_800_000_000.0
    assert doc["tokens"] is None  # deleted above
    assert doc["version"] == 1
    with pytest.raises(ValueError, match="unexpected OAuth storage collection"):
        asyncio.run(kv.get("x", collection="bogus"))


def test_runtime_helper_kv_is_byte_compatible_with_the_cli_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The emitted helper's _TokenKV and rote's ServerTokenKV implement one
    contract: the same operation sequence must produce identical files —
    an emitted app and the CLI share the store at runtime."""
    from rote.mcp import _runtime_helper

    ops = [
        ("mcp-oauth-token", {"access_token": "a", "refresh_token": "r"}),
        ("mcp-oauth-client-info", {"client_id": "c"}),
        ("mcp-oauth-token-expiry", {"expires_at": 1_800_000_000.0}),
    ]

    async def run(kv: Any) -> None:
        for collection, value in ops:
            await kv.put("k", value, collection=collection)

    dir_a = tmp_path / "a"
    monkeypatch.setenv("ROTE_MCP_TOKEN_DIR", str(dir_a))
    asyncio.run(run(_runtime_helper._TokenKV("s", "https://u.example/mcp")))

    dir_b = tmp_path / "b"
    asyncio.run(run(ServerTokenKV("s", "https://u.example/mcp", directory=dir_b)))

    a = json.loads((dir_a / "s.json").read_text(encoding="utf-8"))
    b = json.loads((dir_b / "s.json").read_text(encoding="utf-8"))
    assert a == b


# ───────── Emitted helper resolution ─────────


def test_helper_resolve_url_order(monkeypatch: pytest.MonkeyPatch) -> None:
    from rote.mcp import _runtime_helper as helper

    save_registry(
        McpRegistry(servers={"slack": McpServerConfig(url="https://registry.example/slack")})
    )
    # Env beats registry beats the pipeline-recorded URL at runtime —
    # deployment context outranks graduation-time capture.
    monkeypatch.setenv("ROTE_MCP_SLACK_URL", "https://env.example/slack")
    assert helper.resolve_url("slack", "https://ir.example/slack") == "https://env.example/slack"
    monkeypatch.delenv("ROTE_MCP_SLACK_URL")
    assert helper.resolve_url("slack", "https://ir.example/slack") == (
        "https://registry.example/slack"
    )
    assert helper.resolve_url("gmail", "https://ir.example/gmail") == "https://ir.example/gmail"
    with pytest.raises(RuntimeError, match="rote mcp add nowhere"):
        helper.resolve_url("nowhere", None)


def test_helper_token_usability(monkeypatch: pytest.MonkeyPatch) -> None:
    import time

    from rote.mcp import _runtime_helper as helper

    assert helper.token_is_usable("slack") is False
    write_token_file("slack", _doc(expires_at=time.time() + 3600))
    assert helper.token_is_usable("slack") is True
    write_token_file("slack", _doc(expires_at=time.time() - 10, refresh=True))
    assert helper.token_is_usable("slack") is True  # stale but refreshable
    write_token_file("slack", _doc(expires_at=time.time() - 10, refresh=False))
    assert helper.token_is_usable("slack") is False


def test_dbos_emits_the_helper_verbatim(tmp_path: Path) -> None:
    """The emitted extracted/_rote_mcp.py IS rote.mcp._runtime_helper's
    source — one tested implementation, no drift possible."""
    from rote.adapters import get_adapter
    from rote.mcp import _runtime_helper

    pipeline = Pipeline.model_validate(
        {
            "name": "helper_demo",
            "input": {"type": "In"},
            "nodes": [
                {
                    "id": "pull",
                    "kind": "external_call",
                    "description": "pull data",
                    "mcp": {"server": "vendor", "tool": "get_data"},
                }
            ],
            "edges": [],
        }
    )
    get_adapter("dbos").emit(pipeline, tmp_path)
    emitted = (tmp_path / "extracted" / "_rote_mcp.py").read_text(encoding="utf-8")
    source = Path(_runtime_helper.__file__).read_text(encoding="utf-8")
    assert emitted == source


# ───────── CLI ─────────


def test_cli_mcp_add_list_headers_remove(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli_main(["mcp", "add", "keyd", "https://api.example/mcp", "--header", "X-K: v1"]) == 0
    assert cli_main(["mcp", "add", "slack", "https://mcp.example/slack"]) == 0
    capsys.readouterr()

    assert cli_main(["mcp", "list", "--json"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert {s["name"]: s["auth"] for s in listed["servers"]} == {
        "keyd": "static headers",
        "slack": "not authenticated",
    }

    assert cli_main(["mcp", "headers", "keyd"]) == 0
    assert json.loads(capsys.readouterr().out) == {"X-K": "v1"}

    assert cli_main(["mcp", "remove", "keyd"]) == 0
    assert cli_main(["mcp", "remove", "keyd"]) == 2  # already gone


def test_cli_mcp_add_rejects_bad_input(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli_main(["mcp", "add", "bad-name!", "https://x.example"]) == 2
    assert "valid identifier" in capsys.readouterr().err
    assert cli_main(["mcp", "add", "ok", "https://x.example", "--header", "noseparator"]) == 2


def test_cli_mcp_headers_requires_login(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli_main(["mcp", "add", "slack", "https://mcp.example/slack"]) == 0
    capsys.readouterr()
    assert cli_main(["mcp", "headers", "slack"]) == 1
    assert "rote mcp login slack" in capsys.readouterr().err


def test_cli_mcp_login_unknown_server(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli_main(["mcp", "login", "ghost"]) == 2
    assert "rote mcp add ghost" in capsys.readouterr().err


def test_cli_mcp_list_shows_expired_refreshable(capsys: pytest.CaptureFixture[str]) -> None:
    import time

    assert cli_main(["mcp", "add", "slack", "https://mcp.example/slack"]) == 0
    write_token_file("slack", _doc(expires_at=time.time() - 10))
    capsys.readouterr()
    assert cli_main(["mcp", "list", "--json"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed["servers"][0]["auth"] == "expired (refreshable)"


# ───────── eval --run wiring ─────────


def _bound_pipeline() -> Pipeline:
    return Pipeline.model_validate(
        {
            "name": "wired",
            "input": {"type": "In"},
            "nodes": [
                {
                    "id": "pull",
                    "kind": "external_call",
                    "description": "pull",
                    "mcp": {"server": "slack", "tool": "get_messages"},
                }
            ],
            "edges": [],
        }
    )


def test_eval_wiring_resolves_url_from_registry() -> None:
    from rote.eval.empirical import mcp_servers_for_pipeline

    save_registry(
        McpRegistry(servers={"slack": McpServerConfig(url="https://registry.example/slack")})
    )
    servers, missing = mcp_servers_for_pipeline(_bound_pipeline())
    assert missing == []
    assert servers["slack"]["url"] == "https://registry.example/slack"
    assert "headersHelper" not in servers["slack"]  # not logged in


def test_eval_wiring_adds_headers_helper_when_logged_in() -> None:
    import sys

    from rote.eval.empirical import mcp_servers_for_pipeline

    save_registry(
        McpRegistry(servers={"slack": McpServerConfig(url="https://registry.example/slack")})
    )
    write_token_file("slack", _doc(expires_at=None))
    servers, _ = mcp_servers_for_pipeline(_bound_pipeline())
    helper = servers["slack"]["headersHelper"]
    assert helper.endswith("-m rote mcp headers slack") or "-m rote mcp headers slack" in helper
    assert sys.executable in helper


def test_eval_wiring_static_headers_win_over_helper() -> None:
    from rote.eval.empirical import mcp_servers_for_pipeline

    save_registry(
        McpRegistry(
            servers={
                "slack": McpServerConfig(url="https://registry.example/slack", headers={"X-K": "v"})
            }
        )
    )
    write_token_file("slack", _doc(expires_at=None))
    servers, _ = mcp_servers_for_pipeline(_bound_pipeline())
    assert servers["slack"]["headers"] == {"X-K": "v"}
    assert "headersHelper" not in servers["slack"]
