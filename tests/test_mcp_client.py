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
    token_file,
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
    assert doc is not None and doc["version"] == 2
    assert doc["tokens"]["access_token"] == "tok-123"
    assert clear_token_file("slack") is True
    assert read_token_file("slack") is None
    assert clear_token_file("slack") is False


def test_a_version_1_token_file_still_reads_and_is_rewritten_as_version_2() -> None:
    """Contract v2 only adds ``issuer``. A file a pre-v2 CLI wrote must keep
    working, and the next write must stamp the current version rather than
    carry the old one forward (``{"version": 1, **doc}`` would)."""
    v1 = token_file("slack")
    v1.parent.mkdir(parents=True, exist_ok=True)
    v1.write_text(json.dumps({"version": 1, **_doc(expires_at=None)}), encoding="utf-8")
    doc = read_token_file("slack")
    assert doc is not None and doc["version"] == 1
    assert access_token_state(doc) == ("tok-123", True)
    assert doc.get("issuer") is None

    write_token_file("slack", doc)
    rewritten = read_token_file("slack")
    assert rewritten is not None and rewritten["version"] == 2
    assert rewritten["tokens"] == doc["tokens"]


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
    assert doc["version"] == 2
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
    # deployment context outranks compilation-time capture.
    monkeypatch.setenv("ROTE_MCP_SLACK_URL", "https://env.example/slack")
    assert helper.resolve_url("slack", "https://ir.example/slack") == "https://env.example/slack"
    monkeypatch.delenv("ROTE_MCP_SLACK_URL")
    assert helper.resolve_url("slack", "https://ir.example/slack") == (
        "https://registry.example/slack"
    )
    assert helper.resolve_url("gmail", "https://ir.example/gmail") == "https://ir.example/gmail"
    with pytest.raises(RuntimeError, match="rote mcp add nowhere"):
        helper.resolve_url("nowhere", None)


def test_helper_missing_endpoint_is_a_config_error_naming_the_variable() -> None:
    """The emitted steps exclude RoteMcpConfigError from their retry
    budget, so a missing endpoint must raise exactly that, name the env
    var to set, and survive pickling (DBOS persists step errors)."""
    import pickle

    from rote.mcp import _runtime_helper as helper

    with pytest.raises(helper.RoteMcpConfigError) as info:
        helper.resolve_url("vendor", None)
    assert "set ROTE_MCP_VENDOR_URL" in str(info.value)
    assert not isinstance(info.value, helper.RoteMcpAuthNeeded)
    restored = pickle.loads(pickle.dumps(info.value))
    assert type(restored) is helper.RoteMcpConfigError
    assert str(restored) == str(info.value)


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
    # The emitted README names the fastmcp range the helper was verified
    # against, not a bare `pip install fastmcp` that could resolve to a
    # generation the helper cannot run on.
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert f"pip install '{_runtime_helper.FASTMCP_REQUIREMENT}'" in readme
    assert "pip install fastmcp\n" not in readme


def test_helper_fastmcp_requirement_matches_the_package_extras() -> None:
    """One verified range: the extras rote installs with and the range the
    emitted helper advertises are the same string."""
    import tomllib

    from rote.mcp import _runtime_helper

    pyproject = tomllib.loads(
        (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text(encoding="utf-8")
    )
    extras = pyproject["project"]["optional-dependencies"]
    pins = {
        dep for name in ("mcp", "serve", "dev") for dep in extras[name] if dep.startswith("fastmcp")
    }
    assert pins == {_runtime_helper.FASTMCP_REQUIREMENT}


# ───────── CLI ─────────


def test_cli_mcp_add_list_headers_remove(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli_main(["mcp", "add", "keyd", "https://api.example/mcp", "--header", "X-K: v1"]) == 0
    assert cli_main(["mcp", "add", "slack", "https://mcp.example/slack"]) == 0
    capsys.readouterr()

    assert cli_main(["mcp", "list", "--json"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert {s["name"]: s["auth"] for s in listed["servers"]} == {
        "keyd": "static headers",
        "slack": "auth unknown",
    }

    assert cli_main(["mcp", "headers", "keyd"]) == 0
    assert json.loads(capsys.readouterr().out) == {"X-K": "v1"}

    assert cli_main(["mcp", "remove", "keyd"]) == 0
    assert cli_main(["mcp", "remove", "keyd"]) == 2  # already gone


def test_registry_keeps_fields_and_values_it_does_not_know() -> None:
    """The registry is shared by every rote version on the machine. A newer
    version's field, or a transport value this one does not know, must
    neither fail this load nor vanish when this version writes it back."""
    from rote.mcp.registry import registry_path

    path = registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    original = {
        "version": 7,
        "future_top_level": {"x": 1},
        "servers": {
            "open": {
                "url": "https://o.example/mcp",
                "transport": "websocket-2030",
                "future_field": ["a", 2],
            }
        },
    }
    path.write_text(json.dumps(original), encoding="utf-8")

    registry = load_registry()
    assert registry.servers["open"].transport == "websocket-2030"
    save_registry(registry)
    assert json.loads(path.read_text(encoding="utf-8")) == original


def test_learning_auth_never_touches_the_registry() -> None:
    """Released rote versions (0.14.0 and older) reject unknown registry
    fields, so a login must leave the registry byte-for-byte unchanged."""
    from rote.mcp.learned import learned_auth, learned_path, record_auth
    from rote.mcp.registry import registry_path

    save_registry(McpRegistry(servers={"open": McpServerConfig(url="https://o.example/mcp")}))
    before = registry_path().read_bytes()
    record_auth("open", "https://o.example/mcp", "none", source="login")
    assert registry_path().read_bytes() == before
    assert stat.S_IMODE(learned_path().stat().st_mode) == 0o600
    learned = learned_auth("open", "https://O.example/mcp/")  # same endpoint, other spelling
    assert learned is not None and learned.auth == "none"


def test_learned_auth_is_per_endpoint_and_carries_its_date() -> None:
    import datetime

    from rote.mcp.learned import learned_auth, record_auth

    record_auth("open", "https://o.example/mcp", "none", source="login")
    learned = learned_auth("open", "https://o.example/mcp")
    assert learned is not None
    assert learned.learned_at == datetime.datetime.now(datetime.UTC).date().isoformat()
    assert learned.source == "login"
    # Repointing the name invalidates the finding: it was about another server.
    assert learned_auth("open", "https://elsewhere.example/mcp") is None


def test_the_sidecar_keeps_unknown_fields_values_and_a_higher_version() -> None:
    """A newer rote may add fields, auth values, or a version. This one keeps
    them on write and reads an unknown auth value as nothing learned."""
    from rote.mcp.learned import learned_auth, learned_path, record_auth

    learned_path().parent.mkdir(parents=True, exist_ok=True)
    learned_path().write_text(
        json.dumps(
            {
                "version": 9,
                "future": True,
                "servers": {
                    "a": {"url": "https://a.example/mcp", "auth": "mtls-2030", "extra": 1},
                    "b": {"url": "https://b.example/mcp", "auth": "none"},
                },
            }
        ),
        encoding="utf-8",
    )
    assert learned_auth("a", "https://a.example/mcp") is None
    record_auth("b", "https://b.example/mcp", "oauth", source="park")
    saved = json.loads(learned_path().read_text(encoding="utf-8"))
    assert saved["version"] == 9
    assert saved["future"] is True
    assert saved["servers"]["a"] == {
        "url": "https://a.example/mcp",
        "auth": "mtls-2030",
        "extra": 1,
    }
    assert saved["servers"]["b"]["auth"] == "oauth"


def test_a_park_retires_a_learned_none() -> None:
    from rote.mcp.learned import learned_auth, note_auth_park, record_auth

    save_registry(McpRegistry(servers={"open": McpServerConfig(url="https://o.example/mcp")}))
    record_auth("open", "https://o.example/mcp", "none", source="login")
    note_auth_park("open")
    learned = learned_auth("open", "https://o.example/mcp")
    assert learned is not None
    assert (learned.auth, learned.source) == ("oauth", "park")


def test_a_token_file_outranks_a_learned_none() -> None:
    """A server that moved from open to OAuth: once a token for this endpoint
    exists, its state is the token's, never "no auth required"."""
    from rote.mcp.tokens import auth_status

    url = "https://o.example/mcp"
    config = McpServerConfig(url=url)
    assert auth_status(config, None, "none") == "no auth required"
    expired = {"server_url": url, "tokens": {"access_token": "t"}, "expires_at": 1.0}
    assert auth_status(config, expired, "none") == "expired"


def test_a_token_for_another_endpoint_counts_as_no_token() -> None:
    """After a repoint, the old login is no login here: a learned `none` for
    the new endpoint wins, and without one the name still needs a login."""
    from rote.mcp.tokens import auth_status

    config = McpServerConfig(url="https://new.example/mcp")
    old = {"server_url": "https://old.example/mcp", "tokens": {"access_token": "t"}}
    assert auth_status(config, old, "none") == "no auth required"
    assert auth_status(config, old) == "not authenticated"


def test_live_tools_only_skip_headers_for_none_without_a_token() -> None:
    from rote.mcp.learned import record_auth
    from rote.mcp.live_tools import registry_server_specs

    save_registry(
        McpRegistry(
            servers={
                "open": McpServerConfig(url="https://o.example/mcp"),
                "moved": McpServerConfig(url="https://m.example/mcp"),
            }
        )
    )
    record_auth("open", "https://o.example/mcp", "none", source="login")
    record_auth("moved", "https://m.example/mcp", "none", source="login")
    write_token_file(
        "moved",
        {"server_url": "https://m.example/mcp", "tokens": {"access_token": "tok-1"}},
    )
    specs = {s["name"]: s for s in asyncio.run(registry_server_specs())}
    assert specs["open"] == {"name": "open", "url": "https://o.example/mcp"}
    assert specs["moved"]["headers"] == {"Authorization": "Bearer tok-1"}


def test_token_writes_keep_a_newer_contract_version() -> None:
    from rote.mcp.tokens import TOKEN_CONTRACT_VERSION, token_file

    write_token_file("v", {"version": TOKEN_CONTRACT_VERSION + 5, "tokens": None})
    assert json.loads(token_file("v").read_text())["version"] == TOKEN_CONTRACT_VERSION + 5
    write_token_file("w", {"tokens": None})
    assert json.loads(token_file("w").read_text())["version"] == TOKEN_CONTRACT_VERSION


@pytest.mark.parametrize(
    ("context", "expected"),
    [
        (
            {"protected_resource_metadata": None, "oauth_metadata": None, "auth_server_url": None},
            False,
        ),
        ({"protected_resource_metadata": object(), "oauth_metadata": None}, True),
        (
            {"protected_resource_metadata": None, "oauth_metadata": None, "auth_server_url": "x"},
            True,
        ),
        (None, True),  # provider shape unknown: never infer "no auth"
    ],
)
def test_no_auth_is_inferred_only_when_the_provider_saw_no_challenge(
    context: dict[str, Any] | None, expected: bool
) -> None:
    from types import SimpleNamespace

    from rote.mcp.auth import _saw_auth_challenge

    oauth = SimpleNamespace(context=SimpleNamespace(**context) if context is not None else None)
    assert _saw_auth_challenge(oauth) is expected


def test_auth_label_dates_a_learned_no_auth() -> None:
    from rote.cli import _auth_label

    assert (
        _auth_label({"auth": "no auth required", "auth_learned_at": "2026-10-18"})
        == "no auth required (learned 2026-10-18)"
    )
    assert _auth_label({"auth": "authenticated", "auth_learned_at": "2026-10-18"}) == (
        "authenticated"
    )


def test_cli_mcp_add_records_the_auth_requirement(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli_main(["mcp", "add", "open", "https://o.example/mcp", "--auth", "none"]) == 0
    assert cli_main(["mcp", "add", "locked", "https://l.example/mcp", "--auth", "oauth"]) == 0
    assert cli_main(["mcp", "add", "unsure", "https://u.example/mcp"]) == 0
    capsys.readouterr()
    assert cli_main(["mcp", "list", "--json"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert {s["name"]: s["auth_source"] for s in listed["servers"]} == {
        "open": "user",
        "locked": "user",
        "unsure": None,
    }
    assert {s["name"]: s["auth"] for s in listed["servers"]} == {
        "open": "no auth required",
        "locked": "not authenticated",
        "unsure": "auth unknown",
    }


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

    assert cli_main(["mcp", "add", "slack", "https://mcp.example.com/slack"]) == 0
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


# ───────── rote mcp export (Cloudflare provisioning) ─────────


def test_cli_mcp_export_turns_a_login_into_secrets(capsys: pytest.CaptureFixture[str]) -> None:
    save_registry(
        McpRegistry(servers={"vendor": McpServerConfig(url="https://mcp.example.com/vendor")})
    )
    write_token_file(
        "vendor",
        {
            "server_url": "https://mcp.example.com/vendor",
            "tokens": {"access_token": "at", "refresh_token": "rt-1"},
            "expires_at": None,
            "client_info": {"client_id": "cid-1", "client_secret": "shh"},
            "token_endpoint": "https://auth.example.com/token",
        },
    )
    assert cli_main(["mcp", "export", "vendor", "--json"]) == 0
    secrets = json.loads(capsys.readouterr().out)
    assert secrets == {
        "ROTE_MCP_VENDOR_REFRESH_TOKEN": "rt-1",
        "ROTE_MCP_VENDOR_CLIENT_ID": "cid-1",
        "ROTE_MCP_VENDOR_CLIENT_SECRET": "shh",
        "ROTE_MCP_VENDOR_TOKEN_ENDPOINT": "https://auth.example.com/token",
        "ROTE_MCP_VENDOR_URL": "https://mcp.example.com/vendor",
    }
    # dotenv form for .dev.vars
    assert cli_main(["mcp", "export", "vendor"]) == 0
    out = capsys.readouterr().out
    assert "ROTE_MCP_VENDOR_REFRESH_TOKEN=rt-1" in out


def test_cli_mcp_export_requires_a_refreshable_login(
    capsys: pytest.CaptureFixture[str],
) -> None:
    save_registry(
        McpRegistry(servers={"vendor": McpServerConfig(url="https://mcp.example.com/vendor")})
    )
    write_token_file(
        "vendor",
        {
            "server_url": "https://mcp.example.com/vendor",
            "tokens": {"access_token": "at"},  # no refresh token
            "expires_at": None,
            "client_info": {"client_id": "cid-1"},
            "token_endpoint": "https://auth.example.com/token",
        },
    )
    assert cli_main(["mcp", "export", "vendor"]) == 1
    assert "refresh token" in capsys.readouterr().err


# ───────── call_mcp_tool: auth preflight + 401 mapping ─────────


def test_call_mcp_tool_preflights_dead_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """An expired token with no refresh path raises RoteMcpAuthNeeded
    *before* any network I/O — the workflow parks on it."""
    import time

    from rote.mcp import _runtime_helper as helper

    write_token_file("slack", _doc(expires_at=time.time() - 10, refresh=False))

    def _boom(*a: Any, **kw: Any) -> Any:  # network must not be touched
        raise AssertionError("preflight should raise before opening a client")

    monkeypatch.setattr(helper, "mcp_client", _boom)
    with pytest.raises(helper.RoteMcpAuthNeeded, match="rote mcp login slack") as exc_info:
        asyncio.run(helper.call_mcp_tool("slack", "https://mcp.example.com/slack", "t", {}))
    assert exc_info.value.server == "slack"


class _StatusServer:
    """A localhost HTTP endpoint answering every request with one status.

    A real socket, not a mocked client: under MCP SDK v2 the status code
    of a non-2xx answer never reaches the caller's exception, so the only
    honest test of 401 detection is a real HTTP 401 through the real
    fastmcp transport.
    """

    def __init__(self, status: int) -> None:
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        class _Handler(BaseHTTPRequestHandler):
            def _answer(self) -> None:
                self.send_response(status)
                if status == 401:
                    self.send_header("WWW-Authenticate", 'Bearer realm="test"')
                self.send_header("Content-Length", "0")
                self.end_headers()

            do_GET = do_POST = do_DELETE = _answer

            def log_message(self, *args: Any) -> None:
                return None

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}/mcp"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.mark.parametrize("status", [401, 500])
def test_call_mcp_tool_maps_a_real_401_to_auth_needed(status: int) -> None:
    """An HTTP 401 becomes RoteMcpAuthNeeded (the workflow parks); any
    other failure status passes through as an ordinary error. Under
    fastmcp 4 both arrive as the same bare MCPError, so this pins the
    distinction the transport hook has to make."""
    pytest.importorskip("fastmcp")
    from rote.mcp import _runtime_helper as helper

    server = _StatusServer(status)
    try:
        if status == 401:
            with pytest.raises(helper.RoteMcpAuthNeeded, match="401") as exc_info:
                asyncio.run(helper.call_mcp_tool("slack", server.url, "t", {}))
            assert exc_info.value.server == "slack"
        else:
            with pytest.raises(Exception) as other:
                asyncio.run(helper.call_mcp_tool("slack", server.url, "t", {}))
            assert not isinstance(other.value, helper.RoteMcpAuthNeeded)
    finally:
        server.close()


@pytest.mark.parametrize("interrupt", [asyncio.CancelledError, KeyboardInterrupt])
def test_cancellation_after_a_401_cancels_instead_of_parking(
    monkeypatch: pytest.MonkeyPatch, interrupt: type[BaseException]
) -> None:
    """A cancel or Ctrl-C that lands after the server answered 401 must
    propagate as itself. Mapping it to RoteMcpAuthNeeded would park a
    workflow somebody was trying to stop."""
    from rote.mcp import _runtime_helper as helper

    def _client_seeing_401_then(*a: Any, http_status: Any = None, **kw: Any) -> Any:
        class _Ctx:
            async def __aenter__(self) -> Any:
                http_status.unauthorized = True  # the transport hook saw a 401
                raise interrupt()

            async def __aexit__(self, *exc: Any) -> None:
                return None

        return _Ctx()

    monkeypatch.setattr(helper, "mcp_client", _client_seeing_401_then)
    with pytest.raises(interrupt):
        asyncio.run(helper.call_mcp_tool("slack", "https://mcp.example.com/slack", "t", {}))

    # Negative half: the same 401 with an ordinary failure still parks.
    def _client_seeing_401_then_error(*a: Any, http_status: Any = None, **kw: Any) -> Any:
        class _Ctx:
            async def __aenter__(self) -> Any:
                http_status.unauthorized = True
                raise RuntimeError("Server returned an error response")

            async def __aexit__(self, *exc: Any) -> None:
                return None

        return _Ctx()

    monkeypatch.setattr(helper, "mcp_client", _client_seeing_401_then_error)
    with pytest.raises(helper.RoteMcpAuthNeeded, match="401"):
        asyncio.run(helper.call_mcp_tool("slack", "https://mcp.example.com/slack", "t", {}))


def test_call_mcp_tool_unwraps_auth_needed_from_the_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The OAuth redirect handler raises RoteMcpAuthNeeded inside the
    transport's task group, so it can surface wrapped. It must come out
    as itself; unrelated errors pass through untouched."""
    from rote.mcp import _runtime_helper as helper

    def _client_raising(exc: BaseException) -> Any:
        class _Ctx:
            async def __aenter__(self) -> Any:
                raise exc

            async def __aexit__(self, *a: Any) -> None:
                return None

        return _Ctx()

    needed = helper.RoteMcpAuthNeeded("slack", "the server requires an interactive OAuth flow")
    wrapped = ExceptionGroup("transport", [RuntimeError("x"), needed])
    monkeypatch.setattr(helper, "mcp_client", lambda *a, **kw: _client_raising(wrapped))
    with pytest.raises(helper.RoteMcpAuthNeeded, match="interactive") as exc_info:
        asyncio.run(helper.call_mcp_tool("slack", "https://mcp.example.com/slack", "t", {}))
    assert exc_info.value is needed

    monkeypatch.setattr(
        helper, "mcp_client", lambda *a, **kw: _client_raising(RuntimeError("boom"))
    )
    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(helper.call_mcp_tool("slack", "https://mcp.example.com/slack", "t", {}))


def test_result_payload_is_plain_json_from_a_real_fastmcp_server() -> None:
    """Every shape a real fastmcp server produces comes back as plain,
    picklable JSON: a durable runtime (DBOS) pickles the step result.

    ``result.data`` fails that for a typed object schema: fastmcp
    hydrates it into a generated ``types.Root`` class that cannot be
    pickled (found live against a hosted server). The wire value
    (structured content) is used instead, with fastmcp's
    ``{"result": ...}`` wrapper removed for non-object returns.
    """
    import pickle

    fastmcp = pytest.importorskip("fastmcp")
    from fastmcp.tools import ToolResult
    from mcp.types import TextContent

    from rote.mcp import _runtime_helper as helper

    server = fastmcp.FastMCP("payloads")

    # Keys that are not Python identifiers, as many non-Python servers
    # declare. fastmcp hydrates this into Root(display_name=..., class_=...).
    org_schema = {
        "type": "object",
        "properties": {"display-name": {"type": "string"}, "class": {"type": "string"}},
        "required": ["display-name", "class"],
    }

    @server.tool(output_schema=org_schema)
    def typed_object():  # type: ignore[no-untyped-def]  # noqa: ANN202
        return ToolResult(structured_content={"display-name": "Vaib Studio", "class": "org"})

    @server.tool
    def a_list() -> list[dict[str, int]]:
        return [{"a": 1}, {"a": 2}]

    @server.tool
    def a_scalar() -> int:
        return 7

    @server.tool(output_schema=None)
    def text_only():  # type: ignore[no-untyped-def]  # noqa: ANN202 — no annotation to resolve
        # The TS-SDK shape: no output schema, JSON riding in a text block.
        return ToolResult(content=[TextContent(type="text", text='[{"id": "abc"}]')])

    async def _payloads() -> dict[str, Any]:
        async with fastmcp.Client(server) as client:
            schemas = {t.name: t.output_schema for t in await client.list_tools()}
            out = {}
            for name in ("typed_object", "a_list", "a_scalar", "text_only"):
                result = await client.call_tool(name, {})
                out[name] = helper._result_payload(result, schemas.get(name))
            return out

    payloads = asyncio.run(_payloads())
    assert payloads == {
        # The wire key, not hydration's renamed attribute.
        "typed_object": {"display-name": "Vaib Studio", "class": "org"},
        "a_list": [{"a": 1}, {"a": 2}],
        "a_scalar": 7,
        "text_only": [{"id": "abc"}],
    }
    for value in payloads.values():
        assert pickle.loads(pickle.dumps(value)) == value
        json.dumps(value)


def test_result_payload_text_fallback_shapes() -> None:
    """Text-content edge cases no server fixture needs to produce."""
    from types import SimpleNamespace

    from rote.mcp import _runtime_helper as helper

    def _result(**kw: Any) -> Any:
        fields: dict[str, Any] = {
            "data": None,
            "structured_content": None,
            "meta": None,
            "content": [],
            "is_error": False,
        }
        fields.update(kw)
        return SimpleNamespace(**fields)

    orgs = [{"id": "abc", "name": "Vaib Studio"}]
    text_block = SimpleNamespace(text=json.dumps(orgs))

    # Hydrated data is never the payload, even when present.
    hydrated = _result(data=object(), structured_content={"wire": True})
    assert helper._result_payload(hydrated) == {"wire": True}
    # The wrapper is removed only when fastmcp marked it.
    wrapped = {"result": [1, 2]}
    assert helper._result_payload(_result(structured_content=wrapped)) == wrapped
    marked = _result(structured_content=wrapped, meta={"fastmcp": {"wrap_result": True}})
    assert helper._result_payload(marked) == [1, 2]
    flagged = {"x-fastmcp-wrap-result": True}
    assert helper._result_payload(_result(structured_content=wrapped), flagged) == [1, 2]
    # JSON in a single text block, non-JSON verbatim.
    assert helper._result_payload(_result(content=[text_block])) == orgs
    plain = _result(content=[SimpleNamespace(text="plain words")])
    assert helper._result_payload(plain) == "plain words"
    # Non-text blocks (image/audio) are skipped, not crashed on.
    image_block = SimpleNamespace(mime_type="image/png")
    assert helper._result_payload(_result(content=[image_block, text_block])) == orgs
    # Several text blocks parse independently.
    two = _result(content=[SimpleNamespace(text='{"a": 1}'), SimpleNamespace(text="raw")])
    assert helper._result_payload(two) == [{"a": 1}, "raw"]
    # Nothing usable at all stays None.
    assert helper._result_payload(_result()) is None


def test_auth_needed_survives_pickling() -> None:
    """DBOS serializes step errors across queue workers — the two-arg
    __init__ needs the explicit __reduce__ to round-trip."""
    import pickle

    from rote.mcp._runtime_helper import RoteMcpAuthNeeded

    exc = pickle.loads(pickle.dumps(RoteMcpAuthNeeded("vendor", "expired")))
    assert exc.server == "vendor"
    assert "rote mcp login vendor" in str(exc)


def test_oauth_client_never_opens_a_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    """The helper's OAuth subclass raises RoteMcpAuthNeeded instead of
    invoking the redirect handler — emitted workflows run unattended."""
    import time
    import webbrowser

    from rote.mcp import _runtime_helper as helper

    write_token_file("slack", _doc(expires_at=time.time() + 3600))

    def _no_browser(*a: Any, **kw: Any) -> bool:
        raise AssertionError("workflow code must never open a browser")

    monkeypatch.setattr(webbrowser, "open", _no_browser)
    client = helper.mcp_client("slack", "https://mcp.example.com/slack")
    redirect = client.transport.auth.redirect_handler
    with pytest.raises(helper.RoteMcpAuthNeeded, match="interactive"):
        asyncio.run(redirect("https://auth.example.com/authorize?x=1"))


# ───────── the CIMD document and the login's registration rules ─────────

_CIMD_DOCUMENT = Path(__file__).resolve().parent.parent / "docs" / "oauth" / "rote-cli.json"


def test_cimd_document_matches_what_the_login_offers() -> None:
    """The published document and the CLI must agree byte for byte on the
    client id and the redirect URIs. An AS compares both by exact string
    match, so a drift here fails every CIMD login, on servers we cannot
    test against before publishing."""
    from rote.mcp.auth import (
        CALLBACK_PORTS,
        DEFAULT_CLIENT_METADATA_URL,
        callback_redirect_uris,
    )

    doc = json.loads(_CIMD_DOCUMENT.read_text(encoding="utf-8"))
    assert doc["client_id"] == DEFAULT_CLIENT_METADATA_URL
    assert DEFAULT_CLIENT_METADATA_URL == "https://roteskills.com/oauth/rote-cli.json"
    assert doc["redirect_uris"] == callback_redirect_uris()
    assert {f"http://localhost:{p}/callback" for p in CALLBACK_PORTS} <= set(doc["redirect_uris"])
    # A CLI is a public client; the CIMD draft forbids shared-secret methods.
    assert doc["token_endpoint_auth_method"] == "none"
    assert set(doc["grant_types"]) == {"authorization_code", "refresh_token"}
    assert doc["response_types"] == ["code"]
    assert doc["client_name"]


def test_client_metadata_url_override_and_opt_out(monkeypatch: pytest.MonkeyPatch) -> None:
    from rote.mcp.auth import (
        CLIENT_METADATA_URL_ENV_VAR,
        DEFAULT_CLIENT_METADATA_URL,
        client_metadata_url,
    )

    monkeypatch.delenv(CLIENT_METADATA_URL_ENV_VAR, raising=False)
    assert client_metadata_url() == DEFAULT_CLIENT_METADATA_URL
    monkeypatch.setenv(CLIENT_METADATA_URL_ENV_VAR, "https://127.0.0.1:9/oauth/x.json")
    assert client_metadata_url() == "https://127.0.0.1:9/oauth/x.json"
    monkeypatch.setenv(CLIENT_METADATA_URL_ENV_VAR, "")
    assert client_metadata_url() is None


def test_login_errors_when_every_published_callback_port_is_taken(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rote.mcp import auth

    monkeypatch.setattr(auth, "_port_is_free", lambda port: False)
    with pytest.raises(auth.McpAuthError, match="33418, 33419, 33420, 33421"):
        auth.select_callback_port()
    taken = {auth.CALLBACK_PORTS[0], auth.CALLBACK_PORTS[1]}
    monkeypatch.setattr(auth, "_port_is_free", lambda port: port not in taken)
    assert auth.select_callback_port() == auth.CALLBACK_PORTS[2]


def test_input_required_error_pickles_and_is_not_an_auth_error() -> None:
    """DBOS pickles step errors; the emitted workflow must still see the
    class (to fail instead of parking) after the round trip."""
    import pickle

    from rote.mcp._runtime_helper import RoteMcpAuthNeeded, RoteMcpInputRequired

    err = RoteMcpInputRequired("vendor", "lookup", {"approval": "elicitation/create"})
    back = pickle.loads(pickle.dumps(err))
    assert isinstance(back, RoteMcpInputRequired)
    assert not isinstance(back, RoteMcpAuthNeeded)
    assert (back.server, back.tool, back.requests) == ("vendor", "lookup", err.requests)
    assert "approval (elicitation/create)" in str(back)
    assert "rote mcp login" not in str(back)


def test_packaged_client_metadata_is_the_published_document() -> None:
    """Installed CLIs compare the live document with the packaged copy, so
    the copy in the wheel and docs/oauth/rote-cli.json must be one file."""
    packaged = Path(__file__).resolve().parent.parent / "src" / "rote" / "mcp"
    assert (packaged / "client_metadata.json").read_bytes() == _CIMD_DOCUMENT.read_bytes()
    from rote.mcp.auth import packaged_client_metadata

    assert packaged_client_metadata() == json.loads(_CIMD_DOCUMENT.read_text(encoding="utf-8"))


# ───────── a login belongs to one endpoint ─────────


@pytest.mark.parametrize(
    ("a", "b", "same"),
    [
        ("https://mcp.example.com/mcp", "https://mcp.example.com/mcp/", True),
        ("https://MCP.Example.com:443/mcp", "https://mcp.example.com/mcp", True),
        ("http://127.0.0.1:80/mcp", "http://127.0.0.1/mcp", True),
        ("https://mcp.example.com/mcp", "https://mcp.example.com:8443/mcp", False),
        ("https://mcp.example.com/mcp", "https://other.example.com/mcp", False),
        ("https://mcp.example.com/mcp", "http://mcp.example.com/mcp", False),
        ("https://mcp.example.com/v1/mcp", "https://mcp.example.com/v2/mcp", False),
    ],
)
def test_normalize_server_url(a: str, b: str, same: bool) -> None:
    from rote.mcp._runtime_helper import normalize_server_url

    assert (normalize_server_url(a) == normalize_server_url(b)) is same


def _login_for(url: str) -> dict[str, Any]:
    return {
        **_doc(expires_at=None),
        "server_url": url,
        "issuer": "https://auth.example.com",
    }


def test_a_login_for_another_endpoint_is_not_read_as_this_ones() -> None:
    """The file is keyed by name; after `rote mcp add slack <new-url>` the
    old endpoint's bearer token must not reach the new host."""
    from rote.mcp import _runtime_helper

    write_token_file("slack", _login_for("https://old.example.com/mcp"))

    async def read(kv: Any) -> Any:
        return await kv.get("k", collection="mcp-oauth-token")

    new = "https://new.example.com/mcp"
    assert asyncio.run(read(ServerTokenKV("slack", new))) is None
    assert asyncio.run(read(_runtime_helper._TokenKV("slack", new))) is None
    assert asyncio.run(read(ServerTokenKV("slack", "https://old.example.com/mcp/"))) is not None
    assert not _runtime_helper.has_stored_login("slack", new)
    assert _runtime_helper.login_for_another_url("slack", new) == "https://old.example.com/mcp"


def test_the_cli_refuses_to_mint_headers_for_a_repointed_server() -> None:
    from rote.mcp import McpAuthError
    from rote.mcp.auth import fresh_access_token

    write_token_file("slack", _login_for("https://old.example.com/mcp"))
    config = McpServerConfig(url="https://new.example.com/mcp")
    with pytest.raises(McpAuthError, match=r"is for https://old.example.com/mcp.*rote mcp login"):
        asyncio.run(fresh_access_token("slack", config))
    from rote.mcp import auth_status

    assert auth_status(config, read_token_file("slack")) == "not authenticated"


def test_the_emitted_helper_parks_instead_of_sending_another_endpoints_token() -> None:
    from rote.mcp import _runtime_helper

    write_token_file("slack", _login_for("https://old.example.com/mcp"))
    # Port 9 refuses connections: reaching the network would be a different error.
    with pytest.raises(_runtime_helper.RoteMcpAuthNeeded, match="stored login is for"):
        asyncio.run(_runtime_helper.call_mcp_tool("slack", "http://127.0.0.1:9/mcp", "t", {}))


def test_a_stored_client_without_an_issuer_takes_the_files() -> None:
    """The SDK reads a binding only from client_info.issuer and treats a
    missing one as bound to nothing."""
    from rote.mcp import _runtime_helper

    url = "https://mcp.example.com/mcp"
    write_token_file("slack", _login_for(url))

    async def client_info(kv: Any) -> Any:
        return await kv.get("k", collection="mcp-oauth-client-info")

    for kv in (ServerTokenKV("slack", url), _runtime_helper._TokenKV("slack", url)):
        assert asyncio.run(client_info(kv)) == {
            "client_id": "abc",
            "issuer": "https://auth.example.com",
        }


def test_mcp_add_issuer_needs_a_client_id_and_is_stored() -> None:
    assert cli_main(["mcp", "add", "gh", "https://h.example/mcp", "--issuer", "https://x"]) == 2
    assert (
        cli_main(
            [
                "mcp",
                "add",
                "gh",
                "https://h.example/mcp",
                "--client-id",
                "app",
                "--issuer",
                "https://github.com/login/oauth",
            ]
        )
        == 0
    )
    assert load_registry().servers["gh"].issuer == "https://github.com/login/oauth"


# ───────── hardening ─────────


@pytest.mark.parametrize(
    ("url", "trusted"),
    [
        ("https://auth.example.com", True),
        ("http://127.0.0.1:8000/as", True),
        ("http://[::1]:8000/as", True),
        ("http://localhost/as", True),
        ("http://auth.example.com", False),
        ("http://127.0.0.1.attacker.example/as", False),
        ("ftp://auth.example.com", False),
    ],
)
def test_only_https_or_loopback_issuers_are_trusted(url: str, trusted: bool) -> None:
    from rote.mcp.auth import is_trusted_transport

    assert is_trusted_transport(url) is trusted


def test_issuer_discovery_does_not_trust_a_plain_http_resource() -> None:
    """Over plain HTTP a network attacker writes the metadata, and with it
    the server a client secret is later offered to. No request is made."""
    from rote.mcp.auth import discover_issuer

    found = asyncio.run(discover_issuer("http://mcp.example.com/mcp"))
    assert found.issuer is None
    assert "not HTTPS" in (found.problem or "")


def test_mcp_add_refuses_a_plain_http_issuer(capsys: pytest.CaptureFixture[str]) -> None:
    args = ["mcp", "add", "gh", "https://h.example/mcp", "--client-id", "app"]
    assert cli_main([*args, "--issuer", "http://auth.example.com"]) == 2
    assert "must be an https URL" in capsys.readouterr().err
    assert cli_main([*args, "--issuer", "http://127.0.0.1:9000/as"]) == 0


def test_an_unparseable_stored_url_parks_instead_of_crashing() -> None:
    """A bad port in the stored server_url made the URL comparison raise
    ValueError and crash the run. It names no endpoint, so it is not this one."""
    from rote.mcp import _runtime_helper

    write_token_file("slack", _login_for("https://old.example.com:99999/mcp"))
    with pytest.raises(_runtime_helper.RoteMcpAuthNeeded, match="stored login is for"):
        asyncio.run(_runtime_helper.call_mcp_tool("slack", "http://127.0.0.1:9/mcp", "t", {}))


def test_export_takes_client_id_and_secret_from_one_source(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A dynamically registered client must not be exported with the
    registry's pre-registered secret."""
    url = "https://mcp.example.com/slack"
    save_registry(
        McpRegistry(
            servers={"slack": McpServerConfig(url=url, client_id="pre", client_secret="pre-secret")}
        )
    )
    doc = {
        **_doc(expires_at=None),
        "server_url": url,
        "client_info": {"client_id": "dcr-1"},
        "tokens": {"access_token": "a", "token_type": "Bearer", "refresh_token": "r"},
    }
    write_token_file("slack", doc)
    assert cli_main(["mcp", "export", "slack", "--json"]) == 0
    exported = json.loads(capsys.readouterr().out)
    assert exported["ROTE_MCP_SLACK_CLIENT_ID"] == "dcr-1"
    assert "ROTE_MCP_SLACK_CLIENT_SECRET" not in exported

    doc["client_info"] = None
    write_token_file("slack", doc)
    assert cli_main(["mcp", "export", "slack", "--json"]) == 0
    exported = json.loads(capsys.readouterr().out)
    assert exported["ROTE_MCP_SLACK_CLIENT_ID"] == "pre"
    assert exported["ROTE_MCP_SLACK_CLIENT_SECRET"] == "pre-secret"
