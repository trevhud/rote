"""A login requests the scopes the user set, never the server's wider list (#316).

On a 401 the MCP SDK replaces the requested scopes with the challenge's, else
every scope the protected-resource or authorization-server metadata lists as
supported, and never looks at the configured ones. A real login configured
with six read-only scopes asked the user to consent to 26, Gmail send among
them, while the registry still showed six. These run the real rote, fastmcp
and SDK login path against the local authorization server
(tests/_oauth_as_server.py) and read the ``scope`` the authorization request
actually carried.
"""

# ruff: noqa: F811 (pytest fixtures imported from the registration suite)

from __future__ import annotations

import asyncio

import pytest

pytestmark = pytest.mark.slow

from rote.mcp import McpAuthError, read_token_file, write_token_file  # noqa: E402
from rote.mcp.auth import login  # noqa: E402
from tests._oauth_fixture import OAuthFixture  # noqa: E402
from tests.test_mcp_oauth_registration_e2e import (  # noqa: E402, F401
    _call_through_emitted_helper,
    _config,
    fixture_server,
    fx,
)

WIDE = ["read_data", "write_data", "delete_data"]


def _requested(fx: OAuthFixture) -> set[str | None]:
    """The scope every authorization request carried (fastmcp pre-flights
    the URL before the browser opens it, so each login logs it twice)."""
    return {e["scope"] for e in fx.events("authorize")}


def test_configured_scopes_win_over_advertised_ones(fx: OAuthFixture) -> None:
    fx.configure(prm_scopes=WIDE, as_scopes=WIDE)
    config = _config(fx, scopes=["read_data"])
    asyncio.run(login("vendor", config))
    assert _requested(fx) == {"read_data"}
    assert _call_through_emitted_helper(fx) == {"authenticated": True}


def test_with_no_configured_scopes_the_server_decides(fx: OAuthFixture) -> None:
    """The negative half: discovery still picks the scopes when the user set none."""
    fx.configure(prm_scopes=WIDE)
    asyncio.run(login("vendor", _config(fx)))
    assert _requested(fx) == {" ".join(WIDE)}


def test_a_challenge_requiring_more_stops_before_the_consent_page(fx: OAuthFixture) -> None:
    fx.configure(prm_scopes=WIDE, challenge_scope="read_data write_data")
    config = _config(fx, scopes=["read_data"])
    with pytest.raises(McpAuthError) as caught:
        asyncio.run(login("vendor", config))
    message = str(caught.value)
    assert "requires scope(s) write_data beyond the ones configured for it (read_data)" in message
    assert "rote did not open the consent page" in message
    assert _requested(fx) == set()


def test_a_challenge_within_the_configured_scopes_is_fine(fx: OAuthFixture) -> None:
    fx.configure(prm_scopes=WIDE, challenge_scope="read_data")
    asyncio.run(login("vendor", _config(fx, scopes=["read_data", "write_data"])))
    assert _requested(fx) == {"read_data write_data"}


def test_offline_access_is_kept_for_a_refresh_token(fx: OAuthFixture) -> None:
    """The SDK adds offline_access when the server supports it (SEP-2207): it
    asks for a refresh token, grants no access, and unattended runs need it."""
    fx.configure(prm_scopes=WIDE, as_scopes=[*WIDE, "offline_access"])
    asyncio.run(login("vendor", _config(fx, scopes=["read_data"])))
    assert _requested(fx) == {"read_data offline_access"}


def test_a_login_with_a_stored_client_keeps_the_configured_scopes(fx: OAuthFixture) -> None:
    """A second login reuses the registered client from the token file; the
    scopes must not be lost on that path either."""
    fx.configure(prm_scopes=WIDE, as_scopes=WIDE)
    config = _config(fx, scopes=["read_data"])
    asyncio.run(login("vendor", config))
    stored = read_token_file("vendor") or {}
    assert stored.get("client_info")
    write_token_file("vendor", {k: v for k, v in stored.items() if k != "tokens"})
    before = len(fx.events("authorize"))
    asyncio.run(login("vendor", config))
    assert len(fx.events("authorize")) > before  # it really authorized again
    assert _requested(fx) == {"read_data"}


def test_a_dynamic_registration_declares_only_the_configured_scopes(fx: OAuthFixture) -> None:
    """The SDK builds the DCR body from the scopes it chose on the 401, before
    the authorization step: the client would declare the server's whole list
    (PR #317 review)."""
    fx.configure(cimd=False, prm_scopes=WIDE, as_scopes=WIDE)
    asyncio.run(login("vendor", _config(fx, scopes=["read_data"])))
    assert [e["scope"] for e in fx.events("register")] == ["read_data"]
    assert _requested(fx) == {"read_data"}


def test_a_refused_login_registers_no_client(fx: OAuthFixture) -> None:
    fx.configure(cimd=False, prm_scopes=WIDE, challenge_scope="read_data write_data")
    with pytest.raises(McpAuthError, match="requires scope"):
        asyncio.run(login("vendor", _config(fx, scopes=["read_data"])))
    assert fx.events("register") == []


def test_a_challenge_naming_offline_access_is_not_a_refusal(fx: OAuthFixture) -> None:
    fx.configure(prm_scopes=WIDE, challenge_scope="read_data offline_access")
    asyncio.run(login("vendor", _config(fx, scopes=["read_data"])))
    assert _requested(fx) == {"read_data offline_access"}


def test_a_scope_entry_holding_several_words_counts_each(fx: OAuthFixture) -> None:
    """`--scope "read_data write_data"` is one entry; a challenge for one of
    its words is within what the user configured (PR #317 review)."""
    fx.configure(prm_scopes=WIDE, challenge_scope="write_data")
    asyncio.run(login("vendor", _config(fx, scopes=["read_data write_data"])))
    assert _requested(fx) == {"read_data write_data"}


def test_a_discovery_fetch_answering_401_does_not_set_the_requirement(fx: OAuthFixture) -> None:
    """Only the MCP request's own 401 says what the resource requires. A
    gateway that 401s a well-known path it lacks, with its own scope=, must
    neither clear that requirement nor invent one (PR #317 review)."""
    fx.configure(prm_path=False, prm_scopes=WIDE, well_known_401_scope="admin")
    asyncio.run(login("vendor", _config(fx, scopes=["read_data"])))
    assert _requested(fx) == {"read_data"}
