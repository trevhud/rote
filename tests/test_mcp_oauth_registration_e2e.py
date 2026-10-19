"""`rote mcp login` against every registration mode the ecosystem has.

The 2026-07-28 spec orders client registration: a pre-registered client,
then a Client ID Metadata Document (CIMD) when the authorization server
advertises it, then dynamic client registration (DCR), then asking the
user. Probing 14 hosted authorization servers found every one of those
shapes in the wild (docs/design/mcp-2026-07-28.md, A1): Linear-class
servers offer CIMD and DCR, Atlassian-class only DCR, GitHub-class
neither. This suite runs rote's real login against a local authorization
server that can be switched into each shape (``tests/_oauth_as_server.py``),
serving rote's real CIMD document from a local TLS URL.

Every test asserts on the authorization server's hit log as well as the
outcome, because the failures that matter here are requests that should
never have been sent: a registration the spec ordering skips, a guessed
registration URL, a token request after a bad ``iss``.

Slow: a real AS process and real loopback OAuth callbacks.
"""

from __future__ import annotations

import asyncio
import json
import socket
import time
from collections.abc import Iterator

import pytest

pytestmark = pytest.mark.slow

pytest.importorskip("fastmcp")
pytest.importorskip("trustme")

from rote.mcp import (  # noqa: E402
    McpAuthError,
    McpRegistry,
    McpServerConfig,
    read_token_file,
    save_registry,
    token_file,
    write_token_file,
)
from rote.mcp.auth import (  # noqa: E402
    CALLBACK_HOST,
    CALLBACK_PORTS,
    CLIENT_METADATA_URL_ENV_VAR,
    fresh_access_token,
    login,
    registration_of,
)
from tests._oauth_fixture import (  # noqa: E402
    CIMD_DOCUMENT,
    OAuthFixture,
    fake_browser_open,
    free_port,
)

PRE_ID = "github-style-app"


@pytest.fixture(scope="module")
def fixture_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[OAuthFixture]:
    from tests._oauth_fixture import oauth_fixture

    with oauth_fixture(tmp_path_factory.mktemp("oauth-as")) as fx:
        yield fx


@pytest.fixture()
def fx(fixture_server: OAuthFixture, monkeypatch: pytest.MonkeyPatch) -> Iterator[OAuthFixture]:
    """The fixture reset to its defaults, with rote pointed at its CIMD URL."""
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
    monkeypatch.setenv(CLIENT_METADATA_URL_ENV_VAR, fixture_server.cimd_url)
    monkeypatch.setenv("SSL_CERT_FILE", str(fixture_server.ca_pem))
    monkeypatch.setattr("webbrowser.open", fake_browser_open())
    # rote checks the published document once per process; each test
    # changes what the fixture serves.
    monkeypatch.setattr("rote.mcp.auth._document_checks", {})
    yield fixture_server


def _config(fx: OAuthFixture, **extra: object) -> McpServerConfig:
    config = McpServerConfig(url=fx.mcp_url, **extra)  # type: ignore[arg-type]
    save_registry(McpRegistry(servers={"vendor": config}))
    return config


def _preregister(fx: OAuthFixture) -> None:
    fx.configure(
        preregistered={PRE_ID: {"redirect_uris": [f"http://{CALLBACK_HOST}:33418/callback"]}}
    )


def _call_through_emitted_helper(fx: OAuthFixture) -> dict:
    """A real tool call from the stored login, the way emitted workflows run."""
    from rote.mcp import _runtime_helper

    return asyncio.run(_runtime_helper.call_mcp_tool("vendor", fx.mcp_url, "whoami", {}))


def test_cimd_is_used_when_advertised_and_no_registration_is_sent(fx: OAuthFixture) -> None:
    fx.configure(dcr=False)
    config = _config(fx)

    doc = asyncio.run(login("vendor", config))

    assert doc["client_info"]["client_id"] == fx.cimd_url
    assert registration_of(doc, config) == "cimd"
    assert fx.events("register") == []
    # The AS really resolved the document and matched the redirect URI
    # against it: rote's published port set is what makes this pass.
    assert {e["url"] for e in fx.events("cimd_fetch")} == {fx.cimd_url}
    document = json.loads(CIMD_DOCUMENT.read_text(encoding="utf-8"))
    redirects = {e["redirect_uri"] for e in fx.events("authorize")}
    assert redirects and redirects <= set(document["redirect_uris"])
    assert doc["issuer"] == fx.issuer("a")
    assert doc["version"] == 2
    assert _call_through_emitted_helper(fx) == {"authenticated": True}

    # Refresh with a CIMD client: the client_id on the refresh grant is the URL.
    stale = read_token_file("vendor")
    assert stale is not None
    stale["expires_at"] = time.time() - 10
    write_token_file("vendor", stale)
    asyncio.run(fresh_access_token("vendor", config))
    refreshes = [e for e in fx.events("token") if e["grant_type"] == "refresh_token"]
    assert [e["client_id"] for e in refreshes] == [fx.cimd_url]


def test_cimd_wins_over_dcr_when_both_are_offered(fx: OAuthFixture) -> None:
    doc = asyncio.run(login("vendor", _config(fx)))
    assert doc["client_info"]["client_id"] == fx.cimd_url
    assert fx.events("register") == []


def test_dcr_is_the_fallback_when_cimd_is_not_advertised(fx: OAuthFixture) -> None:
    fx.configure(cimd=False)
    config = _config(fx)
    doc = asyncio.run(login("vendor", config))

    client_id = doc["client_info"]["client_id"]
    assert client_id.startswith("dcr-")
    assert registration_of(doc, config) == "dcr"
    assert len(fx.events("register")) == 1
    assert fx.events("cimd_fetch") == []
    assert {e["client_id"] for e in fx.events("authorize")} == {client_id}
    assert doc["issuer"] == fx.issuer("a")


def test_a_pre_registered_client_wins_even_when_cimd_and_dcr_are_offered(
    fx: OAuthFixture,
) -> None:
    _preregister(fx)
    config = _config(fx, client_id=PRE_ID)
    doc = asyncio.run(login("vendor", config))

    assert doc["client_info"]["client_id"] == PRE_ID
    assert registration_of(doc, config) == "pre-registered"
    assert fx.events("register") == []
    assert fx.events("cimd_fetch") == []
    assert {e["client_id"] for e in fx.events("authorize")} == {PRE_ID}


def test_a_pre_registered_client_works_where_nothing_else_does(fx: OAuthFixture) -> None:
    """GitHub's shape: no registration endpoint, no CIMD."""
    fx.configure(cimd=False, dcr=False)
    _preregister(fx)
    doc = asyncio.run(login("vendor", _config(fx, client_id=PRE_ID)))
    assert doc["client_info"]["client_id"] == PRE_ID
    assert _call_through_emitted_helper(fx) == {"authenticated": True}


def test_no_registration_path_names_the_fix_and_guesses_no_url(fx: OAuthFixture) -> None:
    """Without the guard the SDK POSTs a guessed `<resource-origin>/register`."""
    fx.configure(cimd=False, dcr=False)
    with pytest.raises(McpAuthError, match=r"rote mcp add vendor <url> --client-id"):
        asyncio.run(login("vendor", _config(fx)))
    assert fx.events("register") == []
    assert [e for e in fx.events("not_found") if e["method"] == "POST"] == []
    assert fx.events("authorize") == []
    assert read_token_file("vendor") is None or not (read_token_file("vendor") or {}).get("tokens")


def test_cimd_can_be_turned_off(fx: OAuthFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CLIENT_METADATA_URL_ENV_VAR, "")
    doc = asyncio.run(login("vendor", _config(fx)))
    assert doc["client_info"]["client_id"].startswith("dcr-")
    assert fx.events("cimd_fetch") == []


def test_a_callback_port_outside_the_published_set_does_not_offer_cimd(
    fx: OAuthFixture,
) -> None:
    """The document lists only CALLBACK_PORTS; offering CIMD with another
    port would be rejected by exactly the servers that support CIMD."""
    port = free_port()
    assert port not in CALLBACK_PORTS
    doc = asyncio.run(login("vendor", _config(fx), callback_port=port))
    assert doc["client_info"]["client_id"].startswith("dcr-")
    assert {e["redirect_uri"] for e in fx.events("authorize")} == {
        f"http://{CALLBACK_HOST}:{port}/callback"
    }


def test_login_moves_to_the_next_published_port_when_the_first_is_taken(
    fx: OAuthFixture,
) -> None:
    with socket.socket() as held:
        held.bind(("127.0.0.1", CALLBACK_PORTS[0]))
        held.listen()
        doc = asyncio.run(login("vendor", _config(fx)))
    assert doc["client_info"]["client_id"] == fx.cimd_url
    assert {e["redirect_uri"] for e in fx.events("authorize")} == {
        f"http://{CALLBACK_HOST}:{CALLBACK_PORTS[1]}/callback"
    }


@pytest.mark.parametrize(
    ("iss", "advertise", "error"),
    [
        ("mismatch", True, "iss mismatch"),
        ("omit", True, "missing iss"),
        # Spec row 3: a present iss is compared even when not advertised.
        ("mismatch", False, "iss mismatch"),
    ],
)
def test_a_bad_authorization_response_iss_is_rejected_before_the_token_request(
    fx: OAuthFixture, iss: str, advertise: bool, error: str
) -> None:
    fx.configure(iss=iss, advertise_iss=advertise)
    with pytest.raises(McpAuthError, match=error):
        asyncio.run(login("vendor", _config(fx)))
    assert fx.events("authorize"), "the flow must reach the authorization response"
    assert fx.events("token") == []
    stored = read_token_file("vendor") or {}
    assert not (stored.get("tokens") or {}).get("access_token")


def test_a_missing_iss_proceeds_when_the_server_does_not_advertise_it(
    fx: OAuthFixture,
) -> None:
    fx.configure(iss="omit", advertise_iss=False)
    doc = asyncio.run(login("vendor", _config(fx)))
    assert doc["tokens"]["access_token"]


def test_dcr_credentials_are_not_offered_to_a_new_authorization_server(
    fx: OAuthFixture,
) -> None:
    """Credentials are keyed by issuer: when the protected-resource
    metadata starts naming another AS, the old client must not be sent there."""
    fx.configure(cimd=False)
    config = _config(fx)
    first = asyncio.run(login("vendor", config))
    old_client = first["client_info"]["client_id"]
    assert first["issuer"] == fx.issuer("a")

    fx.configure(active="b")
    second = asyncio.run(login("vendor", config))

    at_b = [e for e in fx.events("authorize") if e["issuer"] == fx.issuer("b")]
    assert at_b and old_client not in {e["client_id"] for e in at_b}
    assert [e["issuer"] for e in fx.events("register")] == [fx.issuer("a"), fx.issuer("b")]
    assert second["issuer"] == fx.issuer("b")
    assert second["client_info"]["client_id"] != old_client


@pytest.mark.parametrize("cimd", [True, False], ids=["new-as-offers-cimd", "new-as-dcr-only"])
def test_a_pre_registered_client_bound_to_one_issuer_is_not_used_at_another(
    fx: OAuthFixture, cimd: bool
) -> None:
    """The SDK drops the bound client and would pick CIMD or DCR at the new
    server by itself. Both must stop before the browser opens."""
    fx.configure(cimd=cimd)
    _preregister(fx)
    config = _config(fx, client_id=PRE_ID)
    first = asyncio.run(login("vendor", config))
    assert first["issuer"] == fx.issuer("a")

    fx.configure(active="b")
    with pytest.raises(McpAuthError, match=r"--client-id"):
        asyncio.run(login("vendor", config))
    assert [e for e in fx.events("authorize") if e["issuer"] == fx.issuer("b")] == []
    assert [e for e in fx.events("register") if e["issuer"] == fx.issuer("b")] == []
    assert [e for e in fx.events("cimd_fetch") if e["issuer"] == fx.issuer("b")] == []


def test_refresh_never_starts_an_interactive_flow(fx: OAuthFixture) -> None:
    """`rote mcp headers` runs unattended: a dead refresh token must end in
    an error, not a browser or a registration."""
    config = _config(fx)
    asyncio.run(login("vendor", config))
    doc = read_token_file("vendor")
    assert doc is not None
    doc["expires_at"] = time.time() - 10
    doc["tokens"]["refresh_token"] = "revoked"
    write_token_file("vendor", doc)
    authorizes_before = len(fx.events("authorize"))

    with pytest.raises(McpAuthError, match="rote mcp login vendor"):
        asyncio.run(fresh_access_token("vendor", config))
    assert len(fx.events("authorize")) == authorizes_before
    assert fx.events("register") == []


def test_emitted_helper_refreshes_at_the_token_endpoint_login_recorded(
    fx: OAuthFixture,
) -> None:
    """Without discovery the SDK refreshes at `<resource-origin>/token`, which
    404s for this path-issuer AS; the run would then park on a credential
    that was fine."""
    fx.configure(cimd=False)
    asyncio.run(login("vendor", _config(fx)))
    doc = read_token_file("vendor")
    assert doc is not None
    doc["expires_at"] = time.time() - 10
    write_token_file("vendor", doc)

    assert _call_through_emitted_helper(fx) == {"authenticated": True}
    refreshes = [e for e in fx.events("token") if e["grant_type"] == "refresh_token"]
    assert [e["issuer"] for e in refreshes] == [fx.issuer("a")]
    assert [e for e in fx.events("not_found") if e["path"].endswith("/token")] == []


def test_emitted_helper_parks_instead_of_registering_a_client(fx: OAuthFixture) -> None:
    from rote.mcp import _runtime_helper

    fx.configure(cimd=False)
    asyncio.run(login("vendor", _config(fx)))
    doc = read_token_file("vendor")
    assert doc is not None
    doc["client_info"] = None
    doc["expires_at"] = time.time() - 10
    write_token_file("vendor", doc)
    registrations_before = len(fx.events("register"))

    with pytest.raises(_runtime_helper.RoteMcpAuthNeeded, match="no longer valid"):
        _call_through_emitted_helper(fx)
    assert len(fx.events("register")) == registrations_before


# ───────── the published document is checked before it is offered ─────────


@pytest.mark.parametrize("served", ["missing", "tampered"])
def test_an_unusable_published_document_falls_back_to_dcr(
    fx: OAuthFixture, served: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """The SDK does not fall back when a CIMD server rejects the document,
    so offering an unpublished or stale one would break logins that DCR
    serves today (the state before roteskills.com serves the file)."""
    fx.configure(cimd_document=served)
    doc = asyncio.run(login("vendor", _config(fx)))

    assert doc["client_info"]["client_id"].startswith("dcr-")
    assert len(fx.events("register")) == 1
    assert fx.cimd_url not in {e["client_id"] for e in fx.events("authorize")}
    assert fx.events("cimd_fetch") == []  # the AS was never asked to resolve it
    assert len(fx.events("document_check")) == 1
    assert "registering a client with this server instead" in capsys.readouterr().err


def test_a_matching_published_document_is_checked_once_per_process(fx: OAuthFixture) -> None:
    config = _config(fx)
    asyncio.run(login("vendor", config))
    stale = read_token_file("vendor")
    assert stale is not None
    stale["tokens"] = None
    stale["client_info"] = None
    write_token_file("vendor", stale)
    second = asyncio.run(login("vendor", config))

    assert second["client_info"]["client_id"] == fx.cimd_url
    assert len(fx.events("document_check")) == 1


def test_a_server_without_cimd_triggers_no_document_check(fx: OAuthFixture) -> None:
    fx.configure(cimd=False)
    asyncio.run(login("vendor", _config(fx)))
    assert fx.events("document_check") == []


# ───────── issuer binding for clients stored before it existed ─────────


def test_a_v1_dcr_client_is_not_offered_to_a_repointed_authorization_server(
    fx: OAuthFixture,
) -> None:
    """A v1 token file records no issuer anywhere, and the SDK treats an
    unbound client as matching every issuer: after the resource repoints at
    another AS, the next login would send the old DCR client there."""
    fx.configure(cimd=False)
    config = _config(fx)
    first = asyncio.run(login("vendor", config))
    old_client = first["client_info"]["client_id"]

    v1 = read_token_file("vendor")
    assert v1 is not None
    v1.pop("issuer")
    v1["client_info"].pop("issuer", None)
    v1["expires_at"] = time.time() - 10
    v1["tokens"]["refresh_token"] = "revoked"
    path = token_file("vendor")
    path.write_text(json.dumps({**v1, "version": 1}), encoding="utf-8")

    fx.configure(active="b")
    second = asyncio.run(login("vendor", config))

    at_b = [e for e in fx.events() if e.get("issuer") == fx.issuer("b")]
    assert old_client not in {e.get("client_id") for e in at_b}
    assert [e["issuer"] for e in fx.events("register")] == [fx.issuer("a"), fx.issuer("b")]
    assert second["issuer"] == fx.issuer("b")
    assert second["client_info"]["issuer"] == fx.issuer("b")


def test_login_binds_the_stored_client_in_both_places(fx: OAuthFixture) -> None:
    """The SDK records an issuer on a client only when it registers one, so a
    pre-registered client would be stored unbound; readers of the file (the
    emitted helper, `rote mcp export`) see the binding only if login writes it."""
    _preregister(fx)
    doc = asyncio.run(login("vendor", _config(fx, client_id=PRE_ID)))
    assert doc["client_info"]["client_id"] == PRE_ID
    assert doc["issuer"] == doc["client_info"]["issuer"] == fx.issuer("a")


# ───────── a pre-registered client is bound from its first login ─────────


def test_mcp_add_records_the_issuer_a_pre_registered_client_belongs_to(
    fx: OAuthFixture,
) -> None:
    from rote.cli import main as cli_main
    from rote.mcp import load_registry

    _preregister(fx)
    assert cli_main(["mcp", "add", "vendor", fx.mcp_url, "--client-id", PRE_ID]) == 0
    assert load_registry().servers["vendor"].issuer == fx.issuer("a")

    # Repointed before the first login: the recorded issuer stops it.
    fx.configure(active="b")
    config = load_registry().servers["vendor"]
    with pytest.raises(McpAuthError, match=r"--client-id"):
        asyncio.run(login("vendor", config))
    assert [e for e in fx.events("authorize") if e["issuer"] == fx.issuer("b")] == []


# ───────── hardening ─────────


def test_a_dribbling_document_host_cannot_hold_the_login_open(
    fx: OAuthFixture, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """httpx's timeout is per read: a byte every half second never trips it.
    The check needs a total deadline, then falls back to DCR."""
    monkeypatch.setattr("rote.mcp.auth._DOCUMENT_CHECK_TIMEOUT_SECONDS", 1.0)
    fx.configure(cimd_document="dribble")

    async def bounded() -> dict:
        # Bound the test itself so a regression fails instead of hanging.
        return await asyncio.wait_for(login("vendor", _config(fx)), timeout=30)

    started = time.monotonic()
    doc = asyncio.run(bounded())
    assert time.monotonic() - started < 15
    assert doc["client_info"]["client_id"].startswith("dcr-")
    assert "no complete answer within 1s" in capsys.readouterr().err


def test_issuer_discovery_requires_metadata_for_this_resource(fx: OAuthFixture) -> None:
    """RFC 9728 section 3.3: the document must name this resource. On a
    shared host the origin-level document describes some other resource,
    and its issuer would bind the client to the wrong server."""
    from rote.mcp.auth import discover_issuer

    assert asyncio.run(discover_issuer(fx.mcp_url)).issuer == fx.issuer("a")
    fx.configure(prm_path=False, prm_origin_resource=f"{fx.base}/other")
    found = asyncio.run(discover_issuer(fx.mcp_url))
    assert found.issuer is None
    assert f"{fx.base}/other" in (found.problem or "")
    # The origin document is fine when it does describe this resource.
    fx.configure(prm_origin_resource=fx.mcp_url + "/")
    assert asyncio.run(discover_issuer(fx.mcp_url)).issuer == fx.issuer("a")


def test_issuer_discovery_refuses_a_plain_http_issuer(fx: OAuthFixture) -> None:
    from rote.cli import main as cli_main
    from rote.mcp import load_registry
    from rote.mcp.auth import discover_issuer

    fx.configure(prm_issuer="http://auth.attacker.example/as")
    found = asyncio.run(discover_issuer(fx.mcp_url))
    assert found.issuer is None
    assert "not HTTPS" in (found.problem or "")
    assert cli_main(["mcp", "add", "vendor", fx.mcp_url, "--client-id", PRE_ID]) == 0
    assert load_registry().servers["vendor"].issuer is None


def test_a_v1_client_on_a_server_without_dcr_names_the_cause(fx: OAuthFixture) -> None:
    """rote drops an unbound v1 client; when the server no longer allows
    registration the login must say that is what happened, and the fix."""
    fx.configure(cimd=False)
    config = _config(fx)
    asyncio.run(login("vendor", config))
    v1 = read_token_file("vendor")
    assert v1 is not None
    old_client = v1["client_info"]["client_id"]
    v1.pop("issuer")
    v1["client_info"].pop("issuer", None)
    v1["expires_at"] = time.time() - 10
    v1["tokens"]["refresh_token"] = "revoked"
    token_file("vendor").write_text(json.dumps({**v1, "version": 1}), encoding="utf-8")
    fx.configure(dcr=False)

    with pytest.raises(McpAuthError) as caught:
        asyncio.run(login("vendor", config))
    message = str(caught.value)
    assert old_client in message
    assert "older rote" in message
    assert "rote mcp add vendor <url> --client-id" in message
