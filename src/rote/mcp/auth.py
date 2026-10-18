"""OAuth login/refresh against a registered MCP server.

The engine is fastmcp's ``OAuth`` (itself a subclass of the official
MCP SDK v2's ``OAuthClientProvider``, which implements the 2026-07-28
authorization spec: protected-resource discovery, PKCE, RFC 9207 ``iss``
validation, issuer-bound credentials, CIMD and DCR). rote supplies what
the engine leaves open:

- **Registration order.** The spec's priority is a pre-registered client,
  then a Client ID Metadata Document when the authorization server
  advertises ``client_id_metadata_document_supported``, then dynamic
  client registration, then asking the user. The registry's ``client_id``
  is the first, :data:`DEFAULT_CLIENT_METADATA_URL` the second, the SDK's
  DCR the third, and :class:`McpAuthError` naming ``rote mcp add
  --client-id`` the fourth. The SDK on its own would guess a
  ``<resource-origin>/register`` URL for a server that advertises no
  registration endpoint (GitHub's shape); rote refuses that guess.
- **A fixed callback port set.** A CIMD authorization server compares the
  redirect URI against the published document by exact string match, so
  the CLI binds the first free port of :data:`CALLBACK_PORTS`, the same
  set the document lists.
- **Durable storage** (:class:`rote.mcp.tokens.ServerTokenKV`), with the
  authorization server's ``issuer`` stamped beside the tokens.
- A no-browser mode for SSH boxes.

fastmcp is an optional dependency (the `mcp` extra, see `rote.extras`), so
everything here imports it lazily and fails with that hint.
"""

from __future__ import annotations

import os
import socket
import sys
import webbrowser
from collections.abc import AsyncGenerator
from typing import Any, Literal

from rote.extras import install_hint
from rote.mcp._runtime_helper import is_registration_request, retarget_token_request
from rote.mcp.registry import McpServerConfig
from rote.mcp.tokens import (
    ServerTokenKV,
    access_token_state,
    read_token_file,
    write_token_file,
)

#: rote's Client ID Metadata Document. The URL is the client_id a CIMD
#: authorization server sees; the document's source is docs/oauth/rote-cli.json.
DEFAULT_CLIENT_METADATA_URL = "https://roteskills.com/oauth/rote-cli.json"

#: Overrides :data:`DEFAULT_CLIENT_METADATA_URL`. An empty value turns CIMD
#: off, so every login without a pre-registered client goes through DCR.
CLIENT_METADATA_URL_ENV_VAR = "ROTE_MCP_CLIENT_METADATA_URL"

#: The loopback ports the CIMD document lists. A CIMD authorization server
#: matches redirect URIs by exact string comparison (CIMD draft section 4.2),
#: and the spec defines no loopback port wildcard, so a random port would
#: be rejected by any server that is not lenient about it.
CALLBACK_PORTS: tuple[int, ...] = (33418, 33419, 33420, 33421)

#: The host the callback server binds and the redirect URI names. fastmcp's
#: default, kept so DCR-only servers that work today see the same redirect.
CALLBACK_HOST = "localhost"

Registration = Literal["pre-registered", "cimd", "dcr"]


class McpAuthError(RuntimeError):
    """User-actionable auth failure (missing extra, failed dance, no token)."""


def _require_fastmcp() -> Any:
    try:
        import fastmcp
    except ImportError as e:
        raise McpAuthError(
            f"MCP auth requires the fastmcp client — install it with: {install_hint('mcp')}"
        ) from e
    return fastmcp


def client_metadata_url() -> str | None:
    """The CIMD URL logins offer, or None when CIMD is turned off."""
    value = os.environ.get(CLIENT_METADATA_URL_ENV_VAR)
    if value is None:
        return DEFAULT_CLIENT_METADATA_URL
    return value or None


def callback_redirect_uris() -> list[str]:
    """Every redirect URI the CIMD document must list, in document order."""
    return [
        f"http://{host}:{port}/callback"
        for host in (CALLBACK_HOST, "127.0.0.1")
        for port in CALLBACK_PORTS
    ]


def _port_is_free(port: int) -> bool:
    # Bind every address the callback host resolves to: uvicorn binds
    # "localhost" the same way, and a port free on 127.0.0.1 but taken on
    # ::1 would fail there instead of here.
    for family, _, _, _, address in socket.getaddrinfo(
        CALLBACK_HOST, port, type=socket.SOCK_STREAM
    ):
        with socket.socket(family, socket.SOCK_STREAM) as s:
            try:
                s.bind(address)
            except OSError:
                return False
    return True


def select_callback_port() -> int:
    """The first free port of :data:`CALLBACK_PORTS`."""
    for port in CALLBACK_PORTS:
        if _port_is_free(port):
            return port
    ports = ", ".join(str(p) for p in CALLBACK_PORTS)
    raise McpAuthError(
        f"every OAuth callback port rote publishes is in use ({ports}). "
        "Free one, or pass --callback-port for a server that accepts any "
        "loopback port (that login cannot use rote's client metadata document)"
    )


def _oauth_class() -> type[Any]:
    """rote's fastmcp ``OAuth`` subclass, built lazily (fastmcp is optional)."""
    from fastmcp.client.auth import OAuth

    class RoteOAuth(OAuth):
        """fastmcp's OAuth with rote's registration rules.

        The SDK's flow is an async generator of HTTP requests; this class
        wraps it and inspects each request before it is sent, which is the
        only point where a registration can be stopped before it happens.
        """

        def __init__(
            self,
            *,
            server: str,
            interactive: bool,
            no_browser: bool,
            announce_url: bool,
            stored_issuer: str | None,
            token_endpoint: str | None,
            **kwargs: Any,
        ) -> None:
            self._rote_server = server
            self._rote_interactive = interactive
            self._rote_no_browser = no_browser
            self._rote_announce_url = announce_url
            self._rote_stored_issuer = stored_issuer
            self._rote_token_endpoint = token_endpoint
            super().__init__(**kwargs)

        async def _refresh_token(self) -> Any:
            # Same fix as the emitted helper: refresh where login found the
            # token endpoint, not at the SDK's resource-origin guess.
            request = await super()._refresh_token()
            if self.context.oauth_metadata is None and self._rote_token_endpoint:
                return retarget_token_request(request, self._rote_token_endpoint)
            return request

        async def _initialize(self) -> None:
            await super()._initialize()
            # fastmcp rebuilds a pre-registered client without an issuer on
            # every load. Binding it to the issuer of its last login lets
            # the SDK notice when the server moves to another authorization
            # server; _pre_registered_client_moved turns that into an error
            # instead of a CIMD or DCR client at the new server (spec: SHOULD
            # surface an error rather than use mismatched credentials).
            static = self._static_client_info
            if static is not None and self._rote_stored_issuer:
                self.context.client_info = static.model_copy(
                    update={"issuer": self._rote_stored_issuer}
                )

        async def redirect_handler(self, authorization_url: str) -> None:
            if not self._rote_interactive:
                raise McpAuthError(
                    f"the stored credentials for {self._rote_server!r} no longer work "
                    f"and the server wants a new authorization. Re-authenticate with: "
                    f"rote mcp login {self._rote_server}"
                )
            if self._rote_no_browser:
                print(
                    f"\nOpen this URL in any browser to authorize {self._rote_server!r}:\n\n"
                    f"  {authorization_url}\n\n"
                    "Waiting for the callback on this machine "
                    "(port-forward it if you're on SSH)…",
                    file=sys.stderr,
                )
                return
            if self._rote_announce_url:
                await self._announce(authorization_url)
                return
            await super().redirect_handler(authorization_url)

        async def _announce(self, authorization_url: str) -> None:
            # fastmcp's own pre-flight GET, kept: it is how a stale cached
            # client gets detected and re-registered. Then the URL always goes
            # to stderr: fastmcp logs it only at info level and ignores
            # webbrowser.open's result, so on SSH the user would wait on a
            # callback for a URL they never saw.
            from fastmcp.client.auth.oauth import ClientNotFoundError

            async with self.httpx_client_factory() as client:
                response = await client.get(authorization_url, follow_redirects=False)
                if response.status_code == 400:
                    raise ClientNotFoundError(
                        "OAuth client not found - cached credentials may be stale"
                    )
                if response.status_code not in (200, 302, 303, 307, 308):
                    raise RuntimeError(f"Unexpected authorization response: {response.status_code}")
            print(
                f"\nAuthorize {self._rote_server!r} in your browser:\n\n  {authorization_url}\n",
                file=sys.stderr,
                flush=True,
            )
            if not webbrowser.open(authorization_url):
                print(
                    "No browser could be opened here. Open the URL above in any "
                    "browser; if you're on SSH, port-forward the callback port.",
                    file=sys.stderr,
                    flush=True,
                )

        def _pre_registered_client_moved(self) -> McpAuthError:
            server = self._rote_server
            metadata = self.context.oauth_metadata
            issuer = metadata.issuer if metadata is not None else "an unknown issuer"
            return McpAuthError(
                f"the pre-registered client for {server!r} was used with "
                f"{self._rote_stored_issuer}, but the server now names "
                f"{issuer} as its authorization server. Register a client "
                f"there and update it with: rote mcp add {server} <url> --client-id <id>"
            )

        async def _perform_authorization(self) -> Any:
            # Runs before the browser opens. When the SDK dropped an
            # issuer-bound pre-registered client it picks CIMD or DCR on its
            # own; the configured client must not be replaced silently.
            static = self._static_client_info
            current = self.context.client_info
            if static is not None and (current is None or current.client_id != static.client_id):
                raise self._pre_registered_client_moved()
            return await super()._perform_authorization()

        def _refuse_registration(self, outgoing: Any, original: Any) -> None:
            if not is_registration_request(self, outgoing, original):
                return
            server = self._rote_server
            metadata = self.context.oauth_metadata
            if self._static_client_info is not None:
                raise self._pre_registered_client_moved()
            if not self._rote_interactive:
                raise McpAuthError(
                    f"{server!r} wants a new client registration. "
                    f"Re-authenticate with: rote mcp login {server}"
                )
            if metadata is not None and metadata.registration_endpoint is None:
                raise McpAuthError(
                    f"the authorization server for {server!r} ({metadata.issuer}) "
                    "offers neither a client metadata document rote can use nor "
                    "dynamic client registration. Register an OAuth app with it "
                    "(redirect URI "
                    f"http://{CALLBACK_HOST}:{CALLBACK_PORTS[0]}/callback), then: "
                    f"rote mcp add {server} <url> --client-id <id> "
                    "[--client-secret <secret>]"
                )

        async def _auth_flow(self, request: Any) -> AsyncGenerator[Any, Any]:
            flow = super()._auth_flow(request)
            try:
                outgoing = await flow.__anext__()
                while True:
                    self._refuse_registration(outgoing, request)
                    response = yield outgoing
                    outgoing = await flow.asend(response)
            except StopAsyncIteration:
                return
            finally:
                await flow.aclose()

    return RoteOAuth


def build_oauth(
    server: str,
    config: McpServerConfig,
    *,
    interactive: bool = True,
    no_browser: bool = False,
    callback_port: int | None = None,
    announce_url: bool = False,
) -> Any:
    """A fastmcp ``OAuth`` bound to rote's token store for this server.

    ``interactive=False`` builds a refresh-only provider: anything that
    would need a browser or a new registration raises :class:`McpAuthError`.
    ``announce_url`` always prints the authorization URL to stderr and says
    so when no browser could be opened.
    """
    _require_fastmcp()
    metadata_url = client_metadata_url()
    if callback_port is None:
        callback_port = select_callback_port() if interactive else CALLBACK_PORTS[0]
    elif callback_port not in CALLBACK_PORTS and metadata_url is not None:
        # The document lists only CALLBACK_PORTS, so a CIMD server would
        # reject this redirect URI. Offering CIMD would fail the login on
        # exactly the servers that support it.
        print(
            f"note: callback port {callback_port} is not one rote's client metadata "
            "document lists, so this login will not offer it (DCR or a "
            "pre-registered client only)",
            file=sys.stderr,
        )
        metadata_url = None

    # The issuer a pre-registered client is bound to: the one its last
    # login used. A client_id that changed since (the user registered a
    # new app with `rote mcp add`) starts unbound.
    stored = read_token_file(server) or {}
    stored_client = (stored.get("client_info") or {}).get("client_id")
    stored_issuer = stored.get("issuer") if stored_client == config.client_id else None
    return _oauth_class()(
        server=server,
        interactive=interactive,
        no_browser=no_browser,
        announce_url=announce_url,
        stored_issuer=stored_issuer,
        token_endpoint=stored.get("token_endpoint"),
        mcp_url=config.url,
        scopes=config.scopes,
        client_name="rote",
        token_storage=ServerTokenKV(server, config.url),
        callback_port=callback_port,
        callback_host=CALLBACK_HOST,
        client_metadata_url=metadata_url,
        client_id=config.client_id,
        client_secret=config.client_secret,
    )


def _rote_error_in(exc: BaseException) -> BaseException | None:
    """The McpAuthError or SDK OAuth error inside an exception tree.

    Errors raised inside the auth flow reach the caller wrapped in the
    transport's exception groups and the client's own errors.
    """
    from mcp.client.auth import OAuthFlowError, OAuthRegistrationError, OAuthTokenError

    wanted = (McpAuthError, OAuthFlowError, OAuthRegistrationError, OAuthTokenError)
    seen: set[int] = set()
    stack: list[BaseException] = [exc]
    found: BaseException | None = None
    while stack:
        e = stack.pop()
        if id(e) in seen:
            continue
        seen.add(id(e))
        if isinstance(e, McpAuthError):
            return e
        if found is None and isinstance(e, wanted):
            found = e
        if isinstance(e, BaseExceptionGroup):
            stack.extend(e.exceptions)
        if e.__cause__ is not None:
            stack.append(e.__cause__)
        if e.__context__ is not None and not e.__suppress_context__:
            stack.append(e.__context__)
    return found


async def _list_tools(server: str, url: str, oauth: Any) -> None:
    """One authenticated request, which runs whatever auth the server needs.

    Not ping: the sessionless 2026-07-28 era does not route it (MCPError
    "Method not found"), and fastmcp 4 negotiates that era by default.
    tools/list exists on every era.
    """
    fastmcp = _require_fastmcp()
    try:
        async with fastmcp.Client(url, auth=oauth) as client:
            await client.list_tools()
    except Exception as exc:
        cause = _rote_error_in(exc)
        if isinstance(cause, McpAuthError):
            raise cause from exc
        if cause is not None:
            raise McpAuthError(f"OAuth for {server!r} failed: {cause}") from exc
        raise


def registration_of(doc: dict[str, Any], config: McpServerConfig) -> Registration:
    """How the stored client was obtained, for the login summary."""
    client_id = (doc.get("client_info") or {}).get("client_id")
    if config.client_id and client_id == config.client_id:
        return "pre-registered"
    metadata_url = client_metadata_url()
    if metadata_url is not None and client_id == metadata_url:
        return "cimd"
    return "dcr"


async def login(
    server: str,
    config: McpServerConfig,
    *,
    no_browser: bool = False,
    callback_port: int | None = None,
    announce_url: bool = False,
) -> dict[str, Any]:
    """Run the full OAuth dance and persist tokens durably.

    Returns the stored contract document. Also stamps ``token_endpoint``
    and ``issuer`` (from the authorization server metadata the provider
    validated) into the token file: refresh-only clients, the emitted
    TypeScript runtimes, never re-run discovery, and the issuer is what
    the credentials are bound to.
    """
    if config.headers and not config.client_id:
        # Static-header servers don't need a dance; there is nothing to
        # store and `rote mcp headers` serves them straight from config.
        raise McpAuthError(
            f"server {server!r} is configured with static headers — no OAuth login needed"
        )

    oauth = build_oauth(
        server,
        config,
        no_browser=no_browser,
        callback_port=callback_port,
        announce_url=announce_url,
    )
    await _list_tools(server, config.url, oauth)

    doc = read_token_file(server)
    if doc is None or not (doc.get("tokens") or {}).get("access_token"):
        raise McpAuthError(
            f"OAuth flow for {server!r} completed without storing a token — "
            "the server may not require auth, or the flow was interrupted"
        )

    metadata = getattr(oauth.context, "oauth_metadata", None)
    stamped = {
        "token_endpoint": str(metadata.token_endpoint) if metadata is not None else None,
        "issuer": str(metadata.issuer) if metadata is not None else None,
    }
    stamped = {k: v for k, v in stamped.items() if v is not None}
    if any(doc.get(k) != v for k, v in stamped.items()):
        doc.update(stamped)
        write_token_file(server, doc)
    return doc


async def fresh_access_token(server: str, config: McpServerConfig) -> str:
    """A currently-valid access token for ``server``, refreshing if stale.

    The refresh path is the provider's own (refresh grant against the
    stored ``refresh_token``), triggered by making a cheap authenticated
    request. It never opens a browser or registers a client: this backs
    ``rote mcp headers``, which tools call unattended. Raises
    :class:`McpAuthError` with the login hint when there are no stored
    credentials or the refresh fails.
    """
    doc = read_token_file(server)
    if doc is None:
        raise McpAuthError(
            f"no stored credentials for MCP server {server!r} — run: rote mcp login {server}"
        )
    token, fresh = access_token_state(doc)
    if token and fresh:
        return token

    # Stale (or unknown-expiry-missing) — let the provider refresh
    # through the same durable store, then re-read.
    await _list_tools(server, config.url, build_oauth(server, config, interactive=False))
    doc = read_token_file(server)
    token, fresh = access_token_state(doc or {})
    if not token or not fresh:
        raise McpAuthError(
            f"could not refresh the access token for {server!r} — "
            f"re-authenticate with: rote mcp login {server}"
        )
    return token
