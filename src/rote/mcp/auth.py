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

fastmcp is a base dependency, imported lazily so a command that never
touches MCP does not pay for the import; a broken install that lacks it
fails with :func:`rote.extras.dependency_hint`.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import secrets
import socket
import sys
import webbrowser
from collections.abc import AsyncGenerator, Awaitable, Callable
from importlib import resources
from typing import Any, Literal, NamedTuple

from rote.extras import dependency_hint
from rote.mcp._runtime_helper import (
    is_registration_request,
    normalize_server_url,
    retarget_token_request,
    stored_for_url,
)
from rote.mcp.learned import record_auth
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

#: Receives the authorization URL instead of a browser or a terminal: the
#: detached listener of ``rote mcp login --no-wait`` hands it to the
#: command that started it (rote.mcp.pending_login).
Handoff = Callable[[str], Awaitable[None]]


def authorization_state(authorization_url: str) -> str | None:
    """The ``state`` the SDK put in ``authorization_url``, or None."""
    from urllib.parse import parse_qs, urlsplit

    values = parse_qs(urlsplit(authorization_url).query).get("state")
    return values[0] if values else None


class _StateCheckedReady:
    """fastmcp's callback "result ready" event, accepting only the right ``state``.

    fastmcp's callback server keeps the first request it gets, and the SDK
    then compares its ``state``. A request to the loopback port with any
    other state, which anything on this machine can send, would end the
    login, and the SDK's mismatch error quotes the expected state. Here a
    success callback with the wrong state is dropped before it counts: the
    login keeps waiting for the real browser, and the state is never
    compared anywhere it could be printed. Error callbacks still end the
    login (fastmcp does not keep their state), which costs a retry, never
    a credential.
    """

    def __init__(
        self, result: Any, expected_state: str | None, on_rejected: Callable[[], None] | None
    ) -> None:
        import anyio

        self._event = anyio.Event()
        self._result = result
        self._expected = expected_state
        self._on_rejected = on_rejected

    def is_set(self) -> bool:
        return self._event.is_set()

    def set(self) -> None:
        result = self._result
        if result.error is None and self._expected is not None:
            state = result.state
            if not isinstance(state, str) or not secrets.compare_digest(state, self._expected):
                result.code = result.state = result.iss = None
                if self._on_rejected is not None:
                    self._on_rejected()
                return
        self._event.set()

    async def wait(self) -> None:
        await self._event.wait()


class McpAuthError(RuntimeError):
    """User-actionable auth failure (missing extra, failed dance, no token)."""


def _require_fastmcp() -> Any:
    try:
        import fastmcp
    except ImportError as e:
        raise McpAuthError(
            f"MCP auth needs the fastmcp client: {dependency_hint('fastmcp')}"
        ) from e
    return fastmcp


def client_metadata_url() -> str | None:
    """The CIMD URL logins offer, or None when CIMD is turned off."""
    value = os.environ.get(CLIENT_METADATA_URL_ENV_VAR)
    if value is None:
        return DEFAULT_CLIENT_METADATA_URL
    return value or None


#: How long the one fetch that checks the published document may take,
#: in total. httpx's own timeout is per read, so a host that sends a byte
#: every few seconds would hold the login open forever without this.
#: Only authorization servers that advertise CIMD trigger the fetch.
_DOCUMENT_CHECK_TIMEOUT_SECONDS = 5.0

#: Per-process verdicts of :func:`_published_document_problem`, by URL.
_document_checks: dict[str, str | None] = {}


def packaged_client_metadata() -> dict[str, Any]:
    """The client metadata document this CLI expects to find published.

    Shipped in the wheel (``rote/mcp/client_metadata.json``) so an installed
    CLI can compare without the repo; ``docs/oauth/rote-cli.json`` is the
    same file, enforced by a test.
    """
    text = resources.files("rote.mcp").joinpath("client_metadata.json").read_text("utf-8")
    doc: dict[str, Any] = json.loads(text)
    return doc


async def _published_document_problem(url: str) -> str | None:
    """Why the document at ``url`` cannot be offered as a client id, or None.

    A CIMD authorization server fetches the document and rejects the login
    when it is missing or lists other redirect URIs, and the SDK does not
    fall back to DCR when that happens. So before offering it, rote fetches
    it once and requires exactly what this CLI expects: the packaged
    document, with ``client_id`` equal to ``url`` (CIMD draft section 3).
    """
    if url in _document_checks:
        return _document_checks[url]
    import httpx

    expected = {**packaged_client_metadata(), "client_id": url}
    problem: str | None
    try:
        async with (
            asyncio.timeout(_DOCUMENT_CHECK_TIMEOUT_SECONDS),
            httpx.AsyncClient(timeout=_DOCUMENT_CHECK_TIMEOUT_SECONDS) as client,
        ):
            response = await client.get(url, follow_redirects=False)
        if response.status_code != 200:
            problem = f"HTTP {response.status_code}"
        elif response.json() != expected:
            problem = "it differs from the document this version of rote expects"
        else:
            problem = None
    except TimeoutError:
        problem = f"no complete answer within {_DOCUMENT_CHECK_TIMEOUT_SECONDS:g}s"
    except (httpx.HTTPError, ValueError) as e:
        problem = f"{type(e).__name__}: {e}"
    _document_checks[url] = problem
    return problem


async def _advertises_cimd(response: Any) -> bool:
    """Whether ``response`` is authorization server metadata offering CIMD."""
    if response.status_code != 200:
        return False
    try:
        # Read now; the SDK's own aread() afterwards gets the cached body.
        body = json.loads(await response.aread())
    except ValueError:
        return False
    return (
        isinstance(body, dict)
        and "issuer" in body
        and body.get("client_id_metadata_document_supported") is True
    )


def callback_redirect_uris() -> list[str]:
    """Every redirect URI the CIMD document must list, in document order."""
    return [
        f"http://{host}:{port}/callback"
        for host in (CALLBACK_HOST, "127.0.0.1")
        for port in CALLBACK_PORTS
    ]


def bind_callback_sockets(port: int) -> list[socket.socket] | None:
    """Listening sockets for ``port`` on every address localhost resolves to.

    None when any of them is taken. Binding *is* the check: checking first
    and binding later (as fastmcp's callback server would) lets another
    login, or anything else, take the port in between, and the callback
    server then dies with EADDRINUSE after the browser was sent there.
    The sockets stay bound until the callback server takes them over.

    SO_REUSEADDR matches what uvicorn sets: a TIME_WAIT left by an earlier
    login on this port must not block the next, while a live listener
    still does. Every resolved address is bound because a browser may
    reach ``localhost`` on either ::1 or 127.0.0.1.
    """
    sockets: list[socket.socket] = []
    try:
        seen: set[tuple[int, Any]] = set()
        for family, _, _, _, address in socket.getaddrinfo(
            CALLBACK_HOST, port, type=socket.SOCK_STREAM
        ):
            if (family, address) in seen:
                continue
            seen.add((family, address))
            sock = socket.socket(family, socket.SOCK_STREAM)
            sockets.append(sock)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if family == socket.AF_INET6:
                sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            sock.bind(address)
            sock.listen(128)
    except OSError:
        for sock in sockets:
            sock.close()
        return None
    return sockets


def select_callback_port() -> tuple[int, list[socket.socket]]:
    """The first free port of :data:`CALLBACK_PORTS`, already bound."""
    for port in CALLBACK_PORTS:
        sockets = bind_callback_sockets(port)
        if sockets is not None:
            return port, sockets
    ports = ", ".join(str(p) for p in CALLBACK_PORTS)
    raise McpAuthError(
        f"every OAuth callback port rote publishes is in use ({ports}). "
        "Free one, or pass --callback-port for a server that accepts any "
        "loopback port (that login cannot use rote's client metadata document)"
    )


def is_trusted_transport(url: str) -> bool:
    """HTTPS, or plain HTTP to a loopback host (local development)."""
    from urllib.parse import urlsplit

    parts = urlsplit(url)
    if parts.scheme == "https":
        return True
    if parts.scheme != "http" or not parts.hostname:
        return False
    if parts.hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(parts.hostname).is_loopback
    except ValueError:
        return False


class IssuerDiscovery(NamedTuple):
    """What :func:`discover_issuer` found: an issuer, or why there is none."""

    issuer: str | None
    problem: str | None


async def discover_issuer(url: str) -> IssuerDiscovery:
    """The authorization server ``url``'s protected-resource metadata names.

    RFC 9728 discovery: the path-specific well-known document, then the
    origin's. The first ``authorization_servers`` entry is an issuer
    identifier, the same one the SDK will expect at login.

    What gets recorded here decides which server a client secret is later
    offered to, so nothing unauthenticated or ambiguous is accepted: the
    resource and the issuer must be HTTPS (or loopback), and the document
    must describe this resource (RFC 9728 section 3.3: ``resource`` equal to
    the URL). The origin-level document on a shared host describes some
    other resource, and its issuer is not this one's.
    """
    from urllib.parse import urlsplit

    import httpx

    if not is_trusted_transport(url):
        return IssuerDiscovery(None, f"{url} is not HTTPS, so its metadata is not trusted")
    parts = urlsplit(url)
    origin = f"{parts.scheme}://{parts.netloc}"
    path = parts.path.rstrip("/")
    candidates = [f"{origin}/.well-known/oauth-protected-resource{path}"] if path else []
    candidates.append(f"{origin}/.well-known/oauth-protected-resource")
    problem = "the server publishes no protected-resource metadata"
    try:
        async with (
            asyncio.timeout(_DOCUMENT_CHECK_TIMEOUT_SECONDS * len(candidates)),
            httpx.AsyncClient(timeout=_DOCUMENT_CHECK_TIMEOUT_SECONDS) as client,
        ):
            for candidate in candidates:
                try:
                    response = await client.get(candidate)
                    body = response.json() if response.status_code == 200 else None
                except (httpx.HTTPError, ValueError):
                    continue
                if not isinstance(body, dict):
                    continue
                resource = body.get("resource")
                try:
                    same = isinstance(resource, str) and normalize_server_url(
                        resource
                    ) == normalize_server_url(url)
                except ValueError:
                    same = False
                if not same:
                    problem = f"the metadata at {candidate} describes {resource!r}, not {url}"
                    continue
                servers = body.get("authorization_servers")
                if not (isinstance(servers, list) and servers and isinstance(servers[0], str)):
                    problem = f"the metadata at {candidate} names no authorization server"
                    continue
                if not is_trusted_transport(servers[0]):
                    return IssuerDiscovery(
                        None, f"the named authorization server {servers[0]} is not HTTPS"
                    )
                return IssuerDiscovery(servers[0], None)
    except TimeoutError:
        problem = "the server's metadata did not arrive in time"
    return IssuerDiscovery(None, problem)


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
            drop_unbound_client: bool,
            callback_sockets: list[socket.socket],
            handoff: Handoff | None,
            on_rejected_callback: Callable[[], None] | None,
            **kwargs: Any,
        ) -> None:
            self._rote_server = server
            self._rote_handoff = handoff
            self._rote_on_rejected_callback = on_rejected_callback
            self._rote_expected_state: str | None = None
            self._rote_interactive = interactive
            self._rote_no_browser = no_browser
            self._rote_announce_url = announce_url
            self._rote_stored_issuer = stored_issuer
            self._rote_token_endpoint = token_endpoint
            self._rote_drop_unbound_client = drop_unbound_client
            self._rote_callback_sockets = callback_sockets
            self._rote_dropped_client_id: str | None = None
            # The scopes the user set (`rote mcp add --scope`), if any, and
            # the scopes the server's last 401 challenge demanded.
            self._rote_configured_scopes: list[str] = list(kwargs.get("scopes") or [])
            self._rote_challenged_scopes: list[str] = []
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
            # A registered client with no issuer at all (a login from before
            # the token file recorded one) is bound to nothing, and the SDK
            # would offer it, secret included, to whichever authorization
            # server the resource names next. An interactive login drops it
            # and registers again, which binds the new client.
            current = self.context.client_info
            if (
                self._rote_drop_unbound_client
                and static is None
                and current is not None
                and current.issuer is None
                and current.client_id != self.context.client_metadata_url
            ):
                self._rote_dropped_client_id = current.client_id
                self.context.client_info = None

        async def redirect_handler(self, authorization_url: str) -> None:
            if not self._rote_interactive:
                raise McpAuthError(
                    f"the stored credentials for {self._rote_server!r} no longer work "
                    f"and the server wants a new authorization. Re-authenticate with: "
                    f"rote mcp login {self._rote_server}"
                )
            self._rote_expected_state = authorization_state(authorization_url)
            if self._rote_handoff is not None:
                await self._preflight(authorization_url)
                await self._rote_handoff(authorization_url)
                return
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

        def close_callback_sockets(self) -> None:
            """Release the callback port, whether or not a callback ran."""
            for sock in self._rote_callback_sockets:
                sock.close()
            self._rote_callback_sockets = []

        async def callback_handler(self) -> Any:
            """fastmcp's callback server, run on the sockets bound at selection.

            The redirect URI already names ``redirect_port``, so the server must
            listen exactly there. fastmcp's own handler binds the port itself,
            which can fail after the browser was already sent to it.
            """
            import functools

            import anyio
            from fastmcp.client.oauth_callback import (
                OAuthCallbackResult,
                create_oauth_callback_server,
            )
            from mcp.shared.auth import AuthorizationCodeResult

            sockets = self._rote_callback_sockets or bind_callback_sockets(self.redirect_port)
            self._rote_callback_sockets = []
            if not sockets:
                # Only on a second flow in one login (fastmcp retries once
                # with a fresh registration): the first released the port.
                raise McpAuthError(
                    f"the OAuth callback port {self.redirect_port} was taken during the "
                    f"login. Retry: rote mcp login {self._rote_server}"
                )
            result = OAuthCallbackResult()
            ready = _StateCheckedReady(
                result, self._rote_expected_state, self._rote_on_rejected_callback
            )
            server = create_oauth_callback_server(
                port=self.redirect_port,
                host=self._callback_host,
                server_url=self.mcp_url,
                result_container=result,
                result_ready=ready,  # type: ignore[arg-type]  # duck-typed anyio.Event
            )
            try:
                async with anyio.create_task_group() as tg:
                    tg.start_soon(functools.partial(server.serve, sockets=sockets))
                    try:
                        with anyio.fail_after(self._callback_timeout):
                            await ready.wait()
                        if result.error:
                            raise result.error
                        return AuthorizationCodeResult(
                            code=result.code or "", state=result.state, iss=result.iss
                        )
                    except TimeoutError as e:
                        raise TimeoutError(
                            f"OAuth callback timed out after {self._callback_timeout} seconds"
                        ) from e
                    finally:
                        server.should_exit = True
                        await anyio.sleep(0.1)  # let the server finish its response
                        tg.cancel_scope.cancel()
            finally:
                for sock in sockets:
                    sock.close()

        async def _preflight(self, authorization_url: str) -> None:
            # fastmcp's own pre-flight GET, kept: it is how a stale cached
            # client gets detected and re-registered.
            from fastmcp.client.auth.oauth import ClientNotFoundError

            async with self.httpx_client_factory() as client:
                response = await client.get(authorization_url, follow_redirects=False)
                if response.status_code == 400:
                    raise ClientNotFoundError(
                        "OAuth client not found - cached credentials may be stale"
                    )
                if response.status_code not in (200, 302, 303, 307, 308):
                    raise RuntimeError(f"Unexpected authorization response: {response.status_code}")

        async def _announce(self, authorization_url: str) -> None:
            # The URL always goes to stderr: fastmcp logs it only at info
            # level and ignores webbrowser.open's result, so on SSH the user
            # would wait on a callback for a URL they never saw.
            await self._preflight(authorization_url)
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
            self._hold_configured_scopes()
            return await super()._perform_authorization()

        def _hold_configured_scopes(self) -> None:
            """Request the scopes the user set, never the server's wider list.

            On a 401 the SDK replaces the requested scopes with the
            challenge's, else everything the metadata lists as supported, and
            never consults the configured ones (mcp 2.3.0, oauth2.py). A real
            login configured with six read-only scopes asked for 26, Gmail
            send among them (#316). A list the server only *advertises* is not
            a requirement, so the configured scopes win. A challenge that
            *requires* more stops the login naming the extra scopes: widening
            the user's choice is theirs to make. ``offline_access``, which the
            SDK adds when the server supports it, stays: it asks for a refresh
            token and grants no access, and unattended runs need one.
            """
            configured = self._rote_configured_scopes
            if not configured:
                return
            extra = [s for s in self._rote_challenged_scopes if s not in configured]
            if extra:
                raise McpAuthError(
                    f"{self._rote_server!r} requires scope(s) {' '.join(extra)} beyond the "
                    f"ones configured for it ({' '.join(configured)}), so rote did not open "
                    "the consent page. To grant them, add them to the server's scopes "
                    f"(rote mcp add {self._rote_server} <url> --scope ...); to request "
                    "whatever the server asks, remove its scopes"
                )
            selected = (self.context.client_metadata.scope or "").split()
            kept = ["offline_access"] if "offline_access" in selected else []
            requested = list(dict.fromkeys([*configured, *kept]))
            self.context.client_metadata.scope = " ".join(requested)

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
                fix = (
                    "Register an OAuth app with it (redirect URI "
                    f"http://{CALLBACK_HOST}:{CALLBACK_PORTS[0]}/callback), then: "
                    f"rote mcp add {server} <url> --client-id <id> "
                    "[--client-secret <secret>]"
                )
                if self._rote_dropped_client_id is not None:
                    # The login had a client, and rote set it aside: say so,
                    # or the user sees a server that "worked yesterday" refuse.
                    raise McpAuthError(
                        f"the stored client for {server!r} "
                        f"({self._rote_dropped_client_id}) was saved by an older rote "
                        "without the authorization server it belongs to, so rote will "
                        "not offer it to any server, and the authorization server "
                        f"({metadata.issuer}) no longer accepts new client "
                        f"registrations. {fix}"
                    )
                raise McpAuthError(
                    f"the authorization server for {server!r} ({metadata.issuer}) "
                    "offers neither a client metadata document rote can use nor "
                    f"dynamic client registration. {fix}"
                )

        async def _check_offered_document(self, response: Any) -> None:
            # The SDK picks CIMD inside the same step that reads this
            # response, so this is the last moment to withdraw the URL.
            # Checking here, not up front, keeps logins to servers that do
            # not offer CIMD free of the extra fetch and the notice.
            url = self.context.client_metadata_url
            if url is None or not await _advertises_cimd(response):
                return
            problem = await _published_document_problem(url)
            if problem is not None:
                print(
                    f"note: rote's client metadata document {url} is not usable "
                    f"({problem}); registering a client with this server instead",
                    file=sys.stderr,
                )
                self.context.client_metadata_url = None

        async def _auth_flow(self, request: Any) -> AsyncGenerator[Any, Any]:
            flow = super()._auth_flow(request)
            try:
                outgoing = await flow.__anext__()
                while True:
                    self._refuse_registration(outgoing, request)
                    response = yield outgoing
                    if response is not None and response.status_code in (401, 403):
                        from mcp.client.auth.utils import extract_scope_from_www_auth

                        demanded = extract_scope_from_www_auth(response)
                        self._rote_challenged_scopes = (demanded or "").split()
                    if outgoing is not request:
                        await self._check_offered_document(response)
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
    handoff: Handoff | None = None,
    callback_timeout: float | None = None,
    on_rejected_callback: Callable[[], None] | None = None,
) -> Any:
    """A fastmcp ``OAuth`` bound to rote's token store for this server.

    ``interactive=False`` builds a refresh-only provider: anything that
    would need a browser or a new registration raises :class:`McpAuthError`.
    ``announce_url`` always prints the authorization URL to stderr and says
    so when no browser could be opened. ``handoff`` receives the URL
    instead, and nothing is printed or opened. ``callback_timeout`` is how
    long the callback server waits for the browser (fastmcp's default is
    five minutes). ``on_rejected_callback`` runs for each callback dropped
    for a wrong ``state``.
    """
    _require_fastmcp()
    metadata_url = client_metadata_url()
    callback_sockets: list[socket.socket] = []
    if not interactive:
        # A refresh never takes a callback; the port is only a name.
        callback_port = callback_port or CALLBACK_PORTS[0]
    elif callback_port is None:
        callback_port, callback_sockets = select_callback_port()
    else:
        bound = bind_callback_sockets(callback_port)
        if bound is None:
            raise McpAuthError(f"the OAuth callback port {callback_port} is in use")
        callback_sockets = bound
    if callback_port not in CALLBACK_PORTS and metadata_url is not None:
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
    # login used, else the one recorded at `rote mcp add`. A client_id that
    # changed since its last login (a new app registered with `rote mcp
    # add`) takes the recorded one. A file for another endpoint is ignored.
    stored = read_token_file(server) or {}
    if not stored_for_url(stored, config.url):
        stored = {}
    stored_client = (stored.get("client_info") or {}).get("client_id")
    stored_issuer = (
        stored.get("issuer") if stored_client == config.client_id else None
    ) or config.issuer
    return _oauth_class()(
        server=server,
        interactive=interactive,
        no_browser=no_browser,
        announce_url=announce_url,
        stored_issuer=stored_issuer,
        token_endpoint=stored.get("token_endpoint"),
        drop_unbound_client=interactive,
        callback_sockets=callback_sockets,
        handoff=handoff,
        on_rejected_callback=on_rejected_callback,
        **({} if callback_timeout is None else {"callback_timeout": callback_timeout}),
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
    handoff: Handoff | None = None,
    callback_timeout: float | None = None,
    on_rejected_callback: Callable[[], None] | None = None,
) -> dict[str, Any] | None:
    """Run the full OAuth dance and persist tokens durably.

    ``handoff``, ``callback_timeout`` and ``on_rejected_callback`` are
    :func:`build_oauth`'s; ``rote mcp login --no-wait`` runs this same
    function in its detached listener.

    Returns the stored contract document, or None when the server
    answered ``tools/list`` without a 401 (the provider discovered
    nothing, see :func:`_saw_auth_challenge`), so it needs no
    credentials. Either way the finding is recorded in the registry
    (``auth: oauth`` or ``auth: none``), which is what lets ``rote emit``
    stop warning that a no-auth server will park. After a real flow it
    also stamps ``token_endpoint`` and ``issuer`` (from the authorization
    server metadata the provider validated) into the token file:
    refresh-only clients, the emitted TypeScript runtimes, never re-run
    discovery, and the issuer is what the credentials are bound to.
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
        handoff=handoff,
        callback_timeout=callback_timeout,
        on_rejected_callback=on_rejected_callback,
    )
    try:
        await _list_tools(server, config.url, oauth)
    finally:
        # Released on every path: a login whose stored token was still good
        # never starts the callback server, and a failed one may not either.
        oauth.close_callback_sockets()

    doc = read_token_file(server)
    if doc is not None and not stored_for_url(doc, config.url):
        # A login for the endpoint this name used to point at: the flow
        # just now neither used nor replaced it, so it is no login here.
        doc = None
    if doc is None or not (doc.get("tokens") or {}).get("access_token"):
        if not _saw_auth_challenge(oauth):
            record_auth(server, config.url, "none", source="login")
            return None
        # The server challenged (discovery ran) yet nothing usable landed
        # in the store: an interrupted flow, or tokens written somewhere
        # this store does not read. Never call that "no auth required".
        raise McpAuthError(
            f"OAuth flow for {server!r} completed without storing a token; "
            "the flow may have been interrupted. Retry: rote mcp login " + server
        )
    record_auth(server, config.url, "oauth", source="login")

    metadata = getattr(oauth.context, "oauth_metadata", None)
    if metadata is None:
        # The stored token was still good: no flow ran, nothing to stamp.
        return doc
    issuer = str(metadata.issuer)
    before = json.dumps(doc, sort_keys=True)
    doc["token_endpoint"] = str(metadata.token_endpoint)
    doc["issuer"] = issuer
    # Also on the client itself: that is the only place the SDK reads a
    # binding from, and it records one only when it registers.
    client_info = doc.get("client_info")
    if isinstance(client_info, dict) and not client_info.get("issuer"):
        client_info["issuer"] = issuer
    if json.dumps(doc, sort_keys=True) != before:
        write_token_file(server, doc)
    return doc


def _saw_auth_challenge(oauth: Any) -> bool:
    """Whether the OAuth provider ran discovery, i.e. the server sent a 401.

    The SDK's provider discovers metadata only after a challenge, so with
    nothing discovered the request went through unauthenticated. If the
    provider's shape is not the one this reads, assume a challenge: the
    caller then reports a failed login instead of recording ``none``.
    """
    context = getattr(oauth, "context", None)
    if context is None or not hasattr(context, "protected_resource_metadata"):
        return True
    return any(
        getattr(context, attr, None) is not None
        for attr in ("protected_resource_metadata", "oauth_metadata", "auth_server_url")
    )


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
    if not stored_for_url(doc, config.url):
        raise McpAuthError(
            f"the stored login for {server!r} is for {doc.get('server_url')}, but the "
            f"server is now registered at {config.url}. Log in to the new endpoint: "
            f"rote mcp login {server}"
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
