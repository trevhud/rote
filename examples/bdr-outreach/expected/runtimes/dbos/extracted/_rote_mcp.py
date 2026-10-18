"""MCP connection helper — emitted verbatim into generated apps.

This module is special: rote emits its *source text* into every
MCP-backed app (as ``extracted/_rote_mcp.py``), so generated code stays
standalone while sharing one tested implementation. It therefore obeys
two hard rules: stdlib-only imports at module level (fastmcp only
inside functions), and no imports from rote.

What it does, mirroring the rote CLI's behavior exactly:

- **Endpoint resolution**: ``ROTE_MCP_<SERVER>_URL`` env var > the rote
  registry (``~/.config/rote/mcp.json``) > the endpoint recorded in the
  pipeline at compilation time.
- **Auth resolution**: the rote token store
  (``~/.local/share/rote/mcp-tokens/<server>.json``, written by
  ``rote mcp login``) through a durable OAuth provider that refreshes
  in place; else static headers from the registry entry; else an
  unauthenticated client.
- **Never interactive**: emitted workflows run unattended, so this
  module must never open a browser. Anything that would require a human
  (expired token with no refresh path, a 401 from the server, an OAuth
  flow wanting authorization or a new client registration) raises
  :class:`RoteMcpAuthNeeded` instead; the emitted workflow catches it and
  parks durably until ``rote mcp login <server>`` releases it. A server
  that asks for human input in the middle of a call (a 2026-07-28
  ``input_required`` result) gets :class:`RoteMcpInputRequired`, a plain
  failure: no login can answer it, so parking on auth would wait forever.
- **Protocol era**: fastmcp 4 negotiates MCP 2026-07-28 and falls back to
  the 2025 handshake for servers that do not speak it.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, SupportsFloat, TypeVar

_E = TypeVar("_E", bound=BaseException)

#: The fastmcp range this helper is verified against. It reads MCP SDK v2
#: field names and builds an httpx2 client, so fastmcp 3.x cannot run it.
#: Emitted READMEs print it; tests pin it to pyproject's extras.
FASTMCP_REQUIREMENT = "fastmcp>=4.1.1,<5"


def _registry_servers() -> dict[str, Any]:
    path_str = os.environ.get("ROTE_MCP_CONFIG")
    if path_str:
        path = Path(path_str)
    else:
        xdg = os.environ.get("XDG_CONFIG_HOME")
        path = (Path(xdg) if xdg else Path.home() / ".config") / "rote" / "mcp.json"
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    servers = data.get("servers", {})
    return servers if isinstance(servers, dict) else {}


def _token_dir() -> Path:
    override = os.environ.get("ROTE_MCP_TOKEN_DIR")
    if override:
        return Path(override)
    xdg = os.environ.get("XDG_DATA_HOME")
    return (Path(xdg) if xdg else Path.home() / ".local" / "share") / "rote" / "mcp-tokens"


class RoteMcpAuthNeeded(RuntimeError):
    """The MCP server needs a human to (re)authenticate.

    Raised instead of ever starting an interactive OAuth flow from
    workflow code. The emitted workflow treats this as "park durably and
    wait for ``rote mcp login <server>``" — never as a retryable error,
    because no number of retries produces a credential.
    """

    def __init__(self, server: str, reason: str) -> None:
        super().__init__(
            f"MCP server {server!r} needs (re)authentication — {reason}. "
            f"Run: rote mcp login {server}"
        )
        self.server = server
        self.reason = reason

    def __reduce__(self) -> tuple[type, tuple[str, str]]:
        # Default exception pickling calls type(exc)(*args) with the one
        # formatted message — a TypeError for this two-arg __init__. DBOS
        # serializes step errors across queue workers, so this must
        # round-trip.
        return (type(self), (self.server, self.reason))


class RoteMcpInputRequired(RuntimeError):
    """The MCP server asked for human input in the middle of a call.

    A 2026-07-28 server can answer ``tools/call`` with ``input_required``
    (elicitation, sampling, or roots requests) and expect the client to
    retry with the answers. An emitted workflow step has no human, so the
    call fails here with the requests named. It is deliberately not a
    :class:`RoteMcpAuthNeeded`: a login does not answer it, so the workflow
    must fail rather than park, and retries cannot help either.
    """

    def __init__(self, server: str, tool: str | None, requests: dict[str, str]) -> None:
        asked = ", ".join(f"{key} ({method})" for key, method in sorted(requests.items()))
        target = f"tool {tool!r} on MCP server {server!r}" if tool else f"MCP server {server!r}"
        super().__init__(
            f"{target} asked for input mid-call ({asked}). An unattended workflow "
            "step has nobody to answer, so the call fails instead of waiting; "
            "pass the input as tool arguments if the server accepts them"
        )
        self.server = server
        self.tool = tool
        self.requests = requests

    def __reduce__(self) -> tuple[type, tuple[str, str | None, dict[str, str]]]:
        # Same reason as RoteMcpAuthNeeded: DBOS pickles step errors.
        return (type(self), (self.server, self.tool, self.requests))


def resolve_url(server: str, pipeline_url: str | None) -> str:
    env_url = os.environ.get(f"ROTE_MCP_{server.upper()}_URL")
    if env_url:
        return env_url
    entry = _registry_servers().get(server)
    if isinstance(entry, dict) and entry.get("url"):
        return str(entry["url"])
    if pipeline_url:
        return pipeline_url
    raise RuntimeError(
        f"no endpoint for MCP server {server!r} — register it "
        f"(rote mcp add {server} <url>) or set ROTE_MCP_{server.upper()}_URL"
    )


class _TokenKV:
    """The rote token-file contract (versions 1 and 2) as the AsyncKeyValue
    shape fastmcp's OAuth storage adapter expects. One instance per
    server; the storage collection selects the field, keys are ignored.
    Keep byte-compatible with rote.mcp.tokens.ServerTokenKV."""

    _FIELDS = {
        "mcp-oauth-token": "tokens",
        "mcp-oauth-client-info": "client_info",
        "mcp-oauth-token-expiry": "_expiry",
    }

    def __init__(self, server: str, server_url: str) -> None:
        self._server = server
        self._server_url = server_url
        self._path = _token_dir() / f"{server}.json"

    def _load(self) -> dict[str, Any]:
        if self._path.is_file():
            with self._path.open("r", encoding="utf-8") as f:
                loaded: dict[str, Any] = json.load(f)
                return loaded
        return {
            "server_url": self._server_url,
            "tokens": None,
            "expires_at": None,
            "client_info": None,
            "token_endpoint": None,
            "issuer": None,
        }

    def _store(self, doc: dict[str, Any]) -> None:
        doc["server_url"] = self._server_url
        doc.setdefault("version", 2)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self._path.parent, prefix=f".{self._server}-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(doc, f, indent=2)
                f.write("\n")
            os.chmod(tmp, 0o600)
            os.replace(tmp, self._path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

    def _field(self, collection: str | None) -> str:
        field = self._FIELDS.get(collection or "")
        if field is None:
            raise ValueError(f"unexpected OAuth storage collection: {collection!r}")
        return field

    async def get(self, key: str, *, collection: str | None = None) -> dict[str, Any] | None:
        doc = self._load()
        field = self._field(collection)
        if field == "_expiry":
            expires_at = doc.get("expires_at")
            return None if expires_at is None else {"expires_at": expires_at}
        value = doc.get(field)
        return value if isinstance(value, dict) else None

    async def put(
        self,
        key: str,
        value: Mapping[str, Any],
        *,
        collection: str | None = None,
        ttl: SupportsFloat | None = None,
    ) -> None:
        doc = self._load()
        field = self._field(collection)
        dump = getattr(value, "model_dump", None)
        payload = dump(mode="json", exclude_none=True) if dump is not None else dict(value)
        if field == "_expiry":
            doc["expires_at"] = payload.get("expires_at")
        else:
            doc[field] = payload
        self._store(doc)

    async def delete(self, key: str, *, collection: str | None = None) -> bool:
        doc = self._load()
        field = self._field(collection)
        if field == "_expiry":
            had = doc.get("expires_at") is not None
            doc["expires_at"] = None
        else:
            had = doc.get(field) is not None
            doc[field] = None
        self._store(doc)
        return had

    # Remaining AsyncKeyValue protocol surface (unused by the OAuth
    # adapter today; present so the shim is a genuine AsyncKeyValue).

    async def get_many(
        self, keys: Sequence[str], *, collection: str | None = None
    ) -> list[dict[str, Any] | None]:
        return [await self.get(k, collection=collection) for k in keys]

    async def put_many(
        self,
        keys: Sequence[str],
        values: Sequence[Mapping[str, Any]],
        *,
        collection: str | None = None,
        ttl: SupportsFloat | None = None,
    ) -> None:
        for k, v in zip(keys, values, strict=True):
            await self.put(k, v, collection=collection, ttl=float(ttl) if ttl else None)

    async def delete_many(self, keys: Sequence[str], *, collection: str | None = None) -> int:
        return sum([await self.delete(k, collection=collection) for k in keys])

    async def ttl(
        self, key: str, *, collection: str | None = None
    ) -> tuple[dict[str, Any] | None, float | None]:
        return await self.get(key, collection=collection), None

    async def ttl_many(
        self, keys: Sequence[str], *, collection: str | None = None
    ) -> list[tuple[dict[str, Any] | None, float | None]]:
        return [(await self.get(k, collection=collection), None) for k in keys]


def has_stored_login(server: str) -> bool:
    path = _token_dir() / f"{server}.json"
    if not path.is_file():
        return False
    with path.open("r", encoding="utf-8") as f:
        doc = json.load(f)
    return bool((doc.get("tokens") or {}).get("access_token"))


def token_is_usable(server: str) -> bool:
    """Fresh access token, or a stale one with a refresh token behind it."""
    path = _token_dir() / f"{server}.json"
    if not path.is_file():
        return False
    with path.open("r", encoding="utf-8") as f:
        doc = json.load(f)
    tokens = doc.get("tokens") or {}
    if not tokens.get("access_token"):
        return False
    expires_at = doc.get("expires_at")
    if expires_at is None or time.time() < float(expires_at) - 60.0:
        return True
    return bool(tokens.get("refresh_token"))


class _HttpStatusWatch:
    """Remembers whether the server's latest answer was a 401.

    MCP SDK v2 turns a non-2xx HTTP answer into a bare JSON-RPC error
    (``MCPError`` code -32603, "Server returned an error response") with
    no cause chain, so the status code never reaches the caller's
    exception (verified against fastmcp 4.1.1 / mcp 2.3.0). The only
    place a 401 is still visible is the HTTP client itself, so the
    transport is built with an httpx2 response hook that records it.

    A 401 sets the flag and any 2xx clears it, so an OAuth refresh that
    answers a 401 and then succeeds does not count, while a concurrent
    non-2xx (a 405 on the optional GET stream) leaves the flag alone.
    """

    def __init__(self) -> None:
        self.unauthorized = False

    async def _on_response(self, response: Any) -> None:
        if response.status_code == 401:
            self.unauthorized = True
        elif 200 <= response.status_code < 300:
            self.unauthorized = False

    def client_factory(
        self,
        headers: dict[str, str] | None = None,
        timeout: Any = None,
        auth: Any = None,
        **kwargs: Any,
    ) -> Any:
        """An ``McpHttpClientFactory`` (fastmcp also passes follow_redirects)."""
        import httpx2

        return httpx2.AsyncClient(
            headers=headers,
            # The SDK's own default: 30s connect/write/pool, a long read
            # because a server may hold a response stream open.
            timeout=timeout if timeout is not None else httpx2.Timeout(30.0, read=300.0),
            auth=auth,
            event_hooks={"response": [self._on_response]},
            **kwargs,
        )


def _stored_token_endpoint(server: str) -> str | None:
    path = _token_dir() / f"{server}.json"
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        endpoint = json.load(f).get("token_endpoint")
    return str(endpoint) if endpoint else None


def retarget_token_request(request: Any, token_endpoint: str) -> Any:
    """``request`` (an httpx2 token request) sent to ``token_endpoint`` instead.

    The SDK refreshes before it has discovered anything, and with no
    metadata it guesses ``<resource-origin>/token``. That is wrong for
    every authorization server that is not the MCP server's own origin
    root (a path issuer, or another host): the refresh 404s, the provider
    falls into a full authorization, and an unattended caller parks on a
    credential that was fine. ``rote mcp login`` records the real token
    endpoint in the token file; this sends the refresh there.
    """
    import httpx2

    headers = {
        k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")
    }
    return httpx2.Request(request.method, token_endpoint, headers=headers, content=request.content)


def is_registration_request(provider: Any, outgoing: Any, original: Any) -> bool:
    """Whether ``outgoing``, yielded by the SDK's auth flow, registers a client.

    The flow sends exactly one POST while it holds no client: the
    registration. Token requests need a client, and the request being
    authenticated is the original.
    """
    return (
        outgoing is not original
        and outgoing.method == "POST"
        and provider.context.client_info is None
    )


def mcp_client(
    server: str, pipeline_url: str | None, *, http_status: _HttpStatusWatch | None = None
) -> Any:
    """A ready fastmcp ``Client`` for ``server`` — durable OAuth when the
    user has run ``rote mcp login``, static registry headers when
    configured, plain client otherwise. Use as an async context manager.

    The client never asks a human for anything: a server that answers a
    call with ``input_required`` raises :class:`RoteMcpInputRequired`.

    Pass ``http_status`` to learn afterwards whether the server's last
    answer was a 401 (see :class:`_HttpStatusWatch`)."""
    from fastmcp import Client
    from fastmcp.client.transports import StreamableHttpTransport

    url = resolve_url(server, pipeline_url)
    entry = _registry_servers().get(server) or {}
    factory = http_status.client_factory if http_status is not None else None

    class _UnattendedClient(Client[Any]):
        """fastmcp's client with the multi-round-trip input driver refused.

        fastmcp hands every 2026-07-28 ``input_required`` result to a
        driver that answers it through the elicitation / sampling / roots
        callbacks. Here there is nobody to answer, so any result that asks
        for input fails the call. A result carrying only ``request_state``
        asks for nothing (the server wants a retry), and still goes through.
        """

        async def _drive_input_required(self, first: Any, retry: Any) -> Any:
            _refuse_input(first)

            async def guarded(responses: Any, state: Any) -> Any:
                result = await retry(responses, state)
                _refuse_input(result)
                return result

            return await super()._drive_input_required(first, guarded)

    def _refuse_input(result: Any) -> None:
        if getattr(result, "result_type", None) != "input_required":
            return
        requests = getattr(result, "input_requests", None) or {}
        if requests:
            raise RoteMcpInputRequired(
                server, None, {key: str(req.method) for key, req in requests.items()}
            )

    if has_stored_login(server):
        from fastmcp.client.auth import OAuth

        token_endpoint = _stored_token_endpoint(server)

        class _NonInteractiveOAuth(OAuth):
            """OAuth that refreshes silently but never opens a browser.

            Workflow code runs unattended; if the provider falls out of
            the refresh path into a full authorization flow (revoked
            grant, rotated client) or wants to register a new client,
            surface RoteMcpAuthNeeded so the workflow parks instead of
            hanging on a browser that will never open, or minting a client
            nobody asked for.
            """

            async def redirect_handler(self, authorization_url: str) -> None:
                raise RoteMcpAuthNeeded(server, "the server requires an interactive OAuth flow")

            async def _refresh_token(self) -> Any:
                request = await super()._refresh_token()
                if self.context.oauth_metadata is None and token_endpoint:
                    return retarget_token_request(request, token_endpoint)
                return request

            async def _auth_flow(self, request: Any) -> Any:
                flow = super()._auth_flow(request)
                try:
                    outgoing = await flow.__anext__()
                    while True:
                        if is_registration_request(self, outgoing, request):
                            raise RoteMcpAuthNeeded(
                                server, "the stored client is no longer valid for this server"
                            )
                        response = yield outgoing
                        outgoing = await flow.asend(response)
                except StopAsyncIteration:
                    return
                finally:
                    await flow.aclose()

        oauth = _NonInteractiveOAuth(
            mcp_url=url,
            client_name="rote",
            token_storage=_TokenKV(server, url),
            client_id=entry.get("client_id"),
            client_secret=entry.get("client_secret"),
        )
        return _UnattendedClient(
            StreamableHttpTransport(url, auth=oauth, httpx_client_factory=factory)
        )
    headers = entry.get("headers")
    return _UnattendedClient(
        StreamableHttpTransport(
            url,
            headers=dict(headers) if isinstance(headers, dict) and headers else None,
            httpx_client_factory=factory,
        )
    )


def _exception_tree(exc: BaseException) -> Iterator[BaseException]:
    """Every exception in ``exc``'s tree: groups, causes, and contexts.

    The OAuth provider and the input driver raise from inside the
    transport's task group, so an error can arrive wrapped in an
    ExceptionGroup or as the cause of another error rather than at the
    top level.
    """
    seen: set[int] = set()
    stack: list[BaseException] = [exc]
    while stack:
        e = stack.pop()
        if id(e) in seen:
            continue
        seen.add(id(e))
        yield e
        if isinstance(e, BaseExceptionGroup):
            stack.extend(e.exceptions)
        if e.__cause__ is not None:
            stack.append(e.__cause__)
        if e.__context__ is not None and not e.__suppress_context__:
            stack.append(e.__context__)


def _find_in_tree(exc: BaseException, wanted: type[_E]) -> _E | None:
    """The first ``wanted`` error in ``exc``'s tree, if any."""
    return next((e for e in _exception_tree(exc) if isinstance(e, wanted)), None)


#: JSON-RPC code a 2026-07-28 server answers with when a tool needs a client
#: capability (elicitation, sampling, roots) the request did not declare.
_MISSING_CLIENT_CAPABILITY = -32021


def _refused_capabilities(exc: BaseException) -> dict[str, str] | None:
    """The capabilities a server refused the call for, if that is the error.

    This client declares none, so a server that checks before asking
    (the TS SDK does) refuses instead of returning ``input_required``.
    Duck-typed on ``code``/``data``: this module does not import the SDK.
    """
    for e in _exception_tree(exc):
        if getattr(e, "code", None) == _MISSING_CLIENT_CAPABILITY:
            data = getattr(e, "data", None) or {}
            required = data.get("requiredCapabilities") if isinstance(data, dict) else None
            names = sorted(required) if isinstance(required, dict) and required else ["unknown"]
            return {name: "client capability" for name in names}
    return None


def _result_payload(result: Any, output_schema: Mapping[str, Any] | None = None) -> Any:
    """Extract the useful payload from a fastmcp ``CallToolResult``.

    The payload is plain JSON (dicts, lists, scalars) because a durable
    runtime pickles or serializes it at the step boundary. That rules
    out ``result.data``: for a typed output schema fastmcp hydrates it
    into generated classes (``types.Root``) that neither pickle nor
    JSON-encode, and hydration renames keys that are not identifiers.

    In order:

    - ``structured_content``, the wire value. fastmcp servers wrap a
      non-object return as ``{"result": ...}`` and mark it (the
      ``x-fastmcp-wrap-result`` schema flag or ``_meta.fastmcp.wrap_result``);
      that wrapper is removed so a list-returning tool yields the list.
    - text content. Many real-world servers (the TS SDK in particular)
      declare no output schema and return JSON as text; without this
      every such call would silently yield ``None``. Text that isn't
      JSON comes back verbatim, and the workflow step decides what it
      means.
    """
    structured = result.structured_content
    if structured is not None:
        fastmcp_meta = (result.meta or {}).get("fastmcp")
        wrapped = (isinstance(fastmcp_meta, dict) and fastmcp_meta.get("wrap_result")) or bool(
            output_schema and output_schema.get("x-fastmcp-wrap-result")
        )
        return structured.get("result") if wrapped else structured
    # Duck-typed on `.text` so image/audio blocks are skipped without
    # importing mcp.types (this module ships verbatim in emitted apps).
    texts = [block.text for block in result.content if getattr(block, "text", None) is not None]

    def _parse(text: str) -> Any:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    if not texts:
        return None
    if len(texts) == 1:
        return _parse(texts[0])
    return [_parse(text) for text in texts]


async def call_mcp_tool(
    server: str, pipeline_url: str | None, tool: str, arguments: dict[str, Any]
) -> Any:
    """Call one MCP tool and return its structured result data.

    The entry point emitted workflow steps use. Auth problems become
    :class:`RoteMcpAuthNeeded` — raised *before* touching the network
    when the stored token is known-dead (expired with no refresh token),
    and mapped from an HTTP 401 when the server rejects whatever
    credentials we did present. A server asking for input mid-call
    becomes :class:`RoteMcpInputRequired`, naming the tool.
    """
    if has_stored_login(server) and not token_is_usable(server):
        raise RoteMcpAuthNeeded(
            server, "the stored access token has expired and there is no refresh token"
        )
    http_status = _HttpStatusWatch()
    try:
        async with mcp_client(server, pipeline_url, http_status=http_status) as client:
            # The listing is not an extra round trip. The MCP SDK's
            # ClientSession.call_tool validates every successful result
            # against the tool's output schema and lists tools to learn it
            # whenever the session has not, which on a fresh client is
            # every call, text-only results included (measured: exactly one
            # tools/list per step either way; pinned by
            # tests/test_mcp_ts_server_e2e.py). Listing first reuses that
            # cache, gives _result_payload the wrap flag, and moves a
            # listing failure ahead of the call instead of after the tool
            # has already run its side effects.
            schemas = {t.name: t.output_schema for t in await client.list_tools()}
            result = await client.call_tool(tool, arguments)
            return _result_payload(result, schemas.get(tool))
    except Exception as exc:
        # Exception, not BaseException: a cancellation or interrupt that
        # lands after a 401 must cancel the step, not park it. (A
        # BaseExceptionGroup only arises when one of its members is a
        # BaseException, so it falls through here too.)
        input_required = _find_in_tree(exc, RoteMcpInputRequired)
        if input_required is not None:
            raise RoteMcpInputRequired(server, tool, input_required.requests) from exc
        refused = _refused_capabilities(exc)
        if refused is not None:
            raise RoteMcpInputRequired(server, tool, refused) from exc
        auth_needed = _find_in_tree(exc, RoteMcpAuthNeeded)
        if auth_needed is exc:
            raise
        if auth_needed is not None:
            raise auth_needed from exc
        if http_status.unauthorized:
            raise RoteMcpAuthNeeded(server, "the server returned 401 Unauthorized") from exc
        raise
