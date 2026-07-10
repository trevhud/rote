"""OAuth login/refresh against a registered MCP server.

The engine is fastmcp's ``OAuth`` (itself a subclass of the official
MCP SDK's ``OAuthClientProvider``, so the full 2025-11-25 authorization
spec: protected-resource discovery, PKCE, dynamic client registration /
CIMD, token refresh). rote supplies the three things the engine leaves
open: durable storage (:class:`rote.mcp.tokens.ServerTokenKV`),
pre-registered client credentials for DCR-less servers (from the
registry), and a no-browser mode for SSH boxes.

fastmcp is an optional dependency (`pip install 'rote-cli[mcp]'`) —
everything here imports it lazily and fails with that hint.
"""

from __future__ import annotations

import sys
from typing import Any

from rote.mcp.registry import McpServerConfig
from rote.mcp.tokens import (
    ServerTokenKV,
    access_token_state,
    read_token_file,
    write_token_file,
)


class McpAuthError(RuntimeError):
    """User-actionable auth failure (missing extra, failed dance, no token)."""


def _require_fastmcp() -> Any:
    try:
        import fastmcp
    except ImportError as e:
        raise McpAuthError(
            "MCP auth requires the fastmcp client — install it with: pip install 'rote-cli[mcp]'"
        ) from e
    return fastmcp


def build_oauth(
    server: str,
    config: McpServerConfig,
    *,
    no_browser: bool = False,
    callback_port: int | None = None,
) -> Any:
    """A fastmcp ``OAuth`` bound to rote's token store for this server."""
    _require_fastmcp()
    from fastmcp.client.auth import OAuth

    storage = ServerTokenKV(server, config.url)

    oauth_cls: type[Any] = OAuth
    if no_browser:

        class _PrintUrlOAuth(OAuth):
            async def redirect_handler(self, authorization_url: str) -> None:
                print(
                    f"\nOpen this URL in any browser to authorize {server!r}:\n\n"
                    f"  {authorization_url}\n\n"
                    "Waiting for the callback on this machine "
                    "(port-forward it if you're on SSH)…",
                    file=sys.stderr,
                )

        oauth_cls = _PrintUrlOAuth

    return oauth_cls(
        mcp_url=config.url,
        scopes=config.scopes,
        client_name="rote",
        token_storage=storage,
        callback_port=callback_port,
        client_id=config.client_id,
        client_secret=config.client_secret,
    )


async def login(
    server: str,
    config: McpServerConfig,
    *,
    no_browser: bool = False,
    callback_port: int | None = None,
) -> dict[str, Any]:
    """Run the full OAuth dance and persist tokens durably.

    Returns the stored contract document. Also best-effort stamps
    ``token_endpoint`` (from the provider's discovered AS metadata)
    into the token file so refresh-only clients — the emitted
    TypeScript runtimes — never re-run discovery.
    """
    fastmcp = _require_fastmcp()

    if config.headers and not config.client_id:
        # Static-header servers don't need a dance; there is nothing to
        # store and `rote mcp headers` serves them straight from config.
        raise McpAuthError(
            f"server {server!r} is configured with static headers — no OAuth login needed"
        )

    oauth = build_oauth(server, config, no_browser=no_browser, callback_port=callback_port)
    async with fastmcp.Client(config.url, auth=oauth) as client:
        await client.ping()  # triggers 401 → discovery → dance → tokens

    doc = read_token_file(server)
    if doc is None or not (doc.get("tokens") or {}).get("access_token"):
        raise McpAuthError(
            f"OAuth flow for {server!r} completed without storing a token — "
            "the server may not require auth, or the flow was interrupted"
        )

    endpoint = _discovered_token_endpoint(oauth)
    if endpoint and doc.get("token_endpoint") != endpoint:
        doc["token_endpoint"] = endpoint
        write_token_file(server, doc)
    return doc


def _discovered_token_endpoint(oauth: Any) -> str | None:
    """The AS token endpoint the provider discovered, if reachable.

    Reaches into the SDK provider's context — best-effort by design;
    a None simply means TS refresh clients fall back to discovery.
    """
    context = getattr(oauth, "context", None)
    metadata = getattr(context, "oauth_metadata", None)
    endpoint = getattr(metadata, "token_endpoint", None)
    return str(endpoint) if endpoint else None


async def fresh_access_token(server: str, config: McpServerConfig) -> str:
    """A currently-valid access token for ``server``, refreshing if stale.

    The refresh path is the provider's own (refresh grant against the
    stored ``refresh_token``), triggered by making a cheap authenticated
    request. Raises :class:`McpAuthError` with the login hint when
    there are no stored credentials or the refresh fails.
    """
    fastmcp = _require_fastmcp()

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
    oauth = build_oauth(server, config)
    async with fastmcp.Client(config.url, auth=oauth) as client:
        await client.ping()
    doc = read_token_file(server)
    token, fresh = access_token_state(doc or {})
    if not token or not fresh:
        raise McpAuthError(
            f"could not refresh the access token for {server!r} — "
            f"re-authenticate with: rote mcp login {server}"
        )
    return token
