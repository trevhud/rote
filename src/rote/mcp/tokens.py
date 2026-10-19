"""The rote token store — one JSON file per logical MCP server.

This file layout is a **cross-language contract**, not an
implementation detail: the Python CLI writes it (via fastmcp's OAuth
flow), Python runtimes read and refresh through it, and the emitted
TypeScript runtimes (DBOS-TS, Inngest) implement the same refresh-grant
against the same file. Change it only with a version bump and readers
updated in lockstep.

Layout — ``<token_dir>/<server-name>.json``, mode ``0600``::

    {
      "version": 2,
      "server_url": "https://mcp.example.com/mcp",
      "tokens": {              # OAuthToken (MCP SDK shape), verbatim
        "access_token": "...",
        "token_type": "Bearer",
        "expires_in": 3600,
        "refresh_token": "...",
        "scope": "..."
      },
      "expires_at": 1770000000.0,   # absolute epoch; null = unknown
      "client_info": { ... },       # registered client (DCR result or
                                    # pre-registered), incl. client_id
      "token_endpoint": "https://auth.example.com/token", # for non-SDK
                                    # refreshers (the TS runtimes); may
                                    # be null until first login
      "issuer": "https://auth.example.com"  # v2: the authorization
                                    # server the credentials belong to
    }

Version 2 added ``issuer``. Readers accept both versions: a v1 file is a
v2 file whose issuer is unknown. Writers stamp the current version.

``tokens``/``client_info``/``expires_at`` are written by the fastmcp
OAuth provider through :class:`ServerTokenKV` (an ``AsyncKeyValue``
shim: the provider addresses three collections keyed by server URL; a
shim instance is already bound to one server file, so the collection
picks the field and the key is ignored). ``token_endpoint`` and
``issuer`` are stamped by ``rote mcp login`` after a successful dance,
from the authorization server metadata the SDK validated: refresh-only
clients never need to re-run discovery, and the issuer records which
authorization server the stored client and tokens are bound to (the
2026-07-28 spec keys credentials by issuer).

The token dir is ``$ROTE_MCP_TOKEN_DIR`` >
``$XDG_DATA_HOME/rote/mcp-tokens`` > ``~/.local/share/rote/mcp-tokens``.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, SupportsFloat

from rote.mcp._runtime_helper import _with_bound_issuer, stored_for_url

if TYPE_CHECKING:
    from rote.mcp.registry import McpServerConfig

TOKEN_DIR_ENV_VAR = "ROTE_MCP_TOKEN_DIR"

#: The token-file contract version writers stamp. Readers accept 1 and 2.
TOKEN_CONTRACT_VERSION = 2

_COLLECTION_FIELDS = {
    "mcp-oauth-token": "tokens",
    "mcp-oauth-client-info": "client_info",
    "mcp-oauth-token-expiry": "_expiry",  # {"expires_at": epoch} — flattened below
}


def token_dir() -> Path:
    override = os.environ.get(TOKEN_DIR_ENV_VAR)
    if override:
        return Path(override)
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg) if xdg else Path.home() / ".local" / "share"
    return base / "rote" / "mcp-tokens"


def token_file(server: str, directory: Path | None = None) -> Path:
    # Server names are IR-validated identifiers ([A-Za-z_][A-Za-z0-9_]*),
    # so they are safe as file stems by construction.
    return (directory or token_dir()) / f"{server}.json"


def _jsonable(value: Any) -> Any:
    """Pydantic models (the SDK's OAuthToken etc.) → plain JSON data."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, Mapping):
        return {k: _jsonable(v) for k, v in value.items()}
    return value


def read_token_file(server: str, directory: Path | None = None) -> dict[str, Any] | None:
    path = token_file(server, directory)
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
    return data


def write_token_file(server: str, data: dict[str, Any], directory: Path | None = None) -> Path:
    """Atomic 0600 write of the full contract document."""
    path = token_file(server, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{server}-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            # A newer rote's higher contract version is kept, not overwritten.
            stamped = data.get("version")
            version = max(stamped if isinstance(stamped, int) else 0, TOKEN_CONTRACT_VERSION)
            json.dump({**data, "version": version}, f, indent=2)
            f.write("\n")
        os.chmod(tmp_name, 0o600)
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise
    return path


def clear_token_file(server: str, directory: Path | None = None) -> bool:
    path = token_file(server, directory)
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False


def access_token_state(doc: dict[str, Any]) -> tuple[str | None, bool]:
    """(access_token, is_fresh) from a contract document.

    ``is_fresh`` is False when the absolute expiry has passed (with a
    60s safety margin) or when there is no token at all. A stale token
    with a ``refresh_token`` present is refreshable — that's the
    caller's job (the OAuth provider in Python, the refresh-grant
    client in TS).
    """
    tokens = doc.get("tokens") or {}
    access = tokens.get("access_token")
    if not access:
        return None, False
    expires_at = doc.get("expires_at")
    if expires_at is None:
        return access, True  # no expiry recorded — assume long-lived
    return access, time.time() < float(expires_at) - 60.0


AuthStatus = Literal[
    "static headers",
    "no auth required",
    "auth unknown",
    "not authenticated",
    "authenticated",
    "expired (refreshable)",
    "expired",
]
"""The states a registered server's credentials can be in.

- ``static headers`` — the server carries API-key-style ``headers`` and
  never runs the OAuth dance.
- ``no auth required`` — rote learned ``none`` for this endpoint
  (:mod:`rote.mcp.learned`) and holds no token for it: runs connect
  unauthenticated and never park on this server.
- ``auth unknown`` — no token for this endpoint, and nothing learned
  about whether it needs one. Runs park on it only if it does.
- ``not authenticated`` — the server needs OAuth (recorded, or a token
  file exists) but there is no usable token.
- ``authenticated`` — a fresh (unexpired) access token.
- ``expired (refreshable)`` — the access token is stale but a
  ``refresh_token`` is on hand, so a refresh grant can recover it.
- ``expired`` — a stale access token with no way to refresh; a fresh
  ``rote mcp login`` is required.
"""


def auth_status(
    server_config: McpServerConfig,
    token_doc: dict[str, Any] | None,
    learned: Literal["oauth", "none"] | None = None,
) -> AuthStatus:
    """Classify a server's credentials into one of the :data:`AuthStatus` states.

    The single source of truth for auth-state classification, shared by
    ``rote mcp list`` and ``rote doctor``. Combines the server's static
    ``headers`` (API-key schemes never touch OAuth), the presence of a
    token file, :func:`access_token_state`'s freshness verdict, and
    whether a ``refresh_token`` is available to recover a stale token.
    """
    if server_config.headers:
        return "static headers"
    # A login for the endpoint this name used to point at is not one for
    # the endpoint it points at now: that file counts as no file, except
    # that it shows this name needed OAuth before.
    stale = token_doc is not None and not stored_for_url(token_doc, server_config.url)
    if token_doc is None or stale:
        if learned == "none":
            return "no auth required"
        if learned == "oauth" or stale:
            return "not authenticated"
        return "auth unknown"
    # A token file for this endpoint is evidence of OAuth whatever was
    # learned: a stale `none` must not hide a login the server now needs.
    token, fresh = access_token_state(token_doc)
    if token and fresh:
        return "authenticated"
    if token and (token_doc.get("tokens") or {}).get("refresh_token"):
        return "expired (refreshable)"
    if token:
        return "expired"
    return "not authenticated"


class ServerTokenKV:
    """``AsyncKeyValue`` shim binding fastmcp's OAuth storage to one
    server's contract file.

    The fastmcp ``TokenStorageAdapter`` addresses three collections
    with URL-derived keys; an instance of this class is already bound
    to a single server, so the collection selects the field and the
    key is deliberately ignored. Only the methods the adapter calls
    are implemented (``get``/``put``/``delete``) — this is a structural
    protocol, not an inheritance hierarchy.
    """

    def __init__(self, server: str, server_url: str, directory: Path | None = None) -> None:
        self._server = server
        self._server_url = server_url
        self._directory = directory

    def _load(self) -> dict[str, Any]:
        doc = read_token_file(self._server, self._directory)
        # A file for another endpoint is not this endpoint's login: reading
        # it would send that endpoint's bearer token here.
        if doc is not None and stored_for_url(doc, self._server_url):
            return doc
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
        write_token_file(self._server, doc, self._directory)

    @staticmethod
    def _field(collection: str | None) -> str:
        if collection not in _COLLECTION_FIELDS:
            raise ValueError(f"unexpected OAuth storage collection: {collection!r}")
        return _COLLECTION_FIELDS[collection]

    async def get(self, key: str, *, collection: str | None = None) -> dict[str, Any] | None:
        doc = self._load()
        field = self._field(collection)
        if field == "_expiry":
            expires_at = doc.get("expires_at")
            return None if expires_at is None else {"expires_at": expires_at}
        value = doc.get(field)
        if field == "client_info":
            value = _with_bound_issuer(doc, value)
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
        payload = _jsonable(value)
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

    # The remaining AsyncKeyValue protocol surface — the fastmcp adapter
    # never calls these today, but implementing them keeps the shim a
    # genuine AsyncKeyValue (mypy-checked) rather than a lookalike.

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
