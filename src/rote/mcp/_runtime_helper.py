"""MCP connection helper — emitted verbatim into generated apps.

This module is special: rote emits its *source text* into every
MCP-backed app (as ``extracted/_rote_mcp.py``), so generated code stays
standalone while sharing one tested implementation. It therefore obeys
two hard rules: stdlib-only imports at module level (fastmcp only
inside functions), and no imports from rote.

What it does, mirroring the rote CLI's behavior exactly:

- **Endpoint resolution**: ``ROTE_MCP_<SERVER>_URL`` env var > the rote
  registry (``~/.config/rote/mcp.json``) > the endpoint recorded in the
  pipeline at graduation time.
- **Auth resolution**: the rote token store
  (``~/.local/share/rote/mcp-tokens/<server>.json``, written by
  ``rote mcp login``) through a durable OAuth provider that refreshes
  in place; else static headers from the registry entry; else an
  unauthenticated client.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, SupportsFloat


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
    """The rote token-file contract (version 1) as the AsyncKeyValue
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
        }

    def _store(self, doc: dict[str, Any]) -> None:
        doc["server_url"] = self._server_url
        doc.setdefault("version", 1)
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


def mcp_client(server: str, pipeline_url: str | None) -> Any:
    """A ready fastmcp ``Client`` for ``server`` — durable OAuth when the
    user has run ``rote mcp login``, static registry headers when
    configured, plain client otherwise. Use as an async context manager."""
    from fastmcp import Client

    url = resolve_url(server, pipeline_url)
    entry = _registry_servers().get(server) or {}

    if has_stored_login(server):
        from fastmcp.client.auth import OAuth

        return Client(
            url,
            auth=OAuth(
                mcp_url=url,
                client_name="rote",
                token_storage=_TokenKV(server, url),
                client_id=entry.get("client_id"),
                client_secret=entry.get("client_secret"),
            ),
        )
    headers = entry.get("headers")
    if isinstance(headers, dict) and headers:
        from fastmcp.client.transports import StreamableHttpTransport

        return Client(StreamableHttpTransport(url, headers=headers))
    return Client(url)
