"""What rote has learned about each MCP server's auth requirement.

A sidecar next to the registry (``mcp-auth-learned.json``), not a field
in it. Released rote versions load the registry with ``extra="forbid"``,
so a field written on every login would break every older rote on the
machine the first time someone logged in. The registry schema stays
what the user wrote; this file holds what rote observed.

One entry per server name, carrying the normalized endpoint URL it was
learned for. Pointing the name at a new endpoint invalidates the entry
without anyone having to clear it: a finding about one server is not a
finding about another.

The format is read tolerantly, in both directions: unknown fields are
kept on write, and an ``auth`` value this version does not recognize
(a future rote's addition) reads as "not learned" instead of failing.

Written atomically with mode 0600, like the registry and token store.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from rote.mcp._runtime_helper import normalize_server_url
from rote.mcp.registry import load_registry, registry_path

#: The format version this rote writes. A file stamped higher by a newer
#: rote keeps its number: writers take the max.
LEARNED_VERSION = 1

LearnedValue = Literal["oauth", "none"]
_KNOWN: tuple[LearnedValue, ...] = ("oauth", "none")


@dataclass(frozen=True)
class LearnedAuth:
    """A recorded auth requirement for the server's current endpoint."""

    auth: LearnedValue
    learned_at: str | None
    #: "login" (observed by `rote mcp login`), "park" (a run parked on
    #: auth), or "user" (`rote mcp add --auth`). Unknown values pass through.
    source: str | None


def learned_path() -> Path:
    """The sidecar beside the registry, so one override moves both."""
    return registry_path().with_name("mcp-auth-learned.json")


def _load() -> dict[str, Any]:
    path = learned_path()
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object")
    return data


def _same_endpoint(stored: object, url: str) -> bool:
    if not isinstance(stored, str):
        return False
    try:
        return normalize_server_url(stored) == normalize_server_url(url)
    except ValueError:
        return False


def learned_auth(server: str, url: str) -> LearnedAuth | None:
    """What rote learned about ``server`` at ``url``, or None if nothing usable."""
    servers = _load().get("servers")
    entry = servers.get(server) if isinstance(servers, dict) else None
    if not isinstance(entry, dict) or not _same_endpoint(entry.get("url"), url):
        return None
    auth = entry.get("auth")
    if auth not in _KNOWN:
        return None  # a value from a newer rote: unknown here, not an error
    learned_at = entry.get("learned_at")
    source = entry.get("source")
    return LearnedAuth(
        auth=auth,
        learned_at=learned_at if isinstance(learned_at, str) else None,
        source=source if isinstance(source, str) else None,
    )


def record_auth(server: str, url: str, auth: LearnedValue, *, source: str) -> None:
    """Record ``auth`` for ``server`` at ``url``, dated today (UTC).

    No write when the entry already says the same for the same endpoint.
    Fields this version does not know, in the file or the entry, are kept.
    """
    data = _load()
    servers = data.get("servers")
    if not isinstance(servers, dict):
        servers = {}
    found = servers.get(server)
    current: dict[str, Any] = (
        found if isinstance(found, dict) and _same_endpoint(found.get("url"), url) else {}
    )
    if current.get("auth") == auth and current.get("source") == source:
        return
    entry = dict(current)
    entry.update(
        url=normalize_server_url(url),
        auth=auth,
        learned_at=datetime.now(UTC).date().isoformat(),
        source=source,
    )
    servers[server] = entry
    data["servers"] = servers
    version = data.get("version")
    data["version"] = max(version if isinstance(version, int) else 0, LEARNED_VERSION)
    _write(data)


def _write(data: dict[str, Any]) -> None:
    path = learned_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".mcp-auth-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        os.chmod(tmp_name, 0o600)
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def note_auth_park(server: str) -> None:
    """A run parked on ``server`` for auth, so it needs auth now.

    The evidence that retires a stale ``none``: a server that moved from
    open to OAuth would otherwise stay "no auth required" in emit's
    advisory and doctor while every run parked on it.
    """
    config = load_registry().servers.get(server)
    if config is None:
        return
    learned = learned_auth(server, config.url)
    if learned is not None and learned.auth == "none":
        record_auth(server, config.url, "oauth", source="park")


def note_auth_park_safely(server: str) -> str | None:
    """:func:`note_auth_park`, returning a warning instead of raising.

    Callers have already done the part that matters (announced or
    released the park). A corrupt sidecar or registry must not undo that.
    """
    try:
        note_auth_park(server)
    except Exception as exc:  # noqa: BLE001 — bookkeeping only, see above
        return f"could not record that MCP server {server!r} needs auth: {exc}"
    return None
