"""OpenCode's MCP servers, read from its layered config files.

OpenCode moved from sst/opencode to anomalyco/opencode. Verified 2026-10-22
against the released source at tag v1.19.4:
https://github.com/anomalyco/opencode/blob/v1.19.4/packages/opencode/src/config/config.ts,
https://github.com/anomalyco/opencode/blob/v1.19.4/packages/opencode/src/config/paths.ts,
https://github.com/anomalyco/opencode/blob/v1.19.4/packages/opencode/src/config/variable.ts,
https://github.com/anomalyco/opencode/blob/v1.19.4/packages/opencode/src/config/managed.ts,
https://github.com/anomalyco/opencode/blob/v1.19.4/packages/core/src/v1/config/mcp.ts,
and the docs at https://opencode.ai/docs/mcp-servers and
https://opencode.ai/docs/config:

- Layers, lowest first: the global dir (``$XDG_CONFIG_HOME/opencode``, else
  ``~/.config/opencode``: ``config.json`` < ``opencode.json`` <
  ``opencode.jsonc``), ``$OPENCODE_CONFIG``, project ``opencode.json(c)``
  from the git root (the filesystem root outside git) down to the working
  directory (deeper wins, ``.jsonc`` over ``.json``; off with
  ``OPENCODE_DISABLE_PROJECT_CONFIG``), ``.opencode/`` dirs (``~/.opencode``
  and the same walk) and ``$OPENCODE_CONFIG_DIR``,
  ``$OPENCODE_CONFIG_CONTENT`` (inline JSON), and the managed dir
  (``/Library/Application Support/opencode/``, ``/etc/opencode/``,
  ``%ProgramData%\\opencode\\``).
- Layers merge deeply (remeda ``mergeDeep``), so ``mcp.<name>`` merges field
  by field and a bare ``{"enabled": false}`` overlay is valid.
- Entry: ``type: "local"`` (``command`` array, ``environment``, ``cwd``) or
  ``type: "remote"`` (``url``, ``headers``, ``oauth``: ``false`` or
  ``{clientId, clientSecret, scope, callbackPort, redirectUri}``); both take
  ``enabled`` and ``timeout``. An entry with no ``type`` is ignored.
- Substitution runs on the raw text before parsing: ``{env:VAR}`` (unset is
  empty) and ``{file:path}``.
- JSONC (jsonc-parser with trailing commas) for ``.json`` and ``.jsonc``.
- The unreleased v2 (``dev`` branch) nests servers under ``mcp.servers``
  with ``disabled`` and snake_case OAuth keys; that shape is read too.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import json5

from rote.clients.base import ClientContext, ImportCandidate, Scope, SourceReading
from rote.clients.common import (
    STDIO_BLOCKED,
    Reader,
    blocked,
    merge_entries,
    new_candidate,
    not_an_object,
    remote,
    switch_off,
    xdg_config_home,
)
from rote.clients.normalize import PlaceholderSyntax

CLIENT_ID = "opencode"

OPENCODE_SYNTAX = PlaceholderSyntax(
    client=CLIENT_ID,
    pattern=re.compile(r"\{env:(?P<var>[A-Za-z_][A-Za-z0-9_]*)\}"),
    other=re.compile(r"\{file:[^}]*\}"),
)

LIMITS = (
    "remote config fetched from a provider's /.well-known/opencode, account org config and "
    "the macOS MDM profile are not read",
    "a {env:...} outside a string (OpenCode substitutes raw text) makes the file unreadable "
    "here; it is reported as a problem",
)

_INLINE = Path("$OPENCODE_CONFIG_CONTENT")


def _walk(cwd: Path) -> list[Path]:
    """From the git root (the filesystem root outside git) down to ``cwd``."""
    chain: list[Path] = []
    for directory in (cwd, *cwd.parents):
        chain.append(directory)
        if (directory / ".git").exists():
            break
    return list(reversed(chain))


def _pair(directory: Path) -> tuple[Path, Path]:
    return directory / "opencode.json", directory / "opencode.jsonc"


def _layers(ctx: ClientContext) -> Iterator[tuple[Scope, Path]]:
    """Every file, lowest precedence first."""
    env = ctx.env
    global_dir = xdg_config_home(ctx) / "opencode"
    yield "user", global_dir / "config.json"
    yield from (("user", p) for p in _pair(global_dir))
    if env.get("OPENCODE_CONFIG", "").strip():
        yield "user", Path(env["OPENCODE_CONFIG"].strip())
    walk = _walk(ctx.cwd)
    if not env.get("OPENCODE_DISABLE_PROJECT_CONFIG", "").strip():
        for directory in walk:
            yield from (("project", p) for p in _pair(directory))
    yield from (("user", p) for p in _pair(ctx.home / ".opencode"))
    for directory in walk:
        if directory != ctx.home:
            yield from (("project", p) for p in _pair(directory / ".opencode"))
    if env.get("OPENCODE_CONFIG_DIR", "").strip():
        yield from (("user", p) for p in _pair(Path(env["OPENCODE_CONFIG_DIR"].strip())))
    if env.get("OPENCODE_CONFIG_CONTENT", "").strip():
        yield "user", _INLINE
    managed = ctx.system_dir if ctx.platform == "darwin" else ctx.etc_dir
    if managed is not None:
        yield from (("system", p) for p in _pair(managed / "opencode"))


def _servers(path: Path, data: dict[str, Any], reader: Reader) -> dict[str, Any]:
    """``mcp.<name>`` (v1), or ``mcp.servers.<name>`` (v2)."""
    mcp = reader.section(path, data, "mcp")
    servers = mcp.get("servers")
    if isinstance(servers, dict) and "type" not in servers:
        return servers
    return mcp


def _candidate(name: str, entry: Any, *, scope: Scope, origin: Path) -> ImportCandidate:
    base = new_candidate(CLIENT_ID, name, scope=scope, origin=origin)
    if not isinstance(entry, dict):
        return not_an_object(base, entry)
    kind = entry.get("type")
    if kind == "local":
        return blocked(base, "stdio", STDIO_BLOCKED)
    if kind != "remote":
        return blocked(base, "other", f"type {kind!r}: OpenCode runs only local or remote entries")
    withheld: list[str] = []
    scopes: list[str] = []
    oauth = entry.get("oauth")
    if isinstance(oauth, dict):
        if "clientId" in oauth or "client_id" in oauth:
            withheld.append(
                "oauth.clientId: issued to OpenCode; rote registers its own client at "
                f"rote mcp login {base.name}"
            )
        if "clientSecret" in oauth or "client_secret" in oauth:
            withheld.append("oauth.clientSecret: a secret issued to OpenCode, never copied")
        scope_value = oauth.get("scope")
        if isinstance(scope_value, str):
            scopes = scope_value.split()
    return remote(
        base,
        url=entry.get("url"),
        transport="streamable-http",
        syntax=OPENCODE_SYNTAX,
        headers=entry.get("headers"),
        withheld=withheld,
        scopes=scopes,
    )


class OpenCodeSource:
    id = CLIENT_ID
    label = "OpenCode"

    def config_paths(self, ctx: ClientContext) -> tuple[Path, ...]:
        return tuple(path for _, path in reversed(list(_layers(ctx))) if path != _INLINE)

    def detect(self, ctx: ClientContext) -> bool:
        return (xdg_config_home(ctx) / "opencode").is_dir() or any(
            p.is_file() for p in _pair(ctx.cwd)
        )

    def read(self, ctx: ClientContext) -> SourceReading:
        reader = Reader()
        layers: list[tuple[Scope, Path, dict[str, Any]]] = []
        for scope, path in _layers(ctx):
            data = self._inline(ctx, reader) if path == _INLINE else reader.mapping(path, "jsonc")
            layers.append((scope, path, _servers(path, data, reader)))
        candidates: list[ImportCandidate] = []
        off: dict[tuple[Path, str], str] = {}
        for m in merge_entries(layers, deep=True):
            candidates.append(_candidate(m.name, m.entry, scope=m.scope, origin=m.origin))
            if isinstance(m.entry, dict) and (
                m.entry.get("enabled") is False or m.entry.get("disabled") is True
            ):
                off[(m.origin, m.name)] = "disabled in OpenCode (enabled: false)"
        return SourceReading(
            client=CLIENT_ID,
            candidates=tuple(switch_off(candidates, off)),
            files_read=tuple(reader.files_read),
            problems=tuple(reader.problems),
            limits=LIMITS,
        )

    @staticmethod
    def _inline(ctx: ClientContext, reader: Reader) -> dict[str, Any]:
        try:
            value = json5.loads(ctx.env["OPENCODE_CONFIG_CONTENT"])
        except ValueError as e:
            reader.problems.append(f"{_INLINE}: could not read it ({type(e).__name__})")
            return {}
        if not isinstance(value, dict):
            reader.problems.append(f"{_INLINE}: expected an object, got {type(value).__name__}")
            return {}
        return value
