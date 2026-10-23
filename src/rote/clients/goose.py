"""Goose's MCP servers (its "extensions"), read from ``config.yaml``.

Goose moved from block/goose to aaif-goose/goose. Verified 2026-10-22
against the source on main:
https://github.com/aaif-goose/goose/blob/main/crates/goose/src/config/paths.rs,
https://github.com/aaif-goose/goose/blob/main/crates/goose/src/config/base.rs,
https://github.com/aaif-goose/goose/blob/main/crates/goose/src/config/extensions.rs,
https://github.com/aaif-goose/goose/blob/main/crates/goose/src/agents/extension.rs,
and https://github.com/aaif-goose/goose/blob/main/documentation/docs/guides/config-files.md:

- **user**: ``$GOOSE_PATH_ROOT/config/config.yaml`` when set; else
  ``$XDG_CONFIG_HOME/goose/config.yaml`` (default ``~/.config/goose/``) on
  macOS and Linux, ``%APPDATA%\\Block\\goose\\config\\config.yaml`` on Windows.
- **system**: ``/etc/goose/config.yaml`` (``%PROGRAMDATA%\\goose\\`` on
  Windows), then each file in ``GOOSE_ADDITIONAL_CONFIG_FILES``, then the
  user file; later wins, and ``extensions`` merge per entry and then per
  field (``merge_config_values``).
- ``extensions`` maps a key to ``{enabled, type, ...}``. Types: ``stdio``
  (``cmd``, ``args``, ``envs``, ``env_keys``), ``builtin``, ``platform``,
  ``streamable_http`` (``uri``, ``headers``, ``envs``, ``env_keys``,
  ``socket``, ``client_id``, ``client_secret_key``, ``scopes``). ``sse`` is
  no longer a type: Goose warns "SSE is unsupported, migrate to
  streamable_http" and does not load it.
- ``${VAR}`` and ``$VAR`` in ``uri`` and header values are filled only from
  the entry's ``envs`` (inline values) and ``env_keys`` (names whose values
  come from the process environment, else Goose's keyring or
  ``secrets.yaml``); any other reference is sent as written.
- ``enabled`` is required; a disabled extension stays in the file.
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import Any

from rote.clients.base import ClientContext, HeaderTemplate, ImportCandidate, Scope, SourceReading
from rote.clients.common import (
    STDIO_BLOCKED,
    Reader,
    appdata,
    blocked,
    merge_entries,
    new_candidate,
    not_an_object,
    remote,
    str_list,
    xdg_config_home,
)
from rote.clients.normalize import PlaceholderSyntax

CLIENT_ID = "goose"

#: Goose's substitution: ``${VAR}`` or ``$VAR``, no defaults.
GOOSE_SYNTAX = PlaceholderSyntax(
    client=CLIENT_ID,
    pattern=re.compile(r"\$\{(?P<var>[A-Za-z_][A-Za-z0-9_]*)\}|\$(?P<bare>[A-Za-z_][A-Za-z0-9_]*)"),
)

LIMITS = (
    "values Goose keeps in its keyring or secrets.yaml (env_keys) are never read: a header "
    "that needs one is copied as a ${VAR} rote fills from its own environment",
    "environment variables that override config.yaml keys, recipes' own extension lists and "
    "plugin .mcp.json servers (local commands) are not read",
)


def _user_file(ctx: ClientContext) -> Path:
    root = ctx.env.get("GOOSE_PATH_ROOT", "").strip()
    if root:
        return Path(root) / "config" / "config.yaml"
    if ctx.platform == "win32":
        return appdata(ctx) / "Block" / "goose" / "config" / "config.yaml"
    return xdg_config_home(ctx) / "goose" / "config.yaml"


def _layers(ctx: ClientContext) -> list[tuple[Scope, Path]]:
    """Lowest first, as Goose merges them."""
    layers: list[tuple[Scope, Path]] = []
    if ctx.etc_dir is not None:
        layers.append(("system", ctx.etc_dir / "goose" / "config.yaml"))
    separator = ";" if ctx.platform == "win32" else ":"
    for extra in ctx.env.get("GOOSE_ADDITIONAL_CONFIG_FILES", "").split(separator):
        if extra.strip():
            layers.append(("system", Path(extra.strip())))
    layers.append(("user", _user_file(ctx)))
    return layers


def _check_references(
    candidate: ImportCandidate, envs: dict[str, Any], env_keys: list[str]
) -> ImportCandidate:
    """Keep a header only when Goose itself would fill its references."""
    kept: list[HeaderTemplate] = []
    notes = list(candidate.withheld)
    for header in candidate.headers:
        unknown = [v for v in header.env_vars if v not in envs and v not in env_keys]
        if unknown:
            notes.append(
                f"header {header.name}: Goose fills only envs and env_keys, so "
                f"{', '.join(unknown)} is sent as written; set it yourself: rote mcp add "
                f"{candidate.name} <url> --header '{header.name}: <value>'"
            )
            continue
        kept.append(header)
        for var in header.env_vars:
            where = "an inline envs value" if var in envs else "env_keys (maybe Goose's keyring)"
            notes.append(
                f"header {header.name}: Goose reads {var} from {where}; set {var} in the "
                "environment rote runs in"
            )
    return replace(candidate, headers=tuple(kept), withheld=tuple(notes))


def _candidate(key: str, entry: Any, *, scope: Scope, origin: Path) -> ImportCandidate:
    base = new_candidate(CLIENT_ID, key, scope=scope, origin=origin)
    if not isinstance(entry, dict):
        return not_an_object(base, entry)
    kind = entry.get("type")
    if kind == "stdio":
        return blocked(base, "stdio", STDIO_BLOCKED)
    if kind in ("builtin", "platform"):
        return blocked(base, "other", f"a {kind} extension that runs inside Goose")
    if kind not in ("streamable_http", "sse"):
        return blocked(base, "other", f"type {kind!r} is not a server rote can reach")
    if entry.get("socket"):
        return blocked(base, "other", "it connects over a Unix socket; rote connects over http(s)")
    withheld: list[str] = []
    if "client_id" in entry:
        withheld.append(
            "client_id: issued to Goose; rote registers its own client at "
            f"rote mcp login {base.name}"
        )
    if "client_secret_key" in entry:
        withheld.append("client_secret_key: names a secret in Goose's keyring, never read")
    tools = entry.get("available_tools")
    if isinstance(tools, list) and tools:
        withheld.append("available_tools: rote imports the whole server; the filter is not copied")
    envs = entry.get("envs", entry.get("env"))
    candidate = remote(
        base,
        url=entry.get("uri"),
        transport="sse" if kind == "sse" else "streamable-http",
        syntax=GOOSE_SYNTAX,
        headers=entry.get("headers"),
        withheld=withheld,
        scopes=str_list(entry.get("scopes")),
    )
    candidate = _check_references(
        candidate, envs if isinstance(envs, dict) else {}, str_list(entry.get("env_keys"))
    )
    if kind == "sse":
        candidate = replace(
            candidate, inactive="Goose no longer loads SSE extensions (migrate to streamable_http)"
        )
    if entry.get("enabled") is not True:
        candidate = replace(candidate, inactive="disabled in Goose (enabled: false)")
    return candidate


class GooseSource:
    id = CLIENT_ID
    label = "Goose"

    def config_paths(self, ctx: ClientContext) -> tuple[Path, ...]:
        return tuple(path for _, path in reversed(_layers(ctx)))

    def detect(self, ctx: ClientContext) -> bool:
        return _user_file(ctx).parent.is_dir()

    def read(self, ctx: ClientContext) -> SourceReading:
        reader = Reader()
        layers = [
            (scope, path, reader.section(path, reader.mapping(path, "yaml"), "extensions"))
            for scope, path in _layers(ctx)
        ]
        candidates = [
            _candidate(m.name, m.entry, scope=m.scope, origin=m.origin)
            for m in merge_entries(layers, deep=False)
        ]
        return SourceReading(
            client=CLIENT_ID,
            candidates=tuple(candidates),
            files_read=tuple(reader.files_read),
            problems=tuple(reader.problems),
            limits=LIMITS,
        )
