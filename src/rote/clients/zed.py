"""Zed's context servers (its name for MCP servers), read from its settings.

Verified 2026-10-22 against Zed's source and docs on main:
https://github.com/zed-industries/zed/blob/main/crates/paths/src/paths.rs
(``config_dir``),
https://github.com/zed-industries/zed/blob/main/crates/settings_content/src/project.rs
(``ContextServerSettingsContent``),
https://github.com/zed-industries/zed/blob/main/crates/migrator/src/migrations/m_2025_11_25/settings.rs,
https://github.com/zed-industries/zed/blob/main/crates/context_server/src/transport/http.rs,
https://github.com/zed-industries/zed/blob/main/docs/src/ai/mcp.md and
https://github.com/zed-industries/zed/blob/main/docs/src/worktree-trust.md:

- **user**: ``~/.config/zed/settings.json`` on macOS (``XDG_CONFIG_HOME`` is
  not read there); ``$FLATPAK_XDG_CONFIG_HOME``, else ``$XDG_CONFIG_HOME``,
  else ``~/.config``, then ``zed/settings.json`` on Linux;
  ``%APPDATA%\\Zed\\settings.json`` on Windows.
- **project**: ``.zed/settings.json`` in the worktree. Project settings beat
  user settings for that project; servers defined there start only once
  the worktree is trusted.
- ``context_servers`` is a map of name to an untagged entry: a command
  (``command``, ``args``, ``env``, ``timeout``), a remote server (``url``,
  ``headers``, ``timeout``, ``oauth: {client_id, client_secret}``), or an
  extension's server (``settings`` only). Each takes ``enabled`` (default
  true). Older files carry ``"source": "custom" | "extension"`` and a nested
  ``command: {path, args, env}``, which the 2025-11-25 migration flattened.
- The remote transport is Streamable HTTP (http.rs); there is no SSE client.
- No placeholder syntax: values are sent as written
  (:data:`~rote.clients.normalize.NO_PLACEHOLDERS`).
- The file is JSON with comments and trailing commas (serde_json_lenient).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rote.clients.base import ClientContext, ImportCandidate, Scope, SourceReading
from rote.clients.common import (
    STDIO_BLOCKED,
    Reader,
    appdata,
    blocked,
    new_candidate,
    not_an_object,
    project_root,
    rank,
    remote,
    switch_off,
    xdg_config_home,
)
from rote.clients.normalize import NO_PLACEHOLDERS

CLIENT_ID = "zed"

LIMITS = (
    "worktree trust is not read: servers in .zed/settings.json are listed as active even "
    "where Zed's Restricted Mode keeps them from starting",
    "a same-named server in user and project settings is treated as the project entry "
    "replacing the user one; Zed merges settings objects, which could combine fields",
)


def _user_file(ctx: ClientContext) -> Path:
    if ctx.platform == "win32":
        return appdata(ctx) / "Zed" / "settings.json"
    if ctx.platform == "darwin":
        return ctx.home / ".config" / "zed" / "settings.json"
    flatpak = ctx.env.get("FLATPAK_XDG_CONFIG_HOME", "").strip()
    base = Path(flatpak) if flatpak else xdg_config_home(ctx)
    return base / "zed" / "settings.json"


def _project_file(ctx: ClientContext) -> Path:
    return project_root(ctx.cwd) / ".zed" / "settings.json"


def _candidate(name: str, entry: Any, *, scope: Scope, origin: Path) -> ImportCandidate:
    base = new_candidate(CLIENT_ID, name, scope=scope, origin=origin)
    if not isinstance(entry, dict):
        return not_an_object(base, entry)
    if "url" in entry:
        withheld: list[str] = []
        oauth = entry.get("oauth")
        if isinstance(oauth, dict):
            if "client_id" in oauth:
                withheld.append(
                    "oauth.client_id: issued to Zed; rote registers its own client at "
                    f"rote mcp login {base.name}"
                )
            if "client_secret" in oauth:
                withheld.append("oauth.client_secret: a secret issued to Zed, never copied")
        return remote(
            base,
            url=entry["url"],
            transport="streamable-http",
            syntax=NO_PLACEHOLDERS,
            headers=entry.get("headers"),
            withheld=withheld,
        )
    if "command" in entry or entry.get("source") == "custom":
        return blocked(base, "stdio", STDIO_BLOCKED)
    return blocked(
        base,
        "other",
        "a server a Zed extension provides (a local program the extension installs); "
        "rote cannot reach it",
    )


class ZedSource:
    id = CLIENT_ID
    label = "Zed"

    def config_paths(self, ctx: ClientContext) -> tuple[Path, ...]:
        return (_project_file(ctx), _user_file(ctx))

    def detect(self, ctx: ClientContext) -> bool:
        return _user_file(ctx).parent.is_dir() or _project_file(ctx).is_file()

    def read(self, ctx: ClientContext) -> SourceReading:
        reader = Reader()
        groups: list[tuple[str, list[ImportCandidate]]] = []
        off: dict[tuple[Path, str], str] = {}
        files: tuple[tuple[Scope, Path], ...] = (
            ("project", _project_file(ctx)),
            ("user", _user_file(ctx)),
        )
        for scope, path in files:
            settings = reader.mapping(path, "jsonc")
            servers = reader.section(path, settings, "context_servers")
            groups.append(
                (scope, [_candidate(n, e, scope=scope, origin=path) for n, e in servers.items()])
            )
            for name, entry in servers.items():
                if isinstance(entry, dict) and entry.get("enabled") is False:
                    off[(path, name)] = "turned off in Zed (enabled: false)"
        return SourceReading(
            client=CLIENT_ID,
            candidates=tuple(switch_off(rank(groups), off)),
            files_read=tuple(reader.files_read),
            problems=tuple(reader.problems),
            limits=LIMITS,
        )
