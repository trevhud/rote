"""Amp's MCP servers, read from its settings files.

Verified 2026-10-22 against https://ampcode.com/docs/markdown/customize/mcp,
https://ampcode.com/docs/markdown/cli/settings,
https://ampcode.com/docs/markdown/customize/skills, the published schema
https://ampcode.com/cli-settings.schema.json, and the strings in the
published CLI (``@ampcode/cli``; Amp has no public source):

- **user**: ``$XDG_CONFIG_HOME/amp/settings.json`` (default
  ``~/.config/amp/``, on every OS, Windows included), or ``settings.jsonc``
  when the ``.json`` is absent. ``AMP_SETTINGS_FILE`` replaces it.
- **workspace**: the nearest ``.amp/settings.json`` (or ``.jsonc``) from the
  working directory up to the git root; the working directory alone outside
  a repository.
- **managed**: ``managed-settings.json`` under
  ``/Library/Application Support/ampcode/`` (macOS), ``/etc/ampcode/``
  (Linux), ``%ProgramData%\\ampcode\\`` (Windows).
- Keys are flat and prefixed: ``amp.mcpServers`` maps a name to a stdio
  entry (``command``, ``args``, ``env``) or a remote one (``url``,
  ``headers``, ``transport``: ``"http"``, ``"sse"`` or ``"http-first"``,
  ``oauth``: ``clientId``, ``clientSecret``, ``authUrl``, ``tokenUrl``,
  ``scopes``). Workspace entries replace user entries by name.
- Placeholders: ``${VAR}`` in env, command, args, url and headers; no
  default syntax; ``${amp:id-token}`` is Amp's own.
- Workspace servers need approval (``amp mcp approve <name>``), kept in the
  user file: ``amp.mcpTrustedServers`` (``{serverName, specHash, allow}``)
  and ``amp.workspaces`` (``{path, allowAllMcpServers: true}``). User
  servers need none. ``amp.mcpPermissions`` (``{matches: {url | command |
  args}, action: "allow" | "reject"}``, first match wins) can reject one.
- The files are JSON with comments and trailing commas.
"""

from __future__ import annotations

import fnmatch
import re
from collections.abc import Iterable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from rote.clients.base import ClientContext, ImportCandidate, Scope, SourceReading
from rote.clients.common import (
    STDIO_BLOCKED,
    Reader,
    blocked,
    new_candidate,
    not_an_object,
    rank,
    remote,
    str_list,
    switch_off,
    xdg_config_home,
)
from rote.clients.normalize import PlaceholderSyntax

CLIENT_ID = "amp"

AMP_SYNTAX = PlaceholderSyntax(
    client=CLIENT_ID,
    pattern=re.compile(r"\$\{(?P<var>[A-Za-z_][A-Za-z0-9_]*)\}"),
    other=re.compile(r"\$\{amp:[^}]*\}"),
)

LIMITS = (
    "an approval is matched by server name only: Amp also checks a hash of the entry, so an "
    "approval it voids after an edit still counts here",
    "remote servers defined on ampcode.com (amp mcp remote), skills' mcp.json servers, "
    "servers in managed-settings.json (local commands) and --mcp-config are not read",
)

_SERVERS = "amp.mcpServers"


def _user_file(ctx: ClientContext) -> Path:
    override = ctx.env.get("AMP_SETTINGS_FILE", "").strip()
    if override:
        return Path(override)
    base = xdg_config_home(ctx) / "amp"
    if not (base / "settings.json").is_file() and (base / "settings.jsonc").is_file():
        return base / "settings.jsonc"
    return base / "settings.json"


def _workspace_file(cwd: Path) -> tuple[Path | None, Path]:
    """(the file, the workspace dir it belongs to)."""
    for directory in (cwd, *cwd.parents):
        for name in ("settings.json", "settings.jsonc"):
            if (directory / ".amp" / name).is_file():
                return directory / ".amp" / name, directory
        if (directory / ".git").exists():
            return None, directory
    return None, cwd


def _managed_file(ctx: ClientContext) -> Path | None:
    if ctx.platform == "win32":
        return ctx.etc_dir / "ampcode" / "managed-settings.json" if ctx.etc_dir else None
    if ctx.system_dir is None:
        return None
    return ctx.system_dir / "ampcode" / "managed-settings.json"


def _candidate(name: str, entry: Any, *, scope: Scope, origin: Path) -> ImportCandidate:
    base = new_candidate(CLIENT_ID, name, scope=scope, origin=origin)
    if not isinstance(entry, dict):
        return not_an_object(base, entry)
    if "command" in entry:
        return blocked(base, "stdio", STDIO_BLOCKED)
    withheld: list[str] = []
    scopes: list[str] = []
    oauth = entry.get("oauth")
    if isinstance(oauth, dict):
        if "clientId" in oauth:
            withheld.append(
                "oauth.clientId: issued to Amp; rote registers its own client at "
                f"rote mcp login {base.name}"
            )
        if "clientSecret" in oauth:
            withheld.append("oauth.clientSecret: a secret issued to Amp, never copied")
        if "authUrl" in oauth or "tokenUrl" in oauth:
            withheld.append("oauth.authUrl/tokenUrl: rote discovers the authorization server")
        scopes = str_list(oauth.get("scopes"))
    if entry.get("includeTools"):
        withheld.append("includeTools: rote imports the whole server; the filter is not copied")
    return remote(
        base,
        url=entry.get("url"),
        transport="sse" if entry.get("transport") == "sse" else "streamable-http",
        syntax=AMP_SYNTAX,
        headers=entry.get("headers"),
        withheld=withheld,
        scopes=scopes,
    )


def _rejected(url: str | None, rules: Iterable[Any]) -> bool:
    """``amp.mcpPermissions``' verdict for a remote server: the first rule
    whose ``url`` glob matches decides; no match allows."""
    if url is None:
        return False
    for rule in rules:
        if not isinstance(rule, dict) or not isinstance(rule.get("matches"), dict):
            continue
        pattern = rule["matches"].get("url")
        if isinstance(pattern, str) and fnmatch.fnmatchcase(url, pattern):
            return rule.get("action") == "reject"
    return False


def _approval(user: Mapping[str, Any], workspace_dir: Path, name: str) -> str | None:
    """Why Amp would not run workspace server ``name``, or None."""
    for item in user.get("amp.workspaces") or ():
        if (
            isinstance(item, dict)
            and item.get("path") == str(workspace_dir)
            and item.get("allowAllMcpServers") is True
        ):
            return None
    for item in user.get("amp.mcpTrustedServers") or ():
        if isinstance(item, dict) and item.get("serverName") == name:
            return None if item.get("allow") is True else "denied in Amp for this workspace"
    return f"awaiting approval in Amp (amp mcp approve {name})"


class AmpSource:
    id = CLIENT_ID
    label = "Amp"

    def config_paths(self, ctx: ClientContext) -> tuple[Path, ...]:
        workspace, _ = _workspace_file(ctx.cwd)
        managed = _managed_file(ctx)
        return (
            *((workspace,) if workspace else ()),
            _user_file(ctx),
            *((managed,) if managed else ()),
        )

    def detect(self, ctx: ClientContext) -> bool:
        return _user_file(ctx).parent.is_dir() or _workspace_file(ctx.cwd)[0] is not None

    def read(self, ctx: ClientContext) -> SourceReading:
        reader = Reader()
        user_path = _user_file(ctx)
        user = reader.mapping(user_path, "jsonc")
        workspace_path, workspace_dir = _workspace_file(ctx.cwd)
        workspace = reader.mapping(workspace_path, "jsonc") if workspace_path else {}
        managed_path = _managed_file(ctx)
        managed = reader.mapping(managed_path, "jsonc") if managed_path else {}

        groups: list[tuple[str, list[ImportCandidate]]] = []
        off: dict[tuple[Path, str], str] = {}
        if workspace_path is not None:
            servers = reader.section(workspace_path, workspace, _SERVERS)
            groups.append(
                (
                    "workspace",
                    [
                        _candidate(n, e, scope="project", origin=workspace_path)
                        for n, e in servers.items()
                    ],
                )
            )
            for name in servers:
                why = _approval(user, workspace_dir, name)
                if why is not None:
                    off[(workspace_path, name)] = why
        servers = reader.section(user_path, user, _SERVERS)
        groups.append(
            ("user", [_candidate(n, e, scope="user", origin=user_path) for n, e in servers.items()])
        )

        own = workspace.get("amp.mcpPermissions", user.get("amp.mcpPermissions"))
        rules = [*(managed.get("amp.mcpPermissions") or ()), *(own or ())]
        candidates = [
            replace(c, blocked="rejected by amp.mcpPermissions", risk=None)
            if c.blocked is None and _rejected(c.url, rules)
            else c
            for c in switch_off(rank(groups), off)
        ]
        return SourceReading(
            client=CLIENT_ID,
            candidates=tuple(candidates),
            files_read=tuple(reader.files_read),
            problems=tuple(reader.problems),
            limits=LIMITS,
        )
