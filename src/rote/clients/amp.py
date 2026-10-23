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
  servers need none. The trust check and the hash are recovered from the
  CLI bundle (``@ampcode/cli-darwin-arm64@0.0.1792713681-g9b55f0``,
  functions ``rGt``/``U3i`` and ``Ggr.isTrusted``; Amp publishes no source):
  a workspace on the allowlist is trusted; else an entry matching both
  ``serverName`` and ``specHash`` decides by its ``allow``; else Amp asks
  ampcode.com's approvals, which are not in a file. ``specHash`` is the hex
  SHA-256 of ``JSON.stringify`` of ``{type: "url", url, headers: <keys
  sorted>, transport: transport || "http"}`` (``{type: "command", command,
  args, env: <keys sorted>}`` for a command), taken on the entry as written.
- ``amp.mcpPermissions`` (``{matches: {url | command | args}, action:
  "allow" | "reject"}``, first match wins) can reject a server.
- The files are JSON with comments and trailing commas.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from rote.clients.base import ClientContext, ImportCandidate, Scope, SourceReading, Trust
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
    trust_unknown,
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
    "approvals kept on ampcode.com are not read: a workspace server with no local approval "
    "is listed with unknown trust",
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


def _sorted_keys(value: Any) -> Any:
    """Amp's ``jht``: the same object with its keys in sorted order."""
    if not isinstance(value, dict):
        return {}
    return {key: value[key] for key in sorted(value)}


def spec_hash(entry: Mapping[str, Any]) -> str:
    """The ``specHash`` Amp files an approval under (``rGt`` in the CLI).

    Keys whose value is absent are left out, as ``JSON.stringify`` drops
    ``undefined``; ``||`` defaults apply to any falsy value. Matches Amp for
    string values; a number-valued header that JavaScript prints differently
    from Python (``1.0``) would hash differently, which only makes rote report
    an approval as missing, never invent one."""
    shape: dict[str, Any]
    if "command" in entry:
        shape = {
            "type": "command",
            "command": entry.get("command"),
            "args": entry.get("args") or [],
            "env": _sorted_keys(entry.get("env") or {}),
        }
    else:
        shape = {
            "type": "url",
            "url": entry.get("url"),
            "headers": _sorted_keys(entry.get("headers") or {}),
            "transport": entry.get("transport") or "http",
        }
    shape = {k: v for k, v in shape.items() if v is not None}
    text = json.dumps(shape, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _approval(
    user: Mapping[str, Any], workspace_dir: Path, name: str, entry: Any
) -> tuple[Trust, str | None]:
    """Amp's verdict on workspace server ``name``: ``Ggr.isTrusted`` read
    from the user file. ``unknown`` when only ampcode.com could say."""
    for item in user.get("amp.workspaces") or ():
        if (
            isinstance(item, dict)
            and item.get("path") == str(workspace_dir)
            and item.get("allowAllMcpServers") is True
        ):
            return "trusted", None
    if not isinstance(entry, dict):
        return "trusted", None  # blocked already; nothing to approve
    digest = spec_hash(entry)
    stale = False
    for item in user.get("amp.mcpTrustedServers") or ():
        if not isinstance(item, dict) or item.get("serverName") != name:
            continue
        if item.get("specHash") == digest:
            if item.get("allow") is True:
                return "trusted", None
            return "untrusted", "denied in Amp for this workspace"
        stale = True
    if stale:
        why = (
            "Amp's approval is for an earlier version of this entry (its spec hash differs), "
            f"so Amp asks again; amp mcp approve {name}"
        )
    else:
        why = (
            f"no approval in Amp's settings, and ampcode.com's are not read; amp mcp approve {name}"
        )
    return "unknown", why


def _with_trust(c: ImportCandidate, verdict: tuple[Trust, str | None] | None) -> ImportCandidate:
    if verdict is None or verdict[0] == "trusted":
        return c
    state, why = verdict
    if state == "unknown":
        return trust_unknown([c], "Amp", why or "")[0]
    return replace(c, trust="untrusted", trust_note=why)


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
        trust: dict[tuple[Path, str], tuple[Trust, str | None]] = {}
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
            for name, entry in servers.items():
                trust[(workspace_path, name)] = _approval(user, workspace_dir, name, entry)
            off = {key: why for key, (state, why) in trust.items() if state == "untrusted" and why}
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
        candidates = [_with_trust(c, trust.get((c.origin, c.original_name))) for c in candidates]
        return SourceReading(
            client=CLIENT_ID,
            candidates=tuple(candidates),
            files_read=tuple(reader.files_read),
            problems=tuple(reader.problems),
            limits=LIMITS,
        )
