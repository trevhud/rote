"""Claude Code's MCP servers, read from its files.

Where Claude Code keeps them (code.claude.com/docs/en/mcp, "MCP
installation scopes"; settings-reference; plugins-reference; shapes
checked against a real install, key names only, 2026-10-22):

- **user**: ``~/.claude.json`` top-level ``mcpServers``.
- **local**: ``~/.claude.json`` ``projects["<abs project dir>"].mcpServers``.
- **project**: ``<project>/.mcp.json`` ``{"mcpServers": {...}}``. Claude Code
  uses one only when approved: ``enableAllProjectMcpServers`` or a name in
  ``enabledMcpjsonServers``, and not in ``disabledMcpjsonServers``. Those
  keys live in any settings file and in the ``projects`` entry; the
  committed ``.claude/settings.json`` counts for approval only in a folder
  whose trust dialog was accepted, and a disable applies from anywhere.
- **plugin**: each enabled plugin (``enabledPlugins`` in the settings files,
  ``"name@marketplace": true``) at its ``installPath`` from
  ``~/.claude/plugins/installed_plugins.json`` (``{"version": 2,
  "plugins": {"name@mkt": [{"scope", "installPath", ...}]}}``), declaring
  servers in ``.mcp.json`` (wrapped in ``mcpServers`` or a bare map) or in
  ``.claude-plugin/plugin.json`` ``mcpServers`` (a map, a path to a JSON
  file, or a list of both).

Precedence is local > project > user and the whole entry wins, so a name
defined twice is listed once as active and once as shadowed. The ``/mcp``
toggle's ``projects[...].disabledMcpServers`` marks a name inactive.

``CLAUDE_CONFIG_DIR`` moves ``~/.claude`` ("All settings, session history,
and plugins are stored under this path", docs/en/env-vars). With it set,
``.claude.json`` is read from inside it, which is where Claude Code keeps
it in that layout.

Entry shapes: ``{"type": "http"|"streamable-http"|"sse"|"ws", "url",
"headers", "headersHelper", "oauth": {"clientId", "callbackPort",
"authServerMetadataUrl", "scopes"}}`` remote; ``{"type"?: "stdio",
"command", "args", "env"}`` local. An entry with a ``url`` and no ``type`` is
an error to Claude Code itself (a missing type means stdio).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from rote.clients.base import (
    ClientContext,
    ImportCandidate,
    Scope,
    SourceReading,
    Transport,
)
from rote.clients.common import (
    STDIO_BLOCKED,
    WEBSOCKET_BLOCKED,
    Reader,
    blocked,
    expand_home,
    new_candidate,
    not_an_object,
    remote,
    servers_in,
    str_list,
    trust_verified,
)
from rote.clients.normalize import CLAUDE_CODE_SYNTAX

CLIENT_ID = "claude-code"

_TRANSPORTS: dict[str, Transport] = {
    "http": "streamable-http",
    "streamable-http": "streamable-http",
    "sse": "sse",
    "ws": "websocket",
    "stdio": "stdio",
}

#: Scopes in Claude Code's precedence order, highest first.
_PRECEDENCE: tuple[Scope, ...] = ("local", "project", "user")


@dataclass(frozen=True)
class _Paths:
    config_dir: Path
    claude_json: Path
    user_settings: Path
    installed_plugins: Path
    #: Admin-deployed policy (code.claude.com/docs/en/managed-mcp). None when
    #: the context reads no system directory.
    managed_dir: Path | None


#: What the Claude Code source never reads, said on every reading.
LIMITS = (
    "managed settings delivered by an MDM profile, the Windows registry or claude.ai "
    "server-managed settings are not read, so a server they deny can still be listed",
    "allowedMcpServers is not evaluated: a server an allowlist excludes is still listed",
)


def _paths(ctx: ClientContext) -> _Paths:
    override = ctx.env.get("CLAUDE_CONFIG_DIR", "").strip()
    if override:
        config_dir = expand_home(override, ctx)
        claude_json = config_dir / ".claude.json"
    else:
        config_dir = ctx.home / ".claude"
        claude_json = ctx.home / ".claude.json"
    managed_dir: Path | None = None
    if ctx.system_dir is not None:
        sub = "claude-code" if ctx.platform.startswith("linux") else "ClaudeCode"
        managed_dir = ctx.system_dir / sub
    return _Paths(
        config_dir=config_dir,
        claude_json=claude_json,
        user_settings=config_dir / "settings.json",
        installed_plugins=config_dir / "plugins" / "installed_plugins.json",
        managed_dir=managed_dir,
    )


def _url_key(url: str) -> str | None:
    """``scheme://host/path`` with the host lowercased and the port dropped."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    if not parts.hostname:
        return None
    return f"{parts.scheme}://{parts.hostname.rstrip('.')}{parts.path or '/'}"


_PATTERN_PORT = re.compile(r"^([^:/]*://[^/:]*)(:(\*|\d+))?")


def _url_denied(url: str, pattern: str) -> bool:
    """Claude Code's ``serverUrl`` match, read wider where it differs: ports
    are ignored and the whole URL, path included, compares case-insensitively. A deny that
    matches too much only blocks an import; one that matches too little
    would import a server the user's policy blocks."""
    key = _url_key(url)
    if key is None:
        return False
    head = _PATTERN_PORT.match(pattern)
    if head is None:
        return False
    host_part, rest = head.group(1), pattern[head.end() :]
    if rest in ("", "/"):
        rest = "/*"  # a pattern with no path matches any path (docs/en/managed-mcp)
    regex = ".*".join(re.escape(piece) for piece in f"{host_part.lower()}{rest}".split("*"))
    return re.fullmatch(regex, key, flags=re.IGNORECASE) is not None


@dataclass(frozen=True)
class _Denylist:
    names: frozenset[str]
    urls: tuple[str, ...]

    def why(self, candidate: ImportCandidate) -> str | None:
        if candidate.original_name in self.names:
            return "blocked by a deniedMcpServers serverName entry in Claude Code's settings"
        if candidate.url is not None:
            for pattern in self.urls:
                if _url_denied(candidate.url, pattern):
                    return "blocked by a deniedMcpServers serverUrl entry in Claude Code's settings"
        return None


def _denylist(layers: Iterable[Mapping[str, Any]]) -> _Denylist:
    """``deniedMcpServers`` merged from every scope, as Claude Code merges it
    ("Denylists merge from every scope regardless")."""
    names: set[str] = set()
    urls: list[str] = []
    for layer in layers:
        entries = layer.get("deniedMcpServers")
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if isinstance(entry.get("serverName"), str):
                names.add(entry["serverName"])
            if isinstance(entry.get("serverUrl"), str):
                urls.append(entry["serverUrl"])
            # serverCommand entries match stdio servers, which are never importable.
    return _Denylist(frozenset(names), tuple(urls))


def _project_entry(claude_json: Mapping[str, Any], cwd: Path) -> tuple[Path | None, dict[str, Any]]:
    """The ``projects`` entry for ``cwd`` or its nearest ancestor that has one."""
    projects = claude_json.get("projects")
    if not isinstance(projects, dict):
        return None, {}
    for directory in (cwd, *cwd.parents):
        entry = projects.get(str(directory))
        if isinstance(entry, dict):
            return directory, entry
    return None, {}


def _project_root(cwd: Path, known: Path | None) -> Path | None:
    """Where ``.mcp.json`` and ``.claude/`` are looked for: the project Claude
    Code knows, else the nearest ancestor with a ``.mcp.json``, stopping at
    the repository root."""
    if known is not None:
        return known
    for directory in (cwd, *cwd.parents):
        if (directory / ".mcp.json").is_file():
            return directory
        if (directory / ".git").exists():
            return None
    return None


@dataclass(frozen=True)
class _Approval:
    enable_all: bool
    enabled: frozenset[str]
    disabled: frozenset[str]

    def why_inactive(self, name: str) -> str | None:
        if name in self.disabled:
            return "disabled for this project in Claude Code"
        if self.enable_all or name in self.enabled:
            return None
        return "not approved in Claude Code for this project"


def _approval(
    settings: Iterable[Mapping[str, Any]], counted: Iterable[Mapping[str, Any]]
) -> _Approval:
    """``settings`` disable from anywhere; only ``counted`` ones approve."""
    enabled: set[str] = set()
    disabled: set[str] = set()
    enable_all = False
    for layer in settings:
        disabled.update(str_list(layer.get("disabledMcpjsonServers")))
    for layer in counted:
        enabled.update(str_list(layer.get("enabledMcpjsonServers")))
        enable_all = enable_all or layer.get("enableAllProjectMcpServers") is True
    return _Approval(enable_all, frozenset(enabled), frozenset(disabled))


def _candidate(
    original_name: str,
    entry: Any,
    *,
    scope: Scope,
    origin: Path,
    plugin: str | None = None,
) -> ImportCandidate:
    base = new_candidate(CLIENT_ID, original_name, scope=scope, origin=origin, plugin=plugin)
    if not isinstance(entry, dict):
        return not_an_object(base, entry)

    raw_type = entry.get("type")
    url = entry.get("url")
    if raw_type is None:
        if isinstance(url, str):
            return replace(
                base,
                url=url,
                blocked=base.blocked
                or "the entry has a url but no type, which Claude Code itself rejects",
            )
        raw_type = "stdio"
    transport = _TRANSPORTS.get(str(raw_type), "other")
    if transport == "stdio":
        return blocked(base, transport, STDIO_BLOCKED)
    if transport == "websocket":
        return blocked(base, transport, WEBSOCKET_BLOCKED)
    if transport == "other":
        return blocked(base, transport, f"type {raw_type!r} is not a server rote can reach")

    withheld: list[str] = []
    if "headersHelper" in entry:
        withheld.append(
            "headersHelper: it runs a command to make headers, and rote does not run another "
            f"app's commands; set headers with rote mcp add {base.name} <url> --header"
        )
    scopes: tuple[str, ...] = ()
    oauth = entry.get("oauth")
    if isinstance(oauth, dict):
        if "clientId" in oauth:
            withheld.append(
                "oauth.clientId: issued to Claude Code; rote registers its own client at "
                f"rote mcp login {base.name}"
            )
        if "authServerMetadataUrl" in oauth:
            withheld.append("oauth.authServerMetadataUrl: rote discovers the authorization server")
        raw_scopes = oauth.get("scopes")
        if isinstance(raw_scopes, str):
            scopes = tuple(s for s in raw_scopes.split() if s)
    return remote(
        base,
        url=url,
        transport=transport,
        syntax=CLAUDE_CODE_SYNTAX,
        headers=entry.get("headers"),
        withheld=withheld,
        scopes=scopes,
    )


class ClaudeCodeSource:
    id = CLIENT_ID
    label = "Claude Code"

    def config_paths(self, ctx: ClientContext) -> tuple[Path, ...]:
        paths = _paths(ctx)
        project, _ = _project_entry(self._claude_json_quiet(paths), ctx.cwd)
        root = _project_root(ctx.cwd, project)
        project_files: tuple[Path, ...] = ()
        if root is not None:
            project_files = (
                root / ".mcp.json",
                root / ".claude" / "settings.json",
                root / ".claude" / "settings.local.json",
            )
        return (
            paths.claude_json,
            paths.user_settings,
            paths.installed_plugins,
            *project_files,
        )

    @staticmethod
    def _claude_json_quiet(paths: _Paths) -> dict[str, Any]:
        reader = Reader()
        return reader.mapping(paths.claude_json)

    def detect(self, ctx: ClientContext) -> bool:
        paths = _paths(ctx)
        return paths.claude_json.is_file() or paths.config_dir.is_dir()

    def read(self, ctx: ClientContext) -> SourceReading:
        paths = _paths(ctx)
        reader = Reader()
        claude_json = reader.mapping(paths.claude_json)
        project_dir, project_entry = _project_entry(claude_json, ctx.cwd)
        root = _project_root(ctx.cwd, project_dir)

        user_settings = reader.mapping(paths.user_settings)
        project_settings: dict[str, Any] = {}
        local_settings: dict[str, Any] = {}
        if root is not None:
            project_settings = reader.mapping(root / ".claude" / "settings.json")
            local_settings = reader.mapping(root / ".claude" / "settings.local.json")
        trusted = project_entry.get("hasTrustDialogAccepted") is True

        by_scope: dict[Scope, list[ImportCandidate]] = {"local": [], "project": [], "user": []}

        user_servers = claude_json.get("mcpServers")
        if isinstance(user_servers, dict):
            for name, entry in user_servers.items():
                by_scope["user"].append(
                    _candidate(name, entry, scope="user", origin=paths.claude_json)
                )
        local_servers = project_entry.get("mcpServers")
        if isinstance(local_servers, dict):
            for name, entry in local_servers.items():
                by_scope["local"].append(
                    _candidate(name, entry, scope="local", origin=paths.claude_json)
                )
        if root is not None:
            mcp_json_path = root / ".mcp.json"
            mcp_json = reader.load(mcp_json_path)
            project_servers = servers_in(mcp_json) if mcp_json is not None else None
            if mcp_json is not None and project_servers is None:
                reader.problems.append(f"{mcp_json_path}: no mcpServers object")
            approval = _approval(
                (user_settings, project_settings, local_settings, project_entry),
                (
                    user_settings,
                    local_settings,
                    project_entry,
                    *((project_settings,) if trusted else ()),
                ),
            )
            for name, entry in (project_servers or {}).items():
                candidate = _candidate(name, entry, scope="project", origin=mcp_json_path)
                why = approval.why_inactive(name)
                if why is None:
                    [candidate] = trust_verified([candidate])
                else:
                    candidate = replace(candidate, inactive=why, trust="untrusted", trust_note=why)
                by_scope["project"].append(candidate)

        toggled_off = set(str_list(project_entry.get("disabledMcpServers")))
        ordered: list[ImportCandidate] = []
        winner: dict[str, Scope] = {}
        for scope in _PRECEDENCE:
            for candidate in by_scope[scope]:
                key = candidate.original_name
                inactive = candidate.inactive
                if key in winner:
                    inactive = f"shadowed by the {winner[key]} entry of the same name"
                elif inactive is None:
                    winner[key] = scope
                if inactive is None and key in toggled_off:
                    inactive = "turned off in Claude Code (/mcp)"
                ordered.append(replace(candidate, inactive=inactive))

        ordered.extend(
            self._plugin_candidates(
                reader, paths, (user_settings, project_settings, local_settings)
            )
        )

        managed_settings: dict[str, Any] = {}
        exclusive = False
        if paths.managed_dir is not None:
            managed_path = paths.managed_dir / "managed-settings.json"
            managed_settings = reader.mapping(managed_path)
            if "managedMcpServers" in managed_settings:
                reader.problems.append(
                    f"{managed_path}: managedMcpServers (servers your organization provides) "
                    "is not read yet, so those servers are not listed"
                )
            managed_mcp = paths.managed_dir / "managed-mcp.json"
            if managed_mcp.is_file():
                exclusive = True
                reader.problems.append(
                    f"{managed_mcp}: Claude Code loads only the servers it defines on this "
                    "machine; rote does not read it yet, and lists every other server as unused"
                )
        denied = _denylist((user_settings, project_settings, local_settings, managed_settings))
        checked: list[ImportCandidate] = []
        for candidate in ordered:
            why = denied.why(candidate)
            if why is not None:
                candidate = replace(candidate, blocked=why, risk=None)
            elif exclusive and candidate.inactive is None:
                candidate = replace(
                    candidate,
                    inactive="Claude Code loads only managed-mcp.json servers on this machine",
                )
            checked.append(candidate)
        return SourceReading(
            client=CLIENT_ID,
            candidates=tuple(checked),
            files_read=tuple(reader.files_read),
            problems=tuple(reader.problems),
            limits=LIMITS,
        )

    @staticmethod
    def _plugin_candidates(
        reader: Reader, paths: _Paths, settings: Iterable[Mapping[str, Any]]
    ) -> list[ImportCandidate]:
        found: list[ImportCandidate] = []
        for plugin in _enabled_plugins(reader, paths, settings):
            for origin, servers in _plugin_servers(reader, plugin.root):
                for name, entry in servers.items():
                    found.append(
                        _candidate(name, entry, scope="plugin", origin=origin, plugin=plugin.name)
                    )
        return found


@dataclass(frozen=True)
class EnabledPlugin:
    """A Claude Code plugin that is installed and turned on."""

    #: ``name@marketplace``, the key in ``enabledPlugins``.
    key: str
    name: str
    #: The install's ``installPath``: where its ``.mcp.json``, ``skills/``
    #: and ``.claude-plugin/plugin.json`` live.
    root: Path


def _enabled_plugins(
    reader: Reader, paths: _Paths, settings: Iterable[Mapping[str, Any]]
) -> list[EnabledPlugin]:
    enabled: dict[str, bool] = {}
    for layer in settings:  # later layers win: user < project < local
        raw = layer.get("enabledPlugins")
        if isinstance(raw, dict):
            enabled.update({k: v for k, v in raw.items() if isinstance(v, bool)})
    installed = reader.mapping(paths.installed_plugins)
    plugins = installed.get("plugins")
    if not isinstance(plugins, dict):
        return []
    found: list[EnabledPlugin] = []
    for key, installs in sorted(plugins.items()):
        if enabled.get(key) is not True or not isinstance(installs, list):
            continue
        for install in installs:
            if not isinstance(install, dict) or not isinstance(install.get("installPath"), str):
                continue
            found.append(EnabledPlugin(key, key.split("@", 1)[0], Path(install["installPath"])))
            break  # one install per plugin key is the one Claude Code loads
    return found


@dataclass(frozen=True)
class ClaudeCodeLayout:
    """Where Claude Code keeps things for one context, for readers other
    than the MCP source (rote.skills): the config dir (``~/.claude``, or
    ``CLAUDE_CONFIG_DIR``), the project root it resolves from the cwd, the
    enabled plugins, and what could not be read."""

    config_dir: Path
    project_root: Path | None
    plugins: tuple[EnabledPlugin, ...]
    problems: tuple[str, ...]


def claude_code_layout(ctx: ClientContext) -> ClaudeCodeLayout:
    """The same resolution :class:`ClaudeCodeSource` reads servers with:
    the settings layers that decide ``enabledPlugins`` are user, project
    and local, from the project ``~/.claude.json`` knows for the cwd."""
    paths = _paths(ctx)
    reader = Reader()
    project_dir, _ = _project_entry(reader.mapping(paths.claude_json), ctx.cwd)
    root = _project_root(ctx.cwd, project_dir)
    layers = [reader.mapping(paths.user_settings)]
    if root is not None:
        layers.append(reader.mapping(root / ".claude" / "settings.json"))
        layers.append(reader.mapping(root / ".claude" / "settings.local.json"))
    plugins = _enabled_plugins(reader, paths, layers)
    return ClaudeCodeLayout(paths.config_dir, root, tuple(plugins), tuple(reader.problems))


def _plugin_servers(reader: Reader, root: Path) -> list[tuple[Path, dict[str, Any]]]:
    """Every server map a plugin declares, with the file it came from."""
    found: list[tuple[Path, dict[str, Any]]] = []
    mcp_json = root / ".mcp.json"
    servers = servers_in(reader.load(mcp_json))
    if servers:
        found.append((mcp_json, servers))
    manifest_path = root / ".claude-plugin" / "plugin.json"
    manifest = reader.load(manifest_path)
    if not isinstance(manifest, dict) or "mcpServers" not in manifest:
        return found
    declared = manifest["mcpServers"]
    for item in declared if isinstance(declared, list) else [declared]:
        if isinstance(item, dict):
            found.append((manifest_path, item))
        elif isinstance(item, str) and item.endswith(".json"):
            target = (root / item).resolve()
            if not target.is_relative_to(root.resolve()):
                reader.problems.append(f"{manifest_path}: mcpServers path leaves the plugin")
                continue
            servers = servers_in(reader.load(target))
            if servers and target != mcp_json.resolve():
                found.append((target, servers))
        elif isinstance(item, str):
            reader.problems.append(
                f"{manifest_path}: a {Path(item).suffix or 'non-JSON'} bundle, which rote "
                "cannot read"
            )
    return found
