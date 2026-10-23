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

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from rote.clients.base import (
    ClientContext,
    HeaderTemplate,
    ImportCandidate,
    Scope,
    SourceReading,
    Transport,
)
from rote.clients.normalize import (
    CLAUDE_CODE_SYNTAX,
    header_template,
    server_identifier,
    url_problem,
)

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


def _paths(ctx: ClientContext) -> _Paths:
    override = ctx.env.get("CLAUDE_CONFIG_DIR", "").strip()
    if override:
        config_dir = Path(override).expanduser()
        claude_json = config_dir / ".claude.json"
    else:
        config_dir = ctx.home / ".claude"
        claude_json = ctx.home / ".claude.json"
    return _Paths(
        config_dir=config_dir,
        claude_json=claude_json,
        user_settings=config_dir / "settings.json",
        installed_plugins=config_dir / "plugins" / "installed_plugins.json",
    )


class _Reader:
    """Reads JSON files once each and records what it read and what failed."""

    def __init__(self) -> None:
        self.files_read: list[Path] = []
        self.problems: list[str] = []
        self._cache: dict[Path, Any] = {}

    def json(self, path: Path) -> Any:
        if path in self._cache:
            return self._cache[path]
        value: Any = None
        if path.is_file():
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
                self.problems.append(f"{path}: could not read it ({type(e).__name__})")
            else:
                self.files_read.append(path)
        self._cache[path] = value
        return value

    def mapping(self, path: Path) -> dict[str, Any]:
        value = self.json(path)
        if value is None:
            return {}
        if not isinstance(value, dict):
            self.problems.append(f"{path}: expected a JSON object, got {type(value).__name__}")
            return {}
        return value


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


def _str_list(value: Any) -> list[str]:
    return [v for v in value if isinstance(v, str)] if isinstance(value, list) else []


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
        disabled.update(_str_list(layer.get("disabledMcpjsonServers")))
    for layer in counted:
        enabled.update(_str_list(layer.get("enabledMcpjsonServers")))
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
    try:
        name = server_identifier(original_name)
    except ValueError:
        name = "_"
    base = ImportCandidate(
        client=CLIENT_ID,
        name=name,
        original_name=original_name,
        scope=scope,
        origin=origin,
        transport="other",
        plugin=plugin,
    )
    if name == "_":
        return replace(base, blocked="the name has no letters or digits rote can keep")
    if not isinstance(entry, dict):
        return replace(base, blocked=f"the entry is a {type(entry).__name__}, not an object")

    raw_type = entry.get("type")
    url = entry.get("url")
    if raw_type is None:
        if isinstance(url, str):
            return replace(
                base,
                url=url,
                blocked="the entry has a url but no type, which Claude Code itself rejects",
            )
        raw_type = "stdio"
    transport = _TRANSPORTS.get(str(raw_type), "other")
    base = replace(base, transport=transport)
    if transport == "stdio":
        return replace(
            base,
            blocked=(
                "a stdio server (a local command); rote's registry holds HTTP and SSE servers only"
            ),
        )
    if transport == "websocket":
        return replace(base, blocked="a WebSocket server; rote connects over HTTP and SSE only")
    if transport == "other":
        return replace(base, blocked=f"type {raw_type!r} is not a server rote can reach")
    if not isinstance(url, str) or not url:
        return replace(base, blocked="the entry has no url")
    base = replace(base, url=url)

    withheld: list[str] = []
    headers: list[HeaderTemplate] = []
    raw_headers = entry.get("headers")
    if isinstance(raw_headers, dict):
        for header, value in raw_headers.items():
            if not isinstance(value, str):
                withheld.append(f"header {header}: not a string")
                continue
            verdict = header_template(name, str(header), value, CLAUDE_CODE_SYNTAX)
            if verdict.header is not None:
                headers.append(verdict.header)
            if verdict.withheld is not None:
                withheld.append(verdict.withheld)
            withheld.extend(verdict.notes)
    if "headersHelper" in entry:
        withheld.append(
            "headersHelper: it runs a command to make headers, and rote does not run another "
            f"app's commands; set headers with rote mcp add {name} <url> --header"
        )

    scopes: tuple[str, ...] = ()
    oauth = entry.get("oauth")
    if isinstance(oauth, dict):
        if "clientId" in oauth:
            withheld.append(
                "oauth.clientId: issued to Claude Code; rote registers its own client at "
                f"rote mcp login {name}"
            )
        if "authServerMetadataUrl" in oauth:
            withheld.append("oauth.authServerMetadataUrl: rote discovers the authorization server")
        raw_scopes = oauth.get("scopes")
        if isinstance(raw_scopes, str):
            scopes = tuple(s for s in raw_scopes.split() if s)

    return replace(
        base,
        headers=tuple(headers),
        scopes=scopes,
        withheld=tuple(withheld),
        blocked=url_problem(url, CLAUDE_CODE_SYNTAX),
    )


def _servers_in(value: Any) -> dict[str, Any] | None:
    """A ``.mcp.json``'s servers: wrapped in ``mcpServers`` or a bare map."""
    if not isinstance(value, dict):
        return None
    inner = value.get("mcpServers")
    if isinstance(inner, dict):
        return inner
    return value if "mcpServers" not in value else None


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
        reader = _Reader()
        return reader.mapping(paths.claude_json)

    def detect(self, ctx: ClientContext) -> bool:
        paths = _paths(ctx)
        return paths.claude_json.is_file() or paths.config_dir.is_dir()

    def read(self, ctx: ClientContext) -> SourceReading:
        paths = _paths(ctx)
        reader = _Reader()
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
            mcp_json = reader.json(mcp_json_path)
            project_servers = _servers_in(mcp_json) if mcp_json is not None else None
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
                by_scope["project"].append(replace(candidate, inactive=approval.why_inactive(name)))

        toggled_off = set(_str_list(project_entry.get("disabledMcpServers")))
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
        return SourceReading(
            client=CLIENT_ID,
            candidates=tuple(ordered),
            files_read=tuple(reader.files_read),
            problems=tuple(reader.problems),
        )

    @staticmethod
    def _plugin_candidates(
        reader: _Reader, paths: _Paths, settings: Iterable[Mapping[str, Any]]
    ) -> list[ImportCandidate]:
        enabled: dict[str, bool] = {}
        for layer in settings:  # later layers win: user < project < local
            raw = layer.get("enabledPlugins")
            if isinstance(raw, dict):
                enabled.update({k: v for k, v in raw.items() if isinstance(v, bool)})
        installed = reader.mapping(paths.installed_plugins)
        plugins = installed.get("plugins")
        if not isinstance(plugins, dict):
            return []
        found: list[ImportCandidate] = []
        for key, installs in sorted(plugins.items()):
            if enabled.get(key) is not True or not isinstance(installs, list):
                continue
            plugin = key.split("@", 1)[0]
            for install in installs:
                if not isinstance(install, dict) or not isinstance(install.get("installPath"), str):
                    continue
                root = Path(install["installPath"])
                for origin, servers in _plugin_servers(reader, root):
                    for name, entry in servers.items():
                        found.append(
                            _candidate(name, entry, scope="plugin", origin=origin, plugin=plugin)
                        )
                break  # one install per plugin key is the one Claude Code loads
        return found


def _plugin_servers(reader: _Reader, root: Path) -> list[tuple[Path, dict[str, Any]]]:
    """Every server map a plugin declares, with the file it came from."""
    found: list[tuple[Path, dict[str, Any]]] = []
    mcp_json = root / ".mcp.json"
    servers = _servers_in(reader.json(mcp_json))
    if servers:
        found.append((mcp_json, servers))
    manifest_path = root / ".claude-plugin" / "plugin.json"
    manifest = reader.json(manifest_path)
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
            servers = _servers_in(reader.json(target))
            if servers and target != mcp_json.resolve():
                found.append((target, servers))
        elif isinstance(item, str):
            reader.problems.append(
                f"{manifest_path}: a {Path(item).suffix or 'non-JSON'} bundle, which rote "
                "cannot read"
            )
    return found
