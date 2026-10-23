"""Gemini CLI's MCP servers, read from its settings files and extensions.

Verified 2026-10-22 against the source at google-gemini/gemini-cli main
(``a713410a``, 0.67.0-nightly):
https://github.com/google-gemini/gemini-cli/blob/main/packages/cli/src/config/settings.ts,
https://github.com/google-gemini/gemini-cli/blob/main/packages/cli/src/config/settingsSchema.ts,
https://github.com/google-gemini/gemini-cli/blob/main/packages/cli/src/config/trustedFolders.ts,
https://github.com/google-gemini/gemini-cli/blob/main/packages/cli/src/config/mcp/mcpServerEnablement.ts,
https://github.com/google-gemini/gemini-cli/blob/main/packages/cli/src/utils/envVarResolver.ts,
https://github.com/google-gemini/gemini-cli/blob/main/packages/core/src/tools/mcp-client.ts,
and https://github.com/google-gemini/gemini-cli/blob/main/docs/tools/mcp-server.md:

- **user**: ``~/.gemini/settings.json`` (the home is ``$GEMINI_CLI_HOME``
  when set). **workspace**: ``<cwd>/.gemini/settings.json``, the working
  directory only, skipped when it is the home. **system**:
  ``$GEMINI_CLI_SYSTEM_SETTINGS_PATH``, else
  ``/Library/Application Support/GeminiCli/settings.json`` (macOS),
  ``C:\\ProgramData\\gemini-cli\\settings.json`` (Windows),
  ``/etc/gemini-cli/settings.json``; **system defaults**:
  ``$GEMINI_CLI_SYSTEM_DEFAULTS_PATH``, else ``system-defaults.json`` beside
  the system file.
- Precedence, lowest first: system defaults < user < workspace < system.
  ``mcpServers`` is a shallow merge, so a same-named entry is replaced whole.
- A workspace's settings load only in a trusted folder:
  ``~/.gemini/trustedFolders.json`` (``$GEMINI_CLI_TRUSTED_FOLDERS_PATH``),
  ``{"<path>": "TRUST_FOLDER" | "TRUST_PARENT" | "DO_NOT_TRUST"}``, the
  longest matching path winning and no match meaning untrusted, unless
  ``security.folderTrust.enabled`` is false. ``GEMINI_CLI_TRUST_WORKSPACE``
  and ``GEMINI_RESTRICTED_MODE`` decide first.
- **extensions**: ``~/.gemini/extensions/<name>/gemini-extension.json``
  ``mcpServers`` (plain JSON), with ``${extensionPath}``, ``${workspacePath}``,
  ``${/}`` filled by Gemini; settings entries override them.
- Entry: ``httpUrl`` (Streamable HTTP, deprecated), ``url`` with ``type``
  ``"http"`` or ``"sse"`` (no type: Streamable HTTP, falling back to SSE),
  ``command`` (stdio), ``tcp`` (WebSocket); ``headers``, ``oauth``
  (``clientId``, ``clientSecret``, ``scopes``, ...), ``authProviderType``,
  ``includeTools``/``excludeTools``, ``trust`` (skips confirmations; not an
  on/off switch).
- Placeholders: every string is expanded at load, ``$VAR``, ``${VAR}`` and
  ``${VAR:-default}``.
- What can turn a server off: ``admin.mcp.enabled: false``, ``mcp.allowed``
  (only these load) and ``mcp.excluded`` (the highest file that sets each
  wins), and ``~/.gemini/mcp-server-enablement.json``
  (``{"<lowercase name>": {"enabled": false}}``, from ``gemini mcp
  disable``). Names match lowercased and trimmed.
- settings.json allows comments (strip-json-comments).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rote.clients.base import ClientContext, ImportCandidate, Scope, SourceReading
from rote.clients.common import (
    STDIO_BLOCKED,
    WEBSOCKET_BLOCKED,
    Reader,
    blocked,
    expand_home,
    new_candidate,
    not_an_object,
    rank,
    remote,
    str_list,
    switch_off,
    trust_failed,
    trust_verified,
)
from rote.clients.normalize import PlaceholderSyntax

CLIENT_ID = "gemini"

_PATTERN = re.compile(
    r"\$\{(?P<var>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^}]*))?\}|\$(?P<bare>[A-Za-z_][A-Za-z0-9_]*)"
)
GEMINI_SYNTAX = PlaceholderSyntax(client=CLIENT_ID, pattern=_PATTERN)
#: An extension's manifest adds the variables Gemini fills in itself.
GEMINI_EXTENSION_SYNTAX = PlaceholderSyntax(
    client=CLIENT_ID,
    pattern=_PATTERN,
    internal=frozenset({"extensionPath", "workspacePath", "pathSeparator"}),
    other=re.compile(r"\$\{/\}"),
)

LIMITS = (
    "the extension on/off state (extensions/extension-enablement.json) and admin settings "
    "delivered by Google are not read: every installed extension's servers are listed",
    "a settings entry that overrides an extension's server field by field is listed as "
    "replacing it",
)


def _gemini_dir(ctx: ClientContext) -> Path:
    home = ctx.env.get("GEMINI_CLI_HOME", "").strip()
    return (Path(home) if home else ctx.home) / ".gemini"


def _system_file(ctx: ClientContext) -> Path | None:
    override = ctx.env.get("GEMINI_CLI_SYSTEM_SETTINGS_PATH", "").strip()
    if override:
        return Path(override)
    if ctx.platform == "darwin":
        return ctx.system_dir / "GeminiCli" / "settings.json" if ctx.system_dir else None
    return ctx.etc_dir / "gemini-cli" / "settings.json" if ctx.etc_dir else None


def _defaults_file(ctx: ClientContext, system: Path | None) -> Path | None:
    override = ctx.env.get("GEMINI_CLI_SYSTEM_DEFAULTS_PATH", "").strip()
    if override:
        return Path(override)
    return system.parent / "system-defaults.json" if system is not None else None


def _workspace_file(ctx: ClientContext) -> Path | None:
    gemini_home = _gemini_dir(ctx).parent
    if ctx.cwd.resolve() == gemini_home.resolve():
        return None
    return ctx.cwd / ".gemini" / "settings.json"


def _nested(data: Mapping[str, Any], *keys: str) -> Any:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _first(layers: Sequence[Mapping[str, Any]], *keys: str) -> Any:
    """The value from the highest layer that sets it."""
    for layer in layers:
        value = _nested(layer, *keys)
        if value is not None:
            return value
    return None


def _trusted(ctx: ClientContext, reader: Reader, folder_trust: Any) -> bool:
    env = ctx.env
    if (
        env.get("GEMINI_RESTRICTED_MODE") == "true"
        or env.get("GEMINI_CLI_TRUST_WORKSPACE") == "false"
    ):
        return False
    if env.get("GEMINI_CLI_TRUST_WORKSPACE") == "true" or folder_trust is False:
        return True
    override = env.get("GEMINI_CLI_TRUSTED_FOLDERS_PATH", "").strip()
    path = Path(override) if override else _gemini_dir(ctx) / "trustedFolders.json"
    rules = reader.mapping(path, "jsonc")
    cwd = ctx.cwd.resolve()
    best: tuple[int, bool] | None = None
    for raw, verdict in rules.items():
        root = expand_home(raw, ctx)
        if verdict == "TRUST_PARENT":
            root = root.parent
        if cwd != root and root not in cwd.parents:
            continue
        depth = len(Path(raw).parts)
        if best is None or depth > best[0]:
            best = (depth, verdict in ("TRUST_FOLDER", "TRUST_PARENT"))
    return best is not None and best[1]


def _candidate(
    name: str,
    entry: Any,
    *,
    scope: Scope,
    origin: Path,
    syntax: PlaceholderSyntax = GEMINI_SYNTAX,
    plugin: str | None = None,
) -> ImportCandidate:
    base = new_candidate(CLIENT_ID, name, scope=scope, origin=origin, plugin=plugin)
    if not isinstance(entry, dict):
        return not_an_object(base, entry)
    if "httpUrl" in entry:
        url, transport = entry["httpUrl"], "streamable-http"
    elif "url" in entry:
        kind = entry.get("type")
        if kind not in (None, "http", "sse"):
            return blocked(base, "other", f"type {kind!r} is not a server rote can reach")
        url, transport = entry["url"], "sse" if kind == "sse" else "streamable-http"
    elif "command" in entry:
        return blocked(base, "stdio", STDIO_BLOCKED)
    elif "tcp" in entry:
        return blocked(base, "websocket", WEBSOCKET_BLOCKED)
    else:
        return blocked(base, "other", "the entry has no httpUrl, url, command or tcp")
    withheld: list[str] = []
    scopes: list[str] = []
    oauth = entry.get("oauth")
    if isinstance(oauth, dict):
        if "clientId" in oauth:
            withheld.append(
                "oauth.clientId: issued to Gemini CLI; rote registers its own client at "
                f"rote mcp login {base.name}"
            )
        if "clientSecret" in oauth:
            withheld.append("oauth.clientSecret: a secret issued to Gemini CLI, never copied")
        scopes = str_list(oauth.get("scopes"))
    provider = entry.get("authProviderType")
    if provider in ("google_credentials", "service_account_impersonation"):
        withheld.append(
            f"authProviderType {provider}: Gemini CLI signs in with Google credentials, which "
            f"rote does not; rote mcp login {base.name} uses the server's OAuth"
        )
    if entry.get("includeTools") or entry.get("excludeTools"):
        withheld.append(
            "includeTools/excludeTools: rote imports the whole server; the filter is not copied"
        )
    return remote(
        base,
        url=url,
        transport="sse" if transport == "sse" else "streamable-http",
        syntax=syntax,
        headers=entry.get("headers"),
        withheld=withheld,
        scopes=scopes,
    )


class GeminiSource:
    id = CLIENT_ID
    label = "Gemini CLI"

    def config_paths(self, ctx: ClientContext) -> tuple[Path, ...]:
        system = _system_file(ctx)
        defaults = _defaults_file(ctx, system)
        workspace = _workspace_file(ctx)
        directory = _gemini_dir(ctx)
        return tuple(
            p
            for p in (
                system,
                workspace,
                directory / "settings.json",
                defaults,
                directory / "trustedFolders.json",
                directory / "mcp-server-enablement.json",
            )
            if p is not None
        )

    def detect(self, ctx: ClientContext) -> bool:
        return _gemini_dir(ctx).is_dir()

    def read(self, ctx: ClientContext) -> SourceReading:
        reader = Reader()
        directory = _gemini_dir(ctx)
        system_path = _system_file(ctx)
        defaults_path = _defaults_file(ctx, system_path)
        workspace_path = _workspace_file(ctx)
        user_path = directory / "settings.json"
        system = reader.mapping(system_path, "jsonc") if system_path else {}
        defaults = reader.mapping(defaults_path, "jsonc") if defaults_path else {}
        user = reader.mapping(user_path, "jsonc")
        workspace = reader.mapping(workspace_path, "jsonc") if workspace_path else {}

        trusted = bool(workspace) and _trusted(
            ctx, reader, _first((system, user, defaults), "security", "folderTrust", "enabled")
        )
        effective = (system, *((workspace,) if trusted else ()), user, defaults)

        groups: list[tuple[str, list[ImportCandidate]]] = []
        files: list[tuple[str, Scope, Path | None, Mapping[str, Any]]] = [
            ("system", "system", system_path, system),
            ("workspace", "project", workspace_path, workspace),
            ("user", "user", user_path, user),
            ("system defaults", "system", defaults_path, defaults),
        ]
        for label, scope, path, data in files:
            if path is None:
                continue
            servers = reader.section(path, data, "mcpServers")
            found = [_candidate(n, e, scope=scope, origin=path) for n, e in servers.items()]
            if label == "workspace":
                found = trust_verified(found) if trusted else trust_failed(found, _UNTRUSTED)
            groups.append((label, found))
        groups.append(("extension", self._extension_candidates(reader, directory)))
        ranked = rank(groups)

        enablement = reader.mapping(directory / "mcp-server-enablement.json", "jsonc")
        allowed = _first(effective, "mcp", "allowed")
        excluded = _first(effective, "mcp", "excluded")
        admin_off = _first(effective, "admin", "mcp", "enabled") is False
        allowed_names = {n.strip().lower() for n in str_list(allowed)}
        excluded_names = {n.strip().lower() for n in str_list(excluded)}
        off: dict[tuple[Path, str], str] = {}
        for c in ranked:
            key = c.original_name.strip().lower()
            state = enablement.get(key)
            why = None
            if admin_off:
                why = "MCP is turned off by admin.mcp.enabled in Gemini CLI's settings"
            elif isinstance(allowed, list) and key not in allowed_names:
                why = "not in Gemini CLI's mcp.allowed list"
            elif key in excluded_names:
                why = "in Gemini CLI's mcp.excluded list"
            elif isinstance(state, dict) and state.get("enabled") is False:
                why = "disabled in Gemini CLI (gemini mcp disable)"
            if why is not None:
                off[(c.origin, c.original_name)] = why
        return SourceReading(
            client=CLIENT_ID,
            candidates=tuple(switch_off(ranked, off)),
            files_read=tuple(reader.files_read),
            problems=tuple(reader.problems),
            limits=LIMITS,
        )

    @staticmethod
    def _extension_candidates(reader: Reader, directory: Path) -> list[ImportCandidate]:
        root = directory / "extensions"
        if not root.is_dir():
            return []
        found: list[ImportCandidate] = []
        for manifest in sorted(root.glob("*/gemini-extension.json")):
            servers = reader.section(manifest, reader.mapping(manifest), "mcpServers")
            found += [
                _candidate(
                    n,
                    e,
                    scope="plugin",
                    origin=manifest,
                    syntax=GEMINI_EXTENSION_SYNTAX,
                    plugin=manifest.parent.name,
                )
                for n, e in servers.items()
            ]
        return found


_UNTRUSTED = "in a folder Gemini CLI does not trust; it loads none of this folder's settings"
