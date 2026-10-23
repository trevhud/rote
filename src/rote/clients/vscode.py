"""VS Code's MCP servers (GitHub Copilot agent mode), read from its files.

Verified 2026-10-22 against
https://code.visualstudio.com/docs/agents/reference/mcp-configuration (the
old docs/copilot/reference URL redirects there),
https://code.visualstudio.com/docs/agent-customization/mcp-servers,
https://code.visualstudio.com/docs/agents/run/manage-mcp-servers,
https://code.visualstudio.com/docs/reference/variables-reference, and the
VS Code source on main (``src/vs/platform/environment/node/userDataPath.ts``,
``src/vs/workbench/contrib/mcp/common/mcpTypes.ts``,
``src/vs/platform/mcp/common/mcpWorkspaceConfiguration.ts``,
``src/vs/platform/agentPlugins/common/pluginParsers.ts``):

- **workspace**: ``.vscode/mcp.json`` (``{"servers": {...}, "inputs": [...]}``)
  and the workspace-root ``.mcp.json`` in the portable Copilot format
  (``mcpServers`` or bare; see :mod:`rote.clients.copilot`).
- **user**: ``<user data>/User/mcp.json``, where the user data dir is
  ``~/Library/Application Support/Code`` (macOS), ``$XDG_CONFIG_HOME/Code``
  or ``~/.config/Code`` (Linux), ``%APPDATA%\\Code`` (Windows);
  ``VSCODE_PORTABLE`` (``<dir>/user-data``) and ``VSCODE_APPDATA``
  (``<dir>/Code``) move it. Insiders uses ``Code - Insiders``.
- Precedence (``McpCollectionSortOrder``): workspace folder, then workspace,
  then user. Duplicate names match case-insensitively and the lower one is
  disabled (``chat.mcp.collisionBehavior`` default ``"disable"``).
- Entry ``type``: ``"stdio"``, ``"http"`` (Streamable HTTP, falling back to
  SSE), ``"sse"``; ``"streamable-http"`` is read as http, no type is
  inferred from ``command`` vs ``url``, ``"ws"`` is rejected. Remote keys:
  ``url`` (also ``unix:///`` and ``pipe:///``), ``headers``,
  ``oauth.clientId`` (or ``oauthClientId``).
- Placeholders: ``${env:NAME}``, ``${input:id}``, ``${config:...}``,
  ``${command:...}``, ``${workspaceFolder}``, ``${userHome}``...
  (:data:`~rote.clients.normalize.VSCODE_SYNTAX`); only ``${env:...}``
  can be carried.
- Both mcp.json files are JSONC (the schema allows comments and trailing
  commas). The ``mcp`` key in settings.json is legacy: VS Code moves it
  into the user mcp.json and deletes it.
- On/off state is in VS Code's storage (``mcp.enablement``), not in a file;
  ``chat.mcp.access: "none"`` in user settings turns every server off.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from rote.clients.base import ClientContext, ImportCandidate, Scope, SourceReading
from rote.clients.common import (
    STDIO_BLOCKED,
    WEBSOCKET_BLOCKED,
    Reader,
    appdata,
    blocked,
    mac_app_support,
    new_candidate,
    not_an_object,
    project_root,
    rank,
    remote,
    servers_in,
    trust_unknown,
    xdg_config_home,
)
from rote.clients.copilot import portable_candidate
from rote.clients.normalize import VSCODE_SYNTAX

CLIENT_ID = "vscode"
_LABEL = "VS Code"

LIMITS = (
    "a server switched off in VS Code (kept in its storage, not a file) is still listed as active",
    "named profiles (User/profiles/<id>/mcp.json), remote and dev container configs, and "
    "servers extensions register are not read",
    "servers VS Code discovers from other clients (chat.mcp.discovery.enabled, Copilot CLI's "
    "file by default) are listed under those clients",
)

_PRODUCTS = (("user", "Code"), ("Insiders user", "Code - Insiders"))


def _user_data_dir(ctx: ClientContext, product: str) -> Path:
    portable = ctx.env.get("VSCODE_PORTABLE", "").strip()
    if portable:
        return Path(portable) / "user-data"
    moved = ctx.env.get("VSCODE_APPDATA", "").strip()
    if moved:
        return Path(moved) / product
    if ctx.platform == "darwin":
        return mac_app_support(ctx) / product
    if ctx.platform == "win32":
        return appdata(ctx) / product
    return xdg_config_home(ctx) / product


def _vscode_candidate(name: str, entry: Any, *, scope: Scope, origin: Path) -> ImportCandidate:
    base = new_candidate(CLIENT_ID, name, scope=scope, origin=origin)
    if not isinstance(entry, dict):
        return not_an_object(base, entry)
    kind = entry.get("type")
    if kind is None:
        kind = "stdio" if "command" in entry else "http"
    if kind == "stdio":
        return blocked(base, "stdio", STDIO_BLOCKED)
    if kind == "ws":
        return blocked(base, "websocket", WEBSOCKET_BLOCKED)
    if kind not in ("http", "streamable-http", "sse"):
        return blocked(base, "other", f"type {kind!r} is not a server rote can reach")
    withheld: list[str] = []
    oauth = entry.get("oauth")
    if "oauthClientId" in entry or (isinstance(oauth, dict) and "clientId" in oauth):
        withheld.append(
            f"oauth.clientId: issued to VS Code; rote registers its own client at "
            f"rote mcp login {base.name}"
        )
    sse = kind == "sse" or entry.get("transport") == "sse"
    return remote(
        base,
        url=entry.get("url"),
        transport="sse" if sse else "streamable-http",
        syntax=VSCODE_SYNTAX,
        headers=entry.get("headers"),
        withheld=withheld,
    )


class VSCodeSource:
    id = CLIENT_ID
    label = _LABEL

    def _workspace_files(self, ctx: ClientContext) -> tuple[Path, Path]:
        root = project_root(ctx.cwd)
        return root / ".vscode" / "mcp.json", root / ".mcp.json"

    def config_paths(self, ctx: ClientContext) -> tuple[Path, ...]:
        users = tuple(_user_data_dir(ctx, product) / "User" for _, product in _PRODUCTS)
        return (
            *self._workspace_files(ctx),
            *(u / "mcp.json" for u in users),
            *(u / "settings.json" for u in users),
        )

    def detect(self, ctx: ClientContext) -> bool:
        return any(_user_data_dir(ctx, product).is_dir() for _, product in _PRODUCTS) or (
            self._workspace_files(ctx)[0].is_file()
        )

    def read(self, ctx: ClientContext) -> SourceReading:
        reader = Reader()
        folder_file, root_file = self._workspace_files(ctx)
        folder = reader.section(folder_file, reader.mapping(folder_file, "jsonc"), "servers")
        workspace: list[ImportCandidate] = [
            _vscode_candidate(n, e, scope="project", origin=folder_file) for n, e in folder.items()
        ]
        root_data = reader.load(root_file, "jsonc")
        root_servers = servers_in(root_data) if root_data is not None else {}
        if root_servers is None:
            reader.problems.append(f"{root_file}: no mcpServers object")
        workspace += [
            portable_candidate(
                CLIENT_ID, _LABEL, n, e, scope="project", origin=root_file, infer_untyped=True
            )
            for n, e in (root_servers or {}).items()
        ]
        workspace = trust_unknown(
            workspace,
            _LABEL,
            "VS Code loads workspace servers only in a trusted folder (not Restricted Mode), "
            "and keeps that answer in its storage",
        )

        candidates: list[ImportCandidate] = []
        for label, product in _PRODUCTS:
            user_dir = _user_data_dir(ctx, product) / "User"
            user_file = user_dir / "mcp.json"
            servers = reader.section(user_file, reader.mapping(user_file, "jsonc"), "servers")
            user = [
                _vscode_candidate(n, e, scope="user", origin=user_file) for n, e in servers.items()
            ]
            # Stable and Insiders are separate apps sharing the workspace
            # files: each user file ranks against the workspace, and the
            # workspace entries are listed once, with stable's verdict.
            ranked = rank((("workspace", workspace), (label, user)), fold_case=True)
            if candidates:
                ranked = ranked[len(workspace) :]
            settings_file = user_dir / "settings.json"
            settings = reader.mapping(settings_file, "jsonc")
            if "mcp" in settings:
                reader.problems.append(
                    f"{settings_file}: has a legacy mcp key, which VS Code moves into "
                    "mcp.json on start; it is not read"
                )
            if settings.get("chat.mcp.access") == "none":
                reason = f'chat.mcp.access is "none" in {product}\'s user settings'
                ranked = [replace(c, inactive=c.inactive or reason) for c in ranked]
            candidates += ranked
        return SourceReading(
            client=CLIENT_ID,
            candidates=tuple(candidates),
            files_read=tuple(reader.files_read),
            problems=tuple(reader.problems),
            limits=LIMITS,
        )
