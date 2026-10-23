"""GitHub Copilot CLI's MCP servers, and the portable format it defines.

Verified 2026-10-22 against
https://docs.github.com/en/copilot/how-tos/copilot-cli/customize-copilot/add-mcp-servers,
https://docs.github.com/en/copilot/reference/copilot-cli-reference/cli-command-reference
and https://docs.github.com/en/copilot/reference/copilot-cli-reference/cli-config-dir-reference:

- **user**: ``~/.copilot/mcp-config.json``; ``COPILOT_HOME`` replaces all of
  ``~/.copilot``. ``{"mcpServers": {...}}``. VS Code reads this file with a
  strict ``JSON.parse`` (mcpCopilotGlobalConfiguration.ts), so it is read as
  plain JSON.
- **project**: ``.mcp.json`` or ``.github/mcp.json`` in each directory from
  the working directory up to the git root; ``.mcp.json`` wins in one
  directory, a closer file wins on a name, and project files beat the user
  file. Either may be wrapped in ``mcpServers`` or bare. ``.vscode/mcp.json``
  is not read. Project files load only in a trusted folder.
- Entry ``type``: ``"local"`` (the default) or ``"stdio"``; ``"http"``
  (``"streamable-http"`` accepted); ``"sse"``. Remote keys: ``url``,
  ``headers``, ``oauthClientId``, ``oauthScopes``, ``oauthPublicClient``,
  ``oauthGrantType``, ``oidc``, ``tools`` (``["*"]`` or names).
- Placeholders: ``$VAR``, ``${VAR}``, ``${VAR:-default}`` in env and headers
  (:data:`~rote.clients.normalize.COPILOT_SYNTAX`).
- Off switch: ``disabledMcpServers`` (names) in ``~/.copilot/settings.json``
  (JSONC), written by ``copilot mcp disable``. There is no key on the entry.

:func:`portable_candidate` reads one entry of this format for any client
that uses it (VS Code reads the workspace ``.mcp.json`` and this user file
too).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rote.clients.base import ClientContext, ImportCandidate, Scope, SourceReading
from rote.clients.common import (
    STDIO_BLOCKED,
    WEBSOCKET_BLOCKED,
    Reader,
    blocked,
    new_candidate,
    not_an_object,
    rank,
    remote,
    servers_in,
    str_list,
    switch_off,
    trust_unknown,
)
from rote.clients.normalize import COPILOT_SYNTAX

CLIENT_ID = "copilot"

LIMITS = (
    "plugin servers, built-in servers and --additional-mcp-config are not listed",
    "a repository's own Copilot settings can add disabledMcpServers entries; only "
    "~/.copilot/settings.json is read",
)

_REMOTE = {"http": "streamable-http", "streamable-http": "streamable-http", "sse": "sse"}


def copilot_home(ctx: ClientContext) -> Path:
    override = ctx.env.get("COPILOT_HOME", "").strip()
    return Path(override) if override else ctx.home / ".copilot"


def portable_candidate(
    client: str,
    label: str,
    name: str,
    entry: Any,
    *,
    scope: Scope,
    origin: Path,
    infer_untyped: bool,
) -> ImportCandidate:
    """One entry of the portable Copilot format.

    ``infer_untyped``: an entry without ``type`` is remote when it has a
    ``url`` and no ``command`` (VS Code's reader); otherwise it is local,
    Copilot CLI's default."""
    base = new_candidate(client, name, scope=scope, origin=origin)
    if not isinstance(entry, dict):
        return not_an_object(base, entry)
    kind = entry.get("type")
    if kind is None and infer_untyped and "url" in entry and "command" not in entry:
        kind = "http"
    if kind is None and not infer_untyped and "url" in entry and "command" not in entry:
        return blocked(
            base, "other", f"the entry has a url but no type, so {label} runs it as a local command"
        )
    if kind is None or kind in ("local", "stdio"):
        return blocked(base, "stdio", STDIO_BLOCKED)
    if kind == "ws":
        return blocked(base, "websocket", WEBSOCKET_BLOCKED)
    transport = _REMOTE.get(str(kind))
    if transport is None:
        return blocked(base, "other", f"type {kind!r} is not a server rote can reach")
    withheld: list[str] = []
    if "oauthClientId" in entry:
        withheld.append(
            f"oauthClientId: issued to {label}; rote registers its own client at "
            f"rote mcp login {base.name}"
        )
    if entry.get("oauthGrantType") == "client_credentials":
        withheld.append(
            "oauthGrantType client_credentials: a machine credential rote does not copy; "
            f"rote mcp login {base.name} signs in as you"
        )
    if "oidc" in entry:
        withheld.append("oidc: an identity-token exchange set up for this client, not copied")
    tools = entry.get("tools")
    if isinstance(tools, list) and tools != ["*"]:
        withheld.append("tools: rote imports the whole server; the tool filter is not copied")
    raw_scopes = entry.get("oauthScopes")
    scopes = raw_scopes.split() if isinstance(raw_scopes, str) else str_list(raw_scopes)
    return remote(
        base,
        url=entry.get("url"),
        transport="sse" if transport == "sse" else "streamable-http",
        syntax=COPILOT_SYNTAX,
        headers=entry.get("headers"),
        withheld=withheld,
        scopes=scopes,
    )


def _project_files(cwd: Path) -> list[Path]:
    """``.mcp.json`` then ``.github/mcp.json`` per directory, closest first,
    up to the git root."""
    found: list[Path] = []
    for directory in (cwd, *cwd.parents):
        found += [directory / ".mcp.json", directory / ".github" / "mcp.json"]
        if (directory / ".git").exists():
            return found
    # Not in a repository: Copilot looks in the working directory only.
    return found[:2]


class CopilotSource:
    id = CLIENT_ID
    label = "GitHub Copilot CLI"

    def config_paths(self, ctx: ClientContext) -> tuple[Path, ...]:
        home = copilot_home(ctx)
        return (*_project_files(ctx.cwd), home / "mcp-config.json", home / "settings.json")

    def detect(self, ctx: ClientContext) -> bool:
        return copilot_home(ctx).is_dir()

    def read(self, ctx: ClientContext) -> SourceReading:
        reader = Reader()
        groups: list[tuple[str, list[ImportCandidate]]] = []
        for path in _project_files(ctx.cwd):
            data = reader.load(path, "jsonc")
            if data is None:
                continue
            servers = servers_in(data)
            if servers is None:
                reader.problems.append(f"{path}: no mcpServers object")
                continue
            found = trust_unknown(
                self._candidates(servers, "project", path),
                self.label,
                "Copilot CLI loads project MCP files only in a trusted folder, and the docs do "
                "not say where it keeps that answer",
            )
            groups.append(("project", found))
        home = copilot_home(ctx)
        user_path = home / "mcp-config.json"
        user = reader.section(user_path, reader.mapping(user_path), "mcpServers")
        groups.append(("user", self._candidates(user, "user", user_path)))

        settings = reader.mapping(home / "settings.json", "jsonc")
        disabled = set(str_list(settings.get("disabledMcpServers")))
        ranked = rank(groups)
        off = {
            (c.origin, c.original_name): "disabled in Copilot CLI (copilot mcp disable)"
            for c in ranked
            if c.original_name in disabled
        }
        return SourceReading(
            client=CLIENT_ID,
            candidates=tuple(switch_off(ranked, off)),
            files_read=tuple(reader.files_read),
            problems=tuple(reader.problems),
            limits=LIMITS,
        )

    def _candidates(
        self, servers: dict[str, Any], scope: Scope, origin: Path
    ) -> list[ImportCandidate]:
        return [
            portable_candidate(
                CLIENT_ID, self.label, n, e, scope=scope, origin=origin, infer_untyped=False
            )
            for n, e in servers.items()
        ]
