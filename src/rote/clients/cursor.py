"""Cursor's MCP servers, read from its ``mcp.json`` files.

Verified 2026-10-22 against https://cursor.com/docs/mcp.md (the canonical
page; cursor.com/docs/context/mcp still renders),
https://cursor.com/help/customization/mcp.md and
https://cursor.com/docs/reference/permissions.md:

- **user**: ``~/.cursor/mcp.json`` (the docs give no per-OS variant; on
  Windows the home is ``%USERPROFILE%``). No variable moves it.
- **project**: ``.cursor/mcp.json`` in the workspace. "Both files are
  merged. If the same server name appears in both, the project-level config
  takes priority" (help/customization/mcp): the whole entry wins.
- Shape: ``{"mcpServers": {name: entry}}``. stdio: ``type: "stdio"``
  (examples omit it), ``command``, ``args``, ``env``, ``envFile``. Remote:
  ``url``, ``headers``, and a static OAuth ``auth`` object
  (``CLIENT_ID``, ``CLIENT_SECRET``, ``scopes``). The docs list SSE and
  Streamable HTTP but no remote ``type`` value; Cursor picks the transport
  when it connects, so a remote entry is listed as Streamable HTTP unless
  its ``type`` says ``sse``.
- Interpolation in command, args, env, url, headers and auth:
  ``${env:NAME}``, ``${userHome}``, ``${workspaceFolder}``,
  ``${workspaceFolderBasename}``, ``${pathSeparator}``, ``${/}``
  (:data:`~rote.clients.normalize.CURSOR_SYNTAX`).
- On/off state is set in the Customize sidebar and is not in mcp.json; the
  docs do not say where it is kept, so it is not read (a limit).
- The docs show a ``//`` comment above an example, so the file is read as
  JSON with comments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rote.clients.base import ClientContext, ImportCandidate, Scope, SourceReading
from rote.clients.common import (
    STDIO_BLOCKED,
    Reader,
    blocked,
    new_candidate,
    not_an_object,
    project_root,
    rank,
    remote,
    str_list,
    trust_unknown,
)
from rote.clients.normalize import CURSOR_SYNTAX

CLIENT_ID = "cursor"

LIMITS = (
    "a server turned off in Cursor's Customize sidebar is still listed as active: Cursor "
    "keeps that switch outside mcp.json",
    "Cursor chooses SSE or Streamable HTTP when it connects; a remote entry without "
    'type "sse" is listed as Streamable HTTP',
    "servers an extension registers (vscode.cursor.mcp.registerServer) and team servers "
    "from the Cursor dashboard are not in any file and are not listed",
)


#: Cursor's docs do not say whether a project's mcp.json loads before the
#: person approves it, nor where such an answer would be kept.
_PROJECT_TRUST = (
    "whether Cursor loads a project's .cursor/mcp.json without asking is not documented"
)


def _user_file(ctx: ClientContext) -> Path:
    return ctx.home / ".cursor" / "mcp.json"


def _project_file(ctx: ClientContext) -> Path:
    return project_root(ctx.cwd) / ".cursor" / "mcp.json"


def _candidate(name: str, entry: Any, *, scope: Scope, origin: Path) -> ImportCandidate:
    base = new_candidate(CLIENT_ID, name, scope=scope, origin=origin)
    if not isinstance(entry, dict):
        return not_an_object(base, entry)
    kind = entry.get("type")
    if kind == "stdio" or ("command" in entry and "url" not in entry):
        return blocked(base, "stdio", STDIO_BLOCKED)
    if kind not in (None, "http", "streamable-http", "sse"):
        return blocked(base, "other", f"type {kind!r} is not a server rote can reach")
    withheld: list[str] = []
    scopes: list[str] = []
    auth = entry.get("auth")
    if isinstance(auth, dict):
        if "CLIENT_ID" in auth:
            withheld.append(
                f"auth.CLIENT_ID: issued to Cursor; rote registers its own client at "
                f"rote mcp login {base.name}"
            )
        if "CLIENT_SECRET" in auth:
            withheld.append("auth.CLIENT_SECRET: a secret issued to Cursor, never copied")
        scopes = str_list(auth.get("scopes"))
    return remote(
        base,
        url=entry.get("url"),
        transport="sse" if kind == "sse" else "streamable-http",
        syntax=CURSOR_SYNTAX,
        headers=entry.get("headers"),
        withheld=withheld,
        scopes=scopes,
    )


class CursorSource:
    id = CLIENT_ID
    label = "Cursor"

    def config_paths(self, ctx: ClientContext) -> tuple[Path, ...]:
        return (_project_file(ctx), _user_file(ctx))

    def detect(self, ctx: ClientContext) -> bool:
        return (ctx.home / ".cursor").is_dir() or _project_file(ctx).is_file()

    def read(self, ctx: ClientContext) -> SourceReading:
        reader = Reader()
        groups: list[tuple[str, list[ImportCandidate]]] = []
        files: tuple[tuple[Scope, Path], ...] = (
            ("project", _project_file(ctx)),
            ("user", _user_file(ctx)),
        )
        for scope, path in files:
            data = reader.mapping(path, "jsonc")
            servers = reader.section(path, data, "mcpServers")
            found = [_candidate(n, e, scope=scope, origin=path) for n, e in servers.items()]
            if scope == "project":
                found = trust_unknown(found, self.label, _PROJECT_TRUST)
            groups.append((scope, found))
        return SourceReading(
            client=CLIENT_ID,
            candidates=tuple(rank(groups)),
            files_read=tuple(reader.files_read),
            problems=tuple(reader.problems),
            limits=LIMITS,
        )
