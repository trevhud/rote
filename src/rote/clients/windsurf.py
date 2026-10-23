"""Windsurf's MCP servers, which Devin Desktop (its 2026 successor) now reads.

Windsurf became Devin Desktop from Cognition: docs.windsurf.com's MCP page
redirects to https://docs.devin.ai/desktop/cascade/mcp, and the transition
FAQ is https://docs.devin.ai/desktop/devin-desktop-faq.md. Verified
2026-10-22 against those pages, https://docs.devin.ai/cli/extensibility/mcp/configuration.md,
https://docs.devin.ai/cli/reference/configuration/global-vs-local.md and a
Wayback copy of the old Windsurf page (2026-02-28):

- **user**: ``~/.config/devin/mcp_config.json`` (``$XDG_CONFIG_HOME/devin/``
  when set; ``%APPDATA%\\devin\\`` on Windows), shared by Cascade and the
  default "Devin Local" agent.
- **project**: ``.devin/mcp_config.json`` (committed) and **local**:
  ``.devin/mcp_config.local.json`` (gitignored). Precedence local > project
  > user, "merged by name (higher-priority source wins)".
- **legacy**: ``~/.codeium/windsurf/mcp_config.json`` (and ``windsurf-next``),
  where Windsurf kept servers before the rename; the FAQ also names
  ``~/.codeium/mcp_config.json``. The pages disagree on which one Cascade
  reads now, so all three are read, below the Devin files.
- Shape: ``{"mcpServers": {...}}``. stdio: ``command``, ``args``, ``env``.
  Remote: ``serverUrl`` or ``url``, ``headers``, ``transport``
  (``"http"``, the default, tries Streamable HTTP then SSE; ``"sse"``),
  ``oauthClientId``, ``oauthClientSecret``, ``oauthResource``. Both kinds
  take ``disabled: true`` (``devin mcp disable``) and ``disabledTools``.
- Interpolation in command, args, env, serverUrl, url and headers:
  ``${env:VAR}`` and ``${file:/path}``
  (:data:`~rote.clients.normalize.WINDSURF_SYNTAX`).
- The Devin config files are "JSON files (with comment support)".
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
    trust_unknown,
    xdg_config_home,
)
from rote.clients.normalize import WINDSURF_SYNTAX

CLIENT_ID = "windsurf"

LIMITS = (
    "Org and Team servers from the Devin admin console are not in any local file and are "
    "not listed",
    "the team MCP allowlist (windsurf.com team settings) is not read, so a server it blocks "
    "can still be listed",
    "servers Devin imports from other clients (read_config_from) are listed under those "
    "clients, not here",
)


def _devin_user_dir(ctx: ClientContext) -> Path:
    base = appdata(ctx) if ctx.platform == "win32" else xdg_config_home(ctx)
    return base / "devin"


def _files(ctx: ClientContext) -> tuple[tuple[str, Scope, Path], ...]:
    """(label, scope, path), highest precedence first."""
    root = project_root(ctx.cwd)
    codeium = ctx.home / ".codeium"
    return (
        ("local", "local", root / ".devin" / "mcp_config.local.json"),
        ("project", "project", root / ".devin" / "mcp_config.json"),
        ("user", "user", _devin_user_dir(ctx) / "mcp_config.json"),
        ("legacy Windsurf", "user", codeium / "windsurf" / "mcp_config.json"),
        ("legacy Windsurf Next", "user", codeium / "windsurf-next" / "mcp_config.json"),
        ("legacy Windsurf", "user", codeium / "mcp_config.json"),
    )


def _candidate(name: str, entry: Any, *, scope: Scope, origin: Path) -> ImportCandidate:
    base = new_candidate(CLIENT_ID, name, scope=scope, origin=origin)
    if not isinstance(entry, dict):
        return not_an_object(base, entry)
    url = entry.get("serverUrl", entry.get("url"))
    if url is None and "command" in entry:
        candidate = blocked(base, "stdio", STDIO_BLOCKED)
    else:
        transport = entry.get("transport")
        withheld: list[str] = []
        if "oauthClientId" in entry:
            withheld.append(
                "oauthClientId: issued to Devin/Windsurf; rote registers its own client at "
                f"rote mcp login {base.name}"
            )
        if "oauthClientSecret" in entry:
            withheld.append("oauthClientSecret: a secret issued to Devin/Windsurf, never copied")
        if entry.get("disabledTools"):
            withheld.append(
                "disabledTools: rote imports the whole server; the tool filter is not copied"
            )
        if transport not in (None, "http", "sse"):
            candidate = blocked(base, "other", f"transport {transport!r} is not one rote can reach")
        else:
            candidate = remote(
                base,
                url=url,
                transport="sse" if transport == "sse" else "streamable-http",
                syntax=WINDSURF_SYNTAX,
                headers=entry.get("headers"),
                withheld=withheld,
            )
    return candidate


class WindsurfSource:
    id = CLIENT_ID
    label = "Windsurf / Devin Desktop"

    def config_paths(self, ctx: ClientContext) -> tuple[Path, ...]:
        return tuple(path for _, _, path in _files(ctx))

    def detect(self, ctx: ClientContext) -> bool:
        return (
            _devin_user_dir(ctx).is_dir()
            or (ctx.home / ".codeium").is_dir()
            or any(path.is_file() for path in self.config_paths(ctx))
        )

    def read(self, ctx: ClientContext) -> SourceReading:
        reader = Reader()
        groups: list[tuple[str, list[ImportCandidate]]] = []
        off: dict[tuple[Path, str], str] = {}
        for label, scope, path in _files(ctx):
            servers = reader.section(path, reader.mapping(path, "jsonc"), "mcpServers")
            found = [_candidate(n, e, scope=scope, origin=path) for n, e in servers.items()]
            if scope in ("project", "local"):
                found = trust_unknown(
                    found,
                    self.label,
                    "whether Devin loads a repository's .devin/ servers without asking is not "
                    "documented",
                )
            groups.append((label, found))
            for n, e in servers.items():
                if isinstance(e, dict) and e.get("disabled") is True:
                    off[(path, n)] = "disabled in Devin/Windsurf (disabled: true)"
        # The whole entry wins by name, disabled or not.
        candidates = switch_off(rank(groups), off)
        return SourceReading(
            client=CLIENT_ID,
            candidates=tuple(candidates),
            files_read=tuple(reader.files_read),
            problems=tuple(reader.problems),
            limits=LIMITS,
        )
