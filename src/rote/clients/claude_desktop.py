"""Claude Desktop's MCP servers, read from ``claude_desktop_config.json``.

Verified 2026-10-22 against
https://modelcontextprotocol.io/docs/develop/connect-local-servers (source:
github.com/modelcontextprotocol/modelcontextprotocol,
docs/docs/2026-07-28/develop/connect-local-servers.mdx),
https://modelcontextprotocol.io/docs/develop/connect-remote-servers and
https://support.claude.com/en/articles/11175166-getting-started-with-custom-connectors-using-remote-mcp:

- The file is ``~/Library/Application Support/Claude/claude_desktop_config.json``
  (macOS) or ``%APPDATA%\\Claude\\claude_desktop_config.json`` (Windows).
  The docs give no Linux path; ``$XDG_CONFIG_HOME/Claude/`` (else
  ``~/.config/Claude/``) is where VS Code's discovery looks, so it is read
  there.
- ``mcpServers`` holds local servers only: ``command``, ``args``, ``env``.
  The file also holds ``preferences`` and other app state.
- Remote servers are custom connectors, added under Settings > Connectors
  and stored in the claude.ai account ("from Anthropic's cloud
  infrastructure, rather than from your local device"). They are not in this
  file, so this source lists every entry as not importable and says where
  the remote ones are.
- No placeholder syntax and no on/off flag in the file are documented.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rote.clients.base import ClientContext, ImportCandidate, SourceReading
from rote.clients.common import (
    STDIO_BLOCKED,
    Reader,
    appdata,
    blocked,
    mac_app_support,
    new_candidate,
    not_an_object,
    xdg_config_home,
)

CLIENT_ID = "claude-desktop"

LIMITS = (
    "remote MCP servers (custom connectors) live in your claude.ai account, not in this "
    "file, and are not listed; add one with rote mcp add <name> <url>",
    "Desktop Extensions (.mcpb) are local servers and are not read",
)


def _config_dir(ctx: ClientContext) -> Path:
    if ctx.platform == "darwin":
        return mac_app_support(ctx) / "Claude"
    if ctx.platform == "win32":
        return appdata(ctx) / "Claude"
    return xdg_config_home(ctx) / "Claude"


def _candidate(name: str, entry: Any, origin: Path) -> ImportCandidate:
    base = new_candidate(CLIENT_ID, name, scope="user", origin=origin)
    if not isinstance(entry, dict):
        return not_an_object(base, entry)
    if "command" in entry:
        return blocked(base, "stdio", STDIO_BLOCKED)
    return blocked(
        base,
        "other",
        "Claude Desktop runs only local commands from this file; a remote server is a "
        "connector in your claude.ai account",
    )


class ClaudeDesktopSource:
    id = CLIENT_ID
    label = "Claude Desktop"

    def config_paths(self, ctx: ClientContext) -> tuple[Path, ...]:
        return (_config_dir(ctx) / "claude_desktop_config.json",)

    def detect(self, ctx: ClientContext) -> bool:
        return _config_dir(ctx).is_dir()

    def read(self, ctx: ClientContext) -> SourceReading:
        reader = Reader()
        [path] = self.config_paths(ctx)
        servers = reader.section(path, reader.mapping(path), "mcpServers")
        return SourceReading(
            client=CLIENT_ID,
            candidates=tuple(_candidate(n, e, path) for n, e in servers.items()),
            files_read=tuple(reader.files_read),
            problems=tuple(reader.problems),
            limits=LIMITS,
        )
