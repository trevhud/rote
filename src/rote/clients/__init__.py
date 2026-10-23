"""Read-only readers of other AI clients' MCP configs (``rote mcp import``'s input).

Each client is one :class:`~rote.clients.base.ClientSource` in
:data:`SOURCES`. A new client is a new module implementing the protocol
plus one line here; nothing else changes. See
``docs/design/first-run.md``.
"""

from __future__ import annotations

from rote.clients.amp import AmpSource
from rote.clients.base import (
    ClientContext,
    ClientSource,
    HeaderTemplate,
    ImportCandidate,
    SourceReading,
    Trust,
)
from rote.clients.claude_code import ClaudeCodeSource
from rote.clients.claude_desktop import ClaudeDesktopSource
from rote.clients.codex import CodexSource
from rote.clients.collisions import Collision, find_collisions
from rote.clients.copilot import CopilotSource
from rote.clients.cursor import CursorSource
from rote.clients.gemini import GeminiSource
from rote.clients.goose import GooseSource
from rote.clients.opencode import OpenCodeSource
from rote.clients.vscode import VSCodeSource
from rote.clients.windsurf import WindsurfSource
from rote.clients.zed import ZedSource

__all__ = [
    "SOURCES",
    "ClientContext",
    "ClientSource",
    "Collision",
    "HeaderTemplate",
    "ImportCandidate",
    "SourceReading",
    "Trust",
    "find_collisions",
    "ordered_sources",
    "source",
]

#: Every client rote can import from, in the order they are offered when
#: none is detected.
SOURCES: tuple[ClientSource, ...] = (
    ClaudeCodeSource(),
    CodexSource(),
    CursorSource(),
    VSCodeSource(),
    CopilotSource(),
    GeminiSource(),
    OpenCodeSource(),
    ZedSource(),
    AmpSource(),
    GooseSource(),
    WindsurfSource(),
    ClaudeDesktopSource(),
)


def source(client_id: str) -> ClientSource:
    for s in SOURCES:
        if s.id == client_id:
            return s
    known = ", ".join(s.id for s in SOURCES)
    raise KeyError(f"no MCP config reader for {client_id!r}; known: {known}")


def ordered_sources(
    detected: str | None, ctx: ClientContext | None = None
) -> tuple[ClientSource, ...]:
    """:data:`SOURCES` ranked for offering: the client the environment names
    (:func:`rote.interact.agents.detect`'s ``client``, caller or host) first,
    then, with ``ctx``, the clients whose config is on this machine
    (:meth:`ClientSource.detect`), then the rest, each group in
    :data:`SOURCES` order."""

    def rank(s: ClientSource) -> int:
        if s.id == detected:
            return 0
        if ctx is not None and s.detect(ctx):
            return 1
        return 2

    return tuple(sorted(SOURCES, key=rank))
