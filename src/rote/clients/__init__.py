"""Read-only readers of other AI clients' MCP configs (``rote mcp import``'s input).

Each client is one :class:`~rote.clients.base.ClientSource` in
:data:`SOURCES`. A new client is a new module implementing the protocol
plus one line here; nothing else changes. See
``docs/design/first-run.md``.
"""

from __future__ import annotations

from rote.clients.base import (
    ClientContext,
    ClientSource,
    HeaderTemplate,
    ImportCandidate,
    SourceReading,
)
from rote.clients.claude_code import ClaudeCodeSource
from rote.clients.collisions import Collision, find_collisions

__all__ = [
    "SOURCES",
    "ClientContext",
    "ClientSource",
    "Collision",
    "HeaderTemplate",
    "ImportCandidate",
    "SourceReading",
    "find_collisions",
    "ordered_sources",
    "source",
]

#: Every client rote can import from, in the order they are offered when
#: none is detected.
SOURCES: tuple[ClientSource, ...] = (ClaudeCodeSource(),)


def source(client_id: str) -> ClientSource:
    for s in SOURCES:
        if s.id == client_id:
            return s
    known = ", ".join(s.id for s in SOURCES)
    raise KeyError(f"no MCP config reader for {client_id!r}; known: {known}")


def ordered_sources(detected: str | None) -> tuple[ClientSource, ...]:
    """:data:`SOURCES` with the detected client first."""
    return tuple(sorted(SOURCES, key=lambda s: s.id != detected))
