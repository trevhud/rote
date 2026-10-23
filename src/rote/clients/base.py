"""The interface every AI-client config reader implements, and what it returns.

A :class:`ClientSource` reads one client's config files (Claude Code,
Codex, Cursor...) and turns each MCP server entry into an
:class:`ImportCandidate`, rote's one normalized shape. Sources only read:
they never run the client (``claude mcp list`` health-checks every
server), never contact a server, never write, and never resolve a
placeholder. Everything they need from the machine comes through
:class:`ClientContext`, so tests point them at fixture files in a tmp dir
and a source cannot read the real home by accident.

A candidate is display-safe by construction: :meth:`ImportCandidate.to_dict`
carries the host but not the URL (a path or query can hold a key), header
names and ``${VAR}`` templates but never a literal value, and a list of
what was withheld and why. The URL itself is kept for the write
``rote mcp import`` makes, and is excluded from ``repr``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol
from urllib.parse import urlsplit

#: What rote's registry can hold today: ``streamable-http`` and ``sse``.
#: The rest are listed so the user sees them, with the reason.
Transport = Literal["streamable-http", "sse", "stdio", "websocket", "other"]
IMPORTABLE_TRANSPORTS: frozenset[str] = frozenset({"streamable-http", "sse"})

#: Where in the client the entry lives. ``local`` is Claude Code's
#: per-project private scope; ``plugin`` is a server an enabled plugin ships.
Scope = Literal["user", "local", "project", "plugin"]


#: Moves the home directory client sources read. tests/conftest.py and
#: ``scripts/isolated-rote`` point it at an empty dir (it is in
#: ``rote.isolation.STATE_PATHS``), so neither ever sees the user's AI
#: clients.
CLIENT_HOME_ENV = "ROTE_CLIENT_HOME_DIR"

#: Variables that move a client's config dir away from the home. Under
#: :data:`CLIENT_HOME_ENV` they are dropped, or they would lead a source
#: straight back to the real files.
CLIENT_DIR_VARS = ("CLAUDE_CONFIG_DIR", "CODEX_HOME", "XDG_CONFIG_HOME")


@dataclass(frozen=True)
class ClientContext:
    """Everything a source may read from the machine, injected."""

    home: Path
    cwd: Path
    #: Read only for the variables that move a client's config dir
    #: (``CLAUDE_CONFIG_DIR``, ``CODEX_HOME``, ``XDG_CONFIG_HOME``), never to
    #: resolve a placeholder.
    env: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def current(cls) -> ClientContext:
        env = dict(os.environ)
        moved = env.get(CLIENT_HOME_ENV, "").strip()
        if moved:
            for var in CLIENT_DIR_VARS:
                env.pop(var, None)
            return cls(home=Path(moved), cwd=Path.cwd(), env=env)
        return cls(home=Path.home(), cwd=Path.cwd(), env=env)


@dataclass(frozen=True)
class HeaderTemplate:
    """A header rote can copy: its value is environment references plus at
    most an auth scheme word (``Bearer ${GITHUB_TOKEN}``)."""

    name: str
    #: The value in rote's one placeholder form, ``${VAR}``.
    template: str
    env_vars: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "template": self.template, "env_vars": list(self.env_vars)}


@dataclass(frozen=True)
class ImportCandidate:
    """One MCP server found in a client's config, normalized and display-safe."""

    client: str
    #: The rote server name: an identifier (the IR's ``mcp.server`` rule).
    name: str
    #: The name in the client's config, when it differs.
    original_name: str
    scope: Scope
    #: The file the entry was read from.
    origin: Path
    transport: Transport
    #: Never displayed: shown as :attr:`host` only.
    url: str | None = field(default=None, repr=False)
    headers: tuple[HeaderTemplate, ...] = ()
    scopes: tuple[str, ...] = ()
    #: What was found but not copied, each with its reason.
    withheld: tuple[str, ...] = ()
    #: Why the server cannot be imported. None: it can.
    blocked: str | None = None
    #: Why the client itself does not use this entry right now (disabled,
    #: not approved, shadowed by a higher scope). None: the client uses it.
    inactive: str | None = None
    #: The plugin that ships it, for ``scope == "plugin"``.
    plugin: str | None = None

    @property
    def importable(self) -> bool:
        return self.blocked is None

    @property
    def renamed(self) -> bool:
        return self.name != self.original_name

    @property
    def host(self) -> str | None:
        """The URL's host (and port), never its path, query or credentials."""
        if self.url is None:
            return None
        parts = urlsplit(self.url)
        if parts.hostname is None:
            return None
        return parts.hostname if parts.port is None else f"{parts.hostname}:{parts.port}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "client": self.client,
            "name": self.name,
            "original_name": self.original_name,
            "scope": self.scope,
            "plugin": self.plugin,
            "origin": str(self.origin),
            "transport": self.transport,
            "host": self.host,
            "headers": [h.to_dict() for h in self.headers],
            "scopes": list(self.scopes),
            "importable": self.importable,
            "blocked": self.blocked,
            "active": self.inactive is None,
            "inactive": self.inactive,
            "withheld": list(self.withheld),
        }


@dataclass(frozen=True)
class SourceReading:
    """What one source found: candidates, the files it read, and the files
    it could not read (reported, not raised, so one broken project file does
    not hide every other server; the command shows them)."""

    client: str
    candidates: tuple[ImportCandidate, ...] = ()
    files_read: tuple[Path, ...] = ()
    problems: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "client": self.client,
            "candidates": [c.to_dict() for c in self.candidates],
            "files_read": [str(p) for p in self.files_read],
            "problems": list(self.problems),
        }


class ClientSource(Protocol):
    """One AI client's MCP config, read-only."""

    #: Stable id, the ``--from`` value and :mod:`rote.interact.agents`' client id.
    id: str
    label: str

    def config_paths(self, ctx: ClientContext) -> tuple[Path, ...]:
        """Every file this source reads, existing or not, in precedence order."""
        ...

    def detect(self, ctx: ClientContext) -> bool:
        """Whether this client has config on the machine (any of its files exists)."""
        ...

    def read(self, ctx: ClientContext) -> SourceReading:
        """Parse the files into candidates. Never raises for a bad file."""
        ...
