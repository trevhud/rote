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
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol
from urllib.parse import urlsplit

#: What rote's registry can hold today: ``streamable-http`` and ``sse``.
#: The rest are listed so the user sees them, with the reason.
Transport = Literal["streamable-http", "sse", "stdio", "websocket", "other"]
IMPORTABLE_TRANSPORTS: frozenset[str] = frozenset({"streamable-http", "sse"})

#: Where in the client the entry lives. ``local`` is a per-project private
#: scope (Claude Code's ``~/.claude.json`` projects entry, Devin's
#: ``.devin/mcp_config.local.json``); ``plugin`` is a server an enabled
#: plugin or extension ships; ``system`` is a machine-wide file an
#: administrator deploys (Gemini CLI's system settings, Goose's
#: ``/etc/goose/config.yaml``).
Scope = Literal["user", "local", "project", "plugin", "system"]


#: Moves the home directory client sources read. tests/conftest.py and
#: ``scripts/isolated-rote`` point it at an empty dir (it is in
#: ``rote.isolation.STATE_PATHS``), so neither ever sees the user's AI
#: clients.
CLIENT_HOME_ENV = "ROTE_CLIENT_HOME_DIR"

#: Variables that move a client's config files away from the home (or, for
#: ``OPENCODE_CONFIG_CONTENT``, hold the config itself). Under
#: :data:`CLIENT_HOME_ENV` they are dropped, or they would lead a source
#: straight back to the real files. A source that reads a new one adds it
#: here.
CLIENT_DIR_VARS = (
    "CLAUDE_CONFIG_DIR",
    "CODEX_HOME",
    "XDG_CONFIG_HOME",
    "APPDATA",
    "COPILOT_HOME",
    "VSCODE_PORTABLE",
    "VSCODE_APPDATA",
    "FLATPAK_XDG_CONFIG_HOME",
    "GOOSE_PATH_ROOT",
    "GOOSE_ADDITIONAL_CONFIG_FILES",
    "AMP_SETTINGS_FILE",
    "GEMINI_CLI_HOME",
    "GEMINI_CLI_SYSTEM_SETTINGS_PATH",
    "GEMINI_CLI_SYSTEM_DEFAULTS_PATH",
    "GEMINI_CLI_TRUSTED_FOLDERS_PATH",
    "OPENCODE_CONFIG",
    "OPENCODE_CONFIG_DIR",
    "OPENCODE_CONFIG_CONTENT",
)


def _system_dir() -> Path | None:
    """The root admin-deployed client policy lives under on this OS (Claude
    Code: code.claude.com/docs/en/managed-mcp, "Deploy managed-mcp.json")."""
    if sys.platform == "darwin":
        return Path("/Library/Application Support")
    if sys.platform.startswith("linux"):
        return Path("/etc")
    if sys.platform == "win32":
        return Path("C:/Program Files")
    return None


def _etc_dir() -> Path | None:
    """Where machine-wide client config lives that is not under
    :func:`_system_dir`: ``/etc`` on every Unix, macOS included (Codex
    ``/etc/codex``, Goose ``/etc/goose``), and ``%ProgramData%`` on Windows
    (Gemini CLI, OpenCode, Codex)."""
    if sys.platform == "win32":
        return Path(os.environ.get("PROGRAMDATA", "C:/ProgramData"))
    return Path("/etc")


@dataclass(frozen=True)
class ClientContext:
    """Everything a source may read from the machine, injected."""

    home: Path
    cwd: Path
    #: Read only for the variables that move a client's config files
    #: (:data:`CLIENT_DIR_VARS`) and the few that switch a client's files on
    #: or off, never to resolve a placeholder.
    env: Mapping[str, str] = field(default_factory=dict)
    #: The OS-wide directory administrators deploy client policy under
    #: (``/Library/Application Support`` on macOS, ``/etc`` on Linux). Each
    #: source adds its own subdirectory. None: not read, which is what the
    #: isolation override gives.
    system_dir: Path | None = None
    #: ``/etc`` on macOS and Linux, ``%ProgramData%`` on Windows: the other
    #: machine-wide root clients read (see ``_etc_dir``). None: not read.
    etc_dir: Path | None = None
    #: ``sys.platform`` for the machine whose files these are.
    platform: str = sys.platform

    @classmethod
    def current(cls) -> ClientContext:
        env = dict(os.environ)
        moved = env.get(CLIENT_HOME_ENV, "").strip()
        if moved:
            for var in CLIENT_DIR_VARS:
                env.pop(var, None)
            return cls(home=Path(moved), cwd=Path.cwd(), env=env)
        return cls(
            home=Path.home(),
            cwd=Path.cwd(),
            env=env,
            system_dir=_system_dir(),
            etc_dir=_etc_dir(),
        )


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
    #: A risk the user must accept by name before this server is imported
    #: (a credential-shaped URL path segment). Importable, but never picked
    #: by a default or by ``--yes``: ``rote mcp import`` needs a per-server
    #: opt-in that names the risk (``--allow-url-secret <name>``).
    risk: str | None = None
    #: The plugin that ships it, for ``scope == "plugin"``.
    plugin: str | None = None

    @property
    def importable(self) -> bool:
        return self.blocked is None

    @property
    def preselectable(self) -> bool:
        """Whether a default (or ``--yes``) may pick this server: importable,
        used by the client, and carrying no risk to opt into."""
        return self.importable and self.inactive is None and self.risk is None

    @property
    def renamed(self) -> bool:
        return self.name != self.original_name

    @property
    def host(self) -> str | None:
        """The URL's host (and port), never its path, query or credentials."""
        if self.url is None:
            return None
        try:
            parts = urlsplit(self.url)
            port = parts.port
        except ValueError:
            # A host that is a placeholder (``https://${env:HOST}/mcp``) reads
            # as a bad port; such a URL is blocked and has no host to show.
            return None
        if parts.hostname is None:
            return None
        return parts.hostname if port is None else f"{parts.hostname}:{port}"

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
            "risk": self.risk,
            "preselectable": self.preselectable,
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
    #: What this source does not read at all, stated every time so a
    #: listing never implies more than it checked.
    limits: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "client": self.client,
            "candidates": [c.to_dict() for c in self.candidates],
            "files_read": [str(p) for p in self.files_read],
            "problems": list(self.problems),
            "limits": list(self.limits),
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
