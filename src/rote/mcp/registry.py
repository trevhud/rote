"""The MCP server registry — ``rote mcp add``'s persistent store.

One user-level JSON file maps *logical server names* (the same names
compiled pipelines carry in their ``mcp:`` bindings) to endpoints and
client credentials. The registry is deliberately tiny: names, URLs,
transports, and the OAuth escape hatches real servers need
(pre-registered ``client_id``/``client_secret`` for DCR-less servers
like Slack and GitHub, static extra headers for API-key schemes).

Tokens never live here — they live in :mod:`rote.mcp.tokens`, one file
per server, so the registry stays shareable and the secrets stay
separate. ``client_secret`` is the one exception (it is registry
config, not a token), which is why the file is written ``0600``.

Resolution order for a server URL, everywhere in rote (the CLI, the
eval harness, and the code emitted into runtimes):

1. an explicit ``url`` on the IR binding (pipeline-pinned),
2. the registry entry for the binding's logical server name,
3. the ``ROTE_MCP_<SERVER>_URL`` environment variable.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

REGISTRY_ENV_VAR = "ROTE_MCP_CONFIG"


def registry_path() -> Path:
    """``$ROTE_MCP_CONFIG`` > ``$XDG_CONFIG_HOME/rote/mcp.json`` > ``~/.config/rote/mcp.json``."""
    override = os.environ.get(REGISTRY_ENV_VAR)
    if override:
        return Path(override)
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "rote" / "mcp.json"


class McpServerConfig(BaseModel):
    """One registered MCP server."""

    model_config = ConfigDict(extra="forbid")

    url: str
    transport: Literal["streamable-http", "sse"] = "streamable-http"
    client_id: str | None = Field(
        default=None,
        description=(
            "Pre-registered OAuth client id. Set for servers that do not "
            "support dynamic client registration (Slack/GitHub-class)."
        ),
    )
    client_secret: str | None = Field(
        default=None,
        description="Pre-registered client secret (confidential clients).",
    )
    scopes: list[str] | None = Field(
        default=None,
        description="OAuth scopes to request at login. Omit to let the server decide.",
    )
    headers: dict[str, str] | None = Field(
        default=None,
        description=(
            "Static extra headers (API-key schemes). Sent on every request; "
            "not a substitute for OAuth — a server offering OAuth should be "
            "logged in via `rote mcp login` instead."
        ),
    )


class McpRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = 1
    servers: dict[str, McpServerConfig] = Field(default_factory=dict)


def load_registry(path: Path | None = None) -> McpRegistry:
    p = path or registry_path()
    if not p.is_file():
        return McpRegistry()
    with p.open("r", encoding="utf-8") as f:
        return McpRegistry.model_validate(json.load(f))


def save_registry(registry: McpRegistry, path: Path | None = None) -> Path:
    """Atomic 0600 write — the registry may hold client secrets."""
    p = path or registry_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=p.parent, prefix=".mcp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(registry.model_dump(exclude_none=True), f, indent=2)
            f.write("\n")
        os.chmod(tmp_name, 0o600)
        os.replace(tmp_name, p)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise
    return p


def resolve_server_url(
    server: str,
    explicit_url: str | None = None,
    registry: McpRegistry | None = None,
) -> str | None:
    """The one URL-resolution rule (binding → registry → env). None = unresolvable."""
    if explicit_url:
        return explicit_url
    reg = registry if registry is not None else load_registry()
    entry = reg.servers.get(server)
    if entry is not None:
        return entry.url
    return os.environ.get(f"ROTE_MCP_{server.upper()}_URL")
