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

Resolution order for a server URL in the code emitted into runtimes
(``resolve_url`` in :mod:`rote.mcp._runtime_helper` and the Node TS
helper), which is what every step and agent loop uses:

1. the ``ROTE_MCP_<SERVER>_URL`` environment variable,
2. the registry entry for the binding's logical server name,
3. the ``url`` recorded on the IR binding at compilation time.

Deployment context outranks compilation-time capture. Workers have no
registry: the variable (a wrangler var or secret), then the IR ``url``.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from rote.xdg import config_home

REGISTRY_ENV_VAR = "ROTE_MCP_CONFIG"


def registry_path() -> Path:
    """``$ROTE_MCP_CONFIG`` > ``$XDG_CONFIG_HOME/rote/mcp.json`` > ``~/.config/rote/mcp.json``."""
    override = os.environ.get(REGISTRY_ENV_VAR)
    if override:
        return Path(override)
    return config_home() / "rote" / "mcp.json"


#: The registry is shared by every rote version on the machine (a cached
#: ``uvx --from rote-cli`` beside a newer install, the plugin's ``uv
#: run``). From this version on, unknown fields are kept, not rejected:
#: a newer version's field must neither fail this one nor be dropped when
#: this one writes the file back. Released versions up to 0.14.0 still
#: reject unknown fields, which is why learned auth lives in a sidecar
#: (rote.mcp.learned) rather than here.
_FORWARD_COMPATIBLE = ConfigDict(extra="allow")


class McpServerConfig(BaseModel):
    """One registered MCP server."""

    model_config = _FORWARD_COMPATIBLE

    url: str
    #: "streamable-http" or "sse" from this version's `rote mcp add`; kept
    #: as a plain string so a transport a newer rote adds loads here and
    #: is treated as unsupported rather than rejected.
    transport: str = "streamable-http"
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
    issuer: str | None = Field(
        default=None,
        description=(
            "The authorization server a pre-registered client belongs to, recorded "
            "at `rote mcp add`. A login against any other issuer stops before the "
            "browser opens instead of offering the client there."
        ),
    )
    scopes: list[str] | None = Field(
        default=None,
        description=(
            "OAuth scopes to request at login, exactly: never widened to what the "
            "server advertises, and a server that requires more stops the login. "
            "Omit to let the server decide."
        ),
    )
    read_only: bool | None = Field(
        default=None,
        description=(
            "The user's word that no tool on this server writes (`rote mcp add "
            "--read-only`). Read-only runs then call every tool on it, whether or "
            "not the server sets readOnlyHint, which many real servers never do."
        ),
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
    model_config = _FORWARD_COMPATIBLE

    version: int = 1
    servers: dict[str, McpServerConfig] = Field(default_factory=dict)


def declared_read_only(server: str) -> bool:
    """Whether the user registered ``server`` read-only, at the endpoint a call
    to it resolves to now (``ROTE_MCP_<SERVER>_URL`` wins over the registry,
    and the declaration covers the registered endpoint only)."""
    from rote.mcp._runtime_helper import RoteMcpConfigError, declared_read_only_at, resolve_url

    config = load_registry().servers.get(server)
    if config is None or config.read_only is not True:
        return False
    try:
        resolved = resolve_url(server, None)
    except RoteMcpConfigError:
        return False
    return declared_read_only_at(config.url, resolved)


def read_only_servers() -> dict[str, str]:
    """``{name: registered url}`` for every server the user registered read-only."""
    return {
        name: c.url for name, c in sorted(load_registry().servers.items()) if c.read_only is True
    }


def has_references(headers: dict[str, str]) -> bool:
    """Whether any header value holds a ``${NAME}`` reference, which the
    runtime helpers fill from the environment at each connection."""
    from rote.mcp._runtime_helper import HEADER_REFERENCE

    return any(HEADER_REFERENCE.search(value) for value in headers.values())


def load_registry(path: Path | None = None) -> McpRegistry:
    p = path or registry_path()
    if not p.is_file():
        return McpRegistry()
    return parse_registry(p.read_bytes())


def parse_registry(data: bytes | None) -> McpRegistry:
    """The registry in ``data`` (a file's bytes; None: no file yet).

    For a caller that merges into the file and must check, at write time,
    that it still holds these same bytes (``rote.interact.plan.Snapshot``)."""
    if data is None:
        return McpRegistry()
    return McpRegistry.model_validate(json.loads(data.decode("utf-8")))


#: The registry file's permission bits: it may hold client secrets.
REGISTRY_FILE_MODE = 0o600


def render_registry(registry: McpRegistry) -> str:
    """The registry file's text, exactly as :func:`save_registry` writes it."""
    return json.dumps(registry.model_dump(exclude_none=True), indent=2) + "\n"


def save_registry(registry: McpRegistry, path: Path | None = None) -> Path:
    """Atomic 0600 write — the registry may hold client secrets."""
    p = path or registry_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=p.parent, prefix=".mcp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(render_registry(registry))
        os.chmod(tmp_name, REGISTRY_FILE_MODE)
        os.replace(tmp_name, p)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise
    return p
