"""rote's MCP client layer: server registry, durable token store, OAuth.

Three pieces, three files:

- :mod:`rote.mcp.registry` — ``rote mcp add``'s store: logical server
  names → endpoints + client credentials (no tokens).
- :mod:`rote.mcp.tokens` — the per-server token files; the layout is a
  documented cross-language contract shared with the emitted
  TypeScript runtimes.
- :mod:`rote.mcp.auth` — the OAuth dance and refresh, via fastmcp's
  client (optional dependency; ``pip install 'rote-cli[mcp]'``).
"""

from rote.mcp.auth import McpAuthError, fresh_access_token, login
from rote.mcp.registry import (
    McpRegistry,
    McpServerConfig,
    load_registry,
    registry_path,
    resolve_server_url,
    save_registry,
)
from rote.mcp.tokens import (
    AuthStatus,
    access_token_state,
    auth_status,
    clear_token_file,
    read_token_file,
    token_dir,
    token_file,
    write_token_file,
)

__all__ = [
    "AuthStatus",
    "McpAuthError",
    "McpRegistry",
    "McpServerConfig",
    "access_token_state",
    "auth_status",
    "clear_token_file",
    "fresh_access_token",
    "load_registry",
    "login",
    "read_token_file",
    "registry_path",
    "resolve_server_url",
    "save_registry",
    "token_dir",
    "token_file",
    "write_token_file",
]
