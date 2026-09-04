"""Live MCP tools for the in-process compiler drivers.

The ``api`` and ``openai-api`` compiler drivers can expose MCP servers'
read-only tools to the compiler agent alongside the three filesystem
tools. Everything protocol-neutral lives here:

- the ``readOnlyHint`` gate the eval baseline already applies — one
  predicate (:func:`read_only_tools`), shared so the two gates cannot
  drift;
- the ``mcp__<server>__<tool>`` naming convention (:func:`mcp_tool_id`),
  the same ids ``claude -p`` uses for MCP tools;
- :class:`LiveMcpTools`, the tool manager that lists each server's
  tools once at startup and opens a fresh connection per tool call;
- :func:`registry_server_specs`, the CLI's resolution of the local
  registry (``rote mcp add`` / ``rote mcp login``) into driver specs —
  the compile-time analog of the baseline's registry wiring.

fastmcp is an optional dependency (``pip install 'rote-cli[mcp]'``) —
everything here imports it lazily and fails with that hint, matching
:mod:`rote.mcp.auth`.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

#: A single wedged tools/call must become a reportable tool failure, never
#: a frozen compile.
CALL_TIMEOUT_SECONDS = 180.0

MCP_TOOL_PREFIX = "mcp__"


def mcp_tool_id(server: str, tool: str) -> str:
    """The ``mcp__<server>__<tool>`` id for one server's tool.

    The same convention ``claude -p`` uses for MCP tool ids, so the
    baseline's allowlist and the in-process drivers' tool names agree.
    """
    return f"{MCP_TOOL_PREFIX}{server}__{tool}"


def _read_only_hint(annotations: Any) -> bool:
    """The read-only hint under either SDK spelling, snake_case first.

    MCP SDK v2 renamed ``ToolAnnotations.readOnlyHint`` to
    ``read_only_hint`` and made the camelCase name a deprecated alias —
    reading it under fastmcp 4 fires FastMCPDeprecationWarning (which
    surfaced as stderr noise in prod error mail). Reading snake_case
    first never touches the alias on a modern install; SDK v1 objects
    (fastmcp 3.x) have no snake attribute and fall through to camelCase.
    """
    if hasattr(annotations, "read_only_hint"):
        return annotations.read_only_hint is True
    return getattr(annotations, "readOnlyHint", None) is True


def read_only_tools(tools: Sequence[Any]) -> list[Any]:
    """The subset of listed tools whose server-declared read-only hint is true.

    The one side-effect gate for exposing MCP tools to an unattended
    agent: only tools the server annotates as read-only are callable.
    A tool with no annotations (or a false/absent hint) is excluded.
    """
    return [t for t in tools if t.annotations is not None and _read_only_hint(t.annotations)]


def result_text(result: Any) -> str:
    """Flatten an MCP ``CallToolResult`` into the text handed to the model.

    Structured content is JSON-stringified; otherwise the text parts are
    concatenated. An empty result yields an empty string — the model can
    cope with that better than with an invented placeholder.
    """
    structured = getattr(result, "structured_content", None)
    if structured is not None:
        return json.dumps(structured)
    texts = [block.text for block in result.content if getattr(block, "text", None) is not None]
    return "\n".join(texts)


class LiveMcpTools:
    """Read-only MCP tools exposed to an in-process compiler agent.

    ``servers`` entries are plain dicts — ``{"name": str, "url": str,
    "headers": dict[str, str] | None}`` (streamable HTTP only). Use as
    an async context manager: entry connects to each server once to
    list its tools and closes that session immediately; every later
    :meth:`call` opens a fresh connection for that one tools/call.
    Nothing is held open between calls — long-lived streamable-HTTP
    sessions combined with the vendor SDK deterministically froze the
    asyncio loop under the hosted container runtime (found empirically),
    and a per-call connect costs ~100ms against multi-second LLM turns
    while picking up rotated tokens naturally.

    A server that fails to connect, fails to list tools, or declares no
    read-only tools is reported through ``on_warning`` and skipped — it
    never fails the compile. Only a missing fastmcp install raises
    (``RuntimeError`` naming the ``rote-cli[mcp]`` extra): the caller
    asked for MCP explicitly, so silently compiling without it would be
    a wrong answer rather than a degraded one.
    """

    def __init__(
        self,
        servers: Sequence[Mapping[str, Any]],
        on_warning: Callable[[str], None] | None = None,
    ) -> None:
        self._servers = list(servers)
        self._on_warning = on_warning
        #: prefixed tool id → (server spec, bare tool name); the spec is
        #: kept so :meth:`call` can open a fresh client per invocation.
        self._tools: dict[str, tuple[dict[str, Any], str]] = {}
        #: wire-shape-neutral defs, same tuple layout as the drivers'
        #: ``_fs_tools._TOOL_DEFS``: (name, description, json-schema).
        self._defs: list[tuple[str, str, dict[str, Any]]] = []

    def _warn(self, message: str) -> None:
        if self._on_warning is not None:
            self._on_warning(message)

    @staticmethod
    def _client(spec: Mapping[str, Any]) -> Any:
        """A new fastmcp client for one server spec (not yet connected).

        Any: the two constructor forms infer different Client type
        parameters; callers only use the shared surface.
        """
        import fastmcp
        from fastmcp.client.transports import StreamableHttpTransport

        url = str(spec["url"])
        headers = spec.get("headers") or None
        if headers:
            return fastmcp.Client(StreamableHttpTransport(url, headers=dict(headers)))
        return fastmcp.Client(url)

    async def __aenter__(self) -> LiveMcpTools:
        try:
            import fastmcp  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "live MCP tools require the fastmcp client — "
                "install it with: pip install 'rote-cli[mcp]'"
            ) from e

        for spec in self._servers:
            name = str(spec["name"])
            try:
                client = self._client(spec)
                async with client:
                    tools = await client.list_tools()
            except Exception as e:  # noqa: BLE001 — a dead server must not sink the compile
                self._warn(
                    f"MCP server {name!r} unavailable "
                    f"({type(e).__name__}: {e}) — continuing without it"
                )
                continue
            read_only = read_only_tools(tools)
            if not read_only:
                self._warn(
                    f"MCP server {name!r} declares no read-only tools "
                    f"({len(tools)} tools total) — nothing exposed to the compiler"
                )
                continue
            for tool in sorted(read_only, key=lambda t: t.name):
                prefixed = mcp_tool_id(name, tool.name)
                if prefixed in self._tools:
                    continue
                schema = tool.inputSchema or {"type": "object", "properties": {}}
                self._tools[prefixed] = (dict(spec), tool.name)
                self._defs.append(
                    (
                        prefixed,
                        tool.description or f"Tool {tool.name!r} on MCP server {name!r}.",
                        dict(schema),
                    )
                )
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """Nothing to release — no session outlives the call that opened it."""
        return None

    def owns(self, tool_name: str) -> bool:
        """True when ``tool_name`` is one of this manager's exposed tools."""
        return tool_name in self._tools

    def anthropic_tool_schemas(self) -> list[dict[str, Any]]:
        """The exposed tools in Anthropic Messages API shape."""
        return [
            {"name": name, "description": desc, "input_schema": schema}
            for name, desc, schema in self._defs
        ]

    def openai_tool_schemas(self) -> list[dict[str, Any]]:
        """The exposed tools in OpenAI chat/completions function-calling shape."""
        return [
            {
                "type": "function",
                "function": {"name": name, "description": desc, "parameters": schema},
            }
            for name, desc, schema in self._defs
        ]

    async def call(self, tool_name: str, args: dict[str, Any] | None) -> str:
        """Invoke one exposed tool over a fresh connection, result as text.

        Raises on any failure (unknown tool, tool error, transport
        error, timeout) — the driver catches and reports it back to the
        model as an error tool result rather than crashing the loop,
        exactly like the filesystem tools.
        """
        try:
            spec, bare_name = self._tools[tool_name]
        except KeyError:
            raise ValueError(f"Unknown MCP tool: {tool_name}") from None
        client = self._client(spec)
        try:
            async with client:
                result = await asyncio.wait_for(
                    client.call_tool(bare_name, args or {}), timeout=CALL_TIMEOUT_SECONDS
                )
        except TimeoutError:
            raise RuntimeError(
                f"MCP tool {tool_name} timed out after {CALL_TIMEOUT_SECONDS:.0f}s"
            ) from None
        return result_text(result)


async def registry_server_specs(
    on_skip: Callable[[str, str], None] | None = None,
) -> list[dict[str, Any]]:
    """Driver server specs for every registered, authenticated MCP server.

    The compile-time analog of the eval baseline's registry wiring
    (:func:`rote.eval.baseline.mcp_servers_from_registry`), resolved for
    in-process drivers that need concrete headers rather than a
    ``headersHelper`` subprocess: a registry entry with static headers
    contributes them verbatim; a logged-in server contributes a freshly
    refreshed bearer token. A server that is not authenticated, whose
    refresh fails, or that uses an unsupported transport is skipped with
    a reason through ``on_skip`` — never an error.
    """
    from rote.mcp import McpAuthError, fresh_access_token, load_registry, read_token_file
    from rote.mcp.tokens import access_token_state

    def _skip(server: str, reason: str) -> None:
        if on_skip is not None:
            on_skip(server, reason)

    specs: list[dict[str, Any]] = []
    registry = load_registry()
    for name, entry in sorted(registry.servers.items()):
        if entry.transport != "streamable-http":
            _skip(name, f"transport {entry.transport!r} is not supported for live compiler tools")
            continue
        if entry.headers:
            specs.append({"name": name, "url": entry.url, "headers": dict(entry.headers)})
            continue
        doc = read_token_file(name)
        if doc is None or access_token_state(doc)[0] is None:
            _skip(name, f"not authenticated — rote mcp login {name}")
            continue
        try:
            token = await fresh_access_token(name, entry)
        except McpAuthError as e:
            _skip(name, str(e))
            continue
        specs.append(
            {"name": name, "url": entry.url, "headers": {"Authorization": f"Bearer {token}"}}
        )
    return specs


__all__ = [
    "MCP_TOOL_PREFIX",
    "LiveMcpTools",
    "mcp_tool_id",
    "read_only_tools",
    "registry_server_specs",
    "result_text",
]
