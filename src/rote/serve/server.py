"""The ``rote serve`` MCP server.

One FastMCP (>=4.1) server whose tools are sourced from the manifest
registry (:mod:`rote.serve.registry`) via a custom v3 ``Provider``:

* every registry entry becomes one MCP tool named ``entry.name`` whose
  ``inputSchema`` is the entry's stored JSON Schema, and
* a companion ``<entry.name>_status`` tool that polls the workflow.

Freshness has two layers:

1. The provider re-reads the registry file (cheap content-hash check)
   on every ``tools/list`` / ``tools/call``, so a ``rote register``
   issued while the server is running is visible on the next request.
2. A background watcher polls the registry file and, when it changes,
   tells every connected client so clients that honor the notification
   (Claude Code does) refresh without a reconnect. FastMCP does not emit
   this itself when providers change, and the two protocol eras deliver
   it differently, so the server handles both:

   * handshake-era connections (2025-11-25 and earlier) get a plain
     ``notifications/tools/list_changed`` on their standalone channel.
     We track those connections via middleware. Track the SDK
     ``Connection``, not the ``ServerSession``: in MCP SDK v2 a
     ``ServerSession`` is a per-request proxy that dies with its
     request, so a weak set of them is always empty by the time the
     watcher looks.
   * sessionless 2026-07-28 connections have no session to push to. A
     client opts in with a ``subscriptions/listen`` request (SEP-2575)
     and the change rides that request's response stream. FastMCP 4.1
     registers no listen handler, so without ours the server advertises
     ``tools.listChanged: false`` on that era and modern clients would
     never hear about a ``rote register``. We register the SDK's
     ``ListenHandler`` and publish ``ToolsListChanged`` on its bus.

Long-running handling: compiled pipelines run minutes to days (HITL
gates), and their durability lives in Temporal / Cloudflare — not in
this process. Trigger tools therefore return ``{workflow_id, status:
"started"}`` immediately and clients poll the ``_status`` companion.
FastMCP 4 ships background tasks as the optional
``io.modelcontextprotocol/tasks`` extension (``fastmcp-tasks``), but
running the trigger as a task would tie a multi-day workflow's
observability to this process's lifetime, so the poll-tool pattern is
deliberate.

Idle sessions: since fastmcp 4.1 a handshake-era Streamable HTTP
session expires after 30 idle minutes (the SDK default). That is fine
here because nothing about a run lives in the session: the workflow_id
is the handle, and a client whose session expired reconnects and polls
the same id. stdio has no session timeout.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import weakref
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import mcp.types as mt
import mcp_types
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.server.providers import Provider
from fastmcp.tools import Tool, ToolResult
from mcp.server.connection import Connection
from mcp.server.subscriptions import InMemorySubscriptionBus, ListenHandler, ToolsListChanged
from mcp_types.version import MODERN_PROTOCOL_VERSIONS

from rote.serve import backends
from rote.serve.registry import Registry, RegistryEntry

logger = logging.getLogger(__name__)

#: How often the background watcher checks the registry file for changes.
DEFAULT_POLL_INTERVAL_S = 1.0

STATUS_TOOL_SUFFIX = "_status"
SIGNAL_TOOL_SUFFIX = "_signal"


# ───────── Tools ─────────


class PipelineTool(Tool):
    """Triggers the compiled workflow behind one registry entry."""

    entry: RegistryEntry

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            result = await backends.start_workflow(self.entry, arguments)
        except backends.BackendError as e:
            raise ToolError(str(e)) from e
        return ToolResult(structured_content=result)


class PipelineStatusTool(Tool):
    """Polls the status of a workflow started by the companion trigger tool."""

    entry: RegistryEntry

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        workflow_id = arguments["workflow_id"]
        try:
            result = await backends.workflow_status(self.entry, workflow_id)
        except backends.BackendError as e:
            raise ToolError(str(e)) from e
        return ToolResult(structured_content=result)


class PipelineSignalTool(Tool):
    """Resumes a workflow parked at a HITL gate (DBOS runtime only)."""

    entry: RegistryEntry

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            result = await backends.signal_workflow(
                self.entry,
                arguments["workflow_id"],
                arguments["signal"],
                arguments.get("payload", {}),
            )
        except backends.BackendError as e:
            raise ToolError(str(e)) from e
        return ToolResult(structured_content=result)


def _trigger_tool(entry: RegistryEntry) -> PipelineTool:
    return PipelineTool(
        name=entry.name,
        description=(
            f"{entry.description}\n\n"
            f"Starts the compiled '{entry.name}' pipeline on "
            f"{entry.trigger.runtime} and returns immediately with a workflow_id. "
            f"Poll progress with the {entry.name}{STATUS_TOOL_SUFFIX} tool."
        ).strip(),
        parameters=entry.input_schema,
        entry=entry,
    )


def _status_tool(entry: RegistryEntry) -> PipelineStatusTool:
    return PipelineStatusTool(
        name=f"{entry.name}{STATUS_TOOL_SUFFIX}",
        description=(
            f"Poll the status of a '{entry.name}' pipeline run previously "
            f"started with the {entry.name} tool."
        ),
        parameters={
            "type": "object",
            "properties": {
                "workflow_id": {
                    "type": "string",
                    "description": f"The workflow_id returned by the {entry.name} tool",
                },
            },
            "required": ["workflow_id"],
            "additionalProperties": False,
        },
        entry=entry,
    )


def _signal_tool(entry: RegistryEntry) -> PipelineSignalTool:
    trigger = entry.trigger
    gate_signals = getattr(trigger, "gate_signals", [])
    signal_schema: dict[str, Any] = {
        "type": "string",
        "description": "The HITL gate's signal (topic) name from the pipeline IR",
    }
    if gate_signals:
        signal_schema["enum"] = list(gate_signals)
    return PipelineSignalTool(
        name=f"{entry.name}{SIGNAL_TOOL_SUFFIX}",
        description=(
            f"Resume a '{entry.name}' pipeline run parked at a human-in-the-"
            f"loop gate. The payload is delivered as the gate's result and "
            f"flows into downstream nodes. Check {entry.name}"
            f"{STATUS_TOOL_SUFFIX} first: a run parked at a gate reports "
            f"status 'pending'."
        ),
        parameters={
            "type": "object",
            "properties": {
                "workflow_id": {
                    "type": "string",
                    "description": f"The workflow_id returned by the {entry.name} tool",
                },
                "signal": signal_schema,
                "payload": {
                    "type": "object",
                    "description": (
                        "Resume payload delivered to the gate (e.g. "
                        '{"approved": true} or the reviewed/edited data)'
                    ),
                },
            },
            "required": ["workflow_id", "signal"],
            "additionalProperties": False,
        },
        entry=entry,
    )


# ───────── Provider ─────────


class RegistryProvider(Provider):
    """Sources MCP tools from the registry file, re-reading it on change.

    Also owns the background watcher (started via the provider
    ``lifespan``, which FastMCP scopes to the server's lifetime), the set
    of live handshake-era connections to notify, and the subscription bus
    that feeds 2026-07-28 ``subscriptions/listen`` streams. Connections
    are held weakly so closed ones never pin memory or receive sends.
    """

    def __init__(
        self,
        registry_path: str | Path,
        poll_interval: float = DEFAULT_POLL_INTERVAL_S,
    ) -> None:
        super().__init__()
        self.registry_path = Path(registry_path)
        self.poll_interval = poll_interval
        self.connections: weakref.WeakSet[Connection] = weakref.WeakSet()
        self.subscriptions = InMemorySubscriptionBus()
        self.listen_handler = ListenHandler(self.subscriptions)
        self._digest: bytes | None = None
        self._registry = Registry()
        self._load_if_changed()

    # ── registry loading ──

    def _current_digest(self) -> bytes | None:
        try:
            return hashlib.sha256(self.registry_path.read_bytes()).digest()
        except FileNotFoundError:
            return None

    def _load_if_changed(self) -> bool:
        """Reload the registry if the file content changed. Returns True on change.

        The digest is committed only after a successful parse: a corrupt
        file (hand-edit typo, non-atomic write by another tool) must not
        be recorded as "seen", or the server would silently serve the
        previous tool list forever. Leaving the digest uncommitted means
        every subsequent poll retries, so fixing the file recovers the
        server without a restart. `Registry.save` itself is atomic
        (tmp + rename), so `rote register` can never produce a torn read.
        """
        digest = self._current_digest()
        if digest == self._digest:
            return False
        registry = Registry.load(self.registry_path)
        self._digest = digest
        self._registry = registry
        return True

    # ── Provider hooks ──

    async def _list_tools(self) -> Sequence[Tool]:
        self._load_if_changed()
        tools: list[Tool] = []
        for entry in self._registry.entries:
            tools.append(_trigger_tool(entry))
            tools.append(_status_tool(entry))
            # Only the DBOS backend can deliver HITL resume signals
            # (DBOSClient.send is one durable call); other runtimes use
            # their own tooling, so no signal tool is synthesized.
            if entry.trigger.runtime == "dbos":
                tools.append(_signal_tool(entry))
        return tools

    @asynccontextmanager
    async def lifespan(self) -> AsyncIterator[None]:
        watcher = asyncio.create_task(self._watch_registry())
        try:
            yield
        finally:
            # Graceful close: each open listen stream sends its final
            # result frame, so clients see a deliberate end, not a drop.
            self.listen_handler.close()
            watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watcher

    # ── change notification ──

    async def _watch_registry(self) -> None:
        while True:
            await asyncio.sleep(self.poll_interval)
            try:
                changed = self._load_if_changed()
            except Exception:
                # A corrupt registry or transient filesystem error must not
                # kill the watcher task — that would silently end
                # list_changed notifications for the rest of the server's
                # life. The digest was not committed, so the next poll
                # retries and a fixed file recovers automatically.
                logger.warning(
                    "registry reload failed; still serving the previous tool list",
                    exc_info=True,
                )
                continue
            if changed:
                await self._notify_tool_list_changed()

    async def _notify_tool_list_changed(self) -> None:
        # Modern clients: every open listen stream that asked for tool
        # changes. The bus isolates a raising listener itself.
        await self.subscriptions.publish(ToolsListChanged())
        # Handshake-era clients: a notification on each tracked connection.
        # Connection.notify already drops (debug-logged) on a closed
        # stream; the guard covers anything else a torn-down connection
        # raises, so one dead client cannot stall the rest.
        for connection in list(self.connections):
            try:
                await connection.notify("notifications/tools/list_changed", None)
            except Exception:
                logger.debug("failed to notify connection of tool list change", exc_info=True)


class ConnectionTrackingMiddleware(Middleware):
    """Records every handshake-era connection so the watcher can notify it.

    A 2026-07-28 request arrives on a per-request connection that has no
    back-channel (the SDK drops change notifications sent on it), so it
    is not tracked. Those clients hear about changes through
    ``subscriptions/listen`` instead.
    """

    def __init__(self, provider: RegistryProvider) -> None:
        self._provider = provider

    async def on_request(
        self,
        context: MiddlewareContext[mt.Request[Any, Any]],
        call_next: CallNext[mt.Request[Any, Any], Any],
    ) -> Any:
        fastmcp_ctx = context.fastmcp_context
        request_context = fastmcp_ctx.request_context if fastmcp_ctx is not None else None
        if (
            request_context is not None
            and request_context.protocol_version not in MODERN_PROTOCOL_VERSIONS
        ):
            # The SDK exposes the Connection on neither the per-request
            # ServerSession nor ServerRequestContext, so this reads the
            # session's private reference. The legacy-era notification
            # tests in tests/test_serve_server.py fail if it moves.
            self._provider.connections.add(request_context.session._connection)
        return await call_next(context)


# ───────── Server construction ─────────


def build_server(
    registry_path: str | Path,
    poll_interval: float = DEFAULT_POLL_INTERVAL_S,
) -> FastMCP[None]:
    """Build the ``rote serve`` FastMCP server for a registry file."""
    provider = RegistryProvider(registry_path, poll_interval=poll_interval)
    server: FastMCP[None] = FastMCP(
        name="rote",
        instructions=(
            "Tools on this server trigger compiled rote pipelines — "
            "deterministic workflows produced from Anthropic-style skills. "
            "Each pipeline has a trigger tool (returns a workflow_id "
            "immediately) and a companion <name>_status tool for polling, "
            "because pipelines can run for minutes to days (human-in-the-"
            "loop gates)."
        ),
        providers=[provider],
        middleware=[ConnectionTrackingMiddleware(provider)],
    )
    # FastMCP exposes no public hook for spec methods (extension
    # MethodBindings refuse them), so the listen handler goes on the
    # low-level server directly. Registering it is also what makes the
    # SDK advertise tools.listChanged on 2026-07-28 connections.
    # tests/test_serve_server.py pins both the capability and delivery.
    server._mcp_server.add_request_handler(
        "subscriptions/listen",
        mcp_types.SubscriptionsListenRequestParams,
        provider.listen_handler,
    )
    return server
