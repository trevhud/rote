"""The ``rote serve`` MCP server.

One FastMCP (>=3.4) server whose tools are sourced from the manifest
registry (:mod:`rote.serve.registry`) via a custom v3 ``Provider``:

* every registry entry becomes one MCP tool named ``entry.name`` whose
  ``inputSchema`` is the entry's stored JSON Schema, and
* a companion ``<entry.name>_status`` tool that polls the workflow.

Freshness has two layers:

1. The provider re-reads the registry file (cheap content-hash check)
   on every ``tools/list`` / ``tools/call``, so a ``rote register``
   issued while the server is running is visible on the next request.
2. A background watcher polls the registry file and, when it changes,
   sends ``notifications/tools/list_changed`` to every connected
   session so clients that honor the notification (Claude Code does)
   refresh without a reconnect. FastMCP 3.4 does not emit this
   notification itself when providers change, so we track sessions via
   middleware and use the low-level SDK's
   ``ServerSession.send_tool_list_changed()`` directly.

Long-running handling: graduated pipelines run minutes to days (HITL
gates), and their durability lives in Temporal / Cloudflare — not in
this process. Trigger tools therefore return ``{workflow_id, status:
"started"}`` immediately and clients poll the ``_status`` companion.
FastMCP 3.4 does ship server-side MCP Tasks support (SEP-1686), but the
extension is still a spec RC and running the trigger as a task would
tie a multi-day workflow's observability to this process's lifetime,
so the poll-tool pattern is deliberate. Revisit once the 2026-07-28
spec lands and Claude clients request task augmentation.
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
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.server.providers import Provider
from fastmcp.tools import Tool, ToolResult
from mcp.server.session import ServerSession

from rote.serve import backends
from rote.serve.registry import Registry, RegistryEntry

logger = logging.getLogger(__name__)

#: How often the background watcher checks the registry file for changes.
DEFAULT_POLL_INTERVAL_S = 1.0

STATUS_TOOL_SUFFIX = "_status"
SIGNAL_TOOL_SUFFIX = "_signal"


# ───────── Tools ─────────


class PipelineTool(Tool):
    """Triggers the graduated workflow behind one registry entry."""

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
            f"Starts the graduated '{entry.name}' pipeline on "
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
    ``lifespan``, which FastMCP scopes to the server's lifetime) and the
    set of live sessions to notify. Sessions are held weakly so closed
    connections never pin memory or receive sends.
    """

    def __init__(
        self,
        registry_path: str | Path,
        poll_interval: float = DEFAULT_POLL_INTERVAL_S,
    ) -> None:
        super().__init__()
        self.registry_path = Path(registry_path)
        self.poll_interval = poll_interval
        self.sessions: weakref.WeakSet[ServerSession] = weakref.WeakSet()
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
        for session in list(self.sessions):
            try:
                await session.send_tool_list_changed()
            except Exception:
                # A torn-down session must not kill the watcher or block
                # notifying the remaining live sessions.
                logger.debug("failed to notify session of tool list change", exc_info=True)


class SessionTrackingMiddleware(Middleware):
    """Records every session that talks to us so the watcher can notify it."""

    def __init__(self, provider: RegistryProvider) -> None:
        self._provider = provider

    async def on_request(
        self,
        context: MiddlewareContext[mt.Request[Any, Any]],
        call_next: CallNext[mt.Request[Any, Any], Any],
    ) -> Any:
        fastmcp_ctx = context.fastmcp_context
        if fastmcp_ctx is not None:
            # No live session on this context (init edge cases) — skip.
            with contextlib.suppress(AttributeError, ValueError):
                self._provider.sessions.add(fastmcp_ctx.session)
        return await call_next(context)


# ───────── Server construction ─────────


def build_server(
    registry_path: str | Path,
    poll_interval: float = DEFAULT_POLL_INTERVAL_S,
) -> FastMCP[None]:
    """Build the ``rote serve`` FastMCP server for a registry file."""
    provider = RegistryProvider(registry_path, poll_interval=poll_interval)
    return FastMCP(
        name="rote",
        instructions=(
            "Tools on this server trigger graduated rote pipelines — "
            "deterministic workflows produced from Anthropic-style skills. "
            "Each pipeline has a trigger tool (returns a workflow_id "
            "immediately) and a companion <name>_status tool for polling, "
            "because pipelines can run for minutes to days (human-in-the-"
            "loop gates)."
        ),
        providers=[provider],
        middleware=[SessionTrackingMiddleware(provider)],
    )
