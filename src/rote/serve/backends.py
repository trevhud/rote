"""Runtime trigger backends for ``rote serve``.

Each registry entry names a runtime trigger (Temporal or Cloudflare).
This module knows how to *start* a graduated workflow on that runtime
and how to *poll* its status. Tool calls never block on workflow
completion — graduated pipelines run for minutes to days (HITL gates),
and their durability lives in the workflow engine, not in this process.
Starting returns ``{workflow_id, status: "started", ...}`` immediately;
the companion ``<tool>_status`` MCP tool polls via :func:`workflow_status`.

Clients are connected lazily, per call, so ``rote serve`` starts (and
lists tools) even when the runtime is unreachable — the error surfaces
on the tool call that needs it, with a message that says exactly what
was unreachable.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from rote.serve.registry import CloudflareTrigger, RegistryEntry, TemporalTrigger

#: How long to wait for a Temporal frontend connection before declaring
#: it unreachable. Temporal's client otherwise retries indefinitely.
TEMPORAL_CONNECT_TIMEOUT_S = 10.0

#: HTTP timeout for Cloudflare worker trigger/status calls.
CLOUDFLARE_HTTP_TIMEOUT_S = 30.0


class BackendError(RuntimeError):
    """A runtime trigger failed in a way the MCP client should see verbatim."""


# ───────── Public dispatch ─────────


async def start_workflow(entry: RegistryEntry, payload: dict[str, Any]) -> dict[str, Any]:
    """Start the graduated workflow behind ``entry`` with ``payload`` as input."""
    trigger = entry.trigger
    if isinstance(trigger, TemporalTrigger):
        return await _start_temporal(trigger, entry.name, payload)
    return await _start_cloudflare(trigger, payload)


async def workflow_status(entry: RegistryEntry, workflow_id: str) -> dict[str, Any]:
    """Poll the status of a previously started workflow."""
    trigger = entry.trigger
    if isinstance(trigger, TemporalTrigger):
        return await _status_temporal(trigger, workflow_id)
    return await _status_cloudflare(trigger, workflow_id)


# ───────── Temporal ─────────


async def _temporal_client(trigger: TemporalTrigger) -> Any:
    try:
        from temporalio.client import Client
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise BackendError(
            "temporalio is not installed. Install the Temporal extra: pip install 'rote[temporal]'"
        ) from e

    try:
        return await asyncio.wait_for(
            Client.connect(trigger.address, namespace=trigger.namespace),
            timeout=TEMPORAL_CONNECT_TIMEOUT_S,
        )
    except TimeoutError as e:
        raise BackendError(
            f"Temporal at {trigger.address} (namespace {trigger.namespace!r}) is "
            f"unreachable: connection timed out after {TEMPORAL_CONNECT_TIMEOUT_S:.0f}s"
        ) from e
    except Exception as e:
        raise BackendError(
            f"Temporal at {trigger.address} (namespace {trigger.namespace!r}) is unreachable: {e}"
        ) from e


async def _start_temporal(
    trigger: TemporalTrigger, tool_name: str, payload: dict[str, Any]
) -> dict[str, Any]:
    client = await _temporal_client(trigger)
    workflow_id = f"{tool_name}-{uuid.uuid4().hex[:12]}"
    try:
        handle = await client.start_workflow(
            trigger.workflow_name,
            payload,
            id=workflow_id,
            task_queue=trigger.task_queue,
        )
    except Exception as e:
        raise BackendError(
            f"Temporal at {trigger.address} refused to start workflow "
            f"{trigger.workflow_name!r} on task queue {trigger.task_queue!r}: {e}"
        ) from e
    return {
        "workflow_id": workflow_id,
        "run_id": handle.first_execution_run_id or "",
        "status": "started",
        "runtime": "temporal",
    }


async def _status_temporal(trigger: TemporalTrigger, workflow_id: str) -> dict[str, Any]:
    client = await _temporal_client(trigger)
    handle = client.get_workflow_handle(workflow_id)
    try:
        description = await handle.describe()
    except Exception as e:
        raise BackendError(
            f"Temporal at {trigger.address} could not describe workflow {workflow_id!r}: {e}"
        ) from e
    status = description.status.name.lower() if description.status else "unknown"
    return {
        "workflow_id": workflow_id,
        "status": status,
        "runtime": "temporal",
    }


# ───────── Cloudflare ─────────


async def _start_cloudflare(trigger: CloudflareTrigger, payload: dict[str, Any]) -> dict[str, Any]:
    import httpx

    try:
        async with httpx.AsyncClient(timeout=CLOUDFLARE_HTTP_TIMEOUT_S) as http:
            resp = await http.post(trigger.url, json=payload)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as e:
        raise BackendError(f"Cloudflare worker at {trigger.url} failed: {e}") from e
    except ValueError as e:
        raise BackendError(
            f"Cloudflare worker at {trigger.url} returned a non-JSON response "
            f"(HTTP {resp.status_code}): {resp.text[:200]!r}"
        ) from e

    # The emitted src/index.ts responds with {id, status}. A 2xx JSON body
    # without an id means the URL is not the emitted worker (an auth proxy,
    # a different service) — nothing started, so say so instead of
    # fabricating success with an empty workflow_id.
    if not isinstance(data, dict) or not data.get("id"):
        raise BackendError(
            f"Cloudflare worker at {trigger.url} did not return a workflow id. "
            f"Expected the emitted worker's {{id, status}} response, got: "
            f"{str(data)[:200]}"
        )
    return {
        "workflow_id": str(data["id"]),
        "status": "started",
        "runtime": "cloudflare",
        "details": data.get("status"),
    }


async def _status_cloudflare(trigger: CloudflareTrigger, workflow_id: str) -> dict[str, Any]:
    if trigger.status_url is None:
        raise BackendError(
            "This Cloudflare worker has no status endpoint registered. The emitted "
            "worker's default fetch handler only creates instances; check status with "
            f"`wrangler workflows instances describe <workflow> {workflow_id}` or the "
            "Cloudflare dashboard, or register with --cloudflare-status-url if the "
            "worker exposes a status route."
        )

    import httpx

    url = trigger.status_url.format(workflow_id=workflow_id)
    try:
        async with httpx.AsyncClient(timeout=CLOUDFLARE_HTTP_TIMEOUT_S) as http:
            resp = await http.get(url)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as e:
        raise BackendError(f"Cloudflare status endpoint at {url} failed: {e}") from e

    return {
        "workflow_id": workflow_id,
        "status": data.get("status", data),
        "runtime": "cloudflare",
    }
