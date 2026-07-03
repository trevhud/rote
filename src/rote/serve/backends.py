"""Runtime trigger backends for ``rote serve``.

Each registry entry names a runtime trigger (Temporal, Cloudflare, or
DBOS). This module knows how to *start* a graduated workflow on that
runtime, how to *poll* its status, and — where the runtime makes it a
single durable call (DBOS) — how to *signal* a HITL gate. Tool calls
never block on workflow completion — graduated pipelines run for
minutes to days (HITL gates), and their durability lives in the
workflow engine, not in this process. Starting returns
``{workflow_id, status: "started", ...}`` immediately; the companion
``<tool>_status`` MCP tool polls via :func:`workflow_status`.

Clients are connected lazily, per call, so ``rote serve`` starts (and
lists tools) even when the runtime is unreachable — the error surfaces
on the tool call that needs it, with a message that says exactly what
was unreachable.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import uuid
from typing import Any

from rote.serve.registry import (
    CloudflareTrigger,
    DbosTrigger,
    RegistryEntry,
    TemporalTrigger,
)

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
    if isinstance(trigger, DbosTrigger):
        return await _start_dbos(trigger, entry.name, payload)
    return await _start_cloudflare(trigger, payload)


async def workflow_status(entry: RegistryEntry, workflow_id: str) -> dict[str, Any]:
    """Poll the status of a previously started workflow."""
    trigger = entry.trigger
    if isinstance(trigger, TemporalTrigger):
        return await _status_temporal(trigger, workflow_id)
    if isinstance(trigger, DbosTrigger):
        return await _status_dbos(trigger, workflow_id)
    return await _status_cloudflare(trigger, workflow_id)


async def signal_workflow(
    entry: RegistryEntry, workflow_id: str, signal: str, payload: dict[str, Any]
) -> dict[str, Any]:
    """Resume a workflow parked at a HITL gate by delivering its signal.

    Only DBOS triggers support this today — ``DBOSClient.send`` is a
    single durable call from any process that can reach the system
    database. Temporal/Cloudflare gates are resumed with the runtime's
    own tooling for now.
    """
    trigger = entry.trigger
    if isinstance(trigger, DbosTrigger):
        return await _signal_dbos(trigger, workflow_id, signal, payload)
    raise BackendError(
        f"Signaling is only supported for the dbos runtime; tool "
        f"{entry.name!r} runs on {trigger.runtime}. Resume this gate with "
        f"the runtime's own tooling instead."
    )


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


# ───────── DBOS ─────────
#
# DBOS has no orchestrator: the trigger contract is the *system database*
# shared with the emitted app process. `DBOSClient.enqueue` inserts an
# ENQUEUED workflow row; the app (running `python main.py --serve` or
# `dbos start` against the same database, with the matching registered
# workflow name and queue) dequeues and executes it. Enqueueing with no
# app running never fails — the run just stays in `enqueued` status —
# which is why the status tool reports that state verbatim.


def _redacted_db_url(url: str) -> str:
    """Strip the password from a database URL for error messages."""
    return re.sub(r"(://[^/@:]+):[^@/]+@", r"\1:***@", url)


async def _dbos_client(trigger: DbosTrigger) -> Any:
    try:
        from dbos import DBOSClient
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise BackendError(
            "dbos is not installed. Install the DBOS extra: pip install 'rote[dbos]'"
        ) from e

    # Construction connects eagerly (DBOSClient runs a connection check),
    # and it is synchronous — run it off the event loop.
    try:
        return await asyncio.to_thread(DBOSClient, system_database_url=trigger.system_database_url)
    except Exception as e:
        raise BackendError(
            f"DBOS system database at {_redacted_db_url(trigger.system_database_url)} "
            f"is unreachable: {e}"
        ) from e


async def _destroy_dbos_client(client: Any) -> None:
    """Release the client's connection pool; never mask the real error."""
    with contextlib.suppress(Exception):
        await asyncio.to_thread(client.destroy)


async def _start_dbos(
    trigger: DbosTrigger, tool_name: str, payload: dict[str, Any]
) -> dict[str, Any]:
    client = await _dbos_client(trigger)
    workflow_id = f"{tool_name}-{uuid.uuid4().hex[:12]}"
    try:
        await client.enqueue_async(
            {
                "workflow_name": trigger.workflow_name,
                "queue_name": trigger.queue_name,
                "workflow_id": workflow_id,
            },
            payload,
        )
    except Exception as e:
        raise BackendError(
            f"DBOS system database at {_redacted_db_url(trigger.system_database_url)} "
            f"refused to enqueue workflow {trigger.workflow_name!r} on queue "
            f"{trigger.queue_name!r}: {e}"
        ) from e
    finally:
        await _destroy_dbos_client(client)
    return {
        "workflow_id": workflow_id,
        "status": "started",
        "runtime": "dbos",
    }


async def _status_dbos(trigger: DbosTrigger, workflow_id: str) -> dict[str, Any]:
    client = await _dbos_client(trigger)
    try:
        handle = await client.retrieve_workflow_async(workflow_id)
        status = await handle.get_status()
    except Exception as e:
        raise BackendError(
            f"DBOS system database at {_redacted_db_url(trigger.system_database_url)} "
            f"could not retrieve workflow {workflow_id!r}: {e}"
        ) from e
    finally:
        await _destroy_dbos_client(client)
    # DBOS statuses: ENQUEUED, DELAYED, PENDING, SUCCESS, ERROR, CANCELLED,
    # MAX_RECOVERY_ATTEMPTS_EXCEEDED. Lowercased to match the Temporal
    # backend's convention. Note `enqueued` means no app process has
    # dequeued the run yet — the emitted app is probably not running.
    return {
        "workflow_id": workflow_id,
        "status": str(status.status).lower(),
        "runtime": "dbos",
    }


async def _signal_dbos(
    trigger: DbosTrigger, workflow_id: str, signal: str, payload: dict[str, Any]
) -> dict[str, Any]:
    # Sending to a topic no gate ever recv's on succeeds silently (the
    # message just sits in the database), so an unknown signal name would
    # be a debugging trap. The register-time gate list closes it.
    if trigger.gate_signals and signal not in trigger.gate_signals:
        raise BackendError(
            f"Unknown signal {signal!r}: this pipeline's HITL gates listen on "
            f"{trigger.gate_signals}. A message sent to any other topic is "
            f"silently ignored, so it is rejected instead."
        )
    client = await _dbos_client(trigger)
    try:
        await client.send_async(workflow_id, payload, topic=signal)
    except Exception as e:
        raise BackendError(
            f"DBOS system database at {_redacted_db_url(trigger.system_database_url)} "
            f"could not deliver signal {signal!r} to workflow {workflow_id!r}: {e}"
        ) from e
    finally:
        await _destroy_dbos_client(client)
    return {
        "workflow_id": workflow_id,
        "signal": signal,
        "status": "signaled",
        "runtime": "dbos",
    }
