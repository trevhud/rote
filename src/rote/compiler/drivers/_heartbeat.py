"""Keepalive events while awaiting a long LLM request.

The hosted platform kills a compilation container whose event feed goes
quiet (its hang detector). A healthy-but-slow model turn — or a hung
request sitting out the SDK's timeout-and-retry cycle — emits nothing for
minutes, which is indistinguishable from a wedged job. Wrapping the
request await with :func:`await_with_heartbeat` emits a ``log`` event on
an interval while the request is pending, so quiet means dead again.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, TypeVar

from rote.compiler.events import CompilationEvent, EventCallback, emit_safely

T = TypeVar("T")

#: Well under the platform's activity window (minutes) and long enough to
#: be silent for every normal turn.
HEARTBEAT_SECONDS = 120.0


async def await_with_heartbeat(
    pending: "asyncio.Future[T] | Any",
    on_event: EventCallback | None,
    label: str,
    interval: float = HEARTBEAT_SECONDS,
) -> T:
    """Await ``pending``, emitting a log event every ``interval`` seconds.

    ``pending`` is any awaitable (typically the SDK request coroutine).
    The result, and any exception, pass through unchanged.
    """
    task = asyncio.ensure_future(pending)
    waited = 0.0
    while True:
        done, _ = await asyncio.wait({task}, timeout=interval)
        if done:
            return task.result()
        waited += interval
        emit_safely(
            on_event,
            CompilationEvent(
                type="log",
                ts=time.time(),
                message=f"still waiting on {label} ({int(waited)}s)…",
            ),
        )
