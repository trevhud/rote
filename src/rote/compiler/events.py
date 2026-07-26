"""Live progress events for a compilation run.

A compilation is a long (~13 min), real-money agent loop. Both the CLI
and the cloud container runner want to show what's happening while it
runs — which phase the agent is in, how many turns it's taken, which
files it's writing, how many tokens it's burned. This module defines
the wire schema those consumers agree on and the plumbing drivers use
to emit it.

# The wire schema is LOCKED

:class:`CompilationEvent` is json-serialized by the cloud container and
replayed to a browser. Its field names are a contract — a rename here
is a breaking change on the other side of a network boundary. Add
fields, never rename them.

# Two ways phases are detected

The agent announces phase transitions by rewriting ``progress.ndjson``
in its work dir (see the rote-compile SKILL.md). Two paths read it:

* **In-process (api driver):** intercepts the agent's ``write_file`` to
  ``progress.ndjson`` and parses the content directly.
* **Subprocess (claude / codex drivers):** can't see the agent's tool
  calls, so :class:`ProgressFileWatcher` polls the file on disk.

Both funnel through :func:`parse_progress_lines` so a phase event looks
identical regardless of which driver produced it.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

#: The file the agent rewrites to announce phase transitions. One JSON
#: object per line: ``{"phase": N, "name": "..."}``. Lives at the root
#: of the driver's work dir.
PROGRESS_FILENAME = "progress.ndjson"

EventType = Literal[
    "phase",
    "turn",
    "tool",
    "artifact",
    "log",
    "warning",
    "complete",
    "error",
]


@dataclass(frozen=True)
class CompilationEvent:
    """One live progress event from a compilation run.

    LOCKED wire schema — a cloud consumer json-serializes these and
    replays them to a browser. Keep field names exactly as-is; add
    fields rather than renaming.

    Every event carries a pre-rendered, human-readable :attr:`message`
    so a dumb consumer can print it without knowing the event taxonomy.
    """

    type: EventType
    ts: float
    phase: int | None = None
    phase_name: str | None = None
    turn: int | None = None
    #: Cumulative token usage: ``{"input": int, "output": int}``.
    tokens: dict[str, int] | None = None
    tool_name: str | None = None
    #: File path, relative to the work dir when the emitter can manage it.
    path: str | None = None
    message: str = ""


EventCallback = Callable[[CompilationEvent], None]
"""A sink for :class:`CompilationEvent`\\ s. Fired synchronously from the
driver / orchestrator; always invoked through :func:`emit_safely` so a
raising callback can't kill a paid run."""


def emit_safely(cb: EventCallback | None, event: CompilationEvent) -> None:
    """Fire ``cb(event)``, swallowing any exception it raises.

    A progress callback is UI code — a rendering bug in it must never
    take down a compilation that's already spent real money. ``None`` is
    accepted so callers can wire it unconditionally without a guard at
    every call site.
    """
    if cb is None:
        return
    # A UI bug must never kill a paid run. Deliberately no logging: the
    # sink is the thing that failed, so routing an error back through it
    # (or a logger it installed) risks a second failure. Drop it.
    with contextlib.suppress(Exception):
        cb(event)


def relative_display_path(path_str: str | None, work_dir: Path) -> str | None:
    """Best-effort path relative to ``work_dir`` for event display.

    Returns the work-dir-relative form when the path is inside it (the
    common case: the agent's ``pipeline.yaml`` / ``signatures/*`` writes),
    the raw string for paths outside it (reads of the source skill or
    rubric), and ``None`` when no path was given. Shared by every driver
    so a ``tool`` event's ``path`` looks the same regardless of backend.
    """
    if not path_str:
        return None
    try:
        return str(Path(path_str).resolve().relative_to(Path(work_dir).resolve()))
    except (ValueError, OSError):
        return path_str


def parse_progress_lines(text: str) -> list[CompilationEvent]:
    """Parse ``progress.ndjson`` content into ``phase`` events.

    Each non-blank line is expected to be ``{"phase": N, "name": "..."}``.
    Malformed lines (bad JSON, missing/invalid ``phase``) are tolerated
    and skipped — the file is written by an LLM mid-run and a half-
    flushed or fat-fingered line must not abort progress display.

    The returned events are in file order, which is the order the agent
    entered the phases.
    """
    events: list[CompilationEvent] = []
    now = time.time()
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        phase = obj.get("phase")
        if not isinstance(phase, int) or isinstance(phase, bool):
            continue
        name = obj.get("name")
        phase_name = str(name) if name is not None else None
        message = f"phase {phase}" + (f": {phase_name}" if phase_name else "")
        events.append(
            CompilationEvent(
                type="phase",
                ts=now,
                phase=phase,
                phase_name=phase_name,
                message=message,
            )
        )
    return events


class ProgressFileWatcher:
    """Polls ``work_dir/progress.ndjson`` and fires new phase events.

    Subprocess drivers (claude, codex) can't observe the agent's tool
    calls, so they can't intercept the ``progress.ndjson`` write the way
    the in-process api driver does. Instead this watcher polls the file
    every :attr:`poll_interval` seconds and fires a ``phase`` event for
    each line that appeared since the last read.

    Use it as an async context manager around the subprocess wait::

        async with ProgressFileWatcher(work_dir, on_event):
            await proc.communicate()

    On exit it does one final read so a phase the agent wrote just
    before the process ended isn't lost to poll timing.
    """

    def __init__(
        self,
        work_dir: Path,
        on_event: EventCallback | None,
        poll_interval: float = 0.5,
    ) -> None:
        self._path = Path(work_dir) / PROGRESS_FILENAME
        self._on_event = on_event
        self._poll_interval = poll_interval
        self._emitted = 0
        self._task: asyncio.Task[None] | None = None

    def _drain(self) -> None:
        """Emit phase events for any lines beyond what we've already sent."""
        try:
            text = self._path.read_text(encoding="utf-8")
        except (FileNotFoundError, OSError):
            return
        events = parse_progress_lines(text)
        for event in events[self._emitted :]:
            emit_safely(self._on_event, event)
        # Track by count so a partially-written final line (skipped by the
        # parser now, complete on the next poll) is picked up later.
        self._emitted = len(events)

    async def _poll_loop(self) -> None:
        while True:
            self._drain()
            await asyncio.sleep(self._poll_interval)

    def start(self) -> None:
        """Begin polling. Idempotent — a second call is a no-op."""
        if self._task is None:
            self._task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        """Stop polling and do a final read so no trailing phase is lost.

        The final :meth:`_drain` runs here rather than inside the poll
        loop's cancellation path: a task cancelled before it ever started
        never executes its body, so relying on the loop to flush would
        drop a phase written between ``start`` and a fast ``stop``.
        """
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        finally:
            self._task = None
        # Final read: catch a phase written between the last poll and now.
        self._drain()

    async def __aenter__(self) -> ProgressFileWatcher:
        self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()


__all__ = [
    "PROGRESS_FILENAME",
    "EventType",
    "CompilationEvent",
    "EventCallback",
    "emit_safely",
    "parse_progress_lines",
    "relative_display_path",
    "ProgressFileWatcher",
]
