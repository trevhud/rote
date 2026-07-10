"""Tests for the graduation event machinery.

Covers the wire helpers every driver funnels progress through:

* ``parse_progress_lines`` — the progress.ndjson → phase-event parser,
  including its tolerance for malformed / partial lines.
* ``emit_safely`` — a raising sink must never propagate.
* ``relative_display_path`` — work-dir-relative rendering with sane
  fallbacks.
* ``ProgressFileWatcher`` — incremental polling plus the final flush on
  stop that guarantees a phase written just before exit isn't lost.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from rote.graduator.events import (
    PROGRESS_FILENAME,
    GraduationEvent,
    ProgressFileWatcher,
    emit_safely,
    parse_progress_lines,
    relative_display_path,
)

# ───────── parse_progress_lines ─────────


def test_parse_progress_lines_basic() -> None:
    text = '{"phase": 1, "name": "Intake"}\n{"phase": 2, "name": "Node Classification"}\n'
    events = parse_progress_lines(text)
    assert [e.type for e in events] == ["phase", "phase"]
    assert [e.phase for e in events] == [1, 2]
    assert events[0].phase_name == "Intake"
    assert events[0].message == "phase 1: Intake"


def test_parse_progress_lines_tolerates_malformed_and_blank() -> None:
    text = (
        "\n"
        "not json\n"
        '{"phase": 1, "name": "Intake"}\n'
        "{ half-written\n"  # partial line, e.g. mid-flush
        '{"phase": 2}\n'  # no name is fine
        '{"name": "no phase"}\n'  # missing phase → skipped
        '{"phase": "3"}\n'  # phase not an int → skipped
    )
    events = parse_progress_lines(text)
    assert [e.phase for e in events] == [1, 2]
    assert events[1].phase_name is None
    assert events[1].message == "phase 2"


def test_parse_progress_lines_rejects_bool_phase() -> None:
    # bool is an int subclass in Python; a JSON ``true`` must not count.
    assert parse_progress_lines('{"phase": true, "name": "x"}\n') == []


def test_parse_progress_lines_empty() -> None:
    assert parse_progress_lines("") == []


# ───────── emit_safely ─────────


def test_emit_safely_swallows_sink_exceptions() -> None:
    def boom(_event: GraduationEvent) -> None:
        raise RuntimeError("UI bug")

    # Must not raise.
    emit_safely(boom, GraduationEvent(type="log", ts=0.0, message="x"))


def test_emit_safely_none_sink_is_noop() -> None:
    emit_safely(None, GraduationEvent(type="log", ts=0.0, message="x"))


def test_emit_safely_forwards_event() -> None:
    seen: list[GraduationEvent] = []
    event = GraduationEvent(type="turn", ts=1.0, turn=3, message="turn 3")
    emit_safely(seen.append, event)
    assert seen == [event]


# ───────── relative_display_path ─────────


def test_relative_display_path_inside_work_dir(tmp_path: Path) -> None:
    target = tmp_path / "signatures" / "qualify.ts"
    assert relative_display_path(str(target), tmp_path) == str(Path("signatures") / "qualify.ts")


def test_relative_display_path_outside_falls_back_to_raw(tmp_path: Path) -> None:
    outside = "/etc/hosts"
    assert relative_display_path(outside, tmp_path) == outside


def test_relative_display_path_none() -> None:
    assert relative_display_path(None, Path("/tmp")) is None


# ───────── ProgressFileWatcher ─────────


@pytest.mark.asyncio
async def test_watcher_polls_incrementally(tmp_path: Path) -> None:
    events: list[GraduationEvent] = []
    progress = tmp_path / PROGRESS_FILENAME

    async with ProgressFileWatcher(tmp_path, events.append, poll_interval=0.02):
        progress.write_text('{"phase": 1, "name": "Intake"}\n', encoding="utf-8")
        await asyncio.sleep(0.1)
        assert [e.phase for e in events] == [1]

        progress.write_text(
            '{"phase": 1, "name": "Intake"}\n{"phase": 2, "name": "Classify"}\n',
            encoding="utf-8",
        )
        await asyncio.sleep(0.1)
        assert [e.phase for e in events] == [1, 2]


@pytest.mark.asyncio
async def test_watcher_final_flush_on_stop(tmp_path: Path) -> None:
    """A phase written between the last poll and stop is still delivered."""
    events: list[GraduationEvent] = []
    progress = tmp_path / PROGRESS_FILENAME

    # A poll interval far longer than the test guarantees the only read
    # that can see the file is the final flush in stop().
    watcher = ProgressFileWatcher(tmp_path, events.append, poll_interval=1000)
    await watcher.__aenter__()
    progress.write_text('{"phase": 7, "name": "Graduation Report"}\n', encoding="utf-8")
    await watcher.__aexit__()

    assert [e.phase for e in events] == [7]


@pytest.mark.asyncio
async def test_watcher_no_file_is_harmless(tmp_path: Path) -> None:
    events: list[GraduationEvent] = []
    async with ProgressFileWatcher(tmp_path, events.append, poll_interval=0.02):
        await asyncio.sleep(0.05)
    assert events == []


@pytest.mark.asyncio
async def test_watcher_does_not_double_emit(tmp_path: Path) -> None:
    events: list[GraduationEvent] = []
    progress = tmp_path / PROGRESS_FILENAME
    progress.write_text('{"phase": 1, "name": "Intake"}\n', encoding="utf-8")

    async with ProgressFileWatcher(tmp_path, events.append, poll_interval=0.02):
        await asyncio.sleep(0.1)  # several polls of the same unchanged file

    assert [e.phase for e in events] == [1]
