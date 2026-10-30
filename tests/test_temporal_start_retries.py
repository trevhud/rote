"""A Temporal server that misses its 5-second start window is launched again (#261)."""

from __future__ import annotations

import asyncio

import pytest

from rote.runners.temporal import START_ATTEMPTS, start_with_retries

MISSED = RuntimeError(
    "Failed starting Temporal dev server: Failed connecting to test server after 5 seconds"
)


def test_a_missed_start_window_is_launched_again() -> None:
    calls: list[int] = []

    async def start() -> str:
        calls.append(1)
        if len(calls) < START_ATTEMPTS:
            raise MISSED
        return "env"

    assert asyncio.run(start_with_retries(start)) == "env"
    assert len(calls) == START_ATTEMPTS


def test_it_gives_up_after_the_last_attempt() -> None:
    calls: list[int] = []

    async def start() -> str:
        calls.append(1)
        raise MISSED

    with pytest.raises(RuntimeError, match="Failed starting"):
        asyncio.run(start_with_retries(start))
    assert len(calls) == START_ATTEMPTS


def test_any_other_error_is_raised_at_once() -> None:
    calls: list[int] = []

    async def start() -> str:
        calls.append(1)
        raise RuntimeError("temporal binary not found")

    with pytest.raises(RuntimeError, match="binary not found"):
        asyncio.run(start_with_retries(start))
    assert len(calls) == 1
