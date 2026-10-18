"""The ``_call_impl`` / ``_run_async`` helpers emitted into the Python runtimes.

Executed for real here (the e2e suites also run them inside emitted
apps). The synchronous form (python, dbos) and the async form (temporal)
must accept the same shapes: a plain function, an ``async def``, and a
plain wrapper that returns a coroutine. The synchronous form must also
keep async state alive across calls and work from inside a running loop.
"""

from __future__ import annotations

import asyncio
import http.server
import inspect
import os
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx
import pytest

from rote.adapters._py_common import (
    async_runner_helper,
    call_impl_helper,
    has_legacy_judges,
    needs_call_impl,
)
from rote.ir import Pipeline


def _sync_helper() -> Callable[[Any, dict[str, Any]], Any]:
    namespace: dict[str, Any] = {
        "Any": Any,
        "asyncio": asyncio,
        "inspect": inspect,
        "os": os,
        "threading": threading,
    }
    exec(async_runner_helper() + "\n\n" + call_impl_helper(), namespace)
    fn: Callable[[Any, dict[str, Any]], Any] = namespace["_call_impl"]
    return fn


def _async_helper() -> Callable[[Any, dict[str, Any]], Any]:
    namespace: dict[str, Any] = {"Any": Any, "asyncio": asyncio, "inspect": inspect}
    exec(call_impl_helper(is_async=True), namespace)
    fn: Callable[[Any, dict[str, Any]], Any] = namespace["_call_impl"]
    return fn


def plain(**payload: Any) -> dict[str, Any]:
    return {"plain": payload["x"] + 1}


async def coroutine_fn(**payload: Any) -> dict[str, Any]:
    return {"async": payload["x"] * 2}


def wrapper_returning_coroutine(**payload: Any) -> Any:
    return coroutine_fn(**payload)


SHAPES = [
    (plain, {"plain": 2}),
    (coroutine_fn, {"async": 2}),
    (wrapper_returning_coroutine, {"async": 2}),
]


@pytest.mark.parametrize(("fn", "expected"), SHAPES, ids=lambda v: getattr(v, "__name__", ""))
def test_sync_helper_accepts_every_shape(fn: Any, expected: dict[str, Any]) -> None:
    assert _sync_helper()(fn, {"x": 1}) == expected


@pytest.mark.parametrize(("fn", "expected"), SHAPES, ids=lambda v: getattr(v, "__name__", ""))
def test_async_helper_accepts_every_shape(fn: Any, expected: dict[str, Any]) -> None:
    assert asyncio.run(_async_helper()(fn, {"x": 1})) == expected


def test_async_function_is_awaited_in_worker_threads() -> None:
    """Parallel waves call steps from pool threads."""
    call_impl = _sync_helper()
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(call_impl, coroutine_fn, {"x": i}) for i in range(8)]
        assert [f.result() for f in futures] == [{"async": 2 * i} for i in range(8)]


@pytest.fixture
def local_server() -> Iterator[str]:
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *args: Any) -> None:
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/"
    finally:
        server.shutdown()


def test_module_level_async_client_survives_repeated_calls(local_server: str) -> None:
    """A hand-written module often keeps one AsyncClient at module level.
    It binds to the loop it first ran on, so a loop per call (asyncio.run)
    fails every call after the first with "Event loop is closed"; that is
    every fan_out element and every retry."""
    client = httpx.AsyncClient()

    async def fetch(**payload: Any) -> dict[str, Any]:
        response = await client.get(local_server)
        return {"status": response.status_code, "n": payload["n"]}

    call_impl = _sync_helper()
    assert [call_impl(fetch, {"n": n}) for n in range(3)] == [
        {"status": 200, "n": n} for n in range(3)
    ]


def test_sync_helper_works_inside_a_running_loop() -> None:
    """An app that imports run_pipeline into async code (a FastAPI handler)
    calls steps with a loop already running on that thread, where
    asyncio.run raises."""
    call_impl = _sync_helper()

    async def handler() -> Any:
        return call_impl(coroutine_fn, {"x": 4})

    assert asyncio.run(handler()) == {"async": 8}


def test_sync_helper_refuses_to_deadlock_on_its_own_loop() -> None:
    call_impl = _sync_helper()

    async def reenters(**payload: Any) -> Any:
        return call_impl(coroutine_fn, payload)

    with pytest.raises(RuntimeError, match="would deadlock"):
        call_impl(reenters, {"x": 1})


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork()")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")  # fork() with threads alive
def test_a_forked_child_gets_its_own_loop() -> None:
    """Pre-fork servers (gunicorn, uvicorn workers) fork after the parent
    may already have run a step. The child inherits the loop globals but
    not the loop's thread, so without the at-fork reset its first async
    call blocks forever on a loop nothing runs."""
    call_impl = _sync_helper()
    assert call_impl(coroutine_fn, {"x": 1}) == {"async": 2}  # parent's loop is live

    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:  # child: one more async call, report it, exit without cleanup
        try:
            ok = call_impl(coroutine_fn, {"x": 21}) == {"async": 42}
            os.write(write_fd, b"ok" if ok else b"wrong")
        finally:
            os._exit(0)
    os.close(write_fd)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        done, _status = os.waitpid(pid, os.WNOHANG)
        if done:
            break
        time.sleep(0.05)
    else:
        os.kill(pid, 9)
        os.waitpid(pid, 0)
        pytest.fail("the forked child hung on its first async call")
    assert os.read(read_fd, 16) == b"ok"
    os.close(read_fd)
    # The parent's loop is untouched by the child's reset.
    assert call_impl(coroutine_fn, {"x": 3}) == {"async": 6}


def _pipeline(nodes: list[dict[str, Any]]) -> Pipeline:
    return Pipeline.model_validate(
        {"name": "p", "input": {"type": "Brief"}, "nodes": nodes, "edges": []}
    )


def test_helpers_are_emitted_only_when_needed() -> None:
    impl = _pipeline([{"id": "a", "kind": "pure_function", "description": "a", "impl": "m.py:f"}])
    gate = _pipeline([{"id": "g", "kind": "hitl_gate", "description": "g", "signal": "go"}])
    legacy = _pipeline(
        [{"id": "j", "kind": "llm_judge", "description": "j", "signature": "s.py:Judge"}]
    )
    assert needs_call_impl(impl) and not needs_call_impl(gate)
    assert has_legacy_judges(legacy) and not has_legacy_judges(impl)
