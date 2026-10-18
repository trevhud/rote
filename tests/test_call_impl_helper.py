"""The ``_call_impl`` helper emitted into the python and dbos runtimes.

Executed for real here (the e2e suites run it inside emitted apps): a
plain function's result passes through, a coroutine function is driven
to completion, and that holds in a worker thread, which is where both
runtimes run parallel waves.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from rote.adapters._py_common import call_impl_helper, needs_call_impl
from rote.ir import Pipeline


def _helper() -> Any:
    namespace: dict[str, Any] = {"Any": Any}
    exec(call_impl_helper(), namespace)
    return namespace["_call_impl"]


def plain(**payload: Any) -> dict[str, Any]:
    return {"plain": payload["x"] + 1}


async def coroutine_fn(**payload: Any) -> dict[str, Any]:
    return {"async": payload["x"] * 2}


def test_plain_function_result_passes_through() -> None:
    assert _helper()(plain, {"x": 1}) == {"plain": 2}


def test_async_function_is_awaited() -> None:
    assert _helper()(coroutine_fn, {"x": 3}) == {"async": 6}


def test_async_function_is_awaited_in_a_worker_thread() -> None:
    call_impl = _helper()
    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(call_impl, coroutine_fn, {"x": 5})
        b = pool.submit(call_impl, plain, {"x": 5})
        assert (a.result(), b.result()) == ({"async": 10}, {"plain": 6})


def test_helper_is_emitted_only_when_a_node_calls_an_impl() -> None:
    def pipeline(nodes: list[dict[str, Any]]) -> Pipeline:
        return Pipeline.model_validate(
            {"name": "p", "input": {"type": "Brief"}, "nodes": nodes, "edges": []}
        )

    with_impl = pipeline(
        [{"id": "a", "kind": "pure_function", "description": "a", "impl": "m.py:f"}]
    )
    without = pipeline([{"id": "g", "kind": "hitl_gate", "description": "g", "signal": "go"}])
    assert needs_call_impl(with_impl)
    assert not needs_call_impl(without)
