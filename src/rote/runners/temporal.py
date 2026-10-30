"""Local orchestration of an emitted Temporal app.

Unlike the other runtimes there is no server process for the runner to
supervise by hand: ``temporalio``'s ``WorkflowEnvironment.start_local()``
downloads (on first use) and manages a **real** Temporal dev server —
the same one ``temporal server start-dev`` runs — and the worker runs
in-process, because the emitted ``workflow.py`` / ``activities.py`` are
plain Python modules the runner can import straight from the app dir
and register on a throwaway task queue.

Gate delivery is the simplest of all five runtimes: Temporal buffers
signals server-side and the emitted handlers
(``@workflow.signal(name=<IR signal>)``) store the payload the moment it
arrives, so every payload is sent right after start and the
``wait_condition`` wakes whenever the workflow reaches its gate. No
parking detection, no re-broadcast. Which gates the run has reached, and
each one's rendered message, is the workflow's ``rote_gate_status`` query,
which the runner polls while it waits so ``rote run`` can show them.

The worker uses ``UnsandboxedWorkflowRunner``: the default workflow
sandbox restricts imports to a known-good allowlist and would reject
the emitted app's ``extracted.*`` / ``signatures.*`` modules.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import sys
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, TypeVar
from uuid import uuid4

from rote.eval.empirical import EmpiricalError, MeasuredRun
from rote.mcp.parked import (
    GATE_STATUS_EVENT,
    AuthPark,
    GateReached,
    ParkWatchFactory,
    app_reports_gates,
    gates_from_status,
)

T = TypeVar("T")


def _load_app_modules(app_dir: Path) -> tuple[type, list[Any]]:
    """Import the emitted workflow class + activity functions.

    The emitted modules use fixed top-level names (``workflow``,
    ``activities``, ``extracted``, ``signatures``), so stale entries
    from any earlier import are purged before the app dir goes on
    ``sys.path`` — otherwise a previous app's modules would shadow this
    one's.
    """
    if not (app_dir / "workflow.py").is_file() or not (app_dir / "activities.py").is_file():
        raise EmpiricalError(
            f"{app_dir} has no workflow.py/activities.py — pass an emitted temporal runtime dir"
        )
    for base in ("workflow", "activities", "extracted", "signatures"):
        for name in [m for m in sys.modules if m == base or m.startswith(f"{base}.")]:
            del sys.modules[name]
    sys.path.insert(0, str(app_dir))
    wf_mod = importlib.import_module("workflow")
    act_mod = importlib.import_module("activities")

    workflow_classes = [
        obj
        for obj in vars(wf_mod).values()
        if isinstance(obj, type) and hasattr(obj, "__temporal_workflow_definition")
    ]
    if len(workflow_classes) != 1:
        raise EmpiricalError(
            f"expected exactly one @workflow.defn class in {app_dir / 'workflow.py'}, "
            f"found {len(workflow_classes)}"
        )
    activities = [
        obj
        for obj in vars(act_mod).values()
        if callable(obj) and hasattr(obj, "__temporal_activity_definition")
    ]
    if not activities:
        raise EmpiricalError(f"no @activity.defn functions found in {app_dir / 'activities.py'}")
    return workflow_classes[0], activities


class TemporalGateChannel:
    """The gates a Temporal run has reached, from its ``rote_gate_status`` query.

    A Temporal run never parks on auth (the temporal runtime has no MCP
    backend), so :meth:`poll` is always empty. The query is async and the
    watcher is not, so the runner stores each answer in :attr:`status`.
    """

    external_release = False
    remote_fix: Callable[[str], str] | None = None

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.status: Any = None

    def poll(self) -> list[AuthPark]:
        return []

    def poll_gates(self) -> list[GateReached]:
        return gates_from_status(self.status, self.run_id)

    def release(self, park: AuthPark) -> None:
        raise RuntimeError(f"a Temporal run cannot park on auth for {park.server!r}")

    def close(self) -> None:
        pass


async def _run_async(
    workflow_class: type,
    activities: list[Any],
    input_payload: dict[str, Any],
    signals: dict[str, Any],
    timeout_seconds: float,
    park_watch: ParkWatchFactory | None = None,
    reports_gates: bool = False,
) -> MeasuredRun:
    from temporalio.client import WorkflowFailureError
    from temporalio.testing import WorkflowEnvironment
    from temporalio.worker import UnsandboxedWorkflowRunner, Worker

    from rote.runners._node import find_free_port

    print(
        "rote run: starting a local Temporal dev server (first use downloads the binary)…",
        file=sys.stderr,
    )
    # The dev server bundles the Temporal Web UI; an explicit port lets
    # us print a working URL (ui_port=None would pick one we can't see).
    ui_port = find_free_port()
    # dev_server_log_level="error": the dev server's own startup WARN
    # ("Cluster Id in Cluster Metadata config is not a valid uuid") comes
    # from its fresh in-memory persistence on every start, not from rote
    # or the app, and reads like a failure to a first-time user.
    env = await start_with_retries(
        lambda: WorkflowEnvironment.start_local(
            ui=True, ui_port=ui_port, dev_server_log_level="error"
        )
    )
    print(
        f"rote run: temporal dev UI: http://127.0.0.1:{ui_port} (live while the run lasts)",
        file=sys.stderr,
    )
    try:
        task_queue = f"rote-run-{uuid4()}"
        started = time.monotonic()
        async with Worker(
            env.client,
            task_queue=task_queue,
            workflows=[workflow_class],
            activities=activities,
            workflow_runner=UnsandboxedWorkflowRunner(),
        ):
            run_id = f"rote-run-{uuid4()}"
            handle = await env.client.start_workflow(
                workflow_class.run,  # type: ignore[attr-defined]
                input_payload,
                id=run_id,
                task_queue=task_queue,
            )
            # Temporal buffers signals server-side — send everything up
            # front and let each gate consume its payload on arrival.
            for name, payload in signals.items():
                await handle.signal(name, payload)
            channel = TemporalGateChannel(run_id)
            watcher = park_watch(channel) if park_watch is not None and reports_gates else None
            try:
                result = await asyncio.wait_for(
                    _result_watching_gates(handle, channel, watcher), timeout=timeout_seconds
                )
            except TimeoutError:
                return MeasuredRun(
                    wall_seconds=time.monotonic() - started,
                    output=None,
                    error=f"timed out after {timeout_seconds:g}s",
                )
            except WorkflowFailureError as e:
                return MeasuredRun(
                    wall_seconds=time.monotonic() - started,
                    output=None,
                    error=f"workflow failed: {_failure_chain(e)}",
                )
        return MeasuredRun(
            wall_seconds=time.monotonic() - started,
            output=result if isinstance(result, dict) else {"result": result},
        )
    finally:
        await env.shutdown()


async def _result_watching_gates(
    handle: Any, channel: TemporalGateChannel, watcher: Any, poll_interval: float = 0.5
) -> Any:
    """``handle.result()``, polling the gate query for ``watcher`` meanwhile.

    The query is asked once more after the run ends, so a gate reached and
    passed between two polls is still announced. Temporal answers a query on
    a failed run too, so the failure still surfaces as the run's own error.
    """
    result = asyncio.ensure_future(handle.result())
    if watcher is None:
        return await result
    try:
        while True:
            done, _ = await asyncio.wait({result}, timeout=poll_interval)
            channel.status = await handle.query(GATE_STATUS_EVENT)
            if done:
                watcher.drain()
                return result.result()
            watcher.tick()
    finally:
        result.cancel()


def run_temporal(
    app_dir: Path,
    input_payload: dict[str, Any],
    *,
    signals: dict[str, Any],
    timeout_seconds: float = 600.0,
    park_watch: ParkWatchFactory | None = None,
) -> MeasuredRun:
    """Run an emitted Temporal app once on a managed local dev server.

    ``park_watch`` builds the watcher that prints each gate the run
    reaches, with its message (see :class:`TemporalGateChannel`).

    The temporal extra is checked by :func:`rote.runners.run_pipeline`'s
    preflight, the one place every runner's install hint comes from.
    """
    workflow_class, activities = _load_app_modules(app_dir)
    # The python and dbos runners start the app with the app dir as its
    # working directory, so a relative path an extracted module writes
    # (expense-audit's reports/) lands inside the app. The temporal worker
    # runs in this process, so it gets the same directory for the run.
    with contextlib.chdir(app_dir):
        return asyncio.run(
            _run_async(
                workflow_class,
                activities,
                input_payload,
                signals,
                timeout_seconds,
                park_watch,
                app_reports_gates(app_dir),
            )
        )


def _failure_chain(error: BaseException) -> str:
    """Every message in a Temporal failure's ``cause`` chain, outermost first.

    The workflow's own failure is a wrapper ("Activity task failed"), and the
    exception the node's code raised is one or two causes further in. Showing
    only the first level hid it, so a user saw that a step failed but not why."""
    messages: list[str] = []
    current: BaseException | None = getattr(error, "cause", None) or error
    while current is not None:
        text = str(current) or type(current).__name__
        if text not in messages:
            messages.append(text)
        current = getattr(current, "cause", None)
    return ": ".join(messages)


#: How many times to start a Temporal server before giving up.
START_ATTEMPTS = 3


async def start_with_retries(start: Callable[[], Awaitable[T]]) -> T:
    """``start()``, run again when the server it launches does not answer in time.

    temporalio's dev and test servers must answer within 5 seconds of
    launch, a limit set in its Rust core that no Python option changes (as
    of 1.34). A loaded machine misses it now and then, and the launch fails
    with "Failed starting ... server" though nothing is wrong (#261, seen in
    CI). That failure is retried with a fresh launch, up to
    :data:`START_ATTEMPTS`; any other error, and the last one, is raised."""
    for attempt in range(1, START_ATTEMPTS + 1):
        try:
            return await start()
        except RuntimeError as e:
            if attempt == START_ATTEMPTS or "Failed starting" not in str(e):
                raise
    raise AssertionError("unreachable")
