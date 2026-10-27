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
parking detection, no re-broadcast.

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
from pathlib import Path
from typing import Any
from uuid import uuid4

from rote.eval.empirical import EmpiricalError, MeasuredRun


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


async def _run_async(
    workflow_class: type,
    activities: list[Any],
    input_payload: dict[str, Any],
    signals: dict[str, Any],
    timeout_seconds: float,
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
    env = await WorkflowEnvironment.start_local(
        ui=True, ui_port=ui_port, dev_server_log_level="error"
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
            handle = await env.client.start_workflow(
                workflow_class.run,  # type: ignore[attr-defined]
                input_payload,
                id=f"rote-run-{uuid4()}",
                task_queue=task_queue,
            )
            # Temporal buffers signals server-side — send everything up
            # front and let each gate consume its payload on arrival.
            for name, payload in signals.items():
                await handle.signal(name, payload)
            try:
                result = await asyncio.wait_for(handle.result(), timeout=timeout_seconds)
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


def run_temporal(
    app_dir: Path,
    input_payload: dict[str, Any],
    *,
    signals: dict[str, Any],
    timeout_seconds: float = 600.0,
) -> MeasuredRun:
    """Run an emitted Temporal app once on a managed local dev server.

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
            _run_async(workflow_class, activities, input_payload, signals, timeout_seconds)
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
