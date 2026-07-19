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

    print(
        "rote run: starting a local Temporal dev server (first use downloads the binary)…",
        file=sys.stderr,
    )
    env = await WorkflowEnvironment.start_local()
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
                    error=f"workflow failed: {e.cause or e}",
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
    """Run an emitted Temporal app once on a managed local dev server."""
    try:
        import temporalio  # noqa: F401
    except ImportError as e:
        raise EmpiricalError(
            "running a temporal app requires the temporal extra: pip install 'rote-cli[temporal]'"
        ) from e
    workflow_class, activities = _load_app_modules(app_dir)
    return asyncio.run(
        _run_async(workflow_class, activities, input_payload, signals, timeout_seconds)
    )
