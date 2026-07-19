"""Local orchestration of an emitted Cloudflare Workflows app.

``rote run`` on a cloudflare target: ``npm install`` (first run),
``npx wrangler dev --local``, then drive the emitted ``src/index.ts``
routes (``POST /start``, ``GET /status/<id>``,
``POST /event/<id>/<type>``). Mechanics proven live in
``tests/test_cloudflare_e2e.py`` — including the one non-obvious part:
local-dev keeps the instance's top-level ``status`` at ``"running"``
while it is parked inside ``step.waitForEvent`` (there is no
``"waiting"`` enum), so parking is inferred by watching
``__LOCAL_DEV_STEP_OUTPUTS`` stop growing across consecutive polls.
Gate payloads are therefore delivered parked-then-send, one gate per
park, in the pipeline's topological gate order.

Caveat that follows from the stability heuristic: a single step that
stays silent longer than ``stability_seconds`` can be mistaken for a
park, which delivers the next gate event early. Cloudflare buffers
events per instance, so an early event is consumed when its
``waitForEvent`` registers — harmless for distinct gate signals, but
raise ``--timeout``/stability if a pipeline re-uses one signal name in
a loop.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError

from rote.eval.empirical import EmpiricalError, MeasuredRun
from rote.runners._node import (
    ensure_npm_install,
    find_free_port,
    http_get_json,
    http_post_json,
    terminate,
)

#: wrangler dev cold start is ~5-10s; CI machines need headroom.
READY_TIMEOUT_SECONDS = 60.0


def _wait_until_parked_or_complete(
    port: int,
    instance_id: str,
    *,
    deadline: float,
    poll_interval: float = 0.5,
    stability_polls: int = 4,
) -> dict[str, Any]:
    """Poll ``/status/<id>`` until terminal state or inferred parking."""
    last_state: dict[str, Any] = {}
    last_count = -1
    stable = 0
    while time.monotonic() < deadline:
        try:
            last_state = http_get_json(f"http://127.0.0.1:{port}/status/{instance_id}")
        except (URLError, TimeoutError, ConnectionResetError):
            time.sleep(poll_interval)
            continue
        status = last_state.get("status")
        if status in ("complete", "errored", "terminated"):
            return last_state
        count = len(last_state.get("__LOCAL_DEV_STEP_OUTPUTS", []))
        if count == last_count:
            stable += 1
            if stable >= stability_polls:
                return last_state
        else:
            stable = 0
            last_count = count
        time.sleep(poll_interval)
    raise EmpiricalError(
        f"timed out waiting on workflow instance {instance_id!r}; last state: "
        f"{json.dumps(last_state)[:500]}"
    )


def run_cloudflare(
    app_dir: Path,
    input_payload: dict[str, Any],
    *,
    signals: dict[str, Any],
    gate_order: list[str],
    timeout_seconds: float = 600.0,
) -> MeasuredRun:
    """Run an emitted Cloudflare Workflows app once under ``wrangler dev``.

    ``gate_order`` is the pipeline's HITL gate signal names in
    topological order; each inferred park delivers the next one's
    payload from ``signals``. The measured wall clock covers the
    workflow run itself (create → terminal), not npm install or the
    wrangler cold start.
    """
    ensure_npm_install(app_dir)
    port = find_free_port()
    log_path = app_dir / "wrangler-dev.log"
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            ["npx", "wrangler", "dev", "--port", str(port), "--local"],
            cwd=app_dir,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            _wait_for_ready(proc, port, log_path)

            started = time.monotonic()
            deadline = started + timeout_seconds
            create = http_post_json(f"http://127.0.0.1:{port}/start", input_payload)
            instance_id = str(create["id"])
            pending = [(g, signals[g]) for g in gate_order if g in signals]

            while True:
                state = _wait_until_parked_or_complete(port, instance_id, deadline=deadline)
                status = state.get("status")
                if status == "complete":
                    wall = time.monotonic() - started
                    output = state.get("output")
                    if isinstance(output, dict):
                        return MeasuredRun(wall_seconds=wall, output=output)
                    return MeasuredRun(
                        wall_seconds=wall,
                        output={"result": output} if output is not None else None,
                    )
                if status in ("errored", "terminated"):
                    return MeasuredRun(
                        wall_seconds=time.monotonic() - started,
                        output=None,
                        error=f"workflow {status}: {json.dumps(state)[:800]}",
                    )
                if not pending:
                    return MeasuredRun(
                        wall_seconds=time.monotonic() - started,
                        output=None,
                        error=(
                            "workflow is parked but every gate payload has been "
                            f"delivered — last state: {json.dumps(state)[:500]}"
                        ),
                    )
                gate, payload = pending.pop(0)
                http_post_json(
                    f"http://127.0.0.1:{port}/event/{instance_id}/{gate}",
                    payload if isinstance(payload, dict) else {"payload": payload},
                )
        finally:
            terminate(proc)


def _wait_for_ready(proc: subprocess.Popen[Any], port: int, log_path: Path) -> None:
    """Block until wrangler dev answers HTTP (any response counts)."""
    ready_deadline = time.monotonic() + READY_TIMEOUT_SECONDS
    while time.monotonic() < ready_deadline:
        if proc.poll() is not None:
            raise EmpiricalError(
                f"wrangler dev exited prematurely (code {proc.returncode}); log tail:\n"
                f"{log_path.read_text(encoding='utf-8', errors='replace')[-800:]}"
            )
        try:
            http_get_json(f"http://127.0.0.1:{port}/healthz", timeout=2.0)
            return
        except HTTPError:
            return  # server answered (older emitted router without /healthz)
        except (URLError, TimeoutError, ConnectionResetError, json.JSONDecodeError):
            time.sleep(0.5)
    raise EmpiricalError(
        f"wrangler dev did not start within {READY_TIMEOUT_SECONDS:g}s on port {port}; "
        f"log tail:\n{log_path.read_text(encoding='utf-8', errors='replace')[-800:]}"
    )
