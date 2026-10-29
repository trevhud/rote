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
a loop. A step sleeping between retry attempts looks the same, which is
why a "park" after the last gate payload is not an error: the runner
keeps waiting for a terminal state instead.

Local dev reports a failed step as a generic ``WorkflowFatalError``
("a step threw an NonRetryableError"), so the error a user needs (for
example "set ROTE_MCP_VENDOR_URL") is only in workerd's console output.
The runner reads it back from the wrangler log and reports it.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError

from rote.eval.empirical import EmpiricalError, MeasuredRun
from rote.mcp.parked import (
    AuthParkWatcher,
    LogMarkerChannel,
    ParkWatchFactory,
    parse_gate_park_markers,
    watches_park,
)
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
    park_watch: ParkWatchFactory | None = None,
    worker_vars: dict[str, str] | None = None,
) -> MeasuredRun:
    """Run an emitted Cloudflare Workflows app once under ``wrangler dev``.

    ``worker_vars`` reach the Worker's ``env`` as ``wrangler dev --var``
    (workerd does not see this process's environment); ``rote run``
    passes its read-only gate this way.

    ``gate_order`` is the pipeline's HITL gate signal names in
    topological order; each inferred park delivers the next one's
    payload from ``signals``. The measured wall clock covers the
    workflow run itself (create → terminal), not npm install or the
    wrangler cold start.

    ``park_watch`` reports MCP auth parks from the marker the emitted
    park prints (workerd console output lands in the wrangler log). It
    cannot release them: a Worker reads its own secrets, never the local
    token store, so the watcher shows the export fix instead. An auth
    park also looks parked to the stability heuristic, so while one is
    active no gate payload is spent on it.
    """
    ensure_npm_install(app_dir)
    log_path = app_dir / "wrangler-dev.log"
    proc, port = _start_wrangler(app_dir, log_path, worker_vars)
    try:
        started = time.monotonic()
        deadline = started + timeout_seconds
        create = http_post_json(f"http://127.0.0.1:{port}/start", input_payload)
        instance_id = str(create["id"])
        pending = [(g, signals[g]) for g in gate_order if g in signals]

        watcher = None
        if park_watch is not None and watches_park(app_dir):

            def remote_fix(server: str) -> str:
                return (
                    "a Worker reads its own secrets, not the local token store: "
                    f"rote mcp login {server}, then rote mcp export {server} >> "
                    f"{app_dir / '.dev.vars'} and re-run (Ctrl-C this one)"
                )

            watcher = park_watch(
                LogMarkerChannel(log_path, None, run_id=instance_id, remote_fix=remote_fix)
            )

        return _drive_instance(
            port,
            instance_id,
            pending,
            started=started,
            deadline=deadline,
            log_path=log_path,
            watcher=watcher,
        )
    finally:
        terminate(proc)


def _drive_instance(
    port: int,
    instance_id: str,
    pending: list[tuple[str, Any]],
    *,
    started: float,
    deadline: float,
    log_path: Path,
    watcher: AuthParkWatcher | None = None,
) -> MeasuredRun:
    """Deliver gate payloads on each inferred park until the run ends.

    A gate's payload goes out only once that gate has announced itself
    (its :data:`~rote.mcp.parked.GATE_PARK_MARKER` line, printed just before
    it waits). A gate with ``when:`` may be skipped and never announce:
    sending its payload on a park inferred from a slow step after it left
    an event nothing consumes, and the stuck-gate check below then failed a
    run that was fine (PR #313 review).

    A "park" with no announced gate to answer is usually a step sleeping
    between retry attempts or a slow step, so the runner keeps waiting for
    a terminal state (bounded by ``deadline``) and reports what actually
    failed. The exception is a stuck gate: the last payload was never
    consumed (a consumed payload adds a step output) and no step has
    logged an error, so nothing is retrying. That fails at once.
    """
    steps_at_last_payload: int | None = None
    answered: set[str] = set()
    while True:
        try:
            state = _wait_until_parked_or_complete(port, instance_id, deadline=deadline)
        except EmpiricalError as exc:
            if pending:
                raise
            raise EmpiricalError(
                f"every gate payload was delivered but the workflow did not finish: "
                f"{exc}{_step_errors_suffix(log_path)}"
            ) from None
        status = state.get("status")
        if status not in ("complete", "errored", "terminated") and watcher is not None:
            deadline += watcher.tick()
            if watcher.has_active_park:
                continue
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
                error=_failure_message(status, state, log_path),
            )
        steps = len(state.get("__LOCAL_DEV_STEP_OUTPUTS", []))
        announced = {g.signal for g in parse_gate_park_markers(_log_text(log_path))}
        ready = next((i for i, (gate, _) in enumerate(pending) if gate in announced), None)
        unanswerable = sorted(announced - answered - {gate for gate, _ in pending})
        if ready is None and unanswerable:
            return MeasuredRun(
                wall_seconds=time.monotonic() - started,
                output=None,
                error=(
                    f"workflow waits for signal(s) {', '.join(unanswerable)}, and no "
                    "--signal answers them. Check that each --signal name matches its "
                    "gate's signal"
                ),
            )
        if ready is None:
            if steps == steps_at_last_payload and not _step_errors(log_path):
                return MeasuredRun(
                    wall_seconds=time.monotonic() - started,
                    output=None,
                    error=(
                        "workflow is still parked after its last gate payload, which it "
                        "never consumed, and no step is retrying. Check that each "
                        "--signal name matches its gate's signal. Last state: "
                        f"{json.dumps(state)[:500]}"
                    ),
                )
            continue
        steps_at_last_payload = steps
        gate, payload = pending.pop(ready)
        answered.add(gate)
        http_post_json(
            f"http://127.0.0.1:{port}/event/{instance_id}/{gate}",
            payload if isinstance(payload, dict) else {"payload": payload},
        )


def _log_text(log_path: Path) -> str:
    try:
        return log_path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
#: workerd prints an uncaught step error as `✘ [ERROR] Uncaught <Name>: <message>`,
#: sometimes followed by a stray ` Error` (the start of its stack heading).
_UNCAUGHT_RE = re.compile(r"\[ERROR\]\s+Uncaught\s+(.+?)(?:\s+Error)?\s*$")


def _step_errors(log_path: Path) -> list[str]:
    """Distinct uncaught step errors from the wrangler log, newest last.

    Ordered by each error's latest occurrence, so an early failure that
    a retry recovered from never outranks the failure that ended the run.
    """
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return []
    errors: list[str] = []
    for line in _ANSI_RE.sub("", text).splitlines():
        match = _UNCAUGHT_RE.search(line)
        if match:
            if match.group(1) in errors:
                errors.remove(match.group(1))
            errors.append(match.group(1))
    return errors


def _step_errors_suffix(log_path: Path) -> str:
    errors = _step_errors(log_path)
    if not errors:
        return ""
    return "\nstep error: " + errors[-1]


#: workerd's console output reaches the log file after the status API
#: already says "errored", so a failure waits this long for the line.
STEP_ERROR_LOG_GRACE_SECONDS = 3.0


def _failure_message(status: str, state: dict[str, Any], log_path: Path) -> str:
    """``workflow errored: <error>``, with the real step error when the log has it."""
    error = state.get("error")
    if isinstance(error, dict) and error.get("message"):
        name = error.get("name")
        summary = f"{name}: {error['message']}" if name else str(error["message"])
    else:
        summary = json.dumps(state)[:800]
    errors = _step_errors(log_path)
    grace_deadline = time.monotonic() + STEP_ERROR_LOG_GRACE_SECONDS
    while status == "errored" and not errors and time.monotonic() < grace_deadline:
        time.sleep(0.2)
        errors = _step_errors(log_path)
    if errors and errors[-1] not in summary:
        summary += "\nstep error: " + errors[-1]
    return f"workflow {status}: {summary}"


#: How many free ports a start tries. The port is free when chosen and taken
#: a moment later by wrangler, so another process can take it in between:
#: parallel CI test workers did, and the run failed with "Address already in
#: use". A new port is a retry of the same start, never a different setup.
PORT_ATTEMPTS = 3


def _start_wrangler(
    app_dir: Path, log_path: Path, worker_vars: dict[str, str] | None
) -> tuple[subprocess.Popen[Any], int]:
    """``wrangler dev`` up and answering on a free port: ``(process, port)``."""
    for attempt in range(1, PORT_ATTEMPTS + 1):
        port = find_free_port()
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(
                [
                    "npx",
                    "wrangler",
                    "dev",
                    "--port",
                    str(port),
                    "--local",
                    *(
                        f
                        for k, v in (worker_vars or {}).items()
                        if v
                        for f in ("--var", f"{k}:{v}")
                    ),
                ],
                cwd=app_dir,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        try:
            _wait_for_ready(proc, port, log_path)
            return proc, port
        except EmpiricalError:
            terminate(proc)
            lost_port = "Address already in use" in log_path.read_text(
                encoding="utf-8", errors="replace"
            )
            if not lost_port or attempt == PORT_ATTEMPTS:
                raise
        except BaseException:
            terminate(proc)
            raise
    raise AssertionError("unreachable")  # the loop returns or raises


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
