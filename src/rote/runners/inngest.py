"""Local orchestration of an emitted Inngest app.

``rote run`` on an inngest target: npm install (plus ``inngest-cli``,
whose postinstall downloads the platform binary), ``tsc`` build, then
two processes — the emitted serve entrypoint (``node dist/index.js``)
and the Inngest dev server (``inngest-cli dev``). Mechanics proven live
in ``tests/test_inngest_e2e.py``, including its three empirical traps:

* With ``--no-poll``, the dev server's single startup sync can race the
  app boot and never register it — an explicit ``PUT`` to the serve
  handler forces registration (retried until acknowledged).
* ``GET /v1/events/<id>/runs`` is trusted only for the ``run_id`` — its
  status field reported a stale ``Completed`` while the run was parked.
  ``GET /v1/runs/<run_id>`` is the truthful status.
* The run's return value is only available via the dev server's GraphQL
  API (``POST /v0/gql``); the v1 REST ``output`` field comes back empty.

HITL gate delivery: Inngest events broadcast and are **dropped** when no
active ``waitForEvent`` matches (no buffering for unstarted waits), so
instead of inferring parking, the runner re-broadcasts every pending
gate event on each poll tick until the run completes. Each wait consumes
exactly one matching event and unmatched extras vanish, so re-sending is
idempotent for distinct gate signals; a pipeline re-using one signal in
a loop would need per-iteration pacing (not supported here).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import URLError

from rote.eval.empirical import EmpiricalError, MeasuredRun
from rote.ir import Pipeline
from rote.runners._node import (
    ensure_npm_install,
    find_free_port,
    http_json,
    terminate,
)

#: Version pin for the dev-server binary — matches the version the e2e
#: suite proved the GraphQL/registration behavior against.
INNGEST_CLI_SPEC = "inngest-cli@^1.34.0"

READY_TIMEOUT_SECONDS = 90.0
REGISTER_TIMEOUT_SECONDS = 60.0


def _ensure_toolchain(app_dir: Path) -> Path:
    """npm install (+ inngest-cli with its binary postinstall) and tsc build."""
    cli = app_dir / "node_modules" / ".bin" / "inngest-cli"
    if not cli.exists():
        # --ignore-scripts=false: the postinstall downloads the platform
        # binary and must run even if npm config disables scripts globally.
        ensure_npm_install(app_dir, "--ignore-scripts=false", INNGEST_CLI_SPEC)
        if not cli.exists():
            rebuild = subprocess.run(
                ["npm", "rebuild", "--ignore-scripts=false", "inngest-cli"],
                cwd=app_dir,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if rebuild.returncode != 0 or not cli.exists():
                raise EmpiricalError(
                    "inngest-cli binary missing after install + rebuild:\n"
                    f"{(rebuild.stderr or rebuild.stdout)[:500]}"
                )
    if not (app_dir / "dist" / "index.js").is_file():
        build = subprocess.run(
            ["npx", "--no-install", "tsc"],
            cwd=app_dir,
            capture_output=True,
            text=True,
            timeout=180,
        )
        if build.returncode != 0:
            raise EmpiricalError(f"tsc build failed:\n{(build.stdout or build.stderr)[:800]}")
    return cli


def _wait_until_healthy(url: str, proc: subprocess.Popen[Any], log_path: Path, what: str) -> None:
    deadline = time.monotonic() + READY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise EmpiricalError(
                f"{what} exited prematurely (code {proc.returncode}); log tail:\n"
                f"{log_path.read_text(encoding='utf-8', errors='replace')[-800:]}"
            )
        try:
            http_json("GET", url, timeout=2.0)
            return
        except (URLError, TimeoutError, ConnectionResetError, json.JSONDecodeError):
            time.sleep(0.5)
    raise EmpiricalError(
        f"{what} did not become healthy in {READY_TIMEOUT_SECONDS:g}s; log tail:\n"
        f"{log_path.read_text(encoding='utf-8', errors='replace')[-800:]}"
    )


def _force_registration(app_port: int) -> None:
    """PUT the serve handler until the dev server acknowledges the app."""
    deadline = time.monotonic() + REGISTER_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        try:
            ack = http_json("PUT", f"http://127.0.0.1:{app_port}/")
            if "registered" in str(ack.get("message", "")).lower():
                return
        except (URLError, TimeoutError, ConnectionResetError, json.JSONDecodeError):
            pass
        time.sleep(1)
    raise EmpiricalError("the emitted app never registered with the inngest dev server")


def _run_output(dev_base: str, run_id: str) -> dict[str, Any]:
    """The function's return value via GraphQL (v1 REST output is empty)."""
    query = f"{{ run(runID: {json.dumps(run_id)}) {{ status output }} }}"
    out = http_json("POST", f"{dev_base}/v0/gql", {"query": query})
    ops = json.loads(out["data"]["run"]["output"])
    completes = [op for op in ops if op.get("op") == "RunComplete"]
    if not completes:
        raise EmpiricalError(f"run {run_id} completed but no RunComplete op in {ops!r}")
    result = completes[0].get("data")
    return result if isinstance(result, dict) else {"result": result}


def run_inngest(
    app_dir: Path,
    input_payload: dict[str, Any],
    *,
    pipeline: Pipeline,
    signals: dict[str, Any],
    gate_order: list[str],
    timeout_seconds: float = 600.0,
) -> MeasuredRun:
    """Run an emitted Inngest app once against a managed dev server.

    ``pipeline`` is required for event naming: the trigger is
    ``<pipeline>/run.requested`` and each gate waits on
    ``<pipeline>/<signal>``. The measured wall clock covers trigger →
    terminal, not toolchain startup.
    """
    from rote.adapters.inngest import gate_event_name, trigger_event_name

    cli = _ensure_toolchain(app_dir)
    signal_events = {
        n.signal: gate_event_name(pipeline, n) for n in pipeline.nodes if n.signal is not None
    }

    app_port = find_free_port()
    dev_port = find_free_port()
    dev_base = f"http://127.0.0.1:{dev_port}"
    app_log_path = app_dir / "rote-run-app.log"
    dev_log_path = app_dir / "rote-run-dev.log"

    with (
        app_log_path.open("w", encoding="utf-8") as app_log,
        dev_log_path.open("w", encoding="utf-8") as dev_log,
    ):
        app_proc = subprocess.Popen(
            ["node", "dist/index.js"],
            cwd=app_dir,
            stdout=app_log,
            stderr=subprocess.STDOUT,
            env={
                **os.environ,
                "INNGEST_DEV": dev_base,
                "PORT": str(app_port),
            },
        )
        dev_proc = subprocess.Popen(
            [
                str(cli),
                "dev",
                "-u",
                f"http://127.0.0.1:{app_port}",
                "--no-discovery",
                "--no-poll",
                "-p",
                str(dev_port),
            ],
            cwd=app_dir,
            stdout=dev_log,
            stderr=subprocess.STDOUT,
        )
        try:
            _wait_until_healthy(f"{dev_base}/health", dev_proc, dev_log_path, "inngest dev server")
            # The dev server serves its dashboard on the same port.
            print(
                f"rote run: inngest dev UI: {dev_base} (live while the run lasts)",
                file=sys.stderr,
            )
            _wait_until_healthy(
                f"http://127.0.0.1:{app_port}/", app_proc, app_log_path, "emitted app"
            )
            _force_registration(app_port)

            started = time.monotonic()
            deadline = started + timeout_seconds
            sent = http_json(
                "POST",
                f"{dev_base}/e/dev",
                {"name": trigger_event_name(pipeline), "data": input_payload},
            )
            event_id = str(sent["ids"][0])

            run_id: str | None = None
            while run_id is None:
                if time.monotonic() > deadline:
                    raise EmpiricalError(f"no run appeared for trigger event {event_id}")
                runs = http_json("GET", f"{dev_base}/v1/events/{event_id}/runs")["data"]
                if runs:
                    run_id = str(runs[0]["run_id"])
                else:
                    time.sleep(0.5)

            pending = [(g, signals[g]) for g in gate_order if g in signals]
            while True:
                if time.monotonic() > deadline:
                    return MeasuredRun(
                        wall_seconds=time.monotonic() - started,
                        output=None,
                        error=f"timed out after {timeout_seconds:g}s (run {run_id})",
                    )
                status = http_json("GET", f"{dev_base}/v1/runs/{run_id}")["data"]["status"]
                if status == "Completed":
                    return MeasuredRun(
                        wall_seconds=time.monotonic() - started,
                        output=_run_output(dev_base, run_id),
                    )
                if status in ("Failed", "Cancelled"):
                    return MeasuredRun(
                        wall_seconds=time.monotonic() - started,
                        output=None,
                        error=(
                            f"run {status.lower()}; app log tail:\n"
                            f"{app_log_path.read_text(encoding='utf-8', errors='replace')[-500:]}"
                        ),
                    )
                # Re-broadcast every pending gate payload — consumed when
                # its waitForEvent is active, dropped otherwise.
                for gate, payload in pending:
                    http_json(
                        "POST",
                        f"{dev_base}/e/dev",
                        {
                            "name": signal_events.get(gate, f"{pipeline.name}/{gate}"),
                            "data": payload if isinstance(payload, dict) else {"payload": payload},
                        },
                    )
                time.sleep(1.0)
        finally:
            terminate(app_proc, dev_proc)
