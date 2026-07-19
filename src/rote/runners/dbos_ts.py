"""Local orchestration of an emitted DBOS TypeScript app.

The emitted ``main.ts`` mirrors the DBOS Python app's one-shot CLI
contract exactly: ``node dist/main.js '<json>'`` prints
``workflow started: <id>`` to stderr, blocks on ``getResult()``, and
prints the result JSON to stdout. So the runner is the dbos-py flow
with a Node toolchain in front: npm install, ``tsc`` build, spawn,
harvest the workflow id, deliver gate payloads cross-process, wait.

Two DBOS-TS-specific facts (both proven in the e2e suites):

* The TS SDK is **Postgres-only** — no SQLite. The runner resolves the
  system database exactly like the emitted app
  (:func:`rote._dbos.dbos_ts_system_database_url`), probes it, and when
  unreachable falls back to a throwaway ``postgres:16-alpine`` Docker
  container for the duration of the run (loudly — durable state dies
  with the container; point ``DBOS_SYSTEM_DATABASE_URL`` at a real
  Postgres to keep it).
* Cross-language gate delivery (Python ``DBOSClient`` → TS ``DBOS.recv``)
  only works in DBOS's **portable** serialization — the same channel the
  MCP auth-release path uses (``tests/test_mcp_park_ts_e2e.py``). The
  defaults (pickle / superjson) are mutually unreadable.

Payloads are sent up front: DBOS notifications persist per-topic, so
the gate picks them up when it parks — no parking detection needed.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from rote._dbos import _load_dbos_config, dbos_ts_system_database_url
from rote.eval.empirical import EmpiricalError, MeasuredRun, _wait_for_workflow_id
from rote.runners._node import ensure_npm_install, terminate

PG_READY_TIMEOUT_SECONDS = 60.0


def _tcp_reachable(url: str) -> bool:
    import socket

    parts = urlsplit(url)
    host = parts.hostname or "localhost"
    port = parts.port or 5432
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


def _docker_available() -> bool:
    import shutil

    if shutil.which("docker") is None:
        return False
    probe = subprocess.run(["docker", "info"], capture_output=True, timeout=20)
    return probe.returncode == 0


def _start_throwaway_postgres(app_dir: Path) -> tuple[str, str]:
    """Boot a disposable Postgres container; return (container_name, url)."""
    from rote.runners._node import find_free_port

    name = f"rote-run-dbos-ts-{uuid.uuid4().hex[:8]}"
    port = find_free_port()
    run = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--rm",
            "--name",
            name,
            "-p",
            f"{port}:5432",
            "-e",
            "POSTGRES_PASSWORD=dbos",
            "postgres:16-alpine",
        ],
        capture_output=True,
        text=True,
        timeout=300,  # includes a possible first-time image pull
    )
    if run.returncode != 0:
        raise EmpiricalError(f"docker run postgres failed: {(run.stderr or run.stdout)[:500]}")
    deadline = time.monotonic() + PG_READY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        ready = subprocess.run(
            ["docker", "exec", name, "pg_isready", "-U", "postgres"],
            capture_output=True,
            timeout=10,
        )
        if ready.returncode == 0:
            break
        time.sleep(1)
    else:
        subprocess.run(["docker", "stop", name], capture_output=True, timeout=60)
        raise EmpiricalError("throwaway Postgres container did not become ready in 60s")
    config = _load_dbos_config(app_dir)
    import re

    db = re.sub(r"[^a-zA-Z0-9_]", "_", str(config.get("name", "pipeline"))) + "_dbos_sys"
    return name, f"postgresql://postgres:dbos@localhost:{port}/{db}"


def _send_signals_portable(url: str, workflow_id: str, signals: dict[str, Any]) -> None:
    """Deliver gate payloads Python → TS (portable serialization)."""
    try:
        from dbos import DBOSClient, WorkflowSerializationFormat
    except ImportError as e:
        raise EmpiricalError(
            "signaling a gated dbos-ts pipeline requires the dbos extra: "
            "pip install 'rote-cli[dbos]'"
        ) from e
    client = DBOSClient(system_database_url=url)
    for topic, payload in signals.items():
        client.send(
            workflow_id,
            payload,
            topic=topic,
            serialization_type=WorkflowSerializationFormat.PORTABLE,
        )


def _parse_trailing_json(stdout: str) -> dict[str, Any] | None:
    """The result document at the tail of a noisy stdout.

    The TS SDK prints its startup banner ("Running DBOS system database
    migrations...", "Initializing DBOS (v…)") to **stdout**, ahead of the
    emitted app's result JSON — found live, not in any doc. The result is
    the last JSON document starting at column 0, so scan backwards for a
    line opening a parseable document.
    """
    lines = stdout.splitlines()
    for i in range(len(lines) - 1, -1, -1):
        if not lines[i].startswith(("{", "[")):
            continue
        candidate = "\n".join(lines[i:])
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        return parsed if isinstance(parsed, dict) else {"result": parsed}
    return None


def _ensure_build(app_dir: Path) -> None:
    ensure_npm_install(app_dir)
    if (app_dir / "dist" / "main.js").is_file():
        return
    build = subprocess.run(
        ["npx", "--no-install", "tsc"],
        cwd=app_dir,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if build.returncode != 0:
        raise EmpiricalError(f"tsc build failed:\n{(build.stdout or build.stderr)[:800]}")


def run_dbos_ts(
    app_dir: Path,
    input_payload: dict[str, Any],
    *,
    signals: dict[str, Any],
    timeout_seconds: float = 600.0,
) -> MeasuredRun:
    """Run an emitted DBOS TypeScript app once against a real Postgres."""
    import sys

    _ensure_build(app_dir)

    container: str | None = None
    url = dbos_ts_system_database_url(app_dir)
    if not _tcp_reachable(url):
        if os.environ.get("DBOS_SYSTEM_DATABASE_URL"):
            raise EmpiricalError(
                f"DBOS_SYSTEM_DATABASE_URL points at an unreachable Postgres: {url}"
            )
        if not _docker_available():
            raise EmpiricalError(
                f"no Postgres reachable at {url} and Docker is unavailable — "
                "start one (`npx dbos postgres start`) or set "
                "DBOS_SYSTEM_DATABASE_URL"
            )
        print(
            "rote run: no Postgres reachable — booting a throwaway "
            "postgres:16-alpine container (durable state dies with it; set "
            "DBOS_SYSTEM_DATABASE_URL to keep state)…",
            file=sys.stderr,
        )
        container, url = _start_throwaway_postgres(app_dir)

    stdout_path = app_dir / "rote-run-stdout.log"
    stderr_path = app_dir / "rote-run-stderr.log"
    started = time.monotonic()
    try:
        with (
            stdout_path.open("w", encoding="utf-8") as out_f,
            stderr_path.open("w", encoding="utf-8") as err_f,
        ):
            proc = subprocess.Popen(
                ["node", "dist/main.js", json.dumps(input_payload)],
                cwd=app_dir,
                stdout=out_f,
                stderr=err_f,
                env={**os.environ, "DBOS_SYSTEM_DATABASE_URL": url},
            )
            try:
                if signals:
                    workflow_id = _wait_for_workflow_id(
                        stderr_path, proc, deadline=started + min(60.0, timeout_seconds)
                    )
                    _send_signals_portable(url, workflow_id, signals)
                proc.wait(timeout=timeout_seconds - (time.monotonic() - started))
            except subprocess.TimeoutExpired:
                terminate(proc)
                return MeasuredRun(
                    wall_seconds=time.monotonic() - started,
                    output=None,
                    error=f"timed out after {timeout_seconds:g}s",
                )
            except BaseException:
                # A parked workflow with no signal coming must not outlive
                # the runner — same rule as the dbos-py trial runner.
                terminate(proc)
                raise
        wall = time.monotonic() - started

        stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
        stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            return MeasuredRun(
                wall_seconds=wall,
                output=None,
                error=f"app exited {proc.returncode}: {(stderr or stdout)[:800]}",
            )
        output = _parse_trailing_json(stdout)
        if output is None:
            return MeasuredRun(
                wall_seconds=wall,
                output=None,
                error=f"no JSON result found in app stdout: {stdout[-300:]}",
            )
        return MeasuredRun(wall_seconds=wall, output=output)
    finally:
        if container is not None:
            subprocess.run(["docker", "stop", container], capture_output=True, timeout=60)
