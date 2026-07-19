"""Shared toolchain helpers for runners that orchestrate Node processes.

Used by the cloudflare and inngest runners (and dbos-ts when it lands):
free-port allocation, JSON-over-HTTP without extra dependencies, npm
install with a clear failure surface, and process teardown that never
leaves a dev server orphaned.
"""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from rote.eval.empirical import EmpiricalError

NPM_INSTALL_TIMEOUT_SECONDS = 300.0


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port: int = s.getsockname()[1]
        return port


def http_get_json(url: str, timeout: float = 10.0) -> dict[str, Any]:
    with urlopen(url, timeout=timeout) as resp:
        data: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
        return data


def http_json(
    method: str, url: str, body: dict[str, Any] | None = None, timeout: float = 30.0
) -> dict[str, Any]:
    payload = json.dumps(body).encode("utf-8") if body is not None else None
    req = Request(url, data=payload, headers={"Content-Type": "application/json"}, method=method)
    with urlopen(req, timeout=timeout) as resp:
        data: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
        return data


def http_post_json(url: str, body: dict[str, Any] | None, timeout: float = 10.0) -> dict[str, Any]:
    return http_json("POST", url, body or {}, timeout)


def ensure_npm_install(app_dir: Path, *extra_args: str) -> None:
    """Install the emitted app's node dependencies once (idempotent).

    ``extra_args`` extends the npm invocation (e.g. an extra package to
    install, or ``--ignore-scripts=false`` for packages whose postinstall
    downloads a binary). When extras are given, callers gate on their own
    marker instead of ``node_modules`` existing.
    """
    if not extra_args and (app_dir / "node_modules").is_dir():
        return
    if shutil.which("npm") is None:
        raise EmpiricalError("`npm` not found — a Node toolchain is required to run this runtime")
    print("rote run: installing npm dependencies (first run, ~30-60s)…", file=sys.stderr)
    proc = subprocess.run(
        ["npm", "install", "--no-audit", "--no-fund", *extra_args],
        cwd=app_dir,
        capture_output=True,
        text=True,
        timeout=NPM_INSTALL_TIMEOUT_SECONDS,
    )
    if proc.returncode != 0:
        raise EmpiricalError(
            f"npm install failed in {app_dir}:\n{(proc.stderr or proc.stdout)[:800]}"
        )


def terminate(*procs: subprocess.Popen[Any]) -> None:
    """Terminate child processes, escalating to kill — never orphan."""
    for proc in procs:
        proc.terminate()
    for proc in procs:
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
