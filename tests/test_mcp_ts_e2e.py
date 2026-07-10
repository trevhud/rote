"""Cross-language MCP credential e2e: Python logs in, TypeScript runs.

The token store is a contract, not a convention — this test proves it
end-to-end against a real OAuth-protected MCP server:

1. Python (`rote.mcp.auth.login`, the PR1 path) runs the real dance and
   seeds the store.
2. An emitted DBOS-TS app's MCP module — compiled with the app's own
   tsconfig against the real npm dependencies — makes an authenticated
   tool call reading that store.
3. The access token is forced stale on disk; the TS side must execute
   the refresh grant against the stored token_endpoint, succeed, and
   write the rotated credentials back where Python can read them.

Slow: real npm install + tsc + node, real uvicorn OAuth server.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

pytest.importorskip("fastmcp")
import httpx  # noqa: E402

from rote.adapters import get_adapter  # noqa: E402
from rote.ir import Pipeline  # noqa: E402
from rote.mcp import McpServerConfig, read_token_file, write_token_file  # noqa: E402
from rote.mcp.auth import login  # noqa: E402
from tests.test_mcp_oauth_e2e import SERVER_SCRIPT, _free_port, _wait_port  # noqa: E402


def _node_available() -> bool:
    return shutil.which("node") is not None and shutil.which("npm") is not None


@pytest.fixture(scope="module")
def protected_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    script = tmp_path_factory.mktemp("oauth-server") / "server.py"
    script.write_text(SERVER_SCRIPT, encoding="utf-8")
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, str(script), str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_port(port)
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def _headless_login(server_url: str) -> None:
    """PR1's real login path with the browser replaced by a GET thread."""
    import webbrowser

    real_open = webbrowser.open

    def fake_open(url: str, *args: object, **kwargs: object) -> bool:
        def follow() -> None:
            deadline = time.monotonic() + 30
            while True:
                try:
                    with httpx.Client(follow_redirects=True, timeout=30) as client:
                        client.get(url)
                    return
                except httpx.ConnectError:
                    if time.monotonic() > deadline:
                        raise
                    time.sleep(0.5)

        threading.Thread(target=follow, daemon=True).start()
        return True

    webbrowser.open = fake_open  # type: ignore[assignment]
    try:
        asyncio.run(login("vendor", McpServerConfig(url=server_url)))
    finally:
        webbrowser.open = real_open  # type: ignore[assignment]


DRIVER_TS = """\
import { whoAmI } from "./extracted/who_am_i";

whoAmI({ name: process.argv[2] ?? "rote" }).then((result) => {
    console.log(JSON.stringify(result));
});
"""


def _pipeline(server_url: str) -> Pipeline:
    return Pipeline.model_validate(
        {
            "name": "ts_mcp_e2e",
            "input": {"type": "In", "required": ["name"]},
            "nodes": [
                {
                    "id": "who_am_i",
                    "kind": "external_call",
                    "description": "authenticated identity call over MCP",
                    "inputs": {"name": "pipeline.input.name"},
                    "mcp": {
                        "server": "vendor",
                        "tool": "whoami",
                        "args": {"name": "name"},
                        "url": server_url,
                    },
                }
            ],
            "edges": [],
        }
    )


@pytest.mark.skipif(not _node_available(), reason="requires node + npm")
def test_ts_runtime_reads_refreshes_and_rotates_python_tokens(
    protected_server: str, tmp_path: Path
) -> None:
    # 1. Python seeds the store via the real OAuth dance.
    _headless_login(protected_server)
    seeded = read_token_file("vendor")
    assert seeded is not None and seeded["token_endpoint"]
    first_access = seeded["tokens"]["access_token"]

    # 2. Emit the DBOS-TS app, add a driver, install + compile for real.
    app_dir = tmp_path / "app"
    get_adapter("dbos-ts").emit(_pipeline(protected_server), app_dir)
    (app_dir / "src" / "driver.ts").write_text(DRIVER_TS, encoding="utf-8")
    install = subprocess.run(
        ["npm", "install", "--no-audit", "--no-fund"],
        cwd=app_dir,
        capture_output=True,
        text=True,
        timeout=600,
        env={**os.environ, "npm_config_progress": "false"},
    )
    assert install.returncode == 0, f"npm install failed:\n{install.stdout}\n{install.stderr}"
    build = subprocess.run(["npx", "tsc"], cwd=app_dir, capture_output=True, text=True, timeout=300)
    assert build.returncode == 0, f"tsc failed:\n{build.stdout}\n{build.stderr}"

    env = {
        **os.environ,
        "ROTE_MCP_CONFIG": os.environ["ROTE_MCP_CONFIG"],
        "ROTE_MCP_TOKEN_DIR": os.environ["ROTE_MCP_TOKEN_DIR"],
    }

    def run_driver(arg: str) -> dict[str, object]:
        proc = subprocess.run(
            ["node", "dist/driver.js", arg],
            cwd=app_dir,
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )
        assert proc.returncode == 0, f"driver failed:\n{proc.stdout}\n{proc.stderr}"
        result: dict[str, object] = json.loads(proc.stdout.strip().splitlines()[-1])
        return result

    # 3. The emitted TS module authenticates with Python's token.
    assert run_driver("first") == {"hello": "first", "authenticated": True}

    # 4. Force the access token stale on disk; the TS side must refresh
    #    via the stored token_endpoint and still succeed…
    doc = read_token_file("vendor")
    assert doc is not None
    doc["expires_at"] = time.time() - 10
    write_token_file("vendor", doc)
    assert run_driver("second") == {"hello": "second", "authenticated": True}

    # …and the rotated credentials must be durable and Python-readable.
    after = read_token_file("vendor")
    assert after is not None
    assert after["tokens"]["access_token"] != first_access
    assert after["tokens"]["refresh_token"]
    assert after["expires_at"] and after["expires_at"] > time.time()
    assert after["version"] == 1
