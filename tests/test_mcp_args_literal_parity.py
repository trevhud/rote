"""``mcp.args`` literals reach the tool byte-identical, on every runtime.

A real BDR compile wrote fixed tool arguments ('management-levels',
'BDR do not contact') into ``mcp.args``, which accepted only references.
``{literal: ...}`` now carries a string, number or boolean. The value
lands in emitted code, so this sends one built to break whatever it is
spliced into (``EVIL``: quotes, a backtick, ``${}``, newlines, a
character outside the BMP) through a real MCP server that echoes its
arguments back, next to a reference and the other scalar types. Every
runtime must send the identical arguments, and the emitted code must
still compile and run.

Four runtimes speak MCP: dbos here, and cloudflare, inngest and dbos-ts in
``test_mcp_args_literal_parity_ts_e2e.py`` and its DBOS-TS sibling. The
python and temporal adapters have no MCP backend and refuse an MCP-bound
node without an ``impl`` at emit time (asserted below), so ``mcp.args``
never reaches them.

Slow: a real fastmcp server and a DBOS SQLite app.
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

from rote.adapters import get_adapter
from rote.eval.empirical import run_pipeline_trial
from rote.ir import Pipeline
from tests.test_constants_reach_every_caller import EVIL

pytestmark = pytest.mark.slow

pytest.importorskip("fastmcp")

SERVER = "echo"

PIPELINE = Pipeline.model_validate(
    {
        "name": "mcp-literals",
        "version": "0.1.0",
        "input": {"type": "In", "required": ["query"], "optional": []},
        "nodes": [
            {
                "id": "call",
                "kind": "external_call",
                "description": "Record the arguments the tool receives.",
                "mcp": {
                    "server": SERVER,
                    "tool": "record",
                    "args": {
                        "q": "query",
                        "level": {"literal": "management-levels"},
                        "n": {"literal": 3},
                        "f": {"literal": 2.5},
                        "b": {"literal": True},
                        "evil": {"literal": EVIL},
                    },
                },
                "inputs": {"query": "pipeline.input.query"},
            }
        ],
        "edges": [],
        "entry_nodes": ["call"],
        "exit_nodes": ["call"],
    }
)

PAYLOAD = {"query": "invoices"}

#: What the tool must receive, on every runtime.
EXPECTED = {
    "call": {
        "q": "invoices",
        "level": "management-levels",
        "n": 3,
        "f": 2.5,
        "b": True,
        "evil": EVIL,
    }
}

_ECHO_SERVER = """\
import sys

from fastmcp import FastMCP

mcp = FastMCP("echo")


@mcp.tool
def record(q: str, level: str, n: int, f: float, b: bool, evil: str) -> dict:
    \"\"\"Echo the arguments back.\"\"\"
    return {"q": q, "level": level, "n": n, "f": f, "b": b, "evil": evil}


mcp.run(transport="http", host="127.0.0.1", port=int(sys.argv[1]))
"""


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="module")
def echo_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    # Not "mcp.py": a script beside the import path would shadow the package.
    script = tmp_path_factory.mktemp("echo") / "echo_server.py"
    script.write_text(_ECHO_SERVER, encoding="utf-8")
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, str(script), str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.time() + 30
        while time.time() < deadline:
            with socket.socket() as s:
                if s.connect_ex(("127.0.0.1", port)) == 0:
                    break
            time.sleep(0.2)
        else:
            raise RuntimeError("echo server did not start")
        yield f"http://127.0.0.1:{port}/mcp/"
    finally:
        proc.terminate()
        proc.wait(timeout=10)


class _TokenStub(BaseHTTPRequestHandler):
    """Answers a refresh grant: the Workers MCP helper mints a token before
    it will call even an unauthenticated server."""

    def do_POST(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler's API
        self.rfile.read(int(self.headers.get("content-length", 0)))
        raw = json.dumps(
            {"access_token": "local", "token_type": "Bearer", "expires_in": 3600}
        ).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args: object) -> None:
        return


@pytest.fixture(scope="module")
def token_endpoint() -> Iterator[str]:
    server = HTTPServer(("127.0.0.1", 0), _TokenStub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/token"
    finally:
        server.shutdown()
        server.server_close()


def mcp_env(echo_url: str, token_url: str, tmp_path: Path) -> dict[str, str]:
    upper = SERVER.upper()
    return {
        f"ROTE_MCP_{upper}_URL": echo_url,
        "ROTE_MCP_TOKEN_DIR": str(tmp_path / "tokens"),
        f"ROTE_MCP_{upper}_REFRESH_TOKEN": "local",
        f"ROTE_MCP_{upper}_CLIENT_ID": "local",
        f"ROTE_MCP_{upper}_TOKEN_ENDPOINT": token_url,
    }


def test_dbos_sends_literals_byte_identical(
    echo_server: str,
    token_endpoint: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key, value in mcp_env(echo_server, token_endpoint, tmp_path).items():
        monkeypatch.setenv(key, value)
    app = tmp_path / "dbos"
    get_adapter("dbos").emit(PIPELINE, app)
    run = run_pipeline_trial(app, PAYLOAD, timeout_seconds=120)
    assert run.error is None, run.error
    assert run.output == EXPECTED


@pytest.mark.parametrize("runtime", ["python", "temporal"])
def test_runtimes_without_mcp_never_emit_mcp_args(runtime: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no MCP backend"):
        get_adapter(runtime).emit(PIPELINE, tmp_path / runtime)


def as_dev_vars(env: dict[str, Any]) -> str:
    """``.dev.vars`` for wrangler: URLs and tokens only, no quoting traps."""
    return "".join(f'{key}="{value}"\n' for key, value in env.items())
