"""End-to-end: a missing MCP endpoint fails a DBOS run after ONE attempt.

No endpoint for a server (no ``ROTE_MCP_<SERVER>_URL``, nothing in the
registry, none in the IR) is configuration. Nothing changes between
retry attempts, so the step must run once and the run must name the
variable. Before the fix a bound step spent its whole budget (``max: 3``,
about 15 seconds of backoff) and the run reported only
``DBOSMaxStepRetriesExceeded``. An agent loop pinned to the server is the
same gap reached through tool discovery, and BDR's MCP use is entirely
agent loops.

Runs the emitted app as a subprocess against a real DBOS runtime on
SQLite, exactly how users run it. Attempts are counted, not inferred
from a message: the emitted MCP helper is wrapped so every endpoint
resolution for the server appends a line, and each attempt of either
node resolves it exactly once. The TypeScript runtimes have the same
cases in their park e2e suites.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from rote.adapters.dbos import DbosAdapter
from rote.adapters.python import PythonAdapter
from tests._helpers import unconfigured_mcp_pipelines

pytest.importorskip("dbos", reason="dbos not installed (pip install 'rote-cli[dbos]')")
pytest.importorskip("fastmcp")

pytestmark = pytest.mark.slow

_COUNT_RESOLUTIONS = """

# ── test overlay: count endpoint resolutions (one per step attempt) ──
_ROTE_E2E_RESOLVE_URL = resolve_url


def resolve_url(server, pipeline_url):  # noqa: F811
    with open({path!r}, "a", encoding="utf-8") as f:
        f.write(server + "\\n")
    return _ROTE_E2E_RESOLVE_URL(server, pipeline_url)
"""


@pytest.mark.parametrize(
    ("runtime", "node_id"),
    [("dbos", "lookup"), ("dbos", "research"), ("python", "research")],
)
def test_missing_endpoint_runs_the_step_once_and_names_the_variable(
    runtime: str, node_id: str, tmp_path: Path
) -> None:
    """The python runtime has no bound MCP steps (and no retry loop around
    agent loops), but its loops reach MCP through the same helper."""
    out = tmp_path / "app"
    pipeline = unconfigured_mcp_pipelines("config")[node_id]
    adapter = DbosAdapter() if runtime == "dbos" else PythonAdapter()
    adapter.emit(pipeline, out)
    resolutions = tmp_path / "resolutions.log"
    helper = out / "extracted" / "_rote_mcp.py"
    helper.write_text(
        helper.read_text(encoding="utf-8") + _COUNT_RESOLUTIONS.format(path=str(resolutions)),
        encoding="utf-8",
    )

    env = {k: v for k, v in os.environ.items() if k not in ("ROTE_MCP_VENDOR_URL", "PYTHONPATH")}
    # An empty registry, so no machine-local entry can supply the endpoint.
    registry = tmp_path / "mcp.json"
    registry.write_text(json.dumps({"servers": {}}), encoding="utf-8")
    env["ROTE_MCP_CONFIG"] = str(registry)
    env["DBOS_SYSTEM_DATABASE_URL"] = f"sqlite:///{tmp_path / 'e2e.sqlite'}"
    # The agent loop takes the SDK lane at a dead endpoint with a fake key:
    # it must fail at tool discovery, before any request, and if it ever
    # got further it could not reach (or bill) anything.
    env["ROTE_INFERENCE"] = "api"
    env["ANTHROPIC_API_KEY"] = "test-key-not-real"
    env["ROTE_BASE_URL_RESEARCH"] = "http://127.0.0.1:9"

    proc = subprocess.run(
        [sys.executable, str(out / "main.py"), json.dumps({"contact_id": "c1"})],
        cwd=out,
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    assert proc.returncode != 0
    assert resolutions.read_text(encoding="utf-8").splitlines() == ["vendor"], proc.stderr
    assert "RoteMcpConfigError" in proc.stderr
    assert "set ROTE_MCP_VENDOR_URL" in proc.stderr
