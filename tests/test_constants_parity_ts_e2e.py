"""The TS runtimes pass each node's constants to its impl, like Python.

``tests/test_constants_parity.py`` runs python, temporal and dbos for real
on one fixture. This runs the same fixture on cloudflare, inngest and
dbos-ts, with real TS impls that echo what they were called with, so all
six runtimes are held to one answer. Before, the TS runtimes printed the
constants in the stub's doc comment and passed none, so a TS impl had to
hard-code them and an edit to pipeline.yaml did nothing.

TS emits one module per node, so the two fetch nodes get one file each;
each file still has to receive its own node's values.

Slow: npm install and the real dev servers (wrangler, inngest-cli,
DBOS-TS on Docker Postgres).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from rote.adapters import get_adapter
from tests.test_constants_parity import _EXPECTED, _PIPELINE

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("npm") is None, reason="requires node + npm"),
]

_ECHO = """\
export async function {fn}(
    input: Record<string, unknown>,
): Promise<{{ called_with: Record<string, string> }}> {{
    return {{ called_with: input as Record<string, string> }};
}}
"""

_IMPLS = {
    "fetch_containers": _ECHO.format(fn="fetchContainers"),
    "fetch_dwell": _ECHO.format(fn="fetchDwell"),
    "list_docks": (
        "export async function listDocks(_input: unknown): Promise<{ docks: string[] }> {\n"
        '    return { docks: ["d1", "d2", "d3"] };\n'
        "}\n"
    ),
    "label_dock": (
        "export async function labelDock(\n"
        "    input: Record<string, unknown>,\n"
        "): Promise<{ label: string }> {\n"
        "    return { label: `${String(input.prefix)}-${String(input.dock)}` };\n"
        "}\n"
    ),
}


def _run(runtime: str, tmp_path: Path) -> Any:
    app = tmp_path / runtime
    get_adapter(runtime).emit(_PIPELINE, app)
    for node_id, source in _IMPLS.items():
        (app / "src" / "extracted" / f"{node_id}.ts").write_text(source, encoding="utf-8")
    payload = {"region": "west"}
    if runtime == "cloudflare":
        from rote.runners.cloudflare import run_cloudflare

        return run_cloudflare(app, payload, signals={}, gate_order=[], timeout_seconds=180)
    if runtime == "inngest":
        from rote.runners.inngest import run_inngest

        return run_inngest(
            app, payload, pipeline=_PIPELINE, signals={}, gate_order=[], timeout_seconds=180
        )
    from rote.runners.dbos_ts import run_dbos_ts

    return run_dbos_ts(app, payload, signals={}, timeout_seconds=180)


def check_runtime(runtime: str, tmp_path: Path) -> None:
    run = _run(runtime, tmp_path)
    assert run.error is None, run.error
    assert run.output == _EXPECTED


#: dbos-ts needs Docker Postgres, so its case lives in
#: test_constants_parity_dbos_ts_e2e.py, which runs in the DBOS-TS CI job.
@pytest.mark.parametrize("runtime", ["cloudflare", "inngest"])
def test_ts_runtime_passes_each_nodes_own_constants(runtime: str, tmp_path: Path) -> None:
    check_runtime(runtime, tmp_path)
