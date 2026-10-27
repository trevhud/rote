"""The TS runtimes show a gate's rendered message under ``rote run``, like Python.

``tests/test_gate_message_parity.py`` runs temporal and dbos for real on one
fixture. This runs the same fixture on cloudflare and inngest (dbos-ts is in
test_gate_message_parity_dbos_ts_e2e.py, which needs Docker), with TS impls
that return the same values, so every gate-capable runtime is held to one
rendered text. Both runtimes replay workflow code, so the marker line prints
more than once; the announcement must still appear exactly once.

Slow: npm install and the real dev servers (wrangler, inngest-cli,
DBOS-TS on Docker Postgres).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from rote.adapters import get_adapter
from rote.mcp.parked import watch_factory
from tests.test_gate_message_parity import (
    _PIPELINE,
    EXPECTED_OUTPUT,
    INPUT,
    SIGNALS,
    check_announcement,
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("npm") is None, reason="requires node + npm"),
]

_IMPLS = {
    "summarize": (
        "export async function summarize(_input: unknown): Promise<Record<string, unknown>> {\n"
        "    return {\n"
        "        count: 3,\n"
        "        ratio: 2.0,\n"
        '        tags: ["a", "b"],\n'
        '        meta: { b: 1, "2": true },\n'
        "        nothing: null,\n"
        "        big: 1e21,\n"
        "        tiny: 1e-7,\n"
        '        who: "Ana \\u202e reversed",\n'
        "    };\n"
        "}\n"
    ),
    "publish": (
        "export async function publish(\n"
        "    input: Record<string, unknown>,\n"
        "): Promise<{ decision: unknown }> {\n"
        "    return { decision: input.decision };\n"
        "}\n"
    ),
}


def _run(runtime: str, tmp_path: Path) -> Any:
    app = tmp_path / runtime
    get_adapter(runtime).emit(_PIPELINE, app)
    for node_id, source in _IMPLS.items():
        (app / "src" / "extracted" / f"{node_id}.ts").write_text(source, encoding="utf-8")
    watch = watch_factory(False)
    if runtime == "cloudflare":
        from rote.runners.cloudflare import run_cloudflare

        return run_cloudflare(
            app,
            INPUT,
            signals=SIGNALS,
            gate_order=["approval"],
            timeout_seconds=180,
            park_watch=watch,
        )
    if runtime == "inngest":
        from rote.runners.inngest import run_inngest

        return run_inngest(
            app,
            INPUT,
            pipeline=_PIPELINE,
            signals=SIGNALS,
            gate_order=["approval"],
            timeout_seconds=180,
            park_watch=watch,
        )
    from rote.runners.dbos_ts import run_dbos_ts

    return run_dbos_ts(app, INPUT, signals=SIGNALS, timeout_seconds=180, park_watch=watch)


def check_runtime(runtime: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run = _run(runtime, tmp_path)
    assert run.error is None, run.error
    assert run.output == EXPECTED_OUTPUT
    check_announcement(capsys.readouterr().err)


@pytest.mark.parametrize("runtime", ["cloudflare", "inngest"])
def test_ts_run_shows_the_gate_message(
    runtime: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    check_runtime(runtime, tmp_path, capsys)
