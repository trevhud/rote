"""The TS cases of the gate-shows parity test (see test_gate_shows_parity.py).

Cloudflare and Inngest here; DBOS-TS, which needs Docker Postgres, in
test_gate_shows_parity_dbos_ts_e2e.py.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from rote.adapters import get_adapter
from rote.mcp.parked import watch_factory
from tests.test_gate_shows_parity import INPUT, PIPELINE, SIGNALS, check

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("npm") is None, reason="requires node + npm"),
]

_IMPLS = {
    "outline": (
        "export async function outline(_input: unknown): Promise<{ sections: string[] }> {\n"
        '    return { sections: ["intro", "body"] };\n'
        "}\n"
    ),
    "publish": (
        "export async function publish(input: {\n"
        "    decision: { approved: boolean };\n"
        "}): Promise<{ published: boolean }> {\n"
        "    return { published: input.decision.approved };\n"
        "}\n"
    ),
}


def run_case(runtime: str, tmp_path: Path) -> Any:
    app = tmp_path / runtime
    get_adapter(runtime).emit(PIPELINE, app)
    for node_id, source in _IMPLS.items():
        (app / "src" / "extracted" / f"{node_id}.ts").write_text(source, encoding="utf-8")
    watch = watch_factory(False)
    if runtime == "cloudflare":
        from rote.runners.cloudflare import run_cloudflare

        return run_cloudflare(
            app,
            INPUT,
            signals=SIGNALS,
            gate_order=["draft_approved"],
            timeout_seconds=180,
            park_watch=watch,
        )
    if runtime == "inngest":
        from rote.runners.inngest import run_inngest

        return run_inngest(
            app,
            INPUT,
            pipeline=PIPELINE,
            signals=SIGNALS,
            gate_order=["draft_approved"],
            timeout_seconds=180,
            park_watch=watch,
        )
    from rote.runners.dbos_ts import run_dbos_ts

    return run_dbos_ts(app, INPUT, signals=SIGNALS, timeout_seconds=180, park_watch=watch)


@pytest.mark.parametrize("runtime", ["cloudflare", "inngest"])
def test_ts_shows_the_gates_data(
    runtime: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    check(run_case(runtime, tmp_path), capsys.readouterr().err)
