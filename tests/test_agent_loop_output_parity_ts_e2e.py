"""The TS runtimes produce an agent loop's declared output keys too.

Same fixture as ``test_agent_loop_output_parity.py`` (the Python
runtimes): a downstream node reads two declared keys of an agent loop,
the loop runs against a stub Messages endpoint and a real fastmcp server,
and the pipeline's output is what that node made of them. Also typechecks
invoice-push on the three TS runtimes: before the fix, its downstream
read of ``process_invoices_loop_result["invoice_results"]`` failed TS7053
on DBOS-TS and Inngest, and Cloudflare compiled only because it cast the
result to ``Record<string, unknown>``.

Slow: npm install, tsc and the real dev servers (wrangler, inngest-cli,
DBOS-TS on Docker Postgres, the last in its own file).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from rote.adapters import get_adapter
from rote.ir import load_pipeline
from tests.test_agent_loop_output_parity import (
    EXPECTED,
    PIPELINE,
    assert_the_model_was_offered_the_declared_output,
    loop_env,
    mcp_server,  # noqa: F401 — fixture
    messages_stub,  # noqa: F401 — fixture
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("npm") is None, reason="requires node + npm"),
]

REPORT_TS = """\
export async function report(input: Record<string, unknown>): Promise<{ line: string }> {
    return { line: `${String(input.summary)} / ${Number(input.score)}` };
}
"""

INVOICE_PUSH = (
    Path(__file__).resolve().parent.parent
    / "examples"
    / "invoice-push"
    / "expected"
    / "pipeline.yaml"
)


def run_ts(runtime: str, env: dict[str, str], tmp_path: Path) -> Any:
    app = tmp_path / runtime
    get_adapter(runtime).emit(PIPELINE, app)
    (app / "src" / "extracted" / "report.ts").write_text(REPORT_TS, encoding="utf-8")
    payload = {"account": "acme"}
    if runtime == "cloudflare":
        from rote.runners.cloudflare import run_cloudflare

        # workerd reads its bindings from .dev.vars, not the process env.
        lines = [f'{key}="{value}"' for key, value in env.items()]
        (app / ".dev.vars").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return run_cloudflare(app, payload, signals={}, gate_order=[], timeout_seconds=180)
    if runtime == "inngest":
        from rote.runners.inngest import run_inngest

        return run_inngest(
            app, payload, pipeline=PIPELINE, signals={}, gate_order=[], timeout_seconds=180
        )
    from rote.runners.dbos_ts import run_dbos_ts

    return run_dbos_ts(app, payload, signals={}, timeout_seconds=180)


def check_ts_runtime(
    runtime: str,
    stub: str,
    mcp: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = loop_env(stub, mcp, tmp_path)
    # The Node runtimes inherit the process env; Cloudflare gets .dev.vars.
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    run = run_ts(runtime, env, tmp_path)
    assert run.error is None, run.error
    assert run.output == EXPECTED
    assert_the_model_was_offered_the_declared_output()


#: dbos-ts needs Docker Postgres, so its case lives in
#: test_agent_loop_output_parity_dbos_ts_e2e.py (the DBOS-TS CI job).
@pytest.mark.parametrize("runtime", ["cloudflare", "inngest"])
def test_downstream_reads_the_loops_declared_keys(
    runtime: str,
    messages_stub: str,  # noqa: F811
    mcp_server: str,  # noqa: F811
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    check_ts_runtime(runtime, messages_stub, mcp_server, tmp_path, monkeypatch)


@pytest.mark.parametrize("runtime", ["cloudflare", "dbos-ts", "inngest"])
def test_invoice_push_typechecks(runtime: str, tmp_path: Path) -> None:
    app = tmp_path / runtime
    get_adapter(runtime).emit(load_pipeline(INVOICE_PUSH), app)
    install = subprocess.run(
        ["npm", "install", "--no-audit", "--no-fund"],
        cwd=app,
        capture_output=True,
        text=True,
        timeout=600,
        env={**os.environ, "npm_config_progress": "false"},
    )
    assert install.returncode == 0, install.stderr
    tsc = subprocess.run(
        ["npx", "--no-install", "tsc", "--noEmit"],
        cwd=app,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert tsc.returncode == 0, tsc.stdout + tsc.stderr
