"""The TS runtimes send ``mcp.args`` literals byte-identical too.

Same fixture as ``test_mcp_args_literal_parity.py``: a real MCP server
echoes its arguments, one of them a literal built to break emitted code.
Cloudflare and Inngest run here; DBOS-TS (Docker Postgres) runs in its
own file.

Slow: npm install and the real dev servers (wrangler, inngest-cli).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from rote.adapters import get_adapter
from tests.test_mcp_args_literal_parity import (
    EXPECTED,
    PAYLOAD,
    PIPELINE,
    as_dev_vars,
    echo_server,  # noqa: F401 — fixture
    mcp_env,
    token_endpoint,  # noqa: F401 — fixture
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("npm") is None, reason="requires node + npm"),
]


def run_ts(runtime: str, env: dict[str, str], tmp_path: Path) -> Any:
    app = tmp_path / runtime
    get_adapter(runtime).emit(PIPELINE, app)
    if runtime == "cloudflare":
        from rote.runners.cloudflare import run_cloudflare

        # workerd reads its bindings from .dev.vars, not the process env.
        (app / ".dev.vars").write_text(as_dev_vars(env), encoding="utf-8")
        return run_cloudflare(app, PAYLOAD, signals={}, gate_order=[], timeout_seconds=180)
    if runtime == "inngest":
        from rote.runners.inngest import run_inngest

        return run_inngest(
            app, PAYLOAD, pipeline=PIPELINE, signals={}, gate_order=[], timeout_seconds=180
        )
    from rote.runners.dbos_ts import run_dbos_ts

    return run_dbos_ts(app, PAYLOAD, signals={}, timeout_seconds=180)


def check_ts_runtime(
    runtime: str, echo: str, token: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = mcp_env(echo, token, tmp_path)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    run = run_ts(runtime, env, tmp_path)
    assert run.error is None, run.error
    assert run.output == EXPECTED


#: dbos-ts needs Docker Postgres, so its case lives in
#: test_mcp_args_literal_parity_dbos_ts_e2e.py (the DBOS-TS CI job).
@pytest.mark.parametrize("runtime", ["cloudflare", "inngest"])
def test_ts_runtimes_send_literals_byte_identical(
    runtime: str,
    echo_server: str,  # noqa: F811
    token_endpoint: str,  # noqa: F811
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    check_ts_runtime(runtime, echo_server, token_endpoint, tmp_path, monkeypatch)
