"""The TS runtimes attempt a node with no ``retry:`` once, like Python (#271).

``tests/test_retry_parity.py`` holds the contract and runs it on python,
temporal and dbos. This runs the same two pipelines on cloudflare and
inngest. workerd has no file system, so the node counts its calls in a
module variable and fails naming the count (``attempt-N``), and the run's
error says how many attempts ran. Before #271, the no-retry case reached
``attempt-6`` on cloudflare and ``attempt-4`` on inngest.

Slow: npm install and the real dev servers (wrangler, inngest-cli).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from rote.adapters import get_adapter
from tests.test_retry_parity import CASES, pipeline

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("npm") is None, reason="requires node + npm"),
]

_IMPL = """\
let calls = 0;

export async function alwaysFails(_input: unknown): Promise<never> {
    calls += 1;
    throw new Error(`attempt-${calls}`);
}
"""


def run(runtime: str, retry: dict[str, Any] | None, tmp_path: Path) -> Any:
    app = tmp_path / runtime
    ir = pipeline(retry)
    get_adapter(runtime).emit(ir, app)
    (app / "src" / "extracted" / "always_fails.ts").write_text(_IMPL, encoding="utf-8")
    if runtime == "cloudflare":
        from rote.runners.cloudflare import run_cloudflare

        return run_cloudflare(app, {}, signals={}, gate_order=[], timeout_seconds=240)
    if runtime == "inngest":
        from rote.runners.inngest import run_inngest

        return run_inngest(app, {}, pipeline=ir, signals={}, gate_order=[], timeout_seconds=300)
    from rote.runners.dbos_ts import run_dbos_ts

    return run_dbos_ts(app, {}, signals={}, timeout_seconds=240)


def check_runtime(
    runtime: str, retry: dict[str, Any] | None, attempts: int, tmp_path: Path
) -> None:
    error = run(runtime, retry, tmp_path).error
    assert error is not None, "the node always fails, so the run must too"
    assert f"attempt-{attempts}" in error, error
    assert f"attempt-{attempts + 1}" not in error, error


#: dbos-ts needs Docker Postgres, so its case lives in
#: test_retry_parity_dbos_ts_e2e.py, which runs in the DBOS-TS CI job.
@pytest.mark.parametrize("runtime", ["cloudflare", "inngest"])
@pytest.mark.parametrize(("retry", "attempts"), CASES)
def test_ts_runtime_attempts(
    runtime: str, retry: dict[str, Any] | None, attempts: int, tmp_path: Path
) -> None:
    check_runtime(runtime, retry, attempts, tmp_path)
