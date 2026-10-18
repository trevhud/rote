"""Live: a TS run missing a required input fails once, naming the field.

The emission assertions in test_pipeline_input_parity.py prove the check
is emitted. These prove what it does on the real runtimes: the run fails
with the named error before any step runs, and it does not retry. Inngest
retries are function-level, so a plain throw would re-run the same bad
input up to ``retries`` times with backoff. The node here declares a retry
budget so that retrying is possible, and the run must still end at once.

Slow: each runtime installs its toolchain and starts its dev server
(dbos-ts also needs Docker Postgres).
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from rote.adapters import get_adapter
from rote.ir import Pipeline

pytestmark = pytest.mark.slow

_PIPELINE = Pipeline.model_validate(
    {
        "name": "needs-region",
        "version": "0.1.0",
        "input": {"type": "In", "required": ["region"], "optional": ["report_date"]},
        "nodes": [
            {
                "id": "build_report",
                "kind": "pure_function",
                "description": "Build the report.",
                "impl": "extracted/report.py:build_report",
                "retry": {"max": 3, "backoff": "exponential"},
                "inputs": {
                    "region": "pipeline.input.region",
                    "report_date": "pipeline.input.report_date",
                },
            }
        ],
        "edges": [],
        "entry_nodes": ["build_report"],
        "exit_nodes": ["build_report"],
    }
)

_MESSAGE = "pipeline input is missing required field(s): region"


def _run(runtime: str, tmp_path: Path, payload: dict) -> tuple[object, float]:
    out = tmp_path / runtime
    get_adapter(runtime).emit(_PIPELINE, out)
    started = time.monotonic()
    if runtime == "inngest":
        from rote.runners.inngest import run_inngest

        run = run_inngest(
            out, payload, pipeline=_PIPELINE, signals={}, gate_order=[], timeout_seconds=120
        )
    elif runtime == "cloudflare":
        from rote.runners.cloudflare import run_cloudflare

        run = run_cloudflare(out, payload, signals={}, gate_order=[], timeout_seconds=120)
    else:
        from rote.runners.dbos_ts import run_dbos_ts

        run = run_dbos_ts(out, payload, signals={}, timeout_seconds=120)
    return run, time.monotonic() - started


@pytest.mark.parametrize("runtime", ["inngest", "cloudflare", "dbos-ts"])
def test_a_missing_required_input_fails_once_naming_it(runtime: str, tmp_path: Path) -> None:
    run, _wall = _run(runtime, tmp_path, {"report_date": "2026-10-18"})

    assert run.output is None  # type: ignore[attr-defined]
    error = run.error or ""  # type: ignore[attr-defined]
    assert _MESSAGE in error, error
    assert "timed out" not in error


def test_inngest_does_not_retry_a_bad_input(tmp_path: Path) -> None:
    """With ``retries: 3`` a plain throw backs off and re-runs; the run
    would still be going at the 120s timeout. NonRetriableError ends it."""
    run, _wall = _run("inngest", tmp_path, {})
    error = run.error or ""  # type: ignore[attr-defined]
    assert f"run failed: NonRetriableError: {_MESSAGE}" in error, error
    # Terminal well inside the timeout. A retried run is still Running at
    # 120s (checked by swapping NonRetriableError for Error), so the
    # runner would report a timeout instead of the failure above.
    assert run.wall_seconds < 60, run.wall_seconds  # type: ignore[attr-defined]
