"""Cross-runtime contract: a node with no ``retry:`` is attempted once (#271).

Before, an absent ``retry:`` meant one attempt on dbos, python and temporal,
six on cloudflare (``step.do``'s default of 5 retries) and four on inngest
(the SDK's function-level default of 3). For a paid judge or a call that is
not idempotent, one IR then did different things on different runtimes.

Each runtime runs the emitted app for real, and the node's code always
fails. Here it appends to a call log, so the count does not depend on what
a runtime's error says (DBOS's names only its last few attempts, Temporal's
only "Activity task failed"). The TS runtimes, where workerd has no file
system, count from the error instead (``attempt-N``). A declared ``retry: {max: 2}`` is the
positive half: three attempts, not one, so the fix cannot pass by turning
retries off everywhere. The TS runtimes run the same pipelines in
``test_retry_parity_ts_e2e.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from rote.adapters import get_adapter
from rote.eval.empirical import run_pipeline_trial
from rote.ir import Pipeline


def pipeline(retry: dict[str, Any] | None) -> Pipeline:
    node: dict[str, Any] = {
        "id": "always_fails",
        "kind": "external_call",
        "description": "Fail, naming the attempt.",
        "impl": "extracted/flaky.py:always_fails",
    }
    if retry is not None:
        node["retry"] = retry
    return Pipeline.model_validate(
        {
            "name": "retry-parity",
            "version": "0.1.0",
            "input": {"type": "In", "required": [], "optional": []},
            "nodes": [node],
            "edges": [],
            "entry_nodes": ["always_fails"],
            "exit_nodes": ["always_fails"],
        }
    )


#: (retry declared in the IR, the attempt the run's error must name).
CASES = [
    pytest.param(None, 1, id="no-retry"),
    pytest.param({"max": 2, "backoff": "constant"}, 3, id="retry-2"),
]


def _emit(runtime: str, retry: dict[str, Any] | None, tmp_path: Path) -> tuple[Path, Path]:
    """The emitted app, and the log its node appends one byte to per call."""
    log = tmp_path / "calls.log"
    source = tmp_path / "compiled"
    (source / "extracted").mkdir(parents=True)
    (source / "extracted" / "flaky.py").write_text(
        "def always_fails(**_payload):\n"
        f"    with open({str(log)!r}, 'a') as f:\n"
        "        f.write('x')\n"
        "    raise RuntimeError('always fails')\n",
        encoding="utf-8",
    )
    out = tmp_path / runtime
    get_adapter(runtime, extracted_source_dir=source).emit(pipeline(retry), out)
    return out, log


def check(error: str | None, log: Path, attempts: int) -> None:
    assert error is not None, "the node always fails, so the run must too"
    assert log.read_text() == "x" * attempts


@pytest.mark.parametrize(("retry", "attempts"), CASES)
def test_python_runtime(retry: dict[str, Any] | None, attempts: int, tmp_path: Path) -> None:
    app, log = _emit("python", retry, tmp_path)
    run = run_pipeline_trial(app, {}, timeout_seconds=60)
    check(run.error, log, attempts)


@pytest.mark.parametrize(("retry", "attempts"), CASES)
def test_temporal_runtime(retry: dict[str, Any] | None, attempts: int, tmp_path: Path) -> None:
    from rote.runners.temporal import run_temporal

    app, log = _emit("temporal", retry, tmp_path)
    try:
        run = run_temporal(app, {}, signals={}, timeout_seconds=90)
    finally:
        sys.path.remove(str(app))
        for name in [
            m
            for m in sys.modules
            if m.split(".")[0] in ("workflow", "activities", "extracted", "signatures")
        ]:
            del sys.modules[name]
    check(run.error, log, attempts)
    # The runner reports the node's own exception, not only Temporal's wrapper.
    assert run.error is not None and "always fails" in run.error, run.error


@pytest.mark.slow
@pytest.mark.parametrize(("retry", "attempts"), CASES)
def test_dbos_runtime(retry: dict[str, Any] | None, attempts: int, tmp_path: Path) -> None:
    """Slow: launches a real DBOS runtime."""
    app, log = _emit("dbos", retry, tmp_path)
    run = run_pipeline_trial(app, {}, timeout_seconds=120)
    check(run.error, log, attempts)
