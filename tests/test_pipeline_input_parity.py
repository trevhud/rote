"""Cross-runtime contract: optional pipeline inputs may be omitted, required
ones fail naming themselves.

ops-report declares ``report_date`` optional, yet every Python runtime
emitted ``pipeline_input["report_date"]``, so a scheduled run that omitted
it died with a bare ``KeyError`` inside generated code. The TS runtimes read
it as undefined, which is right for an optional field and wrong for a
required one: a missing required field slid through as undefined and
surfaced later as an unrelated error.

Each Python runtime runs the emitted app for real. The TS runtimes are held
to the same check by emission assertions here and by the live TS e2e
suites, which run the BDR app with its guard in place.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from rote.adapters import get_adapter
from rote.eval.empirical import run_pipeline_trial
from rote.ir import Pipeline

_PIPELINE = Pipeline.model_validate(
    {
        "name": "dated-report",
        "version": "0.1.0",
        "input": {
            "type": "ReportInput",
            "required": ["region"],
            "optional": ["report_date"],
            "input_schema": {
                "type": "object",
                "properties": {
                    "region": {"type": "string"},
                    "report_date": {"type": "string"},
                    "notify": {"type": "string"},
                },
                "required": ["region"],
            },
        },
        "nodes": [
            {
                "id": "build_report",
                "kind": "pure_function",
                "description": "Build the report.",
                "impl": "extracted/report.py:build_report",
                "inputs": {
                    "region": "pipeline.input.region",
                    "report_date": "pipeline.input.report_date",
                    # Optional only through the schema: a property that the
                    # schema's `required` leaves out.
                    "notify": "pipeline.input.notify",
                },
            }
        ],
        "edges": [],
        "entry_nodes": ["build_report"],
        "exit_nodes": ["build_report"],
    }
)

# Defaults on the optional parameters: an omitted field must be absent from
# the payload, so these apply. An explicit None would override them.
_REPORT = (
    "def build_report(region, report_date='today', notify='nobody'):\n"
    "    return {'region': region, 'report_date': report_date, 'notify': notify}\n"
)

_FULL = {"region": "west"}
_EXPECTED = {"build_report": {"region": "west", "report_date": "today", "notify": "nobody"}}


def test_required_and_optional_fields_derive_from_the_declarations() -> None:
    assert _PIPELINE.optional_input_fields == frozenset({"report_date", "notify"})
    assert _PIPELINE.required_input_fields == ("region",)


def test_with_no_declared_contract_every_read_field_is_required() -> None:
    """The IR skips the declared-field check when the contract is empty, so
    a node may read a field nothing declares. Nothing says it is optional."""
    data = _PIPELINE.model_dump(mode="json", by_alias=True, exclude_none=True)
    data["input"] = {"type": "In", "required": [], "optional": []}
    assert Pipeline.model_validate(data).required_input_fields == (
        "notify",
        "region",
        "report_date",
    )


def test_a_field_cannot_be_both_required_and_optional() -> None:
    data = _PIPELINE.model_dump(mode="json", by_alias=True, exclude_none=True)
    data["input"]["optional"].append("region")
    with pytest.raises(ValueError, match="region are both required and optional"):
        Pipeline.model_validate(data)


def _emit(runtime: str, tmp_path: Path) -> Path:
    source = tmp_path / "compiled"
    (source / "extracted").mkdir(parents=True)
    (source / "extracted" / "report.py").write_text(_REPORT, encoding="utf-8")
    out = tmp_path / runtime
    get_adapter(runtime, extracted_source_dir=source).emit(_PIPELINE, out)
    return out


def _forget_temporal_app(app: Path) -> None:
    sys.path.remove(str(app))
    for name in [
        m
        for m in sys.modules
        if m.split(".")[0] in ("workflow", "activities", "extracted", "signatures")
    ]:
        del sys.modules[name]


def test_python_runtime_omits_optional_and_names_missing_required(tmp_path: Path) -> None:
    app = _emit("python", tmp_path)

    run = run_pipeline_trial(app, _FULL, timeout_seconds=60)
    assert run.error is None, run.error
    assert run.output == _EXPECTED

    missing = run_pipeline_trial(app, {"report_date": "2026-10-18"}, timeout_seconds=60)
    assert missing.output is None
    assert "pipeline input is missing required field(s): region" in (missing.error or "")

    # A run started with no object at all fails with the same named check,
    # not a TypeError inside generated code.
    no_object = run_pipeline_trial(app, None, timeout_seconds=60)  # type: ignore[arg-type]
    assert "pipeline input must be a JSON object, got NoneType" in (no_object.error or "")


def test_temporal_runtime_omits_optional_and_names_missing_required(tmp_path: Path) -> None:
    from rote.runners.temporal import run_temporal

    app = _emit("temporal", tmp_path)
    try:
        run = run_temporal(app, _FULL, signals={}, timeout_seconds=60)
    finally:
        _forget_temporal_app(app)
    assert run.error is None, run.error
    assert run.output == _EXPECTED

    try:
        # Non-retryable, so this fails at once instead of hitting the timeout.
        missing = run_temporal(app, {"notify": "ops"}, signals={}, timeout_seconds=60)
    finally:
        _forget_temporal_app(app)
    assert missing.output is None
    assert "pipeline input is missing required field(s): region" in (missing.error or "")


@pytest.mark.slow
def test_dbos_runtime_omits_optional_and_names_missing_required(tmp_path: Path) -> None:
    """Slow: launches a real DBOS runtime (system-DB migrations, queues)."""
    app = _emit("dbos", tmp_path)

    run = run_pipeline_trial(app, _FULL, timeout_seconds=120)
    assert run.error is None, run.error
    assert run.output == _EXPECTED

    # Run the app directly: the trial runner keeps only the head of a
    # failed run's output, which DBOS's startup log fills.
    missing = subprocess.run(
        [sys.executable, str(app / "main.py"), json.dumps({"notify": "ops"})],
        cwd=app,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert missing.returncode != 0
    assert "pipeline input is missing required field(s): region" in missing.stderr


@pytest.mark.parametrize("runtime", ["cloudflare", "dbos-ts", "inngest"])
def test_ts_runtimes_check_required_inputs_on_entry(runtime: str, tmp_path: Path) -> None:
    out = tmp_path / runtime
    written = get_adapter(runtime).emit(_PIPELINE, out)
    entry = next(
        path.read_text(encoding="utf-8")
        for path in written.values()
        if path.name in ("workflow.ts", "main.ts", "pipeline.ts")
    )

    assert "function missingInputsError(" in entry
    assert 'missingInputsError(pipelineInput, ["region"])' in entry
    # An optional field is read as-is, which is undefined when omitted.
    assert 'pipelineInput["report_date"]' in entry
