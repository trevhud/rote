"""The expense-audit example is the README's free first run. Keep it working.

It is the one example whose extracted/ modules are real code, so it is
the only proof that ``rote emit`` + ``rote run`` complete end to end
with no credentials and no tokens. These tests drive the exact CLI
commands the README prints and assert the dollar figures, not just that
a run finished: a run that "succeeds" with wrong totals is the failure
this example exists to rule out.

Expected figures, worked by hand from input.json and the 2026-09 rates
in references/fx-rates.md:

- batch total $3,304.15 across 16 lines and 5 employees
- review items EXP-2403 $130.39, EXP-2410 $249.00, EXP-2413 $18.75,
  EXP-2420 $446.25 (sum $844.39)
- block items EXP-2404 $25.97 (no receipt), EXP-2412 $38.20 (duplicate)
  (sum $64.17)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rote.cli import main
from rote.ir import NodeKind, load_pipeline

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "expense-audit"
GATED = EXAMPLE / "expected" / "pipeline.yaml"
DRAFT = EXAMPLE / "expected" / "pipeline-draft.yaml"
INPUT = EXAMPLE / "input.json"

#: The controller decision the README passes on gated runs.
REVIEW = {
    "reviewer": "Priya Shah",
    "approved_ids": ["EXP-2403", "EXP-2413", "EXP-2420"],
    "rejected_ids": ["EXP-2410"],
}

EMPLOYEES = 5


def _emit(pipeline_yaml: Path, runtime: str, out: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["emit", str(pipeline_yaml), "--runtime", runtime, "--out", str(out), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    # Every impl is real code copied from expected/extracted/, never a stub.
    assert payload["unimplemented_stubs"] == []
    assert payload["contract_findings"] == []


def _run(app: Path, capsys: pytest.CaptureFixture[str], *extra: str) -> dict[str, Any]:
    # A short timeout: Temporal retries a failing activity forever by
    # default, so a regression would otherwise hang for the full 600s.
    rc = main(["run", str(app), "--input", str(INPUT), "--timeout", "90", *extra])
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    result: dict[str, Any] = json.loads(captured.out)["file_report"]
    return result


def _assert_final(result: dict[str, Any], report_dir: Path) -> None:
    assert result["status"] == "final"
    assert result["reviewer"] == "Priya Shah"
    assert result["batch_total_usd"] == 3304.15
    # Approved review items are paid; the rejected one joins the blocks.
    assert result["reimbursable_usd"] == 2990.98  # 2395.59 + 130.39 + 18.75 + 446.25
    assert result["pending_usd"] == 0.0
    assert result["returned_usd"] == 313.17  # 64.17 + 249.00
    report = (report_dir / "2026-09-expense-audit.md").read_text(encoding="utf-8")
    assert "Status: **final** (reviewed by Priya Shah)" in report
    assert "| EXP-2410 | Ben Okafor | over_cap | review |" in report
    assert report.count("| rejected |") == 1
    assert report.count("| approved |") == 3
    assert "| pending |" not in report


def test_draft_variant_is_the_gated_pipeline_minus_the_gate() -> None:
    """Two hand-maintained IRs drift; pin the only allowed difference."""
    gated, draft = load_pipeline(GATED), load_pipeline(DRAFT)
    gate = gated.node_by_id("controller_review")
    assert gate.kind is NodeKind.HITL_GATE
    assert gated.requires_durable_execution and not draft.requires_durable_execution

    gated_nodes = {n.id: n for n in gated.nodes if n.id != gate.id}
    draft_nodes = {n.id: n for n in draft.nodes}
    assert gated_nodes.keys() == draft_nodes.keys()
    for node_id, node in gated_nodes.items():
        expected = node.model_dump()
        actual = draft_nodes[node_id].model_dump()
        if node_id == "file_report":
            assert expected["inputs"].pop("review") == "controller_review.output"
            expected["input"].pop("review")
        assert actual == expected, node_id

    def edges(p: Any) -> set[tuple[str, str]]:
        return {(e.from_, e.to) for e in p.edges}

    assert edges(gated) - edges(draft) == {
        ("summarize_audit", "controller_review"),
        ("controller_review", "file_report"),
    }
    assert edges(draft) - edges(gated) == {("summarize_audit", "file_report")}


def test_example_exercises_the_shapes_the_readme_claims() -> None:
    """A parallel entry wave, a fan_out, mandatory checks, no inference."""
    from rote.adapters._common import _execution_waves

    pipeline = load_pipeline(GATED)
    waves = [[n.id for n in wave] for wave in _execution_waves(pipeline)]
    assert sorted(waves[0]) == ["convert_to_usd", "find_duplicates", "find_late_submissions"]
    assert pipeline.node_by_id("audit_report").fan_out
    kinds = {n.kind for n in pipeline.nodes}
    assert NodeKind.LLM_JUDGE not in kinds and NodeKind.AGENT_LOOP not in kinds
    assert {n.id for n in pipeline.nodes if n.mandatory} >= {
        "find_duplicates",
        "find_late_submissions",
    }


def test_python_refuses_the_gated_pipeline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main(["emit", str(GATED), "--runtime", "python", "--out", str(tmp_path / "nope")])
    assert rc == 1
    assert "--runtime dbos" in capsys.readouterr().err


def test_free_first_run_on_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The README's 60-second block, verbatim: emit, run, read the report."""
    report_dir = tmp_path / "reports"
    monkeypatch.setenv("EXPENSE_AUDIT_REPORT_DIR", str(report_dir))
    app = tmp_path / "expense-audit"
    _emit(DRAFT, "python", app, capsys)
    result = _run(app, capsys)

    assert result["status"] == "draft"
    assert result["reviewer"] is None
    assert result["batch_total_usd"] == 3304.15
    assert result["reimbursable_usd"] == 2395.59
    assert result["pending_usd"] == 844.39
    assert result["returned_usd"] == 64.17
    assert result["auto_approved"] == ["Dev Patel", "Emi Tanaka"]
    assert (result["review_items"], result["blocked_items"]) == (4, 2)

    report = (report_dir / "2026-09-expense-audit.md").read_text(encoding="utf-8")
    assert result["report_path"] == str((report_dir / "2026-09-expense-audit.md").resolve())
    assert "Status: **draft** (awaiting controller review)" in report
    # The fan_out audited every employee, not one of them N times.
    employees_section = report.split("## Employees")[1].split("## Review queue")[0]
    employee_rows = [ln for ln in employees_section.splitlines() if ln.startswith("| ")][2:]
    names = [row.split(" | ")[0].removeprefix("| ") for row in employee_rows]
    assert names == ["Ana Ruiz", "Ben Okafor", "Chloe Martin", "Dev Patel", "Emi Tanaka"]
    assert len(names) == EMPLOYEES
    assert "| EXP-2412 | Ben Okafor | duplicate | block |" in report
    assert "| EXP-2404 | Ana Ruiz | receipt_missing | block |" in report
    assert report.count("| pending |") == 4
    assert report.count("| returned |") == 2


def test_controller_cannot_approve_a_block_item() -> None:
    """The skill's "ask again rather than guess" rule fails the step loudly."""
    import importlib.util
    import sys

    sys.path.insert(0, str(EXAMPLE / "expected"))
    try:
        for name in [m for m in sys.modules if m == "extracted" or m.startswith("extracted.")]:
            del sys.modules[name]
        spec = importlib.util.find_spec("extracted.report")
        assert spec is not None and spec.origin == str(EXAMPLE / "expected/extracted/report.py")
        from extracted.report import file_report  # type: ignore[import-not-found]

        summary = {
            "batch_total_usd": 50.0,
            "employees": [],
            "auto_approved": [],
            "review_queue": [
                {
                    "expense_id": "EXP-1",
                    "employee": "A",
                    "rule": "duplicate",
                    "severity": "block",
                    "detail": "",
                }
            ],
            "expenses": [{"id": "EXP-1", "employee": "A", "amount_usd": 50.0}],
        }
        with pytest.raises(ValueError, match="EXP-1"):
            file_report("2026-09", summary, {"reviewer": "x", "approved_ids": ["EXP-1"]})
    finally:
        sys.path.remove(str(EXAMPLE / "expected"))
        for name in [m for m in sys.modules if m == "extracted" or m.startswith("extracted.")]:
            del sys.modules[name]


def test_temporal_activities_run_the_shipped_sync_modules(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Emitted temporal apps carry their own extracted/ and await sync code.

    Two gaps made ``rote run`` die on the first temporal activity: the
    activities imported ``expected.extracted.*`` (a package that exists
    only in the source tree), and they ``await``ed the function call,
    which raises on the plain functions the dbos and python runtimes
    call. Activity functions are ordinary coroutines, so this calls one
    directly with no Temporal server.
    """
    import asyncio
    import sys

    app = tmp_path / "expense-audit-temporal"
    _emit(GATED, "temporal", app, capsys)
    activities_src = (app / "activities.py").read_text(encoding="utf-8")
    assert "from extracted.fx import convert_to_usd" in activities_src
    assert "expected.extracted" not in activities_src

    batch = json.loads(INPUT.read_text(encoding="utf-8"))
    sys.path.insert(0, str(app))
    stale = [m for m in sys.modules if m.split(".")[0] in ("activities", "extracted")]
    for name in stale:
        del sys.modules[name]
    try:
        import activities  # type: ignore[import-not-found]

        result = asyncio.run(
            activities.convert_to_usd(
                {"submissions": batch["submissions"], "period": batch["period"]}
            )
        )
    finally:
        sys.path.remove(str(app))
        for name in [m for m in sys.modules if m.split(".")[0] in ("activities", "extracted")]:
            del sys.modules[name]
    by_id = {e["id"]: e["amount_usd"] for e in result["expenses"]}
    assert by_id["EXP-2420"] == 446.25  # 340.00 GBP x 1.3125
    assert by_id["EXP-2440"] == 104.41  # 15400 JPY x 0.00678, rounded once
    assert len(by_id) == 16


@pytest.mark.slow
def test_gated_run_on_dbos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The durable path: park on controller_review, resume, file final."""
    pytest.importorskip("dbos")
    report_dir = tmp_path / "reports"
    monkeypatch.setenv("EXPENSE_AUDIT_REPORT_DIR", str(report_dir))
    app = tmp_path / "expense-audit-dbos"
    _emit(GATED, "dbos", app, capsys)
    result = _run(app, capsys, "--signal", f"controller_review={json.dumps(REVIEW)}")
    _assert_final(result, report_dir)


@pytest.fixture
def _forget_app_modules() -> Any:
    """Undo the temporal runner's in-process imports after the test.

    The runner puts the app dir on ``sys.path`` and imports its
    top-level ``workflow`` / ``activities`` / ``extracted`` packages.
    Left behind, this app's ``extracted`` shadows the next in-process
    app's (test_dbos_e2e then fails on ``extracted.taxonomy``).
    """
    import sys

    saved_path = list(sys.path)
    yield
    sys.path[:] = saved_path
    for name in [
        m for m in sys.modules if m.split(".")[0] in ("workflow", "activities", "extracted")
    ]:
        del sys.modules[name]


@pytest.mark.slow
@pytest.mark.usefixtures("_forget_app_modules")
def test_gated_run_on_temporal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Temporal awaits the same sync extracted/ modules dbos and python call."""
    pytest.importorskip("temporalio")
    report_dir = tmp_path / "reports"
    monkeypatch.setenv("EXPENSE_AUDIT_REPORT_DIR", str(report_dir))
    app = tmp_path / "expense-audit-temporal"
    _emit(GATED, "temporal", app, capsys)
    assert (app / "extracted" / "audit.py").read_text() == (
        EXAMPLE / "expected" / "extracted" / "audit.py"
    ).read_text()
    result = _run(app, capsys, "--signal", f"controller_review={json.dumps(REVIEW)}")
    _assert_final(result, report_dir)
