"""Tests for the compile-time deliverable check (``rote.compiler.verify``).

The check decides whether a paid compile is sent back to the agent, so
both directions matter: it must catch a deliverable that would not run,
and it must stay quiet on one that would, or every healthy compile pays
for a pointless repair round.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rote.compiler.verify import run_generated_tests, verify_deliverable

PIPELINE = """\
name: two-steps
input:
  type: In
  required: [rows]
nodes:
  - id: count_open
    kind: pure_function
    description: Count rows where Resolved is FALSE.
    impl: extracted/logic.py:count_open
    inputs: {rows: pipeline.input.rows}
  - id: fetch_sheet
    kind: external_call
    description: Read the sheet from the vendor API.
    impl: extracted/vendor.py:fetch_sheet
    inputs: {rows: pipeline.input.rows}
edges: []
entry_nodes: [count_open, fetch_sheet]
exit_nodes: [count_open, fetch_sheet]
"""

REAL = """\
def count_open(rows: list[dict]) -> dict:
    return {"open": sum(1 for r in rows if r["Resolved"] == "FALSE")}
"""

VENDOR_STUB = """\
def fetch_sheet(rows):
    raise NotImplementedError("calls the vendor's sheet API, whose shape the skill does not state")
"""

GOOD_TEST = """\
from extracted.logic import count_open


def test_counts_only_unresolved_rows():
    rows = [{"Resolved": "FALSE"}, {"Resolved": "TRUE"}, {"Resolved": "FALSE"}]
    assert count_open(rows) == {"open": 2}
"""


def _deliverable(root: Path, files: dict[str, str]) -> Path:
    for rel, content in {"pipeline.yaml": PIPELINE, **files}.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(content, encoding="utf-8")
    return root


def test_a_healthy_deliverable_needs_no_repair(tmp_path: Path) -> None:
    root = _deliverable(
        tmp_path,
        {
            "extracted/logic.py": REAL,
            "extracted/vendor.py": VENDOR_STUB,
            "tests/test_logic.py": GOOD_TEST,
        },
    )
    v = verify_deliverable(root)
    assert v.ok, (v.errors, v.tests)
    assert v.tests is not None and v.tests.status == "passed"
    assert v.repair_instruction(root) is None
    statuses = {i.node_id: (i.status, i.reason) for i in v.implementations}
    assert statuses["count_open"] == ("implemented", None)
    # An external_call stub is the documented default, not a problem.
    assert statuses["fetch_sheet"][0] == "stub"
    assert "vendor's sheet API" in str(statuses["fetch_sheet"][1])


def test_red_generated_tests_are_sent_back_with_the_failure(tmp_path: Path) -> None:
    wrong = REAL.replace('== "FALSE"', '== "TRUE"')  # inverts the skill's rule
    root = _deliverable(
        tmp_path,
        {
            "extracted/logic.py": wrong,
            "extracted/vendor.py": VENDOR_STUB,
            "tests/test_logic.py": GOOD_TEST,
        },
    )
    v = verify_deliverable(root)
    assert not v.ok
    assert v.tests is not None and v.tests.status == "failed"
    instruction = v.repair_instruction(root)
    assert instruction is not None
    assert "test_counts_only_unresolved_rows" in instruction
    assert "assert {'open': 1} == {'open': 2}" in instruction
    assert "the skill decides which is wrong" in instruction


@pytest.mark.parametrize(
    ("files", "kind"),
    [
        # The skill states the rule, so the agent may not skip the module.
        ({"extracted/vendor.py": VENDOR_STUB}, "pure_function_not_written"),
        (
            {
                "extracted/logic.py": "def count_open(rows):\n    raise NotImplementedError\n",
                "extracted/vendor.py": VENDOR_STUB,
            },
            "unexplained_stub",
        ),
        (
            {"extracted/logic.py": REAL, "extracted/vendor.py": VENDOR_STUB},
            "untested_implementation",
        ),
    ],
)
def test_a_pure_function_is_implemented_and_tested_or_explained(
    tmp_path: Path, files: dict[str, str], kind: str
) -> None:
    v = verify_deliverable(_deliverable(tmp_path, files))
    assert [f.kind for f in v.errors] == [kind]
    assert v.errors[0].node_id == "count_open"
    assert not v.ok


def test_an_explained_pure_function_stub_is_accepted(tmp_path: Path) -> None:
    explained = (
        "def count_open(rows):\n"
        "    raise NotImplementedError('needs the escalation threshold, which the skill "
        "does not state')\n"
    )
    v = verify_deliverable(
        _deliverable(
            tmp_path, {"extracted/logic.py": explained, "extracted/vendor.py": VENDOR_STUB}
        )
    )
    assert v.ok, v.errors
    [impl] = [i for i in v.implementations if i.node_id == "count_open"]
    assert impl.status == "stub"
    assert impl.reason is not None and "escalation threshold" in impl.reason


def test_a_raise_inside_a_branch_is_real_code_not_a_stub(tmp_path: Path) -> None:
    branchy = (
        "def count_open(rows):\n"
        "    if not isinstance(rows, list):\n"
        "        raise NotImplementedError('only list input is supported')\n"
        "    return {'open': len(rows)}\n"
    )
    v = verify_deliverable(
        _deliverable(
            tmp_path,
            {
                "extracted/logic.py": branchy,
                "extracted/vendor.py": VENDOR_STUB,
                "tests/test_logic.py": "from extracted.logic import count_open\n"
                "def test_it():\n    assert count_open([1])['open'] == 1\n",
            },
        )
    )
    [impl] = [i for i in v.implementations if i.node_id == "count_open"]
    assert impl.status == "implemented"


def test_an_invalid_pipeline_short_circuits_with_the_validation_text(tmp_path: Path) -> None:
    (tmp_path / "pipeline.yaml").write_text("name: x\nnodes: [{id: a, kind: nope}]\n")
    v = verify_deliverable(tmp_path)
    assert v.pipeline is None and v.pipeline_error is not None
    instruction = v.repair_instruction(tmp_path)
    assert instruction is not None and "fails IR validation" in instruction


def test_an_update_is_verified_against_the_previous_code(tmp_path: Path) -> None:
    """The work dir of an update holds only what the agent rewrote.

    Verified alone, every module the update left untouched would read as
    missing and trigger a pointless (paid) repair round.
    """
    previous = _deliverable(
        tmp_path / "previous",
        {
            "extracted/logic.py": REAL,
            "extracted/vendor.py": VENDOR_STUB,
            "tests/test_logic.py": GOOD_TEST,
        },
    )
    work = tmp_path / "work"
    work.mkdir()
    (work / "pipeline.yaml").write_text(PIPELINE, encoding="utf-8")

    assert not verify_deliverable(work).ok
    overlaid = verify_deliverable(work, base_dir=previous)
    assert overlaid.ok, overlaid.errors

    # And the overlay is real: the update's own rewrite wins over the base.
    (work / "extracted").mkdir()
    (work / "extracted" / "logic.py").write_text(REAL.replace('"FALSE"', '"TRUE"'))
    broken = verify_deliverable(work, base_dir=previous)
    assert broken.tests is not None and broken.tests.status == "failed"


def test_no_tests_dir_means_nothing_to_run(tmp_path: Path) -> None:
    assert run_generated_tests(tmp_path) is None


def test_missing_pytest_is_reported_not_passed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_x.py").write_text("def test_x():\n    assert True\n")
    monkeypatch.setattr("rote.compiler.verify._pytest_available", lambda: False)
    result = run_generated_tests(tmp_path)
    assert result is not None and result.status == "unavailable" and not result.passed


def test_the_committed_ops_report_compile_still_verifies() -> None:
    """The snapshot of a real compile under this rubric: every pure_function
    implemented, every generated test green, on whatever Python CI runs.

    Its four stubs are the mcp-bound fetches (vendor calls the default
    backend makes through MCP), which is the documented default.
    """
    snapshot = (
        Path(__file__).resolve().parent.parent
        / "examples/ops-report/runs/2026-10-18-implements-pure-functions"
    )
    v = verify_deliverable(snapshot)
    assert v.ok, (v.errors, v.tests and v.tests.output)
    assert v.tests is not None and v.tests.passed
    by_kind: dict[str, set[str]] = {}
    for impl in v.implementations:
        by_kind.setdefault(impl.kind, set()).add(impl.status)
    assert by_kind == {"pure_function": {"implemented"}, "external_call": {"stub"}}
    assert all(i.mcp_bound for i in v.implementations if i.status == "stub")
