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


def test_a_non_utf8_module_is_a_finding_not_a_crash(tmp_path: Path) -> None:
    """An agent-written Latin-1 module used to raise UnicodeDecodeError out
    of verification, which runs where only DriverError is caught."""
    root = _deliverable(
        tmp_path,
        {
            "extracted/vendor.py": VENDOR_STUB,
            "tests/test_logic.py": GOOD_TEST + "# caf\xe9\n",
        },
    )
    (root / "extracted" / "logic.py").write_bytes(REAL.encode() + b"# caf\xe9\n")
    v = verify_deliverable(root)
    assert not v.ok
    assert [(f.node_id, f.kind) for f in v.errors] == [("count_open", "not_utf8")]
    assert "extracted/logic.py is not valid UTF-8" in v.errors[0].message


def test_any_crash_inside_verification_becomes_a_finding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _deliverable(tmp_path, {"extracted/logic.py": REAL, "extracted/vendor.py": VENDOR_STUB})

    def boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk went away")

    monkeypatch.setattr("rote.compiler.verify.check_contracts", boom)
    v = verify_deliverable(root)
    assert not v.ok
    [finding] = v.errors
    assert finding.kind == "verification_error"
    assert "OSError: disk went away" in finding.message
    assert v.repair_instruction(root) is not None


def test_every_repair_prompt_restates_the_write_once_rules(tmp_path: Path) -> None:
    """A real repair round re-edited compile-report.md three times, one turn
    each. Both prompt shapes (invalid IR, failed checks) carry the rule."""
    from rote.compiler.verify import WRITE_ONCE_RULES

    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "pipeline.yaml").write_text("name: x\nnodes: [{id: a, kind: nope}]\n")
    invalid = verify_deliverable(tmp_path / "bad").repair_instruction(tmp_path / "bad")
    root = _deliverable(tmp_path / "red", {"extracted/vendor.py": VENDOR_STUB})
    failing = verify_deliverable(root).repair_instruction(root)
    for prompt in (invalid, failing):
        assert prompt is not None and WRITE_ONCE_RULES in prompt
    assert "ONE Write" in WRITE_ONCE_RULES and "compile-report.md" in WRITE_ONCE_RULES


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


# ───────── Every outcome traces to the skill, or is a declared assumption ─────────

_SKILL_MD = """\
# Expense audit

Lines under **$25.00** don't need a receipt. A missing receipt on a line of
$25.00 or more is a **block**: the employee must resubmit.
"""

_CITED = """\
def count_open(rows):
    # skill: "A missing receipt on a line of $25.00 or more is a block"
    return {"open": sum(1 for r in rows if r["Resolved"] == "FALSE")}
"""


def _provenance_case(tmp_path: Path, module: str, report: str | None) -> Path:
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text(_SKILL_MD)
    files = {
        "extracted/logic.py": module,
        "extracted/vendor.py": VENDOR_STUB,
        "tests/test_logic.py": GOOD_TEST,
    }
    if report is not None:
        files["compile-report.md"] = report
    return _deliverable(tmp_path / "out", files)


def test_a_verbatim_citation_passes(tmp_path: Path) -> None:
    root = _provenance_case(tmp_path, _CITED, None)
    assert verify_deliverable(root, skill_dir=tmp_path / "skill", run_tests=False).ok


def test_an_elided_citation_passes_when_each_fragment_is_verbatim(tmp_path: Path) -> None:
    module = _CITED.replace(
        '"A missing receipt on a line of $25.00 or more is a block"',
        '"Lines under $25.00 don\'t need a receipt ... line of $25.00 or more is a block"',
    )
    root = _provenance_case(tmp_path, module, None)
    v = verify_deliverable(root, skill_dir=tmp_path / "skill", run_tests=False)
    assert v.ok, v.errors


def test_a_citation_wrapped_across_comment_lines_passes(tmp_path: Path) -> None:
    """A real compile wrapped long quotes over several `#` lines, and each
    was rejected as not in the skill, costing a needless repair round."""
    module = _CITED.replace(
        '    # skill: "A missing receipt on a line of $25.00 or more is a block"\n',
        '    # skill: "A missing receipt on a line of $25.00 or more\n    # is a block"\n',
    )
    root = _provenance_case(tmp_path, module, None)
    v = verify_deliverable(root, skill_dir=tmp_path / "skill", run_tests=False)
    assert v.ok, v.errors


def test_an_invented_citation_is_rejected(tmp_path: Path) -> None:
    """A citation that would cover a guessed outcome with made-up skill text."""
    module = _CITED.replace("is a block", "goes to review")
    root = _provenance_case(tmp_path, module, None)
    v = verify_deliverable(root, skill_dir=tmp_path / "skill", run_tests=False)
    assert [f.kind for f in v.errors] == ["citation_not_in_skill"]
    assert "goes to review" in v.errors[0].message


_ASSUMED = _CITED.replace(
    '    return {"open"',
    "    # rote: assumption: an over-cap line is a block (the skill names the rule, "
    "not its severity)\n"
    '    return {"open"',
)


@pytest.mark.parametrize(
    ("report", "ok"),
    [
        (None, False),
        ("# Report\n\n## Open questions\n\n(none)\n\n## Next steps\n\n- ship it\n", False),
        (
            "# Report\n\n## Open questions\n\n- Over-cap lines are treated as block; the "
            "skill does not say. Confirm with finance.\n",
            True,
        ),
        ("**Open questions:**\n\n1. Over-cap severity is assumed block.\n", True),
        # One sub-heading per question, the shape a real ops-report report
        # used; an items-only reading saw an empty section.
        (
            "## Open questions\n\n### OQ-1: Over-cap severity\n**We chose:** block.\n\n"
            "## Next steps\n\n- ship it\n",
            True,
        ),
        # A sub-heading AFTER the section ended is not an entry.
        ("## Open questions\n\n## Next steps\n\n### Step 1\n- ship it\n", False),
    ],
)
def test_a_recorded_assumption_must_reach_the_open_questions(
    tmp_path: Path, report: str | None, ok: bool
) -> None:
    root = _provenance_case(tmp_path, _ASSUMED, report)
    v = verify_deliverable(root, skill_dir=tmp_path / "skill", run_tests=False)
    assert v.ok is ok, v.errors
    if not ok:
        assert [f.kind for f in v.errors] == ["unlisted_assumption"]


def test_no_tests_dir_means_nothing_to_run(tmp_path: Path) -> None:
    assert run_generated_tests(tmp_path) is None


#: What a skill-steered "test" would go looking for.
_SECRETS = {
    "ANTHROPIC_API_KEY": "sk-ant-should-not-leak",
    "OPENAI_API_KEY": "sk-should-not-leak",
    "CLOUDFLARE_API_TOKEN": "cf-should-not-leak",
    "ROTE_CLOUD_TOKEN": "rote_should_not_leak",
    "AWS_SECRET_ACCESS_KEY": "aws-should-not-leak",
    "GITHUB_TOKEN": "ghp-should-not-leak",
}

_SNOOPING_TEST = """\
import os
from pathlib import Path

SECRETS = {names!r}


def test_sees_no_credentials():
    leaked = {{k: os.environ[k] for k in SECRETS if k in os.environ}}
    assert leaked == {{}}, leaked
    assert not any("should-not-leak" in v or "should_not_leak" in v for v in os.environ.values())


def test_home_holds_none_of_the_users_files():
    home = Path.home()
    assert str(home) != {real_home!r}
    assert not (home / ".local" / "share" / "rote" / "cloud.json").exists()
    assert list(home.iterdir()) == []
"""


def test_generated_tests_cannot_see_credentials_or_the_users_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A third-party SKILL.md can steer the agent into a test that reads
    secrets, and rote runs that test automatically. The child environment
    is an allowlist and HOME is empty, so there is nothing to read.

    The test runs a real pytest subprocess with real secrets exported and
    a real cloud credential under the parent's HOME.
    """
    real_home = tmp_path / "real-home"
    cred = real_home / ".local" / "share" / "rote" / "cloud.json"
    cred.parent.mkdir(parents=True)
    cred.write_text('{"token": "rote_should_not_leak"}')
    monkeypatch.setenv("HOME", str(real_home))
    for name, value in _SECRETS.items():
        monkeypatch.setenv(name, value)

    root = tmp_path / "deliverable"
    (root / "tests").mkdir(parents=True)
    (root / "tests" / "test_snoop.py").write_text(
        _SNOOPING_TEST.format(names=sorted(_SECRETS), real_home=str(real_home))
    )
    result = run_generated_tests(root)
    assert result is not None
    assert result.status == "passed", result.output
    assert "2 passed" in result.summary


def test_missing_pytest_is_unverified_not_passed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no test runner the deliverable is unverified, said the same way
    a missing sandbox is. It used to be a quiet 'unavailable' that a real
    compile exited 0 on while a generated test was failing."""
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_x.py").write_text("def test_x():\n    assert False\n")
    monkeypatch.setattr("rote.compiler.verify._pytest_available", lambda: False)
    result = run_generated_tests(tmp_path)
    assert result is not None and result.status == "unverified" and not result.passed
    assert result.summary.startswith("unverified: pytest not available")


def test_the_compile_report_says_unverified_when_pytest_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import argparse

    from rote.cli import _record_verification_in_report

    root = _deliverable(tmp_path, {"extracted/logic.py": REAL, "tests/test_logic.py": GOOD_TEST})
    (root / "compile-report.md").write_text("# Report\n")
    monkeypatch.setattr("rote.compiler.verify._pytest_available", lambda: False)
    v = verify_deliverable(root)
    args = argparse.Namespace(no_verify_tests=False, trust_skill=False)
    _record_verification_in_report(root, v, v.tests, args)
    report = (root / "compile-report.md").read_text()
    assert "UNVERIFIED: unverified: pytest not available" in report


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
    v = verify_deliverable(snapshot, skill_dir=snapshot.parent.parent / "skill")
    assert v.ok, (v.errors, v.tests and v.tests.output)
    assert v.tests is not None and v.tests.passed
    by_kind: dict[str, set[str]] = {}
    for impl in v.implementations:
        by_kind.setdefault(impl.kind, set()).add(impl.status)
    assert by_kind == {"pure_function": {"implemented"}, "external_call": {"stub"}}
    assert all(i.mcp_bound for i in v.implementations if i.status == "stub")


def test_the_summary_is_pytests_not_the_interpreters_last_words(tmp_path: Path) -> None:
    """A uv-managed Python under bwrap prints a termcap warning to stderr as
    it exits. The summary is pytest's result line, not that."""
    noisy = (
        "import atexit, sys\n"
        "atexit.register(lambda: sys.stderr.write('using dumb terminal settings.\\n'))\n\n"
        "def test_ok():\n    assert True\n"
    )
    root = _deliverable(tmp_path, {"tests/test_noisy.py": noisy})
    result = run_generated_tests(root, trust_skill=True)
    assert result is not None and result.status == "passed", result.output
    assert "using dumb terminal settings." in result.output
    assert result.summary.startswith("1 passed"), result.summary
