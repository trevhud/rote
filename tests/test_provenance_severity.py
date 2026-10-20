"""What each provenance finding costs a compile.

The user's rule, from a real BDR compile that exited 1 on 15 recorded
assumptions: a citation the skill does not contain is a factual error and
fails; an assumption is a declared decision and never fails, it is listed
for review; a missing declaration (an unmarked outcome) is repaired, then
a warning. rote renders the report's Open questions from the markers
itself, so an assumption can no longer be missing from them. CI opts into
failing on all of it with ``--strict-provenance``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rote.compiler import Compiler
from rote.compiler.provenance import assumptions
from rote.compiler.verify import verify_deliverable
from rote.ir import load_pipeline
from tests.test_compiler import VALID_YAML, _RepairableDriver

SKILL_MD = """\
---
name: fake
---

# Fake skill

Every new account starts on the bronze tier until it has three orders.
"""

PIPELINE = (
    VALID_YAML.replace(
        "    impl: extracted/foo.py:only_node\n",
        "    impl: extracted/foo.py:only_node\n"
        "  - id: second_node\n"
        "    kind: pure_function\n"
        "    description: Another node in the same module.\n"
        "    impl: extracted/foo.py:second_node\n",
    )
    .replace("exit_nodes: [only_node]", "exit_nodes: [only_node, second_node]")
    .replace("entry_nodes: [only_node]", "entry_nodes: [only_node, second_node]")
)


#: Three assumptions, every outcome marked.
MODULE = """\
def only_node():
    # rote: assumption: a new account's tier is bronze
    tier = "bronze"
    # rote: assumption: tier names are lowercase
    return {"tier": tier}


def second_node():
    # rote: assumption: an account with no orders is inactive
    return {"status": "inactive"}
"""

FIRST = "    # rote: assumption: a new account's tier is bronze\n"
SECOND_NODE_MARKER = "    # rote: assumption: an account with no orders is inactive\n"


def _module(*, cite: str | None = None, unmarked: bool = False, extra: int = 0) -> str:
    """MODULE, with ``cite`` replacing the first assumption by a citation,
    ``unmarked`` dropping second_node's marker, and ``extra`` more assumptions."""
    module = MODULE
    more = "".join(f"    # rote: assumption: extra rule {i}\n" for i in range(extra))
    module = module.replace(FIRST, FIRST + more)
    if cite is not None:
        module = module.replace(FIRST, f'    # skill: "{cite}"\n')
    if unmarked:
        module = module.replace(SECOND_NODE_MARKER, "")
    return module


TEST = (
    "from extracted.foo import only_node, second_node\n\n"
    "def test_nodes():\n"
    "    assert only_node() == {'tier': 'bronze'}\n"
    "    assert second_node() == {'status': 'inactive'}\n"
)

COVERED = "# Report\n\n## Open questions\n\n" + "".join(f"- Question {i}\n" for i in range(9))
UNCOVERED = "# Report\n\n## Open questions\n\n- Only one question.\n"


def _deliverable(root: Path, module: str, report: str) -> Path:
    for rel, content in {
        "pipeline.yaml": PIPELINE,
        "extracted/foo.py": module,
        "tests/test_foo.py": TEST,
        "compile-report.md": report,
    }.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(content, encoding="utf-8")
    return root


@pytest.fixture
def skill(tmp_path: Path) -> Path:
    d = tmp_path / "skill"
    d.mkdir()
    (d / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")
    return d


# ───────── Verification ─────────


@pytest.mark.parametrize("report", [COVERED, UNCOVERED])
def test_any_number_of_assumptions_never_fails(tmp_path: Path, skill: Path, report: str) -> None:
    """Nor does an agent report that lists fewer of them: rote lists them."""
    root = _deliverable(tmp_path / "out", _module(extra=6), report)
    v = verify_deliverable(root, skill_dir=skill, run_tests=False)
    assert v.ok and v.review_gaps == [] and v.repair_instruction(root) is None
    assert len(v.assumptions) == 9
    strict = verify_deliverable(root, skill_dir=skill, run_tests=False, strict_provenance=True)
    assert strict.ok, strict.errors


def test_each_assumption_names_its_node_file_and_line(tmp_path: Path) -> None:
    root = _deliverable(tmp_path / "out", _module(), COVERED)
    found = assumptions(load_pipeline(root / "pipeline.yaml"), root)
    by_text = {a.text: (a.node_id, a.path, a.line) for a in found}
    assert by_text["a new account's tier is bronze"] == ("only_node", "extracted/foo.py", 2)
    assert by_text["an account with no orders is inactive"][0] == "second_node"
    source = (root / "extracted/foo.py").read_text().splitlines()
    for a in found:
        assert a.text in source[a.line - 1]


def test_a_fabricated_citation_fails_with_or_without_strict(tmp_path: Path, skill: Path) -> None:
    root = _deliverable(
        tmp_path / "out", _module(cite="every account starts on the platinum tier"), COVERED
    )
    for strict in (False, True):
        v = verify_deliverable(root, skill_dir=skill, run_tests=False, strict_provenance=strict)
        assert [f.kind for f in v.errors] == ["citation_not_in_skill"], strict
        assert not v.ok


def test_a_real_citation_passes(tmp_path: Path, skill: Path) -> None:
    root = _deliverable(
        tmp_path / "out", _module(cite="Every new account starts on the bronze tier"), COVERED
    )
    assert verify_deliverable(root, skill_dir=skill, run_tests=False).ok


def test_a_review_gap_is_repaired_then_a_warning_and_strict_makes_it_fail(
    tmp_path: Path, skill: Path
) -> None:
    gap = "uncited_outcome"
    root = _deliverable(tmp_path / "out", _module(unmarked=True), COVERED)
    v = verify_deliverable(root, skill_dir=skill, run_tests=False)
    assert v.ok, v.errors
    assert [f.kind for f in v.review_gaps] == [gap]
    instruction = v.repair_instruction(root)
    assert instruction is not None and "Provenance gaps" in instruction
    strict = verify_deliverable(root, skill_dir=skill, run_tests=False, strict_provenance=True)
    assert [f.kind for f in strict.errors] == [gap] and strict.review_gaps == []


# ───────── The compile ─────────


@pytest.mark.asyncio
async def test_a_gap_the_repairs_do_not_close_still_completes_the_compile(
    tmp_path: Path, skill: Path
) -> None:
    files = {"extracted/foo.py": _module(unmarked=True), "tests/test_foo.py": TEST}
    driver = _RepairableDriver([PIPELINE], files=files)
    compiler_dir = tmp_path / "compiler"
    compiler_dir.mkdir()
    (compiler_dir / "SKILL.md").write_text("# fake compiler")
    events: list[object] = []
    compiler = Compiler(compiler_skill_dir=compiler_dir, on_event=events.append)
    compiler.select_driver = lambda: driver  # type: ignore[method-assign]
    result = await compiler.compile(skill, tmp_path / "out")
    assert len(driver.repair_instructions) == 2
    assert all("Provenance gaps" in i for i in driver.repair_instructions)
    assert result.verification is not None and result.verification.ok
    messages = [getattr(e, "message", "") for e in events]
    assert any(m.startswith("provenance review:") for m in messages)
    assert any("2 assumption(s) the skill does not state" in m for m in messages)


def _run_compile(
    tmp_path: Path,
    skill: Path,
    monkeypatch: pytest.MonkeyPatch,
    files: dict[str, str],
    *extra: str,
) -> tuple[int, Path]:
    from rote.cli import main

    driver = _RepairableDriver([PIPELINE], files=files)
    monkeypatch.setattr(Compiler, "select_driver", lambda self: driver)
    out = tmp_path / "run"
    argv = ["compile", str(skill), "--local", "--runtime", "dbos", "--no-eval", "--out", str(out)]
    return main([*argv, *extra]), out


@pytest.mark.parametrize(("strict", "rc"), [(False, 0), (True, 1)])
def test_rote_compile_lists_assumptions_and_exits_by_policy(
    tmp_path: Path,
    skill: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    strict: bool,
    rc: int,
) -> None:
    files = {"extracted/foo.py": _module(unmarked=True), "tests/test_foo.py": TEST}
    extra = ["--json", *(["--strict-provenance"] if strict else [])]
    code, out = _run_compile(tmp_path, skill, monkeypatch, files, *extra)
    assert code == rc
    captured = capsys.readouterr()
    assert "2 assumption(s) the skill does not state" in captured.err
    assert "  only_node:\n    extracted/foo.py:" in captured.err
    assert ("provenance warning (uncited_outcome)" in captured.err) is not strict
    payload = json.loads(captured.out)
    prov = payload["provenance"]
    assert prov["assumptions"] == 2 and len(prov["assumption_list"]) == 2
    assert {a["node"] for a in prov["assumption_list"]} == {"only_node"}
    assert (prov["review_gaps"], prov["errors"]) == ((0, 1) if strict else (1, 0))
    summary = json.loads((out / "progress.jsonl").read_text().splitlines()[-1])
    assert summary["type"] == "summary"
    assert summary["provenance"]["assumptions"] == 2
    report = (out / "compiled" / "compile-report.md").read_text()
    assert "### OQ-1: only_node" in report and "### OQ-2: only_node" in report
    assert "- Assumptions the skill does not state: 2. Review each one:" in report
    assert "  - only_node:\n    - `extracted/foo.py:" in report


def test_rote_compile_fails_on_a_fabricated_citation_without_strict(
    tmp_path: Path,
    skill: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _module(cite="every account starts on the platinum tier")
    files = {"extracted/foo.py": module, "tests/test_foo.py": TEST, "compile-report.md": COVERED}
    code, _ = _run_compile(tmp_path, skill, monkeypatch, files)
    assert code == 1
    err = capsys.readouterr().err
    assert "unresolved: (provenance): extracted/foo.py line 2 cites the skill" in err, err


@pytest.mark.parametrize(("strict", "rc"), [(False, 0), (True, 1)])
def test_rote_analyze_fails_only_under_strict_provenance(
    tmp_path: Path,
    skill: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    strict: bool,
    rc: int,
) -> None:
    from rote.cli import main

    files = {"extracted/foo.py": _module(unmarked=True), "tests/test_foo.py": TEST}
    driver = _RepairableDriver([PIPELINE], files=files)
    monkeypatch.setattr(Compiler, "select_driver", lambda self: driver)
    argv = ["analyze", str(skill), "--json", *(["--strict-provenance"] if strict else [])]
    assert main(argv) == rc
    report = json.loads(capsys.readouterr().out)
    assert report["provenance"]["assumptions"] == 2
    assert report["provenance"]["review_gaps" if not strict else "errors"] == 1
