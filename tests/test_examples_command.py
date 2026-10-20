"""`rote example`: the free first run without a clone.

The bundled files have one home (examples/<name>/), packaged into the
wheel by pyproject's force-include. These tests pin that the packaged
list and `rote.examples.BUNDLED` agree, drive the command into a temp
dir, then run the python-runtime commands it PRINTS (parsed from its own
output, so the instructions cannot drift from what works) and assert the
example README's totals. The slow test builds a real wheel and checks
the files are inside it.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import pytest

from rote.cli import main
from rote.examples import BUNDLED, examples_root

REPO = Path(__file__).resolve().parent.parent


def _force_include() -> dict[str, str]:
    with (REPO / "pyproject.toml").open("rb") as f:
        build = tomllib.load(f)["tool"]["hatch"]["build"]["targets"]["wheel"]
    return dict(build["force-include"])


def test_the_wheel_ships_exactly_the_bundled_files() -> None:
    expected = {
        f"examples/{name}/{rel}": f"rote/examples/{name}/{rel}"
        for name, example in BUNDLED.items()
        for rel in example.files
    }
    shipped = {k: v for k, v in _force_include().items() if k.startswith("examples/")}
    assert shipped == expected
    for source in expected:
        assert (REPO / source).exists(), source


def test_list_names_every_bundled_example(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["example"]) == 0
    out = capsys.readouterr().out
    assert list(BUNDLED) == ["expense-audit"]
    assert "  expense-audit  Month-end expense audit" in out


def test_unknown_example_names_the_available_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["example", "bdr-outreach"]) == 2
    assert "available: expense-audit" in capsys.readouterr().err
    assert not (tmp_path / "bdr-outreach").exists()


def test_an_existing_directory_is_refused_unless_forced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "expense-audit").mkdir()
    (tmp_path / "expense-audit" / "notes.txt").write_text("mine", encoding="utf-8")
    assert main(["example", "expense-audit"]) == 2
    assert "pass --force" in capsys.readouterr().err
    assert not (tmp_path / "expense-audit" / "input.json").exists()

    assert main(["example", "expense-audit", "--force"]) == 0
    assert (tmp_path / "expense-audit" / "input.json").is_file()
    assert (tmp_path / "expense-audit" / "notes.txt").read_text(encoding="utf-8") == "mine"


def test_copy_then_the_printed_python_commands_reproduce_the_readme_totals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["example", "expense-audit"]) == 0
    out = capsys.readouterr().out
    copy = tmp_path / "expense-audit"
    source = examples_root() / "expense-audit"
    for rel in BUNDLED["expense-audit"].files:
        assert (copy / rel).exists(), rel
    assert (copy / "expected" / "extracted" / "audit.py").read_bytes() == (
        source / "expected" / "extracted" / "audit.py"
    ).read_bytes()

    commands = [line.strip() for line in out.splitlines() if line.startswith("  rote ")]
    python_cmds = [c for c in commands if "app-python" in c]
    assert [c.split()[1] for c in python_cmds] == ["emit", "run"]
    assert any(c.startswith("rote eval ") for c in commands)

    emit_args = shlex.split(python_cmds[0])[1:]
    assert main([*emit_args, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["unimplemented_stubs"] == []
    assert main(shlex.split(python_cmds[1])[1:]) == 0
    result = json.loads(capsys.readouterr().out)["file_report"]

    # The example README's "Draft (python)" column.
    assert result["status"] == "draft"
    assert result["reimbursable_usd"] == 2395.59
    assert result["pending_usd"] == 844.39
    assert result["returned_usd"] == 64.17
    assert result["batch_total_usd"] == 3304.15
    assert (result["review_items"], result["blocked_items"]) == (4, 2)


def test_the_copied_readme_names_the_copys_own_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The README is written for the repo. The copy's version must not send
    a reader to examples/... or /tmp/..., and every file its layout tree
    names must be in the copy."""
    from rote.examples import rewrite_readme

    example = BUNDLED["expense-audit"]
    source = (examples_root() / "expense-audit" / "README.md").read_text(encoding="utf-8")
    for old, _new in example.readme_rewrites:
        assert old in source, f"README no longer contains {old!r}; update readme_rewrites"

    monkeypatch.chdir(tmp_path)
    assert main(["example", "expense-audit"]) == 0
    capsys.readouterr()
    copy = tmp_path / "expense-audit"
    readme = (copy / "README.md").read_text(encoding="utf-8")
    assert readme == rewrite_readme(example, source, "expense-audit")
    assert "examples/expense-audit" not in readme
    assert "/tmp/" not in readme
    assert "From the repo root" not in readme
    assert "rote run expense-audit/app-python --input expense-audit/input.json" in readme

    # Every name in the Layout tree exists in the copy.
    tree = readme.split("## Layout")[1].split("```")[1]
    stack: list[str] = []
    for line in tree.splitlines()[2:]:  # skip the root line
        entry = line.split("#")[0].rstrip()
        if not entry.strip():
            continue
        name = entry.lstrip("│├└─ ").strip()
        depth = (len(entry) - len(entry.lstrip("│├└─ "))) // 4 - 1
        stack = [*stack[:depth], name.rstrip("/")]
        assert (copy / "/".join(stack)).exists(), "/".join(stack)
    assert (copy / "scorecard.md").is_file()
    assert (copy / "runs" / "2026-10-18-first-agent-run" / "pipeline.yaml").is_file()


def _build_wheel(out: Path) -> None:
    """Build rote's wheel with whichever builder this environment has.

    uv (the repo's dev setup), then `python -m build`, then pip (how CI
    installs). A uv-made venv has no pip, so pip alone failed there.
    """
    import importlib.util
    import shutil

    if shutil.which("uv"):
        command = ["uv", "build", "--wheel", "-o", str(out), str(REPO)]
    elif importlib.util.find_spec("build") is not None:
        command = [sys.executable, "-m", "build", "--wheel", "-o", str(out), str(REPO)]
    elif importlib.util.find_spec("pip") is not None:
        command = [sys.executable, "-m", "pip", "wheel", "--no-deps", "-w", str(out), str(REPO)]
    else:
        pytest.skip("no wheel builder: install uv, the `build` package, or pip")
    built = subprocess.run(command, capture_output=True, text=True, timeout=600)
    assert built.returncode == 0, built.stdout + built.stderr


@pytest.mark.slow
def test_a_built_wheel_contains_the_bundled_example(tmp_path: Path) -> None:
    """Build the real wheel and look inside."""
    _build_wheel(tmp_path)
    (wheel,) = tmp_path.glob("rote_cli-*.whl")
    names = set(zipfile.ZipFile(wheel).namelist())
    for name, example in BUNDLED.items():
        source = REPO / "examples" / name
        for rel in example.files:
            path = source / rel
            files = [p for p in path.rglob("*") if p.is_file()] if path.is_dir() else [path]
            for file in files:
                if "__pycache__" in file.parts:
                    continue
                inside = f"rote/examples/{name}/{file.relative_to(source).as_posix()}"
                assert inside in names, inside
    # Nothing beyond the bundled list: the other examples stay out.
    assert not any(n.startswith("rote/examples/bdr-outreach/") for n in names)
