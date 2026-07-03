"""Tests for the rote CLI.

Two flavors:

* **In-process tests** call ``rote.cli.main`` directly with fake argv.
  Fast and lets us assert return codes + captured stdout/stderr. These
  cover argument parsing, error handling, and subcommand dispatch.
* **Subprocess tests** invoke the installed ``rote`` entry point via
  ``python -m rote.cli`` so we catch packaging / import-time issues the
  in-process tests would miss.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from rote.cli import main as cli_main

REPO_ROOT = Path(__file__).resolve().parent.parent
BDR_PIPELINE_YAML = REPO_ROOT / "examples" / "bdr-outreach" / "expected" / "pipeline.yaml"


# ───────── In-process tests ─────────


def test_no_args_prints_help_and_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli_main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "rote" in out
    assert "graduate" in out  # subcommand should be in help
    assert "emit" in out


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    from rote import __version__

    with pytest.raises(SystemExit) as excinfo:
        cli_main(["--version"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert __version__ in out


def test_emit_missing_pipeline_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli_main(
        [
            "emit",
            str(tmp_path / "does-not-exist.yaml"),
            "--runtime",
            "temporal",
            "--out",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "not found" in err


def test_emit_bdr_pipeline_in_process(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Run `rote emit` against the real BDR pipeline.yaml.

    Verifies: the CLI loads the pipeline, dispatches to the temporal
    adapter, writes the expected files, and the emitted files parse as
    valid Python.
    """
    out_dir = tmp_path / "emitted"

    rc = cli_main(
        [
            "emit",
            str(BDR_PIPELINE_YAML),
            "--runtime",
            "temporal",
            "--out",
            str(out_dir),
        ]
    )
    assert rc == 0

    captured = capsys.readouterr().out
    assert "bdr-campaign" in captured
    assert str(out_dir) in captured

    # Files exist
    activities_path = out_dir / "activities.py"
    workflow_path = out_dir / "workflow.py"
    init_path = out_dir / "__init__.py"
    assert activities_path.exists()
    assert workflow_path.exists()
    assert init_path.exists()

    # And parse as valid Python
    ast.parse(activities_path.read_text(encoding="utf-8"))
    ast.parse(workflow_path.read_text(encoding="utf-8"))


def test_emit_rejects_unknown_runtime(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # argparse rejects unknown choices with exit code 2
    with pytest.raises(SystemExit) as excinfo:
        cli_main(
            [
                "emit",
                str(BDR_PIPELINE_YAML),
                "--runtime",
                "nonexistent",
                "--out",
                str(tmp_path / "out"),
            ]
        )
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "nonexistent" in err or "invalid choice" in err.lower()


def test_graduate_rejects_nonexistent_skill_dir(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = cli_main(
        [
            "graduate",
            str(tmp_path / "nonexistent"),
            "--runtime",
            "temporal",
            "--out",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "not a directory" in err


def test_graduate_happy_path_with_mocked_graduator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI should run the graduator, emit the adapter output, and
    print a clean summary. We mock ``Graduator`` so the real agent
    loop never runs.
    """
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: t\n---\n")

    out_dir = tmp_path / "out"

    # The fake graduator will be called by the CLI; have it write the
    # BDR pipeline.yaml as a stand-in (we know it's valid because the
    # IR tests prove it loads). Then the real adapter runs against
    # that valid IR and emits real Temporal code.
    from rote.graduator import GraduationResult
    from rote.ir import load_pipeline

    real_pipeline = load_pipeline(BDR_PIPELINE_YAML)

    class _MockGraduator:
        def __init__(self, agent: str | None = None, **kwargs: object) -> None:
            self.agent = agent

        async def graduate(self, skill_path, output_dir, update=False):  # noqa: ANN001
            # Write a placeholder pipeline.yaml in the output dir so
            # downstream debugging is possible (mimics what a real run
            # produces). Use a copy of the BDR yaml for realism.
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "pipeline.yaml").write_text(
                BDR_PIPELINE_YAML.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            return GraduationResult(
                pipeline=real_pipeline,
                output_dir=output_dir,
                driver_name="mock",
                driver_metadata={"tokens": 1234, "iterations": 7},
            )

    monkeypatch.setattr("rote.cli.Graduator", _MockGraduator)

    rc = cli_main(
        [
            "graduate",
            str(skill_dir),
            "--runtime",
            "temporal",
            "--out",
            str(out_dir),
        ]
    )
    assert rc == 0

    captured = capsys.readouterr().out
    assert "bdr-campaign" in captured
    assert "mock" in captured  # driver name
    assert "tokens=1234" in captured

    # Graduator output (just the pipeline.yaml the mock wrote)
    assert (out_dir / "graduated" / "pipeline.yaml").exists()

    # Emitted Temporal runtime code
    assert (out_dir / "runtime" / "temporal" / "workflow.py").exists()
    assert (out_dir / "runtime" / "temporal" / "activities.py").exists()


def test_graduate_surfaces_graduator_error_with_exit_1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A GraduatorError from the orchestrator should print a clean
    error and exit 1, not crash."""
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: t\n---\n")

    class _ExplodingGraduator:
        def __init__(self, agent: str | None = None, **kwargs: object) -> None:
            pass

        async def graduate(self, skill_path, output_dir, update=False):  # noqa: ANN001
            raise GraduatorError("simulated failure: no agent driver available")

    from rote.cli import GraduatorError  # re-imported here for typing visibility

    monkeypatch.setattr("rote.cli.Graduator", _ExplodingGraduator)

    rc = cli_main(
        [
            "graduate",
            str(skill_dir),
            "--runtime",
            "temporal",
            "--out",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "simulated failure" in err


# ───────── Subprocess tests ─────────


@pytest.fixture(scope="module")
def python_executable() -> str:
    return sys.executable


def test_subprocess_python_m_rote_cli(python_executable: str) -> None:
    """Invoke `python -m rote.cli --version` and verify it runs.

    This catches import-time errors, missing entry points, and other
    packaging issues that in-process tests miss.
    """
    result = subprocess.run(
        [python_executable, "-m", "rote.cli", "--version"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"exit={result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "rote" in result.stdout


def test_subprocess_emit_bdr(python_executable: str, tmp_path: Path) -> None:
    """End-to-end subprocess test: `python -m rote.cli emit ...` on BDR."""
    out_dir = tmp_path / "emitted"
    result = subprocess.run(
        [
            python_executable,
            "-m",
            "rote.cli",
            "emit",
            str(BDR_PIPELINE_YAML),
            "--runtime",
            "temporal",
            "--out",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"exit={result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    assert (out_dir / "workflow.py").exists()
    assert (out_dir / "activities.py").exists()
