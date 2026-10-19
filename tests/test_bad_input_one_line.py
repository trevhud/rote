"""A missing required input is one clean line on every Python runtime.

Temporal always reported it as one line (the workflow fails in-process
with the message). The python and dbos apps raised from inside the run
and printed the whole traceback, about 5 KB on dbos. Both now check the
input where the script reads argv, print `error: <what is missing>` and
exit 2, and `rote run` reports that line alone.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from rote.cli import main

EXPENSE = Path(__file__).resolve().parent.parent / "examples" / "expense-audit" / "expected"
MISSING = "pipeline input is missing required field(s): period, submissions"


def _emit(pipeline: Path, runtime: str, out: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["emit", str(pipeline), "--runtime", runtime, "--out", str(out), "--json"]) == 0
    capsys.readouterr()


def _run_script(app: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(app / "main.py"), "{}"],
        cwd=app,
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, **(env or {})},
    )


def test_python_script_reports_a_missing_input_in_one_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    app = tmp_path / "py"
    _emit(EXPENSE / "pipeline-draft.yaml", "python", app, capsys)
    proc = _run_script(app)
    assert proc.returncode == 2
    assert proc.stderr == f"error: {MISSING}\n"
    assert proc.stdout == ""


def test_rote_run_on_python_reports_only_that_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    app = tmp_path / "py"
    _emit(EXPENSE / "pipeline-draft.yaml", "python", app, capsys)
    (tmp_path / "empty.json").write_text("{}", encoding="utf-8")
    assert main(["run", str(app), "--input", str(tmp_path / "empty.json")]) == 1
    err = capsys.readouterr().err
    assert f"rote run (python): failed: {MISSING}\n" in err
    assert "Traceback" not in err
    assert "pipeline exited" not in err


@pytest.mark.slow
def test_dbos_script_rejects_a_missing_input_before_launching(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("dbos")
    app = tmp_path / "dbos"
    _emit(EXPENSE / "pipeline.yaml", "dbos", app, capsys)
    sqlite = tmp_path / "sys.sqlite"
    proc = _run_script(app, {"DBOS_SYSTEM_DATABASE_URL": f"sqlite:///{sqlite}"})
    assert proc.returncode == 2
    assert proc.stderr.splitlines()[-1] == f"error: {MISSING}"
    assert "Traceback" not in proc.stderr
    # Rejected before launch(): no system database was ever created.
    assert not sqlite.exists()

    (tmp_path / "empty.json").write_text(json.dumps({}), encoding="utf-8")
    review = '{"reviewer": "x", "approved_ids": [], "rejected_ids": []}'
    rc = main(
        [
            "run",
            str(app),
            "--input",
            str(tmp_path / "empty.json"),
            "--signal",
            f"controller_review={review}",
        ]
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert MISSING in err
    assert "Traceback" not in err
    assert "Initializing DBOS" not in err
