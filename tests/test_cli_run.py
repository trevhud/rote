"""CLI-level tests for ``rote run`` (dispatch, exit codes, output shape).

The execution primitives are faked at the runners layer — these tests
prove the command wiring: detection errors exit 2, run failures exit 1,
stdout carries only the output JSON (or the ``--json`` object), and gate
payloads reach the pipeline runner.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from rote.cli import main as cli_main
from rote.eval.empirical import MeasuredRun

REPO_ROOT = Path(__file__).resolve().parent.parent
BDR_PIPELINE_YAML = REPO_ROOT / "examples" / "bdr-outreach" / "expected" / "pipeline.yaml"


def _emitted_dbos_dir(tmp_path: Path) -> Path:
    app = tmp_path / "app"
    app.mkdir()
    (app / "main.py").write_text("", encoding="utf-8")
    (app / "dbos-config.yaml").write_text("", encoding="utf-8")
    return app


def _graduate_out_with_bdr(tmp_path: Path) -> Path:
    out = tmp_path / "out"
    (out / "graduated").mkdir(parents=True)
    shutil.copy(BDR_PIPELINE_YAML, out / "graduated" / "pipeline.yaml")
    runtime = out / "runtime" / "dbos"
    runtime.mkdir(parents=True)
    (runtime / "main.py").write_text("", encoding="utf-8")
    (runtime / "dbos-config.yaml").write_text("", encoding="utf-8")
    return out


def test_run_unrecognizable_path_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "junk").mkdir()
    rc = cli_main(["run", str(tmp_path / "junk")])
    assert rc == 2
    assert "neither a skill" in capsys.readouterr().err


def test_run_temporal_dir_dispatches_to_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    app = tmp_path / "app"
    app.mkdir()
    (app / "workflow.py").write_text("", encoding="utf-8")
    (app / "activities.py").write_text("", encoding="utf-8")
    import rote.runners.temporal as t_runner
    from rote.eval.empirical import MeasuredRun as MR

    monkeypatch.setattr(
        t_runner, "run_temporal", lambda *a, **k: MR(wall_seconds=0.1, output={"t": 1})
    )
    rc = cli_main(["run", str(app), "--input", "{}"])
    captured = capsys.readouterr()
    assert rc == 0
    assert json.loads(captured.out) == {"t": 1}
    assert "rote run (temporal): succeeded" in captured.err


def test_run_pipeline_success_prints_output_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    app = _emitted_dbos_dir(tmp_path)

    def fake_trial(app_dir: Any, payload: Any, **kwargs: Any) -> MeasuredRun:
        assert payload == {"q": 7}
        return MeasuredRun(wall_seconds=0.3, output={"answer": 42})

    monkeypatch.setattr("rote.runners.run_pipeline_trial", fake_trial)
    rc = cli_main(["run", str(app), "--input", '{"q": 7}'])
    captured = capsys.readouterr()
    assert rc == 0
    assert json.loads(captured.out) == {"answer": 42}
    assert "rote run (dbos): succeeded" in captured.err


def test_run_pipeline_failure_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    app = _emitted_dbos_dir(tmp_path)
    monkeypatch.setattr(
        "rote.runners.run_pipeline_trial",
        lambda *a, **k: MeasuredRun(wall_seconds=0.1, output=None, error="boom"),
    )
    rc = cli_main(["run", str(app), "--input", "{}"])
    captured = capsys.readouterr()
    assert rc == 1
    assert "failed: boom" in captured.err
    assert captured.out == ""


def test_run_gated_pipeline_requires_signals_non_interactive(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = _graduate_out_with_bdr(tmp_path)
    rc = cli_main(["run", str(out), "--input", "{}"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "HITL gate" in err
    assert "contact_review_approved" in err
    assert "bdr_enrollment_complete" in err


def test_run_gated_pipeline_delivers_signals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    out = _graduate_out_with_bdr(tmp_path)
    captured: dict[str, Any] = {}

    def fake_trial(app_dir: Any, payload: Any, **kwargs: Any) -> MeasuredRun:
        captured["signals"] = kwargs.get("signals")
        return MeasuredRun(wall_seconds=0.2, output={"ok": True})

    monkeypatch.setattr("rote.runners.run_pipeline_trial", fake_trial)
    rc = cli_main(
        [
            "run",
            str(out),
            "--input",
            "{}",
            "--signal",
            'contact_review_approved={"approved": true}',
            "--signal",
            "bdr_enrollment_complete={}",
        ]
    )
    assert rc == 0
    assert captured["signals"] == {
        "contact_review_approved": {"approved": True},
        "bdr_enrollment_complete": {},
    }
    assert json.loads(capsys.readouterr().out) == {"ok": True}


def test_run_unknown_signal_name_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = _graduate_out_with_bdr(tmp_path)
    rc = cli_main(["run", str(out), "--input", "{}", "--signal", "nope={}"])
    assert rc == 2
    assert "not in the pipeline's gates" in capsys.readouterr().err


def test_run_pipeline_json_flag_wraps_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    app = _emitted_dbos_dir(tmp_path)
    monkeypatch.setattr(
        "rote.runners.run_pipeline_trial",
        lambda *a, **k: MeasuredRun(
            wall_seconds=0.4,
            output={"answer": 1},
            judge_usage=({"model": "m", "input_tokens": 10, "output_tokens": 5},),
        ),
    )
    rc = cli_main(["run", str(app), "--input", "{}", "--json"])
    captured = capsys.readouterr()
    assert rc == 0
    payload = json.loads(captured.out)
    assert payload["kind"] == "pipeline"
    assert payload["runtime"] == "dbos"
    assert payload["output"] == {"answer": 1}
    assert payload["judge_usage"][0]["input_tokens"] == 10
    assert "judge usage: 1 call(s), 10 in / 5 out tokens" in captured.err


def test_run_skill_dispatch_and_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("# a skill", encoding="utf-8")

    from rote.eval.baseline import ObservedToolCall
    from rote.runners import SkillRunOutcome

    def fake_run_skill(skill_dir: Any, payload: Any, **kwargs: Any) -> SkillRunOutcome:
        assert payload == {"task": "t"}
        assert kwargs["allow_writes"] is True
        return SkillRunOutcome(
            run=MeasuredRun(
                wall_seconds=12.0, output={"done": True}, turns=5, cost_usd=0.42, model="m"
            ),
            observations=(ObservedToolCall(server="srv", tool="lookup", input={}),),
            servers_wired=("srv",),
            servers_skipped={},
            read_only=False,
        )

    monkeypatch.setattr("rote.runners.run_skill", fake_run_skill)
    rc = cli_main(["run", str(skill), "--input", '{"task": "t"}', "--allow-writes", "--json"])
    captured = capsys.readouterr()
    assert rc == 0
    payload = json.loads(captured.out)
    assert payload["kind"] == "skill"
    assert payload["output"] == {"done": True}
    assert payload["observed_servers"] == ["srv"]
    assert "writes allowed" in captured.err
    assert "observed MCP traffic: 1 call(s) across srv" in captured.err


def test_run_pipeline_without_input_and_without_skill_exits_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    app = _emitted_dbos_dir(tmp_path)
    rc = cli_main(["run", str(app)])
    assert rc == 2
    assert "no source skill to derive" in capsys.readouterr().err


def test_run_help_mentions_both_sides(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli_main(["run", "--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "claude -p" in out
    assert "--signal" in out
    assert "--allow-writes" in out
