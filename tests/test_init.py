"""Tests for the ``rote init`` wizard (scripted I/O — no TTY needed)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rote.cli import main as cli_main
from rote.config import load_config_file
from rote.init_wizard import WizardAborted, run_wizard

_FAKE_DRIVERS = [
    ("claude", True, ""),
    ("codex", False, "codex CLI not found"),
    ("api", False, "ANTHROPIC_API_KEY not set"),
]


@pytest.fixture(autouse=True)
def _quiet_doctor(monkeypatch: pytest.MonkeyPatch) -> None:
    """The wizard opens with a doctor preflight; keep it canned so these
    tests don't probe the developer's real PATH for claude/codex CLIs."""
    monkeypatch.setattr(
        "rote.cli._build_doctor_report",
        lambda: {"version": "0", "python": "3", "drivers": [], "runtimes": [], "ok": True},
    )
    monkeypatch.setattr("rote.graduator.drivers.available_drivers", lambda: _FAKE_DRIVERS)


class _ScriptedIO:
    """Feeds canned answers to the wizard and records everything echoed."""

    def __init__(self, answers: list[str]) -> None:
        self.answers = answers
        self.echoed: list[str] = []

    def input_fn(self, prompt: str) -> str:
        self.echoed.append(prompt)
        return self.answers.pop(0)

    def echo(self, line: str) -> None:
        self.echoed.append(line)


def _run(answers: list[str], *, project: bool = False, cwd: Path | None = None) -> _ScriptedIO:
    io = _ScriptedIO(answers)
    run_wizard(project=project, input_fn=io.input_fn, echo=io.echo, cwd=cwd)
    return io


def test_cloud_path_already_logged_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from rote.cloud_auth import CloudCredential, save_credential

    save_credential(CloudCredential(url="http://p", token="rote_k", user="t@x"))
    config_path = tmp_path / "cfg" / "config.yaml"
    monkeypatch.setenv("ROTE_CONFIG_PATH", str(config_path))

    # hosting=1 (cloud), driver=default (auto), model=blank
    io = _run(["1", "", ""])
    assert load_config_file(config_path) == {"runtime": "cloudflare", "deploy": "rote-cloud"}
    assert any("already logged in as t@x" in line for line in io.echoed)


def test_cloud_path_offers_login_when_logged_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "cfg" / "config.yaml"
    monkeypatch.setenv("ROTE_CONFIG_PATH", str(config_path))
    called = []
    monkeypatch.setattr("rote.cloud_auth.login", lambda *a, **k: called.append(True))

    # hosting=1, login=y, driver=auto, model=blank
    _run(["1", "y", "", ""])
    assert called == [True]
    assert load_config_file(config_path) == {"runtime": "cloudflare", "deploy": "rote-cloud"}


def test_local_path_picks_runtime_driver_and_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "cfg" / "config.yaml"
    monkeypatch.setenv("ROTE_CONFIG_PATH", str(config_path))

    # hosting=2 (local), runtime=2 (temporal), driver=1 (claude), model=m-9
    _run(["2", "2", "1", "m-9"])
    assert load_config_file(config_path) == {
        "runtime": "temporal",
        "deploy": "none",
        "agent": "claude",
        "model": "m-9",
    }


def test_project_flag_writes_rote_yaml_in_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run(["2", "1", "", ""], project=True, cwd=tmp_path)
    assert load_config_file(tmp_path / "rote.yaml") == {"runtime": "dbos", "deploy": "none"}


def test_declined_overwrite_aborts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "cfg" / "config.yaml"
    config_path.parent.mkdir()
    config_path.write_text("runtime: inngest\n", encoding="utf-8")
    monkeypatch.setenv("ROTE_CONFIG_PATH", str(config_path))

    with pytest.raises(WizardAborted, match="kept existing"):
        _run(["2", "1", "", "", "n"])  # last answer declines the overwrite
    assert load_config_file(config_path) == {"runtime": "inngest"}  # untouched


def test_invalid_answers_reprompt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "cfg" / "config.yaml"
    monkeypatch.setenv("ROTE_CONFIG_PATH", str(config_path))

    # "9" is not a hosting choice — the wizard asks again instead of crashing.
    _run(["9", "2", "1", "", ""])
    assert load_config_file(config_path)["runtime"] == "dbos"


def test_cli_init_requires_a_tty(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    rc = cli_main(["init"])
    assert rc == 2
    assert "interactive" in capsys.readouterr().err


def test_cli_init_dispatches_to_wizard(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    ran = []
    monkeypatch.setattr("rote.init_wizard.run_wizard", lambda **kwargs: ran.append(kwargs))
    rc = cli_main(["init", "--project"])
    assert rc == 0
    assert ran == [{"project": True}]


def test_cli_init_eof_aborts_cleanly(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Ctrl-D mid-wizard must abort with a message, not a traceback."""
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", _raise_eof)
    rc = cli_main(["init"])
    assert rc == 1
    assert "nothing was written" in capsys.readouterr().err


def _raise_eof(prompt: str = "") -> str:
    raise EOFError
