"""``rote init``: scripted answers through the Prompter, and the CLI's modes."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from rote.cli import main as cli_main
from rote.config import load_config_file
from rote.init_wizard import InitError, InitFlags, build_plan, decider_credentials_note
from rote.interact.mode import Mode
from rote.interact.output import Output
from rote.interact.plan import apply_plan
from rote.interact.prompter import DEFAULT, Answer, ScriptedPrompter

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
    monkeypatch.setattr("rote.compiler.drivers.available_drivers", lambda: _FAKE_DRIVERS)
    for var in ("ROTE_CLEF_ACCOUNT_ID", "ROTE_CLEF_API_TOKEN", "TYPESAFE_API_KEY"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def config_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "cfg" / "config.yaml"
    monkeypatch.setenv("ROTE_CONFIG_PATH", str(path))
    return path


def _out() -> tuple[Output, io.StringIO]:
    err = io.StringIO()
    mode = Mode.resolve(env={}, stdin=io.StringIO(), stdout=io.StringIO(), stderr=err)
    return Output(mode, stdout=io.StringIO(), stderr=err), err


def _local_answers(**overrides: object) -> list[Answer]:
    """Every question of the local path, defaults unless overridden."""
    order = [
        "--deploy",
        "--runtime",
        "--agent",
        "--model",
        "--structure",
        "--decider",
        "--tiebreak",
    ]
    kinds = {"--model": "text"}
    values: dict[str, object] = {"--deploy": "none", **overrides}
    return [
        Answer(kinds.get(flag, "select"), flag, values.get(flag, DEFAULT))  # type: ignore[arg-type]
        for flag in order
    ]


_NO_FLAGS = InitFlags()


def _run(answers: list[Answer], flags: InitFlags = _NO_FLAGS, **kw: object) -> ScriptedPrompter:
    prompter = ScriptedPrompter(answers)
    out, _ = _out()
    plan = build_plan(flags, prompter=prompter, out=out, project=False, **kw)  # type: ignore[arg-type]
    assert not prompter.answers, f"unused answers: {prompter.answers}"
    assert [o.status for o in apply_plan(plan)] == ["done"] * len(plan.actions)
    return prompter


def test_local_defaults_write_only_runtime_and_deploy(config_path: Path) -> None:
    _run(_local_answers())
    # auto agent, blank model, haiku structure, auto decider, no tiebreak:
    # all are the built-in defaults, so none is written.
    assert load_config_file(config_path) == {"runtime": "dbos", "deploy": "none"}


def test_local_path_picks_every_setting(config_path: Path) -> None:
    _run(
        _local_answers(
            **{
                "--runtime": "temporal",
                "--agent": "claude",
                "--model": "m-9",
                "--structure": "sonnet",
                "--decider": "jev",
                "--tiebreak": "haiku",
            }
        )
    )
    assert load_config_file(config_path) == {
        "runtime": "temporal",
        "deploy": "none",
        "agent": "claude",
        "model": "m-9",
        "structure": "sonnet",
        "decider": "jev",
        "tiebreak": "haiku",
    }


def test_decider_auto_is_the_default_and_rules_is_written(config_path: Path) -> None:
    prompter = _run(_local_answers(**{"--decider": "rules"}))
    assert load_config_file(config_path)["decider"] == "rules"
    decider_q = [msg for kind, flag, msg in prompter.asked if flag == "--decider"]
    assert decider_q == ["Which decision model double-checks it?"]


@pytest.mark.parametrize(
    ("decider", "named", "not_named"),
    [
        ("clef", ["ROTE_CLEF_ACCOUNT_ID", "ROTE_CLEF_API_TOKEN"], ["TYPESAFE_API_KEY"]),
        ("jev", ["TYPESAFE_API_KEY"], ["ROTE_CLEF_ACCOUNT_ID"]),
        ("auto", ["ROTE_CLEF_ACCOUNT_ID", "ROTE_CLEF_API_TOKEN", "TYPESAFE_API_KEY"], []),
    ],
)
def test_decider_note_names_the_variables_compile_reads(
    decider: str, named: list[str], not_named: list[str]
) -> None:
    note = decider_credentials_note(decider)
    for var in named:
        assert var in note
    for var in not_named:
        assert var not in note
    # Deploy credentials must never opt a compile into inference (CLAUDE.md).
    assert "CLOUDFLARE_" not in note


def test_the_decider_menu_never_names_cloudflare_credentials(config_path: Path) -> None:
    from rote.init_wizard import _decider_menu

    text = " ".join(pitch for _, pitch in _decider_menu())
    assert "CLOUDFLARE_" not in text
    assert "ROTE_CLEF_ACCOUNT_ID" in text and "TYPESAFE_API_KEY" in text


def test_auto_note_reports_the_checker_this_environment_would_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    note = decider_credentials_note("auto")
    assert "Jev" in note
    monkeypatch.delenv("TYPESAFE_API_KEY")
    assert "Jev" not in decider_credentials_note("auto").split(":", 1)[0]


def test_cloud_path_already_logged_in_asks_no_runtime(config_path: Path) -> None:
    from rote.cloud_auth import CloudCredential, save_credential

    save_credential(CloudCredential(url="http://p", token="rote_k", user="t@x"))
    answers = [a for a in _local_answers(**{"--deploy": "rote-cloud"}) if a.flag != "--runtime"]
    prompter = ScriptedPrompter(answers)
    out, _ = _out()
    plan = build_plan(InitFlags(), prompter=prompter, out=out, project=False)
    assert [a.id for a in plan.actions] == ["write-config"]
    assert any("already logged in to rote cloud as t@x" in n for n in plan.notes)
    apply_plan(plan)
    assert load_config_file(config_path) == {"runtime": "cloudflare", "deploy": "rote-cloud"}


def test_cloud_login_is_a_planned_action_not_a_side_effect_of_asking(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called: list[bool] = []
    monkeypatch.setattr("rote.cloud_auth.login", lambda *a, **k: called.append(k["open_browser"]))
    answers = [a for a in _local_answers(**{"--deploy": "rote-cloud"}) if a.flag != "--runtime"]
    prompter = ScriptedPrompter([*answers, Answer("confirm", "--login", True)])
    out, _ = _out()
    plan = build_plan(InitFlags(), prompter=prompter, out=out, project=False)
    assert called == []  # answering yes logged nobody in
    login = plan.actions[1]
    assert login.id == "login-rote-cloud"
    assert [e.kind for e in login.effects] == ["contact", "launch"]
    assert login.effects[0].to_dict()["host"] == "app.roteskills.com"
    apply_plan(plan)
    assert called == [True]


def _cloud_answers() -> list[Answer]:
    return [a for a in _local_answers(**{"--deploy": "rote-cloud"}) if a.flag != "--runtime"]


def test_cloud_without_a_person_never_logs_in_unless_asked(config_path: Path) -> None:
    prompter = ScriptedPrompter(_cloud_answers(), interactive_=False)
    out, _ = _out()
    plan = build_plan(InitFlags(), prompter=prompter, out=out, project=False)
    assert [a.id for a in plan.actions] == ["write-config"]
    assert not [q for q in prompter.asked if q[1] == "--login"]
    assert any("rote login" in n for n in plan.notes)


def test_login_flag_is_refused_without_a_person(config_path: Path) -> None:
    """The device login polls until someone approves in a browser; with
    nobody at the terminal it would block forever."""
    out, _ = _out()
    with pytest.raises(InitError, match="--login needs a person at the terminal"):
        build_plan(
            InitFlags(login=True),
            prompter=ScriptedPrompter(_cloud_answers(), interactive_=False),
            out=out,
            project=False,
        )
    assert not config_path.exists()


def test_flags_answer_their_questions(config_path: Path) -> None:
    flags = InitFlags(deploy="none", runtime="inngest", agent="codex", model="", decider="clef")
    prompter = _run(
        [Answer("select", "--structure", DEFAULT), Answer("select", "--tiebreak", DEFAULT)], flags
    )
    asked = [flag for _, flag, _ in prompter.asked]
    assert asked == ["--structure", "--tiebreak"]
    assert load_config_file(config_path) == {
        "runtime": "inngest",
        "deploy": "none",
        "agent": "codex",
        "decider": "clef",
    }


def test_invalid_flag_values_are_errors_not_fallbacks() -> None:
    out, _ = _out()
    with pytest.raises(InitError, match="--runtime 'clouflare' is not valid"):
        build_plan(
            InitFlags(deploy="none", runtime="clouflare"),
            prompter=ScriptedPrompter([]),
            out=out,
            project=False,
        )
    with pytest.raises(InitError, match="runs the cloudflare runtime"):
        build_plan(
            InitFlags(deploy="rote-cloud", runtime="dbos"),
            prompter=ScriptedPrompter([]),
            out=out,
            project=False,
        )


def test_project_flag_plans_rote_yaml_in_cwd(tmp_path: Path) -> None:
    prompter = ScriptedPrompter(_local_answers())
    out, _ = _out()
    plan = build_plan(InitFlags(), prompter=prompter, out=out, project=True, cwd=tmp_path)
    write = plan.actions[0].effects[0].to_dict()
    assert write["path"] == str(tmp_path / "rote.yaml")
    assert write["replaces"] is False
    apply_plan(plan)
    assert load_config_file(tmp_path / "rote.yaml") == {"runtime": "dbos", "deploy": "none"}


def test_an_existing_config_is_planned_as_a_replacement(config_path: Path) -> None:
    config_path.parent.mkdir()
    config_path.write_text("runtime: inngest\n", encoding="utf-8")
    prompter = ScriptedPrompter(_local_answers())
    out, _ = _out()
    plan = build_plan(InitFlags(), prompter=prompter, out=out, project=False)
    assert plan.needed_flags() == ("--force",)
    assert load_config_file(config_path) == {"runtime": "inngest"}  # planning wrote nothing


# ── The CLI: modes, exit codes, the one-JSON-object rule ──


def _cli(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    rc = cli_main(argv)
    captured = capsys.readouterr()
    return rc, captured.out, captured.err


def test_cli_without_a_terminal_names_the_flag_and_writes_nothing(
    config_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc, out, err = _cli(["init", "--agent-mode", "no"], capsys)
    assert rc == 2
    assert "pass --deploy <rote-cloud|none>" in err
    assert "stdin is not a terminal" in err
    assert out == ""
    assert not config_path.exists()


def test_cli_yes_takes_the_defaults_and_applies(
    config_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc, out, err = _cli(["init", "--deploy", "none", "--yes"], capsys)
    assert rc == 0, err
    assert out == ""  # results go to stderr unless --json
    assert load_config_file(config_path) == {"runtime": "dbos", "deploy": "none"}


def test_cli_json_missing_answer_is_one_action_required_object(
    config_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc, out, _ = _cli(["init", "--deploy", "none", "--json"], capsys)
    assert rc == 2
    payload = json.loads(out)
    assert payload["status"] == "action_required"
    assert payload["reason"] == "missing_answer"
    assert payload["missing"] == ["--runtime"]
    assert payload["choices"][0] == "dbos"
    runtimes = "<dbos|temporal|cloudflare|dbos-ts|inngest|python>"
    assert payload["next"] == [{"command": f"rote init --deploy none --json --runtime {runtimes}"}]
    assert not config_path.exists()


def test_cli_replacing_needs_force_even_with_yes(
    config_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path.parent.mkdir()
    config_path.write_text("runtime: inngest\n", encoding="utf-8")
    rc, out, _ = _cli(["init", "--deploy", "none", "--yes", "--json"], capsys)
    assert rc == 2
    payload = json.loads(out)
    assert payload["reason"] == "confirmation_required"
    assert payload["missing"] == ["--force"]
    assert payload["plan"]["actions"][0]["effects"][0]["replaces"] is True
    assert load_config_file(config_path) == {"runtime": "inngest"}

    rc, out, _ = _cli(["init", "--deploy", "none", "--yes", "--force", "--json"], capsys)
    assert rc == 0
    assert json.loads(out)["outcomes"] == [{"id": "write-config", "status": "done", "error": None}]
    assert load_config_file(config_path) == {"runtime": "dbos", "deploy": "none"}


def test_cli_without_yes_stops_at_the_plan(
    config_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["init", "--deploy", "none", "--runtime", "dbos", "--agent", "auto", "--model", ""]
    argv += ["--structure", "haiku", "--decider", "auto", "--tiebreak", "none", "--json"]
    rc, out, _ = _cli(argv, capsys)
    assert rc == 2
    payload = json.loads(out)
    assert payload["reason"] == "confirmation_required"
    assert payload["missing"] == ["--yes"]
    assert not config_path.exists()


def test_piped_answers_never_reach_a_prompt(tmp_path: Path) -> None:
    """The questionary trap: prompt_toolkit reads a pipe as keystrokes, so
    "\\r\\r\\r" would answer every question with its default and write a
    config. Interactivity is decided before the first question, so the
    pipe is never read."""
    env = {
        "PATH": "/usr/bin:/bin",
        "ROTE_CONFIG_PATH": str(tmp_path / "config.yaml"),
        "ROTE_CLOUD_CRED_PATH": str(tmp_path / "cloud.json"),
        "ROTE_PROJECT_CONFIG_PATH": str(tmp_path / "none.yaml"),
        "HOME": str(tmp_path),
    }
    proc = subprocess.run(
        [sys.executable, "-m", "rote", "init", "--agent-mode", "no"],
        input="\r\r\r\r\r\r\r\r\r\ry\r",
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert proc.returncode == 2, proc.stderr
    assert "stdin is not a terminal" in proc.stderr
    assert "Input is not a terminal" not in proc.stderr  # prompt_toolkit never started
    assert not (tmp_path / "config.yaml").exists()
