"""Tests for the layered config (``rote.config``), ``rote config``, and
the compile/emit default wiring.

The conftest autouse fixture points ``ROTE_CONFIG_PATH`` and
``ROTE_PROJECT_CONFIG_PATH`` at nonexistent files, so each test opts
into exactly the layers it wants by writing those paths (or unsetting
the project override to exercise real discovery).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rote import config as rote_config
from rote.cli import main as cli_main
from rote.config import (
    ConfigError,
    load_config_file,
    load_layers,
    project_config_path,
    resolve,
    user_config_path,
    write_config,
)

# ───────── Paths and discovery ─────────


def test_user_config_path_honors_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROTE_CONFIG_PATH", "/x/y/config.yaml")
    assert user_config_path() == Path("/x/y/config.yaml")


def test_user_config_path_defaults_to_xdg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ROTE_CONFIG_PATH", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", "/xdg")
    assert user_config_path() == Path("/xdg/rote/config.yaml")


def test_project_discovery_walks_up_and_stops_at_git_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ROTE_PROJECT_CONFIG_PATH", raising=False)
    repo = tmp_path / "repo"
    nested = repo / "a" / "b"
    nested.mkdir(parents=True)
    (repo / ".git").mkdir()
    # No rote.yaml anywhere → None, and the .git boundary stops the walk
    # before it could ever see one above the repo.
    (tmp_path / "rote.yaml").write_text("runtime: temporal\n", encoding="utf-8")
    assert project_config_path(nested) is None
    # A rote.yaml at the repo root is found from a nested cwd.
    (repo / "rote.yaml").write_text("runtime: temporal\n", encoding="utf-8")
    assert project_config_path(nested) == repo / "rote.yaml"


def test_project_env_override_beats_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    explicit = tmp_path / "elsewhere" / "rote.yaml"
    explicit.parent.mkdir()
    explicit.write_text("runtime: inngest\n", encoding="utf-8")
    monkeypatch.setenv("ROTE_PROJECT_CONFIG_PATH", str(explicit))
    assert project_config_path(tmp_path) == explicit


# ───────── File validation (strict by design) ─────────


def test_unknown_key_is_a_loud_error(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("runtme: dbos\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown setting `runtme`"):
        load_config_file(path)


def test_invalid_choice_is_a_loud_error(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("runtime: clouflare\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="expected one of"):
        load_config_file(path)


def test_non_mapping_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("- dbos\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="expected a mapping"):
        load_config_file(path)


def test_empty_file_is_fine(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("", encoding="utf-8")
    assert load_config_file(path) == {}


@pytest.mark.parametrize("blank", ["", '""', "'  '", "\n"])
def test_a_blank_value_is_rejected_for_a_free_form_key(tmp_path: Path, blank: str) -> None:
    """`model` takes any string, so the enum check cannot catch a blank.

    A key with `valid_choices()` rejects "" incidentally (it isn't in
    the choices), which is why dropping the emptiness check survived a
    mutation sweep — `model` is the only key where it is load-bearing,
    and a blank one would reach the driver as a model name.
    """
    path = tmp_path / "config.yaml"
    path.write_text(f"model: {blank}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="must be a non-empty string"):
        load_config_file(path)


def test_write_config_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    write_config(path, {"runtime": "temporal", "deploy": "none", "agent": "codex"})
    assert load_config_file(path) == {"runtime": "temporal", "deploy": "none", "agent": "codex"}


def test_write_config_validates(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        write_config(tmp_path / "c.yaml", {"runtime": "not-a-runtime"})
    with pytest.raises(ConfigError):
        write_config(tmp_path / "c.yaml", {"nope": "x"})


# ───────── Precedence ─────────


def _layers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    user: str | None = None,
    project: str | None = None,
) -> rote_config.ConfigLayers:
    user_path = tmp_path / "user" / "config.yaml"
    monkeypatch.setenv("ROTE_CONFIG_PATH", str(user_path))
    if user is not None:
        user_path.parent.mkdir(exist_ok=True)
        user_path.write_text(user, encoding="utf-8")
    project_path = tmp_path / "proj" / "rote.yaml"
    monkeypatch.setenv("ROTE_PROJECT_CONFIG_PATH", str(project_path))
    if project is not None:
        project_path.parent.mkdir(exist_ok=True)
        project_path.write_text(project, encoding="utf-8")
    return load_layers()


def test_resolution_precedence_flag_env_project_user(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layers = _layers(monkeypatch, tmp_path, user="runtime: dbos\n", project="runtime: temporal\n")
    assert resolve("runtime", None, layers=layers).value == "temporal"  # project > user
    monkeypatch.setenv("ROTE_RUNTIME", "inngest")
    assert resolve("runtime", None, layers=layers).value == "inngest"  # env > project
    assert resolve("runtime", "python", layers=layers).value == "python"  # flag > env
    # Sources name the winning layer.
    assert resolve("runtime", "python", layers=layers).source == "flag"
    monkeypatch.delenv("ROTE_RUNTIME")
    assert resolve("runtime", None, layers=layers).source.startswith("project ")


def test_env_values_are_validated_loudly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layers = _layers(monkeypatch, tmp_path)
    monkeypatch.setenv("ROTE_DEPLOY", "sideways")
    with pytest.raises(ConfigError, match="ROTE_DEPLOY"):
        resolve("deploy", None, layers=layers)


def test_unset_everywhere_resolves_to_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layers = _layers(monkeypatch, tmp_path)
    resolved = resolve("model", None, layers=layers)
    assert resolved.value is None
    assert resolved.source == "built-in default"


# ───────── `rote config` CLI ─────────


def test_config_shows_sources_and_login_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _layers(monkeypatch, tmp_path, user="runtime: temporal\ndeploy: none\n")
    rc = cli_main(["config"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "temporal" in out
    assert "user " in out  # the winning layer is named
    assert "logged out" in out


def test_config_json_reflects_login_derived_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from rote.cloud_auth import CloudCredential, save_credential

    _layers(monkeypatch, tmp_path)
    save_credential(CloudCredential(url="http://p", token="rote_k", user="t@x"))
    rc = cli_main(["config", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["settings"]["runtime"] == {
        "value": "cloudflare",
        "source": "built-in default (logged in)",
    }
    assert payload["settings"]["deploy"]["value"] == "rote-cloud"
    assert payload["cloud"] == {"logged_in": True, "url": "http://p", "user": "t@x"}


def test_config_bad_file_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "config.yaml"
    bad.write_text("runtime: 7\n", encoding="utf-8")
    monkeypatch.setenv("ROTE_CONFIG_PATH", str(bad))
    rc = cli_main(["config"])
    assert rc == 2
    assert "must be a non-empty string" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["config", "emit", "analyze", "compile"])
def test_every_config_reading_command_exits_2_on_a_bad_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
) -> None:
    """Strictness is a property of the config layer, not of one command.

    `rote config` was the only command asserting exit 2; flipping
    compile's handler to `return 0` — reporting success while doing
    nothing — survived a mutation sweep. Every command that calls
    `load_layers()` owns this contract, so cover them together and let a
    new one fail here rather than shipping a silent fallback.
    """
    from tests.conftest import BDR_PIPELINE_YAML

    bad = tmp_path / "config.yaml"
    bad.write_text("runtime: clouflare\n", encoding="utf-8")
    monkeypatch.setenv("ROTE_CONFIG_PATH", str(bad))

    skill_dir = _skill(tmp_path)
    argv = {
        "config": ["config"],
        "emit": ["emit", str(BDR_PIPELINE_YAML), "--out", str(tmp_path / "out")],
        "analyze": ["analyze", str(skill_dir)],
        "compile": ["compile", str(skill_dir), "--out", str(tmp_path / "out"), "--no-eval"],
    }[command]

    assert cli_main(argv) == 2
    err = capsys.readouterr().err
    assert "expected one of" in err and "clouflare" in err
    # The run stopped at the config, before any work: nothing emitted.
    assert not (tmp_path / "out").exists()


# ───────── compile/emit honor the config ─────────


def _skill(tmp_path: Path) -> Path:
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: t\n---\n")
    return skill_dir


def test_compile_config_runtime_beats_login_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Logged in (built-in default would be cloudflare) but the user
    config pins temporal — the config wins, and no upload happens."""
    from tests.test_cli import _cloud_logged_in, _fake_cloud_deploy, _install_mock_compiler

    _layers(monkeypatch, tmp_path, user="runtime: temporal\n")
    _install_mock_compiler(monkeypatch)
    _cloud_logged_in()
    calls = _fake_cloud_deploy(monkeypatch)

    # The config pins temporal (not cloudflare), which is itself a local
    # opt-out — the logged-in cloud default steps aside, no --local needed.
    out_dir = tmp_path / "out"
    rc = cli_main(["compile", str(_skill(tmp_path)), "--out", str(out_dir), "--no-eval"])
    assert rc == 0
    assert (out_dir / "runtime" / "temporal" / "workflow.py").exists()
    assert calls == []


def test_compile_flag_beats_config_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.test_cli import _install_mock_compiler

    _layers(monkeypatch, tmp_path, user="runtime: temporal\n")
    _install_mock_compiler(monkeypatch)

    out_dir = tmp_path / "out"
    rc = cli_main(
        ["compile", str(_skill(tmp_path)), "--out", str(out_dir), "--no-eval", "--runtime", "dbos"]
    )
    assert rc == 0
    assert (out_dir / "runtime" / "dbos" / "main.py").exists()


def test_compile_config_deploy_none_suppresses_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.test_cli import _cloud_logged_in, _fake_cloud_deploy, _install_mock_compiler

    _layers(monkeypatch, tmp_path, user="deploy: none\n")
    _install_mock_compiler(monkeypatch)
    _cloud_logged_in()
    calls = _fake_cloud_deploy(monkeypatch)

    # deploy: none is a local opt-out — the logged-in cloud default steps
    # aside (no --local needed), and no upload happens.
    rc = cli_main(["compile", str(_skill(tmp_path)), "--out", str(tmp_path / "out"), "--no-eval"])
    assert rc == 0
    # Still emits cloudflare (the logged-in runtime default) — just no upload.
    assert (tmp_path / "out" / "runtime" / "cloudflare").is_dir()
    assert calls == []


def test_compile_configured_cloud_deploy_fails_fast_when_logged_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`deploy: rote-cloud` + logged out must error BEFORE the agent
    spends money, with the fix in the message."""
    from tests.test_cli import _install_mock_compiler

    _layers(monkeypatch, tmp_path, user="runtime: cloudflare\ndeploy: rote-cloud\n")
    _install_mock_compiler(monkeypatch)

    out_dir = tmp_path / "out"
    rc = cli_main(["compile", str(_skill(tmp_path)), "--out", str(out_dir), "--no-eval"])
    assert rc == 2
    assert "rote login" in capsys.readouterr().err
    assert not out_dir.exists()  # nothing ran, nothing was written


def test_compile_flag_runtime_downgrades_configured_deploy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """--runtime temporal for one run beats `deploy: rote-cloud` — warn
    and skip the upload rather than blocking the run."""
    from tests.test_cli import _cloud_logged_in, _fake_cloud_deploy, _install_mock_compiler

    _layers(monkeypatch, tmp_path, user="deploy: rote-cloud\n")
    _install_mock_compiler(monkeypatch)
    _cloud_logged_in()
    calls = _fake_cloud_deploy(monkeypatch)

    rc = cli_main(
        [
            "compile",
            str(_skill(tmp_path)),
            "--out",
            str(tmp_path / "out"),
            "--no-eval",
            "--runtime",
            "temporal",
        ]
    )
    assert rc == 0
    assert calls == []
    assert "skipping the configured rote-cloud deploy" in capsys.readouterr().err


def test_compile_contradictory_config_is_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """runtime and deploy both from config, and incompatible → the config
    is self-contradictory; fixing the file beats guessing an intent."""
    from tests.test_cli import _cloud_logged_in, _install_mock_compiler

    _layers(monkeypatch, tmp_path, user="runtime: temporal\ndeploy: rote-cloud\n")
    _install_mock_compiler(monkeypatch)
    _cloud_logged_in()

    # config runtime:temporal is a local opt-out, so the contradiction with
    # deploy:rote-cloud surfaces on the local path — no --local needed.
    rc = cli_main(["compile", str(_skill(tmp_path)), "--out", str(tmp_path / "out"), "--no-eval"])
    assert rc == 2
    assert "requires the cloudflare runtime" in capsys.readouterr().err


def test_compile_config_agent_reaches_compiler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.test_cli import _install_mock_compiler

    _layers(monkeypatch, tmp_path, user="agent: codex\nmodel: m-1\n")
    _install_mock_compiler(monkeypatch)
    seen: dict[str, object] = {}

    import rote.cli as cli_mod

    real = cli_mod.Compiler

    def capture(*, agent=None, model=None, **kwargs):  # noqa: ANN001, ANN202
        seen["agent"], seen["model"] = agent, model
        return real(agent=agent, model=model, **kwargs)

    monkeypatch.setattr(cli_mod, "Compiler", capture)
    rc = cli_main(
        [
            "compile",
            str(_skill(tmp_path)),
            "--out",
            str(tmp_path / "out"),
            "--no-eval",
            "--runtime",
            "dbos",
        ]
    )
    assert rc == 0
    assert seen == {"agent": "codex", "model": "m-1"}


def test_emit_honors_config_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.conftest import BDR_PIPELINE_YAML

    _layers(monkeypatch, tmp_path, user="runtime: temporal\n")
    out_dir = tmp_path / "emitted"
    rc = cli_main(["emit", str(BDR_PIPELINE_YAML), "--out", str(out_dir)])
    assert rc == 0
    assert (out_dir / "workflow.py").exists()  # temporal shape, not dbos main.py
    assert not (out_dir / "main.py").exists()


# ───────── inference provider selection ─────────


def test_inference_resolves_through_the_same_layers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Who pays for judges is a configurable default like any other — the
    point of the key is that a user sets it once instead of exporting an
    env var before every run."""
    layers = _layers(monkeypatch, tmp_path, user="inference: api\n")
    assert resolve("inference", None, layers=layers).value == "api"
    monkeypatch.setenv("ROTE_INFERENCE", "rote-cloud")
    assert resolve("inference", None, layers=layers).value == "rote-cloud"
    assert resolve("inference", "claude-cli", layers=layers).value == "claude-cli"


def test_inference_rejects_a_provider_that_does_not_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ConfigError, match="not valid"):
        _layers(monkeypatch, tmp_path, user="inference: free-lunch\n")


def test_config_reports_the_inference_lane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _layers(monkeypatch, tmp_path, user="inference: rote-cloud\n")
    rc = cli_main(["config", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["settings"]["inference"]["value"] == "rote-cloud"


def test_unset_inference_reports_auto_detect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _layers(monkeypatch, tmp_path)
    rc = cli_main(["config"])
    assert rc == 0
    assert "auto-detect" in capsys.readouterr().out
