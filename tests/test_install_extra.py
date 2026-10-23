"""A runtime's missing extra is an install the user confirms, never silent.

The base install carries only the MCP client; dbos and temporal are
extras. Picking one that is missing (``rote compile --runtime``, ``rote
init``, ``rote run`` on an app) builds a plan whose one action starts the
installer :func:`rote.extras.installer` derives. These tests run that flow
for real, with ``_run_installer`` replaced by a fake that records the exact
command and makes the module importable (conftest forbids a real install).
"""

from __future__ import annotations

import importlib.util
import io
import json
import shlex
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

import rote.setup.install_extra as install_extra
from rote import __version__
from rote.cli import main as cli_main
from rote.extras import EXTRAS, installer
from rote.interact.mode import Mode
from rote.interact.output import Output
from rote.interact.prompter import Answer, ScriptedPrompter

UV = "/opt/fake/bin/uv"


class FakeInstaller:
    """Stands in for pip/uv: records each command, then the modules it
    names "land" (find_spec starts finding them) unless ``lands`` is off."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *missing: str, lands: bool = True):
        self.calls: list[tuple[str, ...]] = []
        self.missing = set(missing)
        self.lands = lands
        real_find_spec = importlib.util.find_spec

        def find_spec(name: str, *args: object) -> object:
            if name in self.missing:
                return None
            return real_find_spec(name, *args)  # type: ignore[arg-type]

        monkeypatch.setattr(importlib.util, "find_spec", find_spec)
        monkeypatch.setattr(install_extra, "_run_installer", self)

    def __call__(self, argv: Sequence[str]) -> None:
        self.calls.append(tuple(argv))
        if self.lands:
            self.missing.clear()


@pytest.fixture
def pip_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A plain venv with pip: the installer is this interpreter's pip."""
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a: object() if name == "pip" else real_find_spec(name, *a),
    )


def _pip(extra: str) -> tuple[str, ...]:
    return (sys.executable, "-m", "pip", "install", f"rote-cli[{extra}]=={__version__}")


def _dbos_app(tmp_path: Path) -> Path:
    app = tmp_path / "app"
    app.mkdir()
    (app / "main.py").write_text("", encoding="utf-8")
    (app / "dbos-config.yaml").write_text("", encoding="utf-8")
    return app


# ───────── rote run ─────────


def test_run_without_yes_shows_the_plan_and_installs_nothing(
    tmp_path: Path,
    pip_env: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake = FakeInstaller(monkeypatch, "dbos")
    app = _dbos_app(tmp_path)
    rc = cli_main(["run", str(app), "--input", "{}", "--json"])
    assert rc == 2
    assert fake.calls == []
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "action_required"
    assert payload["reason"] == "confirmation_required"
    assert payload["missing"] == ["--yes"]
    assert payload["next"] == [{"command": f"rote run {app} --input '{{}}' --json --yes"}]
    (action,) = payload["plan"]["actions"]
    assert action["id"] == "install-dbos"
    assert action["command"] == shlex.join(_pip("dbos"))
    assert action["effects"] == [
        {
            "kind": "contact",
            "host": "pypi.org",
            "purpose": "resolve the dbos extra's packages (or the index your pip or uv "
            "config names)",
        },
        {"kind": "contact", "host": "files.pythonhosted.org", "purpose": "download those packages"},
        {
            "kind": "launch",
            "program": shlex.join(_pip("dbos")),
            "purpose": f"install them into this rote's environment ({sys.prefix})",
            "paid": False,
        },
    ]


def test_run_with_yes_installs_then_goes_on(
    tmp_path: Path,
    pip_env: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from rote.runners import TargetError

    fake = FakeInstaller(monkeypatch, "dbos")
    reached: list[Path] = []

    def preflight(target: object, **_: object) -> None:
        reached.append(target.path)  # type: ignore[attr-defined]
        raise TargetError("stopped after the install, at preflight")

    monkeypatch.setattr("rote.runners.preflight_pipeline", preflight)
    app = _dbos_app(tmp_path)
    rc = cli_main(["run", str(app), "--input", "{}", "--yes"])
    err = capsys.readouterr().err
    assert fake.calls == [_pip("dbos")]
    assert reached == [app]
    assert rc == 2
    assert f"installing: {shlex.join(_pip('dbos'))}" in err
    assert "installed the dbos extra" in err


def test_an_install_that_does_not_land_is_a_failure(
    tmp_path: Path,
    pip_env: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The installer exited 0, but the module still does not import (a
    different interpreter's pip, say): that must not read as installed."""
    fake = FakeInstaller(monkeypatch, "temporalio", lands=False)
    app = tmp_path / "app"
    app.mkdir()
    (app / "workflow.py").write_text("", encoding="utf-8")
    (app / "activities.py").write_text("", encoding="utf-8")
    rc = cli_main(["run", str(app), "--input", "{}", "--yes"])
    assert rc == 1
    assert fake.calls == [_pip("temporal")]
    assert capsys.readouterr().err.strip().splitlines()[-1] == (
        f"error: installing the temporal extra failed: RuntimeError: "
        f"`{shlex.join(_pip('temporal'))}` succeeded, but the temporal extra still is "
        f"not importable by {sys.executable}"
    )


def test_a_person_who_declines_a_run_install_stops_the_run(
    tmp_path: Path,
    pip_env: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake = FakeInstaller(monkeypatch, "dbos")
    prompter = ScriptedPrompter([Answer("confirm", "--yes", False)])
    monkeypatch.setattr("rote.interact.prompter.prompter_for", lambda mode: prompter)
    rc = cli_main(["run", str(_dbos_app(tmp_path)), "--input", "{}"])
    assert rc == 1
    assert fake.calls == []
    assert [flag for _, flag, _ in prompter.asked] == ["--yes"]
    assert capsys.readouterr().err.strip().splitlines()[-1] == (
        "nothing was installed; running a dbos app needs the dbos extra: "
        f"pip install 'rote-cli[dbos]=={__version__}'"
    )


def test_an_installed_extra_asks_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Negative half: with dbos importable there is no plan at all."""
    from rote.runners import TargetError

    fake = FakeInstaller(monkeypatch)

    def preflight(target: object, **_: object) -> None:
        raise TargetError("reached preflight")

    monkeypatch.setattr("rote.runners.preflight_pipeline", preflight)
    rc = cli_main(["run", str(_dbos_app(tmp_path)), "--input", "{}"])
    assert rc == 2
    assert fake.calls == []
    assert capsys.readouterr().err.strip() == "error: reached preflight"


# ───────── rote compile ─────────


def _skill(tmp_path: Path) -> Path:
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: s\ndescription: d\n---\n# S\n", encoding="utf-8")
    return skill


def test_compile_asks_before_the_agent_spends_anything(
    tmp_path: Path,
    pip_env: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake = FakeInstaller(monkeypatch, "temporalio")

    def no_compile(*a: object, **k: object) -> None:
        raise AssertionError("the compiler ran before the install was confirmed")

    monkeypatch.setattr("rote.compiler.Compiler.__init__", no_compile)
    skill = _skill(tmp_path)
    out = tmp_path / "out"
    argv = ["compile", str(skill), "--local", "--runtime", "temporal", "--out", str(out), "--json"]
    rc = cli_main(argv)
    assert rc == 2
    assert fake.calls == []
    assert not out.exists()
    payload = json.loads(capsys.readouterr().out)
    assert payload["reason"] == "confirmation_required"
    assert payload["next"] == [{"command": shlex.join(["rote", *argv, "--yes"])}]
    assert payload["plan"]["title"] == "Install the temporal extra: the temporal runtime needs it"


def test_a_declined_compile_install_still_compiles(
    pip_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Emitting temporal code needs no temporalio, so a person who says no
    gets the compile; ensure_extra says go on."""
    fake = FakeInstaller(monkeypatch, "temporalio")
    err = io.StringIO()
    mode = Mode.resolve(env={}, stdin=io.StringIO(), stdout=io.StringIO(), stderr=err)
    prompter = ScriptedPrompter([Answer("confirm", "--yes", False)])
    monkeypatch.setattr("rote.interact.prompter.prompter_for", lambda m: prompter)
    out = Output(mode, stdout=io.StringIO(), stderr=err)
    code = install_extra.ensure_extra(
        "temporal", "the temporal runtime", mode=mode, out=out, argv=[], optional=True
    )
    assert code is None
    assert fake.calls == []
    assert err.getvalue().strip().splitlines()[-1] == (
        "not installed; going on. Running the result needs it: "
        f"pip install 'rote-cli[temporal]=={__version__}'"
    )


# ───────── rote init ─────────

_INIT_FLAGS = [
    "--deploy",
    "none",
    "--runtime",
    "temporal",
    "--agent",
    "auto",
    "--model",
    "",
    "--structure",
    "haiku",
    "--decider",
    "auto",
    "--tiebreak",
    "none",
]


@pytest.fixture
def init_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    config = tmp_path / "cfg" / "config.yaml"
    monkeypatch.setenv("ROTE_CONFIG_PATH", str(config))
    monkeypatch.setattr(
        "rote.cli._build_doctor_report",
        lambda: {"version": "0", "python": "3", "drivers": [], "dependencies": [], "runtimes": []},
    )
    monkeypatch.setattr("rote.compiler.drivers.available_drivers", lambda: [("claude", True, "")])
    return config


def test_init_puts_the_install_in_the_same_plan(
    init_env: Path,
    pip_env: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake = FakeInstaller(monkeypatch, "temporalio")
    rc = cli_main(["init", *_INIT_FLAGS, "--json"])
    assert rc == 2
    payload = json.loads(capsys.readouterr().out)
    assert [a["id"] for a in payload["plan"]["actions"]] == ["write-config", "install-temporal"]
    assert fake.calls == []
    assert not init_env.exists()

    rc = cli_main(["init", *_INIT_FLAGS, "--json", "--yes"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert fake.calls == [_pip("temporal")]
    assert init_env.exists()
    assert [(o["id"], o["status"]) for o in payload["outcomes"]] == [
        ("write-config", "done"),
        ("install-temporal", "done"),
    ]


def test_init_plans_no_install_when_the_extra_is_there(
    init_env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    FakeInstaller(monkeypatch)
    assert cli_main(["init", *_INIT_FLAGS, "--json"]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert [a["id"] for a in payload["plan"]["actions"]] == ["write-config"]


# ───────── rote serve ─────────


def test_serve_on_the_slim_client_names_the_serve_extra(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The base install's fastmcp is the slim client: it imports as
    ``fastmcp`` but cannot serve. ``rote serve`` never prompts (stdio is
    the MCP transport), so it names the fix."""
    import importlib.metadata

    real = importlib.metadata.distribution

    def slim_only(name: str) -> importlib.metadata.Distribution:
        if name == "fastmcp":
            raise importlib.metadata.PackageNotFoundError(name)
        return real(name)

    monkeypatch.setattr(importlib.metadata, "distribution", slim_only)
    assert cli_main(["serve"]) == 2
    assert capsys.readouterr().err.strip() == (
        "error: rote serve needs the serve extra (fastmcp's server half): "
        f"pip install 'rote-cli[serve]=={__version__}'"
    )


def test_serve_extra_is_proven_by_the_full_distribution() -> None:
    """The dev environment has full fastmcp, so serve reads installed; the
    marker is the distribution, not the shared ``fastmcp`` module."""
    from rote.extras import is_installed

    assert EXTRAS["serve"].distribution == "fastmcp"
    assert is_installed("serve") is True


# ───────── the installer, per install method ─────────


def _receipt(tmp_path: Path, body: str) -> Path:
    tool = tmp_path / "uv" / "tools" / "rote-cli"
    tool.mkdir(parents=True)
    (tool / "uv-receipt.toml").write_text(body, encoding="utf-8")
    return tool


@pytest.fixture
def fake_uv(monkeypatch: pytest.MonkeyPatch) -> None:
    import shutil

    real_which = shutil.which
    monkeypatch.setattr(
        shutil, "which", lambda cmd, *a, **k: UV if cmd == "uv" else real_which(cmd, *a, **k)
    )


def test_a_uv_tool_keeps_its_extras_and_with_packages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_uv: None
) -> None:
    """Re-running `uv tool install` replaces the tool's requirements, so the
    command carries what the receipt already has: extras (an alias extra
    dropped, it installs nothing), --with packages and the interpreter."""
    tool = _receipt(
        tmp_path,
        "[tool]\n"
        'requirements = [{ name = "rote-cli", extras = ["temporal", "mcp"], '
        'specifier = "==0.1" }, { name = "rich-click", specifier = ">=1.8" }]\n'
        'python = "3.12"\n',
    )
    monkeypatch.setattr("sys.prefix", str(tool))
    found = installer(["dbos"])
    words = [
        "tool",
        "install",
        "--python",
        "3.12",
        f"rote-cli[dbos,temporal]=={__version__}",
        "--with",
        "rich-click>=1.8",
    ]
    assert found.argv == (UV, *words)
    assert found.hint == (
        f"uv tool install --python 3.12 'rote-cli[dbos,temporal]=={__version__}' "
        "--with 'rich-click>=1.8'"
    )
    assert installer(reinstall=True).hint == (
        f"uv tool install --reinstall --python 3.12 'rote-cli[temporal]=={__version__}' "
        "--with 'rich-click>=1.8'"
    )


def test_a_uv_tool_from_a_checkout_is_never_replaced_from_pypi(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_uv: None
) -> None:
    tool = _receipt(
        tmp_path,
        '[tool]\nrequirements = [{ name = "rote-cli", directory = "/src/rote" }]\n',
    )
    monkeypatch.setattr("sys.prefix", str(tool))
    found = installer(["dbos"])
    assert found.argv is None
    assert found.why_not == "this uv tool was installed from directory /src/rote"


def test_uvx_cannot_be_installed_into(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ephemeral = tmp_path / "cache" / "archive-v0" / "Xq9"
    ephemeral.mkdir(parents=True)
    monkeypatch.setattr("sys.prefix", str(ephemeral))
    found = installer(["temporal"])
    assert found.argv is None
    assert found.hint == f"uvx --from 'rote-cli[temporal]=={__version__}' rote ..."
    assert installer(reinstall=True).hint == (
        f"uvx --reinstall --from 'rote-cli=={__version__}' rote ..."
    )


def test_uvx_turns_the_plan_into_the_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """No environment to install into: exit 2 naming the uvx command."""
    ephemeral = tmp_path / "cache" / "archive-v0" / "Xq9"
    ephemeral.mkdir(parents=True)
    monkeypatch.setattr("sys.prefix", str(ephemeral))
    fake = FakeInstaller(monkeypatch, "dbos")
    rc = cli_main(["run", str(_dbos_app(tmp_path)), "--input", "{}", "--yes"])
    assert rc == 2
    assert fake.calls == []
    assert capsys.readouterr().err.strip() == (
        "error: running a dbos app needs the dbos extra (uvx runs rote in a throwaway "
        "environment; run the next uvx command with the extra named): "
        f"uvx --from 'rote-cli[dbos]=={__version__}' rote ..."
    )


def test_a_venv_without_pip_uses_uv_pip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_uv: None
) -> None:
    """`uv venv` makes environments without pip."""
    monkeypatch.setattr("sys.prefix", str(tmp_path / "venv"))
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a: None if name == "pip" else real_find_spec(name, *a),
    )
    found = installer(["dbos"])
    spec = f"rote-cli[dbos]=={__version__}"
    assert found.argv == (UV, "pip", "install", "--python", sys.executable, spec)
    assert found.hint == f"pip install '{spec}'"


def test_an_editable_checkout_installs_its_own_extra(
    tmp_path: Path, pip_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """From PyPI, the released wheel would replace the checkout."""
    monkeypatch.setattr("sys.prefix", str(tmp_path / "venv"))
    monkeypatch.setattr("rote.extras._direct_url_source", lambda: ("editable", "/src/rote"))
    found = installer(["dbos"])
    assert found.argv == (sys.executable, "-m", "pip", "install", "-e", "/src/rote[dbos]")
    assert found.hint == "pip install -e '/src/rote[dbos]'"
    monkeypatch.setattr("rote.extras._direct_url_source", lambda: ("other", "file:///tmp/rote.whl"))
    other = installer(["dbos"])
    assert other.argv is None
    assert other.why_not == "rote-cli was installed from file:///tmp/rote.whl, not PyPI"


def test_the_direct_url_reader_sees_this_checkout_as_editable() -> None:
    """The suite runs from an editable install (conftest hides it from every
    other test); the real reader must recognize it."""
    from tests.conftest import REAL_DIRECT_URL_SOURCE

    source = REAL_DIRECT_URL_SOURCE()
    assert source is not None and source[0] == "editable"
    assert (Path(source[1]) / "pyproject.toml").is_file()
