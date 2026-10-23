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


# ───────── compile, emit, init: a note, never a stop ─────────


def _skill(tmp_path: Path) -> Path:
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: s\ndescription: d\n---\n# S\n", encoding="utf-8")
    return skill


class _ReachedCompiler(BaseException):
    """Raised from Compiler.__init__: the compile went on to the compiler."""


def _note(command: str, runtime: str, extra: str) -> str:
    return (
        f"rote {command}: note: running {runtime} apps needs the {extra} extra, which this "
        "install lacks; `rote run` offers to install it, or: "
        f"pip install 'rote-cli[{extra}]=={__version__}'"
    )


@pytest.mark.parametrize("missing", [True, False])
def test_compile_never_stops_for_a_missing_runtime_extra(
    missing: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Emitting temporal code needs no temporalio. `compile --json` without
    --yes once exited 2 over it; now it reaches the compiler with a note,
    and with the extra installed it says nothing about it."""
    fake = FakeInstaller(monkeypatch, *(["temporalio"] if missing else []))

    def reached(*a: object, **k: object) -> None:
        raise _ReachedCompiler

    monkeypatch.setattr("rote.compiler.Compiler.__init__", reached)
    argv = ["compile", str(_skill(tmp_path)), "--local", "--runtime", "temporal"]
    with pytest.raises(_ReachedCompiler):
        cli_main([*argv, "--out", str(tmp_path / "out"), "--json"])
    err = capsys.readouterr().err
    assert fake.calls == []
    assert "Plan:" not in err
    assert (_note("compile", "temporal", "temporal") in err.splitlines()) is missing


@pytest.mark.parametrize("missing", [True, False])
def test_emit_notes_a_missing_extra_and_succeeds(
    missing: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake = FakeInstaller(monkeypatch, *(["dbos"] if missing else []))
    pipeline = Path(__file__).parent.parent / "examples/expense-audit/expected/pipeline.yaml"
    rc = cli_main(["emit", str(pipeline), "--runtime", "dbos", "--out", str(tmp_path / "app")])
    assert rc == 0
    assert fake.calls == []
    assert (tmp_path / "app" / "main.py").is_file()
    assert (_note("emit", "dbos", "dbos") in capsys.readouterr().err.splitlines()) is missing


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


@pytest.mark.parametrize("missing", [True, False])
def test_init_saves_the_default_and_only_notes_the_extra(
    missing: bool,
    init_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake = FakeInstaller(monkeypatch, *(["temporalio"] if missing else []))
    rc = cli_main(["init", *_INIT_FLAGS, "--json", "--yes"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert fake.calls == []
    assert init_env.exists()
    assert [a["id"] for a in payload["plan"]["actions"]] == ["write-config"]
    note = (
        "running temporal apps needs the temporal extra, which this install lacks; "
        f"`rote run` offers to install it, or: pip install 'rote-cli[temporal]=={__version__}'"
    )
    assert payload["plan"]["notes"] == ([note] if missing else [])


# ───────── rote serve ─────────


def _without_distribution(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    import importlib.metadata

    real = importlib.metadata.distribution

    def distribution(name: str) -> importlib.metadata.Distribution:
        if name in names:
            raise importlib.metadata.PackageNotFoundError(name)
        return real(name)

    monkeypatch.setattr(importlib.metadata, "distribution", distribution)


def test_serve_on_the_slim_client_names_the_serve_extra(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The base install's fastmcp is the slim client: it imports as
    ``fastmcp`` but cannot serve (cyclopts is one of the server half's
    requirements). ``rote serve`` never prompts (stdio is the MCP
    transport), so it names the fix."""
    _without_distribution(monkeypatch, "fastmcp", "cyclopts")
    assert cli_main(["serve"]) == 2
    assert capsys.readouterr().err.strip() == (
        "error: rote serve needs the serve extra (fastmcp's server half): "
        f"pip install 'rote-cli[serve]=={__version__}'"
    )


def test_serve_accepts_the_slim_server_extra_without_full_fastmcp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``fastmcp-slim[client,server]`` serves as well as full fastmcp: the
    check reads the server half's requirements, not one distribution name."""
    from rote.extras import is_installed

    assert EXTRAS["serve"].requires_extra_of == ("fastmcp-slim", "server")
    assert is_installed("serve") is True
    _without_distribution(monkeypatch, "fastmcp")
    assert is_installed("serve") is True
    _without_distribution(monkeypatch, "fastmcp", "joserfc")
    assert is_installed("serve") is False


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
    """`uv venv` makes environments without pip, where a bare `pip install`
    reaches some other environment's pip: the hint is the runnable uv form."""
    monkeypatch.setattr("sys.prefix", str(tmp_path / "venv"))
    monkeypatch.setattr("rote.extras._has_pip", lambda: False)
    found = installer(["dbos"])
    spec = f"rote-cli[dbos]=={__version__}"
    assert found.argv == (UV, "pip", "install", "--python", sys.executable, spec)
    assert found.hint == f"uv pip install --python {shlex.quote(sys.executable)} '{spec}'"


def test_the_pip_check_reads_this_interpreter() -> None:
    """conftest pins ``_has_pip`` for every other test; the real check is
    find_spec on this interpreter."""
    from tests.conftest import REAL_HAS_PIP

    assert REAL_HAS_PIP() is (importlib.util.find_spec("pip") is not None)


def _pipx_venv(tmp_path: Path, recorded: str, name: str = "rote-cli") -> Path:
    venv = tmp_path / "pipx" / "venvs" / name
    venv.mkdir(parents=True)
    (venv / "pipx_metadata.json").write_text(
        json.dumps({"main_package": {"package_or_url": recorded}, "pipx_metadata_version": "0.5"}),
        encoding="utf-8",
    )
    return venv


@pytest.fixture
def fake_pipx(monkeypatch: pytest.MonkeyPatch) -> str:
    import shutil

    pipx = "/opt/fake/bin/pipx"
    real_which = shutil.which
    monkeypatch.setattr(
        shutil, "which", lambda cmd, *a, **k: pipx if cmd == "pipx" else real_which(cmd, *a, **k)
    )
    return pipx


def test_pipx_injects_the_apps_own_extra_keeping_the_recorded_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_pipx: str
) -> None:
    """The extras the main package was installed with are carried into the
    injected requirement, so the command names every extra the app has."""
    venv = _pipx_venv(tmp_path, "rote-cli[temporal, mcp]==0.1", name="rote-cli-dev")
    monkeypatch.setattr("sys.prefix", str(venv))
    found = installer(["dbos"])
    spec = f"rote-cli[dbos,temporal]=={__version__}"
    # --force is required: without it pipx 1.11.1 calls the app's own
    # distribution already injected, warns, exits 0 and installs nothing.
    # The fake pipx cannot show that; it was found on a real pipx run.
    assert found.argv == (fake_pipx, "inject", "--force", "rote-cli-dev", spec)
    assert found.hint == f"pipx inject --force rote-cli-dev '{spec}'"
    assert installer(reinstall=True).hint == "pipx reinstall rote-cli-dev"


def test_pipx_from_a_path_is_never_replaced_from_pypi(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_pipx: str
) -> None:
    monkeypatch.setattr("sys.prefix", str(_pipx_venv(tmp_path, "/src/rote")))
    found = installer(["dbos"])
    assert found.argv is None
    assert found.why_not == "this pipx app was installed from /src/rote"


def test_pipx_run_installs_through_pipx(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_pipx: str
) -> None:
    from rote.runners import TargetError

    monkeypatch.setattr("sys.prefix", str(_pipx_venv(tmp_path, "rote-cli==0.1")))
    fake = FakeInstaller(monkeypatch, "dbos")

    def preflight(target: object, **_: object) -> None:
        raise TargetError("reached preflight")

    monkeypatch.setattr("rote.runners.preflight_pipeline", preflight)
    assert cli_main(["run", str(_dbos_app(tmp_path)), "--input", "{}", "--yes"]) == 2
    # --force is required: without it pipx 1.11.1 calls the app's own
    # distribution already injected, warns, exits 0 and installs nothing.
    # The fake pipx cannot show that; it was found on a real pipx run.
    spec = f"rote-cli[dbos]=={__version__}"
    assert fake.calls == [(fake_pipx, "inject", "--force", "rote-cli", spec)]


def test_a_with_requirement_keeps_its_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_uv: None
) -> None:
    """Re-running `uv tool install` replaces the requirements, so a marker
    dropped from a --with package would install it on every platform."""
    tool = _receipt(
        tmp_path,
        "[tool]\n"
        'requirements = [{ name = "rote-cli" }, { name = "rich-click", '
        """marker = "sys_platform == 'darwin'", specifier = ">=1.8" }]\n""",
    )
    monkeypatch.setattr("sys.prefix", str(tool))
    found = installer(["dbos"])
    marked = "rich-click>=1.8; sys_platform == 'darwin'"
    assert found.argv == (
        UV,
        "tool",
        "install",
        f"rote-cli[dbos]=={__version__}",
        "--with",
        marked,
    )
    assert found.hint.endswith(f"--with {shlex.quote(marked)}")


def test_an_editable_checkout_installs_its_own_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
