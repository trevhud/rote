"""The explicit live-MCP opt-out stops before registry or inference work."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from rote.cli import main


class _CompilerReached(Exception):
    pass


@pytest.fixture
def compiler_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def capture(**kwargs: Any) -> None:
        calls.append(kwargs)
        raise _CompilerReached

    monkeypatch.setattr("rote.cli.Compiler", capture)
    return calls


def _args(tmp_path: Path, command: str) -> list[str]:
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("# Test skill\n", encoding="utf-8")
    args = [command, str(skill)]
    if command == "compile":
        args += ["--local", "--out", str(tmp_path / "out")]
    return args


@pytest.mark.parametrize("command", ["compile", "analyze"])
@pytest.mark.parametrize("driver", ["api", "openai-api"])
def test_no_mcp_skips_registry_and_passes_empty_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    compiler_calls: list[dict[str, Any]],
    command: str,
    driver: str,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("--no-mcp must not resolve registry credentials")

    monkeypatch.setattr("rote.cli._live_mcp_specs_for_driver", forbidden)
    with pytest.raises(_CompilerReached):
        main(_args(tmp_path, command) + ["--agent", driver, "--no-mcp"])
    assert compiler_calls[0]["agent"] == driver
    assert compiler_calls[0]["driver_kwargs"] == {"mcp_servers": []}


@pytest.mark.parametrize("command", ["compile", "analyze"])
def test_no_mcp_accepts_autodetected_api_and_pins_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    compiler_calls: list[dict[str, Any]],
    command: str,
) -> None:
    monkeypatch.setattr("rote.compiler.drivers.auto_detect", lambda: SimpleNamespace(name="api"))
    with pytest.raises(_CompilerReached):
        main(_args(tmp_path, command) + ["--no-mcp"])
    assert compiler_calls[0]["agent"] == "api"
    assert compiler_calls[0]["driver_kwargs"] == {"mcp_servers": []}


@pytest.mark.parametrize("command", ["compile", "analyze"])
@pytest.mark.parametrize("driver", ["claude", "codex"])
def test_no_mcp_rejects_subprocess_drivers_before_compiling(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    compiler_calls: list[dict[str, Any]],
    command: str,
    driver: str,
) -> None:
    assert main(_args(tmp_path, command) + ["--agent", driver, "--no-mcp"]) == 2
    assert "subprocess drivers manage their own MCP" in capsys.readouterr().err
    assert compiler_calls == []


def test_default_compile_still_discovers_mcp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    compiler_calls: list[dict[str, Any]],
) -> None:
    servers = [{"name": "words", "url": "http://localhost:9/mcp", "headers": None}]
    monkeypatch.setattr("rote.cli._live_mcp_specs_for_driver", lambda _agent: servers)
    with pytest.raises(_CompilerReached):
        main(_args(tmp_path, "compile") + ["--agent", "api"])
    assert compiler_calls[0]["driver_kwargs"] == {"mcp_servers": servers}


@pytest.mark.parametrize("command", ["compile", "analyze"])
def test_no_mcp_uses_configured_driver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    compiler_calls: list[dict[str, Any]],
    command: str,
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("agent: api\n", encoding="utf-8")
    monkeypatch.setenv("ROTE_CONFIG_PATH", str(config))
    with pytest.raises(_CompilerReached):
        main(_args(tmp_path, command) + ["--no-mcp"])
    assert compiler_calls[0]["agent"] == "api"
    assert compiler_calls[0]["driver_kwargs"] == {"mcp_servers": []}


def test_default_analyze_still_has_no_live_tools(
    tmp_path: Path, compiler_calls: list[dict[str, Any]]
) -> None:
    with pytest.raises(_CompilerReached):
        main(_args(tmp_path, "analyze") + ["--agent", "api"])
    assert compiler_calls[0]["driver_kwargs"] is None


def test_no_mcp_rejects_baseline_before_raw_skill_runs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], compiler_calls: list[dict[str, Any]]
) -> None:
    assert main(_args(tmp_path, "compile") + ["--agent", "api", "--no-mcp", "--baseline"]) == 2
    assert "cannot be combined with --baseline" in capsys.readouterr().err
    assert compiler_calls == []


def test_no_mcp_cloud_rejection_points_to_local(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], compiler_calls: list[dict[str, Any]]
) -> None:
    args = _args(tmp_path, "compile")
    args.remove("--local")
    assert main(args + ["--cloud", "--no-mcp"]) == 2
    assert "--no-mcp runs locally — add --local" in capsys.readouterr().err
    assert compiler_calls == []
