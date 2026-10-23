"""What every client source promises, checked across all of them at once."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from rote.clients import SOURCES, ClientContext, ordered_sources, source
from rote.clients.base import CLIENT_DIR_VARS
from rote.interact.agents import AI_AGENT_CLIENTS, SIGNALS
from tests._clients import refuse_real_home, write_json

__all__ = ["refuse_real_home"]

#: Signal client ids with no config file to read: Replit's agent lives in
#: the browser.
NO_SOURCE = {"replit"}


def _empty_machine(tmp_path: Path) -> ClientContext:
    home = tmp_path / "home"
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    home.mkdir()
    return ClientContext(home=home, cwd=repo, env={})


def test_ids_are_unique_and_lookups_work() -> None:
    ids = [s.id for s in SOURCES]
    assert len(ids) == len(set(ids))
    assert {
        "claude-code",
        "codex",
        "cursor",
        "vscode",
        "copilot",
        "gemini",
        "opencode",
        "zed",
        "amp",
        "goose",
        "windsurf",
        "claude-desktop",
    } <= set(ids)
    for s in SOURCES:
        assert source(s.id) is s


def test_every_detected_client_has_a_source() -> None:
    """The env signals name clients by source id, so a detected client is
    always the one offered first."""
    ids = {s.id for s in SOURCES}
    named = {signal.client for signal in SIGNALS} | set(AI_AGENT_CLIENTS.values())
    assert named - NO_SOURCE <= ids, named - NO_SOURCE - ids


@pytest.mark.usefixtures("refuse_real_home")
@pytest.mark.parametrize("client", [s.id for s in SOURCES])
def test_an_empty_machine_reads_nothing_and_detects_nothing(tmp_path: Path, client: str) -> None:
    ctx = _empty_machine(tmp_path)
    src = source(client)
    assert not src.detect(ctx)
    reading = src.read(ctx)
    assert reading.client == client
    assert reading.candidates == () and reading.problems == () and reading.files_read == ()
    assert reading.limits, "every source states what it does not read"


@pytest.mark.usefixtures("refuse_real_home")
@pytest.mark.parametrize("platform", ["darwin", "linux", "win32"])
@pytest.mark.parametrize("client", [s.id for s in SOURCES])
def test_config_paths_stay_inside_the_context(tmp_path: Path, client: str, platform: str) -> None:
    """With no system dirs and no environment, a source looks only under the
    home and the project it was given, on every OS."""
    ctx = _empty_machine(tmp_path)
    ctx = ClientContext(home=ctx.home, cwd=ctx.cwd, env={}, platform=platform)
    for path in source(client).config_paths(ctx):
        assert path.is_relative_to(tmp_path), (client, platform, path)


def test_detected_client_then_configured_clients_come_first(tmp_path: Path) -> None:
    ctx = _empty_machine(tmp_path)
    write_json(ctx.home / ".gemini" / "settings.json", {})
    (ctx.home / ".codex").mkdir()
    order = [s.id for s in ordered_sources("zed", ctx)]
    assert order[0] == "zed"
    assert order[1:3] == ["codex", "gemini"]  # detected on disk, in SOURCES order
    assert [s.id for s in ordered_sources(None)] == [s.id for s in SOURCES]


@pytest.mark.parametrize("client", [s.id for s in SOURCES if s.id != "claude-code"])
def test_every_module_cites_what_it_was_verified_against(client: str) -> None:
    module = sys.modules[type(source(client)).__module__]
    doc = " ".join((module.__doc__ or "").split())
    assert "verified 2026-10-22" in doc.lower(), client
    assert "https://" in doc, client


def test_isolation_drops_every_variable_a_source_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    """A variable that moves a client's files must be dropped under
    ROTE_CLIENT_HOME_DIR, or it leads back to the real configs."""
    import re

    read: set[str] = set()
    for s in SOURCES:
        text = Path(importlib.import_module(type(s).__module__).__file__ or "").read_text()
        read |= set(re.findall(r'env(?:\.get\(|\[)"([A-Z][A-Z0-9_]+)"', text))
    switches = {
        # On/off switches, not locations: they cannot lead to another file.
        "OPENCODE_DISABLE_PROJECT_CONFIG",
        "GEMINI_RESTRICTED_MODE",
        "GEMINI_CLI_TRUST_WORKSPACE",
    }
    assert read - switches <= set(CLIENT_DIR_VARS), read - switches - set(CLIENT_DIR_VARS)
    for var in CLIENT_DIR_VARS:
        monkeypatch.setenv(var, "/real/place")
    ctx = ClientContext.current()
    assert not set(CLIENT_DIR_VARS) & set(ctx.env)
    assert ctx.etc_dir is None and ctx.system_dir is None
