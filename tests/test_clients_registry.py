"""What every client source promises, checked across all of them at once."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from rote.clients import SOURCES, ClientContext, ordered_sources, source
from rote.clients.base import CLIENT_DIR_VARS, UNCHECKED_TRUST
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


def test_unknown_trust_is_never_preselected_and_needs_an_opt_in(tmp_path: Path) -> None:
    from dataclasses import replace

    from rote.clients import ImportCandidate

    base = ImportCandidate(
        client="x",
        name="srv",
        original_name="srv",
        scope="user",
        origin=tmp_path / "f.json",
        transport="streamable-http",
        url="https://srv.example.com/mcp",
    )
    assert base.trust == "trusted" and base.preselectable and base.needs_opt_in == ()
    # A repository-supplied scope fails closed until a source reads its gate.
    project = replace(base, scope="project", trust=None)
    assert project.trust == "unknown" and not project.preselectable
    assert project.trust_note == f"trust in x could not be checked: {UNCHECKED_TRUST}"
    unknown = replace(base, trust="unknown", trust_note="trust in X could not be checked: why")
    assert not unknown.preselectable
    assert unknown.needs_opt_in == ("trust in X could not be checked: why",)
    both = replace(unknown, risk="path segment 2 may be an API key")
    assert both.needs_opt_in == ("path segment 2 may be an API key", unknown.trust_note)
    untrusted = replace(base, trust="untrusted", trust_note="gate closed")
    assert not untrusted.preselectable and untrusted.needs_opt_in == ()
    assert untrusted.to_dict()["trust"] == "untrusted"


def _remote(url: str) -> dict[str, str]:
    return {"type": "http", "url": url}


#: For every source with a repository-supplied scope: write one project
#: server named ``proj`` (with the client's gate, if it has one, passed) and
#: return the context to read it in.
def _claude_code(home: Path, repo: Path) -> None:
    write_json(repo / ".mcp.json", {"mcpServers": {"proj": _remote("https://p.example/mcp")}})
    write_json(repo / ".claude" / "settings.local.json", {"enabledMcpjsonServers": ["proj"]})


def _codex(home: Path, repo: Path) -> None:
    (repo / ".codex").mkdir()
    (repo / ".codex" / "config.toml").write_text(
        '[mcp_servers.proj]\nurl = "https://p.example/mcp"\n'
    )
    (home / ".codex").mkdir()
    (home / ".codex" / "config.toml").write_text(f'[projects."{repo}"]\ntrust_level = "trusted"\n')


def _cursor(home: Path, repo: Path) -> None:
    write_json(
        repo / ".cursor" / "mcp.json", {"mcpServers": {"proj": {"url": "https://p.example/mcp"}}}
    )


def _vscode(home: Path, repo: Path) -> None:
    write_json(
        repo / ".vscode" / "mcp.json", {"servers": {"proj": _remote("https://p.example/mcp")}}
    )


def _copilot(home: Path, repo: Path) -> None:
    write_json(repo / ".mcp.json", {"mcpServers": {"proj": _remote("https://p.example/mcp")}})


def _gemini(home: Path, repo: Path) -> None:
    write_json(
        repo / ".gemini" / "settings.json",
        {"mcpServers": {"proj": {"httpUrl": "https://p.example/mcp"}}},
    )
    write_json(home / ".gemini" / "trustedFolders.json", {str(repo): "TRUST_FOLDER"})


def _opencode(home: Path, repo: Path) -> None:
    write_json(
        repo / "opencode.json",
        {"mcp": {"proj": {"type": "remote", "url": "https://p.example/mcp"}}},
    )


def _zed(home: Path, repo: Path) -> None:
    write_json(
        repo / ".zed" / "settings.json",
        {"context_servers": {"proj": {"url": "https://p.example/mcp"}}},
    )


def _amp(home: Path, repo: Path) -> None:
    write_json(
        repo / ".amp" / "settings.json",
        {"amp.mcpServers": {"proj": {"url": "https://p.example/mcp"}}},
    )


def _windsurf(home: Path, repo: Path) -> None:
    write_json(
        repo / ".devin" / "mcp_config.json",
        {"mcpServers": {"proj": {"url": "https://p.example/mcp"}}},
    )


PROJECT_FIXTURES = {
    "claude-code": _claude_code,
    "codex": _codex,
    "cursor": _cursor,
    "vscode": _vscode,
    "copilot": _copilot,
    "gemini": _gemini,
    "opencode": _opencode,
    "zed": _zed,
    "amp": _amp,
    "windsurf": _windsurf,
}
#: Sources with no repository-supplied scope at all.
NO_PROJECT_SCOPE = {"goose", "claude-desktop"}


def test_every_source_is_classified_for_project_trust() -> None:
    """A new source must either get a project fixture below or say it has
    no project scope; it cannot slip past the fail-closed check."""
    assert {s.id for s in SOURCES} == set(PROJECT_FIXTURES) | NO_PROJECT_SCOPE


@pytest.mark.usefixtures("refuse_real_home")
@pytest.mark.parametrize("client", sorted(PROJECT_FIXTURES))
def test_every_project_candidate_has_trust_set_on_purpose(tmp_path: Path, client: str) -> None:
    """The default for a project candidate is unknown with UNCHECKED_TRUST.
    A source must replace it: trusted after reading a passed gate, or a
    note of its own. A source that forgets fails here."""
    ctx = _empty_machine(tmp_path)
    PROJECT_FIXTURES[client](ctx.home, ctx.cwd)
    project = [c for c in source(client).read(ctx).candidates if c.scope == "project"]
    assert [c.original_name for c in project] == ["proj"], client
    [candidate] = project
    assert UNCHECKED_TRUST not in (candidate.trust_note or ""), client
    if candidate.trust == "trusted":
        assert candidate.trust_note is None
    else:
        assert candidate.trust_note, client
