"""Claude Code's MCP config, read from fixture files shaped like the real ones.

The shapes (key names and nesting) were checked against a real install;
every value here is made up. No test reads the real home: the context is
injected and ``Path.home`` is made to fail.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rote.clients import ClientContext, ImportCandidate, ordered_sources, source
from rote.clients.claude_code import ClaudeCodeSource

SECRET = "sk-literal-NOT-A-REAL-KEY"
DEFAULT_SECRET = "sk-default-NOT-REAL"
QUERY_SECRET = "qk-NOT-REAL"


def _write(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _http(url: str, **extra: Any) -> dict[str, Any]:
    return {"type": "http", "url": url, **extra}


@pytest.fixture(autouse=True)
def _no_real_home(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse() -> Path:
        raise AssertionError("a client source read Path.home(); use ClientContext")

    monkeypatch.setattr(Path, "home", staticmethod(refuse))


@pytest.fixture
def machine(tmp_path: Path) -> tuple[ClientContext, Path]:
    home = tmp_path / "home"
    project = tmp_path / "work" / "repo"
    (project / ".git").mkdir(parents=True)
    _write(
        home / ".claude.json",
        {
            "numStartups": 3,
            "mcpServers": {
                "figma": _http("https://mcp.figma.com/mcp"),
                "acme-crm-prod": _http(
                    "https://oms.example-work.com/mcp",
                    headers={"Authorization": "Bearer ${OMS_TOKEN}"},
                ),
                "literal-key": _http("https://api.example.com/mcp", headers={"X-Api-Key": SECRET}),
                "with-default": _http(
                    "https://d.example.com/mcp",
                    headers={"Authorization": "Bearer ${TOK:-" + DEFAULT_SECRET + "}"},
                ),
                "ga4": {"type": "stdio", "command": "uvx", "args": ["ga4-mcp"], "env": {}},
                "bare-command": {"command": "node", "args": ["server.js"]},
                "sse-one": {"type": "sse", "url": "https://sse.example.com/sse"},
                "sock": {"type": "ws", "url": "wss://ws.example.com"},
                "notype": {"url": "https://nt.example.com/mcp"},
                "query": _http(f"https://q.example.com/mcp?api_key={QUERY_SECRET}"),
                "url-var": _http("https://${HOST_VAR}/mcp"),
                "helper": _http(
                    "https://h.example.com/mcp",
                    headersHelper="print-headers.sh",
                    oauth={"clientId": "issued-to-claude", "scopes": "read write"},
                ),
                "9lives": _http("https://nine.example.com/mcp"),
            },
            "projects": {
                str(project): {
                    "hasTrustDialogAccepted": False,
                    "mcpServers": {"figma": _http("https://figma.local.example.com/mcp")},
                    "enabledMcpjsonServers": ["approved"],
                    "disabledMcpjsonServers": ["blocked-proj"],
                    "disabledMcpServers": ["sse-one"],
                },
                str(tmp_path / "elsewhere"): {"mcpServers": {"other": _http("https://o.example")}},
            },
        },
    )
    _write(
        project / ".mcp.json",
        {
            "mcpServers": {
                "approved": _http("https://approved.example.com/mcp"),
                "pending": _http("https://pending.example.com/mcp"),
                "blocked-proj": _http("https://blocked.example.com/mcp"),
                "from-settings": _http("https://fs.example.com/mcp"),
                "acme-crm-prod": _http("https://oms-project.example-work.com/mcp"),
            }
        },
    )
    _write(project / ".claude" / "settings.json", {"enabledMcpjsonServers": ["from-settings"]})
    plugins = tmp_path / "plugin-cache"
    _write(
        home / ".claude" / "plugins" / "installed_plugins.json",
        {
            "version": 2,
            "plugins": {
                "linear@official": [
                    {"scope": "user", "installPath": str(plugins / "linear"), "version": "1.0.0"}
                ],
                "multi@market": [
                    {"scope": "user", "installPath": str(plugins / "multi"), "version": "2"}
                ],
                "off@official": [
                    {"scope": "user", "installPath": str(plugins / "off"), "version": "1"}
                ],
            },
        },
    )
    _write(
        home / ".claude" / "settings.json",
        {"enabledPlugins": {"linear@official": True, "multi@market": True, "off@official": False}},
    )
    _write(plugins / "linear" / ".mcp.json", {"linear": _http("https://mcp.linear.app/mcp")})
    _write(
        plugins / "multi" / ".claude-plugin" / "plugin.json",
        {
            "name": "multi",
            "mcpServers": [
                "./servers.json",
                {"inline": _http("https://inline.example.com/mcp")},
                "./bundle.mcpb",
                "../../escape.json",
            ],
        },
    )
    _write(
        plugins / "multi" / "servers.json",
        {"mcpServers": {"from-file": _http("https://ff.example")}},
    )
    _write(plugins / "off" / ".mcp.json", {"hidden": _http("https://hidden.example.com/mcp")})
    cwd = project / "src" / "pkg"
    cwd.mkdir(parents=True)
    return ClientContext(home=home, cwd=cwd, env={}), project


def _by(candidates: tuple[ImportCandidate, ...]) -> dict[tuple[str, str], ImportCandidate]:
    return {(c.scope, c.original_name): c for c in candidates}


def test_scopes_and_precedence(machine: tuple[ClientContext, Path]) -> None:
    ctx, project = machine
    found = _by(ClaudeCodeSource().read(ctx).candidates)
    local_figma = found[("local", "figma")]
    assert local_figma.inactive is None
    assert local_figma.host == "figma.local.example.com"
    assert found[("user", "figma")].inactive == "shadowed by the local entry of the same name"
    # The project's acme-crm-prod is not approved, so the user entry wins.
    assert found[("project", "acme-crm-prod")].inactive == (
        "not approved in Claude Code for this project"
    )
    assert found[("user", "acme-crm-prod")].inactive is None
    assert ("local", "other") not in found  # another project's local scope


def test_project_approval_rules(machine: tuple[ClientContext, Path]) -> None:
    ctx, project = machine
    found = _by(ClaudeCodeSource().read(ctx).candidates)
    assert found[("project", "approved")].inactive is None
    assert found[("project", "pending")].inactive == "not approved in Claude Code for this project"
    assert found[("project", "blocked-proj")].inactive == "disabled for this project in Claude Code"
    # Committed settings approve only in a trusted folder.
    assert found[("project", "from-settings")].inactive is not None

    claude_json = json.loads((ctx.home / ".claude.json").read_text())
    claude_json["projects"][str(project)]["hasTrustDialogAccepted"] = True
    _write(ctx.home / ".claude.json", claude_json)
    trusted = _by(ClaudeCodeSource().read(ctx).candidates)
    assert trusted[("project", "from-settings")].inactive is None
    assert trusted[("project", "pending")].inactive is not None


def test_enable_all_approves_but_a_disable_still_wins(machine: tuple[ClientContext, Path]) -> None:
    ctx, project = machine
    _write(project / ".claude" / "settings.local.json", {"enableAllProjectMcpServers": True})
    found = _by(ClaudeCodeSource().read(ctx).candidates)
    assert found[("project", "pending")].inactive is None
    assert found[("project", "blocked-proj")].inactive == "disabled for this project in Claude Code"


def test_the_mcp_toggle_marks_a_server_inactive(machine: tuple[ClientContext, Path]) -> None:
    found = _by(ClaudeCodeSource().read(machine[0]).candidates)
    assert found[("user", "sse-one")].inactive == "turned off in Claude Code (/mcp)"
    assert found[("user", "sse-one")].transport == "sse"
    assert found[("user", "sse-one")].importable


@pytest.mark.parametrize(
    ("name", "transport", "reason"),
    [
        ("ga4", "stdio", "a stdio server"),
        ("bare-command", "stdio", "a stdio server"),
        ("sock", "websocket", "WebSocket"),
        ("notype", "other", "no type, which Claude Code itself rejects"),
        ("query", "streamable-http", "query string"),
        ("url-var", "streamable-http", "depends on HOST_VAR"),
    ],
)
def test_what_cannot_be_imported_says_why(
    machine: tuple[ClientContext, Path], name: str, transport: str, reason: str
) -> None:
    candidate = _by(ClaudeCodeSource().read(machine[0]).candidates)[("user", name)]
    assert not candidate.importable
    assert candidate.transport == transport
    assert candidate.blocked is not None and reason in candidate.blocked


def test_importable_entries_are_not_blocked(machine: tuple[ClientContext, Path]) -> None:
    found = _by(ClaudeCodeSource().read(machine[0]).candidates)
    for name in ("figma", "acme-crm-prod", "literal-key", "with-default", "helper", "9lives"):
        assert found[("user", name)].blocked is None, name


def test_names_become_identifiers_and_record_the_original(
    machine: tuple[ClientContext, Path],
) -> None:
    found = _by(ClaudeCodeSource().read(machine[0]).candidates)
    oms = found[("user", "acme-crm-prod")]
    assert (oms.name, oms.original_name, oms.renamed) == ("acme_crm_prod", "acme-crm-prod", True)
    assert found[("user", "9lives")].name == "_9lives"
    assert not found[("user", "figma")].renamed


def test_env_references_are_kept_as_templates_never_resolved(
    machine: tuple[ClientContext, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OMS_TOKEN", "resolved-NOT-REAL")
    oms = _by(ClaudeCodeSource().read(machine[0]).candidates)[("user", "acme-crm-prod")]
    assert [h.to_dict() for h in oms.headers] == [
        {"name": "Authorization", "template": "Bearer ${OMS_TOKEN}", "env_vars": ["OMS_TOKEN"]}
    ]
    assert "resolved-NOT-REAL" not in json.dumps(oms.to_dict())


def test_literal_secrets_and_other_apps_credentials_are_withheld(
    machine: tuple[ClientContext, Path],
) -> None:
    reading = ClaudeCodeSource().read(machine[0])
    found = _by(reading.candidates)
    literal = found[("user", "literal-key")]
    assert literal.headers == ()
    assert literal.withheld == (
        "header X-Api-Key: a literal value, not copied; set it yourself: "
        "rote mcp add literal_key <url> --header 'X-Api-Key: <value>'",
    )
    defaulted = found[("user", "with-default")]
    assert [h.template for h in defaulted.headers] == ["Bearer ${TOK}"]
    assert any("default for ${TOK} was not copied" in w for w in defaulted.withheld)
    helper = found[("user", "helper")]
    assert helper.scopes == ("read", "write")
    assert any(w.startswith("headersHelper:") for w in helper.withheld)
    assert any(w.startswith("oauth.clientId: issued to Claude Code") for w in helper.withheld)

    # Nothing secret leaves through the display or the JSON form.
    rendered = json.dumps(reading.to_dict()) + repr(reading)
    for secret in (SECRET, DEFAULT_SECRET, QUERY_SECRET, "issued-to-claude", "print-headers.sh"):
        assert secret not in rendered, secret
    # The display carries hosts, never URLs.
    assert "https://" not in json.dumps(reading.to_dict())


def test_enabled_plugins_only_with_every_declaration_form(
    machine: tuple[ClientContext, Path],
) -> None:
    reading = ClaudeCodeSource().read(machine[0])
    plugin = sorted(
        (c.plugin, c.original_name, c.host) for c in reading.candidates if c.scope == "plugin"
    )
    assert plugin == [
        ("linear", "linear", "mcp.linear.app"),
        ("multi", "from-file", "ff.example"),
        ("multi", "inline", "inline.example.com"),
    ]
    assert any("bundle.mcpb" not in p and ".mcpb bundle" in p for p in reading.problems)
    assert any("leaves the plugin" in p for p in reading.problems)


def test_a_broken_file_is_a_problem_not_a_crash(machine: tuple[ClientContext, Path]) -> None:
    ctx, project = machine
    (project / ".mcp.json").write_text("{not json", encoding="utf-8")
    reading = ClaudeCodeSource().read(ctx)
    assert any(".mcp.json: could not read it (JSONDecodeError)" in p for p in reading.problems)
    assert ("user", "figma") in _by(reading.candidates)  # the rest still reads


def test_files_read_lists_what_was_opened(machine: tuple[ClientContext, Path]) -> None:
    ctx, project = machine
    read = set(ClaudeCodeSource().read(ctx).files_read)
    assert ctx.home / ".claude.json" in read
    assert project / ".mcp.json" in read
    assert project / ".claude" / "settings.json" in read


def test_claude_config_dir_moves_everything(tmp_path: Path) -> None:
    config = tmp_path / "claude-work"
    _write(config / ".claude.json", {"mcpServers": {"moved": _http("https://moved.example")}})
    _write(tmp_path / "home" / ".claude.json", {"mcpServers": {"ignored": _http("https://i.ex")}})
    ctx = ClientContext(
        home=tmp_path / "home", cwd=tmp_path, env={"CLAUDE_CONFIG_DIR": str(config)}
    )
    names = [c.original_name for c in ClaudeCodeSource().read(ctx).candidates]
    assert names == ["moved"]


def test_detect_and_an_empty_machine(tmp_path: Path) -> None:
    ctx = ClientContext(home=tmp_path, cwd=tmp_path, env={})
    assert not ClaudeCodeSource().detect(ctx)
    reading = ClaudeCodeSource().read(ctx)
    assert reading.candidates == () and reading.problems == ()
    (tmp_path / ".claude").mkdir()
    assert ClaudeCodeSource().detect(ctx)


def test_registry_lookup_and_order() -> None:
    assert source("claude-code").label == "Claude Code"
    with pytest.raises(KeyError, match="no MCP config reader for 'nope'"):
        source("nope")
    assert ordered_sources("claude-code")[0].id == "claude-code"


def test_current_context_honors_the_isolation_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """conftest moves ROTE_CLIENT_HOME_DIR for every test, so this proves
    the suite never reads the developer's AI clients."""
    import os

    assert Path(os.environ["ROTE_CLIENT_HOME_DIR"]).is_relative_to(tmp_path)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/real/.claude")
    ctx = ClientContext.current()
    assert ctx.home == Path(os.environ["ROTE_CLIENT_HOME_DIR"])
    assert "CLAUDE_CONFIG_DIR" not in ctx.env  # would lead back to the real files


ZAPIER_SHAPED = "https://actions.zapier.com/mcp/sk-ak-9fQ2xLm4Rt7Vb8Np3Kd6Wz1Hc5Jy0Ts/sse"


def _one_server_machine(tmp_path: Path, servers: dict[str, Any], **settings: Any) -> ClientContext:
    home = tmp_path / "home"
    _write(home / ".claude.json", {"mcpServers": servers})
    if settings:
        _write(home / ".claude" / "settings.json", settings)
    return ClientContext(home=home, cwd=tmp_path, env={})


def test_a_credential_shaped_url_path_needs_an_opt_in(tmp_path: Path) -> None:
    ctx = _one_server_machine(
        tmp_path,
        {
            "zapier": {"type": "sse", "url": ZAPIER_SHAPED},
            "figma": _http("https://mcp.figma.com/mcp"),
        },
    )
    found = {c.original_name: c for c in ClaudeCodeSource().read(ctx).candidates}
    zapier = found["zapier"]
    assert zapier.importable  # the user may still opt in, by name
    assert zapier.risk is not None and "may be an API key" in zapier.risk
    assert not zapier.preselectable  # never a default, never --yes
    assert found["figma"].risk is None and found["figma"].preselectable
    shown = json.dumps(zapier.to_dict())
    assert "9fQ2xLm4" not in shown and zapier.to_dict()["host"] == "actions.zapier.com"


def test_denied_servers_are_blocked_by_name_or_url(tmp_path: Path) -> None:
    ctx = _one_server_machine(
        tmp_path,
        {
            "named": _http("https://named.example.com/mcp"),
            "staging": _http("https://staging.example.com:8443/api"),
            "fine": _http("https://fine.example.com/mcp"),
        },
        deniedMcpServers=[
            {"serverName": "named"},
            {"serverUrl": "https://staging.example.com/*"},
            {"serverCommand": ["npx", "x"]},
        ],
    )
    found = {c.original_name: c for c in ClaudeCodeSource().read(ctx).candidates}
    assert found["named"].blocked is not None and "serverName" in found["named"].blocked
    # Claude Code would match port 443 only; rote reads the deny wider.
    assert found["staging"].blocked is not None and "serverUrl" in found["staging"].blocked
    assert found["fine"].blocked is None


def test_managed_policy_files_are_read_from_the_system_dir(tmp_path: Path) -> None:
    ctx = _one_server_machine(tmp_path, {"fine": _http("https://fine.example.com/mcp")})
    system = tmp_path / "system"
    _write(
        system / "ClaudeCode" / "managed-settings.json",
        {
            "deniedMcpServers": [{"serverUrl": "https://*.example.com/*"}],
            "managedMcpServers": {"org": _http("https://org.example.com/mcp")},
        },
    )
    mac = ClientContext(home=ctx.home, cwd=ctx.cwd, env={}, system_dir=system, platform="darwin")
    reading = ClaudeCodeSource().read(mac)
    [fine] = reading.candidates
    assert fine.blocked is not None and "deniedMcpServers" in fine.blocked
    assert any("managedMcpServers" in p and "not read yet" in p for p in reading.problems)

    # Linux keeps it under /etc/claude-code; the macOS dir is not read there.
    linux = ClientContext(home=ctx.home, cwd=ctx.cwd, env={}, system_dir=system, platform="linux")
    assert ClaudeCodeSource().read(linux).candidates[0].blocked is None
    _write(system / "claude-code" / "managed-mcp.json", {"mcpServers": {}})
    exclusive = ClaudeCodeSource().read(linux)
    assert exclusive.candidates[0].inactive == (
        "Claude Code loads only managed-mcp.json servers on this machine"
    )
    assert any("managed-mcp.json" in p for p in exclusive.problems)


def test_every_reading_states_what_it_does_not_read(tmp_path: Path) -> None:
    reading = ClaudeCodeSource().read(ClientContext(home=tmp_path, cwd=tmp_path, env={}))
    assert any("MDM" in limit and "server-managed" in limit for limit in reading.limits)
    assert any("allowedMcpServers is not evaluated" in limit for limit in reading.limits)
    assert reading.to_dict()["limits"] == list(reading.limits)


def test_the_isolated_context_reads_no_system_dir() -> None:
    assert ClientContext.current().system_dir is None
