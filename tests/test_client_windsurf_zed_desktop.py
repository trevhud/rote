"""Windsurf / Devin Desktop, Zed and Claude Desktop, from fixtures shaped like their docs."""

from __future__ import annotations

from pathlib import Path

import pytest

from rote.clients import ClientContext
from rote.clients.claude_desktop import ClaudeDesktopSource
from rote.clients.windsurf import WindsurfSource
from rote.clients.zed import ZedSource
from tests._clients import (
    CLIENT_ID,
    CLIENT_SECRET,
    SECRET,
    assert_display_safe,
    by_name,
    refuse_real_home,
    write_json,
    write_text,
)

__all__ = ["refuse_real_home"]
pytestmark = pytest.mark.usefixtures("refuse_real_home")


def _repo(tmp_path: Path) -> tuple[Path, Path]:
    home = tmp_path / "home"
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    home.mkdir()
    return home, repo


def _windsurf(tmp_path: Path) -> ClientContext:
    home, repo = _repo(tmp_path)
    write_text(
        home / ".config" / "devin" / "mcp_config.json",
        f"""{{
  // Devin's config files allow comments
  "mcpServers": {{
    "github": {{ "serverUrl": "https://api.githubcopilot.com/mcp/",
                 "headers": {{ "Authorization": "Bearer ${{env:GH_TOKEN}}" }} }},
    "filed": {{ "url": "https://f.example.com/mcp",
                "headers": {{ "X-Key": "${{file:/secrets/key}}" }} }},
    "literal": {{ "url": "https://l.example.com/mcp", "headers": {{ "X-Key": "{SECRET}" }} }},
    "events": {{ "url": "https://e.example.com/sse", "transport": "sse",
                 "oauthClientId": "{CLIENT_ID}", "oauthClientSecret": "{CLIENT_SECRET}" }},
    "local": {{ "command": "npx", "args": ["pkg"] }},
    "shared": {{ "url": "https://user.example.com/mcp" }}
  }}
}}""",
    )
    write_json(
        repo / ".devin" / "mcp_config.json",
        {"mcpServers": {"shared": {"url": "https://project.example.com/mcp"}}},
    )
    write_json(
        repo / ".devin" / "mcp_config.local.json",
        {"mcpServers": {"shared": {"url": "https://local.example.com/mcp", "disabled": True}}},
    )
    write_json(
        home / ".codeium" / "windsurf" / "mcp_config.json",
        {"mcpServers": {"old": {"serverUrl": "https://old.example.com/mcp"}}},
    )
    return ClientContext(home=home, cwd=repo, env={})


def test_windsurf_scopes_precedence_and_disabled(tmp_path: Path) -> None:
    reading = WindsurfSource().read(_windsurf(tmp_path))
    shared = [c for c in reading.candidates if c.original_name == "shared"]
    # A disabled local entry still wins by name: the server is off.
    assert [(c.scope, c.inactive) for c in shared] == [
        ("local", "disabled in Devin/Windsurf (disabled: true)"),
        ("project", "shadowed by the local entry of the same name"),
        ("user", "shadowed by the local entry of the same name"),
    ]
    old = by_name(reading)[("user", "old")]
    assert old.importable and old.host == "old.example.com"


def test_windsurf_placeholders_and_secrets(tmp_path: Path) -> None:
    reading = WindsurfSource().read(_windsurf(tmp_path))
    found = by_name(reading)
    assert [h.template for h in found[("user", "github")].headers] == ["Bearer ${GH_TOKEN}"]
    assert found[("user", "filed")].headers == ()
    assert any("${file:/secrets/key}" in w for w in found[("user", "filed")].withheld)
    events = found[("user", "events")]
    assert events.transport == "sse"
    assert any(w.startswith("oauthClientId") for w in events.withheld)
    assert found[("user", "local")].transport == "stdio"
    assert_display_safe(reading)


def test_windsurf_xdg_and_windows_paths(tmp_path: Path) -> None:
    home, repo = _repo(tmp_path)
    xdg = tmp_path / "xdg"
    write_json(
        xdg / "devin" / "mcp_config.json", {"mcpServers": {"a": {"url": "https://a.example"}}}
    )
    ctx = ClientContext(home=home, cwd=repo, env={"XDG_CONFIG_HOME": str(xdg)})
    assert ("user", "a") in by_name(WindsurfSource().read(ctx))
    appdata = tmp_path / "appdata"
    write_json(
        appdata / "devin" / "mcp_config.json", {"mcpServers": {"w": {"url": "https://w.example"}}}
    )
    win = ClientContext(home=home, cwd=repo, env={"APPDATA": str(appdata)}, platform="win32")
    assert ("user", "w") in by_name(WindsurfSource().read(win))


def _zed(tmp_path: Path, platform: str = "darwin") -> ClientContext:
    home, repo = _repo(tmp_path)
    settings = {
        "darwin": home / ".config" / "zed" / "settings.json",
        "linux": home / ".config" / "zed" / "settings.json",
        "win32": home / "AppData" / "Roaming" / "Zed" / "settings.json",
    }[platform]
    write_text(
        settings,
        f"""// Zed settings
{{
  "theme": "One Dark",
  "context_servers": {{
    "remote": {{
      "url": "https://mcp.example.com/mcp",
      "headers": {{ "Authorization": "Bearer ${{TOKEN}}" }},
      "oauth": {{ "client_id": "{CLIENT_ID}", "client_secret": "{CLIENT_SECRET}" }},
    }},
    "off": {{ "url": "https://off.example.com/mcp", "enabled": false }},
    "cmd": {{ "command": "npx", "args": ["pkg"], "env": {{ "K": "{SECRET}" }} }},
    "legacy": {{ "source": "custom", "command": {{ "path": "node", "args": [] }} }},
    "ext": {{ "settings": {{ "key": "{SECRET}" }} }},
    "shared": {{ "url": "https://user.example.com/mcp" }},
  }},
}}""",
    )
    write_json(
        repo / ".zed" / "settings.json",
        {
            "context_servers": {
                "shared": {"url": "https://project.example.com/mcp", "enabled": False}
            }
        },
    )
    return ClientContext(home=home, cwd=repo, env={}, platform=platform)


@pytest.mark.parametrize("platform", ["darwin", "linux", "win32"])
def test_zed_per_os_and_shapes(tmp_path: Path, platform: str) -> None:
    reading = ZedSource().read(_zed(tmp_path, platform))
    found = by_name(reading)
    remote = found[("user", "remote")]
    assert remote.importable and remote.transport == "streamable-http"
    # Zed expands nothing: ${TOKEN} would be sent as written, so it is a literal.
    assert remote.headers == () and "a literal value" in remote.withheld[0]
    assert any(w.startswith("oauth.client_id") for w in remote.withheld)
    assert found[("user", "off")].inactive == "turned off in Zed (enabled: false)"
    assert found[("user", "cmd")].transport == "stdio"
    assert found[("user", "legacy")].transport == "stdio"
    assert "extension" in (found[("user", "ext")].blocked or "")
    assert found[("project", "shared")].inactive == "turned off in Zed (enabled: false)"
    assert found[("user", "shared")].inactive == "shadowed by the project entry of the same name"
    assert_display_safe(reading)


def test_zed_linux_honors_xdg_but_macos_does_not(tmp_path: Path) -> None:
    home, repo = _repo(tmp_path)
    xdg = tmp_path / "xdg"
    write_json(
        xdg / "zed" / "settings.json", {"context_servers": {"x": {"url": "https://x.example"}}}
    )
    env = {"XDG_CONFIG_HOME": str(xdg)}
    linux = ClientContext(home=home, cwd=repo, env=env, platform="linux")
    mac = ClientContext(home=home, cwd=repo, env=env, platform="darwin")
    assert ("user", "x") in by_name(ZedSource().read(linux))
    assert ZedSource().read(mac).candidates == ()


def test_zed_malformed_settings(tmp_path: Path) -> None:
    ctx = _zed(tmp_path)
    write_text(tmp_path / "repo" / ".zed" / "settings.json", "{ context_servers: [ }")
    reading = ZedSource().read(ctx)
    assert any(".zed/settings.json: could not read it" in p for p in reading.problems)


def test_claude_desktop_lists_every_entry_as_not_importable(tmp_path: Path) -> None:
    home, repo = _repo(tmp_path)
    config = home / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    write_json(
        config,
        {
            "mcpServers": {
                "files": {"command": "npx", "args": ["fs"], "env": {"K": SECRET}},
                "remote-try": {"url": "https://r.example.com/mcp"},
            },
            "preferences": {"x": True},
        },
    )
    reading = ClaudeDesktopSource().read(
        ClientContext(home=home, cwd=repo, env={}, platform="darwin")
    )
    found = by_name(reading)
    assert found[("user", "files")].transport == "stdio"
    assert "connector" in (found[("user", "remote-try")].blocked or "")
    assert not any(c.importable for c in reading.candidates)
    assert any("claude.ai account" in limit for limit in reading.limits)
    assert_display_safe(reading)
    win = ClientContext(
        home=home, cwd=repo, env={"APPDATA": str(tmp_path / "ad")}, platform="win32"
    )
    assert ClaudeDesktopSource().config_paths(win) == (
        tmp_path / "ad" / "Claude" / "claude_desktop_config.json",
    )
    write_text(config, "not json")
    broken = ClaudeDesktopSource().read(
        ClientContext(home=home, cwd=repo, env={}, platform="darwin")
    )
    assert any("could not read it (JSONDecodeError)" in p for p in broken.problems)
