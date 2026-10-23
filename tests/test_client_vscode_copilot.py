"""VS Code's mcp.json files and Copilot CLI's, from fixtures shaped like the documented ones."""

from __future__ import annotations

from pathlib import Path

import pytest

from rote.clients import ClientContext
from rote.clients.copilot import CopilotSource
from rote.clients.vscode import VSCodeSource
from tests._clients import (
    CLIENT_ID,
    SECRET,
    assert_display_safe,
    assert_trust_unknown,
    by_name,
    refuse_real_home,
    write_json,
    write_text,
)

__all__ = ["refuse_real_home"]
pytestmark = pytest.mark.usefixtures("refuse_real_home")

WORKSPACE = f"""{{
  "inputs": [{{ "type": "promptString", "id": "api-key", "password": true }}],
  "servers": {{
    "github": {{
      "type": "http",
      "url": "https://api.githubcopilot.com/mcp/",
      "headers": {{ "Authorization": "Bearer ${{env:GITHUB_TOKEN}}" }},
    }},
    "prompted": {{
      "type": "http",
      "url": "https://p.example.com/mcp",
      "headers": {{ "X-Key": "${{input:api-key}}" }}
    }},
    "Shared": {{ "type": "sse", "url": "https://ws-shared.example.com/sse" }},
    "inferred": {{ "url": "https://inferred.example.com/mcp" }},
    "local": {{ "command": "node", "args": ["server.js"] }},
    "socket": {{ "type": "http", "url": "unix:///tmp/mcp.sock" }},
    "sock": {{ "type": "ws", "url": "wss://w.example.com" }},
    "oauthed": {{ "type": "http", "url": "https://o.example.com/mcp",
                  "oauth": {{ "clientId": "{CLIENT_ID}" }} }}
  }}
}}
"""


def _vscode_machine(tmp_path: Path, platform: str = "darwin") -> ClientContext:
    home = tmp_path / "home"
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    write_text(repo / ".vscode" / "mcp.json", WORKSPACE)
    write_json(
        repo / ".mcp.json",
        {
            "portable": {
                "type": "http",
                "url": "https://portable.example.com/mcp",
                "headers": {"Authorization": "Bearer $PORTABLE_TOKEN"},
            }
        },
    )
    user_dir = {
        "darwin": home / "Library" / "Application Support" / "Code" / "User",
        "linux": home / ".config" / "Code" / "User",
        "win32": home / "AppData" / "Roaming" / "Code" / "User",
    }[platform]
    write_json(
        user_dir / "mcp.json",
        {
            "servers": {
                "shared": {"type": "http", "url": "https://user-shared.example.com/mcp"},
                "literal": {
                    "type": "http",
                    "url": "https://l.example.com/mcp",
                    "headers": {"X-Key": SECRET},
                },
            }
        },
    )
    write_text(user_dir / "settings.json", '{\n  // jsonc\n  "mcp": {"servers": {}},\n}')
    return ClientContext(home=home, cwd=repo, env={}, platform=platform)


@pytest.mark.parametrize("platform", ["darwin", "linux", "win32"])
def test_vscode_user_file_per_os(tmp_path: Path, platform: str) -> None:
    found = by_name(VSCodeSource().read(_vscode_machine(tmp_path, platform)))
    assert ("user", "literal") in found


def test_vscode_precedence_is_case_insensitive(tmp_path: Path) -> None:
    found = by_name(VSCodeSource().read(_vscode_machine(tmp_path)))
    assert found[("project", "Shared")].inactive is None
    assert found[("project", "Shared")].transport == "sse"
    assert found[("user", "shared")].inactive == "shadowed by the workspace entry of the same name"


def test_vscode_placeholders_inputs_and_shapes(tmp_path: Path) -> None:
    reading = VSCodeSource().read(_vscode_machine(tmp_path))
    found = by_name(reading)
    assert [h.template for h in found[("project", "github")].headers] == ["Bearer ${GITHUB_TOKEN}"]
    prompted = found[("project", "prompted")]
    assert prompted.headers == ()
    assert any("${input:api-key}" in w for w in prompted.withheld)
    assert found[("project", "inferred")].transport == "streamable-http"
    assert found[("project", "local")].transport == "stdio"
    assert "scheme is unix" in (found[("project", "socket")].blocked or "")
    assert found[("project", "sock")].transport == "websocket"
    assert any(
        w.startswith("oauth.clientId: issued to VS Code")
        for w in found[("project", "oauthed")].withheld
    )
    portable = found[("project", "portable")]
    assert [h.template for h in portable.headers] == ["Bearer ${PORTABLE_TOKEN}"]
    assert any("legacy mcp key" in p for p in reading.problems)
    for key in (("project", "github"), ("project", "portable")):
        assert_trust_unknown(found[key], "VS Code")
    assert found[("user", "literal")].trust == "trusted"
    assert_display_safe(reading)


def test_vscode_access_none_turns_every_server_off(tmp_path: Path) -> None:
    ctx = _vscode_machine(tmp_path)
    settings = ctx.home / "Library" / "Application Support" / "Code" / "User" / "settings.json"
    write_json(settings, {"chat.mcp.access": "none"})
    reading = VSCodeSource().read(ctx)
    assert all(c.inactive is not None for c in reading.candidates)


def test_vscode_portable_moves_the_user_dir(tmp_path: Path) -> None:
    ctx = _vscode_machine(tmp_path)
    portable = tmp_path / "portable"
    write_json(
        portable / "user-data" / "User" / "mcp.json",
        {"servers": {"moved": {"type": "http", "url": "https://m.example.com/mcp"}}},
    )
    moved = ClientContext(ctx.home, ctx.cwd, {"VSCODE_PORTABLE": str(portable)}, platform="darwin")
    found = by_name(VSCodeSource().read(moved))
    assert ("user", "moved") in found and ("user", "literal") not in found


def _copilot_machine(tmp_path: Path, **env: str) -> ClientContext:
    home = tmp_path / "home"
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    cwd = repo / "pkg"
    copilot = Path(env["COPILOT_HOME"]) if "COPILOT_HOME" in env else home / ".copilot"
    write_json(
        copilot / "mcp-config.json",
        {
            "mcpServers": {
                "github": {
                    "type": "http",
                    "url": "https://api.githubcopilot.com/mcp/",
                    "headers": {"Authorization": "Bearer ${GH_TOKEN:-" + SECRET + "}"},
                    "tools": ["*"],
                },
                "local": {"command": "npx", "args": ["pkg"], "tools": ["*"]},
                "untyped": {"url": "https://u.example.com/mcp", "tools": ["*"]},
                "filtered": {
                    "type": "sse",
                    "url": "https://f.example.com/sse",
                    "tools": ["one"],
                    "oauthClientId": CLIENT_ID,
                    "oauthScopes": ["repo"],
                },
                "turned-off": {"type": "http", "url": "https://off.example.com/mcp"},
                "shared": {"type": "http", "url": "https://user.example.com/mcp"},
            }
        },
    )
    write_text(copilot / "settings.json", '{"disabledMcpServers": ["turned-off"], // jsonc\n}')
    write_json(
        repo / ".mcp.json", {"shared": {"type": "http", "url": "https://root.example.com/mcp"}}
    )
    write_json(
        cwd / ".github" / "mcp.json",
        {"mcpServers": {"shared": {"type": "http", "url": "https://closer.example.com/mcp"}}},
    )
    return ClientContext(home=home, cwd=cwd, env=env)


def test_copilot_closer_project_files_win(tmp_path: Path) -> None:
    reading = CopilotSource().read(_copilot_machine(tmp_path))
    shared = [c for c in reading.candidates if c.original_name == "shared"]
    assert_trust_unknown(shared[0], "GitHub Copilot CLI")
    assert shared[2].trust == "trusted"
    assert [(c.host, c.inactive is None) for c in shared] == [
        ("closer.example.com", True),
        ("root.example.com", False),
        ("user.example.com", False),
    ]


def test_copilot_shapes_placeholders_and_disabled(tmp_path: Path) -> None:
    reading = CopilotSource().read(_copilot_machine(tmp_path))
    found = by_name(reading)
    github = found[("user", "github")]
    assert [h.template for h in github.headers] == ["Bearer ${GH_TOKEN}"]
    assert any("default for ${GH_TOKEN} was not copied" in w for w in github.withheld)
    assert not any(w.startswith("tools:") for w in github.withheld)
    assert found[("user", "local")].transport == "stdio"
    assert "no type" in (found[("user", "untyped")].blocked or "")
    filtered = found[("user", "filtered")]
    assert filtered.transport == "sse" and filtered.scopes == ("repo",)
    assert any(w.startswith("tools:") for w in filtered.withheld)
    assert found[("user", "turned-off")].inactive == "disabled in Copilot CLI (copilot mcp disable)"
    assert_display_safe(reading)


def test_copilot_home_moves_everything(tmp_path: Path) -> None:
    moved = tmp_path / "copilot-home"
    reading = CopilotSource().read(_copilot_machine(tmp_path, COPILOT_HOME=str(moved)))
    assert moved / "mcp-config.json" in reading.files_read


def test_copilot_malformed_user_file(tmp_path: Path) -> None:
    ctx = _copilot_machine(tmp_path)
    write_text(ctx.home / ".copilot" / "mcp-config.json", "{,}")
    reading = CopilotSource().read(ctx)
    assert any(
        "mcp-config.json: could not read it (JSONDecodeError)" in p for p in reading.problems
    )
