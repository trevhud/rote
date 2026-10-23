"""Goose's config.yaml and Amp's settings, from fixtures shaped like the documented ones."""

from __future__ import annotations

from pathlib import Path

import pytest

from rote.clients import ClientContext
from rote.clients.amp import AmpSource
from rote.clients.goose import GooseSource
from tests._clients import (
    CLIENT_ID,
    CLIENT_SECRET,
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

GOOSE = f"""
GOOSE_PROVIDER: anthropic
extensions:
  developer:
    enabled: true
    type: builtin
    name: developer
  github:
    enabled: true
    type: streamable_http
    name: GitHub
    uri: https://api.githubcopilot.com/mcp/
    env_keys: [GITHUB_TOKEN]
    headers:
      Authorization: Bearer ${{GITHUB_TOKEN}}
  inline:
    enabled: true
    type: streamable_http
    uri: https://i.example.com/mcp
    envs:
      API_KEY: {SECRET}
    headers:
      X-Key: $API_KEY
  stray:
    enabled: true
    type: streamable_http
    uri: https://s.example.com/mcp
    headers:
      Authorization: Bearer ${{NOT_DECLARED}}
  oauth:
    enabled: true
    type: streamable_http
    uri: https://o.example.com/mcp
    client_id: {CLIENT_ID}
    client_secret_key: MY_SECRET_NAME
    scopes: [read]
  old-sse:
    enabled: true
    type: sse
    uri: https://old.example.com/sse
  off:
    enabled: false
    type: streamable_http
    uri: https://off.example.com/mcp
  local:
    enabled: true
    type: stdio
    cmd: npx
    args: [pkg]
    envs:
      K: {SECRET}
  socketed:
    enabled: true
    type: streamable_http
    uri: http://localhost/mcp
    socket: /tmp/goose.sock
"""


def _goose(tmp_path: Path, **env: str) -> ClientContext:
    home = tmp_path / "home"
    write_text(home / ".config" / "goose" / "config.yaml", GOOSE)
    return ClientContext(home=home, cwd=tmp_path, env=env)


def test_goose_entries(tmp_path: Path) -> None:
    reading = GooseSource().read(_goose(tmp_path))
    found = by_name(reading)
    github = found[("user", "github")]
    assert [h.template for h in github.headers] == ["Bearer ${GITHUB_TOKEN}"]
    assert any("env_keys (maybe Goose's keyring)" in w for w in github.withheld)
    inline = found[("user", "inline")]
    assert [h.template for h in inline.headers] == ["${API_KEY}"]
    assert any("an inline envs value" in w for w in inline.withheld)
    # Goose fills only envs and env_keys, so this reference is sent as written.
    stray = found[("user", "stray")]
    assert stray.headers == () and any(
        "NOT_DECLARED is sent as written" in w for w in stray.withheld
    )
    oauth = found[("user", "oauth")]
    assert oauth.scopes == ("read",)
    assert any(w.startswith("client_id: issued to Goose") for w in oauth.withheld)
    assert any(w.startswith("client_secret_key") for w in oauth.withheld)
    old = found[("user", "old-sse")]
    assert old.transport == "sse" and old.importable and old.inactive is not None
    assert found[("user", "off")].inactive == "disabled in Goose (enabled: false)"
    assert found[("user", "local")].transport == "stdio"
    assert "builtin" in (found[("user", "developer")].blocked or "")
    assert "Unix socket" in (found[("user", "socketed")].blocked or "")
    assert_display_safe(reading, "MY_SECRET_NAME")


def test_goose_layers_merge_per_field(tmp_path: Path) -> None:
    extra = tmp_path / "team.yaml"
    write_text(
        extra,
        "extensions:\n  team:\n    enabled: true\n    type: streamable_http\n"
        "    uri: https://team.example.com/mcp\n",
    )
    etc = tmp_path / "etc"
    write_text(
        etc / "goose" / "config.yaml",
        "extensions:\n  github:\n    enabled: false\n    timeout: 300\n",
    )
    ctx = _goose(tmp_path, GOOSE_ADDITIONAL_CONFIG_FILES=str(extra))
    ctx = ClientContext(ctx.home, ctx.cwd, ctx.env, etc_dir=etc)
    found = by_name(GooseSource().read(ctx))
    assert found[("system", "team")].importable
    # The user file's enabled: true wins over the system's false; the
    # system's other fields merge in.
    assert found[("user", "github")].inactive is None


def test_goose_path_root_and_malformed(tmp_path: Path) -> None:
    root = tmp_path / "goose-root"
    write_text(root / "config" / "config.yaml", "extensions: [unclosed")
    reading = GooseSource().read(_goose(tmp_path, GOOSE_PATH_ROOT=str(root)))
    assert any("config.yaml: could not read it" in p for p in reading.problems)
    assert reading.candidates == ()


AMP_USER = f"""{{
  // Amp settings are JSONC
  "amp.mcpServers": {{
    "linear": {{ "url": "https://mcp.linear.app/mcp", "transport": "sse",
                 "headers": {{ "Authorization": "Bearer ${{LINEAR_TOKEN}}" }} }},
    "idtoken": {{ "url": "https://orb.example.com/mcp",
                  "headers": {{ "Authorization": "Bearer ${{amp:id-token}}" }} }},
    "literal": {{ "url": "https://l.example.com/mcp", "headers": {{ "X-Key": "{SECRET}" }} }},
    "oauth": {{ "url": "https://o.example.com/mcp",
                "oauth": {{ "clientId": "{CLIENT_ID}", "clientSecret": "{CLIENT_SECRET}",
                            "authUrl": "https://o.example.com/a", "tokenUrl": "https://o.example.com/t",
                            "scopes": ["read"] }} }},
    "local": {{ "command": "npx", "args": ["pkg"], "env": {{ "K": "{SECRET}" }} }},
    "shared": {{ "url": "https://user.example.com/mcp" }},
    "rejected": {{ "url": "https://blocked.example.com/mcp" }},
  }},
  "amp.mcpTrustedServers": [
    {{ "serverName": "approved", "allow": true }},
    {{ "serverName": "denied", "allow": false }}
  ],
  "amp.mcpPermissions": [
    {{ "matches": {{ "url": "https://blocked.example.com/*" }}, "action": "reject" }}
  ],
}}"""


def _amp(tmp_path: Path) -> ClientContext:
    home = tmp_path / "home"
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    write_text(home / ".config" / "amp" / "settings.json", AMP_USER)
    write_json(
        repo / ".amp" / "settings.json",
        {
            "amp.mcpServers": {
                "approved": {"url": "https://approved.example.com/mcp"},
                "denied": {"url": "https://denied.example.com/mcp"},
                "shared": {"url": "https://workspace.example.com/mcp"},
            }
        },
    )
    (repo / "src").mkdir()
    return ClientContext(home=home, cwd=repo / "src", env={})


def test_amp_workspace_approvals_are_never_taken_as_proof(tmp_path: Path) -> None:
    found = by_name(AmpSource().read(_amp(tmp_path)))
    # Amp does not document how an approval matches an entry, so an approval
    # by name is not verifiable: unknown, listed, never preselected.
    assert_trust_unknown(found[("project", "approved")], "Amp")
    assert "amp mcp approve approved" in (found[("project", "approved")].trust_note or "")
    # No approval at all reads the same. It still replaces the user's entry.
    assert_trust_unknown(found[("project", "shared")], "Amp")
    assert found[("user", "shared")].inactive == "shadowed by the workspace entry of the same name"
    # A denial is plainly in the settings file.
    denied = found[("project", "denied")]
    assert denied.trust == "untrusted"
    assert denied.inactive == "denied in Amp for this workspace (allow: false in its settings)"
    assert found[("user", "linear")].trust == "trusted"  # user servers need no approval


def test_amp_workspace_allowlist_is_not_read_either(tmp_path: Path) -> None:
    ctx = _amp(tmp_path)
    settings = ctx.home / ".config" / "amp" / "settings.json"
    text = settings.read_text().replace(
        '"amp.mcpPermissions"',
        f'"amp.workspaces": [{{"path": "{tmp_path / "repo"}", "allowAllMcpServers": true}}],\n'
        '  "amp.mcpPermissions"',
    )
    write_text(settings, text)
    found = by_name(AmpSource().read(ctx))
    assert found[("project", "shared")].trust == "unknown"


def test_amp_entries(tmp_path: Path) -> None:
    reading = AmpSource().read(_amp(tmp_path))
    found = by_name(reading)
    linear = found[("user", "linear")]
    assert linear.transport == "sse" and [h.template for h in linear.headers] == [
        "Bearer ${LINEAR_TOKEN}"
    ]
    idtoken = found[("user", "idtoken")]
    assert idtoken.headers == () and any("${amp:id-token}" in w for w in idtoken.withheld)
    oauth = found[("user", "oauth")]
    assert oauth.scopes == ("read",) and any(w.startswith("oauth.clientId") for w in oauth.withheld)
    assert found[("user", "local")].transport == "stdio"
    assert found[("user", "rejected")].blocked == "rejected by amp.mcpPermissions"
    assert_display_safe(reading)


def test_amp_settings_file_override_and_malformed(tmp_path: Path) -> None:
    ctx = _amp(tmp_path)
    moved = write_text(tmp_path / "custom.json", '{"amp.mcpServers": {')
    reading = AmpSource().read(ClientContext(ctx.home, ctx.cwd, {"AMP_SETTINGS_FILE": str(moved)}))
    assert any("custom.json: could not read it" in p for p in reading.problems)
    # The workspace still reads.
    assert ("project", "approved") in by_name(reading)
