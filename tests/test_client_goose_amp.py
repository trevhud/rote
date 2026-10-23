"""Goose's config.yaml and Amp's settings, from fixtures shaped like the documented ones."""

from __future__ import annotations

from pathlib import Path

import pytest

from rote.clients import ClientContext
from rote.clients.amp import AmpSource, spec_hash
from rote.clients.goose import GooseSource
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


#: Computed with Node 22's JSON.stringify and crypto over Amp's own U3i/jht
#: (recovered from the CLI bundle), not with rote's port, so these pin the
#: port to the JavaScript it copies.
APPROVED_HASH = "529bb205000048b739254a326d2e5caefd52993237829b210b07b4634ff2b84b"
DENIED_HASH = "f00a807cf4a039c64c79ba7232bd689437ee96091cc068803868857bdf7a7b99"

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
    {{ "serverName": "approved", "specHash": "{APPROVED_HASH}", "allow": true }},
    {{ "serverName": "denied", "specHash": "{DENIED_HASH}", "allow": false }},
    {{ "serverName": "edited", "specHash": "{APPROVED_HASH}", "allow": true }}
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
                # Approved under the hash of another entry: Amp asks again.
                "edited": {"url": "https://edited.example.com/mcp"},
            }
        },
    )
    (repo / "src").mkdir()
    return ClientContext(home=home, cwd=repo / "src", env={})


@pytest.mark.parametrize(
    ("entry", "digest"),
    [
        ({"url": "https://approved.example.com/mcp"}, APPROVED_HASH),
        ({"url": "https://denied.example.com/mcp"}, DENIED_HASH),
        (
            {
                "url": "https://h.example.com/mcp",
                "transport": "sse",
                "headers": {"X-B": "${B}", "Authorization": "Bearer ${T}", "é": "ü"},
            },
            "11938c5ea1e638681f96347c82087fc78bec8ca90d824993228a7fe954027c93",
        ),
        (
            {"command": "npx", "args": ["-y", "pkg"], "env": {"Z": "1", "A": "2"}},
            "4e31e6ceff2f47494f44fd72054500c3e378787fd13ef3b218b18dace90bd6b1",
        ),
    ],
)
def test_amp_spec_hash_matches_the_javascript(entry: dict[str, object], digest: str) -> None:
    assert spec_hash(entry) == digest


def test_amp_workspace_approval_is_verified_by_hash(tmp_path: Path) -> None:
    found = by_name(AmpSource().read(_amp(tmp_path)))
    approved = found[("project", "approved")]
    assert approved.inactive is None and approved.trust == "trusted" and approved.preselectable
    denied = found[("project", "denied")]
    assert denied.inactive == "denied in Amp for this workspace" and denied.trust == "untrusted"
    # The name is approved, but for another entry: not verifiable as approved.
    edited = found[("project", "edited")]
    assert edited.trust == "unknown" and not edited.preselectable
    assert edited.trust_note is not None
    assert edited.trust_note.startswith("trust in Amp could not be checked: ")
    assert "earlier version of this entry" in edited.trust_note
    # No local approval: only ampcode.com could say. It still replaces the
    # user's entry by name.
    pending = found[("project", "shared")]
    assert pending.trust == "unknown" and "amp mcp approve shared" in (pending.trust_note or "")
    assert pending.needs_opt_in == (pending.trust_note,)
    assert found[("user", "shared")].inactive == "shadowed by the workspace entry of the same name"
    assert found[("user", "linear")].trust == "trusted"  # user servers need no approval


def test_amp_trusting_the_whole_workspace(tmp_path: Path) -> None:
    ctx = _amp(tmp_path)
    settings = ctx.home / ".config" / "amp" / "settings.json"
    text = settings.read_text().replace(
        '"amp.mcpPermissions"',
        f'"amp.workspaces": [{{"path": "{tmp_path / "repo"}", "allowAllMcpServers": true}}],\n'
        '  "amp.mcpPermissions"',
    )
    write_text(settings, text)
    found = by_name(AmpSource().read(ctx))
    for name in ("shared", "denied", "edited"):
        assert found[("project", name)].inactive is None
        assert found[("project", name)].trust == "trusted", name


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
