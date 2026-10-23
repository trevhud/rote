"""Gemini CLI's settings and OpenCode's layered config, from fixtures shaped like the source's."""

from __future__ import annotations

from pathlib import Path

import pytest

from rote.clients import ClientContext
from rote.clients.gemini import GeminiSource
from rote.clients.opencode import OpenCodeSource
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

GEMINI_USER = f"""{{
  // comments are stripped
  "mcpServers": {{
    "github": {{ "httpUrl": "https://api.githubcopilot.com/mcp/",
                 "headers": {{ "Authorization": "Bearer $GITHUB_TOKEN" }} }},
    "events": {{ "url": "https://e.example.com/sse", "type": "sse" }},
    "untyped": {{ "url": "https://u.example.com/mcp",
                  "headers": {{ "Authorization": "Bearer ${{TOK:-{SECRET}}}" }} }},
    "google": {{ "httpUrl": "https://g.example.com/mcp",
                 "authProviderType": "google_credentials" }},
    "oauth": {{ "url": "https://o.example.com/mcp",
                "oauth": {{ "enabled": true, "clientId": "{CLIENT_ID}",
                            "clientSecret": "{CLIENT_SECRET}", "scopes": ["openid"] }} }},
    "local": {{ "command": "python", "args": ["server.py"], "env": {{ "K": "{SECRET}" }} }},
    "socket": {{ "tcp": "ws://localhost:9000" }},
    "excluded-one": {{ "httpUrl": "https://x.example.com/mcp" }},
    "Toggled": {{ "httpUrl": "https://t.example.com/mcp" }},
    "shared": {{ "httpUrl": "https://user.example.com/mcp" }}
  }},
  "mcp": {{ "excluded": ["Excluded-One"] }}
}}"""


def _gemini(tmp_path: Path, trust: str | None = "TRUST_FOLDER", **env: str) -> ClientContext:
    home = tmp_path / "home"
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    write_text(home / ".gemini" / "settings.json", GEMINI_USER)
    write_json(home / ".gemini" / "mcp-server-enablement.json", {"toggled": {"enabled": False}})
    if trust is not None:
        write_json(home / ".gemini" / "trustedFolders.json", {str(repo): trust})
    write_json(
        repo / ".gemini" / "settings.json",
        {"mcpServers": {"shared": {"httpUrl": "https://workspace.example.com/mcp"}}},
    )
    ext = home / ".gemini" / "extensions" / "acme"
    write_json(
        ext / "gemini-extension.json",
        {
            "name": "acme",
            "mcpServers": {
                "acme": {
                    "httpUrl": "https://acme.example.com/mcp",
                    "headers": {"X-Root": "${extensionPath}${/}x"},
                },
                "shared": {"httpUrl": "https://ext.example.com/mcp"},
            },
        },
    )
    return ClientContext(home=home, cwd=repo, env=env)


def test_gemini_transports_and_placeholders(tmp_path: Path) -> None:
    reading = GeminiSource().read(_gemini(tmp_path))
    found = by_name(reading)
    github = found[("user", "github")]
    assert github.transport == "streamable-http"
    assert [h.template for h in github.headers] == ["Bearer ${GITHUB_TOKEN}"]
    assert found[("user", "events")].transport == "sse"
    untyped = found[("user", "untyped")]
    assert untyped.transport == "streamable-http"
    assert any("default for ${TOK} was not copied" in w for w in untyped.withheld)
    assert any(
        w.startswith("authProviderType google_credentials")
        for w in found[("user", "google")].withheld
    )
    oauth = found[("user", "oauth")]
    assert oauth.scopes == ("openid",)
    assert any(w.startswith("oauth.clientId") for w in oauth.withheld)
    assert found[("user", "local")].transport == "stdio"
    assert found[("user", "socket")].transport == "websocket"
    acme = found[("plugin", "acme")]
    assert acme.plugin == "acme" and acme.headers == ()
    assert any("${extensionPath}" in w for w in acme.withheld)
    assert_display_safe(reading)


def test_gemini_precedence_and_switches(tmp_path: Path) -> None:
    found = by_name(GeminiSource().read(_gemini(tmp_path)))
    assert found[("project", "shared")].inactive is None
    assert found[("user", "shared")].inactive == "shadowed by the workspace entry of the same name"
    assert found[("plugin", "shared")].inactive is not None
    # Names match lowercased and trimmed.
    assert found[("user", "excluded-one")].inactive == "in Gemini CLI's mcp.excluded list"
    assert found[("user", "Toggled")].inactive == "disabled in Gemini CLI (gemini mcp disable)"


@pytest.mark.parametrize("trust", [None, "DO_NOT_TRUST"])
def test_gemini_untrusted_workspace_is_not_loaded(tmp_path: Path, trust: str | None) -> None:
    found = by_name(GeminiSource().read(_gemini(tmp_path, trust=trust)))
    assert "does not trust" in (found[("project", "shared")].inactive or "")
    assert found[("project", "shared")].trust == "untrusted"
    assert found[("user", "shared")].inactive is None


def test_gemini_trust_parent_and_env_override(tmp_path: Path) -> None:
    ctx = _gemini(tmp_path, trust=None)
    write_json(
        ctx.home / ".gemini" / "trustedFolders.json", {str(tmp_path / "repo" / "x"): "TRUST_PARENT"}
    )
    assert by_name(GeminiSource().read(ctx))[("project", "shared")].inactive is None
    restricted = ClientContext(ctx.home, ctx.cwd, {"GEMINI_CLI_TRUST_WORKSPACE": "false"})
    assert by_name(GeminiSource().read(restricted))[("project", "shared")].inactive is not None


def test_gemini_trusted_folder_tilde_is_the_contexts_home(tmp_path: Path) -> None:
    ctx = _gemini(tmp_path, trust=None)
    repo = ctx.home / "code" / "repo"
    (repo / ".git").mkdir(parents=True)
    write_json(
        repo / ".gemini" / "settings.json",
        {"mcpServers": {"here": {"httpUrl": "https://here.example.com/mcp"}}},
    )
    write_json(ctx.home / ".gemini" / "trustedFolders.json", {"~/code/repo": "TRUST_FOLDER"})
    found = by_name(GeminiSource().read(ClientContext(ctx.home, repo, {})))
    assert found[("project", "here")].trust == "trusted"


def test_gemini_allowed_list_and_system_scope(tmp_path: Path) -> None:
    ctx = _gemini(tmp_path)
    system = tmp_path / "sys" / "settings.json"
    write_json(
        system,
        {
            "mcpServers": {"org": {"httpUrl": "https://org.example.com/mcp"}},
            "mcp": {"allowed": ["org", "github"]},
        },
    )
    ctx = ClientContext(ctx.home, ctx.cwd, {"GEMINI_CLI_SYSTEM_SETTINGS_PATH": str(system)})
    found = by_name(GeminiSource().read(ctx))
    assert found[("system", "org")].inactive is None
    assert found[("user", "github")].inactive is None
    assert found[("user", "events")].inactive == "not in Gemini CLI's mcp.allowed list"


def test_gemini_home_override_and_malformed(tmp_path: Path) -> None:
    ctx = _gemini(tmp_path)
    write_text(ctx.home / ".gemini" / "settings.json", "{ nope")
    reading = GeminiSource().read(ctx)
    assert any("settings.json: could not read it" in p for p in reading.problems)
    moved = tmp_path / "gemini-home"
    write_json(
        moved / ".gemini" / "settings.json", {"mcpServers": {"m": {"httpUrl": "https://m.ex"}}}
    )
    found = by_name(
        GeminiSource().read(ClientContext(ctx.home, ctx.cwd, {"GEMINI_CLI_HOME": str(moved)}))
    )
    assert ("user", "m") in found


OPENCODE_GLOBAL = f"""{{
  "$schema": "https://opencode.ai/config.json",
  // JSONC, trailing commas allowed
  "mcp": {{
    "sentry": {{ "type": "remote", "url": "https://mcp.sentry.dev/mcp",
                 "headers": {{ "Authorization": "Bearer {{env:SENTRY_TOKEN}}" }},
                 "enabled": true }},
    "filed": {{ "type": "remote", "url": "https://f.example.com/mcp",
                "headers": {{ "X-Key": "{{file:~/.secrets/key}}" }} }},
    "literal-dollar": {{ "type": "remote", "url": "https://d.example.com/mcp",
                         "headers": {{ "Authorization": "Bearer ${{NOT_EXPANDED}}",
                                       "X-Mixed": "{{env:REAL}}${{NOT_EXPANDED}}" }} }},
    "oauth": {{ "type": "remote", "url": "https://o.example.com/mcp",
                "oauth": {{ "clientId": "{CLIENT_ID}", "clientSecret": "{CLIENT_SECRET}",
                            "scope": "read write" }} }},
    "local": {{ "type": "local", "command": ["npx", "pkg"], "environment": {{ "K": "{SECRET}" }} }},
    "overlaid": {{ "type": "remote", "url": "https://global.example.com/mcp",
                   "headers": {{ "X-Global": "{{env:GLOBAL}}" }} }},
    "notype": {{ "url": "https://n.example.com/mcp" }},
  }},
}}"""


def _opencode(tmp_path: Path, **env: str) -> ClientContext:
    home = tmp_path / "home"
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    write_text(home / ".config" / "opencode" / "opencode.json", OPENCODE_GLOBAL)
    write_json(
        repo / "opencode.json",
        {"mcp": {"overlaid": {"enabled": False, "headers": {"X-Project": "{env:PROJECT}"}}}},
    )
    write_json(
        repo / "pkg" / "opencode.jsonc",
        {"mcp": {"deep": {"type": "remote", "url": "https://deep.example.com/mcp"}}},
    )
    return ClientContext(home=home, cwd=repo / "pkg", env=env)


def test_opencode_entries_and_placeholders(tmp_path: Path) -> None:
    reading = OpenCodeSource().read(_opencode(tmp_path))
    found = by_name(reading)
    sentry = found[("user", "sentry")]
    assert [h.to_dict() for h in sentry.headers] == [
        {
            "name": "Authorization",
            "template": "Bearer ${SENTRY_TOKEN}",
            "env_vars": ["SENTRY_TOKEN"],
        }
    ]
    assert found[("user", "filed")].headers == ()
    assert any("{file:~/.secrets/key}" in w for w in found[("user", "filed")].withheld)
    # OpenCode does not expand ${X}; it must not become a reference rote fills.
    dollar = found[("user", "literal-dollar")]
    assert dollar.headers == ()
    assert any(w.startswith("header X-Mixed: a literal value") for w in dollar.withheld)
    oauth = found[("user", "oauth")]
    assert oauth.scopes == ("read", "write")
    assert any(w.startswith("oauth.clientId: issued to OpenCode") for w in oauth.withheld)
    assert any(w.startswith("oauth.clientSecret: a secret") for w in oauth.withheld)
    assert found[("user", "local")].transport == "stdio"
    assert "type None" in (found[("user", "notype")].blocked or "")
    assert ("project", "deep") in found
    assert_display_safe(reading)


def test_opencode_layers_merge_deeply(tmp_path: Path) -> None:
    found = by_name(OpenCodeSource().read(_opencode(tmp_path)))
    # The project's bare {"enabled": false} overlays the global entry.
    overlaid = found[("project", "overlaid")]
    assert overlaid.host == "global.example.com"
    # Nested tables merge too: both layers' headers survive.
    assert [h.name for h in overlaid.headers] == ["X-Global", "X-Project"]
    assert overlaid.inactive == "disabled in OpenCode (enabled: false)"


def test_opencode_env_files_inline_content_and_managed(tmp_path: Path) -> None:
    ctx = _opencode(tmp_path)
    extra = write_json(
        tmp_path / "extra.json",
        {"mcp": {"sentry": {"type": "remote", "url": "https://extra.example.com/mcp"}}},
    )
    inline = '{"mcp": {"inline": {"type": "remote", "url": "https://inline.example.com/mcp"}}}'
    etc = tmp_path / "etc"
    write_json(
        etc / "opencode" / "opencode.json",
        {"mcp": {"inline": {"type": "remote", "url": "https://managed.example.com/mcp"}}},
    )
    env = {"OPENCODE_CONFIG": str(extra), "OPENCODE_CONFIG_CONTENT": inline}
    found = by_name(
        OpenCodeSource().read(ClientContext(ctx.home, ctx.cwd, env, etc_dir=etc, platform="linux"))
    )
    assert found[("user", "sentry")].host == "extra.example.com"
    assert found[("system", "inline")].host == "managed.example.com"


def test_opencode_v2_shape_and_disable_project(tmp_path: Path) -> None:
    ctx = _opencode(tmp_path)
    write_json(
        tmp_path / "repo" / "opencode.json",
        {
            "mcp": {
                "servers": {
                    "v2": {"type": "remote", "url": "https://v2.example.com/mcp", "disabled": True}
                }
            }
        },
    )
    found = by_name(OpenCodeSource().read(ctx))
    assert found[("project", "v2")].inactive == "disabled in OpenCode (enabled: false)"
    off = ClientContext(ctx.home, ctx.cwd, {"OPENCODE_DISABLE_PROJECT_CONFIG": "1"})
    assert not any(c.scope == "project" for c in OpenCodeSource().read(off).candidates)


def test_opencode_malformed(tmp_path: Path) -> None:
    ctx = _opencode(tmp_path)
    write_text(tmp_path / "repo" / "opencode.json", '{"mcp": {"x": {env:X}}}')
    reading = OpenCodeSource().read(ctx)
    assert any("opencode.json: could not read it" in p for p in reading.problems)
    assert ("user", "sentry") in by_name(reading)
