"""Cursor's mcp.json files, from fixtures shaped like the documented ones."""

from __future__ import annotations

from pathlib import Path

import pytest

from rote.clients import ClientContext
from rote.clients.cursor import CursorSource
from tests._clients import (
    CLIENT_ID,
    CLIENT_SECRET,
    SECRET,
    assert_display_safe,
    assert_trust_unknown,
    by_name,
    refuse_real_home,
    write_text,
)

__all__ = ["refuse_real_home"]
pytestmark = pytest.mark.usefixtures("refuse_real_home")

USER = f"""{{
  // comments and trailing commas are allowed
  "mcpServers": {{
    "github": {{
      "url": "https://api.githubcopilot.com/mcp/",
      "headers": {{ "Authorization": "Bearer ${{env:GITHUB_TOKEN}}" }},
    }},
    "literal": {{ "url": "https://l.example.com/mcp", "headers": {{ "X-Key": "{SECRET}" }} }},
    "folder": {{
      "url": "https://f.example.com/mcp",
      "headers": {{ "X-Root": "${{workspaceFolder}}" }}
    }},
    "oauth": {{
      "url": "https://o.example.com/mcp",
      "auth": {{ "CLIENT_ID": "{CLIENT_ID}", "CLIENT_SECRET": "{CLIENT_SECRET}",
                 "scopes": ["read", "write"] }}
    }},
    "events": {{ "type": "sse", "url": "https://e.example.com/sse" }},
    "local": {{ "command": "npx", "args": ["-y", "pkg"], "env": {{ "K": "{SECRET}" }} }},
    "shared": {{ "url": "https://user-shared.example.com/mcp" }},
    "host-var": {{ "url": "https://${{env:HOST}}/mcp" }}
  }}
}}
"""


def _machine(tmp_path: Path) -> ClientContext:
    home = tmp_path / "home"
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    write_text(home / ".cursor" / "mcp.json", USER)
    write_text(
        repo / ".cursor" / "mcp.json",
        '{"mcpServers": {"shared": {"url": "https://project-shared.example.com/mcp"}}}',
    )
    (repo / "src").mkdir()
    return ClientContext(home=home, cwd=repo / "src", env={})


def test_placeholders_become_one_form_and_others_are_withheld(tmp_path: Path) -> None:
    found = by_name(CursorSource().read(_machine(tmp_path)))
    github = found[("user", "github")]
    assert [h.template for h in github.headers] == ["Bearer ${GITHUB_TOKEN}"]
    assert github.headers[0].env_vars == ("GITHUB_TOKEN",)
    folder = found[("user", "folder")]
    assert folder.headers == ()
    assert any("${workspaceFolder}" in w and "only the client" in w for w in folder.withheld)
    host_var = found[("user", "host-var")]
    assert "depends on HOST" in (host_var.blocked or "")
    assert host_var.host is None  # a placeholder is not a host to show
    assert host_var.to_dict()["host"] is None


def test_project_wins_over_user(tmp_path: Path) -> None:
    found = by_name(CursorSource().read(_machine(tmp_path)))
    assert found[("project", "shared")].inactive is None
    assert_trust_unknown(found[("project", "shared")], "Cursor")
    assert found[("user", "github")].trust == "trusted"
    assert found[("user", "shared")].inactive == "shadowed by the project entry of the same name"


def test_transports_secrets_and_stdio(tmp_path: Path) -> None:
    reading = CursorSource().read(_machine(tmp_path))
    found = by_name(reading)
    assert found[("user", "events")].transport == "sse"
    assert found[("user", "github")].transport == "streamable-http"
    local = found[("user", "local")]
    assert local.transport == "stdio" and not local.importable
    literal = found[("user", "literal")]
    assert literal.headers == () and "a literal value" in literal.withheld[0]
    oauth = found[("user", "oauth")]
    assert oauth.scopes == ("read", "write")
    assert any(w.startswith("auth.CLIENT_ID: issued to Cursor") for w in oauth.withheld)
    assert any(w.startswith("auth.CLIENT_SECRET") for w in oauth.withheld)
    assert_display_safe(reading)
    assert any("Customize sidebar" in limit for limit in reading.limits)


def test_a_malformed_file_is_a_problem(tmp_path: Path) -> None:
    ctx = _machine(tmp_path)
    write_text(tmp_path / "repo" / ".cursor" / "mcp.json", '{"mcpServers": {')
    reading = CursorSource().read(ctx)
    assert any(".cursor/mcp.json: could not read it" in p for p in reading.problems)
    assert ("user", "github") in by_name(reading)
