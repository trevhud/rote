"""Codex CLI's config.toml layers, from fixtures shaped like the documented ones."""

from __future__ import annotations

from pathlib import Path

import pytest

from rote.clients import ClientContext
from rote.clients.codex import CodexSource
from tests._clients import (
    CLIENT_SECRET,
    KEYED_URL,
    SECRET,
    assert_display_safe,
    by_name,
    refuse_real_home,
    write_text,
)

__all__ = ["refuse_real_home"]
pytestmark = pytest.mark.usefixtures("refuse_real_home")

USER = f"""
model = "gpt-5.5"

[projects."{{repo}}"]
trust_level = "{{trust}}"

[mcp_servers.figma]
url = "https://mcp.figma.com/mcp"
bearer_token_env_var = "FIGMA_TOKEN"
scopes = ["files:read"]

[mcp_servers.acme-crm]
url = "https://crm.example-work.com/mcp"
env_http_headers = {{ "X-Api-Key" = "CRM_KEY" }}
http_headers = {{ "X-Literal" = "{SECRET}", "X-Templated" = "Bearer ${{NOT_EXPANDED}}" }}
enabled_tools = ["search"]

[mcp_servers.acme-crm.oauth]
client_id = "issued-to-the-client-NOT-REAL"
client_secret = "{CLIENT_SECRET}"

[mcp_servers.helper]
url = "https://h.example.com/mcp"
http_headers_helper = ["print-headers"]
auth = "chatgpt"

[mcp_servers.local]
command = "npx"
args = ["-y", "some-server"]
env = {{ API_KEY = "{SECRET}" }}

[mcp_servers.off]
url = "https://off.example.com/mcp"
enabled = false

[mcp_servers.keyed]
url = "{KEYED_URL}"

[mcp_servers.overridden]
url = "https://user.example.com/mcp"
startup_timeout_sec = 20
"""

PROJECT = """
[mcp_servers.overridden]
enabled = false

[mcp_servers.project-only]
url = "https://project.example.com/mcp"
"""


def _machine(tmp_path: Path, trust: str = "trusted", **env: str) -> ClientContext:
    home = tmp_path / "home"
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    cwd = repo / "pkg"
    cwd.mkdir()
    codex_home = Path(env["CODEX_HOME"]) if "CODEX_HOME" in env else home / ".codex"
    write_text(
        codex_home / "config.toml", USER.replace("{repo}", str(repo)).replace("{trust}", trust)
    )
    write_text(repo / ".codex" / "config.toml", PROJECT)
    return ClientContext(home=home, cwd=cwd, env=env)


def test_remote_entries_and_header_indirection(tmp_path: Path) -> None:
    found = by_name(CodexSource().read(_machine(tmp_path)))
    figma = found[("user", "figma")]
    assert figma.importable and figma.transport == "streamable-http"
    assert [h.to_dict() for h in figma.headers] == [
        {"name": "Authorization", "template": "Bearer ${FIGMA_TOKEN}", "env_vars": ["FIGMA_TOKEN"]}
    ]
    assert figma.scopes == ("files:read",)
    crm = found[("user", "acme-crm")]
    assert crm.name == "acme_crm"
    assert [(h.name, h.template) for h in crm.headers] == [("X-Api-Key", "${CRM_KEY}")]
    # http_headers are literal in Codex: a ${X} there is sent as written.
    assert any(w.startswith("header X-Literal: a literal value") for w in crm.withheld)
    assert any(w.startswith("header X-Templated: a literal value") for w in crm.withheld)
    assert any(w.startswith("oauth.client_id: issued to Codex") for w in crm.withheld)
    assert any(w.startswith("oauth.client_secret") for w in crm.withheld)
    assert any(w.startswith("enabled_tools/disabled_tools") for w in crm.withheld)
    helper = found[("user", "helper")]
    assert any(w.startswith("http_headers_helper:") for w in helper.withheld)
    assert any(w.startswith("auth = chatgpt") for w in helper.withheld)


def test_stdio_disabled_and_keyed_url(tmp_path: Path) -> None:
    found = by_name(CodexSource().read(_machine(tmp_path)))
    local = found[("user", "local")]
    assert local.transport == "stdio" and not local.importable
    assert found[("user", "off")].inactive == "disabled in Codex (enabled = false)"
    keyed = found[("user", "keyed")]
    assert keyed.importable and keyed.risk is not None and not keyed.preselectable


def test_trusted_project_layers_merge_field_by_field(tmp_path: Path) -> None:
    ctx = _machine(tmp_path)
    reading = CodexSource().read(ctx)
    found = by_name(reading)
    # The project set only enabled = false; the user's url survives the merge.
    merged = found[("project", "overridden")]
    assert merged.host == "user.example.com"
    assert merged.inactive == "disabled in Codex (enabled = false)"
    assert ("user", "overridden") not in found
    assert found[("project", "project-only")].inactive is None


def test_an_untrusted_project_is_listed_but_unused(tmp_path: Path) -> None:
    found = by_name(CodexSource().read(_machine(tmp_path, trust="untrusted")))
    assert found[("user", "overridden")].inactive is None  # the project layer is left out
    project_only = found[("project", "project-only")]
    assert project_only.inactive is not None and "does not trust" in project_only.inactive


def test_codex_home_moves_the_user_file(tmp_path: Path) -> None:
    ctx = _machine(tmp_path, CODEX_HOME=str(tmp_path / "elsewhere"))
    reading = CodexSource().read(ctx)
    assert tmp_path / "elsewhere" / "config.toml" in reading.files_read
    assert ("user", "figma") in by_name(reading)


def test_system_layer_sits_below_the_user(tmp_path: Path) -> None:
    ctx = _machine(tmp_path)
    etc = tmp_path / "etc"
    write_text(
        etc / "codex" / "config.toml",
        '[mcp_servers.figma]\nurl = "https://sys.example"\n'
        '[mcp_servers.org]\nurl = "https://org.example.com/mcp"\n',
    )
    found = by_name(CodexSource().read(ClientContext(ctx.home, ctx.cwd, {}, etc_dir=etc)))
    assert found[("user", "figma")].host == "mcp.figma.com"
    assert found[("system", "org")].importable


def test_nothing_secret_is_displayed(tmp_path: Path) -> None:
    assert_display_safe(CodexSource().read(_machine(tmp_path)), "print-headers")


def test_a_malformed_file_is_a_problem(tmp_path: Path) -> None:
    ctx = _machine(tmp_path)
    write_text(ctx.home / ".codex" / "config.toml", "[mcp_servers.x\nurl = ")
    reading = CodexSource().read(ctx)
    assert any("could not read it (TOMLDecodeError)" in p for p in reading.problems)
