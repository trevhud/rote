"""``${NAME}`` in a registry header is filled from the environment, never sent empty.

``rote mcp import`` copies a client's ``Authorization: Bearer ${GITHUB_TOKEN}``
as the template, not the value. Every place that sends a registry header
fills it by one rule (``rote.mcp._runtime_helper.expand_headers``; the TS
twin is checked against it in ``test_mcp_header_template_parity_e2e.py``):
the emitted helper's client, the agent loop's ``--mcp-config``, ``rote mcp
headers`` (Claude Code's headersHelper), compile-time tool listing, and the
skill runs' configs.
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import pytest

from rote.cli import main as cli_main
from rote.mcp import McpRegistry, McpServerConfig, save_registry
from rote.mcp._runtime_helper import RoteMcpConfigError, expand_headers, registry_headers

#: The table both languages run (the parity e2e imports it).
CASES: list[tuple[dict[str, str], dict[str, str], dict[str, str] | str]] = [
    # (headers, environment, expected headers or the error's start)
    ({"Authorization": "Bearer ${TOK}"}, {"TOK": "abc"}, {"Authorization": "Bearer abc"}),
    ({"X-Key": "${A}:${B}"}, {"A": "1", "B": "2"}, {"X-Key": "1:2"}),
    ({"X-Key": "${A}${A}"}, {"A": "z"}, {"X-Key": "zz"}),
    # No reference: sent as written (a literal `rote mcp add --header`).
    ({"X-Key": "literal-value"}, {}, {"X-Key": "literal-value"}),
    # Only ${NAME} is a reference: $NAME, ${1X} and ${A:-d} are text.
    ({"X-Key": "$TOK ${1X} ${A:-d}"}, {"TOK": "t", "A": "a"}, {"X-Key": "$TOK ${1X} ${A:-d}"}),
    # A value is never expanded again.
    ({"X-Key": "${A}"}, {"A": "${B}", "B": "nested"}, {"X-Key": "${B}"}),
    # A $ in the value is kept as is (no replacement patterns, either language).
    ({"X-Key": "${A}"}, {"A": "p$&q$1"}, {"X-Key": "p$&q$1"}),
    (
        {"Authorization": "Bearer ${TOK}"},
        {},
        "MCP server 'srv': header 'Authorization' needs TOK, which is not set. Set TOK in "
        "the environment the run starts in (rote never sends a header with an empty value). "
        "This is configuration, so the step is not retried.",
    ),
    (
        {"Authorization": "Bearer ${TOK}"},
        {"TOK": ""},
        "MCP server 'srv': header 'Authorization' needs TOK, which is not set. Set TOK in "
        "the environment the run starts in (rote never sends a header with an empty value). "
        "This is configuration, so the step is not retried.",
    ),
    (
        {"X-Key": "${TOK}"},
        {"TOK": "a\r\nInjected: yes"},
        "MCP server 'srv': header 'X-Key': TOK holds a line break or NUL, which no header "
        "value may carry. This is configuration, so the step is not retried.",
    ),
    (
        {"X-Key": "${TOK}"},
        {"TOK": "a\x00b"},
        "MCP server 'srv': header 'X-Key': TOK holds a line break or NUL, which no header "
        "value may carry. This is configuration, so the step is not retried.",
    ),
]


@pytest.mark.parametrize(("headers", "env", "expected"), CASES)
def test_expand_headers_follows_the_one_rule(
    headers: dict[str, str], env: dict[str, str], expected: dict[str, str] | str
) -> None:
    if isinstance(expected, str):
        with pytest.raises(RoteMcpConfigError) as e:
            expand_headers("srv", headers, env)
        assert str(e.value) == expected
        # The message names the variable, never a value from the environment.
        for value in env.values():
            if value:
                assert value not in str(e.value)
    else:
        assert expand_headers("srv", headers, env) == expected


def _register(headers: dict[str, str]) -> None:
    save_registry(
        McpRegistry(
            servers={"gh": McpServerConfig(url="https://api.example.com/mcp", headers=headers)}
        )
    )


def test_registry_headers_read_the_process_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    _register({"Authorization": "Bearer ${GH_TOKEN}"})
    monkeypatch.setenv("GH_TOKEN", "from-env")
    assert registry_headers("gh") == {"Authorization": "Bearer from-env"}
    monkeypatch.delenv("GH_TOKEN")
    with pytest.raises(RoteMcpConfigError, match="needs GH_TOKEN"):
        registry_headers("gh")
    assert registry_headers("nope") is None


def test_the_emitted_client_sends_the_filled_header(monkeypatch: pytest.MonkeyPatch) -> None:
    """mcp_client builds its transport from the expanded headers, never the
    template (the client is built, not connected: no network)."""
    from rote.mcp._runtime_helper import mcp_client

    _register({"Authorization": "Bearer ${GH_TOKEN}"})
    monkeypatch.setenv("GH_TOKEN", "from-env")
    client = mcp_client("gh", None)
    assert client.transport.headers == {"Authorization": "Bearer from-env"}
    monkeypatch.delenv("GH_TOKEN")
    with pytest.raises(RoteMcpConfigError, match="needs GH_TOKEN"):
        mcp_client("gh", None)


def _headers_cli(name: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = cli_main(["mcp", "headers", name])
    return rc, out.getvalue(), err.getvalue()


def test_mcp_headers_fills_templates_for_headers_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    _register({"Authorization": "Bearer ${GH_TOKEN}", "X-Static": "kept"})
    monkeypatch.setenv("GH_TOKEN", "from-env")
    rc, out, _ = _headers_cli("gh")
    assert rc == 0
    assert json.loads(out) == {"Authorization": "Bearer from-env", "X-Static": "kept"}
    monkeypatch.delenv("GH_TOKEN")
    rc, out, err = _headers_cli("gh")
    assert (rc, out) == (1, "")  # nothing on stdout: Claude Code must not send a half header
    assert "needs GH_TOKEN, which is not set" in err


def test_skill_runs_fill_templates_at_connection_never_in_a_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A templated header reaches claude -p as a headersHelper, so the
    secret never lands in the --mcp-config file; a literal one is as before."""
    from rote.eval.baseline import mcp_servers_from_registry

    save_registry(
        McpRegistry(
            servers={
                "tmpl": McpServerConfig(
                    url="https://a.example/mcp", headers={"Authorization": "Bearer ${T}"}
                ),
                "lit": McpServerConfig(url="https://b.example/mcp", headers={"X-Key": "abc"}),
            }
        )
    )
    monkeypatch.setenv("T", "secret-value")
    servers: dict[str, Any] = mcp_servers_from_registry()
    assert "headers" not in servers["tmpl"]
    assert servers["tmpl"]["headersHelper"].endswith("-m rote mcp headers tmpl")
    assert servers["lit"]["headers"] == {"X-Key": "abc"}
    assert "secret-value" not in json.dumps(servers)


def test_pipeline_trials_fill_templates_at_connection_never_in_a_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The eval --run pipeline side, the same rule as the skill side."""
    from rote.eval.empirical import mcp_servers_for_pipeline
    from rote.ir import Pipeline

    save_registry(
        McpRegistry(
            servers={
                "tmpl": McpServerConfig(
                    url="https://a.example/mcp", headers={"Authorization": "Bearer ${T}"}
                ),
                "lit": McpServerConfig(url="https://b.example/mcp", headers={"X-Key": "abc"}),
            }
        )
    )
    monkeypatch.setenv("T", "secret-value")
    nodes = [
        {
            "id": f"call_{name}",
            "kind": "external_call",
            "description": "Call it.",
            "mcp": {"server": name, "tool": "t"},
        }
        for name in ("tmpl", "lit")
    ]
    pipeline = Pipeline.model_validate(
        {
            "name": "p",
            "input": {"type": "In", "required": [], "optional": []},
            "nodes": nodes,
            "edges": [],
            "entry_nodes": ["call_tmpl", "call_lit"],
            "exit_nodes": ["call_tmpl", "call_lit"],
        }
    )
    servers, missing = mcp_servers_for_pipeline(pipeline)
    assert missing == []
    assert "headers" not in servers["tmpl"]
    assert servers["tmpl"]["headersHelper"].endswith("-m rote mcp headers tmpl")
    assert servers["lit"]["headers"] == {"X-Key": "abc"}
    assert "secret-value" not in json.dumps(servers)


def _loop_config(
    monkeypatch: pytest.MonkeyPatch, pins: dict[str, str]
) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    """The emitted agent loop's MCP config, with the real helper as the
    app's extracted._rote_mcp."""
    import sys
    import types

    from rote.inference import _runtime_helper as inference
    from rote.mcp import _runtime_helper as mcp_helper

    monkeypatch.setitem(sys.modules, "extracted", types.SimpleNamespace(_rote_mcp=mcp_helper))
    monkeypatch.setitem(sys.modules, "extracted._rote_mcp", mcp_helper)
    servers, allowed, _removed, _withheld = inference._mcp_config_for(["search", "lookup"], pins)
    return servers, allowed


def _two_servers() -> None:
    save_registry(
        McpRegistry(
            servers={
                "gh": McpServerConfig(
                    url="https://api.example.com/mcp", headers={"Authorization": "Bearer ${GH_T}"}
                ),
                "open": McpServerConfig(url="https://open.example.com/mcp"),
            }
        )
    )


def test_an_agent_loop_skips_a_templated_server_it_does_not_need(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """review M16: an unset variable on a server the loop does not need
    failed every loop. It is skipped like a missing endpoint, and raises only
    when the loop's own tools are pinned to that server."""
    _two_servers()
    servers, allowed = _loop_config(monkeypatch, {"search": "open"})
    assert sorted(servers) == ["open"]
    assert sorted(allowed) == [("open", "lookup"), ("open", "search")]
    with pytest.raises(
        RoteMcpConfigError, match=r"agent loop tools lookup are served by 'gh'.*GH_T"
    ):
        _loop_config(monkeypatch, {"search": "open", "lookup": "gh"})


def test_an_agent_loop_never_puts_a_header_value_in_its_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI lane passes the config in argv, where ps shows it: a header is
    a headersHelper that runs the emitted helper file, never a value."""
    import shlex
    import subprocess

    _two_servers()
    monkeypatch.setenv("GH_T", "secret-value")
    servers, _allowed = _loop_config(monkeypatch, {"lookup": "gh"})
    assert "headers" not in servers["gh"]
    assert "secret-value" not in json.dumps(servers)
    helper = servers["gh"]["headersHelper"]
    words = shlex.split(helper)
    assert words[1:3] == ["-I", "-c"] and words[-2:] == ["headers", "gh"]
    assert "secret-value" not in helper
    run = subprocess.run(shlex.split(helper), capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout) == {"Authorization": "Bearer secret-value"}
    monkeypatch.delenv("GH_T")
    run = subprocess.run(shlex.split(helper), capture_output=True, text=True, timeout=60)
    assert (run.returncode, run.stdout) == (1, "")
    assert "needs GH_T, which is not set" in run.stderr


def test_the_headers_helper_ignores_modules_planted_beside_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """review: `python extracted/_rote_mcp.py` puts extracted/ first on
    sys.path, where agent-written modules live, so a planted json.py ran in
    the process holding the filled credential. -P keeps it off."""
    import shlex
    import shutil
    import subprocess
    import sys
    import types

    from rote.inference import _runtime_helper as inference
    from rote.mcp import _runtime_helper as mcp_helper

    extracted = tmp_path / "app" / "extracted"
    extracted.mkdir(parents=True)
    copy = extracted / "_rote_mcp.py"
    shutil.copy(mcp_helper.__file__, copy)
    marker = tmp_path / "planted-ran"
    (extracted / "json.py").write_text(
        f"open({str(marker)!r}, 'w').write('x')\nraise SystemExit('planted json.py ran')\n",
        encoding="utf-8",
    )
    planted = types.SimpleNamespace(**vars(mcp_helper))
    planted.__file__ = str(copy)
    _two_servers()
    monkeypatch.setenv("GH_T", "secret-value")
    monkeypatch.setitem(sys.modules, "extracted", types.SimpleNamespace(_rote_mcp=planted))
    monkeypatch.setitem(sys.modules, "extracted._rote_mcp", planted)
    servers, *_ = inference._mcp_config_for(["lookup"], {"lookup": "gh"})
    run = subprocess.run(
        shlex.split(servers["gh"]["headersHelper"]), capture_output=True, text=True, timeout=60
    )
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout) == {"Authorization": "Bearer secret-value"}
    assert not marker.exists()


def test_compile_time_listing_skips_a_server_whose_variable_is_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    from rote.mcp.live_tools import registry_server_specs

    _register({"Authorization": "Bearer ${GH_TOKEN}"})
    skipped: dict[str, str] = {}
    specs = asyncio.run(registry_server_specs(on_skip=skipped.__setitem__))
    assert specs == []
    assert "needs GH_TOKEN" in skipped["gh"]
    monkeypatch.setenv("GH_TOKEN", "from-env")
    specs = asyncio.run(registry_server_specs())
    assert specs == [
        {
            "name": "gh",
            "url": "https://api.example.com/mcp",
            "headers": {"Authorization": "Bearer from-env"},
        }
    ]
