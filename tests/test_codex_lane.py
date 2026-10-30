"""The Codex skill-run lane: the Claude lane's gate, in Codex's terms (#334).

These pin the translation and the parsing. What each translated setting does
inside Codex was measured on codex-cli 0.165.0 (see ``rote.eval.codex_lane``);
checking it again costs inference, so it is an eval, never a test here.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from rote.eval import codex_lane
from rote.eval.empirical import EmpiricalError, skill_run_file_rules

#: A ChatGPT login as Codex writes it (values made up).
CHATGPT_LOGIN = {
    "auth_mode": "chatgpt",
    "OPENAI_API_KEY": None,
    "tokens": {"access_token": "a", "refresh_token": "r", "id_token": "i", "account_id": "x"},
    "last_refresh": "2026-10-01T00:00:00.000000Z",
}


@pytest.fixture
def fake_codex(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A stub binary (by absolute path: never a bare ``codex`` on PATH) and a login."""
    real = tmp_path / "pkg" / "bin" / "codex"
    real.parent.mkdir(parents=True)
    real.write_text("#!/bin/sh\n")
    real.chmod(0o755)
    home = tmp_path / "codex-home"
    home.mkdir()
    (home / "auth.json").write_text(json.dumps(CHATGPT_LOGIN))
    monkeypatch.setenv("CODEX_HOME", str(home))
    return real


def test_the_gate_is_translated_entry_by_entry(tmp_path: Path) -> None:
    skill = tmp_path / "my skill"
    skill.mkdir()
    allowed = [
        *skill_run_file_rules(skill),
        "mcp__jira__search_issues",
        "mcp__jira__get_issue",
        "mcp__rote-gate__ask_person",
        "mcp__docs__*",
        "Bash(rote skill run orphan-pr *)",
    ]
    gate = codex_lane.translate_gate(allowed, ["jira", "rote-gate", "docs", "unused"])
    assert gate.readable == (skill.resolve(),)
    assert gate.tools == {
        "jira": ("search_issues", "get_issue"),
        "rote-gate": ("ask_person",),
        "docs": None,
    }
    assert gate.commands == (("rote", "skill", "run", "orphan-pr"),)


def test_a_wildcard_lifts_the_gate_whatever_came_first() -> None:
    gate = codex_lane.translate_gate(["mcp__jira__get_issue", "mcp__jira__*"], ["jira"])
    assert gate.tools == {"jira": None}
    gate = codex_lane.translate_gate(["mcp__jira__*", "mcp__jira__get_issue"], ["jira"])
    assert gate.tools == {"jira": None}


def test_a_server_name_with_double_underscores_resolves_to_the_longest_name() -> None:
    gate = codex_lane.translate_gate(["mcp__a__b__tool"], ["a", "a__b"])
    assert gate.tools == {"a__b": ("tool",)}


@pytest.mark.parametrize(
    "entry",
    ["Read", "Bash", "WebSearch", "Read(~/notes/**)", "mcp__nowhere__x", "Bash(curl *)x"],
)
def test_an_entry_with_no_translation_is_refused(entry: str) -> None:
    with pytest.raises(EmpiricalError, match="Codex lane"):
        codex_lane.translate_gate([entry], ["jira"])


def test_the_run_is_isolated_and_gated(fake_codex: Path, tmp_path: Path) -> None:
    skill = tmp_path / "skill"
    skill.mkdir()
    work = tmp_path / "work"
    work.mkdir()
    servers = {
        "jira": {
            "type": "http",
            "url": "https://mcp.example/jira",
            "headersHelper": "py -m rote mcp headers jira",
        },
        "static": {"type": "http", "url": "https://mcp.example/s", "headers": {"X-Key": "k"}},
        "rote-gate": {"type": "stdio", "command": "/usr/bin/python3", "args": ["-m", "gate"]},
        "ungated": {"type": "http", "url": "https://mcp.example/u"},
    }
    allowed = [
        *skill_run_file_rules(skill),
        "mcp__jira__get_issue",
        "mcp__static__*",
        "mcp__rote-gate__ask_person",
        "Bash(rote skill run demo *)",
    ]
    shim = tmp_path / "shim"
    shim.mkdir()
    (shim / "rote").write_text("#!/bin/sh\n")
    (shim / "rote").chmod(0o755)
    base = {
        "HOME": "/Users/me",
        "PATH": f"{shim}{os.pathsep}/bin",
        "CODEX_API_KEY": "k",
        "OPENAI_API_KEY": "o",
        "CODEX_HOME": os.environ["CODEX_HOME"],
        "ROTE_SKILL_RUN_DIR": "/state/runs",
        "ROTE_CLEF_API_TOKEN": "decision-model-secret",
        "DATABASE_URL": "postgres://u:p@h/db",
    }
    root = tmp_path / "root"
    with codex_lane.prepared(
        root,
        workdir=work,
        mcp_servers=servers,
        allowed=allowed,
        model="gpt-5.5",
        base_env=base,
        binary=fake_codex,
    ) as run:
        argv = run.argv
        # Launched by the path given (the binary's real path in use).
        assert argv[:2] == [str(fake_codex), "exec"]
        assert argv[-1] == "-" and argv[argv.index("--model") + 1] == "gpt-5.5"
        assert 'approval_policy="never"' in argv and 'web_search="disabled"' in argv
        assert [argv[i + 1] for i, a in enumerate(argv) if a == "--disable"] == list(
            codex_lane.FEATURES_OFF
        )
        # The plan, not a key, pays; the homes are throwaway, the login is linked.
        assert "CODEX_API_KEY" not in run.env and "OPENAI_API_KEY" not in run.env
        assert run.env["CODEX_HOME"] == str(root / "codex-home")
        assert run.env["HOME"] == str(root / "home")
        link = root / "codex-home" / "auth.json"
        assert link.is_symlink() and link.resolve() == Path(os.environ["CODEX_HOME"]) / "auth.json"
        # The driver leaves the sandbox only as one MCP tool, with the name fixed:
        # a custom profile disables escalation, so no prefix rule could.
        assert not (root / "codex-home" / "rules").exists()
        server = "mcp_servers.rote_skill"
        assert f'{server}.enabled_tools=["run"]' in argv
        args = json.loads(next(a for a in argv if a.startswith(f"{server}.args=")).split("=", 1)[1])
        assert args == ["-I", "-m", "rote.eval.skill_run_server", "demo", str(shim / "rote")]
        env_arg = next(a for a in argv if a.startswith(f"{server}.env="))
        # rote's state paths only: a ROTE_* credential is not a state path.
        assert '"ROTE_SKILL_RUN_DIR"="/state/runs"' in env_arg
        assert "ROTE_CLEF_API_TOKEN" not in env_arg
        timeout = next(a for a in argv if a.startswith(f"{server}.tool_timeout_sec="))
        assert int(timeout.split("=")[1]) > skill_run_server_timeout()
        assert "`run` tool of the `rote_skill` MCP server" in run.note
        joined = "\n".join(argv)
        # Reads: the system, the binary's dir and the skill; temp roots denied; workspace writes.
        fs = next(a for a in argv if a.startswith("permissions.rote_skill_run.filesystem="))
        table = fs.split("=", 1)[1]
        assert f'"{fake_codex.parent}"="read"' in table
        assert f'"{skill.resolve()}"="read"' in table
        assert '":minimal"="read"' in table and '":workspace_roots"="write"' in table
        for root in codex_lane._deny_roots():
            assert f'"{root}"="deny"' in table
        # MCP: gated tools enabled by name, a wildcard unrestricted, approval on.
        assert 'mcp_servers.jira.enabled_tools=["get_issue"]' in argv
        assert "mcp_servers.static.enabled_tools" not in joined
        # A header value rides in Codex's env, by name: argv (ps) never holds it.
        var = codex_lane.header_var("static", 0)
        assert f'mcp_servers.static.env_http_headers={{"X-Key"="{var}"}}' in argv
        assert run.env[var] == "k" and "k" not in [a.split("=", 1)[-1] for a in argv]
        # The agent's shell gets core variables only, not the caller's env.
        assert 'shell_environment_policy.inherit="core"' in argv
        assert "decision-model-secret" not in joined and "postgres://" not in joined
        assert 'mcp_servers.rote-gate.command="/usr/bin/python3"' in argv
        assert 'mcp_servers.jira.default_tools_approval_mode="approve"' in argv
        helper = next(a for a in argv if a.startswith("mcp_servers.jira.http_headers_helper="))
        assert json.loads(helper.split("=", 1)[1]) == (
            "/usr/bin/env HOME=/Users/me py -m rote mcp headers jira"
        )
        # A server the gate gives nothing is not wired at all.
        assert "ungated" not in joined


def test_an_sse_server_is_refused() -> None:
    with pytest.raises(EmpiricalError, match="streamable HTTP or stdio"):
        codex_lane.server_config("s", {"type": "sse", "url": "https://x"}, "/h")


def skill_run_server_timeout() -> float:
    from rote.eval import skill_run_server

    return skill_run_server.TIMEOUT_SECONDS


def test_no_login_file_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(EmpiricalError, match="codex login"):
        codex_lane.login_file({"CODEX_HOME": str(tmp_path)})


@pytest.mark.parametrize(
    "login",
    [
        {"auth_mode": "apikey", "OPENAI_API_KEY": "sk-x"},
        {**CHATGPT_LOGIN, "OPENAI_API_KEY": "sk-x"},
        {"tokens": {}},
        ["not", "an", "object"],
    ],
)
def test_a_login_that_would_bill_a_key_is_refused(tmp_path: Path, login: object) -> None:
    (tmp_path / "auth.json").write_text(json.dumps(login))
    with pytest.raises(EmpiricalError, match="not a ChatGPT login"):
        codex_lane.login_file({"CODEX_HOME": str(tmp_path)})


def test_a_stdio_server_env_beyond_paths_is_refused() -> None:
    with pytest.raises(EmpiricalError, match="cannot pass SECRET to MCP server"):
        codex_lane.server_config(
            "s", {"type": "stdio", "command": "x", "env": {"SECRET": "v", "PATH": "/b"}}, "/h"
        )


def test_an_npm_launcher_reads_its_package(fake_codex: Path, tmp_path: Path) -> None:
    js = tmp_path / "node_modules" / "@openai" / "codex" / "bin" / "codex.js"
    js.parent.mkdir(parents=True)
    js.write_text("")
    with codex_lane.prepared(
        tmp_path / "root", workdir=tmp_path, mcp_servers={}, allowed=[], model=None, binary=js
    ) as run:
        fs = next(a for a in run.argv if a.startswith("permissions.rote_skill_run.filesystem="))
    assert f'"{js.parent.parent}"="read"' in fs


def test_a_refreshed_login_is_written_back_and_a_stale_one_is_not(tmp_path: Path) -> None:
    real = tmp_path / "real" / "auth.json"
    real.parent.mkdir()
    real.write_text(json.dumps({"last_refresh": "2026-10-01T00:00:00Z", "t": "old"}))
    link = tmp_path / "home" / "auth.json"
    link.parent.mkdir()
    # Codex replaced the link with a file holding newer tokens.
    link.write_text(json.dumps({"last_refresh": "2026-10-29T00:00:00Z", "t": "new"}))
    codex_lane._write_back_login(link, real)
    assert json.loads(real.read_text())["t"] == "new"
    assert oct(real.stat().st_mode & 0o777) == "0o600"
    # An older file never replaces the user's.
    link.write_text(json.dumps({"last_refresh": "2026-09-01T00:00:00Z", "t": "older"}))
    codex_lane._write_back_login(link, real)
    assert json.loads(real.read_text())["t"] == "new"
    # Times are compared parsed: more fractional digits is not "later" as text.
    real.write_text(json.dumps({"last_refresh": "2026-10-29T00:00:00.9Z", "t": "kept"}))
    link.write_text(json.dumps({"last_refresh": "2026-10-29T00:00:00.100000Z", "t": "x"}))
    codex_lane._write_back_login(link, real)
    assert json.loads(real.read_text())["t"] == "kept"


def test_the_link_left_in_place_writes_nothing(tmp_path: Path) -> None:
    real = tmp_path / "auth.json"
    real.write_text("{}")
    link = tmp_path / "link.json"
    link.symlink_to(real)
    before = real.stat().st_mtime_ns
    codex_lane._write_back_login(link, real)
    assert real.stat().st_mtime_ns == before


def test_events_are_parsed_into_usage_calls_and_errors() -> None:
    lines = [
        json.dumps({"type": "thread.started", "thread_id": "t"}),
        json.dumps(
            {"type": "item.completed", "item": {"type": "agent_message", "text": "working"}}
        ),
        json.dumps(
            {
                "type": "item.completed",
                "item": {
                    "type": "mcp_tool_call",
                    "server": "probe",
                    "tool": "read_note",
                    "arguments": {"name": "alpha"},
                    "result": {"content": [{"type": "text", "text": "hi"}]},
                    "error": None,
                    "status": "completed",
                },
            }
        ),
        json.dumps(
            {"type": "item.completed", "item": {"type": "command_execution", "command": "ls"}}
        ),
        "not json",
        json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "done"}}),
        json.dumps(
            {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": 35313,
                    "cached_input_tokens": 22144,
                    "cache_write_input_tokens": 0,
                    "output_tokens": 181,
                    "reasoning_output_tokens": 0,
                },
            }
        ),
    ]
    out = codex_lane.parse_events(lines)
    assert out.usage == {
        "input_tokens": 35313 - 22144,
        "output_tokens": 181,
        "cache_write_tokens": 0,
        "cache_read_tokens": 22144,
    }
    assert out.calls == (
        {
            "server": "probe",
            "tool": "read_note",
            "arguments": {"name": "alpha"},
            "result": {"content": [{"type": "text", "text": "hi"}]},
            "error": None,
        },
    )
    assert out.commands == ("ls",) and out.message == "done" and out.error is None


def test_a_run_with_no_usage_has_no_figure_and_a_failure_is_kept() -> None:
    failed = codex_lane.parse_events(
        [json.dumps({"type": "turn.failed", "error": {"message": "usage limit reached"}})]
    )
    assert failed.usage is None and failed.error == "usage limit reached" and failed.failed
    recovered = codex_lane.parse_events([json.dumps({"type": "error", "message": "retrying"})])
    assert recovered.error == "retrying" and not recovered.failed


def test_an_incomplete_usage_report_does_not_erase_a_complete_one() -> None:
    complete = {"input_tokens": 10, "cached_input_tokens": 4, "output_tokens": 2}
    out = codex_lane.parse_events(
        [
            json.dumps({"type": "turn.completed", "usage": complete}),
            json.dumps({"type": "turn.completed", "usage": {"input_tokens": 5}}),
        ]
    )
    assert out.usage is not None and out.usage["output_tokens"] == 2


_STUB = """#!{python}
import json, sys
argv = sys.argv[1:]
work = argv[argv.index("-C") + 1]
prompt = sys.stdin.read()
open(work + "/stdin.txt", "w").write(prompt)
open(work + "/result.json", "w").write(json.dumps({{"answer": 42}}))
for event in [
    {{"type": "thread.started", "thread_id": "t"}},
    {{"type": "item.completed", "item": {{"type": "mcp_tool_call", "server": "codex",
        "tool": "list_mcp_resources", "arguments": {{}}, "result": None, "error": None}}}},
    {{"type": "item.completed", "item": {{"type": "mcp_tool_call", "server": "docs",
        "tool": "search", "arguments": {{"q": "x"}},
        "result": {{"content": [{{"type": "text", "text": "hit"}}]}}, "error": None}}}},
    {{"type": "turn.completed", "usage": {{"input_tokens": 900, "cached_input_tokens": 600,
        "cache_write_input_tokens": 0, "output_tokens": 50, "reasoning_output_tokens": 0}}}},
]:
    print(json.dumps(event))
"""


def test_a_codex_baseline_trial_measures_and_observes_only_the_skills_servers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_codex: Path
) -> None:
    """A real subprocess (a stub ``codex`` by absolute path): the prompt on
    stdin, result.json read back, usage in rote's buckets, no dollar figure,
    and Codex's own ``codex`` resource listing not taken for skill traffic."""
    import sys

    from rote.eval.baseline import run_baseline_trial

    fake_codex.write_text(_STUB.format(python=sys.executable))
    monkeypatch.setattr(codex_lane, "codex_binary", lambda: fake_codex)
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: s\ndescription: d\n---\nAnswer.\n")
    run, observed, lines = run_baseline_trial(
        skill,
        {"q": 1},
        model=None,
        agent="codex",
        mcp_servers={"docs": {"type": "http", "url": "https://mcp.example/docs"}},
        mcp_tool_ids=["mcp__docs__search"],
    )
    assert run.error is None and run.output == {"answer": 42}
    assert (run.agent, run.cost_usd, run.turns) == ("codex", None, None)
    assert (run.input_tokens, run.cache_read_tokens, run.output_tokens) == (300, 600, 50)
    assert [(o.server, o.tool, o.input) for o in observed] == [("docs", "search", {"q": "x"})]
    assert observed[0].result == [{"type": "text", "text": "hit"}]
    assert len(lines) == 4


def test_only_the_driver_may_leave_the_sandbox(fake_codex: Path, tmp_path: Path) -> None:
    with (
        pytest.raises(EmpiricalError, match="only `rote skill run <name>`"),
        codex_lane.prepared(
            tmp_path / "root",
            workdir=tmp_path,
            mcp_servers={},
            allowed=["Bash(curl *)"],
            model=None,
            binary=fake_codex,
        ),
    ):
        pass


def test_the_driver_tool_runs_only_this_skills_driver(tmp_path: Path) -> None:
    from rote.eval import skill_run_server

    rote = tmp_path / "rote"
    rote.write_text('#!/bin/sh\necho "$@"\n')
    rote.chmod(0o755)
    out = skill_run_server.run("demo", str(rote), ["start", '{"a": "b c"}'])
    assert out == {"exit_code": 0, "stdout": 'skill run demo start {"a": "b c"}\n', "stderr": ""}
    assert skill_run_server.run("demo", str(rote), "start")["exit_code"] == 2  # type: ignore[arg-type]


def test_a_server_name_codex_cannot_read_bare_is_refused() -> None:
    """Codex keeps TOML quotes in a -c key, so a quoted name never loads."""
    assert codex_lane._key("jira_cloud-2") == "jira_cloud-2"
    with pytest.raises(EmpiricalError, match="letters, digits"):
        codex_lane._key("my.server")
