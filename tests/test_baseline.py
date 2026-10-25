"""Tests for the pre-compilation baseline runner (rote.eval.baseline).

Nothing here spawns a real agent or contacts a real MCP server: the
subprocess is mocked at ``subprocess.run`` (same style as the claude
driver tests), the MCP client at ``mcp_client``, and the registry/token
stores are pointed at hermetic tmp locations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rote.cli import main as cli_main
from rote.eval.baseline import (
    BASELINE_DIRNAME,
    METRICS_FILENAME,
    OBSERVED_TOOLS_FILENAME,
    ObservedToolCall,
    extract_observations,
    mcp_servers_from_registry,
    run_baseline,
    run_baseline_trial,
)

# ───────── Stream-json fixtures ─────────


def _tool_use_event(block_id: str, name: str, tool_input: dict[str, Any]) -> str:
    return json.dumps(
        {
            "type": "assistant",
            "message": {
                "content": [{"type": "tool_use", "id": block_id, "name": name, "input": tool_input}]
            },
        }
    )


def _tool_result_event(block_id: str, content: Any, is_error: bool = False) -> str:
    return json.dumps(
        {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": block_id,
                        "content": content,
                        "is_error": is_error,
                    }
                ]
            },
        }
    )


def _result_event(**overrides: Any) -> str:
    base: dict[str, Any] = {
        "type": "result",
        "subtype": "success",
        "num_turns": 7,
        "total_cost_usd": 0.42,
        "usage": {
            "input_tokens": 12,
            "output_tokens": 900,
            "cache_read_input_tokens": 40000,
            "cache_creation_input_tokens": 9000,
        },
    }
    base.update(overrides)
    return json.dumps(base)


# ───────── extract_observations (pure) ─────────


def test_extract_observations_pairs_calls_with_results() -> None:
    lines = [
        _tool_use_event("t1", "mcp__gmail__search_threads", {"query": "RFP"}),
        _tool_result_event("t1", [{"type": "text", "text": '{"threads": []}'}]),
        _result_event(),
    ]
    observed = extract_observations(lines)
    assert observed == [
        ObservedToolCall(
            server="gmail",
            tool="search_threads",
            input={"query": "RFP"},
            result=[{"type": "text", "text": '{"threads": []}'}],
            is_error=False,
        )
    ]


def test_extract_observations_ignores_local_tools_and_junk() -> None:
    lines = [
        _tool_use_event("t1", "Read", {"file_path": "/x"}),  # local tool: not observed
        "not json at all",
        json.dumps({"type": "system"}),
        _tool_result_event("t1", "ignored"),
        _result_event(),
    ]
    assert extract_observations(lines) == []


def test_extract_observations_keeps_unanswered_calls() -> None:
    """A call cut off by max-turns still proves the skill reaches the tool."""
    lines = [_tool_use_event("t9", "mcp__slack__slack_read_channel", {"channel": "C1"})]
    observed = extract_observations(lines)
    assert len(observed) == 1
    assert observed[0].server == "slack"
    assert observed[0].result is None


def test_extract_observations_marks_errors() -> None:
    lines = [
        _tool_use_event("t1", "mcp__gmail__get_thread", {"id": "x"}),
        _tool_result_event("t1", "401 unauthorized", is_error=True),
    ]
    assert extract_observations(lines)[0].is_error is True


# ───────── Registry wiring ─────────


def _isolate_mcp_stores(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    registry = tmp_path / "mcp.json"
    monkeypatch.setenv("ROTE_MCP_CONFIG", str(registry))
    monkeypatch.setenv("ROTE_MCP_TOKEN_DIR", str(tmp_path / "tokens"))
    return registry


def test_mcp_servers_from_registry_wires_all_servers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _isolate_mcp_stores(monkeypatch, tmp_path)
    registry.write_text(
        json.dumps(
            {
                "version": 1,
                "servers": {
                    "gmail": {"url": "https://gmail.example/mcp"},
                    "hubspot": {
                        "url": "https://hubspot.example/mcp",
                        "headers": {"Authorization": "Bearer static"},
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    servers = mcp_servers_from_registry()
    assert set(servers) == {"gmail", "hubspot"}
    assert servers["hubspot"]["headers"] == {"Authorization": "Bearer static"}
    # No token file for gmail → wired without auth (server may be public
    # or will 401; the observed traffic reports reality either way).
    assert "headersHelper" not in servers["gmail"]
    assert servers["gmail"]["type"] == "http"


def test_mcp_servers_from_registry_empty_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_mcp_stores(monkeypatch, tmp_path)
    assert mcp_servers_from_registry() == {}


# ───────── One streamed trial (mocked subprocess) ─────────


class _FakeProc:
    def __init__(self, stdout: str, returncode: int = 0, stderr: str = "") -> None:
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr


def _install_fake_claude(
    monkeypatch: pytest.MonkeyPatch,
    stdout_lines: list[str],
    *,
    returncode: int = 0,
    write_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Mock subprocess.run for the trial; captures the invocation."""
    captured: dict[str, Any] = {}

    def fake_run(args: list[str], **kwargs: Any) -> _FakeProc:
        captured["args"] = args
        captured["kwargs"] = kwargs
        # The config lives in the trial's temp workdir, gone after the run:
        # read what the child would have read, at spawn time.
        if "--mcp-config" in args:
            config = Path(args[args.index("--mcp-config") + 1])
            captured["mcp_config"] = json.loads(config.read_text(encoding="utf-8"))
        if write_result is not None:
            (Path(kwargs["cwd"]) / "result.json").write_text(
                json.dumps(write_result), encoding="utf-8"
            )
        return _FakeProc("\n".join(stdout_lines), returncode=returncode)

    monkeypatch.setattr("rote.eval.baseline.subprocess.run", fake_run)
    monkeypatch.setattr("rote.eval.baseline.shutil.which", lambda _: "/usr/bin/claude")
    return captured


@pytest.fixture()
def skill_dir(tmp_path: Path) -> Path:
    d = tmp_path / "skill"
    d.mkdir()
    (d / "SKILL.md").write_text("---\nname: t\n---\nDo the thing.\n")
    return d


def test_trial_measures_and_observes(skill_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lines = [
        _tool_use_event("t1", "mcp__gmail__search_threads", {"query": "RFP"}),
        _tool_result_event("t1", [{"type": "text", "text": "{}"}]),
        _result_event(),
    ]
    captured = _install_fake_claude(monkeypatch, lines, write_result={"ok": True})

    run, observed, transcript = run_baseline_trial(
        skill_dir, {"campaign": "x"}, model="claude-test-model"
    )

    assert run.succeeded
    assert run.turns == 7
    assert run.cost_usd == 0.42
    assert run.output_tokens == 900
    assert run.output == {"ok": True}
    assert [o.tool for o in observed] == ["search_threads"]
    assert transcript == lines
    # stream-json + --verbose is what makes observation possible.
    args = captured["args"]
    assert args[args.index("--output-format") + 1] == "stream-json"
    assert "--verbose" in args


def test_trial_wires_mcp_config_strictly(skill_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_fake_claude(monkeypatch, [_result_event()], write_result={})
    run_baseline_trial(
        skill_dir,
        {},
        model="m",
        mcp_servers={"gmail": {"type": "http", "url": "https://g.example/mcp"}},
        mcp_tool_ids=["mcp__gmail__search_threads"],
    )
    args = captured["args"]
    assert "--strict-mcp-config" in args
    config_path = Path(args[args.index("--mcp-config") + 1])
    assert captured["mcp_config"] == {
        "mcpServers": {"gmail": {"type": "http", "url": "https://g.example/mcp"}}
    }
    allowed = args[args.index("--allowedTools") + 1].split(",")
    assert allowed == [
        "Read(./**)",
        f"Read(/{skill_dir.resolve().as_posix()}/**)",
        "Edit(./**)",
        "mcp__gmail__search_threads",
    ]  # read-only gate: explicit ids only, never mcp__gmail__*
    # Config file existed at spawn time inside the trial workdir.
    assert config_path.name == "mcp-config.json"
    assert config_path.parent == Path(captured["kwargs"]["cwd"])


def test_trial_scopes_file_access_to_the_cwd_and_the_skill(
    skill_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A third-party SKILL.md could have the agent read ~/.ssh and send the
    secret out as a read-only MCP tool's argument. No file tool may be
    allowed unscoped: a bare Read/Glob/Grep (or Write/Edit) allows the whole
    machine. Read rules are what Glob and Grep are checked against, so they
    carry no rule of their own."""
    captured = _install_fake_claude(monkeypatch, [_result_event()], write_result={})
    run_baseline_trial(skill_dir, {}, model="m")
    allowed = captured["args"][captured["args"].index("--allowedTools") + 1].split(",")
    file_rules = [r for r in allowed if not r.startswith("mcp__")]
    for tool in ("Read", "Glob", "Grep", "Write", "Edit"):
        assert tool not in file_rules, f"{tool} is allowed unscoped"
    skill_root = skill_dir.resolve().as_posix()
    assert file_rules == ["Read(./**)", f"Read(/{skill_root}/**)", "Edit(./**)"]
    assert f"Read(/{skill_root}/**)".startswith("Read(//")  # `//` = absolute path


@pytest.mark.parametrize("name", ["skills*", "a(1)", "x) Read (y", "tab\tbed"])
def test_trial_refuses_a_skill_dir_a_rule_would_read_as_a_pattern(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    from rote.eval.empirical import EmpiricalError

    skill = tmp_path / name
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: t\n---\nDo it.\n")
    _install_fake_claude(monkeypatch, [_result_event()], write_result={})
    with pytest.raises(EmpiricalError, match="cannot scope"):
        run_baseline_trial(skill, {}, model="m")


def test_a_skill_dir_with_a_plain_space_is_scoped_not_refused(tmp_path: Path) -> None:
    from rote.eval.empirical import skill_run_file_rules

    skill = tmp_path / "My Skills"
    skill.mkdir()
    root = skill.resolve().as_posix()
    assert skill_run_file_rules(skill) == ["Read(./**)", f"Read(/{root}/**)", "Edit(./**)"]


def test_trial_with_no_servers_is_still_strict_and_sees_none(
    skill_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty registry used to mean no --strict-mcp-config, so the run saw
    every server the user has in Claude Code (work servers, claude.ai
    connectors). Strict with an empty config is the only way to see none."""
    captured = _install_fake_claude(monkeypatch, [_result_event()], write_result={})
    run_baseline_trial(skill_dir, {}, model="m")
    args = captured["args"]
    assert "--strict-mcp-config" in args
    assert captured["mcp_config"] == {"mcpServers": {}}
    allowed = args[args.index("--allowedTools") + 1].split(",")
    assert not [t for t in allowed if t.startswith("mcp__")]
    assert "--disallowedTools" not in args


def test_trial_does_not_inherit_the_users_settings_or_permission_mode(
    skill_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--allowedTools only pre-approves. Under the user's `defaultMode: auto`
    a real run called a server the allowlist excluded and searched the web.
    The child loads none of the user's settings, runs in dontAsk (anything
    not allowed is denied), and has no Bash/WebSearch/WebFetch at all."""
    from rote.inference import CLI_CONTEXT_ISOLATION_FLAGS

    captured = _install_fake_claude(monkeypatch, [_result_event()], write_result={})
    run_baseline_trial(
        skill_dir,
        {},
        model="m",
        mcp_servers={"gmail": {"type": "http", "url": "https://g.example/mcp"}},
        mcp_tool_ids=["mcp__gmail__search_threads"],
        denied_mcp_tool_ids=["mcp__gmail__send_email", "mcp__gmail__delete_thread"],
    )
    args = captured["args"]
    # The same tuple the compiler driver and the judge lane use, in order.
    start = args.index(CLI_CONTEXT_ISOLATION_FLAGS[0])
    assert tuple(args[start : start + len(CLI_CONTEXT_ISOLATION_FLAGS)]) == (
        CLI_CONTEXT_ISOLATION_FLAGS
    )
    assert args[args.index("--setting-sources") + 1] == ""
    assert args[args.index("--permission-mode") + 1] == "dontAsk"
    assert args[args.index("--tools") + 1] == "Read,Write,Edit,Glob,Grep"
    for tool in ("Bash", "WebSearch", "WebFetch"):
        assert tool not in args[args.index("--tools") + 1].split(",")
        assert tool not in args[args.index("--allowedTools") + 1].split(",")
    assert args[args.index("--disallowedTools") + 1] == (
        "mcp__gmail__send_email,mcp__gmail__delete_thread"
    )
    # The skill's own files stay readable: its dir is a working directory
    # and a read rule names it.
    assert args[args.index("--add-dir") + 1] == str(skill_dir.resolve())
    allowed = args[args.index("--allowedTools") + 1].split(",")
    assert f"Read(/{skill_dir.resolve().as_posix()}/**)" in allowed


def test_trial_reports_the_servers_the_skill_names_but_never_called(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        "---\nname: t\n---\nCall `search` on the `mslearn` MCP server, then\n"
        "`read_wiki_structure` on the `deepwiki` MCP server, then\n"
        "`lookup` on the `catalog` MCP server.\n"
    )
    lines = [
        _tool_use_event("t1", "mcp__mslearn__search", {"q": "x"}),
        _tool_result_event("t1", "r"),
        _result_event(
            permission_denials=[
                {"tool_name": "mcp__catalog__lookup", "tool_use_id": "t2", "tool_input": {}}
            ]
        ),
    ]
    _install_fake_claude(monkeypatch, lines, write_result={"ok": True})
    run, _, _ = run_baseline_trial(
        skill,
        {},
        model="m",
        mcp_servers={
            "mslearn": {"type": "http", "url": "https://m.example/mcp"},
            "catalog": {"type": "http", "url": "https://c.example/mcp"},
        },
        mcp_tool_ids=["mcp__mslearn__search"],
        denied_mcp_tool_ids=["mcp__catalog__lookup"],
    )
    assert run.succeeded  # it wrote a result ...
    assert run.missing_mcp_servers == {  # ... without two of its servers
        "catalog": "the read-only gate refused mcp__catalog__lookup",
        "deepwiki": "not registered with rote: rote mcp add deepwiki <url>",
    }
    assert run.denied_tools == ("mcp__catalog__lookup",)
    assert "missing_mcp_servers" in run.flags
    assert not run.representative


def test_trial_that_calls_every_named_server_is_not_degraded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: t\n---\nUse the `deepwiki` MCP server.\n")
    lines = [
        _tool_use_event("t1", "mcp__deepwiki__read_wiki_structure", {"repoName": "a/b"}),
        _tool_result_event("t1", "r"),
        _result_event(),
    ]
    _install_fake_claude(monkeypatch, lines, write_result={"ok": True})
    run, _, _ = run_baseline_trial(
        skill,
        {},
        model="m",
        mcp_servers={"deepwiki": {"type": "http", "url": "https://d.example/mcp"}},
        mcp_tool_ids=["mcp__deepwiki__read_wiki_structure"],
    )
    assert run.missing_mcp_servers == {}
    assert run.denied_tools == ()
    assert run.flags == ()


def test_trial_failure_reports_error_and_flags(
    skill_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_claude(
        monkeypatch,
        [_result_event(subtype="error_max_turns", num_turns=60)],
        returncode=1,
    )
    run, _, _ = run_baseline_trial(skill_dir, {}, model="m")
    assert not run.succeeded
    assert "hit_max_turns" in run.flags


def test_trial_rejects_skill_dir_without_skill_md(tmp_path: Path) -> None:
    from rote.eval.empirical import EmpiricalError

    with pytest.raises(EmpiricalError, match="SKILL.md"):
        run_baseline_trial(tmp_path, {}, model="m")


# ───────── Orchestration + artifacts ─────────


def test_run_baseline_writes_artifacts(
    skill_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_mcp_stores(monkeypatch, tmp_path)  # empty registry → no MCP wiring
    lines = [
        _tool_use_event("t1", "mcp__gmail__search_threads", {"query": "q"}),
        _tool_result_event("t1", "r"),
        _result_event(),
    ]
    _install_fake_claude(monkeypatch, lines, write_result={"done": 1})

    out = tmp_path / "out"
    result = run_baseline(skill_dir, {"a": 1}, out, model="m", trials=2)

    assert len(result.runs) == 2
    assert result.read_only is True
    assert result.observed_servers == ["gmail"]

    baseline_dir = out / BASELINE_DIRNAME
    metrics = json.loads((baseline_dir / METRICS_FILENAME).read_text())
    assert metrics["trials"] == 2
    assert metrics["read_only"] is True
    assert metrics["observed_servers"] == ["gmail"]
    assert metrics["runs"][0]["turns"] == 7

    observed = json.loads((baseline_dir / OBSERVED_TOOLS_FILENAME).read_text())
    assert len(observed) == 2  # one per trial
    assert observed[0] == {
        "server": "gmail",
        "tool": "search_threads",
        "input": {"query": "q"},
        "result": "r",
        "is_error": False,
    }
    assert (baseline_dir / "trial-1.transcript.jsonl").is_file()
    assert (baseline_dir / "trial-2.transcript.jsonl").is_file()


def test_run_baseline_allow_writes_wildcards_all_servers(
    skill_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _isolate_mcp_stores(monkeypatch, tmp_path)
    registry.write_text(
        json.dumps({"version": 1, "servers": {"gmail": {"url": "https://g.example/mcp"}}}),
        encoding="utf-8",
    )
    captured = _install_fake_claude(monkeypatch, [_result_event()], write_result={})

    run_baseline(skill_dir, {}, tmp_path / "out", model="m", allow_writes=True)

    args = captured["args"]
    allowed = args[args.index("--allowedTools") + 1]
    assert "mcp__gmail__*" in allowed.split(",")
    assert "--disallowedTools" not in args  # writes allowed: nothing is held back


def test_run_baseline_read_only_gate_uses_annotations(
    skill_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only readOnlyHint tools are allowlisted; a server declaring none
    is skipped with an actionable reason."""
    registry = _isolate_mcp_stores(monkeypatch, tmp_path)
    registry.write_text(
        json.dumps(
            {
                "version": 1,
                "servers": {
                    "gmail": {"url": "https://g.example/mcp"},
                    "pushy": {"url": "https://p.example/mcp"},
                },
            }
        ),
        encoding="utf-8",
    )

    from mcp.types import Tool, ToolAnnotations

    def _Tool(name: str, read_only: bool | None) -> Tool:  # noqa: N802 — reads like the class it builds
        return Tool(
            name=name,
            input_schema={"type": "object"},
            annotations=None if read_only is None else ToolAnnotations(read_only_hint=read_only),
        )

    class _FakeClient:
        def __init__(self, tools: list[Tool]) -> None:
            self._tools = tools

        async def __aenter__(self) -> _FakeClient:
            return self

        async def __aexit__(self, *exc: object) -> None:
            return None

        async def list_tools(self) -> list[_Tool]:
            return self._tools

    clients = {
        "gmail": _FakeClient([_Tool("search_threads", True), _Tool("send_email", False)]),
        "pushy": _FakeClient([_Tool("push_record", None)]),
    }
    monkeypatch.setattr(
        "rote.mcp._runtime_helper.mcp_client",
        lambda server, url: clients[server],
    )
    captured = _install_fake_claude(monkeypatch, [_result_event()], write_result={})

    result = run_baseline(skill_dir, {}, tmp_path / "out", model="m")

    args = captured["args"]
    allowed = args[args.index("--allowedTools") + 1]
    assert "mcp__gmail__search_threads" in allowed
    assert "send_email" not in allowed
    assert "push_record" not in allowed
    # The excluded tool is denied outright (and so out of the model's
    # context), not merely left off an allowlist that denies nothing.
    assert args[args.index("--disallowedTools") + 1] == "mcp__gmail__send_email"
    # A server contributing no callable tool is not wired at all.
    assert set(captured["mcp_config"]["mcpServers"]) == {"gmail"}
    assert result.servers_wired == ("gmail",)
    assert "pushy" in result.servers_skipped
    assert "readOnlyHint" in result.servers_skipped["pushy"]


# ───────── CLI ─────────


def test_cli_baseline_json_output(
    skill_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    _isolate_mcp_stores(monkeypatch, tmp_path)
    _install_fake_claude(
        monkeypatch,
        [
            _tool_use_event("t1", "mcp__gmail__search_threads", {"q": "x"}),
            _tool_result_event("t1", "r"),
            _result_event(),
        ],
        write_result={"ok": True},
    )
    input_file = tmp_path / "input.json"
    input_file.write_text('{"campaign": "expo"}', encoding="utf-8")

    rc = cli_main(
        [
            "baseline",
            str(skill_dir),
            "--out",
            str(tmp_path / "out"),
            "--input",
            str(input_file),
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["observed_servers"] == ["gmail"]
    assert payload["read_only"] is True
    assert payload["runs"][0]["succeeded"] is True
    assert Path(payload["baseline_dir"]).is_dir()


def test_cli_baseline_rejects_bad_input(skill_dir: Path, tmp_path: Path, capsys: Any) -> None:
    bad = tmp_path / "input.json"
    bad.write_text("[1, 2]", encoding="utf-8")
    rc = cli_main(["baseline", str(skill_dir), "--out", str(tmp_path / "o"), "--input", str(bad)])
    assert rc == 2
    assert "JSON object" in capsys.readouterr().err


# ───────── Input derivation ─────────


def _install_fake_derive_claude(
    monkeypatch: pytest.MonkeyPatch, result_text: str, *, returncode: int = 0
) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def fake_run(args: list[str], **kwargs: Any) -> _FakeProc:
        captured["args"] = args
        return _FakeProc(json.dumps({"type": "result", "result": result_text}), returncode)

    monkeypatch.setattr("rote.eval.baseline.subprocess.run", fake_run)
    monkeypatch.setattr("rote.eval.baseline.shutil.which", lambda _: "/usr/bin/claude")
    return captured


def test_derive_input_payload_parses_plain_json(
    skill_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rote.eval.baseline import derive_input_payload

    captured = _install_fake_derive_claude(monkeypatch, '{"campaign": "expo-2026"}')
    payload = derive_input_payload(skill_dir)
    assert payload == {"campaign": "expo-2026"}
    # Cheap and toolless by construction, and blind to the user's setup:
    # no built-in tool, no MCP server, none of the user's settings.
    args = captured["args"]
    assert args[args.index("--tools") + 1] == ""
    assert args[args.index("--setting-sources") + 1] == ""
    assert "--strict-mcp-config" in args
    assert "--mcp-config" not in args  # strict with nothing passed = no servers
    assert args[args.index("--max-turns") + 1] == "2"


def test_derive_input_payload_strips_markdown_fences(
    skill_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rote.eval.baseline import derive_input_payload

    _install_fake_derive_claude(monkeypatch, '```json\n{"a": 1}\n```')
    assert derive_input_payload(skill_dir) == {"a": 1}


def test_derive_input_payload_rejects_non_object(
    skill_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rote.eval.baseline import derive_input_payload
    from rote.eval.empirical import EmpiricalError

    _install_fake_derive_claude(monkeypatch, "[1, 2, 3]")
    with pytest.raises(EmpiricalError, match="JSON object"):
        derive_input_payload(skill_dir)


def test_derive_input_payload_surfaces_agent_failure(
    skill_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rote.eval.baseline import derive_input_payload
    from rote.eval.empirical import EmpiricalError

    _install_fake_derive_claude(monkeypatch, "", returncode=1)
    with pytest.raises(EmpiricalError, match="exited 1"):
        derive_input_payload(skill_dir)


# ───────── CLI input resolution (derive + confirm) ─────────


def test_cli_baseline_derives_with_yes(
    skill_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    """No --input + --yes: derived proposal is accepted, saved, and used."""
    _isolate_mcp_stores(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "rote.eval.baseline.derive_input_payload",
        lambda *a, **k: {"campaign": "derived-expo"},
    )
    _install_fake_claude(monkeypatch, [_result_event()], write_result={"ok": 1})

    out = tmp_path / "out"
    rc = cli_main(["baseline", str(skill_dir), "--out", str(out), "--yes", "--json"])
    assert rc == 0
    derived = json.loads((out / "baseline" / "derived-input.json").read_text())
    assert derived == {"campaign": "derived-expo"}
    metrics = json.loads((out / "baseline" / "metrics.json").read_text())
    assert metrics["input"] == {"campaign": "derived-expo"}


def test_cli_baseline_requires_yes_when_not_a_tty(
    skill_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    _isolate_mcp_stores(monkeypatch, tmp_path)
    monkeypatch.setattr("rote.eval.baseline.derive_input_payload", lambda *a, **k: {"x": 1})
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)

    rc = cli_main(["baseline", str(skill_dir), "--out", str(tmp_path / "o")])
    assert rc == 2
    assert "--yes" in capsys.readouterr().err


def test_cli_baseline_interactive_decline_exits_cleanly(
    skill_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    _isolate_mcp_stores(monkeypatch, tmp_path)
    monkeypatch.setattr("rote.eval.baseline.derive_input_payload", lambda *a, **k: {"x": 1})
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")

    rc = cli_main(["baseline", str(skill_dir), "--out", str(tmp_path / "o")])
    assert rc == 0
    err = capsys.readouterr().err
    assert "declined" in err
    # The proposal survives for editing.
    assert (tmp_path / "o" / "baseline" / "derived-input.json").is_file()


# ───────── compile --baseline integration ─────────


def test_compile_baseline_runs_and_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    """--baseline: the raw skill runs before the compiler, its measured
    runs land in the --json payload, and artifacts are on disk."""
    from tests.test_cli import _install_mock_compiler

    _isolate_mcp_stores(monkeypatch, tmp_path)
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: t\n---\n")
    input_file = tmp_path / "input.json"
    input_file.write_text('{"a": 1}', encoding="utf-8")

    _install_mock_compiler(monkeypatch)
    _install_fake_claude(
        monkeypatch,
        [
            _tool_use_event("t1", "mcp__gmail__search_threads", {"q": "x"}),
            _tool_result_event("t1", "r"),
            _result_event(),
        ],
        write_result={"ok": True},
    )

    out = tmp_path / "out"
    rc = cli_main(
        [
            "compile",
            str(skill),
            "--runtime",
            "dbos",
            "--out",
            str(out),
            "--no-eval",
            "--json",
            "--baseline",
            "--baseline-input",
            str(input_file),
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["baseline"]["observed_servers"] == ["gmail"]
    assert payload["baseline"]["runs"][0]["turns"] == 7
    assert (out / "baseline" / "metrics.json").is_file()


def test_compile_without_baseline_prints_nudge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    from tests.test_cli import _install_mock_compiler

    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: t\n---\n")
    _install_mock_compiler(monkeypatch)

    rc = cli_main(
        ["compile", str(skill), "--runtime", "dbos", "--out", str(tmp_path / "out"), "--no-eval"]
    )
    assert rc == 0
    assert "--baseline" in capsys.readouterr().err


# ───────── Scorecard section ─────────


def test_render_baseline_markdown() -> None:
    from rote.eval.baseline import BaselineResult, render_baseline_markdown
    from rote.eval.empirical import MeasuredRun

    result = BaselineResult(
        skill_dir=Path("/s"),
        input_payload={},
        model="claude-test",
        read_only=True,
        runs=(
            MeasuredRun(wall_seconds=61.0, output={}, turns=12, cost_usd=0.31, model="claude-test"),
        ),
        observations=(
            ObservedToolCall(server="gmail", tool="search_threads", input={}),
            ObservedToolCall(server="gmail", tool="get_thread", input={}),
        ),
    )
    md = render_baseline_markdown(result)
    assert "## Measured baseline (1/1" in md
    assert "| Wall clock (s) | 61 |" in md
    assert "| Cost (USD) | 0.31 |" in md
    assert "gmail ×2" in md
    assert "read-only MCP gate" in md


# ───────── Inferred schemas artifact + cross-check surfacing ─────────


def test_run_baseline_writes_inferred_schemas(
    skill_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_mcp_stores(monkeypatch, tmp_path)
    _install_fake_claude(
        monkeypatch,
        [
            _tool_use_event("t1", "mcp__gmail__search_threads", {"query": "q"}),
            _tool_result_event("t1", [{"type": "text", "text": '{"threads": [1]}'}]),
            _result_event(),
        ],
        write_result={"ok": 1},
    )
    out = tmp_path / "out"
    run_baseline(skill_dir, {}, out, model="m")

    from rote.eval.baseline import INFERRED_SCHEMAS_FILENAME

    inferred = json.loads((out / "baseline" / INFERRED_SCHEMAS_FILENAME).read_text())
    (entry,) = inferred
    assert entry["server"] == "gmail"
    assert entry["tool"] == "search_threads"
    assert entry["input_schema"]["required"] == ["query"]
    assert entry["output_schema"]["properties"]["threads"]["type"] == "array"


def test_load_observations_round_trips(
    skill_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rote.eval.baseline import load_observations

    _isolate_mcp_stores(monkeypatch, tmp_path)
    _install_fake_claude(
        monkeypatch,
        [
            _tool_use_event("t1", "mcp__gmail__get_thread", {"id": "x"}),
            _tool_result_event("t1", "r"),
            _result_event(),
        ],
        write_result={},
    )
    out = tmp_path / "out"
    result = run_baseline(skill_dir, {}, out, model="m")
    loaded = load_observations(out / "baseline")
    assert loaded == list(result.observations)
    # Absent dir → empty, not an error.
    assert load_observations(tmp_path / "nowhere") == []


def test_compile_baseline_cross_check_flags_unbound_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    """BDR's expected IR has no mcp bindings, so an observed gmail call
    must surface as observed_only — the loud missed-requirement case."""
    from tests.test_cli import _install_mock_compiler

    _isolate_mcp_stores(monkeypatch, tmp_path)
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: t\n---\n")
    input_file = tmp_path / "input.json"
    input_file.write_text('{"a": 1}', encoding="utf-8")

    _install_mock_compiler(monkeypatch)
    _install_fake_claude(
        monkeypatch,
        [
            _tool_use_event("t1", "mcp__gmail__search_threads", {"q": "x"}),
            _tool_result_event("t1", "r"),
            _result_event(),
        ],
        write_result={"ok": True},
    )

    rc = cli_main(
        [
            "compile",
            str(skill),
            "--runtime",
            "dbos",
            "--out",
            str(tmp_path / "out"),
            "--no-eval",
            "--json",
            "--baseline",
            "--baseline-input",
            str(input_file),
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["mcp_cross_check"]["observed_only"] == [
        {"server": "gmail", "tool": "search_threads", "observed_calls": 1}
    ]
    assert "likely a missed requirement" in captured.err


def test_compile_cross_checks_prior_baseline_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    """A baseline run earlier into the same --out is picked up from disk —
    no --baseline flag needed on the compile call."""
    from tests.test_cli import _install_mock_compiler

    _isolate_mcp_stores(monkeypatch, tmp_path)
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: t\n---\n")
    out = tmp_path / "out"

    # Step 1: standalone baseline into the out dir.
    _install_fake_claude(
        monkeypatch,
        [
            _tool_use_event("t1", "mcp__gmail__search_threads", {"q": "x"}),
            _tool_result_event("t1", "r"),
            _result_event(),
        ],
        write_result={"ok": True},
    )
    input_file = tmp_path / "input.json"
    input_file.write_text("{}", encoding="utf-8")
    assert cli_main(["baseline", str(skill), "--out", str(out), "--input", str(input_file)]) == 0
    capsys.readouterr()

    # Step 2: compile later, same out dir, no --baseline.
    _install_mock_compiler(monkeypatch)
    rc = cli_main(
        ["compile", str(skill), "--runtime", "dbos", "--out", str(out), "--no-eval", "--json"]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mcp_cross_check"]["observed_only"][0]["server"] == "gmail"


def test_compile_enriches_contracts_from_baseline_observations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    """End-to-end: observed traffic types the matching node's contracts,
    the rewritten pipeline.yaml carries them, and the emitted stub
    documents them."""
    from rote.compiler import CompilationResult
    from rote.ir import load_pipeline

    _isolate_mcp_stores(monkeypatch, tmp_path)
    repo_root = Path(__file__).resolve().parent.parent
    dm_yaml = repo_root / "examples" / "deal-monitor" / "expected" / "pipeline.yaml"
    dm_pipeline = load_pipeline(dm_yaml)

    class _DealMonitorCompiler:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def compile(self, skill_path, output_dir, update=False, **_kwargs):  # noqa: ANN001
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "pipeline.yaml").write_text(
                dm_yaml.read_text(encoding="utf-8"), encoding="utf-8"
            )
            return CompilationResult(
                pipeline=dm_pipeline,
                output_dir=output_dir,
                driver_name="mock",
                driver_metadata={},
            )

    monkeypatch.setattr("rote.cli.Compiler", _DealMonitorCompiler)

    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: t\n---\n")
    input_file = tmp_path / "input.json"
    input_file.write_text("{}", encoding="utf-8")

    _install_fake_claude(
        monkeypatch,
        [
            _tool_use_event("t1", "mcp__slack__slack_read_channel", {"channel": "C0EXAMPLE000"}),
            _tool_result_event("t1", [{"type": "text", "text": '{"messages": ["deal one"]}'}]),
            _result_event(),
        ],
        write_result={"ok": True},
    )

    out = tmp_path / "out"
    rc = cli_main(
        [
            "compile",
            str(skill),
            "--runtime",
            "dbos",
            "--out",
            str(out),
            "--no-eval",
            "--baseline",
            "--baseline-input",
            str(input_file),
        ]
    )
    assert rc == 0
    err = capsys.readouterr().err
    assert "typed 1 node contract(s)" in err

    # The rewritten IR is the source of truth for the contracts…
    reloaded = load_pipeline(out / "compiled" / "pipeline.yaml")
    node = reloaded.node_by_id("fetch_intake_messages")
    assert node.input_schema["required"] == ["channel"]
    assert node.output_schema["properties"]["messages"]["type"] == "array"
    # …and the emitted stub documents them.
    stub = (out / "runtime" / "dbos" / "extracted" / "slack.py").read_text(encoding="utf-8")
    assert "Output contract (JSON Schema, from observed real payloads):" in stub
    assert '"messages"' in stub


# ───────── Probe artifacts fed into the compiler ─────────


def test_materialize_probe_context_copies_artifacts(tmp_path: Path) -> None:
    from rote.compiler import PROBE_CONTEXT_DIRNAME, Compiler

    probe = tmp_path / "baseline"
    probe.mkdir()
    (probe / "observed-tools.json").write_text("[]", encoding="utf-8")
    (probe / "inferred-schemas.json").write_text("[]", encoding="utf-8")
    work = tmp_path / "work"
    work.mkdir()

    instructions = Compiler._materialize_probe_context(work, probe)
    assert instructions is not None
    assert "GROUND TRUTH" in instructions
    assert "mcp: binding" in instructions
    ctx = work / PROBE_CONTEXT_DIRNAME
    assert (ctx / "observed-tools.json").is_file()
    assert (ctx / "inferred-schemas.json").is_file()


def test_materialize_probe_context_absent_artifacts_is_none(tmp_path: Path) -> None:
    from rote.compiler import Compiler

    work = tmp_path / "work"
    work.mkdir()
    assert Compiler._materialize_probe_context(work, tmp_path / "nowhere") is None
    assert not list(work.iterdir())  # no empty probe-context dir left behind


def test_compile_threads_probe_dir_to_driver_instructions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end through the real Compiler with a fake driver: probe
    artifacts land in the work dir and the prompt instructions."""
    import asyncio

    from rote.compiler import PROBE_CONTEXT_DIRNAME, Compiler
    from rote.compiler.drivers import DriverResult

    repo_root = Path(__file__).resolve().parent.parent
    bdr_yaml = repo_root / "examples" / "bdr-outreach" / "expected" / "pipeline.yaml"

    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: t\n---\n# T\n\n## Step 1\ndo\n")
    probe = tmp_path / "out" / "baseline"
    probe.mkdir(parents=True)
    (probe / "observed-tools.json").write_text(
        '[{"server": "gmail", "tool": "search_threads", "input": {}, '
        '"result": null, "is_error": false}]',
        encoding="utf-8",
    )

    captured: dict[str, Any] = {}

    class _FakeDriver:
        name = "fake"

        def is_available(self):  # noqa: ANN201
            return True, ""

        async def run(self, *, skill_dir, compiler_skill_dir, work_dir, **kwargs):  # noqa: ANN001, ANN201
            captured["extra_instructions"] = kwargs.get("extra_instructions")
            captured["probe_files"] = sorted(
                p.name for p in (Path(work_dir) / PROBE_CONTEXT_DIRNAME).iterdir()
            )
            (Path(work_dir) / "pipeline.yaml").write_text(
                bdr_yaml.read_text(encoding="utf-8"), encoding="utf-8"
            )
            return DriverResult(
                pipeline_yaml_path=Path(work_dir) / "pipeline.yaml",
                work_dir=Path(work_dir),
                driver_name="fake",
            )

    compiler = Compiler(system1=False, agent="claude")
    monkeypatch.setattr(compiler, "select_driver", lambda: _FakeDriver())

    result = asyncio.run(compiler.compile(skill, tmp_path / "out" / "compiled", probe_dir=probe))
    assert result.pipeline.name == "bdr-campaign"
    assert captured["probe_files"] == ["observed-tools.json"]
    assert "GROUND TRUTH" in captured["extra_instructions"]


# ───────── Generated-test verification ─────────


def test_run_generated_tests_none_without_tests(tmp_path: Path) -> None:
    from rote.compiler.verify import run_generated_tests

    assert run_generated_tests(tmp_path) is None
    (tmp_path / "tests").mkdir()
    assert run_generated_tests(tmp_path) is None  # dir but no test files


def test_run_generated_tests_reports_pass_and_fail(tmp_path: Path) -> None:
    from rote.compiler.verify import run_generated_tests

    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_ok.py").write_text("def test_ok():\n    assert 1 + 1 == 2\n")
    result = run_generated_tests(tmp_path)
    assert result is not None and result.passed is True

    (tests / "test_bad.py").write_text("def test_bad():\n    assert False\n")
    result = run_generated_tests(tmp_path)
    assert result is not None and result.status == "failed"
    assert "failed" in result.summary


def test_compile_reports_generated_tests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    """A compilation whose agent wrote tests gets them run and reported."""
    from rote.compiler import CompilationResult
    from rote.ir import load_pipeline

    repo_root = Path(__file__).resolve().parent.parent
    bdr_yaml = repo_root / "examples" / "bdr-outreach" / "expected" / "pipeline.yaml"

    class _TestWritingCompiler:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def compile(self, skill_path, output_dir, update=False, **_kwargs):  # noqa: ANN001
            output_dir = Path(output_dir)
            (output_dir / "tests").mkdir(parents=True, exist_ok=True)
            (output_dir / "pipeline.yaml").write_text(
                bdr_yaml.read_text(encoding="utf-8"), encoding="utf-8"
            )
            (output_dir / "tests" / "test_extracted.py").write_text(
                "def test_fixture():\n    assert True\n"
            )
            return CompilationResult(
                pipeline=load_pipeline(bdr_yaml),
                output_dir=output_dir,
                driver_name="mock",
                driver_metadata={},
            )

    monkeypatch.setattr("rote.cli.Compiler", _TestWritingCompiler)
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: t\n---\n")

    rc = cli_main(
        [
            "compile",
            str(skill),
            "--runtime",
            "dbos",
            "--out",
            str(tmp_path / "out"),
            "--no-eval",
            "--json",
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["generated_tests"]["passed"] is True
    assert "generated tests passed" in captured.err


def test_compile_types_a_wrapped_result_as_the_step_receives_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    """DeepWiki (fastmcp) records `{"result": "..."}` and hands the step the
    string. The compile's own record of the declared outputSchema decides."""
    from rote.compiler import CompilationResult
    from rote.compiler.tool_grounding import DECLARED_FILENAME
    from rote.ir import Pipeline

    _isolate_mcp_stores(monkeypatch, tmp_path)
    fixtures = json.loads(
        (Path(__file__).parent / "fixtures" / "mcp_results" / "docs_digest.json").read_text()
    )
    declared = [
        {"server": d["server"], "tool": d["tool"], "output_schema": d["output_schema"]}
        for d in fixtures["declared"]
    ]
    recorded = next(o for o in fixtures["observations"] if o["server"] == "deepwiki")
    pipeline = Pipeline.model_validate(
        {
            "name": "wiki",
            "version": "0.1.0",
            "input": {"type": "In"},
            "nodes": [
                {
                    "id": "outline",
                    "kind": "external_call",
                    "description": "d",
                    "mcp": {"server": "deepwiki", "tool": "read_wiki_structure"},
                    "output": {"result": "string"},
                }
            ],
            "edges": [],
        }
    )

    class _WikiCompiler:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def compile(self, skill_path, output_dir, update=False, **_kwargs):  # noqa: ANN001
            from rote.ir import save_pipeline

            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            save_pipeline(pipeline, output_dir / "pipeline.yaml")
            (output_dir / DECLARED_FILENAME).write_text(json.dumps(declared), encoding="utf-8")
            return CompilationResult(
                pipeline=pipeline, output_dir=output_dir, driver_name="mock", driver_metadata={}
            )

    monkeypatch.setattr("rote.cli.Compiler", _WikiCompiler)
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: t\n---\n")
    input_file = tmp_path / "input.json"
    input_file.write_text("{}", encoding="utf-8")
    _install_fake_claude(
        monkeypatch,
        [
            _tool_use_event("t1", "mcp__deepwiki__read_wiki_structure", recorded["input"]),
            _tool_result_event("t1", recorded["result"]),
            _result_event(),
        ],
        write_result={"ok": True},
    )
    out = tmp_path / "out"
    argv = ["compile", str(skill), "--runtime", "dbos", "--out", str(out), "--no-eval", "--json"]
    assert cli_main([*argv, "--baseline", "--baseline-input", str(input_file)]) == 0
    payload = json.loads(capsys.readouterr().out)
    from rote.ir import load_pipeline

    node = load_pipeline(out / "compiled" / "pipeline.yaml").node_by_id("outline")
    assert node.output_schema == {"type": "string"}
    # The step gets a string, so the node's declared `result` key cannot come from it.
    [mismatch] = payload["mcp_cross_check"]["output_mismatch"]
    assert mismatch["node"] == "outline"
