"""A skill's bundled tools/list snapshot (``mcp-tools.json``).

A skill may ship its MCP tools' schemas beside its SKILL.md, the way it
ships ``sample-input.json``. A compile with nothing registered grounds tool
arguments in it; a live ``tools/list`` of the same server always wins; when
both exist and differ for a bound tool, the snapshot is stale and the
compile says so; and the summary and report name what grounded each server.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from rote.compiler import Compiler, CompilerError, tool_grounding
from rote.compiler.grounding import Grounding
from rote.compiler.tool_grounding import (
    ToolSchema,
    bundled,
    prefer_live,
    schema_sources,
    stale_bundled,
)
from rote.compiler.verify import Verification, verify_deliverable
from rote.tool_schema import BundledToolsError, load_bundled_tools
from tests.test_tool_grounding import SEARCH, _pipeline, _root

REPO = Path(__file__).resolve().parent.parent

READ = {
    "type": "object",
    "properties": {"file_id": {"type": "string"}},
    "required": ["file_id"],
}


def _snapshot(skill: Path, servers: dict[str, dict[str, dict[str, Any]]]) -> Path:
    skill.mkdir(parents=True, exist_ok=True)
    doc = {
        "servers": {
            server: {
                "tools": [
                    {"name": name, "description": "d", "inputSchema": schema}
                    for name, schema in tools.items()
                ]
            }
            for server, tools in servers.items()
        }
    }
    (skill / "mcp-tools.json").write_text(json.dumps(doc), encoding="utf-8")
    return skill


def _live(server: str, tool: str, schema: dict[str, Any]) -> ToolSchema:
    return ToolSchema(server, tool, schema, f"tools/list on the registered {server!r}", True)


# ───────── reading it ─────────


def test_a_skill_without_a_snapshot_bundles_nothing(tmp_path: Path) -> None:
    assert load_bundled_tools(tmp_path) == {}
    assert bundled(tmp_path) == {}


@pytest.mark.parametrize(
    "text",
    [
        "{not json",
        '{"tools": []}',
        '{"servers": {"gmail": {}}}',
        '{"servers": {"gmail": {"tools": [{"name": "x"}]}}}',
    ],
)
def test_a_malformed_snapshot_is_an_error_not_skipped(tmp_path: Path, text: str) -> None:
    (tmp_path / "mcp-tools.json").write_text(text, encoding="utf-8")
    with pytest.raises(BundledToolsError, match="mcp-tools.json"):
        load_bundled_tools(tmp_path)


def test_the_ops_report_snapshot_is_a_declaration_of_gmails_query() -> None:
    snapshot = bundled(REPO / "examples" / "ops-report" / "skill")
    search = snapshot[("gmail", "search_threads")]
    assert search.declared and search.origin == "bundled"
    assert search.input_schema["required"] == ["query"]
    assert "mcp-tools.json" in search.source


# ───────── live wins ─────────


def test_a_live_listing_replaces_the_snapshot_for_its_server_only() -> None:
    snapshot = {
        ("gmail", "search_threads"): ToolSchema(
            "gmail", "search_threads", SEARCH, "b", True, origin="bundled"
        ),
        ("gmail", "old_tool"): ToolSchema("gmail", "old_tool", READ, "b", True, origin="bundled"),
        ("google_drive", "read_file_content"): ToolSchema(
            "google_drive", "read_file_content", READ, "b", True, origin="bundled"
        ),
    }
    live = {("gmail", "search_threads"): _live("gmail", "search_threads", READ)}
    merged = prefer_live(live, snapshot)
    assert merged[("gmail", "search_threads")].origin == "live"
    # The live server no longer lists it, so the snapshot cannot vouch for it.
    assert ("gmail", "old_tool") not in merged
    assert merged[("google_drive", "read_file_content")].origin == "bundled"


def test_the_compile_grounds_in_the_snapshot_with_nothing_registered(tmp_path: Path) -> None:
    skill = _snapshot(tmp_path / "skill", {"gmail": {"search_threads": SEARCH}})
    (skill / "SKILL.md").write_text("Search Gmail.\n", encoding="utf-8")
    tools, _ = asyncio.run(Compiler()._tool_schemas((), skill, bundled(skill)))
    assert tools[("gmail", "search_threads")].origin == "bundled"
    assert tools[("gmail", "search_threads")].input_schema == SEARCH


def test_the_compile_prefers_a_live_listing_of_a_registered_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rote.mcp.registry import McpRegistry, McpServerConfig, save_registry

    save_registry(McpRegistry(servers={"gmail": McpServerConfig(url="https://gmail.example/mcp")}))
    skill = _snapshot(tmp_path / "skill", {"gmail": {"search_threads": SEARCH}})
    (skill / "SKILL.md").write_text("Search Gmail.\n", encoding="utf-8")

    async def _discover(servers: Any, on_skip: Any = None, **_: Any) -> dict[Any, Any]:
        return {("gmail", "search_threads"): _live("gmail", "search_threads", READ)}

    monkeypatch.setattr(tool_grounding, "discover", _discover)
    tools, _ = asyncio.run(Compiler()._tool_schemas((), skill, bundled(skill)))
    assert tools[("gmail", "search_threads")].origin == "live"
    assert tools[("gmail", "search_threads")].input_schema == READ


# ───────── stale, and which source ─────────


def test_a_differing_live_schema_marks_the_snapshot_stale_for_a_bound_tool(
    tmp_path: Path,
) -> None:
    pipeline = _pipeline(_root(tmp_path))  # binds gmail.search_threads
    snapshot = {
        ("gmail", "search_threads"): ToolSchema(
            "gmail", "search_threads", SEARCH, "b", True, origin="bundled"
        )
    }
    same = {("gmail", "search_threads"): _live("gmail", "search_threads", SEARCH)}
    changed = {("gmail", "search_threads"): _live("gmail", "search_threads", READ)}
    assert stale_bundled(pipeline, same, snapshot) == []
    assert stale_bundled(pipeline, changed, snapshot) == ["gmail.search_threads"]
    # Grounded by the snapshot itself: nothing to compare against.
    assert stale_bundled(pipeline, snapshot, snapshot) == []
    # A tool the pipeline does not bind is not this compile's concern.
    other = {("gmail", "send"): _live("gmail", "send", READ)}
    assert (
        stale_bundled(pipeline, other, {("gmail", "send"): snapshot[("gmail", "search_threads")]})
        == []
    )


def test_verification_names_each_bound_servers_source_and_stale_tools(tmp_path: Path) -> None:
    root = _root(tmp_path)
    snapshot = {
        ("gmail", "search_threads"): ToolSchema(
            "gmail", "search_threads", SEARCH, "b", True, origin="bundled"
        )
    }
    from_snapshot = verify_deliverable(
        root,
        grounding=Grounding(tools=snapshot, bundled=snapshot),
        run_tests=False,
        trust_skill=True,
    )
    assert from_snapshot.tool_sources == {"gmail": "the skill's bundled snapshot"}
    assert from_snapshot.stale_bundled_tools == []
    live = {("gmail", "search_threads"): _live("gmail", "search_threads", READ)}
    from_live = verify_deliverable(
        root, grounding=Grounding(tools=live, bundled=snapshot), run_tests=False, trust_skill=True
    )
    assert from_live.tool_sources == {"gmail": "a live tools/list"}
    assert from_live.stale_bundled_tools == ["gmail.search_threads"]
    assert schema_sources(_pipeline(root), {}) == {"gmail": "no schema"}


def test_the_summary_and_report_name_the_source_and_the_stale_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from rote.cli import main as cli_main
    from tests.test_cli import _install_mock_compiler

    root = _root(tmp_path / "ir")
    pipeline = _pipeline(root)
    _install_mock_compiler(
        monkeypatch,
        root / "pipeline.yaml",
        verification=Verification(
            pipeline=pipeline,
            tool_sources={"gmail": "a live tools/list"},
            stale_bundled_tools=["gmail.search_threads"],
        ),
    )
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: t\n---\n", encoding="utf-8")
    out = tmp_path / "out"
    (out / "compiled").mkdir(parents=True)
    (out / "compiled" / "compile-report.md").write_text("# report\n", encoding="utf-8")
    cli_main(
        ["compile", str(skill), "--local", "--runtime", "dbos", "--no-eval", "--out", str(out)]
    )
    summary = capsys.readouterr().out.split("rote compile: ✓", 1)[1]
    assert "  tool schemas: gmail from a live tools/list" in summary
    assert (
        "WARNING: the skill's bundled mcp-tools.json is stale for gmail.search_threads" in summary
    )
    report = (out / "compiled" / "compile-report.md").read_text(encoding="utf-8")
    assert "- Tool schemas: gmail from a live tools/list." in report
    assert (
        "- WARNING: the skill's bundled mcp-tools.json is stale for gmail.search_threads" in report
    )


def test_a_malformed_snapshot_fails_the_compile_before_any_agent_runs(tmp_path: Path) -> None:
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: t\n---\nSearch Gmail.\n", encoding="utf-8")
    (skill / "mcp-tools.json").write_text("{broken", encoding="utf-8")

    class _Never:
        name = "never"

        def is_available(self) -> tuple[bool, str]:
            return True, ""

        async def run(self, *a: Any, **k: Any) -> Any:
            raise AssertionError("the agent ran")

    compiler = Compiler(system1=False, compiler_skill_dir=tmp_path, trust_skill=True)
    compiler.select_driver = lambda: _Never()  # type: ignore[method-assign,return-value]
    with pytest.raises(CompilerError, match="mcp-tools.json: not valid JSON"):
        asyncio.run(compiler.compile(skill, tmp_path / "out"))


def test_the_fixtures_refuse_a_snapshot_that_lists_other_tools(tmp_path: Path) -> None:
    """cases.yaml and the snapshot are one description of the servers: if
    they name different tools, the bench stops instead of guessing."""
    import shutil

    from rote.bench.fidelity import load_cases

    example = tmp_path / "ops-report"
    shutil.copytree(REPO / "examples" / "ops-report" / "fidelity", example / "fidelity")
    _snapshot(example / "skill", {"gmail": {"search_threads": SEARCH}, "google_drive": {}})
    with pytest.raises(ValueError, match="list different google_drive tools"):
        load_cases(example)
