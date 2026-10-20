"""Grounded MCP tool arguments: the tool's own inputSchema decides the names.

Covers :mod:`rote.compiler.tool_grounding` (discovery over a live
``tools/list``, schemas inferred from recorded calls, the static check,
open questions), the prefix run's value check, and the wiring into the
compiler, the CLI and the bench. No inference: deliverables are written
by hand, and the only servers are the bench's fixture servers.
"""

from __future__ import annotations

import asyncio
import json
import textwrap
from pathlib import Path

import pytest
import yaml

from rote.compiler import Compiler
from rote.compiler.drivers import CompilerDriver, DriverResult
from rote.compiler.grounding import GROUNDING_DIRNAME, Grounding, Sample
from rote.compiler.prefix import run_prefix
from rote.compiler.tool_grounding import (
    TOOL_SCHEMAS_FILENAME,
    ToolSchema,
    brief,
    discover,
    recorded,
    tool_findings,
    tool_questions,
)
from rote.compiler.verify import verify_deliverable
from rote.contracts import implementation_status
from rote.eval.baseline import ObservedToolCall
from rote.ir import Pipeline, load_pipeline

REPO = Path(__file__).resolve().parent.parent
OPS = REPO / "examples" / "ops-report"

SEARCH = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "A Gmail search query"},
        "max_results": {"type": "integer"},
    },
    "required": ["query"],
}
DECLARED = {
    ("gmail", "search_threads"): ToolSchema(
        "gmail", "search_threads", SEARCH, "tools/list on the registered 'gmail'", True
    )
}

_IR = """\
name: ops
input:
  type: Day
  required: [report_date]
nodes:
  - id: search
    kind: external_call
    description: search mail
    impl: extracted/mail.py:search
    mcp: {mcp}
    inputs:
      {param}: pipeline.input.report_date
{constants}
  - id: count
    kind: pure_function
    description: count threads
    impl: extracted/mail.py:count
    inputs:
      threads: search.output
edges:
  - {{from: search, to: count}}
entry_nodes: [search]
exit_nodes: [count]
"""

_MAIL = """\
def search(**kwargs):
    raise NotImplementedError("runs as the mcp binding")


def count(threads):
    return {"n": len(threads)}
"""


def _root(tmp_path: Path, *, param: str = "query", mcp: str = "", constants: str = "") -> Path:
    binding = "{server: gmail, tool: search_threads" + (f", {mcp}" if mcp else "") + "}"
    text = _IR.format(mcp=binding, param=param, constants=constants)
    for rel, body in {"pipeline.yaml": text, "extracted/mail.py": _MAIL}.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(textwrap.dedent(body), encoding="utf-8")
    return tmp_path


def _pipeline(root: Path) -> Pipeline:
    return load_pipeline(root / "pipeline.yaml")


# ───────── the static check ─────────


def test_a_guessed_argument_name_fails_naming_the_tools_parameters(tmp_path: Path) -> None:
    """main's ops-report compile: `search_query` where the tool takes `query`."""
    [finding] = tool_findings(_pipeline(_root(tmp_path, param="search_query")), DECLARED)
    assert finding.kind == "tool_arguments_invalid" and finding.node_id == "search"
    assert "missing required argument(s): query" in finding.message
    assert "`query` (string, required), `max_results` (integer)" in finding.message


def test_the_tools_own_names_pass(tmp_path: Path) -> None:
    assert tool_findings(_pipeline(_root(tmp_path)), DECLARED) == []


def test_mcp_args_maps_an_input_to_the_tools_name(tmp_path: Path) -> None:
    root = _root(tmp_path, param="search_text", mcp="args: {query: search_text}")
    assert tool_findings(_pipeline(root), DECLARED) == []


def test_an_argument_the_tool_does_not_take_fails(tmp_path: Path) -> None:
    root = _root(tmp_path, mcp="args: {query: query, label: {literal: dock}}")
    [finding] = tool_findings(_pipeline(root), DECLARED)
    assert "unknown argument(s): label (the tool takes: max_results, query)" in finding.message
    lenient = {
        k: ToolSchema(v.server, v.tool, {**SEARCH, "additionalProperties": True}, v.source, True)
        for k, v in DECLARED.items()
    }
    assert tool_findings(_pipeline(root), lenient) == []


def test_a_constant_of_the_wrong_type_fails(tmp_path: Path) -> None:
    root = _root(
        tmp_path,
        mcp="args: {query: query, max_results: limit}",
        constants='    constants:\n      limit: "10"',
    )
    [finding] = tool_findings(_pipeline(root), DECLARED)
    assert "argument 'max_results' must be integer, got str" in finding.message


def test_an_inferred_schema_never_fails_a_compile_but_flags_unseen_keys(tmp_path: Path) -> None:
    """Recorded calls show what someone sent, not what the tool takes."""
    observed = (ObservedToolCall("gmail", "search_threads", {"query": "x"}, "[]"),)
    inferred = recorded(observed)
    assert not inferred[("gmail", "search_threads")].declared
    root = _root(tmp_path, param="search_query")
    assert tool_findings(_pipeline(root), inferred) == []
    [(node, text)] = tool_questions(_pipeline(root), inferred)
    assert node == "search" and "`search_query`, which no recorded call used" in text


def test_without_a_schema_the_guessed_arguments_are_an_open_question(tmp_path: Path) -> None:
    [(node, text)] = tool_questions(_pipeline(_root(tmp_path, param="search_query")), {})
    assert node == "search"
    assert "arguments to gmail.search_threads (`search_query`) are guessed" in text
    assert tool_questions(_pipeline(_root(tmp_path / "ok")), DECLARED) == []


def test_verification_sends_a_bad_argument_to_repair(tmp_path: Path) -> None:
    root = _root(tmp_path, param="search_query")
    v = verify_deliverable(root, grounding=Grounding(tools=DECLARED), run_tests=False)
    assert "tool_arguments_invalid" in {f.kind for f in v.errors}
    assert "`query` (string, required)" in (v.repair_instruction(root) or "")
    bare = verify_deliverable(root, grounding=Grounding(), run_tests=False)
    assert [q.node_id for q in bare.questions if "are guessed" in q.text] == ["search"]


def test_the_brief_spells_out_the_tools_the_skill_names() -> None:
    other = ToolSchema("gmail", "send_message", {"type": "object"}, "x", True)
    text = brief(
        {**DECLARED, ("gmail", "send_message"): other},
        "Gmail MCP: search_threads",
        Path("/w/grounding/tool-schemas.json"),
    )
    assert "TOOL CONTRACTS" in text and "/w/grounding/tool-schemas.json" in text
    assert "- `gmail` / `search_threads`: `query` (string, required)" in text
    assert "send_message" not in text  # in the file, not the prompt
    assert brief({}, "anything", Path("x")) == ""


# ───────── the prefix checks values ─────────


def test_the_prefix_checks_the_values_it_would_send(tmp_path: Path) -> None:
    """The static check sees only names for payload-taken arguments."""
    root = _root(tmp_path)
    pipeline = _pipeline(root)
    impls = implementation_status(pipeline, root)
    bad = Sample(payload={"report_date": 20261019}, source="sample.json")
    run = run_prefix(pipeline, root, bad, impls, trust_skill=True, tools=DECLARED)
    [finding] = run.findings
    assert finding.kind == "tool_arguments_invalid"
    assert "argument 'query' must be string, got int" in finding.message
    good = Sample(payload={"report_date": "2026-10-19"}, source="sample.json")
    assert run_prefix(pipeline, root, good, impls, trust_skill=True, tools=DECLARED).findings == []


# ───────── discovery, against the bench's real fixture servers ─────────


def test_discovery_reads_the_fixture_servers_own_tools_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One schema: what the compile reads is what the fixtures enforce."""
    pytest.importorskip("fastmcp")
    from rote.bench.runner import tool_servers

    with tool_servers(OPS, tmp_path, {}) as env:
        monkeypatch.setenv("ROTE_MCP_CONFIG", env["ROTE_MCP_CONFIG"])
        found = asyncio.run(discover())
    cases = yaml.safe_load((OPS / "fidelity" / "cases.yaml").read_text())
    for server, spec in cases["servers"].items():
        for tool, defn in spec["tools"].items():
            schema = found[(server, tool)]
            assert schema.declared
            assert schema.input_schema["properties"] == defn["parameters"]["properties"]
            assert schema.input_schema.get("required") == defn["parameters"].get("required")


def test_an_unreachable_server_is_skipped_with_its_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("fastmcp")
    registry = tmp_path / "mcp.json"
    registry.write_text(
        json.dumps({"version": 1, "servers": {"gone": {"url": "http://127.0.0.1:9/mcp"}}})
    )
    monkeypatch.setenv("ROTE_MCP_CONFIG", str(registry))
    skipped: list[str] = []
    assert asyncio.run(discover(on_skip=lambda s, why: skipped.append(s))) == {}
    assert skipped == ["gone"]


def test_the_bench_gives_each_job_its_own_registry(tmp_path: Path) -> None:
    from rote.bench.runner import tool_servers

    with tool_servers(REPO / "examples" / "invoice-push", tmp_path, {"A": "1"}) as env:
        assert env == {"A": "1"}  # no fixture servers, nothing registered
    pytest.importorskip("fastmcp")
    with tool_servers(OPS, tmp_path, {"A": "1"}) as env:
        registry = json.loads(Path(env["ROTE_MCP_CONFIG"]).read_text())
        assert Path(env["ROTE_MCP_CONFIG"]).is_relative_to(tmp_path)
        assert sorted(registry["servers"]) == ["gmail", "google_drive"]


# ───────── the compiler and the CLI ─────────


class _Driver(CompilerDriver):
    name = "tools"

    def __init__(self) -> None:
        self.instructions: str | None = None
        self.saw_schemas = False

    def is_available(self) -> tuple[bool, str]:
        return (True, "")

    async def run(  # type: ignore[override]
        self, skill_dir, compiler_skill_dir, work_dir, extra_instructions=None, on_event=None
    ) -> DriverResult:
        self.instructions = extra_instructions
        self.saw_schemas = (work_dir / GROUNDING_DIRNAME / TOOL_SCHEMAS_FILENAME).is_file()
        _root(work_dir)
        return DriverResult(
            pipeline_yaml_path=work_dir / "pipeline.yaml", work_dir=work_dir, driver_name="tools"
        )


@pytest.mark.parametrize("discover_tools", [True, False])
def test_the_compiler_hands_the_agent_the_schemas_it_discovered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, discover_tools: bool
) -> None:
    from rote.compiler import tool_grounding

    calls: list[int] = []

    async def _discover(on_skip=None):  # noqa: ANN001, ANN202
        calls.append(1)
        return DECLARED

    monkeypatch.setattr(tool_grounding, "discover", _discover)
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("# s\n\nSearch Gmail (search_threads).\n")
    driver = _Driver()
    compiler = Compiler(
        compiler_skill_dir=tmp_path, trust_skill=True, discover_tools=discover_tools
    )
    compiler.select_driver = lambda: driver  # type: ignore[method-assign]
    asyncio.run(compiler.compile(skill, tmp_path / "out"))
    if discover_tools:
        assert calls == [1] and driver.saw_schemas
        assert "`query` (string, required)" in (driver.instructions or "")
    else:
        assert calls == [] and not driver.saw_schemas
        assert "TOOL CONTRACTS" not in (driver.instructions or "")
    assert not (tmp_path / "out" / GROUNDING_DIRNAME).exists()


def test_no_mcp_turns_discovery_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from rote.cli import main as cli_main
    from rote.compiler import CompilerError

    seen: dict[str, object] = {}

    class _Recorder:
        def __init__(self, *args: object, **kwargs: object) -> None:
            seen.update(kwargs)

        async def compile(self, *args: object, **kwargs: object) -> None:
            raise CompilerError("stop here")

    monkeypatch.setattr("rote.cli.Compiler", _Recorder)
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: t\n---\n")
    # --no-mcp needs an in-process driver (the subprocess ones own their MCP config).
    base = ["compile", str(skill), "--runtime", "dbos", "--local", "--no-eval", "--agent", "api"]
    base.append("--out")
    assert cli_main([*base, str(tmp_path / "a")]) == 1
    assert seen["discover_tools"] is True
    assert cli_main([*base, str(tmp_path / "b"), "--no-mcp"]) == 1
    assert seen["discover_tools"] is False


def test_a_declared_schema_beats_one_inferred_from_recorded_calls() -> None:
    from rote.compiler.tool_grounding import merge

    inferred = recorded((ObservedToolCall("gmail", "search_threads", {"q": "x"}, "[]"),))
    assert merge(DECLARED, inferred)[("gmail", "search_threads")].declared
    assert merge({}, inferred)[("gmail", "search_threads")].declared is False


def test_verification_runs_the_value_check_on_the_sample(tmp_path: Path) -> None:
    root = _root(tmp_path)
    grounding = Grounding(
        sample=Sample(payload={"report_date": 20261019}, source="sample.json"), tools=DECLARED
    )
    v = verify_deliverable(root, grounding=grounding, run_tests=True, trust_skill=True)
    bad = [f for f in v.errors if f.kind == "tool_arguments_invalid"]
    assert len(bad) == 1 and "must be string, got int" in bad[0].message


def test_a_bench_compile_runs_with_the_fixture_servers_registered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The compile subprocess sees a registry naming live fixture servers."""
    pytest.importorskip("fastmcp")
    import argparse

    from rote.bench import runner

    seen: dict[str, object] = {}

    class _Proc:
        async def wait(self) -> int:
            return 0

    async def _exec(*argv: str, env: dict[str, str], **kwargs: object) -> _Proc:
        registry = json.loads(Path(env["ROTE_MCP_CONFIG"]).read_text())
        seen["servers"] = sorted(registry["servers"])
        monkeypatch.setenv("ROTE_MCP_CONFIG", env["ROTE_MCP_CONFIG"])
        seen["found"] = sorted(await discover())
        return _Proc()

    monkeypatch.setattr(runner.asyncio, "create_subprocess_exec", _exec)
    monkeypatch.setattr(runner, "collect_job", lambda *a, **k: {"run": {}})
    from collections import defaultdict

    monkeypatch.setattr(runner, "headline", lambda result: defaultdict(lambda: None))
    args = argparse.Namespace(
        examples=REPO / "examples",
        out=tmp_path,
        runtime="dbos",
        model=None,
        agent=None,
        compile_flag=[],
        no_tsc=True,
        fidelity=False,
    )
    job = runner.Job("ops-report", 1, tmp_path / "ops-report" / "r1")
    asyncio.run(runner._run_job(job, args, {}, asyncio.Semaphore(1), runner._Budget(None)))
    assert seen["servers"] == ["gmail", "google_drive"]
    assert ("gmail", "search_threads") in seen["found"]  # type: ignore[operator]
