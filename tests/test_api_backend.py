"""``--backend api``: the agent is told, gets web tools, and cites the docs it used.

Covers the compile side of the api backend end to end without inference:
the Compiler grants web tools and adds the brief only for ``api``, the
drivers' web tool events are recorded, verification holds every
external_call to a cited, tested implementation, and the compile report
lists each cited page beside whether the agent fetched it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rote.compiler import Compiler, api_docs
from rote.compiler.drivers import DRIVERS, CompilerDriver, DriverResult
from rote.compiler.events import CompilationEvent, emit_safely
from rote.compiler.verify import EmitTarget, verify_deliverable
from rote.ir import load_pipeline

DOCS = "https://developers.google.com/workspace/gmail/api/reference/rest/v1/users.threads/list"

PIPELINE_YAML = """\
name: dock-emails
version: "0.1.0"
description: Count dock appointment emails.
input:
  type: DockInput
nodes:
  - id: search_dock_emails
    kind: external_call
    description: Search Gmail for dock activity threads.
    impl: extracted/gmail.py:search_dock_emails
    mcp:
      server: gmail
      tool: search_threads
    inputs:
      query: pipeline.input.query
edges: []
entry_nodes: [search_dock_emails]
exit_nodes: [search_dock_emails]
"""

CITED_MODULE = f'''"""Gmail threads search. Dependencies: httpx."""

import os

import httpx


# docs: {DOCS}#query-parameters
def search_dock_emails(query: str) -> dict:
    token = os.environ.get("GMAIL_ACCESS_TOKEN")
    if not token:
        raise RuntimeError("GMAIL_ACCESS_TOKEN is not set")
    response = httpx.get(
        "https://gmail.googleapis.com/gmail/v1/users/me/threads",
        params={{"q": query}},
        headers={{"Authorization": f"Bearer {{token}}"}},
    )
    response.raise_for_status()
    return {{"threads": response.json().get("threads", [])}}
'''

UNCITED_MODULE = CITED_MODULE.replace(f"# docs: {DOCS}#query-parameters\n", "")

TEST_MODULE = """from extracted.gmail import search_dock_emails


def test_search_dock_emails_is_named() -> None:
    assert callable(search_dock_emails)
"""


def _write(root: Path, files: dict[str, str]) -> None:
    for rel, content in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(content, encoding="utf-8")


def _deliverable(root: Path, module: str, tests: bool = True) -> Path:
    files = {"pipeline.yaml": PIPELINE_YAML, "extracted/gmail.py": module}
    if tests:
        files["tests/test_gmail.py"] = TEST_MODULE
    _write(root, files)
    return root


def _kinds(verification: Any) -> set[str]:
    return {f.kind for f in verification.errors}


# ───────── The Compiler wires the backend ─────────


@pytest.mark.parametrize(
    ("target", "web_tools"),
    [
        (EmitTarget("dbos", "api"), True),
        (EmitTarget("dbos", "mcp"), None),
        (None, None),
    ],
)
def test_only_the_api_backend_grants_the_driver_web_tools(
    monkeypatch: pytest.MonkeyPatch, target: EmitTarget | None, web_tools: bool | None
) -> None:
    captured: dict[str, Any] = {}

    class _Probe(CompilerDriver):
        name = "probe"

        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        def is_available(self) -> tuple[bool, str]:
            return (True, "")

        async def run(self, *args: Any, **kwargs: Any) -> DriverResult:
            raise AssertionError("not run")

    monkeypatch.setitem(DRIVERS, "probe", _Probe)
    Compiler(agent="probe", emit_target=target).select_driver()
    assert captured.get("web_tools") is web_tools


class _ResearchingDriver(CompilerDriver):
    """Writes a canned deliverable after reporting the web lookups a real run would."""

    name = "researching"

    def __init__(self, module: str, fetched: list[str]) -> None:
        self.module = module
        self.fetched = fetched
        self.instructions: str | None = None
        self.repairs: list[str] = []

    def is_available(self) -> tuple[bool, str]:
        return (True, "")

    async def run(
        self,
        skill_dir: Path,
        compiler_skill_dir: Path,
        work_dir: Path,
        extra_instructions: str | None = None,
        on_event: Any = None,
        repair: Any = None,
    ) -> DriverResult:
        self.instructions = extra_instructions
        emit_safely(
            on_event,
            CompilationEvent(
                type="tool", ts=0.0, tool_name="WebSearch", query="gmail api threads list"
            ),
        )
        for url in self.fetched:
            emit_safely(
                on_event, CompilationEvent(type="tool", ts=0.0, tool_name="WebFetch", url=url)
            )
        _deliverable(work_dir, self.module)
        while repair is not None:
            instruction = repair(work_dir / "pipeline.yaml")
            if instruction is None:
                break
            self.repairs.append(instruction)
        return DriverResult(
            pipeline_yaml_path=work_dir / "pipeline.yaml",
            work_dir=work_dir,
            driver_name=self.name,
            metadata={},
        )


@pytest.fixture
def skill_dirs(tmp_path: Path) -> tuple[Path, Path]:
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: dock\n---\n\n# Dock\n", encoding="utf-8")
    rubric = tmp_path / "rubric"
    rubric.mkdir()
    (rubric / "SKILL.md").write_text("# rubric", encoding="utf-8")
    return skill, rubric


async def _compile(
    driver: _ResearchingDriver, skill_dirs: tuple[Path, Path], out: Path, backend: str
) -> str:
    skill, rubric = skill_dirs
    compiler = Compiler(
        compiler_skill_dir=rubric,
        emit_target=EmitTarget("dbos", backend),
        run_generated_tests=False,
        tool_discovery="none",
    )
    compiler.select_driver = lambda: driver  # type: ignore[method-assign]
    await compiler.compile(skill, out)
    return (out / "compile-report.md").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_an_api_compile_briefs_the_agent_and_reports_what_it_fetched(
    tmp_path: Path, skill_dirs: tuple[Path, Path]
) -> None:
    other = "https://developers.google.com/workspace/gmail/api/guides/filtering"
    driver = _ResearchingDriver(CITED_MODULE, [DOCS + "/", other])
    report = await _compile(driver, skill_dirs, tmp_path / "out", "api")

    assert driver.instructions is not None
    assert api_docs.brief() in driver.instructions
    assert driver.repairs == []
    section = report.split("## Vendor API docs", 1)[1].split("## Open questions", 1)[0]
    # The cited URL (with an anchor) matches the fetched one (trailing slash).
    assert f"| `search_dock_emails` | {DOCS}#query-parameters | `extracted/gmail.py:" in section
    assert "| yes |" in section
    assert "1 web search(es) and fetched 2 page(s)" in section
    assert f"- {other}" in section  # fetched, never cited


@pytest.mark.asyncio
async def test_a_cited_page_the_agent_never_fetched_is_flagged(
    tmp_path: Path, skill_dirs: tuple[Path, Path]
) -> None:
    driver = _ResearchingDriver(CITED_MODULE, ["https://example.com/some-other-page"])
    skill, rubric = skill_dirs
    compiler = Compiler(
        compiler_skill_dir=rubric,
        emit_target=EmitTarget("dbos", "api"),
        run_generated_tests=False,
        tool_discovery="none",
    )
    compiler.select_driver = lambda: driver  # type: ignore[method-assign]
    result = await compiler.compile(skill, tmp_path / "out")
    report = (tmp_path / "out" / "compile-report.md").read_text(encoding="utf-8")
    assert "**no**: check it first" in report
    assert "| yes |" not in report
    # Not only a report column: the agent is sent back to fetch it, and the
    # final verdict fails while the page stays unfetched.
    assert driver.repairs and "which this compile never fetched" in driver.repairs[0]
    assert result.verification is not None
    assert "unfetched_docs_page" in _kinds(result.verification)


@pytest.mark.asyncio
async def test_an_mcp_compile_gets_no_brief_and_no_docs_section(
    tmp_path: Path, skill_dirs: tuple[Path, Path]
) -> None:
    driver = _ResearchingDriver(UNCITED_MODULE, [])
    report = await _compile(driver, skill_dirs, tmp_path / "out", "mcp")
    assert driver.instructions is None or api_docs.brief() not in driver.instructions
    assert "## Vendor API docs" not in report
    assert driver.repairs == []


@pytest.mark.asyncio
async def test_an_uncited_vendor_call_is_sent_back_on_the_api_backend(
    tmp_path: Path, skill_dirs: tuple[Path, Path]
) -> None:
    driver = _ResearchingDriver(UNCITED_MODULE, [DOCS])
    await _compile(driver, skill_dirs, tmp_path / "out", "api")
    assert driver.repairs, "the uncited call was accepted"
    assert (
        "search_dock_emails() calls a vendor API but its module cites no docs page"
        in (driver.repairs[0])
    )


# ───────── Verification ─────────


@pytest.mark.parametrize(
    ("module", "tests", "backend", "expected"),
    [
        (CITED_MODULE, True, "api", set()),
        (UNCITED_MODULE, True, "api", {"uncited_api_call"}),
        (CITED_MODULE, False, "api", {"untested_implementation"}),
        (
            "def search_dock_emails(query):\n    raise NotImplementedError\n",
            True,
            "api",
            {"unexplained_stub"},
        ),
        # The mcp backend never runs this module: none of it is held against it.
        (UNCITED_MODULE, False, "mcp", set()),
        ("def search_dock_emails(query):\n    raise NotImplementedError\n", True, "mcp", set()),
    ],
)
def test_api_backend_holds_every_external_call_to_a_cited_tested_module(
    tmp_path: Path, module: str, tests: bool, backend: str, expected: set[str]
) -> None:
    root = _deliverable(tmp_path / "d", module, tests=tests)
    verification = verify_deliverable(
        root, run_tests=False, emit_target=EmitTarget("dbos", backend)
    )
    assert _kinds(verification) == expected


def test_an_unwritten_vendor_module_is_an_error_on_the_api_backend(tmp_path: Path) -> None:
    root = tmp_path / "d"
    _write(root, {"pipeline.yaml": PIPELINE_YAML})
    api = verify_deliverable(root, run_tests=False, emit_target=EmitTarget("dbos", "api"))
    mcp = verify_deliverable(root, run_tests=False, emit_target=EmitTarget("dbos", "mcp"))
    assert "external_call_not_written" in _kinds(api)
    assert "external_call_not_written" not in _kinds(mcp)


def test_an_api_backend_stub_of_a_bound_node_is_an_open_question(tmp_path: Path) -> None:
    from rote.compiler.report import ReportExtras, render_compile_report

    stub = (
        "def search_dock_emails(query):\n"
        "    raise NotImplementedError('the threads.list page did not confirm paging')\n"
    )
    root = _deliverable(tmp_path / "d", stub)
    pipeline = load_pipeline(root / "pipeline.yaml")
    api = render_compile_report(pipeline, root, ReportExtras(api_backend=True))
    mcp = render_compile_report(pipeline, root, ReportExtras())
    assert "### Stub: search_dock_emails" in api
    assert "### Stub: search_dock_emails" not in mcp


# ───────── Parsing ─────────


def test_docs_markers_and_url_normalization() -> None:
    source = (
        "# docs: https://a.example/x/  \nx = 1  # docs: http://A.example/y).\n# docs: not-a-url\n"
    )
    assert api_docs.module_docs(source) == [(1, "https://a.example/x/"), (2, "http://A.example/y")]
    assert api_docs.normalize_url("http://A.example/x/#frag") == "https://a.example/x"
    assert api_docs.normalize_url("https://a.example/x?b=1") == "https://a.example/x?b=1"
    assert api_docs.normalize_url("https://a.example/x?b=1") != api_docs.normalize_url(
        "https://a.example/x?b=2"
    )


def test_web_research_records_only_web_tool_events() -> None:
    research = api_docs.WebResearch()
    seen: list[CompilationEvent] = []
    tee = research.tee(seen.append)
    for event in (
        CompilationEvent(type="tool", ts=0, tool_name="Read", path="SKILL.md"),
        CompilationEvent(type="tool", ts=0, tool_name="web_fetch", url="https://a.example"),
        CompilationEvent(type="tool", ts=0, tool_name="WebFetch", url="https://a.example"),
        CompilationEvent(type="tool", ts=0, tool_name="web_search", query="q"),
        CompilationEvent(type="turn", ts=0, url="https://ignored.example"),
    ):
        tee(event)
    assert research.fetched == ["https://a.example"]
    assert research.searches == ["q"]
    assert len(seen) == 5  # everything is still forwarded


# ───────── The CLI ─────────


@pytest.mark.parametrize("runtime", ["cloudflare", "inngest", "dbos-ts"])
def test_compile_refuses_api_backend_on_a_typescript_runtime_before_spending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any, runtime: str
) -> None:
    from rote.cli import main

    def _never(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the compiler was constructed")

    monkeypatch.setattr("rote.cli.Compiler", _never)
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: t\n---\n", encoding="utf-8")
    rc = main(
        ["compile", str(skill), "--local", "--runtime", runtime, "--backend", "api"]
        + ["--no-eval", "--out", str(tmp_path / "out")]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert f"the {runtime} runtime (flag) emits TypeScript" in err


def test_compile_hands_the_backend_to_the_compiler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    """The CLI's only job is the emit target; the Compiler derives the rest."""
    from tests.test_cli import _install_mock_compiler

    captured: list[dict[str, Any]] = []
    _install_mock_compiler(monkeypatch)
    original = __import__("rote.cli", fromlist=["Compiler"]).Compiler

    class _Capturing(original):  # type: ignore[misc, valid-type]
        def __init__(self, **kwargs: Any) -> None:
            captured.append(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setattr("rote.cli.Compiler", _Capturing)
    from rote.cli import main

    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: t\n---\n", encoding="utf-8")
    for backend, out in (("api", "o1"), ("mcp", "o2")):
        rc = main(
            ["compile", str(skill), "--runtime", "dbos", "--backend", backend, "--no-eval"]
            + ["--out", str(tmp_path / out)]
        )
        assert rc == 0
    capsys.readouterr()
    assert [c["emit_target"].external_backend for c in captured] == ["api", "mcp"]
    assert all(c["driver_kwargs"] is None for c in captured)


def test_a_web_tool_event_carries_its_page_through_the_jsonl_sink(tmp_path: Path) -> None:
    from rote.cli import _JsonlProgressSink

    path = tmp_path / "progress.jsonl"
    sink = _JsonlProgressSink(path, model_id="claude-sonnet-4-6")
    sink.emit(CompilationEvent(type="tool", ts=1.0, tool_name="WebFetch", url=DOCS))
    sink.close()
    [line] = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]
    assert line["url"] == DOCS


def test_a_citation_belongs_to_the_function_it_sits_on(tmp_path: Path) -> None:
    """A page cited on one function does not cite another in the same module,
    and a page in the module header speaks for every function, once."""
    two = (
        PIPELINE_YAML.replace(
            "exit_nodes: [search_dock_emails]",
            "exit_nodes: [search_dock_emails, other_call]",
        )
        .replace(
            "edges: []",
            "  - id: other_call\n"
            "    kind: external_call\n"
            "    description: Another Gmail call.\n"
            "    impl: extracted/gmail.py:other_call\n"
            "edges: []",
        )
        .replace(
            "entry_nodes: [search_dock_emails]", "entry_nodes: [search_dock_emails, other_call]"
        )
    )
    module = (
        '"""Gmail.\n\n# docs: https://header.example/guide\n"""\n\n'
        "# docs: https://header.example/guide#again\n"
        f"# docs: {DOCS}\n"
        "def search_dock_emails(query):\n"
        "    return {}\n\n\n"
        "def other_call():\n"
        "    return {}\n"
    )
    root = tmp_path / "d"
    _write(root, {"pipeline.yaml": two, "extracted/gmail.py": module})
    cites = api_docs.cited_docs(load_pipeline(root / "pipeline.yaml"), root)
    by_node: dict[str, list[str]] = {}
    for c in cites:
        by_node.setdefault(c.node_id, []).append(c.url)
    assert by_node == {
        "search_dock_emails": ["https://header.example/guide", DOCS],
        "other_call": ["https://header.example/guide"],
    }


def test_the_api_backend_asks_nothing_about_an_uncalled_binding(tmp_path: Path) -> None:
    """No tool-argument or tool-result question, and no MCP server to log in
    to, for a binding the api backend never calls."""
    from rote.cli import _mcp_requirements
    from rote.compiler.grounding import Grounding

    root = _deliverable(tmp_path / "d", CITED_MODULE)
    grounding = Grounding(sample=None, observations=(), tools={})
    mcp = verify_deliverable(
        root, run_tests=False, emit_target=EmitTarget("dbos", "mcp"), grounding=grounding
    )
    api = verify_deliverable(
        root, run_tests=False, emit_target=EmitTarget("dbos", "api"), grounding=grounding
    )
    assert any("gmail.search_threads" in q.text for q in mcp.questions)
    assert not any("gmail.search_threads" in q.text for q in api.questions)

    pipeline = load_pipeline(root / "pipeline.yaml")
    assert [e["server"] for e in _mcp_requirements(pipeline)] == ["gmail"]
    assert _mcp_requirements(pipeline, "api") == []


@pytest.mark.asyncio
@pytest.mark.parametrize("runtime", ["cloudflare", "inngest", "dbos-ts"])
async def test_the_compiler_itself_refuses_api_on_a_typescript_runtime(
    tmp_path: Path, skill_dirs: tuple[Path, Path], runtime: str
) -> None:
    """The container and the bench build a Compiler directly, past the CLI's check."""
    from rote.compiler import CompilerError

    skill, rubric = skill_dirs
    compiler = Compiler(compiler_skill_dir=rubric, emit_target=EmitTarget(runtime, "api"))

    def _never() -> CompilerDriver:
        raise AssertionError("a driver was selected")

    compiler.select_driver = _never  # type: ignore[method-assign]
    with pytest.raises(CompilerError, match=f"the {runtime} runtime emits TypeScript"):
        await compiler.compile(skill, tmp_path / "out")
    # The mcp backend on the same runtime is not refused.
    Compiler(emit_target=EmitTarget(runtime, "mcp"))._refuse_api_backend_on_typescript()


@pytest.mark.parametrize(
    ("module", "fetched", "expected"),
    [
        (CITED_MODULE, [DOCS], set()),
        (CITED_MODULE, ["https://elsewhere.example"], {"unfetched_docs_page"}),
        # A driver that reports no fetches cannot say what was not fetched.
        (CITED_MODULE, None, set()),
        # The skill as the reference: cited, and exempt from the fetch check.
        (
            UNCITED_MODULE.replace(
                "def search_dock_emails", "# docs: skill\ndef search_dock_emails"
            ),
            ["https://elsewhere.example"],
            set(),
        ),
    ],
)
def test_a_cited_page_must_have_been_fetched_when_fetches_are_reported(
    tmp_path: Path, module: str, fetched: list[str] | None, expected: set[str]
) -> None:
    root = _deliverable(tmp_path / "d", module)
    research = None if fetched is None else api_docs.WebResearch(fetched=fetched)
    verification = verify_deliverable(
        root, run_tests=False, emit_target=EmitTarget("dbos", "api"), web_research=research
    )
    assert _kinds(verification) == expected


def test_report_cells_escape_a_pipe_and_name_the_skill_reference(tmp_path: Path) -> None:
    piped = "https://api.example/v1?fields=a|b"
    module = UNCITED_MODULE.replace(
        "def search_dock_emails", f"# docs: {piped}\n# docs: skill\ndef search_dock_emails"
    )
    root = _deliverable(tmp_path / "d", module)
    lines = api_docs.report_section(
        load_pipeline(root / "pipeline.yaml"), root, api_docs.WebResearch(fetched=[piped])
    )
    rows = [line for line in lines if line.startswith("| `search_dock_emails`")]
    assert rows[0].startswith("| `search_dock_emails` | https://api.example/v1?fields=a\\|b |")
    assert rows[0].count(" | ") == 3
    assert "the skill's own text (internal API) |" in rows[1] and rows[1].endswith("| n/a |")
