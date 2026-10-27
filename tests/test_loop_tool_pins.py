"""An agent loop's tools are pinned to the one grounded server that offers them (#262).

Covers :func:`rote.compiler.tool_grounding.loop_tool_pins`, the line edit
that writes the pins into pipeline.yaml, the open question for a tool two
servers offer, the ``loop_tool_unserved`` finding, and the compile wiring.
No inference: deliverables are written by hand and discovery is replaced.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from rote.compiler import Compiler
from rote.compiler.drivers import CompilerDriver, DriverResult
from rote.compiler.events import CompilationEvent
from rote.compiler.grounding import Grounding
from rote.compiler.tool_grounding import (
    LOOP_TOOL_UNSERVED,
    ToolSchema,
    adopt_offered,
    loop_tool_pins,
    tool_questions,
    unserved_loop_tool_findings,
    write_loop_pins,
)
from rote.compiler.verify import Verification, verify_deliverable
from rote.ir import Pipeline, load_pipeline

OPEN = {"type": "object", "properties": {"repoName": {"type": "string"}}}


def _schema(server: str, tool: str, origin: str = "live") -> ToolSchema:
    return ToolSchema(server, tool, OPEN, f"tools/list on {server}", True, origin=origin)  # type: ignore[arg-type]


WIKI = {("deepwiki", "ask_question"): _schema("deepwiki", "ask_question")}
BOTH = {**WIKI, ("mirror", "ask_question"): _schema("mirror", "ask_question")}

_IR = """\
name: research
input: {{type: Q}}
nodes:
  - id: research
    kind: agent_loop
    description: research the repo
    # rote: assumption: three turns are enough to answer
    tools: [{tools}]
{pins}    inputs: {{request: pipeline.input}}
    termination: {{condition: answered, max_iterations: 3}}
  - id: report
    kind: pure_function
    description: write it up
    impl: extracted/r.py:report
    inputs: {{found: research.output}}
edges:
  - {{from: research, to: report}}
entry_nodes: [research]
exit_nodes: [report]
"""


def _write(root: Path, tools: str = "ask_question", pins: str = "") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "pipeline.yaml").write_text(_IR.format(tools=tools, pins=pins), encoding="utf-8")
    (root / "extracted").mkdir(exist_ok=True)
    (root / "extracted" / "r.py").write_text("def report(found):\n    return {}\n")
    return root / "pipeline.yaml"


def _loop(path: Path) -> dict[str, str] | None:
    return {n.id: n for n in load_pipeline(path).nodes}["research"].tool_servers


def _skill(tmp_path: Path, body: str) -> Path:
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text(f"---\nname: r\ndescription: d\n---\n\n# R\n\n{body}\n")
    return skill


# ───────── which server a tool is pinned to ─────────


def test_a_tool_one_grounded_server_offers_is_pinned_to_it(tmp_path: Path) -> None:
    pipeline = load_pipeline(_write(tmp_path))
    assert loop_tool_pins(pipeline, WIKI) == {"research": {"ask_question": "deepwiki"}}
    # A snapshot or a recorded call is evidence of the server too.
    for origin in ("bundled", "recorded"):
        tools = {("deepwiki", "ask_question"): _schema("deepwiki", "ask_question", origin)}
        assert loop_tool_pins(pipeline, tools) == {"research": {"ask_question": "deepwiki"}}


def test_a_tool_two_servers_offer_stays_unpinned_and_is_an_open_question(tmp_path: Path) -> None:
    pipeline = load_pipeline(_write(tmp_path))
    assert loop_tool_pins(pipeline, BOTH) == {}
    [(node_id, text)] = tool_questions(pipeline, BOTH)
    assert node_id == "research"
    assert text.startswith(
        "its tool `ask_question` is offered by deepwiki and mirror, so rote left it unpinned"
    )
    assert tool_questions(pipeline, WIKI) == []


def test_an_authored_pin_always_wins(tmp_path: Path) -> None:
    path = _write(tmp_path, pins="    tool_servers: {ask_question: mirror}\n")
    assert loop_tool_pins(load_pipeline(path), WIKI) == {}
    assert loop_tool_pins(load_pipeline(path), BOTH) == {}
    # Nor is an authored pin a question.
    assert tool_questions(load_pipeline(path), BOTH) == []


# ───────── writing the pins ─────────


def test_the_pins_are_written_without_touching_the_agents_comments(tmp_path: Path) -> None:
    path = _write(tmp_path)
    before = path.read_text()
    pipeline = load_pipeline(path)
    assert write_loop_pins(path, pipeline, loop_tool_pins(pipeline, WIKI))
    after = path.read_text()
    assert _loop(path) == {"ask_question": "deepwiki"}
    assert "# rote: assumption: three turns are enough to answer" in after
    # One line added after the loop's id, every other line as it was.
    lines = before.splitlines()
    lines.insert(4, '    tool_servers: {"ask_question": "deepwiki"}')
    assert after.splitlines() == lines


def test_a_partial_authored_map_is_extended_and_keeps_its_entries(tmp_path: Path) -> None:
    pins = "    tool_servers:\n      read_wiki: deepwiki\n"
    path = _write(tmp_path, tools="ask_question, read_wiki", pins=pins)
    tools = {**BOTH, ("mirror", "read_wiki"): _schema("mirror", "read_wiki")}
    tools.pop(("deepwiki", "ask_question"))
    pipeline = load_pipeline(path)
    new = loop_tool_pins(pipeline, tools)
    assert new == {"research": {"ask_question": "mirror"}}
    assert write_loop_pins(path, pipeline, new)
    assert _loop(path) == {"ask_question": "mirror", "read_wiki": "deepwiki"}
    assert "# rote: assumption:" in path.read_text()


def test_a_layout_the_edit_cannot_place_leaves_the_file_as_it_was(tmp_path: Path) -> None:
    path = tmp_path / "pipeline.yaml"
    pipeline = load_pipeline(_write(tmp_path))
    # The same IR in flow style: no `id:` line to edit after.
    flow = json.dumps(pipeline.model_dump(mode="json", by_alias=True, exclude_none=True))
    path.write_text(flow)
    assert not write_loop_pins(path, pipeline, {"research": {"ask_question": "deepwiki"}})
    assert path.read_text() == flow


def test_an_edit_that_lands_in_the_wrong_place_is_undone(tmp_path: Path) -> None:
    """The loop in flow style, and another node's block text that reads like
    its id line: the edit goes into that text, so the result is checked
    against the IR it should be and put back when it is not."""
    text = """\
name: research
input: {type: Q}
nodes:
  - {id: research, kind: agent_loop, description: d, tools: [ask_question],
     termination: {condition: answered, max_iterations: 3}}
  - id: report
    kind: pure_function
    description: |
      Summarize.
      id: research
    impl: extracted/r.py:report
    inputs: {found: research.output}
edges:
  - {from: research, to: report}
"""
    path = tmp_path / "pipeline.yaml"
    path.write_text(text)
    pipeline = load_pipeline(path)
    assert not write_loop_pins(path, pipeline, {"research": {"ask_question": "deepwiki"}})
    assert path.read_text() == text


# ───────── a loop tool no grounded server offers ─────────


def test_a_loop_tool_no_listed_server_offers_is_an_error(tmp_path: Path) -> None:
    """invoice-push's loop tools are browser CLIs (`chrome_cdp_snapshot`)."""
    pipeline = load_pipeline(_write(tmp_path / "out", tools="chrome_cdp_snapshot"))
    skill = _skill(tmp_path, "Ask the `deepwiki` MCP server, then run chrome_cdp_snapshot.")
    [finding] = unserved_loop_tool_findings(pipeline, skill, WIKI, {})
    assert (finding.node_id, finding.kind, finding.severity) == (
        "research",
        LOOP_TOOL_UNSERVED,
        "error",
    )
    assert finding.message.startswith(
        "research lists `chrome_cdp_snapshot` in its tools, but no server the skill uses "
        "offers it (rote listed deepwiki)"
    )
    # A tool a grounded server offers is fine.
    served = load_pipeline(_write(tmp_path / "ok"))
    assert unserved_loop_tool_findings(served, skill, WIKI, {}) == []


@pytest.mark.parametrize(
    ("tools", "unlisted", "body"),
    [
        # Nothing was listed live: a bundled snapshot can be older than the server.
        (
            {("deepwiki", "ask_question"): _schema("deepwiki", "ask_question", "bundled")},
            {},
            "Ask the `deepwiki` MCP server.",
        ),
        ({}, {}, "Ask the `deepwiki` MCP server."),
        # A skill naming no server, grounded only in a snapshot.
        (
            {("deepwiki", "ask_question"): _schema("deepwiki", "ask_question", "bundled")},
            {},
            "Look it up.",
        ),
        # A server the skill names that rote did not list may offer it.
        (WIKI, {"browser": "tools/list failed"}, "Use the `browser` MCP server and deepwiki."),
        # Named but not registered at all: never listed.
        (WIKI, {}, "Use the `browser` MCP server and the `deepwiki` MCP server."),
    ],
)
def test_without_a_listing_of_every_server_the_skill_uses_nothing_is_said(
    tmp_path: Path, tools: dict, unlisted: dict[str, str], body: str
) -> None:
    pipeline = load_pipeline(_write(tmp_path / "out", tools="chrome_cdp_snapshot"))
    assert unserved_loop_tool_findings(pipeline, _skill(tmp_path, body), tools, unlisted) == []


def test_an_authored_pin_is_not_unserved(tmp_path: Path) -> None:
    """A pin is checked against its server's listing (tool_unknown), not here."""
    pins = "    tool_servers: {chrome_cdp_snapshot: browser}\n"
    pipeline = load_pipeline(_write(tmp_path / "out", tools="chrome_cdp_snapshot", pins=pins))
    skill = _skill(tmp_path, "Ask the `deepwiki` MCP server.")
    assert unserved_loop_tool_findings(pipeline, skill, WIKI, {}) == []


def test_verification_sends_an_unserved_loop_tool_to_repair(tmp_path: Path) -> None:
    root = _write(tmp_path / "out", tools="chrome_cdp_snapshot").parent
    skill = _skill(tmp_path, "Ask the `deepwiki` MCP server.")
    v = verify_deliverable(root, grounding=Grounding(tools=WIKI), run_tests=False, skill_dir=skill)
    assert LOOP_TOOL_UNSERVED in {f.kind for f in v.errors} and not v.ok
    assert "chrome_cdp_snapshot" in (v.repair_instruction(root) or "")


def test_verification_lists_a_tool_two_servers_offer_as_a_question(tmp_path: Path) -> None:
    root = _write(tmp_path / "out").parent
    skill = _skill(tmp_path, "Ask the `deepwiki` MCP server or its `mirror` MCP server.")
    v = verify_deliverable(root, grounding=Grounding(tools=BOTH), run_tests=False, skill_dir=skill)
    assert [q.node_id for q in v.questions if "left it unpinned" in q.text] == ["research"]
    assert LOOP_TOOL_UNSERVED not in {f.kind for f in v.findings}


# ───────── an offered server whose tool a loop lists ─────────


def test_an_offered_server_whose_tool_a_loop_lists_becomes_grounding() -> None:
    other = _schema("figma", "get_screenshot")
    candidates = {("deepwiki", "ask_question"): WIKI[("deepwiki", "ask_question")]}
    candidates[("figma", "get_screenshot")] = other
    adopted, unused = adopt_offered(candidates, [], ["ask_question"])
    assert adopted == {("deepwiki", "ask_question"): WIKI[("deepwiki", "ask_question")]}
    assert unused == ["figma"]


# ───────── the compile ─────────


class _LoopDriver(CompilerDriver):
    name = "loop"
    #: Write pipeline.yaml as one JSON line, a layout the line edit refuses.
    flow = False

    def is_available(self) -> tuple[bool, str]:
        return (True, "")

    async def run(  # type: ignore[override]
        self, skill_dir, compiler_skill_dir, work_dir, extra_instructions=None, on_event=None
    ) -> DriverResult:
        path = _write(work_dir)
        if self.flow:
            dumped = load_pipeline(path).model_dump(mode="json", by_alias=True, exclude_none=True)
            path.write_text(json.dumps(dumped))
        return DriverResult(
            pipeline_yaml_path=work_dir / "pipeline.yaml", work_dir=work_dir, driver_name="loop"
        )


def _compile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, listed: dict, *, flow: bool = False
) -> tuple[Pipeline, Path, Verification, list[CompilationEvent]]:
    from rote.compiler import tool_grounding

    async def _discover(servers, on_skip=None, timeout=0):  # noqa: ANN001, ANN202
        return {k: v for k, v in listed.items() if k[0] in servers}

    registry = tmp_path / "mcp.json"
    servers = {s: {"url": "http://127.0.0.1:9/mcp"} for s in ("deepwiki", "mirror")}
    registry.write_text(json.dumps({"version": 1, "servers": servers}))
    monkeypatch.setenv("ROTE_MCP_CONFIG", str(registry))
    monkeypatch.setattr(tool_grounding, "discover", _discover)
    skill = _skill(tmp_path, "Ask the `deepwiki` MCP server or its `mirror` MCP server.")
    events: list[CompilationEvent] = []
    compiler = Compiler(
        system1=False, compiler_skill_dir=tmp_path, trust_skill=True, on_event=events.append
    )
    driver = _LoopDriver()
    driver.flow = flow
    compiler.select_driver = lambda: driver  # type: ignore[method-assign]
    result = asyncio.run(compiler.compile(skill, tmp_path / "out"))
    assert result.verification is not None
    return result.pipeline, tmp_path / "out" / "pipeline.yaml", result.verification, events


def test_the_compile_pins_a_loop_tool_in_the_shipped_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline, path, _, events = _compile(tmp_path, monkeypatch, WIKI)
    loop = {n.id: n for n in pipeline.nodes}["research"]
    assert loop.tool_servers == {"ask_question": "deepwiki"}
    assert _loop(path) == {"ask_question": "deepwiki"}
    assert "# rote: assumption: three turns are enough to answer" in path.read_text()
    assert "deepwiki" in pipeline.required_mcp_servers
    assert "tool schemas: pinned research's tools: ask_question -> deepwiki" in [
        e.message for e in events
    ]
    assert not [e for e in events if e.type == "warning" and "rewritten whole" in e.message]


def test_a_layout_the_line_edit_refuses_is_rewritten_whole_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline, path, _, events = _compile(tmp_path, monkeypatch, WIKI, flow=True)
    assert _loop(path) == {"ask_question": "deepwiki"}
    assert {n.id: n for n in pipeline.nodes}["research"].tool_servers == {
        "ask_question": "deepwiki"
    }
    assert [e.message for e in events if e.type == "warning" and "rewritten whole" in e.message]


def test_the_compile_leaves_a_tool_two_servers_offer_for_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, path, verification, _ = _compile(tmp_path, monkeypatch, BOTH)
    assert _loop(path) is None
    assert any("left it unpinned" in q.text for q in verification.questions)
    report = (path.parent / "compile-report.md").read_text()
    assert "offered by deepwiki and mirror" in report
