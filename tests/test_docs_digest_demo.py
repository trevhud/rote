"""The bundled first-run demo: docs-digest and its recorded reference.

No network and no inference: everything here reads the committed files.
The strongest check is the replay: the MCP calls the recorded pipeline run
made become the recorded-call stubs of rote's own prefix plan and runner
(:func:`rote.compiler.prefix.prefix_plan`, :mod:`rote.compiler._prefix_runner`,
the interpreter verification uses, which mirrors the emitted step bodies),
and the compiled code must
return the recorded digest. So the reference and the committed code cannot
disagree, and the payload rules checked are the ones that run.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from rote import demo
from rote.adapters import get_adapter
from rote.compiler.grounding import Sample
from rote.compiler.prefix import RUNNER_PATH, prefix_plan
from rote.contracts import implementation_status
from rote.eval.baseline import ObservedToolCall
from rote.ir import Pipeline, load_pipeline

REPO = Path(__file__).resolve().parent.parent
DEMO = REPO / "examples" / "docs-digest"


def _pipeline() -> Pipeline:
    return load_pipeline(demo.pipeline_path())


def test_the_demo_resolves_to_the_repo_copy_in_a_checkout() -> None:
    assert demo.docs_digest_dir() == DEMO
    assert demo.skill_dir() == DEMO / "skill"
    assert (demo.skill_dir() / "SKILL.md").is_file()
    assert demo.pipeline_path() == DEMO / "compiled" / "pipeline.yaml"


def test_the_reference_matches_the_skill_and_the_pipeline() -> None:
    ref = demo.load_reference()
    assert ref.raw["format"] == demo.REFERENCE_FORMAT == "rote-demo-reference/1"
    assert ref.raw["skill"] == demo.DEMO_NAME
    # Both sides ran on the input compile grounds itself in.
    skill_input = json.loads((DEMO / "skill" / "input.json").read_text(encoding="utf-8"))
    assert ref.input == skill_input
    # The servers to register are exactly the pipeline's MCP bindings, and
    # each recorded call is one of them.
    pipeline = _pipeline()
    bindings = {n.mcp.server: n.mcp.tool for n in pipeline.nodes if n.mcp is not None}
    assert {s.name: s.tool for s in ref.servers} == bindings
    assert {s.name: s.url for s in ref.servers} == {
        "mslearn": "https://learn.microsoft.com/api/mcp",
        "deepwiki": "https://mcp.deepwiki.com/mcp",
    }
    assert {(c.node, c.server, c.tool) for c in ref.tool_calls} == {
        (n.id, n.mcp.server, n.mcp.tool) for n in pipeline.nodes if n.mcp is not None
    }
    # The pipeline returns its exit node's result under the node id.
    assert list(ref.pipeline_output) == pipeline.exit_nodes
    for key in ("rote_version", "claude_cli_version", "recorded_on"):
        assert ref.raw[key], key
    assert ref.raw["pipeline_run"]["model_calls"] == 0


def test_the_compiled_constant_is_a_number_the_logic_reads() -> None:
    """The first compile shipped `result_limit: "first five"` beside the
    module's own `RESULT_LIMIT = 5` (dead config). This one passes 5."""
    from rote.compiler.verify import verify_deliverable

    [pick] = [n for n in _pipeline().nodes if n.id == "pick_reading_list"]
    assert pick.constants == {"result_count": 5}
    recorded = demo.load_reference().raw["compile"]["constants"]
    assert recorded == {"pick_reading_list": {"result_count": 5}}
    v = verify_deliverable(DEMO / "compiled", run_tests=False)
    assert v.review_gaps == []


def test_every_field_resolves_and_the_equal_ones_agree() -> None:
    ref = demo.load_reference()
    assert [(f.name, f.expect) for f in ref.fields] == [
        ("sections", "equal"),
        ("reading_list_size", "equal"),
        ("reading_list", "drifts"),
        ("repo_outline", "drifts"),
    ]
    values = {
        f.name: (
            demo.value_at(ref.pipeline_output, f.pipeline, f.section, f.measure),
            demo.value_at(ref.skill_output, f.skill, f.section, f.measure),
        )
        for f in ref.fields
    }
    assert values["sections"] == (["Reading list", "Repo outline"],) * 2
    assert values["reading_list_size"] == (5, 5)
    for field in ref.fields:
        pipeline_value, skill_value = values[field.name]
        assert pipeline_value and skill_value
        if field.expect == "equal":
            assert pipeline_value == skill_value, field.name
    # The recorded runs agreed in full: the skill's digest and the
    # pipeline's are the same text but for the pipeline's trailing newline.
    skill_digest = demo.value_at(ref.skill_output, "/digest")
    pipeline_digest = demo.value_at(ref.pipeline_output, "/write_digest/digest")
    assert pipeline_digest == skill_digest + "\n"


def _replay(compiled: Path, ref: demo.Reference) -> dict[str, dict[str, Any]]:
    """The prefix runner's report for the compiled code on the recording.

    The plan comes from :func:`rote.compiler.prefix.prefix_plan` and runs in
    :mod:`rote.compiler._prefix_runner`, exactly as verification runs it,
    minus the sandbox (this is the repo's own committed code, and
    :func:`run_prefix` drops each step's output from its verdict)."""
    pipeline = load_pipeline(compiled / "pipeline.yaml")
    observations = tuple(
        ObservedToolCall(c.server, c.tool, c.arguments, json.dumps(c.result))
        for c in ref.tool_calls
    )
    plan = prefix_plan(
        pipeline,
        compiled,
        Sample(payload=ref.input, source="reference.json"),
        implementation_status(pipeline, compiled),
        observations,
    )
    plan_path, report_path = compiled.parent / "plan.json", compiled.parent / "report.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    done = subprocess.run(
        [sys.executable, "-S", str(RUNNER_PATH), str(plan_path), str(report_path)],
        capture_output=True,
        text=True,
        timeout=60,
        env={"PYTHONDONTWRITEBYTECODE": "1", "PATH": os.environ.get("PATH", "")},
    )
    assert done.returncode == 0, done.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert isinstance(report, dict)
    return report


def test_the_recorded_calls_replay_through_the_prefix_runner_to_the_recorded_digest(
    tmp_path: Path,
) -> None:
    ref = demo.load_reference()
    compiled = tmp_path / "compiled"
    shutil.copytree(DEMO / "compiled", compiled)  # the runner may write beside the code
    steps = _replay(compiled, ref)
    # Each MCP step was served its recording, matched on the exact arguments
    # the pipeline builds, so every node ran.
    assert {node: step["status"] for node, step in steps.items()} == {
        n.id: "ok" for n in _pipeline().nodes
    }
    assert steps["write_digest"]["output"] == ref.pipeline_output["write_digest"]


def test_editing_the_constant_changes_the_digest(tmp_path: Path) -> None:
    """The IR's constant is live config: 3 in pipeline.yaml keeps three
    results. (The first compile ignored it, and this would have kept five.)"""
    ref = demo.load_reference()
    compiled = tmp_path / "compiled"
    shutil.copytree(DEMO / "compiled", compiled)
    ir = compiled / "pipeline.yaml"
    text = ir.read_text(encoding="utf-8")
    assert text.count("result_count: 5") == 1
    ir.write_text(text.replace("result_count: 5", "result_count: 3"), encoding="utf-8")
    digest = _replay(compiled, ref)["write_digest"]["output"]["digest"]
    assert demo.value_at(digest, "", "Reading list", "bullets") == 3


def test_the_replay_fails_when_the_arguments_differ(tmp_path: Path) -> None:
    """Negative half: a recording is used only for the exact call made, so a
    replay cannot pass with arguments the pipeline would not send."""
    ref = demo.load_reference()
    compiled = tmp_path / "compiled"
    shutil.copytree(DEMO / "compiled", compiled)
    other = demo.Reference(
        **{
            **ref.__dict__,
            "tool_calls": tuple(
                demo.ToolCall(c.node, c.server, c.tool, {"query": "something else"}, c.result)
                if c.tool == "microsoft_docs_search"
                else c
                for c in ref.tool_calls
            ),
        }
    )
    steps = _replay(compiled, other)
    assert steps["search_microsoft_learn"]["status"] == "no_recording"
    assert steps["write_digest"]["status"] == "not_reached"


def test_dbos_runs_the_tool_steps_over_mcp_and_python_would_hit_stubs(tmp_path: Path) -> None:
    """dbos is the recorded runtime: Python, SQLite, no Node or Docker. The
    python runtime has no MCP backend, so it calls the two tool steps'
    direct-API modules, which are stubs (rote does not write vendor calls
    from memory), and `rote run` refuses it up front (README, "Run it")."""
    from rote.adapters._common import unimplemented_stubs

    pipeline = _pipeline()
    get_adapter("dbos", extracted_source_dir=DEMO / "compiled").emit(pipeline, tmp_path / "dbos")
    assert unimplemented_stubs(tmp_path / "dbos") == []
    main_py = (tmp_path / "dbos" / "main.py").read_text(encoding="utf-8")
    assert "Source skill: ../skill" in main_py

    get_adapter("python", extracted_source_dir=DEMO / "compiled").emit(
        pipeline, tmp_path / "python"
    )
    stubs = {p.stem for p in unimplemented_stubs(tmp_path / "python")}
    assert stubs == {n.id for n in pipeline.nodes if n.mcp is not None}


def test_nothing_in_the_demo_names_a_machine() -> None:
    """A recorded run must not carry the recorder's paths or account."""
    for path in sorted(DEMO.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        for marker in ("/Users/", "/home/", "/private/", "/tmp/claude", "C:\\", "@gmail.com"):
            assert marker not in text, f"{path.relative_to(REPO)} contains {marker!r}"


# ───────── the reference reader ─────────


def test_value_at_reads_pointers_sections_and_measures() -> None:
    doc = {"a": {"b/c": ["x", "## One\n\n- a\n- b\n  - nested\n\n## Two\nsecond\n"]}}
    text = "/a/b~1c/1"
    assert demo.value_at(doc, "/a/b~1c/0") == "x"
    assert demo.value_at(doc, text, "One") == "- a\n- b\n  - nested"
    assert demo.value_at(doc, text, "Two") == "second"
    assert demo.value_at(doc, text, measure="headings") == ["One", "Two"]
    assert demo.value_at(doc, text, "One", "bullets") == 2
    assert demo.value_at(doc, "") == doc
    with pytest.raises(demo.DemoError, match="missing 'z'"):
        demo.value_at(doc, "/a/z")
    with pytest.raises(demo.DemoError, match="not a JSON pointer"):
        demo.value_at(doc, "a")
    with pytest.raises(demo.DemoError, match="no '## Three' heading"):
        demo.value_at(doc, text, "Three")
    with pytest.raises(demo.DemoError, match="not markdown text"):
        demo.value_at(doc, "/a", "One")


def _raw() -> dict[str, Any]:
    raw = json.loads(demo.reference_path().read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return raw


def _load(tmp_path: Path, raw: dict[str, Any]) -> demo.Reference:
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return demo.load_reference(path)


def _equal_fields(raw: dict[str, Any]) -> list[dict[str, Any]]:
    return [f for f in raw["fields"] if f["expect"] == "equal"]


def test_the_committed_reference_loads_through_the_same_path(tmp_path: Path) -> None:
    """Positive half for every refusal below: the unmodified copy loads."""
    assert _load(tmp_path, _raw()).fields == demo.load_reference().fields


_BREAKS: dict[str, tuple[Any, str]] = {
    "format": (lambda r: r.update(format="rote-demo-reference/0"), "expected format"),
    "servers as a list": (
        lambda r: r.update(servers=[{"url": "https://x.example/mcp", "tool": "t"}]),
        "servers must be an object, found list",
    ),
    "no servers": (lambda r: r.update(servers={}), "servers is empty"),
    "server not an object": (
        lambda r: r["servers"].update(mslearn="https://learn.microsoft.com/api/mcp"),
        "server 'mslearn' must be an object",
    ),
    "server name": (
        lambda r: r["servers"].update({"ms learn": r["servers"].pop("mslearn")}),
        "server name 'ms learn' is not an identifier",
    ),
    "http url": (
        lambda r: r["servers"]["mslearn"].update(url="http://learn.microsoft.com/api/mcp"),
        "server 'mslearn': the URL is not https",
    ),
    "query string": (
        lambda r: r["servers"]["mslearn"].update(url="https://learn.microsoft.com/api/mcp?k=1"),
        "server 'mslearn': the URL has a query string",
    ),
    "key in the path": (
        lambda r: r["servers"]["mslearn"].update(
            url="https://learn.microsoft.com/api/mcp/s/k7Qz9xV2pL4mN8rT1wY6bC3d/sse"
        ),
        "server 'mslearn': path segment 4 of the URL (24 characters, random-looking)",
    ),
    "placeholder": (
        lambda r: r["servers"]["mslearn"].update(url="https://${HOST}/api/mcp"),
        "server 'mslearn': the URL depends on HOST",
    ),
    "not a url": (
        lambda r: r["servers"]["mslearn"].update(url="learn.microsoft.com/api/mcp"),
        "server 'mslearn': the URL scheme is missing",
    ),
    "tool not a name": (
        lambda r: r["servers"]["deepwiki"].update(tool="read wiki; rm -rf"),
        "tool must be a bare MCP tool name",
    ),
    "tool missing": (
        lambda r: r["servers"]["deepwiki"].pop("tool"),
        "tool must be a bare MCP tool name, found None",
    ),
    "fields empty": (lambda r: r.update(fields=[]), 'no field is "equal"'),
    "fields not a list": (lambda r: r.update(fields={"a": 1}), "fields must be a list"),
    "all drift": (
        lambda r: [f.update(expect="drifts") for f in _equal_fields(r)],
        'no field is "equal"',
    ),
    "duplicate name": (
        lambda r: r["fields"][1].update(name=r["fields"][0]["name"]),
        "field names must be unique; repeated: sections",
    ),
    "unknown expectation": (
        lambda r: r["fields"][0].update(expect="close enough"),
        "expect must be equal or drifts",
    ),
    "unknown measure": (
        lambda r: r["fields"][0].update(measure="vibes"),
        "measure must be one of headings, bullets",
    ),
    "section that does not resolve": (
        lambda r: r["fields"][2].update(section="Not a section"),
        "Not a section",
    ),
    "output not an object": (
        lambda r: r["skill_run"].update(output="text"),
        "skill_run.output must be an object, found str",
    ),
    "tool call to another tool": (
        lambda r: r["pipeline_run"]["tool_calls"][0].update(tool="microsoft_docs_fetch"),
        "is not one of the servers' tools",
    ),
    "tool call without result": (
        lambda r: r["pipeline_run"]["tool_calls"][0].pop("result"),
        "has no result",
    ),
    "tool call twice": (
        lambda r: r["pipeline_run"]["tool_calls"].append(r["pipeline_run"]["tool_calls"][0]),
        "names a node twice",
    ),
}


@pytest.mark.parametrize("name", list(_BREAKS))
def test_a_malformed_reference_is_refused_with_a_reason(tmp_path: Path, name: str) -> None:
    mutate, message = _BREAKS[name]
    raw = _raw()
    mutate(raw)
    with pytest.raises(demo.DemoError) as caught:
        _load(tmp_path, raw)
    assert message in str(caught.value)
