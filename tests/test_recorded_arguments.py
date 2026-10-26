"""MCP arguments bound from what a recorded run sent (rote.compiler.system1.evidence).

The recorded calls and the sample are a real ``--baseline`` run of the
docs-digest skill (examples/docs-digest/skill) against DeepWiki and Microsoft
Learn (fixtures/mcp_results).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from rote.compiler import system1
from rote.compiler.grounding import load_sample
from rote.compiler.system1.evidence import bind_recorded_arguments
from rote.compiler.tool_grounding import ToolSchema, tool_findings
from rote.eval.baseline import ObservedToolCall
from rote.ir import Pipeline

FIXTURES = Path(__file__).parent / "fixtures" / "mcp_results"
DATA = json.loads((FIXTURES / "docs_digest.json").read_text(encoding="utf-8"))
SKILL = Path(__file__).resolve().parent.parent / "examples" / "docs-digest" / "skill"
TOOLS = {
    (d["server"], d["tool"]): ToolSchema(
        d["server"], d["tool"], d["input_schema"], "tools/list", True, d["output_schema"]
    )
    for d in DATA["declared"]
}


def _observations(**overrides: dict[str, Any]) -> list[ObservedToolCall]:
    out = []
    for o in DATA["observations"]:
        out.append(ObservedToolCall(**{**o, **overrides.get(o["tool"], {})}))
    return out


def _skeleton(tmp_path: Path, observations: list[Any], sample: Any = None) -> Pipeline:
    work = tmp_path / "work"
    work.mkdir()
    sample = sample or load_sample(SKILL / "input.json")
    system1.prepare(SKILL, work, sample=sample, tools=TOOLS, observations=observations)
    return Pipeline.model_validate(yaml.safe_load((work / "pipeline.yaml").read_text()))


def _bound(p: Pipeline, tool: str) -> Any:
    return next(n for n in p.nodes if n.mcp and n.mcp.tool == tool)


def test_recorded_values_bind_the_arguments_the_compile_left_out(tmp_path: Path) -> None:
    p = _skeleton(tmp_path, _observations())
    search = _bound(p, "microsoft_docs_search")
    assert search.mcp.args == {"query": "arg_query"}
    assert search.inputs["arg_query"] == "pipeline.input.topic"
    wiki = _bound(p, "read_wiki_structure")
    assert wiki.mcp.args == {"repoName": "arg_repoName"}
    assert wiki.inputs["arg_repoName"] == "pipeline.input.repo"
    # The required argument the hard check refused before is bound now.
    assert tool_findings(p, TOOLS) == []


def test_without_a_recording_nothing_is_bound(tmp_path: Path) -> None:
    p = _skeleton(tmp_path, [])
    assert _bound(p, "read_wiki_structure").mcp.args == {}
    [finding] = tool_findings(p, TOOLS)
    assert finding.kind == "tool_arguments_invalid" and "repoName" in finding.message


def test_a_value_two_fields_hold_binds_nothing(tmp_path: Path) -> None:
    from rote.compiler.grounding import Sample

    repo = "Azure/azure-functions-durable-python"
    sample = Sample({"topic": "t", "repo": repo, "mirror": repo}, "test")
    p = _skeleton(tmp_path, _observations(), sample)
    assert "repoName" not in (_bound(p, "read_wiki_structure").mcp.args or {})


def test_calls_that_disagree_about_the_source_bind_nothing(tmp_path: Path) -> None:
    calls = _observations()
    wiki = next(o for o in calls if o.tool == "read_wiki_structure")
    other = ObservedToolCall(
        server=wiki.server,
        tool=wiki.tool,
        input={"repoName": "someone/else"},
        result=wiki.result,
        is_error=False,
    )
    p = _skeleton(tmp_path, [*calls, other])
    assert "repoName" not in (_bound(p, "read_wiki_structure").mcp.args or {})


def test_an_error_call_is_no_evidence(tmp_path: Path) -> None:
    p = _skeleton(tmp_path, _observations(read_wiki_structure={"is_error": True}))
    assert "repoName" not in (_bound(p, "read_wiki_structure").mcp.args or {})


# ───────── the source kinds, on the plan directly ─────────


def _plan(tmp_path: Path) -> Any:
    from rote.compiler.system1.decide import plan
    from rote.compiler.system1.parse import parse_skill

    return plan(parse_skill(SKILL), None, TOOLS, None)


def _call(tool: str, server: str, input: dict[str, Any], result: Any = "{}") -> ObservedToolCall:
    return ObservedToolCall(server=server, tool=tool, input=input, result=result, is_error=False)


def test_a_constant_and_an_upstream_result_are_sources(tmp_path: Path) -> None:
    p = _plan(tmp_path)
    search = next(n for n in p.nodes if n.mcp and n.mcp["tool"] == "microsoft_docs_search")
    wiki = next(n for n in p.nodes if n.mcp and n.mcp["tool"] == "read_wiki_structure")
    search.constants["topic_text"] = "durable"
    wiki.inputs[search.id] = f"{search.id}.output"
    calls = [
        _call("microsoft_docs_search", "mslearn", {"query": "durable"}, '{"repo": "a/b"}'),
        _call("read_wiki_structure", "deepwiki", {"repoName": "a/b"}),
    ]
    bind_recorded_arguments(p, calls, {})
    assert search.mcp["args"] == {"query": "topic_text"}
    assert "query" not in search.inputs  # a constant needs no input
    assert wiki.mcp["args"] == {"repoName": "arg_repoName"}
    assert wiki.inputs["arg_repoName"] == f"{search.id}.output.repo"
    decision = next(d for d in reversed(wiki.decisions) if d.field == "mcp.args")
    assert decision.confidence == pytest.approx(0.95)
    assert "every recorded call sent" in decision.rule


def test_a_result_of_a_node_not_upstream_is_no_source(tmp_path: Path) -> None:
    """Binding it would add a data dependency the plan does not have."""
    p = _plan(tmp_path)
    wiki = next(n for n in p.nodes if n.mcp and n.mcp["tool"] == "read_wiki_structure")
    wiki.inputs = {"request": "pipeline.input"}
    calls = [
        _call("microsoft_docs_search", "mslearn", {"query": "x"}, '{"repo": "a/b"}'),
        _call("read_wiki_structure", "deepwiki", {"repoName": "a/b"}),
    ]
    bind_recorded_arguments(p, calls, {})
    assert wiki.mcp["args"] == {}


@pytest.mark.parametrize("value", [True, "", None])
def test_an_uninformative_value_binds_nothing(tmp_path: Path, value: Any) -> None:
    p = _plan(tmp_path)
    wiki = next(n for n in p.nodes if n.mcp and n.mcp["tool"] == "read_wiki_structure")
    bind_recorded_arguments(
        p, [_call("read_wiki_structure", "deepwiki", {"repoName": value})], {"repo": value}
    )
    assert wiki.mcp["args"] == {}


def test_an_argument_already_bound_is_kept(tmp_path: Path) -> None:
    p = _plan(tmp_path)
    wiki = next(n for n in p.nodes if n.mcp and n.mcp["tool"] == "read_wiki_structure")
    wiki.mcp = {**wiki.mcp, "args": {"repoName": "chosen"}}
    bind_recorded_arguments(
        p, [_call("read_wiki_structure", "deepwiki", {"repoName": "a/b"})], {"repo": "a/b"}
    )
    assert wiki.mcp["args"] == {"repoName": "chosen"}


def test_a_type_mismatch_is_no_match(tmp_path: Path) -> None:
    """`1` sent is not the sample's `"1"`: the tool would get the other type."""
    p = _plan(tmp_path)
    wiki = next(n for n in p.nodes if n.mcp and n.mcp["tool"] == "read_wiki_structure")
    bind_recorded_arguments(
        p, [_call("read_wiki_structure", "deepwiki", {"repoName": 1})], {"repo": "1"}
    )
    assert wiki.mcp["args"] == {}


def test_calls_that_each_match_a_different_source_bind_nothing(tmp_path: Path) -> None:
    """Each call traces to one field, but not the same one: no single source."""
    p = _plan(tmp_path)
    wiki = next(n for n in p.nodes if n.mcp and n.mcp["tool"] == "read_wiki_structure")
    calls = [
        _call("read_wiki_structure", "deepwiki", {"repoName": "a/b"}),
        _call("read_wiki_structure", "deepwiki", {"repoName": "c/d"}),
    ]
    bind_recorded_arguments(p, calls, {"repo": "a/b", "mirror": "c/d"})
    assert wiki.mcp["args"] == {}


@pytest.mark.asyncio
async def test_a_compile_binds_from_its_baseline_recordings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    from rote.compiler import Compiler
    from rote.compiler.system1 import generate
    from rote.eval.baseline import OBSERVED_TOOLS_FILENAME
    from rote.ir import load_pipeline
    from tests.test_system1_generate import FakeGenerator

    skill = tmp_path / "skill"
    shutil.copytree(SKILL, skill)
    probe = tmp_path / "baseline"
    probe.mkdir()
    (probe / OBSERVED_TOOLS_FILENAME).write_text(json.dumps(DATA["observations"]))
    skeleton = tmp_path / "skeleton"
    skeleton.mkdir()
    system1.prepare(skill, skeleton, tools=TOOLS)
    fake = FakeGenerator({n.id: n for n in load_pipeline(skeleton / "pipeline.yaml").nodes})
    monkeypatch.setattr(Compiler, "select_driver", lambda self: fake)
    monkeypatch.setattr(generate.Settings, "prime_seconds", 0.0)

    async def listed(self: Any, observations: Any, skill_dir: Path, snapshot: Any = None) -> Any:
        return TOOLS, {}, {}

    monkeypatch.setattr(Compiler, "_tool_schemas", listed)
    out = tmp_path / "out"
    await Compiler(
        run_generated_tests=False, system1_tiebreak="none", system1_structure="rules"
    ).compile(skill, out, probe_dir=probe)
    wiki = _bound(load_pipeline(out / "pipeline.yaml"), "read_wiki_structure")
    assert wiki.mcp.args == {"repoName": "arg_repoName"}
    assert wiki.inputs["arg_repoName"] == "pipeline.input.repo"


def test_a_boolean_field_is_no_source_for_a_number(tmp_path: Path) -> None:
    """`True == 1` in Python: a flag in the sample must not source a count."""
    p = _plan(tmp_path)
    wiki = next(n for n in p.nodes if n.mcp and n.mcp["tool"] == "read_wiki_structure")
    bind_recorded_arguments(
        p, [_call("read_wiki_structure", "deepwiki", {"repoName": 1})], {"verbose": True}
    )
    assert wiki.mcp["args"] == {}


def test_the_structure_path_binds_recorded_arguments_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The structure pass leaves `query` unbound; the recorded call fills it, as on rules."""

    from rote.compiler.system1 import structure
    from rote.compiler.system1.tiebreak import Lane

    units = {u.title: u.index for u in structure.section_units(SKILL)}
    none = {"io": "none", "how": "none", "name": "", "server": "", "tool": ""}
    search = {
        "io": "read",
        "how": "declared_tool",
        "name": "mslearn",
        "server": "mslearn",
        "tool": "microsoft_docs_search",
    }

    def node(nid: str, title: str, quote: str, kind: str, **over: Any) -> dict[str, Any]:
        return {
            "id": nid,
            "section": units[title],
            "quote": quote,
            "kind": kind,
            "intent": nid,
            "reads": ["pipeline.input"],
            "fan_out_over": "",
            "in_loop": "",
            "mandatory": True,
            "outputs": [],
            "system": dict(none),
            "args": [],
            "constants": [],
            "gate": {"signal": "", "payload": [], "timeout": ""},
            "loop": {"tools": [], "stop_condition": "", "max_iterations": 0},
            "evidence_field": "",
            "confidence": 0.9,
            "why": "fixture",
            **over,
        }

    answer = {
        "input_fields": [],
        "run_outputs": [],
        "nodes": [
            node(
                "search_learn",
                "1. Search Microsoft Learn",
                "Call `microsoft_docs_search`",
                "external_call",
                system=search,
            ),
            node(
                "write_digest",
                "4. Write the digest",
                "Return a markdown document",
                "pure_function",
                reads=["search_learn"],
            ),
        ],
    }

    def call(**kw: Any) -> tuple[dict[str, Any], list[Any]]:
        return json.loads(json.dumps(answer)), []

    monkeypatch.setattr(structure, "_structured_call", lambda: call)
    rubric = tmp_path / "rubric.md"
    rubric.write_text("rubric")
    work = tmp_path / "work"
    work.mkdir()
    system1.prepare(
        SKILL,
        work,
        sample=load_sample(SKILL / "input.json"),
        tools=TOOLS,
        observations=_observations(),
        structure_lane=Lane("claude-cli"),
        rubric=rubric,
    )
    p = Pipeline.model_validate(yaml.safe_load((work / "pipeline.yaml").read_text()))
    bound = _bound(p, "microsoft_docs_search")
    assert bound.mcp is not None and bound.mcp.args == {"query": "arg_query"}
    assert bound.inputs["arg_query"] == "pipeline.input.topic"


def test_a_taken_bare_key_does_not_drop_the_binding(tmp_path: Path) -> None:
    """The argument binds to `arg_<param>`, so an input named `repoName` is no obstacle."""
    p = _plan(tmp_path)
    wiki = next(n for n in p.nodes if n.mcp and n.mcp["tool"] == "read_wiki_structure")
    wiki.inputs["repoName"] = "pipeline.input"
    bind_recorded_arguments(
        p, [_call("read_wiki_structure", "deepwiki", {"repoName": "a/b"})], {"repo": "a/b"}
    )
    assert wiki.mcp["args"] == {"repoName": "arg_repoName"}
    assert wiki.inputs["arg_repoName"] == "pipeline.input.repo"
    assert wiki.inputs["repoName"] == "pipeline.input"
