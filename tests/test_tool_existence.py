"""A bound MCP tool must be one its server lists (rote.compiler.tool_grounding).

Real schemas: DeepWiki's and Microsoft Learn's live tools/list
(fixtures/mcp_results).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rote.compiler.tool_grounding import ToolSchema, tool_findings, tool_questions
from rote.ir import Pipeline

DATA = json.loads(
    (Path(__file__).parent / "fixtures" / "mcp_results" / "docs_digest.json").read_text()
)
TOOLS = {
    (d["server"], d["tool"]): ToolSchema(
        d["server"], d["tool"], d["input_schema"], "tools/list", True, d["output_schema"]
    )
    for d in DATA["declared"]
}


def _pipeline(**node: Any) -> Pipeline:
    return Pipeline.model_validate(
        {
            "name": "p",
            "version": "0.1.0",
            "input": {"type": "In"},
            "nodes": [{"id": "n", "description": "d", **node}],
            "edges": [],
        }
    )


def _bound(server: str, tool: str, **args: str) -> Pipeline:
    return _pipeline(
        kind="external_call",
        inputs={k: f"pipeline.input.{k}" for k in args.values()},
        mcp={"server": server, "tool": tool, "args": args},
    )


def test_a_tool_its_server_does_not_list_is_an_error_owned_by_the_node() -> None:
    [finding] = tool_findings(_bound("deepwiki", "read_wiki", repoName="repo"), TOOLS)
    assert (finding.node_id, finding.kind, finding.severity) == ("n", "tool_unknown", "error")
    assert "deepwiki's tools/list has no tool 'read_wiki'" in finding.message
    assert "`read_wiki_structure`" in finding.message
    # Not also an open question: it is an error.
    assert tool_questions(_bound("deepwiki", "read_wiki", repoName="repo"), TOOLS) == []


def test_a_listed_tool_is_not_unknown() -> None:
    assert tool_findings(_bound("deepwiki", "read_wiki_structure", repoName="repo"), TOOLS) == []


def test_a_server_that_was_not_listed_proves_nothing() -> None:
    """Unreachable, unregistered or needing a login: an open question, never an error."""
    pipeline = _bound("slack", "anything", channel="c")
    assert tool_findings(pipeline, TOOLS) == []
    [(node_id, text)] = tool_questions(pipeline, TOOLS)
    assert node_id == "n" and "no schema for this tool was available" in text


def test_a_recorded_schema_is_no_listing() -> None:
    """Inferred from a baseline's calls, it shows tools used, not tools offered."""
    recorded = {
        k: ToolSchema(t.server, t.tool, t.input_schema, "recorded", False) for k, t in TOOLS.items()
    }
    assert tool_findings(_bound("deepwiki", "read_wiki", repoName="repo"), recorded) == []


def test_an_agent_loops_tool_must_be_on_its_named_server() -> None:
    loop = _pipeline(
        kind="agent_loop",
        tools=["read_wiki_structure", "microsoft_docs_search"],
        # The second is on the wrong server.
        tool_servers={"read_wiki_structure": "deepwiki", "microsoft_docs_search": "deepwiki"},
        termination={"condition": "done", "max_iterations": 3},
    )
    [finding] = tool_findings(loop, TOOLS)
    assert finding.kind == "tool_unknown" and "microsoft_docs_search" in finding.message


@pytest.mark.parametrize("missing", [{}, {"repo": "repo"}])
def test_a_required_argument_left_unbound_is_an_error_owned_by_the_node(
    missing: dict[str, str],
) -> None:
    [finding] = tool_findings(_bound("deepwiki", "read_wiki_structure", **missing), TOOLS)
    assert (finding.node_id, finding.kind, finding.severity) == (
        "n",
        "tool_arguments_invalid",
        "error",
    )
    assert "repoName" in finding.message


def test_an_argument_from_a_key_the_inputs_never_fill_is_an_error() -> None:
    """cf-limits-brief, a real full-rubric compile: `args: {query: query}` on a
    node whose only input is `product`, so the search went out with no query."""
    pipeline = _pipeline(
        kind="external_call",
        fan_out=True,
        inputs={"product": "pipeline.input.products"},
        mcp={"server": "mslearn", "tool": "microsoft_docs_search", "args": {"query": "query"}},
    )
    [finding] = tool_findings(pipeline, TOOLS)
    assert (finding.node_id, finding.kind, finding.severity) == (
        "n",
        "tool_arguments_unsourced",
        "error",
    )
    assert "`query` <- `query`" in finding.message and "`product`" in finding.message


@pytest.mark.parametrize(
    "node",
    [
        # From an input the node has.
        {"inputs": {"query": "pipeline.input.topic"}, "args": {"query": "query"}},
        # From a constant.
        {"inputs": {}, "constants": {"q": "x"}, "args": {"query": "q"}},
        # A literal.
        {"inputs": {}, "args": {"query": {"literal": "x"}}},
    ],
)
def test_a_sourced_argument_is_fine(node: dict[str, Any]) -> None:
    pipeline = _pipeline(
        kind="external_call",
        inputs=node["inputs"],
        constants=node.get("constants"),
        mcp={"server": "mslearn", "tool": "microsoft_docs_search", "args": node["args"]},
    )
    assert tool_findings(pipeline, TOOLS) == []


def test_a_bundled_snapshot_proves_no_tool_unknown() -> None:
    """A skill's bundled tools/list snapshot may predate a tool the live server has."""
    from dataclasses import replace

    bundled = {k: replace(t, origin="bundled") for k, t in TOOLS.items()}
    pipeline = _bound("deepwiki", "a_tool_added_since", repoName="repo")
    assert tool_findings(pipeline, bundled) == []
    # The same listing, live, does say the tool is not there.
    assert [f.kind for f in tool_findings(pipeline, TOOLS)] == ["tool_unknown"]
