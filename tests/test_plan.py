"""``rote.plan``: the one plan builder the prefix runner and a compiled skill share."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from rote.ir import Pipeline, load_pipeline
from rote.plan import (
    PLAN_FORMAT,
    PlanError,
    build_plan,
    gate_answer_schema,
    judge_references,
    plan_hash,
    plan_text,
)

REPO = Path(__file__).resolve().parent.parent
EXAMPLES = REPO / "examples"


def _steps(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {s["id"]: s for s in plan["steps"]}


def _example(rel: str) -> Pipeline:
    return load_pipeline(EXAMPLES / rel / "pipeline.yaml")


def _mini(nodes: list[dict[str, Any]], **extra: Any) -> Pipeline:
    return Pipeline.model_validate(
        {
            "name": "mini",
            "version": "0.1.0",
            "input": {"type": "In", "required": ["q"]},
            "nodes": nodes,
            "edges": extra.pop("edges", []),
            "entry_nodes": [nodes[0]["id"]],
            "exit_nodes": [nodes[-1]["id"]],
            **extra,
        }
    )


def test_docs_digest_live_plan_calls_every_tool_and_function() -> None:
    plan = build_plan(_example("docs-digest/compiled"), mcp="live")
    assert plan["format"] == PLAN_FORMAT
    assert plan["name"] == "docs-digest"
    assert plan["required_inputs"] == ["repo", "topic"]
    assert plan["exit_nodes"] == ["write_digest"]
    steps = _steps(plan)
    assert {i: s["kind"] for i, s in steps.items()} == {
        "search_microsoft_learn": "mcp",
        "read_repo_outline": "mcp",
        "pick_reading_list": "call",
        "write_digest": "call",
    }
    search = steps["search_microsoft_learn"]
    assert (search["server"], search["tool"], search["wave"]) == (
        "mslearn",
        "microsoft_docs_search",
        0,
    )
    assert search["args"] == [("query", "payload", "arg_query")]
    assert search["retry"] == {"attempts": 4, "backoff": "exponential"}
    assert "observed" not in search  # live: nothing recorded stands in
    pick = steps["pick_reading_list"]
    assert (pick["module"], pick["symbol"]) == ("extracted.pick_reading_list", "pick_reading_list")
    assert pick["constants"] == {"result_count": 5}
    assert pick["bindings"] == [
        ["request", "input", None, None],
        ["search_microsoft_learn", "node", "search_microsoft_learn", None],
    ]
    assert steps["write_digest"]["wave"] == 2


def test_the_api_backend_runs_the_impl_of_a_bound_node() -> None:
    steps = _steps(build_plan(_example("docs-digest/compiled"), mcp="live", backend="api"))
    assert steps["search_microsoft_learn"]["kind"] == "call"
    assert steps["search_microsoft_learn"]["module"] == "extracted.search_microsoft_learn"


def test_an_mcp_only_node_has_nothing_to_run_on_the_api_backend() -> None:
    pipeline = _mini(
        [
            {
                "id": "fetch",
                "kind": "external_call",
                "description": "d",
                "mcp": {"server": "s", "tool": "t"},
            }
        ]
    )
    assert _steps(build_plan(pipeline, mcp="live"))["fetch"]["kind"] == "mcp"
    with pytest.raises(PlanError, match="fetch: external_call has nothing to run on the api"):
        build_plan(pipeline, mcp="live", backend="api")


def test_a_fanned_judge_carries_its_prompt_references_and_element() -> None:
    steps = _steps(build_plan(_example("deal-monitor/expected"), mcp="live"))
    classify = steps["classify_thread_step"]
    assert classify["kind"] == "judge"
    assert classify["fan_param"] == "thread"
    assert classify["references"] == ["thread.account_name", "thread"]
    assert classify["output_schema"]["type"] == "object"
    score = steps["score_new_opportunities"]
    assert score["fan_param"] == "opportunity"
    assert score["references"][0] == "scoring_rubric"
    assert "opportunity" not in score["references"]  # only its fields are read


def test_judge_references_keep_first_use_order_once() -> None:
    assert judge_references("{{ a.b }} then {{a}} and {{ a.b }} {{c_d}}") == ["a.b", "a", "c_d"]
    assert judge_references("no placeholders, {single} braces") == []


def test_a_gate_carries_its_answer_schema_and_deadline() -> None:
    steps = _steps(build_plan(_example("expense-audit/expected"), mcp="live"))
    gate = steps["controller_review"]
    assert gate["kind"] == "gate"
    assert gate["timeout_s"] == 72 * 3600
    assert gate["answer_schema"] == {
        "type": "object",
        "properties": {
            "reviewer": {"type": "string"},
            "approved_ids": {"type": "array", "items": {"type": "string"}},
            "rejected_ids": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["reviewer", "approved_ids", "rejected_ids"],
        "additionalProperties": False,
    }
    assert gate["description"] and gate["signal"]
    assert "payload" not in gate  # live: a person answers it


def test_a_gate_without_a_timeout_takes_the_pipeline_default() -> None:
    nodes = [
        {"id": "ask", "kind": "hitl_gate", "description": "d", "signal": "go"},
    ]
    assert _steps(build_plan(_mini(nodes), mcp="live"))["ask"]["timeout_s"] == 7 * 86400
    custom = _mini(nodes, config={"hitl": {"default_timeout": "2h"}})
    assert _steps(build_plan(custom, mcp="live"))["ask"]["timeout_s"] == 7200
    own = _mini([{**nodes[0], "timeout": "15m"}], config={"hitl": {"default_timeout": "2h"}})
    assert _steps(build_plan(own, mcp="live"))["ask"]["timeout_s"] == 900


def test_the_prefix_never_runs_vendor_code_even_when_written() -> None:
    pipeline = _mini(
        [{"id": "push", "kind": "external_call", "description": "d", "impl": "extracted/p.py:push"}]
    )
    recorded = build_plan(pipeline, mcp="recorded", implementations={"push": "implemented"})
    assert _steps(recorded)["push"]["kind"] == "skip"
    assert _steps(recorded)["push"]["reason"] == "external_call"
    assert _steps(build_plan(pipeline, mcp="live"))["push"]["kind"] == "call"


@pytest.mark.parametrize(
    ("output", "schema"),
    [
        ({}, {"type": "object"}),
        ("bool", {"type": "boolean"}),
        (
            {"ok": "bool"},
            {
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
                "additionalProperties": False,
            },
        ),
    ],
)
def test_gate_answer_schema_follows_the_declared_output(
    output: Any, schema: dict[str, Any]
) -> None:
    pipeline = _mini(
        [{"id": "ask", "kind": "hitl_gate", "description": "d", "signal": "go", "output": output}]
    )
    assert gate_answer_schema(pipeline.nodes[0]) == schema


def test_an_agent_loop_carries_its_tools_cap_and_output_contract() -> None:
    steps = _steps(build_plan(_example("bdr-outreach/expected"), mcp="live"))
    [loop] = [s for s in steps.values() if s["kind"] == "loop"][:1]
    pipeline = _example("bdr-outreach/expected")
    node = pipeline.node_by_id(loop["id"])
    assert loop["tools"] == node.tools
    assert loop["loop_body"] == (node.loop_body or [])
    assert loop["max_iterations"] == (node.termination.max_iterations if node.termination else 10)
    assert loop["output_schema"]["additionalProperties"] is False
    # Loop-body sub-nodes run inside the loop, never as top-level steps.
    assert not set(node.loop_body or ()) & set(steps)


def test_a_legacy_judge_cannot_run_live_and_skips_in_the_prefix() -> None:
    nodes = [
        {"id": "j", "kind": "llm_judge", "description": "d", "signature": "signatures/j.py:J"},
    ]
    with pytest.raises(PlanError, match="j: a judge needs a signature_spec"):
        build_plan(_mini(nodes), mcp="live")
    assert _steps(build_plan(_mini(nodes), mcp="recorded"))["j"] == {
        "id": "j",
        "node_kind": "llm_judge",
        "wave": 0,
        "bindings": [],
        "kind": "skip",
        "reason": "llm_judge",
    }


def test_recorded_mode_runs_only_what_the_prefix_may() -> None:
    """The prefix never runs a judge, a loop, vendor code, an unwritten function
    or an unanswered gate: each is a skip with the reason the report shows."""
    pipeline = _example("deal-monitor/expected")
    plan = build_plan(pipeline, mcp="recorded", implementations={})
    kinds = {s["id"]: (s["kind"], s.get("reason")) for s in plan["steps"]}
    assert kinds["classify_thread_step"] == ("skip", "llm_judge")
    pure = [n.id for n in pipeline.nodes if n.kind == "pure_function"]
    assert pure and all(kinds[i] == ("skip", "pure_function without impl") for i in pure)
    ready = build_plan(pipeline, mcp="recorded", implementations={pure[0]: "implemented"})
    assert _steps(ready)[pure[0]]["kind"] == "call"


def test_every_committed_example_plans_live_or_says_why() -> None:
    planned, refused = [], []
    for path in sorted(EXAMPLES.rglob("pipeline.yaml")):
        if "runtime" in path.parts:
            continue
        rel = str(path.relative_to(EXAMPLES))
        pipeline = Pipeline.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
        try:
            plan = build_plan(pipeline, mcp="live")
        except PlanError as e:
            refused.append((rel, str(e)))
            continue
        json.loads(plan_text(plan))  # plain JSON throughout
        planned.append(rel)
    # Every current pipeline plans; only old snapshots from before data-flow
    # threading are refused, each with its reason, never a crash.
    assert not [r for r, _ in refused if "/runs/" not in r], refused
    assert len(planned) >= 15
    for rel, why in refused:
        assert "signature_spec" in why or "nothing to fan over" in why, (rel, why)


def test_the_plan_hash_follows_the_plan_bytes() -> None:
    pipeline = _example("docs-digest/compiled")
    text = plan_text(build_plan(pipeline, mcp="live"))
    assert plan_hash(text) == plan_hash(plan_text(build_plan(pipeline, mcp="live")))
    edited = pipeline.model_copy(deep=True)
    edited.node_by_id("pick_reading_list").constants = {"result_count": 3}
    assert plan_hash(plan_text(build_plan(edited, mcp="live"))) != plan_hash(text)
