"""``rote.plan``: the one plan builder the prefix runner and a compiled skill share."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from rote.ir import Pipeline, load_pipeline
from rote.plan import (
    PLACEHOLDER,
    PLAN_FORMAT,
    PlanError,
    build_plan,
    declared_answer_schema,
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


def test_an_mcp_only_node_is_the_agents_to_do_on_the_api_backend() -> None:
    """Nothing runs it in code there, so the agent does it with its own tools."""
    pipeline = _mini(
        [
            {
                "id": "fetch",
                "kind": "external_call",
                "description": "Fetch the page.",
                "mcp": {"server": "s", "tool": "t"},
                "constants": {"page": 2},
                "output": {"body": "str"},
            }
        ]
    )
    assert _steps(build_plan(pipeline, mcp="live"))["fetch"]["kind"] == "mcp"
    step = _steps(build_plan(pipeline, mcp="live", backend="api"))["fetch"]
    assert {k: step[k] for k in ("kind", "description", "constants", "fan_param")} == {
        "kind": "do",
        "description": "Fetch the page.",
        "constants": {"page": 2},
        "fan_param": None,
    }
    assert step["output_schema"]["required"] == ["body"]
    assert step["output_schema"]["additionalProperties"] is False


def test_an_unwritten_external_call_is_do_and_an_unwritten_pure_function_is_not() -> None:
    """A stub or absent external_call is agent work. A pure_function stub stays
    a call, refused at emit for its stub: a missing pure function is a compile
    defect (and the IR requires a pure_function to name an impl at all)."""
    call = {"id": "push", "kind": "external_call", "description": "d", "impl": "extracted/c.py:p"}
    pipeline = _mini([call])
    for status in ("stub", "absent"):
        plan = build_plan(pipeline, mcp="live", implementations={"push": status})
        assert _steps(plan)["push"]["kind"] == "do", status
    plan = build_plan(pipeline, mcp="live", implementations={"push": "implemented"})
    assert _steps(plan)["push"]["kind"] == "call"
    pure = {**call, "kind": "pure_function"}
    plan = build_plan(_mini([pure]), mcp="live", implementations={"push": "stub"})
    assert _steps(plan)["push"]["kind"] == "call"
    # The prefix never asks the agent: a recorded plan skips the same node.
    recorded = build_plan(pipeline, mcp="recorded", implementations={"push": "stub"})
    assert _steps(recorded)["push"]["kind"] == "skip"


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
    # Its fields first, then the whole object, which the judge passes through.
    assert score["references"][1] == "opportunity.account_name"
    assert score["references"][-1] == "opportunity"


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
def test_a_gates_answer_schema_follows_the_declared_output(
    output: Any, schema: dict[str, Any]
) -> None:
    pipeline = _mini(
        [{"id": "ask", "kind": "hitl_gate", "description": "d", "signal": "go", "output": output}]
    )
    assert declared_answer_schema(pipeline.nodes[0]) == schema


def test_an_agent_loop_carries_its_tools_cap_and_output_contract() -> None:
    steps = _steps(build_plan(_example("bdr-outreach/expected"), mcp="live"))
    loop = steps["lead_generation_loop"]
    assert loop["kind"] == "loop"
    pipeline = _example("bdr-outreach/expected")
    node = pipeline.node_by_id(loop["id"])
    assert loop["tools"] == node.tools
    assert loop["max_iterations"] == (node.termination.max_iterations if node.termination else 10)
    assert loop["output_schema"]["additionalProperties"] is False
    # Each body sub-node rides along in full, so the driver never reads the IR.
    enrich, vet = loop["body"]
    assert node.loop_body == [enrich["id"], vet["id"]] == ["enrich_contact_batch", "vet_contact"]
    enrich_node = pipeline.node_by_id("enrich_contact_batch")
    assert (enrich["kind"], enrich["module"], enrich["symbol"]) == (
        "call",
        "extracted.zoominfo",
        "enrich_batch",
    )
    assert enrich["constants"] == enrich_node.constants and enrich["constants"]
    vet_spec = pipeline.node_by_id("vet_contact").signature_spec
    assert vet_spec is not None
    assert vet["kind"] == "judge" and vet["prompt"] == vet_spec.prompt
    assert vet["output_schema"] == vet_spec.output_schema
    assert vet["references"] == ["contact", "brief", "intel"]
    assert "bindings" not in enrich and enrich["fan_param"] is None
    # Loop-body sub-nodes run inside the loop, never as top-level steps.
    assert not set(node.loop_body or ()) & set(steps)
    # The agent calls the code body with its payload, constants left out
    # (they are merged last); the judge body's arguments are its input schema.
    assert enrich["tool_schema"] == {
        "type": "object",
        "properties": {
            "contacts": {
                "type": "array",
                "items": enrich["tool_schema"]["properties"]["contacts"]["items"],
            }
        },
        "required": ["contacts"],
    }
    assert not set(enrich_node.constants or {}) & set(enrich["tool_schema"]["properties"])
    assert vet["tool_schema"] == vet_spec.input_schema
    # The task is the loop lanes' prompt, split where the inputs go, less
    # what the LOOP block says itself: the tool list and the answer paragraph.
    from rote.inference._runtime_helper import _agent_task_prompt

    head, tail = loop["task"]
    full = _agent_task_prompt(
        description=node.description,
        task="INPUTS",
        termination=node.termination.condition if node.termination else None,
        local_tool_names=[],
        submit="schema",
    )
    assert full.startswith(head + "INPUTS" + tail + "\n\nWhen you are done")
    assert tail.endswith(f"Stop when: {node.termination.condition}" if node.termination else "")


def test_a_loop_body_step_the_driver_cannot_run_is_refused() -> None:
    """A legacy ``signature:`` judge has no prompt to carry, in a body as anywhere."""
    nodes = [
        {"id": "inner", "kind": "llm_judge", "description": "d", "signature": "signatures/i.py:I"},
        {
            "id": "loop",
            "kind": "agent_loop",
            "description": "d",
            "tools": ["t"],
            "tool_servers": {"t": "s"},
            "loop_body": ["inner"],
        },
    ]
    pipeline = _mini(nodes)
    with pytest.raises(PlanError, match="inner: a judge needs a signature_spec"):
        build_plan(pipeline, mcp="live")
    recorded = _steps(build_plan(pipeline, mcp="recorded"))
    assert recorded["loop"]["kind"] == "skip"  # the prefix never runs a loop


def test_a_single_type_gate_takes_its_output_schema_as_the_answer() -> None:
    """Evidence beats the type name, as for an agent loop's output."""
    evidence = {"type": "array", "items": {"type": "string"}}
    gate = {
        "id": "ask",
        "kind": "hitl_gate",
        "description": "d",
        "signal": "go",
        "output": "list",
        "output_schema": evidence,
    }
    assert declared_answer_schema(_mini([gate]).nodes[0]) == evidence
    named = {k: v for k, v in gate.items() if k != "output_schema"}
    assert declared_answer_schema(_mini([named]).nodes[0]) == {"type": "array"}


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
    # Every current pipeline plans, or has an agent loop whose tools no
    # server is pinned to (the skill runtime never searches for a server);
    # old snapshots from before data-flow threading are refused too, each
    # with its reason, never a crash.
    unpinned = "name no server"
    current = sorted(r for r, why in refused if "/runs/" not in r)
    assert current == [
        "holdout/incident-triage/expected/pipeline.yaml",
        "invoice-push/expected/pipeline.yaml",
    ], refused
    assert len(planned) >= 15
    for rel, why in refused:
        assert "signature_spec" in why or "nothing to fan over" in why or unpinned in why, (
            rel,
            why,
        )


def test_the_plan_hash_follows_the_plan_bytes() -> None:
    pipeline = _example("docs-digest/compiled")
    text = plan_text(build_plan(pipeline, mcp="live"))
    assert plan_hash(text) == plan_hash(plan_text(build_plan(pipeline, mcp="live")))
    edited = pipeline.model_copy(deep=True)
    edited.node_by_id("pick_reading_list").constants = {"result_count": 3}
    assert plan_hash(plan_text(build_plan(edited, mcp="live"))) != plan_hash(text)


def _emitted_interpolate() -> Any:
    """The ``_interpolate`` a real emitted judge module defines."""
    namespace: dict[str, Any] = {}
    exec(compile(_emitted_judge_source(), "judge", "exec"), namespace)
    return namespace["_interpolate"]


def _emitted_judge_source() -> str:
    from rote.adapters._py_common import emit_signature_module
    from rote.ir import Node

    out = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
    node = Node.model_validate(
        {
            "id": "j",
            "kind": "llm_judge",
            "description": "d",
            "signature_spec": {
                "prompt": "p",
                "input_schema": {"type": "object", "properties": {}},
                "output_schema": out,
            },
        }
    )
    return emit_signature_module(
        node,
        anthropic_default_model="m",
        openai_default_model="m",
        generated_by="t",
        regen_command="t",
        context_note="t",
    )


def _values_for(references: list[str]) -> dict[str, Any]:
    """Input under which every reference resolves (a leaf another one reads into is a dict)."""
    values: dict[str, Any] = {}
    for ref in sorted(references, key=lambda r: -r.count(".")):
        *parents, leaf = ref.split(".")
        here = values
        for part in parents:
            here = here.setdefault(part, {})
        here.setdefault(leaf, "v")
    return values


@pytest.mark.parametrize(
    "prompt",
    [
        "Score {{ opportunity.arr }} against {{scoring_rubric}} for {{ opportunity.account_name }}",
        "{{a}}{{ a.b.c }} {{ a.b.c }}",
        "nothing here: {single}, {{ not-a-ref }}, {{ }}",
        "{{ 9lives }} and {{ _x }}",
    ],
)
def test_judge_references_are_exactly_what_the_emitted_judge_reads(prompt: str) -> None:
    """With every reference present the emitted ``_interpolate`` renders the whole
    prompt; with any one missing it raises. So the references are what it reads."""
    interpolate = _emitted_interpolate()
    references = judge_references(prompt)
    values = _values_for(references)
    rendered = interpolate(prompt, values)
    assert not PLACEHOLDER.search(rendered)
    for ref in references:
        top = ref.split(".")[0]
        with pytest.raises(KeyError):
            interpolate(prompt, {k: v for k, v in values.items() if k != top})
    # Text that is not a reference stays as written, on both sides.
    for literal in ("{single}", "{{ not-a-ref }}", "{{ }}"):
        if literal in prompt:
            assert literal in rendered


def test_every_placeholder_regex_comes_from_one_constant() -> None:
    from rote.adapters._common import PROMPT_PLACEHOLDER
    from rote.adapters._ts_common import _INTERPOLATE_HELPER

    assert PLACEHOLDER.pattern == PROMPT_PLACEHOLDER
    assert f"/{PROMPT_PLACEHOLDER}/g" in _INTERPOLATE_HELPER
    assert f're.sub(r"{PROMPT_PLACEHOLDER}", _resolve, template)' in _emitted_judge_source()
