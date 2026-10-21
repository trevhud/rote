"""The whole-skill structure pass: evidence checks, repair, and resolution.

No inference: every model answer is a fixture handed to ``structure.decide``
through its ``call`` parameter.
"""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from rote.compiler.system1 import structure
from rote.compiler.system1.parse import parse_skill
from rote.compiler.system1.skeleton import pipeline_doc
from rote.compiler.system1.tiebreak import Lane
from rote.compiler.tool_grounding import ToolSchema
from rote.ir import Pipeline

SKILL = """\
---
name: inbox-digest
description: Summarize yesterday's dock emails
---

# Inbox digest

## Step 1: Fetch the emails

Search Gmail for messages labeled dock-activity from the last 1 day.

## Step 2: Summarize

Write a three-line summary of each email. Flag totals over $2,500.

## Step 3: Deliver

Send the digest to the finance team.
"""

TOOLS = {
    ("gmail", "search_messages"): ToolSchema(
        server="gmail",
        tool="search_messages",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}, "max_results": {"type": "integer"}},
        },
        source="test",
        declared=True,
    )
}

NONE = {"io": "none", "how": "none", "name": "", "server": "", "tool": ""}
NO_GATE = {"signal": "", "payload": [], "timeout": ""}
NO_LOOP = {"tools": [], "stop_condition": "", "max_iterations": 0}


#: A verbatim passage under each heading of SKILL, by SECTIONS index.
QUOTES = {1: "Search Gmail for messages", 2: "Write a three-line summary", 3: "Send the digest"}


def node(nid: str, section: int, kind: str, **over: Any) -> dict[str, Any]:
    n: dict[str, Any] = {
        "id": nid,
        "section": section,
        "quote": QUOTES.get(section, ""),
        "kind": kind,
        "intent": f"{nid} does its work",
        "reads": ["pipeline.input"],
        "fan_out_over": "",
        "in_loop": "",
        "mandatory": True,
        "outputs": [],
        "system": dict(NONE),
        "args": [],
        "constants": [],
        "gate": dict(NO_GATE),
        "loop": dict(NO_LOOP),
        "confidence": 0.9,
        "why": "fixture",
    }
    n.update(over)
    return n


def good_answer() -> dict[str, Any]:
    return {
        "input_fields": [{"name": "day", "description": "the day to digest"}],
        "nodes": [
            node(
                "fetch_emails",
                1,
                "external_call",
                system={
                    "io": "read",
                    "how": "declared_tool",
                    "name": "gmail",
                    "server": "gmail",
                    "tool": "search_messages",
                },
                args=[{"param": "query", "source": "literal", "value": "label:dock-activity"}],
            ),
            node(
                "summarize_email",
                2,
                "llm_judge",
                reads=["fetch_emails"],
                fan_out_over="fetch_emails",
                constants=[{"name": "flag_over_usd", "value": "$2,500"}],
            ),
            node("render_digest", 3, "pure_function", reads=["summarize_email"]),
        ],
    }


@pytest.fixture
def skill(tmp_path: Path) -> Any:
    d = tmp_path / "skill"
    d.mkdir()
    (d / "SKILL.md").write_text(SKILL)
    return replace(parse_skill(d), units=structure.section_units(d))


class FakeCall:
    """Hands out ``answers`` in order and records each prompt."""

    def __init__(self, *answers: dict[str, Any]) -> None:
        self.answers = list(answers)
        self.prompts: list[str] = []

    def __call__(self, **kw: Any) -> tuple[dict[str, Any], list[tuple[str, dict[str, Any]]]]:
        self.prompts.append(kw["prompt"])
        assert kw["output_schema"] in (structure.SCHEMA, structure.REDECIDE_SCHEMA)
        assert kw["max_tokens"] == structure.MAX_TOKENS
        usage = {"input_tokens": 1000, "output_tokens": 100}
        usage |= {"cache_write_tokens": 0, "cache_read_tokens": 0, "cost_usd": 0.01}
        return copy.deepcopy(self.answers.pop(0)), [(kw["model"], usage)]


def run(skill: Any, *answers: dict[str, Any]) -> tuple[Any, Any, FakeCall]:
    fake = FakeCall(*answers)
    plan, rec = structure.decide(skill, SKILL, "rubric", Lane("claude-cli"), tools=TOOLS, call=fake)
    return plan, rec, fake


def test_a_valid_answer_takes_one_round_and_becomes_the_plan(skill: Any) -> None:
    plan, rec, fake = run(skill, good_answer())
    assert rec.rounds == 1 and rec.contradictions == [[]] and rec.resolved == []
    assert len(fake.prompts) == 1
    by_id = {n.id: n for n in plan.nodes}
    assert [n.id for n in plan.nodes] == ["fetch_emails", "summarize_email", "render_digest"]
    fetch = by_id["fetch_emails"]
    assert fetch.mcp == {
        "server": "gmail",
        "tool": "search_messages",
        "args": {"query": {"literal": "label:dock-activity"}},
    }
    judge = by_id["summarize_email"]
    assert judge.fan_out and judge.element_source == "fetch_emails"
    assert judge.inputs["fetch_emails"] == "fetch_emails.output.items"
    assert fetch.lists_output
    assert judge.constants == {"flag_over_usd": 2500}
    assert plan.input_fields == {"day": "the day to digest"}
    # A fanned node's whole result is described where its reader is written.
    assert by_id["render_digest"].intent == (
        "render_digest does its work `summarize_email` is a list with one result per "
        "element of `fetch_emails`, in the same order; a result does not carry its element."
    )
    Pipeline.model_validate(pipeline_doc(plan))


def test_the_prompt_carries_the_whole_skill_tools_and_steps(skill: Any) -> None:
    _, _, fake = run(skill, good_answer())
    prompt = fake.prompts[0]
    assert SKILL in prompt
    assert '"tool": "search_messages"' in prompt and '"name": "max_results"' in prompt
    assert '"heading": "Step 3: Deliver"' in prompt


def test_an_undeclared_tool_goes_back_for_repair(skill: Any) -> None:
    bad = good_answer()
    bad["nodes"][0]["system"]["tool"] = "gmail_search"
    plan, rec, fake = run(skill, bad, good_answer())
    assert rec.rounds == 2
    assert rec.contradictions[0] == [
        "node fetch_emails: tool 'gmail_search' is not a declared tool"
    ]
    assert rec.contradictions[1] == []
    assert "tool 'gmail_search' is not a declared tool" in fake.prompts[1]
    assert json.dumps(bad, indent=1) in fake.prompts[1]
    assert plan.nodes[0].mcp is not None and plan.nodes[0].mcp["tool"] == "search_messages"
    assert rec.first_answer == bad and rec.answer == good_answer()


def test_a_write_to_an_unidentified_system_ends_as_output(skill: Any) -> None:
    unspecified = {"io": "write", "how": "unspecified", "name": "finance", "server": "", "tool": ""}
    answer = good_answer()
    answer["nodes"].append(node("send_digest", 3, "external_call", system=unspecified))
    plan, rec, fake = run(skill, *[answer] * (structure.REPAIR_ROUNDS + 1))
    assert rec.rounds == structure.REPAIR_ROUNDS + 1 == len(fake.prompts)
    last = plan.nodes[-1]
    assert (last.id, last.kind, last.mcp) == ("send_digest", "pure_function", None)
    assert any("send_digest" in r and "No call was compiled" in r for r in rec.resolved)
    assert any(r.startswith("structure pass: `send_digest`") for r in plan.review)


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (
            lambda a: a["nodes"][1]["constants"].append({"name": "limit", "value": "40"}),
            "node summarize_email: constant 'limit' = '40' does not appear in the skill's text",
        ),
        (
            lambda a: a["nodes"][0]["args"].append(
                {"param": "label", "source": "literal", "value": "x"}
            ),
            "node fetch_emails: search_messages has no parameter 'label' "
            "(it takes ['max_results', 'query'])",
        ),
        (
            lambda a: a["nodes"][2].update(reads=["later_node"]),
            "node render_digest: reads 'later_node', which is not an earlier node",
        ),
        (
            lambda a: a["nodes"][2].update(
                system={"io": "write", "how": "none", "name": "", "server": "", "tool": ""}
            ),
            "node render_digest: a pure_function reaches no outside system (io and how must be "
            "none); a call is its own external_call node",
        ),
        (
            lambda a: a["nodes"].append(
                node("research", 2, "agent_loop", mandatory=True, reads=["fetch_emails"])
            ),
            "node research: an agent_loop cannot be mandatory (IR invariant)",
        ),
        (
            lambda a: a["nodes"].append(
                node(
                    "load_rates",
                    2,
                    "external_call",
                    system={
                        "io": "read",
                        "how": "local_path",
                        "name": "",
                        "server": "",
                        "tool": "",
                    },
                )
            ),
            "node load_rates: how is local_path, but its step gives no path outside the skill "
            "(references/ files are part of the skill: their tables are already code in "
            "extracted/rule_tables.py and their text is in the prompt)",
        ),
        (
            lambda a: a["nodes"][2].update(quote="Search Gmail for messages"),
            "node render_digest: its quote does not appear under the heading "
            "'Step 3: Deliver'; copy the words exactly",
        ),
        (
            lambda a: a["nodes"][2].update(id="request"),
            "node request: the id must be a snake_case identifier, not a reserved name",
        ),
    ],
)
def test_check_names_each_contradiction(skill: Any, change: Any, expected: str) -> None:
    answer = good_answer()
    assert structure.check(answer, skill, SKILL, structure.declared_tools(skill, TOOLS)) == []
    change(answer)
    assert structure.check(answer, skill, SKILL, structure.declared_tools(skill, TOOLS)) == [
        expected
    ]


def test_a_value_must_appear_verbatim() -> None:
    assert structure._appears("$2,500", "flag totals over **$2,500**.")
    assert not structure._appears("2500.00", "flag totals over $2,500.")
    assert structure._appears("label:dock-activity", "labeled `label:dock-activity` here")


def _skill_with(tmp_path: Path, text: str) -> Any:
    d = tmp_path / "skill2"
    d.mkdir()
    (d / "SKILL.md").write_text(text)
    return replace(parse_skill(d), units=structure.section_units(d))


def test_web_ui_evidence_is_under_the_nodes_own_section_or_an_open_session(
    tmp_path: Path,
) -> None:
    ui = {"io": "write", "how": "web_ui", "name": "ops", "server": "", "tool": ""}
    elsewhere = SKILL + "\n## Notes\n\nThe app is at https://ops.example.com/app.\n"
    sk = _skill_with(tmp_path, elsewhere)
    declared = structure.declared_tools(sk, TOOLS)
    answer = good_answer()
    answer["nodes"].append(node("click_send", 3, "external_call", system=ui))
    assert structure.check(answer, sk, elsewhere, declared) == [
        "node click_send: how is web_ui, but neither its section nor an earlier web_ui "
        "step's section gives the app's URL"
    ]
    opened = SKILL.replace(
        "Search Gmail for messages",
        "Open https://ops.example.com/app and search Gmail for messages",
    )
    (tmp_path / "b").mkdir()
    sk = _skill_with(tmp_path / "b", opened)
    answer = good_answer()
    answer["nodes"].insert(
        0,
        node("open_app", 1, "external_call", quote="Open https://ops.example.com/app", system=ui),
    )
    answer["nodes"].append(node("click_send", 3, "external_call", system=ui))
    assert structure.check(answer, sk, opened, structure.declared_tools(sk, TOOLS)) == []


def test_a_public_api_is_named_under_the_nodes_own_section(tmp_path: Path) -> None:
    api = {"io": "write", "how": "public_api", "name": "slack", "server": "", "tool": ""}
    text = SKILL + "\n## Notes\n\nThe team uses Slack.\n"
    sk = _skill_with(tmp_path, text)
    answer = good_answer()
    answer["nodes"].append(node("send_digest", 3, "external_call", system=api))
    assert structure.check(answer, sk, text, structure.declared_tools(sk, TOOLS)) == [
        "node send_digest: the product 'slack' is not named in its section"
    ]


def test_resolve_never_ships_what_check_rejects(skill: Any) -> None:
    """After failed repairs: no missing required arg, no unquoted node, sample gate keys."""
    from rote.compiler.grounding import Sample

    declared = structure.declared_tools(skill, REQUIRED)
    sample = Sample({"day": "x"}, "test", gates={"review": {"approved": True, "notes": ""}})
    answer = good_answer()
    answer["nodes"][0]["args"] = [{"param": "query", "source": "input_field", "value": "nope"}]
    answer["nodes"].append(node("invented", 2, "pure_function", quote="Not in the skill"))
    gate = {"signal": "digest_ok", "payload": ["approved"], "timeout": ""}
    answer["nodes"].append(node("review_gate", 3, "hitl_gate", gate=gate))
    fixed, notes = structure.resolve(answer, skill, SKILL, declared, sample)
    by_id = {n["id"]: n for n in fixed["nodes"]}
    assert by_id["fetch_emails"]["kind"] == "pure_function"
    assert any(n.startswith("`fetch_emails` would call search_messages") for n in notes)
    assert "invented" not in by_id
    assert any("dropped node 'invented'" in n for n in notes)
    assert sorted(by_id["review_gate"]["gate"]["payload"]) == ["approved", "notes"]
    two = Sample({"day": "x"}, "test", gates={"a": {"x": 1}, "b": {"y": 2}})
    with pytest.raises(structure.StructureUnavailable, match="answers 2 gate"):
        structure.resolve(answer, skill, SKILL, declared, two)


def test_unchecked_arguments_are_listed(skill: Any) -> None:
    unlisted = {("gmail", "search_messages"): None}
    fixed, notes = structure.resolve(good_answer(), skill, SKILL, unlisted)
    assert (
        "`fetch_emails`: the arguments to gmail.search_messages were not checked: no "
        "tools/list was reached for it, so rote has no schema"
    ) in notes


def test_a_reference_file_is_not_a_local_path() -> None:
    assert structure._outside_paths("see references/fx-rates.md") == []
    assert structure._outside_paths("save to ~/reports/out.csv") == ["~/reports/out.csv"]


def test_a_model_without_a_valid_skeleton_fails_loudly(skill: Any) -> None:
    broken = good_answer()
    broken["nodes"] = [node("x", 99, "pure_function")]
    with pytest.raises(structure.StructureUnavailable, match="no valid skeleton"):
        run(skill, *[broken] * (structure.REPAIR_ROUNDS + 1))


def test_lane_with_names_the_model_in_the_lanes_own_form() -> None:
    assert structure.lane_with(Lane("api"), "sonnet").model == "claude-sonnet-5-5"
    gw = Lane("api", None, "anthropic/claude-haiku-5.5")
    assert structure.lane_with(gw, "sonnet").model == "anthropic/claude-sonnet-5.5"


def test_every_heading_is_a_section_with_its_own_text(tmp_path: Path) -> None:
    d = tmp_path / "skill"
    d.mkdir()
    (d / "SKILL.md").write_text(
        SKILL.replace("Search Gmail", "(Gmail MCP - `search_messages`) Search Gmail")
    )
    units = structure.section_units(d)
    assert [u.title for u in units] == [
        "Inbox digest",
        "Step 1: Fetch the emails",
        "Step 2: Summarize",
        "Step 3: Deliver",
    ]
    assert units[3].text == "Send the digest to the finance team."
    assert units[1].mcp == [("gmail", "search_messages")]


def test_the_sample_decides_how_many_gates_and_their_keys(skill: Any) -> None:
    from rote.compiler.grounding import Sample

    sample = Sample({"day": "x"}, "test", gates={"review": {"approved": True, "notes": ""}})
    declared = structure.declared_tools(skill, TOOLS)
    answer = good_answer()
    assert structure.check(answer, skill, SKILL, declared, sample) == [
        "the sample input answers 1 gate(s), in order, with keys [['approved', 'notes']]; "
        "the skeleton has 0 gate(s). A gate is one pause the person answers once, with all "
        "of its keys"
    ]
    gate = {"signal": "digest_approved", "payload": ["approved"], "timeout": ""}
    answer["nodes"].append(node("review_gate", 3, "hitl_gate", gate=gate, reads=["render_digest"]))
    assert structure.check(answer, skill, SKILL, declared, sample) == [
        "node review_gate: the sample answers this gate with keys ['approved', 'notes'], "
        "not ['approved']"
    ]
    gate["payload"] = ["notes", "approved"]
    assert structure.check(answer, skill, SKILL, declared, sample) == []
    assert structure.check(answer, skill, SKILL, declared, None) == []


REQUIRED = {
    ("gmail", "search_messages"): ToolSchema(
        server="gmail",
        tool="search_messages",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}, "max_results": {"type": "integer"}},
            "required": ["query"],
        },
        source="test",
        declared=True,
    )
}


def test_an_argument_binds_from_an_input_field_or_an_upstream_field(skill: Any) -> None:
    from rote.compiler.grounding import Sample

    answer = good_answer()
    answer["nodes"][0]["args"] = [{"param": "query", "source": "input_field", "value": "day"}]
    answer["nodes"].append(
        node(
            "refine",
            3,
            "external_call",
            reads=["fetch_emails"],
            system=dict(answer["nodes"][0]["system"]),
            args=[
                {"param": "query", "source": "upstream_field", "value": "fetch_emails.next_query"}
            ],
        )
    )
    fake = FakeCall(answer)
    plan, rec = structure.decide(
        skill,
        SKILL,
        "rubric",
        Lane("claude-cli"),
        tools=REQUIRED,
        sample=Sample({"day": "2026-10-19"}, "test"),
        call=fake,
    )
    assert rec.contradictions == [[]]
    fetch, refine = plan.nodes[0], plan.nodes[-1]
    assert fetch.inputs["day"] == "pipeline.input.day"
    assert fetch.mcp == {"server": "gmail", "tool": "search_messages", "args": {"query": "day"}}
    assert refine.inputs["fetch_emails_next_query"] == "fetch_emails.output.next_query"
    assert refine.mcp is not None
    assert refine.mcp["args"] == {"query": "fetch_emails_next_query"}
    Pipeline.model_validate(pipeline_doc(plan))


def test_a_required_parameter_must_be_bound_from_a_real_source(skill: Any) -> None:
    from rote.compiler.grounding import Sample

    declared = structure.declared_tools(skill, REQUIRED)
    sample = Sample({"day": "x"}, "test")
    answer = good_answer()
    answer["nodes"][0]["args"] = []
    assert structure.check(answer, skill, SKILL, declared, sample) == [
        "node fetch_emails: search_messages requires ['query']; bind each one in args"
    ]
    answer["nodes"][0]["args"] = [{"param": "query", "source": "input_field", "value": "topic"}]
    assert structure.check(answer, skill, SKILL, declared, sample) == [
        "node fetch_emails: argument 'query' reads input field 'topic'; the input has ['day']"
    ]
    answer["nodes"][0]["args"] = [
        {"param": "query", "source": "upstream_field", "value": "render_digest.q"}
    ]
    assert structure.check(answer, skill, SKILL, declared, sample) == [
        "node fetch_emails: argument 'query' reads 'render_digest.q'; write "
        "'<node id>.<field>' for a node it reads"
    ]


def test_a_fanned_call_binds_the_element_and_tells_the_builder(skill: Any) -> None:
    answer = good_answer()
    answer["nodes"][0] = node(
        "build_queries",
        1,
        "pure_function",
        intent="Build one query per label.",
    )
    answer["nodes"].insert(
        1,
        node(
            "fetch_emails",
            1,
            "external_call",
            reads=["build_queries"],
            fan_out_over="build_queries",
            system={
                "io": "read",
                "how": "declared_tool",
                "name": "gmail",
                "server": "gmail",
                "tool": "search_messages",
            },
            args=[{"param": "query", "source": "fan_element", "value": "build_queries"}],
        ),
    )
    plan, rec, _ = run(skill, answer)
    assert rec.contradictions == [[]]
    by_id = {n.id: n for n in plan.nodes}
    fetch = by_id["fetch_emails"]
    assert fetch.mcp is not None and fetch.mcp["args"] == {"query": "build_queries"}
    assert fetch.inputs["build_queries"] == "build_queries.output.items"
    assert by_id["build_queries"].intent == (
        "Build one query per label. Its `items` are the `query` values search_messages "
        "is called with, one string per element."
    )
    assert by_id["build_queries"].output_schema == {
        "type": "object",
        "properties": {"items": {"type": "array", "items": {"type": "string"}}},
        "required": ["items"],
    }
    assert pipeline_doc(plan)["nodes"][0]["output_schema"] == by_id["build_queries"].output_schema
    wrong = copy.deepcopy(answer)
    wrong["nodes"][1]["fan_out_over"] = ""
    declared = structure.declared_tools(skill, TOOLS)
    assert (
        "node fetch_emails: argument 'query' binds a fan_element of 'build_queries', but the "
        "node fans out over ''"
    ) in structure.check(wrong, skill, SKILL, declared)


def test_a_fan_over_a_fanned_node_reads_its_whole_result(skill: Any) -> None:
    """A fan_out node's result is the list of per-element results, with no `items` key.

    The MCP demo's risk judge fanned over the per-product search and read
    `search_docs.output.items`, which failed every live run with a TypeError."""
    answer = good_answer()
    answer["nodes"][2] = node(
        "rate_email",
        3,
        "llm_judge",
        reads=["summarize_email"],
        fan_out_over="summarize_email",
    )
    plan, rec, _ = run(skill, answer)
    assert rec.contradictions == [[]]
    by_id = {n.id: n for n in plan.nodes}
    assert by_id["rate_email"].inputs["summarize_email"] == "summarize_email.output"
    assert not by_id["summarize_email"].lists_output
    doc = pipeline_doc(plan)
    rate = next(n for n in doc["nodes"] if n["id"] == "rate_email")
    assert rate["signature_spec"]["input_schema"]["properties"]["summarize_email"] == {
        "description": "one element of summarize_email.output"
    }
    Pipeline.model_validate(doc)


def test_a_nodes_named_outputs_reach_the_skeleton(skill: Any) -> None:
    answer = good_answer()
    answer["nodes"][2]["outputs"] = ["digest", "missing_limits_page"]
    answer["nodes"][0]["outputs"] = ["threads"]
    plan, rec, _ = run(skill, answer)
    doc = {n["id"]: n for n in pipeline_doc(plan)["nodes"]}
    assert doc["render_digest"]["output"] == {"digest": "Any", "missing_limits_page": "Any"}
    # A fan_out source keeps its `items` list beside the keys it names.
    assert doc["fetch_emails"]["output"] == {"items": "list", "threads": "Any"}
    answer["nodes"][2]["outputs"] = ["Bad Key"]
    assert structure.check(answer, skill, SKILL, structure.declared_tools(skill, TOOLS)) == [
        "node render_digest: output keys ['Bad Key'] are not snake_case identifiers"
    ]


def test_redecide_rebinds_only_the_named_node_and_only_with_evidence(
    skill: Any, tmp_path: Path
) -> None:
    from rote.compiler.grounding import Sample
    from rote.compiler.system1.skeleton import write

    sample = Sample({"day": "2026-10-19"}, "test")
    plan, _, _ = run(skill, good_answer())
    plan.sample = sample
    work = tmp_path / "work"
    work.mkdir()
    write(plan, work)
    path = work / "system1" / "skeleton.yaml"
    before = path.read_text()
    findings = {"fetch_emails": ["search_messages requires ['query']"]}

    wrong = {"nodes": [{"id": "fetch_emails", "server": "gmail", "tool": "nope", "args": []}]}
    changes, usage = structure.redecide(
        work, findings, skill, SKILL, Lane("claude-cli"), REQUIRED, sample, call=FakeCall(wrong)
    )
    assert changes == [
        "`fetch_emails`: the new binding was not applied: node fetch_emails: tool 'nope' is "
        "not a declared tool"
    ]
    assert path.read_text() == before and len(usage) == 1

    right = {
        "nodes": [
            {
                "id": "fetch_emails",
                "server": "gmail",
                "tool": "search_messages",
                "args": [{"param": "query", "source": "input_field", "value": "day"}],
            },
            {"id": "render_digest", "server": "gmail", "tool": "search_messages", "args": []},
        ]
    }
    changes, _ = structure.redecide(
        work, findings, skill, SKILL, Lane("claude-cli"), REQUIRED, sample, call=FakeCall(right)
    )
    assert changes == [
        "`fetch_emails`: re-bound to gmail.search_messages with args {'query': 'day'}"
    ]
    doc = {n["id"]: n for n in yaml.safe_load(path.read_text())["nodes"]}
    assert doc["fetch_emails"]["mcp"] == {
        "server": "gmail",
        "tool": "search_messages",
        "args": {"query": "day"},
    }
    assert doc["fetch_emails"]["inputs"]["day"] == "pipeline.input.day"
    assert "mcp" not in doc["render_digest"]


def test_binding_findings_are_the_structure_passs_to_fix() -> None:
    from types import SimpleNamespace

    from rote.compiler.system1.generate import binding_findings
    from rote.contracts import ContractFinding

    def finding(node: str, kind: str) -> ContractFinding:
        return ContractFinding(node_id=node, kind=kind, severity="error", message=f"{node} {kind}")

    verification = SimpleNamespace(
        errors=[
            finding("search", "tool_arguments_invalid"),
            finding("search", "tool_unknown"),
            finding("summarize", "tool_arguments_invalid"),
            finding("search", "untested_implementation"),
        ]
    )
    assert binding_findings(verification, ["summarize"]) == {
        "search": ["search tool_arguments_invalid", "search tool_unknown"]
    }


def test_every_run_output_reaches_an_exit_node(skill: Any) -> None:
    declared = structure.declared_tools(skill, TOOLS)
    answer = good_answer()
    answer["run_outputs"] = ["digest", "missing_limits_page"]
    answer["nodes"][1]["outputs"] = ["missing_limits_page"]
    answer["nodes"][2]["outputs"] = ["digest"]
    assert structure.check(answer, skill, SKILL, declared) == [
        "run output 'missing_limits_page' is not returned by an exit node (exit nodes: "
        "['render_digest']); add it to an exit node's outputs, or add a final node that reads "
        "its producer and returns it"
    ]
    answer["nodes"][2]["outputs"] = ["digest", "missing_limits_page"]
    assert structure.check(answer, skill, SKILL, declared) == []
