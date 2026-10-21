"""Grounded contracts: a compile's input, row keys and gate payloads come from data.

Covers :mod:`rote.compiler.grounding` (sample discovery, the extracted
contract, the static checks, open questions), :mod:`rote.compiler.prefix`
(the deterministic prefix run on the sample), and their wiring into
verification, the compiler and the bench. No inference anywhere: the
deliverables are written by hand, the way a compile would leave them.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from rote.compiler import Compiler
from rote.compiler.drivers import CompilerDriver, DriverResult
from rote.compiler.grounding import (
    GROUNDING_DIRNAME,
    Grounding,
    Sample,
    contract_brief,
    contract_questions,
    find_sample,
    gate_findings,
    input_contract,
    input_findings,
    load_sample,
    named_human_items,
    reference_input_findings,
    schema_mismatches,
)
from rote.compiler.prefix import run_prefix
from rote.compiler.verify import verify_deliverable
from rote.contracts import implementation_status
from rote.eval.baseline import ObservedToolCall
from rote.ir import Pipeline, load_pipeline

REPO = Path(__file__).resolve().parent.parent
EXPENSE = REPO / "examples" / "expense-audit"
OPS = REPO / "examples" / "ops-report"


def _sample(payload: dict, gates: dict[str, dict] | None = None) -> Sample:
    return Sample(payload=payload, source="sample.json", gates=gates or {})


def _write(root: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(textwrap.dedent(text), encoding="utf-8")
    return root


def _pipeline(root: Path) -> Pipeline:
    return load_pipeline(root / "pipeline.yaml")


# ───────── finding the sample ─────────


def test_a_sample_bundled_in_the_skill_dir_is_found(tmp_path: Path) -> None:
    skill = _write(tmp_path / "skill", {"SKILL.md": "# s\n", "examples/input.json": '{"a": 1}'})
    sample = find_sample(skill)
    assert sample is not None
    assert sample.payload == {"a": 1}
    assert sample.source.endswith("examples/input.json")


def test_an_explicit_sample_beats_the_bundled_one(tmp_path: Path) -> None:
    skill = _write(tmp_path / "skill", {"SKILL.md": "# s\n", "input.json": '{"bundled": 1}'})
    explicit = _write(tmp_path, {"mine.json": '{"explicit": 1}'}) / "mine.json"
    assert find_sample(skill, explicit).payload == {"explicit": 1}  # type: ignore[union-attr]


def test_a_missing_explicit_sample_is_an_error_not_a_fallback(tmp_path: Path) -> None:
    skill = _write(tmp_path / "skill", {"SKILL.md": "# s\n", "input.json": '{"bundled": 1}'})
    with pytest.raises(ValueError, match="not found"):
        find_sample(skill, tmp_path / "nope.json")


def test_a_bundled_file_that_is_not_an_object_is_skipped(tmp_path: Path) -> None:
    skill = _write(
        tmp_path / "skill",
        {"SKILL.md": "# s\n", "input.json": "[1, 2]", "fixtures/input.json": '{"b": 2}'},
    )
    assert find_sample(skill).payload == {"b": 2}  # type: ignore[union-attr]


def test_the_baseline_input_is_the_last_source(tmp_path: Path) -> None:
    skill = _write(tmp_path / "skill", {"SKILL.md": "# s\n"})
    probe = _write(tmp_path / "baseline", {"metrics.json": json.dumps({"input": {"d": "x"}})})
    assert find_sample(skill) is None
    sample = find_sample(skill, probe_dir=probe)
    assert sample is not None and sample.payload == {"d": "x"}


def test_the_eval_envelope_grounds_the_gate_payloads_too() -> None:
    sample = load_sample(EXPENSE / "eval-input.json")
    assert set(sample.payload) == {"period", "submissions"}
    assert {s: sorted(g) for s, g in sample.gates.items()} == {
        "controller_review": ["approved_ids", "rejected_ids", "reviewer"]
    }


def test_an_input_field_named_input_is_not_read_as_an_envelope() -> None:
    sample = Sample(payload={}, source="x")
    from rote.compiler.grounding import parse_sample

    sample = parse_sample({"input": "free text", "mode": "fast"}, "x")
    assert sample.payload == {"input": "free text", "mode": "fast"}
    assert sample.gates == {}


# ───────── the contract ─────────


def test_the_contract_carries_row_keys_and_which_are_optional() -> None:
    schema = input_contract(load_sample(EXPENSE / "input.json"))
    assert sorted(schema["required"]) == ["period", "submissions"]
    rows = schema["properties"]["submissions"]["items"]
    assert "id" in rows["required"] and "expense_id" not in rows["properties"]
    # Only lodging lines carry nights: present in some rows, so optional.
    assert "nights" in rows["properties"] and "nights" not in rows["required"]


def test_the_brief_names_the_sample_fields_and_the_bundled_files(tmp_path: Path) -> None:
    sample = load_sample(EXPENSE / "input.json")
    brief = contract_brief(sample, EXPENSE / "skill", tmp_path / GROUNDING_DIRNAME)
    assert "`submissions` (list of objects" in brief
    assert "  - `id` (string)" in brief
    assert "  - `nights` (integer, optional)" in brief
    assert "`references/fx-rates.md`" in brief
    assert brief.count("GATE PAYLOAD") == 0  # input.json is not an envelope


def test_the_brief_without_a_sample_still_names_the_skills_gate_nouns() -> None:
    brief = contract_brief(None, OPS / "skill", None)
    assert "INPUT CONTRACT" not in brief
    assert "**Dock Appointment Completion** -> `dock_appointment_completion`" in brief
    assert "`current_dwell`" in brief


# ───────── static checks ─────────

_EXPENSE_IR = """\
name: expense
input:
  type: Batch
  required: [{required}]
  input_schema:
    type: object
    required: [{required}]
    properties: {{{props}}}
nodes:
  - id: dupes
    kind: pure_function
    description: find duplicates
    impl: extracted/checks.py:dupes
    inputs:
{inputs}
edges: []
entry_nodes: [dupes]
exit_nodes: [dupes]
"""


def _expense_ir(fields: list[str]) -> str:
    return _EXPENSE_IR.format(
        required=", ".join(fields),
        props=", ".join(f"{f}: {{}}" for f in fields),
        inputs="\n".join(f"      {f}: pipeline.input.{f}" for f in fields),
    )


def test_an_invented_input_name_fails_naming_the_samples_fields(tmp_path: Path) -> None:
    root = _write(tmp_path, {"pipeline.yaml": _expense_ir(["expense_lines", "reporting_period"])})
    [missing, schema] = input_findings(_pipeline(root), load_sample(EXPENSE / "input.json"))
    assert missing.kind == "input_not_in_sample"
    assert "expense_lines, reporting_period" in missing.message
    assert "it has period, submissions" in missing.message
    assert schema.kind == "sample_rejected_by_schema"


def test_the_samples_own_names_pass(tmp_path: Path) -> None:
    root = _write(tmp_path, {"pipeline.yaml": _expense_ir(["period", "submissions"])})
    assert input_findings(_pipeline(root), load_sample(EXPENSE / "input.json")) == []


def test_a_schema_row_key_the_sample_lacks_is_named_with_its_row() -> None:
    schema = {
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "items": {"type": "object", "required": ["expense_id"]},
            }
        },
    }
    [problem] = schema_mismatches(schema, {"rows": [{"id": "a"}, {"id": "b"}]})
    assert problem == "input.rows[0] has no `expense_id` (has: id)"
    assert schema_mismatches(schema, {"rows": [{"expense_id": "a"}]}) == []


def test_a_bundled_reference_cannot_be_a_required_input(tmp_path: Path) -> None:
    root = _write(tmp_path, {"pipeline.yaml": _expense_ir(["period", "fx_rates_markdown"])})
    [finding] = reference_input_findings(_pipeline(root), EXPENSE / "skill")
    assert finding.kind == "reference_as_input"
    assert "`fx_rates_markdown`" in finding.message and "references/fx-rates.md" in finding.message
    clean = _write(tmp_path / "ok", {"pipeline.yaml": _expense_ir(["period", "submissions"])})
    assert reference_input_findings(_pipeline(clean), EXPENSE / "skill") == []


_GATE_IR = """\
name: ops
input:
  type: Day
  required: [report_date]
nodes:
  - id: ask
    kind: hitl_gate
    description: ask the duty manager
    source:
      section: Manual data to request from user
    output: {output}
    signal: manual_data
edges: []
entry_nodes: [ask]
exit_nodes: [ask]
"""


def test_the_skill_names_what_the_person_provides() -> None:
    [(heading, items)] = named_human_items(OPS / "skill").items()
    assert heading == "Manual data to request from user:"
    assert [k for _, k in items] == [
        "dock_appointment_completion",
        "sorter_metrics",
        "current_dwell",
    ]


def test_a_paraphrased_gate_key_fails_with_the_skills_noun(tmp_path: Path) -> None:
    output = "{dock_appointments: dict, sorter_metrics: list, current_dwell: list}"
    root = _write(tmp_path, {"pipeline.yaml": _GATE_IR.format(output=output)})
    [finding] = gate_findings(_pipeline(root), OPS / "skill", None)
    assert finding.kind == "gate_payload_not_skill_nouns"
    assert "**Dock Appointment Completion** -> `dock_appointment_completion`" in finding.message
    assert "Sorter Metrics" not in finding.message  # that key is right


def test_a_gate_payload_with_no_declared_keys_fails(tmp_path: Path) -> None:
    root = _write(tmp_path, {"pipeline.yaml": _GATE_IR.format(output="ManualData")})
    [finding] = gate_findings(_pipeline(root), OPS / "skill", None)
    assert "payload keys (none)" in finding.message


def test_the_skills_nouns_pass(tmp_path: Path) -> None:
    output = "{dock_appointment_completion: dict, sorter_metrics: list, current_dwell: list}"
    root = _write(tmp_path, {"pipeline.yaml": _GATE_IR.format(output=output)})
    assert gate_findings(_pipeline(root), OPS / "skill", None) == []


def test_a_sample_gate_payload_fixes_the_gate_keys(tmp_path: Path) -> None:
    root = _write(tmp_path, {"pipeline.yaml": _GATE_IR.format(output="{decisions: dict}")})
    sample = _sample(
        {"report_date": "x"}, gates={"sign_off": {"reviewer": "a", "approved_ids": []}}
    )
    [finding] = gate_findings(_pipeline(root), None, sample)
    assert finding.kind == "gate_payload_not_sample"
    assert "approved_ids, reviewer" in finding.message
    good = _GATE_IR.format(output="{reviewer: str, approved_ids: list}")
    root = _write(tmp_path / "ok", {"pipeline.yaml": good})
    assert gate_findings(_pipeline(root), None, sample) == []


_TWO_GATES_IR = """\
name: two
input:
  type: In
nodes:
  - id: first
    kind: hitl_gate
    description: an earlier check-in the sample has no answer for
    output: {{acknowledged: bool}}
    signal: check_in
  - id: second
    kind: hitl_gate
    description: the review the sample answers
    output: {output}
    signal: {signal}
edges:
  - {{from: first, to: second}}
entry_nodes: [first]
exit_nodes: [second]
"""


@pytest.mark.parametrize("signal", ["controller_review", "named_otherwise"])
def test_a_sample_gate_finds_its_gate_by_signal_then_by_keys_not_by_order(
    tmp_path: Path, signal: str
) -> None:
    """An extra earlier gate must not shift the sample's answer onto it."""
    sample = _sample({}, gates={"controller_review": {"reviewer": "a", "approved_ids": []}})
    good = _TWO_GATES_IR.format(output="{reviewer: str, approved_ids: list}", signal=signal)
    root = _write(tmp_path, {"pipeline.yaml": good})
    assert gate_findings(_pipeline(root), None, sample) == []
    # A real mismatch on the matched gate still fails, and names that gate.
    bad = _TWO_GATES_IR.format(output="{reviewer: str}", signal="controller_review")
    root = _write(tmp_path / "bad", {"pipeline.yaml": bad})
    [finding] = gate_findings(_pipeline(root), None, sample)
    assert finding.node_id == "second" and "Declare approved_ids" in finding.message


def test_the_signal_name_beats_a_bigger_key_overlap(tmp_path: Path) -> None:
    """The gate named by the sample's signal is the one checked, even when
    another gate happens to declare more of the sample's keys."""
    ir = """\
name: two
input:
  type: In
nodes:
  - id: decoy
    kind: hitl_gate
    description: another gate with overlapping keys
    output: {reviewer: str, approved_ids: list}
    signal: something_else
  - id: review
    kind: hitl_gate
    description: the gate the sample answers
    output: {reviewer: str}
    signal: controller_review
edges:
  - {from: decoy, to: review}
entry_nodes: [decoy]
exit_nodes: [review]
"""
    root = _write(tmp_path, {"pipeline.yaml": ir})
    sample = _sample({}, gates={"controller_review": {"reviewer": "a", "approved_ids": []}})
    [finding] = gate_findings(_pipeline(root), None, sample)
    assert finding.node_id == "review" and "Declare approved_ids" in finding.message


def test_bold_choice_options_are_not_payload_keys(tmp_path: Path) -> None:
    skill = _write(
        tmp_path / "skill",
        {
            "SKILL.md": (
                "# s\n\n## Review\n\nAsk the lead to decide each vendor:\n\n"
                "- **Approve**: keep paying\n- **Reject**: cancel it\n"
                "- **Renewal Date**: when it renews\n"
            )
        },
    )
    [(_, items)] = named_human_items(skill).items()
    assert [k for _, k in items] == ["renewal_date"]


def test_a_one_word_reference_flags_only_its_own_name(tmp_path: Path) -> None:
    skill = _write(tmp_path / "skill", {"SKILL.md": "# s\n", "references/policy.md": "x"})
    root = _write(tmp_path, {"pipeline.yaml": _expense_ir(["policy_id", "period"])})
    assert reference_input_findings(_pipeline(root), skill) == []
    root = _write(tmp_path / "bad", {"pipeline.yaml": _expense_ir(["policy", "period"])})
    [finding] = reference_input_findings(_pipeline(root), skill)
    assert "`policy`" in finding.message


@pytest.mark.parametrize("name", ['x"); import os  #', "a b", "1st", "field.sub", ""])
def test_a_hostile_input_field_name_is_rejected_at_the_ir_boundary(name: str) -> None:
    from pydantic import ValidationError

    from rote.ir import PipelineInput

    with pytest.raises(ValidationError, match="must be an identifier"):
        PipelineInput(type="T", required=[name])
    with pytest.raises(ValidationError, match="must be an identifier"):
        PipelineInput(type="T", optional=[name])
    PipelineInput(type="T", required=["period"], optional=["_private2"])


def test_verification_reports_the_sample_mismatch(tmp_path: Path) -> None:
    """Through grounding_findings and verify_deliverable, not the helper alone."""
    from rote.compiler.grounding import grounding_findings

    root = _write(tmp_path, {"pipeline.yaml": _expense_ir(["expense_lines", "period"])})
    sample = load_sample(EXPENSE / "input.json")
    kinds = {f.kind for f in grounding_findings(_pipeline(root), None, sample)}
    assert {"input_not_in_sample", "sample_rejected_by_schema"} <= kinds
    v = verify_deliverable(root, grounding=Grounding(sample=sample), run_tests=False)
    assert "input_not_in_sample" in {f.kind for f in v.errors}


# ───────── open questions ─────────

_MCP_IR = """\
name: ops
input:
  type: Day
  required: [report_date]
nodes:
  - id: search
    kind: external_call
    description: search mail
    impl: extracted/mail.py:search
    mcp: {server: gmail, tool: search_threads}
    inputs:
      query: pipeline.input.report_date
  - id: count
    kind: pure_function
    description: count threads
    impl: extracted/mail.py:count
    inputs:
      threads: search.output.threads
edges:
  - {from: search, to: count}
entry_nodes: [search]
exit_nodes: [count]
"""


def test_without_a_sample_the_chosen_contract_is_listed_with_its_wording(tmp_path: Path) -> None:
    root = _write(tmp_path, {"pipeline.yaml": _expense_ir(["expense_lines", "reporting_period"])})
    [question] = contract_questions(_pipeline(root), EXPENSE / "skill", Grounding())
    assert "`expense_lines`, `reporting_period`" in question.text
    assert "The input is the month's export: one row per expense line" in question.text


def test_an_unrecorded_tool_result_shape_is_listed_not_guessed(tmp_path: Path) -> None:
    root = _write(tmp_path, {"pipeline.yaml": _MCP_IR})
    grounding = Grounding(sample=_sample({"report_date": "d"}))
    [question] = contract_questions(_pipeline(root), None, grounding)
    assert question.node_id == "search"
    assert "gmail.search_threads" in question.text and "key `threads`" in question.text
    recorded = Grounding(
        sample=grounding.sample,
        observations=(ObservedToolCall("gmail", "search_threads", {"query": "d"}, "[]"),),
    )
    assert contract_questions(_pipeline(root), None, recorded) == []


# ───────── the deterministic prefix ─────────

_PREFIX_IR = """\
name: expense
input:
  type: Batch
  required: [period, submissions]
nodes:
  - id: dupes
    kind: pure_function
    description: find duplicates
    impl: extracted/checks.py:dupes
    inputs:
      submissions: pipeline.input.submissions
  - id: per_line
    kind: pure_function
    description: one result per line
    impl: extracted/checks.py:per_line
    fan_out: true
    inputs:
      line: pipeline.input.submissions
    constants:
      cap: 25
  - id: summary
    kind: pure_function
    description: summarize the duplicates
    impl: extracted/checks.py:summary
    inputs:
      flagged: dupes.output.flagged
      cap: dupes.output.cap
  - id: review
    kind: hitl_gate
    description: controller review
    signal: controller_review
    inputs:
      flagged: dupes.output.flagged
  - id: after_gate
    kind: pure_function
    description: needs the gate
    impl: extracted/checks.py:after
    inputs:
      decision: review.output
edges:
  - {from: dupes, to: summary}
  - {from: dupes, to: review}
  - {from: review, to: after_gate}
entry_nodes: [dupes, per_line]
exit_nodes: [after_gate, per_line, summary]
"""

_CHECKS = """\
def dupes(submissions):
    seen, flagged = set(), []
    for line in submissions:
        key = (line["employee"], line["amount"])
        if key in seen:
            flagged.append(line["{row_key}"])
        seen.add(key)
    return {{"flagged": flagged{cap}}}


def per_line(line, cap):
    return {{"id": line["id"], "over": line["amount"] > cap}}


def summary(flagged, cap):
    return {{"count": len(flagged), "cap": cap}}


def after(decision):
    return decision
"""

_TESTS = """\
from extracted.checks import after, dupes, per_line, summary


def test_names():
    assert callable(dupes) and callable(per_line) and callable(summary) and callable(after)
"""

_ROWS = {
    "period": "2026-09",
    "submissions": [
        {"id": "E1", "employee": "Ana", "amount": 10},
        {"id": "E2", "employee": "Ana", "amount": 10},
        {"id": "E3", "employee": "Ben", "amount": 50},
    ],
}


def _prefix_root(tmp_path: Path, *, row_key: str = "id", cap: bool = True) -> Path:
    checks = _CHECKS.format(row_key=row_key, cap=', "cap": 25' if cap else "")
    return _write(
        tmp_path,
        {
            "pipeline.yaml": _PREFIX_IR,
            "extracted/checks.py": checks,
            "tests/test_checks.py": _TESTS,
        },
    )


def _run(root: Path, payload: dict | None = None):
    pipeline = _pipeline(root)
    return run_prefix(
        pipeline,
        root,
        _sample(payload or _ROWS),
        implementation_status(pipeline, root),
        trust_skill=True,
    )


def test_a_row_key_the_sample_does_not_have_fails_at_its_line(tmp_path: Path) -> None:
    run = _run(_prefix_root(tmp_path, row_key="expense_id"))
    [finding] = run.findings
    assert finding.kind == "prefix_run_failed" and finding.node_id == "dupes"
    assert "KeyError: 'expense_id'" in finding.message
    assert 'at extracted/checks.py:6 `flagged.append(line["expense_id"])`' in finding.message
    assert run.steps["summary"]["status"] == "not_reached"


def test_the_prefix_runs_up_to_the_gate_and_no_further(tmp_path: Path) -> None:
    run = _run(_prefix_root(tmp_path))
    assert run.findings == [] and run.status == "ran"
    assert run.steps["dupes"]["status"] == "ok"
    # fan_out ran per element; a gate ends the prefix on its branch.
    assert run.steps["per_line"]["status"] == "ok"
    assert run.steps["review"]["status"] == "not_deterministic"
    assert run.steps["summary"]["status"] == "ok"
    assert run.steps["after_gate"] == {"status": "not_reached", "upstream": ["review"]}
    assert (
        run.summary == "3 of 5 node(s) ran on the sample input (1 not deterministic, 1 not reached)"
    )


def test_a_binding_the_upstream_result_cannot_satisfy_names_both_sides(tmp_path: Path) -> None:
    run = _run(_prefix_root(tmp_path, cap=False))
    [finding] = run.findings
    assert finding.kind == "prefix_binding_failed" and finding.node_id == "summary"
    assert "`cap` is bound to dupes.output.cap, but dupes returned keys flagged" in finding.message


def test_a_result_that_is_not_plain_json_fails_by_path(tmp_path: Path) -> None:
    """A real compile's duplicate check returned a Python set: it pickles in
    DBOS and silently becomes its repr anywhere JSON carries it."""
    root = _prefix_root(tmp_path)
    checks = root / "extracted" / "checks.py"
    checks.write_text(
        checks.read_text().replace(
            'return {"flagged": flagged, "cap": 25}', 'return {"flagged": set(flagged), "cap": 25}'
        )
    )
    run = _run(root)
    [finding] = run.findings
    assert finding.kind == "prefix_not_json" and finding.node_id == "dupes"
    assert "result.flagged is a set" in finding.message
    # Nothing downstream runs on a result no runtime could carry.
    assert run.steps["summary"]["status"] == "not_reached"


def test_a_non_string_key_is_not_plain_json(tmp_path: Path) -> None:
    """JSON turns 1 into "1": the result a later step reads is not the one returned."""
    root = _prefix_root(tmp_path)
    checks = root / "extracted" / "checks.py"
    checks.write_text(
        checks.read_text().replace(
            'return {"flagged": flagged, "cap": 25}', 'return {"flagged": {1: flagged}, "cap": 25}'
        )
    )
    [finding] = _run(root).findings
    assert finding.kind == "prefix_not_json" and "result.flagged has a int key 1" in finding.message


def test_a_fanned_input_that_is_not_a_list_fails(tmp_path: Path) -> None:
    run = _run(_prefix_root(tmp_path), {**_ROWS, "submissions": {"E1": {}}})
    kinds = {f.node_id: f.kind for f in run.findings}
    assert kinds["per_line"] == "prefix_binding_failed"
    assert "expected a list, got dict" in run.steps["per_line"]["message"]


def test_a_recorded_tool_result_stands_in_for_the_call(tmp_path: Path) -> None:
    root = _write(
        tmp_path,
        {
            "pipeline.yaml": _MCP_IR,
            "extracted/mail.py": "def search(query):\n    raise NotImplementedError('mcp')\n\n"
            "def count(threads):\n    return {'n': len(threads)}\n",
        },
    )
    pipeline = _pipeline(root)
    impls = implementation_status(pipeline, root)
    observed = ObservedToolCall(
        "gmail", "search_threads", {"query": "2026-10-19"}, [{"type": "text", "text": "[1, 2]"}]
    )
    sample = _sample({"report_date": "2026-10-19"})
    run = run_prefix(pipeline, root, sample, impls, observations=(observed,), trust_skill=True)
    # The real Gmail result was a bare list, so `.output.threads` cannot bind.
    assert run.steps["search"] == {
        "status": "ok",
        "recorded": True,
        "arguments": {"query": "2026-10-19"},
    }
    [finding] = run.findings
    assert "bound to search.output.threads, but search returned a list" in finding.message
    # A call with other arguments was never recorded: nothing is guessed.
    other = run_prefix(
        pipeline,
        root,
        _sample({"report_date": "x"}),
        impls,
        observations=(observed,),
        trust_skill=True,
    )
    assert other.steps["search"]["status"] == "no_recording" and other.findings == []


@pytest.mark.parametrize(
    ("count_body", "flagged"),
    [
        # 7 of 9 real ops-report compiles: a dict assumed, a list returned.
        ("return {'n': len(found.get('threads', []))}", "a JSON list"),
        ("return {'n': len(found['threads'])}", "a JSON list"),
        # A list handled, text not: Drive's CSV reaches the code as a string.
        (
            "threads = found if isinstance(found, list) else found.get('threads', [])\n"
            "    return {'n': len(threads)}",
            "text",
        ),
        # Every shape: nothing to report.
        (
            "if isinstance(found, str):\n        found = []\n"
            "    threads = found if isinstance(found, list) else found.get('threads', [])\n"
            "    return {'n': len(threads)}",
            None,
        ),
    ],
)
def test_a_consumer_that_assumes_a_dict_is_probed_with_a_list(
    count_body: str, flagged: str | None, tmp_path: Path
) -> None:
    ir = _MCP_IR.replace("threads: search.output.threads", "found: search.output")
    root = _write(
        tmp_path,
        {
            "pipeline.yaml": ir,
            "extracted/mail.py": "def search(query):\n    raise NotImplementedError('mcp')\n\n"
            f"def count(found):\n    {count_body}\n",
        },
    )
    pipeline = _pipeline(root)
    run = run_prefix(
        pipeline,
        root,
        _sample({"report_date": "2026-10-19"}),
        implementation_status(pipeline, root),
        trust_skill=True,
    )
    assert run.steps["search"]["status"] == "no_recording"
    kinds = [f.kind for f in run.findings]
    if flagged:
        assert kinds == ["prefix_shape_assumption"]
        assert f"count crashes when search returns {flagged}:" in run.findings[0].message
    else:
        assert kinds == [] and run.steps["count"]["status"] == "not_reached"


def _probe_run(tmp_path: Path, count_body: str, output_schema: dict | None = None):
    from rote.compiler.tool_grounding import ToolSchema

    ir = _MCP_IR.replace("threads: search.output.threads", "found: search.output")
    root = _write(
        tmp_path,
        {
            "pipeline.yaml": ir,
            "extracted/mail.py": "def search(query):\n    raise NotImplementedError('mcp')\n\n"
            f"def count(found):\n    {count_body}\n",
        },
    )
    pipeline = _pipeline(root)
    tools = {
        ("gmail", "search_threads"): ToolSchema(
            "gmail",
            "search_threads",
            {"type": "object", "properties": {"query": {"type": "string"}}},
            "test",
            True,
            output_schema,
        )
    }
    return run_prefix(
        pipeline,
        root,
        _sample({"report_date": "2026-10-19"}),
        implementation_status(pipeline, root),
        trust_skill=True,
        tools=tools,
    )


def test_a_declared_output_shape_is_the_only_one_probed(tmp_path: Path) -> None:
    """A list-only consumer is right when the tool declares an array result."""
    lists_only = "return {'n': len([t for t in found if t])}"
    array = {"type": "array", "items": {"type": "object"}}
    assert _probe_run(tmp_path / "a", lists_only, array).findings == []
    # FastMCP wraps a non-object result; rote's helper unwraps it.
    wrapped = {
        "type": "object",
        "x-fastmcp-wrap-result": True,
        "properties": {"result": array},
    }
    assert _probe_run(tmp_path / "b", lists_only, wrapped).findings == []
    # Declared as an object: a dict reader is right, nothing to report.
    dict_reader = "return {'n': len(found.get('threads', []))}"
    assert _probe_run(tmp_path / "c", dict_reader, {"type": "object"}).findings == []
    # Undeclared: the same dict reader fails on a list.
    [finding] = _probe_run(tmp_path / "d", dict_reader).findings
    assert finding.kind == "prefix_shape_assumption"


_TWO_UPSTREAMS_IR = """\
name: ops
input:
  type: Day
  required: [report_date]
nodes:
  - id: search
    kind: external_call
    description: search mail
    impl: extracted/mail.py:search
    mcp: {server: gmail, tool: search_threads}
    inputs:
      query: pipeline.input.report_date
  - id: other
    kind: pure_function
    description: a step that does not run on the sample
    impl: extracted/mail.py:other
    inputs:
      day: pipeline.input.report_date
  - id: count
    kind: pure_function
    description: count threads
    impl: extracted/mail.py:count
    constants: {cap: 5}
    inputs:
      found: search.output
      prior: other.output
edges:
  - {from: search, to: count}
  - {from: other, to: count}
entry_nodes: [search, other]
exit_nodes: [count]
"""


def _two_upstream_run(tmp_path: Path, other_body: str):
    root = _write(
        tmp_path,
        {
            "pipeline.yaml": _TWO_UPSTREAMS_IR,
            "extracted/mail.py": "def search(query):\n    raise NotImplementedError('mcp')\n\n"
            f"def other(day):\n    {other_body}\n\n"
            "def count(found, cap, prior=None):\n"
            "    return {'n': min(cap, len(found.get('threads', [])))}\n",
        },
    )
    pipeline = _pipeline(root)
    return run_prefix(
        pipeline,
        root,
        _sample({"report_date": "2026-10-19"}),
        implementation_status(pipeline, root),
        trust_skill=True,
    )


def test_a_consumer_is_probed_only_when_its_other_inputs_ran(tmp_path: Path) -> None:
    """The probe passes made-up data for the unrecorded result only. A
    consumer whose other upstream did not run is never called, even when a
    default would let it run (and fail on the dict read, falsely)."""
    ran = _two_upstream_run(tmp_path / "ran", "return {'ok': True}")
    # Its constant is passed too: without it the call fails on the signature.
    assert [f.kind for f in ran.findings] == ["prefix_shape_assumption"]
    stub = _two_upstream_run(tmp_path / "stub", "raise NotImplementedError('later')")
    assert stub.steps["other"]["status"] != "ok"
    assert stub.findings == [] and stub.steps["count"]["status"] == "not_reached"


def test_a_wrapped_text_result_is_probed_as_text(tmp_path: Path) -> None:
    """FastMCP wraps a string result as {"result": ...}; the helper unwraps it."""
    wrapped = {
        "type": "object",
        "x-fastmcp-wrap-result": True,
        "properties": {"result": {"type": "string"}},
    }
    dict_reader = "return {'n': len(found.get('threads', []))}"
    [finding] = _probe_run(tmp_path, dict_reader, wrapped).findings
    assert "returns text:" in finding.message


def test_a_field_binding_or_a_fanned_consumer_is_not_probed(tmp_path: Path) -> None:
    """Only a consumer that reads the whole result is probed: a field of an
    unknown result, or one element of it, says nothing about the shape."""

    field = _MCP_IR
    root = _write(
        tmp_path / "field",
        {
            "pipeline.yaml": field,
            "extracted/mail.py": "def search(query):\n    raise NotImplementedError('mcp')\n\n"
            "def count(threads):\n    return {'n': len(threads.get('items', []))}\n",
        },
    )
    pipeline = _pipeline(root)
    run = run_prefix(
        pipeline,
        root,
        _sample({"report_date": "2026-10-19"}),
        implementation_status(pipeline, root),
        trust_skill=True,
    )
    assert run.findings == [] and run.steps["count"]["status"] == "not_reached"

    fanned = _MCP_IR.replace("threads: search.output.threads", "found: search.output").replace(
        "    impl: extracted/mail.py:count\n",
        "    impl: extracted/mail.py:count\n    fan_out: true\n",
    )
    root = _write(
        tmp_path / "fanned",
        {
            "pipeline.yaml": fanned,
            "extracted/mail.py": "def search(query):\n    raise NotImplementedError('mcp')\n\n"
            "def count(found):\n    return {'subject': found.get('subject')}\n",
        },
    )
    pipeline = _pipeline(root)
    run = run_prefix(
        pipeline,
        root,
        _sample({"report_date": "2026-10-19"}),
        implementation_status(pipeline, root),
        trust_skill=True,
    )
    assert run.findings == [] and run.steps["count"]["status"] == "not_reached"


@pytest.mark.parametrize(
    "count_body",
    [
        "return {'n': int(found)}",  # TypeError, but not a dict assumption
        "return {'n': found.total}",  # AttributeError on list: not dict-shaped access
    ],
)
def test_only_dict_assumptions_count_as_shape_findings(count_body: str, tmp_path: Path) -> None:
    run = _probe_run(tmp_path, count_body)
    assert [f.kind for f in run.findings] == []
    assert run.steps["count"]["status"] == "not_reached"


def test_the_prefix_runs_in_the_os_sandbox(tmp_path: Path) -> None:
    from rote.compiler import sandbox

    if sandbox.detect()[0] is None:
        pytest.skip("no OS sandbox on this machine")
    root = _prefix_root(tmp_path, row_key="expense_id")
    pipeline = _pipeline(root)
    run = run_prefix(pipeline, root, _sample(_ROWS), implementation_status(pipeline, root))
    assert [f.kind for f in run.findings] == ["prefix_run_failed"]


def test_without_a_sandbox_the_prefix_is_unverified_not_passed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rote.compiler import sandbox

    monkeypatch.setattr(sandbox, "detect", lambda: (None, "none in this test"))
    root = _prefix_root(tmp_path, row_key="expense_id")
    pipeline = _pipeline(root)
    run = run_prefix(pipeline, root, _sample(_ROWS), implementation_status(pipeline, root))
    assert run.status == "unverified" and run.findings == []


_FORGER = """\
import json, os, sys

def dupes(submissions):
    forged = {{"dupes": {{"status": "ok"}}, "per_line": {{"status": "ok"}}}}
    print("ROTE_PREFIX_REPORT " + json.dumps(forged))
    sys.stdout.flush()
    {exit}

def per_line(line, cap):
    return {{}}

def summary(flagged, cap):
    return {{}}

def after(decision):
    return decision
"""


@pytest.mark.parametrize("exit_call", ["os._exit(0)", "return {'flagged': [1]}"])
def test_agent_code_cannot_forge_the_prefix_report(tmp_path: Path, exit_call: str) -> None:
    """The report comes from a file the runner opened before importing agent
    code, so a printed report is just output. Exiting early leaves the file
    empty, which is a crash; returning normally lets the real report show
    the binding the forged one claimed was fine."""
    root = _prefix_root(tmp_path)
    (root / "extracted" / "checks.py").write_text(_FORGER.format(exit=exit_call))
    run = _run(root)
    assert run.status == "failed"
    kinds = {f.kind for f in run.findings}
    if exit_call.startswith("os."):
        assert kinds == {"prefix_crashed"}
    else:
        assert "prefix_binding_failed" in kinds  # dupes returned no `cap`


def test_a_stub_ends_its_branch_and_is_never_called(tmp_path: Path) -> None:
    """A valid explained stub must not fail the compile: it is skipped, and
    its dependents are not reached. (Calling it would raise, which the
    runner would report as a stub, but a stub with side effects or a slow
    body must not run at all.)"""
    root = _prefix_root(tmp_path)
    checks = (root / "extracted" / "checks.py").read_text()
    stub = (
        "def dupes(submissions):\n"
        "    open('called', 'w').write('x')\n"
        "    raise NotImplementedError('the duplicate rule is not stated in the skill')\n"
    )
    (root / "extracted" / "checks.py").write_text(
        checks.replace(checks[: checks.index("def per_line")], stub + "\n\n")
    )
    pipeline = _pipeline(root)
    impls = implementation_status(pipeline, root)
    assert {i.node_id: i.status for i in impls}["dupes"] == "stub"
    run = run_prefix(pipeline, root, _sample(_ROWS), impls, trust_skill=True)
    assert run.findings == []
    assert run.steps["dupes"] == {"status": "not_deterministic", "reason": "pure_function stub"}
    assert run.steps["summary"]["status"] == "not_reached"
    assert not (root / "called").exists()


# ───────── verification and the compiler ─────────


def test_verification_sends_grounding_failures_to_repair(tmp_path: Path) -> None:
    root = _prefix_root(tmp_path, row_key="expense_id")
    grounding = Grounding(sample=_sample(_ROWS))
    v = verify_deliverable(root, grounding=grounding, trust_skill=True, run_tests=True)
    assert v.prefix is not None and v.prefix.status == "failed"
    assert not v.ok
    instruction = v.repair_instruction(root)
    assert instruction is not None and "KeyError: 'expense_id'" in instruction
    # --no-verify-tests executes nothing the agent wrote: no prefix run.
    quiet = verify_deliverable(root, grounding=grounding, trust_skill=True, run_tests=False)
    assert quiet.prefix is None


def test_the_expected_expense_audit_runs_on_its_own_sample() -> None:
    """The hand-written gold pipeline is the positive control."""
    grounding = Grounding(sample=load_sample(EXPENSE / "input.json"))
    v = verify_deliverable(
        EXPENSE / "expected",
        skill_dir=EXPENSE / "skill",
        grounding=grounding,
        trust_skill=True,
        run_tests=True,
    )
    assert v.prefix is not None and v.prefix.findings == []
    assert v.prefix.summary.startswith("6 of 8 node(s) ran")
    assert [f for f in v.findings if f.kind.startswith(("input_", "reference_", "gate_"))] == []


class _GroundedDriver(CompilerDriver):
    """Writes a deliverable that renames the sample's row key, then fixes it on repair."""

    name = "grounded"

    def __init__(self) -> None:
        self.instructions: str | None = None
        self.saw_sample: bool = False
        self.repairs: list[str] = []

    def is_available(self) -> tuple[bool, str]:
        return (True, "")

    async def run(  # type: ignore[override]
        self,
        skill_dir,
        compiler_skill_dir,
        work_dir,
        extra_instructions=None,
        on_event=None,
        repair=None,
    ) -> DriverResult:
        self.instructions = extra_instructions
        self.saw_sample = (work_dir / GROUNDING_DIRNAME / "sample-input.json").is_file()
        _prefix_root(work_dir, row_key="expense_id")
        while repair is not None:
            instruction = repair(work_dir / "pipeline.yaml")
            if instruction is None:
                break
            self.repairs.append(instruction)
            _prefix_root(work_dir)
        return DriverResult(
            pipeline_yaml_path=work_dir / "pipeline.yaml", work_dir=work_dir, driver_name=self.name
        )


@pytest.mark.asyncio
async def test_the_compiler_grounds_the_agent_and_repairs_a_renamed_row_key(
    tmp_path: Path,
) -> None:
    skill = _write(tmp_path / "skill", {"SKILL.md": "# s\n", "input.json": json.dumps(_ROWS)})
    driver = _GroundedDriver()
    compiler = Compiler(system1=False, compiler_skill_dir=tmp_path, trust_skill=True)
    compiler.select_driver = lambda: driver  # type: ignore[method-assign]

    result = await compiler.compile(skill, tmp_path / "out")

    assert driver.saw_sample
    assert driver.instructions is not None and "INPUT CONTRACT" in driver.instructions
    [instruction] = driver.repairs
    assert "KeyError: 'expense_id'" in instruction
    assert result.verification is not None and result.verification.ok
    # The sample was the agent's reference material, not output.
    assert not (tmp_path / "out" / GROUNDING_DIRNAME).exists()
    # Grounded, so the report asks no question about the input contract.
    assert "### Contract:" not in (tmp_path / "out" / "compile-report.md").read_text()


@pytest.mark.asyncio
async def test_without_a_sample_the_report_lists_the_chosen_contract(tmp_path: Path) -> None:
    skill = _write(
        tmp_path / "skill", {"SKILL.md": "# s\n\nThe input is a batch of expense lines.\n"}
    )
    driver = _GroundedDriver()
    compiler = Compiler(system1=False, compiler_skill_dir=tmp_path, trust_skill=True)
    compiler.select_driver = lambda: driver  # type: ignore[method-assign]

    result = await compiler.compile(skill, tmp_path / "out")

    assert not driver.saw_sample and driver.repairs == []
    assert result.verification is not None and result.verification.prefix is None
    report = (tmp_path / "out" / "compile-report.md").read_text()
    assert "### Contract: (pipeline input)" in report
    assert "`period`, `submissions`" in report
    assert "The input is a batch of expense lines." in report


@pytest.mark.asyncio
async def test_a_missing_sample_input_file_fails_before_any_run(tmp_path: Path) -> None:
    from rote.compiler import CompilerError

    skill = _write(tmp_path / "skill", {"SKILL.md": "# s\n"})
    compiler = Compiler(system1=False, compiler_skill_dir=tmp_path)
    compiler.select_driver = lambda: pytest.fail("no driver may run")  # type: ignore[method-assign]
    with pytest.raises(CompilerError, match="--sample-input file not found"):
        await compiler.compile(skill, tmp_path / "out", sample_input=tmp_path / "nope.json")


@pytest.mark.parametrize(
    "module",
    ["rote.eval.empirical", "rote.eval.baseline", "rote.compiler", "rote.compiler.verify"],
)
def test_each_module_imports_first_in_a_fresh_interpreter(module: str) -> None:
    """Import order hides cycles inside one test process: the e2e suites
    import rote.eval.empirical first, and a module-level import of
    rote.eval.baseline in rote.compiler made that a circular import."""
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-c", f"import {module}"], capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 0, proc.stderr[-1500:]


# ───────── the bench ─────────


@pytest.mark.parametrize("example", ["expense-audit", "ops-report"])
def test_a_plain_compile_of_an_example_finds_its_sample(example: str) -> None:
    """`rote compile examples/<name>/skill` with no flags is grounded in the
    example's sample run, and the bench compiles exactly that: one file,
    inside the skill dir, read by the same lookup a user's skill gets."""
    import argparse

    from rote.bench.runner import compile_argv

    skill = REPO / "examples" / example / "skill"
    found = find_sample(skill)
    assert found is not None
    assert found.source == str(skill / "sample-input.json")
    # One copy: no second sample beside the skill for the bench to pass.
    assert not (skill.parent / "sample.json").exists()
    args = argparse.Namespace(runtime="dbos", model=None, agent=None, compile_flag=[])
    assert "--sample-input" not in compile_argv(skill, Path("out"), args)


def test_every_example_with_fidelity_cases_bundles_a_sample() -> None:
    """A fidelity case run on a compile that invented its input contract
    fails every case, so an example the bench scores must ship a sample."""
    for cases in sorted((REPO / "examples").glob("**/fidelity/cases.yaml")):
        skill = cases.parent.parent / "skill"
        assert find_sample(skill) is not None, f"{skill} has fidelity cases and no sample"


def _leaves(value: object) -> set[object]:
    """Every scalar inside ``value``, as ``(type, value)`` so 1 and True differ."""
    if isinstance(value, dict):
        return set().union(*(_leaves(v) for v in value.values())) if value else set()
    if isinstance(value, list):
        return set().union(*(_leaves(v) for v in value)) if value else set()
    return {(type(value).__name__, value)}


def _keys(value: object, path: str = "") -> set[str]:
    """Every key path inside ``value``, list rows merged as ``[]``."""
    if isinstance(value, dict):
        return set().union(*({f"{path}.{k}"} | _keys(v, f"{path}.{k}") for k, v in value.items()))
    if isinstance(value, list):
        return set().union(*(_keys(v, f"{path}[]") for v in value)) if value else set()
    return set()


#: Vocabulary the skill itself fixes, so a sample and a case must share it:
#: currency codes the rate table lists, the policy's category codes, and
#: booleans. Everything else (people, ids, merchants, amounts, dates,
#: counts, sites) must differ.
_SHARED_VOCABULARY = {
    *(("str", c) for c in ("USD", "EUR", "GBP", "CAD", "JPY")),
    *(
        ("str", c)
        for c in (
            "airfare",
            "lodging",
            "meals",
            "ground_transport",
            "software",
            "office_supplies",
            "entertainment",
        )
    ),
    ("bool", True),
    ("bool", False),
}


@pytest.mark.parametrize("example", ["expense-audit", "ops-report"])
def test_the_bench_sample_shares_the_cases_keys_and_none_of_their_values(example: str) -> None:
    from rote.bench.fidelity import load_cases

    sample = load_sample(REPO / "examples" / example / "skill" / "sample-input.json")
    casefile = load_cases(REPO / "examples" / example)
    cases = casefile.cases
    case_values: set[object] = set()
    for case in cases:
        case_values |= _leaves(case.input) | _leaves(case.gates)
    # The answers too: a sample value equal to a total the skill computed
    # on a case would hand the compile that case's expected output.
    reference = json.loads(casefile.reference_path.read_text(encoding="utf-8"))
    for recorded in reference["cases"].values():
        for run in recorded["runs"]:
            case_values |= _leaves(run.get("output"))
    shared = (_leaves(sample.payload) | _leaves(list(sample.gates.values()))) & case_values
    assert shared <= _SHARED_VOCABULARY, f"fidelity case values in the sample: {shared}"
    # Same contract: every key the cases use, input and gates alike.
    base = cases[0]
    assert _keys(sample.payload) >= _keys(base.input)
    assert [_keys(g) for g in sample.gates.values()] == [_keys(g) for g in base.gates]


def test_rote_compile_passes_sample_input_to_the_compiler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rote.cli import main as cli_main
    from rote.compiler import CompilerError

    seen: dict[str, object] = {}

    class _Recorder:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def compile(self, skill_path, output_dir, **kwargs):  # noqa: ANN001, ANN201
            seen.update(kwargs)
            raise CompilerError("stop here")

    monkeypatch.setattr("rote.cli.Compiler", _Recorder)
    skill = _write(tmp_path / "skill", {"SKILL.md": "---\nname: t\n---\n"})
    sample = _write(tmp_path, {"s.json": '{"a": 1}'}) / "s.json"
    argv = ["compile", str(skill), "--runtime", "dbos", "--local", "--no-eval"]
    rc = cli_main([*argv, "--out", str(tmp_path / "out"), "--sample-input", str(sample)])
    assert rc == 1 and seen["sample_input"] == str(sample)


def test_system1_gives_each_gate_its_own_sample_answer_whatever_the_order(
    tmp_path: Path,
) -> None:
    """The sample lists the contact-review answer before the campaign-brief
    one. Assigning by position would give the brief gate the contact keys,
    and the check would then pass, matching rote's own wrong assignment.
    Both go through one matcher: signal, then the step's own words, then
    execution order."""
    from rote.compiler import system1
    from rote.ir import NodeKind

    bdr = REPO / "examples" / "bdr-outreach" / "skill"
    sample = _sample(
        {},
        gates={
            "contact_changes": {"added": [], "removed": [], "reprioritized": []},
            # Ties the brief gate and the handoff gate on words (approved,
            # notes): the earlier gate in execution order takes it.
            "brief_sign_off": {"approved": True, "brief_notes": ""},
        },
    )
    system1.prepare(bdr, tmp_path, sample=sample)
    gates = [
        n
        for wave in _pipeline(tmp_path).execution_waves()
        for n in wave
        if n.kind is NodeKind.HITL_GATE
    ]
    keys = {g.id: g.output_keys for g in gates}
    assert keys["contact_review_gate_gate"] == {"added", "removed", "reprioritized"}
    assert keys["campaign_brief_gate"] == {"approved", "brief_notes"}
    # The third gate has no answer in the sample and keeps the step's words.
    assert keys["handoff_gate"] == {"approved", "notes"}
    assert [g.id for g in gates][0] == "campaign_brief_gate"


def test_a_sample_field_no_reference_can_bind_is_an_open_question(tmp_path: Path) -> None:
    """`Invoice #` cannot be `pipeline.input.<field>`, but it is real data, so
    the sample is kept: the agent is told to read it from the whole input,
    and a reviewer sees it as an open question."""
    sample = _sample({"Invoice #": "INV-9", "vendor": "Acme", "2nd_line": "x"})
    brief = contract_brief(sample, tmp_path, tmp_path / GROUNDING_DIRNAME)
    assert "not identifiers" in brief and "`2nd_line`, `Invoice #`" in brief
    root = _write(tmp_path, {"pipeline.yaml": _expense_ir(["vendor"])})
    questions = contract_questions(_pipeline(root), None, Grounding(sample=sample))
    [question] = [q for q in questions if "not identifiers" in q.text]
    assert "`2nd_line`, `Invoice #`" in question.text
    # Identifier fields raise nothing.
    plain = Grounding(sample=_sample({"vendor": "Acme"}))
    assert not [q for q in contract_questions(_pipeline(root), None, plain) if "identif" in q.text]
    assert "not identifiers" not in contract_brief(plain.sample, tmp_path, tmp_path)


def _grounding_kinds(pipeline: Pipeline, sample: Sample) -> set[str]:
    from rote.compiler.grounding import grounding_findings

    return {f.kind for f in grounding_findings(pipeline, EXPENSE / "skill", sample)}


def test_system1_takes_its_input_schema_and_gate_keys_from_the_sample(tmp_path: Path) -> None:
    """In --system1 mode rote decides the IR's input and gate payloads, and the
    agent cannot change them, so the sample must feed those decisions: a
    check the agent cannot satisfy would only burn repair rounds."""
    from rote.compiler import system1
    from rote.ir import NodeKind

    sample = load_sample(EXPENSE / "skill" / "sample-input.json")
    (tmp_path / "with").mkdir()
    (tmp_path / "without").mkdir()
    system1.prepare(EXPENSE / "skill", tmp_path / "with", sample=sample)
    grounded = _pipeline(tmp_path / "with")
    assert grounded.input.input_schema == input_contract(sample)
    [gate] = [n for n in grounded.nodes if n.kind is NodeKind.HITL_GATE]
    assert gate.output_keys == {"reviewer", "approved_ids", "rejected_ids"}
    assert _grounding_kinds(grounded, sample) == set()
    # The compiler hands the sample it found to System-1.
    import asyncio

    from rote.compiler import CompilerError

    seen: dict[str, object] = {}

    def _record(*args: object, **kwargs: object) -> None:
        seen.update(kwargs)
        raise system1.UnsafeSkillText("stop here")

    skill = _write(tmp_path / "skill", {"SKILL.md": "# s\n", "input.json": json.dumps(_ROWS)})
    compiler = Compiler(compiler_skill_dir=tmp_path, system1=True, system1_tiebreak="none")
    compiler.select_driver = lambda: _GroundedDriver()  # type: ignore[method-assign]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(system1, "prepare", _record)
        with pytest.raises(CompilerError, match="stop here"):
            asyncio.run(compiler.compile(skill, tmp_path / "out"))
    assert isinstance(seen["sample"], Sample) and seen["sample"].payload == _ROWS
    # Without a sample, the wording decides, and the sample's check would fail.
    system1.prepare(EXPENSE / "skill", tmp_path / "without")
    assert "gate_payload_not_sample" in _grounding_kinds(_pipeline(tmp_path / "without"), sample)


_FAN_MCP_IR = """\
name: brief
input:
  type: Products
  required: [names]
nodes:
  - id: build
    kind: pure_function
    description: one search query per product
    impl: extracted/brief.py:build
    inputs:
      names: pipeline.input.names
  - id: search
    kind: external_call
    description: search the docs once per query
    impl: extracted/brief.py:search
    fan_out: true
    mcp: {server: docs, tool: search_docs, args: {query: query}}
    inputs:
      query: build.output
  - id: count
    kind: pure_function
    description: count the results
    impl: extracted/brief.py:count
    inputs:
      results: search.output
edges:
  - {from: build, to: search, fan_out: true}
  - {from: search, to: count}
entry_nodes: [build]
exit_nodes: [count]
"""

_FAN_MCP_MODULE = """\
def build(names):
    return [f"{n} limits" for n in names] if names != ["bad"] else [1, "x"]


def search(query):
    raise NotImplementedError("mcp")


def count(results):
    return {"n": len(results)}
"""


def _fan_tools() -> dict[tuple[str, str], object]:
    from rote.compiler.tool_grounding import ToolSchema

    schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }
    return {("docs", "search_docs"): ToolSchema("docs", "search_docs", schema, "tools/list", True)}


def test_a_fanned_mcp_node_is_checked_one_call_per_element(tmp_path: Path) -> None:
    """The emitted code calls the tool once per element, so the prefix checks
    each element's arguments: the whole list is not a `query` (the demo's
    compile failed `must be string, got list` on correct code)."""
    root = _write(tmp_path, {"pipeline.yaml": _FAN_MCP_IR, "extracted/brief.py": _FAN_MCP_MODULE})
    pipeline = _pipeline(root)
    impls = implementation_status(pipeline, root)
    sample = _sample({"names": ["R2", "Durable Objects"]})
    run = run_prefix(pipeline, root, sample, impls, trust_skill=True, tools=_fan_tools())
    assert [f for f in run.findings if f.kind == "tool_arguments_invalid"] == []
    assert run.steps["search"]["arguments"] == [
        {"query": "R2 limits"},
        {"query": "Durable Objects limits"},
    ]
    # A real per-element mistake is still caught, and names the element.
    bad = run_prefix(
        pipeline, root, _sample({"names": ["bad"]}), impls, trust_skill=True, tools=_fan_tools()
    )
    [finding] = [f for f in bad.findings if f.kind == "tool_arguments_invalid"]
    assert "search[0] would call docs.search_docs with {'query': 1}" in finding.message


def test_recorded_calls_stand_in_for_each_fanned_element(tmp_path: Path) -> None:
    root = _write(tmp_path, {"pipeline.yaml": _FAN_MCP_IR, "extracted/brief.py": _FAN_MCP_MODULE})
    pipeline = _pipeline(root)
    observed = tuple(
        ObservedToolCall("docs", "search_docs", {"query": q}, [{"type": "text", "text": "[1]"}])
        for q in ("R2 limits", "Durable Objects limits")
    )
    run = run_prefix(
        pipeline,
        root,
        _sample({"names": ["R2", "Durable Objects"]}),
        implementation_status(pipeline, root),
        observations=observed,
        trust_skill=True,
        tools=_fan_tools(),
    )
    assert run.steps["search"]["status"] == "ok"
    # The consumer ran on the per-element results, as on every runtime.
    assert run.steps["count"]["status"] == "ok"
    # With one element unrecorded the node is unrecorded, never half-faked.
    partial = run_prefix(
        pipeline,
        root,
        _sample({"names": ["R2", "D1"]}),
        implementation_status(pipeline, root),
        observations=observed,
        trust_skill=True,
        tools=_fan_tools(),
    )
    assert partial.steps["search"]["status"] == "no_recording"
