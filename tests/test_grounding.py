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


def _sample(payload: dict, gates: tuple[dict, ...] = ()) -> Sample:
    return Sample(payload=payload, source="sample.json", gates=gates)


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
    assert [sorted(g) for g in sample.gates] == [["approved_ids", "rejected_ids", "reviewer"]]


def test_an_input_field_named_input_is_not_read_as_an_envelope() -> None:
    sample = Sample(payload={}, source="x")
    from rote.compiler.grounding import parse_sample

    sample = parse_sample({"input": "free text", "mode": "fast"}, "x")
    assert sample.payload == {"input": "free text", "mode": "fast"}
    assert sample.gates == ()


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
    sample = _sample({"report_date": "x"}, gates=({"reviewer": "a", "approved_ids": []},))
    [finding] = gate_findings(_pipeline(root), None, sample)
    assert finding.kind == "gate_payload_not_sample"
    assert "approved_ids, reviewer" in finding.message
    good = _GATE_IR.format(output="{reviewer: str, approved_ids: list}")
    root = _write(tmp_path / "ok", {"pipeline.yaml": good})
    assert gate_findings(_pipeline(root), None, sample) == []


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
    assert run.steps["search"] == {"status": "ok", "recorded": True}
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
    compiler = Compiler(compiler_skill_dir=tmp_path, trust_skill=True)
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
    compiler = Compiler(compiler_skill_dir=tmp_path, trust_skill=True)
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
    compiler = Compiler(compiler_skill_dir=tmp_path)
    compiler.select_driver = lambda: pytest.fail("no driver may run")  # type: ignore[method-assign]
    with pytest.raises(CompilerError, match="--sample-input file not found"):
        await compiler.compile(skill, tmp_path / "out", sample_input=tmp_path / "nope.json")


# ───────── the bench ─────────


def test_the_bench_grounds_each_compile_in_the_examples_sample() -> None:
    import argparse

    from rote.bench.runner import compile_argv, sample_input

    args = argparse.Namespace(runtime="dbos", model=None, agent=None)
    sample = sample_input(REPO / "examples", "expense-audit")
    assert sample == EXPENSE / "input.json"
    argv = compile_argv(Path("skill"), Path("out"), args, sample)
    assert argv[argv.index("--sample-input") + 1] == str(EXPENSE / "input.json")
    # ops-report ships no sample, so its compile is not given one.
    assert sample_input(REPO / "examples", "ops-report") is None
    assert "--sample-input" not in compile_argv(Path("skill"), Path("out"), args, None)


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
