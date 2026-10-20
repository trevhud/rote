"""System-1 against hostile skills and against its own weakest rules.

A skill is untrusted input (invariant #7): everything System-1 derives from
it and writes into code must stay data. The second half pins rules a #150
review mutation sweep found unpinned (M1-M3, M7, M12, M13, M19).
"""

from __future__ import annotations

import ast
import os
import shutil
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
import yaml

from rote.adapters import get_adapter
from rote.compiler import Compiler, CompilerError, system1
from rote.compiler.drivers import CompilerDriver, DriverResult
from rote.compiler.system1.decide import REVIEW_BELOW
from rote.compiler.system1.enforce import (
    _decided_fields,
    check_proposals,
    enforce,
)
from rote.compiler.system1.parse import RuleTable
from rote.compiler.system1.skeleton import tables_module
from rote.ir import Node, load_pipeline

REPO = Path(__file__).resolve().parent.parent
PWNED = "open('PWNED', 'w')"


def _skill(tmp_path: Path, body: str, *, refs: dict[str, str] | None = None) -> Path:
    skill = tmp_path / "skill"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(f"---\nname: tiny\ndescription: x\n---\n\n# Tiny\n\n{body}\n")
    if refs:
        (skill / "references").mkdir()
        for name, text in refs.items():
            (skill / "references" / name).write_text(text)
    return skill


def _prepare(tmp_path: Path, skill: Path) -> tuple[system1.Plan, dict[str, Any]]:
    work = tmp_path / "work"
    work.mkdir()
    plan = system1.prepare(skill, work)
    return plan, yaml.safe_load((work / "pipeline.yaml").read_text())


def _calls_pwned(source: str) -> bool:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and any(
            isinstance(a, ast.Constant) and a.value == "PWNED" for a in node.args
        ):
            return True
    return False


# ── hostile input ─────────────────────────────────────────────────────────


@pytest.mark.skipif(os.name == "nt", reason="a newline in a file name is POSIX-only")
def test_a_reference_file_name_cannot_inject_code(tmp_path: Path) -> None:
    """The #150 review's proof: this name used to become a live line of rule_tables.py."""
    table = "| Category | Cap |\n| --- | --- |\n| Meals | $75.00 |\n| Hotel | $300.00 |\n"
    skill = _skill(
        tmp_path,
        "## Step 1: Total the batch\n\nTotal every line.\n",
        refs={f"caps\n{PWNED}\n#.md": table},
    )
    with pytest.raises(system1.UnsafeSkillText, match="reference file name"):
        _prepare(tmp_path, skill)


@pytest.mark.asyncio
async def test_compile_reports_an_unsafe_skill_as_a_compile_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    skill = _skill(tmp_path, "## Step 1: Total\n\nTotal every line.\n", refs={"a b.md": "x"})
    monkeypatch.setattr(Compiler, "select_driver", lambda self: _Proposer())
    with pytest.raises(CompilerError, match="reference file name"):
        await Compiler(system1=True, system1_tiebreak="none").compile(skill, tmp_path / "out")


def test_generated_comments_cannot_end_their_line(tmp_path: Path) -> None:
    """Defense in depth: skill text in a generated comment stays one line,
    whatever line terminator it carries."""
    hostile = [f"| Meals{sep}{PWNED} | $75.00 |" for sep in ("\n", "\r", " ", "\x0b")]
    table = RuleTable(
        file="references/caps.md",
        heading="Caps",
        line=1,
        header=["Category", "Cap"],
        rows=[["Meals", "$75.00"], ["Hotel", "$300.00"]],
        raw_rows=hostile,
    )
    rendered = tables_module([table])
    assert rendered is not None
    module, _, names = rendered
    assert not _calls_pwned(module)
    namespace: dict[str, Any] = {}
    cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        exec(compile(module, "rule_tables.py", "exec"), namespace)  # noqa: S102
    finally:
        os.chdir(cwd)
    assert not (tmp_path / "PWNED").exists()
    assert namespace[names[0]] == {"Meals": Decimal("75.00"), "Hotel": Decimal("300.00")}


@pytest.mark.parametrize("runtime", ["dbos", "temporal"])
def test_hostile_prose_stays_data_in_emitted_code(tmp_path: Path, runtime: str) -> None:
    """Headings, step prose and placeholders all reach the skeleton (section,
    description, input fields, gate payload keys). None may become code."""
    breakout = f'"""\n{PWNED}\n"""'
    skill = _skill(
        tmp_path,
        f"Run it for [x {breakout}] and <y {breakout}>, for [drug name] and <report date>.\n\n"
        f"## Step 1: Count rows {breakout}\n\nCount every row. {breakout}\n\n"
        f"## Step 2: Review {breakout}\n\nAsk the user to review it.\n\n"
        f"1. **Totals {breakout}** check\n2. **Notes** count\n",
    )
    _, doc = _prepare(tmp_path, skill)
    pipeline = load_pipeline(tmp_path / "work" / "pipeline.yaml")
    out = tmp_path / "emit"
    get_adapter(runtime, extracted_source_dir=tmp_path / "work").emit(pipeline, out)
    for py in out.rglob("*.py"):
        assert not _calls_pwned(py.read_text()), py
    for node in doc["nodes"]:
        assert node["id"].isidentifier()
        if isinstance(node.get("output"), dict):
            assert all(k.isidentifier() for k in node["output"])
    names = list(doc["input"]["input_schema"].get("properties", {}))
    assert {"drug_name", "report_date"} <= set(names)
    assert all(name.isidentifier() for name in names)


# ── rules the review sweep found unpinned ─────────────────────────────────


def test_explicit_mandatory_wins_over_conditional_wording(tmp_path: Path) -> None:
    """M3: "MANDATORY" and "you may skip" in one step is mandatory, at 0.95."""
    skill = _skill(
        tmp_path,
        "## Step 1: Check duplicates (MANDATORY)\n\n"
        "Check every line for duplicates. You may skip lines already flagged.\n",
    )
    plan, doc = _prepare(tmp_path, skill)
    assert doc["nodes"][0]["mandatory"] is True
    (d,) = [d for d in plan.nodes[0].decisions if d.field == "mandatory"]
    assert (d.value, d.confidence) == (True, 0.95)


def test_review_lists_exactly_the_decisions_below_threshold(tmp_path: Path) -> None:
    """M7 + M13: what plan.review holds, and what the report repeats."""
    skill = _skill(
        tmp_path,
        "## Step 1: Total the batch\n\nTotal every line in USD.\n\n"
        "## Step 2: Format the totals\n\nOptionally format the totals as a table.\n\n"
        "## Step 3: Wrap up\n\nThank the team.\n",
    )
    plan, _ = _prepare(tmp_path, skill)
    # A decision AT the threshold is settled, not listed: "optionally" sets
    # mandatory False at exactly REVIEW_BELOW.
    at = [d for n in plan.nodes for d in n.decisions if d.confidence == REVIEW_BELOW]
    assert at
    assert not any(f"(confidence {REVIEW_BELOW:.2f})" in r for r in plan.review)
    below = [
        (n.id, d.field) for n in plan.nodes for d in n.decisions if d.confidence < REVIEW_BELOW
    ]
    assert below  # "Thank the team" has no signal: kind 0.3, data flow 0.5
    assert len(plan.review) == len(below)
    for (nid, fld), line in zip(below, plan.review, strict=True):
        assert line.startswith(f"`{nid}` {fld} = ")
    # An unmarked step is mandatory at 0.6: listed, so a person can clear it.
    assert any(" mandatory = True (confidence 0.60)" in r for r in plan.review)
    work = tmp_path / "work"
    system1.settle(work, plan, final=True)
    questions = system1.report_extras(work, plan).questions
    listed = [d for t, d in questions if t.startswith("System-1 decision below")]
    assert listed == plan.review


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("Ask the user to approve the list.", "hitl_gate"),  # 1 gate, 0 judge
        ("Ask the user to approve; classify the list.", "hitl_gate"),  # tie: the gate wins
        ("Ask the user. Classify and score the rubric list.", "llm_judge"),  # judge outvotes
        (
            "Ask the user to approve it, then run the autonomous loop until the quota is met.",
            "hitl_gate",
        ),
    ],
)
def test_gate_judge_and_loop_precedence(tmp_path: Path, text: str, kind: str) -> None:
    """M1 (gate vs judge tie) and M2 (gate and loop in one step)."""
    _, doc = _prepare(tmp_path, _skill(tmp_path, f"## Step 1: Handle it\n\n{text}\n"))
    assert doc["nodes"][0]["kind"] == kind


def test_a_judge_over_a_plural_heading_does_not_fan_out_without_per_item_words(
    tmp_path: Path,
) -> None:
    """M6: the plural-heading rule is gone; only per-item wording fans out."""
    skill = _skill(
        tmp_path,
        "## Step 1: Pull leads\n\nSearch HubSpot for leads using the HubSpot MCP tool.\n\n"
        "## Step 2: Score Leads\n\nScore the leads against the rubric.\n",
    )
    _, doc = _prepare(tmp_path, skill)
    (judge,) = [n for n in doc["nodes"] if n["kind"] == "llm_judge"]
    assert "fan_out" not in judge


def test_pipeline_level_decisions_come_back(tmp_path: Path) -> None:
    """M12: the agent rewrites name, input, edges and entry/exit nodes."""
    work = tmp_path / "work"
    work.mkdir()
    system1.prepare(REPO / "examples" / "expense-audit" / "skill", work)
    skeleton = yaml.safe_load((work / "pipeline.yaml").read_text())
    doc = yaml.safe_load((work / "pipeline.yaml").read_text())
    doc["name"] = "renamed"
    doc["input"] = {"type": "Other", "input_schema": {"type": "object", "properties": {"z": {}}}}
    doc["edges"] = []
    doc["entry_nodes"] = [doc["nodes"][-1]["id"]]
    doc["exit_nodes"] = [doc["nodes"][0]["id"]]
    (work / "pipeline.yaml").write_text(yaml.safe_dump(doc, sort_keys=False))
    enforce(work)
    after = yaml.safe_load((work / "pipeline.yaml").read_text())
    for key in ("name", "input", "edges", "entry_nodes", "exit_nodes"):
        assert after[key] == skeleton[key], key


class _Proposer(CompilerDriver):
    name = "fake-proposer"

    def is_available(self) -> tuple[bool, str]:
        return (True, "")

    async def run(
        self,
        skill_dir: Path,
        compiler_skill_dir: Path,
        work_dir: Path,
        extra_instructions: str | None = None,
        on_event: object = None,
    ) -> DriverResult:
        doc = yaml.safe_load((work_dir / "pipeline.yaml").read_text())
        first = doc["nodes"][0]
        proposal = [{"node": first["id"], "field": "kind", "value": "llm_judge", "reason": "fuzzy"}]
        (work_dir / "system1" / "proposals.yaml").write_text(yaml.safe_dump(proposal))
        return DriverResult(work_dir / "pipeline.yaml", work_dir, self.name, {"cost_usd": 0.0})


@pytest.mark.asyncio
async def test_a_proposal_is_reported_and_never_applied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M19: an enforce that applied proposals.yaml would ship the LLM's decision."""
    skill = tmp_path / "skill"
    shutil.copytree(REPO / "examples" / "expense-audit" / "skill", skill)
    monkeypatch.setattr(Compiler, "select_driver", lambda self: _Proposer())
    out = tmp_path / "out"
    result = await Compiler(
        system1=True, system1_tiebreak="none", run_generated_tests=False
    ).compile(skill, out)
    first = result.pipeline.nodes[0]
    assert first.kind.value == "pure_function"
    report = (out / "compile-report.md").read_text()
    assert "### Proposed by the agent, not applied" in report
    assert f"> `{first.id}` kind -> 'llm_judge'" in report


def test_every_field_the_agent_writes_survives_enforcement(tmp_path: Path) -> None:
    """#150 blocker 2: input_schema, output_schema, eval_set ... vanished silently.

    Walks Node.model_fields, so a field the IR gains later is covered the day
    it lands: either it is a decision (restored) or it is carried as written."""
    work = tmp_path / "work"
    work.mkdir()
    system1.prepare(REPO / "examples" / "invoice-push" / "skill", work)
    skeleton = yaml.safe_load((work / "pipeline.yaml").read_text())
    doc = yaml.safe_load((work / "pipeline.yaml").read_text())
    node = doc["nodes"][0]
    fields = [f for f in Node.model_fields if f not in ("id", "output")]
    for f in fields:
        node[f] = f"SENTINEL-{f}"
    (work / "pipeline.yaml").write_text(yaml.safe_dump(doc, sort_keys=False))
    changes = enforce(work)
    after = yaml.safe_load((work / "pipeline.yaml").read_text())["nodes"][0]
    decided = _decided_fields(skeleton["nodes"][0])
    for f in fields:
        if f in decided:
            assert after.get(f) == skeleton["nodes"][0].get(f), f
            assert any(f"`{node['id']}` {f}:" in c for c in changes), f
        else:
            assert after.get(f) == f"SENTINEL-{f}", f


def test_a_malformed_proposals_file_is_reported_not_raised(tmp_path: Path) -> None:
    """#150 blocker 3: one bad YAML file used to lose the whole paid run."""
    work = tmp_path / "work"
    work.mkdir()
    plan = system1.prepare(REPO / "examples" / "expense-audit" / "skill", work)
    (work / "system1" / "proposals.yaml").write_text("- node: x\n  reason: a: b: c\n")
    (verdict,) = check_proposals(work)
    assert "did not parse" in verdict
    system1.settle(work, plan, final=True)
    details = [d for _, d in system1.report_extras(work, plan).questions]
    assert any("proposals.yaml did not parse" in d for d in details)


@pytest.mark.asyncio
async def test_a_failure_while_settling_keeps_the_paid_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    skill = tmp_path / "skill"
    shutil.copytree(REPO / "examples" / "expense-audit" / "skill", skill)
    monkeypatch.setattr(Compiler, "select_driver", lambda self: _Proposer())

    def boom(*args: object, **kwargs: object) -> list[str]:
        raise RuntimeError("settle exploded")

    monkeypatch.setattr(system1, "settle", boom)
    out = tmp_path / "out"
    with pytest.raises(CompilerError, match="settle exploded"):
        await Compiler(system1=True, system1_tiebreak="none", run_generated_tests=False).compile(
            skill, out
        )
    kept = tmp_path / "out.rejected"
    assert (kept / "pipeline.yaml").is_file()
    assert (kept / "system1" / "proposals.yaml").is_file()


def test_a_failure_before_a_repair_verification_does_not_escape(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    system1.prepare(REPO / "examples" / "expense-audit" / "skill", work)

    def boom(_: Path) -> None:
        raise RuntimeError("enforce exploded")

    repair = Compiler(
        system1=True, system1_tiebreak="none", run_generated_tests=False
    )._deliverable_repairer(None, REPO / "examples" / "expense-audit" / "skill", before_verify=boom)
    repair(work / "pipeline.yaml")  # verification still runs; nothing raises


@pytest.mark.parametrize(
    ("text", "kinds"),
    [
        # No vendor list: any retrieval verb with a source is a call.
        ("Pull open tickets from Zendesk.", ["external_call"]),
        (
            "Download the invoices from the billing portal and total them.",
            ["external_call", "pure_function"],
        ),
        # The skill's own reference files are not a service.
        ("Look up the rates in references/rates.md and total every line.", ["pure_function"]),
        # Anyone asked to act on the run parks it.
        ("Ask the finance lead to sign off on the batch.", ["hitl_gate"]),
        ("Send the draft to legal and wait for approval.", ["hitl_gate"]),
        ("The batch needs manager approval before it ships.", ["hitl_gate"]),
        # Asking someone FOR a fact is an error path, not a gate.
        ("Total every line. If a rate is missing, ask finance for the rate.", ["pure_function"]),
    ],
)
def test_general_fetch_and_gate_wording(tmp_path: Path, text: str, kinds: list[str]) -> None:
    """Wording outside the five examples, from the #150 review."""
    _, doc = _prepare(tmp_path, _skill(tmp_path, f"## Step 1: Do the step\n\n{text}\n"))
    assert [n["kind"] for n in doc["nodes"]] == kinds


def _loop_skill(tmp_path: Path, wording: str) -> Path:
    return _skill(
        tmp_path,
        "## Step 1: Research the market\n\n"
        f"Run an autonomous loop of web research over the `web_search` tool. {wording}\n\n"
        "## Available Tools\n\n| Tool | Purpose |\n|---|---|\n| `web_search` | search |\n",
    )


def test_a_stated_stop_condition_and_bound_become_the_loops_termination(tmp_path: Path) -> None:
    skill = _loop_skill(tmp_path, "Stop when ten vetted leads are found, at most 5 rounds.")
    _, doc = _prepare(tmp_path, skill)
    (loop,) = [n for n in doc["nodes"] if n["kind"] == "agent_loop"]
    assert loop["termination"] == {
        "condition": "Stop when ten vetted leads are found, at most 5 rounds",
        "max_iterations": 5,
    }
    # A decision: the agent cannot change it.
    work = tmp_path / "work"
    agent = yaml.safe_load((work / "pipeline.yaml").read_text())
    agent["nodes"][0]["termination"] = {"condition": "never", "max_iterations": 99}
    (work / "pipeline.yaml").write_text(yaml.safe_dump(agent, sort_keys=False))
    enforce(work)
    after = load_pipeline(work / "pipeline.yaml").nodes[0]
    assert after.termination is not None and after.termination.max_iterations == 5


def test_without_a_stated_bound_the_text_agent_writes_termination(tmp_path: Path) -> None:
    """#150 re-review: restoring "unset" shipped loops with no stop condition."""
    skill = _loop_skill(tmp_path, "Keep going until the quota is met.")
    plan, doc = _prepare(tmp_path, skill)
    (loop,) = [n for n in doc["nodes"] if n["kind"] == "agent_loop"]
    assert "termination" not in loop
    assert any("termination = None (confidence 0.50)" in r for r in plan.review)
    work = tmp_path / "work"
    stop = {"condition": "until the quota is met", "max_iterations": 8}
    (work / "system1" / "text.yaml").write_text(yaml.safe_dump({loop["id"]: {"termination": stop}}))
    enforce(work)
    after = load_pipeline(work / "pipeline.yaml").nodes[0]
    assert after.termination is not None
    assert (after.termination.condition, after.termination.max_iterations) == (
        "until the quota is met",
        8,
    )


@pytest.mark.parametrize(
    ("text", "kinds"),
    [
        # Data the step already has is not a call.
        ("Get the total from the table and round it.", ["pure_function"]),
        ("Read the status in the row and flag it.", ["pure_function"]),
        # A program is not a person.
        ("Ask the API to check the batch, then total it.", ["pure_function"]),
        # Work the skill says a person does is a gate, whatever the tool.
        ("Enrollment must be done manually in the CRM.", ["hitl_gate"]),
    ],
)
def test_general_rules_do_not_over_fire(tmp_path: Path, text: str, kinds: list[str]) -> None:
    _, doc = _prepare(tmp_path, _skill(tmp_path, f"## Step 1: Do the step\n\n{text}\n"))
    assert [n["kind"] for n in doc["nodes"]] == kinds


def test_agent_text_cannot_reshape_the_report(tmp_path: Path) -> None:
    """A proposal's node value is the agent's text: a newline plus '## ...' in it
    must stay inside its entry, not become a heading of the report."""
    from rote.compiler.provenance import open_questions
    from rote.compiler.report import render_compile_report

    work = tmp_path / "work"
    work.mkdir()
    plan = system1.prepare(REPO / "examples" / "expense-audit" / "skill", work)
    hostile = "x\n\n## Injected heading\n\n- injected item"
    (work / "system1" / "proposals.yaml").write_text(
        yaml.safe_dump(
            [{"node": hostile, "field": "kind", "value": "llm_judge", "reason": hostile}]
        )
    )
    extras = system1.report_extras(work, plan)
    assert all("\n" not in t and "\n" not in d for t, d in extras.questions)
    report = render_compile_report(load_pipeline(work / "pipeline.yaml"), work, extras)
    assert "\n## Injected heading" not in report
    assert not any("injected item" in e and e.startswith("- ") for e in open_questions(report))
