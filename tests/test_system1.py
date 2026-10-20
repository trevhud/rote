"""System-1 compile (`rote compile --system1`): rules decide, the agent only writes text.

No inference anywhere: the skeleton is pure code, and the compiler test uses
a fake driver. The expense-audit assertions pin decisions the skill states
outright ("Steps 1 to 3 do not depend on each other", the MANDATORY
markers, "up to three days", "For each employee"), so a rule that stops
reading them fails here rather than in a paid bench run.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from rote.adapters import get_adapter
from rote.bench.runner import _parser, compile_argv
from rote.compiler import Compiler, system1
from rote.compiler.drivers import CompilerDriver, DriverResult
from rote.compiler.system1.enforce import check_proposals, enforce
from rote.ir import load_pipeline

REPO = Path(__file__).resolve().parent.parent
EXAMPLES = REPO / "examples"
SKILLS = ["bdr-outreach", "deal-monitor", "expense-audit", "ops-report", "invoice-push"]


def _skeleton(tmp_path: Path, skill: str) -> tuple[system1.Plan, dict[str, Any]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    plan = system1.prepare(EXAMPLES / skill / "skill", tmp_path)
    return plan, yaml.safe_load((tmp_path / "pipeline.yaml").read_text())


@pytest.mark.parametrize("skill", SKILLS)
def test_skeleton_is_a_valid_ir_that_emits(tmp_path: Path, skill: str) -> None:
    system1.prepare(EXAMPLES / skill / "skill", tmp_path)
    pipeline = load_pipeline(tmp_path / "pipeline.yaml")
    assert pipeline.nodes
    out = tmp_path / "emit"
    get_adapter("dbos", extracted_source_dir=tmp_path).emit(pipeline, out)
    assert (out / "main.py").is_file()


@pytest.mark.parametrize("skill", SKILLS)
def test_every_node_records_its_decisions(tmp_path: Path, skill: str) -> None:
    plan, _ = _skeleton(tmp_path, skill)
    for n in plan.nodes:
        fields = {d.field for d in n.decisions}
        assert {"kind", "inputs"} <= fields, n.id
        assert all(0.0 <= d.confidence <= 1.0 and d.rule for d in n.decisions)


def test_expense_audit_decisions_follow_what_the_skill_states(tmp_path: Path) -> None:
    _, doc = _skeleton(tmp_path, "expense-audit")
    nodes = {n["source"]["section"]: [] for n in doc["nodes"]}
    for n in doc["nodes"]:
        nodes[n["source"]["section"]].append(n)
    step = {sec.split(":")[0]: ns for sec, ns in nodes.items()}

    # "Steps 1 to 3 do not depend on each other": each reads only the request.
    for s in ("Step 1", "Step 2", "Step 3"):
        assert [n["inputs"] for n in step[s]] == [{"request": "pipeline.input"}], s
    # Step 4 reads all three, groups, then fans out "for each employee".
    group, each = step["Step 4"]
    assert group["kind"] == "pure_function" and group["output"] == {"items": "list"}
    assert set(group["inputs"]) == {
        "request",
        *(step[s][0]["id"] for s in ("Step 1", "Step 2", "Step 3")),
    }
    assert each["fan_out"] is True
    assert each["inputs"][group["id"]] == f"{group['id']}.output.items"
    assert {"from": group["id"], "to": each["id"], "fan_out": True} in doc["edges"]
    # Every step the skill always runs is mandatory (no optional wording), the
    # marked ones at high confidence; the gate is mandatory implicitly, not flagged.
    assert [n.get("mandatory", False) for s in ("Step 1", "Step 2", "Step 3") for n in step[s]] == [
        True,
        True,
        True,
    ]
    # "Hold the audit open for up to three days": the gate parks for 3 days.
    (gate,) = step["Step 6"]
    assert gate["kind"] == "hitl_gate" and "mandatory" not in gate
    assert gate["signal"] == "controller_review_approved"
    # "wait for a decision on every review item: approve or reject"
    assert gate["output"] == {"approved": "Any", "rejected": "Any"}
    assert gate["timeout"] == "3d"
    # "more than 60 days" and "60 days" are one constant, not two.
    assert list(step["Step 3"][0]["constants"].values()) == [60]


def test_rule_tables_are_exact_code(tmp_path: Path) -> None:
    system1.prepare(EXAMPLES / "expense-audit" / "skill", tmp_path)
    spec = importlib.util.spec_from_file_location(
        "s1_tables", tmp_path / "extracted" / "rule_tables.py"
    )
    assert spec is not None and spec.loader is not None
    tables = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tables)
    fx = importlib.util.spec_from_file_location(
        "expected_fx", EXAMPLES / "expense-audit" / "expected" / "extracted" / "fx.py"
    )
    assert fx is not None and fx.loader is not None
    expected = importlib.util.module_from_spec(fx)
    fx.loader.exec_module(expected)
    # The generated FX table equals the hand-written one, value for value.
    assert tables.FX_RATES_USD_PER_UNIT == expected.MONTH_END_RATES
    caps = yaml.safe_load((EXAMPLES / "expense-audit" / "expected" / "pipeline.yaml").read_text())
    want = next(n for n in caps["nodes"] if n["id"] == "audit_report")["constants"][
        "category_caps_usd"
    ]
    assert sorted(float(v) for v in tables.CATEGORY_CAPS_CAP.values()) == sorted(want.values())


def test_generated_rule_table_test_passes(tmp_path: Path) -> None:
    """Run it the way verification does: its own process, the work dir as the root."""
    system1.prepare(EXAMPLES / "expense-audit" / "skill", tmp_path)
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "tests/test_rule_tables.py",
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert "1 passed" in done.stdout


def test_enforce_restores_decisions_and_keeps_text(tmp_path: Path) -> None:
    system1.prepare(EXAMPLES / "expense-audit" / "skill", tmp_path)
    doc = yaml.safe_load((tmp_path / "pipeline.yaml").read_text())
    first, second = doc["nodes"][0], doc["nodes"][1]
    first["kind"] = "llm_judge"  # a decision: must come back
    first["inputs"] = {"oops": "pipeline.input"}  # a decision: must come back
    first["description"] = "Rewritten by the agent."  # text: must stay
    first["output"] = {"converted": "list[Expense]"}  # text on a plain output: stays
    removed = doc["nodes"].pop(1)
    doc["nodes"].append({"id": "extra_node", "kind": "pure_function", "description": "x"})
    (tmp_path / "pipeline.yaml").write_text(yaml.safe_dump(doc, sort_keys=False))

    changes = enforce(tmp_path)

    after = load_pipeline(tmp_path / "pipeline.yaml")
    by_id = {n.id: n for n in after.nodes}
    assert by_id[first["id"]].kind.value == "pure_function"
    assert by_id[first["id"]].inputs == {"request": "pipeline.input"}
    assert by_id[first["id"]].description == "Rewritten by the agent."
    assert by_id[first["id"]].output == {"converted": "list[Expense]"}
    assert removed["id"] in by_id and second["id"] == removed["id"]
    assert "extra_node" not in by_id
    assert any("kind" in c for c in changes)
    assert any("removed node `extra_node`" in c for c in changes)
    assert any(f"restored node `{removed['id']}`" in c for c in changes)
    assert system1.all_changes(tmp_path) == changes


def test_enforce_keeps_skeleton_output_keys_for_fan_out_sources(tmp_path: Path) -> None:
    system1.prepare(EXAMPLES / "expense-audit" / "skill", tmp_path)
    doc = yaml.safe_load((tmp_path / "pipeline.yaml").read_text())
    group = next(n for n in doc["nodes"] if n["output"] == {"items": "list"})
    group["output"] = {"reports": "list[Report]"}
    (tmp_path / "pipeline.yaml").write_text(yaml.safe_dump(doc, sort_keys=False))
    enforce(tmp_path)
    after = {n.id: n for n in load_pipeline(tmp_path / "pipeline.yaml").nodes}
    assert after[group["id"]].output == {"items": "list"}


def test_proposals_are_checked_never_applied(tmp_path: Path) -> None:
    system1.prepare(EXAMPLES / "expense-audit" / "skill", tmp_path)
    skeleton = (tmp_path / "pipeline.yaml").read_text()
    node = yaml.safe_load(skeleton)["nodes"][0]["id"]
    (tmp_path / "system1" / "proposals.yaml").write_text(
        yaml.safe_dump(
            [
                {"node": node, "field": "mandatory", "value": True, "reason": "payroll"},
                {"node": node, "field": "description", "value": "x", "reason": "y"},
                {"node": "nope", "field": "kind", "value": "llm_judge", "reason": "z"},
            ]
        )
    )
    verdicts = check_proposals(tmp_path)
    assert "valid IR if applied" in verdicts[0]
    assert "not a decision field" in verdicts[1]
    assert "no such node" in verdicts[2]
    assert (tmp_path / "pipeline.yaml").read_text() == skeleton


def test_text_rubric_is_system1_md_and_two_references(tmp_path: Path) -> None:
    bundle = system1.materialize_text_rubric(REPO / "skills" / "rote-compile", tmp_path / "rubric")
    assert (bundle / "SKILL.md").read_text() == (
        REPO / "skills" / "rote-compile" / "SYSTEM1.md"
    ).read_text()
    assert sorted(p.name for p in (bundle / "references").iterdir()) == sorted(
        system1.TEXT_RUBRIC_REFERENCES
    )


class _TextAgent(CompilerDriver):
    """Writes what the text agent writes, and also edits a decision it may not."""

    name = "fake-text"

    def __init__(self) -> None:
        self.rubric: Path | None = None

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
        self.rubric = compiler_skill_dir
        doc = yaml.safe_load((work_dir / "pipeline.yaml").read_text())
        for n in doc["nodes"]:
            if "impl" in n:
                path, fn = n["impl"].split(":")
                params = [*n["inputs"], *(n.get("constants") or {})]
                (work_dir / path).parent.mkdir(exist_ok=True)
                (work_dir / path).write_text(
                    f"def {fn}({', '.join(params)}):\n    return {{'items': []}}\n"
                )
        doc["nodes"][0]["fan_out"] = True  # not the agent's to decide
        (work_dir / "pipeline.yaml").write_text(yaml.safe_dump(doc, sort_keys=False))
        (work_dir / "compile-report.md").write_text("# Report\n\n## Open questions\n\n- none\n")
        return DriverResult(work_dir / "pipeline.yaml", work_dir, self.name, {"cost_usd": 0.0})


@pytest.mark.asyncio
async def test_compile_system1_runs_text_agent_under_the_small_rubric(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    skill = tmp_path / "skill"
    shutil.copytree(EXAMPLES / "expense-audit" / "skill", skill)
    agent = _TextAgent()
    monkeypatch.setattr(Compiler, "select_driver", lambda self: agent)
    out = tmp_path / "out"
    result = await Compiler(system1=True, run_generated_tests=False).compile(skill, out)

    assert agent.rubric is not None and agent.rubric.name.startswith("rote-system1-rubric-")
    first = result.pipeline.nodes[0]
    assert first.fan_out is False  # restored after the agent's edit
    report = (out / "compile-report.md").read_text()
    assert "## Open questions" in report
    assert (
        f"The agent edited a decision; rote restored it: `{first.id}` fan_out: "
        "agent wrote True, kept None"
    ) in report
    assert (out / "system1" / "decisions.json").is_file()


@pytest.mark.asyncio
async def test_system1_refuses_update(tmp_path: Path) -> None:
    with pytest.raises(Exception, match="--system1 does not support --update"):
        await Compiler(system1=True).compile(
            EXAMPLES / "expense-audit" / "skill", tmp_path, update=True
        )


def test_bench_passes_extra_compile_flags(tmp_path: Path) -> None:
    args = _parser().parse_args(["--compile-flag=--system1"])
    argv = compile_argv(tmp_path / "skill", tmp_path / "out", args)
    assert argv[-1] == "--system1"
    plain = compile_argv(tmp_path / "skill", tmp_path / "out", _parser().parse_args([]))
    assert "--system1" not in plain


def test_repair_rounds_verify_the_enforced_pipeline(tmp_path: Path) -> None:
    """A resuming driver's repair callback restores decisions BEFORE it verifies."""
    work = tmp_path / "work"
    work.mkdir()
    system1.prepare(EXAMPLES / "expense-audit" / "skill", work)
    doc = yaml.safe_load((work / "pipeline.yaml").read_text())
    doc["nodes"][0]["fan_out"] = True  # an agent edit the repair round must not see
    (work / "pipeline.yaml").write_text(yaml.safe_dump(doc, sort_keys=False))
    compiler = Compiler(system1=True, run_generated_tests=False)
    repair = compiler._deliverable_repairer(
        None, EXAMPLES / "expense-audit" / "skill", before_verify=enforce
    )
    repair(work / "pipeline.yaml")
    after = yaml.safe_load((work / "pipeline.yaml").read_text())
    assert "fan_out" not in after["nodes"][0]


def test_unparseable_agent_yaml_goes_to_repair_then_falls_back_to_the_skeleton(
    tmp_path: Path,
) -> None:
    """A real compile wrote `description: Reads ... (tab: Dwell Log - Form Responses). R`.

    Enforcement must not crash on it: during repair the file is left for
    verification to report, and the final pass restores the skeleton."""
    system1.prepare(EXAMPLES / "ops-report" / "skill", tmp_path)
    broken = "nodes:\n- id: x\n  description: Reads it (tab: Dwell Log). R: y\n"
    (tmp_path / "pipeline.yaml").write_text(broken)
    assert enforce(tmp_path) == ["pipeline.yaml is not valid YAML; sent back for repair"]
    assert (tmp_path / "pipeline.yaml").read_text() == broken
    (change,) = enforce(tmp_path, final=True)
    assert "did not parse" in change
    assert (tmp_path / "pipeline.yaml").read_text() == (
        tmp_path / "system1" / "skeleton.yaml"
    ).read_text()
    assert (tmp_path / "system1" / "agent-pipeline.invalid.yaml").read_text() == broken
    load_pipeline(tmp_path / "pipeline.yaml")


def test_report_lists_every_assumption_so_no_repair_round_is_spent_on_it(tmp_path: Path) -> None:
    """The commonest repair cause: an assumption with no Open questions entry."""
    from rote.compiler.provenance import open_questions
    from rote.compiler.verify import verify_deliverable

    plan = system1.prepare(EXAMPLES / "expense-audit" / "skill", tmp_path)
    doc = yaml.safe_load((tmp_path / "pipeline.yaml").read_text())
    node = next(n for n in doc["nodes"] if n.get("impl"))
    path, fn = node["impl"].split(":")
    params = ", ".join([*node["inputs"], *(node.get("constants") or {})])
    (tmp_path / path).write_text(
        f"def {fn}({params}):\n"
        "    # rote: assumption: an empty batch returns no lines\n"
        "    # rote: assumption: currency codes are upper case\n"
        "    return {}\n"
    )
    system1.settle(tmp_path, plan, final=True)
    report = (tmp_path / "compile-report.md").read_text()
    entries = open_questions(report)
    assert sum("assumed" in e for e in entries) == 2
    v = verify_deliverable(
        tmp_path, run_tests=False, skill_dir=EXAMPLES / "expense-audit" / "skill"
    )
    assert not [f for f in v.findings if f.kind == "unlisted_assumption"]


def test_text_yaml_carries_judge_specs_without_editing_pipeline_yaml(tmp_path: Path) -> None:
    system1.prepare(EXAMPLES / "deal-monitor" / "skill", tmp_path)
    doc = yaml.safe_load((tmp_path / "pipeline.yaml").read_text())
    judge = next(n for n in doc["nodes"] if n["kind"] == "llm_judge")
    spec = {
        "input_schema": {"type": "object", "properties": {k: {} for k in judge["inputs"]}},
        "output_schema": {"type": "object", "properties": {"step": {"type": "integer"}}},
        "prompt": "Classify: {{ request }}",
        "client": "anthropic",
    }
    (tmp_path / "system1" / "text.yaml").write_text(
        yaml.safe_dump({judge["id"]: {"signature_spec": spec, "kind": "pure_function"}})
    )
    enforce(tmp_path)
    after = {n.id: n for n in load_pipeline(tmp_path / "pipeline.yaml").nodes}
    assert after[judge["id"]].signature_spec is not None
    assert after[judge["id"]].signature_spec.prompt == "Classify: {{ request }}"
    assert after[judge["id"]].kind.value == "llm_judge"  # text.yaml cannot carry a decision


def _tiny_skill(tmp_path: Path, body: str) -> Path:
    skill = tmp_path / "tiny"
    skill.mkdir()
    (skill / "SKILL.md").write_text(f"---\nname: tiny\ndescription: x\n---\n\n# Tiny\n\n{body}\n")
    return skill


def test_conditional_wording_is_the_only_way_out_of_mandatory(tmp_path: Path) -> None:
    skill = _tiny_skill(
        tmp_path,
        "## Step 1: Total the batch\n\nTotal every line in USD.\n\n"
        "## Step 2: Format the table\n\nOptionally format the totals as a table.\n",
    )
    out = tmp_path / "work"
    out.mkdir()
    doc = yaml.safe_load((system1.prepare(skill, out) and out / "pipeline.yaml").read_text())
    assert [n.get("mandatory", False) for n in doc["nodes"]] == [True, False]


def test_gate_payload_is_what_the_step_asks_for(tmp_path: Path) -> None:
    _, doc = _skeleton(tmp_path, "ops-report")
    (gate,) = [n for n in doc["nodes"] if n["kind"] == "hitl_gate"]
    assert list(gate["output"]) == [
        "dock_appointment_completion",
        "sorter_metrics",
        "current_dwell",
    ]
    assert gate["signal"].endswith("_provided")


def test_input_fields_come_from_the_skills_own_words(tmp_path: Path) -> None:
    _, bdr = _skeleton(tmp_path / "b", "bdr-outreach")
    props = bdr["input"]["input_schema"]["properties"]
    # "Confirm with the user: drug name (brand + generic), condition ..., therapeutic area,
    # manufacturer, campaign type (...), and optional job focus"
    assert {"therapeutic_area", "manufacturer", "campaign_type", "job_focus"} <= set(props)
    _, invoice = _skeleton(tmp_path / "i", "invoice-push")
    # `[start date] - [end date]` is computed by the skill, inside code: not an input.
    assert "properties" not in invoice["input"]["input_schema"]


def test_per_item_container_becomes_a_loop_over_its_sub_steps(tmp_path: Path) -> None:
    """invoice-push "Step 7: Process Each Invoice Row" / "For each visible row:" + 7a..7g.

    Run once, the sub-steps would push one invoice instead of every row: a
    behavioral difference, not a naming one."""
    _, doc = _skeleton(tmp_path, "invoice-push")
    nodes = {n["id"]: n for n in doc["nodes"]}
    (loop,) = [n for n in doc["nodes"] if n["kind"] == "agent_loop"]
    assert loop["source"]["section"].startswith("Step 7")
    body = loop["loop_body"]
    assert len(body) >= 7
    prefixes = {"7a", "7b", "7c", "7d", "7e", "7f", "7g"}
    assert all(nodes[b]["source"]["section"][:2] in prefixes for b in body)
    # Nothing outside the loop reads a body node; the step after reads the loop.
    outside = [n for n in doc["nodes"] if n["id"] not in body]
    assert not [n for n in outside for ref in n["inputs"].values() if ref.split(".")[0] in body]
    after = next(n for n in outside if n["source"]["section"].startswith("Step 8"))
    assert loop["id"] in after["inputs"]
    load_pipeline(tmp_path / "pipeline.yaml")


def test_descriptive_sub_headings_and_checklists_are_not_steps(tmp_path: Path) -> None:
    _, doc = _skeleton(tmp_path, "invoice-push")
    sections = [n["source"]["section"] for n in doc["nodes"]]
    # "Tab 1 / Tab 2" describe the report Step 9 writes.
    assert not [s for s in sections if s.startswith("Tab ")]
    # "1. **Browser-automation plugin** 2. **User is logged in**" is one check.
    assert sections.count("Prerequisites Check") == 1
