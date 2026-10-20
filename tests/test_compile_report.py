"""compile-report.md is rendered by rote from the deliverable.

The agent used to write it, which cost 17 to 66 seconds of output per
compile and a repair round whenever its Open questions under-counted the
code's assumption markers. Every fact it held is in pipeline.yaml or the
extracted code, so these tests pin that each one reaches the report.
"""

from __future__ import annotations

from pathlib import Path

from rote.compiler.provenance import open_questions
from rote.compiler.report import render_compile_report
from rote.ir import load_pipeline

PIPELINE = """\
name: returns
input:
  type: Batch
  required: [returns]
entry_nodes: [route]
exit_nodes: [notify]
nodes:
  - id: route
    kind: pure_function
    description: Route each return by the skill's | table of reasons.
    impl: extracted/route.py:route
    inputs: {returns: pipeline.input.returns}
    mandatory: true
    source: {section: Review}
  - id: approve
    kind: hitl_gate
    description: A manager approves the routed batch.
    signal: batch_approved
    timeout: 2d
    inputs: {routed: route.output}
  - id: notify
    kind: external_call
    description: Post the batch to the finance channel.
    impl: extracted/notify.py:notify
    inputs: {routed: route.output}
edges:
  - {from: route, to: approve}
  - {from: approve, to: notify}
"""

ROUTE = '''\
"""Route returns by reason."""


def route(returns):
    # skill: "A return over $50.00 without a receipt goes to manual review"
    if returns and returns[0]["amount"] > 50:
        return {"route": "manual_review"}
    # rote: assumption: an unlisted reason is refunded at once
    return {"route": "refund"}
'''

NOTIFY = """\
def notify(routed):
    raise NotImplementedError("the skill never names the finance channel's API")
"""


def _deliverable(root: Path) -> Path:
    for rel, text in {
        "pipeline.yaml": PIPELINE,
        "extracted/route.py": ROUTE,
        "extracted/notify.py": NOTIFY,
    }.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")
    return root


def _render(root: Path) -> str:
    return render_compile_report(load_pipeline(root / "pipeline.yaml"), root)


def test_the_summary_counts_every_kind_and_names_the_gate(tmp_path: Path) -> None:
    report = _render(_deliverable(tmp_path))
    assert report.startswith("# Compile report: returns\n")
    assert (
        "- 3 node(s): 1 `pure_function`, 1 `external_call`, 0 `llm_judge`, "
        "0 `agent_loop`, 1 `hitl_gate`." in report
    )
    assert "- Codified (`pure_function` + `external_call`): 2 of 2 non-gate node(s), 100%." in (
        report
    )
    assert "- Implemented impl node(s): 1 of 2." in report
    assert "- Mandatory: `route`." in report
    assert (
        "- HITL gate `approve` (signal `batch_approved`, timeout 2d): "
        "A manager approves the routed batch." in report
    )


def test_each_node_is_a_row_with_its_status_section_and_reason(tmp_path: Path) -> None:
    report = _render(_deliverable(tmp_path))
    assert (
        "| `route` | pure_function | `extracted/route.py:route` (implemented) | Review "
        "| Route each return by the skill's \\| table of reasons. |" in report
    )
    assert "| `notify` | external_call | `extracted/notify.py:notify` (stub) |" in report
    assert "| `approve` | hitl_gate | signal `batch_approved` |" in report


def test_the_crystallization_log_quotes_each_citation(tmp_path: Path) -> None:
    report = _render(_deliverable(tmp_path))
    assert "- `extracted/route.py`: Route returns by reason. (1 skill citation(s))" in report
    assert '  - line 5: "A return over $50.00 without a receipt goes to manual review"' in report
    assert "- `extracted/notify.py` (0 skill citation(s))" in report


def test_open_questions_hold_every_assumption_and_every_stub(tmp_path: Path) -> None:
    report = _render(_deliverable(tmp_path))
    assert open_questions(report) == [
        "### OQ-1: route",
        "### Stub: notify (`extracted/notify.py:notify`, stub)",
    ]
    assert "an unlisted reason is refunded at once" in report
    assert "Marked `# rote: assumption:` at `extracted/route.py:8`." in report
    assert "the skill never names the finance channel's API" in report


def test_nothing_to_question_says_so_and_counts_zero(tmp_path: Path) -> None:
    root = _deliverable(tmp_path)
    (root / "extracted/route.py").write_text(
        ROUTE.replace("    # rote: assumption: an unlisted reason is refunded at once\n", ""),
        encoding="utf-8",
    )
    (root / "extracted/notify.py").write_text("def notify(routed):\n    return routed\n")
    report = _render(root)
    assert open_questions(report) == []
    assert "None: the code records no assumption and leaves no stub." in report


def test_rendering_is_a_pure_function_of_the_deliverable(tmp_path: Path) -> None:
    root = _deliverable(tmp_path)
    assert _render(root) == _render(root)
