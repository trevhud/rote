"""Slow integration test: emit → launch → run the BDR workflow on real DBOS.

This is the DBOS analog of ``test_temporal_e2e.py`` /
``test_cloudflare_e2e.py``: it proves the emitted code actually executes
on the real durable-execution machinery, not just that it parses.

Concretely, against a real DBOS 2.x runtime with a SQLite system database
(no Docker/Postgres needed — SQLite is DBOS's supported local-dev mode):

1. Emit the BDR pipeline, then overlay the ``extracted/`` stubs and
   ``signatures/`` judges with recorder mocks (same technique as the
   Cloudflare e2e's ``_write_test_overlay``).
2. Start the workflow; verify the pre-gate steps run in wave order and
   the workflow then *parks* on ``contact_review_gate`` — exactly the
   pre-gate step set recorded, status ``PENDING``, nothing else running.
3. Deliver ``DBOS.send(..., topic="contact_review_approved")`` and watch
   phases 4–7 execute; verify it parks again on
   ``manual_enrollment_handoff``.
4. Deliver the second signal; verify completion and that the gate payload
   flowed into the workflow's exit-node result.
5. Verify data-flow threading empirically: the pipeline input reached
   the entry nodes, the first gate's resume payload flowed into
   ``hubspot_upsert``, and downstream steps received real upstream
   outputs (not ``{}``).
6. Verify durability: the system database's step log contains a
   ``DBOS.recv`` checkpoint per gate (the park survives restarts because
   it is *in the database*, not in memory).

Gated behind ``@pytest.mark.slow`` because it launches a real DBOS
runtime (queue workers, system DB migrations) — a few seconds, plus the
``dbos`` package requirement (``pip install rote[dbos]``).
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import ModuleType

import pytest

from rote.adapters.dbos import DbosAdapter, _extracted_layout
from rote.adapters.temporal import _execution_waves, _impl_path_parts, _to_pascal_case
from rote.ir import NodeKind, Pipeline, load_pipeline

dbos = pytest.importorskip("dbos", reason="dbos not installed (pip install rote[dbos])")

REPO_ROOT = Path(__file__).resolve().parent.parent
BDR_PIPELINE_YAML = REPO_ROOT / "examples" / "bdr-outreach" / "expected" / "pipeline.yaml"

pytestmark = pytest.mark.slow


# ───────── Recorder overlay ─────────
#
# The emitted extracted/ stubs raise NotImplementedError and the
# signatures/ judges call real LLM APIs. For an integration test that
# exercises orchestration (not I/O), each is replaced by a mock that
# appends its node id *and the payload it received* to a JSONL file and
# returns a canned dict shaped like the node's declared output — so the
# test can assert data-flow threading, not just completion (same
# technique as the Temporal e2e's CAPTURED_PAYLOADS).

_RECORDER_PRELUDE = """\
import json
import time

_RECORD_PATH = {record_path!r}


def _record(name, payload):
    with open(_RECORD_PATH, "a") as f:
        f.write(
            json.dumps(
                {{"node": name, "t": time.monotonic(), "payload": payload}},
                default=str,
            )
            + "\\n"
        )
"""

# Canned outputs shaped like each node's declared output, so downstream
# `<node>.output.<field>` references resolve. Nodes absent here have no
# downstream field references and get a generic marker dict.
_MOCK_OUTPUTS: dict[str, dict] = {
    "lead_generation_loop": {"vetted_contacts": [], "discarded_summary": {}},
    "hubspot_upsert": {"upserted": [{"vid": "hs-1"}]},
    "hubspot_create_list": {"list_id": "list-abc123"},
    "exclusion_check_dnc": {"passed": [], "excluded": []},
    "exclusion_check_recent": {"passed": [], "excluded": []},
    "exclusion_check_sequence": {"passed": [], "excluded": []},
    "create_sales_template": {"template_ids": ["t1", "t2"]},
}


def _mock_output(node_id: str) -> dict:
    return _MOCK_OUTPUTS.get(node_id, {"mocked": True, "node": node_id})


def _write_test_overlay(out_dir: Path, pipeline: Pipeline, record_path: Path) -> None:
    # Extracted modules (pure_function / external_call / agent_loop),
    # grouped exactly the way the adapter grouped them.
    for module_name, nodes in _extracted_layout(pipeline).items():
        src = _RECORDER_PRELUDE.format(record_path=str(record_path))
        for node in nodes:
            if node.kind is NodeKind.AGENT_LOOP:
                func = node.id
            else:
                assert node.impl is not None
                _, func = _impl_path_parts(node.impl)
            src += (
                f"\n\ndef {func}(**payload):\n"
                f'    _record("{node.id}", payload)\n'
                f"    return {_mock_output(node.id)!r}\n"
            )
        (out_dir / "extracted" / f"{module_name}.py").write_text(src, encoding="utf-8")

    # Generated signature judges: keep the class/forward/model_dump shape
    # the emitted step calls, minus Pydantic validation and the LLM.
    for node in pipeline.nodes_by_kind(NodeKind.LLM_JUDGE):
        if node.signature_spec is None:
            continue
        pascal = _to_pascal_case(node.id)
        src = _RECORDER_PRELUDE.format(record_path=str(record_path))
        src += (
            f"\n\nclass {pascal}Input:\n"
            f"    def __init__(self, **kwargs):\n"
            f"        self.kwargs = kwargs\n"
            f"\n\nclass _Result:\n"
            f"    def model_dump(self):\n"
            f"        return {_mock_output(node.id)!r}\n"
            f"\n\nclass {pascal}:\n"
            f"    def forward(self, inputs):\n"
            f'        _record("{node.id}", inputs.kwargs)\n'
            f"        return _Result()\n"
        )
        (out_dir / "signatures" / f"{node.id}.py").write_text(src, encoding="utf-8")


def _pre_and_post_gate_nodes(pipeline: Pipeline) -> tuple[set[str], set[str]]:
    """Split the top-level non-HITL nodes into before-first-gate and the
    rest, derived from the execution waves (robust to pipeline edits)."""
    pre: set[str] = set()
    post: set[str] = set()
    seen_gate = False
    for wave in _execution_waves(pipeline):
        for node in wave:
            if node.kind is NodeKind.HITL_GATE:
                seen_gate = True
            elif seen_gate:
                post.add(node.id)
            else:
                pre.add(node.id)
    return pre, post


# ───────── Fixtures ─────────


@pytest.fixture(scope="module")
def bdr_pipeline() -> Pipeline:
    return load_pipeline(BDR_PIPELINE_YAML)


@pytest.fixture(scope="module")
def dbos_app(bdr_pipeline: Pipeline, tmp_path_factory: pytest.TempPathFactory):  # noqa: ANN201
    """Emit + overlay + import the app and launch one DBOS runtime.

    Module-scoped: DBOS is a process-wide singleton, so one launch serves
    every test in the module. Teardown destroys the runtime and undoes
    the sys.path/module mutations.
    """
    out = tmp_path_factory.mktemp("dbos-e2e")
    record_path = out / "record.jsonl"
    DbosAdapter().emit(bdr_pipeline, out)
    _write_test_overlay(out, bdr_pipeline, record_path)

    # The emitted main.py resolves its system DB from this env var; point
    # it at a SQLite file inside the test dir.
    os.environ["DBOS_SYSTEM_DATABASE_URL"] = f"sqlite:///{out / 'e2e.sqlite'}"
    sys.path.insert(0, str(out))
    try:
        spec = importlib.util.spec_from_file_location("bdr_dbos_main", out / "main.py")
        assert spec is not None and spec.loader is not None
        main_module: ModuleType = importlib.util.module_from_spec(spec)
        sys.modules["bdr_dbos_main"] = main_module
        spec.loader.exec_module(main_module)

        from dbos import DBOS

        DBOS.launch()
        yield main_module, record_path
    finally:
        from dbos import DBOS

        DBOS.destroy()
        sys.path.remove(str(out))
        for mod in list(sys.modules):
            if mod in ("bdr_dbos_main", "extracted", "signatures") or mod.startswith(
                ("extracted.", "signatures.")
            ):
                del sys.modules[mod]
        os.environ.pop("DBOS_SYSTEM_DATABASE_URL", None)


# ───────── Helpers ─────────


def _recorded(record_path: Path) -> list[dict]:
    if not record_path.exists():
        return []
    return [json.loads(line) for line in record_path.read_text().splitlines()]


def _wait_for_node_set(
    record_path: Path,
    expected: set[str],
    *,
    timeout: float = 60.0,
    poll: float = 0.2,
) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if {r["node"] for r in _recorded(record_path)} >= expected:
            return
        time.sleep(poll)
    raise AssertionError(
        f"timeout waiting for nodes {sorted(expected)}; "
        f"recorded so far: {sorted({r['node'] for r in _recorded(record_path)})}"
    )


# ───────── The end-to-end test ─────────


def test_bdr_workflow_runs_through_hitl_gates(
    dbos_app,  # noqa: ANN001
    bdr_pipeline: Pipeline,
) -> None:
    from dbos import DBOS

    main_module, record_path = dbos_app
    pre_gate, post_gate = _pre_and_post_gate_nodes(bdr_pipeline)
    assert pre_gate == {"target_research", "taxonomy_lookup", "lead_generation_loop"}

    # A complete brief matching the pipeline's input contract. The emitted
    # workflow now threads real payloads, so every referenced field must
    # exist. Values reuse the fictionalized examples from the IR's comments.
    brief = {
        "drug_brand": "Orladeyo",
        "drug_generic": "berotralstat",
        "condition_full": "hereditary angioedema",
        "condition_acronym": "HAE",
        "therapeutic_area": "rare disease, hematology",
        "manufacturer": "BioCryst Pharmaceuticals",
        "campaign_type": "drug-specific",
        "target_quota": 3,
    }

    handle = DBOS.start_workflow(main_module.run_pipeline, brief)

    # ── Park at gate 1 ──
    _wait_for_node_set(record_path, pre_gate)
    # Stability window: nothing beyond the pre-gate set may run while the
    # workflow is parked on DBOS.recv. (DBOS reports PENDING while parked
    # — like Cloudflare local-dev's "running" — so the step set is the
    # authoritative parking signal, status is corroboration.)
    time.sleep(1.5)
    recorded_names = {r["node"] for r in _recorded(record_path)}
    assert recorded_names == pre_gate, (
        f"workflow should be parked at contact_review_gate after {sorted(pre_gate)}; "
        f"recorded: {sorted(recorded_names)}"
    )
    assert handle.get_status().status == "PENDING"

    # ── Wave order: the loop consumed both wave-1 results before starting ──
    order = [r["node"] for r in _recorded(record_path)]
    assert order.index("lead_generation_loop") > order.index("target_research")
    assert order.index("lead_generation_loop") > order.index("taxonomy_lookup")

    # ── Resume gate 1 with the IR's signal name (catches name drift) ──
    contact_gate = bdr_pipeline.node_by_id("contact_review_gate")
    assert contact_gate.signal == "contact_review_approved"
    DBOS.send(
        handle.workflow_id,
        {"approved_contacts": [{"id": "c1"}]},
        topic=contact_gate.signal,
    )

    # ── Park at gate 2 ──
    _wait_for_node_set(record_path, pre_gate | post_gate)
    time.sleep(1.5)
    recorded_names = {r["node"] for r in _recorded(record_path)}
    assert recorded_names == pre_gate | post_gate
    assert handle.get_status().status == "PENDING"

    # ── Resume gate 2 and complete ──
    handoff_gate = bdr_pipeline.node_by_id("manual_enrollment_handoff")
    assert handoff_gate.signal == "bdr_enrollment_complete"
    handoff_payload = {"enrolled": True, "enrolled_count": 1}
    DBOS.send(handle.workflow_id, handoff_payload, topic=handoff_gate.signal)

    result = handle.get_result()
    assert result == {"manual_enrollment_handoff": handoff_payload}, (
        "the second gate's payload must flow through to the exit-node result"
    )
    assert handle.get_status().status == "SUCCESS"

    # ── Data-flow threading: real payloads reached the steps ──
    payloads = {r["node"]: r["payload"] for r in _recorded(record_path)}

    # The pipeline input reached the entry node intact.
    assert payloads["target_research"] == {"brief": brief}

    # The fan-in loop consumed both wave-1 results plus an input field.
    loop_payload = payloads["lead_generation_loop"]
    assert loop_payload["brief"] == brief
    assert loop_payload["taxonomy"] == _mock_output("taxonomy_lookup")
    assert loop_payload["target_quota"] == brief["target_quota"]

    # The first HITL gate's resume payload flowed into hubspot_upsert via
    # `contacts: contact_review_gate.output.approved_contacts`.
    assert payloads["hubspot_upsert"] == {"contacts": [{"id": "c1"}]}

    # hubspot_upsert's mocked result flowed into the DNC check via
    # `contacts: hubspot_upsert.output.upserted`.
    assert payloads["exclusion_check_dnc"] == {"contacts": [{"vid": "hs-1"}]}

    # The report node received a fan-in of upstream results:
    # pipeline input field + two different upstream nodes.
    report_payload = payloads["pre_enrollment_report"]
    assert report_payload["campaign_name"] == brief["drug_brand"]
    assert report_payload["passed_contacts"] == []
    assert report_payload["template_ids"] == ["t1", "t2"]

    # ── Durability: the gates were checkpointed recv steps in the system
    # database, not in-memory waits. A crashed process would recover the
    # workflow and resume waiting at the same recv.
    step_names = [s["function_name"] for s in DBOS.list_workflow_steps(handle.workflow_id)]
    assert step_names.count("DBOS.recv") == 2, (
        f"expected 2 durable recv checkpoints, got step log: {step_names}"
    )
    # The parallel wave-1 nodes ran as enqueued child workflows: their
    # handles appear in the step log as getResult checkpoints.
    assert any("getResult" in s for s in step_names), step_names
