"""End-to-end test: emit → register → run the BDR workflow in Temporal.

This is the load-bearing milestone test. It proves that rote can take
the BDR skill (via its hand-drafted IR), emit Temporal code, and that
the emitted code actually runs to completion inside Temporal's
time-skipping test environment.

The real activity implementations are stubs that raise
NotImplementedError — they document the MCP origin but don't call
real APIs. For this test we replace each activity with a mock that
returns a canned value, registers under the same
``@activity.defn(name=...)``, and lets us assert the workflow
orchestration works independently of the activity bodies.

This is exactly the level of testing that would exist in a production
pipeline: the orchestration is tested with mocked I/O; the I/O is
tested in isolation against the real vendor APIs.
"""

from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

import pytest
from temporalio import activity
from temporalio.client import WorkflowFailureError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import UnsandboxedWorkflowRunner, Worker

from tests._helpers import FAN_OUT_ELEMENTS, fan_out_element

REPO_ROOT = Path(__file__).resolve().parent.parent
BDR_EXAMPLE_PKG_ROOT = REPO_ROOT / "examples" / "bdr-outreach"


# ───────── Mocked activities (shadowing the real stubs) ─────────
#
# The emitted activities.py calls ``from expected.extracted.taxonomy
# import resolve_taxonomy_ids`` which raises NotImplementedError. For
# the end-to-end test we register a parallel set of mock activities
# under the same activity names. Temporal dispatches by name, so the
# workflow will resolve to the mocks.
#
# Each mock records the payload it receives so the test can assert the
# workflow threads real data between activities (data-flow threading),
# not just that the orchestration completes.

CAPTURED_PAYLOADS: dict[str, dict] = {}
#: Every (node, payload) in call order. CAPTURED_PAYLOADS is keyed by
#: node so it keeps only the last call — structurally unable to see a
#: fan_out node's N invocations.
CAPTURE_LOG: list[tuple[str, dict]] = []


def _capture(name: str, payload: dict) -> None:
    CAPTURED_PAYLOADS[name] = payload
    CAPTURE_LOG.append((name, payload))


@activity.defn(name="target_research")
async def mock_target_research(payload: dict) -> dict:
    _capture("target_research", payload)
    return {
        "pipeline_summary": "Mocked pipeline summary for the test campaign.",
        "rwe_signals": ["signal_a", "signal_b"],
        "key_programs": ["program_x"],
        "prior_interactions": "",
        "relevant_experience": "",
        "messaging_angles": ["angle_1"],
    }


@activity.defn(name="taxonomy_lookup")
async def mock_taxonomy_lookup(payload: dict) -> dict:
    return {
        "vp_level_id": 1,
        "director_level_id": 2,
        "pharma_industry_id": 10,
        "biotech_industry_id": 11,
        "medical_dept_id": 20,
    }


@activity.defn(name="lead_generation_loop")
async def mock_lead_generation_loop(payload: dict) -> dict:
    return {
        "vetted_contacts": [],
        "discarded_summary": {
            "total_searched": 0,
            "total_kept": 0,
            "total_discarded": 0,
            "discard_counts": {},
            "tier_breakdown": {},
        },
    }


@activity.defn(name="enrich_contact_batch")
async def mock_enrich_contact_batch(payload: dict) -> dict:
    # Never called directly by the workflow in this test — only from
    # inside lead_generation_loop — but registered so the worker accepts
    # it if the workflow ever does dispatch it.
    return {"enriched": []}


@activity.defn(name="vet_contact")
async def mock_vet_contact(payload: dict) -> dict:
    return {
        "decision": "keep",
        "tier": "strong",
        "discard_reason": None,
        "relevance_evidence": "mocked",
    }


@activity.defn(name="hubspot_upsert")
async def mock_hubspot_upsert(payload: dict) -> dict:
    _capture("hubspot_upsert", payload)
    return {"upserted": [{"vid": "hs-1"}]}


@activity.defn(name="hubspot_create_list")
async def mock_hubspot_create_list(payload: dict) -> dict:
    return {"list_id": "list-abc123"}


@activity.defn(name="exclusion_check_dnc")
async def mock_exclusion_check_dnc(payload: dict) -> dict:
    _capture("exclusion_check_dnc", payload)
    return {"passed": [], "excluded": []}


@activity.defn(name="exclusion_check_recent")
async def mock_exclusion_check_recent(payload: dict) -> dict:
    return {"passed": [], "excluded": []}


@activity.defn(name="exclusion_check_sequence")
async def mock_exclusion_check_sequence(payload: dict) -> dict:
    # Non-empty: personalize_email fans over this list, and an empty
    # one makes the fan a correct no-op that no assertion can catch.
    return {
        "passed": [fan_out_element(i) for i in range(FAN_OUT_ELEMENTS)],
        "excluded": [],
    }


@activity.defn(name="personalize_email")
async def mock_personalize_email(payload: dict) -> dict:
    _capture("personalize_email", payload)
    return {"opening_line": "Hi there,", "ta_callout": "rare disease"}


@activity.defn(name="create_sales_template")
async def mock_create_sales_template(payload: dict) -> dict:
    return {"template_ids": ["t1", "t2"]}


@activity.defn(name="pre_enrollment_report")
async def mock_pre_enrollment_report(payload: dict) -> dict:
    _capture("pre_enrollment_report", payload)
    return {"report_markdown": "# Pre-Enrollment Report\n\nMocked."}


MOCK_ACTIVITIES = [
    mock_target_research,
    mock_taxonomy_lookup,
    mock_lead_generation_loop,
    mock_enrich_contact_batch,
    mock_vet_contact,
    mock_hubspot_upsert,
    mock_hubspot_create_list,
    mock_exclusion_check_dnc,
    mock_exclusion_check_recent,
    mock_exclusion_check_sequence,
    mock_personalize_email,
    mock_create_sales_template,
    mock_pre_enrollment_report,
]


# ───────── Fixtures ─────────


@pytest.fixture(scope="module")
def bdr_workflow_class():  # noqa: ANN201
    """Import the emitted BdrCampaignWorkflow.

    The emitted file lives outside the src/rote package, so we have to
    put the example root on sys.path so its ``expected.*`` imports
    resolve. We also need to ensure the worker is configured with the
    UnsandboxedWorkflowRunner because the default sandbox restricts
    imports to a known-good allowlist and would reject
    ``expected.extracted.*``.
    """
    sys.path.insert(0, str(BDR_EXAMPLE_PKG_ROOT))
    try:
        # Ensure clean import
        for mod in list(sys.modules):
            if mod.startswith("expected."):
                del sys.modules[mod]
        from expected.runtimes.temporal.workflow import (  # type: ignore[import-not-found]
            BdrCampaignWorkflow,
        )

        yield BdrCampaignWorkflow
    finally:
        sys.path.remove(str(BDR_EXAMPLE_PKG_ROOT))


# ───────── The end-to-end test ─────────


@pytest.mark.asyncio
async def test_bdr_workflow_runs_to_completion(bdr_workflow_class) -> None:  # noqa: ANN001
    """Run the full BDR workflow with mocked activities.

    The workflow has two HITL gates. The test:

    1. Starts the workflow with a complete campaign brief (the emitted
       workflow now threads real payloads, so every referenced field
       must exist).
    2. Waits for the first gate by polling the workflow's status until
       it's blocked on the first signal.
    3. Sends ``contact_review_approved`` to unblock phase 3.
    4. Waits for the second gate.
    5. Sends ``bdr_enrollment_complete`` to unblock phase 7.
    6. Awaits the final result and asserts the exit node's payload is
       what we signaled — and that data-flow threading delivered the
       right payloads to the mocked activities along the way.
    """
    CAPTURED_PAYLOADS.clear()
    CAPTURE_LOG.clear()

    # A complete brief matching the pipeline's input contract. Values
    # reuse the fictionalized examples already present in the IR's
    # comments.
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

    async with await WorkflowEnvironment.start_time_skipping() as env:
        task_queue = f"rote-bdr-test-{uuid4()}"

        async with Worker(
            env.client,
            task_queue=task_queue,
            workflows=[bdr_workflow_class],
            activities=MOCK_ACTIVITIES,
            # The default sandbox blocks imports of non-stdlib modules.
            # Our workflow is pure — it only uses asyncio and temporalio
            # — so unsandboxed is safe and simplifies the test.
            workflow_runner=UnsandboxedWorkflowRunner(),
        ):
            # Kick off the workflow. Note: start_workflow uses the
            # @workflow.defn(name=...) registered name, not the Python
            # class name.
            handle = await env.client.start_workflow(
                bdr_workflow_class.run,
                brief,
                id=f"bdr-campaign-{uuid4()}",
                task_queue=task_queue,
            )

            # ─── First HITL signal ───
            # Don't wait for the workflow to "reach" the gate — signals
            # are buffered by Temporal until the workflow reads them,
            # so we can send early.
            await handle.signal(
                bdr_workflow_class.contact_review_approved,
                {"approved_contacts": [{"id": "c1"}], "reviewer": "test"},
            )

            # ─── Second HITL signal ───
            await handle.signal(
                bdr_workflow_class.bdr_enrollment_complete,
                {"enrolled": True, "enrolled_count": 1},
            )

            # ─── Await completion ───
            try:
                result = await handle.result()
            except WorkflowFailureError as e:
                pytest.fail(f"Workflow failed: {e}\n  cause: {e.cause}")

            # The workflow returns a dict keyed by exit node IDs.
            assert "manual_enrollment_handoff" in result
            assert result["manual_enrollment_handoff"] == {
                "enrolled": True,
                "enrolled_count": 1,
            }

    # ─── Data-flow threading assertions ───
    # The pipeline input reached the entry node intact.
    assert CAPTURED_PAYLOADS["target_research"] == {"brief": brief}

    # The first HITL gate's signal payload flowed into hubspot_upsert
    # via `contacts: contact_review_gate.output.approved_contacts`.
    assert CAPTURED_PAYLOADS["hubspot_upsert"] == {"contacts": [{"id": "c1"}]}

    # hubspot_upsert's mocked result flowed into the DNC check via
    # `contacts: hubspot_upsert.output.upserted`.
    assert CAPTURED_PAYLOADS["exclusion_check_dnc"] == {"contacts": [{"vid": "hs-1"}]}

    # The report node received a fan-in of upstream results:
    # pipeline input field + two different upstream nodes.
    report_payload = CAPTURED_PAYLOADS["pre_enrollment_report"]
    assert report_payload["campaign_name"] == brief["drug_brand"]
    assert report_payload["passed_contacts"] == [
        fan_out_element(i) for i in range(FAN_OUT_ELEMENTS)
    ]
    assert report_payload["template_ids"] == ["t1", "t2"]

    # ─── fan_out: personalize_email ran once per surviving contact ───
    # CAPTURED_PAYLOADS is keyed by node, so it can only ever show one
    # invocation; the ordered log is what makes a fan observable.
    fan_calls = [p for (n, p) in CAPTURE_LOG if n == "personalize_email"]
    assert len(fan_calls) == FAN_OUT_ELEMENTS, (
        f"personalize_email is a fan_out node over "
        f"exclusion_check_sequence.passed; expected {FAN_OUT_ELEMENTS} "
        f"activity executions, got {len(fan_calls)}"
    )
    # Each invocation got ONE contact, not the whole list, and the
    # non-fanned inputs are shared verbatim across all of them.
    assert sorted(c["contact"]["fanElement"] for c in fan_calls) == list(range(FAN_OUT_ELEMENTS))
    assert {c["campaign_type"] for c in fan_calls} == {brief["campaign_type"]}
