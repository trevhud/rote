"""Slow integration test: the DBOS serve backend against a real app process.

This is the empirical proof of the ``rote serve`` → DBOS operational
contract, using the exact topology production uses:

* The emitted BDR app runs as a **separate OS process** in worker mode
  (``python main.py --serve`` — the same command ``dbos start`` runs),
  with recorder overlays instead of real vendor APIs (same technique as
  ``test_dbos_e2e.py``).
* The test process plays the ``rote serve`` role: it calls the *actual*
  backend functions (:func:`rote.serve.backends.start_workflow` /
  ``workflow_status`` / ``signal_workflow``), which construct a real
  ``DBOSClient`` against the shared system database.

What it proves, in order:

1. ``start_workflow`` enqueues externally and the app process dequeues
   and executes the run (registered workflow name + queue name match the
   emitted code — this is the whole trigger contract).
2. The run executes to the first HITL gate and parks; ``workflow_status``
   reports the non-terminal state.
3. ``signal_workflow`` (``DBOSClient.send`` on the gate's topic) resumes
   the parked run from outside the app process, twice, and the resume
   payload threads into downstream nodes.
4. ``workflow_status`` reports the terminal ``success``.

The system database is SQLite — verified working cross-process with
dbos 2.26 (DBOS labels SQLite "for development and testing"; production
should use Postgres, but the client/app contract is identical).

Gated behind ``@pytest.mark.slow``: spawns a Python subprocess and a
real DBOS runtime.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from rote.adapters._common import _pipeline_hash, _to_pascal_case
from rote.adapters.dbos import DbosAdapter
from rote.ir import Pipeline
from rote.serve import backends
from rote.serve.registry import DbosTrigger, RegistryEntry

from .test_dbos_e2e import (
    _mock_output,
    _pre_and_post_gate_nodes,
    _recorded,
    _write_test_overlay,
)

pytest.importorskip("dbos", reason="dbos not installed (pip install rote[dbos])")

REPO_ROOT = Path(__file__).resolve().parent.parent
BDR_PIPELINE_YAML = REPO_ROOT / "examples" / "bdr-outreach" / "expected" / "pipeline.yaml"

pytestmark = pytest.mark.slow

#: Emitted by main.py's --serve mode once DBOS.launch() has completed —
#: the readiness signal that migrations ran and queue workers are live.
_SERVE_READY_MARKER = "serving: waiting for enqueued runs"


@pytest.fixture(scope="module")
def serve_app(
    bdr_pipeline: Pipeline, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[tuple[RegistryEntry, Path]]:
    """Emit + overlay the BDR app, run it as a ``--serve`` subprocess, and
    build the registry entry `rote register --runtime dbos` would derive."""
    out = tmp_path_factory.mktemp("dbos-serve-e2e")
    record_path = out / "record.jsonl"
    DbosAdapter().emit(bdr_pipeline, out)
    _write_test_overlay(out, bdr_pipeline, record_path)

    system_db_url = f"sqlite:///{out / 'serve-e2e.sqlite'}"
    env = {**os.environ, "DBOS_SYSTEM_DATABASE_URL": system_db_url}
    log_path = out / "app.log"
    with log_path.open("wb") as log:
        proc = subprocess.Popen(
            [sys.executable, "main.py", "--serve"],
            cwd=out,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    try:
        # Wait for launch: the client cannot enqueue before the app has
        # migrated the system database and started its queue workers.
        deadline = time.time() + 60.0
        while _SERVE_READY_MARKER not in log_path.read_text(encoding="utf-8", errors="replace"):
            if proc.poll() is not None:
                raise AssertionError(
                    f"app process exited rc={proc.returncode} before ready:\n"
                    f"{log_path.read_text(encoding='utf-8', errors='replace')}"
                )
            if time.time() > deadline:
                raise AssertionError(
                    f"app process not ready after 60s:\n"
                    f"{log_path.read_text(encoding='utf-8', errors='replace')}"
                )
            time.sleep(0.2)

        # The trigger coordinates exactly as `rote register --runtime dbos`
        # derives them: versioned workflow name, '<name>-queue', IR gates.
        entry = RegistryEntry(
            name="bdr-campaign",
            description="BDR outreach campaign pipeline",
            pipeline_yaml=str(BDR_PIPELINE_YAML),
            input_schema={"type": "object"},
            trigger=DbosTrigger(
                system_database_url=system_db_url,
                workflow_name=(
                    f"{_to_pascal_case(bdr_pipeline.name)}_{_pipeline_hash(bdr_pipeline)}"
                ),
                queue_name=f"{bdr_pipeline.name}-queue",
                gate_signals=["contact_review_approved", "bdr_enrollment_complete"],
            ),
        )
        yield entry, record_path
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)


async def _wait_for_node_set(
    record_path: Path, expected: set[str], *, timeout: float = 60.0
) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if {r["node"] for r in _recorded(record_path)} >= expected:
            return
        await asyncio.sleep(0.2)
    raise AssertionError(
        f"timeout waiting for nodes {sorted(expected)}; "
        f"recorded so far: {sorted({r['node'] for r in _recorded(record_path)})}"
    )


async def _wait_for_status(
    entry: RegistryEntry, workflow_id: str, terminal: str, *, timeout: float = 60.0
) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        polled = await backends.workflow_status(entry, workflow_id)
        if polled["status"] == terminal:
            return
        await asyncio.sleep(0.2)
    raise AssertionError(f"workflow {workflow_id} never reached status {terminal!r}")


@pytest.mark.asyncio
async def test_serve_backend_drives_bdr_through_hitl_gates(
    serve_app: tuple[RegistryEntry, Path], bdr_pipeline: Pipeline
) -> None:
    entry, record_path = serve_app
    pre_gate, post_gate = _pre_and_post_gate_nodes(bdr_pipeline)

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

    # ── Trigger: external enqueue reaches the separate app process ──
    started = await backends.start_workflow(entry, brief)
    assert started["status"] == "started"
    assert started["runtime"] == "dbos"
    workflow_id = started["workflow_id"]
    assert workflow_id.startswith("bdr-campaign-")

    # ── The app executes to the first gate and parks ──
    await _wait_for_node_set(record_path, pre_gate)
    polled = await backends.workflow_status(entry, workflow_id)
    # PENDING while parked on DBOS.recv (there is no distinct "waiting"
    # state); the recorded step set is the authoritative parking signal.
    assert polled == {"workflow_id": workflow_id, "status": "pending", "runtime": "dbos"}

    # ── Resume gate 1 from outside the app process ──
    resume_1 = {"approved_contacts": [{"id": "c1"}]}
    signaled = await backends.signal_workflow(
        entry, workflow_id, "contact_review_approved", resume_1
    )
    assert signaled["status"] == "signaled"

    # ── Post-gate waves run; the run parks at gate 2 ──
    await _wait_for_node_set(record_path, pre_gate | post_gate)
    polled = await backends.workflow_status(entry, workflow_id)
    assert polled["status"] == "pending"

    # ── Resume gate 2 and reach the terminal state ──
    await backends.signal_workflow(
        entry, workflow_id, "bdr_enrollment_complete", {"enrolled": True}
    )
    await _wait_for_status(entry, workflow_id, "success")

    # ── The gate resume payload threaded into downstream nodes ──
    payloads = {r["node"]: r["payload"] for r in _recorded(record_path)}
    assert payloads["target_research"] == {"brief": brief}
    assert payloads["hubspot_upsert"] == {"contacts": resume_1["approved_contacts"]}
    assert payloads["exclusion_check_dnc"] == {
        "contacts": _mock_output("hubspot_upsert")["upserted"]
    }
