"""Regression test: the compiler's output on BDR vs. hand-drafted expected/.

This is the moment-of-truth test for rote. It compares the
compiler agent's real output against the hand-drafted
``examples/bdr-outreach/expected/pipeline.yaml``, which we treat as
the ground truth.

# Why this test is different from the others

Every other test in the suite mocks the LLM. This one does not — it
reads a previously-captured `pipeline.yaml` produced by a real
compiler run and asserts that the compiler made sensible
classifications for the BDR skill. Running a fresh compilation on
every pytest invocation would:

* Cost real money (or subscription budget)
* Take 5–10 minutes per run
* Be flaky across rubric iterations

Instead, we commit snapshots of real runs under
``examples/bdr-outreach/runs/<timestamp>/`` and compare those to the
hand-drafted expected. When the rubric is updated we re-compile
manually, commit a new snapshot, and adjust the test assertions if
needed.

# What "pass" means

The test intentionally does *semantic* comparison, not textual.
Reasonable agents can make different reasonable judgment calls —
what matters is whether the classification structure is sensible:

* All 5 node kinds appear somewhere
* The MANDATORY exclusion checks are marked ``mandatory: true``
* HITL gates are present with the right number of signals
* The pipeline actually loads via ``rote.ir.load_pipeline``
* Every ``external_call`` / ``pure_function`` node references an
  ``impl:`` that points at a real file in the compiled output
* Every ``llm_judge`` node references a ``signature:`` that points
  at a real file in the compiled output

We are NOT asserting:

* Exact node IDs (the agent might name ``lead_generation_loop`` as
  ``zoominfo_search_loop`` — both are fine)
* Exact node count (the agent might split ``exclusion_check`` into
  three separate nodes or keep them merged)
* Exact field values (timeouts, retry policies, etc.)

# Adding a new snapshot

1. Run: ``rote compile examples/bdr-outreach/skill --runtime temporal
   --out /tmp/bdr-fresh --agent claude --model claude-sonnet-4-6``
2. Inspect the output. If it looks reasonable:
   ``cp -r /tmp/bdr-fresh/compiled examples/bdr-outreach/runs/<date>``
3. Re-run this test file. If it fails on legitimate improvements
   (e.g., the new run is BETTER than the previous snapshot), update
   the assertions and explain why in the commit message.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rote.ir import NodeKind, load_pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
BDR_RUNS_DIR = REPO_ROOT / "examples" / "bdr-outreach" / "runs"
BDR_EXPECTED = REPO_ROOT / "examples" / "bdr-outreach" / "expected" / "pipeline.yaml"


def _latest_snapshot() -> Path | None:
    """Return the most recent compiler run snapshot, if any exist."""
    if not BDR_RUNS_DIR.is_dir():
        return None
    snapshots = sorted(
        (p for p in BDR_RUNS_DIR.iterdir() if p.is_dir()),
        reverse=True,
    )
    return snapshots[0] if snapshots else None


@pytest.fixture(scope="module")
def latest_snapshot_dir() -> Path:
    snapshot = _latest_snapshot()
    if snapshot is None:
        pytest.skip(
            "No compiler run snapshots committed yet. Run "
            "`rote compile examples/bdr-outreach/skill --out /tmp/bdr "
            "--agent claude` and copy the output to examples/bdr-outreach/runs/."
        )
    return snapshot


@pytest.fixture(scope="module")
def snapshot_pipeline(latest_snapshot_dir: Path):  # noqa: ANN201
    pipeline_yaml = latest_snapshot_dir / "pipeline.yaml"
    if not pipeline_yaml.is_file():
        pytest.skip(f"Snapshot at {latest_snapshot_dir} is missing pipeline.yaml")
    return load_pipeline(pipeline_yaml)


# ───────── Structural assertions ─────────


def test_snapshot_loads_cleanly(snapshot_pipeline) -> None:  # noqa: ANN001
    """The produced pipeline.yaml validates against the IR schema."""
    assert snapshot_pipeline.name


def test_snapshot_covers_all_five_node_kinds(snapshot_pipeline) -> None:  # noqa: ANN001
    """A well-classified BDR compilation touches all five kinds."""
    present = {n.kind for n in snapshot_pipeline.nodes}
    missing = set(NodeKind) - present
    assert not missing, (
        f"Compiler missed node kinds: {[k.value for k in missing]}. "
        f"BDR has exclusion checks (pure/external), vetting (llm_judge), "
        f"HITL gates, and research loops (agent_loop) — all five should appear."
    )


def test_snapshot_has_at_least_two_hitl_gates(snapshot_pipeline) -> None:  # noqa: ANN001
    """BDR's SKILL.md explicitly marks Phase 3 (contact review) and
    Phase 7 (manual enrollment handoff) as HITL — both should survive
    compilation."""
    gates = snapshot_pipeline.nodes_by_kind(NodeKind.HITL_GATE)
    assert len(gates) >= 2, (
        f"Expected at least 2 HITL gates (Phase 3 review + Phase 7 handoff), "
        f"got {len(gates)}: {[g.id for g in gates]}"
    )


def test_snapshot_marks_mandatory_exclusion_checks(snapshot_pipeline) -> None:  # noqa: ANN001
    """BDR's Phase 5 has three MANDATORY exclusion checks (do-not-contact,
    recently-emailed, active-sequence). The compiler must mark at least
    one of these nodes as ``mandatory: true`` — they're the highest-leverage
    crystallization pattern in the whole skill."""
    mandatory = [n for n in snapshot_pipeline.nodes if n.mandatory]
    assert mandatory, (
        "No nodes marked mandatory: true. BDR's exclusion checks are "
        "explicitly marked MANDATORY in the source skill and should be "
        "codified with that flag."
    )


def test_snapshot_impls_and_signatures_point_at_real_files(
    snapshot_pipeline, latest_snapshot_dir: Path
) -> None:  # noqa: ANN001
    """Every pure_function / external_call node with an ``impl:`` should
    point at a file that actually exists in the compiled output."""
    for node in snapshot_pipeline.nodes:
        if node.kind in (NodeKind.PURE_FUNCTION, NodeKind.EXTERNAL_CALL):
            assert node.impl, f"Node {node.id} has no impl"
            # ``impl:`` is a relative path like "extracted/foo.py:bar"
            file_part = node.impl.split(":", 1)[0]
            target = latest_snapshot_dir / file_part
            assert target.is_file(), (
                f"Node {node.id} impl points at {file_part!r} which doesn't "
                f"exist in the snapshot at {latest_snapshot_dir}"
            )
        elif node.kind is NodeKind.LLM_JUDGE:
            assert node.signature, f"Node {node.id} has no signature"
            file_part = node.signature.split(":", 1)[0]
            target = latest_snapshot_dir / file_part
            assert target.is_file(), (
                f"Node {node.id} signature points at {file_part!r} which "
                f"doesn't exist in the snapshot at {latest_snapshot_dir}"
            )


def test_snapshot_reasonable_codifiable_percentage(snapshot_pipeline) -> None:  # noqa: ANN001
    """The compiler should move a meaningful fraction of BDR's work
    into deterministic code. Too low a number suggests the rubric isn't
    being followed; too high may mean over-crystallization of genuinely
    fuzzy work.

    BDR's hand-drafted expected/ has ~65% codifiable (8 of 12
    non-HITL nodes are pure_function or external_call). The compiler
    should land within a generous window of that.
    """
    non_hitl = [n for n in snapshot_pipeline.nodes if n.kind is not NodeKind.HITL_GATE]
    codifiable = [n for n in non_hitl if n.kind in (NodeKind.PURE_FUNCTION, NodeKind.EXTERNAL_CALL)]
    pct = len(codifiable) / len(non_hitl) * 100 if non_hitl else 0
    assert 30 <= pct <= 90, (
        f"Codifiable percentage {pct:.0f}% is outside the reasonable 30-90% "
        f"window. Lower than 30% suggests the compiler isn't following the "
        f"crystallization heuristics; higher than 90% suggests over-"
        f"crystallization of genuinely fuzzy work. "
        f"({len(codifiable)}/{len(non_hitl)} codifiable nodes)"
    )
