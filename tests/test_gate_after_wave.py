"""A hitl_gate waits after every other node in its wave, on every runtime.

The IR lets a gate share a wave with reads it does not depend on (a real
ops-report compile put its gate beside the four fetches, taking only the
pipeline input). Every adapter emits the gate's wait after the wave's other
results are in hand, so the person is asked after the data arrives. That
holds only by emission convention (documented on Pipeline.execution_waves),
so this test pins it for each gate-capable runtime: the line that binds a
sibling's finished result comes before the line where the gate waits.

The waves sort by id and ``approve`` sorts before both reads, so an adapter
that emitted wave members in wave order, gate included, fails here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from rote.adapters import get_adapter
from rote.ir import Pipeline

#: Where each runtime's workflow body lives, and the line where its gate
#: starts to wait (the topic, event or condition the wait names).
_RUNTIMES = {
    "dbos": ("main.py", r'topic="approved"'),
    "temporal": ("workflow.py", r"self\._approved_payload is not None"),
    "cloudflare": ("src/workflow.ts", r'type: "approved"'),
    "inngest": ("src/inngest/pipeline.ts", r'event: "gate_wave/approved"'),
    "dbos-ts": ("src/main.ts", r'^\s*"approved",'),
}

_READS = ("fetch_rows", "fetch_totals")


def _pipeline() -> Pipeline:
    reads = [
        {
            "id": rid,
            "kind": "external_call",
            "description": f"read the {rid.removeprefix('fetch_')}",
            "impl": f"extracted/m.py:{rid}",
            "inputs": {"period": "pipeline.input.period"},
            "output": {"data": "list"},
        }
        for rid in _READS
    ]
    return Pipeline.model_validate(
        {
            "name": "gate_wave",
            "input": {"type": "In", "required": ["period"]},
            "nodes": [
                *reads,
                {
                    "id": "approve",
                    "kind": "hitl_gate",
                    "description": "a person approves",
                    "signal": "approved",
                    "output": {"ok": "bool"},
                },
                {
                    "id": "publish",
                    "kind": "external_call",
                    "description": "publish",
                    "impl": "extracted/m.py:publish",
                    "inputs": {
                        "rows": "fetch_rows.output.data",
                        "totals": "fetch_totals.output.data",
                        "ok": "approve.output.ok",
                    },
                },
            ],
            "edges": [
                {"from": "fetch_rows", "to": "publish"},
                {"from": "fetch_totals", "to": "publish"},
                {"from": "approve", "to": "publish"},
            ],
        }
    )


def _first_line(lines: list[str], pattern: str) -> int:
    rx = re.compile(pattern)
    return next(i for i, line in enumerate(lines) if rx.search(line))


def _gate_waits_after_reads(source: str, wait_pattern: str) -> bool:
    """The gate's wait line comes after the first line binding each read's
    finished result (``<id>_result``; a handle or promise is named apart)."""
    lines = source.splitlines()
    wait = _first_line(lines, wait_pattern)
    return all(_first_line(lines, rf"\b{rid}_result\b") < wait for rid in _READS)


def test_the_fixture_puts_the_gate_in_the_reads_wave_first_by_id() -> None:
    waves = [[n.id for n in w] for w in _pipeline().execution_waves()]
    assert waves == [["approve", "fetch_rows", "fetch_totals"], ["publish"]]


@pytest.mark.parametrize("runtime", sorted(_RUNTIMES))
def test_a_gate_waits_after_the_reads_in_its_wave(runtime: str, tmp_path: Path) -> None:
    get_adapter(runtime).emit(_pipeline(), tmp_path)
    rel, wait_pattern = _RUNTIMES[runtime]
    assert _gate_waits_after_reads((tmp_path / rel).read_text(encoding="utf-8"), wait_pattern)


def test_the_check_fails_on_a_gate_that_waits_first() -> None:
    """The negative half: the same check on a body whose gate waits before
    one read's result is bound."""
    early = "\n".join(
        [
            "fetch_rows_result = fetch_rows(payload)",
            'approve_result = DBOS.recv(topic="approved")',
            "fetch_totals_result = fetch_totals(payload)",
        ]
    )
    assert not _gate_waits_after_reads(early, r'topic="approved"')
