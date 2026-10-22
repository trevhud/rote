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


def _assignment(source: str, rid: str) -> int:
    """Offset just past the first statement that assigns ``<rid>_result``.

    A plain or annotated assignment at the start of a line
    (``x_result = ...``, ``const x_result: T = ...``, ``a_result, x_result =``),
    or a destructure across lines (``const [a_result, x_result] = ...``,
    ``(a_result, x_result) = ...``). A bare declaration such as
    ``let x_result;`` assigns nothing, so it never counts: an adapter that
    declares the result above the wave and assigns it after the gate fails.
    """
    name = rf"{rid}_result"
    others = r"(?:[\w$]+\s*,\s*)*"
    rest = r"(?:\s*,\s*[\w$]+)*"
    line = re.compile(
        rf"^[ \t]*(?:const|let|var)?[ \t]*{others}\b{name}\b{rest}[ \t]*(?::[^=\n]*)?=(?!=)",
        re.M,
    )
    destructure = re.compile(rf"[\[(]\s*{others}\b{name}\b{rest}\s*,?\s*[\])]\s*=(?!=)")
    ends = [m.end() for rx in (line, destructure) if (m := rx.search(source))]
    assert ends, f"no assignment of {name}"
    return min(ends)


def _gate_waits_after_reads(source: str, wait_pattern: str) -> bool:
    """The gate's wait comes after the statement that assigns each read's
    finished result (``<id>_result``; a handle or promise is named apart)."""
    wait = re.compile(wait_pattern, re.M).search(source)
    assert wait, wait_pattern
    return all(_assignment(source, rid) < wait.start() for rid in _READS)


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


def test_a_forward_declaration_does_not_count_as_the_result() -> None:
    """Declaring the result above the gate and assigning it after is a gate
    that waits first; the first mention alone would have passed it."""
    declared = "\n".join(
        [
            "let fetch_rows_result;",
            "let fetch_totals_result;",
            'const approve_event = await step.waitForEvent("approve", { type: "approved" });',
            "[fetch_rows_result, fetch_totals_result] = await Promise.all([a, b]);",
        ]
    )
    assert not _gate_waits_after_reads(declared, r'type: "approved"')
