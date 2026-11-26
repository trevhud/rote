"""Cross-runtime contract: a step with ``when:`` runs only when its condition holds.

A skill that does a step only when something holds ("post the summary to
the channel if chat is connected") used to compile to a step every run had
to do. ``when: <node>.output.<field>`` names a boolean an earlier node
computes: true, the step runs as before; false, it is skipped and its
result is null, which the node after it reads whole. Any other value fails
the run naming the step, on every runtime.

The pipeline puts two conditional steps (one of them a fan_out) in a wave
with an unconditional one, so each emitter's "leaves its wave" path and its
fan_out block both run. Each runtime runs the emitted app for real, three
times: the condition true, false, and a string. The TS runtimes live in
test_conditional_step_parity_ts_e2e.py and
test_conditional_step_parity_dbos_ts_e2e.py.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from rote.adapters import get_adapter
from rote.eval.empirical import run_pipeline_trial
from rote.ir import Pipeline
from rote.mcp.parked import watch_factory

PEOPLE = ["ana", "bo", "cy"]

PIPELINE = Pipeline.model_validate(
    {
        "name": "conditional-step",
        "version": "0.1.0",
        "input": {"type": "In", "required": ["notes", "flag"]},
        "nodes": [
            {
                "id": "check",
                "kind": "pure_function",
                "description": "See whether chat is connected.",
                "impl": "extracted/steps.py:check",
                "inputs": {"flag": "pipeline.input.flag"},
                "output": {"connected": "bool", "channel": "str", "people": "list"},
            },
            {
                "id": "draft",
                "kind": "pure_function",
                "description": "Draft the summary.",
                "impl": "extracted/steps.py:draft",
                "inputs": {"notes": "pipeline.input.notes"},
                "output": {"text": "str"},
            },
            {
                "id": "post",
                "kind": "pure_function",
                "description": "Post to the channel, when chat is connected.",
                "impl": "extracted/steps.py:post",
                "when": "check.output.connected",
                "inputs": {"channel": "check.output.channel", "notes": "pipeline.input.notes"},
                "output": {"posted_to": "str"},
            },
            {
                "id": "ping",
                "kind": "pure_function",
                "description": "Ping each person, when chat is connected.",
                "impl": "extracted/steps.py:ping",
                "when": "check.output.connected",
                "fan_out": True,
                "inputs": {"person": "check.output.people"},
            },
            {
                "id": "recap",
                "kind": "pure_function",
                "description": "Say what was done.",
                "impl": "extracted/steps.py:recap",
                "inputs": {
                    "text": "draft.output.text",
                    "post": "post.output",
                    "pings": "ping.output",
                },
            },
        ],
        "edges": [
            {"from": "check", "to": "draft"},
            {"from": "check", "to": "post"},
            {"from": "check", "to": "ping", "fan_out": True},
            {"from": "draft", "to": "recap"},
            {"from": "post", "to": "recap"},
            {"from": "ping", "to": "recap"},
        ],
        "entry_nodes": ["check"],
        "exit_nodes": ["recap"],
    }
)

STEPS = (
    "def check(flag):\n"
    f"    return {{'connected': flag, 'channel': '#ops', 'people': {PEOPLE!r}}}\n"
    "\n"
    "\n"
    "def draft(notes):\n"
    "    return {'text': notes.upper()}\n"
    "\n"
    "\n"
    "def post(channel, notes):\n"
    "    return {'posted_to': channel}\n"
    "\n"
    "\n"
    "def ping(person):\n"
    "    return 'pinged ' + person\n"
    "\n"
    "\n"
    "def recap(text, post, pings):\n"
    "    return {\n"
    "        'text': text,\n"
    "        'posted_to': None if post is None else post['posted_to'],\n"
    "        'pings': pings,\n"
    "    }\n"
)

#: (pipeline input, the run's recap or None, a fragment of its error or None)
CASES: dict[str, tuple[dict[str, Any], dict[str, Any] | None, str | None]] = {
    "runs": (
        {"notes": "kickoff", "flag": True},
        {"text": "KICKOFF", "posted_to": "#ops", "pings": [f"pinged {p}" for p in PEOPLE]},
        None,
    ),
    "skips": (
        {"notes": "kickoff", "flag": False},
        {"text": "KICKOFF", "posted_to": None, "pings": None},
        None,
    ),
    "refuses": (
        {"notes": "kickoff", "flag": "yes"},
        None,
        "' runs only when check.output.connected is true, and it is",
    ),
}


def check(case: str, run: Any) -> None:
    """The run's output, or its named failure."""
    _, expected, error = CASES[case]
    if error is not None:
        assert run.error is not None and error in run.error, run.error
        return
    assert run.error is None, run.error
    assert run.output == {"recap": expected}


def _emit(runtime: str, tmp_path: Path) -> Path:
    source = tmp_path / "compiled"
    (source / "extracted").mkdir(parents=True)
    (source / "extracted" / "steps.py").write_text(STEPS, encoding="utf-8")
    out = tmp_path / runtime
    get_adapter(runtime, extracted_source_dir=source).emit(PIPELINE, out)
    return out


@pytest.mark.parametrize(
    "runtime", ["cloudflare", "dbos", "dbos-ts", "inngest", "python", "temporal"]
)
def test_a_conditional_step_leaves_its_wave_and_the_rest_stays_parallel(
    runtime: str, tmp_path: Path
) -> None:
    """Emitted text, both halves: the conditional steps sit behind their
    condition, and the wave's unconditional step is not behind any."""
    out = tmp_path / runtime
    get_adapter(runtime).emit(PIPELINE, out)
    name = {
        "cloudflare": "src/workflow.ts",
        "dbos": "main.py",
        "dbos-ts": "src/main.ts",
        "inngest": "src/inngest/pipeline.ts",
        "python": "main.py",
        "temporal": "workflow.py",
    }[runtime]
    text = (out / name).read_text(encoding="utf-8")
    helper = "conditionError(" if name.endswith(".ts") else "_condition_error("
    # One definition and one call per conditional step.
    assert text.count(helper) == 3, text
    body = text[text.index("Wave 1") :]
    draft, post, ping = (body.index(f"{n}_result = ") for n in ("draft", "post", "ping"))
    first_check = body.index(helper)
    assert draft < first_check < min(post, ping), body


@pytest.mark.parametrize("case", sorted(CASES))
def test_python_runs_a_step_only_when_its_condition_holds(case: str, tmp_path: Path) -> None:
    run = run_pipeline_trial(_emit("python", tmp_path), CASES[case][0], timeout_seconds=60)
    check(case, run)


@pytest.mark.parametrize("case", sorted(CASES))
def test_temporal_runs_a_step_only_when_its_condition_holds(case: str, tmp_path: Path) -> None:
    from rote.runners.temporal import run_temporal

    app = _emit("temporal", tmp_path)
    try:
        run = run_temporal(
            app, CASES[case][0], signals={}, timeout_seconds=60, park_watch=watch_factory(False)
        )
    finally:
        sys.path.remove(str(app))
        for name in [
            m
            for m in sys.modules
            if m.split(".")[0] in ("workflow", "activities", "extracted", "signatures")
        ]:
            del sys.modules[name]
    check(case, run)


@pytest.mark.slow
@pytest.mark.parametrize("case", sorted(CASES))
def test_dbos_runs_a_step_only_when_its_condition_holds(case: str, tmp_path: Path) -> None:
    """Slow: launches a real DBOS runtime (system-DB migrations, queues)."""
    run = run_pipeline_trial(
        _emit("dbos", tmp_path),
        CASES[case][0],
        signals={},
        timeout_seconds=120,
        park_watch=watch_factory(False),
    )
    check(case, run)


def _with_step(step: dict[str, Any], *, gate: bool = False) -> Pipeline:
    """``check`` and one conditional ``post`` of another kind, optionally
    beside a gate in ``post``'s wave."""
    raw = PIPELINE.model_dump(mode="json", by_alias=True, exclude_none=True)
    nodes = [n for n in raw["nodes"] if n["id"] == "check"]
    nodes.append({"id": "post", "description": "Post it.", "when": "check.output.connected"} | step)
    edges = [{"from": "check", "to": "post"}]
    if gate:
        nodes.append(
            {
                "id": "approve",
                "kind": "hitl_gate",
                "description": "Approve the channel.",
                "signal": "approved",
                "inputs": {"channel": "check.output.channel"},
                "notify": {"message_template": "Post to {channel}?"},
            }
        )
        edges.append({"from": "check", "to": "approve"})
    return Pipeline.model_validate(
        raw | {"nodes": nodes, "edges": edges, "exit_nodes": [n["id"] for n in nodes[1:]]}
    )


_MCP_STEP = {
    "kind": "external_call",
    "mcp": {"server": "chat", "tool": "post_message"},
    "inputs": {"channel": "check.output.channel"},
}
_LOOP_STEP = {
    "kind": "agent_loop",
    "tools": ["search"],
    "inputs": {"channel": "check.output.channel"},
}

_WORKFLOW_FILE = {
    "cloudflare": "src/workflow.ts",
    "dbos": "main.py",
    "dbos-ts": "src/main.ts",
    "inngest": "src/inngest/pipeline.ts",
    "python": "main.py",
    "temporal": "workflow.py",
}


@pytest.mark.parametrize(
    ("runtime", "step"),
    [
        *((r, "mcp") for r in ("cloudflare", "dbos", "dbos-ts", "inngest")),
        *((r, "loop") for r in sorted(_WORKFLOW_FILE)),
    ],
)
def test_a_conditional_tool_call_or_agent_loop_emits_behind_its_condition(
    runtime: str, step: str, tmp_path: Path
) -> None:
    """Found by review: Cloudflare declares a step that may park with ``let``,
    and the wrapper insisted on ``const``, so any conditional MCP step or
    agent loop failed the emit with a bare AssertionError."""
    pipeline = _with_step(_MCP_STEP if step == "mcp" else _LOOP_STEP)
    get_adapter(runtime).emit(pipeline, tmp_path)
    text = (tmp_path / _WORKFLOW_FILE[runtime]).read_text(encoding="utf-8")
    helper = (
        "conditionError("
        if runtime in ("cloudflare", "dbos-ts", "inngest")
        else "_condition_error("
    )
    body = text[text.index("Wave 1") :]
    assert body.index(helper) < body.index("post_result")


def test_a_gate_in_the_wave_waits_after_the_conditional_step() -> None:
    """Found by review: every emitter runs a wave's gates last, so a step
    that left the wave ran after the gate and sat behind a person's answer
    it does not read. The gate moves with it; a wave with no conditional step
    is untouched."""
    from rote.adapters._common import emission_waves

    waves = [[n.id for n in w] for w in emission_waves(_with_step(_LOOP_STEP, gate=True))]
    assert waves == [["check"], ["post"], ["approve"]]
    plain = Pipeline.model_validate(
        _with_step(_LOOP_STEP, gate=True).model_dump(mode="json", by_alias=True, exclude_none=True)
        | {
            "nodes": [
                n if n["id"] != "post" else {k: v for k, v in n.items() if k != "when"}
                for n in _with_step(_LOOP_STEP, gate=True).model_dump(
                    mode="json", by_alias=True, exclude_none=True
                )["nodes"]
            ]
        }
    )
    assert [sorted(n.id for n in w) for w in emission_waves(plain)] == [
        ["check"],
        ["approve", "post"],
    ]
