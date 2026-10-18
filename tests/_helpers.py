"""Shared helpers for adapter tests.

Everything here exists in 3+ adapter test files otherwise — and the
copies had already drifted (the Cloudflare MCP scan was missing the
backtick template-literal alternative, so a legitimate "mcp" mention
inside a template string failed only that adapter's test).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from rote.adapters._common import strip_ts_comments
from rote.ir import Node, Pipeline

# ───────── MCP invariant scan (TS) ─────────
#
# Comments and JSDoc may *mention* MCP to explain the compilation
# history; only executable code is forbidden from referencing it. So
# strip /* ... */ blocks, // line comments, and string literals —
# including backtick template literals — before scanning.

_JS_STRING = re.compile(r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|`(?:[^`\\]|\\.)*`')


def assert_no_mcp_in_ts(
    emit_result: dict[str, Path], *, min_files: int, expected: set[str] | None = None
) -> None:
    """Assert no emitted .ts file references MCP in executable code.

    ``expected`` names the labels that legitimately DO — the MCP helper
    itself, and any agent_loop module, whose ``tools:`` are MCP tool
    names by the IR's own definition. Naming them (rather than skipping
    every file that happens to mention MCP) keeps the scan a real
    constraint: an unexpected MCP reference still fails, and an expected
    site that stops referencing MCP fails too.

    ``min_files`` guards against the scan silently checking nothing
    (e.g. a refactor that changes emitted file suffixes).
    """
    expected = expected or set()
    checked = 0
    for label, path in emit_result.items():
        if not str(path).endswith(".ts"):
            continue
        if label in expected:
            assert "mcp" in path.read_text(encoding="utf-8").lower(), (
                f"{label} was declared an expected MCP site but references no MCP"
            )
            continue
        src = path.read_text(encoding="utf-8")
        cleaned = _JS_STRING.sub('""', strip_ts_comments(src))
        assert "mcp" not in cleaned.lower(), (
            f"{label} ({path.name}) contains forbidden substring 'mcp' in executable code"
        )
        checked += 1
    assert checked >= min_files, f"MCP scan only saw {checked} .ts files (expected >= {min_files})"


# ───────── Minimal pipeline factory ─────────


def mini_pipeline(node: Node) -> Pipeline:
    """A single-node pipeline for focused emission tests."""
    return Pipeline(
        name="mini",
        input={"type": "In", "required": [], "optional": []},
        nodes=[node],
        edges=[],
        entry_nodes=[node.id],
        exit_nodes=[node.id],
    )


# ───────── fan_out mock support (live e2e) ─────────
#
# Every live e2e replaces node implementations with recording mocks. A
# fan_out node dispatches once per element of an upstream list, so the
# mock standing in for that upstream node has to return a real list —
# otherwise the fan either dispatches zero times (empty list, node
# silently never runs) or trips the fanOutList guard (missing key).
#
# Derived from the IR rather than hardcoded per suite: the three TS e2e
# files had already encoded `exclusion_check_sequence` returning
# `passed: []`, which made the BDR fan a no-op that no assertion caught.

#: Elements per fanned list in mocked runs. Must exceed 1, or
#: per-element dispatch is indistinguishable from batch dispatch.
FAN_OUT_ELEMENTS = 3


def fan_out_element(index: int) -> dict[str, int]:
    """One canned element of a mocked fanned list."""
    return {"fanElement": index}


def fan_out_source_keys(pipeline: Pipeline) -> dict[str, dict[str, list[dict[str, int]]]]:
    """``{upstream_node: {field: [element, ...]}}`` for every fan_out node.

    Uses the adapters' own element-param resolution, so a mock can never
    disagree with the emitted code about which input is the fanned list.
    """
    from rote.ir import fan_out_element_param, parse_input_ref

    extra: dict[str, dict[str, list[dict[str, int]]]] = {}
    for node in pipeline.nodes:
        if not node.fan_out or not node.inputs:
            continue
        parsed = parse_input_ref(node.inputs[fan_out_element_param(node, pipeline)])
        if parsed.node_id is None or parsed.field is None:
            continue
        extra.setdefault(parsed.node_id, {})[parsed.field] = [
            fan_out_element(i) for i in range(FAN_OUT_ELEMENTS)
        ]
    return extra


# ───────── Unconfigured MCP endpoint (config-error e2e cases) ─────────
#
# A node bound to (or a loop pinned to) a server with no URL anywhere: no
# IR url, no registry entry, no ROTE_MCP_VENDOR_URL. Every runtime must
# run the step exactly ONCE and fail naming the variable. The count is
# the assertion that matters: a message can be right while the step
# still burned its retry budget.

ATTEMPT_MARKER = "ROTE_E2E_ATTEMPT"


def unconfigured_mcp_pipelines(name: str) -> dict[str, Pipeline]:
    """One bound-step pipeline and one agent-loop pipeline, both with retries."""
    from rote.ir import MCPBinding, NodeKind, PipelineInput, RetryPolicy, TerminationConfig

    bound = Node(
        id="lookup",
        kind=NodeKind.EXTERNAL_CALL,
        description="Look up a contact on the vendor MCP server.",
        input={"contact_id": "str"},
        inputs={"contact_id": "pipeline.input.contact_id"},
        output="dict",
        mcp=MCPBinding(server="vendor", tool="enrich_contact", args={"contact_id": "contact_id"}),
        retry=RetryPolicy(max=3, backoff="exponential"),
    )
    loop = Node(
        id="research",
        kind=NodeKind.AGENT_LOOP,
        description="Research the contact with the vendor's tools.",
        input={"contact_id": "str"},
        inputs={"contact_id": "pipeline.input.contact_id"},
        output="dict",
        tools=["enrich_contact"],
        tool_servers={"enrich_contact": "vendor"},
        termination=TerminationConfig(condition="the contact is enriched", max_iterations=3),
        retry=RetryPolicy(max=3, backoff="exponential"),
    )
    return {
        node.id: Pipeline(
            name=f"{name}_{node.id}",
            description="Unconfigured MCP endpoint e2e pipeline.",
            input=PipelineInput(type="Req", required=["contact_id"]),
            nodes=[node],
            edges=[],
            entry_nodes=[node.id],
            exit_nodes=[node.id],
        )
        for node in (bound, loop)
    }


def count_ts_step_attempts(module: Path, node_id: str) -> None:
    """Make an emitted TS node module log one marker line per invocation.

    The marker goes first in the exported function, so it counts step
    attempts whatever the step does next (bound call or tool discovery).
    """
    src = module.read_text(encoding="utf-8")
    match = re.search(r"export async function \w+\([\s\S]*?\)[^\n]*\{\n", src)
    assert match, f"no exported async function in {module}"
    marker = f'    console.log("{ATTEMPT_MARKER} {node_id}");\n'
    module.write_text(src[: match.end()] + marker + src[match.end() :], encoding="utf-8")


def attempts_in(text: str, node_id: str) -> int:
    """How many times :func:`count_ts_step_attempts`'s marker was printed."""
    return text.count(f"{ATTEMPT_MARKER} {node_id}")


def payload_dict_keys(payload: ast.Dict) -> set[str]:
    """The keys an emitted payload dict literal can carry.

    A pipeline input field is spread in as
    ``**_input_field(pipeline_input, "<key>", "<field>")``, so its key is
    the call's second argument rather than a literal dict key.
    """
    keys: set[str] = set()
    for key, value in zip(payload.keys, payload.values, strict=True):
        if isinstance(key, ast.Constant):
            keys.add(key.value)
        elif (
            key is None
            and isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "_input_field"
            and isinstance(value.args[1], ast.Constant)
        ):
            keys.add(value.args[1].value)
    return keys
