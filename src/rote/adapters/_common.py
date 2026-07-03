"""Helpers shared by every runtime adapter.

These functions are runtime-agnostic: they operate purely on the
:class:`rote.ir.Pipeline` IR and on plain strings. Anything that encodes
a *runtime's* semantics (Temporal retry policies, Cloudflare step
configs, …) stays in the adapter that owns it.

Extracted from ``rote.adapters.temporal`` / ``rote.adapters.cloudflare``
once the second adapter proved the helpers were genuinely shared.
"""

from __future__ import annotations

import hashlib
import re

from rote.ir import Node, Pipeline, parse_input_ref

# ───────── Case conversion ─────────


def _to_pascal_case(s: str) -> str:
    """Convert kebab-case or snake_case to PascalCase."""
    parts = s.replace("-", "_").split("_")
    return "".join(p.capitalize() for p in parts if p)


def _to_camel_case(s: str) -> str:
    pascal = _to_pascal_case(s)
    if not pascal:
        return ""
    return pascal[0].lower() + pascal[1:]


# ───────── Pipeline identity ─────────


def _pipeline_hash(pipeline: Pipeline) -> str:
    """Stable 8-char hash of the pipeline's identity.

    Used to version emitted artifacts (e.g. the Temporal workflow type
    name) so a regenerated pipeline becomes a new type. Old in-flight
    workflows continue on the old code; new workflows use the new code.
    """
    payload = f"{pipeline.name}|{pipeline.version}|{len(pipeline.nodes)}|{len(pipeline.edges)}"
    return hashlib.sha256(payload.encode()).hexdigest()[:8]


# ───────── Duration strings ─────────

_IR_DURATION_RE = re.compile(r"^(\d+(?:\.\d+)?)(ms|s|m|h|d)$")

_UNIT_TO_HUMAN = {
    "ms": "milliseconds",
    "s": "seconds",
    "m": "minutes",
    "h": "hours",
    "d": "days",
}


def ir_duration_to_human(s: str) -> str:
    """Convert an IR duration string ('5m', '30s', '7d') to a human form.

    Returns e.g. '5 minutes', '30 seconds', '7 days' — the format
    Cloudflare's step config accepts directly. Strings that don't match
    the IR shorthand pattern are passed through unchanged (they're
    assumed to already be in an acceptable form).
    """
    s = s.strip()
    m = _IR_DURATION_RE.fullmatch(s)
    if not m:
        return s
    return f"{m.group(1)} {_UNIT_TO_HUMAN[m.group(2)]}"


# ───────── Topological waves ─────────


def _execution_waves(pipeline: Pipeline) -> list[list[Node]]:
    """Topologically sort the pipeline into parallel-execution waves.

    A "wave" is a set of nodes whose dependencies have all been satisfied
    by the time the wave starts; nodes within a wave can run in parallel.

    Nodes that appear inside another node's ``loop_body`` are excluded
    from the top-level waves — they're orchestrated from inside the
    parent loop activity, not by the workflow.
    """
    nested_ids: set[str] = set()
    for n in pipeline.nodes:
        if n.loop_body:
            nested_ids.update(n.loop_body)

    eligible = [n for n in pipeline.nodes if n.id not in nested_ids]
    eligible_ids = {n.id for n in eligible}

    in_degree: dict[str, int] = {n.id: 0 for n in eligible}
    for edge in pipeline.edges:
        if edge.from_ in eligible_ids and edge.to in eligible_ids:
            in_degree[edge.to] += 1

    waves: list[list[Node]] = []
    remaining: set[str] = set(in_degree.keys())

    while remaining:
        # Sort by id for stable, deterministic emission order.
        wave_ids = sorted(nid for nid in remaining if in_degree[nid] == 0)
        if not wave_ids:
            raise ValueError(
                f"Cycle detected in pipeline {pipeline.name!r}; "
                f"remaining nodes: {sorted(remaining)}"
            )
        waves.append([pipeline.node_by_id(nid) for nid in wave_ids])
        for nid in wave_ids:
            remaining.remove(nid)
            for edge in pipeline.edges:
                if edge.from_ == nid and edge.to in eligible_ids:
                    in_degree[edge.to] -= 1

    return waves


# ───────── Data-flow threading ─────────


def check_input_refs_available(node: Node, available: set[str]) -> None:
    """Emit-time guard: every node-output reference must already be bound.

    Adapters emit nodes wave-by-wave, binding each node's result to a
    local as they go. A node whose ``inputs`` reference a node in a
    *later* wave (or a loop-body sub-node, which never binds a top-level
    result) would produce code that references an undefined variable —
    fail loudly at emit time instead.

    IR validation already guarantees the referenced node *exists*; this
    check is about emission ordering, which is an adapter concern.
    """
    if not node.inputs:
        return
    for param, ref in node.inputs.items():
        parsed = parse_input_ref(ref)
        if parsed.node_id is not None and parsed.node_id not in available:
            raise ValueError(
                f"Node {node.id!r} input {param!r} references {ref!r}, but node "
                f"{parsed.node_id!r} has no result available at this point in the "
                f"workflow (it runs in a later wave, or is a loop-body sub-node "
                f"with no top-level result)."
            )
