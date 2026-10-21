"""Bind MCP arguments from what a recorded run actually sent.

A ``rote baseline`` run records every tool call the skill made, with its
arguments. When a recorded argument's value is exactly one value the
pipeline also has (a field of the sample input the baseline ran on, one of
the node's own constants, or a top-level field of an upstream tool's
recorded result) that is where the argument comes from: evidence, not a
reading of the skill's prose. In the measured docs-digest run, Microsoft
Learn was sent ``query: "Azure Functions durable orchestration"``, the
sample's ``topic``, and DeepWiki ``repoName``, the sample's ``repo``; the
compile had bound neither, and the live calls returned nothing or failed.

Code only, no model. An argument already bound is kept. A value that
matches no source, or more than one, or calls that disagree about the
source, bind nothing: the gap stays for review and the hard check
(``tool_arguments_invalid``) instead of a guess.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from rote.compiler.system1.decide import NodePlan, Plan

#: Confidence of a binding every recorded call agrees on.
RECORDED_CONFIDENCE = 0.95


def _matches(value: Any, candidates: Mapping[str, Any]) -> list[str]:
    """The candidate names holding exactly ``value``.

    Booleans and empty values match nothing: ``True`` or ``""`` equal to
    some field says nothing about where an argument comes from.
    """
    if isinstance(value, bool) or value in (None, "", [], {}):
        return []
    return [
        name for name, held in candidates.items() if not isinstance(held, bool) and held == value
    ]


def _sources(
    value: Any, node: NodePlan, sample: Mapping[str, Any], upstream: Mapping[str, Mapping[str, Any]]
) -> list[tuple[str, str]]:
    found = [("input", f) for f in _matches(value, sample)]
    found += [("constant", c) for c in _matches(value, node.constants)]
    for up_id, fields in sorted(upstream.items()):
        found += [(up_id, f) for f in _matches(value, fields)]
    return found


def bind_recorded_arguments(
    plan: Plan, observations: Iterable[Any], sample_payload: Mapping[str, Any] | None
) -> None:
    """Bind each MCP node's recorded arguments to their one matching source."""
    from rote.probe import parse_tool_result

    calls = [o for o in observations if not getattr(o, "is_error", False)]
    sample = dict(sample_payload or {})
    by_tool: dict[tuple[str, str], list[Any]] = {}
    for o in calls:
        by_tool.setdefault((o.server, o.tool), []).append(o)
    # What each MCP node's tool returned, top-level fields only: the IR's
    # input references name one field of an upstream output.
    results: dict[str, Mapping[str, Any]] = {}
    for n in plan.nodes:
        if n.mcp is None:
            continue
        recorded = [
            parse_tool_result(o.result) for o in by_tool.get((n.mcp["server"], n.mcp["tool"]), [])
        ]
        objects = [r for r in recorded if isinstance(r, dict)]
        if len(objects) == 1:
            results[n.id] = objects[0]
    for n in plan.nodes:
        if n.mcp is None:
            continue
        recorded_inputs = [
            o.input
            for o in by_tool.get((n.mcp["server"], n.mcp["tool"]), [])
            if isinstance(o.input, dict)
        ]
        if not recorded_inputs:
            continue
        upstream = {
            src: results[src]
            for src in (ref.split(".", 1)[0] for ref in n.inputs.values())
            if src in results and src != n.id
        }
        args = dict(n.mcp.get("args") or {})
        bound: list[str] = []
        for param in sorted({p for call in recorded_inputs for p in call}):
            if param in args or not param.isidentifier():
                continue
            per_call = [_sources(call.get(param), n, sample, upstream) for call in recorded_inputs]
            if any(len(found) != 1 for found in per_call) or len(set(map(tuple, per_call))) != 1:
                continue
            origin, name = per_call[0][0]
            if origin == "constant":
                args[param] = name
            else:
                ref = f"pipeline.input.{name}" if origin == "input" else f"{origin}.output.{name}"
                if n.inputs.get(param, ref) != ref:
                    continue  # the name already carries something else
                n.inputs[param] = ref
                args[param] = param
            bound.append(f"`{param}` <- {'constant' if origin == 'constant' else ref}")
        if bound:
            n.mcp = {**n.mcp, "args": args}
            n.decide(
                "mcp.args",
                dict(args),
                RECORDED_CONFIDENCE,
                "every recorded call sent these values: " + ", ".join(bound),
            )
