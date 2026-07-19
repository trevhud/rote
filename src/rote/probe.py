"""Schema inference from observed MCP traffic — ground truth over guesses.

The baseline run (:mod:`rote.eval.baseline`) records every MCP tool call
the source skill actually made, with real input and result payloads.
This module turns those observations into JSON Schemas:

* :func:`infer_schema` — merge N observed JSON values into one schema.
  Object ``required`` is the *intersection* of keys across samples (a key
  every observation carried is evidence of a contract; a sometimes-key is
  optional), and mixed-type observations become ``anyOf`` rather than
  collapsing to the loosest common type.
* :func:`parse_tool_result` — unwrap the MCP content-block envelope
  (``[{"type": "text", "text": "..."}]``) back into the JSON value the
  tool actually returned.
* :func:`infer_tool_schemas` — group a baseline's observations by
  ``(server, tool)`` and infer an input + output schema per tool.

Inference is deliberately conservative: it never invents fields it did
not see, and a schema inferred from one sample is still just one sample —
callers surface ``samples`` so downstream consumers can weigh confidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from rote.eval.baseline import ObservedToolCall

# ───────── Value → schema ─────────


def _scalar_type(value: Any) -> str | None:
    # bool before int: bool is an int subclass in Python, not in JSON.
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if value is None:
        return "null"
    return None


def infer_schema(values: list[Any]) -> dict[str, Any]:
    """One JSON Schema that all observed ``values`` conform to.

    Merging rules, in the order they matter:

    * All samples the same scalar type → that type (``integer`` and
      ``number`` mixed → ``number``).
    * Objects → property-wise recursive merge; ``required`` is the key
      intersection across samples.
    * Arrays → ``items`` is the merge of every element seen anywhere.
    * Mixed types (including ``null``) → ``anyOf`` of the per-type merges,
      deterministic order.
    """
    if not values:
        return {}

    by_kind: dict[str, list[Any]] = {}
    for v in values:
        scalar = _scalar_type(v)
        if scalar is not None:
            by_kind.setdefault(scalar, []).append(v)
        elif isinstance(v, dict):
            by_kind.setdefault("object", []).append(v)
        elif isinstance(v, list):
            by_kind.setdefault("array", []).append(v)
        else:  # non-JSON value (shouldn't happen from parsed payloads)
            by_kind.setdefault("unknown", []).append(v)

    # integer + number collapse to number before branching on kind count.
    if "integer" in by_kind and "number" in by_kind:
        by_kind["number"].extend(by_kind.pop("integer"))

    if len(by_kind) == 1:
        kind, samples = next(iter(by_kind.items()))
        return _schema_for_kind(kind, samples)

    variants = [
        _schema_for_kind(kind, samples)
        for kind, samples in sorted(by_kind.items(), key=lambda kv: kv[0])
    ]
    return {"anyOf": variants}


def _schema_for_kind(kind: str, samples: list[Any]) -> dict[str, Any]:
    if kind == "object":
        keys_seen: set[str] = set()
        keys_in_all: set[str] | None = None
        per_key: dict[str, list[Any]] = {}
        for obj in samples:
            keys = set(obj)
            keys_seen |= keys
            keys_in_all = keys if keys_in_all is None else (keys_in_all & keys)
            for k, v in obj.items():
                per_key.setdefault(k, []).append(v)
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {k: infer_schema(per_key[k]) for k in sorted(keys_seen)},
        }
        required = sorted(keys_in_all or set())
        if required:
            schema["required"] = required
        return schema
    if kind == "array":
        elements = [item for arr in samples for item in arr]
        if not elements:
            return {"type": "array"}
        return {"type": "array", "items": infer_schema(elements)}
    if kind == "unknown":
        return {}
    return {"type": kind}


# ───────── MCP result envelope → JSON value ─────────


def parse_tool_result(result: Any) -> Any:
    """The JSON value behind an observed tool result, best-effort.

    MCP tool results arrive as content blocks; structured payloads ride
    inside ``text`` blocks as serialized JSON. Unwraps:

    * a list of blocks → concatenated ``text`` of the ``type: "text"``
      blocks, then JSON-parsed if possible;
    * a bare string → JSON-parsed if possible, else the string;
    * anything else → returned as-is.

    Returns ``None`` for an empty/missing result — callers must treat
    that as "no sample", not "the tool returns null".
    """
    if result is None:
        return None
    if isinstance(result, list):
        texts = [
            b.get("text", "") for b in result if isinstance(b, dict) and b.get("type") == "text"
        ]
        if not texts:
            return result
        result = "".join(texts)
    if isinstance(result, str):
        stripped = result.strip()
        if not stripped:
            return None
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return result
    return result


# ───────── Observations → per-tool schemas ─────────


@dataclass(frozen=True)
class InferredToolSchema:
    """Input/output schema evidence for one ``(server, tool)``."""

    server: str
    tool: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    samples: int
    error_samples: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "server": self.server,
            "tool": self.tool,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "samples": self.samples,
            "error_samples": self.error_samples,
        }


def cross_check(pipeline: Any, observations: list[ObservedToolCall]) -> dict[str, Any]:
    """Static MCP bindings vs. observed traffic — the two-source check.

    The graduator *infers* the skill's MCP usage from prose; the baseline
    *observed* it. Disagreements mean different things:

    * ``observed_only`` — the skill called a tool the pipeline doesn't
      bind. The graduator likely missed a requirement; the strongest
      signal here, surfaced first.
    * ``static_only`` — bound but never observed. Often benign (a branch
      the baseline task didn't take, or a write tool the read-only gate
      held back), so it's a note, not an alarm.
    * ``confirmed`` — bound and observed; the binding is evidenced.

    ``pipeline`` is a :class:`rote.ir.Pipeline`; typed as ``Any`` to keep
    this module import-light (it must not pull the IR at import time).
    """
    static: dict[tuple[str, str], list[str]] = {}
    for node in pipeline.nodes:
        if node.mcp is not None:
            static.setdefault((node.mcp.server, node.mcp.tool), []).append(node.id)

    observed_counts: dict[tuple[str, str], int] = {}
    for o in observations:
        observed_counts[(o.server, o.tool)] = observed_counts.get((o.server, o.tool), 0) + 1

    confirmed = [
        {
            "server": server,
            "tool": tool,
            "nodes": sorted(static[(server, tool)]),
            "observed_calls": count,
        }
        for (server, tool), count in sorted(observed_counts.items())
        if (server, tool) in static
    ]
    observed_only = [
        {"server": server, "tool": tool, "observed_calls": count}
        for (server, tool), count in sorted(observed_counts.items())
        if (server, tool) not in static
    ]
    static_only = [
        {"server": server, "tool": tool, "nodes": sorted(nodes)}
        for (server, tool), nodes in sorted(static.items())
        if (server, tool) not in observed_counts
    ]
    return {
        "observed_only": observed_only,
        "static_only": static_only,
        "confirmed": confirmed,
    }


def infer_tool_schemas(
    observations: list[ObservedToolCall],
) -> list[InferredToolSchema]:
    """Per-(server, tool) schemas inferred from a baseline's observations.

    Error results are excluded from output inference (a 401 body is not
    the tool's contract) but their *inputs* still count — the skill
    reached for the tool with that shape either way. Tools observed only
    in error get an empty output schema.
    """
    grouped: dict[tuple[str, str], list[ObservedToolCall]] = {}
    for o in observations:
        grouped.setdefault((o.server, o.tool), []).append(o)

    inferred: list[InferredToolSchema] = []
    for (server, tool), calls in sorted(grouped.items()):
        inputs = [c.input for c in calls]
        outputs = [
            parsed
            for c in calls
            if not c.is_error and (parsed := parse_tool_result(c.result)) is not None
        ]
        inferred.append(
            InferredToolSchema(
                server=server,
                tool=tool,
                input_schema=infer_schema(inputs),
                output_schema=infer_schema(outputs),
                samples=len(calls),
                error_samples=sum(1 for c in calls if c.is_error),
            )
        )
    return inferred
