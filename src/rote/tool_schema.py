"""Whether arguments satisfy an MCP tool's input schema. Stdlib only.

One reading of a tool's ``inputSchema``, shared by everything that sends
or answers a tool call outside a real server: the bench's fixture servers
refuse a call a real server would refuse, and verification
(:mod:`rote.compiler.grounding`, :mod:`rote.compiler.prefix`) refuses an
``mcp:`` binding that would send one. If the two drifted, a compile could
pass verification and still be refused by the fixtures, or the reverse.

The reading is the part of JSON Schema a tool's flat object schema uses:
required keys present, no key the schema does not list unless it allows
extra properties, and each value of its declared JSON type. An argument
a server does not declare is one it rejects or ignores, never acts on.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
    "null": (type(None),),
}


def is_json_type(value: Any, want: str | list[str]) -> bool:
    """Whether ``value`` is of the JSON type(s) ``want`` (a bool is not a number)."""
    for name in [want] if isinstance(want, str) else want:
        if name in ("integer", "number") and isinstance(value, bool):
            continue
        if isinstance(value, _JSON_TYPES.get(name, (object,))):
            return True
    return False


def allows_extra(schema: Mapping[str, Any]) -> bool:
    """Whether the schema accepts properties it does not list.

    Absent counts as no: a tool lists the parameters it acts on, and
    MCP servers routinely leave ``additionalProperties`` unset.
    """
    return schema.get("additionalProperties", False) not in (False, None)


def names_problem(
    schema: Mapping[str, Any], names: Iterable[str], *, check_unknown: bool = True
) -> str | None:
    """Why sending arguments named ``names`` breaks the schema, or ``None``.

    The static half: which keys go out, not their values. ``check_unknown``
    is off for a schema inferred from recorded calls, which only lists the
    keys someone happened to send.
    """
    sent = set(names)
    properties: Mapping[str, Any] = schema.get("properties") or {}
    missing = [k for k in schema.get("required") or [] if k not in sent]
    if missing:
        return f"missing required argument(s): {', '.join(missing)}"
    if check_unknown and not allows_extra(schema):
        unknown = sorted(sent - set(properties))
        if unknown:
            return (
                f"unknown argument(s): {', '.join(unknown)} "
                f"(the tool takes: {', '.join(sorted(properties)) or 'none'})"
            )
    return None


def argument_problem(
    schema: Mapping[str, Any], arguments: Mapping[str, Any], *, check_unknown: bool = True
) -> str | None:
    """Why ``arguments`` break the tool's input schema, or ``None``."""
    problem = names_problem(schema, arguments, check_unknown=check_unknown)
    if problem is not None:
        return problem
    properties: Mapping[str, Any] = schema.get("properties") or {}
    for key, value in arguments.items():
        want = (properties.get(key) or {}).get("type")
        if want is not None and not is_json_type(value, want):
            return f"argument {key!r} must be {want}, got {type(value).__name__}"
    return None
