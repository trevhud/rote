"""Check a JSON value against a judge's or gate's schema. Stdlib only.

rote emits this file verbatim beside a compiled skill's driver (as
``scripts/_rote_schema.py``), so it imports nothing from rote and nothing
outside the standard library.

An answer that reaches a compiled skill comes from an agent, not from a
schema-locked decoder, so it is checked here before anything downstream
reads it. The rule is the compiled pipeline's own: what the emitted judge's
Pydantic model accepts (``rote.adapters._py_common._SchemaToPydantic``),
read strictly, plus the JSON Schema constraints that model does not carry.

- **The Pydantic model's rules.** Types, ``required`` (a property with a
  ``default`` is not required, and an optional property may be ``null``,
  as the model's ``X | None = None`` field allows), ``additionalProperties:
  false`` (``extra="forbid"``; otherwise undeclared keys are dropped),
  ``enum`` (``Literal``), ``anyOf`` / ``oneOf`` (a union: one matching
  branch is enough), ``items``, local ``$ref`` into ``$defs``.
- **Read strictly.** No coercion: ``"3"`` is not an integer, ``1`` is not a
  boolean. An integral float (``3.0``) is an integer, as JSON says.
- **Plus the constraints the model drops.** ``const``, ``minimum`` /
  ``maximum`` / exclusive bounds, ``multipleOf``, ``minLength`` /
  ``maxLength``, ``pattern``, ``minItems`` / ``maxItems``, ``uniqueItems``,
  ``allOf``, and a map's typed values (``additionalProperties: <schema>`` on
  an object that declares no properties, which the model reads as
  ``dict[str, Any]``). Structured outputs move these into the description,
  so the emitted judge never checks them; here they are checked.

:func:`project` is the other half: the value the emitted judge's
``model_dump(by_alias=True)`` returns for an accepted value, so a step
downstream reads the same keys on every runtime (an omitted optional key
is ``None``, a default is filled in, undeclared keys are dropped).
``tests/test_plan_schema.py`` runs both against the real emitted models
for every judge schema in ``examples/``.
"""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping
from typing import Any

_JSON_TYPES = {
    "string": "string",
    "integer": "integer",
    "number": "number",
    "boolean": "boolean",
    "null": "null",
    "object": "object",
    "array": "array",
}


def json_type(value: Any) -> str:
    """The JSON name of ``value``'s type, for messages an agent reads."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _is_type(kind: str, value: Any) -> bool:
    if kind == "string":
        return isinstance(value, str)
    if kind == "integer":
        if isinstance(value, bool):
            return False
        if isinstance(value, int):
            return True
        return isinstance(value, float) and math.isfinite(value) and value.is_integer()
    if kind == "number":
        if isinstance(value, bool):
            return False
        return isinstance(value, int) or (isinstance(value, float) and math.isfinite(value))
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "null":
        return value is None
    if kind == "object":
        return isinstance(value, dict)
    if kind == "array":
        return isinstance(value, list)
    raise ValueError(f"unknown JSON Schema type {kind!r}")


def _resolve(schema: Mapping[str, Any], root: Mapping[str, Any]) -> Mapping[str, Any]:
    """Follow ``$ref`` chains into the root's ``$defs`` (the only form rote emits)."""
    seen: set[str] = set()
    while "$ref" in schema:
        ref = schema["$ref"]
        if not isinstance(ref, str) or not ref.startswith("#/$defs/"):
            raise ValueError(f"unsupported $ref {ref!r}: only #/$defs/<name> is followed")
        if ref in seen:
            raise ValueError(f"cyclic $ref {ref!r}")
        seen.add(ref)
        name = ref[len("#/$defs/") :]
        defs = root.get("$defs") or {}
        if name not in defs:
            raise ValueError(f"unknown $ref target {ref!r}")
        schema = defs[name]
    return schema


def _short(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= 60 else text[:57] + "..."


def _defaulted(prop: Any) -> bool:
    return isinstance(prop, Mapping) and "default" in prop


def _optional_props(schema: Mapping[str, Any]) -> set[str]:
    """Properties the Pydantic model makes optional: not required, or defaulted."""
    required = set(schema.get("required") or ())
    return {
        key
        for key, prop in (schema.get("properties") or {}).items()
        if key not in required or _defaulted(prop)
    }


def _nullable_props(schema: Mapping[str, Any]) -> set[str]:
    """Properties the model types ``X | None = None``: optional with no default."""
    required = set(schema.get("required") or ())
    return {
        key
        for key, prop in (schema.get("properties") or {}).items()
        if key not in required and not _defaulted(prop)
    }


def schema_errors(
    schema: Any, value: Any, path: str = "answer", root: Mapping[str, Any] | None = None
) -> list[str]:
    """Every place ``value`` breaks ``schema``, as ``path: what`` lines. Empty when valid."""
    if schema is True or schema == {}:
        return []
    if schema is False:
        return [f"{path}: no value is allowed here"]
    if not isinstance(schema, Mapping):
        raise ValueError(f"{path}: a schema must be an object or a boolean")
    root = schema if root is None else root
    schema = _resolve(schema, root)
    errors: list[str] = []
    for sub in schema.get("allOf") or ():
        errors.extend(schema_errors(sub, value, path, root))
    for key in ("anyOf", "oneOf"):
        if key in schema:
            options = [schema_errors(option, value, path, root) for option in schema[key]]
            if not any(not found for found in options):
                # A branch of the value's own type is the closer miss.
                closest = min(options, key=lambda found: (_wrong_type_at(found, path), len(found)))
                errors.append(
                    f"{path}: matches none of the allowed shapes (closest: "
                    + "; ".join(closest[:3])
                    + ")"
                )
                return errors
    if "const" in schema and not _equal(value, schema["const"]):
        errors.append(f"{path}: must be {_short(schema['const'])}, got {_short(value)}")
        return errors
    if "enum" in schema and not any(_equal(value, option) for option in schema["enum"]):
        errors.append(f"{path}: {_short(value)} is not one of {_short(schema['enum'])}")
        return errors
    kinds = schema.get("type")
    if kinds is not None:
        allowed = [kinds] if isinstance(kinds, str) else list(kinds)
        if not any(_is_type(kind, value) for kind in allowed):
            expected = " or ".join(allowed)
            return [*errors, f"{path}: expected {expected}, got {json_type(value)}"]
    # Each keyword applies to the values of its own type, declared or not
    # (``{"minimum": 3}`` inside an ``allOf`` constrains a number).
    if isinstance(value, dict):
        errors.extend(_object_errors(schema, value, path, root))
    elif isinstance(value, list):
        errors.extend(_array_errors(schema, value, path, root))
    elif isinstance(value, str):
        errors.extend(_string_errors(schema, value, path))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        errors.extend(_number_errors(schema, value, path))
    return errors


def _wrong_type_at(errors: list[str], path: str) -> bool:
    return any(e.startswith(f"{path}: expected ") for e in errors)


def _equal(a: Any, b: Any) -> bool:
    """JSON equality: ``True`` is not ``1``, ``1`` is ``1.0``."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    return bool(a == b)


def _object_errors(
    schema: Mapping[str, Any], value: dict[str, Any], path: str, root: Mapping[str, Any]
) -> list[str]:
    errors: list[str] = []
    properties: Mapping[str, Any] = schema.get("properties") or {}
    optional = _optional_props(schema)
    nullable = _nullable_props(schema)
    for key in schema.get("required") or ():
        if key not in value and key not in optional:
            errors.append(f"{path}: missing key {key!r}")
    extra = schema.get("additionalProperties")
    for key, item in value.items():
        where = f"{path}.{key}"
        if key in properties:
            if item is None and key in nullable:
                continue  # the model's optional field is `X | None = None`
            errors.extend(schema_errors(properties[key], item, where, root))
        elif extra is False:
            errors.append(f"{path}: unexpected key {key!r}")
        elif isinstance(extra, Mapping) and not properties:
            errors.extend(schema_errors(extra, item, where, root))
    return errors


def _array_errors(
    schema: Mapping[str, Any], value: list[Any], path: str, root: Mapping[str, Any]
) -> list[str]:
    errors: list[str] = []
    if "minItems" in schema and len(value) < schema["minItems"]:
        errors.append(f"{path}: needs at least {schema['minItems']} items, got {len(value)}")
    if "maxItems" in schema and len(value) > schema["maxItems"]:
        errors.append(f"{path}: allows at most {schema['maxItems']} items, got {len(value)}")
    if schema.get("uniqueItems"):
        for i, item in enumerate(value):
            if any(_equal(item, other) for other in value[:i]):
                errors.append(f"{path}[{i}]: repeats an earlier item")
                break
    items = schema.get("items")
    if isinstance(items, (Mapping, bool)):
        for i, item in enumerate(value):
            errors.extend(schema_errors(items, item, f"{path}[{i}]", root))
    return errors


def _string_errors(schema: Mapping[str, Any], value: str, path: str) -> list[str]:
    errors: list[str] = []
    if "minLength" in schema and len(value) < schema["minLength"]:
        errors.append(f"{path}: needs at least {schema['minLength']} characters")
    if "maxLength" in schema and len(value) > schema["maxLength"]:
        errors.append(f"{path}: allows at most {schema['maxLength']} characters")
    if "pattern" in schema and re.search(schema["pattern"], value) is None:
        errors.append(f"{path}: {_short(value)} does not match {schema['pattern']!r}")
    return errors


def _number_errors(schema: Mapping[str, Any], value: float, path: str) -> list[str]:
    errors: list[str] = []
    if "minimum" in schema and value < schema["minimum"]:
        errors.append(f"{path}: {value} is below the minimum {schema['minimum']}")
    if "maximum" in schema and value > schema["maximum"]:
        errors.append(f"{path}: {value} is above the maximum {schema['maximum']}")
    if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
        errors.append(f"{path}: {value} must be above {schema['exclusiveMinimum']}")
    if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
        errors.append(f"{path}: {value} must be below {schema['exclusiveMaximum']}")
    step = schema.get("multipleOf")
    if step and not math.isclose(value / step, round(value / step), abs_tol=1e-9):
        errors.append(f"{path}: {value} is not a multiple of {step}")
    return errors


def project(schema: Any, value: Any, root: Mapping[str, Any] | None = None) -> Any:
    """``value`` as the emitted judge's model dumps it. Call only on a valid value.

    Declared properties only (undeclared keys dropped), an omitted optional
    property as its ``default`` or ``None``, a union through its first
    matching branch, an integral float under ``integer`` as an int.
    """
    if not isinstance(schema, Mapping) or not schema:
        return copy.deepcopy(value)
    root = schema if root is None else root
    schema = _resolve(schema, root)
    for key in ("anyOf", "oneOf"):
        if key in schema:
            for option in schema[key]:
                if not schema_errors(option, value, "value", root):
                    return project(option, value, root)
            raise ValueError("project() needs a value that matches the schema")
    kinds = schema.get("type")
    allowed = [kinds] if isinstance(kinds, str) else list(kinds or ())
    if "integer" in allowed and isinstance(value, float) and _is_type("integer", value):
        return int(value)
    if isinstance(value, dict) and ("object" in allowed or not allowed):
        properties: Mapping[str, Any] = schema.get("properties") or {}
        if not properties:
            return copy.deepcopy(value)
        out: dict[str, Any] = {}
        for key, prop in properties.items():
            if key in value:
                item = value[key]
                out[key] = None if item is None else project(prop, item, root)
            elif isinstance(prop, Mapping) and "default" in prop:
                out[key] = copy.deepcopy(prop["default"])
            else:
                out[key] = None
        return out
    if isinstance(value, list) and isinstance(schema.get("items"), Mapping):
        return [project(schema["items"], item, root) for item in value]
    return copy.deepcopy(value)
