"""A small JSON Schema checker for the subset vendor specs use.

Enough to hold a fixture body to its operation's response schema: ``type``,
``properties``, ``additionalProperties``, ``items``, ``enum``, ``required``
and ``$ref`` into an :class:`~rote.vendor_specs.model.ApiSpec`'s schemas.
Strict where a vendor spec is: an object whose schema declares
``properties`` and no ``additionalProperties`` accepts no other key, so a
fixture cannot carry a field the real API never returns.
"""

from __future__ import annotations

from typing import Any

from rote.vendor_specs.model import ApiSpec

_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "object": (dict,),
    "array": (list,),
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
}


def problems(spec: ApiSpec, schema: dict[str, Any] | None, value: Any, at: str = "$") -> list[str]:
    """Every way ``value`` breaks ``schema``, as ``"<json path>: <what>"`` lines."""
    resolved = spec.resolve(schema)
    if resolved is None:
        return []
    out: list[str] = []
    kind = resolved.get("type")
    if isinstance(kind, str) and kind in _JSON_TYPES:
        ok = isinstance(value, _JSON_TYPES[kind]) and not (
            kind in ("integer", "number") and isinstance(value, bool)
        )
        # Discovery encodes int64 / uint64 as JSON strings.
        if not ok and kind == "string" and isinstance(value, int) and not isinstance(value, bool):
            ok = resolved.get("format") in ("int64", "uint64")
        if not ok:
            return [f"{at}: expected {kind}, got {type(value).__name__}"]
    enum = resolved.get("enum")
    if enum is not None and value not in enum:
        out.append(f"{at}: {value!r} is not one of {', '.join(map(str, enum))}")
    if isinstance(value, dict):
        props = resolved.get("properties")
        extra = resolved.get("additionalProperties")
        for key in resolved.get("required") or []:
            if key not in value:
                out.append(f"{at}: missing required field {key!r}")
        for key, item in value.items():
            if props is not None and key in props:
                out += problems(spec, props[key], item, f"{at}.{key}")
            elif isinstance(extra, dict):
                out += problems(spec, extra, item, f"{at}.{key}")
            elif props is not None and not extra:
                out.append(f"{at}: field {key!r} is not in the schema")
    elif isinstance(value, list) and isinstance(resolved.get("items"), dict):
        for i, item in enumerate(value):
            out += problems(spec, resolved["items"], item, f"{at}[{i}]")
    return out


def minimal(spec: ApiSpec, schema: dict[str, Any] | None) -> Any:
    """The smallest value ``schema`` accepts: what an API returns when nothing matched.

    Discovery marks nearly nothing required, so a list response with no
    results is ``{}`` (Gmail omits an empty ``threads``), exactly as the
    real API sends it.
    """
    resolved = spec.resolve(schema) or {}
    kind = resolved.get("type", "object")
    if kind == "object":
        props = resolved.get("properties") or {}
        return {k: minimal(spec, props[k]) for k in resolved.get("required") or [] if k in props}
    return {"array": [], "string": "", "integer": 0, "number": 0, "boolean": False}.get(kind)


__all__ = ["minimal", "problems"]
