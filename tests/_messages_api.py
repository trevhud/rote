"""What the real Anthropic Messages API refuses, for the local stubs.

Every Messages stub in the suite answers through these checks first, so
a request the real API would 400 gets a 400 here too. Without that, a
stub accepts whatever rote sends and a request shape that is dead on
arrival against the real API (forced ``tool_choice`` on
claude-sonnet-5-5) passes every test.

The rules were measured against the live API on 2026-10-20, with
claude-haiku-5-5 and claude-sonnet-5-5 giving identical answers. They
are written independently of :func:`rote.inference.structured_output_schema`
on purpose: a check that called the normalizer would agree with it by
construction.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler
from typing import Any

#: The model the stubs reject forced tool choice for, as the real API
#: does. claude-haiku-5-5 and claude-sonnet-4-6 still accepted it on
#: 2026-10-20, so tests that guard against forcing run on this one.
CURRENT_SONNET = "claude-sonnet-5-5"

FORCED_TOOL_CHOICE_ERROR = 'tool_choice: type "tool" and "any" are not supported for this model.'

#: Keywords the live API rejected inside ``output_config.format.schema``.
_REJECTED_KEYWORDS = frozenset(
    {
        "oneOf",
        "not",
        "patternProperties",
        "maxItems",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
    }
)
_FORMATS = frozenset(
    {"date-time", "time", "date", "duration", "email", "hostname", "uri", "ipv4", "ipv6", "uuid"}
)
_TYPELESS_OK = ("enum", "const", "$ref", "anyOf", "allOf")


def format_schema_problem(schema: Any, path: str = "#") -> str | None:
    """The live API's 400 message for an ``output_config`` schema, or None."""
    if not isinstance(schema, dict):
        return f"{path}: schema must be an object"
    for key in schema:
        if key in _REJECTED_KEYWORDS:
            return f"{path}: Schema keyword {key!r} is not supported"
    if schema.get("format") is not None and schema["format"] not in _FORMATS:
        return f"{path}: format {schema['format']!r} is not supported"
    if schema.get("minItems") not in (None, 0, 1):
        return f"{path}: 'minItems' values other than 0 or 1 are not supported"
    kind = schema.get("type")
    kinds = kind if isinstance(kind, list) else [kind]
    if "object" in kinds and schema.get("additionalProperties") is not False:
        return f"{path}: For 'object' type, 'additionalProperties' must be explicitly set to false"
    if kind is None and not any(key in schema for key in _TYPELESS_OK):
        return f"{path}: Schema type is missing"
    children: list[tuple[str, Any]] = []
    for name, sub in (schema.get("properties") or {}).items():
        children.append((f"{path}/properties/{name}", sub))
    for name, sub in (schema.get("$defs") or {}).items():
        children.append((f"{path}/$defs/{name}", sub))
    for key in ("anyOf", "allOf"):
        for index, sub in enumerate(schema.get(key) or []):
            children.append((f"{path}/{key}/{index}", sub))
    if isinstance(schema.get("items"), dict):
        children.append((f"{path}/items", schema["items"]))
    for child_path, sub in children:
        problem = format_schema_problem(sub, child_path)
        if problem is not None:
            return problem
    return None


#: Sampling parameters API version 2026-10-21 removed. Any of them, even
#: at its default value, is a 400 there (the migration guide,
#: platform.claude.com/docs/en/api/migration-2026-10-21#sampling-parameters).
#: Documented, not measured: no inference was spent on it, so the stub's
#: message text is its own, not the API's.
SAMPLING_PARAMETERS = ("temperature", "top_p", "top_k")

#: Current-generation models that refuse a non-default ``temperature`` on
#: any API version, with the live API's message for each. Measured; models
#: older than 4.7 (claude-haiku-4-5, claude-sonnet-4-6) still accept one.
TEMPERATURE_REJECTIONS = {
    CURRENT_SONNET: "`temperature` is not supported on this model",
    "claude-haiku-5-5": "`temperature` is deprecated for this model.",
}

#: The API version anthropic 2.x and @anthropic-ai/sdk 1.x send.
SAMPLING_FREE_API_VERSION = "2026-10-21"


def api_rejection(payload: dict[str, Any], api_version: str | None = None) -> str | None:
    """The 400 message the real API returns for ``payload``, or None.

    ``api_version`` is the request's ``anthropic-version`` header.
    """
    sent = [name for name in SAMPLING_PARAMETERS if name in payload]
    if sent and api_version == SAMPLING_FREE_API_VERSION:
        return f"`{sent[0]}` is not supported in API version {api_version}"
    rejected = TEMPERATURE_REJECTIONS.get(payload.get("model", ""))
    if rejected is not None and payload.get("temperature", 1.0) != 1.0:
        return rejected
    choice = payload.get("tool_choice") or {}
    if payload.get("model") == CURRENT_SONNET and choice.get("type") in ("tool", "any"):
        return FORCED_TOOL_CHOICE_ERROR
    output_format = (payload.get("output_config") or {}).get("format")
    if output_format is not None:
        problem = format_schema_problem(output_format.get("schema"))
        if problem is not None:
            return f"output_config.format.schema: {problem}"
    return None


def send_json(handler: BaseHTTPRequestHandler, body: dict[str, Any], status: int = 200) -> None:
    raw = json.dumps(body).encode()
    handler.send_response(status)
    handler.send_header("content-type", "application/json")
    handler.send_header("content-length", str(len(raw)))
    handler.end_headers()
    handler.wfile.write(raw)


def reject_like_the_api(handler: BaseHTTPRequestHandler, payload: dict[str, Any]) -> bool:
    """Send the real API's 400 when it would refuse ``payload``; True if sent."""
    message = api_rejection(payload, handler.headers.get("anthropic-version"))
    if message is None:
        return False
    send_json(
        handler,
        {"type": "error", "error": {"type": "invalid_request_error", "message": message}},
        status=400,
    )
    return True


def structured_message(payload: dict[str, Any], answer: Any, **fields: Any) -> dict[str, Any]:
    """A structured-output reply: the answer as JSON text, ``end_turn``."""
    return {
        "id": "msg_structured",
        "type": "message",
        "role": "assistant",
        "model": payload.get("model", "stub"),
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": json.dumps(answer)}],
        "usage": {"input_tokens": 10, "output_tokens": 5},
        **fields,
    }
