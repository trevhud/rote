"""An agent loop's declared-output contract, derived once for every runtime."""

from __future__ import annotations

from typing import Any

import pytest

from rote.adapters._common import agent_loop_output, ir_type
from rote.ir import Node


@pytest.mark.parametrize(
    ("name", "schema", "ts"),
    [
        ("str", {"type": "string"}, "string"),
        ("int", {"type": "integer"}, "number"),
        ("float", {"type": "number"}, "number"),
        ("bool", {"type": "boolean"}, "boolean"),
        ("str | null", {"anyOf": [{"type": "string"}, {"type": "null"}]}, "string | null"),
        ("Optional[int]", {"anyOf": [{"type": "integer"}, {"type": "null"}]}, "number | null"),
        ("list[str]", {"type": "array", "items": {"type": "string"}}, "string[]"),
        (
            "list[str | null]",
            {"type": "array", "items": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
            "(string | null)[]",
        ),
        ("dict[str, int]", {"type": "object"}, "{ [key: string]: JsonValue }"),
        ("InvoiceResult", {"description": "Value of type InvoiceResult."}, "JsonValue"),
        (
            "list[InvoiceResult]",
            {"type": "array", "items": {"description": "Value of type InvoiceResult."}},
            "JsonValue[]",
        ),
    ],
)
def test_ir_type_maps_to_a_schema_and_a_ts_type(name: str, schema: Any, ts: str) -> None:
    assert ir_type(name) == (schema, ts)


def _loop(**fields: Any) -> Node:
    return Node.model_validate(
        {"id": "research", "kind": "agent_loop", "description": "d", "tools": ["t"], **fields}
    )


def test_declared_keys_are_required_and_closed() -> None:
    contract = agent_loop_output(_loop(output={"summary": "str", "score": "int"}))
    assert contract.schema == {
        "type": "object",
        "properties": {"summary": {"type": "string"}, "score": {"type": "integer"}},
        "required": ["summary", "score"],
        "additionalProperties": False,
    }
    assert contract.unwrap_key is None
    assert contract.ts_type == '{ "summary": string; "score": number }'


def test_output_schema_evidence_beats_the_type_name() -> None:
    observed = {"type": "object", "properties": {"id": {"type": "string"}}}
    contract = agent_loop_output(
        _loop(
            output={"invoice": "InvoiceResult", "ok": "bool"},
            output_schema={"type": "object", "properties": {"invoice": observed}},
        )
    )
    assert contract.schema is not None
    assert contract.schema["properties"] == {"invoice": observed, "ok": {"type": "boolean"}}


def test_a_single_type_is_submitted_under_result_and_unwrapped() -> None:
    contract = agent_loop_output(_loop(output="IntelBrief"))
    assert contract.schema is not None
    assert contract.schema["required"] == ["result"]
    assert contract.unwrap_key == "result"
    assert contract.ts_type == "JsonValue"


def test_no_declared_output_keeps_the_final_text() -> None:
    contract = agent_loop_output(_loop())
    assert (contract.schema, contract.unwrap_key, contract.ts_type) == (
        None,
        None,
        "{ result: string }",
    )
