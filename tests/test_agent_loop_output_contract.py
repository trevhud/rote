"""An agent loop's declared-output contract, derived once for every runtime."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rote.adapters import get_adapter
from rote.adapters._common import agent_loop_output, ir_type
from rote.cli import main
from rote.ir import Node, load_pipeline


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
        # A named type the IR never defines: any JSON value, and an alias on TS.
        ("InvoiceResult", {"description": "Value of type InvoiceResult."}, "InvoiceResult"),
        (
            "list[InvoiceResult]",
            {"type": "array", "items": {"description": "Value of type InvoiceResult."}},
            "InvoiceResult[]",
        ),
        ("Weird-Name", {"description": "Value of type Weird-Name."}, "JsonValue"),
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
    assert contract.ts_type == "IntelBrief"
    assert contract.named_types == {"result": ["IntelBrief"]}
    assert contract.strict is False


def test_no_declared_output_keeps_the_final_text() -> None:
    contract = agent_loop_output(_loop())
    assert (contract.schema, contract.unwrap_key, contract.ts_type) == (
        None,
        None,
        "{ result: string }",
    )


# ───────── undefined named types: an alias on TS, and a note at emit ─────────

_INVOICE_PUSH = (
    Path(__file__).resolve().parent.parent
    / "examples"
    / "invoice-push"
    / "expected"
    / "pipeline.yaml"
)


def test_strict_needs_every_leaf_typed() -> None:
    typed = agent_loop_output(_loop(output={"summary": "str", "tags": "list[str]"}))
    named = agent_loop_output(_loop(output={"summary": "str", "rows": "list[Row]"}))
    assert typed.strict is True
    # A named type is description-only, outside the strict subset.
    assert named.strict is False
    assert agent_loop_output(_loop()).strict is False


@pytest.mark.parametrize("runtime", ["cloudflare", "dbos-ts", "inngest"])
def test_ts_runtimes_declare_each_named_type_once_in_types_ts(runtime: str, tmp_path: Path) -> None:
    out = tmp_path / runtime
    get_adapter(runtime).emit(load_pipeline(_INVOICE_PUSH), out)
    types_ts = (out / "src" / "types.ts").read_text(encoding="utf-8")
    assert [line for line in types_ts.splitlines() if line.startswith("export type")] == [
        "export type InvoiceResult = JsonValue;"
    ]
    loop = (out / "src" / "extracted" / "process_invoices_loop.ts").read_text(encoding="utf-8")
    assert 'import type { InvoiceResult } from "../types";' in loop
    assert '"invoice_results": InvoiceResult[]' in loop


def test_a_narrowed_types_ts_survives_re_emit(tmp_path: Path) -> None:
    pipeline = load_pipeline(_INVOICE_PUSH)
    out = tmp_path / "inngest"
    get_adapter("inngest").emit(pipeline, out)
    types_ts = out / "src" / "types.ts"
    narrowed = "export interface InvoiceResult {\n    batch_id: string;\n}\n"
    types_ts.write_text(narrowed, encoding="utf-8")
    get_adapter("inngest").emit(pipeline, out)
    assert types_ts.read_text(encoding="utf-8") == narrowed
    assert (out / "src" / "types.ts.new").is_file()


def test_python_runtimes_emit_no_types_module(tmp_path: Path) -> None:
    out = tmp_path / "dbos"
    get_adapter("dbos").emit(load_pipeline(_INVOICE_PUSH), out)
    assert not list(out.rglob("types.ts"))


@pytest.mark.parametrize(
    ("runtime", "advice"),
    [
        ("dbos-ts", "narrow it in src/types.ts"),
        ("dbos", "declare its fields in the node's output_schema"),
    ],
)
def test_emit_notes_each_unchecked_output_type(
    runtime: str, advice: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / runtime
    assert main(["emit", str(_INVOICE_PUSH), "--runtime", runtime, "--out", str(out)]) == 0
    expected = (
        "process_invoices_loop.invoice_results: type InvoiceResult is not defined in the IR; "
        f"shape unchecked, {advice}"
    )
    lines = [line.strip() for line in capsys.readouterr().out.splitlines()]
    assert lines.count(f"note: {expected}") == 1

    assert (
        main(["emit", str(_INVOICE_PUSH), "--runtime", runtime, "--out", str(out), "--json"]) == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["unchecked_output_types"] == [expected]


def test_output_schema_evidence_clears_the_note() -> None:
    from rote.adapters._common import unchecked_output_notes
    from rote.ir import Pipeline

    data = load_pipeline(_INVOICE_PUSH).model_dump(mode="json", by_alias=True, exclude_none=True)
    loop = next(n for n in data["nodes"] if n["id"] == "process_invoices_loop")
    loop["output_schema"] = {
        "type": "object",
        "properties": {"invoice_results": {"type": "array", "items": {"type": "object"}}},
    }
    assert unchecked_output_notes(Pipeline.model_validate(data), types_module=None) == []
