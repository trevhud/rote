"""Tests for schema inference from observed MCP traffic (rote.probe)."""

from __future__ import annotations

import json

from rote.eval.baseline import ObservedToolCall
from rote.probe import (
    cross_check,
    infer_schema,
    infer_tool_schemas,
    parse_tool_result,
)

# ───────── infer_schema ─────────


def test_scalar_types() -> None:
    assert infer_schema(["a", "b"]) == {"type": "string"}
    assert infer_schema([1, 2]) == {"type": "integer"}
    assert infer_schema([1.5]) == {"type": "number"}
    assert infer_schema([True, False]) == {"type": "boolean"}
    assert infer_schema([None]) == {"type": "null"}


def test_bool_is_not_integer() -> None:
    """Python bools are ints; JSON booleans are not integers."""
    assert infer_schema([True, 1]) == {"anyOf": [{"type": "boolean"}, {"type": "integer"}]}


def test_int_and_float_collapse_to_number() -> None:
    assert infer_schema([1, 2.5]) == {"type": "number"}


def test_nullable_becomes_anyof() -> None:
    assert infer_schema(["x", None]) == {"anyOf": [{"type": "null"}, {"type": "string"}]}


def test_object_required_is_key_intersection() -> None:
    schema = infer_schema(
        [
            {"id": 1, "name": "a", "note": "sometimes"},
            {"id": 2, "name": "b"},
        ]
    )
    assert schema == {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "name": {"type": "string"},
            "note": {"type": "string"},
        },
        "required": ["id", "name"],
    }


def test_nested_objects_and_arrays() -> None:
    schema = infer_schema(
        [
            {"threads": [{"id": "t1", "score": 3}, {"id": "t2", "score": 4.5}]},
            {"threads": []},
        ]
    )
    assert schema == {
        "type": "object",
        "properties": {
            "threads": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}, "score": {"type": "number"}},
                    "required": ["id", "score"],
                },
            }
        },
        "required": ["threads"],
    }


def test_empty_array_has_no_items() -> None:
    assert infer_schema([[]]) == {"type": "array"}


def test_no_samples_is_empty_schema() -> None:
    assert infer_schema([]) == {}


# ───────── parse_tool_result ─────────


def test_parse_content_blocks_with_json_text() -> None:
    result = [{"type": "text", "text": '{"word": "graduate", "length": 8}'}]
    assert parse_tool_result(result) == {"word": "graduate", "length": 8}


def test_parse_bare_json_string() -> None:
    assert parse_tool_result('{"a": 1}') == {"a": 1}


def test_parse_non_json_text_stays_text() -> None:
    assert parse_tool_result("stored the word") == "stored the word"


def test_parse_empty_is_none() -> None:
    assert parse_tool_result(None) is None
    assert parse_tool_result("") is None
    assert parse_tool_result([{"type": "text", "text": "  "}]) is None


def test_parse_multiple_text_blocks_concatenate() -> None:
    result = [
        {"type": "text", "text": '{"a": '},
        {"type": "text", "text": "1}"},
    ]
    assert parse_tool_result(result) == {"a": 1}


def test_parse_non_text_blocks_pass_through() -> None:
    blocks = [{"type": "image", "data": "…"}]
    assert parse_tool_result(blocks) == blocks


# ───────── infer_tool_schemas ─────────


def _call(
    tool: str = "lookup_word",
    server: str = "wordbank",
    word: str = "graduate",
    result: dict | None = None,
    is_error: bool = False,
) -> ObservedToolCall:
    payload = result if result is not None else {"word": word, "length": len(word)}
    return ObservedToolCall(
        server=server,
        tool=tool,
        input={"word": word},
        result=[{"type": "text", "text": json.dumps(payload)}],
        is_error=is_error,
    )


def test_infer_tool_schemas_groups_and_infers() -> None:
    observations = [
        _call(word="graduate"),
        _call(word="rote"),
        ObservedToolCall(server="slack", tool="read_channel", input={"channel": "C1"}),
    ]
    inferred = {(s.server, s.tool): s for s in infer_tool_schemas(observations)}
    wordbank = inferred[("wordbank", "lookup_word")]
    assert wordbank.samples == 2
    assert wordbank.input_schema == {
        "type": "object",
        "properties": {"word": {"type": "string"}},
        "required": ["word"],
    }
    assert wordbank.output_schema["properties"]["length"] == {"type": "integer"}
    # No result observed for slack → empty output schema, input still typed.
    slack = inferred[("slack", "read_channel")]
    assert slack.output_schema == {}
    assert slack.input_schema["required"] == ["channel"]


def test_infer_tool_schemas_excludes_error_outputs() -> None:
    observations = [
        _call(word="ok"),
        ObservedToolCall(
            server="wordbank",
            tool="lookup_word",
            input={"word": "denied"},
            result="401 unauthorized",
            is_error=True,
        ),
    ]
    (inferred,) = infer_tool_schemas(observations)
    assert inferred.samples == 2
    assert inferred.error_samples == 1
    # The error body must not pollute the output contract…
    assert inferred.output_schema["properties"]["word"] == {"type": "string"}
    assert "401" not in json.dumps(inferred.output_schema)
    # …but the errored call's input still counts as contract evidence.
    assert inferred.input_schema["required"] == ["word"]


# ───────── cross_check ─────────


def test_cross_check_classifies_three_ways() -> None:
    from pathlib import Path

    from rote.ir import load_pipeline

    repo_root = Path(__file__).resolve().parent.parent
    pipeline = load_pipeline(repo_root / "examples" / "deal-monitor" / "expected" / "pipeline.yaml")
    observations = [
        # Bound in the pipeline AND observed → confirmed.
        ObservedToolCall(server="slack", tool="slack_read_channel", input={}),
        ObservedToolCall(server="slack", tool="slack_read_channel", input={}),
        # Observed but never bound → the loud case.
        ObservedToolCall(server="salesforce", tool="get_account", input={}),
    ]
    result = cross_check(pipeline, observations)

    assert result["confirmed"] == [
        {
            "server": "slack",
            "tool": "slack_read_channel",
            "nodes": ["fetch_intake_messages"],
            "observed_calls": 2,
        }
    ]
    assert result["observed_only"] == [
        {"server": "salesforce", "tool": "get_account", "observed_calls": 1}
    ]
    # Gmail bindings exist but weren't observed.
    static_only_tools = {(e["server"], e["tool"]) for e in result["static_only"]}
    assert ("gmail", "search_threads") in static_only_tools
    assert ("gmail", "get_thread") in static_only_tools


def test_cross_check_empty_observations() -> None:
    from pathlib import Path

    from rote.ir import load_pipeline

    repo_root = Path(__file__).resolve().parent.parent
    pipeline = load_pipeline(repo_root / "examples" / "deal-monitor" / "expected" / "pipeline.yaml")
    result = cross_check(pipeline, [])
    assert result["confirmed"] == []
    assert result["observed_only"] == []
    assert len(result["static_only"]) == 3  # slack + 2 distinct gmail tools


# ───────── enrich_pipeline + typed stub contracts ─────────


def _wordbank_schema() -> object:
    from rote.probe import InferredToolSchema

    return InferredToolSchema(
        server="slack",
        tool="slack_read_channel",
        input_schema={
            "type": "object",
            "properties": {"channel": {"type": "string"}},
            "required": ["channel"],
        },
        output_schema={
            "type": "object",
            "properties": {"messages": {"type": "array", "items": {"type": "string"}}},
            "required": ["messages"],
        },
        samples=2,
        error_samples=0,
    )


def test_enrich_pipeline_fills_only_missing_contracts() -> None:
    from pathlib import Path

    from rote.ir import load_pipeline
    from rote.probe import enrich_pipeline

    repo_root = Path(__file__).resolve().parent.parent
    pipeline = load_pipeline(repo_root / "examples" / "deal-monitor" / "expected" / "pipeline.yaml")
    enriched, ids = enrich_pipeline(pipeline, [_wordbank_schema()])

    assert ids == ["fetch_intake_messages"]
    node = enriched.node_by_id("fetch_intake_messages")
    assert node.input_schema["required"] == ["channel"]
    assert node.output_schema["properties"]["messages"]["type"] == "array"
    # Original untouched; unmatched nodes unchanged.
    assert pipeline.node_by_id("fetch_intake_messages").input_schema is None
    assert enriched.node_by_id("search_gmail_standard").input_schema is None


def test_enrich_pipeline_never_overwrites_existing_contract() -> None:
    from pathlib import Path

    from rote.ir import load_pipeline
    from rote.probe import enrich_pipeline

    repo_root = Path(__file__).resolve().parent.parent
    pipeline = load_pipeline(repo_root / "examples" / "deal-monitor" / "expected" / "pipeline.yaml")
    preset = {"type": "object", "properties": {"authored": {"type": "string"}}}
    nodes = [
        n.model_copy(update={"input_schema": preset}) if n.id == "fetch_intake_messages" else n
        for n in pipeline.nodes
    ]
    pipeline = pipeline.model_copy(update={"nodes": nodes})

    enriched, ids = enrich_pipeline(pipeline, [_wordbank_schema()])
    # output_schema was still missing → enriched; input_schema kept.
    assert ids == ["fetch_intake_messages"]
    node = enriched.node_by_id("fetch_intake_messages")
    assert node.input_schema == preset
    assert node.output_schema is not None


def test_save_pipeline_round_trips_enriched_contracts(tmp_path) -> None:
    from pathlib import Path

    from rote.ir import load_pipeline, save_pipeline
    from rote.probe import enrich_pipeline

    repo_root = Path(__file__).resolve().parent.parent
    pipeline = load_pipeline(repo_root / "examples" / "deal-monitor" / "expected" / "pipeline.yaml")
    enriched, _ = enrich_pipeline(pipeline, [_wordbank_schema()])
    out = tmp_path / "pipeline.yaml"
    save_pipeline(enriched, out)
    reloaded = load_pipeline(out)
    assert reloaded == enriched
    assert reloaded.node_by_id("fetch_intake_messages").output_schema is not None


def test_emitted_stub_carries_observed_contract(tmp_path) -> None:
    """The dbos-emitted extracted stub documents the observed contracts."""
    from pathlib import Path

    from rote.adapters import get_adapter
    from rote.ir import load_pipeline
    from rote.probe import enrich_pipeline

    repo_root = Path(__file__).resolve().parent.parent
    pipeline = load_pipeline(repo_root / "examples" / "deal-monitor" / "expected" / "pipeline.yaml")
    enriched, _ = enrich_pipeline(pipeline, [_wordbank_schema()])
    written = get_adapter("dbos").emit(enriched, tmp_path / "out")

    stub_src = written["extracted/slack"].read_text(encoding="utf-8")
    assert "Input contract (JSON Schema, from observed real payloads):" in stub_src
    assert '"channel"' in stub_src
    assert "Output contract (JSON Schema, from observed real payloads):" in stub_src
    assert '"messages"' in stub_src
    # Unenriched stubs stay contract-free.
    gmail_src = written["extracted/gmail"].read_text(encoding="utf-8")
    assert "Input contract" not in gmail_src
