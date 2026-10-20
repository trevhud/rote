"""The rewrite of a judge's output schema into Anthropic structured outputs.

Every claim here was measured against the live API on 2026-10-20 (see
``tests/_messages_api.py``). The API check is an independent
implementation of those measurements, so "the rewrite passes the check"
is not the normalizer agreeing with itself.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from rote.adapters import ADAPTERS
from rote.compiler.verify import EmitTarget, emit_findings
from rote.inference import structured_output_schema
from rote.ir import Pipeline
from tests._messages_api import format_schema_problem

REPO = Path(__file__).resolve().parent.parent


def _example_judge_schemas() -> list[tuple[str, dict[str, Any]]]:
    found = []
    for path in sorted(REPO.glob("examples/**/pipeline.yaml")):
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        for node in doc.get("nodes", []):
            spec = node.get("signature_spec")
            if spec:
                label = f"{path.relative_to(REPO).parent}:{node['id']}"
                found.append((label, spec["output_schema"]))
    return found


EXAMPLES = _example_judge_schemas()


def test_the_example_corpus_is_not_degenerate() -> None:
    assert len(EXAMPLES) >= 20
    # Five of the 23 are 400s as written (the live API and the stand-in
    # check agreed on all 23 on 2026-10-20), which is what makes the
    # corpus a test of the rewrite rather than of the identity function.
    rejected = [label for label, schema in EXAMPLES if format_schema_problem(schema) is not None]
    assert len(rejected) >= 5


@pytest.mark.parametrize(("label", "schema"), EXAMPLES, ids=[label for label, _ in EXAMPLES])
def test_every_example_judge_schema_rewrites_into_one_the_api_accepts(
    label: str, schema: dict[str, Any]
) -> None:
    """All 23 were also sent to claude-haiku-5-5 on 2026-10-20: every one
    returned end_turn with JSON."""
    rewritten = structured_output_schema(schema, node_id="judge")
    assert format_schema_problem(rewritten) is None


@pytest.mark.parametrize(
    ("schema", "problem"),
    [
        ({"type": "object", "properties": {"a": {"type": "string"}}}, "additionalProperties"),
        (
            {"type": "object", "properties": {}, "additionalProperties": True},
            "additionalProperties",
        ),
        ({"type": "integer", "minimum": 1}, "'minimum'"),
        ({"type": "string", "maxLength": 3}, "'maxLength'"),
        ({"anyOf": [{"type": "string"}], "oneOf": [{"type": "string"}]}, "'oneOf'"),
        ({"type": "string", "format": "phone"}, "format 'phone'"),
        ({"type": "array", "items": {"type": "string"}, "minItems": 2}, "minItems"),
        ({"type": "array", "items": {"type": "string"}, "maxItems": 3}, "'maxItems'"),
        ({"type": "string", "not": {"const": "x"}}, "'not'"),
        ({"description": "any"}, "type is missing"),
    ],
)
def test_the_stand_in_api_refuses_what_the_live_api_refused(
    schema: dict[str, Any], problem: str
) -> None:
    found = format_schema_problem(schema)
    assert found is not None and problem in found


def test_objects_with_properties_are_closed_at_every_depth() -> None:
    schema = {
        "type": "object",
        "properties": {
            "outer": {"type": "object", "properties": {"inner": {"type": "string"}}},
            "list": {"type": "array", "items": {"type": "object", "properties": {"x": {}}}},
        },
        "$defs": {"D": {"type": "object", "properties": {"y": {"type": "integer"}}}},
    }
    schema["properties"]["list"]["items"]["properties"]["x"] = {"type": "number"}
    out = structured_output_schema(schema, node_id="n")
    assert out["additionalProperties"] is False
    assert out["properties"]["outer"]["additionalProperties"] is False
    assert out["properties"]["list"]["items"]["additionalProperties"] is False
    assert out["$defs"]["D"]["additionalProperties"] is False
    # A schema that admitted extra keys is closed too: it narrows what the
    # model may write, and the answer still fits the original.
    opened = {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "additionalProperties": True,
    }
    assert structured_output_schema(opened, node_id="n")["additionalProperties"] is False


def test_unsupported_constraints_move_into_the_description() -> None:
    schema = {
        "type": "object",
        "properties": {
            "score": {"type": "integer", "minimum": 1, "maximum": 10, "description": "Score."},
            "code": {"type": "string", "pattern": "^[A-Z]{3}$", "format": "phone"},
            "tags": {"type": "array", "items": {"type": "string"}, "minItems": 2},
            "one": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "at": {"type": "string", "format": "date-time"},
        },
        "required": ["score"],
        "additionalProperties": False,
    }
    out = structured_output_schema(schema, node_id="n")
    score = out["properties"]["score"]
    assert score == {
        "type": "integer",
        "description": 'Score.\n\n{"maximum": 10, "minimum": 1}',
    }
    code = out["properties"]["code"]
    assert code == {"type": "string", "description": '{"format": "phone", "pattern": "^[A-Z]{3}$"}'}
    assert out["properties"]["tags"] == {
        "type": "array",
        "items": {"type": "string"},
        "description": '{"minItems": 2}',
    }
    # What the API accepts stays where it is.
    assert out["properties"]["one"]["minItems"] == 1
    assert out["properties"]["at"] == {"type": "string", "format": "date-time"}
    assert out["required"] == ["score"]


def test_one_of_becomes_any_of() -> None:
    schema = {"oneOf": [{"type": "string"}, {"type": "integer"}]}
    assert structured_output_schema(schema, node_id="n") == {
        "anyOf": [{"type": "string"}, {"type": "integer"}]
    }


def test_what_the_api_accepts_as_written_is_left_alone() -> None:
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "title": "Out",
        "properties": {
            "kind": {"enum": ["a", "b"]},
            "fixed": {"const": 3},
            "maybe": {"type": ["string", "null"]},
            "ref": {"$ref": "#/$defs/Kind", "description": "Which."},
            "both": {"allOf": [{"$ref": "#/$defs/Kind"}]},
        },
        "required": ["kind"],
        "additionalProperties": False,
        "$defs": {"Kind": {"type": "string", "enum": ["x", "y"]}},
    }
    assert structured_output_schema(schema, node_id="n") == schema


@pytest.mark.parametrize(
    ("schema", "path"),
    [
        ({"type": "object"}, "#"),
        ({"type": "object", "properties": {}}, "#"),
        ({"type": "object", "additionalProperties": {"type": "string"}}, "#"),
        (
            {
                "type": "object",
                "properties": {"meta": {"type": "object", "additionalProperties": True}},
                "additionalProperties": False,
            },
            "#/properties/meta",
        ),
        (
            {
                "type": "object",
                "properties": {"items": {"type": "array", "items": {}}},
                "additionalProperties": False,
            },
            "#/properties/items/items",
        ),
        (
            {
                "type": "object",
                "properties": {"x": {"description": "Value of type Account."}},
                "additionalProperties": False,
            },
            "#/properties/x",
        ),
    ],
)
def test_shapes_the_subset_cannot_express_raise_naming_the_path(
    schema: dict[str, Any], path: str
) -> None:
    """Closing a free-form map would make the model answer ``{}``; a value
    with no type has no grammar. Neither is approximated."""
    with pytest.raises(RuntimeError) as excinfo:
        structured_output_schema(schema, node_id="vet_contact")
    assert str(excinfo.value).startswith(f"vet_contact: output schema at {path} ")


def test_an_explicitly_empty_object_is_expressible() -> None:
    schema = {"type": "object", "properties": {}, "additionalProperties": False}
    assert structured_output_schema(schema, node_id="n") == schema


def test_the_input_schema_is_not_mutated() -> None:
    schema = {
        "type": "object",
        "properties": {"a": {"type": "integer", "minimum": 0}},
        "$defs": {"D": {"type": "object", "properties": {"b": {"type": "string"}}}},
    }
    before = copy.deepcopy(schema)
    structured_output_schema(schema, node_id="n")
    assert json.dumps(schema, sort_keys=True) == json.dumps(before, sort_keys=True)


# ───────── the compile's dry emit sees the refusal ─────────

#: What a System-1 skeleton judge looks like before the agent fills in its
#: output schema: a bare object.
_UNFILLED_JUDGE = Pipeline.model_validate(
    {
        "name": "unfilled-judge",
        "version": "0.1.0",
        "source_skill": "tests/fixtures/unfilled-judge",
        "description": "One judge whose output schema was never filled in.",
        "input": {"type": "Essay", "required": ["essay"], "optional": []},
        "nodes": [
            {
                "id": "grade_essay",
                "kind": "llm_judge",
                "description": "Grade an essay.",
                "inputs": {"essay": "pipeline.input.essay"},
                "signature_spec": {
                    "input_schema": {
                        "type": "object",
                        "properties": {"essay": {"type": "string"}},
                        "required": ["essay"],
                    },
                    "output_schema": {"type": "object"},
                    "prompt": "Grade: {{ essay }}",
                },
            }
        ],
        "edges": [],
        "entry_nodes": ["grade_essay"],
        "exit_nodes": ["grade_essay"],
    }
)


@pytest.mark.parametrize("runtime", sorted(ADAPTERS))
def test_an_unexpressible_judge_schema_reaches_the_compile_repair_loop(
    runtime: str, tmp_path: Path
) -> None:
    """Every runtime refuses it, and the compile's dry emit reports the
    refusal as an error finding, so the agent fixes it before the compile
    says it succeeded rather than at `rote emit` afterwards."""
    findings = emit_findings(_UNFILLED_JUDGE, tmp_path, EmitTarget(runtime))
    (finding,) = findings
    assert finding.kind == "emit_refused"
    assert finding.severity == "error"
    assert "grade_essay: output schema at # is an object with no declared properties" in (
        finding.message
    )
