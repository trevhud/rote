"""One schema validator, proven the same in Python, TS and the emitted models.

``schema_errors`` (``rote.inference._runtime_helper``, the verbatim
``_rote_inference.py``) checks every answer rote validates outside a
schema-locked decoder, on every lane: a judge's answer and an agent loop's
declared output on every Python runtime, and a compiled skill's judge, gate
and loop answers. Its TS twin, ``SCHEMA_ERRORS_TS`` (emitted as
``signatures/_roteSchema.ts``), checks judges and agent loops on the TS
runtimes. Three legs, one corpus:

- **Python against the emitted Pydantic models**, for every judge schema
  under ``examples/`` (input and output): valid values and one mutation of
  each kind the model enforces get the same verdict, and every accepted value
  comes out of :func:`project` exactly as ``model_dump(by_alias=True)``
  returns it. The constraints the model does not carry (bounds, ``pattern``,
  ``const``, item counts) are stricter here on purpose, tested rule by rule.
- **Python against TS, run by real node**: the same corpus plus every agent
  loop's and gate's schema in ``examples/``, and a positive and a negative
  case for each rule (:data:`RULES`), must get the same verdict and name the
  same paths. Slow: it needs node.
"""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pydantic
import pytest
import yaml

from rote.adapters._common import agent_loop_output
from rote.adapters._py_common import emit_signature_module
from rote.adapters._ts_common import SCHEMA_ERRORS_TS
from rote.inference._runtime_helper import project, schema_errors
from rote.ir import Node, NodeKind, Pipeline
from rote.plan import declared_answer_schema

REPO = Path(__file__).resolve().parent.parent


def _pipelines() -> list[tuple[str, Pipeline]]:
    found: list[tuple[str, Pipeline]] = []
    for path in sorted((REPO / "examples").rglob("pipeline.yaml")):
        if "runtime" in path.parts:
            continue
        pipeline = Pipeline.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
        found.append((str(path.relative_to(REPO / "examples").parent), pipeline))
    return found


PIPELINES = _pipelines()


def _judges() -> list[tuple[str, Node]]:
    return [
        (f"{rel}:{node.id}", node)
        for rel, pipeline in PIPELINES
        for node in pipeline.nodes
        if node.kind is NodeKind.LLM_JUDGE and node.signature_spec is not None
    ]


JUDGES = _judges()


def _models(label: str, node: Node) -> tuple[type[pydantic.BaseModel], type[pydantic.BaseModel]]:
    """The emitted judge module's input and output models, imported for real."""
    source = emit_signature_module(
        node,
        anthropic_default_model="m",
        openai_default_model="m",
        generated_by="t",
        regen_command="t",
        context_note="t",
    )
    name = "_parity_" + "".join(c if c.isalnum() else "_" for c in label)
    module = types.ModuleType(name)
    sys.modules[name] = module
    try:
        exec(compile(source, name, "exec"), module.__dict__)
        pascal = "".join(p.capitalize() for p in node.id.split("_"))
        inp = module.__dict__[f"{pascal}Input"]
        out = module.__dict__[f"{pascal}Output"]
        inp.model_rebuild(_types_namespace=module.__dict__)
        out.model_rebuild(_types_namespace=module.__dict__)
        return inp, out
    finally:
        del sys.modules[name]


def _resolve(schema: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    while "$ref" in schema:
        schema = root["$defs"][schema["$ref"].rsplit("/", 1)[1]]
    return schema


def _example(schema: Any, root: dict[str, Any], *, full: bool) -> Any:
    """A value ``schema`` accepts: every property (``full``) or only the required ones."""
    if not isinstance(schema, dict) or not schema:
        return "anything"
    schema = _resolve(schema, root)
    if "enum" in schema:
        return schema["enum"][0]
    for key in ("anyOf", "oneOf"):
        if key in schema:
            options = [o for o in schema[key] if _resolve(o, root).get("type") != "null"]
            return _example(options[0], root, full=full) if options else None
    kind = schema.get("type")
    if isinstance(kind, list):
        kind = next(k for k in kind if k != "null")
    if kind == "object":
        required = set(schema.get("required") or ())
        props = schema.get("properties") or {}
        extra = schema.get("additionalProperties")
        free = _example(extra, root, full=full) if isinstance(extra, dict) else 1
        return {
            k: _example(p, root, full=full) for k, p in props.items() if full or k in required
        } or {"free": free}
    if kind == "array":
        items = schema.get("items")
        return [_example(items, root, full=full) for _ in range(2)] if items else [1, "a"]
    if kind == "string":
        return "text"
    if kind == "integer":
        return int(schema.get("minimum", 1))
    if kind == "number":
        return float(schema.get("minimum", 1.5))
    if kind == "boolean":
        return True
    if kind == "null":
        return None
    return "anything"


def _wrong_value(schema: Any, root: dict[str, Any]) -> Any | None:
    """A value of a type ``schema`` cannot coerce, or None when there is none to pick."""
    if not isinstance(schema, dict) or not schema:
        return None
    schema = _resolve(schema, root)
    if "anyOf" in schema or "oneOf" in schema:
        return None
    kind = schema.get("type")
    kinds = kind if isinstance(kind, list) else [kind]
    if None in kinds:
        return None
    if "object" in kinds:
        return [1]
    if "array" in kinds:
        return {"x": 1}
    return {"x": 1}


def _corpus(schema: dict[str, Any]) -> list[tuple[str, Any]]:
    root = schema
    full = _example(schema, root, full=True)
    minimal = _example(schema, root, full=False)
    cases: list[tuple[str, Any]] = [("full", full), ("minimal", minimal)]
    props = schema.get("properties") or {}
    required = set(schema.get("required") or ())
    for key, prop in props.items():
        defaulted = isinstance(prop, dict) and "default" in prop
        if key in required and not defaulted:
            cases.append((f"drop {key}", {k: v for k, v in full.items() if k != key}))
        else:
            cases.append((f"null {key}", {**full, key: None}))
        wrong = _wrong_value(prop, root)
        if wrong is not None:
            cases.append((f"wrong type {key}", {**full, key: wrong}))
        resolved = _resolve(prop, root) if isinstance(prop, dict) else {}
        if "enum" in resolved and all(isinstance(e, str) for e in resolved["enum"]):
            cases.append((f"enum {key}", {**full, key: "not-one-of-them"}))
        # A map (no declared properties) is the stricter case, tested below.
        nested = (
            resolved if resolved.get("type") == "object" and resolved.get("properties") else None
        )
        if nested and isinstance(full.get(key), dict):
            inner = copy.deepcopy(full)
            inner[key]["zz_extra"] = 1
            cases.append((f"extra in {key}", inner))
            for sub in nested.get("required") or ():
                dropped = copy.deepcopy(full)
                dropped[key].pop(sub, None)
                cases.append((f"drop {key}.{sub}", dropped))
    cases.append(("extra key", {**full, "zz_extra": 1}))
    cases.append(("not an object", [full]))
    return cases


def _pydantic_verdict(model: type[pydantic.BaseModel], value: Any) -> tuple[bool, Any]:
    try:
        return True, model.model_validate(value).model_dump(by_alias=True)
    except pydantic.ValidationError:
        return False, None


def _same(a: Any, b: Any) -> bool:
    """Equal values of the same types throughout: ``3`` is not ``3.0`` here."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b, strict=True))
    return bool(a == b)


def _check(label: str, schema: dict[str, Any], model: type[pydantic.BaseModel], cases: Any) -> None:
    for case, value in cases:
        accepted, dumped = _pydantic_verdict(model, value)
        errors = schema_errors(schema, value)
        assert (not errors) == accepted, (label, case, errors)
        if accepted:
            ours = project(schema, value)
            assert _same(ours, dumped), (label, case, ours, dumped)


@pytest.mark.parametrize(("label", "node"), JUDGES, ids=[label for label, _ in JUDGES])
def test_verdicts_and_projections_match_the_emitted_models(label: str, node: Node) -> None:
    assert node.signature_spec is not None
    input_model, output_model = _models(label, node)
    for side, schema, model in (
        ("input", node.signature_spec.input_schema, input_model),
        ("output", node.signature_spec.output_schema, output_model),
    ):
        cases = _corpus(schema)
        assert len(cases) >= 4, (label, side)
        _check(f"{label} {side}", schema, model, cases)


_A = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}
_AB = {
    "type": "object",
    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
    "required": ["a"],
}

#: Shapes the example corpus lacks, where a union's branch or a number's type
#: decides what the model dumps. ``(name, the value's schema, cases)``.
PROBES: list[tuple[str, dict[str, Any], list[tuple[str, Any]]]] = [
    # Smart mode: the branch with the most fields set wins, not the first match.
    ("more fields win", {"anyOf": [_A, _AB]}, [("a and b", {"a": 1, "b": 2}), ("a", {"a": 1})]),
    ("tie goes left", {"anyOf": [_AB, _A]}, [("a", {"a": 1}), ("a and b", {"a": 1, "b": 2})]),
    ("number dumps a float", {"type": "number"}, [("int", 3), ("float", 3.0), ("half", 3.5)]),
    ("integer dumps an int", {"type": "integer"}, [("int", 3), ("integral float", 3.0)]),
    (
        "exact number type wins",
        {"anyOf": [{"type": "integer"}, {"type": "number"}]},
        [("int", 3), ("float", 3.0), ("half", 3.5)],
    ),
    (
        "exact number type wins, reversed",
        {"anyOf": [{"type": "number"}, {"type": "integer"}]},
        [("int", 3), ("float", 3.0)],
    ),
    (
        "oneOf is a union",
        {"oneOf": [{"type": "string"}, {"type": "integer"}]},
        [("string", "x"), ("int", 2), ("neither", [1]), ("null", None)],
    ),
    (
        "nullable enum",
        {"anyOf": [{"type": "string", "enum": ["a", "b"]}, {"type": "null"}]},
        [("a", "a"), ("null", None), ("other", "c")],
    ),
]


@pytest.mark.parametrize(("name", "schema", "cases"), PROBES, ids=[p[0] for p in PROBES])
def test_probes_match_the_emitted_model(
    name: str, schema: dict[str, Any], cases: list[tuple[str, Any]]
) -> None:
    root = {"type": "object", "properties": {"v": schema}, "required": ["v"]}
    node = Node.model_validate(
        {
            "id": "probe",
            "kind": "llm_judge",
            "description": "d",
            "signature_spec": {"prompt": "p", "input_schema": root, "output_schema": root},
        }
    )
    _, model = _models(name, node)
    _check(name, root, model, [(case, {"v": value}) for case, value in cases])


def test_the_probes_pin_what_the_model_does() -> None:
    """The cases above are not degenerate: these are the dumps they compare."""
    assert project({"anyOf": [_A, _AB]}, {"a": 1, "b": 2}) == {"a": 1, "b": 2}
    assert project({"anyOf": [_AB, _A]}, {"a": 1}) == {"a": 1, "b": None}
    assert type(project({"type": "number"}, 3)) is float
    assert type(project({"anyOf": [{"type": "number"}, {"type": "integer"}]}, 3)) is int
    assert type(project({"anyOf": [{"type": "integer"}, {"type": "number"}]}, 3.0)) is float


def test_the_corpus_covers_every_kind_of_judge_and_rejects_something() -> None:
    """Non-degenerate: many judges, and mutations that both sides reject."""
    assert len(JUDGES) >= 20
    rejected = 0
    for _, node in JUDGES:
        assert node.signature_spec is not None
        for _, value in _corpus(node.signature_spec.output_schema):
            rejected += bool(schema_errors(node.signature_spec.output_schema, value))
    assert rejected >= 3 * len(JUDGES)


# ───────── the rules, one by one ─────────

OBJ = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["keep", "drop"]},
        "score": {"type": "integer", "minimum": 1, "maximum": 5},
        "note": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "level": {"type": "string", "default": "low"},
    },
    "required": ["label", "score"],
    "additionalProperties": False,
}


def test_a_valid_answer_has_no_errors_and_projects_like_the_model() -> None:
    value = {"label": "keep", "score": 3}
    assert schema_errors(OBJ, value) == []
    # An omitted optional key is None, a default is filled in.
    assert project(OBJ, value) == {
        "label": "keep",
        "score": 3,
        "note": None,
        "tags": None,
        "level": "low",
    }


def test_null_is_allowed_exactly_where_the_model_types_x_or_none() -> None:
    """An optional property with no default is ``X | None = None`` in the model;
    a defaulted or required one is not."""
    assert schema_errors(OBJ, {"label": "keep", "score": 3, "note": None}) == []
    assert schema_errors(OBJ, {"label": "keep", "score": 3, "level": None}) == [
        "answer.level: expected string, got null"
    ]
    assert schema_errors(OBJ, {"label": None, "score": 3}) == [
        "answer.label: None is not one of ['keep', 'drop']"
    ]


def test_each_break_is_named_with_its_path() -> None:
    errors = schema_errors(
        OBJ, {"label": "maybe", "score": 9, "tags": ["a", 2], "extra": 1}, "answer.3"
    )
    assert errors == [
        "answer.3.label: 'maybe' is not one of ['keep', 'drop']",
        "answer.3.score: 9 is above the maximum 5",
        "answer.3.tags[1]: expected string, got integer",
        "answer.3: unexpected key 'extra'",
    ]
    assert schema_errors(OBJ, {"label": "keep"}) == ["answer: missing key 'score'"]


@pytest.mark.parametrize(
    ("schema", "bad", "good", "message"),
    [
        ({"const": "v1"}, "v2", "v1", "must be 'v1'"),
        ({"type": "number", "exclusiveMinimum": 0}, 0, 0.5, "must be above 0"),
        ({"type": "integer", "multipleOf": 5}, 7, 10, "not a multiple of 5"),
        ({"type": "string", "pattern": "^[A-Z]{3}$"}, "usd", "USD", "does not match"),
        ({"type": "string", "minLength": 2}, "a", "ab", "at least 2 characters"),
        ({"type": "string", "maxLength": 2}, "abc", "ab", "at most 2 characters"),
        ({"type": "array", "minItems": 2}, [1], [1, 2], "at least 2 items"),
        ({"type": "array", "maxItems": 1}, [1, 2], [1], "at most 1 items"),
        ({"type": "array", "uniqueItems": True}, [1, 1], [1, 2], "repeats an earlier item"),
        ({"allOf": [{"type": "integer"}, {"minimum": 3}]}, 2, 3, "below the minimum 3"),
    ],
)
def test_constraints_the_model_drops_are_checked(
    schema: dict[str, Any], bad: Any, good: Any, message: str
) -> None:
    [error] = schema_errors(schema, bad)
    assert message in error
    assert schema_errors(schema, good) == []


@pytest.mark.parametrize(
    ("kind", "bad", "got"),
    [
        ("integer", "3", "string"),
        ("integer", True, "boolean"),
        ("boolean", 1, "integer"),
        ("number", "1.5", "string"),
        ("string", 3, "integer"),
    ],
)
def test_no_coercion(kind: str, bad: Any, got: str) -> None:
    """Stricter than Pydantic's lax mode: the agent writes JSON, so it can write the type."""
    assert schema_errors({"type": kind}, bad) == [f"answer: expected {kind}, got {got}"]


def test_an_integral_float_is_an_integer_and_projects_to_int() -> None:
    assert schema_errors({"type": "integer"}, 3.0) == []
    assert project({"type": "integer"}, 3.0) == 3
    assert isinstance(project({"type": "integer"}, 3.0), int)
    assert schema_errors({"type": "integer"}, 3.5) == ["answer: expected integer, got number"]


def test_a_union_matches_any_branch_and_names_the_closest_miss() -> None:
    union = {"anyOf": [{"type": "string"}, {"type": "object", "required": ["a"]}]}
    assert schema_errors(union, "x") == []
    assert schema_errors(union, {"a": 1}) == []
    [error] = schema_errors(union, {"b": 1})
    assert error == "answer: matches none of the allowed shapes (closest: answer: missing key 'a')"


def test_refs_resolve_into_defs_and_foreign_refs_refuse() -> None:
    schema = {
        "type": "object",
        "properties": {"tier": {"$ref": "#/$defs/Tier"}},
        "required": ["tier"],
        "$defs": {"Tier": {"type": "string", "enum": ["A", "B"]}},
    }
    assert schema_errors(schema, {"tier": "A"}) == []
    assert schema_errors(schema, {"tier": "C"}) == ["answer.tier: 'C' is not one of ['A', 'B']"]
    with pytest.raises(ValueError, match="only #/\\$defs"):
        schema_errors({"$ref": "https://example.com/s.json"}, 1)


def test_a_typed_map_checks_its_values() -> None:
    """Stricter than the model's ``dict[str, Any]`` (rote-compile's reference_files)."""
    files = {"type": "object", "additionalProperties": {"type": "string"}}
    assert schema_errors(files, {"a.md": "text"}) == []
    assert schema_errors(files, {"a.md": 1}) == ["answer.a.md: expected string, got integer"]


def test_an_open_object_keeps_and_a_declared_one_drops_undeclared_keys() -> None:
    assert project({"type": "object"}, {"a": 1}) == {"a": 1}
    declared = {"type": "object", "properties": {"a": {"type": "integer"}}}
    assert schema_errors(declared, {"a": 1, "b": 2}) == []
    assert project(declared, {"a": 1, "b": 2}) == {"a": 1}


# ───────── Python against TS ─────────

_OPTIONALS = {
    "type": "object",
    "properties": {
        "req": {"type": "string"},
        "opt": {"type": "string"},
        "dflt": {"type": "string", "default": "d"},
    },
    "required": ["req", "dflt"],
}

_NESTED = {
    "type": "object",
    "properties": {
        "o": {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}
    },
}

_EMOJI = "\U0001f600"

#: Every rule both validators enforce: ``(rule, schema, accepted, rejected)``.
#: Each rejected value breaks that rule and nothing else.
RULES: list[tuple[str, Any, list[Any], list[Any]]] = [
    ("type string", {"type": "string"}, ["", "x"], [3, None, True, [], {}]),
    ("type integer", {"type": "integer"}, [0, -4, 3.0], [3.5, "3", True, None]),
    ("type number", {"type": "number"}, [0, 1.5, -2], ["1.5", False, None]),
    ("type boolean", {"type": "boolean"}, [True, False], [0, 1, "true", None]),
    ("type null", {"type": "null"}, [None], [0, "", False]),
    ("type object", {"type": "object"}, [{}, {"a": 1}], [[], "x", None]),
    ("type array", {"type": "array"}, [[], [1]], [{}, "x", None]),
    ("type list", {"type": ["string", "null"]}, ["x", None], [1, []]),
    ("required", _OPTIONALS, [{"req": "a"}, {"req": "a", "dflt": "b"}], [{}, {"opt": "a"}]),
    (
        "nullable optional",
        _OPTIONALS,
        [{"req": "a", "opt": None}],
        [{"req": None}, {"req": "a", "dflt": None}],
    ),
    (
        "additionalProperties false",
        {"type": "object", "properties": {"a": {}}, "additionalProperties": False},
        [{}, {"a": 1}],
        [{"b": 1}],
    ),
    (
        "undeclared keys pass an open object",
        {"type": "object", "properties": {"a": {"type": "integer"}}},
        [{"a": 1, "b": "x"}],
        [{"a": "x", "b": "x"}],
    ),
    (
        "typed map",
        {"type": "object", "additionalProperties": {"type": "integer"}},
        [{}, {"a": 1, "b": 2}],
        [{"a": "1"}],
    ),
    ("nested", _NESTED, [{"o": {"n": 1}}], [{"o": {}}, {"o": {"n": "1"}}]),
    ("enum", {"enum": ["a", 1, None]}, ["a", 1, 1.0, None], ["b", True, 2]),
    (
        "enum deep",
        {"enum": [[1, {"k": True}]]},
        [[1, {"k": True}]],
        [[1, {"k": 1}], [True, {"k": True}], [1]],
    ),
    ("const", {"const": {"v": 1}}, [{"v": 1}, {"v": 1.0}], [{"v": True}, {"v": 1, "w": 2}]),
    ("const bool", {"const": False}, [False], [0, None]),
    (
        "anyOf",
        {"anyOf": [{"type": "string"}, {"type": "object", "required": ["a"]}]},
        ["x", {"a": 1}],
        [{"b": 1}, 3],
    ),
    ("oneOf", {"oneOf": [{"type": "string"}, {"type": "integer"}]}, ["x", 2], [1.5, None]),
    ("allOf", {"allOf": [{"type": "integer"}, {"minimum": 3}]}, [3, 9], [2, 3.5]),
    ("items", {"type": "array", "items": {"type": "integer"}}, [[], [1, 2]], [[1, "2"]]),
    ("items false", {"type": "array", "items": False}, [[]], [[1]]),
    (
        "$ref",
        {
            "type": "object",
            "properties": {"t": {"$ref": "#/$defs/T"}},
            "$defs": {"T": {"$ref": "#/$defs/U"}, "U": {"enum": ["A"]}},
        },
        [{"t": "A"}],
        [{"t": "B"}],
    ),
    ("minimum", {"type": "number", "minimum": 1}, [1, 2.5], [0.5]),
    ("maximum", {"type": "number", "maximum": 1}, [1, -3], [1.5]),
    ("exclusiveMinimum", {"type": "number", "exclusiveMinimum": 0}, [0.5], [0, -1]),
    ("exclusiveMaximum", {"type": "number", "exclusiveMaximum": 0}, [-0.5], [0, 1]),
    ("multipleOf", {"type": "number", "multipleOf": 0.1}, [0.3, 1, 0], [0.35]),
    ("multipleOf integer", {"type": "integer", "multipleOf": 5}, [10, -5], [7]),
    ("minLength", {"type": "string", "minLength": 2}, ["ab", _EMOJI * 2], ["a", _EMOJI]),
    ("maxLength", {"type": "string", "maxLength": 2}, ["ab", _EMOJI * 2], ["abc", _EMOJI + "ab"]),
    ("pattern", {"type": "string", "pattern": "^[A-Z]{3}$"}, ["USD"], ["usd", "USDX"]),
    ("pattern search", {"type": "string", "pattern": "[0-9]+"}, ["a1b"], ["abc"]),
    ("minItems", {"type": "array", "minItems": 2}, [[1, 2]], [[1]]),
    ("maxItems", {"type": "array", "maxItems": 1}, [[], [1]], [[1, 2]]),
    (
        "uniqueItems",
        {"type": "array", "uniqueItems": True},
        [[1, 2], [{"a": 1}, {"a": True}], [1, True]],
        [[1, 1.0], [{"a": 1, "b": 2}, {"b": 2, "a": 1}]],
    ),
    ("untyped keyword", {"minLength": 2}, ["ab", 1, None], ["a"]),
    ("description only", {"description": "a named type"}, [None, 1, {"a": [1]}], []),
    ("true", True, [None, 1], []),
    ("false", False, [], [None, 1]),
    ("empty", {}, [None, {"a": 1}], []),
]

#: Schemas both validators refuse to read at all.
REFUSED: list[tuple[str, Any]] = [
    ("unknown type", {"type": "date"}),
    ("foreign $ref", {"$ref": "https://example.com/s.json"}),
    ("cyclic $ref", {"$ref": "#/$defs/A", "$defs": {"A": {"$ref": "#/$defs/A"}}}),
    ("missing $ref target", {"$ref": "#/$defs/Nope"}),
    ("not a schema", 3),
    ("pattern neither engine compiles", {"type": "string", "pattern": "(unclosed"}),
]

#: Patterns only one engine compiles: refused, by name, by the other.
PYTHON_ONLY_PATTERN = "(?P<code>[A-Z]+)"  # JS spells a named group (?<code>...)
JS_ONLY_PATTERN = "(?<code>[A-Z]+)"


def _schema_cases(schema: Any) -> list[tuple[str, Any]]:
    """Values for an example schema: :func:`_corpus` for an object, else three."""
    example = _example(schema, schema, full=True)
    if isinstance(example, dict) and isinstance(schema, dict) and schema.get("properties"):
        return _corpus(schema)
    return [("example", example), ("wrong type", [example]), ("null", None)]


def _cross_corpus() -> list[tuple[str, Any, Any]]:
    """``(label, schema, value)`` for every schema the examples declare, and :data:`RULES`."""
    corpus: list[tuple[str, Any, Any]] = []
    for label, node in JUDGES:
        assert node.signature_spec is not None
        for side in ("input_schema", "output_schema"):
            schema = getattr(node.signature_spec, side)
            corpus += [(f"{label} {side} {c}", schema, v) for c, v in _corpus(schema)]
    for rel, pipeline in PIPELINES:
        for node in pipeline.nodes:
            if node.kind is NodeKind.AGENT_LOOP:
                schema = agent_loop_output(node).schema
            elif node.kind is NodeKind.HITL_GATE:
                schema = declared_answer_schema(node)
            else:
                continue
            if schema is not None:
                label = f"{rel}:{node.id}"
                corpus += [(f"{label} {c}", schema, v) for c, v in _schema_cases(schema)]
    for name, schema, cases in PROBES:
        corpus += [(f"probe {name} {c}", schema, v) for c, v in cases]
    for rule, schema, good, bad in RULES:
        corpus += [(f"{rule} accepts {v!r}", schema, v) for v in good]
        corpus += [(f"{rule} rejects {v!r}", schema, v) for v in bad]
    # Round-trip through JSON so both sides read exactly the same input.
    loaded: list[tuple[str, Any, Any]] = json.loads(json.dumps(corpus))
    return loaded


CROSS = _cross_corpus()

_TS_MAIN = """
import { readFileSync } from "node:fs";

const cases = JSON.parse(readFileSync(0, "utf8")) as [string, unknown, unknown][];
const out = cases.map(([, schema, value]) => {
    try {
        return schemaErrors(schema, value, "answer");
    } catch {
        return null;
    }
});
process.stdout.write(JSON.stringify(out));
"""


def _paths(errors: list[str]) -> list[str]:
    """Where each error is: the part both languages word the same way."""
    return sorted(e.split(": ", 1)[0] for e in errors)


def _python_verdict(schema: Any, value: Any) -> list[str] | None:
    """The sorted error paths, ``[]`` when valid, None when the schema is refused."""
    try:
        return _paths(schema_errors(schema, value, "answer"))
    except ValueError:
        return None


def _ts_verdicts(tmp: Path, cases: list[tuple[str, Any, Any]]) -> list[list[str] | None]:
    """The same, from ``SCHEMA_ERRORS_TS`` run by real node (which strips the types)."""
    node = shutil.which("node")
    assert node, "node is required to run the TS validator"
    script = tmp / "schema_parity.mts"
    script.write_text(SCHEMA_ERRORS_TS + _TS_MAIN, encoding="utf-8")
    done = subprocess.run(
        [node, str(script)], input=json.dumps(cases), capture_output=True, text=True, timeout=60
    )
    assert done.returncode == 0, done.stderr
    return [None if e is None else _paths(e) for e in json.loads(done.stdout)]


def test_every_rule_has_both_halves_and_the_corpus_is_not_degenerate() -> None:
    """Python agrees with each rule's own labels, and the corpus both accepts
    and rejects plenty, so a TS validator that said yes (or no) to everything
    could not pass the parity test below."""
    for rule, schema, good, bad in RULES:
        assert good or bad, rule
        for value in good:
            assert schema_errors(schema, value) == [], (rule, value)
        for value in bad:
            assert schema_errors(schema, value), (rule, value)
    for _, schema in REFUSED:
        with pytest.raises(ValueError):
            schema_errors(schema, "x")
    verdicts = [_python_verdict(schema, value) for _, schema, value in CROSS]
    assert sum(1 for v in verdicts if v) >= 200
    assert sum(1 for v in verdicts if v == []) >= 200


@pytest.mark.slow
def test_python_and_ts_reach_the_same_verdicts(tmp_path: Path) -> None:
    cases = CROSS + [(f"refused {name}", schema, "x") for name, schema in REFUSED]
    theirs = _ts_verdicts(tmp_path, cases)
    differ = [
        (label, ours, ts)
        for (label, schema, value), ts in zip(cases, theirs, strict=True)
        if (ours := _python_verdict(schema, value)) != ts
    ]
    assert not differ, differ[:10]


def test_a_pattern_python_cannot_compile_is_refused_by_name() -> None:
    schema = {"type": "string", "pattern": JS_ONLY_PATTERN}
    with pytest.raises(ValueError, match="pattern '\\(\\?<code>.*Python can compile"):
        schema_errors(schema, "ABC", "output.code")
    assert schema_errors({"type": "string", "pattern": PYTHON_ONLY_PATTERN}, "ABC") == []


def test_ir_load_refuses_a_pattern_python_cannot_compile() -> None:
    """At load, not at the first answer: every judge, node and input schema."""
    bad = {
        "type": "object",
        "properties": {"code": {"anyOf": [{"type": "string", "pattern": "(unclosed"}]}},
    }
    with pytest.raises(pydantic.ValidationError, match="Python can compile"):
        Node.model_validate(
            {
                "id": "probe",
                "kind": "llm_judge",
                "description": "d",
                "signature_spec": {"prompt": "p", "input_schema": {}, "output_schema": bad},
            }
        )
    with pytest.raises(pydantic.ValidationError, match="Python can compile"):
        Node.model_validate(
            {
                "id": "probe",
                "kind": "pure_function",
                "description": "d",
                "impl": "extracted/p.py:p",
                "output_schema": bad,
            }
        )
    # A property that is merely *named* pattern is not a pattern.
    named = {"type": "object", "properties": {"pattern": {"type": "string"}}}
    Node.model_validate(
        {
            "id": "probe",
            "kind": "llm_judge",
            "description": "d",
            "signature_spec": {"prompt": "p", "input_schema": {}, "output_schema": named},
        }
    )


@pytest.mark.slow
def test_a_pattern_javascript_cannot_compile_is_refused_by_name(tmp_path: Path) -> None:
    node = shutil.which("node")
    assert node, "node is required to run the TS validator"
    python_only = json.dumps({"type": "string", "pattern": PYTHON_ONLY_PATTERN})
    js_only = json.dumps({"type": "string", "pattern": JS_ONLY_PATTERN})
    script = tmp_path / "pattern.mts"
    script.write_text(
        SCHEMA_ERRORS_TS
        + "\ntry {\n"
        + f'    schemaErrors({python_only}, "ABC", "output.code");\n'
        + '    console.log("accepted");\n'
        + "} catch (err) {\n"
        + "    console.log((err as Error).message);\n"
        + "}\n"
        + f'console.log(JSON.stringify(schemaErrors({js_only}, "ABC")));\n',
        encoding="utf-8",
    )
    done = subprocess.run([node, str(script)], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    refused, accepted = done.stdout.strip().splitlines()
    assert refused.startswith(
        'output.code: the schema\'s pattern "(?P<code>[A-Z]+)" is not a regular '
        "expression JavaScript can compile ("
    ), refused
    assert accepted == "[]"
