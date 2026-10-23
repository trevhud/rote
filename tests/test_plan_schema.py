"""``rote.plan._schema`` against the emitted judges' own Pydantic models.

A compiled skill checks an agent's answer with :func:`schema_errors` and
hands downstream steps :func:`project`'s value. On every other runtime the
same answer goes through the judge's generated Pydantic model. So for every
judge schema committed under ``examples/`` (input and output), a corpus of
values built from the schema (valid ones, and one mutation each of the
kinds both sides enforce) must get the same verdict from both, and every
accepted value must come out of :func:`project` exactly as
``model_dump(by_alias=True)`` returns it.

The constraints the model does not carry (bounds, ``pattern``, ``const``,
item counts) are stricter here on purpose, and tested on their own below.
"""

from __future__ import annotations

import copy
import sys
import types
from pathlib import Path
from typing import Any

import pydantic
import pytest
import yaml

from rote.adapters._py_common import emit_signature_module
from rote.ir import Node, NodeKind, Pipeline
from rote.plan._schema import project, schema_errors

REPO = Path(__file__).resolve().parent.parent


def _judges() -> list[tuple[str, Node]]:
    found: list[tuple[str, Node]] = []
    for path in sorted((REPO / "examples").rglob("pipeline.yaml")):
        if "runtime" in path.parts:
            continue
        pipeline = Pipeline.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
        for node in pipeline.nodes:
            if node.kind is NodeKind.LLM_JUDGE and node.signature_spec is not None:
                rel = path.relative_to(REPO / "examples").parent
                found.append((f"{rel}:{node.id}", node))
    return found


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
