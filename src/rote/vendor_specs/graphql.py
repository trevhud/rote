"""GraphQL APIs: a request is checked against the vendor's own schema.

A GraphQL API has one endpoint and a schema; the request is a document. So
the check is graphql-core's own validation of the request document against
the schema (unknown fields, arguments, types, missing required arguments,
undeclared variables), plus rote's: the variables the code sends against
the ones the document declares, and the response fields the code reads
against the ones the document selects.

:data:`REGISTRY` maps an endpoint to the vendor's published schema: an SDL
file the vendor hosts, or the vendor's own endpoint introspected (a
GraphQL server describes itself; the introspection query needs no
credential where the API itself needs none).

https://spec.graphql.org/October2021/
https://graphql.org/learn/introspection/
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from graphql import (
    DocumentNode,
    FieldNode,
    FragmentDefinitionNode,
    FragmentSpreadNode,
    GraphQLError,
    GraphQLInterfaceType,
    GraphQLNamedType,
    GraphQLObjectType,
    GraphQLScalarType,
    GraphQLSchema,
    InlineFragmentNode,
    OperationDefinitionNode,
    VariableDefinitionNode,
    build_client_schema,
    build_schema,
    get_introspection_query,
    get_named_type,
    is_input_type,
    is_leaf_type,
    parse,
    print_schema,
    specified_scalar_types,
    type_from_ast,
    validate,
)
from graphql.utilities import validate_input_value

from rote.vendor_specs.extract import UNKNOWN, Expr
from rote.vendor_specs.model import ApiSpec, Operation

if TYPE_CHECKING:
    from graphql import IntrospectionQuery


@dataclass(frozen=True)
class Source:
    """One vendor GraphQL endpoint and where its schema comes from."""

    vendor: str
    api: str
    version: str
    endpoint: str
    #: A vendor-hosted SDL file, or ``None`` to introspect ``endpoint``.
    sdl_url: str | None
    checked: str


#: Endpoint URL -> the vendor's schema. Endpoints are matched exactly.
REGISTRY: dict[str, Source] = {
    "https://api.github.com/graphql": Source(
        "github",
        "graphql",
        "v4",
        "https://api.github.com/graphql",
        "https://docs.github.com/public/fpt/schema.docs.graphql",
        "2026-10-20",
    ),
    "https://graphql.anilist.co": Source(
        "anilist", "graphql", "v2", "https://graphql.anilist.co", None, "2026-10-20"
    ),
}

INTROSPECTION_QUERY = get_introspection_query(descriptions=False)


def by_key(key: str) -> Source | None:
    return next((s for s in REGISTRY.values() if f"{s.vendor}/{s.api}/{s.version}" == key), None)


def sdl_from(document: Any) -> str:
    """The SDL text for a fetched schema: SDL as is, or an introspection result printed."""
    if isinstance(document, str):
        return document
    data = document.get("data", document) if isinstance(document, dict) else None
    if not isinstance(data, dict) or "__schema" not in data:
        raise ValueError("not an SDL document or an introspection result")
    return print_schema(build_client_schema(cast("IntrospectionQuery", data)))


def to_api_spec(sdl: str, source: Source) -> ApiSpec:
    # Builds it once here, so an SDL that does not build is refused before
    # anything uses it.
    _build(sdl)
    return ApiSpec(
        vendor=source.vendor,
        api=source.api,
        version=source.version,
        revision=_revision(sdl),
        source="graphql-sdl",
        source_url=source.sdl_url or source.endpoint,
        operations=[
            Operation(id="graphql", method="POST", url_template=source.endpoint, params={})
        ],
        graphql_schema=sdl,
    )


def _revision(sdl: str) -> str:
    """The schema's own content as a revision: two schemas of one length differ."""
    return "sdl:" + hashlib.sha256(sdl.encode("utf-8")).hexdigest()[:16]


_SCHEMAS: dict[str, GraphQLSchema] = {}


def _build(sdl: str) -> GraphQLSchema:
    """The schema, built without graphql-core's schema validation.

    That validation is for a schema's author, not its client: GitHub's
    public schema fails it (a deprecated field implementing a field its
    interface does not deprecate), and every GitHub check would raise. A
    request is still validated against the schema in full.
    """
    key = _revision(sdl)
    if key not in _SCHEMAS:
        _SCHEMAS[key] = build_schema(sdl, assume_valid=True)
    return _SCHEMAS[key]


def _schema(spec: ApiSpec) -> GraphQLSchema:
    assert spec.graphql_schema is not None
    return _build(spec.graphql_schema)


@dataclass
class Result:
    """What one GraphQL request is, and what is wrong with it."""

    #: ``query`` / ``mutation`` / ``subscription``; ``None`` when no
    #: operation could be chosen.
    operation: str | None
    #: ``(kind, message)`` per violation.
    problems: list[tuple[str, str]]
    #: What could not be checked, and why.
    unchecked: list[str]


def check(
    spec: ApiSpec,
    query: str,
    variables: dict[str, object] | None,
    variables_complete: bool,
    reads: list[tuple[tuple[str, ...], int]],
    operation_name: object = None,
) -> Result:
    """Everything wrong with one GraphQL request.

    ``operation_name`` is the request's ``operationName``: ``None`` when
    it sends none, a string, or :data:`~rote.vendor_specs.extract.UNKNOWN`
    when it is computed at run time.
    """
    try:
        document = parse(query)
    except GraphQLError as e:
        return Result(None, [("graphql_invalid", f"the query does not parse: {e.message}")], [])
    schema = _schema(spec)
    invalid = [("graphql_invalid", error.message) for error in validate(schema, document)]
    if invalid:
        return Result(_kind(document, operation_name), invalid, [])
    operation, why = _choose(document, operation_name)
    if operation is None:
        if why is None:  # a computed operationName: cannot tell which
            return Result(
                None,
                [],
                ["the operation: the document has several, and operationName is computed"],
            )
        return Result(None, [("graphql_invalid", why)], [])
    out: list[tuple[str, str]] = []
    unchecked: list[str] = []
    declared = {v.variable.name.value: v for v in operation.variable_definitions or ()}
    sent = variables or {}
    for name in sorted(set(sent) - set(declared)):
        out.append(
            (
                "graphql_unknown_variable",
                f"sends variable ${name}, which the query does not declare",
            )
        )
    for name, definition in sorted(declared.items()):
        required = definition.type.kind == "non_null_type" and definition.default_value is None
        if name not in sent:
            if required and variables_complete:
                out.append(("graphql_missing_variable", f"required variable ${name} is not sent"))
            continue
        problem = _variable_problem(schema, definition, sent[name])
        if problem:
            out.append(("graphql_variable_value", f"variable ${name}: {problem}"))
    selected = _selected(schema, document, operation)
    reported: list[tuple[str, ...]] = []
    for path, line in sorted(reads, key=lambda r: len(r[0])):
        if not path or path[0] != "data":
            continue
        fields = tuple(k for k in path[1:] if k != "[]")
        for i in range(1, len(fields) + 1):
            if fields[:i] in selected:
                continue
            # One mistake, once: `x.y` under an unselected `x` is the same one.
            if any(fields[: len(r)] == r for r in reported):
                break
            reported.append(fields[:i])
            above = selected.get(fields[: i - 1]) if i > 1 else None
            dotted = ".".join(("data", *fields))
            if above is not None and is_leaf_type(above):
                if isinstance(above, GraphQLScalarType) and above.name not in _BUILT_IN:
                    # A custom scalar (GitHub's URI is a string, AniList's
                    # Json is any JSON): what is inside is not described.
                    unchecked.append(
                        f"line {line} reads {dotted}: {'.'.join(fields[: i - 1])} is the "
                        f"scalar {above.name}, which the schema does not describe"
                    )
                else:
                    out.append(
                        (
                            "spec_unknown_response_field",
                            f"line {line} reads {dotted}, but {'.'.join(fields[: i - 1])} is "
                            f"a {above.name}, which has no fields",
                        )
                    )
            else:
                out.append(
                    (
                        "spec_unknown_response_field",
                        f"line {line} reads {dotted}, which the query does not select",
                    )
                )
            break
    return Result(operation.operation.value, out, unchecked)


_BUILT_IN = set(specified_scalar_types)


def _variable_problem(
    schema: GraphQLSchema, definition: VariableDefinitionNode, value: object
) -> str | None:
    """Why a literal variable value is not of the declared type, if it is not."""
    # A computed value is a whole UNKNOWN or Expr: the extractor reads a
    # dict or list only when every element of it is a literal.
    if value is UNKNOWN or isinstance(value, Expr):
        return None
    typ = type_from_ast(schema, definition.type)
    if typ is None or not is_input_type(typ):
        return None
    errors: list[str] = []
    validate_input_value(value, typ, lambda error, path: errors.append(error.message))
    return errors[0] if errors else None


def operation_type(query: str, operation_name: object = None) -> str | None:
    """``query`` / ``mutation`` / ``subscription`` of the operation the
    request runs; ``None`` when it cannot be told."""
    try:
        document = parse(query)
    except GraphQLError:
        return None
    return _kind(document, operation_name)


def _kind(document: DocumentNode, operation_name: object) -> str | None:
    operation, _ = _choose(document, operation_name)
    return operation.operation.value if operation is not None else None


def _choose(
    document: DocumentNode, operation_name: object
) -> tuple[OperationDefinitionNode | None, str | None]:
    """The operation a request runs, as GraphQL's GetOperation picks it.

    One operation and no name: that one. A name: the operation of that
    name. Several and no name: a request error. A name computed at run
    time: ``(None, None)``, since which one runs cannot be told here.
    """
    ops = [d for d in document.definitions if isinstance(d, OperationDefinitionNode)]
    if operation_name is None:
        if len(ops) == 1:
            return ops[0], None
        return None, (
            "the document has no operation"
            if not ops
            else f"the document has {len(ops)} operations and the request names none "
            f"(send operationName)"
        )
    if not isinstance(operation_name, str):
        return None, None
    named = [o for o in ops if o.name is not None and o.name.value == operation_name]
    if not named:
        return None, f"operationName {operation_name!r} names no operation in the document"
    return named[0], None


def _selected(
    schema: GraphQLSchema, document: DocumentNode, operation: OperationDefinitionNode
) -> dict[tuple[str, ...], GraphQLNamedType | None]:
    """Every response path (alias, else field name) the operation selects,
    with the type there, inline and named fragments followed.

    Only for a document that validated: every fragment it spreads exists,
    and none spreads itself.
    """
    fragments = {
        d.name.value: d for d in document.definitions if isinstance(d, FragmentDefinitionNode)
    }
    paths: dict[tuple[str, ...], GraphQLNamedType | None] = {}

    def walk(selections: Any, parent: GraphQLNamedType | None, prefix: tuple[str, ...]) -> None:
        for sel in selections or ():
            if isinstance(sel, FieldNode):
                here = (*prefix, (sel.alias or sel.name).value)
                named = _field_type(schema, parent, sel.name.value)
                if here not in paths or paths[here] is None:
                    paths[here] = named
                if sel.selection_set is not None:
                    walk(sel.selection_set.selections, named, here)
            elif isinstance(sel, InlineFragmentNode):
                condition = (
                    schema.get_type(sel.type_condition.name.value)
                    if sel.type_condition is not None
                    else parent
                )
                walk(sel.selection_set.selections, condition, prefix)
            elif isinstance(sel, FragmentSpreadNode):
                fragment = fragments[sel.name.value]
                walk(
                    fragment.selection_set.selections,
                    schema.get_type(fragment.type_condition.name.value),
                    prefix,
                )

    root = {
        "query": schema.query_type,
        "mutation": schema.mutation_type,
        "subscription": schema.subscription_type,
    }[operation.operation.value]
    walk(operation.selection_set.selections, root, ())
    return paths


def _field_type(
    schema: GraphQLSchema, parent: GraphQLNamedType | None, name: str
) -> GraphQLNamedType | None:
    if name == "__typename":
        return schema.get_type("String")
    if not isinstance(parent, GraphQLObjectType | GraphQLInterfaceType):
        return None
    field = parent.fields.get(name)
    return get_named_type(field.type) if field is not None else None


__all__ = [
    "INTROSPECTION_QUERY",
    "REGISTRY",
    "Source",
    "by_key",
    "Result",
    "check",
    "operation_type",
    "sdl_from",
    "to_api_spec",
]
