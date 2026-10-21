"""Vendor APIs that publish an OpenAPI 3 description.

Two parts:

* :data:`REGISTRY`: vendor hosts mapped to the OpenAPI document the vendor
  itself publishes. Curated on purpose: a community directory
  (APIs.guru) lags vendors by years for some APIs (its Slack entry comes
  from an archived 2021 spec), and a spec the vendor no longer maintains
  checks code against the wrong API. Each entry was fetched and parsed
  when it was added; ``checked`` says when. Every host the document's
  ``servers`` name (document, path or operation level) is registered.
* :func:`from_openapi`: an OpenAPI 3.0 / 3.1 document as an
  :class:`~rote.vendor_specs.model.ApiSpec` (operations from ``paths`` at
  their own ``servers``, path and query parameters with type / enum /
  required / repeated / deepObject members, the request body (JSON, form
  or multipart) with the media types it accepts, every 2xx JSON response,
  ``components.schemas`` as ``$defs``).

https://spec.openapis.org/oas/v3.1.0
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin

from rote.vendor_specs.model import ApiSpec, Operation, ParamSpec, ParamType

_METHODS = ("get", "post", "put", "patch", "delete", "head")
_TYPES: dict[str, ParamType] = {
    "string": "string",
    "integer": "integer",
    "number": "number",
    "boolean": "boolean",
    "object": "object",
    "array": "array",
}
#: Request body media types in the order the body schema is taken from.
_BODY_MEDIA = ("json", "application/x-www-form-urlencoded", "multipart/form-data")
_SUCCESS = re.compile(r"2(\d\d|XX)", re.IGNORECASE)


@dataclass(frozen=True)
class Source:
    """One vendor-published OpenAPI document."""

    vendor: str
    api: str
    version: str
    url: str
    #: When the entry was last fetched and parsed by hand.
    checked: str
    #: Every host the document's ``servers`` name.
    hosts: tuple[str, ...]


_SOURCES = (
    Source(
        "clinicaltrials",
        "studies",
        "v2",
        "https://clinicaltrials.gov/api/oas/v2",
        "2026-10-20",
        ("clinicaltrials.gov",),
    ),
    Source(
        "stripe",
        "api",
        "v1",
        "https://raw.githubusercontent.com/stripe/openapi/master/openapi/spec3.json",
        "2026-10-20",
        # files.stripe.com: the file upload's own server (POST /v1/files).
        ("api.stripe.com", "files.stripe.com"),
    ),
    Source(
        "github",
        "rest",
        "v3",
        "https://raw.githubusercontent.com/github/rest-api-description/main/descriptions/"
        "api.github.com/api.github.com.json",
        "2026-10-20",
        # uploads.github.com: release asset uploads.
        ("api.github.com", "uploads.github.com"),
    ),
)

#: Host -> the vendor's own document. Hosts are matched exactly.
REGISTRY: dict[str, Source] = {host: s for s in _SOURCES for host in s.hosts}


def by_key(key: str) -> Source | None:
    """The registry entry for a spec key (``<vendor>/<api>/<version>``)."""
    return next((s for s in _SOURCES if f"{s.vendor}/{s.api}/{s.version}" == key), None)


class OpenApiError(ValueError):
    """A document rote cannot read as OpenAPI 3."""


def from_openapi(doc: Any, source: Source) -> ApiSpec:
    if not isinstance(doc, dict) or not str(doc.get("openapi", "")).startswith("3."):
        version = doc.get("swagger") if isinstance(doc, dict) else None
        raise OpenApiError(
            f"{source.url} is not OpenAPI 3"
            + (f" (Swagger {version} is not supported)" if version else "")
        )
    doc_servers = _list(doc.get("servers"), "servers", source) or [{"url": "/"}]
    components = _dict(doc.get("components"), "components", source)
    operations: list[Operation] = []
    for path, raw_item in _dict(doc.get("paths"), "paths", source).items():
        item = _dict(_deref(raw_item, doc), f"paths[{path!r}]", source)
        path_servers = _list(item.get("servers"), f"paths[{path!r}].servers", source)
        path_params = _list(item.get("parameters"), f"paths[{path!r}].parameters", source)
        for method in _METHODS:
            where = f"paths[{path!r}].{method}"
            op = item.get(method)
            if op is None:
                continue
            op = _dict(op, where, source)
            servers = (
                _list(op.get("servers"), f"{where}.servers", source) or path_servers or doc_servers
            )
            base = _server_url(_dict(servers[0], f"{where}.servers[0]", source), source.url)
            params: dict[str, ParamSpec] = {}
            template = str(path)
            for raw in [*path_params, *_list(op.get("parameters"), f"{where}.parameters", source)]:
                declared = _deref(raw, doc)
                p = _param(declared, doc)
                if p is None:
                    continue
                params[p.name] = p
                if p.location == "path" and declared.get("x-multi-segment"):
                    # GitHub's own extension: the value spans segments
                    # (contents/docs/README.md, git/ref/heads/main).
                    template = template.replace("{" + p.name + "}", "{+" + p.name + "}")
            media, request = _request_body(_deref(op.get("requestBody"), doc), doc)
            operations.append(
                Operation(
                    id=str(op.get("operationId") or f"{method.upper()} {path}"),
                    method=method.upper(),  # type: ignore[arg-type]
                    url_template=base.rstrip("/") + "/" + template.lstrip("/"),
                    params=params,
                    request_schema=request,
                    request_media_types=media,
                    response_schema=_response(
                        _dict(op.get("responses"), f"{where}.responses", source), doc
                    ),
                )
            )
    return ApiSpec(
        vendor=source.vendor,
        api=source.api,
        version=source.version,
        revision=str(_dict(doc.get("info"), "info", source).get("version") or ""),
        source="openapi",
        source_url=source.url,
        operations=operations,
        schemas={
            str(name): _refs(_dict(s, f"components.schemas[{name!r}]", source))
            for name, s in _dict(components.get("schemas"), "components.schemas", source).items()
        },
    )


def _dict(value: Any, where: str, source: Source) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise OpenApiError(f"{source.url}: {where} is not an object")
    return value


def _list(value: Any, where: str, source: Source) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise OpenApiError(f"{source.url}: {where} is not a list")
    return value


def _server_url(server: dict[str, Any], doc_url: str) -> str:
    url = str(server.get("url") or "/")
    variables = server.get("variables")
    for name, var in (variables if isinstance(variables, dict) else {}).items():
        default = var.get("default", "") if isinstance(var, dict) else ""
        url = url.replace("{" + str(name) + "}", str(default))
    return urljoin(doc_url, url)


def _deref(node: Any, doc: dict[str, Any]) -> Any:
    """``node`` with a local ``$ref`` (``#/components/...``) followed, once per hop.

    A reference rote cannot follow (another file, a cycle) is ``{}``: a
    schema that says nothing, which the check reads as unconstrained.
    """
    seen: set[str] = set()
    while isinstance(node, dict) and isinstance(node.get("$ref"), str):
        ref = node["$ref"]
        if ref in seen or not ref.startswith("#/"):
            return {}
        seen.add(ref)
        target: Any = doc
        for part in ref[2:].split("/"):
            target = (
                target.get(part.replace("~1", "/").replace("~0", "~"), {})
                if isinstance(target, dict)
                else {}
            )
        node = target
    return node


def _schema(node: Any, doc: dict[str, Any]) -> dict[str, Any]:
    """``node`` as a schema object, references followed; ``{}`` when it is not one."""
    found = _deref(node, doc)
    return found if isinstance(found, dict) else {}


def _param(p: Any, doc: dict[str, Any]) -> ParamSpec | None:
    if not isinstance(p, dict) or p.get("in") not in ("path", "query") or not p.get("name"):
        return None
    location = p["in"]
    schema = _schema(p.get("schema"), doc)
    style = p.get("style", "form" if location == "query" else "simple")
    required = bool(p.get("required")) or location == "path"
    if style == "deepObject":
        return _deep_object(str(p["name"]), schema, required, doc)
    kind = _type(schema)
    repeated = False
    enum = _enum(schema)
    if kind == "array":
        # form + explode (the default) repeats the name per value; anything
        # else (explode: false, pipeDelimited...) is one delimited string.
        exploded = p.get("explode", style == "form")
        repeated = bool(exploded) and style == "form"
        items = _schema(schema.get("items"), doc)
        kind = _type(items) if repeated else "string"
        enum = _enum(items) if repeated else None
    return ParamSpec(
        name=str(p["name"]),
        location=location,
        type=_TYPES.get(str(kind), "any"),
        required=required,
        repeated=repeated,
        enum=enum,
        format=schema["format"] if isinstance(schema.get("format"), str) else None,
    )


def _deep_object(
    name: str, schema: dict[str, Any], required: bool, doc: dict[str, Any]
) -> ParamSpec:
    """A ``style: deepObject`` parameter: ``name[key]=...`` / ``name[]=...``.

    Each ``anyOf`` / ``oneOf`` branch adds what it allows: an object branch
    its members (``None``: any key), an array branch its item, a plain
    branch the bare ``name=...``.
    """
    alternatives = schema.get("anyOf") or schema.get("oneOf")
    branches = [
        _schema(b, doc) for b in (alternatives if isinstance(alternatives, list) else [schema])
    ]
    members: dict[str, ParamSpec] | None = {}
    items: ParamSpec | None = None
    bare: ParamSpec | None = None
    for branch in branches:
        kind = _type(branch)
        if kind == "array":
            items = _value(name, _schema(branch.get("items"), doc))
        elif kind == "object" or "properties" in branch:
            props = branch.get("properties")
            if members is not None:
                if not isinstance(props, dict) or branch.get("additionalProperties") not in (
                    None,
                    False,
                ):
                    members = None
                else:
                    members |= {str(k): _value(str(k), _schema(v, doc)) for k, v in props.items()}
        else:
            bare = _value(name, branch)
    return ParamSpec(
        name=name,
        location="query",
        type=bare.type if bare is not None else "any",
        required=required,
        enum=bare.enum if bare is not None else None,
        deep_object=True,
        bare=bare is not None,
        members=members,
        items=items,
    )


def _value(name: str, schema: dict[str, Any]) -> ParamSpec:
    """The value one deepObject key takes: its type and enum."""
    return ParamSpec(
        name=name,
        location="query",
        type=_TYPES.get(str(_type(schema)), "any"),
        enum=_enum(schema),
    )


def _enum(schema: dict[str, Any]) -> tuple[str, ...] | None:
    enum = schema.get("enum")
    return tuple(str(e) for e in enum) if isinstance(enum, list) and enum else None


def _type(schema: dict[str, Any]) -> str | None:
    kind = schema.get("type")
    if isinstance(kind, list):  # 3.1: ["string", "null"]
        kind = next((k for k in kind if k != "null"), "string")
    return kind if isinstance(kind, str) else None


def _request_body(body: Any, doc: dict[str, Any]) -> tuple[tuple[str, ...], dict[str, Any] | None]:
    """The media types the body may be sent as, and the schema of the first one rote reads."""
    content = body.get("content") if isinstance(body, dict) else None
    if not isinstance(content, dict):
        return (), None
    media = tuple(str(m) for m in content)
    for wanted in _BODY_MEDIA:
        for name, value in content.items():
            if wanted in str(name) and isinstance(value, dict):
                schema = value.get("schema")
                return media, _refs(schema) if isinstance(schema, dict) else {}
    return media, None


def _response(responses: dict[str, Any], doc: dict[str, Any]) -> dict[str, Any] | None:
    """Every 2xx JSON response's schema, as one (an ``anyOf`` when they differ).

    ``default`` is read only when the document lists no other response:
    elsewhere it is the error shape (Stripe's ``error``). A 2xx JSON
    response without a schema is ``{}``: anything.
    """
    codes = [c for c in responses if _SUCCESS.fullmatch(str(c))]
    if not codes and [str(c) for c in responses] == ["default"]:
        codes = list(responses)
    schemas: list[dict[str, Any]] = []
    for code in codes:
        content = _deref(responses[code], doc)
        content = content.get("content") if isinstance(content, dict) else None
        if not isinstance(content, dict):
            continue  # 204: no body
        # application/json itself over a JSON variant beside it
        # (ClinicalTrials.gov's application/fhir+json, asked for by format=).
        media = sorted(
            (str(m) for m, v in content.items() if "json" in str(m) and isinstance(v, dict)),
            key=lambda m: m.split(";", 1)[0].strip() != "application/json",
        )
        if media:
            schema = content[media[0]].get("schema")
            found = _refs(schema) if isinstance(schema, dict) else {}
            if found not in schemas:
                schemas.append(found)
    if not schemas:
        return None
    return schemas[0] if len(schemas) == 1 else {"anyOf": schemas}


def _refs(node: Any) -> Any:
    """``#/components/schemas/X`` -> ``#/$defs/X``, everywhere in ``node``."""
    if isinstance(node, list):
        return [_refs(n) for n in node]
    if not isinstance(node, dict):
        return node
    out: dict[str, Any] = {}
    for key, value in node.items():
        if key == "$ref" and isinstance(value, str) and value.startswith("#/components/schemas/"):
            out[key] = "#/$defs/" + value.rsplit("/", 1)[-1]
        else:
            out[key] = _refs(value)
    return out


__all__ = ["REGISTRY", "OpenApiError", "Source", "by_key", "from_openapi"]
