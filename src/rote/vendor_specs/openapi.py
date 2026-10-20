"""Vendor APIs that publish an OpenAPI 3 description.

Two parts:

* :data:`REGISTRY`: vendor hosts mapped to the OpenAPI document the vendor
  itself publishes. Curated on purpose: a community directory
  (APIs.guru) lags vendors by years for some APIs (its Slack entry comes
  from an archived 2021 spec), and a spec the vendor no longer maintains
  checks code against the wrong API. Each entry was fetched and parsed
  when it was added; ``checked`` says when.
* :func:`from_openapi`: an OpenAPI 3.0 / 3.1 document as an
  :class:`~rote.vendor_specs.model.ApiSpec` (operations from ``paths``,
  path and query parameters with type / enum / required / repeated, the
  JSON request body and the 2xx JSON response, ``components.schemas`` as
  ``$defs``).

https://spec.openapis.org/oas/v3.1.0
"""

from __future__ import annotations

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


@dataclass(frozen=True)
class Source:
    """One vendor-published OpenAPI document."""

    vendor: str
    api: str
    version: str
    url: str
    #: When the entry was last fetched and parsed by hand.
    checked: str


#: Host -> the vendor's own OpenAPI document. Hosts are matched exactly.
REGISTRY: dict[str, Source] = {
    "clinicaltrials.gov": Source(
        "clinicaltrials", "studies", "v2", "https://clinicaltrials.gov/api/oas/v2", "2026-10-20"
    ),
    "api.stripe.com": Source(
        "stripe",
        "api",
        "v1",
        "https://raw.githubusercontent.com/stripe/openapi/master/openapi/spec3.json",
        "2026-10-20",
    ),
    "api.github.com": Source(
        "github",
        "rest",
        "v3",
        "https://raw.githubusercontent.com/github/rest-api-description/main/descriptions/"
        "api.github.com/api.github.com.json",
        "2026-10-20",
    ),
}


def by_key(key: str) -> Source | None:
    """The registry entry for a spec key (``<vendor>/<api>/<version>``)."""
    return next((s for s in REGISTRY.values() if f"{s.vendor}/{s.api}/{s.version}" == key), None)


class OpenApiError(ValueError):
    """A document rote cannot read as OpenAPI 3."""


def from_openapi(doc: Any, source: Source) -> ApiSpec:
    if not isinstance(doc, dict) or not str(doc.get("openapi", "")).startswith("3."):
        version = doc.get("swagger") if isinstance(doc, dict) else None
        raise OpenApiError(
            f"{source.url} is not OpenAPI 3"
            + (f" (Swagger {version} is not supported)" if version else "")
        )
    servers = doc.get("servers") or [{"url": "/"}]
    base = _server_url(servers[0], source.url)
    components = doc.get("components") or {}
    shared = components.get("parameters") or {}
    operations: list[Operation] = []
    for path, item in (doc.get("paths") or {}).items():
        if not isinstance(item, dict):
            continue
        item = _deref(item, doc)
        path_params = item.get("parameters") or []
        for method in _METHODS:
            op = item.get(method)
            if not isinstance(op, dict):
                continue
            params: dict[str, ParamSpec] = {}
            for raw in [*path_params, *(op.get("parameters") or [])]:
                p = _param(_deref(raw, doc, shared))
                if p is not None:
                    params[p.name] = p
            operations.append(
                Operation(
                    id=str(op.get("operationId") or f"{method.upper()} {path}"),
                    method=method.upper(),  # type: ignore[arg-type]
                    url_template=base.rstrip("/") + "/" + path.lstrip("/"),
                    params=params,
                    request_schema=_json_schema(_deref(op.get("requestBody") or {}, doc)),
                    response_schema=_response(op.get("responses") or {}, doc),
                )
            )
    return ApiSpec(
        vendor=source.vendor,
        api=source.api,
        version=source.version,
        revision=str((doc.get("info") or {}).get("version") or ""),
        source="openapi",
        source_url=source.url,
        operations=operations,
        schemas={name: _refs(s) for name, s in (components.get("schemas") or {}).items()},
    )


def _server_url(server: dict[str, Any], doc_url: str) -> str:
    url = str(server.get("url") or "/")
    for name, var in (server.get("variables") or {}).items():
        url = url.replace("{" + name + "}", str(var.get("default", "")))
    return urljoin(doc_url, url)


def _deref(node: Any, doc: dict[str, Any], shared: dict[str, Any] | None = None) -> Any:
    """``node`` with a local ``$ref`` (``#/components/...``) followed, once per hop."""
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


def _param(p: Any) -> ParamSpec | None:
    if not isinstance(p, dict) or p.get("in") not in ("path", "query") or not p.get("name"):
        return None
    schema = p.get("schema") or {}
    kind = schema.get("type")
    if isinstance(kind, list):  # 3.1: ["string", "null"]
        kind = next((k for k in kind if k != "null"), "string")
    repeated = False
    enum = schema.get("enum")
    if kind == "array":
        # form + explode (the default) repeats the name per value; anything
        # else (explode: false, pipeDelimited...) is one delimited string.
        exploded = p.get("explode", p.get("style", "form") == "form")
        style = p.get("style", "form")
        repeated = bool(exploded) and style == "form"
        items = schema.get("items") or {}
        kind = items.get("type", "string") if repeated else "string"
        enum = items.get("enum") if repeated else None
    return ParamSpec(
        name=str(p["name"]),
        location=p["in"],
        type=_TYPES.get(str(kind), "any"),
        required=bool(p.get("required")) or p["in"] == "path",
        repeated=repeated,
        enum=tuple(str(e) for e in enum) if enum else None,
        format=schema.get("format"),
    )


def _json_schema(body: Any) -> dict[str, Any] | None:
    content = body.get("content") if isinstance(body, dict) else None
    if not isinstance(content, dict):
        return None
    for media, value in content.items():
        if "json" in media and isinstance(value, dict) and "schema" in value:
            return _refs(value["schema"])  # type: ignore[no-any-return]
    return None


def _response(responses: dict[str, Any], doc: dict[str, Any]) -> dict[str, Any] | None:
    for code in ("200", "201", "202", "2XX", "2xx", "default"):
        if code in responses:
            return _json_schema(_deref(responses[code], doc))
    return None


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
