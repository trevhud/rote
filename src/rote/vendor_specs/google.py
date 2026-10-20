"""Google APIs, from Google's Discovery service.

Every Google API publishes a Discovery document: each method's HTTP verb,
path, parameters (with location, type, required, repeated, enum), request
and response schemas, and OAuth scopes. The directory lists them all, and
neither needs a credential:

* directory: ``https://discovery.googleapis.com/discovery/v1/apis``
  (``items[]`` with ``name``, ``version``, ``discoveryRestUrl``)
* a document, e.g. Gmail v1: ``https://gmail.googleapis.com/$discovery/rest?version=v1``,
  Drive v3: ``https://www.googleapis.com/discovery/v1/apis/drive/v3/rest``

https://developers.google.com/discovery/v1/reference/apis/list
https://developers.google.com/discovery/v1/reference/apis
"""

from __future__ import annotations

import re
from typing import Any

from rote.vendor_specs.model import ApiSpec, Operation, ParamSpec, ParamType, host_of

DIRECTORY_URL = "https://discovery.googleapis.com/discovery/v1/apis"

_TYPES: dict[str, ParamType] = {
    "string": "string",
    "integer": "integer",
    "number": "number",
    "boolean": "boolean",
    "object": "object",
    "array": "array",
    "any": "any",
}

_VERSION_SEGMENT = re.compile(r"^v\d+(?:[a-z]+\d*)?$")


def api_for_url(url: str) -> tuple[str, str] | None:
    """``(api, version)`` a Google API URL names, or ``None`` for a non-Google URL.

    ``gmail.googleapis.com/gmail/v1/...`` names the API by host;
    ``www.googleapis.com/drive/v3/...`` (and ``/upload/drive/v3/...``) by its
    first path segment. The version is the first ``v<N>`` segment. The
    directory confirms the pair exists before anything is fetched.
    """
    host = host_of(url)
    if host is None or not host.endswith(".googleapis.com"):
        return None
    segments = [s for s in re.sub(r"^[a-z]+://[^/]+", "", url).split("?")[0].split("/") if s]
    sub = host.removesuffix(".googleapis.com")
    if sub in ("www", "content"):
        if segments and segments[0] == "upload":
            segments = segments[1:]
        if not segments:
            return None
        api, segments = segments[0], segments[1:]
    else:
        api = sub
    version = next((s for s in segments if _VERSION_SEGMENT.match(s)), None)
    return (api, version) if version else None


def discovery_url(directory: dict[str, Any], api: str, version: str) -> str | None:
    """The ``discoveryRestUrl`` the directory lists for ``api`` ``version``."""
    for item in directory.get("items") or []:
        if item.get("name") == api and item.get("version") == version:
            url = item.get("discoveryRestUrl")
            return str(url) if url else None
    return None


def from_discovery(doc: dict[str, Any], source_url: str) -> ApiSpec:
    """A Discovery document as an :class:`ApiSpec`."""
    base = f"{doc['rootUrl']}{doc.get('servicePath') or ''}"
    operations: list[Operation] = []
    _walk(doc.get("resources") or {}, doc.get("methods") or {}, base, operations)
    return ApiSpec(
        vendor="google",
        api=str(doc["name"]),
        version=str(doc["version"]),
        revision=str(doc.get("revision") or ""),
        source="google-discovery",
        source_url=source_url,
        global_params=_params(doc.get("parameters") or {}),
        operations=operations,
        schemas={name: _schema(s) for name, s in (doc.get("schemas") or {}).items()},
    )


def _walk(
    resources: dict[str, Any], methods: dict[str, Any], base: str, out: list[Operation]
) -> None:
    for method in methods.values():
        request = method.get("request")
        response = method.get("response")
        out.append(
            Operation(
                id=str(method["id"]),
                method=str(method["httpMethod"]).upper(),  # type: ignore[arg-type]
                url_template=base + str(method.get("flatPath") or method["path"]),
                params=_params(method.get("parameters") or {}),
                request_schema=_schema(request) if request else None,
                response_schema=_schema(response) if response else None,
                media_download=bool(method.get("supportsMediaDownload")) and response is None,
                supports_media=bool(method.get("supportsMediaDownload")),
                scopes=tuple(method.get("scopes") or ()),
            )
        )
    for resource in resources.values():
        _walk(resource.get("resources") or {}, resource.get("methods") or {}, base, out)


def _params(params: dict[str, Any]) -> dict[str, ParamSpec]:
    out: dict[str, ParamSpec] = {}
    for name, p in params.items():
        location = p.get("location", "query")
        if location not in ("path", "query"):
            continue
        out[name] = ParamSpec(
            name=name,
            location=location,
            type=_TYPES.get(str(p.get("type", "string")), "any"),
            required=bool(p.get("required")),
            repeated=bool(p.get("repeated")),
            enum=tuple(p["enum"]) if p.get("enum") else None,
            format=p.get("format"),
        )
    return out


def _schema(node: Any) -> Any:
    """A Discovery schema fragment as JSON Schema (``$ref`` into ``#/$defs``)."""
    if isinstance(node, list):
        return [_schema(n) for n in node]
    if not isinstance(node, dict):
        return node
    out: dict[str, Any] = {}
    for key, value in node.items():
        if key == "$ref":
            out["$ref"] = f"#/$defs/{value}"
        elif key == "id" or key == "type" and value == "any":
            continue
        elif key in ("properties", "patternProperties"):
            out[key] = {name: _schema(v) for name, v in value.items()}
        else:
            out[key] = _schema(value)
    return out


__all__ = ["DIRECTORY_URL", "api_for_url", "discovery_url", "from_discovery"]
