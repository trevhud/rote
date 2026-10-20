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

import keyword
import re
from dataclasses import dataclass
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


@dataclass(frozen=True)
class UrlApi:
    """The Google API a URL names, with the version strings it could mean."""

    api: str
    #: Candidates in preference order: ``admin/directory/v1`` could be
    #: ``directory_v1`` (what the directory lists) or ``v1``. Only one the
    #: directory lists is used; the URL alone never decides.
    versions: tuple[str, ...]
    #: ``<api>.googleapis.com``, where every path is that API's. On a shared
    #: host (``www.googleapis.com``) an unlisted version can be an endpoint
    #: Discovery does not describe, such as OAuth2 v3's userinfo.
    dedicated_host: bool


def api_for_url(url: str) -> UrlApi | None:
    """The API and candidate versions a Google API URL names; ``None`` off Google.

    ``gmail.googleapis.com/gmail/v1/...`` names the API by host;
    ``www.googleapis.com/drive/v3/...`` (and its ``/upload/`` and
    ``/resumable/upload/`` forms) by its first path segment. Each ``v<N>``
    segment is a candidate version, preceded by the version its previous
    segment joins into (``directory`` + ``v1`` is ``directory_v1``).
    """
    host = host_of(url)
    if host is None or not host.endswith(".googleapis.com"):
        return None
    segments = [s for s in re.sub(r"^[a-z]+://[^/]+", "", url).split("?")[0].split("/") if s]
    sub = host.removesuffix(".googleapis.com")
    shared = sub in ("www", "content")
    if shared:
        while segments and segments[0] in ("upload", "resumable"):
            segments = segments[1:]
        if not segments:
            return None
        api, segments = segments[0], segments[1:]
    else:
        api = sub
    versions: list[str] = []
    for i, seg in enumerate(segments):
        if not _VERSION_SEGMENT.match(seg):
            continue
        if i > 0 and not _VERSION_SEGMENT.match(segments[i - 1]):
            versions.append(f"{segments[i - 1]}_{seg}")
        versions.append(seg)
    return UrlApi(api, tuple(versions), dedicated_host=not shared)


# ── Google's client library (google-api-python-client) names things its own way ──
# The rules below are its own (googleapiclient.discovery.key2param,
# fix_method_name, STACK_QUERY_PARAMETERS); a check that used the Discovery
# names refused correct client code.

#: Query parameters the client library adds to every method.
CLIENT_STACK_PARAMS = frozenset({"trace", "pp", "userip", "strict"})


def client_param(name: str) -> str:
    """The keyword argument the client library takes for Discovery parameter ``name``.

    ``requestMask.includeField`` -> ``requestMask_includeField``,
    ``$.xgafv`` -> ``x__xgafv`` (googleapiclient's ``key2param``).
    """
    prefix = "" if name[:1].isalpha() else "x"
    return prefix + "".join(c if c.isalnum() else "_" for c in name)


def client_method(name: str) -> str:
    """The client library's name for a Discovery resource or method (``import`` -> ``import_``)."""
    name = name.replace("$", "_").replace("-", "_")
    return f"{name}_" if keyword.iskeyword(name) or name == "body" else name


def listed_versions(directory: dict[str, Any], api: str) -> tuple[str, ...]:
    """Every version the directory lists for ``api``."""
    return tuple(
        str(item["version"])
        for item in directory.get("items") or []
        if item.get("name") == api and item.get("version")
    )


def discovery_url(directory: dict[str, Any], api: str, version: str) -> str | None:
    """The ``discoveryRestUrl`` the directory lists for ``api`` ``version``."""
    for item in directory.get("items") or []:
        if item.get("name") == api and item.get("version") == version:
            url = item.get("discoveryRestUrl")
            return str(url) if url else None
    return None


def from_discovery(doc: dict[str, Any], source_url: str) -> ApiSpec:
    """A Discovery document as an :class:`ApiSpec`."""
    root = str(doc["rootUrl"])
    base = f"{root}{doc.get('servicePath') or ''}"
    operations: list[Operation] = []
    _walk(doc.get("resources") or {}, doc.get("methods") or {}, root, base, operations)
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
    resources: dict[str, Any],
    methods: dict[str, Any],
    root: str,
    base: str,
    out: list[Operation],
) -> None:
    for method in methods.values():
        request = method.get("request")
        response = method.get("response")
        protocols = (method.get("mediaUpload") or {}).get("protocols") or {}
        uploads = tuple(
            root.rstrip("/") + str(p["path"])
            for p in protocols.values()
            if isinstance(p, dict) and p.get("path")
        )
        out.append(
            Operation(
                id=str(method["id"]),
                method=str(method["httpMethod"]).upper(),  # type: ignore[arg-type]
                # ``path``, not ``flatPath``: its parameter names are the
                # declared ones. People's flatPath ``v1/people/{peopleId}``
                # names a parameter the method does not declare; its path
                # ``v1/{+resourceName}`` binds the one it requires.
                url_template=base + str(method["path"]),
                upload_url_templates=uploads,
                params=_params(method.get("parameters") or {}),
                request_schema=_schema(request) if request else None,
                response_schema=_schema(response) if response else None,
                media_download=bool(method.get("supportsMediaDownload")) and response is None,
                supports_media=bool(method.get("supportsMediaDownload")),
                supports_media_upload=bool(method.get("supportsMediaUpload")),
                scopes=tuple(method.get("scopes") or ()),
            )
        )
    for resource in resources.values():
        _walk(resource.get("resources") or {}, resource.get("methods") or {}, root, base, out)


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


__all__ = [
    "CLIENT_STACK_PARAMS",
    "DIRECTORY_URL",
    "client_method",
    "client_param",
    "UrlApi",
    "api_for_url",
    "discovery_url",
    "from_discovery",
    "listed_versions",
]
