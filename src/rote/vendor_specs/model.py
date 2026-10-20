"""One vendor API version, normalized from whatever machine-readable spec the vendor publishes.

Every source format (Google Discovery today; OpenAPI and GraphQL SDL to
follow) converts into this model, and everything downstream (the request
check, the HTTP fixtures, the live shape diff) reads only this model. It is
also the unit a shared store keys and holds: one ``ApiSpec`` per
``(vendor, api, version)``, serialized as JSON with ``format_version``, the
vendor's own ``revision``, and the ``source_url`` it came from.

Schemas are JSON Schema (2020-12 subset), with references as
``{"$ref": "#/$defs/<Name>"}`` into :attr:`ApiSpec.schemas`.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

FORMAT_VERSION = 1

ParamType = Literal["string", "integer", "number", "boolean", "object", "array", "any"]


class ParamSpec(BaseModel):
    """One request parameter: in the path or the query string."""

    model_config = ConfigDict(frozen=True)

    name: str
    location: Literal["path", "query"]
    type: ParamType = "string"
    required: bool = False
    #: A query parameter that may appear more than once (``labelIds=a&labelIds=b``).
    repeated: bool = False
    enum: tuple[str, ...] | None = None
    format: str | None = None


class Operation(BaseModel):
    """One endpoint: method + URL template, its parameters, and its body shapes."""

    model_config = ConfigDict(frozen=True)

    #: The vendor's own id where it has one (``gmail.users.threads.list``).
    id: str
    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"]
    #: Absolute URL with ``{name}`` path parameters (one segment each) and
    #: ``{+name}`` ones (Discovery's reserved expansion: one or more
    #: segments, slashes included). The names are the declared parameters'.
    url_template: str
    #: Discovery's ``flatPath`` form of the URL, absolute, when it differs:
    #: literal segments where ``url_template`` has ``{+name}``. It tells apart
    #: operations whose ``url_template`` is the same (People's people.get and
    #: contactGroups.get are both ``v1/{+resourceName}``).
    flat_url_template: str | None = None
    #: Where the same operation takes an upload (Discovery ``mediaUpload``:
    #: ``/upload/drive/v3/files``, ``/resumable/upload/...``), absolute.
    upload_url_templates: tuple[str, ...] = ()
    params: dict[str, ParamSpec] = Field(default_factory=dict)
    request_schema: dict[str, Any] | None = None
    response_schema: dict[str, Any] | None = None
    #: The response is always a file's bytes, not JSON (Drive's export).
    media_download: bool = False
    #: The operation can return the file's bytes instead of its JSON
    #: (Discovery ``supportsMediaDownload``): with ``alt=media`` over HTTP,
    #: and as ``<method>_media`` in Google's client library.
    supports_media: bool = False
    #: The operation accepts an uploaded file (Discovery
    #: ``supportsMediaUpload``): at an upload URL over HTTP, and as
    #: ``media_body`` / ``media_mime_type`` in Google's client library.
    supports_media_upload: bool = False
    scopes: tuple[str, ...] = ()

    def segments(self) -> list[str]:
        return path_segments(self.url_template)

    def templates(self) -> tuple[str, ...]:
        """Every URL the operation is called at: its own, then its upload URLs."""
        return (self.url_template, *self.upload_url_templates)


class AuthSpec(BaseModel):
    """The one credential a vendor's requests carry: where it goes and where it comes from.

    The live check sends this credential and no other: a ``{$env:VAR}`` the
    code reads anywhere else is refused, so code a hostile skill steered
    cannot send another secret from the user's shell.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The environment variable holding it.
    env: str
    location: Literal["header", "query"]
    #: The header (``Authorization``) or query parameter (``key``) it goes in.
    name: str


class ApiSpec(BaseModel):
    """One vendor API version, in the shareable form."""

    format_version: int = FORMAT_VERSION
    vendor: str
    api: str
    version: str
    #: The vendor's own revision of the spec (Discovery's ``revision``).
    revision: str
    source: Literal["google-discovery", "openapi", "graphql-sdl", "docs"]
    source_url: str
    #: Every page a ``docs`` spec was written from (``source="docs"``).
    sources: list[str] = Field(default_factory=list)
    #: Parameters every operation accepts (Discovery's top-level
    #: ``parameters``: ``fields``, ``alt``, ``key``, ...).
    global_params: dict[str, ParamSpec] = Field(default_factory=dict)
    operations: list[Operation] = Field(default_factory=list)
    schemas: dict[str, dict[str, Any]] = Field(default_factory=dict)
    #: The credential its requests carry, when the spec declares one (a docs
    #: spec's ``auth``). Google's is the OAuth token, handled by the live check.
    auth: AuthSpec | None = None

    @property
    def key(self) -> str:
        """The store key: ``<vendor>/<api>/<version>``."""
        return f"{self.vendor}/{self.api}/{self.version}"

    def operation(self, op_id: str) -> Operation | None:
        return next((o for o in self.operations if o.id == op_id), None)

    def resolve(self, schema: dict[str, Any] | None) -> dict[str, Any] | None:
        """``schema`` with a top-level ``$ref`` followed (once per hop, cycle-safe)."""
        seen: set[str] = set()
        while schema is not None and "$ref" in schema:
            ref = str(schema["$ref"])
            if ref in seen:
                return None
            seen.add(ref)
            schema = self.schemas.get(ref.rsplit("/", 1)[-1])
        return schema


_SCHEME_HOST = re.compile(r"^[a-z]+://[^/]+")


def path_segments(url: str) -> list[str]:
    """The path of ``url`` as segments, scheme, host, query and empty parts dropped."""
    path = _SCHEME_HOST.sub("", url).split("?", 1)[0]
    return [s for s in path.split("/") if s]


def host_of(url: str) -> str | None:
    match = _SCHEME_HOST.match(url)
    return match.group(0).split("://", 1)[1].lower() if match else None


__all__ = [
    "FORMAT_VERSION",
    "AuthSpec",
    "path_segments",
    "ApiSpec",
    "Operation",
    "ParamSpec",
    "ParamType",
    "host_of",
]
