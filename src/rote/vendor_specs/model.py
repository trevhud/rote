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
    #: Absolute URL with ``{name}`` path parameters; ``{+name}`` matches the
    #: rest of the path (Discovery's reserved expansion).
    url_template: str
    params: dict[str, ParamSpec] = Field(default_factory=dict)
    request_schema: dict[str, Any] | None = None
    response_schema: dict[str, Any] | None = None
    #: The response is a file's bytes, not JSON (``alt=media`` / export).
    media_download: bool = False
    scopes: tuple[str, ...] = ()

    def segments(self) -> list[str]:
        return _path_segments(self.url_template)


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
    #: Parameters every operation accepts (Discovery's top-level
    #: ``parameters``: ``fields``, ``alt``, ``key``, ...).
    global_params: dict[str, ParamSpec] = Field(default_factory=dict)
    operations: list[Operation] = Field(default_factory=list)
    schemas: dict[str, dict[str, Any]] = Field(default_factory=dict)

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


def _path_segments(url: str) -> list[str]:
    """The path of ``url`` as segments, scheme, host, query and empty parts dropped."""
    path = _SCHEME_HOST.sub("", url).split("?", 1)[0]
    return [s for s in path.split("/") if s]


def host_of(url: str) -> str | None:
    match = _SCHEME_HOST.match(url)
    return match.group(0).split("://", 1)[1].lower() if match else None


__all__ = [
    "FORMAT_VERSION",
    "_path_segments",
    "ApiSpec",
    "Operation",
    "ParamSpec",
    "ParamType",
    "host_of",
]
