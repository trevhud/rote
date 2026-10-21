"""Specs written from documentation, where a vendor publishes none (lower trust).

Most vendors publish a machine-readable spec (:mod:`rote.vendor_specs.google`;
OpenAPI and GraphQL SDL to follow). Some do not: Slack's Web API is
documented as pages. For those, and only for those, the compile agent
writes the spec itself from the pages it fetched, as one file per API
version in the deliverable::

    vendor_specs/<vendor>.<api>.<version>.json
    {"vendor": "slack", "api": "web", "version": "v1",
     "sources": ["https://docs.slack.dev/reference/methods/conversations.history"],
     "operations": [{
         "id": "conversations.history", "method": "GET",
         "url_template": "https://slack.com/api/conversations.history",
         "params": {"channel": {"location": "query", "type": "string", "required": true},
                    "limit": {"location": "query", "type": "integer"}},
         "response_example": {"ok": true,
                              "messages": [{"type": "message", "ts": "1.2", "text": "..."}]}}]}

rote turns it into an :class:`~rote.vendor_specs.model.ApiSpec` with
``source="docs"``. The response schema is inferred from the documented
example, so the same request check and the same fake run on it. Two rules
keep it honest: every source must be a page fetched in this compile (or
``skill``, for an internal API the skill documents), and every place that
reports it says ``docs (lower trust)``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from rote.contracts import ContractFinding
from rote.vendor_specs import graphql, openapi
from rote.vendor_specs.model import ApiSpec, AuthSpec, Operation, ParamSpec

DOCS_DIRNAME = "vendor_specs"

#: Vendors whose own machine-readable spec rote reads: a docs spec may not
#: claim one.
PUBLISHED_VENDORS = frozenset(
    {
        "google",
        *(s.vendor for s in openapi.REGISTRY.values()),
        *(s.vendor for s in graphql.REGISTRY.values()),
    }
)


class _DocsParam(BaseModel):
    model_config = ConfigDict(extra="forbid")

    location: str = "query"
    type: str = "string"
    required: bool = False
    repeated: bool = False
    enum: list[str] | None = None


class _DocsOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    method: str
    url_template: str
    params: dict[str, _DocsParam] = Field(default_factory=dict)
    request_example: Any = None
    response_example: Any = None
    media_download: bool = False


class DocsSpecFile(BaseModel):
    """The file the agent writes under ``vendor_specs/``."""

    model_config = ConfigDict(extra="forbid")

    vendor: str
    api: str
    version: str
    sources: list[str] = Field(min_length=1)
    operations: list[_DocsOperation] = Field(min_length=1)
    #: The one credential the vendor's requests carry, as its docs describe:
    #: ``{"env": "SLACK_BOT_TOKEN", "location": "header", "name": "Authorization"}``.
    auth: AuthSpec | None = None

    @field_validator("vendor")
    @classmethod
    def _not_a_published_vendor(cls, vendor: str) -> str:
        # A docs spec stands in for a missing vendor spec. One that claims a
        # vendor rote reads a published spec for is the agent standing in
        # for the vendor (and, for Google, asking for the user's token).
        if vendor.strip().lower() in PUBLISHED_VENDORS:
            raise ValueError(
                f"{vendor.strip()} publishes a spec rote reads; check requests against it, "
                f"not a docs spec"
            )
        return vendor


class DocsSpecError(ValueError):
    """A ``vendor_specs/`` file that is not a docs spec."""


def load(root: Path) -> list[tuple[Path, ApiSpec]]:
    """Every docs spec in ``<root>/vendor_specs/``, as ``(file, ApiSpec)``.

    A file that does not parse raises :class:`DocsSpecError` naming it, so
    the agent fixes the file rather than having its requests go unchecked.
    """
    out: list[tuple[Path, ApiSpec]] = []
    folder = root / DOCS_DIRNAME
    if not folder.is_dir():
        return out
    for path in sorted(folder.glob("*.json")):
        try:
            raw = DocsSpecFile.model_validate(json.loads(path.read_text(encoding="utf-8")))
        except (ValueError, ValidationError) as e:
            raise DocsSpecError(f"{path.relative_to(root)}: {e}") from e
        out.append((path, to_api_spec(raw, path.relative_to(root).as_posix())))
    return out


def to_api_spec(raw: DocsSpecFile, where: str) -> ApiSpec:
    from rote.probe import infer_schema

    operations = []
    for op in raw.operations:
        params = {
            name: ParamSpec(
                name=name,
                location="path" if p.location == "path" else "query",
                type=p.type if p.type in _TYPES else "any",  # type: ignore[arg-type]
                required=p.required,
                repeated=p.repeated,
                enum=tuple(p.enum) if p.enum else None,
            )
            for name, p in op.params.items()
        }
        operations.append(
            Operation(
                id=op.id,
                method=op.method.upper(),  # type: ignore[arg-type]
                url_template=op.url_template,
                params=params,
                request_schema=_loose(infer_schema([op.request_example]))
                if op.request_example is not None
                else None,
                response_schema=_loose(infer_schema([op.response_example]))
                if op.response_example is not None
                else None,
                media_download=op.media_download,
            )
        )
    return ApiSpec(
        vendor=raw.vendor,
        api=raw.api,
        version=raw.version,
        revision=f"docs:{where}",
        source="docs",
        source_url=raw.sources[0],
        sources=list(raw.sources),
        operations=operations,
        auth=raw.auth,
    )


_TYPES = {"string", "integer", "number", "boolean", "object", "array", "any"}


def _loose(schema: Any) -> Any:
    """An inferred schema minus ``required``: one documented example cannot say
    which fields are optional."""
    if isinstance(schema, dict):
        return {k: _loose(v) for k, v in schema.items() if k != "required"}
    if isinstance(schema, list):
        return [_loose(v) for v in schema]
    return schema


def unfetched_sources(root: Path, was_fetched: Callable[[str], bool]) -> list[ContractFinding]:
    """A docs spec source the agent never fetched in this compile.

    A spec "from the docs" whose docs were never read is a spec from
    memory with a URL attached. ``skill`` (the skill documents the API) is
    exempt.
    """
    findings: list[ContractFinding] = []
    try:
        loaded = load(root)
    except DocsSpecError:
        return findings  # reported by the request check
    for path, spec in loaded:
        rel = path.relative_to(root).as_posix()
        for source in spec.sources:
            if source == "skill" or was_fetched(source):
                continue
            findings.append(
                ContractFinding(
                    node_id="(deliverable)",
                    kind="docs_spec_unfetched_source",
                    severity="error",
                    message=(
                        f"{rel} cites {source}, which this compile never fetched. Fetch it and "
                        f"check the spec against it, or cite the page you did read"
                    ),
                    path=rel,
                )
            )
    return findings


__all__ = [
    "DOCS_DIRNAME",
    "unfetched_sources",
    "DocsSpecError",
    "DocsSpecFile",
    "load",
    "to_api_spec",
]
