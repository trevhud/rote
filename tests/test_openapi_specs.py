"""Vendor APIs that publish OpenAPI 3: read from the vendor's own document.

The fixture is the official ClinicalTrials.gov document, trimmed (tests/
fixtures/vendor_specs/clinicaltrials_v2.openapi.json); nothing here
reaches the network.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

import pytest

from rote.ir import load_pipeline
from rote.vendor_specs import SpecStore, SpecUnavailable, check_deliverable
from rote.vendor_specs.openapi import REGISTRY, OpenApiError, from_openapi
from rote.vendor_specs.store import parse_document

FIXTURE = Path(__file__).parent / "fixtures" / "vendor_specs" / "clinicaltrials_v2.openapi.json"
SOURCE = REGISTRY["clinicaltrials.gov"]


def _doc() -> Any:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture
def store(tmp_path: Path) -> SpecStore:
    served = {SOURCE.url: _doc()}

    def fetch(url: str) -> Any:
        if url not in served:
            raise SpecUnavailable(f"not served: {url}")
        return served[url]

    return SpecStore(root=tmp_path / "specs", fetch=fetch)


def test_an_openapi_document_becomes_typed_absolute_operations() -> None:
    spec = from_openapi(_doc(), SOURCE)
    assert (spec.key, spec.revision, spec.source) == (
        "clinicaltrials/studies/v2",
        "2.0.5",
        "openapi",
    )
    listing = spec.operation("listStudies")
    assert listing is not None
    assert (listing.method, listing.url_template) == (
        "GET",
        "https://clinicaltrials.gov/api/v2/studies",
    )
    assert listing.params["pageSize"].type == "integer"
    assert listing.params["countTotal"].type == "boolean"
    # pipeDelimited, explode: false: one delimited string, not a repeated name.
    status = listing.params["filter.overallStatus"]
    assert (status.type, status.repeated) == ("string", False)
    assert listing.response_schema == {"$ref": "#/$defs/PagedStudies"}
    assert spec.schemas["PagedStudies"]["additionalProperties"] is False
    fetch = spec.operation("fetchStudy")
    assert fetch is not None and fetch.params["nctId"].required
    assert fetch.url_template == "https://clinicaltrials.gov/api/v2/studies/{nctId}"


def test_swagger_2_is_named_not_misread() -> None:
    with pytest.raises(OpenApiError, match="Swagger 2.0 is not supported"):
        from_openapi({"swagger": "2.0", "paths": {}}, SOURCE)


def test_a_yaml_document_parses_like_json() -> None:
    assert parse_document("openapi: '3.0.3'\npaths: {}\n", "u") == {"openapi": "3.0.3", "paths": {}}


PIPELINE = """\
name: trials
version: "0.1.0"
description: d
input: {type: In}
nodes:
  - id: search
    kind: external_call
    description: Search.
    impl: extracted/trials.py:search
    inputs: {request: pipeline.input}
edges: []
entry_nodes: [search]
exit_nodes: [search]
"""

#: The shape the real trial-scout compile wrote.
MODULE = """
import httpx

URL = "https://clinicaltrials.gov/api/v2/studies"


def search(request):
    params = {"query.cond": request["condition"], "pageSize": 10, "format": "json"}
    data = httpx.get(URL, params=params, timeout=30).json()
    return {"studies": data.get("studies", []), "next": data.get("nextPageToken")}
"""


def _kinds(tmp_path: Path, store: SpecStore, module: str) -> list[str]:
    root = tmp_path / "d"
    (root / "extracted").mkdir(parents=True)
    (root / "pipeline.yaml").write_text(PIPELINE, encoding="utf-8")
    (root / "extracted" / "trials.py").write_text(textwrap.dedent(module), encoding="utf-8")
    [check] = check_deliverable(load_pipeline(root / "pipeline.yaml"), root, store)
    assert check.spec == "clinicaltrials/studies/v2 rev 2.0.5"
    return [f.kind for f in check.findings]


@pytest.mark.parametrize(
    ("old", "new", "kinds"),
    [
        (None, None, []),
        ('"query.cond"', '"query.condition"', ["spec_unknown_param"]),
        ('"pageSize": 10', '"pageSize": "ten"', ["spec_param_value"]),
        ('"format": "json"', '"format": "xml"', ["spec_param_value"]),
        ('data.get("nextPageToken")', 'data.get("pageToken")', ["spec_unknown_response_field"]),
        ("api/v2/studies", "api/v2/trials", ["spec_unknown_operation"]),
    ],
)
def test_requests_are_checked_against_the_vendors_openapi(
    tmp_path: Path, store: SpecStore, old: str | None, new: str | None, kinds: list[str]
) -> None:
    module = MODULE if old is None else MODULE.replace(old, new or "", 1)
    assert _kinds(tmp_path, store, module) == kinds


def test_the_vendors_openapi_outranks_a_docs_spec_for_the_same_host(
    tmp_path: Path, store: SpecStore
) -> None:
    """The real trial-scout compile wrote a docs spec from third-party pages
    claiming `pageToken`; the vendor's document says `nextPageToken`."""
    root = tmp_path / "d"
    (root / "extracted").mkdir(parents=True)
    (root / "vendor_specs").mkdir()
    (root / "pipeline.yaml").write_text(PIPELINE, encoding="utf-8")
    (root / "extracted" / "trials.py").write_text(
        textwrap.dedent(MODULE.replace('data.get("nextPageToken")', 'data.get("pageToken")')),
        encoding="utf-8",
    )
    (root / "vendor_specs" / "clinicaltrials.studies.v2.json").write_text(
        json.dumps(
            {
                "vendor": "clinicaltrials",
                "api": "studies",
                "version": "v2",
                "sources": ["https://dev.to/some-post"],
                "operations": [
                    {
                        "id": "studies.search",
                        "method": "GET",
                        "url_template": "https://clinicaltrials.gov/api/v2/studies",
                        "params": {
                            "query.cond": {"location": "query"},
                            "pageSize": {"location": "query", "type": "integer"},
                            "format": {"location": "query"},
                        },
                        "response_example": {"studies": [], "pageToken": "x"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    [check] = check_deliverable(load_pipeline(root / "pipeline.yaml"), root, store)
    assert check.operation == "listStudies"
    assert [f.kind for f in check.findings] == ["spec_unknown_response_field"]


def test_the_store_caches_the_vendors_document_and_refetches_a_moved_one(tmp_path: Path) -> None:
    from dataclasses import replace

    calls: list[str] = []

    def fetch(url: str) -> Any:
        calls.append(url)
        return _doc()

    root = tmp_path / "specs"
    SpecStore(root=root, fetch=fetch).openapi(SOURCE)
    cached = json.loads((root / "clinicaltrials" / "studies" / "v2.json").read_text())
    assert cached["body"]["source_url"] == SOURCE.url
    assert cached["body"]["document"]["info"]["version"] == "2.0.5"
    SpecStore(root=root, fetch=fetch).openapi(SOURCE)
    assert calls == [SOURCE.url]
    moved = replace(SOURCE, url="https://clinicaltrials.gov/api/oas/v2.1")
    SpecStore(root=root, fetch=fetch).openapi(moved)
    assert calls == [SOURCE.url, moved.url]


def test_a_path_parameter_is_required_even_when_the_document_omits_it() -> None:
    """OpenAPI requires `required: true` on path parameters; some documents
    leave it out, and the path cannot be built without one anyway."""
    from rote.vendor_specs.openapi import _param

    p = _param({"name": "id", "in": "path", "schema": {"type": "string"}})
    assert p is not None and p.required
    q = _param({"name": "q", "in": "query", "schema": {"type": "string"}})
    assert q is not None and not q.required
    assert _param({"name": "X-Key", "in": "header", "schema": {"type": "string"}}) is None
