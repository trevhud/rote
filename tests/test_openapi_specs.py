"""Vendor APIs that publish OpenAPI 3: read from the vendor's own document.

The fixtures are the vendors' official documents, trimmed (tests/fixtures/
vendor_specs/*.openapi.json: ClinicalTrials.gov, Stripe, GitHub REST);
nothing here reaches the network. ``correct_calls/`` holds requests written
the way the vendors' docs say to make them: every one must pass.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

import pytest

from rote.ir import load_pipeline
from rote.vendor_specs import SpecStore, SpecUnavailable, check_deliverable
from rote.vendor_specs.check import RequestCheck, report_section
from rote.vendor_specs.model import ApiSpec, Operation, host_of
from rote.vendor_specs.openapi import REGISTRY, OpenApiError, by_key, from_openapi
from rote.vendor_specs.store import parse_document

FIXTURES = Path(__file__).parent / "fixtures" / "vendor_specs"
FIXTURE = FIXTURES / "clinicaltrials_v2.openapi.json"
SOURCE = REGISTRY["clinicaltrials.gov"]
STRIPE = REGISTRY["api.stripe.com"]
GITHUB = REGISTRY["api.github.com"]
#: Each registered document's trimmed fixture.
DOCUMENTS = {
    SOURCE.url: FIXTURE,
    STRIPE.url: FIXTURES / "stripe_v1.openapi.json",
    GITHUB.url: FIXTURES / "github_rest.openapi.json",
}
CORRECT_CALLS = FIXTURES / "correct_calls"


def _doc() -> Any:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


_PARSED: dict[str, Any] = {}


def _document(url: str) -> Any:
    if url not in _PARSED:
        _PARSED[url] = json.loads(DOCUMENTS[url].read_text(encoding="utf-8"))
    return _PARSED[url]


def _serve(url: str) -> Any:
    if url not in DOCUMENTS:
        raise SpecUnavailable(f"not served: {url}")
    return _document(url)


@pytest.fixture
def store(tmp_path: Path) -> SpecStore:
    return SpecStore(root=tmp_path / "specs", fetch=_serve)


@pytest.fixture(scope="module")
def stripe() -> ApiSpec:
    return from_openapi(_document(STRIPE.url), STRIPE)


@pytest.fixture(scope="module")
def github() -> ApiSpec:
    return from_openapi(_document(GITHUB.url), GITHUB)


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


def test_a_docs_spec_for_a_vendor_with_openapi_is_refused_and_the_vendors_spec_checks(
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
    refused, check = check_deliverable(load_pipeline(root / "pipeline.yaml"), root, store)
    assert [f.kind for f in refused.findings] == ["docs_spec_invalid"]
    assert "clinicaltrials publishes a spec rote reads" in refused.findings[0].message
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

    p = _param({"name": "id", "in": "path", "schema": {"type": "string"}}, {})
    assert p is not None and p.required
    q = _param({"name": "q", "in": "query", "schema": {"type": "string"}}, {})
    assert q is not None and not q.required
    assert _param({"name": "X-Key", "in": "header", "schema": {"type": "string"}}, {}) is None


# ── Stripe, GitHub: the shapes ClinicalTrials.gov does not have ──


def _op(spec: ApiSpec, op_id: str) -> Operation:
    op = spec.operation(op_id)
    assert op is not None, op_id
    return op


def test_a_deep_object_parameter_keeps_what_each_bracket_form_takes(stripe: ApiSpec) -> None:
    """Stripe's `expand` is `expand[]=...` only; `created` is a timestamp or
    `created[gte]=...`; `limit` is a plain integer."""
    params = _op(stripe, "GetCustomers").params
    expand, created, limit = params["expand"], params["created"], params["limit"]
    assert (expand.deep_object, expand.bare, expand.members) == (True, False, {})
    assert expand.items is not None and expand.items.type == "string"
    assert (created.deep_object, created.bare, created.type, created.items) == (
        True,
        True,
        "integer",
        None,
    )
    assert created.members is not None
    assert {k: m.type for k, m in created.members.items()} == {
        "gt": "integer",
        "gte": "integer",
        "lt": "integer",
        "lte": "integer",
    }
    assert (limit.deep_object, limit.bare, limit.type) == (False, True, "integer")


def test_a_multi_segment_path_parameter_spans_segments(github: ApiSpec) -> None:
    """GitHub's `x-multi-segment`: a file path or a ref keeps its slashes."""
    content = _op(github, "repos/get-content").url_template
    assert content == "https://api.github.com/repos/{owner}/{repo}/contents/{+path}"
    assert _op(github, "git/get-ref").url_template.endswith("/git/ref/{+ref}")
    assert _op(github, "repos/get").url_template.endswith("/repos/{owner}/{repo}")


def test_an_operation_is_at_its_own_server(stripe: ApiSpec, github: ApiSpec) -> None:
    """Stripe uploads files at files.stripe.com, GitHub release assets at
    uploads.github.com: the operation's `servers`, not the document's."""
    assert _op(stripe, "PostFiles").url_template == "https://files.stripe.com/v1/files"
    assert _op(stripe, "GetFiles").url_template == "https://api.stripe.com/v1/files"
    upload = _op(github, "repos/upload-release-asset").url_template
    assert upload.startswith("https://uploads.github.com/repos/")
    listing = _op(github, "repos/list-release-assets").url_template
    assert listing.startswith("https://api.github.com/repos/")


def test_every_host_a_registered_document_names_is_registered() -> None:
    for url, path in DOCUMENTS.items():
        doc = json.loads(path.read_text(encoding="utf-8"))
        servers = [*doc.get("servers", [])]
        for item in doc["paths"].values():
            servers += item.get("servers", [])
            servers += [
                s for op in item.values() if isinstance(op, dict) for s in op.get("servers", [])
            ]
        hosts = {host_of(s["url"].rstrip("/") + "/") for s in servers}
        assert hosts and all(REGISTRY[h or ""].url == url for h in hosts), hosts


def test_the_request_body_keeps_the_media_types_it_is_sent_as(
    stripe: ApiSpec, github: ApiSpec
) -> None:
    create = _op(stripe, "PostCustomers")
    assert create.request_media_types == ("application/x-www-form-urlencoded",)
    assert create.request_schema is not None
    assert "email" in create.request_schema["properties"]
    files = _op(stripe, "PostFiles")
    assert files.request_media_types == ("multipart/form-data",)
    assert files.request_schema is not None and "file" in files.request_schema["properties"]
    assert _op(github, "issues/create").request_media_types == ("application/json",)


def test_every_success_response_is_read_and_the_error_default_is_not() -> None:
    doc = parse_document(
        textwrap.dedent(
            """
            openapi: 3.0.3
            info: {version: "1"}
            servers: [{url: "https://clinicaltrials.gov/api/v2"}]
            paths:
              /a:
                get:
                  responses:
                    200:
                      content:
                        application/json:
                          schema: {type: object, properties: {done: {type: boolean}}}
                    201:
                      content:
                        application/json:
                          schema: {type: object, properties: {made: {type: string}}}
                    default:
                      content:
                        application/json:
                          schema: {type: object, properties: {error: {type: string}}}
              /b:
                get:
                  responses:
                    default:
                      content:
                        application/json:
                          schema: {type: object, properties: {only: {type: string}}}
            """
        ),
        "u",
    )
    spec = from_openapi(doc, SOURCE)
    a, b = spec.operations
    # YAML reads an unquoted 200 as an integer: still a success response.
    assert a.response_schema == {
        "anyOf": [
            {"type": "object", "properties": {"done": {"type": "boolean"}}},
            {"type": "object", "properties": {"made": {"type": "string"}}},
        ]
    }
    assert b.response_schema == {"type": "object", "properties": {"only": {"type": "string"}}}


@pytest.mark.parametrize(
    "doc",
    [
        {"openapi": "3.0.0", "paths": []},
        {"openapi": "3.0.0", "paths": {"/a": "x"}},
        {"openapi": "3.0.0", "paths": {"/a": {"get": "x"}}},
        {"openapi": "3.0.0", "paths": {"/a": {"get": {"parameters": {"a": 1}}}}},
        {"openapi": "3.0.0", "paths": {"/a": {"get": {"responses": []}}}},
        {"openapi": "3.0.0", "servers": "x", "paths": {}},
        {"openapi": "3.0.0", "components": {"schemas": {"A": []}}, "paths": {}},
        {"openapi": "3.0.0", "info": [], "paths": {}},
    ],
)
def test_a_malformed_document_is_a_spec_error_that_names_where(doc: Any) -> None:
    with pytest.raises(OpenApiError, match=r"is not (an object|a list)"):
        from_openapi(doc, SOURCE)


def test_a_malformed_document_leaves_the_request_not_checked(tmp_path: Path) -> None:
    """The compile goes on; the report says why nothing was checked."""
    store = SpecStore(
        root=tmp_path / "specs", fetch=lambda url: {"openapi": "3.0.0", "paths": {"/a": "x"}}
    )
    [check] = _run(tmp_path, store, CORRECT_CALLS / "clinicaltrials" / "meta.py")[:1]
    assert check.status == "spec_unavailable"
    assert check.note is not None and "paths['/a'] is not an object" in check.note
    assert report_section([check])[-1].endswith("is not an object |")


def _run(
    tmp_path: Path, store: SpecStore, module: Path, source: str | None = None
) -> list[RequestCheck]:
    root = tmp_path / module.parent.name / module.stem
    (root / "extracted").mkdir(parents=True)
    (root / "pipeline.yaml").write_text(PIPELINE, encoding="utf-8")
    text = module.read_text(encoding="utf-8") if source is None else source
    (root / "extracted" / "trials.py").write_text(
        text.replace("def f(", "def search("), encoding="utf-8"
    )
    return check_deliverable(load_pipeline(root / "pipeline.yaml"), root, store)


#: The one correct call whose response the spec does not fully describe:
#: GitHub's attestation bundle allows fields it does not list.
PARTIAL = {("github", "attestations.py")}


@pytest.mark.parametrize(
    "module", sorted(CORRECT_CALLS.glob("*/*.py")), ids=lambda p: f"{p.parent.name}/{p.name}"
)
def test_correct_real_world_calls_report_no_findings(
    tmp_path: Path, store: SpecStore, module: Path
) -> None:
    checks = _run(tmp_path, store, module)
    assert checks
    for check in checks:
        assert check.findings == [], [f.message for f in check.findings]
        assert check.operation is not None, check.note
        expected = "partial" if (module.parent.name, module.name) in PARTIAL else "ok"
        assert check.status == expected, (check.line, check.unchecked)


@pytest.mark.parametrize(
    ("module", "old", "new", "kinds"),
    [
        # Stripe rejects a bare `expand`: it is `expand[]` (deepObject).
        ("stripe/customers.py", '"expand[]": "data', '"expand": "data', ["spec_param_value"]),
        ("stripe/customers.py", '"expand[]": [', '"expand": [', ["spec_param_value"]),
        ("stripe/subscriptions.py", "expand[]=data", "expand=data", ["spec_param_value"]),
        ("stripe/charges.py", '"created[gte]"', '"created[gtee]"', ["spec_unknown_param"]),
        ("stripe/charges.py", '"created[gte]"', '"created[]"', ["spec_unknown_param"]),
        (
            "stripe/charges.py",
            '"created[gte]": request["since"]',
            '"created[gte]": "soon"',
            ["spec_param_value"],
        ),
        ("stripe/charges.py", '"limit": 100', '"created": "soon"', ["spec_param_value"]),
        (
            "stripe/charges.py",
            '"customer": request',
            '"customer[id]": request',
            ["spec_unknown_param"],
        ),
        (
            "stripe/customers.py",
            '"metadata[user_id]"',
            '"meta[user_id]"',
            ["spec_unknown_body_field"],
        ),
        ("stripe/customers.py", '"address[city]"', '"address[town]"', ["spec_unknown_body_field"]),
        (
            "stripe/payments.py",
            'data={\n            "amount"',
            'json={\n            "amount"',
            # As JSON, a bracketed form key is a field of that literal name:
            # automatic_payment_methods[enabled], metadata[order_id].
            ["spec_body_media", "spec_unknown_body_field", "spec_unknown_body_field"],
        ),
        ("stripe/files.py", "        files=", "        x=", ["spec_body_media"]),
        (
            "stripe/customers.py",
            'one.get("email")',
            'one.get("emial")',
            ["spec_unknown_response_field"],
        ),
        ("stripe/charges.py", 'charge["paid"]', 'charge["paied"]', ["spec_unknown_response_field"]),
        (
            "stripe/subscriptions.py",
            '["price"]["id"]',
            '["price"]["idd"]',
            ["spec_unknown_response_field"],
        ),
        (
            "stripe/customers.py",
            '"query": f',
            '"q": f',
            ["spec_unknown_param", "spec_missing_required"],
        ),
        ("github/issues.py", '"state": "open"', '"state": "opened"', ["spec_param_value"]),
        (
            "github/writes.py",
            '"conclusion": "success"',
            '"conclusions": "success"',
            ["spec_unknown_body_field"],
        ),
        (
            "github/writes.py",
            '"name": "report.zip"',
            '"label": "report.zip"',
            ["spec_missing_required"],
        ),
        ("github/repos.py", 'readme["sha"]', 'readme["sha1"]', ["spec_unknown_response_field"]),
        (
            "github/repos.py",
            'ref["object"]["sha"]',
            'ref["object"]["shaa"]',
            ["spec_unknown_response_field"],
        ),
        ("github/attestations.py", 'a["bundle"]', 'a["bundles"]', ["spec_unknown_response_field"]),
        (
            "clinicaltrials/studies.py",
            '"sort": ["LastUpdatePostDate:desc"]',
            '"sort": ["LastUpdatePostDate:desc", "EnrollmentCount"]',
            ["spec_param_value"],
        ),
    ],
)
def test_a_wrong_call_is_caught(
    tmp_path: Path, store: SpecStore, module: str, old: str, new: str, kinds: list[str]
) -> None:
    path = CORRECT_CALLS / module
    source = path.read_text(encoding="utf-8")
    assert source.count(old) == 1, old
    checks = _run(tmp_path, store, path, source.replace(old, new))
    assert [f.kind for c in checks for f in c.findings] == kinds


def test_a_partly_checked_request_is_reported_as_not_checked(
    tmp_path: Path, store: SpecStore
) -> None:
    [check] = _run(tmp_path, store, CORRECT_CALLS / "github" / "attestations.py")
    assert check.unchecked == [
        "line 9 reads attestations[].bundle.verificationMaterial.tlogEntries: the spec does "
        "not list the fields of attestations[].bundle.verificationMaterial, so 'tlogEntries' "
        "is not checked"
    ]
    row = next(line for line in report_section([check]) if line.startswith("| `"))
    assert "| no violations; not checked: line 9 reads" in row
    assert "| ok |" not in row


def test_a_response_with_no_json_schema_leaves_its_reads_not_checked(
    tmp_path: Path, store: SpecStore
) -> None:
    module = """
import httpx

def f(request):
    r = httpx.delete(f"https://api.github.com/repos/{request['o']}/{request['r']}")
    return r.json()["message"]
"""
    [check] = _run(tmp_path, store, CORRECT_CALLS / "github" / "x.py", module)
    assert (check.operation, check.status, check.findings) == ("repos/delete", "partial", [])
    assert "the spec gives no JSON response" in check.unchecked[0]


def test_a_spec_that_cannot_be_fetched_is_not_checked_never_ok(tmp_path: Path) -> None:
    def down(url: str) -> Any:
        raise SpecUnavailable(f"could not fetch {url}: timed out")

    store = SpecStore(root=tmp_path / "specs", fetch=down)
    checks = _run(tmp_path, store, CORRECT_CALLS / "stripe" / "files.py")
    assert [(c.status, c.note) for c in checks] == [
        ("spec_unavailable", f"could not fetch {STRIPE.url}: timed out")
    ]
    # Memoized: the second request does not refetch, and is not checked either.
    assert store.openapi.__self__ is store  # type: ignore[attr-defined]
    with pytest.raises(SpecUnavailable):
        store.openapi(STRIPE)


def test_the_vendors_spec_outranks_a_docs_spec_for_the_bench_and_the_live_check(
    tmp_path: Path, store: SpecStore
) -> None:
    from rote.bench.http_fixtures import specs_for
    from rote.vendor_specs.fixtures import FixtureSet
    from rote.vendor_specs.live import _specs_of

    root = tmp_path / "c"
    (root / "vendor_specs").mkdir(parents=True)
    (root / "vendor_specs" / "stripe.api.v1.json").write_text(
        json.dumps(
            {
                "vendor": "stripe",
                "api": "api",
                "version": "v1",
                "sources": ["https://example.com/blog"],
                "operations": [
                    {
                        "id": "customers.list",
                        "method": "GET",
                        "url_template": "https://api.stripe.com/v1/customers",
                        "response_example": {"results": []},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    from rote.vendor_specs import docs

    # The file itself is refused (Stripe publishes OpenAPI), so the bench
    # stops rather than fake Stripe from it.
    with pytest.raises(docs.DocsSpecError, match="stripe publishes a spec rote reads"):
        specs_for({"stripe/api/v1": FixtureSet(spec="stripe/api/v1")}, store, root)
    # Past the loader, the precedence still holds: the vendor's own spec.
    raw = json.loads((root / "vendor_specs" / "stripe.api.v1.json").read_text())
    raw["vendor"] = "elsewhere"
    written = [
        docs.to_api_spec(docs.DocsSpecFile.model_validate(raw), "x").model_copy(
            update={"vendor": "stripe"}
        )
    ]
    assert written[0].key == "stripe/api/v1" and written[0].source == "docs"
    [spec] = specs_for({"stripe/api/v1": FixtureSet(spec="stripe/api/v1")}, store, None)
    assert (spec.source, spec.key) == ("openapi", "stripe/api/v1")
    check = RequestCheck((), "x", 1, "ok", spec="stripe/api/v1 rev x")
    [live] = _specs_of([check], store, written)
    assert live.source == "openapi"
    assert by_key("stripe/api/v1") is STRIPE


# ── the live check, against a vendor's own OpenAPI document ──


def test_live_drift_follows_any_of_and_nullable_fields(tmp_path: Path, stripe: ApiSpec) -> None:
    """Stripe's retrieve-customer response is a customer or a deleted one,
    and `email` is nullable: a correct live response is no drift."""
    from rote.vendor_specs.fixture_store import FixtureStore
    from rote.vendor_specs.live import _compare

    op = _op(stripe, "GetCustomersCustomer")

    def drift(body: dict[str, Any]) -> list[str]:
        raw = json.dumps(body).encode()
        fixtures = FixtureStore(tmp_path / "fx")
        return _compare("n", stripe, op, 200, "application/json", raw, fixtures).schema_drift

    customer = {"id": "cus_1", "object": "customer", "email": None, "created": 1, "livemode": False}
    assert drift(customer) == []
    assert drift({"id": "cus_1", "object": "customer", "deleted": True}) == []
    assert drift({**customer, "emial": "x"}) == ["emial: returned live, not in the spec"]
    assert drift({**customer, "created": "soon"}) == ["created: expected integer, got str"]


def test_a_partly_checked_read_still_gets_its_live_check(tmp_path: Path, store: SpecStore) -> None:
    """Fields the spec does not list stay unchecked; they do not block the call."""
    from rote.compiler.backend import write_backend
    from rote.vendor_specs.fixture_store import FixtureStore
    from rote.vendor_specs.live import LiveRequest, live_check

    root = tmp_path / "compiled"
    (root / "extracted").mkdir(parents=True)
    (root / "pipeline.yaml").write_text(
        PIPELINE.replace("inputs: {request: pipeline.input}", "inputs: {org: pipeline.input.org}"),
        encoding="utf-8",
    )
    (root / "extracted" / "trials.py").write_text(
        textwrap.dedent(
            """
            import httpx

            def search(org):
                r = httpx.get(f"https://api.github.com/orgs/{org}/attestations/sha256:ab")
                found = r.json()["attestations"]
                return [a["bundle"]["verificationMaterial"]["tlogEntries"] for a in found]
            """
        ),
        encoding="utf-8",
    )
    write_backend(root, "api")
    sent: list[LiveRequest] = []

    def send(request: LiveRequest) -> Any:
        sent.append(request)
        return 200, "application/json", json.dumps({"attestations": []}).encode()

    report = live_check(
        root,
        {"org": "acme"},
        store,
        FixtureStore(tmp_path / "fx"),
        env={},
        send=send,
        confirm=lambda plan: True,
    )
    assert [r.url for r in sent] == ["https://api.github.com/orgs/acme/attestations/sha256:ab"]
    [check] = report.checked
    assert (check.operation, check.schema_drift) == ("orgs/list-attestations", [])


# ── small documents for the shapes the vendors' own use rarely ──

#: An OpenAPI 3.1 document at ClinicalTrials.gov's registered URL, so a
#: request to clinicaltrials.gov is checked against it.
SMALL: dict[str, Any] = {
    "openapi": "3.1.0",
    "info": {"version": "t"},
    # A relative server is resolved against the document's own URL.
    "servers": [{"url": "/api/{stage}", "variables": {"stage": {"default": "v2"}}}],
    "paths": {
        "/items": {"$ref": "#/components/pathItems/Items"},
        "/items/{id}": {
            "parameters": [{"$ref": "#/components/parameters/per~1page"}],
            "get": {
                "parameters": [{"name": "id", "in": "path", "schema": {"type": "string"}}],
                "responses": {
                    "200": {
                        "content": {
                            "application/fhir+json": {"schema": {}},
                            "application/json": {"schema": {"$ref": "#/components/schemas/Item"}},
                        }
                    }
                },
            },
            "post": {
                "requestBody": {"content": {"application/octet-stream": {}}},
                "responses": {"204": {}},
            },
            "put": {
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {"oneOf": [{"$ref": "#/components/schemas/Item"}, {}]}
                        }
                    }
                },
                "responses": {
                    "200": {
                        "content": {
                            "application/json": {"schema": {"$ref": "#/components/schemas/Loop"}}
                        }
                    }
                },
            },
        },
    },
    "components": {
        "pathItems": {
            "Items": {
                "get": {
                    "parameters": [
                        {"name": "n", "in": "query", "schema": {"type": ["null", "integer"]}},
                        # A reference cycle is no parameter, and ends.
                        {"$ref": "#/components/parameters/Loop"},
                        {
                            "name": "tag",
                            "in": "query",
                            "schema": {
                                "type": "array",
                                "items": {"type": "string", "enum": ["a", "b"]},
                            },
                        },
                        {
                            "name": "filter",
                            "in": "query",
                            "required": True,
                            "style": "deepObject",
                            "explode": True,
                            "schema": {"type": "object", "additionalProperties": True},
                        },
                    ],
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/Page"}
                                }
                            }
                        }
                    },
                }
            }
        },
        "parameters": {
            "per/page": {"name": "per_page", "in": "query", "schema": {"type": "integer"}},
            "Loop": {"$ref": "#/components/parameters/Loop"},
        },
        "schemas": {
            "Item": {"type": "object", "properties": {"id": {"type": "string"}}},
            "Page": {
                "allOf": [
                    {
                        "type": "object",
                        "properties": {"meta": {"type": "object", "properties": {"a": {}}}},
                    },
                    {
                        "type": "object",
                        "properties": {"meta": {"type": "object", "properties": {"b": {}}}},
                    },
                    {
                        "type": "object",
                        "properties": {
                            "free": {"type": "object"},
                            "any": {"type": "object", "additionalProperties": {}},
                            "other": {"$ref": "other.json#/Item"},
                        },
                    },
                ]
            },
            "Loop": {"$ref": "#/components/schemas/Loop2"},
            "Loop2": {"$ref": "#/components/schemas/Loop"},
        },
    },
}


@pytest.fixture
def small(tmp_path: Path) -> SpecStore:
    return SpecStore(root=tmp_path / "specs", fetch=lambda url: SMALL)


def _small(tmp_path: Path, store: SpecStore, body: str) -> list[RequestCheck]:
    module = "import httpx\n\nB = 'https://clinicaltrials.gov/api/v2'\n\n\ndef f(request):\n"
    module += textwrap.indent(textwrap.dedent(body).strip(), "    ") + "\n"
    return _run(tmp_path, store, CORRECT_CALLS / "small" / "m.py", module)


def test_a_small_31_document_converts_every_shape() -> None:
    spec = from_openapi(SMALL, SOURCE)
    items = _op(spec, "GET /items")
    # A relative server with a variable, against the document's own URL.
    assert items.url_template == "https://clinicaltrials.gov/api/v2/items"
    # 3.1's ["null", "integer"] is an integer; a repeated item keeps its enum.
    assert items.params["n"].type == "integer"
    assert (items.params["tag"].repeated, items.params["tag"].enum) == (True, ("a", "b"))
    # The path-level parameter, through a ref whose key holds an escaped "/".
    assert set(_op(spec, "GET /items/{id}").params) == {"per_page", "id"}


@pytest.mark.parametrize(
    ("body", "kinds", "status"),
    [
        # A deepObject parameter with an open object: any member, "0" too;
        # sent as a member, a required one is sent.
        ('httpx.get(B + "/items", params={"filter[any]": 1, "filter[0]": 2})', [], "ok"),
        ('httpx.get(B + "/items", params={"n": 1})', ["spec_missing_required"], "violations"),
        ('httpx.get(B + "/items?filter%5Bx%5D=1&n=2")', [], "ok"),
        (
            'httpx.get(B + "/items", params={"filter[x]": 1, "tag": ["a", "c"]})',
            ["spec_param_value"],
            "violations",
        ),
        (
            'httpx.get(B + "/items", params={"filter[x]": 1, "n": "x"})',
            ["spec_param_value"],
            "violations",
        ),
        # allOf: `meta` is the allOf of both parts.
        (
            'r = httpx.get(B + "/items", params={"filter[x]": 1}).json()\n'
            'return r["meta"]["a"], r["meta"]["b"]',
            [],
            "ok",
        ),
        (
            'r = httpx.get(B + "/items", params={"filter[x]": 1}).json()\nreturn r["meta"]["c"]',
            ["spec_unknown_response_field"],
            "violations",
        ),
        # An object listing no fields, an open map, a ref rote cannot follow:
        # not checked, never a finding.
        (
            'r = httpx.get(B + "/items", params={"filter[x]": 1}).json()\nreturn r["free"]["x"]',
            [],
            "partial",
        ),
        (
            'r = httpx.get(B + "/items", params={"filter[x]": 1}).json()\nreturn r["any"]["x"]',
            [],
            "partial",
        ),
        (
            'r = httpx.get(B + "/items", params={"filter[x]": 1}).json()\nreturn r["other"]["x"]',
            [],
            "partial",
        ),
        # application/json over the JSON variant listed before it.
        ('return httpx.get(B + "/items/1").json()["id"]', [], "ok"),
        (
            'return httpx.get(B + "/items/1").json()["idd"]',
            ["spec_unknown_response_field"],
            "violations",
        ),
        # A body the spec gives no schema for; a schema that is a choice
        # with an open branch; a response that is a reference cycle.
        ('httpx.post(B + "/items/1", json={"a": 1})', ["spec_body_media"], "violations"),
        ('httpx.post(B + "/items/1", data="raw")', [], "ok"),
        ('return httpx.put(B + "/items/1", json={"x": 1}).json()["y"]', [], "partial"),
    ],
)
def test_small_document_requests(
    tmp_path: Path, small: SpecStore, body: str, kinds: list[str], status: str
) -> None:
    [check] = _small(tmp_path, small, body)
    assert ([f.kind for f in check.findings], check.status) == (kinds, status), check.unchecked


def test_what_is_not_checked_is_named(tmp_path: Path, small: SpecStore) -> None:
    [check] = _small(tmp_path, small, 'httpx.post(B + "/items/1", content=b"x", json={"a": 1})')
    assert check.unchecked == ["request body: the spec gives no schema for it"]
    [check] = _small(
        tmp_path / "2", small, 'return httpx.put(B + "/items/1", json={"x": 1}).json()["y"]'
    )
    assert check.unchecked == [
        "request body field 'x': the request schema does not say which fields it takes",
        "line 7 reads y: the spec does not list the fields of the response, so 'y' is not checked",
    ]
    # `a` is listed and holds anything: what is not checked is the read below it.
    read = 'return httpx.get(B + "/items", params={"filter[x]": 1}).json()["meta"]["a"]["x"]'
    [check] = _small(tmp_path / "3", small, read)
    assert check.unchecked == [
        "line 7 reads meta.a.x: the spec does not list the fields of meta.a, so 'x' is not checked"
    ]


def test_a_multipart_body_is_checked_by_its_files_too(tmp_path: Path, store: SpecStore) -> None:
    files = CORRECT_CALLS / "stripe" / "files.py"
    wrong = files.read_text(encoding="utf-8").replace('files={"file":', 'files={"upload":')
    [check] = _run(tmp_path, store, files, wrong)
    assert [f.kind for f in check.findings] == ["spec_unknown_body_field"]
    customers = CORRECT_CALLS / "stripe" / "customers.py"
    multipart = customers.read_text(encoding="utf-8").replace(
        '"preferred_locales[0]": "en",\n        },',
        '"preferred_locales[0]": "en",\n        },\n        files={},',
    )
    kinds = [f.kind for c in _run(tmp_path / "2", store, customers, multipart) for f in c.findings]
    assert kinds == ["spec_body_media"]


def test_a_failed_fetch_is_remembered_not_retried(tmp_path: Path) -> None:
    calls: list[str] = []

    def down(url: str) -> Any:
        calls.append(url)
        raise SpecUnavailable("down")

    store = SpecStore(root=tmp_path / "specs", fetch=down)
    for _ in range(2):
        with pytest.raises(SpecUnavailable):
            store.openapi(STRIPE)
    assert calls == [STRIPE.url]


def test_the_live_check_finds_a_hosts_spec_by_the_registry(store: SpecStore) -> None:
    from rote.vendor_specs.live import _spec_for_url

    docs_spec = from_openapi(_document(SOURCE.url), SOURCE).model_copy(update={"source": "docs"})
    found = _spec_for_url("https://clinicaltrials.gov/api/v2/studies", store, [docs_spec])
    assert found is not None and found.source == "openapi"
    found = _spec_for_url("https://files.stripe.com/v1/files", store, [])
    assert found is not None and found.key == "stripe/api/v1"


def test_a_document_rote_cannot_read_is_not_cached(tmp_path: Path) -> None:
    root = tmp_path / "specs"
    broken = SpecStore(root=root, fetch=lambda url: {"openapi": "3.0.0", "paths": []})
    with pytest.raises(SpecUnavailable, match="paths is not an object"):
        broken.openapi(SOURCE)
    calls: list[str] = []

    def fetch(url: str) -> Any:
        calls.append(url)
        return _doc()

    assert SpecStore(root=root, fetch=fetch).openapi(SOURCE).revision == "2.0.5"
    assert calls == [SOURCE.url]
