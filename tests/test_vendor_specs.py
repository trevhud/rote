"""Vendor requests are checked against the vendor's own machine-readable spec.

The specs are trimmed real Google Discovery documents (tests/fixtures/
vendor_specs, Gmail v1 and Drive v3 rev 20261012), served through an
injected fetch: no test touches the network (conftest blocks it).
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

import pytest

from rote.compiler.verify import EmitTarget, verify_deliverable
from rote.ir import load_pipeline
from rote.vendor_specs import SpecStore, SpecUnavailable, check_deliverable
from rote.vendor_specs.extract import Expr, extract
from rote.vendor_specs.google import DIRECTORY_URL, UrlApi, api_for_url, from_discovery

FIXTURES = Path(__file__).parent / "fixtures" / "vendor_specs"
GMAIL_URL = "https://gmail.googleapis.com/$discovery/rest?version=v1"
DRIVE_URL = "https://www.googleapis.com/discovery/v1/apis/drive/v3/rest"


#: discoveryRestUrl -> the trimmed document served for it.
_DOCUMENTS = {
    GMAIL_URL: "gmail_v1.discovery.json",
    DRIVE_URL: "drive_v3.discovery.json",
    "https://sheets.googleapis.com/$discovery/rest?version=v4": "sheets_v4.discovery.json",
    "https://people.googleapis.com/$discovery/rest?version=v1": "people_v1.discovery.json",
    "https://admin.googleapis.com/$discovery/rest?version=directory_v1": (
        "admin_directory_v1.discovery.json"
    ),
    "https://www.googleapis.com/discovery/v1/apis/oauth2/v2/rest": "oauth2_v2.discovery.json",
}


def _served() -> dict[str, Any]:
    return {
        DIRECTORY_URL: json.loads((FIXTURES / "google_directory.json").read_text()),
        **{url: json.loads((FIXTURES / name).read_text()) for url, name in _DOCUMENTS.items()},
    }


class _Fetch:
    def __init__(self) -> None:
        self.served = _served()
        self.calls: list[str] = []

    def __call__(self, url: str) -> Any:
        self.calls.append(url)
        if url not in self.served:
            raise SpecUnavailable(f"not served: {url}")
        return self.served[url]


@pytest.fixture
def store(tmp_path: Path) -> SpecStore:
    return SpecStore(root=tmp_path / "specs", fetch=_Fetch())


PIPELINE = """\
name: dock
version: "0.1.0"
description: d
input:
  type: In
nodes:
  - id: fetch_emails
    kind: external_call
    description: Search Gmail.
    impl: extracted/vendor.py:fetch_emails
    mcp: {server: gmail, tool: search_threads}
  - id: fetch_sheet
    kind: external_call
    description: Export a sheet.
    impl: extracted/vendor.py:fetch_sheet
edges: []
entry_nodes: [fetch_emails, fetch_sheet]
exit_nodes: [fetch_emails, fetch_sheet]
"""

#: The shape the real ops-report compile wrote (claude, legacy path).
GOOD = """
import os

import httpx

_THREADS = "https://gmail.googleapis.com/gmail/v1/users/me/threads"
_EXPORT = "https://www.googleapis.com/drive/v3/files/{file_id}/export"


def fetch_emails(query: str) -> dict:
    params: dict = {"q": query, "maxResults": 500}
    threads = []
    with httpx.Client(timeout=25.0) as client:
        while True:
            response = client.get(_THREADS, headers={}, params=params)
            body = response.json()
            threads.extend(body.get("threads") or [])
            ids = [t["id"] for t in body.get("threads", [])]
            if not body.get("nextPageToken"):
                break
    return {"threads": threads, "ids": ids}


def fetch_sheet(file_id: str) -> dict:
    url = _EXPORT.format(file_id=file_id)
    response = httpx.get(url, params={"mimeType": "text/csv"}, follow_redirects=True)
    return {"content": response.text}
"""


def _deliverable(tmp_path: Path, module: str) -> Path:
    root = tmp_path / "d"
    (root / "extracted").mkdir(parents=True)
    (root / "pipeline.yaml").write_text(PIPELINE, encoding="utf-8")
    (root / "extracted" / "vendor.py").write_text(textwrap.dedent(module), encoding="utf-8")
    return root


def _kinds(root: Path, store: SpecStore) -> list[str]:
    checks = check_deliverable(load_pipeline(root / "pipeline.yaml"), root, store)
    return sorted(f.kind for c in checks for f in c.findings)


# ───────── Discovery → ApiSpec ─────────


def test_a_discovery_document_becomes_absolute_typed_operations() -> None:
    gmail = from_discovery(_served()[GMAIL_URL], GMAIL_URL)
    assert (gmail.key, gmail.revision) == ("google/gmail/v1", "20261012")
    op = gmail.operation("gmail.users.threads.list")
    assert op is not None
    assert (op.method, op.url_template) == (
        "GET",
        "https://gmail.googleapis.com/gmail/v1/users/{userId}/threads",
    )
    assert op.params["maxResults"].type == "integer"
    assert op.params["labelIds"].repeated and op.params["userId"].location == "path"
    assert op.response_schema == {"$ref": "#/$defs/ListThreadsResponse"}
    get = gmail.operation("gmail.users.threads.get")
    assert get is not None and get.params["format"].enum == ("full", "metadata", "minimal")
    assert "fields" in gmail.global_params
    drive = from_discovery(_served()[DRIVE_URL], DRIVE_URL)
    export = drive.operation("drive.files.export")
    assert export is not None and export.media_download and export.response_schema is None
    assert export.url_template == "https://www.googleapis.com/drive/v3/files/{fileId}/export"
    assert export.params["mimeType"].required


@pytest.mark.parametrize(
    ("url", "named"),
    [
        (
            "https://gmail.googleapis.com/gmail/v1/users/me/threads",
            UrlApi("gmail", ("gmail_v1", "v1"), dedicated_host=True),
        ),
        (
            "https://www.googleapis.com/drive/v3/files/{id}/export",
            UrlApi("drive", ("v3",), dedicated_host=False),
        ),
        ("https://www.googleapis.com/upload/drive/v3/files", UrlApi("drive", ("v3",), False)),
        (
            "https://www.googleapis.com/resumable/upload/drive/v3/files",
            UrlApi("drive", ("v3",), False),
        ),
        (
            "https://sheets.googleapis.com/v4/spreadsheets/{id}/values/{r}",
            UrlApi("sheets", ("v4",), True),
        ),
        # The directory lists Admin SDK Directory as `directory_v1`.
        (
            "https://admin.googleapis.com/admin/directory/v1/users",
            UrlApi("admin", ("directory_v1", "v1"), True),
        ),
        ("https://www.googleapis.com/oauth2/v3/userinfo", UrlApi("oauth2", ("v3",), False)),
        ("https://oauth2.googleapis.com/token", UrlApi("oauth2", (), True)),
        ("https://api.hubapi.com/crm/v3/objects", None),
        ("{?}/threads", None),
    ],
)
def test_a_google_url_names_its_api_and_candidate_versions(url: str, named: Any) -> None:
    assert api_for_url(url) == named


# ───────── The store ─────────


def test_the_store_caches_by_vendor_api_version_and_never_reads_stale(tmp_path: Path) -> None:
    fetch = _Fetch()
    root = tmp_path / "specs"
    assert SpecStore(root=root, fetch=fetch).google("gmail", "v1") is not None
    cached = json.loads((root / "google" / "gmail" / "v1.json").read_text())
    # The vendor's document is what is cached, not rote's conversion of it.
    assert cached["body"]["document"]["revision"] == "20261012"
    assert cached["body"]["source_url"] == GMAIL_URL
    assert len(fetch.calls) == 2  # the directory, then the document

    def offline(url: str) -> Any:
        raise SpecUnavailable("offline")

    # Fresh cache: no fetch at all.
    assert SpecStore(root=root, fetch=offline).google("gmail", "v1") is not None
    # Expired cache: refetched, and a failed refetch is an error, not a stale read.
    with pytest.raises(SpecUnavailable):
        SpecStore(root=root, fetch=offline, ttl_seconds=-1).google("gmail", "v1")
    # An API the directory does not list is None, not an error.
    assert SpecStore(root=root, fetch=_Fetch()).google("gmail", "v9") is None


# ───────── Extraction ─────────


def test_extraction_reads_requests_and_response_fields_from_raw_http() -> None:
    sites = extract(textwrap.dedent(GOOD))
    emails, sheet = sites
    assert (emails.method, emails.url) == (
        "GET",
        "https://gmail.googleapis.com/gmail/v1/users/me/threads",
    )
    assert emails.params == {"q": Expr("{query}"), "maxResults": 500}
    assert {p for p, _ in emails.response_reads} == {
        ("threads",),
        ("threads", "[]", "id"),
        ("nextPageToken",),
    }
    assert sheet.url == "https://www.googleapis.com/drive/v3/files/{file_id}/export"
    assert sheet.params == {"mimeType": "text/csv"}


def test_extraction_follows_googles_client_library() -> None:
    """The three shapes real compiles wrote: build() inline, a helper that
    wraps build(), and a resource bound to a name with **kwargs."""
    source = textwrap.dedent(
        """
        from googleapiclient.discovery import build

        def _service(api, version, scope):
            return build(api, version, credentials=None)

        def a():
            service = build("gmail", "v1")
            page = service.users().threads().list(userId="me", q="x").execute()
            return [t["snippet"] for t in page.get("threads", [])]

        def b():
            threads = _service("gmail", "v1", "s").users().threads()
            args = {"userId": "me"}
            args["pageToken"] = "t"
            got = threads.get(**args, id="1", format="metadata").execute()
            return got["messages"][0]["payload"]["headers"]
        """
    )
    sites = extract(source)
    assert [s.google_method_id for s in sites] == [
        "gmail.users.threads.list",
        "gmail.users.threads.get",
    ]
    listed, got = sites
    assert ("threads", "[]", "snippet") in {p for p, _ in listed.response_reads}
    assert got.params == {"userId": "me", "pageToken": "t", "id": "1", "format": "metadata"}
    assert ("messages", "[]", "payload", "headers") in {p for p, _ in got.response_reads}


# ───────── The check ─────────


def test_correct_requests_pass(tmp_path: Path, store: SpecStore) -> None:
    root = _deliverable(tmp_path, GOOD)
    checks = check_deliverable(load_pipeline(root / "pipeline.yaml"), root, store)
    assert [(c.status, c.operation) for c in checks] == [
        ("ok", "gmail.users.threads.list"),
        ("ok", "drive.files.export"),
    ]


@pytest.mark.parametrize(
    ("old", "new", "kind"),
    [
        (
            '{"q": query, "maxResults": 500}',
            '{"query": query, "maxResults": 500}',
            "spec_unknown_param",
        ),
        ('"maxResults": 500', '"maxResults": "lots"', "spec_param_value"),
        ('"maxResults": 500', '"maxResults": [1, 2]', "spec_param_value"),
        ('params={"mimeType": "text/csv"}', "params={}", "spec_missing_required"),
        ("/export", "/exportCsv", "spec_unknown_operation"),
        ('t["id"]', 't["subject"]', "spec_unknown_response_field"),
        ('body.get("nextPageToken")', 'body.get("next_page_token")', "spec_unknown_response_field"),
        (
            'return {"content": response.text}',
            'return response.json()["rows"]',
            "spec_unknown_response_field",
        ),
        ("gmail/v1", "gmail/v9", "spec_unknown_api"),
        ("client.get(_THREADS", "client.post(_THREADS", "spec_unknown_operation"),
    ],
)
def test_each_spec_violation_is_found(
    tmp_path: Path, store: SpecStore, old: str, new: str, kind: str
) -> None:
    assert old in GOOD
    root = _deliverable(tmp_path, GOOD.replace(old, new, 1))
    assert _kinds(root, store) == [kind]


def test_googles_client_library_is_checked_by_method_id(tmp_path: Path, store: SpecStore) -> None:
    module = """
    from googleapiclient.discovery import build

    def fetch_emails(query):
        svc = build("gmail", "v1")
        got = svc.users().threads().get(userId="me", id="1", format="headers").execute()
        return {"subject": got["subject"], "x": svc.users().threads().search(q=query).execute()}

    def fetch_sheet(file_id):
        return build("drive", "v3").files().export(fileId=file_id).execute()
    """
    root = _deliverable(tmp_path, module)
    assert _kinds(root, store) == [
        "spec_missing_required",  # drive export without mimeType
        "spec_param_value",  # format="headers" is not in the enum
        "spec_unknown_operation",  # threads.search does not exist
        "spec_unknown_response_field",  # a Thread has no subject
    ]


def test_what_cannot_be_checked_is_listed_not_failed(tmp_path: Path, tmp_path_factory: Any) -> None:
    module = """
    import httpx

    def fetch_emails(query):
        return httpx.get("https://api.hubapi.com/crm/v3/objects/contacts").json()

    def fetch_sheet(file_id):
        return httpx.get("https://gmail.googleapis.com/gmail/v1/users/me/threads").json()
    """
    root = _deliverable(tmp_path, module)

    def offline(url: str) -> Any:
        raise SpecUnavailable("discovery is down")

    store = SpecStore(root=tmp_path_factory.mktemp("specs"), fetch=offline)
    checks = check_deliverable(load_pipeline(root / "pipeline.yaml"), root, store)
    # A vendor with no spec rote reads asks for one written from the docs;
    # an unreachable spec service is listed, never failed.
    assert [(c.status, [f.kind for f in c.findings]) for c in checks] == [
        ("violations", ["spec_missing"]),
        ("spec_unavailable", []),
    ]
    assert "api.hubapi.com" in checks[0].findings[0].message


# ───────── In the compile ─────────


def test_verification_holds_api_requests_to_the_spec(tmp_path: Path, store: SpecStore) -> None:
    bad = GOOD.replace('t["id"]', 't["subject"]')
    root = _deliverable(tmp_path, bad)
    api = verify_deliverable(
        root, run_tests=False, emit_target=EmitTarget("dbos", "api"), spec_store=store
    )
    mcp = verify_deliverable(
        root, run_tests=False, emit_target=EmitTarget("dbos", "mcp"), spec_store=store
    )
    assert "spec_unknown_response_field" in {f.kind for f in api.errors}
    assert [c.status for c in api.spec_checks] == ["violations", "ok"]
    assert not mcp.spec_checks
    assert "spec_unknown_response_field" not in {f.kind for f in mcp.errors}
    message = next(f.message for f in api.errors if f.kind == "spec_unknown_response_field")
    assert "threads[].subject" in message and "gmail.users.threads.list" in message


@pytest.mark.asyncio
async def test_a_compile_bounces_a_spec_violation_and_reports_every_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rote.compiler import Compiler
    from rote.compiler.drivers import CompilerDriver, DriverResult

    monkeypatch.setattr("rote.vendor_specs.store.http_get_json", _Fetch())
    versions = [GOOD.replace('t["id"]', 't["subject"]'), GOOD]

    class _Driver(CompilerDriver):
        name = "fake"
        repairs: list[str] = []

        def is_available(self) -> tuple[bool, str]:
            return (True, "")

        async def run(  # type: ignore[override]
            self,
            skill_dir: Path,
            compiler_skill_dir: Path,
            work_dir: Path,
            extra_instructions: str | None = None,
            on_event: Any = None,
            repair: Any = None,
        ) -> DriverResult:
            _write_into(work_dir, versions.pop(0))
            while (msg := repair(work_dir / "pipeline.yaml")) is not None:
                self.repairs.append(msg)
                if versions:
                    _write_into(work_dir, versions.pop(0))
            return DriverResult(work_dir / "pipeline.yaml", work_dir, self.name, {})

    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: dock\n---\n\n# Dock\n", encoding="utf-8")
    rubric = tmp_path / "rubric"
    rubric.mkdir()
    (rubric / "SKILL.md").write_text("# r", encoding="utf-8")
    driver = _Driver()
    compiler = Compiler(
        system1=False,  # a fake rubric
        compiler_skill_dir=rubric,
        emit_target=EmitTarget("dbos", "api"),
        run_generated_tests=False,
        tool_discovery="none",
    )
    compiler.select_driver = lambda: driver  # type: ignore[method-assign]
    await compiler.compile(skill, tmp_path / "out")
    assert driver.repairs and "threads[].subject" in driver.repairs[0]
    report = (tmp_path / "out" / "compile-report.md").read_text(encoding="utf-8")
    assert "### Vendor API spec checks" in report
    assert "| gmail.users.threads.list | google/gmail/v1 rev 20261012 | ok |" in report


def _write_into(work_dir: Path, module: str) -> None:
    (work_dir / "extracted").mkdir(parents=True, exist_ok=True)
    (work_dir / "pipeline.yaml").write_text(PIPELINE, encoding="utf-8")
    (work_dir / "extracted" / "vendor.py").write_text(textwrap.dedent(module), encoding="utf-8")
    (work_dir / "tests").mkdir(exist_ok=True)
    (work_dir / "tests" / "test_vendor.py").write_text(
        "from extracted.vendor import fetch_emails, fetch_sheet\n"
        "def test_named():\n    assert fetch_emails and fetch_sheet\n",
        encoding="utf-8",
    )


def test_the_client_librarys_media_variants_are_real_operations(
    tmp_path: Path, store: SpecStore
) -> None:
    """files().export_media() and files().get_media() are drive.files.export /
    drive.files.get returning bytes. A real compile was bounced for
    export_media when the check did not know the variant."""
    module = """
    from googleapiclient.discovery import build

    def fetch_emails(query):
        return {"x": build("gmail", "v1").users().threads().list_media(userId="me").execute()}

    def fetch_sheet(file_id):
        drive = build("drive", "v3")
        data = drive.files().export_media(fileId=file_id, mimeType="text/csv").execute()
        raw = drive.files().get_media(fileId=file_id).execute()
        return {"content": data.decode("utf-8"), "raw": raw}
    """
    root = _deliverable(tmp_path, module)
    checks = check_deliverable(load_pipeline(root / "pipeline.yaml"), root, store)
    assert [(c.operation, c.status) for c in checks] == [
        (None, "violations"),  # threads.list has no media download
        ("drive.files.export", "ok"),
        ("drive.files.get", "ok"),
    ]


# ───────── Correct Google code passes (review6's false positives) ─────────

#: Each shape a correct compile writes that the check used to refuse.
CORRECT_GOOGLE = """
import io

import httpx
import requests
from google.auth.transport.requests import AuthorizedSession
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from httpx import Client
from requests import get as requests_get

SHEETS = "https://sheets.googleapis.com/v4/spreadsheets"


def list_pages(folder):
    drive = build("drive", "v3")
    request = drive.files().list(q=f"'{folder}' in parents", pageSize=100)
    names = []
    while request is not None:
        page = request.execute()
        names += [f["name"] for f in page.get("files", [])]
        request = drive.files().list_next(request, page)
    return names


def upload_with_client(data):
    drive = build("drive", "v3")
    media = MediaIoBaseUpload(io.BytesIO(data), mimetype="text/csv")
    created = drive.files().create(
        body={"name": "report.csv", "mimeType": "text/csv"}, media_body=media, fields="id"
    ).execute()
    return created["id"]


def upload_over_http(data, token):
    response = httpx.post(
        "https://www.googleapis.com/upload/drive/v3/files",
        params={"uploadType": "media"},
        headers={"Authorization": f"Bearer {token}"},
        content=data,
    )
    return response.json()["id"]


def append_rows(sid, rng, rows):
    r = httpx.post(
        f"{SHEETS}/{sid}/values/{rng}:append",
        params={"valueInputOption": "RAW"},
        json={"values": rows},
    )
    return r.json()["updates"]["updatedRows"]


def batch_update(sid, reqs):
    return httpx.post(f"{SHEETS}/{sid}:batchUpdate", json={"requests": reqs}).json()["replies"]


def directory_users():
    return requests.get(
        "https://admin.googleapis.com/admin/directory/v1/users",
        params={"customer": "my_customer", "maxResults": 100},
    ).json()["users"]


def userinfo_v2(token):
    return httpx.get("https://www.googleapis.com/oauth2/v2/userinfo").json()["email"]


def person(token):
    body = httpx.get(
        "https://people.googleapis.com/v1/people/me", params={"personFields": "names"}
    ).json()
    return body["names"]


def connections():
    return httpx.get(
        "https://people.googleapis.com/v1/people/me/connections",
        params={"personFields": "names", "pageSize": 100},
    ).json()["connections"]


def person_with_client():
    people = build("people", "v1")
    return people.people().get(resourceName="people/me", personFields="names").execute()["names"]


def authorized_session(creds):
    session = AuthorizedSession(creds)
    threads = session.get(
        "https://gmail.googleapis.com/gmail/v1/users/me/threads", params={"q": "label:x"}
    ).json()
    return threads["threads"]


def imported_client():
    with Client() as c:
        return c.get(
            "https://gmail.googleapis.com/gmail/v1/users/me/threads", params={"q": "x"}
        ).json()["resultSizeEstimate"]


def imported_function():
    return requests_get(
        "https://www.googleapis.com/drive/v3/files", params={"q": "trashed = false"}
    ).json()["files"]
"""

_CORRECT_OPS = [
    "drive.files.list",
    "drive.files.list",  # list_next is the next page of the same list call
    "drive.files.create",
    "drive.files.create",  # the upload URL is the same operation
    "sheets.spreadsheets.values.append",
    "sheets.spreadsheets.batchUpdate",
    "directory.users.list",
    "oauth2.userinfo.get",
    "people.people.get",
    "people.people.connections.list",
    "people.people.get",
    "gmail.users.threads.list",
    "gmail.users.threads.list",
    "drive.files.list",
]


def test_correct_google_code_of_every_shape_passes(tmp_path: Path, store: SpecStore) -> None:
    root = _deliverable(tmp_path, CORRECT_GOOGLE)
    checks = check_deliverable(load_pipeline(root / "pipeline.yaml"), root, store)
    assert [(c.operation, c.status) for c in checks] == [(op, "ok") for op in _CORRECT_OPS], [
        f.message for c in checks for f in c.findings
    ]


def test_an_endpoint_discovery_does_not_describe_is_not_checked_not_failed(
    tmp_path: Path, store: SpecStore
) -> None:
    """OAuth2 v3's userinfo is a real OpenID endpoint Discovery lists no
    document for: unresolved, on a shared host, rather than a wrong version."""
    module = """
    import httpx

    def userinfo(token):
        return httpx.get("https://www.googleapis.com/oauth2/v3/userinfo").json()["email"]
    """
    root = _deliverable(tmp_path, module)
    [check] = check_deliverable(load_pipeline(root / "pipeline.yaml"), root, store)
    assert (check.status, check.findings) == ("unresolved", [])
    assert "describes no oauth2 v3" in (check.note or "")


@pytest.mark.parametrize(
    ("old", "new", "kind"),
    [
        # Each correct shape, broken once: the checks still bite.
        ('f["name"]', 'f["title"]', "spec_unknown_response_field"),
        (
            "drive.files().list(q=f\"'{folder}' in parents\", pageSize=100)",
            "drive.files().list(q=f\"'{folder}' in parents\", pageSize=100, media_body=1)",
            "spec_unknown_param",
        ),
        ('created["id"]', 'created["fileId"]', "spec_unknown_response_field"),
        ("/upload/drive/v3/files", "/upload/drive/v3/filez", "spec_unknown_operation"),
        ('{"uploadType": "media"}', '{"upload": "media"}', "spec_unknown_param"),
        ('{rng}:append"', '{rng}:appendRows"', "spec_unknown_operation"),
        ('["updatedRows"]', '["rowsUpdated"]', "spec_unknown_response_field"),
        ('{"requests": reqs}', '{"request": reqs}', "spec_unknown_body_field"),
        ('"maxResults": 100}', '"maxResults": "lots"}', "spec_param_value"),
        ('.json()["users"]', '.json()["members"]', "spec_unknown_response_field"),
        ('["email"]', '["mail"]', "spec_unknown_response_field"),
        ('{"personFields": "names"}', '{"fields_of_person": "names"}', "spec_unknown_param"),
        ('"pageSize": 100}', '"page_size": 100}', "spec_unknown_param"),
        (
            'personFields="names").execute()["names"]',
            'personFields="names").execute()["nom"]',
            "spec_unknown_response_field",
        ),
        ('params={"q": "label:x"}', 'params={"query": "label:x"}', "spec_unknown_param"),
        ('params={"q": "x"}', 'params={"search": "x"}', "spec_unknown_param"),
        (
            'params={"q": "trashed = false"}',
            'params={"filter": "trashed = false"}',
            "spec_unknown_param",
        ),
    ],
)
def test_every_correct_shape_still_catches_its_own_mistake(
    tmp_path: Path, store: SpecStore, old: str, new: str, kind: str
) -> None:
    assert CORRECT_GOOGLE.count(old) == 1, old
    root = _deliverable(tmp_path, CORRECT_GOOGLE.replace(old, new))
    assert _kinds(root, store) == [kind]


def test_the_most_specific_operation_wins_and_binds_every_path_parameter() -> None:
    from rote.vendor_specs.check import best_operation, match_path

    people = from_discovery(json.loads((FIXTURES / "people_v1.discovery.json").read_text()), "u")
    op, bound = best_operation(
        people.operations, "GET", "https://people.googleapis.com/v1/people/me/connections"
    )
    assert op is not None and op.id == "people.people.connections.list"
    assert bound == {"resourceName": "people/me"}
    op, bound = best_operation(
        people.operations, "PATCH", "https://people.googleapis.com/v1/people/c1:updateContact"
    )
    assert op is not None and op.id == "people.people.updateContact"
    assert bound == {"resourceName": "people/c1"}
    assert match_path(
        ["v4", "spreadsheets", "{spreadsheetId}:batchUpdate"],
        ["v4", "spreadsheets", "S:batchUpdate"],
    ) == {"spreadsheetId": "S"}
    assert match_path(["v1", "{+name}"], ["v1"]) is None
    assert match_path(["files", "{fileId}"], ["files", "a", "b"]) is None


_GMAIL_THREADS = "https://gmail.googleapis.com/gmail/v1/users/me/threads"
_DRIVE_FILE = "https://www.googleapis.com/drive/v3/files/F1"


@pytest.mark.parametrize(
    ("body", "kinds"),
    [
        # Query parameters written into the URL are checked like params=.
        (f'httpx.get("{_GMAIL_THREADS}?q=x&maxResults=lots")', ["spec_param_value"]),
        (f'httpx.get("{_GMAIL_THREADS}?query=x")', ["spec_unknown_param"]),
        (f'httpx.get("{_GMAIL_THREADS}?q=x&maxResults=5")', []),
        # A path parameter is not a query parameter.
        (f'httpx.get("{_GMAIL_THREADS}", params={{"userId": "me"}})', ["spec_unknown_param"]),
        # Booleans are booleans.
        (
            f'httpx.get("{_GMAIL_THREADS}", params={{"includeSpamTrash": "yes"}})',
            ["spec_param_value"],
        ),
        (f'httpx.get("{_GMAIL_THREADS}", params={{"includeSpamTrash": True}})', []),
        # alt=media returns the file's bytes: no JSON fields to read.
        (
            f'httpx.get("{_DRIVE_FILE}", params={{"alt": "media"}}).json()["name"]',
            ["spec_unknown_response_field"],
        ),
        (f'httpx.get("{_DRIVE_FILE}").json()["name"]', []),
        # Arguments spread from a dict the AST cannot see may hold the
        # required one: not reported missing.
        ('httpx.get("https://www.googleapis.com/drive/v3/files/F1/export", **opts)', []),
        (
            'httpx.get("https://www.googleapis.com/drive/v3/files/F1/export")',
            ["spec_missing_required"],
        ),
        ('build("drive", "v3").files().export(fileId="F1", **opts).execute()', []),
        (
            'httpx.get("https://www.googleapis.com/drive/v3/files/F1/export", '
            'params={"fields": "x", **opts})',
            [],
        ),
        ('build("drive", "v3").files().export(fileId="F1").execute()', ["spec_missing_required"]),
        # list_next's keyword form takes the previous request, not parameters.
        (
            'build("drive", "v3").files().list_next(previous_request=r, previous_response=p)',
            [],
        ),
        # A client built inline is a client.
        (
            f'httpx.Client().get("{_GMAIL_THREADS}", params={{"query": "x"}})',
            ["spec_unknown_param"],
        ),
    ],
)
def test_every_rule_has_a_case(
    tmp_path: Path, store: SpecStore, body: str, kinds: list[str]
) -> None:
    module = f"""
    import httpx
    from googleapiclient.discovery import build

    def fetch_emails(opts, r, p):
        return {body}
    """
    root = _deliverable(tmp_path, module)
    assert _kinds(root, store) == kinds


def test_a_read_through_an_or_default_is_followed(tmp_path: Path, store: SpecStore) -> None:
    module = f"""
    import httpx

    def fetch_emails():
        body = httpx.get("{_GMAIL_THREADS}").json()
        threads = body.get("threads") or []
        return [t["subject"] for t in threads]
    """
    root = _deliverable(tmp_path, module)
    assert _kinds(root, store) == ["spec_unknown_response_field"]


# ───────── review6, second round ─────────


def test_the_client_librarys_own_names_are_accepted(tmp_path: Path, store: SpecStore) -> None:
    """googleapiclient renames parameters (key2param), adds its own, and
    suffixes reserved method names; correct client code uses those names."""
    module = """
    from googleapiclient.discovery import build

    def person():
        people = build("people", "v1")
        return people.people().get(
            resourceName="people/me",
            personFields="names",
            requestMask_includeField="person.names",
            x__xgafv="2",
            trace="token:x",
        ).execute()["names"]
    """
    root = _deliverable(tmp_path, module)
    assert _kinds(root, store) == []
    # The Discovery spelling is not what the library takes.
    broken = module.replace("requestMask_includeField=", "requestMaskIncludeField=")
    assert _kinds(_deliverable(tmp_path / "b", broken), store) == ["spec_unknown_param"]


def test_client_names_follow_googleapiclient() -> None:
    from rote.vendor_specs.google import client_method, client_param

    assert client_param("requestMask.includeField") == "requestMask_includeField"
    assert client_param("$.xgafv") == "x__xgafv"
    assert client_param("max-results") == "max_results"
    assert client_param("userId") == "userId"
    assert client_method("import") == "import_"
    assert client_method("body") == "body_"
    assert client_method("list") == "list"


def test_a_reserved_method_name_resolves_to_its_operation(tmp_path: Path) -> None:
    from rote.vendor_specs.check import _client_named
    from rote.vendor_specs.model import ApiSpec, Operation

    spec = ApiSpec(
        vendor="google",
        api="gmail",
        version="v1",
        revision="r",
        source="google-discovery",
        source_url="u",
        operations=[
            Operation(
                id="gmail.users.messages.import",
                method="POST",
                url_template="https://gmail.googleapis.com/gmail/v1/users/{userId}/messages/import",
            )
        ],
    )
    op = _client_named(spec, "gmail.users.messages.import_")
    assert op is not None and op.id == "gmail.users.messages.import"
    assert _client_named(spec, "gmail.users.messages.importt") is None


def test_an_executed_chain_is_a_response_not_a_resource(tmp_path: Path, store: SpecStore) -> None:
    """`results = svc.files().list().execute()` made `results.get(...)` a bogus
    drive.files.list.execute.get operation."""
    module = """
    from googleapiclient.discovery import build

    def files():
        svc = build("drive", "v3")
        results = svc.files().list().execute()
        return [f["name"] for f in results.get("files", [])]
    """
    root = _deliverable(tmp_path, module)
    checks = check_deliverable(load_pipeline(root / "pipeline.yaml"), root, store)
    assert [(c.operation, c.status) for c in checks] == [("drive.files.list", "ok")]
    broken = module.replace('results.get("files", [])', 'results.get("filez", [])')
    assert _kinds(_deliverable(tmp_path / "b", broken), store) == ["spec_unknown_response_field"]


def test_each_read_is_checked_against_the_binding_at_its_line(
    tmp_path: Path, store: SpecStore
) -> None:
    module = """
    from googleapiclient.discovery import build
    import httpx

    def threads():
        svc = build("gmail", "v1")
        resp = svc.users().threads().list(userId="me").execute()
        ids = [t["id"] for t in resp["threads"]]
        resp = svc.users().threads().get(userId="me", id=ids[0]).execute()
        first = resp["messages"]
        resp = {"anything": 1}
        return first, resp["anything"]
    """
    root = _deliverable(tmp_path, module)
    checks = check_deliverable(load_pipeline(root / "pipeline.yaml"), root, store)
    assert [(c.operation, c.status) for c in checks] == [
        ("gmail.users.threads.list", "ok"),
        ("gmail.users.threads.get", "ok"),
    ], [f.message for c in checks for f in c.findings]
    # The list response read the get's field: caught on the right call.
    broken = module.replace('for t in resp["threads"]', 'for t in resp["messages"]')
    root = _deliverable(tmp_path / "b", broken)
    checks = check_deliverable(load_pipeline(root / "pipeline.yaml"), root, store)
    assert [(c.operation, [f.kind for f in c.findings]) for c in checks] == [
        ("gmail.users.threads.list", ["spec_unknown_response_field"]),
        ("gmail.users.threads.get", []),
    ]


@pytest.mark.parametrize(
    ("url", "status", "note"),
    [
        # A segment built at run time could be "threads" or "messages".
        (f"{_GMAIL_THREADS[: -len('/threads')]}/{{kind}}", "unresolved", "built at run time"),
        # Relative to a client's base_url.
        ("/gmail/v1/users/me/threads", "unresolved", "no host here"),
        # A placeholder does not excuse a literal no operation has.
        (f"{_GMAIL_THREADS}z/{{tid}}", "violations", None),
    ],
)
def test_a_url_rote_cannot_pin_down_is_not_checked(
    tmp_path: Path, store: SpecStore, url: str, status: str, note: str | None
) -> None:
    module = f"""
    import httpx

    def fetch(kind, tid):
        return httpx.get(f"{url}").json()
    """
    root = _deliverable(tmp_path, module)
    [check] = check_deliverable(load_pipeline(root / "pipeline.yaml"), root, store)
    assert check.status == status, check
    if note is not None:
        assert note in (check.note or "") and check.findings == []


def test_a_module_with_no_recognised_request_gets_a_not_checked_row(
    tmp_path: Path, store: SpecStore
) -> None:
    module = """
    import aiohttp

    async def fetch_emails(q):
        async with aiohttp.ClientSession() as s:
            return await (await s.get("https://gmail.googleapis.com/gmail/v1/users/me/threads")).json()
    """
    root = _deliverable(tmp_path, module)
    [check] = check_deliverable(load_pipeline(root / "pipeline.yaml"), root, store)
    assert (check.status, check.path) == ("unresolved", "extracted/vendor.py")
    assert "recognises no vendor request" in (check.note or "")


def test_the_report_never_shows_an_unchecked_request_as_ok() -> None:
    from rote.contracts import ContractFinding
    from rote.vendor_specs.check import RequestCheck, report_section

    bad = ContractFinding(
        node_id="n", kind="spec_unknown_param", severity="error", message="m1", path="p"
    )
    lines = report_section(
        [
            RequestCheck(("n",), "a.py", 1, "ok", operation="op.a", spec="s"),
            RequestCheck(("n",), "a.py", 2, "unresolved", note="the URL is built at run time"),
            RequestCheck(("n",), "a.py", 3, "spec_unavailable", note="discovery is down"),
            RequestCheck(("n",), "a.py", 4, "violations", operation="op.b", findings=[bad]),
        ]
    )
    rows = [line for line in lines if line.startswith("| `a.py")]
    assert rows == [
        "| `a.py:1` | op.a | s | ok |",
        "| `a.py:2` |  |  | not checked: the URL is built at run time |",
        "| `a.py:3` |  |  | not checked: discovery is down |",
        "| `a.py:4` | op.b |  | **1 violation(s)** |",
    ]
    assert "- m1" in lines
    assert report_section([]) == ["No vendor request was found in the external_call modules."]


@pytest.mark.parametrize(
    ("body", "kinds"),
    [
        # A URL built with +: the operation is still resolved.
        (
            'httpx.get("https://gmail.googleapis.com/gmail/v1/users/" + "me/threads")'
            '.json()["nope"]',
            ["spec_unknown_response_field"],
        ),
        # An integer parameter sent as a digit string is an integer.
        (f'httpx.get("{_GMAIL_THREADS}", params={{"maxResults": "5"}})', []),
        (f'httpx.get("{_GMAIL_THREADS}", params={{"maxResults": "5x"}})', ["spec_param_value"]),
    ],
)
def test_more_rules_each_with_a_case(
    tmp_path: Path, store: SpecStore, body: str, kinds: list[str]
) -> None:
    module = f"""
    import httpx

    def fetch():
        return {body}
    """
    assert _kinds(_deliverable(tmp_path, module), store) == kinds


def test_a_for_loop_over_a_response_checks_each_items_fields(
    tmp_path: Path, store: SpecStore
) -> None:
    module = f"""
    import httpx

    def fetch():
        out = []
        for t in httpx.get("{_GMAIL_THREADS}").json()["threads"]:
            out.append(t["subject"])
        return out
    """
    assert _kinds(_deliverable(tmp_path, module), store) == ["spec_unknown_response_field"]


def test_specs_are_cached_for_a_day_by_default(tmp_path: Path) -> None:
    from rote.vendor_specs.store import DEFAULT_TTL_SECONDS

    assert DEFAULT_TTL_SECONDS == 24 * 3600
    assert SpecStore(root=tmp_path).ttl_seconds == DEFAULT_TTL_SECONDS


def _synthetic_spec() -> Any:
    from rote.vendor_specs.model import ApiSpec, Operation, ParamSpec

    return ApiSpec(
        vendor="google",
        api="mail",
        version="v1",
        revision="r",
        source="google-discovery",
        source_url="u",
        operations=[
            Operation(
                id="mail.users.messages.import",
                method="POST",
                url_template="https://mail.googleapis.com/v1/users/{userId}/messages/import",
                params={
                    "userId": ParamSpec(name="userId", location="path", required=True),
                    "page.size": ParamSpec(
                        name="page.size", location="query", type="integer", required=True
                    ),
                },
            )
        ],
    )


def test_a_required_parameter_is_found_under_its_client_name() -> None:
    """`page.size` is sent as `page_size`; the import_ method is `import`."""
    from rote.vendor_specs.check import _operation, _violations
    from rote.vendor_specs.extract import CallSite

    spec = _synthetic_spec()
    site = CallSite(
        line=1,
        function=None,
        kind="google_client",
        google_api=("mail", "v1"),
        google_chain=("users", "messages", "import_"),
        params={"userId": "me", "page_size": 5},
    )
    op, bound = _operation(site, spec)
    assert op is not None and op.id == "mail.users.messages.import"
    assert _violations(site, op, spec, bound) == []
    missing = CallSite(**{**site.__dict__, "params": {"userId": "me"}})
    assert [k for k, _ in _violations(missing, op, spec, bound)] == ["spec_missing_required"]


@pytest.mark.parametrize(
    ("tail", "kinds"),
    [
        # Read straight off the first binding, before the name is rebound.
        ('n = resp["threads"]\n    resp = {"x": 1}\n    return n, resp["x"]', []),
        (
            'n = resp["threadz"]\n    resp = {"x": 1}\n    return n, resp["x"]',
            ["spec_unknown_response_field"],
        ),
        # Rebound to something else: reads off it are not the request's.
        ('resp = {"x": 1}\n    return resp["threadz"]', []),
    ],
)
def test_a_direct_read_resolves_against_the_binding_at_its_line(
    tmp_path: Path, store: SpecStore, tail: str, kinds: list[str]
) -> None:
    module = (
        "from googleapiclient.discovery import build\n\n"
        "def threads():\n"
        '    resp = build("gmail", "v1").users().threads().list(userId="me").execute()\n'
        f"    {tail}\n"
    )
    root = tmp_path / "d"
    (root / "extracted").mkdir(parents=True)
    (root / "pipeline.yaml").write_text(PIPELINE, encoding="utf-8")
    (root / "extracted" / "vendor.py").write_text(module, encoding="utf-8")
    assert _kinds(root, store) == kinds


def test_a_request_rebound_before_it_runs_is_not_read_as_that_request(
    tmp_path: Path, store: SpecStore
) -> None:
    module = """
    from googleapiclient.discovery import build

    def threads(other):
        req = build("gmail", "v1").users().threads().list(userId="me")
        req = other
        return req.execute()["threadz"]
    """
    assert _kinds(_deliverable(tmp_path, module), store) == []


@pytest.mark.parametrize(
    ("url", "status", "operation"),
    [
        # Same `path` (v1/{+resourceName}); flatPath tells them apart.
        (
            "https://people.googleapis.com/v1/people/me?personFields=names",
            "ok",
            "people.people.get",
        ),
        (
            "https://people.googleapis.com/v1/contactGroups/g1?groupFields=name",
            "ok",
            "people.contactGroups.get",
        ),
        (
            "https://admin.googleapis.com/admin/directory/v1/customers/C/chrome/printers/P",
            "ok",
            "admin.customers.chrome.printers.get",
        ),
        (
            "https://admin.googleapis.com/admin/directory/v1/customers/C/chrome/printServers/S",
            "ok",
            "admin.customers.chrome.printServers.get",
        ),
        # Both {+name} operations match and no flatPath does: not checked.
        (
            "https://admin.googleapis.com/admin/directory/v1/customers/C/chrome/faxes/F",
            "unresolved",
            None,
        ),
    ],
)
def test_operations_with_one_path_are_told_apart_by_flat_path(
    tmp_path: Path, store: SpecStore, url: str, status: str, operation: str | None
) -> None:
    """review6 I5: a strict `>` gave the tie to whichever came first, so GET
    /v1/people/me was checked as contactGroups.get."""
    module = f"""
    import httpx

    def fetch():
        return httpx.get("{url}").json()
    """
    root = _deliverable(tmp_path, module)
    [check] = check_deliverable(load_pipeline(root / "pipeline.yaml"), root, store)
    assert (check.status, check.operation) == (status, operation), check
    if status == "unresolved":
        assert "matches several operations equally" in (check.note or "")


def test_a_boolean_is_not_an_integer(tmp_path: Path, store: SpecStore) -> None:
    module = f"""
    import httpx

    def fetch():
        return httpx.get("{_GMAIL_THREADS}", params={{"maxResults": True}})
    """
    assert _kinds(_deliverable(tmp_path, module), store) == ["spec_param_value"]


_PAGED = """
from googleapiclient.discovery import build

def names():
    drive = build("drive", "v3")
    request = drive.files().list()
    out = []
    while request is not None:
        resp = request.execute()
        out += [f["name"] for f in resp.get("files", [])]
        request = drive.files().list_next(request, resp)
    return out

def second_page():
    drive = build("drive", "v3")
    first = drive.files().list()
    page = first.execute()
    following = drive.files().list_next(first, page)
    return following.execute()["nextPageToken"]
"""


def test_a_zero_argument_request_bound_to_a_name_is_a_request(
    tmp_path: Path, store: SpecStore
) -> None:
    """`request = drive.files().list()` reads like a resource step; its
    `.execute()` says it is a request. It was dropped, reads, pages and all."""
    sites = extract(textwrap.dedent(_PAGED))
    assert [(s.function, s.google_method_id) for s in sites] == [
        ("names", "drive.files.list"),
        ("names", "drive.files.list_next"),
        ("second_page", "drive.files.list"),
        ("second_page", "drive.files.list_next"),
    ]
    assert [p for p, _ in sites[0].response_reads] == [("files",), ("files", "[]", "name")]
    assert [p for p, _ in sites[3].response_reads] == [("nextPageToken",)]
    root = _deliverable(tmp_path, _PAGED)
    checks = check_deliverable(load_pipeline(root / "pipeline.yaml"), root, store)
    assert [(c.operation, c.status) for c in checks] == [("drive.files.list", "ok")] * 4, [
        f.message for c in checks for f in c.findings
    ]
    for old, new, kind in (
        ('f["name"]', 'f["title"]', "spec_unknown_response_field"),
        ('["nextPageToken"]', '["nextToken"]', "spec_unknown_response_field"),
        ("first = drive.files().list()", "first = drive.files().lisst()", "spec_unknown_operation"),
    ):
        broken = _PAGED.replace(old, new)
        assert _kinds(_deliverable(tmp_path / new, broken), store) == [kind]


def test_a_named_resource_is_still_a_resource() -> None:
    sites = extract(
        textwrap.dedent(
            """
            from googleapiclient.discovery import build

            def threads():
                svc = build("gmail", "v1")
                t = svc.users().threads()
                return t.list(userId="me").execute()["threads"]
            """
        )
    )
    assert [s.google_method_id for s in sites] == ["gmail.users.threads.list"]


@pytest.mark.parametrize(
    "body",
    [
        # Only run.
        'req = drive.files().list()\n    return req.execute()["files"]',
        # Only paged: a page token from elsewhere, the request never run itself.
        "req = drive.files().list()\n    return drive.files().list_next(req, previous).execute()",
    ],
)
def test_either_use_makes_a_name_a_request(body: str) -> None:
    source = (
        "from googleapiclient.discovery import build\n\n"
        f"def f(previous):\n    drive = build('drive', 'v3')\n    {body}\n"
    )
    assert extract(source)[0].google_method_id == "drive.files.list"


# ───────── Specs written from the docs ─────────

SLACK_SPEC = {
    "vendor": "slack",
    "api": "web",
    "version": "v1",
    "sources": ["https://docs.slack.dev/reference/methods/conversations.history"],
    "operations": [
        {
            "id": "conversations.history",
            "method": "GET",
            "url_template": "https://slack.com/api/conversations.history",
            "params": {
                "channel": {"location": "query", "type": "string", "required": True},
                "limit": {"location": "query", "type": "integer"},
            },
            "response_example": {
                "ok": True,
                "messages": [{"type": "message", "ts": "1.2", "text": "hi", "user": "U1"}],
                "has_more": False,
            },
        }
    ],
}

SLACK_PIPELINE = """\
name: intake
version: "0.1.0"
description: d
input: {type: In}
nodes:
  - id: read_intake
    kind: external_call
    description: Read the channel.
    impl: extracted/slack.py:read_intake
edges: []
entry_nodes: [read_intake]
exit_nodes: [read_intake]
"""

SLACK_MODULE = """
import os

import httpx


def read_intake(channel: str) -> dict:
    body = httpx.get(
        "https://slack.com/api/conversations.history",
        params={"channel": channel, "limit": 50},
        headers={"Authorization": f"Bearer {os.environ['SLACK_TOKEN']}"},
    ).json()
    return {"texts": [m["text"] for m in body.get("messages", [])]}
"""


def _slack(tmp_path: Path, module: str, spec: dict[str, Any] | None) -> Path:
    root = tmp_path / "d"
    (root / "extracted").mkdir(parents=True)
    (root / "pipeline.yaml").write_text(SLACK_PIPELINE, encoding="utf-8")
    (root / "extracted" / "slack.py").write_text(textwrap.dedent(module), encoding="utf-8")
    if spec is not None:
        (root / "vendor_specs").mkdir()
        (root / "vendor_specs" / "slack.web.v1.json").write_text(json.dumps(spec), encoding="utf-8")
    return root


@pytest.mark.parametrize(
    ("old", "new", "kinds"),
    [
        (None, None, []),
        ('"limit": 50', '"count": 50', ["spec_unknown_param"]),
        ('"limit": 50', '"limit": "fifty"', ["spec_param_value"]),
        (
            'params={"channel": channel, "limit": 50}',
            'params={"limit": 50}',
            ["spec_missing_required"],
        ),
        ('m["text"]', 'm["body"]', ["spec_unknown_response_field"]),
        ("conversations.history", "conversations.historyy", ["spec_unknown_operation"]),
    ],
)
def test_a_docs_spec_checks_requests_like_a_vendors_own(
    tmp_path: Path, store: SpecStore, old: str | None, new: str | None, kinds: list[str]
) -> None:
    module = SLACK_MODULE if old is None else SLACK_MODULE.replace(old, new or "", 1)
    root = _slack(tmp_path, module, SLACK_SPEC)
    [check] = check_deliverable(load_pipeline(root / "pipeline.yaml"), root, store)
    assert [f.kind for f in check.findings] == kinds
    assert check.spec == "slack/web/v1 (docs, lower trust)"


def test_a_host_with_no_spec_asks_for_one_from_the_docs(tmp_path: Path, store: SpecStore) -> None:
    root = _slack(tmp_path, SLACK_MODULE, None)
    [check] = check_deliverable(load_pipeline(root / "pipeline.yaml"), root, store)
    [finding] = check.findings
    assert finding.kind == "spec_missing" and "slack.com publishes no spec" in finding.message
    assert "vendor_specs/<vendor>.<api>.<version>.json" in finding.message


def test_a_docs_spec_that_does_not_parse_is_reported_by_file(
    tmp_path: Path, store: SpecStore
) -> None:
    bad = {**SLACK_SPEC, "sources": []}
    root = _slack(tmp_path, SLACK_MODULE, bad)
    checks = check_deliverable(load_pipeline(root / "pipeline.yaml"), root, store)
    kinds = [f.kind for c in checks for f in c.findings]
    assert kinds[0] == "docs_spec_invalid"
    assert "slack.web.v1.json" in checks[0].findings[0].message


@pytest.mark.parametrize(
    ("fetched", "kinds"),
    [
        (["https://docs.slack.dev/reference/methods/conversations.history"], []),
        (["https://elsewhere.example"], ["docs_spec_unfetched_source"]),
    ],
)
def test_a_docs_spec_must_come_from_pages_fetched_in_this_compile(
    tmp_path: Path, store: SpecStore, fetched: list[str], kinds: list[str]
) -> None:
    from rote.compiler.api_docs import WebResearch

    module = SLACK_MODULE.replace("def read_intake", "# docs: skill\ndef read_intake")
    root = _slack(tmp_path, module, SLACK_SPEC)
    result = verify_deliverable(
        root,
        run_tests=False,
        emit_target=EmitTarget("dbos", "api"),
        spec_store=store,
        web_research=WebResearch(fetched=fetched),
    )
    assert [f.kind for f in result.errors if f.kind.startswith(("docs_", "spec_"))] == kinds
