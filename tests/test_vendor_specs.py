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
from rote.vendor_specs.extract import UNKNOWN, extract
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
    assert emails.params == {"q": UNKNOWN, "maxResults": 500}
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
    assert [(c.status, c.findings) for c in checks] == [
        ("unresolved", []),
        ("spec_unavailable", []),
    ]
    assert "api.hubapi.com" in (checks[0].note or "")


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
