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
from rote.vendor_specs.google import DIRECTORY_URL, api_for_url, from_discovery

FIXTURES = Path(__file__).parent / "fixtures" / "vendor_specs"
GMAIL_URL = "https://gmail.googleapis.com/$discovery/rest?version=v1"
DRIVE_URL = "https://www.googleapis.com/discovery/v1/apis/drive/v3/rest"


def _served() -> dict[str, Any]:
    return {
        DIRECTORY_URL: json.loads((FIXTURES / "google_directory.json").read_text()),
        GMAIL_URL: json.loads((FIXTURES / "gmail_v1.discovery.json").read_text()),
        DRIVE_URL: json.loads((FIXTURES / "drive_v3.discovery.json").read_text()),
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
        ("https://gmail.googleapis.com/gmail/v1/users/me/threads", ("gmail", "v1")),
        ("https://www.googleapis.com/drive/v3/files/{id}/export", ("drive", "v3")),
        ("https://www.googleapis.com/upload/drive/v3/files", ("drive", "v3")),
        ("https://sheets.googleapis.com/v4/spreadsheets/{id}/values/{r}", ("sheets", "v4")),
        ("https://api.hubapi.com/crm/v3/objects", None),
        ("{?}/threads", None),
    ],
)
def test_a_google_url_names_its_api_and_version(url: str, named: Any) -> None:
    assert api_for_url(url) == named


# ───────── The store ─────────


def test_the_store_caches_by_vendor_api_version_and_never_reads_stale(tmp_path: Path) -> None:
    fetch = _Fetch()
    root = tmp_path / "specs"
    assert SpecStore(root=root, fetch=fetch).google("gmail", "v1") is not None
    assert (root / "google" / "gmail" / "v1.json").is_file()
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
