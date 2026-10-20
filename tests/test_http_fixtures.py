"""HTTP fixtures for ``--backend api`` fidelity: a spec-built fake vendor API.

The specs are the trimmed real Discovery documents in tests/fixtures/
vendor_specs; nothing here reaches the network.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from rote.bench.http_fixtures import (
    INTERCEPT_DIR,
    HttpFixtureError,
    api_calls,
    credential_variables,
    load_http,
    vendor_requirements,
)
from rote.vendor_specs import SpecStore
from rote.vendor_specs.fake import FakeApi, read_log, serving
from rote.vendor_specs.fixtures import FixtureError, FixtureSet
from tests.test_vendor_specs import _Fetch

AUTH = {"Authorization": "Bearer t"}

GMAIL_FIXTURES = {
    "operations": {
        "gmail.users.threads.list": [
            {
                "when_contains": {"q": "dock-activity"},
                "body": {
                    "threads": [{"id": "t-1", "snippet": "approved", "historyId": "9"}],
                    "resultSizeEstimate": 1,
                },
                "reads": "gmail.search_threads[0]",
            }
        ],
        "gmail.users.threads.get": [
            {
                "when": {"id": "t-1"},
                "body": {"id": "t-1", "snippet": "approved", "messages": []},
                "reads": "gmail.search_threads[0]",
            }
        ],
    }
}
DRIVE_FIXTURES = {
    "operations": {
        "drive.files.export": [
            {"when": {"fileId": "F1"}, "body": "Site,Active\nEAST1,4\n", "reads": "g.read[0]"}
        ],
        "drive.files.get": [
            {"when": {"fileId": "F1", "alt": "media"}, "body": "raw bytes"},
        ],
    }
}


@pytest.fixture
def store(tmp_path: Path) -> SpecStore:
    return SpecStore(root=tmp_path / "specs", fetch=_Fetch())


def _api(
    store: SpecStore, log: Path, gmail: Any = GMAIL_FIXTURES, drive: Any = DRIVE_FIXTURES
) -> FakeApi:
    sets = load_http({"google/gmail/v1": gmail, "google/drive/v3": drive})
    gmail_spec = store.google("gmail", "v1")
    drive_spec = store.google("drive", "v3")
    assert gmail_spec is not None and drive_spec is not None
    return FakeApi(
        [(gmail_spec, sets["google/gmail/v1"]), (drive_spec, sets["google/drive/v3"])], log
    )


# ───────── The fake ─────────


@pytest.mark.parametrize(
    ("method", "target", "headers", "status", "body"),
    [
        (
            "GET",
            "/gmail.googleapis.com/gmail/v1/users/me/threads?q=label:dock-activity",
            AUTH,
            200,
            {
                "threads": [{"id": "t-1", "snippet": "approved", "historyId": "9"}],
                "resultSizeEstimate": 1,
            },
        ),
        # No fixture matched: the smallest spec-shaped answer, as Gmail sends it.
        ("GET", "/gmail.googleapis.com/gmail/v1/users/me/threads?q=other", AUTH, 200, {}),
        ("GET", "/gmail.googleapis.com/gmail/v1/users/me/threads", {}, 401, None),
        ("GET", "/gmail.googleapis.com/gmail/v1/users/me/threads?query=x", AUTH, 400, None),
        ("GET", "/gmail.googleapis.com/gmail/v1/users/me/threads?maxResults=lots", AUTH, 400, None),
        (
            "GET",
            "/gmail.googleapis.com/gmail/v1/users/me/threads/t-1?format=headers",
            AUTH,
            400,
            None,
        ),
        ("GET", "/gmail.googleapis.com/gmail/v1/users/me/threadz", AUTH, 404, None),
        ("POST", "/gmail.googleapis.com/gmail/v1/users/me/threads", AUTH, 404, None),
        ("GET", "/www.googleapis.com/drive/v3/files/F1/export", AUTH, 400, None),  # mimeType
        (
            "GET",
            "/www.googleapis.com/drive/v3/files/F1/export?mimeType=text/csv",
            AUTH,
            200,
            "Site,Active\nEAST1,4\n",
        ),
        ("GET", "/www.googleapis.com/drive/v3/files/F9/export?mimeType=text/csv", AUTH, 404, None),
        ("GET", "/www.googleapis.com/drive/v3/files/F1?alt=media", AUTH, 200, "raw bytes"),
        # A key parameter is a credential too.
        ("GET", "/gmail.googleapis.com/gmail/v1/users/me/threads?q=other&key=k", {}, 200, {}),
    ],
)
def test_the_fake_answers_like_the_api_would(
    tmp_path: Path,
    store: SpecStore,
    method: str,
    target: str,
    headers: dict[str, str],
    status: int,
    body: Any,
) -> None:
    api = _api(store, tmp_path / "log.jsonl")
    got_status, ctype, raw = api.answer(method, target, headers)
    assert got_status == status
    if body is not None:
        # A file comes back as its bytes, JSON as JSON: never one dressed as the other.
        assert ctype.startswith("text/") is isinstance(body, str)
        decoded: Any = raw.decode() if ctype.startswith("text/") else json.loads(raw)
        assert decoded == body
    else:
        assert json.loads(raw)["error"]["code"] == status
    [entry] = read_log(tmp_path / "log.jsonl")
    assert entry["status"] == status


def test_the_log_carries_what_each_request_read(tmp_path: Path, store: SpecStore) -> None:
    log = tmp_path / "log.jsonl"
    api = _api(store, log)
    api.answer("GET", "/gmail.googleapis.com/gmail/v1/users/me/threads?q=dock-activity", AUTH)
    api.answer("GET", "/gmail.googleapis.com/gmail/v1/users/me/threads/t-1", AUTH)
    api.answer("GET", "/gmail.googleapis.com/gmail/v1/users/me/threads?q=other", AUTH)
    calls = api_calls(log)
    assert [(c["server"], c["tool"], c["response"]) for c in calls] == [
        ("gmail", "search_threads", 0),
        ("gmail", "search_threads", 0),
        ("google/gmail/v1", "gmail.users.threads.list", None),
    ]
    assert calls[1]["arguments"] == {"userId": "me", "id": "t-1"}


@pytest.mark.parametrize(
    ("bad", "says"),
    [
        (
            {"gmail.users.threads.list": [{"body": {"threads": [{"id": "t", "subject": "s"}]}}]},
            "field 'subject' is not in the schema",
        ),
        (
            {"gmail.users.threads.list": [{"body": {"resultSizeEstimate": "many"}}]},
            "expected integer",
        ),
        ({"gmail.users.threads.search": [{"body": {}}]}, "not an operation"),
        (
            {"gmail.users.threads.list": [{"when": {"query": "x"}, "body": {}}]},
            "undeclared parameters",
        ),
    ],
)
def test_a_fixture_that_does_not_fit_the_spec_is_refused(
    tmp_path: Path, store: SpecStore, bad: dict[str, Any], says: str
) -> None:
    with pytest.raises(FixtureError, match=says):
        _api(store, tmp_path / "log.jsonl", gmail={"operations": bad})


def test_a_media_fixture_must_be_the_files_text(tmp_path: Path, store: SpecStore) -> None:
    drive = {"operations": {"drive.files.export": [{"when": {"fileId": "F"}, "body": {"a": 1}}]}}
    with pytest.raises(FixtureError, match="media download's body is the file's text"):
        _api(store, tmp_path / "log.jsonl", drive=drive)


def test_a_fixture_set_round_trips_as_its_shareable_json() -> None:
    fs = load_http({"google/gmail/v1": GMAIL_FIXTURES})["google/gmail/v1"]
    again = FixtureSet.model_validate_json(fs.model_dump_json())
    assert again == fs and again.spec == "google/gmail/v1" and again.trust == "spec"


# ───────── Interception, in a real interpreter ─────────


def test_a_pipeline_process_reaches_the_fake_and_nothing_else_of_the_vendor(
    tmp_path: Path, store: SpecStore
) -> None:
    pytest.importorskip("httpx")
    api = _api(store, tmp_path / "log.jsonl")
    script = textwrap.dedent(
        """
        import json, urllib.request, httpx
        out = {}
        r = httpx.get(
            "https://gmail.googleapis.com/gmail/v1/users/me/threads",
            params={"q": "label:dock-activity"}, headers={"Authorization": "Bearer x"},
        )
        out["httpx"] = [r.status_code, r.json()]
        req = urllib.request.Request(
            "https://www.googleapis.com/drive/v3/files/F1/export?mimeType=text/csv",
            headers={"Authorization": "Bearer x"},
        )
        out["urllib"] = urllib.request.urlopen(req).read().decode()
        try:
            httpx.get("https://sheets.googleapis.com/v4/spreadsheets/S")
        except Exception as e:
            out["blocked"] = type(e).__name__
        import sitecustomize
        out["other_host"] = sitecustomize.rewrite("https://api.anthropic.com/v1/messages")
        print(json.dumps(out))
        """
    )
    with serving(api) as base:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            env={
                "PATH": "/usr/bin:/bin",
                "PYTHONPATH": str(INTERCEPT_DIR),
                "ROTE_HTTP_FIXTURE_URL": base,
                "ROTE_HTTP_FIXTURE_HOSTS": ",".join(sorted(api.hosts)),
                "ROTE_HTTP_FIXTURE_BLOCK": "googleapis.com",
            },
            capture_output=True,
            text=True,
            timeout=60,
        )
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["httpx"][0] == 200 and out["httpx"][1]["resultSizeEstimate"] == 1
    assert out["urllib"] == "Site,Active\nEAST1,4\n"
    assert out["blocked"] == "NetworkBlocked"
    # A judge's inference endpoint is not a vendor in play: left alone.
    assert out["other_host"] == "https://api.anthropic.com/v1/messages"
    assert [e["operation"] for e in read_log(tmp_path / "log.jsonl")] == [
        "gmail.users.threads.list",
        "drive.files.export",
    ]


#: One client per snippet, each run in a fresh interpreter with the bench's
#: interception on. A snippet assigns ``out``: the answer it got, or the
#: text of the error that refused it.
_THREADS = "https://gmail.googleapis.com/gmail/v1/users/me/threads?q=label:dock-activity"
_FAKED = {
    "httpx": f"""
import httpx
out = httpx.get("{_THREADS}", headers={{"Authorization": "Bearer x"}}).json()
""",
    "httpx-async": f"""
import asyncio, httpx
async def go():
    async with httpx.AsyncClient() as c:
        return (await c.get("{_THREADS}", headers={{"Authorization": "Bearer x"}})).json()
out = asyncio.run(go())
""",
    "requests": f"""
import requests
out = requests.get("{_THREADS}", headers={{"Authorization": "Bearer x"}}).json()
""",
    "httplib2": f"""
import json, httplib2
_, body = httplib2.Http().request("{_THREADS}", headers={{"Authorization": "Bearer x"}})
out = json.loads(body)
""",
    "urllib": f"""
import json, urllib.request
req = urllib.request.Request("{_THREADS}", headers={{"Authorization": "Bearer x"}})
out = json.loads(urllib.request.urlopen(req).read())
""",
    # Google's own client: google.auth.default -> AuthorizedHttp -> httplib2.
    "googleapiclient": """
import google.auth
from googleapiclient.discovery import build
creds, _ = google.auth.default()
gmail = build("gmail", "v1", credentials=creds, static_discovery=True, cache_discovery=False)
out = gmail.users().threads().list(userId="me", q="label:dock-activity").execute()
""",
}
#: Clients the module does not patch: refused when they resolve the name.
_UNPATCHED = {
    "http.client": """
import http.client
conn = http.client.HTTPSConnection("gmail.googleapis.com", timeout=5)
conn.request("GET", "/gmail/v1/users/me/threads")
""",
    "httpcore": f"""
import httpcore
httpcore.ConnectionPool().request("GET", "{_THREADS}")
""",
    "urllib3": f"""
import urllib3
urllib3.PoolManager(retries=False).request("GET", "{_THREADS}")
""",
}


def _intercepted(api: FakeApi, snippet: str) -> Any:
    script = (
        "import json, traceback\n"
        "try:\n"
        + textwrap.indent(textwrap.dedent(snippet).strip(), "    ")
        + "\nexcept Exception as e:\n"
        "    chain = []\n"
        "    while e is not None:\n"
        "        chain.append(f'{type(e).__name__}: {e}')\n"
        "        e = e.__cause__ or e.__context__\n"
        "    out = {'refused': ' <- '.join(chain)}\n"
        "print(json.dumps(out))\n"
    )
    with serving(api) as base:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            env={
                "PATH": "/usr/bin:/bin",
                "PYTHONPATH": str(INTERCEPT_DIR),
                "ROTE_HTTP_FIXTURE_URL": base,
                "ROTE_HTTP_FIXTURE_HOSTS": ",".join(sorted(api.hosts)),
                "ROTE_HTTP_FIXTURE_BLOCK": "googleapis.com",
            },
            capture_output=True,
            text=True,
            timeout=60,
        )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("client", sorted(_FAKED))
def test_each_patched_client_is_answered_by_the_fake(
    tmp_path: Path, store: SpecStore, client: str
) -> None:
    api = _api(store, tmp_path / "log.jsonl")
    out = _intercepted(api, _FAKED[client])
    assert out.get("resultSizeEstimate") == 1, out
    assert [e["operation"] for e in read_log(tmp_path / "log.jsonl")] == [
        "gmail.users.threads.list"
    ]


@pytest.mark.parametrize("client", sorted(_UNPATCHED))
def test_an_unpatched_client_is_refused_before_it_reaches_the_vendor(
    tmp_path: Path, store: SpecStore, client: str
) -> None:
    """Without the name-resolution guard each of these opened a connection
    to the real gmail.googleapis.com."""
    api = _api(store, tmp_path / "log.jsonl")
    out = _intercepted(api, _UNPATCHED[client])
    assert "the fidelity bench blocks gmail.googleapis.com" in out.get("refused", ""), out
    assert "reached name resolution" in out["refused"]
    assert read_log(tmp_path / "log.jsonl") == []


def test_name_resolution_is_left_alone_for_a_host_not_in_play(
    tmp_path: Path, store: SpecStore
) -> None:
    api = _api(store, tmp_path / "log.jsonl")
    out = _intercepted(
        api,
        """
import socket
out = {"resolved": bool(socket.getaddrinfo("localhost", 80))}
""",
    )
    assert out == {"resolved": True}


# ───────── The app's environment ─────────


def _app(tmp_path: Path, source: str) -> Path:
    app = tmp_path / "app"
    (app / "extracted").mkdir(parents=True)
    (app / "extracted" / "vendor.py").write_text(textwrap.dedent(source), encoding="utf-8")
    return app


def test_credential_variables_are_the_ones_the_modules_read(tmp_path: Path) -> None:
    app = _app(
        tmp_path,
        """
        import os
        a = os.environ.get("GMAIL_TOKEN")
        b = os.environ["DRIVE_KEY"]
        c = os.getenv('SHEETS_TOKEN')
        d = os.environ.get("ROTE_MCP_CONFIG")
        """,
    )
    assert credential_variables(app) == {"GMAIL_TOKEN", "DRIVE_KEY", "SHEETS_TOKEN"}


def test_vendor_requirements_name_known_distributions_and_refuse_guesses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "rote.bench.http_fixtures._importable", lambda name: name.split(".")[0] in ("httpx", "json")
    )
    app = _app(
        tmp_path,
        """
        import json, httpx
        from googleapiclient.discovery import build
        import google.auth
        """,
    )
    assert vendor_requirements(app) == ["google-api-python-client", "google-auth"]
    (app / "extracted" / "more.py").write_text("import hubspot_sdk\n", encoding="utf-8")
    with pytest.raises(HttpFixtureError, match="hubspot_sdk"):
        vendor_requirements(app)


# ───────── Fidelity, end to end ─────────


@pytest.mark.slow
@pytest.mark.parametrize("variant", ["fits", "bad_request", "bad_fixture"])
def test_an_api_pipeline_is_scored_against_the_skill_through_the_fake(
    tmp_path: Path, store: SpecStore, variant: str
) -> None:
    """The api pipeline's Gmail read keys the same as the skill's MCP read,
    so reads score 1.0; a request the spec refuses fails the node. A fixture
    the spec refuses is the bench's own fault: it stops the bench instead of
    scoring the pipeline 0.0 (it was a ValueError, scored as a dead run)."""
    bad = variant == "bad_request"
    pytest.importorskip("dbos")
    pytest.importorskip("httpx")
    import yaml

    from rote.bench.fidelity import pipeline_fidelity
    from rote.compiler.backend import write_backend

    example = tmp_path / "example"
    (example / "fidelity").mkdir(parents=True)
    (example / "fidelity" / "cases.yaml").write_text(
        yaml.safe_dump(
            {
                "output_fields": {"thread_ids": "ids of the dock threads"},
                "servers": {},
                "cases": [
                    {
                        "id": "c",
                        "input": {"q": "label:dock-activity"},
                        "fixtures": {},
                        "http": {
                            "google/gmail/v1": GMAIL_FIXTURES
                            if variant != "bad_fixture"
                            else {"operations": {"gmail.users.threads.search": [{"body": {}}]}}
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    read = {
        "server": "gmail",
        "tool": "search_threads",
        "arguments": {},
        "side_effect": False,
        "response": 0,
    }
    run = {"output": {"thread_ids": ["t-1"], "outcome": "completed"}, "calls": [read]}
    (example / "fidelity" / "reference.json").write_text(
        json.dumps({"cases": {"c": {"runs": [run, run]}}}), encoding="utf-8"
    )
    compiled = tmp_path / "compiled"
    (compiled / "extracted").mkdir(parents=True)
    param = "query" if bad else "q"
    (compiled / "extracted" / "gmail.py").write_text(
        textwrap.dedent(
            f"""
            import os

            import httpx


            def search(q):
                token = os.environ["GMAIL_TOKEN"]
                body = httpx.get(
                    "https://gmail.googleapis.com/gmail/v1/users/me/threads",
                    params={{"{param}": q}},
                    headers={{"Authorization": f"Bearer {{token}}"}},
                ).raise_for_status().json()
                return {{"thread_ids": [t["id"] for t in body.get("threads", [])]}}
            """
        ),
        encoding="utf-8",
    )
    (compiled / "pipeline.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "dock",
                "input": {"type": "In", "required": ["q"]},
                "nodes": [
                    {
                        "id": "fetch_dock_emails",
                        "kind": "external_call",
                        "description": "search the dock threads",
                        "impl": "extracted/gmail.py:search",
                        "inputs": {"q": "pipeline.input.q"},
                        "mcp": {"server": "gmail", "tool": "search_threads"},
                        "retry": {"max": 0},
                    }
                ],
                "edges": [],
            }
        ),
        encoding="utf-8",
    )
    write_backend(compiled, "api")

    if variant == "bad_fixture":
        with pytest.raises(HttpFixtureError, match="not an operation"):
            pipeline_fidelity(compiled, example, tmp_path / "work", spec_store=store)
        return
    report = pipeline_fidelity(compiled, example, tmp_path / "work", spec_store=store)
    case = report["cases"]["c"]
    if bad:
        assert case["failure"]["node"] == "fetch_dock_emails"
        assert "400" in (case["run_error"] or "")
    else:
        assert case["run_error"] is None
        assert case["reads"] == 1.0
        assert case["score"] == 1.0
