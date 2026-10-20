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
@pytest.mark.parametrize("bad", [False, True])
def test_an_api_pipeline_is_scored_against_the_skill_through_the_fake(
    tmp_path: Path, store: SpecStore, bad: bool
) -> None:
    """The api pipeline's Gmail read keys the same as the skill's MCP read,
    so reads score 1.0; a request the spec refuses fails the node."""
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
                        "http": {"google/gmail/v1": GMAIL_FIXTURES},
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

    report = pipeline_fidelity(compiled, example, tmp_path / "work", spec_store=store)
    case = report["cases"]["c"]
    if bad:
        assert case["failure"]["node"] == "fetch_dock_emails"
        assert "400" in (case["run_error"] or "")
    else:
        assert case["run_error"] is None
        assert case["reads"] == 1.0
        assert case["score"] == 1.0
