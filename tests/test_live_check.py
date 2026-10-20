"""``rote run --live-check``: one real, read-only call per entry vendor read, shapes compared.

The "real" vendor here is a local HTTP server described by a docs spec, so
the whole path runs offline. The live check never runs the deliverable's
code: it builds each read itself from the static extraction and sends only
that. These tests hold it to that against review6's reproducers (code that
also writes through urllib, through a helper module, to a host alias), and
to the promise that no value, credential or live key reaches the report.
"""

from __future__ import annotations

import json
import textwrap
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from rote.compiler.backend import write_backend
from rote.vendor_specs import SpecStore
from rote.vendor_specs.fixture_store import FixtureStore
from rote.vendor_specs.fixtures import FixtureSet
from rote.vendor_specs.live import LiveCheckRefused, LiveRequest, live_check

SECRET = "value-from-the-users-account"


class _Api:
    def __init__(self) -> None:
        self.requests: list[tuple[str, str]] = []
        self.headers: list[dict[str, str]] = []

    @property
    def handler(self) -> type[BaseHTTPRequestHandler]:
        api = self

        class Handler(BaseHTTPRequestHandler):
            def _answer(self) -> None:
                api.requests.append((self.command, self.path))
                api.headers.append(dict(self.headers))
                if self.path.startswith("/v1/moved"):
                    self.send_response(302)
                    self.send_header("Location", "/v1/items/purge")
                    self.end_headers()
                    return
                body = json.dumps(
                    {"items": [{"id": "a1", "name": SECRET, "extra": SECRET}], "count": 1}
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            do_GET = do_POST = do_DELETE = _answer  # noqa: N815

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

        return Handler


@pytest.fixture
def api() -> Iterator[tuple[_Api, str]]:
    state = _Api()
    server = ThreadingHTTPServer(("127.0.0.1", 0), state.handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield state, f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def _compiled(tmp_path: Path, base: str, method: str = "get", module: str | None = None) -> Path:
    root = tmp_path / "compiled"
    (root / "extracted").mkdir(parents=True)
    (root / "vendor_specs").mkdir()
    (root / "pipeline.yaml").write_text(
        textwrap.dedent(
            """\
            name: items
            version: "0.1.0"
            description: d
            input: {type: In}
            nodes:
              - id: fetch_items
                kind: external_call
                description: Fetch items.
                impl: extracted/items.py:fetch_items
                inputs: {q: pipeline.input.q}
              - id: summarize
                kind: external_call
                description: Uses an upstream result.
                impl: extracted/items.py:summarize
                inputs: {items: fetch_items.output}
            edges: [{from: fetch_items, to: summarize}]
            entry_nodes: [fetch_items]
            exit_nodes: [summarize]
            """
        ),
        encoding="utf-8",
    )
    (root / "extracted" / "items.py").write_text(
        textwrap.dedent(
            module.replace("BASEURL", base)
            if module is not None
            else f"""
            import httpx

            BASE = "{base}/v1/items"


            def fetch_items(q):
                body = httpx.{method}(BASE, params={{"q": q}}).json()
                return {{"names": [i["name"] for i in body.get("items", [])],
                        "titles": [i.get("title") for i in body.get("items", [])]}}


            def summarize(items):
                return {{"n": len(items["names"])}}
            """
        ),
        encoding="utf-8",
    )
    (root / "vendor_specs" / "local.items.v1.json").write_text(
        json.dumps(
            {
                "vendor": "local",
                "api": "items",
                "version": "v1",
                "sources": ["skill"],
                "operations": [
                    {
                        "id": "items.list",
                        "method": method.upper(),
                        "url_template": f"{base}/v1/items",
                        "params": {"q": {"location": "query", "type": "string"}},
                        "response_example": {
                            "items": [{"id": "x", "name": "y", "title": "t"}],
                            "count": 0,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    write_backend(root, "api")
    return root


def _check(tmp_path: Path, root: Path, **kwargs: Any) -> Any:
    return live_check(
        root,
        kwargs.pop("payload", {"q": "x"}),
        SpecStore(root=tmp_path / "specs"),
        FixtureStore(tmp_path / "fx"),
        env=kwargs.pop("env", {}),
        **kwargs,
    )


def test_a_live_read_is_compared_with_the_spec_and_the_fixtures_and_nothing_is_kept(
    tmp_path: Path, api: tuple[_Api, str]
) -> None:
    state, base = api
    root = _compiled(tmp_path, base)
    fixtures = FixtureStore(tmp_path / "fx")
    from rote.vendor_specs.docs import load

    [(_, spec)] = load(root)
    # The docs spec has no "legacy", so a set with it cannot be saved. Save one
    # that fits, then add the stale field by hand, as an out-of-date set has.
    fits = FixtureSet(
        spec="local/items/v1",
        trust="docs",
        operations={"items.list": [{"body": {"items": [{"id": "x", "name": "y"}], "count": 1}}]},  # type: ignore[list-item]
    )
    path = fixtures.save(fits, spec, name="old", origin="t")
    raw = json.loads(path.read_text())
    raw["operations"]["items.list"][0]["body"]["items"][0]["legacy"] = 1
    path.write_text(json.dumps(raw))

    report = live_check(root, {"q": "x"}, SpecStore(root=tmp_path / "specs"), fixtures, env={})
    [check] = report.checked
    assert (check.method, check.status, check.operation) == ("GET", 200, "items.list")
    # The operation's path template, not the concrete request path.
    assert check.path == "/v1/items"
    # The docs spec claimed a title the live API does not return, and the code
    # reads it: the finding that would break the pipeline.
    assert check.reads_missing == ["items[].title"]
    assert check.spec_only == ["items[].title"]
    # A docs spec lists only what the code needs, so extra live fields are
    # not drift; for a vendor's own spec they are (below).
    assert check.schema_drift == []
    assert check.fixture_only == ["items[].legacy"]
    assert check.live_only == ["items[].extra"]
    assert report.skipped == {"summarize": "its inputs come from fetch_items"}
    assert report.drifted and not report.errors and not report.unchecked
    assert SECRET not in json.dumps(report.as_dict())
    assert state.requests == [("GET", "/v1/items?q=x")]


def test_a_deliverable_that_writes_is_refused_before_any_call(
    tmp_path: Path, api: tuple[_Api, str]
) -> None:
    state, base = api
    root = _compiled(tmp_path, base, method="post")
    with pytest.raises(
        LiveCheckRefused, match="read-only, and this deliverable writes: items.list"
    ):
        _check(tmp_path, root)
    assert state.requests == []


def test_a_request_that_does_not_pass_the_spec_check_is_refused(
    tmp_path: Path, api: tuple[_Api, str]
) -> None:
    state, base = api
    root = _compiled(tmp_path, base)
    module = root / "extracted" / "items.py"
    module.write_text(module.read_text().replace('{"q": q}', '{"query": q}'))
    with pytest.raises(LiveCheckRefused, match="must resolve to a spec operation"):
        _check(tmp_path, root)
    assert state.requests == []


# ───────── The deliverable's code never runs (review6's reproducers) ─────────


@pytest.mark.parametrize(
    ("name", "module", "helper"),
    [
        (
            "a checked GET, then an unextracted urllib POST to the same host",
            """
            import httpx, urllib.request
            BASE = "BASEURL/v1/items"
            def fetch_items(q):
                httpx.get(BASE, params={"q": q}).json()
                urllib.request.urlopen("BASEURL/v1/items/delete_all", data=b"{}")
                return {}
            def summarize(items):
                return {}
            """,
            None,
        ),
        (
            "a helper module writing through requests to a host alias",
            """
            import httpx
            BASE = "BASEURL/v1/items"
            def fetch_items(q):
                httpx.get(BASE, params={"q": q}).json()
                from extracted._h import write; write()
                return {}
            def summarize(items):
                return {}
            """,
            "import requests\ndef write():\n    requests.post('ALIAS/v1/items/purge')\n",
        ),
        (
            "an http.client DELETE alongside the read",
            """
            import http.client, httpx
            BASE = "BASEURL/v1/items"
            def fetch_items(q):
                httpx.get(BASE, params={"q": q}).json()
                c = http.client.HTTPConnection("127.0.0.1", PORT)
                c.request("DELETE", "/v1/items/all")
                return {}
            def summarize(items):
                return {}
            """,
            None,
        ),
    ],
)
def test_only_the_statically_built_read_is_ever_sent(
    tmp_path: Path, api: tuple[_Api, str], name: str, module: str, helper: str | None
) -> None:
    """Each of these sent a POST or DELETE to the live host when live mode ran
    the deliverable's code (three real live_check() runs did)."""
    state, base = api
    port = base.rsplit(":", 1)[1]
    root = _compiled(tmp_path, base, module=module.replace("PORT", port))
    if helper is not None:
        alias = base.replace("127.0.0.1", "localhost")
        (root / "extracted" / "_h.py").write_text(helper.replace("ALIAS", alias))
    report = _check(tmp_path, root)
    assert state.requests == [("GET", "/v1/items?q=x")], name
    assert [c.operation for c in report.checked] == ["items.list"]


def test_a_module_whose_requests_the_check_cannot_see_sends_nothing(
    tmp_path: Path, api: tuple[_Api, str]
) -> None:
    state, base = api
    module = """
    import urllib.request
    def fetch_items(q):
        urllib.request.urlopen("BASEURL/v1/messages/send", data=b"{}")
        return {}
    def summarize(items):
        return {}
    """
    root = _compiled(tmp_path, base, module=module)
    # The spec check lists the module as not checked, so the live check
    # refuses before anything is built.
    with pytest.raises(LiveCheckRefused, match="recognises no vendor request"):
        _check(tmp_path, root)
    assert state.requests == []


def test_a_node_whose_request_is_in_a_helper_function_sends_nothing(
    tmp_path: Path, api: tuple[_Api, str]
) -> None:
    state, base = api
    module = """
    import httpx
    BASE = "BASEURL/v1/items"
    def _get(q):
        return httpx.get(BASE, params={"q": q}).json()
    def fetch_items(q):
        return _get(q)
    def summarize(items):
        return {}
    """
    root = _compiled(tmp_path, base, module=module)
    report = _check(tmp_path, root)
    assert state.requests == []
    assert report.checked == [] and "nothing was sent" in report.unchecked["fetch_items"]


@pytest.mark.parametrize(
    ("change", "why"),
    [
        # A value computed at run time cannot be sent as the code would.
        ('params={"q": q}', 'params={"q": q.upper()}'),
        # A method-override header turns the GET into a write at the server.
        ('params={"q": q}', 'params={"q": q}, headers={"X-HTTP-Method-Override": "DELETE"}'),
        # An auth object the check cannot rebuild.
        ('params={"q": q}', 'params={"q": q}, auth=("u", "p")'),
        # A credential from the environment that is not set.
        (
            'params={"q": q}',
            'params={"q": q}, headers={"Authorization": os.environ["ITEMS_TOKEN"]}',
        ),
        # Parameters spread from a dict the AST cannot see.
        ('params={"q": q}', 'params={"q": q}, **EXTRA'),
    ],
)
def test_a_read_the_check_cannot_build_exactly_is_not_sent(
    tmp_path: Path, api: tuple[_Api, str], change: str, why: str
) -> None:
    state, base = api
    root = _compiled(tmp_path, base)
    module = root / "extracted" / "items.py"
    module.write_text("import os\nEXTRA = {}\n" + module.read_text().replace(change, why))
    report = _check(tmp_path, root)
    assert state.requests == []
    [(where, reason)] = report.unchecked.items()
    assert where.startswith("fetch_items:extracted/items.py:") and reason


def test_the_users_credential_is_sent_from_the_environment_the_code_names(
    tmp_path: Path, api: tuple[_Api, str]
) -> None:
    state, base = api
    module = """
    import os
    import httpx

    BASE = "BASEURL/v1/items"


    def _headers():
        return {"Authorization": f"Bearer {os.environ['ITEMS_TOKEN']}"}


    def fetch_items(q):
        return httpx.get(BASE, params={"q": q}, headers=_headers()).json()


    def summarize(items):
        return {}
    """
    root = _compiled(tmp_path, base, module=module)
    report = _check(tmp_path, root, env={"ITEMS_TOKEN": "tok-123"})
    assert state.requests == [("GET", "/v1/items?q=x")]
    assert state.headers[0]["Authorization"] == "Bearer tok-123"
    assert "tok-123" not in json.dumps(report.as_dict())


def test_a_redirect_is_not_followed(tmp_path: Path, api: tuple[_Api, str]) -> None:
    state, base = api
    root = _compiled(tmp_path, base)
    from rote.vendor_specs.live import _send

    status, _, _ = _send(LiveRequest("GET", f"{base}/v1/moved", {}))
    assert status == 302
    assert state.requests == [("GET", "/v1/moved")]
    with pytest.raises(LiveCheckRefused, match="refusing to send POST"):
        _send(LiveRequest("POST", f"{base}/v1/items", {}))
    assert root.is_dir()


def test_an_error_reports_its_type_never_its_text(tmp_path: Path, api: tuple[_Api, str]) -> None:
    """The text of a failed request carries its URL, and so any query credential."""
    _, base = api
    module = """
    import os
    import httpx
    BASE = "BASEURL/v1/items"
    def fetch_items(q):
        return httpx.get(BASE, params={"q": q, "key": os.environ["ITEMS_KEY"]}).json()
    def summarize(items):
        return {}
    """
    root = _compiled(tmp_path, base, module=module)
    spec_path = root / "vendor_specs" / "local.items.v1.json"
    raw = json.loads(spec_path.read_text())
    raw["operations"][0]["params"]["key"] = {"location": "query", "type": "string"}
    spec_path.write_text(json.dumps(raw))

    def failing(request: LiveRequest) -> Any:
        raise OSError(f"connection reset while fetching {request.url}")

    report = _check(tmp_path, root, env={"ITEMS_KEY": "AIzaSECRET-API-KEY"}, send=failing)
    assert report.errors == {"fetch_items:extracted/items.py:6": "OSError"}
    assert "AIzaSECRET" not in json.dumps(report.as_dict())


def test_rote_run_live_check_wants_an_api_compile_and_an_input(
    tmp_path: Path, api: tuple[_Api, str], capsys: Any
) -> None:
    from rote.cli import main

    _, base = api
    root = _compiled(tmp_path, base)
    assert main(["run", str(root.parent), "--live-check"]) == 2
    assert "needs --input" in capsys.readouterr().err
    write_backend(root, "mcp")
    assert main(["run", str(root), "--live-check", "--input", '{"q": "x"}']) == 2
    assert "was not compiled for it" in capsys.readouterr().err


# ───────── Google, by the client library's own request ─────────


def test_a_google_client_call_is_sent_as_the_library_would_send_it(tmp_path: Path) -> None:
    from tests.test_vendor_specs import _Fetch

    root = tmp_path / "compiled"
    (root / "extracted").mkdir(parents=True)
    (root / "pipeline.yaml").write_text(
        textwrap.dedent(
            """\
            name: drive
            version: "0.1.0"
            description: d
            input: {type: In}
            nodes:
              - id: fetch_file
                kind: external_call
                description: Get a file's metadata.
                impl: extracted/drive.py:fetch_file
                inputs: {file_id: pipeline.input.file_id}
            edges: []
            entry_nodes: [fetch_file]
            exit_nodes: [fetch_file]
            """
        )
    )
    (root / "extracted" / "drive.py").write_text(
        textwrap.dedent(
            """
            from googleapiclient.discovery import build

            def fetch_file(file_id):
                f = build("drive", "v3").files().get(fileId=file_id, fields="id,name").execute()
                return {"name": f["name"]}
            """
        )
    )
    write_backend(root, "api")
    sent: list[LiveRequest] = []

    def capture(request: LiveRequest) -> Any:
        sent.append(request)
        return 200, "application/json", json.dumps({"id": "F", "name": SECRET}).encode()

    report = live_check(
        root,
        {"file_id": "F/1"},
        SpecStore(root=tmp_path / "specs", fetch=_Fetch()),
        FixtureStore(tmp_path / "fx"),
        env={"GOOGLE_OAUTH_ACCESS_TOKEN": "ya29.token"},
        send=capture,
    )
    assert sent == [
        LiveRequest(
            "GET",
            "https://www.googleapis.com/drive/v3/files/F%2F1?fields=id%2Cname",
            {"Authorization": "Bearer ya29.token"},
        )
    ]
    [check] = report.checked
    assert (check.operation, check.path) == ("drive.files.get", "/drive/v3/files/{fileId}")
    assert SECRET not in json.dumps(report.as_dict())


def test_live_keys_never_reach_the_report(tmp_path: Path) -> None:
    """A map's keys are user data (Drive appProperties); so is a live field
    name that is not one (an email the spec does not know)."""
    from rote.vendor_specs.live import _compare
    from tests.test_vendor_specs import _Fetch

    specs = SpecStore(root=tmp_path / "specs", fetch=_Fetch())
    drive = specs.google("drive", "v3")
    gmail = specs.google("gmail", "v1")
    assert drive is not None and gmail is not None
    get = drive.operation("drive.files.get")
    body = {"id": "1", "appProperties": {"alice@example.com": "x"}}
    assert get is not None
    # A stored fixture for files.get, so the live paths are reported.
    store = FixtureStore(tmp_path / "fx")
    store.save(
        FixtureSet(spec="google/drive/v3", operations={"drive.files.get": [{"body": {"id": "2"}}]}),  # type: ignore[list-item]
        drive,
        name="f",
        origin="t",
    )
    check = _compare("n", drive, get, 200, "application/json", json.dumps(body).encode(), store)
    assert check.live_only == ["appProperties", "appProperties.{key}"]
    listing = gmail.operation("gmail.users.threads.list")
    assert listing is not None
    threads = {"threads": [{"id": "t", "alice.private@example.com": 1}]}
    check2 = _compare(
        "n",
        gmail,
        listing,
        200,
        "application/json",
        json.dumps(threads).encode(),
        FixtureStore(tmp_path / "fx"),
    )
    text = json.dumps([check.__dict__, check2.__dict__])
    assert "alice" not in text
    assert "threads[].{key}: returned live, not in the spec" in check2.schema_drift


def test_against_a_vendors_own_spec_every_unknown_live_field_is_drift(tmp_path: Path) -> None:
    from rote.vendor_specs.live import _compare
    from tests.test_vendor_specs import _Fetch

    specs = SpecStore(root=tmp_path / "specs", fetch=_Fetch())
    gmail = specs.google("gmail", "v1")
    assert gmail is not None
    op = gmail.operation("gmail.users.threads.list")
    assert op is not None
    body = {"threads": [{"id": "t", "snippet": SECRET, "labels": ["x"]}], "resultSizeEstimate": "3"}
    check = _compare(
        "n",
        gmail,
        op,
        200,
        "application/json; charset=UTF-8",
        json.dumps(body).encode(),
        FixtureStore(tmp_path / "fx"),
        {},
    )
    assert check.operation == "gmail.users.threads.list"
    assert check.schema_drift == [
        "threads[].labels: returned live, not in the spec",
        "resultSizeEstimate: expected integer, got str",
    ]
    assert "nextPageToken" in check.spec_only
    assert SECRET not in json.dumps(check.__dict__)


def test_a_docs_operation_named_like_a_write_is_refused_whatever_its_method(
    tmp_path: Path, api: tuple[_Api, str]
) -> None:
    """A docs spec's method is whatever the agent wrote: GET messages.send is a write."""
    state, base = api
    root = _compiled(tmp_path, base)
    spec_path = root / "vendor_specs" / "local.items.v1.json"
    raw = json.loads(spec_path.read_text())
    raw["operations"][0]["id"] = "messages.send"
    spec_path.write_text(json.dumps(raw))
    with pytest.raises(LiveCheckRefused, match="this deliverable writes: messages.send"):
        _check(tmp_path, root)
    assert state.requests == []


def test_a_non_scalar_input_is_not_sent(tmp_path: Path, api: tuple[_Api, str]) -> None:
    state, base = api
    root = _compiled(tmp_path, base)
    report = _check(tmp_path, root, payload={"q": {"nested": 1}})
    assert state.requests == [] and len(report.unchecked) == 1


def test_a_path_value_is_quoted_as_one_segment(tmp_path: Path, api: tuple[_Api, str]) -> None:
    state, base = api
    module = """
    import httpx
    def fetch_items(q):
        return httpx.get(f"BASEURL/v1/items/{q}").json()
    def summarize(items):
        return {}
    """
    root = _compiled(tmp_path, base, module=module)
    spec_path = root / "vendor_specs" / "local.items.v1.json"
    raw = json.loads(spec_path.read_text())
    raw["operations"][0].update(
        {
            "id": "items.get",
            "url_template": f"{base}/v1/items/{{q}}",
            "params": {"q": {"location": "path", "type": "string", "required": True}},
        }
    )
    spec_path.write_text(json.dumps(raw))
    _check(tmp_path, root, payload={"q": "a/b"})
    assert state.requests == [("GET", "/v1/items/a%2Fb")]
