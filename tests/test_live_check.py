"""``rote run --live-check``: one real, read-only call per entry vendor read, shapes compared.

The "real" vendor here is a local HTTP server described by a docs spec, so
the whole path runs offline: the static refusal, the intercept's runtime
read-only guard, the recording, the comparison with the spec and with the
stored fixtures, and the promise that no value reaches the report.
"""

from __future__ import annotations

import json
import subprocess
import sys
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
from rote.vendor_specs.live import INTERCEPT_DIR, LiveCheckRefused, live_check

SECRET = "value-from-the-users-account"


class _Api:
    def __init__(self) -> None:
        self.requests: list[tuple[str, str]] = []

    @property
    def handler(self) -> type[BaseHTTPRequestHandler]:
        api = self

        class Handler(BaseHTTPRequestHandler):
            def _answer(self) -> None:
                api.requests.append((self.command, self.path))
                body = json.dumps(
                    {"items": [{"id": "a1", "name": SECRET, "extra": SECRET}], "count": 1}
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            do_GET = do_POST = _answer  # noqa: N815

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


def _compiled(tmp_path: Path, base: str, method: str = "get") -> Path:
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
            f"""
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


def test_a_live_read_is_compared_with_the_spec_and_the_fixtures_and_nothing_is_kept(
    tmp_path: Path, api: tuple[_Api, str]
) -> None:
    pytest.importorskip("httpx")
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

    report = live_check(root, {"q": "x"}, SpecStore(root=tmp_path / "specs"), fixtures)
    [check] = report.checked
    assert (check.method, check.status, check.operation) == ("GET", 200, "items.list")
    # The docs spec claimed a title the live API does not return, and the code
    # reads it: the finding that would break the pipeline.
    assert check.reads_missing == ["items[].title"]
    assert check.spec_only == ["items[].title"]
    # A docs spec lists only what the code needs, so extra live fields are
    # not drift; for a vendor's own spec they are (next test).
    assert check.schema_drift == []
    assert check.fixture_only == ["items[].legacy"]
    assert check.live_only == ["items[].extra"]
    assert report.skipped == {"summarize": "its inputs come from fetch_items"}
    assert report.drifted and not report.errors
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
        live_check(
            root, {"q": "x"}, SpecStore(root=tmp_path / "specs"), FixtureStore(tmp_path / "fx")
        )
    assert state.requests == []


def test_a_request_that_does_not_pass_the_spec_check_is_refused(
    tmp_path: Path, api: tuple[_Api, str]
) -> None:
    state, base = api
    root = _compiled(tmp_path, base)
    module = root / "extracted" / "items.py"
    module.write_text(module.read_text().replace('{"q": q}', '{"query": q}'))
    with pytest.raises(LiveCheckRefused, match="must resolve to a spec operation"):
        live_check(
            root, {"q": "x"}, SpecStore(root=tmp_path / "specs"), FixtureStore(tmp_path / "fx")
        )
    assert state.requests == []


def test_the_intercept_refuses_a_write_at_run_time(tmp_path: Path, api: tuple[_Api, str]) -> None:
    """A write the static check could not see still never leaves the process."""
    pytest.importorskip("httpx")
    state, base = api
    record = tmp_path / "rec.jsonl"
    script = (
        "import httpx\n"
        f"print(httpx.get('{base}/v1/items').status_code)\n"
        "try:\n"
        f"    httpx.post('{base}/v1/items', json={{}})\n"
        "except Exception as e:\n"
        "    print(type(e).__name__, e)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        env={
            "PATH": "/usr/bin:/bin",
            "PYTHONPATH": str(INTERCEPT_DIR),
            "ROTE_HTTP_LIVE_HOSTS": "127.0.0.1",
            "ROTE_HTTP_LIVE_RECORD": str(record),
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "NetworkBlocked the live check is read-only: refused POST 127.0.0.1" in proc.stdout
    assert state.requests == [("GET", "/v1/items")]
    [line] = record.read_text().splitlines()
    assert json.loads(line)["method"] == "GET"


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


def test_against_a_vendors_own_spec_every_unknown_live_field_is_drift(tmp_path: Path) -> None:
    from rote.vendor_specs.live import _compare
    from tests.test_vendor_specs import _Fetch

    specs = SpecStore(root=tmp_path / "specs", fetch=_Fetch())
    body = {"threads": [{"id": "t", "snippet": SECRET, "labels": ["x"]}], "resultSizeEstimate": "3"}
    rec = {
        "method": "GET",
        "url": "https://gmail.googleapis.com/gmail/v1/users/me/threads?q=x",
        "status": 200,
        "content_type": "application/json; charset=UTF-8",
        "body": json.dumps(body),
    }
    check = _compare("n", rec, specs, [], FixtureStore(tmp_path / "fx"), {})
    assert check.operation == "gmail.users.threads.list"
    assert check.schema_drift == [
        "threads[].labels: returned live, not in the spec",
        "resultSizeEstimate: expected integer, got str",
    ]
    assert "nextPageToken" in check.spec_only
    assert SECRET not in json.dumps(check.__dict__)


@pytest.mark.parametrize(
    "call",
    [
        # http.client, httpcore and urllib3 are not patched: their method was
        # never checked, so the live host does not resolve for them.
        "import http.client\n"
        "c = http.client.HTTPConnection('127.0.0.1', PORT, timeout=5)\n"
        "c.request('POST', '/v1/items', body='{}')\n",
        "import httpcore\nhttpcore.ConnectionPool().request('POST', 'BASE/v1/items')\n",
        "import urllib3\nurllib3.PoolManager(retries=False).request('GET', 'BASE/v1/items')\n",
    ],
)
def test_an_unguarded_client_cannot_reach_a_live_host(
    tmp_path: Path, api: tuple[_Api, str], call: str
) -> None:
    pytest.importorskip("httpx")
    state, base = api
    port = base.rsplit(":", 1)[1].split("/")[0]
    script = (
        "import httpx\n"
        f"print(httpx.get('{base}/v1/items').status_code)\n"
        "try:\n"
        + "".join(
            f"    {line}\n"
            for line in call.replace("PORT", port).replace("BASE", base).splitlines()
        )
        + "except Exception as e:\n"
        "    chain = []\n"
        "    while e is not None:\n"
        "        chain.append(str(e))\n"
        "        e = e.__cause__ or e.__context__\n"
        "    print('REFUSED', ' <- '.join(chain))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        env={
            "PATH": "/usr/bin:/bin",
            "PYTHONPATH": str(INTERCEPT_DIR),
            "ROTE_HTTP_LIVE_HOSTS": "127.0.0.1",
            "ROTE_HTTP_LIVE_RECORD": str(tmp_path / "rec.jsonl"),
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "REFUSED" in proc.stdout and "an HTTP client it does not guard" in proc.stdout
    # The guarded read went through; the unguarded call never arrived.
    assert state.requests == [("GET", "/v1/items")]
