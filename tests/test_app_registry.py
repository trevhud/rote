"""App registry + parked-workflow release (the CLI side of park-on-auth).

Hermetic: the autouse conftest fixture points ROTE_APPS_PATH at a
per-test tmp file, and the DBOS client is faked via sys.modules — the
real cross-process park→release loop runs in tests/test_mcp_park_e2e.py
(slow).
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import pytest

from rote.app_registry import apps_path, record_app, registered_apps
from rote.cli import main as cli_main
from rote.mcp.release import ReleaseUnavailable, release_parked_workflows

# ───────── Registry ─────────


def test_record_and_list_round_trip(tmp_path: Path) -> None:
    record_app(tmp_path / "app-a", "dbos", "demo_a")
    record_app(tmp_path / "app-b", "temporal", "demo_b")
    apps = registered_apps()
    assert [(a.runtime, a.pipeline) for a in apps] == [("dbos", "demo_a"), ("temporal", "demo_b")]
    assert all(a.path.is_absolute() for a in apps)


def test_record_dedupes_by_resolved_path(tmp_path: Path) -> None:
    record_app(tmp_path / "app", "dbos", "old_name")
    record_app(tmp_path / "app", "dbos", "new_name")
    apps = registered_apps()
    assert len(apps) == 1
    assert apps[0].pipeline == "new_name"


def test_apps_path_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    override = tmp_path / "elsewhere" / "apps.json"
    monkeypatch.setenv("ROTE_APPS_PATH", str(override))
    assert apps_path() == override
    record_app(tmp_path / "app", "dbos", "demo")
    assert override.is_file()


def test_cli_emit_records_the_app(tmp_path: Path) -> None:
    """`rote emit` registers its output dir so `rote mcp login` can later
    discover parked workflows without being told where apps live."""
    yaml_path = tmp_path / "pipeline.yaml"
    yaml_path.write_text(
        """\
name: registry_demo
version: "0.1.0"
description: App-registry recording test.
input:
  type: Req
  required: [x]
nodes:
  - id: step_one
    kind: pure_function
    description: One step.
    input:
      x: str
    inputs:
      x: pipeline.input.x
    output: dict
    impl: extracted/one.py:one
edges: []
entry_nodes: [step_one]
exit_nodes: [step_one]
"""
    )
    out = tmp_path / "out"
    assert cli_main(["emit", str(yaml_path), "--runtime", "dbos", "--out", str(out)]) == 0
    apps = registered_apps()
    assert [(a.path, a.runtime, a.pipeline) for a in apps] == [
        (out.resolve(), "dbos", "registry_demo")
    ]


# ───────── Release scan ─────────


@dataclass
class _FakeWorkflowStatus:
    workflow_id: str


@dataclass
class _FakeDBOSClient:
    """Stands in for dbos.DBOSClient: two PENDING workflows, one parked
    awaiting 'vendor', one busy computing (no rote_auth_status event)."""

    system_database_url: str
    sent: list[tuple[str, Any, str]] = field(default_factory=list)
    destroyed: bool = False

    created: ClassVar[list[_FakeDBOSClient]] = []

    def __post_init__(self) -> None:
        _FakeDBOSClient.created.append(self)

    def list_workflows(
        self, *, status: str, load_input: bool = True, load_output: bool = True
    ) -> list[_FakeWorkflowStatus]:
        assert status == "PENDING"
        # The release path must not deserialize inputs/outputs — a TS
        # app's superjson values are unreadable from Python.
        assert load_input is False and load_output is False
        return [_FakeWorkflowStatus("wf-parked"), _FakeWorkflowStatus("wf-busy")]

    def get_event(self, workflow_id: str, key: str, timeout_seconds: float) -> Any:
        assert key == "rote_auth_status"
        assert timeout_seconds == 0
        return {"awaiting": "vendor"} if workflow_id == "wf-parked" else None

    def send(
        self, workflow_id: str, message: Any, topic: str, *, serialization_type: Any = None
    ) -> None:
        # Portable serialization is load-bearing: the Python default
        # (pickle) is opaque to a TS app's DBOS.recv.
        assert serialization_type == _FAKE_PORTABLE
        self.sent.append((workflow_id, message, topic))

    def destroy(self) -> None:
        self.destroyed = True


_FAKE_PORTABLE = "portable"


def _fake_dbos_module(client_cls: type) -> types.ModuleType:
    module = types.ModuleType("dbos")
    module.DBOSClient = client_cls  # type: ignore[attr-defined]
    module.WorkflowSerializationFormat = types.SimpleNamespace(  # type: ignore[attr-defined]
        PORTABLE=_FAKE_PORTABLE
    )
    return module


@pytest.fixture
def fake_dbos(monkeypatch: pytest.MonkeyPatch) -> type[_FakeDBOSClient]:
    _FakeDBOSClient.created = []
    monkeypatch.setitem(sys.modules, "dbos", _fake_dbos_module(_FakeDBOSClient))
    return _FakeDBOSClient


def _make_app(tmp_path: Path, name: str, *, ran: bool = True) -> Path:
    app = tmp_path / name
    app.mkdir()
    (app / "dbos-config.yaml").write_text(f"name: {name}\n", encoding="utf-8")
    if ran:
        (app / f"{name}.dbos.sqlite").touch()
    return app


def test_release_sends_only_to_matching_parked_workflows(
    tmp_path: Path, fake_dbos: type[_FakeDBOSClient]
) -> None:
    app = _make_app(tmp_path, "demo")
    record_app(app, "dbos", "demo")

    report = release_parked_workflows("vendor")

    assert [(r.app, r.workflow_id) for r in report.released] == [(app.resolve(), "wf-parked")]
    assert report.skipped == ()
    client = fake_dbos.created[0]
    assert client.sent == [
        ("wf-parked", {"released_by": "rote mcp login vendor"}, "rote:auth:vendor")
    ]
    assert client.destroyed  # connection cleaned up even on success


def test_release_skips_stale_and_never_ran_apps(
    tmp_path: Path, fake_dbos: type[_FakeDBOSClient]
) -> None:
    record_app(tmp_path / "gone", "dbos", "gone")  # dir doesn't exist
    record_app(_make_app(tmp_path, "unrun", ran=False), "dbos", "unrun")
    record_app(tmp_path / "not-dbos", "temporal", "other")  # wrong runtime

    report = release_parked_workflows("vendor")

    assert report.released == ()
    reasons = {s.app.name: s.reason for s in report.skipped}
    assert "directory no longer exists" in reasons["gone"]
    assert "never ran" in reasons["unrun"]
    assert "not-dbos" not in reasons  # non-DBOS runtimes aren't scanned
    assert fake_dbos.created == []  # nothing reachable → no client built


def test_release_without_dbos_extra_is_a_loud_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DBOS apps registered but the dbos package missing → ReleaseUnavailable
    with the install hint; no apps registered → quietly empty (no import)."""
    monkeypatch.setitem(sys.modules, "dbos", None)  # import raises ImportError
    assert release_parked_workflows("vendor").released == ()

    record_app(_make_app(tmp_path, "demo"), "dbos", "demo")
    with pytest.raises(ReleaseUnavailable, match="rote-cli\\[dbos\\]"):
        release_parked_workflows("vendor")


def test_release_reports_unreachable_databases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A registry entry pointing at a dead database is reported in skipped
    with its reason — one stale app must not sink the scan (the login that
    triggered it already succeeded)."""

    class _ExplodingClient:
        def __init__(self, *, system_database_url: str) -> None:
            raise ConnectionError("connection refused")

    monkeypatch.setitem(sys.modules, "dbos", _fake_dbos_module(_ExplodingClient))

    record_app(_make_app(tmp_path, "demo"), "dbos", "demo")
    report = release_parked_workflows("vendor")
    assert report.released == ()
    assert len(report.skipped) == 1
    assert "connection refused" in report.skipped[0].reason


def test_release_scans_dbos_ts_apps_with_postgres_urls(
    tmp_path: Path, fake_dbos: type[_FakeDBOSClient], monkeypatch: pytest.MonkeyPatch
) -> None:
    """dbos-ts apps release through the same DBOSClient protocol, with the
    TS SDK's URL resolution (Postgres default — no SQLite branch)."""
    for var in ("DBOS_SYSTEM_DATABASE_URL", "PGHOST", "PGPORT", "PGUSER", "PGPASSWORD"):
        monkeypatch.delenv(var, raising=False)
    app = tmp_path / "tsapp"
    app.mkdir()
    (app / "dbos-config.yaml").write_text("name: ts_demo\nlanguage: node\n", encoding="utf-8")
    record_app(app, "dbos-ts", "ts_demo")

    report = release_parked_workflows("vendor")

    client = fake_dbos.created[0]
    assert (
        client.system_database_url == "postgresql://postgres:dbos@localhost:5432/ts_demo_dbos_sys"
    )
    assert [r.workflow_id for r in report.released] == ["wf-parked"]


def test_dbos_ts_url_prefers_config_key_with_env_expansion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rote._dbos import dbos_ts_system_database_url

    monkeypatch.delenv("DBOS_SYSTEM_DATABASE_URL", raising=False)
    app = tmp_path / "tsapp"
    app.mkdir()
    (app / "dbos-config.yaml").write_text(
        "name: ts_demo\nsystem_database_url: ${MY_PG_URL}\n", encoding="utf-8"
    )
    # Unset var expands to empty — the SDK behavior — so the default wins.
    monkeypatch.delenv("MY_PG_URL", raising=False)
    assert dbos_ts_system_database_url(app).endswith("/ts_demo_dbos_sys")
    monkeypatch.setenv("MY_PG_URL", "postgresql://u:p@db.example:5432/custom")
    assert dbos_ts_system_database_url(app) == "postgresql://u:p@db.example:5432/custom"
    # The env override outranks everything, matching emitted main.ts.
    monkeypatch.setenv("DBOS_SYSTEM_DATABASE_URL", "postgresql://o:o@override:5432/sys")
    assert dbos_ts_system_database_url(app) == "postgresql://o:o@override:5432/sys"


def test_release_broadcasts_to_inngest_apps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inngest release = one fan-out event per registered pipeline; no
    per-run discovery (events wake every matching waiter). Endpoint
    resolution: ROTE_INNGEST_EVENT_URL > INNGEST_EVENT_KEY (cloud) >
    the local dev server with its unvalidated key."""
    import httpx

    from rote.mcp.release import _inngest_event_endpoint

    posts: list[tuple[str, dict]] = []

    def _fake_post(url: str, *, json: dict, timeout: float) -> httpx.Response:
        posts.append((url, json))
        return httpx.Response(200, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", _fake_post)
    monkeypatch.delenv("ROTE_INNGEST_EVENT_URL", raising=False)
    monkeypatch.delenv("INNGEST_EVENT_KEY", raising=False)

    app = tmp_path / "inngest-app"
    app.mkdir()
    record_app(app, "inngest", "invoice_push")
    record_app(tmp_path / "inngest-dup", "inngest", "invoice_push")  # same pipeline → deduped

    report = release_parked_workflows("vendor")

    assert [(b.event, b.endpoint) for b in report.broadcasts] == [
        ("invoice_push/rote.auth.vendor", "http://127.0.0.1:8288/e/dev")
    ]
    assert posts == [
        (
            "http://127.0.0.1:8288/e/dev",
            {
                "name": "invoice_push/rote.auth.vendor",
                "data": {"server": "vendor", "released_by": "rote mcp"},
            },
        )
    ]
    assert report.released == ()

    # Endpoint resolution.
    monkeypatch.setenv("INNGEST_EVENT_KEY", "prod-key")
    assert _inngest_event_endpoint() == "https://inn.gs/e/prod-key"
    monkeypatch.setenv("ROTE_INNGEST_EVENT_URL", "http://10.0.0.5:8288/e/custom")
    assert _inngest_event_endpoint() == "http://10.0.0.5:8288/e/custom"


def test_release_reports_unreachable_inngest_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import httpx

    def _refuse(url: str, *, json: dict, timeout: float) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "post", _refuse)
    monkeypatch.delenv("ROTE_INNGEST_EVENT_URL", raising=False)
    monkeypatch.delenv("INNGEST_EVENT_KEY", raising=False)

    app = tmp_path / "inngest-app"
    app.mkdir()
    record_app(app, "inngest", "invoice_push")

    report = release_parked_workflows("vendor")
    assert report.broadcasts == ()
    assert len(report.skipped) == 1
    assert "connection refused" in report.skipped[0].reason


def test_release_blasts_cloudflare_instances(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cloudflare release = the rote_auth_<server> event sent to every
    NON-TERMINAL instance (no broadcast exists; buffering makes blasting
    safe — parked instances consume it, others buffer it harmlessly)."""
    import httpx

    calls: list[tuple[str, str]] = []

    def _fake_get(url: str, *, headers: dict, params: dict, timeout: float) -> httpx.Response:
        calls.append(("GET", url))
        assert headers["Authorization"] == "Bearer cf-token"
        return httpx.Response(
            200,
            json={
                "result": [
                    {"id": "inst-waiting", "status": "waiting"},
                    {"id": "inst-running", "status": "running"},
                    {"id": "inst-done", "status": "complete"},
                ]
            },
            request=httpx.Request("GET", url),
        )

    def _fake_post(url: str, *, headers: dict, json: dict, timeout: float) -> httpx.Response:
        calls.append(("POST", url))
        return httpx.Response(200, json={}, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "get", _fake_get)
    monkeypatch.setattr(httpx, "post", _fake_post)
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-token")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct-1")

    app = tmp_path / "cf-app"
    app.mkdir()
    record_app(app, "cloudflare", "invoice_push")

    report = release_parked_workflows("vendor")

    assert [r.workflow_id for r in report.released] == ["inst-waiting", "inst-running"]
    post_urls = [u for m, u in calls if m == "POST"]
    assert all(u.endswith("/events/rote_auth_vendor") for u in post_urls)
    assert "inst-done" not in str(post_urls)  # terminal instances skipped


def test_release_cloudflare_without_credentials_points_at_wrangler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    app = tmp_path / "cf-app"
    app.mkdir()
    record_app(app, "cloudflare", "invoice_push")

    report = release_parked_workflows("vendor")
    assert report.released == ()
    assert len(report.skipped) == 1
    assert "CLOUDFLARE_API_TOKEN" in report.skipped[0].reason
    assert "send-event" in report.skipped[0].reason  # the wrangler dev path
