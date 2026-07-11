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

    def list_workflows(self, *, status: str) -> list[_FakeWorkflowStatus]:
        assert status == "PENDING"
        return [_FakeWorkflowStatus("wf-parked"), _FakeWorkflowStatus("wf-busy")]

    def get_event(self, workflow_id: str, key: str, timeout_seconds: float) -> Any:
        assert key == "rote_auth_status"
        assert timeout_seconds == 0
        return {"awaiting": "vendor"} if workflow_id == "wf-parked" else None

    def send(self, workflow_id: str, message: Any, topic: str) -> None:
        self.sent.append((workflow_id, message, topic))

    def destroy(self) -> None:
        self.destroyed = True


@pytest.fixture
def fake_dbos(monkeypatch: pytest.MonkeyPatch) -> type[_FakeDBOSClient]:
    _FakeDBOSClient.created = []
    module = types.ModuleType("dbos")
    module.DBOSClient = _FakeDBOSClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "dbos", module)
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

    module = types.ModuleType("dbos")
    module.DBOSClient = _ExplodingClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "dbos", module)

    record_app(_make_app(tmp_path, "demo"), "dbos", "demo")
    report = release_parked_workflows("vendor")
    assert report.released == ()
    assert len(report.skipped) == 1
    assert "connection refused" in report.skipped[0].reason
