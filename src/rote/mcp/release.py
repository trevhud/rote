"""Release workflows parked on MCP auth — the CLI side of park-on-auth.

Emitted DBOS apps park durably when an MCP credential is missing or dead:
the workflow advertises what it's waiting for via the ``rote_auth_status``
workflow event (DBOS has no distinct WAITING status — a parked workflow is
just PENDING, so it must say what it's blocked on), then blocks on a
``DBOS.recv`` over the ``rote:auth:<server>`` topic. This module is the
other half: after ``rote mcp login <server>`` succeeds, scan the app
registry for DBOS apps, find PENDING workflows awaiting that server, and
send each one the release message.

DBOS notifications persist per-topic, so releasing a workflow that hasn't
quite reached its ``recv`` yet is safe — the message waits for it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rote._dbos import dbos_system_database_url, dbos_ts_system_database_url
from rote.app_registry import RegisteredApp, registered_apps


class ReleaseUnavailable(RuntimeError):
    """DBOS apps are registered but the ``dbos`` package isn't installed."""


@dataclass(frozen=True)
class ReleasedWorkflow:
    app: Path
    workflow_id: str


@dataclass(frozen=True)
class SkippedApp:
    app: Path
    reason: str


@dataclass(frozen=True)
class ReleaseReport:
    released: tuple[ReleasedWorkflow, ...]
    skipped: tuple[SkippedApp, ...]


#: Runtimes whose parked workflows this module can release. Both DBOS
#: SDKs share the same system-DB schema, status strings, and event/
#: message tables — one DBOSClient protocol serves Python and TS apps.
_DBOS_RUNTIMES = ("dbos", "dbos-ts")


def release_parked_workflows(server: str) -> ReleaseReport:
    """Wake every registered DBOS workflow parked waiting for ``server``.

    Covers both the Python (``dbos``) and TypeScript (``dbos-ts``)
    runtimes — the park contract (``rote_auth_status`` event +
    ``rote:auth:<server>`` topic) is language-neutral, and messages are
    sent in DBOS's *portable* serialization format, which both SDKs
    deserialize (the Python default, pickle, would be unreadable from
    TS).

    Registry entries are allowed to be stale (moved/deleted app dirs,
    apps that never ran, an unreachable Postgres) — each such app is
    reported in ``skipped`` with its reason rather than failing the
    whole scan; the login that triggered this already succeeded.
    """
    dbos_apps = [a for a in registered_apps() if a.runtime in _DBOS_RUNTIMES]
    if not dbos_apps:
        return ReleaseReport(released=(), skipped=())

    try:
        from dbos import DBOSClient
    except ImportError as e:
        raise ReleaseUnavailable(
            "DBOS apps are registered but releasing their parked workflows "
            "requires the dbos extra: pip install 'rote-cli[dbos]'"
        ) from e

    released: list[ReleasedWorkflow] = []
    skipped: list[SkippedApp] = []
    seen_urls: set[str] = set()

    for app in dbos_apps:
        if not app.path.is_dir():
            skipped.append(SkippedApp(app.path, "directory no longer exists"))
            continue
        if not (app.path / "dbos-config.yaml").is_file():
            skipped.append(SkippedApp(app.path, "no dbos-config.yaml"))
            continue
        url = (
            dbos_ts_system_database_url(app.path)
            if app.runtime == "dbos-ts"
            else dbos_system_database_url(app.path)
        )
        if url in seen_urls:
            continue
        seen_urls.add(url)
        sqlite_prefix = "sqlite:///"
        if url.startswith(sqlite_prefix) and not Path(url[len(sqlite_prefix) :]).is_file():
            skipped.append(SkippedApp(app.path, "no system database yet (app never ran)"))
            continue
        try:
            released.extend(_release_in_app(DBOSClient, app, url, server))
        except Exception as e:  # noqa: BLE001 — a stale entry must not sink the scan
            skipped.append(SkippedApp(app.path, f"could not reach its system database: {e}"))
    return ReleaseReport(released=tuple(released), skipped=tuple(skipped))


def _release_in_app(
    client_cls: type, app: RegisteredApp, url: str, server: str
) -> list[ReleasedWorkflow]:
    from dbos import WorkflowSerializationFormat

    client = client_cls(system_database_url=url)
    try:
        released: list[ReleasedWorkflow] = []
        # load_input/output off: a TS app's superjson-serialized values
        # are unreadable from Python, and we only need ids + the event.
        for wf in client.list_workflows(status="PENDING", load_input=False, load_output=False):
            status = client.get_event(wf.workflow_id, "rote_auth_status", timeout_seconds=0)
            if isinstance(status, dict) and status.get("awaiting") == server:
                client.send(
                    wf.workflow_id,
                    {"released_by": f"rote mcp login {server}"},
                    topic=f"rote:auth:{server}",
                    # Portable JSON: readable by BOTH SDKs' recv (the
                    # Python default, pickle, is TS-opaque).
                    serialization_type=WorkflowSerializationFormat.PORTABLE,
                )
                released.append(ReleasedWorkflow(app.path, wf.workflow_id))
        return released
    finally:
        client.destroy()
