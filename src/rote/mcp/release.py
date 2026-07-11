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

import os
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
class BroadcastRelease:
    """A fan-out release: one event sent, every parked run wakes.

    Inngest has no parked-run discovery API and doesn't need one —
    events fan out to all matching ``waitForEvent`` waiters, so the
    release is a single send per (endpoint, event) rather than one
    message per workflow.
    """

    app: Path
    event: str
    endpoint: str


@dataclass(frozen=True)
class SkippedApp:
    app: Path
    reason: str


@dataclass(frozen=True)
class ReleaseReport:
    released: tuple[ReleasedWorkflow, ...]
    skipped: tuple[SkippedApp, ...]
    broadcasts: tuple[BroadcastRelease, ...] = ()


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
    apps = registered_apps()
    dbos_apps = [a for a in apps if a.runtime in _DBOS_RUNTIMES]
    inngest_apps = [a for a in apps if a.runtime == "inngest"]

    released: list[ReleasedWorkflow] = []
    skipped: list[SkippedApp] = []
    broadcasts: list[BroadcastRelease] = []

    if inngest_apps:
        _release_inngest_apps(server, inngest_apps, broadcasts, skipped)
    if not dbos_apps:
        return ReleaseReport(released=(), skipped=tuple(skipped), broadcasts=tuple(broadcasts))

    try:
        from dbos import DBOSClient
    except ImportError as e:
        raise ReleaseUnavailable(
            "DBOS apps are registered but releasing their parked workflows "
            "requires the dbos extra: pip install 'rote-cli[dbos]'"
        ) from e

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
    return ReleaseReport(
        released=tuple(released), skipped=tuple(skipped), broadcasts=tuple(broadcasts)
    )


def _inngest_event_endpoint() -> str:
    """Where to POST release events.

    ``ROTE_INNGEST_EVENT_URL`` (a full ``…/e/<key>`` URL) wins;
    ``INNGEST_EVENT_KEY`` selects Inngest Cloud; otherwise the local dev
    server, which accepts any key ("the Dev Server does not validate
    keys locally").
    """
    override = os.environ.get("ROTE_INNGEST_EVENT_URL")
    if override:
        return override
    event_key = os.environ.get("INNGEST_EVENT_KEY")
    if event_key:
        return f"https://inn.gs/e/{event_key}"
    return "http://127.0.0.1:8288/e/dev"


def _release_inngest_apps(
    server: str,
    apps: list[RegisteredApp],
    broadcasts: list[BroadcastRelease],
    skipped: list[SkippedApp],
) -> None:
    """Broadcast the release event for every registered Inngest pipeline.

    Event name mirrors :func:`rote.adapters.inngest.auth_event_name`:
    ``<pipeline>/rote.auth.<server>``. One send wakes every run parked
    on it (Inngest events fan out to all matching waiters); a send to a
    pipeline with nothing parked is harmless — Inngest does not buffer
    events for waits that haven't started, and the emitted park loop's
    retry-once covers a release that lands in that gap.
    """
    try:
        import httpx
    except ImportError:
        skipped.extend(
            SkippedApp(a.path, "releasing Inngest runs requires httpx (rote-cli[mcp])")
            for a in apps
        )
        return

    endpoint = _inngest_event_endpoint()
    seen_events: set[str] = set()
    for app in apps:
        if not app.pipeline:
            skipped.append(SkippedApp(app.path, "registry entry has no pipeline name"))
            continue
        event = f"{app.pipeline}/rote.auth.{server}"
        if event in seen_events:
            continue
        seen_events.add(event)
        try:
            response = httpx.post(
                endpoint,
                json={"name": event, "data": {"server": server, "released_by": "rote mcp"}},
                timeout=10.0,
            )
            response.raise_for_status()
        except Exception as e:  # noqa: BLE001 — a dead endpoint must not sink the scan
            skipped.append(SkippedApp(app.path, f"could not send {event!r} to {endpoint}: {e}"))
            continue
        broadcasts.append(BroadcastRelease(app.path, event, endpoint))


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
