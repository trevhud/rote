"""Auth-park visibility during ``rote run`` and the read-only listing.

Unit level: channels, the watcher's TTY and non-TTY branches (all I/O
injected), the emitted advertisement each runtime produces, and
``rote mcp parked``. The live proof is the park e2e suites
(``tests/test_mcp_park_e2e.py`` drives ``rote run``'s runner end to end).
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from rote.app_registry import record_app
from rote.mcp.auth import McpAuthError
from rote.mcp.parked import (
    AUTH_PARK_MARKER,
    AuthPark,
    AuthParkWatcher,
    DbosAuthChannel,
    LogMarkerChannel,
    list_parked_workflows,
    parse_auth_park_markers,
    release_hint,
    wait_for_exit,
)
from rote.mcp.registry import McpRegistry, McpServerConfig, save_registry
from tests.test_app_registry import (  # reuse the proven fake DBOSClient
    _FAKE_PORTABLE,
    _fake_dbos_module,
    _FakeDBOSClient,
    _make_app,
)

# ───────── Fakes ─────────


@pytest.fixture
def fake_dbos(monkeypatch: pytest.MonkeyPatch) -> type[_FakeDBOSClient]:
    _FakeDBOSClient.created = []
    monkeypatch.setitem(sys.modules, "dbos", _fake_dbos_module(_FakeDBOSClient))
    return _FakeDBOSClient


@dataclass
class _FakeChannel:
    """A channel whose parks the test sets directly."""

    parks: list[AuthPark] = field(default_factory=list)
    released: list[AuthPark] = field(default_factory=list)
    external_release: bool = False
    remote_fix: Any = None

    def poll(self) -> list[AuthPark]:
        return list(self.parks)

    def release(self, park: AuthPark) -> None:
        self.released.append(park)
        self.parks = [p for p in self.parks if p.server != park.server]

    def close(self) -> None:
        pass


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _park(server: str = "vendor", node: str | None = "enrich", key: str = "k1") -> AuthPark:
    return AuthPark(server=server, node=node, run_id="wf-1", key=key)


def _no_input(prompt: str) -> str:
    raise AssertionError(f"non-interactive watcher prompted: {prompt!r}")


def _no_login(server: str) -> None:
    raise AssertionError(f"non-interactive watcher ran a login for {server!r}")


def _watcher(channel: _FakeChannel, lines: list[str], **kwargs: Any) -> AuthParkWatcher:
    defaults: dict[str, Any] = {
        "interactive": False,
        "input_fn": _no_input,
        "login_fn": _no_login,
        "echo": lines.append,
        "fingerprint_fn": lambda server: ("v1",),
        "hint_fn": lambda server: f"rote mcp login {server}",
        "settled_fn": lambda server: True,
    }
    defaults.update(kwargs)
    return AuthParkWatcher(channel=channel, **defaults)


# ───────── Non-TTY: announce, never prompt, release on credential change ─────────


def test_non_tty_announces_server_node_and_release_command_once() -> None:
    channel = _FakeChannel(parks=[_park()])
    lines: list[str] = []
    watcher = _watcher(channel, lines)

    assert watcher.tick() == 0.0
    text = "\n".join(lines)
    assert "node 'enrich'" in text
    assert "server 'vendor'" in text
    assert "run wf-1" in text
    assert "release it: rote mcp login vendor" in text
    assert "resumes on its own" in text
    assert watcher.has_active_park

    # Same park on the next tick: no second announcement, no release.
    count = len(lines)
    watcher.tick()
    assert len(lines) == count
    assert channel.released == []


def test_non_tty_releases_its_own_run_when_credentials_change() -> None:
    channel = _FakeChannel(parks=[_park()])
    lines: list[str] = []
    version = ["v1"]
    watcher = _watcher(channel, lines, fingerprint_fn=lambda server: (version[0],))

    watcher.tick()
    assert channel.released == []
    version[0] = "v2"  # `rote mcp login vendor` in another terminal
    watcher.tick()
    assert [p.server for p in channel.released] == ["vendor"]
    assert any("credentials for 'vendor' changed" in line for line in lines)
    assert not watcher.has_active_park


def test_a_login_in_progress_does_not_release_until_its_token_lands() -> None:
    """Found live: a login's first write is the client registration, with
    no token. Releasing then re-parks the run and strands it."""
    channel = _FakeChannel(parks=[_park()])
    version = ["v1"]
    settled = [False]
    watcher = _watcher(
        channel,
        [],
        fingerprint_fn=lambda server: (version[0],),
        settled_fn=lambda server: settled[0],
    )
    watcher.tick()
    version[0] = "v2"  # client_info written, tokens not yet
    watcher.tick()
    assert channel.released == []
    settled[0] = True
    watcher.tick()
    assert [p.server for p in channel.released] == ["vendor"]


def test_credentials_settled_reads_the_token_file() -> None:
    from rote.mcp.parked import credentials_settled
    from rote.mcp.tokens import write_token_file

    assert credentials_settled("vendor")  # no file: nothing in flight
    write_token_file("vendor", {"server_url": "u", "tokens": None, "client_info": {"id": "c"}})
    assert not credentials_settled("vendor")
    write_token_file("vendor", {"server_url": "u", "tokens": {"access_token": "a"}})
    assert credentials_settled("vendor")


def test_external_release_channels_wait_out_the_grace_before_sending() -> None:
    """DBOS apps are released by `rote mcp login` itself; the watcher only
    sends its own release when that one has not landed."""
    channel = _FakeChannel(parks=[_park()], external_release=True)
    clock = _Clock()
    version = ["v1"]
    watcher = _watcher(
        channel,
        [],
        fingerprint_fn=lambda server: (version[0],),
        clock=clock,
        external_grace_seconds=5.0,
    )
    watcher.tick()
    version[0] = "v2"
    watcher.tick()
    clock.now += 4.9
    watcher.tick()
    assert channel.released == []
    clock.now += 0.2
    watcher.tick()
    assert [p.server for p in channel.released] == ["vendor"]


def test_park_released_elsewhere_is_reported_and_a_new_park_is_announced_again() -> None:
    channel = _FakeChannel(parks=[_park(key="k1")])
    lines: list[str] = []
    watcher = _watcher(channel, lines)
    watcher.tick()
    channel.parks = []
    watcher.tick()
    assert lines[-1] == "rote run: auth park released, the run is resuming"
    channel.parks = [_park(key="k2")]
    watcher.tick()
    assert sum("needs MCP credentials" in line for line in lines) == 2


# ───────── TTY: offer the login in place ─────────


def test_tty_accepting_logs_in_releases_and_reports_the_time_spent() -> None:
    channel = _FakeChannel(parks=[_park()])
    clock = _Clock()
    prompts: list[str] = []
    logins: list[str] = []

    def answer(prompt: str) -> str:
        prompts.append(prompt)
        return ""  # Enter = yes

    def login(server: str) -> None:
        logins.append(server)
        clock.now += 30.0  # the browser dance

    lines: list[str] = []
    watcher = _watcher(
        channel, lines, interactive=True, input_fn=answer, login_fn=login, clock=clock
    )
    blocked = watcher.tick()

    assert prompts == ["Log in to 'vendor' now and resume this run? [Y/n] "]
    assert logins == ["vendor"]
    assert [p.server for p in channel.released] == ["vendor"]
    assert blocked == pytest.approx(30.0)
    assert any("authenticated 'vendor', releasing" in line for line in lines)
    # The non-TTY "leave it running" line is not shown to a human who
    # is about to be asked.
    assert not any("resumes on its own once" in line for line in lines)


def test_tty_declining_never_logs_in_and_falls_back_to_watching() -> None:
    channel = _FakeChannel(parks=[_park()])
    version = ["v1"]
    lines: list[str] = []
    watcher = _watcher(
        channel,
        lines,
        interactive=True,
        input_fn=lambda prompt: "n",
        fingerprint_fn=lambda server: (version[0],),
    )
    watcher.tick()
    assert channel.released == []
    assert any("in another terminal" in line for line in lines)
    version[0] = "v2"
    watcher.tick()
    assert [p.server for p in channel.released] == ["vendor"]


def test_tty_eof_on_the_prompt_counts_as_declining() -> None:
    def eof(prompt: str) -> str:
        raise EOFError

    channel = _FakeChannel(parks=[_park()])
    watcher = _watcher(channel, [], interactive=True, input_fn=eof)
    watcher.tick()
    assert channel.released == []


def test_tty_failed_login_reports_and_does_not_release() -> None:
    def failing(server: str) -> None:
        raise McpAuthError("the dance was interrupted")

    channel = _FakeChannel(parks=[_park()])
    lines: list[str] = []
    watcher = _watcher(channel, lines, interactive=True, input_fn=lambda p: "y", login_fn=failing)
    watcher.tick()
    assert channel.released == []
    assert any("login failed: the dance was interrupted" in line for line in lines)
    # Not asked again for the same server on the next tick.
    watcher.input_fn = _no_input
    watcher.tick()


# ───────── Remote credentials (Cloudflare) ─────────


def test_remote_fix_channel_never_offers_a_local_login_or_self_releases() -> None:
    """A Worker cannot see the local token store: offering `rote mcp login`
    there would 'succeed' and leave the run parked. Show the real fix."""
    channel = _FakeChannel(
        parks=[_park()], remote_fix=lambda server: f"rote mcp export {server} >> .dev.vars"
    )
    version = ["v1"]
    lines: list[str] = []
    watcher = _watcher(
        channel, lines, interactive=True, fingerprint_fn=lambda server: (version[0],)
    )
    watcher.tick()  # interactive, yet _no_input/_no_login are never hit
    assert "  fix it: rote mcp export vendor >> .dev.vars" in lines
    assert not any("release it:" in line for line in lines)
    version[0] = "v2"
    watcher.tick()
    assert channel.released == []
    assert watcher.has_active_park  # the Cloudflare runner keeps gates for it


# ───────── The in-place login over SSH / without a display ─────────


class _FakeAuthorizeResponse:
    status_code = 302


class _FakeHttpx:
    async def __aenter__(self) -> _FakeHttpx:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def get(self, url: str, follow_redirects: bool) -> _FakeAuthorizeResponse:
        return _FakeAuthorizeResponse()


@pytest.mark.parametrize("opened", [False, True])
def test_announced_login_always_prints_the_url_and_says_when_no_browser_opened(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], opened: bool
) -> None:
    """Found in review: over SSH `webbrowser.open` fails quietly and fastmcp
    logs the URL at info level only, so the user answered Y and then waited
    on a callback for a URL they never saw."""
    import asyncio

    pytest.importorskip("fastmcp")
    from rote.mcp.auth import build_oauth

    opened_urls: list[str] = []

    def fake_open(url: str) -> bool:
        opened_urls.append(url)
        return opened

    monkeypatch.setattr("webbrowser.open", fake_open)
    oauth = build_oauth("vendor", McpServerConfig(url="http://127.0.0.1:9/mcp"), announce_url=True)
    oauth.httpx_client_factory = _FakeHttpx
    url = "http://127.0.0.1:9/authorize?state=abc"
    asyncio.run(oauth.redirect_handler(url))

    err = capsys.readouterr().err
    assert f"  {url}\n" in err
    assert opened_urls == [url]
    assert ("No browser could be opened here" in err) is not opened


def test_in_place_login_asks_for_the_announced_url(monkeypatch: pytest.MonkeyPatch) -> None:
    from rote.mcp.parked import login_server

    save_registry(McpRegistry(servers={"vendor": McpServerConfig(url="http://v.example/mcp")}))
    seen: dict[str, Any] = {}

    async def fake_login(server: str, config: Any, **kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs, server=server)
        return {}

    monkeypatch.setattr("rote.mcp.auth.login", fake_login)
    login_server("vendor")
    assert seen == {"server": "vendor", "announce_url": True}


# ───────── Release hints from the registry ─────────


def test_release_hint_follows_the_registry() -> None:
    assert release_hint("vendor") == "rote mcp add vendor <url>, then rote mcp login vendor"
    save_registry(
        McpRegistry(
            servers={
                "vendor": McpServerConfig(url="https://v.example/mcp"),
                "keyed": McpServerConfig(
                    url="https://k.example/mcp", headers={"X-Api-Key": "secret"}
                ),
            }
        )
    )
    assert release_hint("vendor") == "rote mcp login vendor"
    hint = release_hint("keyed")
    assert hint.startswith("rote mcp add keyed https://k.example/mcp --header")
    assert "secret" not in hint


# ───────── Log-marker channel (Inngest, Cloudflare) ─────────


def _marker(server: str, node: str, wait: str) -> str:
    return f"{AUTH_PARK_MARKER} " + json.dumps({"server": server, "node": node, "wait": wait})


def test_parse_markers_tolerates_dev_server_decoration_and_garbage() -> None:
    text = "\n".join(
        [
            "[wrangler:info] GET /status 200",
            "  " + _marker("vendor", "enrich", "enrich auth wait 1"),
            f"{AUTH_PARK_MARKER} {{not json",
            f"{AUTH_PARK_MARKER} " + json.dumps({"node": "x"}),  # no server/wait
        ]
    )
    assert parse_auth_park_markers(text) == [
        {"server": "vendor", "node": "enrich", "wait": "enrich auth wait 1"}
    ]


def test_log_channel_dedupes_replays_and_releases_per_server(tmp_path: Path) -> None:
    log = tmp_path / "app.log"
    sent: list[str] = []
    channel = LogMarkerChannel(log, lambda park: sent.append(park.server), run_id="run-9")
    assert channel.poll() == []  # no log yet

    log.write_text(
        _marker("vendor", "a", "a-auth-wait-1") + "\n" + _marker("vendor", "a", "a-auth-wait-1")
    )
    parks = channel.poll()
    assert [(p.server, p.node, p.run_id) for p in parks] == [("vendor", "a", "run-9")]

    channel.release(parks[0])
    assert sent == ["vendor"]
    # Replay after release prints the same wait again: still released.
    log.write_text(log.read_text() + "\n" + _marker("vendor", "a", "a-auth-wait-1"))
    assert channel.poll() == []
    # A fresh park (new wait id) is a new park.
    log.write_text(log.read_text() + "\n" + _marker("vendor", "a", "a-auth-wait-3"))
    assert [p.key for p in channel.poll()] == ["a-auth-wait-3"]


# ───────── DBOS channel ─────────


class _EventClient:
    def __init__(self, events: list[Any]) -> None:
        self.events = events
        self.sent: list[tuple[str, Any, str, Any]] = []

    def get_event(self, workflow_id: str, key: str, timeout_seconds: float) -> Any:
        assert (workflow_id, key, timeout_seconds) == ("wf-7", "rote_auth_status", 0)
        return self.events.pop(0)

    def send(self, workflow_id: str, message: Any, topic: str, *, serialization_type: Any) -> None:
        self.sent.append((workflow_id, message, topic, serialization_type))

    def destroy(self) -> None:
        pass


def test_dbos_channel_reads_the_event_and_keys_each_park_episode(
    tmp_path: Path, fake_dbos: type[_FakeDBOSClient]
) -> None:
    stderr = tmp_path / "stderr.txt"
    channel = DbosAuthChannel("sqlite:///unused", stderr)
    assert channel.poll() == []  # no workflow id yet: no client opened
    assert fake_dbos.created == []

    stderr.write_text("booting\nworkflow started: wf-7\n")
    parked = {"awaiting": "vendor", "node": "enrich"}
    client = _EventClient([parked, parked, None, parked])
    channel._client = client
    first = channel.poll()
    assert [(p.server, p.node, p.run_id) for p in first] == [("vendor", "enrich", "wf-7")]
    assert channel.poll()[0].key == first[0].key  # same episode
    assert channel.poll() == []  # released
    assert channel.poll()[0].key != first[0].key  # parked again: new episode

    channel.release(first[0])
    assert client.sent == [
        (
            "wf-7",
            {"released_by": "rote run (credentials for vendor updated)"},
            "rote:auth:vendor",
            _FAKE_PORTABLE,
        )
    ]


def test_release_then_reparking_within_one_poll_is_announced_again(
    tmp_path: Path, fake_dbos: type[_FakeDBOSClient]
) -> None:
    """Found in review: the watcher releases, the credential is still bad
    (wrong account, missing scope), and the workflow clears its event,
    retries and parks again before the next poll. No poll ever sees the
    event cleared, so without the park number the new park reused the
    released key: no message, no prompt, and a run hung to --timeout."""
    stderr = tmp_path / "stderr.txt"
    stderr.write_text("workflow started: wf-7\n")
    channel = DbosAuthChannel("sqlite:///unused", stderr)
    first = {"awaiting": "vendor", "node": "enrich", "park": 1}
    second = {"awaiting": "vendor", "node": "enrich", "park": 2}
    client = _EventClient([first, first, second, second])
    channel._client = client

    version = ["v1"]
    lines: list[str] = []
    watcher = _watcher(
        channel,
        lines,
        fingerprint_fn=lambda server: (version[0],),
        external_grace_seconds=0.0,
    )
    watcher.tick()
    version[0] = "v2"
    watcher.tick()  # credentials changed: released
    assert len(client.sent) == 1
    watcher.tick()  # already parked again, never seen cleared
    assert sum("needs MCP credentials" in line for line in lines) == 2
    assert watcher.has_active_park
    version[0] = "v3"  # fixed for real this time
    watcher.tick()
    assert len(client.sent) == 2


# ───────── wait_for_exit ─────────


def test_wait_for_exit_extends_the_deadline_by_time_spent_with_the_user() -> None:
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(1.5)"])

    class _Blocking:
        ticks = 0

        def tick(self) -> float:
            self.ticks += 1
            # Pretend each tick spent 10s in a browser login.
            return 10.0

    watcher = _Blocking()
    # A 0.3s budget alone would time out; the reported user time extends it.
    wait_for_exit(proc, deadline=time.monotonic() + 0.3, watcher=watcher, poll_interval=0.1)  # type: ignore[arg-type]
    assert proc.returncode == 0
    assert watcher.ticks >= 1


def test_wait_for_exit_times_out_without_user_time() -> None:
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:

        class _Idle:
            def tick(self) -> float:
                return 0.0

        with pytest.raises(subprocess.TimeoutExpired):
            wait_for_exit(proc, deadline=time.monotonic() + 0.3, watcher=_Idle(), poll_interval=0.1)  # type: ignore[arg-type]
    finally:
        proc.kill()
        proc.wait()


# ───────── Emitted advertisements ─────────


def _mcp_pipeline() -> Any:
    from rote.ir import MCPBinding, Node, NodeKind, Pipeline, PipelineInput

    node = Node(
        id="enrich",
        kind=NodeKind.EXTERNAL_CALL,
        description="Enrich a contact.",
        input={"contact_id": "str"},
        inputs={"contact_id": "pipeline.input.contact_id"},
        output="dict",
        mcp=MCPBinding(server="vendor", tool="enrich_contact", args={"contact_id": "contact_id"}),
    )
    return Pipeline(
        name="park_demo",
        description="d",
        input=PipelineInput(type="Req", required=["contact_id"]),
        nodes=[node],
        edges=[],
        entry_nodes=["enrich"],
        exit_nodes=["enrich"],
    )


def test_dbos_ts_park_event_names_the_node(tmp_path: Path) -> None:
    from rote.adapters import get_adapter

    get_adapter("dbos-ts").emit(_mcp_pipeline(), tmp_path)
    src = (tmp_path / "src" / "main.ts").read_text(encoding="utf-8")
    assert "{ awaiting: server, node, park }," in src
    assert "const park = (parkCounts.get(workflowId) ?? 0) + 1;" in src
    assert '"vendor",\n            "enrich",\n        );' in src


def test_inngest_park_prints_the_marker_before_waiting(tmp_path: Path) -> None:
    from rote.adapters import get_adapter

    get_adapter("inngest").emit(_mcp_pipeline(), tmp_path)
    src = (tmp_path / "src" / "inngest" / "pipeline.ts").read_text(encoding="utf-8")
    marker = (
        f'"{AUTH_PARK_MARKER} " +\n'
        "                        JSON.stringify({ server, node: stepId, wait: waitId }),"
    )
    assert marker in src
    # The marker's wait id is the wait's own step id, and it prints first.
    assert src.index(marker) < src.index("await step.waitForEvent(\n                    waitId,")


def test_cloudflare_park_prints_the_marker_before_waiting(tmp_path: Path) -> None:
    from rote.adapters import get_adapter

    get_adapter("cloudflare").emit(_mcp_pipeline(), tmp_path)
    src = (tmp_path / "src" / "workflow.ts").read_text(encoding="utf-8")
    marker_at = src.index(f'"{AUTH_PARK_MARKER} " +')
    block = src[marker_at : src.index("step.waitForEvent", marker_at)]
    assert 'server: enrich_event.slice("rote_auth_".length),' in block
    assert 'node: "enrich",' in block
    assert "wait: enrich_wait," in block
    assert "enrich_wait,\n" in src[marker_at:]


# ───────── Listing (rote mcp parked) ─────────


def test_list_parked_reads_without_releasing(
    tmp_path: Path, fake_dbos: type[_FakeDBOSClient]
) -> None:
    app = _make_app(tmp_path, "demo")
    record_app(app, "dbos", "demo")
    record_app(tmp_path / "ing", "inngest", "ing")

    report = list_parked_workflows()
    assert [(p.workflow_id, p.server, p.runtime) for p in report.parked] == [
        ("wf-parked", "vendor", "dbos")
    ]
    assert fake_dbos.created[0].sent == []  # listing never releases
    assert [s.app for s in report.skipped] == [(tmp_path / "ing").resolve()]
    assert list_parked_workflows("other").parked == ()


def test_cli_mcp_parked_lists_and_points_at_the_release(
    tmp_path: Path, fake_dbos: type[_FakeDBOSClient], capsys: pytest.CaptureFixture[str]
) -> None:
    from rote.cli import main

    app = _make_app(tmp_path, "demo")
    record_app(app, "dbos", "demo")
    assert main(["mcp", "parked"]) == 0
    out = capsys.readouterr().out
    assert f"vendor: workflow wf-parked (dbos, {app.resolve()})" in out
    assert "release with: rote mcp login vendor" in out

    assert main(["mcp", "parked", "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["parked"] == [
        {
            "app": str(app.resolve()),
            "runtime": "dbos",
            "workflow_id": "wf-parked",
            "server": "vendor",
            "node": None,
        }
    ]


# ───────── rote run wiring ─────────


def test_run_pipeline_passes_interactive_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rote.eval.empirical import MeasuredRun
    from rote.runners import RunTarget, run_pipeline

    seen: dict[str, Any] = {}

    def fake_trial(app_dir: Any, payload: Any, **kwargs: Any) -> MeasuredRun:
        seen["watcher"] = kwargs["park_watch"](_FakeChannel())
        return MeasuredRun(wall_seconds=0.1, output={})

    monkeypatch.setattr("rote.runners.run_pipeline_trial", fake_trial)
    run_pipeline(RunTarget(kind="pipeline", path=tmp_path, runtime="dbos"), {}, interactive=True)
    assert seen["watcher"].interactive is True
