"""See an auth park while ``rote run`` drives it, and resolve it in place.

Emitted apps park durably when an MCP credential is missing or dead
(see :mod:`rote.mcp.release`). Unattended, that is the right behavior.
Under ``rote run`` it used to look like a hang: the run sat silent until
``--timeout``. This module is the run-side half:

- A **channel** per runtime answers "is this run parked on auth right
  now, for which server and node?" and knows how to release *this* run:

  - :class:`DbosAuthChannel` (``dbos`` and ``dbos-ts``) reads the
    ``rote_auth_status`` workflow event the emitted park advertises
    (``{"awaiting": server, "node": node}``) and releases by sending the
    ``rote:auth:<server>`` message, in DBOS's portable serialization so
    both SDKs can read it.
  - :class:`LogMarkerChannel` (``inngest`` and ``cloudflare``) scans the
    app's log for the :data:`AUTH_PARK_MARKER` line the emitted park
    prints just before its ``waitForEvent``. Neither local dev server
    exposes a truthful "waiting on event X" status (Inngest's v1 run
    status and Cloudflare's local-dev status both read as running), so
    the emitted code says it instead. The runner supplies the release,
    because only it knows the local dev server's port or instance id.

- :class:`AuthParkWatcher` turns channel observations into what the
  user sees: one message per park naming the server, the node, and the
  command that releases it. On an interactive terminal it offers to run
  the browser login right there and release the run. Off a terminal it
  never prompts and never opens a browser (CI-safe); it watches the
  server's credential files instead and releases the run itself once
  they change, so ``rote mcp login <server>`` in another terminal is
  enough on every runtime, including the local dev servers that the
  registry-wide release cannot reach.

Everything interactive lives here, on the CLI side. Emitted workflow
code stays never-interactive (CLAUDE.md invariant #3): it only parks and
advertises.

:func:`list_parked_workflows` is the read-only sibling of
:func:`rote.mcp.release.release_parked_workflows`, backing
``rote mcp parked``.
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from rote.mcp.learned import note_auth_park_safely

#: Prefix of the line the emitted TypeScript parks (Inngest, Cloudflare)
#: print right before they wait, followed by a JSON object with
#: ``server``, ``node`` and ``wait`` (the wait's step id, unique per park
#: within a run). Both runtimes replay workflow code, so the same line
#: can print again after release; ``wait`` is what de-duplicates it.
AUTH_PARK_MARKER = "rote-auth-park"

_MARKER_RE = re.compile(re.escape(AUTH_PARK_MARKER) + r" (\{.*\})")

#: The workflow event both DBOS adapters advertise a park through.
AUTH_STATUS_EVENT = "rote_auth_status"

#: Where a run says which HITL gates it has reached. On DBOS (both SDKs) a
#: workflow event, and on Temporal a workflow query, of this name, whose
#: value is ``{"reached": [{"gate", "signal", "message"}, ...]}`` in the
#: order the gates were reached. The list keeps every gate, so a poll that
#: comes late (a gate whose payload was already waiting) still sees it.
GATE_STATUS_EVENT = "rote_gate_status"

#: Prefix of the line the emitted TypeScript gates (Inngest, Cloudflare)
#: print right before they wait, followed by a JSON object with ``gate``,
#: ``signal``, ``message`` and ``wait`` (the wait's step id). Replays print
#: it again; ``wait`` de-duplicates it, as for :data:`AUTH_PARK_MARKER`.
GATE_PARK_MARKER = "rote-gate-park"

_GATE_MARKER_RE = re.compile(re.escape(GATE_PARK_MARKER) + r" (\{.*\})")


def auth_topic(server: str) -> str:
    """The DBOS message topic that releases a park on ``server``."""
    return f"rote:auth:{server}"


@dataclass(frozen=True)
class AuthPark:
    """One observed park: a run waiting for credentials to ``server``."""

    server: str
    node: str | None
    run_id: str | None
    key: str
    """Identifies this park episode; a later park of the same run gets a
    new key, so it is announced again."""


@dataclass(frozen=True)
class GateReached:
    """A run that reached a HITL gate, and the gate's rendered message.

    The message is display only: rote sends it nowhere (no channel or
    target in the IR's ``notify`` is ever contacted).
    """

    gate: str
    signal: str
    message: str | None
    key: str
    """Identifies this arrival, so it is announced once."""
    shows: dict[str, Any] | None = None
    """The gate's other bound inputs by name: the data the person decides on
    that its message does not name (#325)."""


def gates_from_status(status: Any, run_id: str) -> list[GateReached]:
    """The gates a :data:`GATE_STATUS_EVENT` value says the run reached."""
    if not isinstance(status, dict) or not isinstance(status.get("reached"), list):
        return []
    found: list[GateReached] = []
    for i, entry in enumerate(status["reached"]):
        if isinstance(entry, dict) and entry.get("gate"):
            message = entry.get("message")
            found.append(
                GateReached(
                    gate=str(entry["gate"]),
                    signal=str(entry.get("signal") or ""),
                    message=message if isinstance(message, str) else None,
                    key=f"{run_id}#{i}",
                    shows=entry["shows"] if isinstance(entry.get("shows"), dict) else None,
                )
            )
    return found


def parse_gate_park_markers(text: str) -> list[GateReached]:
    """Every well-formed :data:`GATE_PARK_MARKER` line in ``text``, in order."""
    found: list[GateReached] = []
    for line in text.splitlines():
        match = _GATE_MARKER_RE.search(line)
        if match is None:
            continue
        try:
            doc = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(doc, dict) and doc.get("gate") and doc.get("wait"):
            message = doc.get("message")
            found.append(
                GateReached(
                    gate=str(doc["gate"]),
                    signal=str(doc.get("signal") or ""),
                    message=message if isinstance(message, str) else None,
                    key=str(doc["wait"]),
                    shows=doc["shows"] if isinstance(doc.get("shows"), dict) else None,
                )
            )
    return found


class ParkChannel(Protocol):
    """How one runtime reports and releases this run's auth parks."""

    #: True when something other than this run's watcher can release the
    #: park (``rote mcp login`` releases registered DBOS apps itself), so
    #: a credential change waits briefly before the watcher sends its own
    #: release instead of doubling it.
    external_release: bool

    #: How this runtime gets its credentials fixed, when a local login is
    #: not it. ``None`` means the app reads the local token store (DBOS,
    #: Inngest), so ``rote mcp login`` is the fix and the watcher may offer
    #: it and release on a credential change. A Cloudflare Worker reads
    #: only its own secrets, so a local login changes nothing it can see:
    #: the watcher then shows this hint and neither offers a login nor
    #: releases the run.
    remote_fix: Callable[[str], str] | None

    def poll(self) -> list[AuthPark]:
        """Parks active right now."""
        ...

    def poll_gates(self) -> list[GateReached]:
        """Every HITL gate the run has reached so far, in order."""
        ...

    def release(self, park: AuthPark) -> None:
        """Wake this run's park (the credential is fixed)."""
        ...

    def close(self) -> None: ...


# ───────── DBOS (Python and TypeScript) ─────────


def _workflow_id_from_stderr(stderr_path: Path) -> str | None:
    """The ``workflow started: <id>`` line both DBOS apps print, if present."""
    marker = "workflow started: "
    try:
        text = stderr_path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return None
    for line in text.splitlines():
        if line.startswith(marker):
            return line[len(marker) :].strip()
    return None


class DbosAuthChannel:
    """Park visibility for an emitted DBOS app (Python or TypeScript).

    The workflow id is harvested lazily from the app's stderr, and the
    ``DBOSClient`` is only opened once it is known, so an app that fails
    before starting a workflow reports its own error rather than one
    from this channel.
    """

    external_release = True
    remote_fix: Callable[[str], str] | None = None

    def __init__(self, system_database_url: str, stderr_path: Path) -> None:
        self._url = system_database_url
        self._stderr_path = stderr_path
        self._workflow_id: str | None = None
        self._client: Any = None
        self._episode = 0
        self._was_parked = False

    def _ensure_client(self) -> Any:
        if self._client is None:
            from dbos import DBOSClient

            self._client = DBOSClient(system_database_url=self._url)
        return self._client

    def poll(self) -> list[AuthPark]:
        if self._workflow_id is None:
            self._workflow_id = _workflow_id_from_stderr(self._stderr_path)
            if self._workflow_id is None:
                return []
        status = self._ensure_client().get_event(
            self._workflow_id, AUTH_STATUS_EVENT, timeout_seconds=0
        )
        if not (isinstance(status, dict) and status.get("awaiting")):
            if self._was_parked:
                self._episode += 1
                self._was_parked = False
            return []
        self._was_parked = True
        server = str(status["awaiting"])
        node = status.get("node")
        # Current apps number each park, which is the only reliable key: a
        # release then an immediate re-park can both land between two
        # polls, so "saw the event clear" never happens. The episode
        # counter only serves apps emitted before the number existed.
        park = status.get("park")
        episode = f"p{park}" if park is not None else f"e{self._episode}"
        return [
            AuthPark(
                server=server,
                node=str(node) if node else None,
                run_id=self._workflow_id,
                key=f"{self._workflow_id}#{episode}:{server}",
            )
        ]

    def poll_gates(self) -> list[GateReached]:
        if self._workflow_id is None:
            self._workflow_id = _workflow_id_from_stderr(self._stderr_path)
            if self._workflow_id is None:
                return []
        status = self._ensure_client().get_event(
            self._workflow_id, GATE_STATUS_EVENT, timeout_seconds=0
        )
        return gates_from_status(status, self._workflow_id)

    def release(self, park: AuthPark) -> None:
        from dbos import WorkflowSerializationFormat

        self._ensure_client().send(
            park.run_id,
            {"released_by": f"rote run (credentials for {park.server} updated)"},
            topic=auth_topic(park.server),
            # Portable: readable by the TS SDK's recv as well as Python's.
            serialization_type=WorkflowSerializationFormat.PORTABLE,
        )

    def close(self) -> None:
        if self._client is not None:
            self._client.destroy()
            self._client = None


# ───────── Log markers (Inngest, Cloudflare) ─────────


def parse_auth_park_markers(text: str) -> list[dict[str, str]]:
    """Every well-formed marker in ``text``, in order.

    The marker is matched anywhere on the line because dev servers
    prefix worker output (wrangler adds its own decoration). A line
    whose JSON does not parse is not a marker.
    """
    found: list[dict[str, str]] = []
    for line in text.splitlines():
        match = _MARKER_RE.search(line)
        if match is None:
            continue
        try:
            doc = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(doc, dict) and doc.get("server") and doc.get("wait"):
            found.append({k: str(v) for k, v in doc.items() if v is not None})
    return found


class LogMarkerChannel:
    """Park visibility from the marker line the emitted TS park prints.

    A marker cannot say when its park ended, so a park stays active until
    this channel releases it. Locally that is accurate: the registry-wide
    ``rote mcp login`` release targets the default Inngest port and the
    Cloudflare REST API, neither of which reaches a ``rote run`` dev
    server, so this run's watcher is the only releaser there is.
    """

    external_release = False

    def __init__(
        self,
        log_path: Path,
        release: Callable[[AuthPark], None] | None,
        *,
        run_id: str | None = None,
        remote_fix: Callable[[str], str] | None = None,
    ) -> None:
        self._log_path = log_path
        self._release = release
        self._run_id = run_id
        self.remote_fix = remote_fix
        self._parks: dict[str, AuthPark] = {}
        self._released: set[str] = set()

    def poll(self) -> list[AuthPark]:
        try:
            text = self._log_path.read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            return []
        for doc in parse_auth_park_markers(text):
            key = doc["wait"]
            if key not in self._parks:
                self._parks[key] = AuthPark(
                    server=doc["server"], node=doc.get("node"), run_id=self._run_id, key=key
                )
        return [p for k, p in self._parks.items() if k not in self._released]

    def poll_gates(self) -> list[GateReached]:
        try:
            text = self._log_path.read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            return []
        return parse_gate_park_markers(text)

    def release(self, park: AuthPark) -> None:
        if self._release is None:
            # The watcher never calls this when remote_fix is set.
            raise RuntimeError(f"this runtime cannot release a park on {park.server!r} itself")
        self._release(park)
        # One release event wakes every wait on that server (Inngest
        # broadcasts; Cloudflare parks one step at a time), so every
        # active park on the server is done.
        for key, other in self._parks.items():
            if other.server == park.server:
                self._released.add(key)

    def close(self) -> None:
        pass


# ───────── The watcher ─────────


def credential_fingerprint(server: str) -> tuple[Any, ...]:
    """What changes when the user fixes ``server``'s credentials.

    The token file (``rote mcp login``) and the registry (``rote mcp
    add --header``, a new URL). Stat data, not contents: the token file
    holds secrets and nothing here needs to read them.
    """
    from rote.mcp.registry import registry_path
    from rote.mcp.tokens import token_file

    def stat(p: Path) -> tuple[int, int] | None:
        try:
            st = p.stat()
        except FileNotFoundError:
            return None
        return (st.st_mtime_ns, st.st_size)

    return (stat(token_file(server)), stat(registry_path()))


def credentials_settled(server: str) -> bool:
    """Whether ``server``'s token file is in a finished state.

    A login writes the token file more than once: the dynamic client
    registration lands first, with no token yet, and the tokens arrive
    when the browser comes back. Releasing on the first write wakes the
    run into the same failure and it parks again, by then with the
    finished credential already counted as seen. So a change only
    counts once the file has an access token, or is gone (an
    unauthenticated server, or one moved to static headers).
    """
    from rote.mcp.tokens import read_token_file

    doc = read_token_file(server)
    return doc is None or bool((doc.get("tokens") or {}).get("access_token"))


def release_hint(server: str) -> str:
    """The command that fixes ``server``'s credentials, for this machine.

    Read from the registry so the hint is the one that applies: a server
    that was never registered needs ``rote mcp add`` first, and a server
    configured with static headers has no OAuth login to run.
    """
    from rote.mcp.registry import load_registry

    config = load_registry().servers.get(server)
    if config is None:
        return f"rote mcp add {server} <url>, then rote mcp login {server}"
    if config.headers and not config.client_id:
        return f"rote mcp add {server} {config.url} --header 'Name: value' (update its headers)"
    return f"rote mcp login {server}"


def login_server(server: str) -> None:
    """Run the browser OAuth login for ``server`` (raises McpAuthError)."""
    from rote.mcp.auth import McpAuthError, login
    from rote.mcp.registry import load_registry

    config = load_registry().servers.get(server)
    if config is None:
        raise McpAuthError(
            f"no MCP server named {server!r} is registered; run: rote mcp add {server} <url>"
        )
    # Always show the URL: the user just said yes at a prompt, and on SSH
    # or a display-less TTY no browser will open by itself.
    asyncio.run(login(server, config, announce_url=True))


def _stderr_echo(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


def _stderr_input(prompt: str) -> str:
    """``input()`` with the prompt on stderr.

    ``input`` writes its prompt to stdout, which under ``rote run`` is
    the result JSON (often redirected to a file), so the user would be
    asked a question they cannot see. Found by running it on a pty.
    """
    print(prompt, end="", file=sys.stderr, flush=True)
    line = sys.stdin.readline()
    if not line:
        raise EOFError
    return line.rstrip("\n")


@dataclass
class AuthParkWatcher:
    """Announce auth parks during ``rote run`` and resolve them.

    ``tick()`` is called from the runner's poll loop and returns the
    seconds it spent waiting on the user (prompt plus browser login), so
    the runner can extend its deadline by that much: time the human
    spends authenticating is not time the pipeline spent running.

    All I/O is injected (``input_fn``, ``echo``, ``login_fn``,
    ``fingerprint_fn``) so both branches are testable without a TTY.
    """

    channel: ParkChannel
    interactive: bool
    input_fn: Callable[[str], str] | None = None
    echo: Callable[[str], None] = _stderr_echo
    login_fn: Callable[[str], None] = login_server
    fingerprint_fn: Callable[[str], tuple[Any, ...]] = credential_fingerprint
    hint_fn: Callable[[str], str] = release_hint
    settled_fn: Callable[[str], bool] = credentials_settled
    #: Seconds a credential change waits for an external release (only on
    #: channels with ``external_release``) before this watcher sends one.
    #: A known race, accepted: if the workflow has not consumed the login's
    #: release within the grace, both messages are sent. DBOS messages
    #: persist per topic, so the spare one is consumed by the next park on
    #: that server in this run. That park returns at once and retries,
    #: costing at most two extra failing attempts before it parks for real.
    #: Nothing is lost and nothing is skipped.
    external_grace_seconds: float = 5.0
    clock: Callable[[], float] = time.monotonic
    _seen: dict[str, tuple[Any, ...]] = field(default_factory=dict, init=False)
    _changed_at: dict[str, float] = field(default_factory=dict, init=False)
    _released: set[str] = field(default_factory=set, init=False)
    _declined: set[str] = field(default_factory=set, init=False)
    _active: set[str] = field(default_factory=set, init=False)
    _gates_announced: set[str] = field(default_factory=set, init=False)

    def tick(self) -> float:
        self._announce_gates()
        active = self.channel.poll()
        active_keys = {p.key for p in active}
        for key in list(self._seen):
            if key not in active_keys:
                if key not in self._released:
                    self.echo("rote run: auth park released, the run is resuming")
                del self._seen[key]
                self._changed_at.pop(key, None)
        blocked = 0.0
        for park in active:
            if park.key in self._released:
                continue
            if park.key not in self._seen:
                self._seen[park.key] = self.fingerprint_fn(park.server)
                self._announce(park)
                warning = note_auth_park_safely(park.server)
                if warning is not None:
                    self.echo(f"rote run: note: {warning}")
                fixable_here = self.channel.remote_fix is None
                if fixable_here and self.interactive and park.server not in self._declined:
                    blocked += self._offer_login(park)
                continue
            if self.channel.remote_fix is None:
                self._release_if_fixed(park)
        self._active = {p.key for p in active if p.key not in self._released}
        return blocked

    @property
    def has_active_park(self) -> bool:
        """Whether the last tick saw a park this watcher has not released."""
        return bool(self._active)

    def close(self) -> None:
        self.channel.close()

    def drain(self) -> None:
        """Announce any gate the run reached since the last tick.

        Runners call it once the run has ended: a gate whose payload was
        already waiting can be reached and passed between two ticks.
        """
        self._announce_gates()

    def _announce_gates(self) -> None:
        for gate in self.channel.poll_gates():
            if gate.key in self._gates_announced:
                continue
            self._gates_announced.add(gate.key)
            self.echo(gate_announcement(gate))

    def _announce(self, park: AuthPark) -> None:
        where = f"node {park.node!r}" if park.node else "a step"
        run = f" (run {park.run_id})" if park.run_id else ""
        self.echo(
            f"rote run: parked{run}: {where} needs MCP credentials for server {park.server!r}"
        )
        remote_fix = self.channel.remote_fix
        if remote_fix is not None:
            self.echo(f"  fix it: {remote_fix(park.server)}")
            return
        self.echo(f"  release it: {self.hint_fn(park.server)}")
        if not self.interactive:
            self.echo(
                "  this run resumes on its own once the credentials change; "
                "leave it running (it gives up at --timeout)"
            )

    def _offer_login(self, park: AuthPark) -> float:
        started = self.clock()
        input_fn = self.input_fn if self.input_fn is not None else _stderr_input
        try:
            answer = input_fn(f"Log in to {park.server!r} now and resume this run? [Y/n] ")
        except EOFError:
            answer = "n"
        if answer.strip().lower() not in {"", "y", "yes"}:
            self._declined.add(park.server)
            self.echo(
                f"  ok, waiting: run `{self.hint_fn(park.server)}` in another "
                "terminal and this run resumes on its own"
            )
            return self.clock() - started
        from rote.mcp.auth import McpAuthError

        try:
            self.login_fn(park.server)
        except McpAuthError as e:
            self._declined.add(park.server)
            self.echo(f"  login failed: {e}")
            self.echo("  waiting: fix the credentials and this run resumes on its own")
            return self.clock() - started
        self._release(park, reason=f"authenticated {park.server!r}")
        return self.clock() - started

    def _release_if_fixed(self, park: AuthPark) -> None:
        if self.fingerprint_fn(park.server) == self._seen[park.key]:
            return
        if not self.settled_fn(park.server):
            return  # a login in progress; wait for it to finish
        if self.channel.external_release:
            first = self._changed_at.setdefault(park.key, self.clock())
            if self.clock() - first < self.external_grace_seconds:
                return
        self._release(park, reason=f"credentials for {park.server!r} changed")

    def _release(self, park: AuthPark, *, reason: str) -> None:
        self.channel.release(park)
        self._released.add(park.key)
        self.echo(f"rote run: {reason}, releasing the parked run")


def gate_announcement(gate: GateReached) -> str:
    """What ``rote run`` prints when the run reaches ``gate``: one line, then
    the message indented (blank lines stay blank, trailing ones go). The
    message is escaped again here, because the runner prints whatever the
    app reported; on a message the emitted renderer wrote, that changes
    nothing."""
    from rote.adapters._common import escape_invisible

    head = f"rote run: waiting at gate {gate.gate!r} for signal {gate.signal!r}"
    text = escape_invisible(gate.message or "").rstrip("\n")
    for name, value in (gate.shows or {}).items():
        # What the gate shows below its message: a string as written (a draft
        # keeps its lines), anything else as compact JSON.
        shown = (
            value
            if isinstance(value, str)
            else json.dumps(value, separators=(",", ":"), ensure_ascii=False)
        )
        shown = escape_invisible(shown).rstrip("\n")
        text += ("\n" if text else "") + (
            f"{escape_invisible(name)}:\n" + "\n".join(f"  {line}" for line in shown.split("\n"))
            if "\n" in shown
            else f"{escape_invisible(name)}: {shown}"
        )
    if not text:
        return head
    body = "\n".join(f"  {line}" if line else "" for line in text.split("\n"))
    return f"{head}:\n{body}"


#: What runners take: given the run's channel, build its watcher.
ParkWatchFactory = Callable[[ParkChannel], "AuthParkWatcher"]


def watch_factory(interactive: bool) -> ParkWatchFactory:
    """What runners take: build a watcher once they know their channel."""

    def build(channel: ParkChannel) -> AuthParkWatcher:
        return AuthParkWatcher(channel=channel, interactive=interactive)

    return build


#: Where each runtime's emitted workflow lives, for :func:`app_reports_gates`.
_WORKFLOW_FILES = (
    "main.py",
    "workflow.py",
    "src/main.ts",
    "src/workflow.ts",
    "src/inngest/pipeline.ts",
)


def app_reports_gates(app_dir: Path) -> bool:
    """Whether an emitted app reports the HITL gates it reaches.

    Lets runners build a watcher for a gated app that never touches MCP.
    """
    for name in _WORKFLOW_FILES:
        path = app_dir / name
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="replace")
            if GATE_STATUS_EVENT in text or GATE_PARK_MARKER in text:
                return True
    return False


def app_parks_on_auth(app_dir: Path) -> bool:
    """Whether an emitted app carries the MCP helper (so it can park).

    Lets runners skip the watcher, and its DBOS client, for the apps that
    never touch MCP.
    """
    return (app_dir / "extracted" / "_rote_mcp.py").is_file() or (
        app_dir / "src" / "extracted" / "_roteMcp.ts"
    ).is_file()


def watches_park(app_dir: Path) -> bool:
    """Whether a runner needs a watcher for this app: it can park on auth,
    or it reports the gates it reaches."""
    return app_parks_on_auth(app_dir) or app_reports_gates(app_dir)


# ───────── Listing parked runs across registered apps ─────────


@dataclass(frozen=True)
class ParkedWorkflow:
    app: Path
    runtime: str
    workflow_id: str
    server: str
    node: str | None


@dataclass(frozen=True)
class ParkedReport:
    parked: tuple[ParkedWorkflow, ...]
    skipped: tuple[Any, ...]
    """:class:`rote.mcp.release.SkippedApp` entries."""


def list_parked_workflows(server: str | None = None) -> ParkedReport:
    """Every registered DBOS workflow parked on auth, without releasing it.

    Only the DBOS runtimes can be listed: their park is a queryable
    workflow event. Inngest and Cloudflare apps are reported as skipped
    with the reason, since neither offers a way to ask which runs are
    waiting on which event (their release is a broadcast or a blast for
    exactly that reason).
    """
    from rote.mcp.release import SkippedApp, scan_dbos_apps

    parked: list[ParkedWorkflow] = []

    def visit(client: Any, app: Any) -> None:
        for wf in client.list_workflows(status="PENDING", load_input=False, load_output=False):
            status = client.get_event(wf.workflow_id, AUTH_STATUS_EVENT, timeout_seconds=0)
            if not (isinstance(status, dict) and status.get("awaiting")):
                continue
            awaiting = str(status["awaiting"])
            if server is not None and awaiting != server:
                continue
            node = status.get("node")
            parked.append(
                ParkedWorkflow(
                    app=app.path,
                    runtime=app.runtime,
                    workflow_id=wf.workflow_id,
                    server=awaiting,
                    node=str(node) if node else None,
                )
            )

    skipped = list(scan_dbos_apps(visit))
    from rote.app_registry import registered_apps

    for app in registered_apps():
        if app.runtime == "inngest":
            skipped.append(
                SkippedApp(app.path, "Inngest runs cannot be listed; rote mcp release wakes them")
            )
        elif app.runtime == "cloudflare":
            skipped.append(
                SkippedApp(
                    app.path, "Cloudflare instances cannot be listed; rote mcp release wakes them"
                )
            )
    return ParkedReport(parked=tuple(parked), skipped=tuple(skipped))


def wait_for_exit(
    proc: subprocess.Popen[Any],
    *,
    deadline: float,
    watcher: AuthParkWatcher | None,
    poll_interval: float = 0.5,
) -> None:
    """``proc.wait`` with the watcher ticking, for the DBOS runners.

    ``deadline`` is a ``time.monotonic()`` instant and moves out by the
    time the watcher spent waiting on the user. Raises
    ``subprocess.TimeoutExpired`` like ``proc.wait`` does.
    """
    if watcher is None:
        proc.wait(timeout=max(0.0, deadline - time.monotonic()))
        return
    while proc.poll() is None:
        if time.monotonic() >= deadline:
            raise subprocess.TimeoutExpired(proc.args, 0)
        deadline += watcher.tick()
        try:
            proc.wait(timeout=poll_interval)
        except subprocess.TimeoutExpired:
            continue
    watcher.drain()
