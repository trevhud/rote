"""``rote mcp login --no-wait`` / ``--complete``: an OAuth login an agent can drive.

A plain ``rote mcp login`` blocks until the browser returns to its loopback
callback. An agent running rote through a non-interactive shell tool cannot
hold that open while its user approves, so the login is split in two, in
the shape of Stripe CLI's pending login (``pkg/login/oauth_pending.go``):

1. ``--no-wait`` starts a **detached listener**: a short-lived child
   process (its own session, no terminal, stdio on ``/dev/null``) that runs
   the same :func:`rote.mcp.auth.login` a blocking login runs. It
   discovers the authorization server, registers a client if one is
   needed, binds the callback port (:func:`~rote.mcp.auth.bind_callback_sockets`
   via the usual :data:`~rote.mcp.auth.CALLBACK_PORTS` selection), and
   hands the authorization URL back to the command that started it over a
   pipe. That command prints the URL and a pending id and exits; the
   listener keeps waiting for the browser.
2. ``--complete [<id>]`` reads the pending record, waits a bounded time for
   the listener's outcome, and reports it. When the browser cannot reach
   this machine's loopback (it runs elsewhere), ``--callback-url`` takes
   the URL the browser ended on and replays it to the listener, which is
   the only process that can redeem it.

Why a listener rather than a paste-back that rebuilds the flow later: the
PKCE verifier and the ``state`` never leave the listener's memory. Nothing
on disk could redeem an authorization code, the record holds only what
``--complete`` needs to find and judge the login, and a pasted callback URL
is useless to anything but the listener (the code is single-use and bound
to that verifier). The listener also checks ``state`` before a callback
counts (:class:`rote.mcp.auth._StateCheckedReady`), so a forged request to
the port is dropped instead of ending the login.

The record (``<token dir>/pending-logins/<id>.json``) is written atomically
with mode 0600 in a 0700 directory. It expires after :data:`PENDING_TTL_SECONDS`:
the listener stops waiting at its deadline, and an expired record is
removed by the next ``--no-wait`` or ``--complete``. Deleting the record
cancels the login: the listener watches for it and exits. Its fields:
``id``, ``server``, ``server_url``, ``created_at``, ``expires_at``,
``status`` (``starting``, ``waiting``, ``done``, ``failed``), the listener's
``pid`` and ``port``, ``error``, a ``result`` summary, and
``rejected_callbacks`` (callbacks dropped for a wrong ``state``).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import queue
import re
import secrets
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import parse_qs, urlsplit

from rote.interact.plan import atomic_write_text
from rote.mcp.tokens import token_dir

if TYPE_CHECKING:
    from rote.mcp.registry import McpServerConfig

#: How long a pending login lives: the user's time to approve in the browser.
PENDING_TTL_SECONDS = 600
#: How long ``--no-wait`` waits for the listener to reach the authorization
#: URL (discovery, the CIMD check and a registration happen before it).
START_TIMEOUT_SECONDS = 60.0
#: The listener stops this long before the record expires, so an expired
#: record never belongs to a listener that is still running.
_LISTENER_MARGIN_SECONDS = 2.0
#: How often the listener checks that its record still exists.
_WATCH_INTERVAL_SECONDS = 0.25

Status = Literal["starting", "waiting", "done", "failed"]
TERMINAL: frozenset[str] = frozenset({"done", "failed"})

_ID_RE = re.compile(r"^[a-z0-9]{16}$")


class PendingLoginError(RuntimeError):
    """A pending login that cannot be started, found or completed. User-actionable."""


def pending_dir() -> Path:
    """Where pending records live: beside the token files they lead to."""
    return token_dir() / "pending-logins"


def new_pending_id() -> str:
    return secrets.token_hex(8)


def valid_pending_id(pending_id: str) -> bool:
    return bool(_ID_RE.match(pending_id))


def pending_path(pending_id: str) -> Path:
    if not valid_pending_id(pending_id):
        # Ids name files; never let one be a path.
        raise PendingLoginError(f"{pending_id!r} is not a pending login id")
    return pending_dir() / f"{pending_id}.json"


@dataclass
class PendingLogin:
    id: str
    server: str
    server_url: str
    created_at: float
    expires_at: float
    status: Status = "starting"
    pid: int | None = None
    port: int | None = None
    error: str | None = None
    result: dict[str, Any] | None = None
    rejected_callbacks: int = 0
    #: Not stored: set by :func:`read_pending` from the file it read.
    path: Path | None = field(default=None, compare=False, repr=False)

    @classmethod
    def new(cls, server: str, server_url: str, *, now: float | None = None) -> PendingLogin:
        created = time.time() if now is None else now
        return cls(
            id=new_pending_id(),
            server=server,
            server_url=server_url,
            created_at=created,
            expires_at=created + PENDING_TTL_SECONDS,
        )

    @property
    def file(self) -> Path:
        return pending_path(self.id)

    def expired(self, now: float | None = None) -> bool:
        return (time.time() if now is None else now) >= self.expires_at

    @property
    def live(self) -> bool:
        return self.status not in TERMINAL and not self.expired()

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("path")
        return data


def write_pending(pending: PendingLogin) -> Path:
    """Write the record atomically, owner-only, in an owner-only directory."""
    directory = pending_dir()
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(directory, 0o700)
    path = pending.file
    atomic_write_text(path, json.dumps(pending.to_json(), indent=2) + "\n", file_mode=0o600)
    return path


def read_pending(pending_id: str) -> PendingLogin | None:
    path = pending_path(pending_id)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as e:
        raise PendingLoginError(f"the pending login record {path} is unreadable: {e}") from e
    known = {k: v for k, v in data.items() if k in PendingLogin.__dataclass_fields__}
    known.pop("path", None)
    try:
        pending = PendingLogin(**known)
    except TypeError as e:
        raise PendingLoginError(f"the pending login record {path} is malformed: {e}") from e
    if pending.id != pending_id:
        raise PendingLoginError(f"the pending login record {path} names another id")
    pending.path = path
    return pending


def delete_pending(pending_id: str) -> bool:
    try:
        pending_path(pending_id).unlink()
        return True
    except FileNotFoundError:
        return False


def list_pending() -> list[PendingLogin]:
    """Every readable record, oldest first. Unreadable files are skipped."""
    directory = pending_dir()
    if not directory.is_dir():
        return []
    found: list[PendingLogin] = []
    for path in sorted(directory.glob("*.json")):
        try:
            pending = read_pending(path.stem)
        except PendingLoginError:
            continue
        if pending is not None:
            found.append(pending)
    return sorted(found, key=lambda p: p.created_at)


def sweep_expired(now: float | None = None) -> list[str]:
    """Remove expired records (their listeners have stopped). Returns their ids."""
    removed = []
    for pending in list_pending():
        if pending.expired(now) and delete_pending(pending.id):
            removed.append(pending.id)
    return removed


def live_for_server(server: str) -> PendingLogin | None:
    for pending in list_pending():
        if pending.server == server and pending.live:
            return pending
    return None


def _pid_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# ── start ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Started:
    """What ``--no-wait`` got back from its listener."""

    #: ``authorize``: the user must approve at ``authorization_url``.
    #: ``done``: nothing to approve (a stored token still worked, or the
    #: server asks for no auth); ``result`` says which.
    kind: Literal["authorize", "done"]
    pending: PendingLogin
    authorization_url: str | None = None
    result: dict[str, Any] | None = None


def start_pending_login(
    server: str, config: McpServerConfig, *, timeout: float = START_TIMEOUT_SECONDS
) -> Started:
    """Start the detached listener and wait until it has an authorization URL.

    Raises :class:`PendingLoginError` when a login for ``server`` is already
    pending, and :class:`rote.mcp.auth.McpAuthError` with the listener's
    reason when the login failed before it reached the browser step.
    """
    from rote.mcp.auth import McpAuthError

    sweep_expired()
    existing = live_for_server(server)
    if existing is not None:
        raise PendingLoginError(
            f"a login for {server!r} is already pending (id {existing.id}, expires "
            f"{_clock(existing.expires_at)}). Finish it: rote mcp login --complete "
            f"{existing.id}, or cancel it: rote mcp login --cancel {existing.id}"
        )
    pending = PendingLogin.new(server, config.url)
    write_pending(pending)
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "rote.mcp.pending_login", pending.id],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            # The listener's output is never shown or kept: fastmcp logs the
            # authorization URL, and nobody reads a detached process's stderr.
            stderr=subprocess.DEVNULL,
            # Its own session: the shell tool that ran this command can end,
            # and its process group with it, while the listener waits.
            start_new_session=True,
            close_fds=True,
        )
    except OSError:
        delete_pending(pending.id)
        raise
    assert proc.stdout is not None
    lines: queue.Queue[str] = queue.Queue()

    def read_one() -> None:
        assert proc.stdout is not None
        lines.put(proc.stdout.readline().decode("utf-8", "replace"))

    threading.Thread(target=read_one, daemon=True).start()
    try:
        line = lines.get(timeout=timeout)
    except queue.Empty:
        line = ""
        why = f"the login listener did not reach the authorization step within {timeout:g}s"
    else:
        why = "the login listener stopped before reaching the authorization step"
    finally:
        proc.stdout.close()
    try:
        message = json.loads(line) if line.strip() else None
    except ValueError:
        message = None
    if not isinstance(message, dict):
        delete_pending(pending.id)  # stops a listener that is still running
        with contextlib.suppress(OSError):
            proc.kill()
        raise McpAuthError(f"{why}. Retry: rote mcp login {server} --no-wait")
    event = message.get("event")
    if event == "authorize":
        current = read_pending(pending.id) or pending
        return Started("authorize", current, authorization_url=str(message["url"]))
    if event == "done":
        delete_pending(pending.id)
        return Started("done", pending, result=message.get("result"))
    delete_pending(pending.id)
    raise McpAuthError(str(message.get("error") or why))


def _clock(epoch: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(epoch))


# ── complete ─────────────────────────────────────────────────────────────


def resolve_pending(pending_id: str | None, server: str | None) -> PendingLogin:
    """The record ``--complete`` / ``--cancel`` acts on.

    With an id, that record. Without one, the only pending login (for
    ``server`` when one is named); several is an error listing them.
    """
    if pending_id:
        pending = read_pending(pending_id)
        if pending is None:
            raise PendingLoginError(
                f"no pending login {pending_id}: it was completed, cancelled, or expired "
                "and removed. Start a new one: rote mcp login <server> --no-wait"
            )
        if server is not None and pending.server != server:
            raise PendingLoginError(
                f"pending login {pending_id} is for {pending.server!r}, not {server!r}"
            )
        return pending
    candidates = [p for p in list_pending() if server is None or p.server == server]
    if not candidates:
        scope = f" for {server!r}" if server else ""
        raise PendingLoginError(
            f"no pending login{scope}. Start one: rote mcp login {server or '<server>'} --no-wait"
        )
    if len(candidates) > 1:
        listed = ", ".join(f"{p.id} ({p.server})" for p in candidates)
        raise PendingLoginError(f"several logins are pending ({listed}); name one by id")
    return candidates[0]


def replay_callback(pending: PendingLogin, callback_url: str) -> None:
    """Deliver the URL a browser on another machine ended on to the listener.

    Only a loopback ``/callback`` URL on the listener's own port is sent,
    so this can never be pointed at another host. And only while this
    login's listener is alive and waiting: once it has stopped, its port is
    free, and whatever binds it next would receive the code and state.
    """
    import httpx

    parts = urlsplit(callback_url)
    if parts.scheme != "http" or parts.hostname not in ("localhost", "127.0.0.1"):
        raise PendingLoginError(
            "--callback-url must be the http://localhost:<port>/callback?... URL the "
            "browser was sent back to"
        )
    try:
        port = parts.port
    except ValueError as e:
        raise PendingLoginError(f"--callback-url has an invalid port ({e})") from e
    current = read_pending(pending.id)
    if current is None or current.status != "waiting" or current.expired():
        state = "gone" if current is None else ("expired" if current.expired() else current.status)
        raise PendingLoginError(
            f"pending login {pending.id} is {state}, so its listener no longer waits for "
            f"a callback; nothing was sent. Start again: rote mcp login {pending.server} "
            "--no-wait"
        )
    if not _pid_alive(current.pid):
        raise PendingLoginError(
            f"the listener for pending login {pending.id} has stopped; nothing was sent. "
            f"Start again: rote mcp login {pending.server} --no-wait"
        )
    pending = current
    if pending.port is None or port != pending.port or parts.path != "/callback":
        raise PendingLoginError(
            f"--callback-url is not this login's callback (expected port {pending.port}, "
            "path /callback): was it copied from another login?"
        )
    if not parse_qs(parts.query).get("code") and not parse_qs(parts.query).get("error"):
        raise PendingLoginError("--callback-url carries no code: copy the whole URL")
    target = f"http://127.0.0.1:{pending.port}/callback?{parts.query}"
    try:
        httpx.get(target, timeout=10, follow_redirects=False)
    except httpx.HTTPError as e:
        raise PendingLoginError(
            f"the login listener for {pending.server!r} did not answer ({type(e).__name__}); "
            "it may have stopped. Start again: rote mcp login "
            f"{pending.server} --no-wait"
        ) from e


@dataclass(frozen=True)
class Completion:
    kind: Literal["done", "failed", "waiting", "expired"]
    pending: PendingLogin


def wait_for_outcome(
    pending: PendingLogin, *, wait: float, poll: float = 0.2, now: Any = time.time
) -> Completion:
    """Wait up to ``wait`` seconds for the listener's outcome.

    Terminal and expired records are removed here: each id is completed
    once. A waiting record is kept, so ``--complete`` can be run again.
    """
    deadline = time.monotonic() + max(wait, 0.0)
    current = pending
    while True:
        latest = read_pending(current.id)
        if latest is None:
            raise PendingLoginError(
                f"pending login {current.id} was removed while waiting (cancelled, or "
                "completed by another command)"
            )
        current = latest
        if current.status in TERMINAL:
            delete_pending(current.id)
            return Completion("done" if current.status == "done" else "failed", current)
        if current.expired(now()):
            delete_pending(current.id)
            return Completion("expired", current)
        if current.status == "waiting" and not _pid_alive(current.pid):
            delete_pending(current.id)
            current.status = "failed"
            current.error = "the login listener stopped without a result"
            return Completion("failed", current)
        if time.monotonic() >= deadline:
            return Completion("waiting", current)
        time.sleep(poll)


# ── the listener (python -m rote.mcp.pending_login <id>) ────────────────


def _update(pending_id: str, **changes: Any) -> None:
    """Change the record, unless it is gone (gone means cancelled)."""
    current = read_pending(pending_id)
    if current is None:
        return
    for key, value in changes.items():
        setattr(current, key, value)
    write_pending(current)


def result_summary(doc: dict[str, Any] | None, config: McpServerConfig) -> dict[str, Any]:
    """What a finished login stored, without a single credential in it."""
    from rote.mcp.auth import registration_of

    if doc is None:
        return {"auth_required": False}
    return {
        "auth_required": True,
        "registration": registration_of(doc, config),
        "issuer": doc.get("issuer"),
        "refresh_token": bool((doc.get("tokens") or {}).get("refresh_token")),
        "expires_at": doc.get("expires_at"),
    }


def _redact(text: str, secrets_seen: list[str]) -> str:
    for value in secrets_seen:
        if value:
            text = text.replace(value, "<redacted>")
    return text


class _Deadline(Exception):
    """The record's expiry arrived while the listener was still working."""


def _caused_by_timeout(exc: BaseException) -> bool:
    """Whether a TimeoutError is anywhere in ``exc``'s chain or groups
    (fastmcp wraps the callback timeout in its own connection error)."""
    stack: list[BaseException] = [exc]
    seen: set[int] = set()
    while stack:
        e = stack.pop()
        if id(e) in seen:
            continue
        seen.add(id(e))
        if isinstance(e, TimeoutError):
            return True
        if isinstance(e, BaseExceptionGroup):
            stack.extend(e.exceptions)
        stack.extend(x for x in (e.__cause__, e.__context__) if x is not None)
    return False


def _listen(pending_id: str) -> int:
    from rote.mcp.auth import authorization_state, login
    from rote.mcp.registry import load_registry

    handshake = sys.stdout
    announced = False
    seen: list[str] = []

    def send(message: dict[str, Any]) -> None:
        nonlocal handshake
        if handshake is None:
            return
        with contextlib.suppress(OSError, ValueError):
            handshake.write(json.dumps(message) + "\n")
            handshake.flush()
            handshake.close()
        handshake = None
        # Nothing may reach the pipe again: its reader has gone.
        sys.stdout = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115

    pending = read_pending(pending_id)
    if pending is None:
        return 1
    _update(pending_id, status="waiting", pid=os.getpid())

    async def handoff(url: str) -> None:
        nonlocal announced
        state = authorization_state(url)
        if state:
            seen.append(state)
        redirect = (parse_qs(urlsplit(url).query).get("redirect_uri") or [""])[0]
        port = urlsplit(redirect).port
        _update(pending_id, port=port)
        announced = True
        send({"event": "authorize", "url": url, "port": port})

    def rejected() -> None:
        current = read_pending(pending_id)
        if current is not None:
            _update(pending_id, rejected_callbacks=current.rejected_callbacks + 1)

    async def run() -> dict[str, Any] | None:
        config = load_registry().servers.get(pending.server)
        if config is None or config.url != pending.server_url:
            raise PendingLoginError(
                f"{pending.server!r} is no longer registered at the URL this login started for"
            )
        timeout = pending.expires_at - time.time() - _LISTENER_MARGIN_SECONDS
        task = asyncio.ensure_future(
            login(
                pending.server,
                config,
                handoff=handoff,
                callback_timeout=max(timeout, 1.0),
                on_rejected_callback=rejected,
            )
        )
        deadline = pending.expires_at - _LISTENER_MARGIN_SECONDS
        while not task.done():
            await asyncio.wait({task}, timeout=_WATCH_INTERVAL_SECONDS)
            if task.done():
                break
            # The callback server's own timeout starts only once discovery,
            # registration and the preflight are done; this deadline holds
            # from the start, so an expired record never has a live listener.
            expired = time.time() >= deadline
            if expired or not pending_path(pending_id).exists():
                task.cancel()  # the record was deleted: the login was cancelled
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
                if expired:
                    raise _Deadline
                raise asyncio.CancelledError
        doc = task.result()
        return result_summary(doc, config)

    try:
        summary = asyncio.run(run())
    except _Deadline:
        error = "it expired before the browser returned"
        _update(pending_id, status="failed", error=error)
        send({"event": "failed", "error": error})
        return 1
    except asyncio.CancelledError:
        send({"event": "failed", "error": "the login was cancelled"})
        return 1
    except Exception as e:  # the record carries the reason; nothing else would see it
        if announced and _caused_by_timeout(e):
            # The callback server's deadline, which is the record's expiry.
            error = "it expired before the browser returned"
            _update(pending_id, status="failed", error=error)
            send({"event": "failed", "error": error})
            return 1
        error = _redact(str(e) or type(e).__name__, seen)
        if not isinstance(e, (PendingLoginError, RuntimeError)) or type(e) is RuntimeError:
            error = f"{type(e).__name__}: {error}"
        _update(pending_id, status="failed", error=error)
        send({"event": "failed", "error": error})
        return 1
    _update(pending_id, status="done", result=summary)
    if not announced:
        send({"event": "done", "result": summary})
    return 0


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1 or not valid_pending_id(args[0]):
        print("usage: python -m rote.mcp.pending_login <pending id>", file=sys.stderr)
        return 2
    return _listen(args[0])


if __name__ == "__main__":
    sys.exit(main())
