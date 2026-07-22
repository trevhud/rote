"""Server-side ("cloud") graduation client for ``rote graduate``.

When logged in to rote cloud, ``rote graduate`` runs the graduation on
the platform instead of on the user's machine: the skill bundle is
synced up, a graduation job is started, its live progress is streamed
back (the same :class:`~rote.graduator.events.GraduationEvent` wire
schema the local path emits), and the finished artifacts are downloaded
into the ``--out`` directory. Full-mode graduations auto-deploy on the
server, so the CLI never runs the local ``deploy_rote_cloud`` leg.

This module is the transport layer for that flow — stdlib ``urllib``
only, module-level functions, style-matched to
:mod:`rote.deploy_rote_cloud`. Endpoint + token resolution reuses the
``deploy_rote_cloud`` precedence (flag > ``ROTE_CLOUD_*`` env > stored
login). Every endpoint lives under ``{url}/v1``, speaks JSON, and
authenticates with ``Authorization: Bearer rote_…``. Server error bodies
are ``{"error": "…"}`` (some also carry a ``"code"``).

``urlopen`` is imported at module scope so tests can stub
``rote.cloud_graduate.urlopen`` the way ``tests/test_deploy.py`` stubs
``rote.deploy_rote_cloud.urlopen``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import time
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast, get_args
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from rote.graduator.events import EventCallback, EventType, GraduationEvent, emit_safely

#: Client-side bundle caps (enforced before any upload, with actionable
#: errors) mirroring what the platform accepts.
MAX_BUNDLE_FILES = 300
MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_BUNDLE_BYTES = 25 * 1024 * 1024

#: Directory/file names never uploaded as part of a skill bundle.
_SKIP_DIR_NAMES = frozenset({".git", "node_modules", "__pycache__"})

#: Bundle-root paths the server reserves and rejects in ``files``:
#: ``SKILL.md`` rides in ``skill_md``, ``bundle.json`` is the server's manifest.
_RESERVED_BUNDLE_PATHS = frozenset({"SKILL.md", "bundle.json"})

#: The event ``type`` values the wire schema allows; anything else in a
#: server event is treated as garbage and skipped by :func:`_event_from_wire`.
_EVENT_TYPES = frozenset(get_args(EventType))


class CloudGraduateError(RuntimeError):
    """A server-side graduation failed. ``exit_code`` is the CLI status."""

    def __init__(self, message: str, *, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code = exit_code


class ActiveGraduationExists(CloudGraduateError):
    """A graduation is already running for this skill.

    Raised *before* any mutation so the caller can attach to the running
    job instead of starting (or failing) a new one.
    """

    def __init__(
        self,
        message: str,
        *,
        graduation_id: str | None,
        skill_id: str | None,
        exit_code: int = 1,
    ) -> None:
        super().__init__(message, exit_code=exit_code)
        self.graduation_id = graduation_id
        self.skill_id = skill_id


class NoChanges(CloudGraduateError):
    """Update-mode no-op — the caller treats this as success (exit 0)."""


@dataclass(frozen=True)
class CloudEndpoint:
    """A resolved rote-cloud base URL + tenant bearer token."""

    url: str
    token: str


def resolve_cloud_endpoint(
    url_flag: str | None = None, token_flag: str | None = None
) -> CloudEndpoint:
    """Resolve the endpoint + token via the ``deploy_rote_cloud`` chain.

    Reuses :func:`rote.deploy_rote_cloud.resolve_endpoint` /
    ``resolve_token`` so cloud graduation and cloud deploy share one
    precedence. A missing credential is a usage error (exit 2): the fix
    is ``rote login`` or ``ROTE_CLOUD_*`` env, not a retry.
    """
    from rote.deploy import DeployError
    from rote.deploy_rote_cloud import resolve_endpoint, resolve_token

    try:
        url = resolve_endpoint(url_flag)
        token = resolve_token(token_flag)
    except DeployError as e:
        raise CloudGraduateError(str(e), exit_code=2) from e
    return CloudEndpoint(url=url, token=token)


def _request_json(
    method: str,
    url: str,
    body: dict[str, Any] | None = None,
    *,
    token: str,
    timeout: float = 60.0,
) -> tuple[int, dict[str, Any]]:
    """``(status, parsed body)`` — 4xx bodies are data here, not exceptions.

    Modeled on :func:`rote.cloud_auth._request_json`: the platform
    signals ``insufficient_credits`` / ``no_changes`` / ``unknown model``
    as 4xx JSON, so the transport hands those back for interpretation.
    Only a transport failure (:class:`URLError`) raises — as a
    :class:`CloudGraduateError` the caller can catch (poll treats it as a
    transient network blip).
    """
    req = Request(
        url,
        data=json.dumps(body).encode("utf-8") if body is not None else None,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
        method=method,
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8") or "{}")
    except HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, {"error": f"http_{e.code}", "detail": raw[:300]}
    except URLError as e:
        raise CloudGraduateError(f"could not reach {url}: {e.reason}") from e


# ───────────────────────── models ─────────────────────────


def fetch_models(ep: CloudEndpoint) -> dict[str, Any]:
    """``{models: [{id, label, …}], default}`` from the server."""
    status, doc = _request_json("GET", f"{ep.url}/v1/graduations/models", token=ep.token)
    if status != 200:
        raise CloudGraduateError(
            f"could not fetch graduation models (HTTP {status}): {doc.get('error') or doc}"
        )
    return doc


def resolve_model(ep: CloudEndpoint, model_flag: str | None) -> str:
    """Validate ``model_flag`` against the server lineup, or take the default.

    ``None`` resolves to the server's ``default`` (else the first offered
    id). An unknown id is a usage error (exit 2) whose message lists the
    valid ids.
    """
    doc = fetch_models(ep)
    ids = [str(m["id"]) for m in doc.get("models") or [] if isinstance(m, dict) and m.get("id")]
    if model_flag is None:
        default = doc.get("default")
        if default:
            return str(default)
        if ids:
            return ids[0]
        raise CloudGraduateError(f"{ep.url} offered no graduation models")
    if model_flag in ids:
        return model_flag
    offered = ", ".join(ids) if ids else "(none)"
    raise CloudGraduateError(
        f"unknown model {model_flag!r} — {ep.url} offers: {offered}", exit_code=2
    )


# ───────────────────────── skill sync ─────────────────────────


def _skill_name_from_md(text: str, fallback: str) -> str:
    """Read ``name:`` from a SKILL.md YAML frontmatter block, else fallback.

    The source skill's identity on the platform is its frontmatter
    ``name`` (the same field the local path keys on); a skill without a
    frontmatter ``name`` falls back to the directory name.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return fallback
    for line in lines[1:]:
        if line.strip() == "---":
            break
        m = re.match(r"\s*name\s*:\s*(.+?)\s*$", line)
        if m:
            value = m.group(1).strip().strip("'\"")
            if value:
                return value
    return fallback


def _collect_bundle_files(skill_dir: Path) -> list[tuple[str, bytes]]:
    """``[(posix_relpath, bytes)]`` for every supporting file to upload.

    Skips ``.git`` / ``node_modules`` / ``__pycache__``, any hidden file
    or directory (a path component starting with ``.``), and the two
    bundle-root paths the server reserves: ``SKILL.md`` (uploaded
    separately as ``skill_md``) and ``bundle.json`` (the server's own
    manifest). Every remaining file — text or binary — is read as raw
    bytes for a binary-safe base64 upload. Sorted by path for a
    deterministic bundle hash.
    """
    files: list[tuple[str, bytes]] = []
    for path in sorted(skill_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(skill_dir)
        if any(part in _SKIP_DIR_NAMES or part.startswith(".") for part in rel.parts):
            continue
        posix = rel.as_posix()
        if posix in _RESERVED_BUNDLE_PATHS:
            continue
        files.append((posix, path.read_bytes()))
    return files


def _enforce_bundle_caps(files: list[tuple[str, bytes]]) -> None:
    if len(files) > MAX_BUNDLE_FILES:
        raise CloudGraduateError(
            f"skill bundle has {len(files)} files, over the {MAX_BUNDLE_FILES}-file limit "
            "— trim the bundle or graduate with --local",
            exit_code=2,
        )
    total = 0
    for posix, content in files:
        if len(content) > MAX_FILE_BYTES:
            raise CloudGraduateError(
                f"{posix} is {len(content) / 1024 / 1024:.1f} MB, over the "
                f"{MAX_FILE_BYTES // 1024 // 1024} MB per-file limit — graduate with --local",
                exit_code=2,
            )
        total += len(content)
    if total > MAX_BUNDLE_BYTES:
        raise CloudGraduateError(
            f"skill bundle is {total / 1024 / 1024:.1f} MB, over the "
            f"{MAX_BUNDLE_BYTES // 1024 // 1024} MB total limit — graduate with --local",
            exit_code=2,
        )


def _bundle_sha256(files: list[tuple[str, bytes]]) -> str | None:
    """Deterministic bundle digest, or ``None`` when there are no files.

    ``sha256`` of the concatenation of ``"{path}\\n{sha256(content)}\\n"``
    over files sorted by path (hex digests throughout) — the exact recipe
    the server uses, so a matching digest means an identical bundle and
    the upload can be skipped.
    """
    if not files:
        return None
    h = hashlib.sha256()
    for posix, content in sorted(files, key=lambda f: f[0]):
        h.update(f"{posix}\n{hashlib.sha256(content).hexdigest()}\n".encode())
    return h.hexdigest()


def _encode_files(files: list[tuple[str, bytes]]) -> list[dict[str, str]]:
    return [{"path": p, "content_b64": base64.b64encode(c).decode("ascii")} for p, c in files]


def sync_skill(ep: CloudEndpoint, skill_dir: Path) -> str:
    """Ensure the skill exists on the server with the local bundle; return its id.

    Reads ``SKILL.md`` (error if missing), collects supporting files,
    enforces the bundle caps, then reconciles with the server:

    * If a skill of the same name has an active graduation →
      :class:`ActiveGraduationExists` is raised **before any mutation**.
    * If it exists and both the skill_md hash and bundle hash already
      match → no upload (return the id).
    * If it exists but differs → ``PUT`` the full bundle.
    * If it does not exist → ``POST`` a new skill.
    """
    skill_md_path = skill_dir / "SKILL.md"
    if not skill_md_path.is_file():
        raise CloudGraduateError(f"{skill_dir} has no SKILL.md — not a skill bundle", exit_code=2)
    skill_md = skill_md_path.read_text(encoding="utf-8")
    name = _skill_name_from_md(skill_md, fallback=skill_dir.resolve().name)
    files = _collect_bundle_files(skill_dir)
    _enforce_bundle_caps(files)

    status, doc = _request_json("GET", f"{ep.url}/v1/skills", token=ep.token)
    if status != 200:
        raise CloudGraduateError(
            f"could not list skills (HTTP {status}): {doc.get('error') or doc}"
        )
    existing: dict[str, Any] | None = None
    for entry in doc.get("skills") or []:
        if isinstance(entry, dict) and entry.get("name") == name:
            existing = entry
            break

    if existing is not None:
        skill_id = str(existing.get("id"))
        active = existing.get("active_graduation")
        if active is not None:
            grad_id = active.get("id") if isinstance(active, dict) else active
            raise ActiveGraduationExists(
                f"skill {name!r} has an active graduation",
                graduation_id=str(grad_id) if grad_id is not None else None,
                skill_id=skill_id,
            )
        local_bundle = _bundle_sha256(files)
        local_md = hashlib.sha256(skill_md.encode("utf-8")).hexdigest()
        bstatus, bdoc = _request_json(
            "GET", f"{ep.url}/v1/skills/{skill_id}/bundle", token=ep.token
        )
        remote_bundle = bdoc.get("bundle_sha256") if bstatus == 200 else None
        if existing.get("skill_md_sha256") == local_md and remote_bundle == local_bundle:
            return skill_id
        pstatus, pdoc = _request_json(
            "PUT",
            f"{ep.url}/v1/skills/{skill_id}/bundle",
            {"skill_md": skill_md, "files": _encode_files(files)},
            token=ep.token,
        )
        if pstatus == 409:
            raise ActiveGraduationExists(
                f"skill {name!r} started an active graduation — cannot update its bundle",
                graduation_id=str(pdoc.get("graduation_id")) if pdoc.get("graduation_id") else None,
                skill_id=skill_id,
            )
        if pstatus not in (200, 201):
            raise CloudGraduateError(
                f"could not update skill bundle (HTTP {pstatus}): {pdoc.get('error') or pdoc}"
            )
        return skill_id

    cstatus, cdoc = _request_json(
        "POST",
        f"{ep.url}/v1/skills",
        {"name": name, "skill_md": skill_md, "files": _encode_files(files)},
        token=ep.token,
    )
    if cstatus == 409:
        raise CloudGraduateError(f"a skill named {name!r} already exists on {ep.url}")
    if cstatus not in (200, 201):
        raise CloudGraduateError(
            f"could not create skill (HTTP {cstatus}): {cdoc.get('error') or cdoc}"
        )
    new_id = cdoc.get("id")
    if not new_id:
        raise CloudGraduateError(f"{ep.url} did not return a skill id on create")
    return str(new_id)


# ───────────────────────── graduation lifecycle ─────────────────────────


def start_graduation(ep: CloudEndpoint, skill_id: str, *, mode: str, model: str) -> dict[str, Any]:
    """Start a graduation; return its record ``{id, skill_id, status, mode}``.

    Maps the platform's refusal codes to actionable CLI errors:
    402 → top-up hint; 503 → "not enabled, run --local"; 409/no_changes
    → :class:`NoChanges` (caller exits 0); 409 → :class:`ActiveGraduationExists`
    (caller attaches); 400 → unknown model (usage error).
    """
    status, doc = _request_json(
        "POST",
        f"{ep.url}/v1/skills/{skill_id}/graduations",
        {"mode": mode, "model": model},
        token=ep.token,
    )
    if status in (200, 201):
        return doc
    code = doc.get("code")
    if status == 402:
        raise CloudGraduateError(f"insufficient credits — top up at {ep.url}/billing")
    if status == 503:
        raise CloudGraduateError(
            f"server-side graduation is not enabled on {ep.url} — run with --local"
        )
    if status == 409 and code == "no_changes":
        raise NoChanges("no changes to graduate")
    if status == 409:
        grad = doc.get("graduation")
        grad_id = doc.get("graduation_id") or (grad.get("id") if isinstance(grad, dict) else None)
        raise ActiveGraduationExists(
            f"skill {skill_id} has an active graduation",
            graduation_id=str(grad_id) if grad_id is not None else None,
            skill_id=skill_id,
        )
    if status == 400:
        raise CloudGraduateError(str(doc.get("error") or "bad graduation request"), exit_code=2)
    raise CloudGraduateError(
        f"could not start graduation (HTTP {status}): {doc.get('error') or doc}"
    )


def _str_or_none(value: Any) -> str | None:
    return str(value) if value is not None else None


def _opt_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _event_from_wire(wire: Any) -> GraduationEvent | None:
    """Reconstruct a :class:`GraduationEvent` from a wire dict, tolerantly.

    The wire form is ``dataclasses.asdict(GraduationEvent(...))``. This is
    deliberately forgiving — a half-written or malformed event from the
    server must be skipped (``None``), never crash a poll loop watching a
    paid run.
    """
    if not isinstance(wire, dict):
        return None
    etype = wire.get("type")
    if etype not in _EVENT_TYPES:
        return None
    ts_raw = wire.get("ts")
    try:
        ts = float(ts_raw) if ts_raw is not None else time.time()
    except (TypeError, ValueError):
        ts = time.time()
    tokens = wire.get("tokens")
    if not (isinstance(tokens, dict) and all(isinstance(v, int) for v in tokens.values())):
        tokens = None
    try:
        return GraduationEvent(
            type=cast(EventType, etype),
            ts=ts,
            phase=_opt_int(wire.get("phase")),
            phase_name=_str_or_none(wire.get("phase_name")),
            turn=_opt_int(wire.get("turn")),
            tokens=tokens,
            tool_name=_str_or_none(wire.get("tool_name")),
            path=_str_or_none(wire.get("path")),
            message=str(wire.get("message") or ""),
        )
    except (TypeError, ValueError):
        return None


def _cursor_advances(old: Any, new: Any) -> bool:
    """Whether ``new`` is a strictly later cursor than ``old``.

    Cursors are opaque; when they are comparable (ints, sortable strings)
    only a forward move is honored so a stale/duplicate page can't rewind
    us and replay events. Incomparable types replace unconditionally.
    """
    try:
        return bool(new > old)
    except TypeError:
        return True


def poll_graduation(
    ep: CloudEndpoint,
    graduation_id: str,
    on_event: EventCallback | None,
    *,
    interval: float = 2.0,
    timeout: float = 45 * 60,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Poll a graduation to a terminal state, replaying events as they land.

    ``GET /v1/graduations/:id?after=<cursor>`` each tick; every decoded
    event fires ``on_event`` (guarded — a UI bug can't kill a paid run);
    the cursor only advances forward. Returns the final doc when
    ``status`` is terminal (``complete`` / ``error`` / ``canceled``).

    A transient transport failure or 5xx backs off exponentially (2→30s)
    and gives up after 5 consecutive failures with a re-attach hint — the
    job keeps running server-side. ``KeyboardInterrupt`` propagates so the
    caller can detach or cancel.
    """
    cursor: Any = None
    deadline = time.monotonic() + timeout
    failures = 0
    backoff = 2.0
    while True:
        if time.monotonic() > deadline:
            raise CloudGraduateError(
                f"graduation {graduation_id} did not finish within {timeout / 60:.0f} min — "
                f"it continues server-side; re-run rote graduate to re-attach"
            )
        url = f"{ep.url}/v1/graduations/{graduation_id}"
        if cursor is not None:
            url += f"?after={urllib.parse.quote(str(cursor))}"
        try:
            status: int | None
            status, doc = _request_json("GET", url, token=ep.token)
        except CloudGraduateError:
            status, doc = None, {}
        if status is None or status >= 500:
            failures += 1
            if failures >= 5:
                raise CloudGraduateError(
                    f"lost contact with {ep.url} — the graduation continues server-side; "
                    "re-run rote graduate to re-attach, or watch it in the web app"
                )
            sleep(backoff)
            backoff = min(backoff * 2, 30.0)
            continue
        if status != 200:
            raise CloudGraduateError(
                f"could not poll graduation (HTTP {status}): {doc.get('error') or doc}"
            )
        failures = 0
        backoff = 2.0
        for wire in doc.get("events") or []:
            event = _event_from_wire(wire)
            if event is not None:
                emit_safely(on_event, event)
        new_cursor = doc.get("cursor")
        if new_cursor is not None and (cursor is None or _cursor_advances(cursor, new_cursor)):
            cursor = new_cursor
        if doc.get("status") in ("complete", "error", "canceled"):
            return doc
        sleep(interval)


def cancel_graduation(ep: CloudEndpoint, graduation_id: str) -> bool:
    """Best-effort ``DELETE`` of a running graduation. Never raises."""
    try:
        status, _doc = _request_json(
            "DELETE", f"{ep.url}/v1/graduations/{graduation_id}", token=ep.token
        )
    except CloudGraduateError:
        return False
    return status in (200, 204)


# ───────────────────────── artifact download ─────────────────────────


def _fetch_file(ep: CloudEndpoint, graduation_id: str, rel: str) -> str:
    """Download one artifact's text content, with a single retry."""
    quoted = urllib.parse.quote(rel)  # default safe='/' keeps path separators
    url = f"{ep.url}/v1/graduations/{graduation_id}/files/{quoted}"
    last: CloudGraduateError | None = None
    for _attempt in range(2):
        try:
            status, doc = _request_json("GET", url, token=ep.token)
        except CloudGraduateError as e:
            last = e
            continue
        if status == 200:
            return str(doc.get("content") or "")
        last = CloudGraduateError(
            f"could not download {rel} (HTTP {status}): {doc.get('error') or doc}"
        )
    assert last is not None
    raise last


def download_artifacts(ep: CloudEndpoint, graduation_id: str, out_dir: Path) -> dict[str, Path]:
    """Download every graduation artifact into ``out_dir``; return relpath→abs.

    Paths come from the server; any absolute path or one containing a
    ``..`` component is rejected outright (path-traversal guard) before a
    single byte is written. Text content is written verbatim, parents
    created as needed.
    """
    status, doc = _request_json(
        "GET", f"{ep.url}/v1/graduations/{graduation_id}/files", token=ep.token
    )
    if status != 200:
        raise CloudGraduateError(
            f"could not list graduation files (HTTP {status}): {doc.get('error') or doc}"
        )
    written: dict[str, Path] = {}
    for entry in doc.get("files") or []:
        if not isinstance(entry, dict):
            continue
        rel = entry.get("path")
        if not isinstance(rel, str) or not rel:
            continue
        candidate = Path(rel)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise CloudGraduateError(f"refusing unsafe artifact path from server: {rel!r}")
        content = _fetch_file(ep, graduation_id, rel)
        dest = out_dir / candidate
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")
        written[rel] = dest
    return written


__all__ = [
    "MAX_BUNDLE_FILES",
    "MAX_FILE_BYTES",
    "MAX_BUNDLE_BYTES",
    "CloudGraduateError",
    "ActiveGraduationExists",
    "NoChanges",
    "CloudEndpoint",
    "resolve_cloud_endpoint",
    "fetch_models",
    "resolve_model",
    "sync_skill",
    "start_graduation",
    "poll_graduation",
    "cancel_graduation",
    "download_artifacts",
]
