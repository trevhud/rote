"""What the user vouched for in a compiled skill, and the check every run makes against it.

``rote skill run <name>`` runs code an allow rule pre-approves
(``Bash(rote skill run <name> *)``), so the rule must approve exactly the
files the user installed, not whatever sits in a directory that happens to
use a registered name. Three pieces, all here:

- **The record.** ``rote skill install <dir>`` (the first trust point) and
  ``rote skill trust <name>`` (every later one) write an install record into
  the app registry entry (:mod:`rote.app_registry`): the sha256 of every file
  in the skill dir, ``.rote-manifest.json`` and SKILL.md included, the state
  paths the user vouched for at that moment (:func:`bindings`), and any write
  consent (``--allow-writes``), bound to the plan it was given for.
- **The check** (:func:`check`), before every run: the dir holds exactly the
  recorded files with the recorded hashes. A changed file, an unlisted one
  (a dropped-in ``scripts/json.py`` would shadow an import; a stale
  ``__pycache__`` would too), a missing one (an import would fall through to
  site-packages), a symlink (its target can change after the check) or a
  special file is refused, and so is any binding that differs from the
  record. Each file is read once; the run starts from those bytes
  (:class:`Verified`, :mod:`rote.skill_target.launch`).
- **The copies.** Each trusted file's bytes are kept by hash beside the
  registry (:func:`store_dir`), so ``rote skill trust`` can show what changed
  as a diff, not only that a hash differs. A copy is used only while it still
  hashes to its name.

Besides the state paths, the record binds ``ROTE_CLOUD_CRED_PATH``, the
proxy variables, the CA bundles (path and hash), the servers the plan
calls that the user registered read-only (name and URL), and the names of
the environment variables the plan's code reads (``Node.env``), which a run
passes through from the caller's environment. A read-only declaration lets
every tool on a server run, so it is write consent by another name and is
bound the same way.

The state paths are bound, not merely read, because each can widen trust
without touching a file in the skill: a forged ``ROTE_MCP_CONFIG`` can
declare a server read-only (so its write tools run) or point it at another
endpoint, a forged ``ROTE_MCP_TOKEN_DIR`` can supply somebody else's login
(the skill's writes then land in their account), and a forged
``ROTE_SKILL_RUN_DIR`` can hold a journal whose gate a person never
answered. The override variables stay honored, because the test suite and
``scripts/isolated-rote`` need them; a run whose paths moved since the
install is refused, naming ``rote skill trust``. The registry itself
(``ROTE_APPS_PATH``) is the root of trust and cannot be bound to itself:
see ``docs/design/compiled-skill-target.md`` for why that is the line.
"""

from __future__ import annotations

import difflib
import errno
import hashlib
import json
import os
import stat
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rote.app_registry import RegisteredApp, apps_path
from rote.env_vars import CA_VARS, PROXY_VARS, reserved_reason

#: The app registry runtime a compiled skill is recorded under.
SKILL_RUNTIME = "skill"
#: The install record's format.
TRUST_FORMAT = 1
MANIFEST_FILE = ".rote-manifest.json"
PLAN_FILE = "scripts/plan.json"

#: The state paths an install is bound to: ``{key: (the override, what it holds)}``.
BINDINGS: dict[str, tuple[str, str]] = {
    "mcp_config": ("ROTE_MCP_CONFIG", "the MCP registry"),
    "token_dir": ("ROTE_MCP_TOKEN_DIR", "the MCP login store"),
    "run_dir": ("ROTE_SKILL_RUN_DIR", "the run folders"),
    "cloud_cred": ("ROTE_CLOUD_CRED_PATH", "the rote cloud login a helper judge pays with"),
}


class SkillTrustError(Exception):
    """A skill command cannot go on: the cause, and the command that fixes it."""

    def __init__(self, cause: str, fix: str, *, usage: bool = False) -> None:
        super().__init__(cause)
        self.cause = cause
        self.fix = fix
        self.usage = usage


# ───────── the files ─────────


@dataclass(frozen=True)
class Scan:
    """A skill dir as it is now: ``{posix relative path: sha256}`` of every regular
    file, plus every symlink and special file, which no record can vouch for."""

    files: dict[str, str]
    links: tuple[str, ...] = ()
    special: tuple[str, ...] = ()
    #: Each file's bytes, as hashed: a run starts from these, never from a
    #: second read of the dir (:mod:`rote.skill_target.launch`).
    data: dict[str, bytes] = field(default_factory=dict, repr=False, compare=False)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def scan(root: Path) -> Scan:
    """Every entry under ``root``, never following a symlink."""
    files: dict[str, str] = {}
    data: dict[str, bytes] = {}
    links: list[str] = []
    special: list[str] = []

    def walk(directory: Path, prefix: str) -> None:
        with os.scandir(directory) as entries:
            for entry in sorted(entries, key=lambda e: e.name):
                rel = f"{prefix}{entry.name}"
                if entry.is_symlink():
                    links.append(rel)
                elif entry.is_dir(follow_symlinks=False):
                    walk(Path(entry.path), f"{rel}/")
                elif entry.is_file(follow_symlinks=False):
                    # O_NOFOLLOW: a file swapped for a symlink since the
                    # listing is refused, not followed.
                    try:
                        fd = os.open(entry.path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                    except OSError as e:
                        if e.errno != errno.ELOOP:
                            raise
                        links.append(rel)
                        continue
                    with os.fdopen(fd, "rb") as f:
                        if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):
                            special.append(rel)
                            continue
                        content = f.read()
                    files[rel] = _sha(content)
                    data[rel] = content
                else:
                    special.append(rel)

    walk(root, "")
    return Scan(files, tuple(links), tuple(special), data)


@dataclass(frozen=True)
class Drift:
    """How a skill dir differs from the files a record vouches for."""

    changed: tuple[str, ...] = ()
    unlisted: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    links: tuple[str, ...] = ()
    special: tuple[str, ...] = ()

    @property
    def clean(self) -> bool:
        return not (self.changed or self.unlisted or self.missing or self.links or self.special)

    @property
    def untrustable(self) -> tuple[str, ...]:
        """What no ``rote skill trust`` can record: it must be removed first."""
        return self.links + self.special

    def describe(self) -> str:
        """One clause per kind, for an ERROR line."""
        parts = []
        for label, paths in (
            ("changed", self.changed),
            ("not in the install record", self.unlisted),
            ("missing", self.missing),
            ("a symlink", self.links),
            ("not a regular file", self.special),
        ):
            if paths:
                shown = ", ".join(paths[:5]) + (
                    f" and {len(paths) - 5} more" if len(paths) > 5 else ""
                )
                parts.append(f"{label}: {shown}")
        return "; ".join(parts)

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "changed": list(self.changed),
            "unlisted": list(self.unlisted),
            "missing": list(self.missing),
            "symlinks": list(self.links),
            "special": list(self.special),
        }


def drift(now: Scan, recorded: Mapping[str, str]) -> Drift:
    return Drift(
        changed=tuple(p for p, h in now.files.items() if p in recorded and recorded[p] != h),
        unlisted=tuple(p for p in now.files if p not in recorded),
        missing=tuple(p for p in recorded if p not in now.files),
        links=now.links,
        special=now.special,
    )


# ───────── the state paths ─────────


def _file_sha(path: str) -> str:
    try:
        return _sha(Path(path).read_bytes())
    except OSError as e:
        return f"unreadable ({e.strerror})"


def network_env(environ: Mapping[str, str]) -> dict[str, Any]:
    """The proxy and CA variables set in ``environ``, the CA files by path and hash."""
    return {
        "proxy": {v: environ[v] for v in PROXY_VARS if environ.get(v)},
        "ca": {
            v: f"{environ[v]} sha256:{_file_sha(environ[v])}" for v in CA_VARS if environ.get(v)
        },
    }


def plan_servers(plan: Mapping[str, Any]) -> set[str]:
    """Every MCP server a plan can call: its steps, a loop's tools and body steps."""
    used: set[str] = set()
    for step in plan.get("steps") or []:
        if step.get("kind") == "mcp":
            used.add(str(step["server"]))
        for body in step.get("body") or []:
            if body.get("kind") == "mcp":
                used.add(str(body["server"]))
        used.update(str(s) for s in (step.get("tool_servers") or {}).values())
    return used


def plan_env(plan: Mapping[str, Any]) -> list[str]:
    """The environment variables the plan's code reads (``Node.env``), sorted:
    a run passes exactly these from the caller's environment, beyond the
    allowlist. A name no node may declare (:func:`rote.env_vars.reserved_reason`)
    is refused: the IR refuses it too, so only an edited plan carries one."""
    names: set[str] = set()
    for step in plan.get("steps") or []:
        for each in (step, *(step.get("body") or [])):
            names.update(str(n) for n in each.get("env") or [])
    for name in sorted(names):
        reason = reserved_reason(name)
        if reason is not None:
            raise SkillTrustError(
                f"the skill's plan passes {name} through from the caller's environment, "
                f"which no skill may: {reason}.",
                "re-emit the skill from its pipeline.yaml (rote emit --runtime skill)",
            )
    return sorted(names)


def relied_read_only(plan: Mapping[str, Any]) -> dict[str, str]:
    """``{server: registered url}`` for each server the plan calls that the user
    registered read-only. A read-only run calls every tool on those, so the set
    acts as write consent and is bound like it."""
    from rote.mcp.registry import load_registry

    servers = load_registry().servers
    used = plan_servers(plan)
    return {
        name: config.url
        for name, config in sorted(servers.items())
        if name in used and config.read_only is True
    }


def bindings(plan: Mapping[str, Any], environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """What a run of ``plan`` depends on beyond its files, as this process resolves it:
    the state paths, the proxy and CA variables, and the servers trusted read-only."""
    from rote.cloud_auth import credential_path
    from rote.mcp.registry import registry_path
    from rote.mcp.tokens import token_dir
    from rote.skill_target._driver import run_root

    environ = os.environ if environ is None else environ
    return {
        "mcp_config": str(registry_path().resolve()),
        "token_dir": str(token_dir().resolve()),
        "run_dir": str(run_root().resolve()),
        "cloud_cred": str(credential_path().resolve()),
        **network_env(environ),
        "read_only": relied_read_only(plan),
        "env": plan_env(plan),
    }


def moved_bindings(recorded: Mapping[str, Any], now: Mapping[str, Any]) -> list[str]:
    """One line per binding that differs from the record."""
    lines = []
    for key, (var, what) in BINDINGS.items():
        if recorded.get(key) != now[key]:
            lines.append(f"{what} ({var}) was {recorded.get(key)} at install and is {now[key]} now")
    for key, what in (
        ("proxy", "the proxy variables"),
        ("ca", "the CA bundle variables"),
        ("read_only", "the servers registered read-only that this skill calls"),
    ):
        if (recorded.get(key) or {}) != (now.get(key) or {}):
            lines.append(f"{what} were {recorded.get(key) or {}} at install and are {now[key]} now")
    if (recorded.get("env") or []) != (now.get("env") or []):
        lines.append(
            "the environment variables passed through from the caller were "
            f"{recorded.get('env') or []} at install and are {now.get('env') or []} now"
        )
    return lines


def _loopback(host: str | None) -> bool:
    import ipaddress

    if not host:
        return False
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def proxied_plain_http(plan: Mapping[str, Any], network: Mapping[str, Any]) -> list[str]:
    """The non-loopback ``http://`` endpoints a set proxy would see in clear text."""
    from urllib.parse import urlparse

    from rote.mcp.registry import load_registry

    if not network.get("proxy"):
        return []
    servers = load_registry().servers
    urls: dict[str, str] = {}
    for step in plan.get("steps") or []:
        if step.get("kind") == "mcp":
            config = servers.get(str(step["server"]))
            urls[str(step["server"])] = config.url if config is not None else str(step.get("url"))
    for server in plan_servers(plan) - urls.keys():
        config = servers.get(server)
        if config is not None:
            urls[server] = config.url
    bad = []
    for server, url in sorted(urls.items()):
        parsed = urlparse(url)
        if parsed.scheme == "http" and not _loopback(parsed.hostname):
            bad.append(f"{server} at {url}")
    return bad


# ───────── the copies ─────────


def store_dir() -> Path:
    """Where trusted files' bytes are kept by hash: beside the app registry,
    so whatever moves the registry (``ROTE_APPS_PATH``) moves these too."""
    return apps_path().parent / "skill-files"


def keep_copies(data: Mapping[str, bytes]) -> None:
    """Keep each file's bytes in the store under its hash. A kept copy that no
    longer matches its name is replaced: only verified bytes are ever read back."""
    store = store_dir()
    store.mkdir(parents=True, exist_ok=True, mode=0o700)
    for content in data.values():
        digest = _sha(content)
        target = store / digest
        if _copy(digest) is not None:
            continue
        tmp = store / f".{digest}.tmp"
        tmp.write_bytes(content)
        os.replace(tmp, target)


def _copy(digest: str) -> bytes | None:
    """The kept bytes for ``digest``, only when they still hash to it."""
    try:
        data = (store_dir() / digest).read_bytes()
    except OSError:
        return None
    return data if _sha(data) == digest else None


def _lines(data: bytes | None) -> list[str] | None:
    if data is None:
        return None
    try:
        return data.decode("utf-8").splitlines()
    except UnicodeDecodeError:
        return None


def _old_copy_problem(digest: str) -> str:
    """Why no diff is possible for a file recorded with ``digest``."""
    path = store_dir() / digest
    if not path.exists():
        return "no copy of the trusted version is kept"
    return "the kept copy of the trusted version does not match its hash (it was altered)"


def diff_summary(now: Scan, found: Drift, recorded: Mapping[str, str]) -> list[str]:
    """One line per difference: lines added and removed, where a verified copy of the
    trusted version is kept. Otherwise it says why, and :func:`unified_diffs` shows
    the whole current file instead of a diff."""
    lines = []
    for rel in found.changed:
        old = _lines(_copy(recorded[rel]))
        new = _lines(now.data[rel])
        if new is None:
            lines.append(f"changed  {rel}  (not text, {len(now.data[rel])} bytes)")
            continue
        if old is None:
            lines.append(
                f"changed  {rel}  ({_old_copy_problem(recorded[rel])}; the whole file is shown)"
            )
            continue
        added = removed = 0
        for line in difflib.ndiff(old, new):
            if line.startswith("+ "):
                added += 1
            elif line.startswith("- "):
                removed += 1
        lines.append(f"changed  {rel}  (+{added} -{removed} lines)")
    for rel in found.unlisted:
        new = _lines(now.data[rel])
        size = f"{len(new)} lines" if new is not None else f"{len(now.data[rel])} bytes"
        lines.append(f"new      {rel}  ({size})")
    lines += [f"missing  {rel}" for rel in found.missing]
    lines += [f"symlink  {rel}  (remove it: a symlink cannot be trusted)" for rel in found.links]
    lines += [f"special  {rel}  (remove it: not a regular file)" for rel in found.special]
    return lines


def whole_files(now: Scan, found: Drift, recorded: Mapping[str, str]) -> list[str]:
    """Every changed text file whose trusted version has no verified copy, in full:
    shown whether or not ``--diff`` was asked for, since its summary has no numbers."""
    out: list[str] = []
    for rel in found.changed:
        new = _lines(now.data[rel])
        if new is not None and _copy(recorded[rel]) is None:
            out.append(f"--- whole file: {rel} ({_old_copy_problem(recorded[rel])})")
            out += [f"  {line}" for line in new]
    return out


def unified_diffs(now: Scan, found: Drift, recorded: Mapping[str, str]) -> list[str]:
    """The changed and new text files as unified diffs against the verified copies."""
    out: list[str] = []
    for rel in (*found.changed, *found.unlisted):
        new = _lines(now.data[rel])
        if new is None:
            out.append(f"not text: {rel}")
            continue
        if rel in recorded:
            old = _lines(_copy(recorded[rel]))
            if old is None:
                continue  # whole_files shows it
        else:
            old = []
        out += list(difflib.unified_diff(old, new, f"a/{rel}", f"b/{rel}", lineterm=""))
    return out


# ───────── the skill and its record ─────────


@dataclass(frozen=True)
class SkillDir:
    """A compiled skill on disk, as its manifest describes it."""

    root: Path
    name: str
    pipeline: str
    manifest_files: dict[str, str] = field(repr=False)

    @property
    def scripts(self) -> Path:
        return self.root / "scripts"


def read_skill_dir(path: Path) -> SkillDir:
    """The compiled skill at ``path``; refused when it is not one rote emitted."""
    from rote.skill_target._driver import SKILL_NAME, SKILL_NAME_MAX

    root = Path(path)
    if root.is_symlink():
        raise SkillTrustError(
            f"{root} is a symlink, and a skill is installed from its real directory.",
            f"rote skill install {root.resolve()}",
            usage=True,
        )
    manifest = root / MANIFEST_FILE
    if not manifest.is_file():
        raise SkillTrustError(
            f"{root} is not a compiled skill: it has no {MANIFEST_FILE}.",
            "emit one with rote emit <pipeline.yaml> --runtime skill --out <dir>, or "
            "rote compile <skill> --runtime skill",
            usage=True,
        )
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise SkillTrustError(
            f"{manifest} cannot be read: {e}.", "re-emit the skill (rote emit --runtime skill)"
        ) from e
    record = data.get("skill") if isinstance(data, dict) else None
    files = data.get("files") if isinstance(data, dict) else None
    if not isinstance(record, dict) or not isinstance(files, dict):
        raise SkillTrustError(
            f"{root} was not emitted for --runtime skill (its manifest records no skill).",
            "emit it with rote emit <pipeline.yaml> --runtime skill --out <dir>",
            usage=True,
        )
    name = record.get("name")
    if not isinstance(name, str) or len(name) > SKILL_NAME_MAX or not SKILL_NAME.fullmatch(name):
        raise SkillTrustError(
            f"{manifest} records {name!r} as the skill's name, which is not a skill name.",
            "re-emit the skill (rote emit --runtime skill)",
        )
    if not (root / PLAN_FILE).is_file():
        raise SkillTrustError(
            f"{root} has no {PLAN_FILE}.", "re-emit the skill (rote emit --runtime skill)"
        )
    return SkillDir(
        root=root.resolve(),
        name=name,
        pipeline=str(record.get("pipeline") or name),
        manifest_files={str(k): str(v) for k, v in files.items()},
    )


def against_manifest(skill: SkillDir, now: Scan) -> Drift:
    """How the dir differs from what rote emitted into it (the first trust point).

    The manifest attests to every file rote wrote; a file it lists that is
    gone is not a difference (an emit carries entries forward)."""
    listed = {**skill.manifest_files, MANIFEST_FILE: now.files.get(MANIFEST_FILE, "")}
    found = drift(now, listed)
    return Drift(found.changed, found.unlisted, (), found.links, found.special)


def installed(name: str, apps: list[RegisteredApp]) -> RegisteredApp:
    """The install of ``name``: its one registry entry with an install record."""
    from rote.skill_target._driver import SKILL_NAME, SKILL_NAME_MAX

    if len(name) > SKILL_NAME_MAX or not SKILL_NAME.fullmatch(name):
        raise SkillTrustError(
            f"{name!r} is not a skill name (lowercase letters, digits and single hyphens).",
            "rote skill list shows the installed skills",
            usage=True,
        )
    named = [a for a in apps if a.runtime == SKILL_RUNTIME and a.name == name]
    trusted = [a for a in named if a.trust is not None]
    if len(trusted) == 1:
        return trusted[0]
    if len(trusted) > 1:
        paths = ", ".join(str(a.path) for a in trusted)
        raise SkillTrustError(
            f"{name} is installed from more than one directory ({paths}).",
            "rote skill install <the one to keep> --force (it replaces the others)",
        )
    if named:
        return_dir = named[-1].path
        raise SkillTrustError(
            f"{name} was emitted to {return_dir} but is not installed, so nothing vouches "
            "for its files yet.",
            f"ask the user to review it and run: rote skill install {return_dir}",
        )
    raise SkillTrustError(
        f"no skill named {name} is installed.",
        "ask the user to run: rote skill install <the compiled skill's dir>",
    )


def scanned_plan(now: Scan) -> dict[str, Any]:
    """The plan from the verified bytes, never a second read of the dir."""
    loaded: dict[str, Any] = json.loads(now.data[PLAN_FILE].decode("utf-8"))
    return loaded


def install_record(
    now: Scan, *, writes: dict[str, Any] | None, environ: Mapping[str, str] | None = None
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "format": TRUST_FORMAT,
        "files": dict(sorted(now.files.items())),
        "bindings": bindings(scanned_plan(now), environ),
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if writes is not None:
        record["writes"] = writes
    return record


@dataclass(frozen=True)
class Verified:
    """A skill that passed :func:`check`: its dir, the bytes that were checked, its plan."""

    skill: SkillDir
    scan: Scan
    plan: dict[str, Any]
    bound: dict[str, Any]


def check(app: RegisteredApp, environ: Mapping[str, str] | None = None) -> Verified:
    """Refuse to run ``app`` unless its dir is exactly what its record vouches for,
    and what it depends on (the state paths, the proxy and CA variables, the
    servers trusted read-only) is what the record says."""
    assert app.trust is not None and app.name is not None
    name = app.name
    fix = f"ask the user to review the change and run: rote skill trust {name}"
    root = app.path
    try:
        st = os.lstat(root)
    except FileNotFoundError:
        raise SkillTrustError(
            f"{name} is installed from {root}, which no longer exists.",
            "ask the user to install it again: rote skill install <dir>",
        ) from None
    if stat.S_ISLNK(st.st_mode) or Path(os.path.realpath(root)) != root:
        raise SkillTrustError(
            f"{root} (where {name} is installed from) is now reached through a symlink, "
            "which can point somewhere else after this check.",
            f"replace the symlink with the directory itself, then: rote skill install {root}",
        )
    files = app.trust.get("files")
    recorded = app.trust.get("bindings")
    if not isinstance(files, dict) or not isinstance(recorded, dict):
        raise SkillTrustError(
            f"{name}'s install record is malformed.", f"rote skill install {root}"
        )
    now = scan(root)
    found = drift(now, files)
    if not found.clean:
        raise SkillTrustError(
            f"{name}'s files are not the ones the user installed ({root}): {found.describe()}.",
            fix,
        )
    plan = scanned_plan(now)
    bound = bindings(plan, environ)
    moved = moved_bindings(recorded, bound)
    if moved:
        raise SkillTrustError(
            f"{name} was installed against another setup than this run sees: "
            + "; ".join(moved)
            + ". These decide which servers count as read-only, whose logins a run "
            "uses, where its traffic goes and which journals it reads.",
            fix,
        )
    exposed = proxied_plain_http(plan, bound)
    if exposed:
        raise SkillTrustError(
            f"a proxy is set, and {name} calls {', '.join(exposed)} over plain http, so the "
            "proxy would read every request and login token in clear text.",
            "register the server at an https:// URL (rote mcp add <server> <url>), or run "
            "without the proxy variables",
        )
    return Verified(read_skill_dir(root), now, plan, bound)


# ───────── write consent ─────────


def plan_sha256(now: Scan) -> str:
    """The hash of the plan in the checked bytes."""
    return now.files[PLAN_FILE]


@dataclass(frozen=True)
class WriteTools:
    """The tools a skill's plan calls, split by whether a read-only run lets them through."""

    #: ``server.tool`` for each one that may write.
    may_write: tuple[str, ...]
    #: ``server.tool`` for each one on a server the user registered read-only.
    read_only: tuple[str, ...]


def write_tools(plan: Mapping[str, Any]) -> WriteTools:
    """Every tool the plan can call, minus those on servers the user registered read-only.

    Offline: rote does not list tools to read their ``readOnlyHint``, so a
    tool is listed as one that may write unless the user's own declaration
    (``rote mcp add --read-only``, at the endpoint the run calls) covers it.
    The consent covers this list and this plan only."""
    from rote.mcp.registry import load_registry

    servers = load_registry().servers
    calls: set[str] = set()
    for step in plan.get("steps") or []:
        if step.get("kind") == "mcp":
            calls.add(f"{step['server']}.{step['tool']}")
        elif step.get("kind") == "loop":
            mapped = step.get("tool_servers") or {}
            for tool in step.get("tools") or []:
                calls.add(f"{mapped.get(tool, '?')}.{tool}")
            for body in step.get("body") or []:
                if body.get("kind") == "mcp":
                    calls.add(f"{body['server']}.{body['tool']}")
    may_write: list[str] = []
    read_only: list[str] = []
    for call in sorted(calls):
        server = call.split(".", 1)[0]
        config = servers.get(server)
        declared = config is not None and config.read_only is True
        (read_only if declared else may_write).append(call)
    return WriteTools(tuple(may_write), tuple(read_only))


def write_consent(app: RegisteredApp, now: Scan) -> bool:
    """Whether the user let this skill's tools write, for the plan it has now."""
    writes = (app.trust or {}).get("writes")
    return isinstance(writes, dict) and writes.get("plan_sha256") == plan_sha256(now)
