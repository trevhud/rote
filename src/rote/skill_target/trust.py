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
  ``__pycache__`` would too, which is why the driver runs with ``-B``), a
  missing one (an import would fall through to site-packages), a symlink
  (its target can change after the check) or a special file is refused,
  and so is a state path that no longer resolves where it did at install.
- **The copies.** Each trusted file's bytes are kept by hash beside the
  registry (:func:`store_dir`), so ``rote skill trust`` can show what changed
  as a diff, not only that a hash differs.

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


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def scan(root: Path) -> Scan:
    """Every entry under ``root``, never following a symlink."""
    files: dict[str, str] = {}
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
                    files[rel] = _sha(Path(entry.path).read_bytes())
                else:
                    special.append(rel)

    walk(root, "")
    return Scan(files, tuple(links), tuple(special))


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


def bindings() -> dict[str, str]:
    """Where this process's MCP registry, login store and run folders resolve."""
    from rote.mcp.registry import registry_path
    from rote.mcp.tokens import token_dir
    from rote.skill_target._driver import run_root

    return {
        "mcp_config": str(registry_path().resolve()),
        "token_dir": str(token_dir().resolve()),
        "run_dir": str(run_root().resolve()),
    }


def moved_bindings(recorded: Mapping[str, str], now: Mapping[str, str]) -> list[str]:
    """One line per state path that resolves elsewhere than the record says."""
    lines = []
    for key, (var, what) in BINDINGS.items():
        if recorded.get(key) != now[key]:
            lines.append(f"{what} ({var}) was {recorded.get(key)} at install and is {now[key]} now")
    return lines


# ───────── the copies ─────────


def store_dir() -> Path:
    """Where trusted files' bytes are kept by hash: beside the app registry,
    so whatever moves the registry (``ROTE_APPS_PATH``) moves these too."""
    return apps_path().parent / "skill-files"


def keep_copies(root: Path, files: Mapping[str, str]) -> None:
    """Copy each of ``files`` into the store, unless a copy with its hash is there."""
    store = store_dir()
    store.mkdir(parents=True, exist_ok=True, mode=0o700)
    for rel, digest in files.items():
        target = store / digest
        if target.is_file():
            continue
        data = (root / rel).read_bytes()
        if _sha(data) != digest:
            raise SkillTrustError(
                f"{root / rel} changed while it was being recorded.",
                "run the command again",
            )
        tmp = store / f".{digest}.tmp"
        tmp.write_bytes(data)
        os.replace(tmp, target)


def _copy(digest: str) -> bytes | None:
    try:
        return (store_dir() / digest).read_bytes()
    except OSError:
        return None


def _lines(data: bytes | None) -> list[str] | None:
    if data is None:
        return None
    try:
        return data.decode("utf-8").splitlines()
    except UnicodeDecodeError:
        return None


def diff_summary(root: Path, found: Drift, recorded: Mapping[str, str]) -> list[str]:
    """One line per difference: lines added and removed where the old copy is kept."""
    lines = []
    for rel in found.changed:
        old = _lines(_copy(recorded[rel]))
        new = _lines((root / rel).read_bytes())
        if old is None or new is None:
            lines.append(f"changed  {rel}")
            continue
        added = removed = 0
        for line in difflib.ndiff(old, new):
            if line.startswith("+ "):
                added += 1
            elif line.startswith("- "):
                removed += 1
        lines.append(f"changed  {rel}  (+{added} -{removed} lines)")
    for rel in found.unlisted:
        new = _lines((root / rel).read_bytes())
        size = f"{len(new)} lines" if new is not None else f"{(root / rel).stat().st_size} bytes"
        lines.append(f"new      {rel}  ({size})")
    lines += [f"missing  {rel}" for rel in found.missing]
    lines += [f"symlink  {rel}  (remove it: a symlink cannot be trusted)" for rel in found.links]
    lines += [f"special  {rel}  (remove it: not a regular file)" for rel in found.special]
    return lines


def unified_diffs(root: Path, found: Drift, recorded: Mapping[str, str]) -> list[str]:
    """The changed and new text files as unified diffs against the kept copies."""
    out: list[str] = []
    for rel in (*found.changed, *found.unlisted):
        old = _lines(_copy(recorded[rel])) if rel in recorded else []
        new = _lines((root / rel).read_bytes())
        if old is None or new is None:
            out.append(f"Binary or unrecorded: {rel}")
            continue
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

    def plan(self) -> dict[str, Any]:
        loaded: dict[str, Any] = json.loads((self.root / PLAN_FILE).read_text(encoding="utf-8"))
        return loaded


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


def install_record(files: Mapping[str, str], *, writes: dict[str, Any] | None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "format": TRUST_FORMAT,
        "files": dict(sorted(files.items())),
        "bindings": bindings(),
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if writes is not None:
        record["writes"] = writes
    return record


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


def check(app: RegisteredApp) -> SkillDir:
    """Refuse to run ``app`` unless its dir is exactly what its record vouches for."""
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
    moved = moved_bindings(recorded, bindings())
    if moved:
        raise SkillTrustError(
            f"{name} was installed against other state paths than this run sees: "
            + "; ".join(moved)
            + ". Those paths decide which servers count as read-only, whose logins a run "
            "uses and which journals it reads.",
            fix,
        )
    found = drift(scan(root), files)
    if not found.clean:
        raise SkillTrustError(
            f"{name}'s files are not the ones the user installed ({root}): {found.describe()}.",
            fix,
        )
    return read_skill_dir(root)


# ───────── write consent ─────────


def plan_sha256(skill: SkillDir) -> str:
    return _sha((skill.root / PLAN_FILE).read_bytes())


@dataclass(frozen=True)
class WriteTools:
    """The tools a skill's plan calls, split by whether a read-only run lets them through."""

    #: ``server.tool`` for each one that may write.
    may_write: tuple[str, ...]
    #: ``server.tool`` for each one on a server the user registered read-only.
    read_only: tuple[str, ...]


def write_tools(skill: SkillDir) -> WriteTools:
    """Every tool the plan can call, minus those on servers the user registered read-only.

    Offline: rote does not list tools to read their ``readOnlyHint``, so a
    tool is listed as one that may write unless the user's own declaration
    (``rote mcp add --read-only``, at the endpoint the run calls) covers it.
    The consent covers this list and this plan only."""
    from rote.mcp.registry import load_registry

    servers = load_registry().servers
    calls: set[str] = set()
    for step in skill.plan().get("steps") or []:
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


def write_consent(app: RegisteredApp, skill: SkillDir) -> bool:
    """Whether the user let this skill's tools write, for the plan it has now."""
    writes = (app.trust or {}).get("writes")
    return isinstance(writes, dict) and writes.get("plan_sha256") == plan_sha256(skill)
