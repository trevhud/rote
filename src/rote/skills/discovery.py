"""Find the skills on this machine and rank them by how ready they are to compile.

First-run setup offers the user a skill to compile. The useful first pick
is one that can run *now*: the bundled demo, then a skill whose MCP
servers are all registered and logged in, then one that is a login away.
:func:`discover` finds every skill the user's AI clients load and returns
them in that order, each with a structured :class:`Readiness`.

Roots, in the order a duplicate is credited to (code.claude.com/docs/en/skills;
learn.chatgpt.com/docs/build-skills; geminicli.com/docs/cli/skills;
opencode.ai/docs/skills; agentskills.io, checked 2026-10-22):

- the bundled examples that ship a skill (:data:`rote.examples.BUNDLED`);
- Claude Code: ``.claude/skills`` in the cwd and each parent up to the
  repository root, ``<config dir>/skills`` (``~/.claude``, or
  ``CLAUDE_CONFIG_DIR``), and ``skills/`` of each enabled plugin, found
  exactly as the MCP import finds them
  (:func:`rote.clients.claude_code.claude_code_layout`);
- the cross-client ``.agents/skills`` (project, walked the same way) and
  ``~/.agents/skills``, which Codex, Gemini CLI and OpenCode read;
- Codex's older ``$CODEX_HOME/skills`` (``~/.codex/skills``), still read
  for compatibility; Gemini CLI's ``.gemini/skills`` and
  ``~/.gemini/skills``; OpenCode's ``.opencode/skills`` and
  ``$XDG_CONFIG_HOME/opencode/skills`` (``~/.config/opencode/skills``).

A skill is ``<root>/<name>/SKILL.md``, one level down, which is what every
one of these clients loads. Skills are deduplicated by resolved path, so
a skill symlinked into two roots is one skill listed with both.

This module only reads. It never contacts a server, and it reads a
skill's ``SKILL.md`` frontmatter (bounded) plus, for readiness, the text
:func:`rote.compiler.tool_grounding.servers_named` reads. Everything comes
through :class:`~rote.clients.base.ClientContext`, so tests use fixture
dirs and ``scripts/isolated-rote`` never sees the real home.
"""

from __future__ import annotations

import os
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from rote.clients.base import ClientContext

Client = Literal["rote", "claude-code", "agents", "codex", "gemini", "opencode"]
Scope = Literal["bundled", "project", "user", "plugin"]
State = Literal["bundled", "ready", "needs_login", "needs_server", "unreadable"]

#: Ranking order of :attr:`Readiness.state`, best first.
STATE_ORDER: tuple[State, ...] = ("bundled", "ready", "needs_login", "needs_server", "unreadable")

#: An auth state (:func:`rote.mcp.tokens.auth_status`) a run can use as is.
USABLE_AUTH: frozenset[str] = frozenset(
    {"authenticated", "expired (refreshable)", "static headers", "no auth required"}
)

#: How much of a SKILL.md is read for its frontmatter. A frontmatter that
#: does not close within it is reported, not read further.
FRONTMATTER_BYTES = 64 * 1024
#: Displayed text is cut to this many characters.
DESCRIPTION_CHARS = 300

SKILL_FILE = "SKILL.md"


@dataclass(frozen=True)
class SkillRoot:
    """A directory whose ``<name>/SKILL.md`` children are skills."""

    client: Client
    scope: Scope
    path: Path
    #: The plugin that ships it, for ``scope == "plugin"``.
    plugin: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "client": self.client,
            "scope": self.scope,
            "path": str(self.path),
            "plugin": self.plugin,
        }


@dataclass(frozen=True)
class ServerReadiness:
    """One MCP server a skill names, against the local registry."""

    name: str
    #: Why rote thinks the skill uses it (``servers_named``'s evidence).
    evidence: str
    registered: bool
    #: :func:`rote.mcp.tokens.auth_status` when registered.
    auth: str | None

    @property
    def usable(self) -> bool:
        return self.registered and self.auth in USABLE_AUTH

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "evidence": self.evidence,
            "registered": self.registered,
            "auth": self.auth,
            "usable": self.usable,
        }


@dataclass(frozen=True)
class Readiness:
    state: State
    #: One sentence a person can act on.
    reason: str
    servers: tuple[ServerReadiness, ...] = ()
    #: The commands that move the skill to ``ready``, in order.
    next: tuple[str, ...] = ()

    @property
    def logins_needed(self) -> int:
        return sum(1 for s in self.servers if s.registered and not s.usable)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "reason": self.reason,
            "servers": [s.to_dict() for s in self.servers],
            "next": list(self.next),
        }


@dataclass(frozen=True)
class Skill:
    name: str
    description: str | None
    #: The skill directory, resolved: the identity duplicates collapse on.
    path: Path
    #: Every root it was found under, the first being the one credited.
    found_in: tuple[SkillRoot, ...]
    readiness: Readiness
    problems: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "path": str(self.path),
            "client": self.found_in[0].client,
            "scope": self.found_in[0].scope,
            "plugin": self.found_in[0].plugin,
            "found_in": [r.to_dict() for r in self.found_in],
            "readiness": self.readiness.to_dict(),
            "problems": list(self.problems),
        }


@dataclass(frozen=True)
class Discovery:
    skills: tuple[Skill, ...]
    #: Every root looked at, existing or not.
    roots: tuple[SkillRoot, ...]
    problems: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "skills": [s.to_dict() for s in self.skills],
            "roots": [{**r.to_dict(), "exists": r.path.is_dir()} for r in self.roots],
            "problems": list(self.problems),
        }


# ── roots ───────────────────────────────────────────────────────────────


def _project_dirs(cwd: Path) -> list[Path]:
    """The cwd and each parent up to the repository root (the nearest
    ancestor holding ``.git``), nearest first. No repository: the cwd only."""
    walked: list[Path] = []
    for directory in (cwd, *cwd.parents):
        walked.append(directory)
        if (directory / ".git").exists():
            return walked
    return [cwd]


def bundled_roots() -> list[tuple[str, Path]]:
    """``(example name, skill dir)`` for each bundled example that ships a skill."""
    from rote.examples import BUNDLED, ExampleError, examples_root

    try:
        root = examples_root()
    except ExampleError:
        return []
    return [
        (name, root / name / "skill")
        for name, example in BUNDLED.items()
        if "skill" in example.files and (root / name / "skill" / SKILL_FILE).is_file()
    ]


def skill_roots(ctx: ClientContext) -> tuple[list[SkillRoot], list[str]]:
    """Every root, in crediting order, and what could not be read finding them."""
    from rote.clients.claude_code import claude_code_layout

    layout = claude_code_layout(ctx)
    projects = _project_dirs(ctx.cwd)
    home = ctx.home
    codex_home = Path(ctx.env["CODEX_HOME"]).expanduser() if ctx.env.get("CODEX_HOME") else None
    xdg = ctx.env.get("XDG_CONFIG_HOME", "").strip()
    config_home = Path(xdg).expanduser() if xdg else home / ".config"

    roots: list[SkillRoot] = []
    roots += [SkillRoot("claude-code", "project", d / ".claude" / "skills") for d in projects]
    roots.append(SkillRoot("claude-code", "user", layout.config_dir / "skills"))
    roots += [
        SkillRoot("claude-code", "plugin", p.root / "skills", plugin=p.name) for p in layout.plugins
    ]
    roots += [SkillRoot("agents", "project", d / ".agents" / "skills") for d in projects]
    roots.append(SkillRoot("agents", "user", home / ".agents" / "skills"))
    roots.append(SkillRoot("codex", "user", (codex_home or home / ".codex") / "skills"))
    roots += [SkillRoot("gemini", "project", d / ".gemini" / "skills") for d in projects]
    roots.append(SkillRoot("gemini", "user", home / ".gemini" / "skills"))
    roots += [SkillRoot("opencode", "project", d / ".opencode" / "skills") for d in projects]
    roots.append(SkillRoot("opencode", "user", config_home / "opencode" / "skills"))
    return roots, list(layout.problems)


# ── one skill ───────────────────────────────────────────────────────────


def _display(text: object, limit: int = DESCRIPTION_CHARS) -> str | None:
    """Text from a third-party file, made safe to print: one line, no
    control or format characters (a bidi override, a zero-width space),
    and bounded."""
    if not isinstance(text, str):
        return None
    cleaned = "".join(
        " " if ch.isspace() else ch
        for ch in text
        if ch.isspace() or unicodedata.category(ch) not in ("Cc", "Cf", "Cs", "Zl", "Zp")
    )
    cleaned = " ".join(cleaned.split())
    if not cleaned:
        return None
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 1].rstrip() + "…"


def read_frontmatter(skill_md: Path) -> tuple[dict[str, Any], str | None]:
    """The YAML frontmatter of ``skill_md`` and a problem, if any.

    Reads at most :data:`FRONTMATTER_BYTES`, parses with ``safe_load``, and
    never raises: a file with no frontmatter is ``{}`` (Claude Code then
    uses the directory name), and a bad one is ``{}`` plus the reason.
    """
    try:
        with skill_md.open("rb") as f:
            head = f.read(FRONTMATTER_BYTES)
    except OSError as e:
        return {}, f"unreadable ({type(e).__name__})"
    text = head.decode("utf-8", errors="replace").lstrip("﻿")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            block = "\n".join(lines[1:index])
            break
    else:
        return {}, f"its frontmatter does not close within {FRONTMATTER_BYTES // 1024} KB"
    try:
        loaded = yaml.safe_load(block) if block.strip() else {}
    except yaml.YAMLError as e:
        mark = getattr(e, "problem_mark", None)
        where = f" (line {mark.line + 2})" if mark is not None else ""
        return {}, f"its frontmatter is not valid YAML{where}"
    if loaded is None:
        return {}, None
    if not isinstance(loaded, dict):
        return {}, "its frontmatter is not a mapping"
    return loaded, None


@dataclass(frozen=True)
class RegistryView:
    """What readiness is judged against: each registered server's auth state."""

    auth: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def current(cls) -> RegistryView:
        from rote.mcp import auth_status, load_registry, read_token_file
        from rote.mcp.learned import learned_auth

        registry = load_registry()
        states: dict[str, str] = {}
        for name, config in registry.servers.items():
            learned = learned_auth(name, config.url)
            states[name] = auth_status(
                config, read_token_file(name), learned.auth if learned else None
            )
        return cls(states)


def readiness_of(skill_dir: Path, registry: RegistryView, *, bundled: bool) -> Readiness:
    """Where ``skill_dir`` stands: its named servers against ``registry``."""
    from rote.compiler.tool_grounding import servers_named

    try:
        named = servers_named(skill_dir, registry.auth.keys())
    except Exception as e:  # one broken skill is reported, never hides the rest
        return Readiness("unreadable", f"rote could not read the skill ({type(e).__name__})")
    servers = tuple(
        ServerReadiness(name, evidence, name in registry.auth, registry.auth.get(name))
        for name, evidence in named.items()
    )
    missing = [s.name for s in servers if not s.registered]
    logins = [s.name for s in servers if s.registered and not s.usable]
    nexts = tuple(
        [f"rote mcp add {name} <url>" for name in missing]
        + [f"rote mcp login {name}" for name in [*missing, *logins]]
    )
    if bundled:
        reason = "the bundled demo: runs precompiled, no login or tokens needed"
        if servers and not missing and not logins:
            reason += "; its servers are ready for the live run too"
        return Readiness("bundled", reason, servers, nexts)
    if not servers:
        return Readiness("ready", "it names no MCP server", servers)
    if missing:
        listed = ", ".join(missing)
        return Readiness(
            "needs_server",
            f"names {len(missing)} server(s) rote has not registered: {listed}",
            servers,
            nexts,
        )
    if logins:
        listed = ", ".join(f"{s.name} ({s.auth})" for s in servers if s.name in logins)
        return Readiness(
            "needs_login",
            f"{len(logins)} login(s) away: {listed}",
            servers,
            nexts,
        )
    names = ", ".join(s.name for s in servers)
    return Readiness(
        "ready", f"every server it names is registered and logged in: {names}", servers
    )


def _read_skill(skill_dir: Path, found_in: tuple[SkillRoot, ...], registry: RegistryView) -> Skill:
    meta, problem = read_frontmatter(skill_dir / SKILL_FILE)
    problems = [problem] if problem else []
    name = _display(meta.get("name"), 80) or _display(skill_dir.name, 80) or skill_dir.name
    bundled = found_in[0].scope == "bundled"
    readiness = readiness_of(skill_dir, registry, bundled=bundled)
    return Skill(
        name=name,
        description=_display(meta.get("description")),
        path=skill_dir,
        found_in=found_in,
        readiness=readiness,
        problems=tuple(problems),
    )


# ── all of them ─────────────────────────────────────────────────────────


def _children(root: SkillRoot) -> Iterable[Path]:
    try:
        entries = sorted(os.scandir(root.path), key=lambda e: e.name)
    except OSError:  # missing, not a dir, or unreadable: nothing here
        return []
    found = []
    for entry in entries:
        if entry.name.startswith("."):
            continue
        candidate = Path(entry.path)
        if (candidate / SKILL_FILE).is_file():
            found.append(candidate)
    return found


def rank_key(skill: Skill, bundled_order: Mapping[Path, int]) -> tuple[Any, ...]:
    r = skill.readiness
    return (
        STATE_ORDER.index(r.state),
        bundled_order.get(skill.path, 0),
        r.logins_needed,
        sum(1 for s in r.servers if not s.registered),
        skill.name.lower(),
        str(skill.path),
    )


def discover(
    ctx: ClientContext | None = None,
    *,
    registry: RegistryView | None = None,
    include_bundled: bool = True,
) -> Discovery:
    """Every skill under :func:`skill_roots`, ranked best first.

    The bundled demos come first, then ``ready``, then ``needs_login``
    (fewest logins first), then ``needs_server``, then ``unreadable``;
    names break ties.
    """
    ctx = ClientContext.current() if ctx is None else ctx
    registry = RegistryView.current() if registry is None else registry
    roots, problems = skill_roots(ctx)
    bundled: list[tuple[SkillRoot, Path]] = []
    if include_bundled:
        for _name, skill_dir in bundled_roots():
            root = SkillRoot("rote", "bundled", skill_dir.parent)
            bundled.append((root, skill_dir))

    by_path: dict[Path, list[SkillRoot]] = {}
    bundled_order: dict[Path, int] = {}
    for index, (root, skill_dir) in enumerate(bundled):
        resolved = skill_dir.resolve()
        by_path.setdefault(resolved, []).append(root)
        bundled_order[resolved] = index
    for root in roots:
        for skill_dir in _children(root):
            try:
                resolved = skill_dir.resolve()
            except OSError as e:
                problems.append(f"{skill_dir}: could not resolve it ({type(e).__name__})")
                continue
            by_path.setdefault(resolved, []).append(root)

    # Serial on purpose: the work is regex over text already read, and a
    # thread pool measured no faster (178 skills, 0.33 s either way).
    skills = [_read_skill(path, tuple(found), registry) for path, found in by_path.items()]
    skills.sort(key=lambda s: rank_key(s, bundled_order))
    return Discovery(skills=tuple(skills), roots=tuple(roots), problems=tuple(problems))
