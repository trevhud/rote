"""``rote setup``: the guided first run, from a fresh install to a verdict.

One command that chains the steps rote already has (docs/design/first-run.md):

1. **mcp-import**: the servers in the AI client rote runs under, then the
   others on this machine (:class:`~rote.setup.mcp_import.McpImportStep`).
2. **login**: the registered servers that cannot connect yet
   (:class:`LoginStep`).
3. **skill**: one of the user's skills, ranked (:mod:`rote.skills`), its
   input, and a yes for each server the comparison calls live
   (:class:`SkillStep`).
4. **compile**, 5. **install**, 6. **eval**: ``rote try``'s three steps
   (:mod:`rote.setup.try_skill`), on the picked skill.

Then the verdict, in ``rote try``'s words, and what to do next.

Every step is a plan the person confirms, run by :func:`rote.setup.run_steps`.
Resumable: each step's status is read from the world (the registry, the
token store, ``rote try``'s files), plus the one thing the world does not
hold, the picked skill, which the skill step saves (:func:`state_path`) as
an Action in its plan.

**Agents.** Nobody at the terminal (an agent is the caller, ``--json``, a
pipe), every question and every consent stops the run with one
``action_required`` object (exit 2) that says what to ask the user
(``ask``) and the command that carries the answer (``next``). Consent is
per plan: ``--approve <step>=<code>`` approves the plan the user saw
(:func:`plan_code`), once, so an agent's yes for the import never covers
the compile, and a replay or a changed plan asks again; a paid step needs
``--live`` too, and a step that only a person may approve (``rote skill
install`` relying on read-only servers) names the command the user runs in
their own terminal. ``--skip <step>`` is the user's no.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rote.interact.plan import (
    FORCE_FLAG,
    YES_FLAG,
    Action,
    ConfirmationRequired,
    Plan,
    snapshot,
    write_file_action,
)
from rote.interact.prompter import Choice, Prompter, PromptRequired
from rote.interact.report import (
    EXIT_ACTION_REQUIRED,
    PersonRequired,
    action_required,
    authorization_pending,
    next_command,
    person_required,
)
from rote.setup import SetupContext, Step, StepReport, StepStatus, run_steps
from rote.setup.try_skill import DEFAULT_TRIALS, LIVE_FLAG, TryError

if TYPE_CHECKING:
    from rote.interact.mode import Mode
    from rote.interact.output import Output
    from rote.mcp.pending_login import Started
    from rote.mcp.registry import McpServerConfig
    from rote.setup.mcp_import import ImportPlanner
    from rote.setup.try_skill import Session

#: ``$ROTE_SETUP_STATE_PATH`` overrides where the picked skill is kept.
STATE_ENV = "ROTE_SETUP_STATE_PATH"
STATE_FORMAT = "rote-setup/1"

#: Every step, in order. ``--approve`` and ``--skip`` take these.
STEP_IDS: tuple[str, ...] = ("mcp-import", "login", "skill", "compile", "install", "eval")
#: Steps whose skip leaves the rest runnable. Skipping any other ends setup
#: there: the later steps build on it.
CONTINUES_WHEN_SKIPPED = frozenset({"mcp-import", "login"})
APPROVE_FLAG = "--approve"
#: How many of the user's own skills the skill step offers.
OFFERED_SKILLS = 10
SKIP_FLAG = "--skip"

#: How to pass each answer, for the agent relaying a question.
ANSWER_FORMAT: Mapping[str, str] = {
    "--only": "comma-separated server names from choices (or --skip mcp-import for none)",
    "--work-domain": "comma-separated domains, or none",
    "--login": "comma-separated server names from choices (or --skip login for none)",
    "--skill": "one skill path from choices",
    "--input": "the path of a JSON file",
}


class SetupError(RuntimeError):
    """``rote setup`` cannot go on with what it was given; the message names the fix."""


# ───────── What setup remembers ─────────


def state_path() -> Path:
    from rote.xdg import data_home

    override = os.environ.get(STATE_ENV)
    return Path(override) if override else data_home() / "rote" / "setup.json"


@dataclass(frozen=True)
class SetupState:
    """What setup keeps between runs: the picked skill, its input and work
    dir (written by the skill step's plan), the user's answers (so ``next``
    stays short: a re-run reads them instead of carrying them on argv), and
    what the user was offered and passed on, per step (so a partial import
    or login is not asked again)."""

    skill: Path | None = None
    input: Path | None = None
    out: Path | None = None
    answers: Mapping[str, Any] = field(default_factory=dict)
    passed_on: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    #: How many plans of each step an approval has applied. Part of the next
    #: approval's code, so a replayed ``--approve`` never applies twice.
    approvals: Mapping[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": STATE_FORMAT,
            "skill": None if self.skill is None else str(self.skill),
            "input": None if self.input is None else str(self.input),
            "out": None if self.out is None else str(self.out),
            "answers": dict(self.answers),
            "passed_on": {k: list(v) for k, v in self.passed_on.items()},
            "approvals": dict(self.approvals),
        }


def read_state() -> SetupState:
    path = state_path()
    if not path.is_file():
        return SetupState()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise SetupError(f"{path} does not parse ({e}). Fix: rote setup --reset") from None
    if not isinstance(data, dict) or data.get("format") != STATE_FORMAT:
        raise SetupError(f"{path} is not rote setup's. Fix: rote setup --reset")

    def as_path(key: str) -> Path | None:
        value = data.get(key)
        return Path(value) if isinstance(value, str) and value else None

    answers = data.get("answers")
    passed = data.get("passed_on")
    approvals = data.get("approvals")
    return SetupState(
        as_path("skill"),
        as_path("input"),
        as_path("out"),
        answers if isinstance(answers, dict) else {},
        {
            k: tuple(str(x) for x in v)
            for k, v in (passed if isinstance(passed, dict) else {}).items()
            if isinstance(v, list)
        },
        {
            k: v
            for k, v in (approvals if isinstance(approvals, dict) else {}).items()
            if isinstance(v, int)
        },
    )


def save_state(state: SetupState) -> None:
    """Setup's own bookkeeping file (owner-only), outside any plan: it holds
    only what the user answered and what setup did, never a decision."""
    from rote.interact.plan import atomic_write_text

    atomic_write_text(state_path(), json.dumps(state.to_dict(), indent=2) + "\n", file_mode=0o600)


# ───────── The answers given as flags ─────────


@dataclass(frozen=True)
class SetupFlags:
    from_client: str | None = None
    only: tuple[str, ...] | None = None
    work_domains: tuple[str, ...] = ()
    allow_url_secret: tuple[str, ...] = ()
    allow_untrusted: tuple[str, ...] = ()
    #: ``--login a,b``; None: ask.
    login: tuple[str, ...] | None = None
    skill: str | None = None
    input: str | None = None
    read_only: tuple[str, ...] = ()
    out: str | None = None
    trials: int = 2
    #: ``--approve <step>=<code>``: the user's yes to the plan whose code it is.
    approve: Mapping[str, str] = field(default_factory=dict)
    skip: frozenset[str] = frozenset()
    live: bool = False
    force: bool = False

    @classmethod
    def from_args(cls, args: Any, saved: Mapping[str, Any] | None = None) -> SetupFlags:
        """The flags of this run over the answers saved by earlier ones: a
        value flag given now wins, a list of yeses (``--read-only``,
        ``--skip``, ...) adds to the saved one."""
        from rote.setup.mcp_import import split_names

        saved = saved or {}

        def names(values: Sequence[str] | None) -> tuple[str, ...]:
            return split_names(values or [])

        def merged(key: str, given: Sequence[str] | None) -> tuple[str, ...]:
            return tuple(dict.fromkeys((*saved.get(key, ()), *names(given))))

        def latest(key: str, given: Any) -> Any:
            return given if given is not None else saved.get(key)

        approve: dict[str, str] = {}
        for item in names(args.approve):
            step, sep, code = item.partition("=")
            if step not in STEP_IDS:
                raise SetupError(
                    f"{APPROVE_FLAG} {item}: not a step. Fix: one of {', '.join(STEP_IDS)}"
                )
            if not sep or not code:
                raise SetupError(
                    f"{APPROVE_FLAG} {item}: an approval names the plan it approves "
                    f"({APPROVE_FLAG} {step}=<code>). Fix: run rote setup without it, "
                    "show the user the plan, and pass the code it prints"
                )
            approve[step] = code
        unknown = sorted(set(names(args.skip)) - set(STEP_IDS))
        if unknown:
            raise SetupError(
                f"{SKIP_FLAG} {unknown[0]}: not a step. Fix: one of {', '.join(STEP_IDS)}"
            )
        trials = latest("trials", args.trials) or DEFAULT_TRIALS
        if trials < 1:
            raise SetupError(f"--trials must be at least 1, got {trials}")
        only = latest("only", names(args.only) if args.only else None)
        login = latest("login", names(args.login) if args.login else None)
        return cls(
            from_client=args.from_client,
            only=None if only is None else tuple(only),
            work_domains=merged("work_domains", args.work_domain),
            allow_url_secret=merged("allow_url_secret", args.allow_url_secret),
            allow_untrusted=merged("allow_untrusted", args.allow_untrusted),
            login=None if login is None else tuple(login),
            skill=latest("skill", args.skill),
            input=latest("input", args.input),
            read_only=merged("read_only", args.read_only),
            out=latest("out", args.out),
            trials=trials,
            approve=approve,
            skip=frozenset(merged("skip", args.skip)),
            live=args.live,
            force=args.force,
        )

    def answers(self) -> dict[str, Any]:
        """What :meth:`from_args` reads back on the next run. Consent
        (``--approve``, ``--live``, ``--force``) is never saved: it is for
        the plan this run shows."""
        values: dict[str, Any] = {
            "only": None if self.only is None else list(self.only),
            "work_domains": list(self.work_domains),
            "allow_url_secret": list(self.allow_url_secret),
            "allow_untrusted": list(self.allow_untrusted),
            "login": None if self.login is None else list(self.login),
            "skill": self.skill,
            "input": self.input,
            "read_only": list(self.read_only),
            "out": self.out,
            "trials": self.trials,
            "skip": sorted(self.skip),
        }
        return {k: v for k, v in values.items() if v not in (None, [])}

    @property
    def granted(self) -> set[str]:
        return {f for f, given in ((LIVE_FLAG, self.live), (FORCE_FLAG, self.force)) if given}


# ───────── Step 2: log in ─────────


@dataclass(frozen=True)
class NeedsLogin:
    name: str
    config: McpServerConfig
    host: str
    #: :func:`rote.mcp.tokens.auth_status`.
    auth: str
    work: bool


def needs_login(work_domains: Sequence[str]) -> list[NeedsLogin]:
    """Every registered server a run could not connect to as things stand:
    no static headers, and no usable credential (:data:`rote.skills.discovery.USABLE_AUTH`)."""
    from urllib.parse import urlsplit

    from rote.clients.normalize import is_work_host
    from rote.mcp import auth_status, load_registry, read_token_file
    from rote.mcp.learned import learned_auth
    from rote.skills.discovery import USABLE_AUTH

    found = []
    for name, config in sorted(load_registry().servers.items()):
        if config.headers:
            continue
        learned = learned_auth(name, config.url)
        auth = auth_status(config, read_token_file(name), learned.auth if learned else None)
        if auth in USABLE_AUTH:
            continue
        host = urlsplit(config.url).hostname or config.url
        found.append(NeedsLogin(name, config, host, auth, is_work_host(host, tuple(work_domains))))
    return found


@dataclass
class LoginStep:
    """Log in to the registered servers that need it.

    A person gets ``rote mcp login``'s blocking browser login per server.
    Nobody at the terminal gets ``rote mcp login --no-wait``'s actions
    (:mod:`rote.setup.mcp_login`): the listener starts, and setup stops with
    the authorization URL for the agent to hand over (:attr:`started`)."""

    flags: SetupFlags
    work_domains: Callable[[], Sequence[str]]
    #: The blocking login for a person; injected by tests.
    login: Callable[[str], object] | None = None
    #: Servers the user was offered a login for and did not pick, remembered.
    passed_on: Callable[[], Collection[str]] = frozenset
    id: str = "login"
    title: str = "Log in to the servers that need it"
    command: str = "rote mcp login <name>"
    #: Filled while planning, for :func:`cmd_setup` to read after the apply.
    offered: list[NeedsLogin] = field(default_factory=list)
    chosen: list[str] = field(default_factory=list)
    started: dict[str, list[Started]] = field(default_factory=dict)
    #: Servers whose ``--no-wait`` login is already waiting for the browser.
    waiting: list[Any] = field(default_factory=list)

    def status(self, ctx: SetupContext) -> StepStatus:
        pending = needs_login(self.work_domains())
        passed = set(self.passed_on())
        left = [n for n in pending if n.name not in passed]
        if left:
            names = ", ".join(n.name for n in left)
            return StepStatus("pending", f"{len(left)} not logged in: {names}")
        if pending:
            names = ", ".join(n.name for n in pending)
            return StepStatus("done", f"you chose not to log in to {names} (rote mcp login <name>)")
        return StepStatus("done", "every registered server can connect")

    def plan(self, ctx: SetupContext, prompter: Prompter) -> Plan:
        from rote.mcp.pending_login import live_for_server
        from rote.setup.mcp_import import login_action
        from rote.setup.mcp_login import no_wait_actions

        passed = set(self.passed_on())
        self.offered = [n for n in needs_login(self.work_domains()) if n.name not in passed]
        by_name = {n.name: n for n in self.offered}
        if self.flags.login is not None:
            unknown = [n for n in self.flags.login if n not in by_name]
            if unknown:
                raise SetupError(
                    f"--login {unknown[0]}: not a registered server that needs a login "
                    f"(these do: {', '.join(by_name) or 'none'})"
                )
            chosen = list(self.flags.login)
        else:
            choices = [
                Choice(n.name, n.name, f"{n.host}  ·  {n.auth}" + ("  ·  WORK" if n.work else ""))
                for n in self.offered
            ]
            chosen = prompter.checkbox(
                "Log in to which servers (each one opens your browser to approve rote)?",
                choices,
                flag="--login",
                default=[n.name for n in self.offered if not n.work],
            )
        actions: list[Action] = []
        self.waiting = []
        self.chosen = list(chosen)
        for name in chosen:
            n = by_name[name]
            if prompter.interactive:
                actions.append(login_action(name, n.host, self.login))
                continue
            already = live_for_server(name)
            if already is not None:
                self.waiting.append(already)
                continue
            more, started = no_wait_actions(name, n.config)
            self.started[name] = started
            actions.extend(more)
        skipped = [n.name for n in self.offered if n.name not in chosen]
        notes = [f"{name}: later: rote mcp login {name}" for name in skipped]
        notes += [f"{p.server}: a login is already waiting for the browser" for p in self.waiting]
        return Plan("Log in to MCP servers", tuple(actions), tuple(notes))

    def ask(self, plan: Plan) -> str:
        """The login plan as one question for an agent to put to its user."""
        from rote.mcp.pending_login import PENDING_TTL_SECONDS

        names = [
            a.id.removeprefix("start-login-")
            for a in plan.actions
            if a.id.startswith("start-login-")
        ]
        names = names or [a.id.removeprefix("login-") for a in plan.actions]
        listed = (
            " and ".join(names) if len(names) <= 2 else ", ".join(names[:-1]) + f" and {names[-1]}"
        )
        return (
            f"Log in to {listed}? rote opens a browser page for each one for you to approve, "
            f"and waits up to {PENDING_TTL_SECONDS // 60} min."
        )

    def authorizations(self) -> list[Started]:
        """The logins that now wait for the user's browser."""
        return [s for started in self.started.values() for s in started if s.kind == "authorize"]


# ───────── Step 3: pick a skill ─────────


@dataclass(frozen=True)
class Offered:
    path: Path
    name: str
    #: The one line that explains the rank.
    reason: str
    where: str
    bundled: bool
    #: What ``--skill`` takes for it: the name, or ``name@where`` when two
    #: offered skills share a name. Short, so ``next`` carries no path.
    key: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "path": str(self.path),
            "name": self.name,
            "reason": self.reason,
            "where": self.where,
            "demo": self.bundled,
        }


def offered_skills(limit: int) -> list[Offered]:
    """The user's own skills, best ranked first (:func:`rote.skills.discover`),
    then rote's demo, so a first run with no skills still has one to try."""
    from rote.skills import discover

    mine: list[Offered] = []
    demos: list[Offered] = []
    for s in discover().skills:
        if s.readiness.state == "unreadable":
            continue
        root = s.found_in[0]
        bundled = root.scope == "bundled"
        where = (
            "rote's demo"
            if bundled
            else root.client + (f" plugin {root.plugin}" if root.plugin else f" {root.scope}")
        )
        (demos if bundled else mine).append(
            Offered(s.path, s.name, s.readiness.reason, where, bundled)
        )
    offered = [*mine[:limit], *demos[:1]]
    counts: dict[str, int] = {}
    for o in offered:
        counts[o.name] = counts.get(o.name, 0) + 1

    def key(o: Offered) -> str:
        if counts[o.name] == 1:
            return o.name
        where = "demo" if o.bundled else re.sub(r"[^a-z0-9]+", "-", o.where.lower()).strip("-")
        return f"{o.name}@{where}"

    return [replace(o, key=key(o)) for o in offered]


def _registry() -> tuple[Any, Any]:
    from rote.mcp.registry import parse_registry, registry_path

    read = snapshot(registry_path())
    return read, parse_registry(read.data)


@dataclass
class SkillStep:
    """Pick the skill, its input, and say yes to each server the comparison
    calls live. Its plan saves the pick (:func:`state_path`) and marks those
    servers read-only in rote's registry."""

    flags: SetupFlags
    id: str = "skill"
    title: str = "Pick a skill to compile"
    command: str = "rote skills list"
    offered: list[Offered] = field(default_factory=list)

    def _flag_skill(self, prompter: Prompter) -> Path | None:
        from rote.setup.try_skill import choose_skill

        if self.flags.skill is None:
            return None
        keyed = [o.path for o in offered_skills(OFFERED_SKILLS) if o.key == self.flags.skill]
        if keyed:
            return keyed[0]
        try:
            return choose_skill(self.flags.skill, prompter)
        except TryError as e:
            raise SetupError(f"--skill {e}") from None

    def status(self, ctx: SetupContext) -> StepStatus:
        from rote.interact.prompter import NonInteractivePrompter
        from rote.setup.try_skill import require_live_servers

        state = read_state()
        if state.skill is None or not (state.skill / "SKILL.md").is_file():
            return StepStatus("pending", "no skill picked")
        flagged = self._flag_skill(NonInteractivePrompter(why="reading status"))
        if flagged is not None and flagged.resolve() != state.skill.resolve():
            return StepStatus("pending", f"--skill names {flagged}, not {state.skill}")
        try:
            require_live_servers(state.skill)
        except TryError:
            return StepStatus("pending", f"{state.skill.name}: a server it calls is not read-only")
        return StepStatus("done", f"picked {state.skill.name} ({state.skill})")

    def plan(self, ctx: SetupContext, prompter: Prompter) -> Plan:
        from rote.adapters.skill import read_identity
        from rote.compiler.tool_grounding import servers_named
        from rote.setup.try_skill import default_out, input_path, load_task

        skill_dir = self._flag_skill(prompter)
        self.offered = offered_skills(OFFERED_SKILLS)
        if skill_dir is None:
            if not self.offered:
                raise SetupError("no skills found. Fix: rote setup --skill <dir with a SKILL.md>")
            best = self.offered[0]
            why = (
                "none of your skills could be read, so rote's demo"
                if best.bundled
                else f"{best.reason}"
            )
            ctx.out.say(f"  Recommended: {best.name} ({best.where}): {why}")
            choices = [Choice(o.key, o.name, f"{o.where}  ·  {o.reason}") for o in self.offered]
            picked = prompter.select(
                "Which skill should rote compile and compare?",
                choices,
                flag="--skill",
                default=choices[0].value,
            )
            skill_dir = next(o.path for o in self.offered if o.key == picked)
        skill_dir = skill_dir.resolve()
        name = read_identity(skill_dir).name
        task_path = input_path(skill_dir, self.flags.input, prompter)
        load_task(task_path)  # a bad input is refused now, not after the compile

        read, registry = _registry()
        named = servers_named(skill_dir, registry.servers.keys())
        marks: list[str] = []
        notes: list[str] = [f"input: {task_path}"]
        for server in sorted(named):
            entry = registry.servers.get(server)
            if entry is None:
                notes.append(
                    f"{server}: {name} calls it and it is not registered; the compile waits "
                    f"for: rote mcp add {server} <url> --read-only"
                )
                continue
            if entry.read_only is True:
                continue
            from urllib.parse import urlsplit

            host = urlsplit(entry.url).hostname or entry.url
            if server in self.flags.read_only:
                marks.append(server)
                continue
            marked = _tools_marked_read_only(server)
            if marked:
                # The server says which of its tools only read: the comparison
                # calls only those, so there is nothing to vouch for.
                notes.append(
                    f"{server}: the comparison calls only the {marked} tool(s) it marks read-only"
                )
            elif self._vouch(prompter, server, host):
                marks.append(server)
            else:
                notes.append(
                    f"{server}: not marked read-only, so the comparison cannot call it; the "
                    "compile waits for it"
                )
        unknown = sorted(set(self.flags.read_only) - set(named))
        if unknown:
            raise SetupError(
                f"--read-only {unknown[0]}: {name} does not call it "
                f"(it calls: {', '.join(sorted(named)) or 'no MCP server'})"
            )
        state = read_state()
        out = state.out if state.skill == skill_dir and state.out else default_out(name)
        if self.flags.out is not None:
            out = Path(self.flags.out).resolve()
        new = replace(state, skill=skill_dir, input=task_path, out=out)
        state_read = snapshot(state_path())
        actions = [
            write_file_action(
                id="remember-skill",
                title=f"Remember {name} for the next steps",
                read=state_read,
                content=json.dumps(new.to_dict(), indent=2) + "\n",
                summary=f"skill {name}, its input, and the work dir {out}",
                replaces=False,
                file_mode=0o600,
                command=None,
            )
        ]
        if marks:
            actions.append(self._mark_read_only(read, registry, marks))
        return Plan(f"Pick {name}", tuple(actions), tuple(notes))

    @staticmethod
    def _vouch(prompter: Prompter, server: str, host: str) -> bool:
        """The user's word that ``server`` only reads. A person is asked
        (default no). Nobody at the terminal stops for it, ``--yes`` too: a
        default is not anyone's word."""
        question = (
            f"Let rote call any tool on {server} ({host}) without asking? Say yes only if "
            "none of its tools can change anything."
        )
        flag = f"--read-only {server}"
        if not prompter.interactive:
            raise PromptRequired(
                question,
                flag,
                "only the user can vouch for a server, so --yes does not answer this",
                takes_value=False,
            )
        return prompter.confirm(question, flag=flag, default=False)

    @staticmethod
    def _mark_read_only(read: Any, registry: Any, servers: Sequence[str]) -> Action:
        from rote.mcp.registry import REGISTRY_FILE_MODE, render_registry

        merged = registry.model_copy(deep=True)
        for server in servers:
            merged.servers[server].read_only = True
        return write_file_action(
            id="mark-read-only",
            title=f"Mark {', '.join(servers)} read-only",
            read=read,
            content=render_registry(merged),
            summary=f"read_only: true on {', '.join(servers)}",
            replaces=False,
            file_mode=REGISTRY_FILE_MODE,
        )


# ───────── The run ─────────


class AuthorizationPending(RuntimeError):
    """Logins started for an agent's user: setup waits for their browser."""

    def __init__(self, started: Sequence[Started], waiting: Sequence[Any]) -> None:
        self.started = tuple(started)
        self.waiting = tuple(waiting)
        super().__init__("a login waits for the browser")


@dataclass
class Stopped:
    """Setup ended before the verdict, by choice or by failure."""

    step_id: str
    why: str
    failed: bool


@dataclass
class Runner:
    mode: Mode
    out: Output
    prompter: Prompter
    flags: SetupFlags
    argv: list[str]
    ctx: SetupContext
    planners: list[ImportPlanner] = field(default_factory=list)
    reports: list[StepReport] = field(default_factory=list)
    current: Step | None = None
    index: int = 0
    #: Steps whose ``--approve`` code was taken this run.
    used: set[str] = field(default_factory=set)

    def one(self, step: Step, index: int) -> Stopped | None:
        self.current, self.index = step, index
        total = len(STEP_IDS)
        if step.id in self.flags.skip:
            self.out.say(f"Step {index}/{total}  {step.title}", style="bold")
            self.out.say(f"  skipped ({SKIP_FLAG} {step.id})")
            self.reports.append(StepReport(step.id, "skipped", "skipped by the user"))
            return None if step.id in CONTINUES_WHEN_SKIPPED else Stopped(step.id, "skipped", False)
        (report,) = run_steps(
            [step],
            self.ctx,
            self.prompter,
            granted=self.flags.granted,
            yes=self._approved,
            first=index,
            total=total,
        )
        self.reports.append(report)
        if report.state in ("applied", "failed"):
            self._count_used(step)
        if report.state in ("applied", "declined"):
            self._remember_passed_on(step)
        for outcome in report.outcomes:
            if outcome.status != "done":
                self.out.say(f"  {outcome.status}: {outcome.action_id} ({outcome.error})")
        if report.state == "declined":
            if step.id in CONTINUES_WHEN_SKIPPED:
                self.out.say(f"  skipped; later: {step.command}")
                return None
            return Stopped(step.id, "not applied", False)
        if report.state == "failed":
            return Stopped(step.id, report.detail, True)
        if isinstance(step, LoginStep) and not self.prompter.interactive:
            pending = step.authorizations()
            if pending or step.waiting:
                raise AuthorizationPending(pending, step.waiting)
        return None

    def _approved(self, step: Step, plan: Plan) -> bool:
        """``--yes``, or an ``--approve`` whose code is this plan's. A used
        approval is counted, which changes the next plan's code."""
        if self.mode.yes:
            return True
        given = self.flags.approve.get(step.id)
        if given is None or given != plan_code(step.id, plan):
            return False
        if any(f not in self.flags.granted for f in plan.needed_flags()):
            # Not applied yet (a spend's flag is missing): the approval stays
            # unused, and the stop asks for it again with the same code.
            return False
        # Counted once the plan ran (one() does it): counting here would
        # change the state file under a plan that writes it.
        self.used.add(step.id)
        return True

    def _count_used(self, step: Step) -> None:
        if step.id not in self.used:
            return
        state = read_state()
        used = dict(state.approvals)
        used[step.id] = used.get(step.id, 0) + 1
        save_state(replace(state, approvals=used))

    def _remember_passed_on(self, step: Step) -> None:
        """What the user saw at this step and did not take, so the next run
        does not offer it again."""
        from rote.mcp import load_registry
        from rote.mcp._runtime_helper import normalize_server_url
        from rote.setup.mcp_import import _url_key, candidate_key

        if step.id == "mcp-import" and self.planners:
            # Offered and not registered now: what the user passed on. A
            # picked server is not in it, so removing it later offers it again.
            registered = {normalize_server_url(c.url) for c in load_registry().servers.values()}
            seen = [
                candidate_key(c) for c in self.planners[-1].primary if _url_key(c) not in registered
            ]
        elif isinstance(step, LoginStep):
            seen = [n.name for n in step.offered if n.name not in step.chosen]
        else:
            return
        state = read_state()
        passed = dict(state.passed_on)
        passed[step.id] = tuple(dict.fromkeys((*passed.get(step.id, ()), *seen)))
        save_state(replace(state, passed_on=passed))

    # ── the stops, as one JSON object each ──

    def _context(self) -> dict[str, Any]:
        step = self.current
        payload: dict[str, Any] = {
            "step": None
            if step is None
            else {"id": step.id, "title": step.title, "index": self.index, "total": len(STEP_IDS)},
            "steps": [
                {"id": r.step_id, "state": r.state, "detail": r.detail} for r in self.reports
            ],
        }
        if isinstance(step, LoginStep):
            payload["servers"] = [
                {"name": n.name, "host": n.host, "auth": n.auth, "work": n.work}
                for n in step.offered
            ]
        if isinstance(step, SkillStep) and step.offered:
            payload["skills"] = [o.to_dict() for o in step.offered]
        if step is not None and step.id == "mcp-import" and self.planners:
            payload["servers"] = [c.to_dict() for c in self.planners[-1].found]
        return payload

    def asked(self, error: PromptRequired) -> dict[str, Any]:
        payload = action_required(error, self.argv)
        payload.update(self._context())
        labels = [str(c) for c in error.choices]
        if error.flag == "--skill" and "skills" in payload:
            by_key = {s["key"]: s for s in payload["skills"]}
            labels = [f"{c} ({by_key[c]['where']})" if c in by_key else c for c in error.choices]
        choices = f" Choices: {'; '.join(labels)}." if labels else ""
        payload["ask"] = f"{error.question.rstrip()}{choices}"
        how = ANSWER_FORMAT.get(error.flag)
        if how:
            payload["answer_with"] = f"{error.flag} <{how}>"
        return payload

    def confirm(self, error: ConfirmationRequired) -> dict[str, Any]:
        step = self.current
        assert step is not None
        # --yes would approve every later plan too. --approve <step>=<code>
        # approves this plan, as shown, once.
        needs = [f for f in error.missing if f != YES_FLAG]
        code = plan_code(step.id, error.plan)
        mine = [f"{APPROVE_FLAG} {step.id}={code}"] if YES_FLAG in error.missing else []
        approve = [w for f in (*mine, *needs) for w in f.split()]
        payload = action_required(error, self.argv)
        payload["missing"] = [*mine, *needs]
        payload["message"] = (
            f"{error.plan.title}: not applied. Ask the user; pass {' '.join(approve)} "
            "if they say yes (nothing was written, contacted or spent)"
        )
        payload["ask"] = (
            step.ask(error.plan) if isinstance(step, LoginStep) else plain_plan(error.plan)
        )
        payload["approval"] = {"step": step.id, "code": code}
        payload["next"] = [
            {"command": next_command(self.argv, approve), "when": "the user says yes"},
            {
                "command": next_command(self.argv, [SKIP_FLAG, step.id]),
                "when": "the user says no"
                + ("" if step.id in CONTINUES_WHEN_SKIPPED else " (setup ends at this step)"),
            },
        ]
        payload.update(self._context())
        return payload

    def person(self, error: PersonRequired) -> dict[str, Any]:
        payload = person_required(error)
        payload["ask"] = (
            f"Run this in your own terminal, then tell me when it is done: {error.command}"
        )
        payload["next"] = [
            *payload["next"],
            {"command": next_command(self.argv, []), "when": "the user ran it"},
        ]
        payload.update(self._context())
        return payload

    def authorizing(self, error: AuthorizationPending) -> dict[str, Any]:
        logins = [
            authorization_pending(
                server=s.pending.server,
                pending_id=s.pending.id,
                authorization_url=s.authorization_url,
                expires_at=s.pending.expires_at,
            )
            for s in error.started
        ]
        logins += [
            authorization_pending(
                server=p.server,
                pending_id=p.id,
                authorization_url=None,
                expires_at=p.expires_at,
                still_waiting=True,
            )
            for p in error.waiting
        ]
        urls = [
            f"{x['server']}: {x['authorization_url']}" for x in logins if x["authorization_url"]
        ]
        completes = [x["next"][0] for x in logins]
        return {
            "ok": False,
            "status": "action_required",
            "reason": "authorization_pending",
            "message": "a login waits for the user to approve it in a browser",
            "ask": "Approve rote's access in your browser"
            + (f" ({'; '.join(urls)})" if urls else "")
            + ", then tell me when you are done.",
            "logins": logins,
            "next": [
                *completes,
                {"command": next_command(self.argv, []), "when": "every login above completed"},
            ],
            **self._context(),
        }


def _tools_marked_read_only(server: str) -> int:
    """How many of ``server``'s tools it marks ``readOnlyHint`` itself (a
    read-only ``tools/list``). 0 when it marks none or cannot be listed now,
    which leaves the question to the user."""
    import asyncio

    from rote.eval.baseline import read_only_allowlist

    allowed, _denied, skipped = asyncio.run(read_only_allowlist([server]))
    return 0 if server in skipped else len(allowed)


def plan_code(step_id: str, plan: Plan) -> str:
    """What ``--approve <step>=<code>`` must name: every action and effect of
    the plan (paths, hosts, the spend range), the picked skill, and how many
    approvals the step has used. A different plan, another skill or a replay
    of a used approval has another code, so it stops to be asked again."""
    import hashlib

    state = read_state()
    material = {
        "step": step_id,
        "plan": plan.to_dict(),
        "skill": None if state.skill is None else str(state.skill),
        "used": state.approvals.get(step_id, 0),
    }
    digest = hashlib.sha256(json.dumps(material, sort_keys=True, default=str).encode())
    return digest.hexdigest()[:12]


def plain_plan(plan: Plan) -> str:
    """The plan in two or three sentences, for an agent to put to its user:
    what each action does, the money and time, and the hosts it contacts.
    Every effect is still in the payload's ``plan``."""
    from rote.interact.plan import Contact, Spend

    did = "; ".join(a.title[0].lower() + a.title[1:] for a in plan.actions)
    spend = [e.describe() for e in plan.effects if isinstance(e, Spend)]
    hosts = sorted({e.host for e in plan.effects if isinstance(e, Contact)})
    timing = [n for n in plan.notes if n.startswith(("already spent", "in all", "takes"))]
    parts = [f"{plan.title}: {did}."]
    if spend:
        parts.append(f"It will {' and '.join(spend)}.")
    if timing:
        parts.append(" ".join(n[0].upper() + n[1:] + "." for n in timing))
    if hosts:
        parts.append(f"It contacts {', '.join(hosts)}.")
    parts.append("Go ahead?")
    return " ".join(parts)


def _client_line(mode: Mode) -> str:
    from rote.clients import source

    detected = mode.detected
    if detected is None:
        return "No AI client in the environment: rote reads every client config on this machine."
    try:
        label = source(detected.client).label
    except KeyError:
        label = detected.client
    if mode.agent:
        return (
            f"Running under {label} ({detected.var}): each question comes back as JSON for "
            f"{label} to ask you; nothing is decided for you."
        )
    return f"In {label}'s terminal ({detected.var}): its MCP servers are offered first."


def _passed_on(step_id: str) -> Callable[[], Collection[str]]:
    return lambda: read_state().passed_on.get(step_id, ())


def _import_step(mode: Mode, flags: SetupFlags, planners: list[ImportPlanner]) -> Step:
    """``rote mcp import``'s step, with its planner kept in ``planners`` so a
    stop can show the agent what was found."""
    from rote import config as rote_config
    from rote.clients import ClientContext
    from rote.mcp.registry import registry_path
    from rote.setup.mcp_import import ImportFlags, ImportPlanner, McpImportStep

    layers = rote_config.load_layers()
    import_flags = ImportFlags(
        from_client=flags.from_client,
        only=flags.only,
        allow_url_secret=flags.allow_url_secret,
        allow_untrusted=flags.allow_untrusted,
        work_domains=flags.work_domains,
        # Logins are setup's own step, after the import.
        login=(),
    )

    def make(prompter: Prompter, out: Output) -> ImportPlanner:
        planner = ImportPlanner(
            flags=import_flags,
            prompter=prompter,
            out=out,
            ctx=ClientContext.current(),
            layers=layers,
            registry_path=registry_path(),
            user_config_path=rote_config.user_config_path(),
            detected_client=mode.detected.client if mode.detected is not None else None,
        )
        planners.append(planner)
        return planner

    return McpImportStep(make, passed_on=_passed_on("mcp-import"))


def _work_domains() -> Sequence[str]:
    from rote import config as rote_config

    return rote_config.resolve_list("work_domains", layers=rote_config.load_layers()).values


def _session(flags: SetupFlags, out: Output) -> Session:
    from rote.setup.try_skill import load_task, session_for

    state = read_state()
    assert state.skill is not None and state.input is not None
    return session_for(
        state.skill,
        load_task(state.input),
        out=flags.out or state.out,
        trials=flags.trials,
        log=out.say,
    )


#: Why ``rote init``'s defaults are not a step here: each one it sets
#: (runtime, deploy target, driver, model) is one ``rote try`` pins itself.
INIT_SKIPPED = (
    "rote init's defaults are not a step here: rote try pins its own runtime, driver "
    "and model. rote init sets them for your later compiles."
)


def base_argv(args: Any) -> list[str]:
    """The words every ``next`` command starts with: the run's mode, never
    an answer (those are saved), so the command stays short."""
    words = ["setup"]
    if args.agent_mode is not None:
        words += ["--agent-mode", args.agent_mode]
    if args.json:
        words.append("--json")
    if args.from_client is not None:
        words += ["--from", args.from_client]
    return words


def cmd_setup(args: Any) -> int:
    """``rote setup``. Exit 0: the verdict says the compiled skill gave the
    same answer. 1: it did not, setup stopped before the verdict, or a step
    failed. 2: an answer or a consent is missing (``action_required``)."""
    from rote.config import ConfigError
    from rote.interact.mode import Mode
    from rote.interact.output import Output
    from rote.interact.prompter import PromptCancelled, prompter_for
    from rote.mcp.auth import McpAuthError
    from rote.mcp.pending_login import PendingLoginError
    from rote.setup.mcp_import import McpImportError
    from rote.setup.try_skill import next_steps, render, try_steps
    from rote.skill_target.trust import SkillTrustError

    try:
        mode = Mode.resolve(json=args.json, yes=args.yes, agent_mode=args.agent_mode)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    out = Output(mode)
    try:
        if args.reset:
            state_path().unlink(missing_ok=True)
            out.say("rote setup: forgot the saved answers and the picked skill")
        state = read_state()
        flags = SetupFlags.from_args(args, state.answers)
        if flags.answers() != dict(state.answers):
            save_state(replace(state, answers=flags.answers()))
    except SetupError as e:
        out.error(str(e))
        out.result({"ok": False, "error": str(e)})
        return 2
    prompter = prompter_for(mode)
    runner = Runner(mode, out, prompter, flags, base_argv(args), SetupContext(mode, out))
    out.say(f"rote setup: {_client_line(mode)}")
    out.say(f"  {INIT_SKIPPED}")
    session: Session | None = None
    stopped: Stopped | None = None
    try:
        first: list[Step] = [
            _import_step(mode, flags, runner.planners),
            LoginStep(flags, _work_domains, passed_on=_passed_on("login")),
            SkillStep(flags),
        ]
        for index, step in enumerate(first, start=1):
            stopped = runner.one(step, index)
            if stopped is not None:
                break
        if stopped is None:
            session = _session(flags, out)
            out.say(f"  working in {session.out}")
            for index, step in enumerate(try_steps(session), start=len(first) + 1):
                stopped = runner.one(step, index)
                if stopped is not None:
                    break
    except PromptRequired as e:
        out.error(str(e))
        out.result(runner.asked(e))
        return EXIT_ACTION_REQUIRED
    except ConfirmationRequired as e:
        payload = runner.confirm(e)
        out.error(payload["message"])
        out.result(payload)
        return EXIT_ACTION_REQUIRED
    except PersonRequired as e:
        out.error(str(e))
        out.result(runner.person(e))
        return EXIT_ACTION_REQUIRED
    except AuthorizationPending as e:
        payload = runner.authorizing(e)
        for login in payload["logins"]:
            if login["authorization_url"]:
                out.say(f"  approve {login['server']}: {login['authorization_url']}")
        out.say(f"  then: {payload['next'][0]['command']}, and rote setup again")
        out.result(payload)
        return EXIT_ACTION_REQUIRED
    except (
        SetupError,
        TryError,
        McpImportError,
        ConfigError,
        McpAuthError,
        PendingLoginError,
        ValueError,
    ) as e:
        out.error(str(e))
        out.result({"ok": False, "error": str(e), **runner._context()})
        return 1
    except SkillTrustError as e:
        message = f"{e.cause} Fix: {e.fix}"
        out.error(message)
        out.result({"ok": False, "error": message, **runner._context()})
        return 1
    except (PromptCancelled, KeyboardInterrupt):
        out.say("rote setup: stopped. Run rote setup again to pick up from this step")
        return 130
    steps_json = [r.to_dict() for r in runner.reports]
    if stopped is not None:
        verb = "failed at" if stopped.failed else "stopped at"
        out.say(f"rote setup: {verb} {stopped.step_id} ({stopped.why}). rote setup picks up here")
        out.result(
            {"ok": False, "stopped_at": stopped.step_id, "why": stopped.why, "steps": steps_json}
        )
        return 1
    assert session is not None
    summary = session.result()
    if summary is None:
        out.error("the comparison finished but recorded no result. Fix: run rote setup again")
        return 1
    follow = next_steps(session, summary, command="rote setup")
    out.say()
    for line in render(summary, name=session.name, next_steps=follow):
        out.say(line)
    out.result(
        {
            "ok": bool(summary["ok"]),
            "status": "done",
            "out": str(session.out),
            "steps": steps_json,
            "next": [{"command": step} for step in follow],
            **{k: v for k, v in summary.items() if k != "report"},
        }
    )
    return 0 if summary["ok"] else 1
