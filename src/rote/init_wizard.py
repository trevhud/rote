"""``rote init``: choose the defaults every later command reads, once.

Asks where compiled pipelines run (rote cloud or a local runtime), which
compiler driver and model compile them, and who decides a local
compile's structure (the structure pass, the decision model that checks
it, the tie-break). Then it shows a plan (the config file it writes, and a
rote cloud login when one was asked for) and applies it only on a yes. A
chosen runtime whose extra is missing gets a note with the install hint.

Built on :mod:`rote.interact`, so every question has a flag
(``--deploy``, ``--runtime``, ``--agent``, ``--model``, ``--structure``,
``--decider``, ``--tiebreak``, ``--login``) and the same code runs for a
person, a script and an agent: a person is asked what no flag answered, a
script with ``--yes`` takes the defaults, and anything else stops naming
the flag to pass (:mod:`rote.interact.prompter`). ``rote setup`` will
absorb this as one of its steps; the questions and the plan are the seam.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from rote.config import CONFIG_KEYS, render_config, user_config_path
from rote.extras import dependency_hint, install_hint
from rote.interact.output import Output
from rote.interact.plan import Action, Contact, Launch, Plan, snapshot, write_file_action
from rote.interact.prompter import Choice, Prompter
from rote.setup.install_extra import missing_extra_note

#: Runtime menu: (name, one-line pitch). Order = recommendation order,
#: the compile default (rote.cli.DEFAULT_COMPILE_RUNTIME, skill) first.
RUNTIME_MENU: tuple[tuple[str, str], ...] = (
    (
        "skill",
        "a skill your agent runs, its fixed steps in code: the original's answer "
        "in fewer tokens (recommended)",
    ),
    ("dbos", "durable Python, no orchestrator process to run"),
    ("temporal", "workflow.py + activities.py for a Temporal cluster"),
    ("cloudflare", "TypeScript Workflows on Cloudflare Workers"),
    ("dbos-ts", "durable TypeScript for DBOS Transact (Postgres)"),
    ("inngest", "TypeScript function app that mounts into Node/Next.js"),
    ("python", "plain script, maximum legibility (no HITL gates)"),
)

#: The value of a question whose answer is "leave the setting unset".
AUTO = "auto"


def _structure_menu() -> tuple[tuple[str, str], ...]:
    from rote.compiler.system1.structure import MODELS

    return (
        (
            "haiku",
            f"one {MODELS['haiku']} call reads the whole skill, checked by code (default)",
        ),
        ("sonnet", f"the same pass on {MODELS['sonnet']}: slower, costs more"),
        ("rules", "rules over the skill's text: no call"),
    )


def _decider_menu() -> tuple[tuple[str, str], ...]:
    from rote.compiler.system1 import decision_model as dm

    order = " else ".join(f"{m.label}" for m in dm.CHECKER_ORDER)
    return (
        (
            AUTO,
            f"the first decision model with credentials ({order}), else no check (default)",
        ),
        ("rules", "no check: nothing leaves this machine"),
        (
            "clef",
            f"Cloudflare's Clef, needs {dm.CLEF_ACCOUNT_ENV} and {dm.CLEF_TOKEN_ENV} "
            f"(~${dm.CLEF.price_per_m_input}/M input tokens)",
        ),
        (
            "jev",
            f"Typesafe's Jev, needs {dm.TYPESAFE_KEY_ENV} (~${dm.JEV.price_per_m_input}/M input "
            "tokens)",
        ),
    )


def decider_credentials_note(decider: str) -> str:
    """What the chosen checker needs, named from the decision-model module
    itself, so this text cannot drift from the variables a compile reads."""
    from rote.compiler.system1 import decision_model as dm

    if decider == "clef":
        return (
            f"Clef runs on your Cloudflare account through Workers AI: set {dm.CLEF_ACCOUNT_ENV} "
            f"and {dm.CLEF_TOKEN_ENV} before compiling (a compile without them fails rather "
            "than skipping the check)"
        )
    if decider == "jev":
        return (
            f"Jev runs on Typesafe's API: set {dm.TYPESAFE_KEY_ENV} before compiling (a compile "
            "without it fails rather than skipping the check)"
        )
    if decider == AUTO:
        _, why = dm.default_checker()
        return f"with this machine's environment today: {why}"
    return "the structure pass runs without a second opinion"


@dataclass(frozen=True)
class InitFlags:
    """The answers given as flags. None: not given, so ask (or take the default)."""

    deploy: str | None = None
    runtime: str | None = None
    agent: str | None = None
    model: str | None = None
    structure: str | None = None
    decider: str | None = None
    tiebreak: str | None = None
    #: True: log in to rote cloud; False: don't; None: ask a person, and
    #: skip when nobody is at the terminal (a login waits on a browser).
    login: bool | None = None


class InitError(ValueError):
    """Flags that contradict each other."""


def _menu(items: tuple[tuple[str, str], ...]) -> list[Choice[str]]:
    return [Choice(name, name, pitch) for name, pitch in items]


def _check_flag(flag: str, value: str | None, choices: list[Choice[str]]) -> None:
    if value is not None and value not in {c.value for c in choices}:
        known = ", ".join(c.value for c in choices)
        raise InitError(f"{flag} {value!r} is not valid; expected one of: {known}")


def build_plan(
    flags: InitFlags,
    *,
    prompter: Prompter,
    out: Output,
    project: bool,
    cwd: Path | None = None,
) -> Plan:
    """Ask what the flags left open, then plan the write (and the login)."""
    from rote.cli import _build_doctor_report
    from rote.cloud_auth import DEFAULT_CLOUD_URL, load_credential
    from rote.compiler.drivers import available_drivers

    out.say("rote init: set your defaults once; every command reads them.")
    report = _build_doctor_report()
    out.say("Compiler drivers (at least one required; `rote doctor` has the full preflight):")
    for driver in report["drivers"]:
        mark = "✓" if driver["available"] else "✗"
        suffix = "" if driver["available"] else f" ({driver['reason']})"
        out.say(f"  {mark} {driver['name']}{suffix}")
    for dep in report.get("dependencies", []):
        if not dep["installed"]:
            out.warn(f"this install lacks {dependency_hint(str(dep['module']))}")
    for runtime in report["runtimes"]:
        if not runtime["installed"]:
            out.say(f"  extra not installed: {runtime['name']} ({install_hint(runtime['name'])})")

    values: dict[str, str] = {}

    # ── Hosting ──
    hosting = [
        Choice("none", "locally", "a compiled skill or a workflow runtime (recommended)"),
        Choice(
            "rote-cloud",
            "rote cloud",
            "hosted Cloudflare Workflows; compile and deploy in one step",
        ),
    ]
    runtimes = _menu(RUNTIME_MENU)
    _check_flag("--deploy", flags.deploy, hosting)
    _check_flag("--runtime", flags.runtime, runtimes)
    deploy = flags.deploy or prompter.select(
        "Where should compiled pipelines run?", hosting, flag="--deploy", default="none"
    )
    values["deploy"] = deploy
    if deploy == "rote-cloud":
        if flags.runtime not in (None, "cloudflare"):
            raise InitError(
                f"--deploy rote-cloud runs the cloudflare runtime, not {flags.runtime}; "
                "pass --deploy none to emit another runtime locally"
            )
        values["runtime"] = "cloudflare"
    else:
        values["runtime"] = flags.runtime or prompter.select(
            "Which runtime should rote emit by default?",
            runtimes,
            flag="--runtime",
            default=RUNTIME_MENU[0][0],
        )

    # ── Compiler driver and model ──
    drivers = available_drivers()
    agent_choices = [Choice(AUTO, AUTO, "first available at run time (recommended)")] + [
        Choice(name, name, "available" if ok else reason) for name, ok, reason in drivers
    ]
    _check_flag("--agent", flags.agent, agent_choices)
    agent = flags.agent or prompter.select(
        "Which agent should run compilations?", agent_choices, flag="--agent", default=AUTO
    )
    if agent != AUTO:
        values["agent"] = agent
    model = (
        flags.model
        if flags.model is not None
        else prompter.text(
            "Compiler model (blank for the driver default)", flag="--model", default=""
        )
    )
    if model.strip():
        values["model"] = model.strip()

    # ── Structure, its checker, and the tie-break (local compiles) ──
    structure_choices = _menu(_structure_menu())
    _check_flag("--structure", flags.structure, structure_choices)
    structure = flags.structure or prompter.select(
        "Who drafts the pipeline's structure?",
        structure_choices,
        flag="--structure",
        default="haiku",
    )
    if structure != "haiku":
        values["structure"] = structure

    decider_choices = _menu(_decider_menu())
    _check_flag("--decider", flags.decider, decider_choices)
    decider = flags.decider or prompter.select(
        "Which decision model double-checks it?",
        decider_choices,
        flag="--decider",
        default=AUTO,
    )
    if decider != AUTO:
        values["decider"] = decider
    out.say(f"  decider {decider}: {decider_credentials_note(decider)}")

    tiebreak_choices = [
        Choice("none", "none", "the conservative default for unsure steps (default)"),
        Choice("haiku", "haiku", "one small Haiku call a compile, billed like the compile"),
    ]
    _check_flag("--tiebreak", flags.tiebreak, tiebreak_choices)
    tiebreak = flags.tiebreak or prompter.select(
        "Who decides the steps still unsure after that?",
        tiebreak_choices,
        flag="--tiebreak",
        default="none",
    )
    if tiebreak != "none":
        values["tiebreak"] = tiebreak

    # ── The plan ──
    target = (cwd or Path.cwd()) / "rote.yaml" if project else user_config_path()
    read = snapshot(target)
    actions: list[Action] = [
        write_file_action(
            id="write-config",
            title=f"Save {len(values)} default{'s' if len(values) != 1 else ''}",
            read=read,
            content=render_config(values),
            summary=", ".join(f"{k}: {values[k]}" for k in _ordered(values)),
            replaces=read.existed,
        )
    ]
    notes: list[str] = []
    # Saving a default needs nothing installed; `rote run` offers the
    # extra when an app first needs it.
    extra_note = missing_extra_note(values["runtime"])
    if extra_note is not None:
        notes.append(extra_note)
    if deploy == "rote-cloud":
        credential = load_credential()
        if credential is not None:
            notes.append(f"already logged in to rote cloud as {credential.user or credential.url}")
        else:
            wants_login = flags.login
            if wants_login and not prompter.interactive:
                # The device flow polls until someone approves in a browser;
                # with nobody at the terminal it would block with no one to
                # see the code. Lift this when `rote login` gets the
                # --no-wait / --complete split (docs/design/first-run.md).
                raise InitError(
                    "--login needs a person at the terminal (the device login waits for "
                    "approval in a browser); run `rote login` in a terminal, or pass "
                    "--no-login and log in later"
                )
            if wants_login is None and prompter.interactive:
                wants_login = prompter.confirm(
                    "Log in to rote cloud now?", flag="--login", default=True
                )
            if wants_login:
                actions.append(_login_action(DEFAULT_CLOUD_URL, prompter.interactive))
            else:
                notes.append("not logged in to rote cloud: run `rote login` before compiling")
    return Plan(title="Save rote defaults", actions=tuple(actions), notes=tuple(notes))


def _ordered(values: dict[str, str]) -> list[str]:
    return [k.name for k in CONFIG_KEYS if k.name in values]


def _login_action(url: str, open_browser: bool) -> Action:
    from rote.cloud_auth import login

    host = urlsplit(url).hostname or url
    effects: tuple[Contact | Launch, ...] = (
        Contact(host=host, purpose="device login, then a stored API key"),
    )
    if open_browser:
        effects += (Launch(program="your browser", purpose="approve the login"),)

    def run() -> None:
        login(open_browser=open_browser)

    return Action(
        id="login-rote-cloud",
        title="Log in to rote cloud",
        effects=effects,
        run=run,
        command="rote login" if open_browser else "rote login --device",
    )
