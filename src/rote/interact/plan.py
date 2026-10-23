"""Every side effect is an Action in a Plan, shown and confirmed before it runs.

An onboarding command never writes a file, contacts a server, opens a
browser or spends money inline. It builds a :class:`Plan`: a list of
:class:`Action` objects, each carrying the :data:`Effect` records that say
what it will do, and the exact command a script or an agent could run to
do the same thing. The plan is then

1. rendered (:func:`render_plan` for people, :meth:`Plan.to_dict` for
   ``--json``; one model, two renderings),
2. confirmed (:func:`confirm_plan`: a prompt for a person, ``--yes`` plus
   any stronger flag an effect names for everyone else),
3. applied in order (:func:`apply_plan`), stopping at the first failure.

This is Terraform's plan/apply split at the size of a CLI wizard. The point
is that the confirmation is about the real effects, not a description of
them written somewhere else: the code that applies an action is attached to
the record that previews it.

Two effects need more than ``--yes`` when nobody is at the terminal:
replacing an existing file needs ``--force``, and spending money needs the
flag the :class:`Spend` names (``--live``, say). ``--yes`` accepts the
defaults, and a default never replaces a file or spends money.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from rote.interact.prompter import Prompter

#: The flag that lets a non-interactive run replace a file that exists.
FORCE_FLAG = "--force"
#: The flag that approves a plan when nobody is at the terminal.
YES_FLAG = "--yes"


@dataclass(frozen=True)
class FileWrite:
    """A file this action creates or replaces."""

    path: Path
    #: What changes, in a few words ("add 3 servers", "set runtime: dbos").
    summary: str
    #: Whether the file existed when the plan was made.
    existed: bool = False
    #: True when existing content will be lost. A merge that keeps
    #: everything already there is an update, not a replacement.
    replaces: bool = False
    #: The permission bits the file is written with.
    file_mode: int = 0o644

    kind: Literal["write"] = "write"

    def __post_init__(self) -> None:
        if self.replaces and not self.existed:
            raise ValueError(f"{self.path}: cannot replace a file that does not exist")

    @property
    def needs(self) -> str | None:
        return FORCE_FLAG if self.replaces else None

    def describe(self) -> str:
        verb = "replace" if self.replaces else ("update" if self.existed else "create")
        return f"{verb} {self.path} ({self.summary})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "path": str(self.path),
            "summary": self.summary,
            "existed": self.existed,
            "replaces": self.replaces,
        }


@dataclass(frozen=True)
class Contact:
    """A host this action sends a request to.

    ``host`` is a host name and nothing more: never a URL, whose path or
    query can carry a credential."""

    host: str
    purpose: str

    kind: Literal["contact"] = "contact"

    def __post_init__(self) -> None:
        if not self.host or any(c in self.host for c in "/?#@ "):
            raise ValueError(f"Contact.host must be a bare host name, got {self.host!r}")

    @property
    def needs(self) -> str | None:
        return None

    def describe(self) -> str:
        return f"contact {self.host} ({self.purpose})"

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "host": self.host, "purpose": self.purpose}


@dataclass(frozen=True)
class Launch:
    """A program this action starts on the user's machine (a browser, ``claude -p``)."""

    program: str
    purpose: str

    kind: Literal["launch"] = "launch"

    @property
    def needs(self) -> str | None:
        return None

    def describe(self) -> str:
        return f"start {self.program} ({self.purpose})"

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "program": self.program, "purpose": self.purpose}


@dataclass(frozen=True)
class Spend:
    """Money this action spends, as an estimated range, and who pays."""

    low_usd: float
    high_usd: float
    #: Who is billed ("your Claude subscription", "your Anthropic API key").
    payer: str
    #: The flag that opts into this spend when nobody is at the terminal.
    #: ``--yes`` alone never spends money.
    flag: str

    kind: Literal["spend"] = "spend"

    def __post_init__(self) -> None:
        if self.low_usd < 0 or self.high_usd < self.low_usd:
            raise ValueError(
                f"Spend range must be 0 <= low <= high, got {self.low_usd}..{self.high_usd}"
            )
        if not self.flag.startswith("--"):
            raise ValueError(f"Spend.flag must be a long flag, got {self.flag!r}")

    @property
    def needs(self) -> str | None:
        return self.flag

    def describe(self) -> str:
        if self.low_usd == self.high_usd:
            amount = f"about ${self.low_usd:.2f}"
        else:
            amount = f"${self.low_usd:.2f} to ${self.high_usd:.2f}"
        return f"spend {amount}, billed to {self.payer}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "low_usd": self.low_usd,
            "high_usd": self.high_usd,
            "payer": self.payer,
            "flag": self.flag,
        }


Effect = FileWrite | Contact | Launch | Spend


@dataclass(frozen=True)
class Action:
    """One step of a plan: what it does, how a script would do it, and the code that does it."""

    #: Stable id for ``--json`` consumers ("write-config", "login-figma").
    id: str
    title: str
    effects: tuple[Effect, ...]
    #: Runs the action. Raises on failure; the error is the report.
    run: Callable[[], None] = field(repr=False, compare=False)
    #: The exact command that does the same thing, for an agent or a
    #: script to run later. None when no single command does it.
    command: str | None = None

    def __post_init__(self) -> None:
        if not self.effects:
            raise ValueError(
                f"action {self.id!r} has no effects: an action with nothing to "
                "preview is not a side effect and does not belong in a plan"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "command": self.command,
            "effects": [e.to_dict() for e in self.effects],
        }


@dataclass(frozen=True)
class Plan:
    """The actions a command will take, in order, plus what it decided not to do."""

    title: str
    actions: tuple[Action, ...] = ()
    #: Things the user should know that are not actions ("figma: already
    #: registered, skipped").
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        ids = [a.id for a in self.actions]
        if len(ids) != len(set(ids)):
            raise ValueError(f"plan {self.title!r} repeats an action id: {ids}")

    @property
    def effects(self) -> tuple[Effect, ...]:
        return tuple(e for a in self.actions for e in a.effects)

    def needed_flags(self) -> tuple[str, ...]:
        """Flags beyond ``--yes`` this plan needs when nobody is at the terminal."""
        seen: dict[str, None] = {}
        for effect in self.effects:
            if effect.needs is not None:
                seen.setdefault(effect.needs, None)
        return tuple(seen)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "actions": [a.to_dict() for a in self.actions],
            "notes": list(self.notes),
            "needs": [YES_FLAG, *self.needed_flags()] if self.actions else [],
        }


def render_plan(plan: Plan) -> list[str]:
    """The plan as lines for a person to read, every effect spelled out."""
    lines = [f"Plan: {plan.title}"]
    if not plan.actions:
        lines.append("  nothing to do")
    for index, action in enumerate(plan.actions, start=1):
        lines.append(f"  {index}. {action.title}")
        for effect in action.effects:
            lines.append(f"       - {effect.describe()}")
    for note in plan.notes:
        lines.append(f"  note: {note}")
    return lines


class ConfirmationRequired(RuntimeError):
    """A non-interactive run reached a plan it may not apply without more flags."""

    def __init__(self, plan: Plan, missing: Sequence[str]) -> None:
        self.plan = plan
        self.missing = tuple(missing)
        super().__init__(
            f"{plan.title}: not applied. Pass {' '.join(self.missing)} to apply it "
            "(nothing was written, contacted or spent)"
        )


def confirm_plan(
    plan: Plan,
    prompter: Prompter,
    *,
    yes: bool,
    granted: Collection[str] = (),
) -> bool:
    """Whether ``plan`` may be applied.

    A person at the terminal is asked, and the default answer is no when
    the plan replaces a file or spends money. Otherwise ``--yes`` is
    required, plus every flag an effect needs (``--force``, a spend's
    flag), which the caller passes in ``granted`` when the user gave it.
    Raises :class:`ConfirmationRequired` naming what is missing; returns
    False only when a person declined.
    """
    if not plan.actions:
        return True
    needed = plan.needed_flags()
    if prompter.interactive:
        return prompter.confirm(
            "Apply this plan?",
            flag=YES_FLAG,
            default=not needed,
        )
    missing = [] if yes else [YES_FLAG]
    missing += [flag for flag in needed if flag not in granted]
    if missing:
        raise ConfirmationRequired(plan, missing)
    return True


@dataclass(frozen=True)
class Outcome:
    """What happened to one action when the plan was applied."""

    action_id: str
    status: Literal["done", "failed", "not-run"]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.action_id, "status": self.status, "error": self.error}


def apply_plan(plan: Plan) -> list[Outcome]:
    """Run each action in order. The first failure stops the rest, which
    are reported as not run: a later action may depend on an earlier one
    (log in to a server only after it is registered)."""
    outcomes: list[Outcome] = []
    failed = False
    for action in plan.actions:
        if failed:
            outcomes.append(Outcome(action.id, "not-run"))
            continue
        try:
            action.run()
        except Exception as e:  # the report carries it; the caller decides the exit code
            outcomes.append(Outcome(action.id, "failed", f"{type(e).__name__}: {e}"))
            failed = True
        else:
            outcomes.append(Outcome(action.id, "done"))
    return outcomes


def atomic_write_text(path: Path, text: str, *, file_mode: int = 0o644) -> None:
    """Write ``text`` to ``path`` all at once: a crash leaves the old file or the new one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.chmod(tmp_name, file_mode)
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def write_file_action(
    *,
    id: str,
    title: str,
    path: Path,
    content: str,
    summary: str,
    replaces: bool,
    file_mode: int = 0o644,
    command: str | None = None,
) -> Action:
    """An action that writes ``content`` to ``path`` atomically."""
    return Action(
        id=id,
        title=title,
        effects=(
            FileWrite(
                path=path,
                summary=summary,
                existed=path.exists(),
                replaces=replaces,
                file_mode=file_mode,
            ),
        ),
        run=lambda: atomic_write_text(path, content, file_mode=file_mode),
        command=command,
    )
