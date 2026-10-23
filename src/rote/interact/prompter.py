"""Asking the user a question, behind one interface with three implementations.

Commands ask through a :class:`Prompter` and never call ``input()`` or a
prompt library directly. Which implementation they get is decided once,
from the :class:`~rote.interact.mode.Mode`, before the first question
(:func:`prompter_for`):

- :class:`QuestionaryPrompter` for a person at a terminal. Prompts are
  drawn on stderr, so stdout stays the command's result.
- :class:`NonInteractivePrompter` for agents, CI, ``--json`` and pipes. It
  never reads stdin. With ``--yes`` it takes each question's default, and
  a question without a default raises :class:`PromptRequired` naming the
  flag that answers it. Without ``--yes`` every question raises.
- :class:`ScriptedPrompter` for tests: canned answers, each checked
  against the question it answers.

Every question names its flag. That is the gh rule: a wizard and a script
are one code path, and the error a script hits tells it what to pass.
The command checks its flag first and asks only when the flag is absent::

    runtime = args.runtime or prompter.select(
        "Runtime?", RUNTIMES, flag="--runtime", default="dbos"
    )

Deciding interactivity up front is not a nicety. prompt_toolkit reads a
piped stdin as keystrokes, so a questionary prompt fed by a pipe returns
whatever the bytes select, or the default, with only a warning on stderr.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, Literal, Protocol, TypeVar

if TYPE_CHECKING:
    from rote.interact.mode import Mode

T = TypeVar("T")

Kind = Literal["select", "checkbox", "confirm", "text"]


@dataclass(frozen=True)
class Choice(Generic[T]):
    """One option of a select or checkbox question."""

    value: T
    label: str
    #: Shown beside the label (a host, a scope, "recommended").
    hint: str = ""
    #: Why the option cannot be picked. Disabled options are shown, so the
    #: user sees what was found and why it is not offered.
    disabled: str | None = None


class PromptRequired(RuntimeError):
    """A question was reached with nobody to answer it and no default to take."""

    def __init__(
        self,
        message: str,
        flag: str,
        why: str,
        choices: Sequence[str] = (),
        *,
        takes_value: bool = True,
    ) -> None:
        self.question = message
        self.flag = flag
        self.why = why
        #: The values the flag takes, when the question was a choice.
        self.choices = tuple(choices)
        #: False for a yes/no question: its flag is a switch (``--login``),
        #: so the command to run next passes it bare, with no ``<value>``.
        self.takes_value = takes_value
        values = f" <{'|'.join(self.choices)}>" if self.choices else ""
        super().__init__(f"{message.rstrip('?: ')}: pass {flag}{values} ({why})")


class PromptCancelled(RuntimeError):
    """The person cancelled a question (Ctrl-C or Ctrl-D)."""


class Prompter(Protocol):
    """How a command asks. Every method names the flag that answers it."""

    @property
    def interactive(self) -> bool:
        """True when a person answers. Commands use it to skip what only a
        person needs (opening a browser), never to pick a different flow."""
        ...

    def select(
        self, message: str, choices: Sequence[Choice[T]], *, flag: str, default: T | None = None
    ) -> T: ...

    def checkbox(
        self,
        message: str,
        choices: Sequence[Choice[T]],
        *,
        flag: str,
        default: Sequence[T] = (),
    ) -> list[T]: ...

    def confirm(self, message: str, *, flag: str, default: bool) -> bool: ...

    def text(
        self,
        message: str,
        *,
        flag: str,
        default: str | None = None,
        validate: Callable[[str], str | None] | None = None,
    ) -> str: ...


def _check_choices(message: str, choices: Sequence[Choice[Any]], default: Any) -> None:
    if not choices:
        raise ValueError(f"{message!r}: a question needs at least one choice")
    if default is not None:
        match = [c for c in choices if c.value == default]
        if not match:
            raise ValueError(f"{message!r}: default {default!r} is not one of the choices")
        if match[0].disabled is not None:
            raise ValueError(f"{message!r}: default {default!r} is disabled ({match[0].disabled})")


def _check_checkbox_default(
    message: str, choices: Sequence[Choice[Any]], default: Sequence[Any]
) -> None:
    if not choices:
        raise ValueError(f"{message!r}: a question needs at least one choice")
    enabled = {repr(c.value) for c in choices if c.disabled is None}
    for value in default:
        if repr(value) not in enabled:
            raise ValueError(f"{message!r}: default {value!r} is not an enabled choice")


@dataclass
class NonInteractivePrompter:
    """Answers from defaults under ``--yes``; otherwise names the flag to pass."""

    #: Why nobody can be asked ("stdin is not a terminal", "running under Claude Code").
    why: str
    #: ``--yes``: take each question's default.
    accept_defaults: bool = False

    @property
    def interactive(self) -> bool:
        return False

    def _default_or_raise(
        self,
        message: str,
        flag: str,
        default: Any,
        choices: Sequence[Choice[Any]] = (),
        *,
        takes_value: bool = True,
    ) -> Any:
        if self.accept_defaults and default is not None:
            return default
        if self.accept_defaults:
            why = f"{self.why}; this question has no default"
        elif default is not None:
            why = f"{self.why}; or --yes to take the default"
        else:
            why = self.why
        values = [str(c.value) for c in choices if c.disabled is None]
        raise PromptRequired(message, flag, why, values, takes_value=takes_value)

    def select(
        self, message: str, choices: Sequence[Choice[T]], *, flag: str, default: T | None = None
    ) -> T:
        _check_choices(message, choices, default)
        value: T = self._default_or_raise(message, flag, default, choices)
        return value

    def checkbox(
        self,
        message: str,
        choices: Sequence[Choice[T]],
        *,
        flag: str,
        default: Sequence[T] = (),
    ) -> list[T]:
        _check_checkbox_default(message, choices, default)
        # An empty default is a real default ("import nothing"), so --yes takes it.
        value: Sequence[T] = self._default_or_raise(message, flag, default, choices)
        return list(value)

    def confirm(self, message: str, *, flag: str, default: bool) -> bool:
        value: bool = self._default_or_raise(message, flag, default, takes_value=False)
        return value

    def text(
        self,
        message: str,
        *,
        flag: str,
        default: str | None = None,
        validate: Callable[[str], str | None] | None = None,
    ) -> str:
        value: str = self._default_or_raise(message, flag, default)
        if validate is not None:
            problem = validate(value)
            if problem is not None:
                raise ValueError(f"{message!r}: default {value!r} fails its own check: {problem}")
        return value


@dataclass(frozen=True)
class Answer:
    """One scripted answer: the question kind and flag it must match, and the value."""

    kind: Kind
    flag: str
    value: Any


@dataclass
class ScriptedPrompter:
    """Canned answers for tests. Each answer names the question it answers
    (its kind and flag), so a test fails where the flow asked something
    else, not three questions later. Use :data:`DEFAULT` to take the
    question's default.

    ``asked`` records ``(kind, flag, message)`` for every question."""

    answers: list[Answer]
    interactive_: bool = True
    asked: list[tuple[Kind, str, str]] = field(default_factory=list)

    @property
    def interactive(self) -> bool:
        return self.interactive_

    def _next(self, kind: Kind, message: str, flag: str, default: Any) -> Any:
        self.asked.append((kind, flag, message))
        if not self.answers:
            raise AssertionError(f"unscripted {kind} question {message!r} ({flag})")
        answer = self.answers.pop(0)
        if (answer.kind, answer.flag) != (kind, flag):
            raise AssertionError(
                f"scripted answer for {answer.kind} {answer.flag} reached {kind} "
                f"question {message!r} ({flag})"
            )
        if answer.value is DEFAULT:
            if default is None:
                raise AssertionError(f"{message!r} ({flag}) has no default to take")
            return default
        if isinstance(answer.value, BaseException):
            raise answer.value
        return answer.value

    def select(
        self, message: str, choices: Sequence[Choice[T]], *, flag: str, default: T | None = None
    ) -> T:
        _check_choices(message, choices, default)
        value: T = self._next("select", message, flag, default)
        enabled = [c.value for c in choices if c.disabled is None]
        if value not in enabled:
            raise AssertionError(f"{message!r}: {value!r} is not an enabled choice")
        return value

    def checkbox(
        self,
        message: str,
        choices: Sequence[Choice[T]],
        *,
        flag: str,
        default: Sequence[T] = (),
    ) -> list[T]:
        _check_checkbox_default(message, choices, default)
        value: Sequence[T] = self._next("checkbox", message, flag, list(default))
        enabled = [c.value for c in choices if c.disabled is None]
        for item in value:
            if item not in enabled:
                raise AssertionError(f"{message!r}: {item!r} is not an enabled choice")
        return list(value)

    def confirm(self, message: str, *, flag: str, default: bool) -> bool:
        value: bool = self._next("confirm", message, flag, default)
        return value

    def text(
        self,
        message: str,
        *,
        flag: str,
        default: str | None = None,
        validate: Callable[[str], str | None] | None = None,
    ) -> str:
        value: str = self._next("text", message, flag, default)
        if validate is not None and validate(value) is not None:
            raise AssertionError(f"{message!r}: scripted {value!r} fails validation")
        return value


class _Default:
    def __repr__(self) -> str:
        return "DEFAULT"


#: A scripted answer that takes the question's default.
DEFAULT: Any = _Default()


class QuestionaryPrompter:
    """A person at a terminal, through questionary, drawn on stderr.

    Construct it only through :func:`prompter_for`, which has already
    checked that stdin and stderr are terminals. ``input`` and ``output``
    are prompt_toolkit objects, injected by tests (``create_pipe_input``
    and ``DummyOutput``); by default the prompts read the terminal and
    draw on stderr."""

    def __init__(self, *, input: Any = None, output: Any = None) -> None:
        from prompt_toolkit.output import create_output

        self._input = input
        # questionary draws on stdout by default, and stdout is the
        # command's result (JSON under --json, a file under redirection).
        self._output = output if output is not None else create_output(stdout=sys.stderr)

    @property
    def interactive(self) -> bool:
        return True

    def _ask(self, question: Any) -> Any:
        # unsafe_ask, not ask: ask() prints "Cancelled by user" to stdout
        # and returns None, which a caller cannot tell from an answer.
        try:
            return question.unsafe_ask()
        except (KeyboardInterrupt, EOFError) as e:
            raise PromptCancelled("cancelled") from e

    def _io(self) -> dict[str, Any]:
        return {"input": self._input, "output": self._output}

    @staticmethod
    def _choices(choices: Sequence[Choice[Any]]) -> list[Any]:
        import questionary

        return [
            questionary.Choice(
                title=f"{c.label}  {c.hint}" if c.hint else c.label,
                value=index,
                disabled=c.disabled,
            )
            for index, c in enumerate(choices)
        ]

    def select(
        self, message: str, choices: Sequence[Choice[T]], *, flag: str, default: T | None = None
    ) -> T:
        import questionary

        _check_choices(message, choices, default)
        options = self._choices(choices)
        start = next((o for o, c in zip(options, choices, strict=True) if c.value == default), None)
        index: int = self._ask(
            questionary.select(message, choices=options, default=start, **self._io())
        )
        return choices[index].value

    def checkbox(
        self,
        message: str,
        choices: Sequence[Choice[T]],
        *,
        flag: str,
        default: Sequence[T] = (),
    ) -> list[T]:
        import questionary

        _check_checkbox_default(message, choices, default)
        options = self._choices(choices)
        for option, choice in zip(options, choices, strict=True):
            option.checked = choice.value in default
        picked = self._ask(questionary.checkbox(message, choices=options, **self._io()))
        return [choices[i].value for i in picked]

    def confirm(self, message: str, *, flag: str, default: bool) -> bool:
        import questionary

        return bool(self._ask(questionary.confirm(message, default=default, **self._io())))

    def text(
        self,
        message: str,
        *,
        flag: str,
        default: str | None = None,
        validate: Callable[[str], str | None] | None = None,
    ) -> str:
        import questionary

        def check(value: str) -> bool | str:
            if validate is None:
                return True
            problem = validate(value)
            return True if problem is None else problem

        return str(
            self._ask(
                questionary.text(message, default=default or "", validate=check, **self._io())
            )
        )


def prompter_for(mode: Mode) -> Prompter:
    """The one place that turns a :class:`Mode` into a prompter."""
    if mode.interactive:
        return QuestionaryPrompter()
    why = mode.why_not_interactive
    assert why is not None
    return NonInteractivePrompter(why=why, accept_defaults=mode.yes)
