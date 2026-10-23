"""The run's mode, decided once, before anything is asked or printed.

gh decides ``CanPrompt()`` in one place (``pkg/iostreams``) and every
command asks it; Vercel's ``client.nonInteractive`` defaults to "an agent
is calling". :class:`Mode` is both, frozen: the terminal facts, the agent
in the environment, and the flags that override them. A command resolves
it at entry (:meth:`Mode.resolve`) and hands it to
:func:`rote.interact.prompter.prompter_for`; nothing below that reads
``isatty`` or ``os.environ`` for this again.

A person is asked only when all of these hold:

- stdin and stderr are terminals (prompts read stdin and draw on stderr,
  so ``rote setup --json | jq`` can be piped and still not prompt: --json
  is below);
- no agent is the caller, unless ``--agent-mode no`` says a person is here;
- not ``--json`` (one JSON object on stdout, nothing to answer);
- not ``--yes`` (take the defaults, approve the plan);
- ``ROTE_PROMPT_DISABLED`` is not set (gh's ``GH_PROMPT_DISABLED``).

The flag is ``--agent-mode``, not the design's ``--agent``: ``--agent``
already names the compiler driver on compile, analyze and init.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, TextIO

from rote.interact.agents import Detected, detect

AgentMode = Literal["auto", "yes", "no"]
AGENT_MODES: tuple[AgentMode, ...] = ("auto", "yes", "no")

#: Set to anything non-empty to turn every prompt off.
PROMPT_DISABLED_ENV = "ROTE_PROMPT_DISABLED"
#: ``--agent-mode``'s environment twin.
AGENT_MODE_ENV = "ROTE_AGENT_MODE"


def _isatty(stream: TextIO | None) -> bool:
    if stream is None:
        return False
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError):  # closed, or not a real stream
        return False


@dataclass(frozen=True)
class Mode:
    stdin_tty: bool
    stdout_tty: bool
    stderr_tty: bool
    #: The AI client in the environment, whatever ``--agent-mode`` says.
    detected: Detected | None
    agent_mode: AgentMode
    json: bool
    yes: bool
    prompt_disabled: bool
    #: Color on stderr: a terminal, and ``NO_COLOR`` unset.
    color: bool

    @classmethod
    def resolve(
        cls,
        *,
        json: bool = False,
        yes: bool = False,
        agent_mode: AgentMode | None = None,
        env: Mapping[str, str] | None = None,
        stdin: TextIO | None = None,
        stdout: TextIO | None = None,
        stderr: TextIO | None = None,
    ) -> Mode:
        """Read the terminal and the environment once. Every argument is
        injectable; the defaults are this process's own."""
        env = os.environ if env is None else env
        if agent_mode is None:
            raw = env.get(AGENT_MODE_ENV, "auto").strip().lower() or "auto"
            if raw not in AGENT_MODES:
                choices = ", ".join(AGENT_MODES)
                raise ValueError(
                    f"{AGENT_MODE_ENV}={raw!r} is not valid; expected one of {choices}"
                )
            agent_mode = raw
        assert agent_mode is not None
        stderr_tty = _isatty(stderr if stderr is not None else sys.stderr)
        return cls(
            stdin_tty=_isatty(stdin if stdin is not None else sys.stdin),
            stdout_tty=_isatty(stdout if stdout is not None else sys.stdout),
            stderr_tty=stderr_tty,
            detected=detect(env),
            agent_mode=agent_mode,
            json=json,
            yes=yes,
            prompt_disabled=bool(env.get(PROMPT_DISABLED_ENV, "").strip()),
            color=stderr_tty and not env.get("NO_COLOR"),
        )

    @property
    def agent(self) -> bool:
        """An agent is the caller: by flag, or detected under ``auto``."""
        if self.agent_mode == "yes":
            return True
        if self.agent_mode == "no":
            return False
        return self.detected is not None and self.detected.role == "caller"

    @property
    def why_not_interactive(self) -> str | None:
        """The first reason nobody can be asked, or None when a person can be."""
        if self.json:
            return "--json was passed"
        if self.yes:
            return "--yes was passed"
        if self.prompt_disabled:
            return f"{PROMPT_DISABLED_ENV} is set"
        if self.agent:
            if self.agent_mode == "yes":
                return "--agent-mode yes was passed"
            assert self.detected is not None
            return f"running under {self.detected.label}; --agent-mode no if a person is here"
        if not self.stdin_tty:
            return "stdin is not a terminal"
        if not self.stderr_tty:
            return "stderr is not a terminal"
        return None

    @property
    def interactive(self) -> bool:
        return self.why_not_interactive is None

    def to_dict(self) -> dict[str, object]:
        return {
            "interactive": self.interactive,
            "why_not_interactive": self.why_not_interactive,
            "agent": self.agent,
            "detected_client": None if self.detected is None else self.detected.client,
            "detected_role": None if self.detected is None else self.detected.role,
            "detected_by": None if self.detected is None else self.detected.var,
        }
