"""Where an onboarding command's words go: stderr for people, stdout for the result.

uv's ``Printer`` is the model: a command writes to a channel and never
branches on verbosity or format itself. Here the channels are

- :meth:`Output.say` / :meth:`Output.warn` / :meth:`Output.plan`: stderr,
  for the person (or the agent's log). Styled through rich when stderr is
  a terminal and ``NO_COLOR`` is unset.
- :meth:`Output.result`: stdout, the command's one result. Under ``--json``
  it is one JSON object (rote's ``--json`` contract); otherwise nothing,
  since a person already read the stderr lines.

Every line is printed as a rich ``Text``, never a markup string, and the
console is built with ``markup=False`` and ``highlight=False`` besides:
server names, hosts and paths come from other apps' config files, and
markup would let a name like ``[link=...]`` restyle or relink the line it
is printed in. Either layer alone stops that; both are kept so a later
``print(str)`` does not reopen it.

The console is built from the streams at call time, not import time, so
``cli._run_json_command``'s stream redirection and pytest's capture see
every write.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TextIO

from rich.console import Console
from rich.text import Text

from rote.interact.mode import Mode
from rote.interact.plan import Plan, render_plan


@dataclass
class Output:
    mode: Mode
    #: Streams, resolved per write when None (see the module docstring).
    stdout: TextIO | None = None
    stderr: TextIO | None = None

    def _err(self) -> Console:
        stream = self.stderr if self.stderr is not None else sys.stderr
        return Console(
            file=stream,
            no_color=not self.mode.color,
            force_terminal=self.mode.color,
            markup=False,
            highlight=False,
            emoji=False,
            soft_wrap=True,
        )

    def say(self, line: str = "", *, style: str | None = None) -> None:
        self._err().print(Text(line, style=style or ""))

    def warn(self, line: str) -> None:
        self._err().print(Text(f"warning: {line}", style="yellow"))

    def error(self, line: str) -> None:
        self._err().print(Text(f"error: {line}", style="red"))

    def plan(self, plan: Plan) -> None:
        console = self._err()
        lines = render_plan(plan)
        console.print(Text(lines[0], style="bold"))
        for line in lines[1:]:
            console.print(Text(line))

    def result(self, payload: Mapping[str, Any]) -> None:
        """The command's one stdout object, under ``--json`` only."""
        if not self.mode.json:
            return
        stream = self.stdout if self.stdout is not None else sys.stdout
        stream.write(json.dumps(payload, indent=2, default=str) + "\n")
        stream.flush()
