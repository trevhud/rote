"""What a non-interactive run prints when it needs something it was not given.

Vercel's ``action_required`` payload is the model
(``packages/cli/src/util/agent-output.ts``): a machine-readable reason, the
missing flags, and the exact command to run next, so an agent relays one
question to its user and then runs one command. rote's ``--json`` contract
(one object on stdout, ``ok`` first) still holds; this is the object for
the two stops a non-interactive onboarding run can reach:

- ``missing_answer``: a question with no flag and no default to take
  (:class:`~rote.interact.prompter.PromptRequired`).
- ``confirmation_required``: a plan that needs ``--yes`` or a stronger flag
  (:class:`~rote.interact.plan.ConfirmationRequired`). The plan is included
  in full, so the agent can show the user what would happen.

The exit code for both is 2, a usage error (gh's ``FlagError``): nothing
ran, and the same command with the named flags would.
"""

from __future__ import annotations

import shlex
from collections.abc import Sequence
from typing import Any, Literal

from rote.interact.plan import ConfirmationRequired
from rote.interact.prompter import PromptRequired

#: The exit code a run that stopped for a missing answer or consent returns.
EXIT_ACTION_REQUIRED = 2

Reason = Literal["missing_answer", "confirmation_required"]


def next_command(argv: Sequence[str], extra: Sequence[str]) -> str:
    """``rote <argv> <extra>``, shell-quoted except ``<placeholder>`` words."""
    words = ["rote", *argv, *extra]
    return " ".join(w if w.startswith("<") else shlex.quote(w) for w in words)


def action_required(
    error: PromptRequired | ConfirmationRequired, argv: Sequence[str]
) -> dict[str, Any]:
    """The ``--json`` object for a run that stopped for ``error``."""
    reason: Reason
    if isinstance(error, PromptRequired):
        reason = "missing_answer"
        placeholder = f"<{'|'.join(error.choices)}>" if error.choices else "<value>"
        missing = [error.flag]
        command = next_command(argv, [error.flag, placeholder])
        extra: dict[str, Any] = {"question": error.question, "choices": list(error.choices)}
    else:
        reason = "confirmation_required"
        missing = list(error.missing)
        command = next_command(argv, missing)
        extra = {"plan": error.plan.to_dict()}
    return {
        "ok": False,
        "status": "action_required",
        "reason": reason,
        "message": str(error),
        "missing": missing,
        "next": [{"command": command}],
        **extra,
    }
