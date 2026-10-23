"""Install a workflow runtime's extra into rote's own environment, with consent.

The base install carries only the MCP client. dbos and temporal are
extras, because a user who picks temporal should not download dbos. So a
command that runs a runtime whose extra is missing (``rote run`` on a
dbos app) does not fail with a hint, and does not install anything
silently either. Commands that only write the runtime's code (compile,
emit, init) need none of it: they print the hint and go on. It builds a
:class:`~rote.interact.plan.Plan` with one Action: start the installer
:func:`rote.extras.installer` derives for this install (uv tool, pip, uv
pip, an editable checkout), which contacts the package index.

The consent model is :mod:`rote.interact`'s: a person confirms the plan;
``--yes`` applies it with nobody at the terminal; ``--json`` and an agent
caller without ``--yes`` get the ``action_required`` object and exit 2,
having installed nothing.
"""

from __future__ import annotations

import importlib
import shlex
import subprocess
import sys
from collections.abc import Sequence

from rote.extras import EXTRAS, RUNTIME_EXTRAS, installer, is_installed
from rote.interact.mode import Mode
from rote.interact.output import Output
from rote.interact.plan import (
    Action,
    ConfirmationRequired,
    Contact,
    Launch,
    Plan,
    apply_plan,
    confirm_plan,
)

__all__ = ["ensure_extra", "install_extra_action", "missing_extra_note"]

#: How long an install may take before it is reported as failed.
INSTALL_TIMEOUT_SECONDS = 900.0
#: Lines of the installer's output kept in a failure's message.
_FAILURE_TAIL_LINES = 15


def _run_installer(argv: Sequence[str]) -> None:
    """Run the installer; raise with the end of its output when it fails.

    Its output is captured, not streamed: under ``--json`` stdout is the
    command's one result object, and pip writes its progress there."""
    proc = subprocess.run(
        list(argv),
        capture_output=True,
        text=True,
        timeout=INSTALL_TIMEOUT_SECONDS,
        check=False,
    )
    if proc.returncode != 0:
        lines = (proc.stderr.strip() or proc.stdout.strip()).splitlines()
        tail = "\n".join(lines[-_FAILURE_TAIL_LINES:])
        raise RuntimeError(f"`{shlex.join(argv)}` exited {proc.returncode}:\n{tail}")


def missing_extra_note(runtime: str | None) -> str | None:
    """The note a command that only writes ``runtime``'s code (compile,
    emit, init) prints when the runtime's extra is missing, or None.

    It says ``rote run`` offers the install only where it can (not under
    uvx, or for an install from a path)."""
    extra = RUNTIME_EXTRAS.get(runtime or "")
    if extra is None or is_installed(extra):
        return None
    found = installer((extra,))
    offer = "`rote run` offers to install it, or: " if found.argv is not None else "install it: "
    lacks = f"running {runtime} apps needs the {extra} extra, which this install lacks"
    return f"{lacks}; {offer}{found.hint}"


def install_extra_action(extra: str) -> Action | None:
    """The plan step that installs ``extra``, or None when rote cannot
    install into its own environment (uvx, a non-PyPI source, no pip)."""
    found = installer((extra,))
    if found.argv is None:
        return None
    argv = found.argv
    command = shlex.join(argv)

    def run() -> str:
        _run_installer(argv)
        # The new distribution landed in a site-packages dir already on
        # sys.path; the import system only needs to forget its cached
        # directory listings.
        importlib.invalidate_caches()
        if not is_installed(extra):
            raise RuntimeError(
                f"`{command}` succeeded, but the {extra} extra still is not importable "
                f"by {sys.executable}"
            )
        return extra

    return Action(
        id=f"install-{extra}",
        title=f"Install rote-cli's {extra} extra ({EXTRAS[extra].used_for})",
        effects=(
            Contact(
                host="pypi.org",
                purpose=f"resolve the {extra} extra's packages (or the index your "
                "pip or uv config names)",
            ),
            Contact(host="files.pythonhosted.org", purpose="download those packages"),
            Launch(
                program=command,
                purpose=f"install them into this rote's environment ({sys.prefix})",
            ),
        ),
        run=run,
        command=command,
    )


def ensure_extra(
    extra: str,
    purpose: str,
    *,
    mode: Mode,
    out: Output,
    argv: Sequence[str],
) -> int | None:
    """Make sure ``extra`` is installed before ``purpose`` goes on.

    For a command that runs the runtime (``rote run``). A command that only
    writes its code (compile, emit, init) never calls this: it prints the
    hint and goes on.

    Returns None to go on (installed already, or installed now), else the
    exit code the command returns: 2 when the plan needs a flag nobody gave (the
    ``action_required`` object is printed under ``--json``) or rote cannot
    install here, 1 when the person declined or the install failed, 130
    when they cancelled.

    ``purpose`` finishes "<purpose> needs the <extra> extra", e.g.
    ``"running a dbos app"``.
    """
    from rote.interact.prompter import PromptCancelled, prompter_for
    from rote.interact.report import EXIT_ACTION_REQUIRED, action_required

    if is_installed(extra):
        return None
    action = install_extra_action(extra)
    if action is None:
        found = installer((extra,))
        out.error(f"{purpose} needs the {extra} extra ({found.why_not}): {found.hint}")
        return 2
    plan = Plan(title=f"Install the {extra} extra: {purpose} needs it", actions=(action,))
    out.plan(plan)
    try:
        approved = confirm_plan(plan, prompter_for(mode), yes=mode.yes)
    except ConfirmationRequired as e:
        out.error(str(e))
        out.result(action_required(e, argv))
        return EXIT_ACTION_REQUIRED
    except PromptCancelled:
        out.say("cancelled; nothing was installed")
        return 130
    if not approved:
        hint = installer((extra,)).hint
        out.say(f"nothing was installed; {purpose} needs the {extra} extra: {hint}")
        return 1
    out.say(f"installing: {action.command}")
    outcomes = apply_plan(plan)
    failed = [o for o in outcomes if o.status == "failed"]
    if failed:
        out.error(f"installing the {extra} extra failed: {failed[0].error}")
        return 1
    out.say(f"installed the {extra} extra")
    return None
