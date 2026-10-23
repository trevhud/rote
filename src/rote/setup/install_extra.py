"""Install a workflow runtime's extra into rote's own environment, with consent.

The base install carries only the MCP client. dbos and temporal are
extras, because a user who picks temporal should not download dbos. So a
command that needs a runtime's extra (``rote compile --runtime dbos``,
``rote init`` picking temporal, ``rote run`` on a dbos app) does not fail
with a hint, and does not install anything silently either. It builds a
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

from rote.extras import EXTRAS, installer, is_installed
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

__all__ = ["ensure_extra", "install_extra_action"]

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
    optional: bool = False,
) -> int | None:
    """Make sure ``extra`` is installed before ``purpose`` goes on.

    Returns None to go on (installed already, or installed now, or the
    person declined an ``optional`` install), else the exit code the
    command returns: 2 when the plan needs a flag nobody gave (the
    ``action_required`` object is printed under ``--json``) or rote cannot
    install here, 1 when the person declined or the install failed, 130
    when they cancelled.

    ``purpose`` finishes "<purpose> needs the <extra> extra", e.g.
    ``"running a dbos app"``. ``optional``: the command can do its work
    without the extra (a compile emits dbos code without dbos installed),
    so declining goes on with a note instead of stopping.
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
        if optional:
            out.say(f"not installed; going on. Running the result needs it: {hint}")
            return None
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
