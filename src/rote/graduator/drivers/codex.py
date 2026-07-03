"""Codex CLI driver — spawns ``codex exec`` for ChatGPT subscription users.

This is the ChatGPT Plus/Pro subscription path. It shells out to the
``codex`` CLI in its non-interactive ``exec`` subcommand.

# Auth

``codex exec`` reuses the cached login from ``~/.codex/auth.json``.
The user runs ``codex login`` once interactively with their ChatGPT
account; tokens auto-refresh. No env scrubbing is needed — unlike
Claude Code, Codex defaults to the cached OAuth session as long as
the user hasn't overridden it with ``OPENAI_API_KEY``.

# Gotchas

* Codex insists the workspace directory be a git repo by default;
  rote's scratch work dir is not, so we always pass
  ``--skip-git-repo-check``.
* An active ChatGPT login *blocks* ``OPENAI_API_KEY``. rote does not
  try to support both auth modes through this driver; if the user
  wants API-key auth, they should use the ``api`` driver (which is
  Anthropic, not OpenAI).
* Progress goes to stderr; final agent message goes to stdout. With
  ``--json``, stdout is NDJSON event stream instead.
"""

from __future__ import annotations

from pathlib import Path
from shutil import which

from rote.graduator.drivers import DriverResult, GraduatorDriver


class CodexDriver(GraduatorDriver):
    name: str = "codex"

    def is_available(self) -> tuple[bool, str]:
        """Check whether ``codex`` CLI is installed on PATH.

        Auth is not checked here — that surfaces at ``run()`` time.
        """
        if which("codex") is None:
            return (
                False,
                "The `codex` CLI is not installed. "
                "Install from https://developers.openai.com/codex/ "
                "and run `codex login` once to enable subscription auth.",
            )
        return (True, "")

    async def run(
        self,
        skill_dir: Path,
        graduator_skill_dir: Path,
        work_dir: Path,
    ) -> DriverResult:
        """Spawn ``codex exec`` and wait for it to produce a pipeline.yaml.

        Not yet implemented. Task #13.

        Planned invocation (per ``docs/agent-runtime.md``)::

            codex exec \\
                --cd "<work_dir>" \\
                --add-dir "<skill_dir>" \\
                --add-dir "<graduator_skill_dir>" \\
                --sandbox workspace-write \\
                --ask-for-approval never \\
                --skip-git-repo-check \\
                --json \\
                "<prompt with inlined graduator SKILL.md>"

        Notes:

        * No ``--append-system-prompt`` flag exists for ``codex exec``;
          the graduator's SKILL.md contents are inlined into the prompt
          argument directly.
        * ``--sandbox workspace-write`` confines writes to ``work_dir``
          + ``/tmp`` and disables network access (consistent with our
          read-the-skill / write-pipeline.yaml contract).
        """
        raise NotImplementedError("CodexDriver.run: to be implemented in task #13")
