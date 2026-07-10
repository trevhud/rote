"""Codex CLI driver — spawns ``codex exec`` in non-interactive mode.

This is the path for users of OpenAI's Codex CLI. It shells out to
``codex exec`` (the non-interactive entry point) and requires only that
the user has authenticated once — either ``codex login`` with a ChatGPT
Plus/Pro account, or an ``OPENAI_API_KEY`` that Codex is configured to
use.

# Sandbox model

``codex exec`` defaults to a **read-only** sandbox, which cannot write
the pipeline.yaml — so we explicitly select ``--sandbox workspace-write``:

* **Reads** are full-access across the filesystem in this mode, so the
  agent can read the source skill and the rote-graduate rubric wherever
  they live — no grant needed.
* **Writes** are confined to the working directory (``--cd <work_dir>``)
  plus the system temp dir — exactly the "read the skill, write the
  pipeline.yaml into work_dir" contract every driver honors.
* **Network** is denied for model-generated shell commands. Codex's own
  model API calls are not subject to this (they're the CLI's connection,
  not a sandboxed command), so the graduator loop still works.

We deliberately do **not** pass ``--add-dir`` for the skill/rubric dirs:
that flag appends to the sandbox's *writable* roots, and the source skill
must stay read-only. Reads already work under ``workspace-write``.

``codex exec`` is headless: its approval policy defaults to *never ask*,
and ``--ask-for-approval`` is not even a valid flag there (it hard-errors
on the interactive command only). So there is nothing to configure — we
just don't pass any approval flag.

# Auth: no env scrubbing (contrast with ClaudeDriver)

``ClaudeDriver`` scrubs ``ANTHROPIC_API_KEY`` to force subscription auth,
because rote ships a separate ``api`` driver for API-key users. There is
no OpenAI-API driver — so a user whose only credential is
``OPENAI_API_KEY`` must still be able to graduate through Codex. This
driver therefore passes the environment through untouched and defers to
Codex's own auth precedence.

Note the asymmetry with Claude: for Codex, a stored ``codex login``
session is only overridden by ``CODEX_API_KEY`` / ``CODEX_ACCESS_TOKEN``
— *not* by ``OPENAI_API_KEY`` (which only matters as a model-provider key
at request time). So scrubbing ``OPENAI_API_KEY`` would be pointless, and
scrubbing ``CODEX_API_KEY`` would break users who intend to use it.

See ``docs/agent-runtime.md`` for the full design record.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import tempfile
from pathlib import Path
from shutil import which
from typing import Any

from rote.graduator.drivers import DriverError, DriverResult, GraduatorDriver
from rote.graduator.events import EventCallback, ProgressFileWatcher

# ───────── Defaults ─────────

DEFAULT_SANDBOX = "workspace-write"
"""Sandbox policy for the graduator run.

``workspace-write`` allows global reads (so the agent can read the source
skill and rubric in place), confines writes to the work dir, and denies
network for shell commands. It's the tightest policy that still lets the
"read a skill, write a pipeline.yaml" contract run unattended. Note that
``codex exec`` defaults to ``read-only``, so passing this is mandatory —
without it the agent cannot write its output."""

DEFAULT_MODEL: str | None = None
"""No hardcoded model id.

Codex model ids move fast, and the right default depends on the user's
account and ``~/.codex/config.toml``. Leaving this ``None`` means we omit
``--model`` entirely and let Codex use its configured default; a user who
wants a specific model passes ``rote graduate --model <id>``, which flows
through to ``--model``."""


class CodexDriver(GraduatorDriver):
    name: str = "codex"

    def __init__(
        self,
        model: str | None = DEFAULT_MODEL,
        sandbox: str = DEFAULT_SANDBOX,
        codex_executable: str = "codex",
        **_ignored: Any,
    ) -> None:
        """
        Parameters
        ----------
        model
            Codex model id (``--model``). ``None`` omits the flag and uses
            Codex's configured default.
        sandbox
            Sandbox policy passed to ``--sandbox``.
        codex_executable
            Name/path of the Codex CLI binary.

        Extra keyword arguments (e.g. ``max_turns``, which Codex exec has
        no equivalent for) are absorbed and ignored, per the
        driver-registry ``**kwargs`` convention.
        """
        self.model = model
        self.sandbox = sandbox
        self.codex_executable = codex_executable

    def is_available(self) -> tuple[bool, str]:
        """Check whether the ``codex`` CLI is installed on PATH.

        Auth is not checked here — that surfaces at ``run()`` time.
        Probing auth would mean spawning a process, which ``is_available``
        (called during auto-detect) must stay cheap enough to avoid.
        """
        if which(self.codex_executable) is None:
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
        extra_instructions: str | None = None,
        on_event: EventCallback | None = None,
    ) -> DriverResult:
        """Spawn ``codex exec`` and wait for it to produce pipeline.yaml.

        Codex's stdout is not stream-parsed for turn/tool events — the
        ``codex exec`` output format isn't a stable machine contract we
        want to depend on. Phase events still flow: a
        :class:`~rote.graduator.events.ProgressFileWatcher` polls the
        ``progress.ndjson`` the agent writes, exactly as it does for the
        claude driver.
        """
        skill_dir = skill_dir.resolve()
        graduator_skill_dir = graduator_skill_dir.resolve()
        work_dir = work_dir.resolve()
        work_dir.mkdir(parents=True, exist_ok=True)

        skill_md = graduator_skill_dir / "SKILL.md"
        if not skill_md.is_file():
            raise DriverError(
                f"rote-graduate SKILL.md not found at {skill_md}. "
                f"Pass an explicit graduator_skill_dir to the orchestrator."
            )

        prompt = self._build_prompt(
            skill_dir,
            graduator_skill_dir,
            work_dir,
            skill_md.read_text(encoding="utf-8"),
            extra_instructions,
        )

        # Capture the agent's final message for metadata. Written to a
        # temp file *outside* work_dir so the orchestrator's move of
        # work_dir → user output doesn't sweep it up as an artifact.
        last_message_fd, last_message_path = tempfile.mkstemp(
            prefix="rote-codex-last-", suffix=".txt"
        )
        os.close(last_message_fd)

        try:
            args = [
                self.codex_executable,
                "exec",
                "--cd",
                str(work_dir),
                "--sandbox",
                self.sandbox,
                "--skip-git-repo-check",
                "--ephemeral",
                "--color",
                "never",
                "--output-last-message",
                last_message_path,
            ]
            if self.model:
                args += ["--model", self.model]
            args.append(prompt)

            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=os.environ.copy(),
            )
            async with ProgressFileWatcher(work_dir, on_event):
                stdout_bytes, stderr_bytes = await proc.communicate()

            stdout = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
            stderr = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""

            pipeline_yaml = work_dir / "pipeline.yaml"

            # The deliverable is a file on disk, not a clean exit code.
            # Mirror ClaudeDriver: check for pipeline.yaml FIRST so a
            # nonzero exit *after* the file was written (a transient error
            # on a final self-check turn) doesn't discard a real run.
            if not pipeline_yaml.is_file():
                if proc.returncode != 0:
                    raise DriverError(
                        f"codex CLI exited with code {proc.returncode}",
                        details=stderr or stdout or "(no output)",
                    )
                raise DriverError(
                    f"codex CLI finished successfully but did not produce {pipeline_yaml}.",
                    details=stdout,
                )

            metadata = self._build_metadata(last_message_path)
            if proc.returncode != 0:
                metadata["subprocess_warning"] = (
                    f"codex CLI exited with code {proc.returncode} but "
                    f"pipeline.yaml was produced. Treating as success."
                )

            return DriverResult(
                pipeline_yaml_path=pipeline_yaml,
                work_dir=work_dir,
                driver_name=self.name,
                metadata=metadata,
            )
        finally:
            # Best-effort cleanup of the out-of-tree last-message file.
            with contextlib.suppress(OSError):
                os.unlink(last_message_path)

    # ───────── Private helpers ─────────

    def _build_prompt(
        self,
        skill_dir: Path,
        graduator_skill_dir: Path,
        work_dir: Path,
        graduator_skill_md: str,
        extra_instructions: str | None,
    ) -> str:
        """Build the single positional prompt for ``codex exec``.

        Codex has no ``--append-system-prompt`` equivalent, so the
        rote-graduate rubric's SKILL.md is inlined directly into the
        prompt (its reference files are read from disk by the agent under
        the workspace-write sandbox's global read access). The task
        framing and the read/write paths mirror ClaudeDriver's prompts.
        """
        task = (
            f"You are the rote graduator. Follow the procedure and rubric below "
            f"to graduate the skill at {skill_dir}.\n\n"
            f"Paths for this run:\n"
            f"  - Source skill (read): {skill_dir}\n"
            f"  - Rote-graduate rubric (read): {graduator_skill_dir}\n"
            f"  - Work directory (write): {work_dir}\n\n"
            f"Begin by reading {graduator_skill_dir}/SKILL.md and its reference "
            f"files under {graduator_skill_dir}/references/, then follow the "
            f"procedure it describes.\n\n"
            f"Your final deliverable is {work_dir}/pipeline.yaml. Write any "
            f"extracted Python modules to {work_dir}/extracted/ and any signature "
            f"stubs to {work_dir}/signatures/, as the rubric instructs. Write "
            f"only inside the work directory."
        )
        if extra_instructions:
            task = f"{task}\n\n{extra_instructions}"
        return (
            f"{task}\n\n"
            "================== ROTE GRADUATE SKILL ==================\n\n"
            f"{graduator_skill_md}"
        )

    def _build_metadata(self, last_message_path: str) -> dict[str, Any]:
        """Assemble driver metadata from the captured final message.

        Tolerant by design: a missing/empty last-message file just yields
        the driver name. The final message can be large, so it's clipped.
        """
        metadata: dict[str, Any] = {"driver": self.name}
        try:
            text = Path(last_message_path).read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            text = ""
        if text:
            metadata["last_message"] = text[:1000]
        return metadata
