"""Claude Code CLI driver — spawns ``claude -p`` in subscription mode.

This driver is the primary subscription path. It shells out to the
``claude`` CLI (Claude Code) in non-interactive "print" mode and
requires only that the user has run ``claude login`` interactively at
least once (or set ``CLAUDE_CODE_OAUTH_TOKEN`` for automation).

# The env-var gotcha

Claude Code's print mode (``claude -p``) has a documented behavior
where ``ANTHROPIC_API_KEY`` and ``ANTHROPIC_AUTH_TOKEN`` *always win*
over any active OAuth session. To force subscription auth, the
subprocess environment must not contain those variables.

This driver always scrubs both env vars from the child environment.
If the user wants API-key auth to Anthropic's servers, they should
use the ``api`` driver (which uses the ``anthropic`` SDK directly);
``ClaudeDriver`` is specifically the subscription path.

``CLAUDE_CODE_OAUTH_TOKEN`` — a long-lived token Anthropic issues for
automation — is always passed through if set. That's the cleanest
way to use ``ClaudeDriver`` in CI, where no interactive ``claude login``
has been run.

See ``docs/agent-runtime.md`` for the full rationale.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from shutil import which
from typing import Any

from rote.graduator.drivers import DriverError, DriverResult, GraduatorDriver

# ───────── Defaults ─────────

DEFAULT_MODEL = "claude-sonnet-4-6"
"""Default model for the graduator agent.

**Why Sonnet, not Opus:** Claude Code defaults to Opus 4.6, which is
overkill for the graduator's task. Sonnet 4.6 follows structured
rubrics just as reliably for this kind of work and is ~5× cheaper on
both the Anthropic API and the Claude Max/Pro subscription budget.

**Measured on BDR outreach (2026-04-07):** two runs with Opus burned
~$3.50 each in subscription accounting (2.6M cache-read tokens, 31K
output tokens, 31 turns). At that rate a single Max/Pro user's "extra
usage" allowance is exhausted in 2-3 runs. The Sonnet switch brings
per-run cost to ~$0.70, which makes iterative rubric tuning actually
feasible.

Users can override with ``ClaudeDriver(model=...)`` or the CLI flag
``rote graduate --model claude-opus-4-6`` for complex skills where
Opus's extra reasoning ability is worth the 5× cost.
"""

DEFAULT_MAX_TURNS = 60
"""Agents need a fairly generous turn budget for complex skills.

Measured on the BDR outreach skill (7 phases, 6 reference files,
~14 nodes, ~8 extracted modules, ~2 signatures): a clean run needs
roughly 25 tool calls minimum (reads + writes), realistically 40–50
with exploration, list_directory calls, and between-phase thinking.
30 is not enough; we observed an ``error_max_turns`` failure at that
limit with BDR. 60 leaves headroom for more complex skills and for
the agent to recover from tool errors."""

DEFAULT_ALLOWED_TOOLS = "Read,Write,Edit,Glob,Grep"
"""Conservative tool allowlist for the graduator. No Bash (no shell
access needed), no WebFetch / WebSearch (no external network), no
TodoWrite. Read + Write + Edit cover file I/O; Glob + Grep cover
discovery of source skill structure and reference content."""


def build_subscription_env() -> dict[str, str]:
    """Child environment for a subscription-billed ``claude -p`` spawn.

    Critical: scrub ``ANTHROPIC_API_KEY`` and ``ANTHROPIC_AUTH_TOKEN``.
    In ``claude -p`` mode these env vars always win over an active
    OAuth session, which defeats the whole point of the subscription
    path. Callers who want API-key auth should use
    ``AnthropicApiDriver`` instead — this helper is specifically about
    reusing the user's Claude Max/Pro subscription.
    """
    env = os.environ.copy()
    env.pop("ANTHROPIC_API_KEY", None)
    env.pop("ANTHROPIC_AUTH_TOKEN", None)
    # Silence spinners in non-interactive output so stdout is just
    # the JSON result and stderr is just the error text (if any).
    env["CLAUDE_CODE_DISABLE_NONINTERACTIVE_ANIMATIONS"] = "1"
    # CLAUDE_CODE_OAUTH_TOKEN, if set in the parent env, is
    # preserved by env.copy() — this is the automation-friendly
    # path for CI environments.
    return env


class ClaudeDriver(GraduatorDriver):
    name: str = "claude"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_turns: int = DEFAULT_MAX_TURNS,
        allowed_tools: str = DEFAULT_ALLOWED_TOOLS,
        claude_executable: str = "claude",
    ) -> None:
        self.model = model
        self.max_turns = max_turns
        self.allowed_tools = allowed_tools
        self.claude_executable = claude_executable

    def is_available(self) -> tuple[bool, str]:
        """Check whether ``claude`` CLI is installed on PATH.

        We intentionally do **not** check auth here — doing so would
        require spawning a ``claude`` process, which is expensive. Auth
        errors surface at ``run()`` time with their own clear messages.
        """
        if which(self.claude_executable) is None:
            return (
                False,
                "The `claude` CLI is not installed. "
                "Install Claude Code from https://code.claude.com/download "
                "and run `claude login` once to enable subscription auth.",
            )
        return (True, "")

    async def run(
        self,
        skill_dir: Path,
        graduator_skill_dir: Path,
        work_dir: Path,
        extra_instructions: str | None = None,
    ) -> DriverResult:
        """Spawn ``claude -p`` and wait for it to produce pipeline.yaml."""
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

        system_prompt = self._build_system_prompt(skill_md.read_text(encoding="utf-8"))
        user_prompt = self._build_user_prompt(skill_dir, graduator_skill_dir, work_dir)
        if extra_instructions:
            user_prompt = f"{user_prompt}\n\n{extra_instructions}"

        env = self._build_child_env()

        args = [
            self.claude_executable,
            "-p",
            user_prompt,
            "--model",
            self.model,
            "--append-system-prompt",
            system_prompt,
            "--add-dir",
            str(skill_dir),
            "--add-dir",
            str(graduator_skill_dir),
            "--add-dir",
            str(work_dir),
            "--allowedTools",
            self.allowed_tools,
            "--output-format",
            "json",
            "--max-turns",
            str(self.max_turns),
        ]

        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        stdout_bytes, stderr_bytes = await proc.communicate()

        stdout = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
        stderr = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""

        pipeline_yaml = work_dir / "pipeline.yaml"

        # The agent's deliverable is a file on disk, not a clean exit
        # code. A real-world failure mode: the agent writes a complete
        # pipeline.yaml at turn N, then runs an extra validation step at
        # turn N+1 that hits a transient API error (ECONNRESET, rate
        # limit, etc.) — the subprocess returns nonzero but the work is
        # done. Treating that as fatal would discard a $2+ run over a
        # blip, so we check for the file FIRST and only fail if it's
        # missing.
        if not pipeline_yaml.is_file():
            if proc.returncode != 0:
                raise DriverError(
                    f"claude CLI exited with code {proc.returncode}",
                    details=stderr or stdout or "(no output)",
                )
            raise DriverError(
                f"claude CLI finished successfully but did not produce {pipeline_yaml}.",
                details=stdout,
            )

        metadata = self._parse_metadata(stdout)
        if proc.returncode != 0:
            metadata["subprocess_warning"] = (
                f"claude CLI exited with code {proc.returncode} but "
                f"pipeline.yaml was produced. Treating as success."
            )

        return DriverResult(
            pipeline_yaml_path=pipeline_yaml,
            work_dir=work_dir,
            driver_name=self.name,
            metadata=metadata,
        )

    # ───────── Private helpers ─────────

    def _build_child_env(self) -> dict[str, str]:
        """Build the environment for the subprocess.

        Delegates to :func:`build_subscription_env` — shared with the
        eval harness's skill runner, which spawns ``claude -p`` under
        the same billing rules.
        """
        return build_subscription_env()

    def _build_system_prompt(self, skill_md_text: str) -> str:
        """Build the ``--append-system-prompt`` content.

        Claude Code has a default coding-focused system prompt that
        we're happy to keep — we just *append* the rote-graduate
        SKILL.md content on top of it. The agent reads reference
        files via its Read tool; we don't inline them here to keep
        the command-line argument small and to let Claude cache
        reference reads.
        """
        return (
            "You are the rote graduator. Follow the procedure and rubric "
            "below to graduate the skill identified in the user prompt.\n\n"
            "When asked to read reference files, use the Read tool. When "
            "producing the pipeline.yaml and any extracted Python modules "
            "or signature stubs, use the Write tool.\n\n"
            "================== ROTE GRADUATE SKILL ==================\n\n"
            f"{skill_md_text}"
        )

    def _build_user_prompt(
        self,
        skill_dir: Path,
        graduator_skill_dir: Path,
        work_dir: Path,
    ) -> str:
        """Build the ``-p`` prompt.

        Short and task-oriented; the heavy lifting is in the system
        prompt. The goal is to tell the agent which specific paths to
        read from and write to, and what the final deliverable is.
        """
        return (
            f"Graduate the skill at {skill_dir}.\n\n"
            f"Paths for this run:\n"
            f"  - Source skill (read): {skill_dir}\n"
            f"  - Rote-graduate rubric (read): {graduator_skill_dir}\n"
            f"  - Work directory (write): {work_dir}\n\n"
            f"Begin by reading {graduator_skill_dir}/SKILL.md and its "
            f"reference files under {graduator_skill_dir}/references/, "
            f"then follow the procedure it describes.\n\n"
            f"Your final deliverable is {work_dir}/pipeline.yaml. Write "
            f"any extracted Python modules to {work_dir}/extracted/ and "
            f"any signature stubs to {work_dir}/signatures/, as the "
            f"rubric instructs."
        )

    def _parse_metadata(self, stdout: str) -> dict[str, Any]:
        """Parse ``claude -p``'s ``--output-format json`` stdout.

        Per the Claude Code headless docs, the output is one JSON
        object with fields ``result``, ``cost_usd``, ``duration_ms``,
        ``num_turns``, ``session_id``. We strip ``result`` from the
        metadata (it can be large) and keep the numeric/id fields.

        If the stdout isn't parseable JSON — e.g. because there's a
        banner before it, or animations slipped through — we fall back
        to parsing the last non-empty line, then finally to returning
        a truncated raw output for debugging.
        """
        if not stdout.strip():
            return {"driver": self.name}

        data: Any = None
        try:
            data = json.loads(stdout)
        except json.JSONDecodeError:
            # Try the last non-empty line — sometimes there's preamble
            for line in reversed([line for line in stdout.split("\n") if line.strip()]):
                try:
                    data = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue

        if not isinstance(data, dict):
            return {
                "driver": self.name,
                "raw_output": stdout[:500],
            }

        return {
            "driver": self.name,
            "cost_usd": data.get("cost_usd"),
            "duration_ms": data.get("duration_ms"),
            "num_turns": data.get("num_turns"),
            "session_id": data.get("session_id"),
        }
