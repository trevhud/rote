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
import time
from pathlib import Path
from shutil import which
from typing import Any

from rote.graduator.drivers import DriverError, DriverResult, GraduatorDriver
from rote.graduator.events import (
    EventCallback,
    GraduationEvent,
    ProgressFileWatcher,
    emit_safely,
    relative_display_path,
)

# ───────── Defaults ─────────

#: StreamReader line-buffer cap for the subprocess stdout. Claude Code's
#: stream-json emits one JSON object per line, and an ``assistant`` line
#: carrying a Write tool_use block inlines the *entire* file content —
#: easily past asyncio's 64 KiB default, which would raise
#: ``LimitOverrunError`` mid-run. 16 MiB comfortably covers a graduated
#: module or a large pipeline.yaml.
_STREAM_LINE_LIMIT = 16 * 1024 * 1024

#: Turn-event message cap, matching the api driver's snippet width.
_TURN_SNIPPET_CHARS = 120

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
access needed), no WebFetch / WebSearch by default (no external
network), no TodoWrite. Read + Write + Edit cover file I/O; Glob +
Grep cover discovery of source skill structure and reference content."""

WEB_TOOLS = "WebSearch,WebFetch"
"""Research tools appended when ``web_tools=True``: the agent may look
up *current* vendor API docs instead of trusting training-data memory.
Required for reliable ``--backend api`` output — SDK shapes drift faster
than model knowledge — and off by default because ordinary graduation
needs no network and shouldn't pay the latency."""


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


# ───────── stream-json parsing (pure, testable) ─────────


def _stream_text_snippet(content: list[Any]) -> str:
    """First ~120 chars of an assistant message's text, or ``"thinking…"``.

    ``content`` is the raw block list from a stream-json ``assistant``
    message (plain dicts, not SDK objects). A turn made entirely of
    tool_use blocks has no prose to quote and reports as thinking.
    """
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            text = str(block.get("text") or "").strip()
            if text:
                return text[:_TURN_SNIPPET_CHARS]
    return "thinking…"


def _usage_delta(message: object) -> tuple[int, int]:
    """``(input_tokens, output_tokens)`` from a stream-json ``message.usage``.

    Claude Code stamps each ``assistant`` message with the token usage of
    that single model response. Missing / non-integer fields count as 0 so
    a malformed line never poisons the running total. Cache-token fields
    are intentionally ignored to match the api driver, which sums only the
    ``input_tokens`` / ``output_tokens`` totals.
    """
    if not isinstance(message, dict):
        return 0, 0
    usage = message.get("usage")
    if not isinstance(usage, dict):
        return 0, 0

    def _int(value: object) -> int:
        return value if isinstance(value, int) and not isinstance(value, bool) else 0

    return _int(usage.get("input_tokens")), _int(usage.get("output_tokens"))


def _map_stream_line(
    raw: str,
    turn: int,
    work_dir: Path | None = None,
    totals: dict[str, int] | None = None,
) -> tuple[list[GraduationEvent], int, dict[str, Any] | None]:
    """Map one stream-json NDJSON line to progress events.

    Pure and side-effect-free so it can be unit-tested against canned
    lines. Returns ``(events, new_turn, parsed_obj)``:

    * ``events`` — a ``turn`` event for an ``assistant`` message (with a
      text snippet and cumulative ``tokens``), plus one ``tool`` event per
      ``tool_use`` block in it.
    * ``new_turn`` — ``turn`` + 1 for an assistant message (each is one
      run turn), otherwise unchanged.
    * ``parsed_obj`` — the decoded dict (or ``None`` for a non-JSON line),
      so the caller can pick out the final ``result`` object for metadata
      without parsing the line a second time.

    ``totals`` is the caller's running ``{"input", "output"}`` token
    tally, mutated in place across lines so every ``turn`` event carries
    *cumulative* usage — the stream-json ``usage`` is per-message, so the
    driver accumulates it exactly the way the api driver does. When
    omitted (pure-function callers) a fresh tally is used, so a single
    line still reports its own usage.

    Phase events are deliberately NOT produced here — a
    :class:`~rote.graduator.events.ProgressFileWatcher` owns those for
    subprocess drivers, off the ``progress.ndjson`` file. ``work_dir``,
    when given, relativizes ``tool`` event paths for display.
    """
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return [], turn, None
    if not isinstance(obj, dict):
        return [], turn, None
    if obj.get("type") != "assistant":
        return [], turn, obj

    message = obj.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, list):
        content = []

    if totals is None:
        totals = {"input": 0, "output": 0}
    delta_in, delta_out = _usage_delta(message)
    totals["input"] += delta_in
    totals["output"] += delta_out

    turn += 1
    events: list[GraduationEvent] = [
        GraduationEvent(
            type="turn",
            ts=time.time(),
            turn=turn,
            tokens=dict(totals),
            message=f"turn {turn}: {_stream_text_snippet(content)}",
        )
    ]
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "tool_use":
            continue
        tool_name = str(block.get("name") or "tool")
        tool_input = block.get("input")
        raw_path = None
        if isinstance(tool_input, dict):
            # Claude Code's file tools key the path as ``file_path``.
            raw_path = tool_input.get("file_path") or tool_input.get("path")
        rel = relative_display_path(raw_path, work_dir) if work_dir else raw_path
        events.append(
            GraduationEvent(
                type="tool",
                ts=time.time(),
                turn=turn,
                tool_name=tool_name,
                path=rel,
                message=f"{tool_name} {rel}".rstrip() if rel else tool_name,
            )
        )
    return events, turn, obj


def _result_detail(result_obj: dict[str, Any] | None) -> str:
    """Render the final result object as an error-detail string.

    Prefers the human-readable ``result`` text the CLI attaches to a
    failed run; falls back to a compact JSON dump, clipped so a giant
    payload can't flood the error.
    """
    if not isinstance(result_obj, dict):
        return ""
    text = result_obj.get("result")
    if isinstance(text, str) and text.strip():
        return text[:2000]
    return json.dumps(result_obj)[:2000]


class ClaudeDriver(GraduatorDriver):
    name: str = "claude"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_turns: int = DEFAULT_MAX_TURNS,
        allowed_tools: str = DEFAULT_ALLOWED_TOOLS,
        claude_executable: str = "claude",
        web_tools: bool = False,
    ) -> None:
        self.model = model
        self.max_turns = max_turns
        if web_tools:
            allowed_tools = ",".join([allowed_tools, WEB_TOOLS])
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
        on_event: EventCallback | None = None,
    ) -> DriverResult:
        """Spawn ``claude -p`` and wait for it to produce pipeline.yaml.

        Runs the CLI in ``stream-json`` mode and consumes its NDJSON
        stdout incrementally: each ``assistant`` message becomes a
        ``turn`` event (with the tool_use blocks inside it fanned out to
        ``tool`` events), and the final ``result`` object supplies the
        run metadata. A :class:`~rote.graduator.events.ProgressFileWatcher`
        runs concurrently for phase events off ``progress.ndjson``.

        The stream is parsed the same way whether or not ``on_event`` is
        wired — a single code path is simpler than a stream/no-stream
        fork, and the metadata still comes from the same ``result``
        object either way.
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
            "stream-json",
            "--verbose",
            "--max-turns",
            str(self.max_turns),
        ]

        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            limit=_STREAM_LINE_LIMIT,
        )

        async with ProgressFileWatcher(work_dir, on_event):
            result_obj, stderr, totals = await self._consume_stream(proc, on_event, work_dir)
        await proc.wait()

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
            detail = stderr or _result_detail(result_obj) or "(no output)"
            if proc.returncode != 0:
                raise DriverError(
                    f"claude CLI exited with code {proc.returncode}",
                    details=detail,
                )
            raise DriverError(
                f"claude CLI finished successfully but did not produce {pipeline_yaml}.",
                details=detail,
            )

        metadata = self._parse_metadata(result_obj, totals)
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

    async def _consume_stream(
        self,
        proc: asyncio.subprocess.Process,
        on_event: EventCallback | None,
        work_dir: Path,
    ) -> tuple[dict[str, Any] | None, str, dict[str, int]]:
        """Drain the subprocess's stream-json stdout and its stderr.

        Reads stdout line-by-line, mapping each to progress events via
        :func:`_map_stream_line` and firing them through ``on_event``.
        stderr is drained concurrently (a full pipe would otherwise
        deadlock the child). Returns the final ``result`` object (for
        metadata), the decoded stderr text (for error details), and the
        cumulative ``{"input", "output"}`` token tally accumulated across
        the assistant messages (per-message usage summed the way the api
        driver sums it).
        """
        assert proc.stdout is not None
        assert proc.stderr is not None

        result_obj: dict[str, Any] | None = None
        turn = 0
        totals: dict[str, int] = {"input": 0, "output": 0}

        async def read_stdout() -> None:
            nonlocal result_obj, turn
            async for raw in proc.stdout:  # type: ignore[union-attr]
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                events, turn, obj = _map_stream_line(line, turn, work_dir, totals)
                for event in events:
                    emit_safely(on_event, event)
                if isinstance(obj, dict) and obj.get("type") == "result":
                    result_obj = obj

        async def read_stderr() -> bytes:
            return await proc.stderr.read()  # type: ignore[union-attr]

        _, stderr_bytes = await asyncio.gather(read_stdout(), read_stderr())
        stderr = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""
        return result_obj, stderr, totals

    def _parse_metadata(
        self, result: dict[str, Any] | None, totals: dict[str, int]
    ) -> dict[str, Any]:
        """Turn the stream-json ``result`` object into driver metadata.

        The final ``{"type": "result", ...}`` line of the stream carries
        the same run summary the old ``--output-format json`` mode
        returned as its whole payload: ``cost_usd``, ``duration_ms``,
        ``num_turns``, ``session_id``. We keep those numeric/id fields and
        drop the (potentially large) ``result`` text. ``input_tokens`` /
        ``output_tokens`` come from the streamed-usage ``totals`` and
        mirror the api driver's final metadata, so a consumer can price a
        subscription run the same way it prices an api one.

        A missing result object — the stream ended without a summary line
        — degrades to the driver name plus the token totals, same shape
        an unparseable payload produces.
        """
        metadata = {
            "driver": self.name,
            "input_tokens": totals["input"],
            "output_tokens": totals["output"],
        }
        if not isinstance(result, dict):
            return metadata
        metadata.update(
            cost_usd=result.get("cost_usd"),
            duration_ms=result.get("duration_ms"),
            num_turns=result.get("num_turns"),
            session_id=result.get("session_id"),
        )
        return metadata
