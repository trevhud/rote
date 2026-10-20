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
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from shutil import which
from typing import Any

from rote.compiler.drivers import CompilerDriver, DriverError, DriverResult
from rote.compiler.events import (
    CompilationEvent,
    EventCallback,
    ProgressFileWatcher,
    emit_safely,
    relative_display_path,
)

# Imported, not defined here: emitted apps run the identical scrub
# (their `claude-cli` judge provider spawns the same subprocess) and
# cannot import rote, so the one definition lives in the stdlib-only
# helper that ships into them verbatim. A second copy here is exactly
# how the rule drifts back to API-key billing in one place and not the
# other.
from rote.inference import CLI_CONTEXT_ISOLATION_FLAGS, build_subscription_env

# ───────── Defaults ─────────

#: StreamReader line-buffer cap for the subprocess stdout. Claude Code's
#: stream-json emits one JSON object per line, and an ``assistant`` line
#: carrying a Write tool_use block inlines the *entire* file content —
#: easily past asyncio's 64 KiB default, which would raise
#: ``LimitOverrunError`` mid-run. 16 MiB comfortably covers a compiled
#: module or a large pipeline.yaml.
_STREAM_LINE_LIMIT = 16 * 1024 * 1024

#: Turn-event message cap, matching the api driver's snippet width.
_TURN_SNIPPET_CHARS = 120

DEFAULT_MODEL = "claude-sonnet-4-6"
"""Default model for the compiler agent.

**Why Sonnet, not Opus:** Claude Code defaults to Opus 4.6, which is
overkill for the compiler's task. Sonnet 4.6 follows structured
rubrics just as reliably for this kind of work and is ~5× cheaper on
both the Anthropic API and the Claude Max/Pro subscription budget.

**Measured on BDR outreach (2026-04-07):** two runs with Opus burned
~$3.50 each in subscription accounting (2.6M cache-read tokens, 31K
output tokens, 31 turns). At that rate a single Max/Pro user's "extra
usage" allowance is exhausted in 2-3 runs. The Sonnet switch brings
per-run cost to ~$0.70, which makes iterative rubric tuning actually
feasible.

Users can override with ``ClaudeDriver(model=...)`` or the CLI flag
``rote compile --model claude-opus-4-6`` for complex skills where
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

DEFAULT_EFFORT = "low"
"""Claude Code effort level for the compiler (``--effort``).

Effort is what sizes adaptive thinking on Sonnet 4.6, and Claude Code's
default for it is ``high``. Measured on six real compiles (2026-10-18,
expense-audit and ops-report, Sonnet 4.6 at the default): thinking was
62 to 76 percent of all output tokens (62,901 of 94,428; 123,597 of
162,506), and output tokens were most of the bill and nearly all of the
wall time, since the CLI streams them at about 75 per second. Most of it
came in one block right after the rubric was read: the agent drafted the
whole compile in its head, hit the per-response output cap with no tool
call made, and the CLI discarded the block and started over. One ops-report
compile did that twice, 64,000 tokens and 14 minutes gone before its
first Write.

Measured on the "think in files" rubric (2026-10-19, rote bench scorer),
against a same-day ``high`` baseline of $2.25 and 986 s on expense-audit:
``medium`` was $1.91 and 916 s (25k thinking tokens), ``low`` was $1.31
and $1.20, 513 s and 423 s (under 10k), and ops-report at ``low`` was
$0.87 and 329 s against $2.46 to $3.54 and 17 to 30 minutes at ``high``.
Every compile verified clean, with every pure function implemented and
node and kind accuracy at or above the ``high`` runs'. The rubric does
the structuring that the thinking used to do.

A user who wants more for a hard skill passes ``rote compile --effort``
(or sets ``ROTE_EFFORT`` / the ``effort`` config key). The child's
``CLAUDE_CODE_EFFORT_LEVEL`` is scrubbed, because Claude Code ranks that
env var above ``--effort``: one exported in a shell profile for
interactive use would otherwise silently re-price every compile, and
rote's own precedence (flag > env > project > user > this default) would
be a lie. Pinned by ``tests/test_cost_defaults.py``.
https://code.claude.com/docs/en/model-config"""

EFFORT_LEVELS: tuple[str, ...] = ("low", "medium", "high")
"""The levels ``--effort`` accepts. Claude Code also has ``xhigh`` and
``max``; they are left out on purpose, since Sonnet 4.6 has no ``xhigh``
(it runs as ``high``) and ``max`` is the costliest level there is."""

PROMPT_CACHE_ENV: dict[str, str] = {"FORCE_PROMPT_CACHING_5M": "1"}
"""Forces Claude Code's 5-minute prompt-cache TTL in the compiler child.

On a subscription, Claude Code writes the cache with the 1-hour TTL, which
bills a write at 2x the input rate against 1.25x for 5 minutes. A
compile's calls follow each other within seconds, and every read renews
the TTL, so the extra hour buys nothing. Profiled compiles (2026-10-19,
Sonnet 4.6 and 5.5) put 35 to 40 percent of ``total_cost_usd`` in 1-hour
cache writes. The variable outranks every other TTL setting, including
a user's own ``promptCacheTtl``.
https://code.claude.com/docs/en/prompt-caching#choose-the-ttl-yourself"""

DEFAULT_ALLOWED_TOOLS = "Read,Write,Edit,Glob,Grep"
"""Conservative tool allowlist for the compiler. No Bash (no shell
access needed), no WebFetch / WebSearch by default (no external
network), no TodoWrite. Read + Write + Edit cover file I/O; Glob +
Grep cover discovery of source skill structure and reference content."""

#: Claude Code's per-response output cap for the compiler subprocess.
#: Unset, it is 32000 for a model the CLI cannot resolve, and a real
#: invoice-push compile died on it after 28 minutes and 8 turns ("Claude's
#: response exceeded the 32000 output token maximum"), on one large Write.
#: 64000 is the default compile model's (claude-sonnet-4-6) own maximum;
#: Claude Code lowers a value above a model's cap to the cap, so a larger
#: model is never asked for more than it supports. Higher is not free: the
#: docs note it shrinks the context left before auto-compaction. A value
#: already set in the caller's environment is respected.
#: https://code.claude.com/docs/en/env-vars
MAX_OUTPUT_TOKENS = "64000"

#: The CLI's message when one response hits that cap. The run ends there,
#: but the session is intact, so it is resumable.
_OUTPUT_CAP_RE = re.compile(r"exceeded the \d+ output token maximum")

#: The one user turn a capped session is resumed with. The cap is hit by a
#: single huge tool call (a whole module in one Write), so the fix is to
#: write in pieces, not to retry the same call.
OUTPUT_CAP_RESUME = (
    "Your last response was cut off: it exceeded the output-token limit for a "
    "single response, so nothing in it took effect. Continue the compilation "
    "from where you were. Write any large file in pieces: Write a first part, "
    "then add the rest with Edit, keeping each tool call well under 20000 "
    "tokens. Do not rewrite files that are already complete."
)

WEB_TOOLS = "WebSearch,WebFetch"
"""Research tools appended when ``web_tools=True``: the agent may look
up *current* vendor API docs instead of trusting training-data memory.
Required for reliable ``--backend api`` output — SDK shapes drift faster
than model knowledge — and off by default because ordinary compilation
needs no network and shouldn't pay the latency."""


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


#: The four token buckets a run accumulates. ``cache_write`` /
#: ``cache_read`` carry nearly all of a ``claude -p`` run's input
#: volume; see :func:`_usage_delta`.
_EMPTY_TOTALS: dict[str, int] = {"input": 0, "output": 0, "cache_write": 0, "cache_read": 0}


def _usage_delta(message: object) -> dict[str, int]:
    """Per-bucket token usage from a stream-json ``message.usage``.

    Claude Code repeats input/cache usage on each content block of one
    response; callers deduplicate by message.id. Per-block output usage
    is provisional, so only final result usage may supply output totals.
    Missing / non-integer fields count as 0, matching the existing decoder.

    All four buckets are captured, and the two cache buckets are the
    point. ``claude -p`` always runs prompt-cached, so ``input_tokens``
    holds only the handful of tokens that were neither written to nor read
    from the cache: a measured one-word reply reported ``input_tokens: 3``
    beside ``cache_creation_input_tokens: 122956``. Summing the plain
    fields alone therefore undercounts a real compilation by orders of
    magnitude, and anything priced from that total is wrong by the same
    factor. The api driver can sum two fields because it sends no
    ``cache_control`` at all; copying its policy here was the bug.
    """
    empty = dict(_EMPTY_TOTALS)
    if not isinstance(message, dict):
        return empty
    usage = message.get("usage")
    if not isinstance(usage, dict):
        return empty

    def _int(value: object) -> int:
        return value if isinstance(value, int) and not isinstance(value, bool) else 0

    return {
        "input": _int(usage.get("input_tokens")),
        "output": _int(usage.get("output_tokens")),
        "cache_write": _int(usage.get("cache_creation_input_tokens")),
        "cache_read": _int(usage.get("cache_read_input_tokens")),
    }


def _map_stream_line(
    raw: str,
    turn: int,
    work_dir: Path | None = None,
    totals: dict[str, int] | None = None,
    message_turns: dict[str, int] | None = None,
) -> tuple[list[CompilationEvent], int, dict[str, Any] | None]:
    """Map one stream-json NDJSON line to progress events.

    Returns ``(events, new_turn, parsed_obj)`` while updating the caller's
    token totals and response-ID map:

    * ``events`` — a ``turn`` event for a new assistant response ID (with a
      text snippet and cumulative ``tokens``), plus one ``tool`` event per
      ``tool_use`` block in it.
    * ``new_turn`` — increments per distinct response, not per content
      block. The provider's final ``num_turns`` remains separate metadata.
    * ``parsed_obj`` — the decoded dict (or ``None`` for a non-JSON line),
      so the caller can pick out the final ``result`` object for metadata
      without parsing the line a second time.

    Stream events carry deduplicated input/cache totals only. Output is
    pending until the final result, because assistant output_tokens is a
    message-start placeholder. All tool blocks still emit events, even
    when their response was already counted. Older ID-less records retain
    the existing per-record behavior; completed result usage supersedes it.

    Phase events are deliberately NOT produced here — a
    :class:`~rote.compiler.events.ProgressFileWatcher` owns those for
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
        totals = dict(_EMPTY_TOTALS)
    message_id = message.get("id") if isinstance(message, dict) else None
    known_turn = (
        message_turns.get(message_id)
        if message_turns is not None and isinstance(message_id, str)
        else None
    )
    events: list[CompilationEvent] = []
    if known_turn is None:
        # Claude emits multiple content blocks with the same response ID and
        # usage. Only the first contributes input/cache tokens and a new turn.
        # Per-block output_tokens is a message-start placeholder, not usage.
        for bucket, delta in _usage_delta(message).items():
            if bucket != "output":
                totals[bucket] = totals.get(bucket, 0) + delta
        turn += 1
        known_turn = turn
        if message_turns is not None and isinstance(message_id, str) and message_id:
            message_turns[message_id] = turn
        events.append(
            CompilationEvent(
                type="turn",
                ts=time.time(),
                turn=turn,
                tokens={key: value for key, value in totals.items() if key != "output"},
                usage_complete=False,
                message=f"turn {turn}: {_stream_text_snippet(content)}",
            )
        )
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
            CompilationEvent(
                type="tool",
                ts=time.time(),
                turn=known_turn,
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


def _hit_output_cap(result_obj: dict[str, Any] | None) -> bool:
    """Whether a run ended on Claude Code's per-response output-token cap."""
    if not isinstance(result_obj, dict):
        return False
    text = result_obj.get("result")
    return isinstance(text, str) and bool(_OUTPUT_CAP_RE.search(text))


@dataclass(frozen=True)
class _Invocation:
    """One ``claude -p`` subprocess: the first run or a resumed repair round."""

    result: dict[str, Any] | None
    stderr: str
    totals: dict[str, int]
    returncode: int | None
    turn: int


class ClaudeDriver(CompilerDriver):
    name: str = "claude"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_turns: int = DEFAULT_MAX_TURNS,
        allowed_tools: str = DEFAULT_ALLOWED_TOOLS,
        claude_executable: str = "claude",
        web_tools: bool = False,
        effort: str = DEFAULT_EFFORT,
        **_ignored: Any,
    ) -> None:
        """``**_ignored`` absorbs registry kwargs meant for other drivers
        (``mcp_servers`` for the in-process ones), per the driver-factory
        convention."""
        if effort not in EFFORT_LEVELS:
            raise ValueError(f"effort must be one of {', '.join(EFFORT_LEVELS)}, got {effort!r}")
        self.model = model
        self.max_turns = max_turns
        self.effort = effort
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
        compiler_skill_dir: Path,
        work_dir: Path,
        extra_instructions: str | None = None,
        on_event: EventCallback | None = None,
        repair: Callable[[Path], str | None] | None = None,
    ) -> DriverResult:
        """Spawn ``claude -p`` and wait for it to produce pipeline.yaml.

        ``repair`` is the orchestrator's verification callback. Once the
        agent has produced pipeline.yaml, it is called with that path; a
        string return is sent back into the same conversation with
        ``claude -p --resume <session_id>`` and the loop repeats until it
        returns ``None``. Resuming keeps the model that wrote the code,
        with the skill and its own reasoning still in context, as the one
        that fixes it.

        Runs the CLI in ``stream-json`` mode and consumes its NDJSON
        stdout incrementally: each ``assistant`` message becomes a
        ``turn`` event (with the tool_use blocks inside it fanned out to
        ``tool`` events), and the final ``result`` object supplies the
        run metadata. A :class:`~rote.compiler.events.ProgressFileWatcher`
        runs concurrently for phase events off ``progress.ndjson``.

        The stream is parsed the same way whether or not ``on_event`` is
        wired — a single code path is simpler than a stream/no-stream
        fork, and the metadata still comes from the same ``result``
        object either way.
        """
        skill_dir = skill_dir.resolve()
        compiler_skill_dir = compiler_skill_dir.resolve()
        work_dir = work_dir.resolve()
        work_dir.mkdir(parents=True, exist_ok=True)

        skill_md = compiler_skill_dir / "SKILL.md"
        if not skill_md.is_file():
            raise DriverError(
                f"rote-compile SKILL.md not found at {skill_md}. "
                f"Pass an explicit compiler_skill_dir to the orchestrator."
            )

        system_prompt = self._build_system_prompt(skill_md.read_text(encoding="utf-8"))
        user_prompt = self._build_user_prompt(skill_dir, compiler_skill_dir, work_dir)
        if extra_instructions:
            user_prompt = f"{user_prompt}\n\n{extra_instructions}"

        env = self._build_child_env()

        def args_for(prompt: str, resume: str | None = None) -> list[str]:
            args = [
                self.claude_executable,
                "-p",
                prompt,
                "--model",
                self.model,
                "--effort",
                self.effort,
                "--append-system-prompt",
                system_prompt,
                "--add-dir",
                str(skill_dir),
                "--add-dir",
                str(compiler_skill_dir),
                "--add-dir",
                str(work_dir),
                "--allowedTools",
                self.allowed_tools,
                # Isolation from the invoking user's Claude Code setup, the
                # same flags the emitted judge and agent-loop lanes use (one
                # definition, in the helper, so the copies cannot drift).
                # Each was found by a real compile: the user's connectors put
                # 357 tools into the first request (past 200k tokens, which a
                # subscription rejected before the compile started); a
                # prose-lint hook in the user's settings cost 12 of 48 turns
                # in one-line compile-report.md edits; and the same settings
                # granted Bash, which the allowlist deliberately omits.
                *CLI_CONTEXT_ISOLATION_FLAGS,
                # The compiler's own tool surface: exactly the allowlist.
                # A VALUE is required, as for the judges' empty one: the flag
                # is variadic, and a bare `--tools` is rejected outright.
                "--tools",
                self.allowed_tools,
                "--output-format",
                "stream-json",
                "--verbose",
                "--max-turns",
                str(self.max_turns),
            ]
            if resume is not None:
                args += ["--resume", resume]
            return args

        async def invoke(prompt: str, resume: str | None, start_turn: int) -> _Invocation:
            proc = await asyncio.create_subprocess_exec(
                *args_for(prompt, resume),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                limit=_STREAM_LINE_LIMIT,
                # The work dir, not the caller's cwd: run from a repo, the
                # agent globbed its way into that repo's earlier compiled
                # output of the same skill and read it as a template, so the
                # result depended on where `rote compile` was typed. It is
                # also where `--resume` looks the session up, so every
                # invocation of one compile must share it.
                cwd=work_dir,
            )
            result_obj, stderr, totals, turn = await self._consume_stream(
                proc, on_event, work_dir, start_turn=start_turn
            )
            await proc.wait()
            return _Invocation(result_obj, stderr, totals, proc.returncode, turn)

        # One watcher for the whole session: a watcher per invocation
        # re-announced every phase already in progress.ndjson at the start
        # of each repair round, so a repaired compile read as if it had
        # restarted from Phase 1.
        cap_resumed = False

        async def resume_if_capped(inv: _Invocation) -> _Invocation | None:
            """Resume once, compile-wide, a session that hit the output cap."""
            nonlocal cap_resumed
            if cap_resumed or not _hit_output_cap(inv.result):
                return None
            assert inv.result is not None
            session_id = inv.result.get("session_id")
            if not isinstance(session_id, str) or not session_id:
                return None
            cap_resumed = True
            emit_safely(
                on_event,
                CompilationEvent(
                    type="warning",
                    ts=time.time(),
                    message=(
                        "a response hit the output-token cap; resuming the session "
                        "once with an instruction to write large files in pieces"
                    ),
                ),
            )
            resumed = await invoke(OUTPUT_CAP_RESUME, session_id, inv.turn)
            invocations.append(resumed)
            return resumed

        invocations: list[_Invocation] = []
        async with ProgressFileWatcher(work_dir, on_event):
            first = await invoke(user_prompt, None, 0)
            invocations.append(first)
            # A capped first run is continued, and the continuation then
            # stands in for it in everything below.
            first = await resume_if_capped(first) or first
            result_obj, stderr = first.result, first.stderr
            # The first run's exit status decides recovery: a repair round that
            # dies is reported as a repair_warning, not a failed compilation.
            returncode = first.returncode

            pipeline_yaml = work_dir / "pipeline.yaml"
            repair_warning: str | None = None
            while repair is not None and pipeline_yaml.is_file():
                instruction = repair(pipeline_yaml)
                if instruction is None:
                    break
                last = invocations[-1]
                session_id = (
                    last.result.get("session_id") if isinstance(last.result, dict) else None
                )
                if not isinstance(session_id, str) or not session_id:
                    repair_warning = (
                        "the deliverable needed repair but the claude CLI reported no "
                        "session_id to resume, so it was left as the agent wrote it"
                    )
                    break
                repaired = await invoke(instruction, session_id, last.turn)
                invocations.append(repaired)
                repaired = await resume_if_capped(repaired) or repaired
                if repaired.returncode != 0:
                    repair_warning = (
                        f"a repair round exited with code {repaired.returncode}; the "
                        f"deliverable is whatever it had written by then"
                    )
                    break
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
            # The result envelope exists on most failures (an API error, the
            # output cap, max turns) and carries the session's cost. It used
            # to be dropped here, so a 28-minute failed compile reported no
            # cost anywhere.
            usage = self._session_metadata(invocations)
            if returncode != 0:
                raise DriverError(
                    f"claude CLI exited with code {returncode}",
                    details=detail,
                    metadata=usage,
                )
            raise DriverError(
                f"claude CLI finished successfully but did not produce {pipeline_yaml}.",
                details=detail,
                metadata=usage,
            )

        metadata = self._session_metadata(invocations)
        if repair_warning is not None:
            metadata["repair_warning"] = repair_warning
        if returncode != 0:
            metadata["subprocess_warning"] = (
                f"claude CLI exited with code {returncode} but "
                f"pipeline.yaml was produced. Treating as success."
            )

        return DriverResult(
            pipeline_yaml_path=pipeline_yaml,
            work_dir=work_dir,
            driver_name=self.name,
            metadata=metadata,
        )

    # ───────── Private helpers ─────────

    def _session_metadata(self, invocations: list[_Invocation]) -> dict[str, Any]:
        """Usage for the whole compile, across the first run and its repair rounds.

        A resumed ``claude -p`` reports the WHOLE session in
        ``total_cost_usd`` and ``modelUsage`` (measured on claude 2.1.302:
        a resume's cost was the first run's plus its own), while
        ``num_turns`` and ``duration_ms`` are per invocation. That is
        observed behavior with no version floor here, and it is least
        likely to hold exactly when it matters: a round after an abnormal
        exit. So it is checked, not assumed:

        * the rounds' costs never decrease: they are cumulative, and the
          last result is the session's usage;
        * otherwise each round reported only itself, and the rounds are
          summed (``usage_basis`` says which, ``usage_note`` why);
        * a round that produced no result leaves the total incomplete, so
          ``usage_complete`` is False. A number that silently omits a
          round reads as a small number, not a missing one.
        """
        per = [self._parse_metadata(inv.result, inv.totals) for inv in invocations]
        if len(invocations) == 1:
            return per[0]
        reported = [m for inv, m in zip(invocations, per, strict=True) if inv.result is not None]
        costs = [m.get("cost_usd") for m in reported]
        numeric = [c for c in costs if isinstance(c, int | float) and not isinstance(c, bool)]
        cumulative = len(numeric) == len(costs) and all(
            later >= earlier for earlier, later in zip(numeric, numeric[1:], strict=False)
        )
        if not reported:
            metadata = per[-1]
        elif cumulative:
            metadata = dict(reported[-1])
            metadata["usage_basis"] = "cumulative_last_result"
        else:
            metadata = dict(reported[-1])
            for key in ("input_tokens", "cache_write_tokens", "cache_read_tokens"):
                metadata[key] = sum(int(m.get(key) or 0) for m in reported)
            outputs = [m.get("output_tokens") for m in reported]
            known = [o for o in outputs if isinstance(o, int)]
            metadata["output_tokens"] = sum(known) if len(known) == len(outputs) else None
            metadata["cost_usd"] = sum(numeric) if numeric else None
            metadata["usage_basis"] = "summed_per_round"
            metadata["usage_note"] = (
                f"repair rounds reported non-cumulative costs {costs}; summed them"
            )
        missing = len(invocations) - len(reported)
        if missing:
            metadata["usage_complete"] = False
            metadata["usage_note"] = (
                f"{missing} of {len(invocations)} claude invocations produced no result; "
                f"usage omits them"
            )
        metadata["repair_rounds"] = len(invocations) - 1
        for key in ("num_turns", "duration_ms"):
            values = [r.get(key) for r in (inv.result for inv in invocations) if r is not None]
            if values and all(isinstance(v, int) for v in values):
                metadata[key] = sum(v for v in values if isinstance(v, int))
        return metadata

    def _build_child_env(self) -> dict[str, str]:
        """Build the environment for the subprocess.

        Delegates to :func:`build_subscription_env` — shared with the
        eval harness's skill runner, which spawns ``claude -p`` under
        the same billing rules.
        """
        env = build_subscription_env()
        env.setdefault("CLAUDE_CODE_MAX_OUTPUT_TOKENS", MAX_OUTPUT_TOKENS)
        # Claude Code ranks this above --effort; see DEFAULT_EFFORT.
        env.pop("CLAUDE_CODE_EFFORT_LEVEL", None)
        env.update(PROMPT_CACHE_ENV)
        return env

    def _build_system_prompt(self, skill_md_text: str) -> str:
        """Build the ``--append-system-prompt`` content.

        Claude Code has a default coding-focused system prompt that
        we're happy to keep — we just *append* the rote-compile
        SKILL.md content on top of it. The agent reads reference
        files via its Read tool; we don't inline them here to keep
        the command-line argument small and to let Claude cache
        reference reads.
        """
        return (
            "You are the rote compiler. Follow the procedure and rubric "
            "below to compile the skill identified in the user prompt.\n\n"
            "When asked to read reference files, use the Read tool. When "
            "producing the pipeline.yaml and any extracted Python modules "
            "or signature stubs, use the Write tool.\n\n"
            "================== ROTE COMPILE SKILL ==================\n\n"
            f"{skill_md_text}"
        )

    def _build_user_prompt(
        self,
        skill_dir: Path,
        compiler_skill_dir: Path,
        work_dir: Path,
    ) -> str:
        """Build the ``-p`` prompt.

        Short and task-oriented; the heavy lifting is in the system
        prompt. The goal is to tell the agent which specific paths to
        read from and write to, and what the final deliverable is.
        """
        return (
            f"Compile the skill at {skill_dir}.\n\n"
            f"Paths for this run:\n"
            f"  - Source skill (read): {skill_dir}\n"
            f"  - Rote-compile rubric (read): {compiler_skill_dir}\n"
            f"  - Work directory (write): {work_dir}\n\n"
            f"The rubric's SKILL.md is already in your system prompt, so do "
            f"not Read it again. In your first turn, Read every reference "
            f"file under {compiler_skill_dir}/references/ and every file of "
            f"the source skill, all in that one turn, then follow the "
            f"procedure.\n\n"
            f"Your final deliverable is {work_dir}/pipeline.yaml. Write "
            f"any extracted Python modules to {work_dir}/extracted/, "
            f"their tests to {work_dir}/tests/, and any signature stubs "
            f"to {work_dir}/signatures/, as the rubric instructs."
        )

    async def _consume_stream(
        self,
        proc: asyncio.subprocess.Process,
        on_event: EventCallback | None,
        work_dir: Path,
        start_turn: int = 0,
    ) -> tuple[dict[str, Any] | None, str, dict[str, int], int]:
        """Drain the subprocess's stream-json stdout and its stderr.

        Reads stdout line-by-line, mapping each to progress events via
        :func:`_map_stream_line` and firing them through ``on_event``.
        stderr is drained concurrently (a full pipe would otherwise
        deadlock the child). Returns the final ``result`` object (for
        metadata), the decoded stderr text (for error details), the
        deduplicated input/cache tally, and the last turn number (a
        resumed run continues the count from ``start_turn``). Output is
        supplied by the final result rather than the assistant records'
        placeholders.
        """
        assert proc.stdout is not None
        assert proc.stderr is not None

        result_obj: dict[str, Any] | None = None
        turn = start_turn
        totals: dict[str, int] = dict(_EMPTY_TOTALS)
        message_turns: dict[str, int] = {}

        async def read_stdout() -> None:
            nonlocal result_obj, turn
            async for raw in proc.stdout:  # type: ignore[union-attr]
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                events, turn, obj = _map_stream_line(line, turn, work_dir, totals, message_turns)
                for event in events:
                    emit_safely(on_event, event)
                if isinstance(obj, dict) and obj.get("type") == "result":
                    result_obj = obj

        async def read_stderr() -> bytes:
            return await proc.stderr.read()  # type: ignore[union-attr]

        _, stderr_bytes = await asyncio.gather(read_stdout(), read_stderr())
        stderr = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""
        return result_obj, stderr, totals, turn

    def _parse_metadata(
        self, result: dict[str, Any] | None, totals: dict[str, int]
    ) -> dict[str, Any]:
        """Turn the stream-json ``result`` object into driver metadata.

        The final ``{"type": "result", ...}`` line of the stream carries
        the run summary: ``total_cost_usd``, ``duration_ms``,
        ``num_turns``, ``session_id``. We keep those numeric/id fields and
        drop the (potentially large) ``result`` text.

        ``total_cost_usd`` is the CLI's estimate, not billing data. Prefer
        it to rote's estimate because it accounts for multiple models.
        Token totals prefer result.modelUsage (including subagents), then
        result.usage (main loop). A missing result preserves the existing
        artifact-recovery path with input/cache totals, but output is null
        and usage_complete is false. Never report placeholders as usage.

        Contract: https://code.claude.com/docs/en/agent-sdk/cost-tracking
        """

        def valid_output(value: object) -> bool:
            return isinstance(value, int) and not isinstance(value, bool) and value >= 0

        usage_source = "stream"
        if isinstance(result, dict):
            model_usage = result.get("modelUsage")
            if isinstance(model_usage, dict) and model_usage:
                # Whole-query counts include subagents; result.usage only
                # covers the main agent loop. Both are SDK-documented fields.
                final_totals = dict(_EMPTY_TOTALS)
                complete = True
                for usage in model_usage.values():
                    if not isinstance(usage, dict) or not valid_output(usage.get("outputTokens")):
                        complete = False
                        break
                    for bucket, field in (
                        ("input", "inputTokens"),
                        ("output", "outputTokens"),
                        ("cache_write", "cacheCreationInputTokens"),
                        ("cache_read", "cacheReadInputTokens"),
                    ):
                        value = usage.get(field)
                        if isinstance(value, int) and not isinstance(value, bool):
                            final_totals[bucket] += value
                if complete:
                    totals = final_totals
                    usage_source = "modelUsage"
            elif isinstance(result.get("usage"), dict) and valid_output(
                result["usage"].get("output_tokens")
            ):
                totals = _usage_delta({"usage": result["usage"]})
                usage_source = "usage"
        metadata: dict[str, Any] = {
            "driver": self.name,
            "input_tokens": totals.get("input", 0),
            "output_tokens": totals.get("output", 0) if usage_source != "stream" else None,
            "cache_write_tokens": totals.get("cache_write", 0),
            "cache_read_tokens": totals.get("cache_read", 0),
            "usage_source": usage_source,
            "usage_complete": usage_source != "stream",
        }
        if not isinstance(result, dict):
            return metadata
        metadata.update(
            cost_usd=result.get("total_cost_usd"),
            duration_ms=result.get("duration_ms"),
            num_turns=result.get("num_turns"),
            session_id=result.get("session_id"),
        )
        return metadata
