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
  agent can read the source skill and the rote-compile rubric wherever
  they live — no grant needed.
* **Writes** are confined to the working directory (``--cd <work_dir>``)
  plus the system temp dir — exactly the "read the skill, write the
  pipeline.yaml into work_dir" contract every driver honors.
* **Network** is denied for model-generated shell commands. Codex's own
  model API calls are not subject to this (they're the CLI's connection,
  not a sandboxed command), so the compiler loop still works.

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
``OPENAI_API_KEY`` must still be able to compile through Codex. This
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
import json
import os
import re
import tempfile
import time
from pathlib import Path
from shutil import which
from typing import Any

from rote.compiler.drivers import CompilerDriver, DriverError, DriverResult
from rote.compiler.events import (
    CompilationEvent,
    EventCallback,
    ProgressFileWatcher,
    emit_safely,
)

# ───────── Defaults ─────────

DEFAULT_SANDBOX = "workspace-write"
"""Sandbox policy for the compiler run.

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
wants a specific model passes ``rote compile --model <id>``, which flows
through to ``--model``."""

#: ``web_tools=True`` (``--backend api``): Codex's native Responses
#: ``web_search`` tool in live mode. ``codex exec`` has no ``--search``
#: flag (that one is the interactive command's), so it is set through the
#: config override, whose value is parsed as TOML. Live, not the default
#: ``cached`` index: the point is the vendor's current docs.
#: https://learn.chatgpt.com/docs/config-file/config-reference
WEB_SEARCH_CONFIG = 'web_search="live"'

#: A ``web_search`` result Codex opened as a page (``turn1view0``), as
#: against a search hit (``turn0search0``). Codex reports a page open as an
#: action of type ``other``; the opened page is the result carrying this
#: kind of ref id. Observed on codex-cli 0.163.0.
_OPENED_REF_RE = re.compile(r"^turn\d+view\d+$")


def web_events(stdout: str) -> list[CompilationEvent]:
    """The searches and page opens in a ``codex exec --json`` event stream.

    One ``tool`` event per search (``query``) and per opened page (``url``),
    named ``web_search`` / ``web_fetch`` like the Messages API's server
    tools, so :class:`rote.compiler.api_docs.WebResearch` records them the
    same way. Lines that are not JSON, and items of any other type, are
    skipped: the stream is Codex's, and only these items are read from it.
    """
    events: list[CompilationEvent] = []
    for line in stdout.splitlines():
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or obj.get("type") != "item.completed":
            continue
        item = obj.get("item")
        if not isinstance(item, dict):
            continue
        if item.get("type") != "web_search":
            continue
        raw_action = item.get("action")
        action: dict[str, Any] = raw_action if isinstance(raw_action, dict) else {}
        query = action.get("query") or item.get("query")
        if action.get("type") == "search" and isinstance(query, str) and query:
            events.append(_web_event("web_search", query=query))
            continue
        urls = [action["url"]] if isinstance(action.get("url"), str) else []
        for result in item.get("results") or []:
            if (
                isinstance(result, dict)
                and isinstance(result.get("url"), str)
                and _OPENED_REF_RE.match(str(result.get("ref_id") or ""))
            ):
                urls.append(result["url"])
        events += [_web_event("web_fetch", url=url) for url in dict.fromkeys(urls)]
    return events


def turn_usage(stdout: str) -> dict[str, int] | None:
    """The run's token usage from a ``codex exec --json`` event stream, or None.

    Each ``turn.completed`` carries the thread's running total, not the
    turn's own: codex-rs's JSONL processor fills it from the last
    ``ThreadTokenUsageUpdated`` notification's ``total`` (rust-v0.164.0,
    ``event_processor_with_jsonl_output.rs``). So the last one is the run,
    and summing them would count every earlier turn again.

    Its ``input_tokens`` is the Responses API's, which already holds the
    cached reads and the cache writes (``input_tokens_details``); both are
    split out so no token sits in two buckets. ``output_tokens`` already
    holds the reasoning tokens. A stream with no ``turn.completed`` (a
    failed turn, a crash) reported nothing, and that is None, never zero.
    """
    usage: dict[str, Any] | None = None
    for line in stdout.splitlines():
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("type") == "turn.completed":
            reported = obj.get("usage")
            usage = reported if isinstance(reported, dict) else None
    if usage is None:
        return None

    def count(key: str) -> int | None:
        value = usage.get(key)
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    total, output = count("input_tokens"), count("output_tokens")
    if total is None or output is None:
        return None
    cache_read = count("cached_input_tokens") or 0
    cache_write = count("cache_write_input_tokens") or 0
    return {
        "input_tokens": max(0, total - cache_read - cache_write),
        "output_tokens": output,
        "cache_write_tokens": cache_write,
        "cache_read_tokens": cache_read,
    }


#: Most of a failed run's stdout an error carries. With ``--json`` stdout is
#: the whole event stream, search results included: megabytes, of which the
#: last lines say why the run ended.
DETAILS_MAX_CHARS = 4000


def _clip(text: str) -> str:
    """The tail of ``text``, at most :data:`DETAILS_MAX_CHARS` characters."""
    if len(text) <= DETAILS_MAX_CHARS:
        return text
    return (
        f"[... {len(text) - DETAILS_MAX_CHARS} earlier characters cut]\n"
        + text[-DETAILS_MAX_CHARS:]
    )


def _web_event(tool: str, url: str | None = None, query: str | None = None) -> CompilationEvent:
    target = url or query
    return CompilationEvent(
        type="tool",
        ts=time.time(),
        tool_name=tool,
        url=url,
        query=query,
        message=f"{tool} {target}",
    )


class CodexDriver(CompilerDriver):
    name: str = "codex"

    def __init__(
        self,
        model: str | None = DEFAULT_MODEL,
        sandbox: str = DEFAULT_SANDBOX,
        codex_executable: str = "codex",
        web_tools: bool = False,
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
        web_tools
            ``--backend api``: enable Codex's live web search
            (:data:`WEB_SEARCH_CONFIG`) and read its searches and page
            opens off ``--json`` (:func:`web_events`).

        Extra keyword arguments (e.g. ``max_turns``, which Codex exec has
        no equivalent for) are absorbed and ignored, per the
        driver-registry ``**kwargs`` convention.
        """
        self.model = model
        self.sandbox = sandbox
        self.codex_executable = codex_executable
        self.web_tools = web_tools

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
        compiler_skill_dir: Path,
        work_dir: Path,
        extra_instructions: str | None = None,
        on_event: EventCallback | None = None,
    ) -> DriverResult:
        """Spawn ``codex exec`` and wait for it to produce pipeline.yaml.

        Codex's stdout is read once the run ends, not streamed: for its
        token usage (:func:`turn_usage`) and, under ``web_tools``, its web
        research (:func:`web_events`). Phase events still flow: a
        :class:`~rote.compiler.events.ProgressFileWatcher` polls the
        ``progress.ndjson`` the agent writes, exactly as it does for the
        claude driver.
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

        prompt = self._build_prompt(
            skill_dir,
            compiler_skill_dir,
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
            # --json always: its turn.completed events are the only usage
            # report codex exec makes (turn_usage), and web research is
            # read from the same stream.
            args.append("--json")
            if self.web_tools:
                args += ["-c", WEB_SEARCH_CONFIG]
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
            if self.web_tools:
                # Reported once the run ends: stdout is read whole, not streamed.
                for event in web_events(stdout):
                    emit_safely(on_event, event)

            pipeline_yaml = work_dir / "pipeline.yaml"

            # The deliverable is a file on disk, not a clean exit code.
            # Mirror ClaudeDriver: check for pipeline.yaml FIRST so a
            # nonzero exit *after* the file was written (a transient error
            # on a final self-check turn) doesn't discard a real run.
            if not pipeline_yaml.is_file():
                if proc.returncode != 0:
                    raise DriverError(
                        f"codex CLI exited with code {proc.returncode}",
                        details=stderr or _clip(stdout) or "(no output)",
                    )
                raise DriverError(
                    f"codex CLI finished successfully but did not produce {pipeline_yaml}.",
                    details=_clip(stdout),
                )

            metadata = self._build_metadata(last_message_path)
            usage = turn_usage(stdout)
            if usage is not None:
                metadata.update(usage, usage_source="turn.completed", usage_complete=True)
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
        compiler_skill_dir: Path,
        work_dir: Path,
        compiler_skill_md: str,
        extra_instructions: str | None,
    ) -> str:
        """Build the single positional prompt for ``codex exec``.

        Codex has no ``--append-system-prompt`` equivalent, so the
        rote-compile rubric's SKILL.md is inlined directly into the
        prompt (its reference files are read from disk by the agent under
        the workspace-write sandbox's global read access). The task
        framing and the read/write paths mirror ClaudeDriver's prompts.
        """
        task = (
            f"You are the rote compiler. Follow the procedure and rubric below "
            f"to compile the skill at {skill_dir}.\n\n"
            f"Paths for this run:\n"
            f"  - Source skill (read): {skill_dir}\n"
            f"  - Rote-compile rubric (read): {compiler_skill_dir}\n"
            f"  - Work directory (write): {work_dir}\n\n"
            f"Begin by reading {compiler_skill_dir}/SKILL.md and its reference "
            f"files under {compiler_skill_dir}/references/, then follow the "
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
            "================== ROTE COMPILE SKILL ==================\n\n"
            f"{compiler_skill_md}"
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
