"""Anthropic API driver — in-process tool-use loop via the bare SDK.

For users who prefer (or are required to use) Anthropic API-key auth.
Runs the compiler agent in-process using the ``anthropic`` Python
SDK's Messages API with a minimal filesystem tool surface.

# Why the bare SDK and not ``claude-agent-sdk``?

Anthropic's terms of service explicitly forbid third-party agents
built on the Claude Agent SDK from using claude.ai login credentials
without prior approval. That's a policy blocker, not a technical
limitation. Since we already have a separate subscription path via
the ``claude`` driver (which spawns Claude Code directly), depending
on ``claude-agent-sdk`` would only add weight without adding value.

# Optional dependency

The ``anthropic`` package is an optional extra. Users who only use
the subprocess drivers (``claude`` / ``codex``) don't pay the import
cost::

    pip install "rote[api]"

# Filesystem tool surface

The driver implements three minimal tools for the LLM:

* ``read_file(path)`` — scoped to ``skill_dir`` and
  ``compiler_skill_dir`` (read-only).
* ``list_directory(path)`` — read-only listing in the same scope.
* ``write_file(path, content)`` — scoped to ``work_dir`` (write-only).

This is the same effective capability set the subprocess drivers give
their CLIs via ``--add-dir`` and ``--allowedTools``, just implemented
as a small Python shim instead of deferring to a heavier runtime.

Path traversal is blocked at the tool level: every path is resolved
to an absolute form and then checked against the list of allowed
roots via ``Path.relative_to``.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from rote.compiler.drivers import CompilerDriver, DriverError, DriverResult
from rote.compiler.drivers._fs_tools import (
    TURN_SNIPPET_CHARS,
    anthropic_tool_schemas,
    build_system_prompt,
    dispatch_tool,
    emit_progress_phases,
)
from rote.compiler.drivers._fs_tools import (
    handle_list_directory as _handle_list_directory,
)
from rote.compiler.drivers._fs_tools import (
    handle_read_file as _handle_read_file,
)
from rote.compiler.drivers._fs_tools import (
    handle_write_file as _handle_write_file,
)
from rote.compiler.drivers._heartbeat import await_with_heartbeat
from rote.compiler.events import (
    CompilationEvent,
    EventCallback,
    emit_safely,
    relative_display_path,
)

if TYPE_CHECKING:
    from anthropic.types import MessageParam, ToolResultBlockParam

# Re-exported filesystem-tool handlers (shared with the OpenAI driver via
# rote.compiler.drivers._fs_tools). Kept importable from this module for
# the path-jailing unit tests that predate the extraction.
__all__ = [
    "AnthropicApiDriver",
    "DEFAULT_MODEL",
    "_handle_list_directory",
    "_handle_read_file",
    "_handle_write_file",
]

# Detect the optional dep at import time so is_available() can report
# a specific "pip install rote[api]" message instead of a cryptic
# ImportError.
try:
    import anthropic  # noqa: F401

    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False


# ───────── Defaults ─────────

DEFAULT_MODEL = "claude-sonnet-4-6"
"""Default model for the in-process compiler loop.

Matches the ``ClaudeDriver`` default for consistency: Sonnet 4.6
follows structured rubrics reliably and is ~5× cheaper than Opus.
Users with complex skills can override via
``AnthropicApiDriver(model="claude-opus-4-6")`` or the CLI's
``--model`` flag.
"""

DEFAULT_MAX_ITERATIONS = 60

#: Per-turn output-token cap. This is a ceiling, not a spend — only a
#: runaway turn pays it — so it's set generously: Claude 5-family models
#: think at length, and an 8192 cap starved them into hitting
#: ``stop_reason == "max_tokens"`` mid-thought (which the loop now treats
#: as "continue", not completion — see :meth:`AnthropicApiDriver.run`).
DEFAULT_MAX_TOKENS_PER_TURN = 32768

#: Cloudflare AI Gateway's authenticated-gateway header. When BYOK
#: (bring-your-own-key) is enabled on the gateway, this header carries
#: the gateway token and Anthropic provider auth is replaced by the
#: gateway's stored key — so the SDK needs no real ANTHROPIC_API_KEY.
_GATEWAY_AUTH_HEADER = "cf-aig-authorization"

#: Placeholder passed as ``api_key`` in gateway-BYOK mode. The SDK
#: requires *some* api_key value even though provider auth is handled by
#: the gateway; this makes the requirement explicit rather than smuggling
#: in a fake-looking key.
_GATEWAY_BYOK_PLACEHOLDER = "rote-gateway-byok"


def _has_gateway_auth(default_headers: dict[str, str] | None) -> bool:
    """True when ``default_headers`` carries the AI Gateway auth header.

    Matched case-insensitively — HTTP header names are case-insensitive
    and a caller might spell it ``Cf-Aig-Authorization``.
    """
    if not default_headers:
        return False
    return any(k.lower() == _GATEWAY_AUTH_HEADER for k in default_headers)


def _assistant_snippet(content: list[Any]) -> str:
    """First ~120 chars of any assistant text block, or ``"thinking…"``.

    A turn whose content is all tool_use blocks (no prose) has nothing
    to quote, so it reports as thinking — which is exactly what the
    agent is doing when it acts without narrating.
    """
    for block in content:
        if getattr(block, "type", None) == "text":
            text = (getattr(block, "text", "") or "").strip()
            if text:
                return text[:TURN_SNIPPET_CHARS]
    return "thinking…"


# ───────── The driver ─────────


class AnthropicApiDriver(CompilerDriver):
    name: str = "api"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        max_tokens_per_turn: int = DEFAULT_MAX_TOKENS_PER_TURN,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        mcp_servers: list[dict[str, Any]] | None = None,
        **_ignored: Any,
    ) -> None:
        """
        Parameters
        ----------
        model, max_iterations, max_tokens_per_turn
            Loop knobs (see the module constants for defaults).
        **_ignored
            Cross-driver kwargs this driver doesn't implement yet are
            swallowed per the registry contract — notably ``web_tools``
            (ClaudeDriver only for now; the api driver's equivalent is
            the server-side web_search tool, a planned follow-up).
        base_url
            Override the Anthropic API base URL — point the SDK at a
            proxy or a Cloudflare AI Gateway endpoint. ``None`` lets the
            SDK use its own default (``ANTHROPIC_BASE_URL`` or the public
            API).
        default_headers
            Extra headers sent on every request. The cloud runner uses
            this to pass the gateway's ``cf-aig-authorization`` token.
            When that header is present, the driver runs in AI Gateway
            BYOK mode and needs no ``ANTHROPIC_API_KEY`` (see
            :meth:`is_available` and :meth:`run`).
        mcp_servers
            Live MCP servers whose read-only tools are exposed to the
            compiler agent as ``mcp__<server>__<tool>`` tools — each
            entry ``{"name": str, "url": str, "headers": dict[str, str]
            | None}`` (streamable HTTP only). Requires the ``mcp`` extra
            (``pip install 'rote-cli[mcp]'``). Only tools whose server
            declares ``readOnlyHint`` are exposed; a server that cannot
            be reached is skipped with a warning event, never an error.
        """
        self.model = model
        self.max_iterations = max_iterations
        self.max_tokens_per_turn = max_tokens_per_turn
        self.base_url = base_url
        self.default_headers = default_headers
        self.mcp_servers = mcp_servers

    def _client_kwargs(self) -> dict[str, Any]:
        """Assemble ``AsyncAnthropic(...)`` kwargs.

        Omits ``base_url`` / ``default_headers`` when unset so the SDK's
        own env-based defaults (``ANTHROPIC_BASE_URL``, etc.) still
        apply. In AI Gateway BYOK mode — the gateway auth header present
        but no ``ANTHROPIC_API_KEY`` — passes the placeholder key the SDK
        requires; the gateway supplies the real provider credential.
        """
        kwargs: dict[str, Any] = {}
        if self.base_url is not None:
            kwargs["base_url"] = self.base_url
        if self.default_headers is not None:
            kwargs["default_headers"] = self.default_headers
        if not os.environ.get("ANTHROPIC_API_KEY") and _has_gateway_auth(self.default_headers):
            kwargs["api_key"] = _GATEWAY_BYOK_PLACEHOLDER
        # The SDK refuses non-streaming requests whose worst-case duration
        # (scaled from max_tokens) exceeds 10 minutes UNLESS the client has
        # an explicit timeout — and a 32k thinking-friendly turn cap crosses
        # that line. Size the timeout from the cap (the SDK's own 128k-tokens
        # -per-hour model) with headroom, floored at the 10-minute default.
        kwargs["timeout"] = max(600.0, 60 * 60 * self.max_tokens_per_turn / 128_000 * 1.5)
        return kwargs

    def is_available(self) -> tuple[bool, str]:
        if not _ANTHROPIC_AVAILABLE:
            return (
                False,
                "The `anthropic` package is not installed. "
                "Install the optional extra with: `pip install rote[api]`.",
            )
        if os.environ.get("ANTHROPIC_API_KEY"):
            return (True, "")
        # AI Gateway BYOK: the gateway holds the provider key, so the
        # gateway auth header standing in for ANTHROPIC_API_KEY is enough.
        if _has_gateway_auth(self.default_headers):
            return (True, "")
        return (
            False,
            "The ANTHROPIC_API_KEY environment variable is not set. "
            "Export it or use the `claude` / `codex` driver to authenticate "
            "via your existing Claude Max/Pro or ChatGPT subscription.",
        )

    async def run(
        self,
        skill_dir: Path,
        compiler_skill_dir: Path,
        work_dir: Path,
        extra_instructions: str | None = None,
        on_event: EventCallback | None = None,
    ) -> DriverResult:
        if not _ANTHROPIC_AVAILABLE:
            raise DriverError("anthropic package is not installed. Run: pip install rote[api]")

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
        skill_md_text = skill_md.read_text(encoding="utf-8")

        # Live MCP tools list servers once here; each tool call then
        # opens a fresh connection — long-lived streamable-HTTP sessions
        # deterministically froze the asyncio loop under the hosted
        # container runtime (see rote.mcp.live_tools).
        live = self._live_mcp_tools(on_event)
        if live is not None:
            try:
                await live.__aenter__()
            except RuntimeError as e:
                # The mcp extra is missing but mcp_servers were requested:
                # compiling without the requested tools would be a silently
                # wrong run, not a degraded one.
                raise DriverError(str(e)) from e
        try:
            return await self._run_agent(
                skill_dir=skill_dir,
                compiler_skill_dir=compiler_skill_dir,
                work_dir=work_dir,
                skill_md_text=skill_md_text,
                extra_instructions=extra_instructions,
                on_event=on_event,
                live=live,
            )
        finally:
            if live is not None:
                await live.__aexit__(None, None, None)

    def _live_mcp_tools(self, on_event: EventCallback | None) -> Any:
        """A connected-tools manager for this run, or ``None`` without servers.

        Import is local so the optional-dependency convention holds: the
        subprocess-free default path never touches :mod:`rote.mcp`.
        """
        if not self.mcp_servers:
            return None
        from rote.mcp.live_tools import LiveMcpTools

        def _warn(message: str) -> None:
            emit_safely(
                on_event,
                CompilationEvent(type="warning", ts=time.time(), message=message),
            )

        return LiveMcpTools(self.mcp_servers, on_warning=_warn)

    async def _run_agent(
        self,
        *,
        skill_dir: Path,
        compiler_skill_dir: Path,
        work_dir: Path,
        skill_md_text: str,
        extra_instructions: str | None,
        on_event: EventCallback | None,
        live: Any,
    ) -> DriverResult:
        """The tool-use loop body, bracketed by ``run``'s MCP lifecycle."""
        read_roots = [skill_dir, compiler_skill_dir]
        write_root = work_dir

        system_prompt = build_system_prompt(skill_dir, compiler_skill_dir, work_dir, skill_md_text)

        client = anthropic.AsyncAnthropic(**self._client_kwargs())
        tool_schemas = anthropic_tool_schemas()
        if live is not None:
            tool_schemas = tool_schemas + live.anthropic_tool_schemas()

        task_prompt = (
            f"Compile the skill at {skill_dir}. "
            f"Begin by reading {skill_dir}/SKILL.md and the reference "
            f"files in {compiler_skill_dir}/references/, then produce "
            f"the pipeline.yaml at {work_dir}/pipeline.yaml."
        )
        if extra_instructions:
            task_prompt = f"{task_prompt}\n\n{extra_instructions}"

        messages: list[MessageParam] = [{"role": "user", "content": task_prompt}]

        total_input_tokens = 0
        total_output_tokens = 0
        last_text = ""
        completed_iterations = 0
        #: Phase events already emitted from the agent's progress.ndjson
        #: writes — the loop only fires events for lines beyond this.
        emitted_phases = 0

        for iteration in range(1, self.max_iterations + 1):
            completed_iterations = iteration

            response = await await_with_heartbeat(
                client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens_per_turn,
                    system=system_prompt,
                    # The shared builder returns plain dicts (wire-shape-
                    # neutral, shared with the OpenAI driver); the SDK
                    # validates them at runtime. Cast past the SDK's
                    # TypedDict union.
                    tools=cast("Any", tool_schemas),
                    messages=messages,
                ),
                on_event,
                f"the model (turn {iteration})",
            )

            usage = getattr(response, "usage", None)
            if usage is not None:
                total_input_tokens += getattr(usage, "input_tokens", 0) or 0
                total_output_tokens += getattr(usage, "output_tokens", 0) or 0

            # The AI Gateway unified endpoint can return `content: null` for
            # a turn that was entirely (stripped) thinking blocks — normalize
            # so nothing downstream trips over None.
            content = list(response.content or [])

            snippet = _assistant_snippet(content)
            emit_safely(
                on_event,
                CompilationEvent(
                    type="turn",
                    ts=time.time(),
                    turn=iteration,
                    tokens={"input": total_input_tokens, "output": total_output_tokens},
                    message=f"turn {iteration}: {snippet}",
                ),
            )

            # An empty assistant content list is a wire-contract violation on
            # replay, so only append turns that actually said/did something.
            if content:
                messages.append({"role": "assistant", "content": content})

            has_tool_use = any(block.type == "tool_use" for block in content)

            if not content and response.stop_reason != "max_tokens":
                # A turn with no visible output at a natural stop: through
                # the AI Gateway this is an all-thinking turn whose blocks
                # were stripped, not completion. Nudge and keep going.
                emit_safely(
                    on_event,
                    CompilationEvent(
                        type="warning",
                        ts=time.time(),
                        turn=iteration,
                        message=f"turn {iteration} returned no visible content — continuing",
                    ),
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your last turn produced no visible output. "
                            "Continue with the task — use the tools to make progress."
                        ),
                    }
                )
                continue

            if response.stop_reason == "max_tokens" and not has_tool_use:
                # The turn was cut off at the output-token limit before the
                # agent produced any action — Claude 5-family models can
                # spend a whole turn thinking. This is NOT completion: nudge
                # it to keep going and spend another iteration (the
                # max-iterations cap still bounds the loop).
                emit_safely(
                    on_event,
                    CompilationEvent(
                        type="warning",
                        ts=time.time(),
                        turn=iteration,
                        message=(
                            f"turn {iteration} truncated at the output-token limit — continuing"
                        ),
                    ),
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your last turn hit the output-token limit. "
                            "Continue exactly where you left off."
                        ),
                    }
                )
                continue

            if not has_tool_use:
                # A natural stop (end_turn, etc.) with nothing left to do.
                for block in content:
                    if block.type == "text":
                        last_text = block.text or last_text
                break

            # Dispatch tool calls
            tool_results: list[ToolResultBlockParam] = []
            for block in content:
                if block.type != "tool_use":
                    continue

                tool_name = block.name
                tool_input = cast(dict[str, Any], block.input or {})

                rel_path = relative_display_path(tool_input.get("path"), work_dir)
                emit_safely(
                    on_event,
                    CompilationEvent(
                        type="tool",
                        ts=time.time(),
                        turn=iteration,
                        tool_name=tool_name,
                        path=rel_path,
                        message=f"{tool_name} {rel_path}".rstrip() if rel_path else tool_name,
                    ),
                )

                try:
                    if live is not None and live.owns(tool_name):
                        # An exposed MCP tool: dispatch to its server and
                        # hand the result back as text. Failures fall to
                        # the error branch below — reported to the model,
                        # never allowed to crash the compile.
                        result_text = await live.call(tool_name, tool_input)
                    else:
                        result_text = dispatch_tool(tool_name, tool_input, read_roots, write_root)
                        if tool_name == "write_file":
                            # The agent announces phase transitions by writing
                            # progress.ndjson. Intercept that write and turn
                            # the new lines into phase events — the in-process
                            # analog of the subprocess drivers' file watcher.
                            emitted_phases = emit_progress_phases(
                                tool_input["path"],
                                tool_input["content"],
                                work_dir,
                                on_event,
                                emitted_phases,
                            )
                    is_error = False
                except Exception as e:
                    result_text = f"Error: {type(e).__name__}: {e}"
                    is_error = True

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_text,
                        "is_error": is_error,
                    }
                )

            messages.append({"role": "user", "content": tool_results})
        else:
            raise DriverError(
                f"Agent did not complete within {self.max_iterations} iterations.",
                details=f"Last assistant text: {last_text!r}",
            )

        pipeline_yaml = work_dir / "pipeline.yaml"
        if not pipeline_yaml.is_file():
            raise DriverError(
                f"Agent finished but did not produce {pipeline_yaml}.",
                details=f"Last assistant text: {last_text!r}",
            )

        return DriverResult(
            pipeline_yaml_path=pipeline_yaml,
            work_dir=work_dir,
            driver_name=self.name,
            metadata={
                "model": self.model,
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "iterations": completed_iterations,
            },
        )
