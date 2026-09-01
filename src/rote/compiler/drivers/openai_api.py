"""OpenAI-compatible API driver — in-process tool-use loop via the OpenAI SDK.

The counterpart to :mod:`rote.compiler.drivers.anthropic_api` for models
that speak the OpenAI ``chat/completions`` wire shape rather than the
Anthropic Messages API. The cloud platform routes non-Anthropic compiler
models (e.g. ``openai/gpt-5.5``, ``@cf/zai-org/glm-5.2``,
``moonshotai/kimi-k2.6``) through Cloudflare AI Gateway's OpenAI-compatible
endpoint with Unified Billing, which the Anthropic SDK can't call.

Both drivers share the same filesystem tool surface — the three tools,
path jailing, read caps, system prompt, and ``progress.ndjson`` phase
interception all live in :mod:`rote.compiler.drivers._fs_tools`. This
module owns only the OpenAI-specific protocol glue: function-calling tool
declarations, the ``assistant``/``tool`` message shapes, ``usage``
accounting, and ``max_completion_tokens``.

# Optional dependency

The ``openai`` package is an optional extra so users of the subprocess
drivers don't pay the import cost::

    pip install "rote[openai-api]"

# No default model

Unlike the Anthropic driver, there is no sensible universal default
model — the endpoint serves many vendors. The model is a required
constructor argument; the CLI (``--model``) and the cloud runner always
supply one.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from rote.compiler.drivers import CompilerDriver, DriverError, DriverResult
from rote.compiler.drivers._fs_tools import (
    TURN_SNIPPET_CHARS,
    build_system_prompt,
    dispatch_tool,
    emit_progress_phases,
    openai_tool_schemas,
)
from rote.compiler.events import (
    CompilationEvent,
    EventCallback,
    emit_safely,
    relative_display_path,
)

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam

# Detect the optional dep at import time so is_available() can report a
# specific "pip install rote[openai-api]" message instead of a cryptic
# ImportError.
try:
    import openai  # noqa: F401

    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False


# ───────── Defaults ─────────

DEFAULT_MAX_ITERATIONS = 60

#: Per-turn output-token cap, sent as ``max_completion_tokens``. A ceiling,
#: not a spend — only a runaway turn pays it — so it's generous: reasoning
#: models think at length, and a low cap truncates a turn mid-thought
#: (``finish_reason == "length"``, which the loop treats as "continue", not
#: completion — see :meth:`OpenAIApiDriver.run`).
DEFAULT_MAX_TOKENS_PER_TURN = 32768

#: Header names that authenticate a Cloudflare AI Gateway call. Either one
#: standing in for ``OPENAI_API_KEY`` means the driver is running against
#: the gateway (Unified Billing supplies the provider credential): the
#: gateway's own ``cf-aig-authorization``, or the ``Authorization`` bearer
#: the compat route reads.
_GATEWAY_AUTH_HEADERS = ("cf-aig-authorization", "authorization")

#: Placeholder passed as ``api_key`` in gateway mode. The SDK requires
#: *some* api_key value even though provider auth is handled by the
#: gateway; this makes the requirement explicit rather than smuggling in a
#: fake-looking key.
_GATEWAY_PLACEHOLDER = "rote-gateway"


def _has_gateway_auth(default_headers: dict[str, str] | None) -> bool:
    """True when ``default_headers`` carries a gateway auth header.

    Matched case-insensitively — HTTP header names are case-insensitive
    and a caller might spell it ``Cf-Aig-Authorization`` or ``authorization``.
    """
    if not default_headers:
        return False
    lowered = {k.lower() for k in default_headers}
    return any(h in lowered for h in _GATEWAY_AUTH_HEADERS)


def _assistant_snippet(content: str | None) -> str:
    """First ~120 chars of the assistant's text, or ``"thinking…"``.

    An OpenAI ``chat/completions`` message carries content as a plain
    string (or ``None`` when the turn is pure tool calls / reasoning),
    unlike Anthropic's typed block list — so the snippet is trivial here.
    """
    text = (content or "").strip()
    return text[:TURN_SNIPPET_CHARS] if text else "thinking…"


# ───────── The driver ─────────


class OpenAIApiDriver(CompilerDriver):
    name: str = "openai-api"

    def __init__(
        self,
        model: str,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        max_tokens_per_turn: int = DEFAULT_MAX_TOKENS_PER_TURN,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        mcp_servers: list[dict[str, Any]] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        model
            REQUIRED — the model id to send (e.g. ``openai/gpt-5.5``,
            ``@cf/zai-org/glm-5.2``, ``moonshotai/kimi-k2.6``). There is
            no default: the endpoint serves many vendors and no single
            id is universally right.
        max_iterations, max_tokens_per_turn
            Loop knobs. ``max_tokens_per_turn`` is sent as
            ``max_completion_tokens`` (reasoning models reject the legacy
            ``max_tokens``).
        base_url
            Override the API base URL — point the SDK at a proxy or a
            Cloudflare AI Gateway OpenAI-compatible endpoint. ``None`` lets
            the SDK use its own default (``OPENAI_BASE_URL`` or the public
            API).
        default_headers
            Extra headers sent on every request. The cloud runner passes
            the gateway's ``cf-aig-authorization`` / ``Authorization``
            token here; when either is present the driver runs in gateway
            mode and needs no ``OPENAI_API_KEY`` (see :meth:`is_available`).
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
        """Assemble ``AsyncOpenAI(...)`` kwargs.

        Omits ``base_url`` / ``default_headers`` when unset so the SDK's
        own env-based defaults (``OPENAI_BASE_URL``, etc.) still apply. In
        gateway mode — a gateway auth header present but no
        ``OPENAI_API_KEY`` — passes the placeholder key the SDK requires;
        the gateway supplies the real provider credential.
        """
        kwargs: dict[str, Any] = {}
        if self.base_url is not None:
            kwargs["base_url"] = self.base_url
        if self.default_headers is not None:
            kwargs["default_headers"] = self.default_headers
        if not os.environ.get("OPENAI_API_KEY") and _has_gateway_auth(self.default_headers):
            kwargs["api_key"] = _GATEWAY_PLACEHOLDER
        # Long reasoning turns at the 32k cap can outlast the SDK's default
        # 10-minute timeout; scale it with the cap (same heuristic as the
        # anthropic driver), floored at that default.
        kwargs["timeout"] = max(600.0, 60 * 60 * self.max_tokens_per_turn / 128_000 * 1.5)
        return kwargs

    def is_available(self) -> tuple[bool, str]:
        if not _OPENAI_AVAILABLE:
            return (
                False,
                "The `openai` package is not installed. "
                "Install the optional extra with: `pip install rote[openai-api]`.",
            )
        if os.environ.get("OPENAI_API_KEY"):
            return (True, "")
        # Gateway mode: the gateway holds the provider key, so a gateway
        # auth header standing in for OPENAI_API_KEY is enough.
        if _has_gateway_auth(self.default_headers):
            return (True, "")
        return (
            False,
            "The OPENAI_API_KEY environment variable is not set. "
            "Export it, or route through a Cloudflare AI Gateway by passing "
            "a cf-aig-authorization / Authorization header.",
        )

    async def run(
        self,
        skill_dir: Path,
        compiler_skill_dir: Path,
        work_dir: Path,
        extra_instructions: str | None = None,
        on_event: EventCallback | None = None,
    ) -> DriverResult:
        if not _OPENAI_AVAILABLE:
            raise DriverError("openai package is not installed. Run: pip install rote[openai-api]")

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

        # Live MCP tools stay connected across the whole agent loop —
        # reconnecting per call would redo the HTTP handshake (and lose
        # any server-side session state) on every tool invocation.
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
        """A connected-tools manager for this run, or ``None`` without servers."""
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

        client = openai.AsyncOpenAI(**self._client_kwargs())
        tool_schemas = openai_tool_schemas()
        if live is not None:
            tool_schemas = tool_schemas + live.openai_tool_schemas()

        task_prompt = (
            f"Compile the skill at {skill_dir}. "
            f"Begin by reading {skill_dir}/SKILL.md and the reference "
            f"files in {compiler_skill_dir}/references/, then produce "
            f"the pipeline.yaml at {work_dir}/pipeline.yaml."
        )
        if extra_instructions:
            task_prompt = f"{task_prompt}\n\n{extra_instructions}"

        # OpenAI carries the system prompt as the first message rather than
        # a separate top-level parameter.
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_prompt},
        ]

        total_input_tokens = 0
        total_output_tokens = 0
        last_text = ""
        completed_iterations = 0
        #: Phase events already emitted from the agent's progress.ndjson
        #: writes — the loop only fires events for lines beyond this.
        emitted_phases = 0

        for iteration in range(1, self.max_iterations + 1):
            completed_iterations = iteration

            response = await client.chat.completions.create(
                model=self.model,
                max_completion_tokens=self.max_tokens_per_turn,
                messages=messages,
                # The shared builder returns plain dicts (wire-shape-neutral,
                # shared with the Anthropic driver); the SDK validates them
                # at runtime. Cast past the SDK's TypedDict union.
                tools=cast("Any", tool_schemas),
            )

            usage = getattr(response, "usage", None)
            if usage is not None:
                total_input_tokens += getattr(usage, "prompt_tokens", 0) or 0
                total_output_tokens += getattr(usage, "completion_tokens", 0) or 0

            if not response.choices:
                raise DriverError(
                    "OpenAI-compatible endpoint returned no choices.",
                    details=f"turn {iteration}",
                )
            choice = response.choices[0]
            message = choice.message
            content = getattr(message, "content", None)
            tool_calls = getattr(message, "tool_calls", None) or []
            finish_reason = getattr(choice, "finish_reason", None)

            emit_safely(
                on_event,
                CompilationEvent(
                    type="turn",
                    ts=time.time(),
                    turn=iteration,
                    tokens={"input": total_input_tokens, "output": total_output_tokens},
                    message=f"turn {iteration}: {_assistant_snippet(content)}",
                ),
            )

            # Rebuild the assistant turn as a plain dict from just content +
            # tool_calls. This drops any reasoning/thinking fields some
            # models (kimi, glm) attach — we don't echo them back, and the
            # API rejects unknown fields on replay. Coerce a null content to
            # "" on non-tool turns: some endpoints reject an assistant
            # message with both content null and no tool_calls on replay.
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": content if tool_calls else (content or ""),
            }
            if tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ]
            messages.append(cast("ChatCompletionMessageParam", assistant_msg))

            if finish_reason == "length" and not tool_calls:
                # The turn was cut off at the output-token limit before the
                # agent produced any action (reasoning models can spend a
                # whole turn thinking). NOT completion: nudge it to keep
                # going and spend another iteration (the max-iterations cap
                # still bounds the loop).
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

            if not tool_calls:
                # A natural stop (finish_reason "stop", no tool calls) → done.
                last_text = content or last_text
                break

            for tc in tool_calls:
                tool_name = tc.function.name
                raw_arguments = tc.function.arguments or ""
                try:
                    args = json.loads(raw_arguments) if raw_arguments else {}
                    parse_error: str | None = None
                except json.JSONDecodeError as e:
                    args = None
                    parse_error = f"could not parse tool arguments as JSON: {e}"

                arg_path = args.get("path") if isinstance(args, dict) else None
                rel_path = relative_display_path(arg_path, work_dir)
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

                if parse_error is not None:
                    # Malformed arguments must not crash the loop — report
                    # them back so the model can retry.
                    result_text = f"Error: {parse_error}"
                else:
                    typed_args = cast("dict[str, Any]", args)
                    try:
                        if live is not None and live.owns(tool_name):
                            # An exposed MCP tool: dispatch to its server
                            # and hand the result back as text. Failures
                            # fall to the error branch below — reported to
                            # the model, never allowed to crash the compile.
                            result_text = await live.call(tool_name, typed_args)
                        else:
                            result_text = dispatch_tool(
                                tool_name, typed_args, read_roots, write_root
                            )
                            if tool_name == "write_file":
                                # The agent announces phase transitions by
                                # writing progress.ndjson; intercept that write
                                # and turn the new lines into phase events.
                                emitted_phases = emit_progress_phases(
                                    typed_args["path"],
                                    typed_args["content"],
                                    work_dir,
                                    on_event,
                                    emitted_phases,
                                )
                    except Exception as e:
                        result_text = f"Error: {type(e).__name__}: {e}"

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_text,
                    }
                )
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
