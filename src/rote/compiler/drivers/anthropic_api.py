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
cost. It is the ``api`` extra; :func:`rote.extras.install_hint` spells
the install command.

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

import importlib.util
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from rote.compiler.drivers import (
    CompilerDriver,
    DriverError,
    DriverResult,
    Generation,
    GenerationRequest,
)
from rote.compiler.drivers._fs_tools import (
    TURN_SNIPPET_CHARS,
    anthropic_tool_schemas,
    build_system_prompt,
    build_task_prompt,
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
from rote.compiler.drivers.claude import DEFAULT_EFFORT, EFFORT_LEVELS
from rote.compiler.events import (
    CompilationEvent,
    EventCallback,
    emit_safely,
    relative_display_path,
)
from rote.extras import install_hint
from rote.inference import structured_answer, structured_output_schema

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

# Whether the optional dep is installed, so is_available() can report a
# specific install hint instead of a cryptic ImportError. Found, not
# imported: importing anthropic costs a few hundred ms, and every rote
# command builds the parser that imports this module.
_ANTHROPIC_AVAILABLE = importlib.util.find_spec("anthropic") is not None


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

#: ``web_tools=True`` (``--backend api``): Anthropic's server-side web
#: search and web fetch, the API's equivalent of Claude Code's WebSearch /
#: WebFetch. The API runs them inside the request, so the loop never
#: dispatches them. web_fetch only opens a URL that already appeared in the
#: conversation, so the agent searches first.
#:
#: The basic versions, on purpose. The newer ones (web_search_20260209+,
#: web_fetch_20260209+) turn on "dynamic filtering", for which the API
#: provisions a code execution container on its own. In a real ops-report
#: compile the model then used that container as a second workspace: 70+
#: code_execution / bash calls inside two requests, billed as 3.69M and
#: 2.91M input tokens ($8.04 and $6.18), for a run that costs about $1 on
#: the claude driver. The basic versions run no code. ``max_content_tokens``
#: bounds what one fetched page adds to a context this loop resends every
#: turn; a vendor reference page is typically 10k to 25k tokens.
#: ``max_uses`` caps each tool per request (a search is billed per call).
#: The pinned pair is the only server tool this driver will send
#: (:func:`_check_server_tools`) or accept back (``_ALLOWED_SERVER_TOOLS``).
#: https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool
#: https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-fetch-tool
WEB_TOOL_SCHEMAS: tuple[dict[str, Any], ...] = (
    {"type": "web_search_20250305", "name": "web_search", "max_uses": 5},
    {
        "type": "web_fetch_20250910",
        "name": "web_fetch",
        "max_uses": 10,
        "max_content_tokens": 32000,
    },
)

#: Server tool names a response may carry. Anything else (``code_execution``
#: above all) means the API provisioned a tool this driver did not ask
#: for, and the run stops before it spends more.
_ALLOWED_SERVER_TOOLS = frozenset(str(t["name"]) for t in WEB_TOOL_SCHEMAS)

#: Input tokens one request may bill before the run stops. A normal turn of
#: this loop is bounded by the context window; the dynamic-filtering run
#: above billed 3.69M in ONE request because the server ran its own loop
#: inside it. The check runs on the response, so it stops the next request,
#: not the one that crossed it.
DEFAULT_MAX_INPUT_TOKENS_PER_REQUEST = 400_000


def _check_server_tools(tool_schemas: list[dict[str, Any]]) -> None:
    """Refuse any server tool (a schema with a top-level ``type``) not pinned above."""
    pinned = [dict(t) for t in WEB_TOOL_SCHEMAS]
    for schema in tool_schemas:
        if "type" in schema and schema not in pinned:
            raise DriverError(
                f"refusing to send server tool {schema.get('type')!r}: the api driver "
                f"only sends the pinned web tools ({', '.join(str(t['type']) for t in pinned)}). "
                f"Newer web tool versions provision code execution, which one real "
                f"compile used for $14 of input tokens in two requests."
            )


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


async def _streamed_message(client: Any, request_kwargs: dict[str, Any]) -> Any:
    """One streamed Messages request, returned in ``messages.create()`` shape.

    Streaming is load-bearing, not a nicety: a multi-minute non-streaming
    generation sends no bytes until it finishes, and intermediaries time
    the silent connection out — the 32k thinking-friendly turn cap made
    single turns long enough to die (a real compile's pipeline-authoring
    turn sat 720s "waiting on the model" and failed with
    APIConnectionError, and the SDK's retries hit the same wall).
    Streaming keeps bytes flowing for the whole generation.

    Timeout semantics, verified against anthropic 0.116 / httpx: the
    client ``timeout`` becomes an httpx read timeout, which on a stream
    bounds the gap between received chunks — NOT the total generation —
    so a slow-but-alive turn survives while a stalled stream still
    times out.
    """
    async with client.messages.stream(**request_kwargs) as stream:
        # Drain the events; get_final_message() then hands back the
        # accumulated Message — identical shape to messages.create(),
        # so content/stop_reason/usage handling downstream is untouched.
        async for _ in stream:
            pass
        return await stream.get_final_message()


#: The prompt cache breakpoint rote sets (5-minute TTL; reads bill at a
#: tenth of the input rate on Sonnet 4.6, writes at 1.25x).
#: https://platform.claude.com/docs/en/build-with-claude/prompt-caching
_CACHE = {"type": "ephemeral"}


def cached_tools(tool_schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The tool list with a breakpoint on its last entry: tools are the head
    of the cached prefix (``tools`` -> ``system`` -> ``messages``)."""
    if not tool_schemas:
        return tool_schemas
    return [*tool_schemas[:-1], {**tool_schemas[-1], "cache_control": _CACHE}]


def cached_system(system_prompt: str) -> list[dict[str, Any]]:
    """The system prompt as one text block carrying a breakpoint."""
    return [{"type": "text", "text": system_prompt, "cache_control": _CACHE}]


def cached_messages(messages: list[Any]) -> list[Any]:
    """``messages`` with breakpoints on its last two user turns' last blocks.

    Without them every turn re-bills the whole conversation at the full input
    rate: a measured expense-audit compile spent $3.67 over 28 turns on 948k
    uncached input tokens. The last user turn's breakpoint writes this
    request's prefix. The one before it is where the previous request wrote
    its own, so it reads that entry however wide the turn in between was. A
    breakpoint only looks back 20 positions for an earlier entry. A run of
    consecutive tool_use (or tool_result) blocks is one position, so plain
    parallel calls never reach that; alternating blocks do: text between
    calls, or web research, whose server_tool_use and result blocks
    interleave with text. Measured live: a 24-block alternating turn lost
    the previous entry under one rolling breakpoint and kept it with two.

    With the tools' and the system's that is all four breakpoints the API
    allows. The history itself is not changed (a copy is returned), so the
    breakpoints never accumulate. A trailing assistant turn (a resumed
    ``pause_turn``) keeps them on the user turns before it.
    """
    out = list(messages)
    marked = 0
    for i in range(len(out) - 1, -1, -1):
        if marked == 2:
            break
        message = out[i]
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message["content"]
        blocks = [{"type": "text", "text": content}] if isinstance(content, str) else list(content)
        if not blocks or not isinstance(blocks[-1], dict):
            continue
        blocks[-1] = {**blocks[-1], "cache_control": _CACHE}
        out[i] = {**message, "content": blocks}
        marked += 1
    return out


def usage_buckets(usage: Any) -> dict[str, int]:
    """One response's four token buckets (``input`` is after the last breakpoint)."""
    return {
        "input": getattr(usage, "input_tokens", 0) or 0,
        "output": getattr(usage, "output_tokens", 0) or 0,
        "cache_write": getattr(usage, "cache_creation_input_tokens", 0) or 0,
        "cache_read": getattr(usage, "cache_read_input_tokens", 0) or 0,
    }


def web_search_usd(model: str, searches: int) -> float | None:
    """What ``searches`` server-side web searches cost on ``model``, or ``None``
    when no live source prices them (see ``PricingCatalog.web_search_usd_per_request``)."""
    from rote.eval.pricing import PricingError, fetch_catalog

    if not searches:
        return 0.0
    try:
        rate = fetch_catalog(provider="anthropic").web_search_usd_per_request(model)
    except PricingError:
        return None
    return round(rate * searches, 6)


def run_cost_usd(model: str, totals: dict[str, int], web_searches: int = 0) -> float | None:
    """The run's cost from the live price catalog, or ``None`` when it cannot be priced.

    Tokens in all four buckets, plus each server-side web search, which
    bills per call on top of tokens (web fetch bills tokens only).
    ``None`` rather than a guess, as everywhere in rote: no catalog
    (offline), an unlisted model, cache tokens with no published cache
    rate, or searches with no live search price. The model id is
    normalized by the catalog (a gateway's ``anthropic/claude-haiku-5.5``
    prices as ``claude-haiku-5-5``).
    """
    from rote.eval.pricing import PricingError, fetch_catalog

    search_usd = web_search_usd(model, web_searches)
    if search_usd is None:
        return None
    try:
        rates = fetch_catalog(provider="anthropic").rates_for(model)
    except PricingError:
        return None
    if rates is None:
        return None
    inp, out, cache_read, cache_write = rates
    if (totals["cache_read"] and cache_read is None) or (
        totals["cache_write"] and cache_write is None
    ):
        return None
    cost = (
        totals["input"] * inp
        + totals["output"] * out
        + totals["cache_read"] * (cache_read or 0.0)
        + totals["cache_write"] * (cache_write or 0.0)
    ) / 1e6
    return round(cost + search_usd, 6)


def _server_tool_event(block: Any, turn: int) -> CompilationEvent:
    """A ``tool`` event for a ``server_tool_use`` block: a fetch's url, a search's query."""
    name = str(getattr(block, "name", "") or "server_tool")
    tool_input = getattr(block, "input", None)
    tool_input = tool_input if isinstance(tool_input, dict) else {}
    url = tool_input.get("url") if name == "web_fetch" else None
    query = tool_input.get("query") if name == "web_search" else None
    url = url if isinstance(url, str) else None
    query = query if isinstance(query, str) else None
    target = url or query
    return CompilationEvent(
        type="tool",
        ts=time.time(),
        turn=turn,
        tool_name=name,
        url=url,
        query=query,
        message=f"{name} {target}" if target else name,
    )


# ───────── The driver ─────────


class AnthropicApiDriver(CompilerDriver):
    name: str = "api"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        max_tokens_per_turn: int = DEFAULT_MAX_TOKENS_PER_TURN,
        max_input_tokens_per_request: int = DEFAULT_MAX_INPUT_TOKENS_PER_REQUEST,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        mcp_servers: list[dict[str, Any]] | None = None,
        web_tools: bool = False,
        effort: str = DEFAULT_EFFORT,
        **_ignored: Any,
    ) -> None:
        """
        Parameters
        ----------
        model, max_iterations, max_tokens_per_turn
            Loop knobs (see the module constants for defaults).
        **_ignored
            Cross-driver kwargs this driver doesn't implement are
            swallowed per the registry contract.
        web_tools
            ``--backend api``: add the server-side web search and web
            fetch tools (:data:`WEB_TOOL_SCHEMAS`).
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
            (see :mod:`rote.extras`). Only tools whose server
            declares ``readOnlyHint`` are exposed; a server that cannot
            be reached is skipped with a warning event, never an error.
        effort
            ``rote compile --effort``, sent as ``output_config.effort`` on
            :meth:`generate` calls. Unset, the API runs at its own default,
            and a measured ops-report compile's pieces spent 12k to 21k
            output tokens each, against 2k to 3k at ``low``, the claude
            driver's default (:data:`rote.compiler.drivers.claude.DEFAULT_EFFORT`).
        """
        if effort not in EFFORT_LEVELS:
            raise ValueError(f"effort must be one of {', '.join(EFFORT_LEVELS)}, got {effort!r}")
        self.effort = effort
        self.model = model
        self.max_iterations = max_iterations
        self.max_tokens_per_turn = max_tokens_per_turn
        self.max_input_tokens_per_request = max_input_tokens_per_request
        self.base_url = base_url
        self.default_headers = default_headers
        self.mcp_servers = mcp_servers
        self.web_tools = web_tools

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
                f"Install the optional extra with: `{install_hint('api')}`.",
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

    async def generate(self, request: GenerationRequest) -> Generation:
        """One schema-locked call: structured outputs, the answer as JSON text.

        ``output_config.format`` constrains decoding to ``request.schema``.
        A forced tool would do the same on older models, and current ones
        (Sonnet 5.5, Haiku 5.5) reject ``tool_choice`` ``tool`` and ``any``
        outright with a 400. The shared prefix is the system prompt, marked
        for the prompt cache, so the calls of one compile after the first
        read it at the cache rate, and usage keeps all four buckets for that
        reason. No ``temperature``: current models reject it. The API
        reports no cost, so ``cost_usd`` is the call's own model's list price
        on its four buckets (:func:`run_cost_usd`), or ``None`` when that
        cannot be priced.
        """
        if not _ANTHROPIC_AVAILABLE:
            raise DriverError(f"anthropic package is not installed. Run: {install_hint('api')}")
        try:
            # The one rewrite into the subset structured outputs accepts
            # (shared with emitted judges): it refuses an open object rather
            # than closing it.
            schema = structured_output_schema(request.schema, node_id=request.label)
        except RuntimeError as e:
            raise DriverError(f"cannot lock the answer for {request.label}: {e}") from e
        started = time.monotonic()
        import anthropic

        client = anthropic.AsyncAnthropic(**self._client_kwargs())
        try:
            message = await _streamed_message(
                client,
                {
                    "model": request.model,
                    "max_tokens": self.max_tokens_per_turn,
                    "system": cached_system(request.system),
                    "messages": [{"role": "user", "content": request.prompt}],
                    "output_config": {
                        "format": {"type": "json_schema", "schema": schema},
                        "effort": self.effort,
                    },
                },
            )
        except Exception as e:  # noqa: BLE001 (any SDK failure is this call's)
            raise DriverError(f"Anthropic API call failed for {request.label}: {e}") from e
        finally:
            await client.close()
        totals = usage_buckets(message.usage)
        metadata = {f"{bucket}_tokens": count for bucket, count in totals.items()}
        try:
            payload = structured_answer(message, node_id=request.label)
        except RuntimeError as e:  # truncated, refused, or not JSON: still a paid call
            raise DriverError(f"Anthropic API returned no answer: {e}", metadata=metadata) from e
        if not isinstance(payload, dict):
            raise DriverError(
                f"Anthropic API returned no answer for {request.label}: a JSON "
                f"{type(payload).__name__}, not an object",
                metadata=metadata,
            )
        return Generation(
            payload=payload,
            model=str(message.model),
            cost_usd=run_cost_usd(str(message.model), totals),
            duration_ms=int((time.monotonic() - started) * 1000),
            **metadata,
        )

    async def run(
        self,
        skill_dir: Path,
        compiler_skill_dir: Path,
        work_dir: Path,
        extra_instructions: str | None = None,
        on_event: EventCallback | None = None,
        repair: Callable[[Path], str | None] | None = None,
    ) -> DriverResult:
        """Run the compiler agent (see :class:`CompilerDriver`).

        ``repair`` is the orchestrator's validation callback: called with
        the produced pipeline.yaml at each natural stop; a returned string
        is appended to the SAME conversation as one user turn and the
        loop continues, ``None`` accepts the deliverable. Passed only by
        orchestrators that saw it in this signature.
        """
        if not _ANTHROPIC_AVAILABLE:
            raise DriverError(f"anthropic package is not installed. Run: {install_hint('api')}")

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
                # fastmcp is missing but mcp_servers were requested:
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
                repair=repair,
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
        repair: Callable[[Path], str | None] | None,
    ) -> DriverResult:
        """The tool-use loop body, bracketed by ``run``'s MCP lifecycle."""
        # The work directory too: System-1 writes the skeleton pipeline.yaml
        # and system1/brief.md there before the agent starts, and an agent
        # that cannot read them writes its own plan beside System-1's.
        read_roots = [skill_dir, compiler_skill_dir, work_dir]
        write_root = work_dir

        system_prompt = build_system_prompt(skill_dir, compiler_skill_dir, work_dir, skill_md_text)

        import anthropic

        client = anthropic.AsyncAnthropic(**self._client_kwargs())
        tool_schemas = anthropic_tool_schemas()
        if live is not None:
            tool_schemas = tool_schemas + live.anthropic_tool_schemas()
        if self.web_tools:
            tool_schemas = tool_schemas + [dict(t) for t in WEB_TOOL_SCHEMAS]
        _check_server_tools(tool_schemas)

        task_prompt = build_task_prompt(skill_dir, compiler_skill_dir, work_dir)
        if extra_instructions:
            task_prompt = f"{task_prompt}\n\n{extra_instructions}"

        messages: list[MessageParam] = [{"role": "user", "content": task_prompt}]

        totals = {"input": 0, "output": 0, "cache_write": 0, "cache_read": 0}
        tools_param = cached_tools(tool_schemas)
        system_param = cached_system(system_prompt)
        #: Server tool calls: searches bill per call on top of tokens; fetches are free.
        server_tools = {"web_search": 0, "web_fetch": 0}
        last_text = ""
        completed_iterations = 0
        #: Repair turns the orchestrator's verifier bounced back (the claude
        #: driver reports the same key from its --resume invocations).
        repair_rounds = 0
        #: Phase events already emitted from the agent's progress.ndjson
        #: writes — the loop only fires events for lines beyond this.
        emitted_phases = 0

        for iteration in range(1, self.max_iterations + 1):
            completed_iterations = iteration

            # Streamed, never create(): long turns die on silent
            # connections (see _streamed_message). The heartbeat around
            # the whole stream consumption now only fires when the
            # stream itself stalls >120s — the new hang signal, not a
            # healthy slow turn.
            response = await await_with_heartbeat(
                _streamed_message(
                    client,
                    {
                        "model": self.model,
                        "max_tokens": self.max_tokens_per_turn,
                        # Prompt-cached: tools, system and the latest user
                        # turn each carry a breakpoint (see cached_messages).
                        "system": system_param,
                        # The shared builder returns plain dicts
                        # (wire-shape-neutral, shared with the OpenAI
                        # driver); the SDK validates them at runtime.
                        "tools": tools_param,
                        "messages": cached_messages(messages),
                    },
                ),
                on_event,
                f"the model (turn {iteration})",
            )

            usage = getattr(response, "usage", None)
            if usage is not None:
                for bucket, n in usage_buckets(usage).items():
                    totals[bucket] += n
                server = getattr(usage, "server_tool_use", None)
                server_tools["web_search"] += getattr(server, "web_search_requests", 0) or 0
                server_tools["web_fetch"] += getattr(server, "web_fetch_requests", 0) or 0
                billed = sum(
                    getattr(usage, key, 0) or 0
                    for key in (
                        "input_tokens",
                        "cache_creation_input_tokens",
                        "cache_read_input_tokens",
                    )
                )
                if billed > self.max_input_tokens_per_request:
                    raise DriverError(
                        f"turn {iteration} billed {billed} input tokens in one request, "
                        f"over the {self.max_input_tokens_per_request} ceiling; stopping "
                        f"before the next one. A server-side loop inside a request is "
                        f"the usual cause.",
                        metadata=self._metadata(
                            totals, server_tools, completed_iterations, repair_rounds
                        ),
                    )

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
                    tokens=dict(totals),
                    message=f"turn {iteration}: {snippet}",
                ),
            )

            # An empty assistant content list is a wire-contract violation on
            # replay, so only append turns that actually said/did something.
            if content:
                messages.append({"role": "assistant", "content": content})

            # The API ran these itself (web search / fetch); they are only
            # reported, and any other server tool stops the run.
            for block in content:
                if block.type == "server_tool_use":
                    emit_safely(on_event, _server_tool_event(block, iteration))
                    if block.name not in _ALLOWED_SERVER_TOOLS:
                        raise DriverError(
                            f"the API ran server tool {block.name!r}, which the api "
                            f"driver did not ask for; stopping before it spends more.",
                            metadata=self._metadata(
                                totals, server_tools, completed_iterations, repair_rounds
                            ),
                        )

            if response.stop_reason == "pause_turn":
                # A long server-tool turn the API paused: send the paused
                # assistant turn back unchanged, with nothing after it, and
                # the API resumes it. Not a stop.
                continue

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
                natural_stop_pipeline = work_dir / "pipeline.yaml"
                if repair is not None and natural_stop_pipeline.is_file():
                    # Let the orchestrator's validator bounce the
                    # deliverable back into the SAME conversation —
                    # pydantic errors the model can trivially fix must
                    # not fail a paid run one-shot. Repair turns spend
                    # the same iteration budget, so the loop stays
                    # bounded.
                    repair_instruction = repair(natural_stop_pipeline)
                    if repair_instruction is not None:
                        repair_rounds += 1
                        messages.append({"role": "user", "content": repair_instruction})
                        continue
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
                metadata=self._metadata(totals, server_tools, completed_iterations, repair_rounds),
            )

        pipeline_yaml = work_dir / "pipeline.yaml"
        if not pipeline_yaml.is_file():
            raise DriverError(
                f"Agent finished but did not produce {pipeline_yaml}.",
                details=f"Last assistant text: {last_text!r}",
                metadata=self._metadata(totals, server_tools, completed_iterations, repair_rounds),
            )

        return DriverResult(
            pipeline_yaml_path=pipeline_yaml,
            work_dir=work_dir,
            driver_name=self.name,
            metadata=self._metadata(totals, server_tools, completed_iterations, repair_rounds),
        )

    def _metadata(
        self,
        totals: dict[str, int],
        server_tools: dict[str, int],
        iterations: int,
        repair_rounds: int,
    ) -> dict[str, Any]:
        """Usage in the keys every driver reports, priced into ``cost_usd``.

        The cost lands in compile-metrics.json, which is what ``rote bench
        --budget`` sums: without it an api-driver compile counted as $0.
        """
        return {
            "model": self.model,
            "input_tokens": totals["input"],
            "output_tokens": totals["output"],
            "cache_write_tokens": totals["cache_write"],
            "cache_read_tokens": totals["cache_read"],
            "cost_usd": run_cost_usd(self.model, totals, server_tools["web_search"]),
            "iterations": iterations,
            "repair_rounds": repair_rounds,
            **(
                {
                    "web_search_requests": server_tools["web_search"],
                    "web_fetch_requests": server_tools["web_fetch"],
                    "web_search_usd": web_search_usd(self.model, server_tools["web_search"]),
                }
                if self.web_tools
                else {}
            ),
        }
