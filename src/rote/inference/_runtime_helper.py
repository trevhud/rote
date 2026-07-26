"""Judge inference helper — emitted verbatim into generated apps.

This module is special: rote emits its *source text* into every Python
app that contains an ``llm_judge`` node (as
``signatures/_rote_inference.py``), so generated code stays standalone
while sharing one tested implementation. It therefore obeys two hard
rules, exactly like :mod:`rote.mcp._runtime_helper`: **stdlib-only
imports at module level** (vendor SDKs only inside functions), and **no
imports from rote**.

What it decides
---------------

An emitted judge knows *what* to ask (the prompt and the output schema
come from the IR) but not *who pays for the call*. That is a deployment
fact about one machine on one day, so it is resolved here at runtime,
never baked into ``pipeline.yaml``. Three providers:

======================  ===========  ==========================  ============
Provider                Transport    Billing                     Runtimes
======================  ===========  ==========================  ============
``claude-cli``          subprocess   the user's Claude subscription  local only
``api``                 vendor SDK   the user's API key           all
``rote-cloud``          vendor SDK   the tenant's rote account    all
======================  ===========  ==========================  ============

``api`` and ``rote-cloud`` are **not two implementations** — they are
the same SDK call at a different ``(base_url, auth, model)``. That was
established empirically: pointing a judge at a Cloudflare AI Gateway
with Unified Billing (Cloudflare holds the provider credential and
bills the account) needed zero code, only environment. So a provider is
a *resolver*, and only two transports exist to test.

Selection, first match wins::

    ROTE_INFERENCE_<NODE_ID> > ROTE_INFERENCE > auto-detect

Auto-detect is ordered cheapest-to-the-user: a subscription the user
already pays for, then their own key at provider rates, then rote cloud.
It is one order everywhere — a deployed image simply has no ``claude``
binary, so the first candidate drops out on its own rather than needing
a separate "deployed" branch.

Why every failure is a plain ``RuntimeError``
---------------------------------------------

A durable runtime persists a failed step by serializing the exception,
and the vendor SDKs' error classes take keyword-only ``response`` /
``body`` arguments that ``BaseException.__reduce__`` cannot rebuild —
so the real status and body get replaced by a bare ``TypeError`` at the
join. This module therefore re-raises through :func:`_durable_error`,
and deliberately defines **no exception subclass of its own**: a custom
class only unpickles where its module is importable, and the observer
of a failed step is not always the process that raised it (the fan_out
enqueue path is the worked example). A plain ``RuntimeError`` carrying
the detail in its message is reconstructable everywhere.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

#: Every provider.
PROVIDERS: tuple[str, ...] = ("claude-cli", "api", "rote-cloud")

#: Auto-detect order for a **judge** — one structured call, no tools.
#: A key serves it first because the CLI's per-call overhead (process
#: spawn plus the agent harness) dominates a single request: measured
#: 3.6–5.0s per call against ~1.3s through an SDK endpoint, for
#: identical output. ``claude-cli`` still precedes ``rote-cloud`` so a
#: subscriber with no API key runs their pipeline for free rather than
#: falling off a cliff at the moment graduation is supposed to pay off.
_JUDGE_ORDER: tuple[str, ...] = ("api", "claude-cli", "rote-cloud")

#: Auto-detect order for an **agent loop** — the inverse, and for the
#: same reason read the other way. A bounded multi-turn loop amortizes
#: the harness cost over every iteration instead of paying it once per
#: answer, and agent loops are the expensive part of a pipeline, so the
#: subscription is where they belong when one is available.
_AGENT_ORDER: tuple[str, ...] = ("claude-cli", "api", "rote-cloud")

#: Wall-clock ceiling for one judge call. A judge that has not answered
#: in five minutes is wedged, not slow; the durable runtime's own retry
#: policy is the recovery path.
_DEFAULT_TIMEOUT_S = 300.0

#: Agent loops run many turns and may call slow tools, so they get a
#: larger budget than a single judge call.
_DEFAULT_AGENT_TIMEOUT_S = 1800.0

#: Replaces Claude Code's default system prompt on the ``claude-cli``
#: judge path. Without it every judge call re-sends the full coding-agent
#: preamble — measured at ~37k cache-creation tokens and 11.5s versus
#: ~740 tokens and 7.4s with this one line.
_CLI_SYSTEM_PROMPT = (
    "You are a precise evaluator. Answer only by populating the "
    "structured output schema. Do not explain your process."
)

#: System prompt for an ``agent_loop``. Unlike the judge path this one
#: keeps tools — that is the point — but it still replaces Claude Code's
#: coding-agent preamble, because the agent is executing one pipeline
#: step, not operating a developer's machine.
_AGENT_SYSTEM_PROMPT = (
    "You are executing one bounded step of an automated pipeline. Use the "
    "tools you are given to accomplish the task, then report the result. "
    "You are running unattended: never ask a question, never wait for "
    "confirmation, and stop as soon as the task is done."
)


# ───────── subscription environment ─────────


def build_subscription_env() -> dict[str, str]:
    """Child environment for a subscription-billed ``claude -p`` spawn.

    Critical: scrub ``ANTHROPIC_API_KEY`` and ``ANTHROPIC_AUTH_TOKEN``.
    In ``claude -p`` mode these env vars always win over an active
    OAuth session, which defeats the whole point of the subscription
    path. Callers who want API-key auth should use the ``api`` provider
    (or, for the graduator, ``AnthropicApiDriver``) — this helper is
    specifically about reusing the user's Claude Max/Pro subscription.

    Lives here rather than beside the graduator's Claude driver because
    emitted apps need the identical rule and cannot import rote: one
    definition, copied verbatim, instead of two that drift.
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


# ───────── rote cloud credential (read-only mirror of rote.cloud_auth) ─────────


def _cloud_credential() -> tuple[str, str] | None:
    """``(base_url, token)`` for rote cloud, or None when logged out.

    Reads the same store ``rote login`` writes
    (``~/.local/share/rote/cloud.json``); env vars win, matching the
    CLI's own resolution order. Never refreshes and never prompts —
    emitted workflows run unattended.
    """
    url = os.environ.get("ROTE_CLOUD_URL")
    token = os.environ.get("ROTE_CLOUD_TOKEN")
    if url and token:
        return url, token

    override = os.environ.get("ROTE_CLOUD_CRED_PATH")
    if override:
        path = Path(override)
    else:
        xdg = os.environ.get("XDG_DATA_HOME")
        base = Path(xdg) if xdg else Path.home() / ".local" / "share"
        path = base / "rote" / "cloud.json"
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, ValueError):
        return None
    stored_url = doc.get("base_url") or doc.get("url")
    stored_token = doc.get("token")
    if not stored_url or not stored_token:
        return None
    return str(url or stored_url), str(token or stored_token)


# ───────── provider selection ─────────


def provider_availability(
    provider: str,
    client: str,
    *,
    base_url: str | None = None,
    workload: str = "judge",
    local_tools: bool = False,
) -> tuple[bool, str]:
    """``(available, reason)`` for one provider serving one node.

    ``reason`` is the setup step when unavailable, so a failure can list
    what to do for every provider rather than naming one.
    """
    if provider == "claude-cli":
        if client != "anthropic":
            return False, (
                f"a Claude subscription cannot serve a {client!r}-client {workload}; "
                f"set an API key or use rote cloud"
            )
        if local_tools:
            # `claude -p` reaches tools only over MCP, so an in-process
            # Python callable (a loop_body sub-node) cannot be handed to
            # it without standing up an MCP bridge. The SDK tool runner
            # takes callables directly, so that lane serves these loops.
            return False, (
                "this agent loop drives loop_body sub-nodes as in-process "
                "callables, which the CLI can only reach over MCP; the SDK "
                "tool runner binds them directly"
            )
        if base_url:
            return False, (
                f"an explicit endpoint is configured for this {workload} "
                f"(base_url / ROTE_BASE_URL_<NODE_ID>), which the CLI cannot honor"
            )
        if shutil.which("claude") is None:
            return (
                False,
                "the `claude` CLI is not on PATH — install Claude Code and run `claude login`",
            )
        return True, "claude -p, billed to the Claude subscription"
    if provider == "api":
        if client == "openai":
            if os.environ.get("OPENAI_API_KEY"):
                return True, "OPENAI_API_KEY"
            return False, "set OPENAI_API_KEY"
        if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
            return True, "ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN"
        return False, "set ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN for a gateway)"
    if provider == "rote-cloud":
        if _cloud_credential() is not None:
            return True, "billed to the rote cloud tenant"
        return False, "run `rote login`"
    return False, f"unknown provider {provider!r}"


def select_provider(
    node_id: str,
    client: str,
    *,
    base_url: str | None = None,
    workload: str = "judge",
    local_tools: bool = False,
) -> str:
    """The provider serving this node: explicit choice, else auto-detect.

    ``workload`` picks the auto-detect order — ``"judge"`` for a single
    structured call, ``"agent"`` for a bounded multi-turn loop. They are
    inverses of each other, and deliberately so: the CLI's harness
    overhead is a tax on one answer and an amortized cost across twenty.

    An explicitly chosen provider that cannot serve the node is an error
    naming why — never a silent downgrade to a billing lane the operator
    did not pick.
    """
    order = _AGENT_ORDER if workload == "agent" else _JUDGE_ORDER
    chosen = os.environ.get(f"ROTE_INFERENCE_{node_id.upper()}") or os.environ.get("ROTE_INFERENCE")
    if chosen:
        chosen = chosen.strip()
        if chosen not in PROVIDERS:
            raise RuntimeError(
                f"unknown inference provider {chosen!r} for {node_id}: "
                f"expected one of {', '.join(PROVIDERS)}"
            )
        ok, reason = provider_availability(
            chosen, client, base_url=base_url, workload=workload, local_tools=local_tools
        )
        if not ok:
            raise RuntimeError(f"inference provider {chosen!r} cannot serve {node_id}: {reason}")
        return chosen

    reasons = []
    for provider in order:
        ok, reason = provider_availability(
            provider, client, base_url=base_url, workload=workload, local_tools=local_tools
        )
        if ok:
            return provider
        reasons.append(f"  {provider}: {reason}")
    raise RuntimeError(
        f"no inference provider available for {node_id} — nothing can serve this "
        f"{workload}:\n" + "\n".join(reasons)
    )


# ───────── error translation ─────────


def _durable_error(exc: Exception, *, node_id: str, provider: str, model: str) -> RuntimeError:
    """Vendor SDK errors do not survive a durable step boundary.

    Their classes take keyword-only ``response``/``body`` arguments, so
    a runtime that persists a failed step by pickling the exception
    cannot rebuild it — the real status and body would be replaced by a
    bare ``TypeError``. Carry the detail in a plain exception instead.
    """
    status = getattr(exc, "status_code", None)
    where = f" (HTTP {status})" if status is not None else ""
    return RuntimeError(
        f"{type(exc).__name__}{where} calling {model} for {node_id} via {provider}: {exc}"
    )


# ───────── usage log ─────────


def _log_usage(
    *, node_id: str, provider: str, model: str, input_tokens: Any, output_tokens: Any
) -> None:
    """Append token usage as JSONL to ``$ROTE_USAGE_LOG``, if set.

    What lets ``rote eval --run`` measure a graduated pipeline's actual
    LLM footprint instead of estimating it, and a zero-setup
    observability tap in production. The provider is recorded next to
    the tokens because the same token count costs differently per
    billing lane — a subscription trial and a key trial are not
    interchangeable data points. A logging failure never breaks the call.
    """
    path = os.environ.get("ROTE_USAGE_LOG")
    if not path:
        return
    try:
        record = {
            "node": node_id,
            "provider": provider,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except OSError:
        pass


# ───────── transport: vendor SDK ─────────


def _sdk_target(provider: str, client: str, base_url: str | None) -> tuple[str | None, str | None]:
    """``(base_url, explicit token)`` for an SDK call under ``provider``.

    For ``api`` the token is None **on purpose**: the SDK reads its own
    env vars, which is what makes an operator's ``ROTE_BASE_URL_<ID>`` +
    ``ANTHROPIC_AUTH_TOKEN`` gateway config work unchanged.

    For ``rote-cloud`` the token is passed explicitly, which in the
    Anthropic SDK also *suppresses* every credential env read ("explicit
    ctor args are total") — so an ambient ``ANTHROPIC_API_KEY`` cannot
    ride along to our proxy as a second header.
    """
    if provider != "rote-cloud":
        return base_url, None
    credential = _cloud_credential()
    if credential is None:  # pragma: no cover — select_provider checks first
        raise RuntimeError("rote cloud inference selected but no credential is stored")
    cloud_url, token = credential
    return f"{cloud_url.rstrip('/')}/v1/inference/{client}", token


def _call_anthropic(
    *,
    node_id: str,
    provider: str,
    model: str,
    base_url: str | None,
    prompt: str,
    output_schema: dict[str, Any],
    tool_description: str,
    sampling: dict[str, Any],
    max_tokens: int,
) -> tuple[dict[str, Any], Any, Any]:
    import anthropic

    endpoint, token = _sdk_target(provider, "anthropic", base_url)
    client = (
        anthropic.Anthropic(base_url=endpoint)
        if token is None
        else anthropic.Anthropic(base_url=endpoint, auth_token=token)
    )
    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            **sampling,
            tools=[
                {
                    "name": node_id,
                    "description": tool_description,
                    "input_schema": output_schema,
                }
            ],
            tool_choice={"type": "tool", "name": node_id},
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.APIError as exc:
        raise _durable_error(exc, node_id=node_id, provider=provider, model=model) from None
    usage = getattr(response, "usage", None)
    for block in response.content:
        if block.type == "tool_use":
            return (
                dict(block.input),
                getattr(usage, "input_tokens", None),
                getattr(usage, "output_tokens", None),
            )
    raise RuntimeError(f"LLM did not return a tool_use block for {node_id}")


def _call_openai(
    *,
    node_id: str,
    provider: str,
    model: str,
    base_url: str | None,
    prompt: str,
    output_schema: dict[str, Any],
    sampling: dict[str, Any],
) -> tuple[dict[str, Any], Any, Any]:
    import openai

    endpoint, token = _sdk_target(provider, "openai", base_url)
    client = (
        openai.OpenAI(base_url=endpoint)
        if token is None
        else openai.OpenAI(base_url=endpoint, api_key=token)
    )
    try:
        response = client.chat.completions.create(
            model=model,
            **sampling,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": node_id, "schema": output_schema, "strict": True},
            },
            messages=[{"role": "user", "content": prompt}],
        )
    except openai.APIError as exc:
        raise _durable_error(exc, node_id=node_id, provider=provider, model=model) from None
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError(f"OpenAI returned no content for {node_id}")
    usage = getattr(response, "usage", None)
    return (
        json.loads(content),
        getattr(usage, "prompt_tokens", None),
        getattr(usage, "completion_tokens", None),
    )


# ───────── transport: claude -p subprocess ─────────


def _cli_model(model: str) -> str:
    """The model id ``claude -p`` understands.

    Gateway-qualified ids (``anthropic/claude-sonnet-5``) are meaningful
    only to the endpoint that routes them; the CLI wants the bare id or
    an alias (``sonnet``).
    """
    return model.rsplit("/", 1)[-1]


def _call_claude_cli(
    *,
    node_id: str,
    model: str,
    prompt: str,
    output_schema: dict[str, Any],
    timeout: float,
) -> tuple[dict[str, Any], Any, Any]:
    """One schema-locked judge call through the Claude Code CLI.

    ``--json-schema`` makes this a *forced tool call* under the hood
    (the envelope reports ``stop_reason: "tool_use"``), so the
    subscription path has the same structural guarantee as the SDK path
    — not prose that happens to parse.

    The prompt goes on stdin, not argv: judge prompts carry whole
    documents, and argv is both size-limited and visible in ``ps``.
    """
    binary = shutil.which("claude")
    if binary is None:  # pragma: no cover — select_provider checks first
        raise RuntimeError("the `claude` CLI vanished between provider selection and the call")
    command = [
        binary,
        "-p",
        "--output-format",
        "json",
        "--json-schema",
        json.dumps(output_schema),
        "--model",
        _cli_model(model),
        "--system-prompt",
        _CLI_SYSTEM_PROMPT,
        # A judge reasons over its prompt and answers; it must not read
        # files, run commands, or inherit the user's settings. Both flags
        # also cut the request from ~37k tokens to ~740.
        "--tools",
        "--setting-sources",
        "",
    ]
    try:
        completed = subprocess.run(
            command,
            input=prompt,
            capture_output=True,
            text=True,
            env=build_subscription_env(),
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"claude -p did not answer within {timeout:.0f}s for {node_id}"
        ) from None

    try:
        envelope = json.loads(completed.stdout)
    except ValueError:
        detail = (completed.stderr or completed.stdout or "").strip()[:400]
        raise RuntimeError(
            f"claude -p returned no JSON envelope for {node_id} "
            f"(exit {completed.returncode}): {detail}"
        ) from None

    if envelope.get("is_error"):
        reason = envelope.get("result") or envelope.get("terminal_reason") or "unknown error"
        raise RuntimeError(f"claude -p failed for {node_id}: {str(reason)[:400]}")

    payload = envelope.get("structured_output")
    if payload is None:
        raw = envelope.get("result")
        try:
            payload = json.loads(raw) if isinstance(raw, str) else None
        except ValueError:
            payload = None
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"claude -p returned no structured output for {node_id}; "
            f"got: {str(envelope.get('result'))[:200]}"
        )
    usage = envelope.get("usage") or {}
    return payload, usage.get("input_tokens"), usage.get("output_tokens")


# ───────── the one entry point emitted judges call ─────────


def call_judge(
    *,
    node_id: str,
    client: str,
    model: str,
    prompt: str,
    output_schema: dict[str, Any],
    base_url: str | None = None,
    tool_description: str = "",
    sampling: dict[str, Any] | None = None,
    max_tokens: int = 4096,
) -> dict[str, Any]:
    """Run one schema-locked judge call and return its raw output dict.

    The caller owns validation (it holds the generated Pydantic model);
    this returns the decoded payload so the same helper serves every
    judge regardless of its schema.
    """
    provider = select_provider(node_id, client, base_url=base_url)
    sampling = dict(sampling or {})
    timeout = float(os.environ.get("ROTE_INFERENCE_TIMEOUT") or _DEFAULT_TIMEOUT_S)

    if provider == "claude-cli":
        payload, tokens_in, tokens_out = _call_claude_cli(
            node_id=node_id,
            model=model,
            prompt=prompt,
            output_schema=output_schema,
            timeout=timeout,
        )
    elif client == "anthropic":
        payload, tokens_in, tokens_out = _call_anthropic(
            node_id=node_id,
            provider=provider,
            model=model,
            base_url=base_url,
            prompt=prompt,
            output_schema=output_schema,
            tool_description=tool_description,
            sampling=sampling,
            max_tokens=max_tokens,
        )
    elif client == "openai":
        payload, tokens_in, tokens_out = _call_openai(
            node_id=node_id,
            provider=provider,
            model=model,
            base_url=base_url,
            prompt=prompt,
            output_schema=output_schema,
            sampling=sampling,
        )
    else:
        raise RuntimeError(f"unsupported LLM client {client!r} for {node_id}")

    _log_usage(
        node_id=node_id,
        provider=provider,
        model=_cli_model(model) if provider == "claude-cli" else model,
        input_tokens=tokens_in,
        output_tokens=tokens_out,
    )
    return payload


# ───────── agent loops ─────────


def _mcp_config_for(tools: Sequence[str]) -> tuple[dict[str, Any], list[str]]:
    """``(mcp-config servers, allowlisted tool ids)`` for an agent's tools.

    An ``agent_loop`` declares bare tool names (``zoominfo_search_contacts``)
    rather than a server binding — one loop may reach several servers, and
    :class:`MCPBinding` is a single ``server``/``tool`` pair. So the names
    are resolved the way ``rote run`` resolves a skill baseline's: wire
    every registered server and let ``--allowedTools`` be the boundary.
    Over-wiring is safe *because* the allowlist is the actual constraint;
    the agent can only call what the IR declared.

    Endpoint and credential resolution is delegated to the MCP helper
    already emitted beside this one, so there is one definition of "where
    does server X live and how do I authenticate to it" per app.
    """
    if not tools:
        return {}, []
    try:
        from extracted import _rote_mcp  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover — emitted together or not at all
        raise RuntimeError(
            "this agent loop declares MCP tools but extracted/_rote_mcp.py is "
            "missing; re-emit the pipeline"
        ) from None

    servers: dict[str, Any] = {}
    for name, entry in sorted(_rote_mcp._registry_servers().items()):
        try:
            url = _rote_mcp.resolve_url(name, None)
        except Exception:  # unresolvable server: skip, the allowlist still binds
            continue
        config: dict[str, Any] = {"type": "http", "url": url}
        headers = entry.get("headers") if isinstance(entry, dict) else None
        if headers:
            config["headers"] = dict(headers)
        servers[name] = config
    # The IR names tools without servers, so allow each declared name on
    # every wired server; a name no server provides simply never resolves.
    allowed = [f"mcp__{server}__{tool}" for server in sorted(servers) for tool in sorted(tools)]
    return servers, allowed


def _agent_task_prompt(
    *, description: str, task: str, termination: str | None, local_tool_names: Sequence[str]
) -> str:
    parts = [description.strip(), "", task.strip()]
    if local_tool_names:
        parts += ["", "Pipeline steps available to you as tools: " + ", ".join(local_tool_names)]
    if termination:
        parts += ["", f"Stop when: {termination}"]
    parts += [
        "",
        "When you are done, state your final answer as the last message. "
        "Do not ask follow-up questions — nobody is available to answer.",
    ]
    return "\n".join(parts)


def _run_agent_via_cli(
    *,
    node_id: str,
    model: str,
    prompt: str,
    tools: Sequence[str],
    max_iterations: int,
    timeout: float,
) -> tuple[str, Any, Any, int]:
    """A bounded agent loop on the Claude subscription.

    Unlike the judge path this one *keeps* tools — that is the entire
    point — but still replaces the coding-agent system prompt and ignores
    the user's settings files, so the loop gets the pipeline's tools and
    nothing else.
    """
    binary = shutil.which("claude")
    if binary is None:  # pragma: no cover — select_provider checks first
        raise RuntimeError("the `claude` CLI vanished between provider selection and the call")
    servers, allowed = _mcp_config_for(tools)
    command = [
        binary,
        "-p",
        "--output-format",
        "json",
        "--model",
        _cli_model(model),
        "--system-prompt",
        _AGENT_SYSTEM_PROMPT,
        "--setting-sources",
        "",
        "--max-turns",
        str(max_iterations),
    ]
    if allowed:
        command += ["--allowedTools", ",".join(allowed)]
        command += ["--mcp-config", json.dumps({"mcpServers": servers})]
    else:
        command.append("--tools")

    try:
        completed = subprocess.run(
            command,
            input=prompt,
            capture_output=True,
            text=True,
            env=build_subscription_env(),
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"claude -p agent loop for {node_id} exceeded {timeout:.0f}s") from None

    try:
        envelope = json.loads(completed.stdout)
    except ValueError:
        detail = (completed.stderr or completed.stdout or "").strip()[:400]
        raise RuntimeError(
            f"claude -p returned no JSON envelope for {node_id} "
            f"(exit {completed.returncode}): {detail}"
        ) from None
    if envelope.get("is_error"):
        reason = envelope.get("result") or envelope.get("terminal_reason") or "unknown error"
        raise RuntimeError(f"claude -p agent loop failed for {node_id}: {str(reason)[:400]}")
    usage = envelope.get("usage") or {}
    return (
        str(envelope.get("result") or ""),
        usage.get("input_tokens"),
        usage.get("output_tokens"),
        int(envelope.get("num_turns") or 0),
    )


def _run_agent_via_sdk(
    *,
    node_id: str,
    provider: str,
    model: str,
    base_url: str | None,
    prompt: str,
    local_tools: Mapping[str, Any],
    tools: Sequence[str],
    max_iterations: int,
    timeout: float,
) -> tuple[str, Any, Any, int]:
    """A bounded agent loop on the vendor SDK's own tool runner.

    ``anthropic.lib.tools`` ships a tool loop, so this needs no agent
    framework and no dependency the emitted app does not already have for
    its judges. ``@beta_tool`` derives each tool's JSON schema from the
    Python signature of the pipeline step it wraps, which means the
    agent's tool contract and the step's real contract cannot drift.
    """
    import anthropic
    from anthropic.lib.tools import beta_tool

    endpoint, token = _sdk_target(provider, "anthropic", base_url)
    client = (
        anthropic.Anthropic(base_url=endpoint, timeout=timeout)
        if token is None
        else anthropic.Anthropic(base_url=endpoint, auth_token=token, timeout=timeout)
    )
    # The tool's name is the sub-node's id and its schema is derived from
    # the emitted step's own signature, so the contract the agent sees and
    # the contract the step enforces are the same object — as rich as the
    # IR's input_schema made that step, no richer and no staler.
    bound = [beta_tool(name=name)(fn) for name, fn in local_tools.items()]
    servers, _allowed = _mcp_config_for(tools)
    mcp_servers = [
        {"type": "url", "url": config["url"], "name": name} for name, config in servers.items()
    ]

    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": 4096,
        "tools": bound,
        "max_iterations": max_iterations,
        "system": _AGENT_SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": prompt}],
    }
    if mcp_servers:
        kwargs["mcp_servers"] = mcp_servers
    try:
        runner = client.beta.messages.tool_runner(**kwargs)
        final = runner.until_done()
    except anthropic.APIError as exc:
        raise _durable_error(exc, node_id=node_id, provider=provider, model=model) from None

    text = "\n".join(
        b.text for b in getattr(final, "content", []) if getattr(b, "type", "") == "text"
    )
    usage = getattr(final, "usage", None)
    return (
        text,
        getattr(usage, "input_tokens", None),
        getattr(usage, "output_tokens", None),
        max_iterations,
    )


def run_agent_loop(
    *,
    node_id: str,
    description: str,
    task: str,
    model: str,
    tools: Sequence[str] = (),
    local_tools: Mapping[str, Any] | None = None,
    max_iterations: int = 10,
    termination: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Run one bounded, tool-restricted agent loop and return its result.

    This is what an ``agent_loop`` node executes. It is a real loop on
    both lanes — never a stub — because a pipeline that hands its one
    genuinely agentic step back to the user has not graduated that step.

    ``local_tools`` are the node's ``loop_body`` sub-nodes: already
    emitted, already working pipeline steps, bound as callables so the
    agent drives them per iteration exactly as the IR describes.
    """
    local_tools = dict(local_tools or {})
    provider = select_provider(
        node_id,
        "anthropic",
        base_url=base_url,
        workload="agent",
        local_tools=bool(local_tools),
    )
    timeout = float(os.environ.get("ROTE_AGENT_TIMEOUT") or _DEFAULT_AGENT_TIMEOUT_S)
    prompt = _agent_task_prompt(
        description=description,
        task=task,
        termination=termination,
        local_tool_names=sorted(local_tools),
    )

    if provider == "claude-cli":
        text, tokens_in, tokens_out, turns = _run_agent_via_cli(
            node_id=node_id,
            model=model,
            prompt=prompt,
            tools=tools,
            max_iterations=max_iterations,
            timeout=timeout,
        )
    else:
        text, tokens_in, tokens_out, turns = _run_agent_via_sdk(
            node_id=node_id,
            provider=provider,
            model=model,
            base_url=base_url,
            prompt=prompt,
            local_tools=local_tools,
            tools=tools,
            max_iterations=max_iterations,
            timeout=timeout,
        )

    _log_usage(
        node_id=node_id,
        provider=provider,
        model=_cli_model(model) if provider == "claude-cli" else model,
        input_tokens=tokens_in,
        output_tokens=tokens_out,
    )
    return {"result": text, "provider": provider, "iterations": turns}
