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
import re
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

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

#: Context isolation shared by every ``claude -p`` this helper spawns,
#: judge and agent loop alike. One tuple rather than two copies, because
#: the agent-loop copy is the one that drifted every time before. Each
#: entry was measured on Claude Code 2.1.302 by removing it from a real
#: "say hi" judge call (input + cache_write + cache_read on the envelope):
#:
#: * ``--setting-sources ""`` keeps the user's and project's settings,
#:   CLAUDE.md files, hooks and plugins out. Without it: 25.6k tokens.
#: * ``--strict-mcp-config`` loads only servers passed in ``--mcp-config``.
#:   Without it the user's claude.ai connectors and plugin MCP servers
#:   ride along: 30.3k tokens and $0.183 a cold call, and the judge can see
#:   (if not call) tools like Gmail's send_message. Connectors load
#:   asynchronously, so the overhead also varies call to call.
#: * ``autoMemoryEnabled: false`` keeps the auto-memory index of whatever
#:   repository the pipeline runs from out of the prompt: 1.8k tokens of
#:   someone's private notes (one judge greeted the developer by name).
#:   ``--setting-sources ""`` does not cover it; ``--settings`` does.
#: * ``--tools ""`` removes the built-in coding tools. It does not touch
#:   MCP tools, so an agent loop keeps the servers it declared. Without it
#:   the loop carried 31 built-in tools: 31k tokens on every turn.
#:
#: With all four a judge costs 1.25k tokens and an agent loop 1.4k a
#: turn. The empty VALUE after ``--tools`` is required, not decorative:
#: the flag is variadic (``--tools <tools...>``), so a bare ``--tools`` is
#: rejected outright ("option '--tools <tools...>' argument missing",
#: exit 1, on every call).
#:
#: The first three flags are the context half, shared with rote's own
#: compiler driver (``rote.compiler.drivers.claude``), which imports them
#: from here for the same reason it imports ``build_subscription_env``: a
#: second copy is how a fix lands in one place and not the other. Only the
#: tool surface differs: judges and loops take ``--tools ""``, the compiler
#: ``--tools <its allowlist>``.
CLI_CONTEXT_ISOLATION_FLAGS = (
    "--setting-sources",
    "",
    "--strict-mcp-config",
    "--settings",
    '{"autoMemoryEnabled": false}',
)
_CLI_ISOLATION_FLAGS = (*CLI_CONTEXT_ISOLATION_FLAGS, "--tools", "")


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


def _usage_field(usage: Any, name: str) -> Any:
    """One usage field from a dict (CLI envelope) or an SDK object."""
    value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _anthropic_usage(usage: Any) -> dict[str, Any]:
    """The four token buckets of an Anthropic-shaped ``usage``.

    All four, because a prompt-cached call reports almost nothing in
    ``input_tokens``: ``claude -p`` always caches, and a measured judge
    call reported ``input_tokens: 3`` beside 30k cache tokens. Logging the
    plain pair alone made the subscription lane look ~10,000x cheaper
    than it was. ``input_tokens`` keeps the vendor's meaning (uncached
    input only), so the four buckets add up to the whole prompt.
    """
    return {
        "input_tokens": _usage_field(usage, "input_tokens"),
        "output_tokens": _usage_field(usage, "output_tokens"),
        "cache_write_tokens": _usage_field(usage, "cache_creation_input_tokens"),
        "cache_read_tokens": _usage_field(usage, "cache_read_input_tokens"),
    }


def _cli_usage_rows(
    envelope: Mapping[str, Any], requested_model: str
) -> list[tuple[str, dict[str, Any]]]:
    """``(model, usage)`` per model a ``claude -p`` call actually ran.

    The model asked for is often an alias (``sonnet``), which no price
    catalog lists, so a row logged under it could never be priced. The
    envelope's ``modelUsage`` names the concrete model per entry
    (``claude-sonnet-5-5``) with its four buckets and the CLI's own cost
    estimate, covering the whole query. Without ``modelUsage`` the row
    keeps the requested id and the main-loop ``usage``, and pricing
    refuses an alias rather than guess which model it meant.
    """
    by_model = envelope.get("modelUsage")
    rows: list[tuple[str, dict[str, Any]]] = []
    if isinstance(by_model, dict):
        for name, entry in by_model.items():
            if not isinstance(entry, dict):
                continue
            cost = entry.get("costUSD")
            rows.append(
                (
                    str(name),
                    {
                        "input_tokens": _usage_field(entry, "inputTokens"),
                        "output_tokens": _usage_field(entry, "outputTokens"),
                        "cache_write_tokens": _usage_field(entry, "cacheCreationInputTokens"),
                        "cache_read_tokens": _usage_field(entry, "cacheReadInputTokens"),
                        "cost_usd": (
                            float(cost)
                            if isinstance(cost, (int, float)) and not isinstance(cost, bool)
                            else None
                        ),
                    },
                )
            )
    if rows:
        return rows
    return [(_cli_model(requested_model), _anthropic_usage(envelope.get("usage")))]


def _openai_usage(usage: Any) -> dict[str, Any]:
    """The same four buckets from an OpenAI chat-completions ``usage``.

    OpenAI counts cached tokens *inside* ``prompt_tokens`` and has no
    separate write charge, so they are split out to match the Anthropic
    meaning: ``input_tokens`` is the uncached remainder.
    """
    prompt = _usage_field(usage, "prompt_tokens")
    details = getattr(usage, "prompt_tokens_details", None)
    cached = _usage_field(details, "cached_tokens") or 0
    return {
        "input_tokens": None if prompt is None else prompt - cached,
        "output_tokens": _usage_field(usage, "completion_tokens"),
        "cache_write_tokens": 0,
        "cache_read_tokens": cached,
    }


def _log_usage(
    *,
    node_id: str,
    provider: str,
    model: str,
    usage: Mapping[str, Any],
    iterations: int | None = None,
) -> None:
    """Append token usage as JSONL to ``$ROTE_USAGE_LOG``, if set.

    What lets ``rote eval --run`` measure a graduated pipeline's actual
    LLM footprint instead of estimating it, and a zero-setup
    observability tap in production. The provider is recorded next to
    the tokens because the same token count costs differently per
    billing lane — a subscription trial and a key trial are not
    interchangeable data points. ``usage`` carries the four buckets from
    :func:`_anthropic_usage` / :func:`_openai_usage`. A logging failure
    never breaks the call.
    """
    path = os.environ.get("ROTE_USAGE_LOG")
    if not path:
        return
    try:
        record = {
            "node": node_id,
            "provider": provider,
            "model": model,
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "cache_write_tokens": usage.get("cache_write_tokens"),
            "cache_read_tokens": usage.get("cache_read_tokens"),
        }
        if usage.get("cost_usd") is not None:
            # The CLI's own estimate. It knows what a catalog cannot, such
            # as a 1h cache write billing at twice the 5m rate.
            record["cost_usd"] = usage["cost_usd"]
        if iterations is not None:
            # An agent loop's turns: its result is its declared output, so
            # this log is where the count is observable.
            record["iterations"] = iterations
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except OSError:
        pass


# ───────── structured outputs (Anthropic) ─────────

#: String formats Anthropic structured outputs accepts. Any other format
#: is a 400 ("format 'phone' is not supported"), so it moves into the
#: description instead.
_STRUCTURED_FORMATS = frozenset(
    {"date-time", "time", "date", "duration", "email", "hostname", "uri", "ipv4", "ipv6", "uuid"}
)

#: Keywords kept as they are, on any schema node. Everything else that
#: is not handled below (``minimum``, ``maxLength``, ``maxItems``,
#: ``pattern``, ``default``, ``not``, ``patternProperties`` and so on)
#: either 400s or is outside what the anthropic SDK's own
#: ``transform_schema`` keeps, so it is written into the description,
#: where the model can still read it. The judge's own validator, built
#: from the original schema, is what enforces it.
_STRUCTURED_KEPT = frozenset({"type", "enum", "const", "description", "title", "$ref", "required"})


def structured_output_schema(schema: Mapping[str, Any], *, node_id: str) -> dict[str, Any]:
    """``schema`` rewritten into the subset Anthropic structured outputs accepts.

    ``output_config.format`` constrains decoding to the schema, which is
    why it only accepts a subset of JSON Schema. Measured against the
    live API on 2026-10-20 (claude-haiku-5-5 and claude-sonnet-5-5 gave
    identical answers):

    * every object needs ``additionalProperties: false``; omitting it and
      ``true`` are both a 400. An object that omits it gets it set here,
      which only narrows what the model may write.
    * ``oneOf`` is a 400 and becomes ``anyOf``, which accepts a superset.
    * numeric and string bounds, ``maxItems``, ``minItems`` above 1,
      ``not`` and ``patternProperties`` are a 400 and move into the
      description.
    * ``$defs``, ``$ref``, ``anyOf``, ``allOf``, ``enum`` and ``const``
      (with or without a ``type``), type arrays and optional properties
      are all accepted as they are.

    Two shapes cannot be expressed at all, and raise rather than being
    approximated: an object that admits keys it does not declare (no
    declared properties, or ``additionalProperties`` set to ``true`` or a
    schema, as in a fixed key plus a typed map), and a value with no type.
    Closing either would make the model drop data, or guess a type, with
    nothing downstream noticing. Both errors contain
    :data:`STRUCTURED_INEXPRESSIBLE`.

    The caller validates the answer against the original schema, which
    is what enforces the constraints that moved into descriptions.
    """
    return _structured_node(schema, "#", node_id)


def _structured_node(schema: Any, path: str, node_id: str) -> dict[str, Any]:
    if not isinstance(schema, Mapping):
        raise RuntimeError(f"{node_id}: output schema at {path} is not a JSON Schema object")
    out: dict[str, Any] = {}
    moved: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "$defs":
            out[key] = {
                name: _structured_node(sub, f"{path}/$defs/{name}", node_id)
                for name, sub in value.items()
            }
        elif key in ("anyOf", "oneOf", "allOf"):
            target = "allOf" if key == "allOf" else "anyOf"
            out[target] = [
                _structured_node(sub, f"{path}/{key}/{i}", node_id) for i, sub in enumerate(value)
            ]
        elif key == "properties":
            out[key] = {
                name: _structured_node(sub, f"{path}/properties/{name}", node_id)
                for name, sub in value.items()
            }
        elif key == "items" and isinstance(value, Mapping):
            out[key] = _structured_node(value, f"{path}/items", node_id)
        elif _structured_keeps(key, value, path):
            out[key] = value
        elif key == "additionalProperties":
            continue  # decided below, from the node's type
        else:
            moved[key] = value

    kind = schema.get("type")
    kinds = kind if isinstance(kind, list) else [kind]
    if "object" in kinds:
        extra = schema.get("additionalProperties", False)
        if extra is not False:
            raise RuntimeError(
                f"{node_id}: output schema at {path} is an object that allows keys it "
                f"does not declare (additionalProperties: {json.dumps(extra)}), "
                f"{STRUCTURED_INEXPRESSIBLE} (every object must set "
                "additionalProperties: false). Declare every key as a property, or "
                "turn the map into a list of objects, in the IR's "
                "signature_spec.output_schema."
            )
        if not schema.get("properties") and "additionalProperties" not in schema:
            raise RuntimeError(
                f"{node_id}: output schema at {path} is an object with no declared "
                f"properties, {STRUCTURED_INEXPRESSIBLE} (every object must set "
                "additionalProperties: false). Declare the object's properties in "
                "the IR's signature_spec.output_schema."
            )
        out["additionalProperties"] = False
    if kind is None and not any(k in schema for k in _STRUCTURED_UNTYPED_OK):
        raise RuntimeError(
            f"{node_id}: output schema at {path} has no type, "
            f"{STRUCTURED_INEXPRESSIBLE}. Give it a type in the IR's "
            "signature_spec.output_schema."
        )
    if moved:
        note = json.dumps(moved, sort_keys=True, default=str)
        described = out.get("description")
        out["description"] = f"{described}\n\n{note}" if described else note
    return out


#: In every error :func:`structured_output_schema` raises for a shape the
#: subset cannot express, and in no other error, so a caller with another
#: way to ask (an agent loop's submit tool) can tell the two apart. A
#: substring rather than an exception class: a custom class only unpickles
#: where this module is importable (see the module docstring).
STRUCTURED_INEXPRESSIBLE = "which Anthropic structured outputs cannot express"

#: Keywords that give a node a value space without a ``type``.
_STRUCTURED_UNTYPED_OK = ("enum", "const", "$ref", "anyOf", "oneOf", "allOf")


def _structured_keeps(key: str, value: Any, path: str) -> bool:
    """Whether ``key: value`` passes through to the request schema as is."""
    if key == "format":
        return value in _STRUCTURED_FORMATS
    if key == "minItems":
        return value in (0, 1)
    return key in _STRUCTURED_KEPT or (key == "$schema" and path == "#")


def structured_answer(response: Any, *, node_id: str) -> Any:
    """The JSON answer of a structured-output response, or a RuntimeError.

    A schema-locked response puts its answer in text blocks and ends on
    ``end_turn``. Any other stop reason means the text is not a complete
    answer: ``max_tokens`` truncated it, ``refusal`` replaced it. Neither
    may reach a downstream node as data, so both raise with the reason.
    """
    stop = getattr(response, "stop_reason", None)
    if stop != "end_turn":
        raise RuntimeError(
            f"structured output for {node_id} ended with stop_reason {stop!r}, not "
            "end_turn, so it is not a complete answer"
        )
    text = "".join(
        block.text
        for block in getattr(response, "content", [])
        if getattr(block, "type", "") == "text"
    )
    try:
        return json.loads(text)
    except ValueError:
        raise RuntimeError(f"structured output for {node_id} is not JSON: {text[:200]!r}") from None


# ───────── effort ─────────

#: The effort a judge or agent loop runs at when no operator sets one.
#: None sends no effort, which the API runs as ``high``. Measured on
#: 2026-10-20 over 8 example judges x 6 inputs: ``low`` cut output tokens
#: 15% on claude-sonnet-4-6 and 33% on claude-sonnet-5-5, but changed
#: decisions the default made the same way twice (it dropped an ARR and a
#: go-live date stated in the message, and missed a deploy as the cause),
#: so accuracy keeps the API's default.
DEFAULT_JUDGE_EFFORT: str | None = None


def node_effort(node_id: str) -> str | None:
    """The effort this judge or agent loop runs at, or None for none.

    Every lane reads it: the SDK lanes send it as ``output_config.effort``
    and the ``claude-cli`` lane as ``claude -p --effort`` (see
    :func:`_cli_effort_flags`), so the knob is never a silent no-op on a
    subscription.

    ``ROTE_EFFORT_<NODE_ID>`` > ``ROTE_JUDGE_EFFORT`` > :data:`DEFAULT_JUDGE_EFFORT`,
    the same shape as the ``ROTE_TEMPERATURE_<NODE_ID>`` knob: a variable
    that is set but empty omits the parameter, so an operator can always
    get back to the API's own default without a re-emit. The value is not
    checked here; the levels differ per model (claude-sonnet-4-6 has no
    ``xhigh``), and the API's 400 names the ones it accepts.

    The pipeline-wide name is not ``ROTE_EFFORT``: that one is the
    compiler's (``rote compile --effort``), and a developer who sets it
    for compiles must not silently change every judge they run.
    """
    for var in (f"ROTE_EFFORT_{node_id.upper()}", "ROTE_JUDGE_EFFORT"):
        value = os.environ.get(var)
        if value is not None:
            return value.strip() or None
    return DEFAULT_JUDGE_EFFORT


def _cli_effort_flags(node_id: str, binary: str) -> list[str]:
    """``claude -p`` flags for this node's effort: ``--effort <level>`` or none.

    One function for the judge and the agent-loop command lines, because
    their flag blocks are separate copies and the agent-loop copy is the
    one that drifted every time before (see ``_CLI_ISOLATION_FLAGS``).

    The CLI does not fail on an unknown level: it warns on stderr and runs
    at its default, exit 0 (Claude Code 2.1.304), so a bad value would be a
    silent no-op here while the SDK lanes 400 on it. The level is therefore
    checked before any call is paid for, by :func:`_check_cli_effort`.
    The helper never holds its own list of levels, which would go stale;
    the CLI is asked.
    """
    effort = node_effort(node_id)
    if effort is None:
        return []
    _check_cli_effort(binary, effort, node_id)
    return ["--effort", effort]


#: A level no CLI will ever accept, for :func:`_check_cli_effort`'s self-check.
_EFFORT_PROBE = "rote-self-check-not-a-level"

#: ``(binary, level)`` pairs already checked in this process.
_CHECKED_EFFORTS: set[tuple[str, str]] = set()

_EFFORT_WORD = re.compile(r"\beffort\b", re.IGNORECASE)
_REJECTION_WORD = re.compile(
    r"\b(unknown|invalid|unsupported|unrecognized|not supported|not a valid|not valid)\b",
    re.IGNORECASE,
)


def _effort_warning(stderr: str | None) -> str | None:
    """The CLI's line saying it rejected an effort level, or None.

    Matched by meaning, not by exact text: any line naming effort and
    saying unknown, invalid, unsupported (and so on). Today's wording is
    "Warning: Unknown --effort value 'x' — ignoring it and using the
    default effort."
    """
    for line in (stderr or "").splitlines():
        if _EFFORT_WORD.search(line) and _REJECTION_WORD.search(line):
            return line.strip()[:300]
    return None


def _cli_effort_stderr(binary: str, level: str) -> str:
    """stderr of ``claude --effort <level> --version``: no prompt, no cost.

    ``--version`` exits before any model call, measured at about 50ms,
    and the CLI validates ``--effort`` first, so this reports on a level
    without spending a token.
    """
    try:
        completed = subprocess.run(
            [binary, "--effort", level, "--version"],
            capture_output=True,
            text=True,
            env=build_subscription_env(),
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"`{binary} --effort {level} --version` did not exit in 30s") from None
    return completed.stderr or ""


def _check_cli_effort(binary: str, level: str, node_id: str) -> None:
    """Fail before a paid call when the CLI would ignore ``level``.

    Two checks, once per process each. The self-check gives the CLI a level
    nothing accepts and requires a recognizable rejection: if a CLI release
    changes its wording past :func:`_effort_warning`, every effort check
    would silently pass, so that case fails loudly instead. Then the real
    level, which must draw no rejection.
    """
    if (binary, _EFFORT_PROBE) not in _CHECKED_EFFORTS:
        stderr = _cli_effort_stderr(binary, _EFFORT_PROBE)
        if _effort_warning(stderr) is None:
            raise RuntimeError(
                f"claude -p for {node_id}: cannot verify the effort level. Given an "
                f"invalid --effort, this claude CLI printed no warning rote recognizes "
                f"(stderr: {stderr.strip()[:200]!r}), so an unsupported level would be "
                "ignored silently. Update rote, or unset ROTE_EFFORT_<NODE_ID> / "
                "ROTE_JUDGE_EFFORT to run at the CLI's default."
            )
        _CHECKED_EFFORTS.add((binary, _EFFORT_PROBE))
    if (binary, level) not in _CHECKED_EFFORTS:
        warning = _effort_warning(_cli_effort_stderr(binary, level))
        if warning is not None:
            raise RuntimeError(f"claude -p for {node_id}: {warning}")
        _CHECKED_EFFORTS.add((binary, level))


def _refuse_ignored_effort(node_id: str, stderr: str | None) -> None:
    """Raise when a ``claude -p`` run says it ignored its ``--effort``.

    The pre-check in :func:`_check_cli_effort` should make this
    unreachable; it stays because the run's own stderr is the last word.
    """
    warning = _effort_warning(stderr)
    if warning is not None:
        raise RuntimeError(f"claude -p for {node_id}: {warning}")


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
    sampling: dict[str, Any],
    max_tokens: int,
    sdk_client: Any = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """One schema-locked judge call through ``output_config.format``.

    Not a forced tool call: claude-sonnet-5-5 rejects ``tool_choice``
    types ``tool`` and ``any`` outright (HTTP 400, verified 2026-10-20),
    while structured outputs works on every current model. One mechanism
    for every model, so nothing here branches on a model id.
    """
    import anthropic

    if sdk_client is not None:
        client = sdk_client
    else:
        endpoint, token = _sdk_target(provider, "anthropic", base_url)
        client = (
            anthropic.Anthropic(base_url=endpoint)
            if token is None
            else anthropic.Anthropic(base_url=endpoint, auth_token=token)
        )
    request_schema = structured_output_schema(output_schema, node_id=node_id)
    effort = node_effort(node_id)
    # Any: the SDK types this as a TypedDict, which a dict built at run
    # time cannot satisfy statically.
    output_config: Any = {"format": {"type": "json_schema", "schema": request_schema}}
    if effort is not None:
        output_config["effort"] = effort
    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            **sampling,
            output_config=output_config,
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.APIError as exc:
        raise _durable_error(exc, node_id=node_id, provider=provider, model=model) from None
    answer = structured_answer(response, node_id=node_id)
    if not isinstance(answer, dict):
        raise RuntimeError(f"structured output for {node_id} is not a JSON object")
    return answer, _anthropic_usage(getattr(response, "usage", None))


def _call_openai(
    *,
    node_id: str,
    provider: str,
    model: str,
    base_url: str | None,
    prompt: str,
    output_schema: dict[str, Any],
    sampling: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
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
    return json.loads(content), _openai_usage(getattr(response, "usage", None))


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
) -> tuple[dict[str, Any], list[tuple[str, dict[str, Any]]]]:
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
        # files, run commands, or see anything of the user's setup.
        *_CLI_ISOLATION_FLAGS,
        *_cli_effort_flags(node_id, binary),
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

    _refuse_ignored_effort(node_id, completed.stderr)
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
    return payload, _cli_usage_rows(envelope, model)


# ───────── the one entry point emitted judges call ─────────


def call_judge(
    *,
    node_id: str,
    client: str,
    model: str,
    prompt: str,
    output_schema: dict[str, Any],
    base_url: str | None = None,
    sampling: dict[str, Any] | None = None,
    max_tokens: int = 4096,
) -> dict[str, Any]:
    """Run one schema-locked judge call and return its raw output dict.

    The caller owns validation (it holds the generated Pydantic model);
    this returns the decoded payload so the same helper serves every
    judge regardless of its schema.
    """
    provider = select_provider(node_id, client, base_url=base_url)
    payload, rows = structured_call(
        node_id=node_id,
        provider=provider,
        client=client,
        model=model,
        prompt=prompt,
        output_schema=output_schema,
        base_url=base_url,
        sampling=sampling,
        max_tokens=max_tokens,
    )
    for row_model, row_usage in rows:
        _log_usage(node_id=node_id, provider=provider, model=row_model, usage=row_usage)
    return payload


def structured_call(
    *,
    node_id: str,
    provider: str,
    client: str,
    model: str,
    prompt: str,
    output_schema: dict[str, Any],
    base_url: str | None = None,
    sampling: dict[str, Any] | None = None,
    max_tokens: int = 4096,
    sdk_client: Any = None,
) -> tuple[dict[str, Any], list[tuple[str, dict[str, Any]]]]:
    """One schema-locked call on a provider already chosen: ``(payload, usage rows)``.

    :func:`call_judge` is this plus provider selection and the usage log.
    Each usage row is ``(concrete model, the four token buckets)``, and a
    ``claude-cli`` row also carries the CLI's own ``cost_usd``.
    ``sdk_client`` is an Anthropic client the caller already built (rote's
    compiler passes its api driver's, so a call bills and routes exactly
    like the compile); without it one is built for ``provider``.
    """
    sampling = dict(sampling or {})
    timeout = float(os.environ.get("ROTE_INFERENCE_TIMEOUT") or _DEFAULT_TIMEOUT_S)

    if provider == "claude-cli":
        payload, rows = _call_claude_cli(
            node_id=node_id,
            model=model,
            prompt=prompt,
            output_schema=output_schema,
            timeout=timeout,
        )
    elif client == "anthropic":
        payload, usage = _call_anthropic(
            node_id=node_id,
            provider=provider,
            model=model,
            base_url=base_url,
            prompt=prompt,
            output_schema=output_schema,
            sampling=sampling,
            max_tokens=max_tokens,
            sdk_client=sdk_client,
        )
    elif client == "openai":
        payload, usage = _call_openai(
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
    if provider != "claude-cli":
        rows = [(model, usage)]
    return payload, rows


# ───────── agent loops ─────────


class AgentMcp(NamedTuple):
    """An agent loop's MCP wiring, as ``(server, tool)`` pairs.

    Pairs, never ``mcp__server__tool`` strings: a server name may itself
    contain ``__``, and splitting the id back apart then names the wrong
    server and tool.
    """

    #: ``{server: mcp-config entry}`` for every server wired to the loop.
    servers: dict[str, Any]
    #: What the loop may call.
    allowed: list[tuple[str, str]]
    #: What a read-only run took away because it may write.
    removed: list[tuple[str, str]]
    #: ``{tool: why}`` for removed tools no wired server still offers.
    withheld: dict[str, str]


def _tool_id(server: str, tool: str) -> str:
    """Claude Code's name for an MCP tool."""
    return f"mcp__{server}__{tool}"


def _mcp_config_for(
    tools: Sequence[str], tool_servers: Mapping[str, str] | None = None
) -> AgentMcp:
    """The servers, allowlist and read-only withholdings for an agent's tools.

    An ``agent_loop`` declares bare tool names (``zoominfo_search_contacts``)
    rather than a server binding — one loop may reach several servers, and
    :class:`MCPBinding` is a single ``server``/``tool`` pair. So the names
    are resolved the way ``rote run`` resolves a skill baseline's: wire
    every registered server and let ``--allowedTools`` be the boundary.
    Over-wiring is safe *because* the allowlist is the actual constraint;
    the agent can only call what the IR declared.

    ``tool_servers`` narrows that: a tool the IR resolved to a server is
    allowlisted on THAT server only. Without it the allowlist is a full
    cross product, which allowlists every tool on every server — mostly
    phantom entries, and a real hazard when two servers export the same
    tool name and the agent may pick either.

    Endpoint and credential resolution is delegated to the MCP helper
    already emitted beside this one, so there is one definition of "where
    does server X live and how do I authenticate to it" per app.
    """
    if not tools:
        return AgentMcp({}, [], [], {})
    try:
        from extracted import _rote_mcp  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover — emitted together or not at all
        raise RuntimeError(
            "this agent loop declares MCP tools but extracted/_rote_mcp.py is "
            "missing; re-emit the pipeline"
        ) from None

    pins = dict(tool_servers or {})
    registry = _rote_mcp._registry_servers()
    servers: dict[str, Any] = {}
    # Every registered server, plus each server the IR pinned a tool to:
    # a pinned server may have no registry entry and resolve from its
    # ROTE_MCP_<SERVER>_URL variable alone.
    for name in sorted(set(registry) | set(pins.values())):
        try:
            url = _rote_mcp.resolve_url(name, None)
        except _rote_mcp.RoteMcpConfigError as exc:
            needed = sorted(tool for tool, server in pins.items() if server == name)
            if needed:
                # The loop is known to need this server, and no attempt
                # will find an endpoint that is not configured.
                raise _rote_mcp.RoteMcpConfigError(
                    f"agent loop tools {', '.join(needed)} are served by {name!r}: {exc}"
                ) from None
            continue  # not known to be needed; the allowlist still binds
        config: dict[str, Any] = {"type": "http", "url": url}
        entry = registry.get(name)
        headers = entry.get("headers") if isinstance(entry, dict) else None
        if headers:
            config["headers"] = dict(headers)
        servers[name] = config
    allowed = [
        (server, tool)
        for server in sorted(servers)
        for tool in sorted(tools)
        # Resolved: this server only. Unresolved: every wired server, since
        # a name no server provides simply never resolves anyway.
        if pins.get(tool, server) == server
    ]
    removed, withheld = _withhold_writes(_rote_mcp, servers, allowed)
    return AgentMcp(servers, allowed, removed, withheld)


def _withhold_writes(
    _rote_mcp: Any, servers: Mapping[str, Any], allowed: list[tuple[str, str]]
) -> tuple[list[tuple[str, str]], dict[str, str]]:
    """In a read-only run, drop from ``allowed`` (in place) every tool that may write.

    The loop calls its tools through its own MCP client, so the run holds
    it to read-only calls by what it hands over: each server is listed and
    a tool passes only by the rule a pipeline step's own call meets
    (``_rote_mcp.agent_tool_refusals``). Returns the removed pairs, and
    ``{tool: why}`` for the removed tools no wired server still offers. A
    loop left with no MCP
    tool at all is refused: it would run, but not as declared. ``_rote_mcp``
    is the emitted MCP helper the caller already imported.
    """
    if os.environ.get(_rote_mcp.READ_ONLY_ENV) != "1" or not allowed:
        return [], {}
    import asyncio

    removed: list[tuple[str, str]] = []
    reasons: dict[str, str] = {}
    for name in sorted(servers):
        offered = [tool for server, tool in allowed if server == name]
        refused = asyncio.run(_rote_mcp.agent_tool_refusals(name, None, offered))
        for tool, why in refused.items():
            allowed.remove((name, tool))
            removed.append((name, tool))
            reasons.setdefault(tool, why)
    kept = {tool for _server, tool in allowed}
    withheld = {tool: why for tool, why in reasons.items() if tool not in kept}
    if not allowed:
        raise _rote_mcp.RoteMcpConfigError(
            "this agent loop has no MCP tool left to call, so it does not start. "
            + " ".join(withheld.values())
        )
    return removed, withheld


#: The Messages API's MCP connector beta. Without it ``mcp_servers`` is a
#: 400 ("this parameter requires anthropic-beta"), which every SDK-lane loop
#: with MCP tools hit; the newest version the API names, verified live.
MCP_CONNECTOR_BETA = "mcp-client-2026-09-15"


def _withheld_note(withheld: Mapping[str, str]) -> str:
    """What a read-only run tells a loop about the declared tools it withheld.

    Without it the loop is simply never offered them, and a model asked to
    "publish the brief" reported it published. The same text as the TS
    helper's withheldNote.
    """
    return (
        "This run is read-only, so these tools are not available to you because "
        f"they may write: {', '.join(sorted(withheld))}. Do not say you used them or that "
        "their part of the task is done: say plainly which part you could not do."
    )


#: The tool an agent loop calls once, at the end, with its declared output
#: (rote.adapters._common.AGENT_SUBMIT_TOOL; the TS twin uses the same name).
_SUBMIT_TOOL = "submit_result"


def _agent_task_prompt(
    *,
    description: str,
    task: str,
    termination: str | None,
    local_tool_names: Sequence[str],
    submit: str | None = None,
) -> str:
    parts = [description.strip(), "", task.strip()]
    if local_tool_names:
        parts += ["", "Pipeline steps available to you as tools: " + ", ".join(local_tool_names)]
    if termination:
        parts += ["", f"Stop when: {termination}"]
    if submit == "tool":
        parts += [
            "",
            f"When you are done, call the {_SUBMIT_TOOL} tool exactly once with the "
            "final answer: its arguments are this step's output, and the next steps "
            "read them. Do not ask follow-up questions — nobody is available to answer.",
        ]
    elif submit == "schema":
        parts += [
            "",
            "When you are done, give the final answer in the required structured "
            "form: it is this step's output, and the next steps read it. Do not ask "
            "follow-up questions — nobody is available to answer.",
        ]
    else:
        parts += [
            "",
            "When you are done, state your final answer as the last message. "
            "Do not ask follow-up questions — nobody is available to answer.",
        ]
    return "\n".join(parts)


def _schema_errors(schema: Mapping[str, Any], value: Any, path: str) -> list[str]:
    """Where ``value`` breaks ``schema``: types, required keys, extra keys.

    The subset rote's agent-loop schemas use (``rote.adapters._common
    .ir_type``): ``type``, ``properties`` / ``required`` /
    ``additionalProperties: false``, ``items``, ``anyOf`` and ``enum``. A
    description-only schema (a named type the IR never defines) accepts
    anything. Stdlib only: this module ships verbatim in emitted apps.
    """
    if "anyOf" in schema:
        options = [_schema_errors(option, value, path) for option in schema["anyOf"]]
        if any(not errors for errors in options):
            return []
        return [f"{path}: matches none of its allowed types"]
    if "enum" in schema and value not in schema["enum"]:
        return [f"{path}: {value!r} is not one of {schema['enum']!r}"]
    kind = schema.get("type")
    checks = {
        "string": lambda v: isinstance(v, str),
        "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        "boolean": lambda v: isinstance(v, bool),
        "null": lambda v: v is None,
        "object": lambda v: isinstance(v, dict),
        "array": lambda v: isinstance(v, list),
    }
    if kind is None:
        return []
    if kind in checks and not checks[kind](value):
        return [f"{path}: expected {kind}, got {type(value).__name__}"]
    errors: list[str] = []
    if kind == "object":
        properties = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}: missing key {key!r}")
        for key, item in value.items():
            if key in properties:
                errors.extend(_schema_errors(properties[key], item, f"{path}.{key}"))
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}: unexpected key {key!r}")
    if kind == "array" and isinstance(schema.get("items"), dict):
        for index, item in enumerate(value):
            errors.extend(_schema_errors(schema["items"], item, f"{path}[{index}]"))
    return errors


def _declared_output(
    node_id: str, provider: str, schema: Mapping[str, Any], answer: Any
) -> dict[str, Any]:
    """The loop's final answer, validated against its declared output.

    Downstream nodes index these keys, so a missing key, a wrong type or an
    undeclared key fails here, naming the node, rather than as a KeyError
    or a bad value in a later step. Every lane ends here, with the same
    message as the TS twin.
    """
    if answer is None:
        reason = f"it never called {_SUBMIT_TOOL}, even when asked once more"
        if provider == "claude-cli":
            reason = "the CLI returned no structured output"
        raise RuntimeError(
            f"agent loop {node_id} ({provider}) did not hand back its declared output: {reason}"
        )
    errors = _schema_errors(schema, answer, "output")
    if errors:
        raise RuntimeError(
            f"agent loop {node_id} ({provider}) did not hand back its declared output: "
            + "; ".join(errors)
        )
    return dict(answer)


def _run_agent_via_cli(
    *,
    node_id: str,
    model: str,
    prompt: str,
    tools: Sequence[str],
    tool_servers: Mapping[str, str] | None,
    max_iterations: int,
    timeout: float,
    output_schema: Mapping[str, Any] | None = None,
) -> tuple[Any, list[tuple[str, dict[str, Any]]], int | None]:
    """A bounded agent loop on the Claude subscription.

    Unlike the judge path this one *keeps* its declared MCP tools (that
    is the entire point) but gets the same isolation otherwise: no
    built-in coding tools, no user settings, memory or connectors. With
    ``--strict-mcp-config`` the servers in ``--mcp-config`` are the only
    ones it can see, so the loop gets the pipeline's tools and nothing
    else. With no server resolved it gets no MCP at all.
    """
    binary = shutil.which("claude")
    if binary is None:  # pragma: no cover — select_provider checks first
        raise RuntimeError("the `claude` CLI vanished between provider selection and the call")
    servers, allowed, removed, withheld = _mcp_config_for(tools, tool_servers)
    if withheld:
        prompt = f"{prompt}\n\n{_withheld_note(withheld)}"
    command = [
        binary,
        "-p",
        "--output-format",
        "json",
        "--model",
        _cli_model(model),
        "--system-prompt",
        _AGENT_SYSTEM_PROMPT,
        *_CLI_ISOLATION_FLAGS,
        *_cli_effort_flags(node_id, binary),
        "--max-turns",
        str(max_iterations),
    ]
    if allowed:
        command += ["--allowedTools", ",".join(_tool_id(*pair) for pair in allowed)]
        command += ["--mcp-config", json.dumps({"mcpServers": servers})]
    if removed:
        # Not just left off the allowlist (in print mode an unlisted tool is
        # still offered, and then refused): denied, so the loop never sees
        # it. Every removed pair, including a tool another server still
        # offers under the same name.
        command += ["--disallowedTools", ",".join(_tool_id(*pair) for pair in removed)]
    if output_schema is not None:
        # The same forced structured output a judge uses: the CLI adds a
        # final tool whose arguments are validated against the schema and
        # returns them in `structured_output`.
        command += ["--json-schema", json.dumps(output_schema)]

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

    _refuse_ignored_effort(node_id, completed.stderr)
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
    answer: Any = (
        envelope.get("structured_output")
        if output_schema is not None
        else str(envelope.get("result") or "")
    )
    return answer, _cli_usage_rows(envelope, model), _cli_turns(envelope)


def _cli_turns(envelope: Mapping[str, Any]) -> int | None:
    """Turns the CLI loop actually ran, in the same unit as ``--max-turns``.

    ``num_turns`` is on every result envelope today. When it is missing the
    count is unknown, and unknown is None (as on the TS Workers AI lane),
    not 0: a zero would read as a loop that never ran.
    """
    turns = envelope.get("num_turns")
    return None if turns is None else int(turns)


def _text_result(fn: Any) -> Any:
    """``fn`` as a tool whose result reaches the model as JSON text.

    Pipeline steps return dicts. The SDK's tool runner does not serialize
    a dict result: it iterates it, so the model saw only the keys and none
    of the values. The TS twin JSON-stringifies non-string results the
    same way. ``functools.wraps`` keeps the step's signature and docstring,
    which is what the tool's schema and description are derived from.
    """
    import functools

    @functools.wraps(fn)
    def tool(*args: Any, **kwargs: Any) -> str:
        result = fn(*args, **kwargs)
        return result if isinstance(result, str) else json.dumps(result, default=str)

    return tool


def _run_agent_via_sdk(
    *,
    node_id: str,
    provider: str,
    model: str,
    base_url: str | None,
    prompt: str,
    local_tools: Mapping[str, Any],
    tools: Sequence[str],
    tool_servers: Mapping[str, str] | None,
    max_iterations: int,
    timeout: float,
    output_schema: Mapping[str, Any] | None = None,
    output_strict: bool = False,
) -> tuple[Any, dict[str, Any], int]:
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
    # Runnable tools, then (below) the connector's server-side toolsets.
    bound: list[Any] = [beta_tool(name=name)(_text_result(fn)) for name, fn in local_tools.items()]
    submitted: list[dict[str, Any]] = []
    if output_schema is not None:

        def submit(**answer: Any) -> str:
            submitted.append(answer)
            return "Recorded. The step is complete; end your turn."

        bound.append(
            beta_tool(
                name=_SUBMIT_TOOL,
                description=(
                    "Hand back this step's final output. Call exactly once, when "
                    "the task is done; the next pipeline steps read these fields."
                ),
                input_schema=dict(output_schema),
                # Grammar-constrained arguments when the schema fits the
                # strict subset (rote.adapters._common.strict_compatible).
                strict=True if output_strict else None,
            )(submit)
        )
    servers, allowed, _removed, withheld = _mcp_config_for(tools, tool_servers)
    if withheld:
        prompt = f"{prompt}\n\n{_withheld_note(withheld)}"
    # The connector exposes a server's tools through one mcp_toolset per
    # server, and an allowlist is the only shape that holds the loop to the
    # tools it declared (and, in a read-only run, the ones that cannot
    # write): every other tool on the server stays disabled.
    enabled: dict[str, list[str]] = {}
    for server, tool in allowed:
        enabled.setdefault(server, []).append(tool)
    mcp_servers = [
        {"type": "url", "url": servers[name]["url"], "name": name} for name in sorted(enabled)
    ]
    bound += [
        {
            "type": "mcp_toolset",
            "mcp_server_name": name,
            "default_config": {"enabled": False},
            "configs": {tool: {"enabled": True} for tool in sorted(enabled[name])},
        }
        for name in sorted(enabled)
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
        kwargs["betas"] = [MCP_CONNECTOR_BETA]
    effort = node_effort(node_id)
    if effort is not None:
        kwargs["output_config"] = {"effort": effort}
    # Drive the runner ourselves before until_done(): each yielded item is
    # one assistant message, i.e. one turn that actually ran. That count is
    # what the result reports, never the cap, which would make every loop
    # look saturated (the TS twin counts the same thing). Usage is summed
    # over every turn for the same reason: the final message's usage covers
    # only the final request. until_done() on the drained runner sends
    # nothing more and returns the turn the run ended on.
    turns = 0
    usage: dict[str, Any] = _anthropic_usage(None)
    try:
        runner = client.beta.messages.tool_runner(**kwargs)
        for message in runner:
            turns += 1
            turn_usage = _anthropic_usage(getattr(message, "usage", None))
            usage = {key: _add_tokens(usage[key], turn_usage[key]) for key in usage}
        final = runner.until_done()
        if output_schema is not None and not submitted:
            # The loop ended (or hit its cap) without handing anything back.
            # One more turn, schema-locked the way a judge is whenever the
            # schema allows it, before failing.
            request_schema = _lockable_schema(output_schema, node_id)
            forced = _force_submission(client, runner, final, kwargs, bound, request_schema)
            turns += 1
            turn_usage = _anthropic_usage(getattr(forced, "usage", None))
            usage = {key: _add_tokens(usage[key], turn_usage[key]) for key in usage}
            if request_schema is not None:
                submitted.append(structured_answer(forced, node_id=node_id))
            else:
                submitted.extend(
                    dict(block.input)
                    for block in getattr(forced, "content", [])
                    if getattr(block, "type", "") == "tool_use" and block.name == _SUBMIT_TOOL
                )
    except anthropic.APIError as exc:
        raise _durable_error(exc, node_id=node_id, provider=provider, model=model) from None

    if output_schema is not None:
        # The last submission wins: a model that corrects itself means it.
        return (submitted[-1] if submitted else None), usage, turns
    text = "\n".join(
        b.text for b in getattr(final, "content", []) if getattr(b, "type", "") == "text"
    )
    return text, usage, turns


def _assistant_param(message: Any) -> dict[str, Any]:
    """A response message as a request message param, on any SDK version.

    ``Message.to_param()`` exists only in newer SDKs, and an emitted app
    runs on whatever version it installed. A content block dumps to its
    wire shape (``{"type": "tool_use", "id", "name", "input"}`` and so on);
    ``parsed_output`` is the SDK's own parse of a text block, never sent.
    """
    return {
        "role": message.role,
        "content": [
            block.model_dump(mode="json", exclude_none=True, exclude={"parsed_output"})
            for block in message.content
        ],
    }


def _force_submission(
    client: Any,
    runner: Any,
    final: Any,
    kwargs: Mapping[str, Any],
    bound: Sequence[Any],
    request_schema: Mapping[str, Any] | None,
) -> Any:
    """One more request for the step's output, schema-locked when possible.

    Never ``tool_choice`` forced to the submit tool: claude-sonnet-5-5
    rejects forced tool choice outright. With a ``request_schema`` (the
    output schema rewritten by :func:`structured_output_schema`),
    ``tool_choice: none`` keeps the tools defined, which the history's
    ``tool_use`` blocks require, while ``output_config.format`` makes the
    text answer the output object: the mechanism :func:`_call_anthropic`
    uses. Without one (an output naming a type the IR never defines has no
    JSON type, so nothing can lock it) the turn only asks the model to call
    the submit tool, and :func:`_declared_output` reports it if it does not.

    The conversation is the runner's, read through its public
    ``set_messages_params``. A turn that ended the loop normally
    (``end_turn``) is not in it yet, so it is appended. A turn that hit the
    iteration cap ended on ``tool_use``, and the runner has already run
    those calls and appended the turn with their results, so the nudge
    joins that last user message instead: nothing runs twice.
    """
    history: list[Any] = []

    def _capture(params: Any) -> Any:
        history.extend(params["messages"])
        return params

    runner.set_messages_params(_capture)
    ask = (
        "Reply now with the step's final answer."
        if request_schema is not None
        else f"Call {_SUBMIT_TOOL} now with the final answer."
    )
    nudge = {"type": "text", "text": ask}
    if getattr(final, "stop_reason", None) == "tool_use" and history[-1].get("role") == "user":
        last = history[-1]
        content = last["content"]
        blocks = list(content) if isinstance(content, list) else [{"type": "text", "text": content}]
        messages = [*history[:-1], {**last, "content": [*blocks, nudge]}]
    else:
        messages = [*history, _assistant_param(final), {"role": "user", "content": [nudge]}]
    request = {key: value for key, value in kwargs.items() if key not in ("max_iterations",)}
    # A runnable tool serializes itself; a connector toolset is already a param.
    request["tools"] = [tool if isinstance(tool, dict) else tool.to_dict() for tool in bound]
    if request_schema is not None:
        request["tool_choice"] = {"type": "none"}
        # Merged, so the loop's effort carries into its last turn.
        request["output_config"] = {
            **request.get("output_config", {}),
            "format": {"type": "json_schema", "schema": request_schema},
        }
    request["messages"] = messages
    return client.beta.messages.create(**request)


def _lockable_schema(output_schema: Mapping[str, Any], node_id: str) -> dict[str, Any] | None:
    """``output_schema`` for ``output_config.format``, or None when the
    subset cannot express it."""
    try:
        return structured_output_schema(output_schema, node_id=node_id)
    except RuntimeError as exc:
        if STRUCTURED_INEXPRESSIBLE not in str(exc):
            raise
        return None


def _add_tokens(total: int | None, more: Any) -> int | None:
    """Sum a per-turn token count, keeping "never reported" as None."""
    if more is None:
        return total
    return (total or 0) + int(more)


def run_agent_loop(
    *,
    node_id: str,
    description: str,
    task: str,
    model: str,
    tools: Sequence[str] = (),
    tool_servers: Mapping[str, str] | None = None,
    local_tools: Mapping[str, Any] | None = None,
    max_iterations: int = 10,
    termination: str | None = None,
    base_url: str | None = None,
    output_schema: Mapping[str, Any] | None = None,
    unwrap_key: str | None = None,
    output_strict: bool = False,
) -> Any:
    """Run one bounded, tool-restricted agent loop and return its output.

    The output is the node's declared output, the same as every other
    node kind returns: ``output_schema`` is the object the loop's final
    answer must be (``rote.adapters._common.agent_loop_output`` derives it
    from the IR), and downstream nodes read its keys. On the SDK lane the
    agent hands it back through a ``submit_result`` tool, on the CLI lane
    as the CLI's forced structured output. ``unwrap_key`` marks a
    single-type output, returned as the bare value. A node that declares
    no output gets ``{"result": <final text>}``. The provider and the
    turns that ran go to the usage log.

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
        submit=(
            None if output_schema is None else "schema" if provider == "claude-cli" else "tool"
        ),
    )

    if provider == "claude-cli":
        answer, rows, turns = _run_agent_via_cli(
            node_id=node_id,
            model=model,
            prompt=prompt,
            tools=tools,
            tool_servers=tool_servers,
            max_iterations=max_iterations,
            timeout=timeout,
            output_schema=output_schema,
        )
    else:
        answer, usage, turns = _run_agent_via_sdk(
            node_id=node_id,
            provider=provider,
            model=model,
            base_url=base_url,
            prompt=prompt,
            local_tools=local_tools,
            tools=tools,
            tool_servers=tool_servers,
            max_iterations=max_iterations,
            timeout=timeout,
            output_schema=output_schema,
            output_strict=output_strict,
        )
        rows = [(model, usage)]

    for index, (row_model, row_usage) in enumerate(rows):
        _log_usage(
            node_id=node_id,
            provider=provider,
            model=row_model,
            usage=row_usage,
            iterations=turns if index == 0 else None,
        )
    if output_schema is None:
        return {"result": answer}
    declared = _declared_output(node_id, provider, output_schema, answer)
    return declared[unwrap_key] if unwrap_key is not None else declared
