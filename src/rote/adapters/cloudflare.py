"""Cloudflare Workflows adapter — emits a TypeScript ``WorkflowEntrypoint``.

Layer 3 of rote: takes a validated :class:`rote.ir.Pipeline` and emits a
deployable Cloudflare Workflow as TypeScript. The output directory is
``wrangler deploy``-ready: a workflow class file, a fetch handler, typed
LLM signatures backed by Zod + the Anthropic SDK, stubs for deterministic
extracted modules, and the supporting ``wrangler.jsonc`` / ``package.json``
/ ``tsconfig.json``.

Two key design choices vs. the Temporal adapter:

* **Single workflow class, not workflow + activities.** Cloudflare's
  programming model is a class extending ``WorkflowEntrypoint`` whose
  ``run(event, step)`` calls ``step.do(...)`` for each unit of work. There
  is no separate "activity" registration.

* **No BAML runtime in emitted code.** BAML's TS client requires a Rust
  native binary that does not run on Workers (V8 isolates). Signatures
  are emitted as Zod schemas + direct Anthropic SDK calls with
  structured-output tool use. The IR's ``signature_spec`` (JSON Schema +
  prompt) is the cross-language source of truth.

The emitted code never imports MCP runtime — same architectural invariant
as the Temporal adapter, enforced by AST tests.
"""

from __future__ import annotations

import json
import re
import textwrap
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from rote.adapters._common import (
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_OPENAI_MODEL,
    EmitResult,
    EmitWriter,
    _execution_waves,
    _pipeline_hash,
    _to_camel_case,
    check_input_refs_available,
    fan_out_nodes,
    pipeline_identity,
    safe_block_comment_line,
    workflow_class_name,
)
from rote.adapters._common import (
    ir_duration_to_human as _ir_duration_to_cf,
)
from rote.adapters._ts_common import (
    AI_UTILS_NPM_VERSION,
    ANTHROPIC_SDK_NPM_VERSION,
    FAN_OUT_LIST_HELPER_TS,
    MCP_SDK_NPM_VERSION,
    ROTE_INFERENCE_HELPER_TS,
    ROTE_MCP_BINDING_HELPER_TS,
    ROTE_MCP_WORKERS_HELPER_TS,
    WORKERS_AI_AGENT_MODEL,
    agent_tool_servers,
    emit_agent_loop_module,
    emit_ts_signature_module,
    emit_workers_mcp_call_module,
    fan_out_ts_binding,
    llm_clients,
    mcp_backed_nodes,
    module_imports,
    override_env_vars,
    payload_ts_literal,
)
from rote.ir import Node, NodeKind, Pipeline, parse_input_ref

# ───────── Adapter configuration ─────────


@dataclass(frozen=True)
class CloudflareAdapterConfig:
    """Per-emission knobs for the Cloudflare adapter.

    Defaults work out-of-the-box for the BDR example. Production users
    will typically override ``workflow_binding`` and the model defaults.
    """

    workflow_binding: str = "PIPELINE"
    compatibility_date: str = "2026-04-25"
    anthropic_default_model: str = DEFAULT_ANTHROPIC_MODEL
    openai_default_model: str = DEFAULT_OPENAI_MODEL
    external_backend: Literal["mcp", "api"] = "mcp"
    """"mcp" (default): external_call nodes with an ``mcp:`` binding emit a
    working call authenticated from provisioned Worker secrets
    (`rote mcp export`) with KV-cached tokens. "api": direct-SDK stubs."""
    mcp_client: Literal["direct", "binding"] = "direct"
    """"direct" (default): emitted MCP code carries its own OAuth refresh —
    per-server Worker secrets plus the ROTE_MCP_TOKENS KV cache. "binding":
    the emitted helper delegates every MCP operation to a
    platform-provisioned ``ROTE_MCP`` service binding — no secrets, no
    token store, no MCP SDK dependency."""
    # Defaults use IR shorthand (5m / 7d) so they round-trip through
    # ``_ir_duration_to_cf`` without re-conversion.
    default_step_timeout: str = "10m"
    default_hitl_timeout: str = "7d"
    default_step_retry_delay: str = "5s"


# ───────── Misc helpers ─────────


_SIGNAL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_VALID_BACKOFFS = {"constant", "linear", "exponential"}


def _validate_signal_name(name: str, node_id: str) -> None:
    """Cloudflare's waitForEvent ``type`` field accepts only [A-Za-z0-9_-].

    The IR now pins ``signal`` to a valid identifier (a strict subset of
    this pattern), so a validated pipeline never trips this check. It is
    retained as defense-in-depth tied to Cloudflare's specific runtime
    contract: if the IR constraint were ever loosened, emit still fails
    loudly here instead of surfacing a runtime ``workflow.invalid_event_type``.
    """
    if not _SIGNAL_NAME_RE.fullmatch(name):
        raise ValueError(
            f"Cloudflare adapter: hitl_gate {node_id!r} signal {name!r} "
            f"contains invalid characters. Cloudflare waitForEvent types "
            f"must match {_SIGNAL_NAME_RE.pattern!r} (no dots, spaces, etc.)."
        )


def _step_config_literal(node: Node, cfg: CloudflareAdapterConfig) -> str:
    """Render a TS object literal for a step.do(...) config arg.

    Maps the IR's ``RetryPolicy`` onto Cloudflare's ``WorkflowStepConfig``:

    | IR field       | Cloudflare field        |
    |----------------|-------------------------|
    | ``max``        | ``retries.limit``       |
    | ``backoff``    | ``retries.backoff``     |
    | (none)         | ``retries.delay`` (default from cfg) |
    | ``timeout``    | ``timeout``             |

    The IR ``backoff`` enum values match Cloudflare's exactly, so the
    mapping is lossless — unlike Temporal where ``backoff_coefficient``
    is numeric.
    """
    timeout = node.timeout or cfg.default_step_timeout
    parts = [f"timeout: {json.dumps(_ir_duration_to_cf(timeout))}"]
    if node.retry:
        backoff = node.retry.backoff if node.retry.backoff in _VALID_BACKOFFS else "exponential"
        retry_delay = _ir_duration_to_cf(cfg.default_step_retry_delay)
        retry_obj = (
            "retries: { "
            f"limit: {node.retry.max}, "
            f"delay: {json.dumps(retry_delay)}, "
            f"backoff: {json.dumps(backoff)}"
            " }"
        )
        parts.insert(0, retry_obj)
    return "{ " + ", ".join(parts) + " }"


# ───────── Workflow.ts emission ─────────


def _emit_step_call(node: Node, cfg: CloudflareAdapterConfig, *, pass_env: bool) -> str:
    fn_name = _to_camel_case(node.id)
    config = _step_config_literal(node, cfg)
    payload = payload_ts_literal(node, indent=" " * 12)
    args = f"{payload}, this.env" if pass_env else payload
    return (
        f"        const {node.id}_result = await step.do(\n"
        f"            {json.dumps(node.id)},\n"
        f"            {config},\n"
        f"            async () => {fn_name}({args}),\n"
        f"        );\n"
    )


def auth_event_type(server: str) -> str:
    """The waitForEvent type that releases instances parked on ``server``.

    ``rote_auth_<server>`` — Cloudflare event types must match
    ``^[A-Za-z0-9_][A-Za-z0-9-_]*$`` (no dots, no colons), so neither
    the DBOS runtimes' ``rote:auth:`` topic nor Inngest's dotted event
    fits. Collision-safe by prefix: a gate signal named
    ``rote_auth_...`` would be pathological but harmless — the release
    payload simply resumes it like any approval.
    """
    return f"rote_auth_{server}"


def _emit_step_call_mcp_parkable(
    node: Node,
    cfg: CloudflareAdapterConfig,
    *,
    pipeline: Pipeline,
    payload: str | None = None,
    name_suffix: str = "",
) -> str:
    """MCP-backed dispatch: auth failures park the instance durably.

    Three Cloudflare specifics shape this loop: ``NonRetryableError``
    (from ``cloudflare:workflows``) is the retry opt-out — there is no
    should-retry predicate, and a dead credential would otherwise burn
    ``retries.limit`` attempts of delay; ``waitForEvent`` THROWS on
    timeout (default 24h — hence the explicit long timeout), failing
    the run after 30 unreleased days, which is the intended terminal
    behavior; and events sent before the wait starts are buffered
    per-instance, so `rote mcp release` can blast every non-terminal
    instance without a race.

    ``payload`` and ``name_suffix`` are supplied by the fan_out path:
    one element per call, and a step-name fragment (``[${_index}]``)
    that keeps each element's step name unique within the instance.
    """
    fn_name = _to_camel_case(node.id)
    config = _step_config_literal(node, cfg)
    if payload is None:
        payload = payload_ts_literal(node, indent=" " * 24)
    nid = node.id
    if node.mcp is not None:
        # One binding, one server: the release event is known at emit time.
        event_expr = json.dumps(auth_event_type(node.mcp.server))
        release_hint = (
            f"        // `rote mcp release {node.mcp.server}` "
            f"(after re-provisioning secrets)\n"
            f"        // sends the {auth_event_type(node.mcp.server)!r} event to every "
            f"non-terminal instance.\n"
        )
    else:
        # An agent loop reaches whatever servers provide its tools, and
        # ROTE_MCP_SERVERS can add more at run time — so the release event
        # is derived from the failure itself rather than baked in.
        servers = sorted(agent_tool_servers(pipeline))
        fallback = auth_event_type(servers[0]) if servers else "rote_auth_unknown"
        event_expr = f"authEventType(err, {json.dumps(fallback)})"
        release_hint = (
            "        // Agent loop: the parked server names itself in the auth error,\n"
            "        // so `rote mcp release <server>` releases the right instances.\n"
        )
    label = "agent loop" if node.mcp is None else "MCP-backed"
    # A fan_out element's step names carry its index; every step name
    # must be unique within an instance or the elements collapse onto
    # one cached step.
    first_name = f"`{nid}{name_suffix}`" if name_suffix else json.dumps(nid)
    return (
        f"        // ─── {nid} ({label}): park on dead credentials ───\n"
        f"{release_hint}"
        f"        let {nid}_result!: Record<string, unknown>;\n"
        f"        for (let {nid}_attempt = 0; ; {nid}_attempt++) {{\n"
        f"            try {{\n"
        f"                {nid}_result = (await step.do(\n"
        f"                    {nid}_attempt === 0\n"
        f"                        ? {first_name}\n"
        f"                        : `{nid}{name_suffix} (auth retry ${{{nid}_attempt}})`,\n"
        f"                    {config},\n"
        f"                    async () => {{\n"
        f"                        try {{\n"
        f"                            return await {fn_name}({payload}, this.env);\n"
        f"                        }} catch (err) {{\n"
        f"                            // No retry mints a credential — skip the budget.\n"
        f"                            throw isRoteMcpAuthNeeded(err)\n"
        f"                                ? new NonRetryableError(\n"
        f"                                      err instanceof Error ? err.message : String(err),\n"
        f"                                  )\n"
        f"                                : err;\n"
        f"                        }}\n"
        f"                    }},\n"
        f"                )) as Record<string, unknown>;\n"
        f"                break;\n"
        f"            }} catch (err) {{\n"
        f"                if (!stepNeedsAuth(err)) throw err;\n"
        f"                // Retry once immediately (another isolate may have\n"
        f"                // refreshed the KV token cache); park on the second\n"
        f"                // consecutive failure. Fresh step names per attempt.\n"
        f"                if ({nid}_attempt % 2 === 1) {{\n"
        f"                    await step.waitForEvent<any>(\n"
        f"                        `{nid}{name_suffix} auth wait ${{{nid}_attempt}}`,\n"
        f'                        {{ type: {event_expr}, timeout: "30 days" }},\n'
        f"                    );\n"
        f"                }}\n"
        f"            }}\n"
        f"        }}\n"
    )


_STEP_NEEDS_AUTH_HELPER = """\
/** A failed step's error carries the auth signal. Cloudflare serializes
 * step errors across hibernation/replay — class identity is lost, so
 * detection is by `name` with the message text as fallback. */
function stepNeedsAuth(err: unknown): boolean {
    if (isRoteMcpAuthNeeded(err)) return true;
    const msg = err instanceof Error ? err.message : String(err);
    return msg.includes("needs (re)authentication");
}

/** Which server's release event unparks this failure.
 *
 * A step bound to one MCP server knows its event at emit time; an agent
 * loop may reach several (and ROTE_MCP_SERVERS can add more at run time),
 * so the server is read back off the error. The name survives the
 * NonRetryableError re-throw because RoteMcpAuthNeeded puts it in the
 * message — own properties do not reliably cross a step boundary. */
function authEventType(err: unknown, fallback: string): string {
    const direct = (err as { server?: unknown })?.server;
    if (typeof direct === "string" && direct.length > 0) return `rote_auth_${direct}`;
    const msg = err instanceof Error ? err.message : String(err);
    const match = /MCP server '([A-Za-z0-9_-]+)'/.exec(msg);
    return match ? `rote_auth_${match[1]}` : fallback;
}
"""


def _emit_step_call_pure_or_external(node: Node, cfg: CloudflareAdapterConfig) -> str:
    return _emit_step_call(node, cfg, pass_env=False)


def _emit_step_call_llm_judge(node: Node, cfg: CloudflareAdapterConfig) -> str:
    return _emit_step_call(node, cfg, pass_env=True)


def _emit_step_call_agent_loop(node: Node, cfg: CloudflareAdapterConfig) -> str:
    return _emit_step_call(node, cfg, pass_env=True)


def _emit_parallel_step_calls(nodes: list[Node], cfg: CloudflareAdapterConfig) -> str:
    """Emit one wave of independent nodes as a single ``Promise.all``.

    A promise combinator over ``step.do`` is Cloudflare's documented
    concurrency primitive for Workflows — the "Rules of Workflows" good
    example is exactly this shape, and the platform's own visualizer has
    a ``ParallelNode`` for it. There is no documented per-instance cap on
    concurrent steps.

    Deliberately NOT wrapped in an outer ``step.do``. That advice is
    specific to ``Promise.race``/``any``, where losing branches are
    discarded and their cached steps go unconsumed; wrapping a whole wave
    would burn an extra step, subject the aggregate to the 1 MiB
    step-result cap, and collapse per-node retry policies into one.

    Fail-fast (``Promise.all``, not ``allSettled``) is the right semantic
    for a DAG wave: reaching the rejection means that node already
    exhausted its own retries, and every sibling's result is persisted
    under its own step name — so a workflow retry replays the survivors
    from cache instead of re-running them.
    """
    lines = ["        // Independent nodes — dispatched concurrently."]
    lines.append("        const [")
    for n in nodes:
        lines.append(f"            {n.id}_result,")
    lines.append("        ] = await Promise.all([")
    for n in nodes:
        pass_env = n.kind in (NodeKind.LLM_JUDGE, NodeKind.AGENT_LOOP)
        # Closing brace lines up with its `async () =>`, as in the
        # single-node form (which passes its own call indent).
        payload = payload_ts_literal(n, indent=" " * 16)
        args = f"{payload}, this.env" if pass_env else payload
        lines.append("            step.do(")
        lines.append(f"                {json.dumps(n.id)},")
        lines.append(f"                {_step_config_literal(n, cfg)},")
        lines.append(f"                async () => {_to_camel_case(n.id)}({args}),")
        lines.append("            ),")
    lines.append("        ]);")
    return "\n".join(lines) + "\n"


def _emit_fan_out_parallel(node: Node, cfg: CloudflareAdapterConfig, *, pipeline: Pipeline) -> str:
    """Emit a plain ``fan_out`` node as one ``step.do`` per element.

    Same ``Promise.all`` reasoning as a parallel wave: it is
    Cloudflare's documented concurrency primitive, each element persists
    under its own step name, and a rejection means that element already
    exhausted its own retries.

    Step names are index-suffixed. Cloudflare caches a step's result by
    name, so reusing one name for every element would return the first
    element's result for all of them — the failure would look like a
    correct run with suspiciously uniform output.
    """
    payload, list_expr = fan_out_ts_binding(node, pipeline, indent=" " * 12)
    pass_env = node.kind in (NodeKind.LLM_JUDGE, NodeKind.AGENT_LOOP)
    args = f"{payload}, this.env" if pass_env else payload
    return "\n".join(
        [
            f"        // fan_out: {node.id} runs once per element of the bound list,",
            "        // each as its own durable step (names are index-suffixed —",
            "        // Cloudflare caches step results by name).",
            f"        const {node.id}_result = await Promise.all(",
            f"            {list_expr}.map((_item, _index) =>",
            "                step.do(",
            f"                    `{node.id}[${{_index}}]`,",
            f"                    {_step_config_literal(node, cfg)},",
            f"                    async () => {_to_camel_case(node.id)}({args}),",
            "                ),",
            "            ),",
            "        );",
        ]
    )


def _emit_fan_out_parkable(node: Node, cfg: CloudflareAdapterConfig, *, pipeline: Pipeline) -> str:
    """Emit a parkable ``fan_out`` node: one element per iteration.

    Sequential, for the same reason a parkable step leaves its parallel
    wave — it can suspend on ``waitForEvent``, whose behavior inside a
    promise combinator is undocumented and whose timeout *throws*, which
    would reject every other element.
    """
    payload, list_expr = fan_out_ts_binding(node, pipeline, indent=" " * 8)
    inner = _emit_step_call_mcp_parkable(
        node, cfg, pipeline=pipeline, payload=payload, name_suffix="[${_index}]"
    )
    return "\n".join(
        [
            f"        // fan_out: {node.id} runs once per element of the bound list.",
            "        // Parkable steps stay sequential — a waitForEvent inside a",
            "        // promise combinator would reject every sibling on timeout.",
            f"        const {node.id}_results: Record<string, unknown>[] = [];",
            f"        for (const [_index, _item] of {list_expr}.entries()) {{",
            textwrap.indent(inner.rstrip("\n"), "    "),
            f"            {node.id}_results.push({node.id}_result);",
            "        }",
            f"        const {node.id}_result = {node.id}_results;",
        ]
    )


def _emit_hitl_gate(node: Node, cfg: CloudflareAdapterConfig) -> str:
    assert node.signal is not None
    _validate_signal_name(node.signal, node.id)
    timeout = node.timeout or cfg.default_hitl_timeout
    timeout_cf = _ir_duration_to_cf(timeout)
    return (
        f"        // ─── HITL gate: {node.id} ───\n"
        f"        // Workflow suspends here until an event of type {node.signal!r}\n"
        f"        // arrives. Survives hibernation; events that arrive before this\n"
        f"        // line is reached are buffered and delivered when reached.\n"
        f"        const {node.id}_event = await step.waitForEvent<any>(\n"
        f"            {json.dumps(node.id)},\n"
        f"            {{ type: {json.dumps(node.signal)}, timeout: {json.dumps(timeout_cf)} }},\n"
        f"        );\n"
        f"        const {node.id}_result = {node.id}_event.payload;\n"
    )


def emit_workflow(pipeline: Pipeline, cfg: CloudflareAdapterConfig | None = None) -> str:
    """Render the workflow.ts source for a pipeline."""
    cfg = cfg or CloudflareAdapterConfig()
    class_name = workflow_class_name(pipeline)
    pipeline_h = _pipeline_hash(pipeline)
    waves = _execution_waves(pipeline)
    mcp_backed = bool(mcp_backed_nodes(pipeline, cfg.external_backend))
    # Anything that can park on an auth failure needs the helpers and the
    # NonRetryableError import — an agent loop's tools are MCP tools.
    parks_on_auth = mcp_backed or any(n.tools for n in pipeline.nodes_by_kind(NodeKind.AGENT_LOOP))

    if mcp_backed and cfg.mcp_client == "binding":
        arch_note = (
            " * Architecture note: deterministic steps wrap functions from the\n"
            " * `extracted/` modules; MCP-backed steps call the tool the source\n"
            " * skill used through the platform's `ROTE_MCP` service binding,\n"
            " * which owns endpoints and credentials. When the platform reports\n"
            " * a dead connection the instance parks durably on a\n"
            " * `rote_auth_<server>` event until it is re-authorized."
        )
    elif mcp_backed:
        arch_note = (
            " * Architecture note: deterministic steps wrap functions from the\n"
            " * `extracted/` modules; MCP-backed steps call the tool the source\n"
            " * skill used, over Streamable HTTP, authenticated from provisioned\n"
            " * Worker secrets. When a credential is missing or dead the instance\n"
            " * parks durably on a `rote_auth_<server>` event — re-provision with\n"
            " * `rote mcp export`, then `rote mcp release <server>` wakes it."
        )
    else:
        arch_note = (
            " * Architecture note: every external_call step in this workflow wraps a\n"
            " * deterministic API call from the `extracted/` modules. None of them call\n"
            " * MCP tools at runtime — those calls were compiled into direct API calls\n"
            " * during the rote emission step."
        )
    # The header f-string below is dedent()ed AFTER interpolation; match
    # the template's 8-space indentation or dedent becomes a no-op.
    arch_note = textwrap.indent(arch_note, " " * 8)

    park_import = (
        '\nimport { NonRetryableError } from "cloudflare:workflows";' if parks_on_auth else ""
    )

    header = textwrap.dedent(
        f"""\
        /**
         * Auto-generated by rote.adapters.cloudflare.
         *
         * Pipeline: {pipeline.name} v{pipeline.version}
         * Source skill: {pipeline.source_skill or "unknown"}
         * Pipeline hash: {pipeline_h}
         *
         * DO NOT EDIT BY HAND. Re-run `rote emit --runtime cloudflare` to regenerate.
         *
{arch_note}
         */

        import {{
            WorkflowEntrypoint,
            WorkflowEvent,
            WorkflowStep,
        }} from "cloudflare:workers";{park_import}

        """
    )

    imports = module_imports(pipeline)
    if parks_on_auth:
        if cfg.mcp_client == "binding":
            # The Env interface names the platform proxy's structural type.
            imports += (
                '\nimport { isRoteMcpAuthNeeded, type RoteMcpBinding } from "./extracted/_roteMcp";'
            )
        else:
            imports += '\nimport { isRoteMcpAuthNeeded } from "./extracted/_roteMcp";'

    # The Env interface carries only the credentials the pipeline's judges
    # actually use: an API key per SDK client, and the Workers AI binding
    # (`AI`) when any judge runs on Workers AI (no key — the binding is the
    # auth). A judge-free pipeline declares no credentials at all.
    clients = llm_clients(pipeline)
    env_field_lines: list[str] = []
    if "anthropic" in clients:
        env_field_lines.append("    ANTHROPIC_API_KEY: string;")
    if "openai" in clients:
        env_field_lines.append("    OPENAI_API_KEY: string;")
    if "workers-ai" in clients:
        env_field_lines.append("    AI: Ai;")
    elif pipeline.nodes_by_kind(NodeKind.AGENT_LOOP):
        # Optional: an agent loop offers the Workers AI lane but never
        # requires it — the operator may pay through any other lane.
        env_field_lines.append("    AI?: Ai;")
    if cfg.mcp_client == "binding":
        # Platform-managed MCP: one RPC service binding replaces the whole
        # per-server secret surface and the KV token cache.
        if parks_on_auth:
            env_field_lines.append(
                "    // Platform-managed MCP proxy — an RPC service binding the host provisions."
            )
            env_field_lines.append("    ROTE_MCP: RoteMcpBinding;")
    else:
        for server in sorted(
            {n.mcp.server for n in mcp_backed_nodes(pipeline, cfg.external_backend) if n.mcp}
        ):
            upper = server.upper()
            env_field_lines.append(
                f"    // MCP server {server!r} — provisioned by `rote mcp export {server}`."
            )
            env_field_lines.append(f"    ROTE_MCP_{upper}_REFRESH_TOKEN: string;")
            env_field_lines.append(f"    ROTE_MCP_{upper}_CLIENT_ID: string;")
            env_field_lines.append(f"    ROTE_MCP_{upper}_CLIENT_SECRET?: string;")
            env_field_lines.append(f"    ROTE_MCP_{upper}_TOKEN_ENDPOINT: string;")
            env_field_lines.append(f"    ROTE_MCP_{upper}_URL?: string;")
        if mcp_backed_nodes(pipeline, cfg.external_backend):
            env_field_lines.append(
                "    // Rotated-token cache: npx wrangler kv namespace create rote-mcp-tokens"
            )
            env_field_lines.append("    ROTE_MCP_TOKENS?: KVNamespace;")
    env_field_lines.append(f"    {cfg.workflow_binding}: Workflow<Params>;")
    env_fields = "\n".join(env_field_lines)
    env_block = (
        "\nexport interface Env {\n"
        f"{env_fields}\n"
        "}\n\n"
        "export type Params = Record<string, unknown>;\n\n"
    )

    body_lines: list[str] = []

    # Bind the instance params once when any top-level node's inputs
    # reference the pipeline input.
    wave_nodes = [n for wave in waves for n in wave]
    needs_pipeline_input = any(
        parse_input_ref(ref).node_id is None
        for n in wave_nodes
        if n.inputs
        for ref in n.inputs.values()
    )
    if needs_pipeline_input:
        body_lines.append("        const pipelineInput = event.payload;")
        body_lines.append("")

    # Node ids whose results are bound by the time each wave starts —
    # used to reject inputs that reference a later wave at emit time.
    available: set[str] = set()

    def _is_parkable(node: Node) -> bool:
        # MCP-backed calls read credentials from env (Worker secrets + the
        # KV token cache) and park the instance durably when the
        # credential is missing or dead.
        if node.kind is NodeKind.AGENT_LOOP:
            # Its tools are MCP tools; bindAgentTools and every tool call
            # inside the loop raise the same auth signal.
            return bool(node.tools)
        return (
            node.kind in (NodeKind.PURE_FUNCTION, NodeKind.EXTERNAL_CALL)
            and node.mcp is not None
            and cfg.external_backend == "mcp"
        )

    for wave_idx, wave in enumerate(waves, start=1):
        body_lines.append(f"        // ─── Wave {wave_idx} ───")

        # A wave's members have no dependency on each other, so they may
        # run in any order or concurrently. Two shapes stay sequential:
        # gates and parkable steps both suspend on `waitForEvent`, whose
        # behavior inside a promise combinator is undocumented — and
        # whose timeout throw would reject the entire wave.
        gates = [n for n in wave if n.kind is NodeKind.HITL_GATE]
        parkable = [n for n in wave if _is_parkable(n)]
        plain = [n for n in wave if n.kind is not NodeKind.HITL_GATE and not _is_parkable(n)]

        # fan_out nodes dispatch once per element of their bound list —
        # they never share the single/parallel payload shapes below, but
        # they keep the parkable/plain split (a parkable fan_out runs its
        # elements sequentially, for the same waitForEvent reason).
        fanned_parkable, parkable = fan_out_nodes(parkable)
        fanned_plain, plain = fan_out_nodes(plain)

        for node in plain + parkable + fanned_plain + fanned_parkable:
            check_input_refs_available(node, available)

        if len(plain) > 1:
            body_lines.append(_emit_parallel_step_calls(plain, cfg).rstrip("\n"))
        elif plain:
            node = plain[0]
            if node.kind is NodeKind.LLM_JUDGE:
                body_lines.append(_emit_step_call_llm_judge(node, cfg).rstrip("\n"))
            elif node.kind is NodeKind.AGENT_LOOP:
                body_lines.append(_emit_step_call_agent_loop(node, cfg).rstrip("\n"))
            else:
                body_lines.append(_emit_step_call_pure_or_external(node, cfg).rstrip("\n"))

        for node in parkable:
            body_lines.append(
                _emit_step_call_mcp_parkable(node, cfg, pipeline=pipeline).rstrip("\n")
            )

        for node in fanned_plain:
            body_lines.append(_emit_fan_out_parallel(node, cfg, pipeline=pipeline))

        for node in fanned_parkable:
            body_lines.append(_emit_fan_out_parkable(node, cfg, pipeline=pipeline))

        for node in gates:
            body_lines.append(_emit_hitl_gate(node, cfg).rstrip("\n"))

        body_lines.append("")
        available.update(n.id for n in wave)

    # Build return object. Cast `unknown` step results so the workflow's
    # declared return type stays serializable.
    body_lines.append("        return {")
    for exit_id in pipeline.exit_nodes:
        cast = "as Record<string, unknown>"
        body_lines.append(f"            {json.dumps(exit_id)}: {exit_id}_result {cast},")
    body_lines.append("        };")

    body = "\n".join(body_lines)

    class_block = (
        f"export class {class_name} extends WorkflowEntrypoint<Env, Params> {{\n"
        f"    async run(event: WorkflowEvent<Params>, step: WorkflowStep) {{\n"
        f"{body}\n"
        f"    }}\n"
        f"}}\n"
    )

    helper_block = f"{_STEP_NEEDS_AUTH_HELPER}\n" if parks_on_auth else ""
    if any(n.fan_out for n in pipeline.nodes):
        helper_block += f"{FAN_OUT_LIST_HELPER_TS}\n"

    return header + imports + "\n" + env_block + helper_block + class_block


# ───────── index.ts emission ─────────


def emit_index(pipeline: Pipeline, cfg: CloudflareAdapterConfig) -> str:
    class_name = workflow_class_name(pipeline)
    template = textwrap.dedent(
        """\
        /**
         * Auto-generated by rote.adapters.cloudflare.
         *
         * Driver surface for the emitted workflow — the routes `rote run`
         * (and `wrangler dev` users) drive:
         *
         *   POST /  or  /start        create an instance (body = Params)
         *   GET  /healthz             liveness probe
         *   GET  /status/<id>         instance status
         *   POST /event/<id>/<type>   deliver a HITL gate event
         *
         * Replace with your own trigger surface (cron, queue consumer,
         * etc.) as needed.
         */

        import { __CLASS__, type Env, type Params } from "./workflow";

        export { __CLASS__ };

        export default {
            async fetch(req: Request, env: Env): Promise<Response> {
                const url = new URL(req.url);
                if (url.pathname === "/healthz") {
                    return Response.json({ ok: true });
                }
                if (req.method === "POST" && (url.pathname === "/" || url.pathname === "/start")) {
                    const raw = await req.json().catch(() => ({}));
                    const params = (raw ?? {}) as Params;
                    const instance = await env.__BINDING__.create({ params });
                    return Response.json({
                        id: instance.id,
                        status: await instance.status(),
                    });
                }
                let m = url.pathname.match(/^\\/status\\/([^/]+)$/);
                if (m) {
                    const instance = await env.__BINDING__.get(m[1]);
                    return Response.json(await instance.status());
                }
                m = url.pathname.match(/^\\/event\\/([^/]+)\\/([^/]+)$/);
                if (m && req.method === "POST") {
                    const instance = await env.__BINDING__.get(m[1]);
                    const payload = await req.json().catch(() => ({}));
                    await instance.sendEvent({ type: m[2], payload });
                    return Response.json({ ok: true });
                }
                return new Response("not found", { status: 404 });
            },
        } satisfies ExportedHandler<Env>;
        """
    )
    return template.replace("__CLASS__", class_name).replace("__BINDING__", cfg.workflow_binding)


# ───────── signatures/<id>.ts emission ─────────


def emit_signature_module(node: Node, cfg: CloudflareAdapterConfig) -> str:
    """Emit src/signatures/<node_id>.ts for an llm_judge node.

    Delegates to the shared TS signature emitter
    (:func:`rote.adapters._ts_common.emit_ts_signature_module`) with this
    adapter's identity string and model defaults.
    """
    return emit_ts_signature_module(
        node,
        anthropic_default_model=cfg.anthropic_default_model,
        openai_default_model=cfg.openai_default_model,
        generated_by="rote.adapters.cloudflare",
    )


# ───────── extracted/<id>.ts emission ─────────


def emit_extracted_module(node: Node) -> str:
    """Emit src/extracted/<node_id>.ts — a stub for deterministic Python equivalents.

    Cloudflare workers don't share Python modules with the Temporal
    runtime; users implement these stubs in TypeScript directly (or call
    out to a separately-deployed Python worker via service binding).
    """
    fn_name = _to_camel_case(node.id)
    desc_first = safe_block_comment_line(node.description, fallback=node.id)

    doc: list[str] = ["/**"]
    if node.kind is NodeKind.AGENT_LOOP:
        doc.append(f" * Stub for agent_loop node: {node.id}")
    else:
        doc.append(f" * Stub for {node.kind.value} node: {node.id}")
    doc.extend([" *", f" * {desc_first}"])

    if node.kind is NodeKind.AGENT_LOOP:
        doc.extend(
            [
                " *",
                " * Agent loops require an LLM agent runtime (e.g. the Anthropic Agent SDK",
                " * with bounded iterations). Implement this against the agent harness your",
                " * project already uses — the workflow only cares that the function",
                " * resolves with the loop's terminal output.",
            ]
        )
        if node.tools:
            doc.extend([" *", " * Tools the agent should be allowed to call:"])
            doc.extend(f" *   - {t}" for t in node.tools)
        if node.loop_body:
            doc.extend([" *", " * Loop body sub-nodes (call once per iteration):"])
            doc.extend(f" *   - {sn}" for sn in node.loop_body)
    else:
        doc.extend(
            [
                " *",
                " * Replace this stub with the deterministic API call. Direct vendor SDKs",
                " * are preferred over MCP wrappers — the rote compiler removes the MCP",
                " * layer at emit time, so production code calls Salesforce / HubSpot /",
                " * ZoomInfo / etc. directly.",
            ]
        )

    if node.mandatory:
        doc.extend(
            [
                " *",
                " * MANDATORY: this node was marked mandatory in the source skill.",
                " * The workflow always calls it; do not make it conditional.",
            ]
        )
    if node.constants:
        doc.extend([" *", " * Constants from the source skill (lifted into the IR):"])
        # k is an identifier (IR-validated); json.dumps does NOT escape */, so
        # run the value through safe_block_comment_line to neutralize it.
        doc.extend(
            f" *   {k} = {safe_block_comment_line(json.dumps(v))}"
            for k, v in node.constants.items()
        )

    doc.append(" */")

    body: list[str]
    # Stubs declare Promise<never> — honest for a function that always
    # throws, and `never` is the one type that both satisfies step.do's
    # `Rpc.Serializable<T>` constraint and stays castable at the
    # workflow's data-flow reference sites (see `ref_to_ts_expr`).
    # Note: `Promise<Record<string, unknown>>` would NOT work here —
    # `unknown` values aren't structurally serializable, which breaks
    # step.do overload resolution (verified via the tsc e2e test).
    # Replace the annotation with your concrete output type when you
    # fill in the implementation.
    if node.kind is NodeKind.AGENT_LOOP:
        msg = f'"agent_loop {node.id}: requires an agent runtime — implement me"'
        body = [
            "",
            'import { type Env } from "../workflow";',
            "",
            f"export async function {fn_name}(",
            "    _input: unknown,",
            "    _env: Env,",
            "): Promise<never> {",
            f"    throw new Error({msg});",
            "}",
            "",
        ]
    else:
        msg = f'"{node.kind.value} {node.id}: stub not implemented"'
        body = [
            "",
            f"export async function {fn_name}(_input: unknown): Promise<never> {{",
            f"    throw new Error({msg});",
            "}",
            "",
        ]
    return "\n".join(doc + body)


# ───────── wrangler / package / tsconfig emission ─────────


def emit_wrangler(pipeline: Pipeline, cfg: CloudflareAdapterConfig) -> str:
    class_name = workflow_class_name(pipeline)
    obj = {
        "$schema": "node_modules/wrangler/config-schema.json",
        "name": pipeline.name,
        "main": "src/index.ts",
        "compatibility_date": cfg.compatibility_date,
        "observability": {"enabled": True},
        "workflows": [
            {
                "name": pipeline.name,
                "binding": cfg.workflow_binding,
                "class_name": class_name,
            }
        ],
    }
    # Workers AI judges need the AI binding (inference auth + gateway
    # routing). An agent loop gets it too: the binding is what makes
    # "run this on my own Cloudflare account" available at all, and it
    # bills nothing until a run actually selects that lane.
    if "workers-ai" in llm_clients(pipeline) or pipeline.nodes_by_kind(NodeKind.AGENT_LOOP):
        obj["ai"] = {"binding": "AI"}
    # MCP-backed nodes cache refreshed/rotated OAuth tokens in KV so every
    # isolate sees the latest credentials. Create the namespace with
    # `npx wrangler kv namespace create rote-mcp-tokens` and paste its id.
    # Binding mode has no token cache to provision — the platform's
    # ROTE_MCP service binding owns credentials.
    if cfg.mcp_client != "binding" and mcp_backed_nodes(pipeline, cfg.external_backend):
        obj["kv_namespaces"] = [
            {
                "binding": "ROTE_MCP_TOKENS",
                "id": "REPLACE-run: npx wrangler kv namespace create rote-mcp-tokens",
            }
        ]
    body = json.dumps(obj, indent=2)
    return (
        "// Auto-generated by rote.adapters.cloudflare. Hand-edit at your own risk\n"
        "// (re-running `rote emit --runtime cloudflare` will overwrite).\n"
        f"{body}\n"
    )


def emit_package_json(
    pipeline: Pipeline, *, with_mcp: bool = False, with_agent_loops: bool = False
) -> str:
    obj = {
        "name": pipeline.name,
        "version": pipeline.version,
        "private": True,
        "type": "module",
        "scripts": {
            "deploy": "wrangler deploy",
            "dev": "wrangler dev",
            "typecheck": "tsc --noEmit",
        },
        "dependencies": {
            # The agent loop needs beta.messages.toolRunner, which the
            # older judge-era pin predates.
            "@anthropic-ai/sdk": ANTHROPIC_SDK_NPM_VERSION,
            "openai": "^6.0.0",
            "zod": "^4.0.0",
            **({"@modelcontextprotocol/sdk": MCP_SDK_NPM_VERSION} if with_mcp else {}),
            # Cloudflare's embedded function-calling toolkit — the
            # workers-ai lane's tool runner.
            **({"@cloudflare/ai-utils": AI_UTILS_NPM_VERSION} if with_agent_loops else {}),
        },
        "devDependencies": {
            "@cloudflare/workers-types": "^5.20260710.1",
            "typescript": "^5.6.0",
            "wrangler": "^4.85.0",
        },
    }
    return json.dumps(obj, indent=2) + "\n"


def emit_tsconfig() -> str:
    obj = {
        "compilerOptions": {
            "target": "ES2022",
            "lib": ["ES2022"],
            "module": "ES2022",
            "moduleResolution": "Bundler",
            "strict": True,
            "skipLibCheck": True,
            "esModuleInterop": True,
            "isolatedModules": True,
            "verbatimModuleSyntax": False,
            "noEmit": True,
            "types": ["@cloudflare/workers-types"],
        },
        "include": ["src/**/*.ts"],
    }
    return json.dumps(obj, indent=2) + "\n"


# ───────── README.md + .dev.vars.example emission ─────────

# "Deploy to Cloudflare" button spec, per
# https://developers.cloudflare.com/workers/platform/deploy-buttons/:
# a markdown image link whose target is the deploy service with the
# public GitHub/GitLab repo URL passed via the ``url`` query parameter.
# The repo URL is unknowable at emission time, so the emitted README
# carries an explicit placeholder for the user to substitute.
_DEPLOY_BUTTON_IMAGE = "https://deploy.workers.cloudflare.com/button"
_DEPLOY_BUTTON_BASE = "https://deploy.workers.cloudflare.com/?url="
_REPO_URL_PLACEHOLDER = "REPLACE-WITH-YOUR-REPO-URL"


def _secret_names(pipeline: Pipeline) -> list[str]:
    """API-key secrets the emitted worker reads, in emission order.

    One per SDK-backed client actually used: ``ANTHROPIC_API_KEY`` for the
    anthropic client, ``OPENAI_API_KEY`` for openai. The workers-ai client
    needs no secret — it authenticates via the ``AI`` binding, not a key —
    so it contributes nothing here.
    """
    clients = llm_clients(pipeline)
    secrets: list[str] = []
    if "anthropic" in clients:
        secrets.append("ANTHROPIC_API_KEY")
    if "openai" in clients:
        secrets.append("OPENAI_API_KEY")
    return secrets


def emit_dev_vars_example(pipeline: Pipeline, cfg: CloudflareAdapterConfig | None = None) -> str:
    """Emit ``.dev.vars.example`` — dotenv-format secret declarations.

    The Deploy to Cloudflare flow reads this file to know which secrets
    to prompt for during one-click setup (per the deploy-buttons docs);
    ``wrangler dev`` users copy it to ``.dev.vars`` for local runs.
    """
    lines = [
        "# Copy to .dev.vars for local `wrangler dev`; the Deploy to Cloudflare",
        "# flow reads this file to prompt for secrets during one-click setup.",
        "# For a manual deploy, set each with `npx wrangler secret put <NAME>`.",
    ]
    lines.extend(f"{name}=" for name in _secret_names(pipeline))
    resolved_cfg = cfg or CloudflareAdapterConfig()
    if resolved_cfg.mcp_client == "binding":
        # Platform-managed MCP: no per-server secrets exist to declare.
        mcp_servers: list[str] = []
    else:
        mcp_servers = sorted(
            {
                n.mcp.server
                for n in mcp_backed_nodes(pipeline, resolved_cfg.external_backend)
                if n.mcp
            }
        )
    for server in mcp_servers:
        lines.append("#")
        lines.append(f"# MCP server {server!r} — fill from: rote mcp export {server}")
        upper = server.upper()
        lines.append(f"ROTE_MCP_{upper}_REFRESH_TOKEN=")
        lines.append(f"ROTE_MCP_{upper}_CLIENT_ID=")
        lines.append(f"# ROTE_MCP_{upper}_CLIENT_SECRET=")
        lines.append(f"ROTE_MCP_{upper}_TOKEN_ENDPOINT=")
        lines.append(f"ROTE_MCP_{upper}_URL=")
    judges = [n for n in pipeline.nodes if n.kind is NodeKind.LLM_JUDGE]
    if judges:
        lines.append("#")
        lines.append("# Optional per-judge overrides: swap the model or point at a")
        lines.append("# different endpoint (proxy, gateway, OpenAI-compatible server)")
        lines.append("# without re-emitting.")
        for node in judges:
            model_var, base_var = override_env_vars(node.id)
            lines.append(f"# {model_var}=")
            lines.append(f"# {base_var}=")
    return "\n".join(lines) + "\n"


def emit_readme(pipeline: Pipeline, cfg: CloudflareAdapterConfig) -> str:
    deploy_url = f"{_DEPLOY_BUTTON_BASE}{_REPO_URL_PLACEHOLDER}"
    button = f"[![Deploy to Cloudflare]({_DEPLOY_BUTTON_IMAGE})]({deploy_url})"

    description = pipeline.description.strip()
    description_block = f"\n{description}\n" if description else ""

    secrets = _secret_names(pipeline)
    secret_puts = "\n".join(f"npx wrangler secret put {name}" for name in secrets)

    gates = [n for n in pipeline.nodes if n.kind is NodeKind.HITL_GATE]
    first_signal = gates[0].signal if gates and gates[0].signal else "example_signal"
    gate_lines = "\n".join(
        f"| `{g.id}` | `{g.signal}` | {_ir_duration_to_cf(g.timeout or cfg.default_hitl_timeout)} |"
        for g in gates
    )

    mcp_servers = sorted(
        {
            n.mcp.server
            for n in mcp_backed_nodes(pipeline, cfg.external_backend)
            if n.mcp is not None
        }
    )
    if mcp_servers and cfg.mcp_client == "binding":
        example_event = auth_event_type(mcp_servers[0])
        mcp_note = f"""
## MCP-backed steps: platform-managed connections

Some `external_call` steps call MCP tools through the platform's
`ROTE_MCP` service binding — the hosting platform owns endpoints,
credentials, and token refresh, so this build carries no MCP secrets
of its own. If a connection is missing or revoked at run time, the
instance does **not** fail — it parks durably on a `{example_event}`
event until the connection is re-authorized on the platform, which
then releases every parked instance.
"""
    elif mcp_servers:
        example_server = mcp_servers[0]
        example_event = auth_event_type(example_server)
        export_lines = "\n".join(
            f"rote mcp login {s} && rote mcp export {s} --json | npx wrangler secret bulk"
            for s in mcp_servers
        )
        mcp_note = f"""
## MCP-backed steps: authentication and parking

Some `external_call` steps call MCP tools over Streamable HTTP,
authenticated from Worker secrets provisioned by `rote mcp export`
(see `.dev.vars.example`). If a credential is missing or dead at run
time, the instance does **not** fail — it parks durably on a
`{example_event}` waitForEvent (events sent early are buffered, so a
release can never race the park). To fix and release:

```sh
{export_lines}
CLOUDFLARE_API_TOKEN=... CLOUDFLARE_ACCOUNT_ID=... rote mcp release {example_server}
```

`rote mcp release` sends the event to every non-terminal instance via
the Cloudflare API. In local dev (`wrangler dev`), send it by hand:

```sh
npx wrangler workflows instances send-event {pipeline.name} <id> \\
    --type {example_event} --local
```

An instance parked longer than 30 days fails the run. Prefer direct
vendor-SDK calls? Re-emit with `rote emit --runtime cloudflare
--backend api`.
"""
    else:
        mcp_note = ""

    # Built flush-left (not textwrap.dedent) because interpolated
    # multi-line values — the pipeline description, gate table rows —
    # contain unindented lines that would defeat dedent's common-prefix
    # detection.
    return f"""\
# {pipeline.name} — Cloudflare Workflows runtime

Auto-generated by `rote emit --runtime cloudflare`. Do not edit
generated files by hand; re-run the emitter to regenerate.
{description_block}
## Deploy to Cloudflare

{button}

Push this directory to a **public GitHub or GitLab repository**,
then replace `{_REPO_URL_PLACEHOLDER}` in the button link above with
the repo URL (a subdirectory path works too). One click clones the
repo into the visitor's account, provisions the workflow, and
prompts for the secrets declared in `.dev.vars.example`.

## Layout

- `src/workflow.ts` — the `WorkflowEntrypoint` class: one `step.do`
  per node, `step.waitForEvent` per HITL gate
- `src/index.ts` — fetch handler that creates a workflow instance
  from the POST body
- `src/extracted/` — stubs for deterministic nodes; fill these in
  with direct vendor API calls (they throw until you do)
- `src/signatures/` — typed LLM judges generated from the pipeline
  IR (Zod schemas + direct vendor SDK calls)
- `wrangler.jsonc` / `package.json` / `tsconfig.json` — deploy-ready
  Workers project config

## Deploy from this machine

```sh
npm install
{secret_puts}
npx wrangler deploy
```

## Trigger a run

POST the pipeline input to the deployed worker; it responds with
the new instance's id and status:

```sh
curl -X POST https://{pipeline.name}.<your-subdomain>.workers.dev \\
    -H 'content-type: application/json' \\
    -d '{{"your": "input"}}'
```

Or trigger directly through wrangler:

```sh
npx wrangler workflows trigger {pipeline.name} '{{"your": "input"}}'
```

## HITL gates

The workflow parks durably at each gate until an event of the
gate's type arrives:

| Gate | Event type | Timeout |
| --- | --- | --- |
{gate_lines}

Resume a parked instance by sending the event (`latest` targets the
most recent instance):

```sh
npx wrangler workflows instances send-event {pipeline.name} latest \\
    --type {first_signal} --payload '{{"approved": true}}'
```

A gate that times out fails the run — silence is not approval.
{mcp_note}
## Trigger from Claude (MCP)

`rote` can expose this deployed pipeline as an MCP tool so any MCP
client triggers the deterministic workflow instead of re-running
the fuzzy skill:

```sh
rote register <compiled-output-dir> \\
    --runtime cloudflare \\
    --url https://{pipeline.name}.<your-subdomain>.workers.dev
rote serve
```

See `docs/mcp-trigger.md` in the rote repository for the full flow.
"""


# ───────── manifest.json emission ─────────

MANIFEST_SCHEMA_VERSION = 1


def _manifest_node_ids(pipeline: Pipeline) -> list[str]:
    """Node ids the cloud runner maps to emitted modules.

    Mirrors ``rote-cloud/upload.mjs``'s derivation exactly: the basenames
    under ``src/signatures/`` (llm_judge nodes) followed by those under
    ``src/extracted/`` (every other emitted node). HITL gates emit no
    module, so they contribute no id — matching what the adapter's
    ``emit`` actually writes.
    """
    signatures = [n.id for n in pipeline.nodes if n.kind is NodeKind.LLM_JUDGE]
    extracted = [
        n.id for n in pipeline.nodes if n.kind not in (NodeKind.LLM_JUDGE, NodeKind.HITL_GATE)
    ]
    return signatures + extracted


def emit_manifest(pipeline: Pipeline) -> str:
    """Emit ``manifest.json`` — the machine-readable deploy descriptor.

    The cloud runner (``rote-cloud``) reads this instead of re-deriving
    the pipeline's identity, node set, and input schema from the emitted
    TypeScript. Everything here is the single source of truth those
    consumers agree on:

    * ``schema`` — manifest format version, for forward-compat.
    * ``name`` / ``version`` / ``pipeline_hash`` / ``class_name`` — from
      :func:`~rote.adapters._common.pipeline_identity`, identical to the
      values baked into ``workflow.ts``.
    * ``node_ids`` — the signature + extracted module basenames, matching
      what ``upload.mjs`` derives from the ``src/`` tree today.
    * ``input_schema`` — the pipeline input's JSON Schema (empty object
      when the pipeline declares only field names), the shape the cloud's
      ``/v1/pipelines`` endpoint expects.
    * ``mcp_servers`` — :attr:`rote.ir.Pipeline.required_mcp_servers`
      (server name → sorted ids of the nodes bound to it). Always
      present, in both MCP client modes; ``{}`` when the pipeline makes
      no MCP calls.
    * ``entry`` — the bundler entry point, always ``src/workflow.ts``.
    """
    identity = pipeline_identity(pipeline)
    obj = {
        "schema": MANIFEST_SCHEMA_VERSION,
        "adapter": "cloudflare",
        "name": identity["name"],
        "version": identity["version"],
        "pipeline_hash": identity["pipeline_hash"],
        "class_name": identity["class_name"],
        "node_ids": _manifest_node_ids(pipeline),
        "input_schema": pipeline.input.input_schema or {},
        "mcp_servers": pipeline.required_mcp_servers,
        "entry": "src/workflow.ts",
    }
    return json.dumps(obj, indent=2) + "\n"


# ───────── Adapter facade ─────────


class CloudflareAdapter:
    """Facade that emits a Cloudflare Workflow from a Pipeline IR.

    Output layout::

        out/
            README.md
            .dev.vars.example
            wrangler.jsonc
            package.json
            tsconfig.json
            src/
                index.ts
                workflow.ts
                signatures/<llm_judge_id>.ts
                extracted/<other_id>.ts

    The directory is ``wrangler deploy``-ready once the user fills in
    the extracted/ stubs and signs in to Cloudflare.
    """

    def __init__(self, config: CloudflareAdapterConfig | None = None) -> None:
        self.config = config or CloudflareAdapterConfig()

    def _emit_agent_loop(self, node: Node, pipeline: Pipeline, cfg: CloudflareAdapterConfig) -> str:
        """One agent_loop node's module, in the Workers calling convention.

        A sub-node takes ``env`` exactly when its own emitted module does
        — judges (credentials) and MCP-backed calls (secrets + the KV
        token cache). A plain extracted stub stays one-argument, so this
        mirrors the predicate the workflow's own step calls use.

        The cast is written as ``Parameters<typeof fn>[1]`` rather than a
        named type: a judge's env parameter is an inline object type
        listing exactly the variables that judge reads, so deriving the
        cast from the function keeps it correct when the judge's
        overrides change — and it needs no import from workflow.ts.
        """
        mcp_ids = {n.id for n in mcp_backed_nodes(pipeline, cfg.external_backend)}

        def env_arg(sub: Node) -> str | None:
            if sub.kind is not NodeKind.LLM_JUDGE and sub.id not in mcp_ids:
                return None
            return f"env as Parameters<typeof {_to_camel_case(sub.id)}>[1]"

        return emit_agent_loop_module(
            node,
            pipeline,
            default_model=cfg.anthropic_default_model,
            generated_by="rote.adapters.cloudflare",
            workers=True,
            sub_node_env_arg=env_arg,
            workers_ai_model=WORKERS_AI_AGENT_MODEL,
        )

    def emit_workflow(self, pipeline: Pipeline) -> str:
        return emit_workflow(pipeline, self.config)

    def emit_index(self, pipeline: Pipeline) -> str:
        return emit_index(pipeline, self.config)

    def emit(
        self,
        pipeline: Pipeline,
        output_dir: str | Path,
        mcp_client: Literal["direct", "binding"] | None = None,
    ) -> EmitResult:
        """Emit the pipeline into ``output_dir``.

        ``mcp_client`` overrides the config's MCP client mode for this
        emission: ``"direct"`` (the default) emits self-contained OAuth
        refresh over provisioned Worker secrets + the ROTE_MCP_TOKENS KV
        cache; ``"binding"`` emits the helper variant that delegates
        every MCP operation to a platform-provisioned ``ROTE_MCP``
        service binding (no secrets, no KV, no MCP SDK dependency).
        """
        if mcp_client is None:
            cfg = self.config
        elif mcp_client in ("direct", "binding"):
            cfg = replace(self.config, mcp_client=mcp_client)
        else:
            raise ValueError(f"mcp_client must be 'direct' or 'binding', got {mcp_client!r}")
        writer = EmitWriter(output_dir)

        written: dict[str, Path] = {}

        written["workflow"] = writer.write(
            "src", "workflow.ts", content=emit_workflow(pipeline, cfg)
        )
        written["index"] = writer.write("src", "index.ts", content=emit_index(pipeline, cfg))

        mcp_ids = {n.id for n in mcp_backed_nodes(pipeline, cfg.external_backend)}
        agent_loops = pipeline.nodes_by_kind(NodeKind.AGENT_LOOP)
        for node in pipeline.nodes:
            if node.kind is NodeKind.HITL_GATE:
                continue
            if node.kind is NodeKind.AGENT_LOOP:
                written[f"extracted/{node.id}"] = writer.write(
                    "src",
                    "extracted",
                    f"{node.id}.ts",
                    content=self._emit_agent_loop(node, pipeline, cfg),
                )
            elif node.kind is NodeKind.LLM_JUDGE:
                written[f"signatures/{node.id}"] = writer.write(
                    "src",
                    "signatures",
                    f"{node.id}.ts",
                    content=emit_signature_module(node, cfg),
                )
            elif node.id in mcp_ids:
                written[f"extracted/{node.id}"] = writer.write(
                    "src",
                    "extracted",
                    f"{node.id}.ts",
                    content=emit_workers_mcp_call_module(
                        node, generated_by="rote.adapters.cloudflare"
                    ),
                )
            else:
                written[f"extracted/{node.id}"] = writer.write(
                    "src",
                    "extracted",
                    f"{node.id}.ts",
                    content=emit_extracted_module(node),
                )
        # An agent_loop that declares tools needs the MCP helper beside it
        # even when no node carries an `mcp:` binding — Node.tools are MCP
        # tool names.
        needs_mcp = bool(mcp_ids) or any(n.tools for n in agent_loops)
        if needs_mcp:
            helper_src = (
                ROTE_MCP_BINDING_HELPER_TS
                if cfg.mcp_client == "binding"
                else ROTE_MCP_WORKERS_HELPER_TS
            )
            written["extracted/_roteMcp"] = writer.write(
                "src", "extracted", "_roteMcp.ts", content=helper_src
            )
        if agent_loops:
            written["signatures/_roteInference"] = writer.write(
                "src", "signatures", "_roteInference.ts", content=ROTE_INFERENCE_HELPER_TS
            )

        written["wrangler"] = writer.write("wrangler.jsonc", content=emit_wrangler(pipeline, cfg))
        written["package.json"] = writer.write(
            "package.json",
            content=emit_package_json(
                pipeline,
                # The binding variant delegates to the platform's RPC stub
                # and never imports the MCP SDK.
                with_mcp=needs_mcp and cfg.mcp_client != "binding",
                with_agent_loops=bool(agent_loops),
            ),
        )
        written["tsconfig.json"] = writer.write("tsconfig.json", content=emit_tsconfig())
        written["README"] = writer.write("README.md", content=emit_readme(pipeline, cfg))
        written[".dev.vars.example"] = writer.write(
            ".dev.vars.example", content=emit_dev_vars_example(pipeline, cfg)
        )
        # Machine-readable deploy descriptor at the output root — the cloud
        # runner reads this instead of scraping identity out of the TS.
        written["manifest.json"] = writer.write("manifest.json", content=emit_manifest(pipeline))

        writer.finalize()
        return EmitResult(written, writer.preserved)
