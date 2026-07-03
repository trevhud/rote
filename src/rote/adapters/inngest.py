"""Inngest adapter — emits TypeScript Inngest functions from a Pipeline IR.

Layer 3 of rote: takes a validated :class:`rote.ir.Pipeline` and emits a
runnable Inngest app in TypeScript. Inngest is the "lives inside your
existing app" target: the emitted functions mount into any Next.js /
Express / plain-Node service via the SDK's serve adapters, and the
Inngest platform (or the open-source dev server) drives execution,
retries, and event delivery from outside the process.

Output layout::

    out/
        src/inngest/client.ts       # the Inngest client (app id)
        src/inngest/pipeline.ts     # one createFunction running the DAG waves
        src/index.ts                # minimal Node serve entrypoint (inngest/node)
        src/signatures/<judge>.ts   # typed Zod + vendor-SDK signatures
        src/extracted/<node>.ts     # stubs for pure_function / external_call / agent_loop
        package.json                # inngest sdk + zod + vendor SDKs
        tsconfig.json
        README.md                   # run, mount in Next.js, resume HITL gates

Key design choices, verified against ``inngest`` v4.11.0 (the installed
package's ``.d.ts`` plus a live probe app against ``inngest-cli`` v1.34.0
are the source of truth; inngest.com/docs corroborates):

* **v4 single-options ``createFunction``.** The v4 SDK takes
  ``createFunction({ id, retries, triggers: [{ event }] }, handler)`` —
  two arguments. The older three-argument form
  ``createFunction(opts, { event }, handler)`` throws at runtime in
  v4.11.0 ("Triggers belong in the first argument").

* **Retries are function-level only.** ``StepOptions`` is
  ``{ id, name?, parallelMode? }`` — there is no per-step retry or
  timeout configuration anywhere in the v4 SDK. The IR's per-node
  :class:`rote.ir.RetryPolicy` therefore maps to a single function-level
  ``retries`` value (the max across nodes, clamped to Inngest's 0–20
  range, default 3), and every per-node delta is documented as an
  emitted comment on that node's step — same convention as the other
  adapters' ``retry_on`` gaps. Backoff is managed by Inngest
  (exponential + jitter, not configurable), so ``backoff`` also
  surfaces as a comment.

* **Parallel waves via ``Promise.all``.** Inngest's documented
  parallelism pattern is ``Promise.all`` over ``step.run`` tool calls —
  the executor discovers and schedules the steps concurrently
  (verified in the live probe: parallel steps are ``StepScheduled``
  together). ``Promise.allSettled`` is not the documented pattern here,
  unlike DBOS TS.

* **HITL gates via ``step.waitForEvent``.** The IR ``signal`` becomes
  an event name namespaced by the pipeline (``<pipeline>/<signal>``).
  ``waitForEvent`` returns ``null`` on timeout → the emitted code
  throws ``NonRetriableError``: silence is not approval, and retrying
  the function cannot conjure the missing human approval (the timed-out
  wait is memoized). Inngest does not enforce an event-name charset
  (the docs recommend lowercase dot/slash notation); the adapter pins
  emitted names to ``[A-Za-z0-9_.-]`` with a single ``/`` separator so
  they stay safe in URLs, CEL expressions, and shell commands.

* **``inngest/node`` serve entrypoint.** The most framework-neutral
  mount: ``http.createServer(serve({ client, functions }))``. The
  stable export is ``serve`` (an ``http.RequestListener``);
  ``createServer`` from the same module is marked EXPERIMENTAL in
  v4.11.0, so the emitted entrypoint composes ``serve`` with
  ``node:http`` directly. The README documents the Next.js mount
  (``inngest/next``) as the drop-in alternative — that integration is
  this adapter's whole pitch.

* **IR durations pass through.** ``waitForEvent``'s ``timeout`` takes
  ``ms``-compatible strings ("30s", "5m", "7d"), which is exactly the
  IR's duration shorthand — no conversion layer.

* **Data-flow threading matches the other adapters.** The handler
  binds the trigger event's ``data`` as ``pipelineInput``, every node's
  result (including gate resume payloads) as ``<id>_result``, and each
  step's payload comes from the node's ``inputs:`` bindings via the
  shared reference grammar (:func:`rote.ir.parse_input_ref`). Forward
  references are rejected at emit time by
  :func:`rote.adapters._common.check_input_refs_available`.

* **Typed LLM signatures as Zod + vendor SDK modules.** Emitted via
  :mod:`rote.adapters._ts_common` — identical machinery to the
  Cloudflare and DBOS TS adapters, so all TS runtimes share prompt
  semantics. ``signature_spec`` is required; the legacy Python-path
  ``signature`` form cannot be transpiled to TypeScript.

The emitted code never imports MCP runtime — same architectural
invariant as the other adapters, enforced by a comment-and-string
stripping scan in the tests.
"""

from __future__ import annotations

import json
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path

from rote.adapters._common import (
    EmitWriter,
    _execution_waves,
    _pipeline_hash,
    _to_camel_case,
    check_input_refs_available,
    safe_block_comment_line,
)
from rote.adapters._ts_common import (
    REQUIRE_ENV_HELPER,
    emit_node_tsconfig,
    emit_ts_signature_module,
    judge_env_arg,
    llm_clients,
    module_imports,
    payload_ts_literal,
)
from rote.ir import Node, NodeKind, Pipeline, parse_input_ref

_GENERATED_BY = "rote.adapters.inngest"

# ───────── Adapter configuration ─────────


@dataclass(frozen=True)
class InngestAdapterConfig:
    """Per-emission knobs for the Inngest adapter.

    Defaults work out-of-the-box for the BDR example against the local
    dev server (``npx inngest-cli dev``); production users set the
    event/signing keys via environment variables at deploy time.
    """

    anthropic_default_model: str = "claude-sonnet-4-6"
    openai_default_model: str = "gpt-4.1"
    #: Default port for the emitted Node serve entrypoint (overridable
    #: at runtime via the PORT environment variable).
    serve_port: int = 3000


# ───────── Event-name mapping / validation ─────────
#
# Inngest does not enforce an event-name charset (the SDK types accept
# any string; the docs recommend lowercase dot notation with a `prefix/`
# namespace). The adapter still pins emitted names to a conservative
# pattern so they stay safe in URLs (`POST /e/<key>` bodies), CEL `if`
# expressions, and the shell commands printed in the README. Same
# defense-in-depth stance as the Cloudflare adapter's signal validation.

_EVENT_SEGMENT_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_SIGNAL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _validate_signal_name(name: str, node_id: str) -> None:
    """The IR pins ``signal`` to an identifier (a strict subset of this
    pattern), so a validated pipeline never trips this check. Retained
    as defense-in-depth tied to this adapter's event-name contract: if
    the IR constraint were ever loosened, emit still fails loudly here
    instead of producing an event name that breaks the README's curl
    commands or a CEL expression."""
    if not _SIGNAL_NAME_RE.fullmatch(name):
        raise ValueError(
            f"Inngest adapter: hitl_gate {node_id!r} signal {name!r} contains "
            f"invalid characters. Emitted Inngest event names must match "
            f"{_SIGNAL_NAME_RE.pattern!r} (no dots, spaces, slashes, etc.)."
        )


def _event_prefix(pipeline: Pipeline) -> str:
    """The pipeline-name namespace all emitted event names live under."""
    if not _EVENT_SEGMENT_RE.fullmatch(pipeline.name):
        raise ValueError(
            f"Inngest adapter: pipeline name {pipeline.name!r} contains "
            f"characters unsafe for an Inngest event-name prefix; must match "
            f"{_EVENT_SEGMENT_RE.pattern!r}."
        )
    return pipeline.name


def trigger_event_name(pipeline: Pipeline) -> str:
    """The event that starts a pipeline run.

    Deliberately *not* versioned with the pipeline hash: a regenerated
    pipeline keeps responding to the same trigger event, so senders
    never need updating. (The function id carries the hash instead.)
    """
    return f"{_event_prefix(pipeline)}/run.requested"


def gate_event_name(pipeline: Pipeline, node: Node) -> str:
    """The event a HITL gate waits for: ``<pipeline>/<signal>``."""
    assert node.signal is not None
    _validate_signal_name(node.signal, node.id)
    return f"{_event_prefix(pipeline)}/{node.signal}"


# ───────── Retry mapping ─────────
#
# | IR field       | Inngest mapping                                       |
# |----------------|-------------------------------------------------------|
# | ``max``        | function-level ``retries`` = max across nodes,        |
# |                | clamped to 0–20 (per-step budgets don't exist in v4)  |
# | ``backoff``    | not mapped (Inngest backoff is managed: exponential   |
# |                | + jitter, not configurable) — surfaced as a comment   |
# | ``retry_on``   | not mapped (narrow by throwing NonRetriableError      |
# |                | inside the step) — surfaced as a comment              |
# | ``timeout``    | not mapped (no per-step timeout in v4; the function-  |
# |                | level ``timeouts.finish`` bounds the whole run) —     |
# |                | surfaced as a comment                                 |

_INNGEST_MAX_RETRIES = 20
_INNGEST_DEFAULT_RETRIES = 3


def _function_retries(pipeline: Pipeline) -> int:
    """Function-level ``retries`` for the emitted createFunction.

    Inngest v4 has no per-step retry configuration (``StepOptions`` is
    ``{ id, name?, parallelMode? }``), so the single function-level
    budget must cover the most retry-hungry node. Nodes that declared a
    smaller budget get the surplus documented as a comment; when no
    node declares a retry policy the SDK default (3) is kept rather
    than forcing 0 — Inngest's platform semantics treat retries as the
    baseline for transient HTTP failures, not an opt-in.
    """
    declared = [n.retry.max for n in pipeline.nodes if n.retry is not None]
    if not declared:
        return _INNGEST_DEFAULT_RETRIES
    return min(max(declared), _INNGEST_MAX_RETRIES)


def _node_policy_comment(node: Node, fn_retries: int, indent: str) -> str:
    """Comment block documenting IR policy this runtime can't express per-step."""
    lines: list[str] = []
    if node.retry is not None and node.retry.max != fn_retries:
        lines.append(f"// IR retry policy: max {node.retry.max} ({node.retry.backoff}). Inngest v4")
        lines.append("// retries are function-level only — this step gets the function's")
        lines.append(f"// budget of {fn_retries} with managed exponential backoff + jitter.")
        lines.append("// Throw NonRetriableError inside the step to stop retries early.")
    elif node.retry is not None and node.retry.backoff != "exponential":
        lines.append(f"// IR backoff {node.retry.backoff!r} is not configurable on Inngest —")
        lines.append("// the platform applies managed exponential backoff + jitter.")
    if node.retry is not None and node.retry.retry_on:
        cats = ", ".join(node.retry.retry_on)
        lines.append(f"// retry_on categories from the IR: {cats}. Inngest retries any")
        lines.append("// thrown error; narrow by throwing NonRetriableError for the rest.")
    if node.timeout:
        lines.append(f"// IR timeout {node.timeout}: Inngest v4 has no per-step timeout —")
        lines.append("// enforce inside the implementation (AbortSignal.timeout) if needed.")
    if not lines:
        return ""
    return "\n".join(indent + line for line in lines) + "\n"


# ───────── signatures/<id>.ts emission ─────────


def emit_signature_module(node: Node, cfg: InngestAdapterConfig) -> str:
    """Emit src/signatures/<node_id>.ts for an llm_judge node.

    Delegates to the shared TS signature emitter
    (:func:`rote.adapters._ts_common.emit_ts_signature_module`) with this
    adapter's identity string and model defaults.
    """
    return emit_ts_signature_module(
        node,
        anthropic_default_model=cfg.anthropic_default_model,
        openai_default_model=cfg.openai_default_model,
        generated_by=_GENERATED_BY,
    )


# ───────── extracted/<id>.ts emission ─────────


def emit_extracted_module(node: Node) -> str:
    """Emit src/extracted/<node_id>.ts — a stub for a deterministic node.

    Same scaffolding convention as the other TS adapters: one module per
    node, throwing until the user fills it in with direct vendor API
    calls. The graduation history (MCP origin) is documented in JSDoc
    only — never in executable code.
    """
    fn_name = _to_camel_case(node.id)
    desc_first = safe_block_comment_line(node.description, fallback=node.id)

    doc: list[str] = ["/**"]
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
                " * are preferred over MCP wrappers — the rote graduator removes the MCP",
                " * layer at emit time, so production code calls the vendor APIs directly.",
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
        doc.extend(f" *   {k} = {json.dumps(v)}" for k, v in node.constants.items())

    doc.append(" */")

    # Stubs declare Promise<never> — honest for a function that always
    # throws, and the one type the workflow's data-flow casts
    # (`(<id>_result as Record<...>)["field"]`) always accept. Replace
    # the annotation with your concrete output type when you fill in
    # the implementation.
    if node.kind is NodeKind.AGENT_LOOP:
        msg = f'"agent_loop {node.id}: requires an agent runtime — implement me"'
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


# ───────── src/inngest/client.ts emission ─────────


def emit_client(pipeline: Pipeline) -> str:
    return textwrap.dedent(
        f"""\
        /**
         * Auto-generated by rote.adapters.inngest.
         *
         * The Inngest client for the {pipeline.name} pipeline. In dev the SDK
         * auto-detects the local dev server via INNGEST_DEV=1; in production
         * set INNGEST_EVENT_KEY (for sends) and INNGEST_SIGNING_KEY (for the
         * serve handler) in the environment.
         */

        import {{ Inngest }} from "inngest";

        export const inngest = new Inngest({{ id: {json.dumps(pipeline.name)} }});
        """
    )


# ───────── src/inngest/pipeline.ts emission ─────────


def _step_call_expr(node: Node, payload_indent: str) -> str:
    """The ``step.run("<id>", async () => fn(payload))`` expression, unterminated."""
    fn_name = _to_camel_case(node.id)
    payload = payload_ts_literal(node, indent=payload_indent)
    if node.kind is NodeKind.LLM_JUDGE:
        call = f"{fn_name}({payload}, {judge_env_arg(node)})"
    else:
        call = f"{fn_name}({payload})"
    return f"step.run({json.dumps(node.id)}, async () => {call})"


def _emit_hitl_wait(node: Node, pipeline: Pipeline) -> str:
    assert node.signal is not None
    event_name = gate_event_name(pipeline, node)
    timeout = node.timeout or pipeline.config.hitl.default_timeout
    quoted_event = json.dumps(event_name)
    return (
        f"        // ─── HITL gate: {node.id} ───\n"
        f"        // The run parks durably until an event named {event_name!r}\n"
        f"        // arrives (inngest.send / POST to the event API). waitForEvent\n"
        f"        // returns null on timeout — and a timed-out wait is memoized, so\n"
        f"        // retrying could never conjure the missing approval. Throw\n"
        f"        // NonRetriableError: silence is not approval.\n"
        f"        const {node.id}_event = await step.waitForEvent(\n"
        f"            {json.dumps(node.id)},\n"
        f"            {{ event: {quoted_event}, timeout: {json.dumps(timeout)} }},\n"
        f"        );\n"
        f"        if ({node.id}_event === null) {{\n"
        f"            throw new NonRetriableError(\n"
        f"                \"HITL gate '{node.id}' timed out after {timeout} waiting \" +\n"
        f"                    \"for event '{event_name}'\",\n"
        f"            );\n"
        f"        }}\n"
        f"        const {node.id}_result = {node.id}_event.data as Record<string, unknown>;\n"
    )


def _emit_workflow_body(pipeline: Pipeline, fn_retries: int) -> str:
    """Render the createFunction handler body (waves + return)."""
    waves = _execution_waves(pipeline)
    lines: list[str] = []

    # Bind the trigger event's data once when any top-level node's
    # inputs reference the pipeline input.
    wave_nodes = [n for wave in waves for n in wave]
    needs_pipeline_input = any(
        parse_input_ref(ref).node_id is None
        for n in wave_nodes
        if n.inputs
        for ref in n.inputs.values()
    )
    if needs_pipeline_input:
        lines.append("        const pipelineInput = event.data as Record<string, unknown>;")

    # Node ids whose results are bound by the time each wave starts —
    # used to reject inputs that reference a later wave at emit time.
    # HITL gates count: their resume payload binds `<id>_result`.
    available: set[str] = set()

    for wave_idx, wave in enumerate(waves, start=1):
        non_hitl = [n for n in wave if n.kind is not NodeKind.HITL_GATE]
        hitl = [n for n in wave if n.kind is NodeKind.HITL_GATE]

        for n in non_hitl:
            check_input_refs_available(n, available)

        lines.append("")
        lines.append(f"        // ─── Wave {wave_idx} ───")

        if len(non_hitl) == 1:
            node = non_hitl[0]
            comment = _node_policy_comment(node, fn_retries, indent=" " * 8)
            if comment:
                lines.append(comment.rstrip("\n"))
            expr = _step_call_expr(node, payload_indent=" " * 12)
            lines.append(f"        const {node.id}_result = await {expr};")
        elif len(non_hitl) > 1:
            lines.append("        // Parallel fan-out: Promise.all over step.run calls is")
            lines.append("        // Inngest's documented in-function parallelism pattern —")
            lines.append("        // the executor schedules the steps concurrently.")
            for node in non_hitl:
                comment = _node_policy_comment(node, fn_retries, indent=" " * 8)
                if comment:
                    lines.append(comment.rstrip("\n"))
            result_names = ", ".join(f"{n.id}_result" for n in non_hitl)
            lines.append(f"        const [{result_names}] = await Promise.all([")
            for node in non_hitl:
                expr = _step_call_expr(node, payload_indent=" " * 16)
                lines.append(f"            {expr},")
            lines.append("        ]);")

        for gate in hitl:
            lines.append(_emit_hitl_wait(gate, pipeline).rstrip("\n"))

        available.update(n.id for n in wave)

    lines.append("")
    lines.append("        return {")
    for exit_id in pipeline.exit_nodes:
        lines.append(
            f"            {json.dumps(exit_id)}: {exit_id}_result as Record<string, unknown>,"
        )
    lines.append("        };")
    return "\n".join(lines)


def emit_pipeline_ts(pipeline: Pipeline, cfg: InngestAdapterConfig | None = None) -> str:
    """Render the src/inngest/pipeline.ts source for a pipeline."""
    cfg = cfg or InngestAdapterConfig()

    pipeline_h = _pipeline_hash(pipeline)
    function_id = f"{pipeline.name}-{pipeline_h}"
    fn_retries = _function_retries(pipeline)
    trigger = trigger_event_name(pipeline)
    desc_first = safe_block_comment_line(pipeline.description, fallback=pipeline.name)
    has_gates = any(n.kind is NodeKind.HITL_GATE for n in pipeline.nodes)

    header = textwrap.dedent(
        f"""\
        /**
         * Auto-generated by rote.adapters.inngest.
         *
         * Pipeline: {pipeline.name} v{pipeline.version}
         * Source skill: {pipeline.source_skill or "unknown"}
         * Pipeline hash: {pipeline_h}
         *
         * DO NOT EDIT BY HAND. Re-run `rote emit --runtime inngest` to regenerate.
         *
         * Function versioning: the function id includes a hash of the pipeline
         * so a regenerated pipeline becomes a new Inngest function. The trigger
         * event name stays stable across versions — senders never change.
         *
         * Architecture note: every step in this function wraps a deterministic
         * function from `extracted/` or a typed LLM signature from
         * `signatures/`. None of them call MCP tools at runtime — the MCP
         * tool calls from the source skill were graduated into direct API
         * calls during the rote emission step.
         */

        """
    )

    inngest_imports = (
        'import { NonRetriableError } from "inngest";\n' if has_gates else ""
    ) + 'import { inngest } from "./client";\n'

    imports_block = module_imports(pipeline, prefix="../")

    has_judges = bool(pipeline.nodes_by_kind(NodeKind.LLM_JUDGE))
    helper_block = f"\n{REQUIRE_ENV_HELPER.rstrip(chr(10))}\n" if has_judges else ""

    body = _emit_workflow_body(pipeline, fn_retries)

    retries_comment = (
        "        // Inngest v4 retries are function-level: every step gets this\n"
        "        // budget (managed exponential backoff + jitter). Set to the max\n"
        "        // across the IR's per-node retry policies; per-node deltas are\n"
        "        // documented on the affected steps below.\n"
    )

    function_block = (
        "/**\n"
        f" * {desc_first}\n"
        " */\n"
        "export const runPipeline = inngest.createFunction(\n"
        "    {\n"
        f"        id: {json.dumps(function_id)},\n"
        f"{retries_comment}"
        f"        retries: {fn_retries},\n"
        f"        triggers: [{{ event: {json.dumps(trigger)} }}],\n"
        "    },\n"
        "    async ({ event, step }) => {\n"
        f"{body}\n"
        "    },\n"
        ");\n"
    )

    sections = [header + inngest_imports + imports_block]
    if helper_block:
        sections.append(helper_block.strip("\n"))
    sections.append(function_block)
    return "\n\n".join(sections)


# ───────── src/index.ts emission ─────────


def emit_index(pipeline: Pipeline, cfg: InngestAdapterConfig) -> str:
    return textwrap.dedent(
        f"""\
        /**
         * Auto-generated by rote.adapters.inngest.
         *
         * Minimal framework-neutral serve entrypoint: a plain Node HTTP server
         * wrapping the Inngest serve handler (which answers on every path).
         * This is what the local dev server
         * (`npx inngest-cli dev -u http://localhost:{cfg.serve_port}`) and
         * Inngest Cloud sync against.
         *
         * Mounting the same functions inside an existing Next.js / Express app
         * instead: see README.md — import {{ runPipeline }} and use the
         * framework's serve adapter (`inngest/next`, `inngest/express`).
         */

        import http from "node:http";
        import {{ serve }} from "inngest/node";
        import {{ inngest }} from "./inngest/client";
        import {{ runPipeline }} from "./inngest/pipeline";

        const handler = serve({{ client: inngest, functions: [runPipeline] }});

        const server = http.createServer(handler);

        const port = Number(process.env.PORT ?? {cfg.serve_port});
        server.listen(port, () => {{
            console.log(`inngest functions served on :${{port}} (app: {pipeline.name})`);
        }});
        """
    )


# ───────── package / tsconfig / README emission ─────────


def emit_package_json(pipeline: Pipeline) -> str:
    """Emit package.json with the current SDK majors (verified on npm).

    CommonJS (no ``"type": "module"``): the inngest package ships dual
    CJS/ESM builds, and CJS keeps the emitted output consistent with the
    other TS adapters' output and trivially importable from either
    module system.
    """
    dependencies = {
        "inngest": "^4.11.0",
        "zod": "^4.4.3",
    }
    clients = llm_clients(pipeline)
    if "anthropic" in clients:
        dependencies["@anthropic-ai/sdk"] = "^0.110.0"
    if "openai" in clients:
        dependencies["openai"] = "^6.45.0"
    obj = {
        "name": pipeline.name,
        "version": pipeline.version,
        "private": True,
        "scripts": {
            "build": "tsc",
            "start": "node dist/index.js",
            "typecheck": "tsc --noEmit",
        },
        "dependencies": dict(sorted(dependencies.items())),
        "devDependencies": {
            "@types/node": "^26.1.0",
            "typescript": "^6.0.3",
        },
    }
    return json.dumps(obj, indent=2) + "\n"


def emit_readme(pipeline: Pipeline, cfg: InngestAdapterConfig) -> str:
    trigger = trigger_event_name(pipeline)
    gates = [n for n in pipeline.nodes if n.kind is NodeKind.HITL_GATE]
    gate_lines = "\n".join(
        f"| `{g.id}` | `{gate_event_name(pipeline, g)}` "
        f"| {g.timeout or pipeline.config.hitl.default_timeout} |"
        for g in gates
    )
    first_gate_event = gate_event_name(pipeline, gates[0]) if gates else "app/example.approved"

    # Built flush-left (not textwrap.dedent) because interpolated
    # multi-line values — the gate table rows — contain unindented lines
    # that would defeat dedent's common-prefix detection. Same fix the
    # DBOS adapters' emit_readme shipped after that exact bug.
    return f"""\
# {pipeline.name} — Inngest runtime

Auto-generated by `rote emit --runtime inngest`. Do not edit generated
files by hand; re-run the emitter to regenerate.

Inngest is the "inside your existing app" target: the pipeline is a
durable Inngest function that mounts into the Node/Next.js/Express
service you already deploy. The Inngest platform (or the open-source
dev server) delivers events, schedules steps, and drives retries from
outside your process — your app only exposes an HTTP handler.

## Layout

- `src/inngest/client.ts` — the Inngest client (app id `{pipeline.name}`)
- `src/inngest/pipeline.ts` — the pipeline as one durable Inngest
  function: `step.run` per node, `step.waitForEvent` per HITL gate
- `src/index.ts` — minimal framework-neutral serve entrypoint
  (`inngest/node` on a plain `node:http` server)
- `src/extracted/` — stubs for deterministic nodes; fill these in with
  direct vendor API calls (they throw until you do)
- `src/signatures/` — typed LLM judges generated from the pipeline IR
  (Zod schemas + direct vendor SDK calls)

## Run locally

Two processes: your app, and the Inngest dev server (single binary, no
account needed).

```sh
npm install
npm run build
INNGEST_DEV=1 node dist/index.js
```

```sh
npx inngest-cli@latest dev -u http://localhost:{cfg.serve_port}
```

(The emitted `src/index.ts` serves the Inngest handler on every path,
so no `/api/inngest` suffix is needed; when you mount inside a
framework instead, point `-u` at the route you mounted.)

The dev UI at http://localhost:8288 shows registered functions, runs,
and step timelines.

LLM judge steps call the vendor SDK directly and read the standard
`ANTHROPIC_API_KEY` / `OPENAI_API_KEY` environment variables.

## Trigger a run

Send the trigger event with the pipeline input as `data` — from code:

```ts
import {{ inngest }} from "./src/inngest/client";

await inngest.send({{ name: "{trigger}", data: {{ your: "input" }} }});
```

or over HTTP against the dev server (any event key works in dev):

```sh
curl -X POST http://localhost:8288/e/dev \\
    -H 'content-type: application/json' \\
    -d '{{"name": "{trigger}", "data": {{"your": "input"}}}}'
```

In production, send to Inngest Cloud (`https://inn.gs/e/<EVENT_KEY>`)
or call `inngest.send()` with `INNGEST_EVENT_KEY` set.

## Mount in an existing Next.js app

This is the point of the Inngest target: the same functions live in
the app you already ship. Copy `src/inngest/`, `src/extracted/`, and
`src/signatures/` into your project, then expose the serve handler as
a route — Inngest reaches your functions through it:

```ts
// app/api/inngest/route.ts
import {{ serve }} from "inngest/next";
import {{ inngest }} from "@/inngest/client";
import {{ runPipeline }} from "@/inngest/pipeline";

export const {{ GET, POST, PUT }} = serve({{
    client: inngest,
    functions: [runPipeline],
}});
```

Express works the same via `inngest/express` (mount at `/api/inngest`
with `express.json()`). Set `INNGEST_SIGNING_KEY` (and
`INNGEST_EVENT_KEY` for sends) in the deployed environment, then sync
the app once from the Inngest dashboard.

## HITL gates

The run parks durably at each gate until its event arrives:

| Gate | Event | Timeout |
| --- | --- | --- |
{gate_lines}

Resume a parked run by sending the gate's event — the payload becomes
the gate's result downstream:

```sh
curl -X POST http://localhost:8288/e/dev \\
    -H 'content-type: application/json' \\
    -d '{{"name": "{first_gate_event}", "data": {{"approved": true}}}}'
```

(From code: `inngest.send({{ name: "{first_gate_event}", data: {{...}} }})`.)
A gate that times out throws `NonRetriableError` and fails the run —
silence is not approval.

## Retry semantics (read before tuning)

Inngest v4 retries are **function-level**: every step shares the
`retries` budget configured in `src/inngest/pipeline.ts` (set to the
max across the IR's per-node policies), with managed exponential
backoff + jitter. Per-node deltas from the IR are documented as
comments on the affected steps. Throw `NonRetriableError` inside a
step to stop retrying early.
"""


# ───────── Adapter facade ─────────


class InngestAdapter:
    """Facade that emits an Inngest TypeScript app from a Pipeline IR.

    Output layout::

        out/
            README.md
            package.json
            tsconfig.json
            src/
                index.ts
                inngest/
                    client.ts
                    pipeline.ts
                signatures/<llm_judge_id>.ts
                extracted/<other_id>.ts

    The directory runs with ``npm install && npm run build &&
    INNGEST_DEV=1 node dist/index.js`` next to ``npx inngest-cli dev``
    once the user fills in the extracted/ stubs; in production the same
    functions mount into an existing Next.js/Express/Node app.
    """

    def __init__(self, config: InngestAdapterConfig | None = None) -> None:
        self.config = config or InngestAdapterConfig()

    def emit_pipeline_ts(self, pipeline: Pipeline) -> str:
        return emit_pipeline_ts(pipeline, self.config)

    def emit(self, pipeline: Pipeline, output_dir: str | Path) -> dict[str, Path]:
        writer = EmitWriter(output_dir)

        written: dict[str, Path] = {}

        written["client"] = writer.write(
            "src", "inngest", "client.ts", content=emit_client(pipeline)
        )
        written["pipeline"] = writer.write(
            "src", "inngest", "pipeline.ts", content=self.emit_pipeline_ts(pipeline)
        )
        written["index"] = writer.write(
            "src", "index.ts", content=emit_index(pipeline, self.config)
        )

        for node in pipeline.nodes:
            if node.kind is NodeKind.HITL_GATE:
                continue
            if node.kind is NodeKind.LLM_JUDGE:
                written[f"signatures/{node.id}"] = writer.write(
                    "src",
                    "signatures",
                    f"{node.id}.ts",
                    content=emit_signature_module(node, self.config),
                )
            else:
                written[f"extracted/{node.id}"] = writer.write(
                    "src",
                    "extracted",
                    f"{node.id}.ts",
                    content=emit_extracted_module(node),
                )

        written["package.json"] = writer.write("package.json", content=emit_package_json(pipeline))
        written["tsconfig.json"] = writer.write("tsconfig.json", content=emit_node_tsconfig())
        written["README"] = writer.write("README.md", content=emit_readme(pipeline, self.config))

        writer.finalize()
        return written
