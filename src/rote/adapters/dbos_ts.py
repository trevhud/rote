"""DBOS TypeScript adapter — emits a durable Node.js app from a Pipeline IR.

Layer 3 of rote: takes a validated :class:`rote.ir.Pipeline` and emits a
runnable DBOS Transact application in TypeScript — the TS/Node analog of
:mod:`rote.adapters.dbos`. DBOS is the "no workflow runner" target:
durable execution as a library checkpointing to Postgres. There is no
orchestrator process to deploy; ``node dist/main.js`` *is* the runtime.

Output layout::

    out/
        src/main.ts                 # DBOS.registerWorkflow + one registerStep per node
        src/signatures/<judge>.ts   # typed Zod + vendor-SDK signatures
        src/extracted/<node>.ts     # stubs for pure_function / external_call / agent_loop
        package.json                # dbos sdk + zod + vendor SDKs
        tsconfig.json
        dbos-config.yaml            # CLI/Cloud tooling config (`npx dbos start`)
        README.md                   # how to run, signal HITL gates, deploy

Key design choices, verified against ``@dbos-inc/dbos-sdk`` v4.23.6 (the
installed package's ``.d.ts`` is the source of truth; docs.dbos.dev
corroborates):

* **Function-wrapper registration, not decorators.** The v4 TS SDK's
  primary API is ``DBOS.registerWorkflow(fn, config)`` /
  ``DBOS.registerStep(fn, config)`` — plain function wrappers. The
  legacy class-static decorator form still exists but the emitted code
  uses the wrapper form the current docs lead with.

* **Postgres only.** Unlike the Python SDK (SQLite since 1.13.0), the
  TS SDK's sole database driver is ``pg`` — there is no SQLite system
  database. Local dev uses ``npx dbos postgres start`` (a Docker
  Postgres matching the SDK's default URL,
  ``postgresql://postgres:dbos@localhost:5432/<name>_dbos_sys``);
  production points ``DBOS_SYSTEM_DATABASE_URL`` at real Postgres.

* **Parallel waves via ``Promise.allSettled``.** The DBOS TS docs'
  documented in-workflow concurrency primitive is settling concurrent
  step calls (an unguarded ``Promise.all`` can crash the Node process
  on unhandled rejections, per the docs' explicit warning). Multi-node
  waves settle every step then unwrap; single-node waves await the
  step directly. Queues exist in the TS SDK but are a workflow-level
  fan-out primitive, not the step-level one.

* **HITL gates via durable messages.** ``DBOS.recv<T>(topic,
  timeoutSeconds)`` parks the workflow in the system database until
  ``DBOS.send(workflowID, payload, topic)`` delivers the resume
  message. The IR ``signal`` name maps to the topic. ``recv`` defaults
  to a 60-second timeout when none is given, so the emitted code
  *always* passes the IR timeout explicitly. Timeout returns ``null``
  → the workflow throws: silence is not approval.

* **Per-step timeouts are real config here.** TS ``StepConfig`` has
  ``timeoutMS`` (the Python SDK has no per-step timeout primitive), so
  the IR ``timeout`` maps to config instead of surviving only as a
  comment. ``retry_on`` still can't be expressed declaratively — the
  TS knob is a ``shouldRetry`` predicate — so it surfaces as a comment
  naming that knob, mirroring the Python adapter.

* **Data-flow threading matches the other adapters.** The workflow
  takes the pipeline input as ``pipelineInput``, binds every node's
  result (including HITL gate resume payloads) as ``<id>_result``, and
  builds each step's payload from the node's ``inputs:`` bindings via
  the shared reference grammar (:func:`rote.ir.parse_input_ref`).
  Forward references are rejected at emit time by
  :func:`rote.adapters._common.check_input_refs_available`.

* **Typed LLM signatures as Zod + vendor SDK modules.** Emitted via
  :mod:`rote.adapters._ts_common` — the same shared machinery as the
  Cloudflare adapter, so the two TS runtimes share prompt semantics
  (including the interpolate helper that throws on unresolvable
  placeholders). ``signature_spec`` is required; the legacy Python-path
  ``signature`` form cannot be transpiled to TypeScript.

The emitted code never imports MCP runtime — same architectural
invariant as the other adapters, enforced by a comment-and-string
stripping scan in the tests.
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass
from pathlib import Path

from rote.adapters._common import (
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_OPENAI_MODEL,
    EmitWriter,
    _duration_to_seconds,
    _execution_waves,
    _pipeline_hash,
    _seconds_literal,
    _to_camel_case,
    _to_pascal_case,
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
from rote.ir import Node, NodeKind, Pipeline

_GENERATED_BY = "rote.adapters.dbos_ts"

# ───────── Adapter configuration ─────────


@dataclass(frozen=True)
class DbosTsAdapterConfig:
    """Per-emission knobs for the DBOS TypeScript adapter.

    Defaults work out-of-the-box for the BDR example: the SDK's default
    local Postgres URL for dev (overridable via
    ``DBOS_SYSTEM_DATABASE_URL``), admin server off.
    """

    anthropic_default_model: str = DEFAULT_ANTHROPIC_MODEL
    openai_default_model: str = DEFAULT_OPENAI_MODEL


# ───────── Duration / retry / timeout mapping ─────────


def _duration_to_ms(s: str) -> int:
    """Convert an IR duration string ('5m', '30s') to integer milliseconds.

    ``StepConfig.timeoutMS`` takes milliseconds; every IR shorthand unit
    converts to a whole number of them.
    """
    return int(_duration_to_seconds(s) * 1000)


def _step_config_literal(node: Node) -> str:
    """Render the config object for a ``DBOS.registerStep`` call.

    Maps the IR's ``RetryPolicy`` onto DBOS TS step options:

    | IR field      | DBOS StepConfig field                          |
    |---------------|------------------------------------------------|
    | ``max``       | ``maxAttempts`` (= max + 1; DBOS counts the    |
    |               | initial attempt, like Temporal and DBOS-Python)|
    | ``backoff``   | ``backoffRate`` (exponential → 2; constant →   |
    |               | 1; linear → 1, the closest approximation —     |
    |               | DBOS delay is ``intervalSeconds *              |
    |               | backoffRate**attempt``)                        |
    | ``timeout``   | ``timeoutMS`` (per-attempt; the TS SDK has a   |
    |               | real step timeout, unlike the Python SDK)      |
    | ``retry_on``  | not mapped (the TS knob is a ``shouldRetry``   |
    |               | predicate, not categories — surfaced as a      |
    |               | comment)                                       |
    """
    parts = [f"name: {json.dumps(node.id)}"]
    if node.retry and node.retry.max > 0:
        backoff_rate = 2 if node.retry.backoff == "exponential" else 1
        parts.append("retriesAllowed: true")
        parts.append(f"maxAttempts: {node.retry.max + 1}")
        parts.append(f"backoffRate: {backoff_rate}")
    if node.timeout:
        parts.append(f"timeoutMS: {_duration_to_ms(node.timeout)}")
    return "{ " + ", ".join(parts) + " }"


def _retry_on_comment(node: Node) -> str:
    """Emit a comment documenting retry_on categories DBOS can't express."""
    if node.retry and node.retry.retry_on:
        cats = ", ".join(node.retry.retry_on)
        return (
            f"// retry_on categories from the IR: {cats}. DBOS retries any\n"
            f"// exception; narrow with the StepConfig `shouldRetry` predicate if needed.\n"
        )
    return ""


# ───────── signatures/<id>.ts emission ─────────


def emit_signature_module(node: Node, cfg: DbosTsAdapterConfig) -> str:
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

    Same scaffolding convention as the Cloudflare adapter: one module
    per node, throwing until the user fills it in with direct vendor
    API calls. The graduation history (MCP origin) is documented in
    JSDoc only — never in executable code.
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


# ───────── src/main.ts emission ─────────


def _emit_step_registration(node: Node) -> str:
    """Emit the ``export const <camel>Step = DBOS.registerStep(...)`` block."""
    fn_name = _to_camel_case(node.id)
    desc_first = safe_block_comment_line(node.description, fallback=node.id)
    config = _step_config_literal(node)

    doc: list[str] = ["/**", f" * {desc_first}", " *"]
    if node.kind is NodeKind.LLM_JUDGE:
        doc.append(" * LLM judge — typed input/output, bounded decision space. The")
        doc.append(" * non-determinism lives inside this step, not the workflow.")
    elif node.kind is NodeKind.AGENT_LOOP:
        doc.append(" * Agent loop — bounded, tool-restricted. The stub in")
        doc.append(f" * `extracted/{node.id}` throws until implemented against an")
        doc.append(" * agent harness.")
    else:
        doc.append(" * Graduated from MCP tool call → deterministic API call. See")
        doc.append(f" * `extracted/{node.id}` for the implementation.")
    if node.mandatory:
        doc.append(" *")
        doc.append(" * MANDATORY: this node was marked mandatory in the source skill.")
        doc.append(" * The workflow always calls it; do not make it conditional.")
    doc.append(" */")

    if node.kind is NodeKind.LLM_JUDGE:
        call = f"{fn_name}(payload, {judge_env_arg(node)})"
    else:
        call = f"{fn_name}(payload)"

    lines = doc
    retry_comment = _retry_on_comment(node)
    if retry_comment:
        lines.extend(retry_comment.rstrip("\n").splitlines())
    lines.extend(
        [
            f"export const {fn_name}Step = DBOS.registerStep(",
            f"    async (payload: Record<string, unknown>) => {call},",
            f"    {config},",
            ");",
        ]
    )
    return "\n".join(lines)


def _emit_hitl_wait(node: Node, pipeline: Pipeline) -> str:
    assert node.signal is not None
    timeout_str = node.timeout or pipeline.config.hitl.default_timeout
    timeout_seconds = _seconds_literal(_duration_to_seconds(timeout_str))
    quoted_signal = json.dumps(node.signal)
    return (
        f"        // ─── HITL gate: {node.id} ───\n"
        f"        // Durable receive: the workflow parks in the system database until\n"
        f"        // DBOS.send(workflowID, payload, {quoted_signal})\n"
        f"        // delivers the resume message. Survives process restarts. The\n"
        f"        // timeout is explicit — DBOS.recv defaults to 60s when omitted.\n"
        f"        const {node.id}_result = await DBOS.recv<Record<string, unknown>>(\n"
        f"            {quoted_signal},\n"
        f"            {timeout_seconds}, // {timeout_str}\n"
        f"        );\n"
        f"        if ({node.id}_result === null) {{\n"
        f"            throw new Error(\n"
        f"                \"HITL gate '{node.id}' timed out after {timeout_str} waiting \" +\n"
        f"                    \"for signal '{node.signal}'\",\n"
        f"            );\n"
        f"        }}\n"
    )


def _emit_workflow_body(pipeline: Pipeline) -> tuple[str, bool]:
    """Render the workflow function body; returns (body, uses_unwrap)."""
    waves = _execution_waves(pipeline)
    lines: list[str] = []
    uses_unwrap = False

    # Node ids whose results are bound by the time each wave starts —
    # used to reject inputs that reference a later wave at emit time.
    # HITL gates count: their DBOS.recv payload binds `<id>_result`, so
    # gate resume payloads participate as that gate's result downstream.
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
            fn_name = _to_camel_case(node.id)
            payload = payload_ts_literal(node, indent=" " * 8)
            if payload == "{}":
                lines.append(f"        const {node.id}_result = await {fn_name}Step({{}});")
            else:
                lines.append(f"        const {node.id}_result = await {fn_name}Step(")
                lines.append(f"            {payload_ts_literal(node, indent=' ' * 12)},")
                lines.append("        );")
        elif len(non_hitl) > 1:
            uses_unwrap = True
            lines.append("        // Parallel fan-out: run every step in the wave concurrently")
            lines.append("        // and settle them all — Promise.allSettled per the DBOS docs")
            lines.append("        // (a bare Promise.all can crash the Node process on unhandled")
            lines.append("        // rejections).")
            settled_names = ", ".join(f"{n.id}_settled" for n in non_hitl)
            lines.append(f"        const [{settled_names}] = await Promise.allSettled([")
            for node in non_hitl:
                fn_name = _to_camel_case(node.id)
                payload = payload_ts_literal(node, indent=" " * 12)
                if payload == "{}":
                    lines.append(f"            {fn_name}Step({{}}),")
                else:
                    lines.append(f"            {fn_name}Step({payload}),")
            lines.append("        ]);")
            for node in non_hitl:
                lines.append(f"        const {node.id}_result = unwrap({node.id}_settled);")

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
    return "\n".join(lines), uses_unwrap


_UNWRAP_HELPER = """\
function unwrap<T>(settled: PromiseSettledResult<T>): T {
    if (settled.status === "rejected") {
        // Rethrow the step's original error — DBOS already logged which
        // step failed, and downstream debugging wants the real cause.
        throw settled.reason;
    }
    return settled.value;
}
"""


def emit_main(pipeline: Pipeline, cfg: DbosTsAdapterConfig | None = None) -> str:
    """Render the src/main.ts source for a pipeline."""
    cfg = cfg or DbosTsAdapterConfig()

    pascal = _to_pascal_case(pipeline.name)
    pipeline_h = _pipeline_hash(pipeline)
    workflow_name = f"{pascal}_{pipeline_h}"
    desc_first = safe_block_comment_line(pipeline.description, fallback=pipeline.name)

    header = textwrap.dedent(
        f"""\
        /**
         * Auto-generated by rote.adapters.dbos_ts.
         *
         * Pipeline: {pipeline.name} v{pipeline.version}
         * Source skill: {pipeline.source_skill or "unknown"}
         * Pipeline hash: {pipeline_h}
         *
         * DO NOT EDIT BY HAND. Re-run `rote emit --runtime dbos-ts` to regenerate.
         *
         * Workflow versioning: the registered workflow name includes a hash of
         * the pipeline so regenerated pipelines become a new workflow type.
         * In-flight workflows recover onto the code they started with; new
         * starts use the new code.
         *
         * Architecture note: every step in this file wraps a deterministic
         * function from `extracted/` or a typed LLM signature from
         * `signatures/`. None of them call MCP tools at runtime — the MCP
         * tool calls from the source skill were graduated into direct API
         * calls during the rote emission step.
         */

        import {{ DBOS }} from "@dbos-inc/dbos-sdk";

        """
    )

    imports = module_imports(pipeline)

    helper_blocks: list[str] = []
    if pipeline.nodes_by_kind(NodeKind.LLM_JUDGE):
        helper_blocks.append(REQUIRE_ENV_HELPER.rstrip("\n"))

    steps_header = (
        "// ───────── Steps ─────────\n"
        "//\n"
        "// One durable step per node (loop-body sub-nodes included, so they\n"
        "// stay testable in isolation). DBOS checkpoints every step result to\n"
        "// the system database; a crashed process resumes after the last\n"
        "// completed step."
    )
    step_blocks = [
        _emit_step_registration(node)
        for node in pipeline.nodes
        if node.kind is not NodeKind.HITL_GATE
    ]

    body, uses_unwrap = _emit_workflow_body(pipeline)
    unwrap_block = f"\n{_UNWRAP_HELPER.rstrip(chr(10))}\n" if uses_unwrap else ""

    workflow_block = (
        "// ───────── Workflow ─────────\n"
        f"{unwrap_block}"
        "\n"
        "/**\n"
        f" * {desc_first}\n"
        " */\n"
        "export const runPipeline = DBOS.registerWorkflow(\n"
        "    async (pipelineInput: Record<string, unknown>): "
        "Promise<Record<string, unknown>> => {"
        f"{body}\n"
        "    },\n"
        f"    {{ name: {json.dumps(workflow_name)} }},\n"
        ");"
    )

    main_block = textwrap.dedent(
        f"""\
        // ───────── Entrypoint ─────────

        export async function main(): Promise<void> {{
            DBOS.setConfig({{
                name: {json.dumps(pipeline.name)},
                // undefined → the SDK default
                // (postgresql://postgres:dbos@localhost:5432/<name>_dbos_sys),
                // which matches `npx dbos postgres start`. The TS SDK is
                // Postgres-only — unlike DBOS Python there is no SQLite mode.
                systemDatabaseUrl: process.env.DBOS_SYSTEM_DATABASE_URL,
                // The admin server (port 3001) is opt-in; enable it when you
                // want the DBOS console / management API.
                runAdminServer: false,
            }});
            await DBOS.launch();
            try {{
                const pipelineInput = JSON.parse(
                    process.argv[2] ?? "{{}}",
                ) as Record<string, unknown>;
                const handle = await DBOS.startWorkflow(runPipeline)(pipelineInput);
                console.error(`workflow started: ${{handle.workflowID}}`);
                console.log(JSON.stringify(await handle.getResult(), null, 2));
            }} finally {{
                await DBOS.shutdown();
            }}
        }}

        if (require.main === module) {{
            main().catch((err) => {{
                console.error(err);
                process.exitCode = 1;
            }});
        }}
        """
    )

    sections = [header + imports]
    sections.extend(helper_blocks)
    sections.append(steps_header)
    sections.extend(step_blocks)
    sections.append(workflow_block)
    sections.append(main_block.rstrip("\n") + "\n")
    return "\n\n".join(sections)


# ───────── package / tsconfig / dbos-config / README emission ─────────


def emit_package_json(pipeline: Pipeline) -> str:
    """Emit package.json with the current SDK majors (verified on npm).

    CommonJS (no ``"type": "module"``): the DBOS SDK ships CJS and the
    docs' templates compile to it; ``require.main === module`` in the
    emitted entrypoint depends on it too.
    """
    dependencies = {
        "@dbos-inc/dbos-sdk": "^4.23.6",
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
            "start": "node dist/main.js",
            "typecheck": "tsc --noEmit",
        },
        "dependencies": dict(sorted(dependencies.items())),
        "devDependencies": {
            "@types/node": "^26.1.0",
            "typescript": "^6.0.3",
        },
    }
    return json.dumps(obj, indent=2) + "\n"


def emit_dbos_config(pipeline: Pipeline) -> str:
    """Emit dbos-config.yaml so ``npx dbos start`` (and DBOS Cloud) work.

    Runtime configuration is programmatic (``DBOS.setConfig`` in
    src/main.ts); this file only feeds the CLI/Cloud tooling.
    """
    return (
        "# Auto-generated by rote.adapters.dbos_ts. Used by the `dbos` CLI and\n"
        "# DBOS Cloud; runtime config lives in src/main.ts's DBOS.setConfig.\n"
        f"name: {pipeline.name}\n"
        "language: node\n"
        "runtimeConfig:\n"
        "  start:\n"
        "    - node dist/main.js\n"
    )


def emit_readme(pipeline: Pipeline, cfg: DbosTsAdapterConfig) -> str:
    gates = [n for n in pipeline.nodes if n.kind is NodeKind.HITL_GATE]
    gate_lines = "\n".join(
        f"| `{g.id}` | `{g.signal}` | {g.timeout or pipeline.config.hitl.default_timeout} |"
        for g in gates
    )
    first_signal = gates[0].signal if gates and gates[0].signal else "example_signal"

    # Built flush-left (not textwrap.dedent) because interpolated
    # multi-line values — the gate table rows — contain unindented lines
    # that would defeat dedent's common-prefix detection. Same fix the
    # Python DBOS adapter's emit_readme shipped after that exact bug.
    return f"""\
# {pipeline.name} — DBOS (TypeScript) runtime

Auto-generated by `rote emit --runtime dbos-ts`. Do not edit generated
files by hand; re-run the emitter to regenerate.

DBOS is durable execution as a library: there is no orchestrator to
deploy. The compiled `dist/main.js` checkpoints every step to a
Postgres system database and resumes from the last completed step
after a crash.

## Layout

- `src/main.ts` — the `DBOS.registerWorkflow` DAG plus one
  `DBOS.registerStep` per node
- `src/extracted/` — stubs for deterministic nodes; fill these in with
  direct vendor API calls (they throw until you do)
- `src/signatures/` — typed LLM judges generated from the pipeline IR
  (Zod schemas + direct vendor SDK calls)
- `dbos-config.yaml` — CLI/Cloud tooling config for `npx dbos start`

## Run locally

The TypeScript SDK's system database is Postgres (unlike DBOS Python
there is no SQLite mode). `npx dbos postgres start` launches a local
Docker Postgres matching the SDK's default connection URL:

```sh
npm install
npx dbos postgres start
npm run build
node dist/main.js '{{"your": "input"}}'
```

For production, point the app at your own Postgres:

```sh
export DBOS_SYSTEM_DATABASE_URL="postgresql://user:pass@host:5432/db"
node dist/main.js '{{...}}'   # or: npx dbos start
```

LLM judge steps call the vendor SDK directly and read the standard
`ANTHROPIC_API_KEY` / `OPENAI_API_KEY` environment variables. Each
judge also honors per-node `ROTE_MODEL_<NODE_ID>` and
`ROTE_BASE_URL_<NODE_ID>` overrides, so operators can swap the model
or point at an OpenAI-compatible endpoint without re-emitting.

## HITL gates

The workflow parks durably at each gate until a message arrives on
the gate's topic:

| Gate | Signal (topic) | Timeout |
| --- | --- | --- |
{gate_lines}

Resume a parked workflow from any process that can reach the system
database:

```ts
import {{ DBOSClient }} from "@dbos-inc/dbos-sdk";

const client = await DBOSClient.create({{ systemDatabaseUrl: "..." }});
await client.send(workflowId, {{ approved: true }}, "{first_signal}");
await client.destroy();
```

(In-process, `DBOS.send(workflowId, payload, "{first_signal}")` does
the same.) A gate that times out throws and fails the run — silence
is not approval.
"""


# ───────── Adapter facade ─────────


class DbosTsAdapter:
    """Facade that emits a DBOS TypeScript application from a Pipeline IR.

    Output layout::

        out/
            README.md
            package.json
            tsconfig.json
            dbos-config.yaml
            src/
                main.ts
                signatures/<llm_judge_id>.ts
                extracted/<other_id>.ts

    The directory runs with ``npm install && npm run build && node
    dist/main.js`` once the user fills in the extracted/ stubs (local
    Postgres via ``npx dbos postgres start``; production via
    ``DBOS_SYSTEM_DATABASE_URL``).
    """

    def __init__(self, config: DbosTsAdapterConfig | None = None) -> None:
        self.config = config or DbosTsAdapterConfig()

    def emit_main(self, pipeline: Pipeline) -> str:
        return emit_main(pipeline, self.config)

    def emit(self, pipeline: Pipeline, output_dir: str | Path) -> dict[str, Path]:
        writer = EmitWriter(output_dir)

        written: dict[str, Path] = {}

        written["main"] = writer.write("src", "main.ts", content=self.emit_main(pipeline))

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
        written["dbos-config"] = writer.write(
            "dbos-config.yaml", content=emit_dbos_config(pipeline)
        )
        written["README"] = writer.write("README.md", content=emit_readme(pipeline, self.config))

        writer.finalize()
        return written
