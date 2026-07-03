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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rote.adapters._common import (
    _execution_waves,
    _pipeline_hash,
    _to_camel_case,
    _to_pascal_case,
    check_input_refs_available,
)
from rote.adapters._common import (
    ir_duration_to_human as _ir_duration_to_cf,
)
from rote.ir import LLMSignature, Node, NodeKind, Pipeline, parse_input_ref

# ───────── Adapter configuration ─────────


@dataclass(frozen=True)
class CloudflareAdapterConfig:
    """Per-emission knobs for the Cloudflare adapter.

    Defaults work out-of-the-box for the BDR example. Production users
    will typically override ``workflow_binding`` and the model defaults.
    """

    workflow_binding: str = "PIPELINE"
    compatibility_date: str = "2026-04-25"
    anthropic_default_model: str = "claude-sonnet-4-6"
    openai_default_model: str = "gpt-4.1"
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

    The IR allows arbitrary signal strings; the adapter enforces the
    Cloudflare constraint at emit time so the user gets a clear error
    instead of a runtime ``workflow.invalid_event_type`` from Cloudflare.
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


# ───────── JSON Schema → Zod ─────────


def _resolve_refs(schema: Any, defs: dict[str, Any]) -> Any:
    """Recursively inline ``$ref`` references using a pre-extracted ``$defs`` map.

    Pydantic emits nested types as ``$ref: "#/$defs/Name"``. Zod schemas
    are constructed inline so we resolve refs eagerly.
    """
    if isinstance(schema, dict):
        if "$ref" in schema:
            ref = schema["$ref"]
            if not ref.startswith("#/$defs/"):
                raise ValueError(f"Unsupported $ref form: {ref!r}")
            name = ref[len("#/$defs/") :]
            if name not in defs:
                raise ValueError(f"Unknown $ref target: {name!r}")
            return _resolve_refs(defs[name], defs)
        return {k: _resolve_refs(v, defs) for k, v in schema.items() if k != "$defs"}
    if isinstance(schema, list):
        return [_resolve_refs(x, defs) for x in schema]
    return schema


def _convert_zod(schema: Any, indent: int = 0) -> str:
    """Convert a (resolved, ref-free) JSON Schema fragment to a Zod expression."""
    if not isinstance(schema, dict):
        raise ValueError(f"Expected schema dict, got {type(schema).__name__}: {schema!r}")

    # Nullable / unions via anyOf or oneOf
    for union_key in ("anyOf", "oneOf"):
        if union_key in schema:
            variants = schema[union_key]
            non_null = [
                v for v in variants if not (isinstance(v, dict) and v.get("type") == "null")
            ]
            has_null = len(non_null) < len(variants)
            if len(non_null) == 1:
                inner = _convert_zod(non_null[0], indent)
                return f"{inner}.nullable()" if has_null else inner
            parts = [_convert_zod(v, indent) for v in non_null]
            union = "z.union([" + ", ".join(parts) + "])"
            return f"{union}.nullable()" if has_null else union

    if "enum" in schema:
        values = schema["enum"]
        if all(isinstance(v, str) for v in values):
            return "z.enum([" + ", ".join(json.dumps(v) for v in values) + "])"
        # Mixed-type enum
        literals = [f"z.literal({json.dumps(v)})" for v in values]
        return "z.union([" + ", ".join(literals) + "])"

    schema_type = schema.get("type")

    if schema_type == "object":
        props = schema.get("properties", {})
        required = set(schema.get("required", []))
        if not props:
            return "z.object({}).strict()"
        pad = "  " * (indent + 1)
        outer = "  " * indent
        lines = []
        for name, prop_schema in props.items():
            inner = _convert_zod(prop_schema, indent + 1)
            if name not in required:
                inner = f"{inner}.optional()"
            lines.append(f"{pad}{json.dumps(name)}: {inner},")
        body = "\n".join(lines)
        return f"z.object({{\n{body}\n{outer}}}).strict()"

    if schema_type == "array":
        items = schema.get("items", {})
        return f"z.array({_convert_zod(items, indent)})"

    if schema_type == "string":
        return "z.string()"
    if schema_type == "integer":
        return "z.number().int()"
    if schema_type == "number":
        return "z.number()"
    if schema_type == "boolean":
        return "z.boolean()"
    if schema_type == "null":
        return "z.null()"

    # Permissive fallback for under-specified schemas.
    return "z.unknown()"


def json_schema_to_zod(schema: dict[str, Any], indent: int = 0) -> str:
    """Public entry point: convert a (possibly ref-laden) JSON Schema to Zod source."""
    defs = schema.get("$defs", {})
    resolved = _resolve_refs(schema, defs)
    return _convert_zod(resolved, indent)


# ───────── Workflow.ts emission ─────────

_TS_IDENT_RE = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*$")


def _ref_to_ts_expr(ref: str) -> str:
    """Render an ``inputs:`` source reference as a TypeScript expression.

    The workflow binds the instance params to ``pipelineInput`` and each
    node's result to ``<node_id>_result``:

    | Reference                  | Expression                                          |
    |----------------------------|-----------------------------------------------------|
    | ``pipeline.input``         | ``pipelineInput``                                   |
    | ``pipeline.input.f``       | ``pipelineInput["f"]``                              |
    | ``foo.output``             | ``foo_result``                                      |
    | ``foo.output.f``           | ``(foo_result as Record<string, unknown>)["f"]``    |

    Node-output field access goes through a Record cast because
    ``step.do``'s ``Rpc.Serializable`` constraint widens inferred result
    types to its constraint union (verified against
    @cloudflare/workers-types via the tsc e2e test) — direct indexing
    doesn't compile. The cast also stays valid for whatever concrete
    return type the user gives a stub later.
    """
    parsed = parse_input_ref(ref)
    if parsed.node_id is None:
        if parsed.field is None:
            return "pipelineInput"
        return f"pipelineInput[{json.dumps(parsed.field)}]"
    base = f"{parsed.node_id}_result"
    if parsed.field is None:
        return base
    return f"({base} as Record<string, unknown>)[{json.dumps(parsed.field)}]"


def _payload_ts_literal(node: Node, indent: str) -> str:
    """Render the step payload object for a node's data-flow bindings.

    Nodes without ``inputs`` keep the empty payload (back-compat).
    ``indent`` is the indentation of the line the literal starts on;
    entries are indented one level deeper.
    """
    if not node.inputs:
        return "{}"
    inner = indent + "    "
    lines = ["{"]
    for param, ref in node.inputs.items():
        key = param if _TS_IDENT_RE.fullmatch(param) else json.dumps(param)
        lines.append(f"{inner}{key}: {_ref_to_ts_expr(ref)},")
    lines.append(indent + "}")
    return "\n".join(lines)


def _emit_step_call(node: Node, cfg: CloudflareAdapterConfig, *, pass_env: bool) -> str:
    fn_name = _to_camel_case(node.id)
    config = _step_config_literal(node, cfg)
    payload = _payload_ts_literal(node, indent=" " * 12)
    args = f"{payload}, this.env" if pass_env else payload
    return (
        f"        const {node.id}_result = await step.do(\n"
        f"            {json.dumps(node.id)},\n"
        f"            {config},\n"
        f"            async () => {fn_name}({args}),\n"
        f"        );\n"
    )


def _emit_step_call_pure_or_external(node: Node, cfg: CloudflareAdapterConfig) -> str:
    return _emit_step_call(node, cfg, pass_env=False)


def _emit_step_call_llm_judge(node: Node, cfg: CloudflareAdapterConfig) -> str:
    return _emit_step_call(node, cfg, pass_env=True)


def _emit_step_call_agent_loop(node: Node, cfg: CloudflareAdapterConfig) -> str:
    return _emit_step_call(node, cfg, pass_env=True)


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


def _module_imports(pipeline: Pipeline) -> str:
    """Build the static-import block for the workflow file.

    One import per non-HITL node, including loop_body sub-nodes. The
    function name is camelCase of the node id; the path is
    ``./signatures/<id>`` for llm_judge and ``./extracted/<id>`` otherwise.
    """
    lines: list[str] = []
    for node in pipeline.nodes:
        if node.kind is NodeKind.HITL_GATE:
            continue
        fn = _to_camel_case(node.id)
        if node.kind is NodeKind.LLM_JUDGE:
            lines.append(f'import {{ {fn} }} from "./signatures/{node.id}";')
        else:
            lines.append(f'import {{ {fn} }} from "./extracted/{node.id}";')
    return "\n".join(lines)


def emit_workflow(pipeline: Pipeline, cfg: CloudflareAdapterConfig | None = None) -> str:
    """Render the workflow.ts source for a pipeline."""
    cfg = cfg or CloudflareAdapterConfig()
    pascal = _to_pascal_case(pipeline.name)
    class_name = f"{pascal}Workflow"
    pipeline_h = _pipeline_hash(pipeline)
    waves = _execution_waves(pipeline)

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
         * Architecture note: every external_call step in this workflow wraps a
         * deterministic API call from the `extracted/` modules. None of them call
         * MCP tools at runtime — those calls were graduated into direct API calls
         * during the rote emission step.
         */

        import {{
            WorkflowEntrypoint,
            WorkflowEvent,
            WorkflowStep,
        }} from "cloudflare:workers";

        """
    )

    imports = _module_imports(pipeline)

    env_block = textwrap.dedent(
        f"""\

        export interface Env {{
            ANTHROPIC_API_KEY: string;
            OPENAI_API_KEY?: string;
            {cfg.workflow_binding}: Workflow<Params>;
        }}

        export type Params = Record<string, unknown>;

        """
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

    for wave_idx, wave in enumerate(waves, start=1):
        body_lines.append(f"        // ─── Wave {wave_idx} ───")
        for node in wave:
            if node.kind is NodeKind.HITL_GATE:
                body_lines.append(_emit_hitl_gate(node, cfg).rstrip("\n"))
            else:
                check_input_refs_available(node, available)
                if node.kind in (NodeKind.PURE_FUNCTION, NodeKind.EXTERNAL_CALL):
                    body_lines.append(_emit_step_call_pure_or_external(node, cfg).rstrip("\n"))
                elif node.kind is NodeKind.LLM_JUDGE:
                    body_lines.append(_emit_step_call_llm_judge(node, cfg).rstrip("\n"))
                elif node.kind is NodeKind.AGENT_LOOP:
                    body_lines.append(_emit_step_call_agent_loop(node, cfg).rstrip("\n"))
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

    return header + imports + "\n" + env_block + class_block


# ───────── index.ts emission ─────────


def emit_index(pipeline: Pipeline, cfg: CloudflareAdapterConfig) -> str:
    pascal = _to_pascal_case(pipeline.name)
    class_name = f"{pascal}Workflow"
    return textwrap.dedent(
        f"""\
        /**
         * Auto-generated by rote.adapters.cloudflare.
         *
         * Default fetch handler. Creates a workflow instance from the request
         * body and returns the instance id + status. Replace with your own
         * trigger surface (cron, queue consumer, etc.) as needed.
         */

        import {{ {class_name}, type Env, type Params }} from "./workflow";

        export {{ {class_name} }};

        export default {{
            async fetch(req: Request, env: Env): Promise<Response> {{
                const raw = await req.json().catch(() => ({{}}));
                const params = (raw ?? {{}}) as Params;
                const instance = await env.{cfg.workflow_binding}.create({{ params }});
                return Response.json({{
                    id: instance.id,
                    status: await instance.status(),
                }});
            }},
        }} satisfies ExportedHandler<Env>;
        """
    )


# ───────── signatures/<id>.ts emission ─────────


def emit_signature_module(node: Node, cfg: CloudflareAdapterConfig) -> str:
    """Emit src/signatures/<node_id>.ts for an llm_judge node.

    The emitted module:
    1. Declares Zod schemas for input + output, derived from the IR's
       ``signature_spec`` JSON Schemas.
    2. Exports a typed function that calls the Anthropic Messages API with
       structured-output tool use, validates the response with Zod, and
       returns the typed output.

    The IR's prompt template is interpolated with the input via a
    minimal ``{{ key }}`` substitution at runtime — kept simple so the
    emitted code has no extra dependencies beyond Zod and the vendor SDK.
    """
    if node.signature_spec is None:
        raise ValueError(
            f"Cloudflare adapter: llm_judge {node.id!r} requires signature_spec "
            f"(structured form). Path-only signature: {node.signature!r} is "
            f"Temporal-specific and cannot be emitted to TypeScript."
        )
    spec = node.signature_spec
    fn_name = _to_camel_case(node.id)
    pascal = _to_pascal_case(node.id)
    input_schema_zod = json_schema_to_zod(spec.input_schema, indent=0)
    output_schema_zod = json_schema_to_zod(spec.output_schema, indent=0)
    model = spec.model or _default_model_for(spec, cfg)
    temperature = spec.temperature

    if spec.client == "anthropic":
        return _emit_signature_anthropic(
            node_id=node.id,
            fn_name=fn_name,
            pascal=pascal,
            description=node.description,
            input_zod=input_schema_zod,
            output_zod=output_schema_zod,
            output_schema_json=json.dumps(spec.output_schema, indent=2),
            prompt_template=spec.prompt,
            model=model,
            temperature=temperature,
        )
    if spec.client == "openai":
        return _emit_signature_openai(
            node_id=node.id,
            fn_name=fn_name,
            pascal=pascal,
            description=node.description,
            input_zod=input_schema_zod,
            output_zod=output_schema_zod,
            output_schema_json=json.dumps(spec.output_schema, indent=2),
            prompt_template=spec.prompt,
            model=model,
            temperature=temperature,
        )
    raise ValueError(f"Unsupported LLM client: {spec.client!r}")


def _default_model_for(spec: LLMSignature, cfg: CloudflareAdapterConfig) -> str:
    if spec.client == "openai":
        return cfg.openai_default_model
    return cfg.anthropic_default_model


_INTERPOLATE_HELPER = """\
function interpolate(template: string, vars: Record<string, unknown>): string {
    return template.replace(/\\{\\{\\s*([\\w.]+)\\s*\\}\\}/g, (_match, key: string) => {
        const value = key.split(".").reduce<unknown>(
            (acc, part) =>
                acc != null && typeof acc === "object"
                    ? (acc as Record<string, unknown>)[part]
                    : undefined,
            vars,
        );
        if (value === undefined) return "";
        return typeof value === "string" ? value : JSON.stringify(value);
    });
}
"""


def _emit_signature_anthropic(
    *,
    node_id: str,
    fn_name: str,
    pascal: str,
    description: str,
    input_zod: str,
    output_zod: str,
    output_schema_json: str,
    prompt_template: str,
    model: str,
    temperature: float | None,
) -> str:
    desc_first = description.strip().splitlines()[0] if description else node_id
    temp_line = f"        temperature: {temperature},\n" if temperature is not None else ""
    quoted_id = json.dumps(node_id)
    quoted_desc = json.dumps(desc_first)
    quoted_model = json.dumps(model)
    quoted_prompt = json.dumps(prompt_template)
    parts = [
        "/**",
        f" * Typed LLM signature: {node_id}",
        " *",
        f" * {desc_first}",
        " *",
        " * Auto-generated by rote.adapters.cloudflare from the IR's",
        " * `signature_spec`. The non-determinism lives inside this module;",
        " * the workflow that calls it stays deterministic.",
        " */",
        "",
        'import Anthropic from "@anthropic-ai/sdk";',
        'import { z } from "zod";',
        "",
        f"export const {pascal}Input = {input_zod};",
        f"export type {pascal}Input = z.infer<typeof {pascal}Input>;",
        "",
        f"export const {pascal}Output = {output_zod};",
        f"export type {pascal}Output = z.infer<typeof {pascal}Output>;",
        "",
        f"const PROMPT = {quoted_prompt};",
        "",
        f"const OUTPUT_JSON_SCHEMA = {output_schema_json};",
        "",
        _INTERPOLATE_HELPER.rstrip("\n"),
        "",
        f"export async function {fn_name}(",
        "    rawInput: unknown,",
        "    env: { ANTHROPIC_API_KEY: string },",
        f"): Promise<{pascal}Output> {{",
        f"    const input = {pascal}Input.parse(rawInput);",
        "    const client = new Anthropic({ apiKey: env.ANTHROPIC_API_KEY });",
        "",
        "    const response = await client.messages.create({",
        f"        model: {quoted_model},",
        "        max_tokens: 4096,",
    ]
    if temp_line:
        parts.append(temp_line.rstrip("\n"))
    schema_cast = 'as { type: "object"; [k: string]: unknown }'
    msg_line = (
        '            { role: "user", '
        "content: interpolate(PROMPT, input as Record<string, unknown>) },"
    )
    parts.extend(
        [
            "        tools: [",
            "            {",
            f"                name: {quoted_id},",
            f"                description: {quoted_desc},",
            f"                input_schema: OUTPUT_JSON_SCHEMA {schema_cast},",
            "            },",
            "        ],",
            f'        tool_choice: {{ type: "tool", name: {quoted_id} }},',
            "        messages: [",
            msg_line,
            "        ],",
            "    });",
            "",
            '    const toolUse = response.content.find((b) => b.type === "tool_use");',
            '    if (!toolUse || toolUse.type !== "tool_use") {',
            f'        throw new Error("LLM did not return a tool_use block for {node_id}");',
            "    }",
            f"    return {pascal}Output.parse(toolUse.input);",
            "}",
            "",
        ]
    )
    return "\n".join(parts)


def _emit_signature_openai(
    *,
    node_id: str,
    fn_name: str,
    pascal: str,
    description: str,
    input_zod: str,
    output_zod: str,
    output_schema_json: str,
    prompt_template: str,
    model: str,
    temperature: float | None,
) -> str:
    desc_first = description.strip().splitlines()[0] if description else node_id
    temp_line = f"        temperature: {temperature},\n" if temperature is not None else ""
    quoted_id = json.dumps(node_id)
    quoted_model = json.dumps(model)
    quoted_prompt = json.dumps(prompt_template)
    parts = [
        "/**",
        f" * Typed LLM signature: {node_id}",
        " *",
        f" * {desc_first}",
        " *",
        " * Auto-generated by rote.adapters.cloudflare from the IR's",
        " * `signature_spec`. Uses OpenAI structured outputs with JSON Schema.",
        " */",
        "",
        'import OpenAI from "openai";',
        'import { z } from "zod";',
        "",
        f"export const {pascal}Input = {input_zod};",
        f"export type {pascal}Input = z.infer<typeof {pascal}Input>;",
        "",
        f"export const {pascal}Output = {output_zod};",
        f"export type {pascal}Output = z.infer<typeof {pascal}Output>;",
        "",
        f"const PROMPT = {quoted_prompt};",
        f"const OUTPUT_JSON_SCHEMA = {output_schema_json};",
        "",
        _INTERPOLATE_HELPER.rstrip("\n"),
        "",
        f"export async function {fn_name}(",
        "    rawInput: unknown,",
        "    env: { OPENAI_API_KEY: string },",
        f"): Promise<{pascal}Output> {{",
        f"    const input = {pascal}Input.parse(rawInput);",
        "    const client = new OpenAI({ apiKey: env.OPENAI_API_KEY });",
        "",
        "    const response = await client.chat.completions.create({",
        f"        model: {quoted_model},",
    ]
    if temp_line:
        parts.append(temp_line.rstrip("\n"))
    schema_inline = (
        f"            json_schema: {{ name: {quoted_id}, "
        "schema: OUTPUT_JSON_SCHEMA, strict: true },"
    )
    msg_line = (
        '            { role: "user", '
        "content: interpolate(PROMPT, input as Record<string, unknown>) },"
    )
    parts.extend(
        [
            "        response_format: {",
            '            type: "json_schema",',
            schema_inline,
            "        },",
            "        messages: [",
            msg_line,
            "        ],",
            "    });",
            "    const content = response.choices[0]?.message?.content;",
            "    if (!content) {",
            f'        throw new Error("OpenAI returned no content for {node_id}");',
            "    }",
            f"    return {pascal}Output.parse(JSON.parse(content));",
            "}",
            "",
        ]
    )
    return "\n".join(parts)


# ───────── extracted/<id>.ts emission ─────────


def emit_extracted_module(node: Node) -> str:
    """Emit src/extracted/<node_id>.ts — a stub for deterministic Python equivalents.

    Cloudflare workers don't share Python modules with the Temporal
    runtime; users implement these stubs in TypeScript directly (or call
    out to a separately-deployed Python worker via service binding).
    """
    fn_name = _to_camel_case(node.id)
    desc_first = node.description.strip().splitlines()[0] if node.description else node.id

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
                " * are preferred over MCP wrappers — the rote graduator removes the MCP",
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
        doc.extend(f" *   {k} = {json.dumps(v)}" for k, v in node.constants.items())

    doc.append(" */")

    body: list[str]
    # Stubs declare Promise<never> — honest for a function that always
    # throws, and `never` is the one type that both satisfies step.do's
    # `Rpc.Serializable<T>` constraint and stays castable at the
    # workflow's data-flow reference sites (see `_ref_to_ts_expr`).
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
    pascal = _to_pascal_case(pipeline.name)
    class_name = f"{pascal}Workflow"
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
    body = json.dumps(obj, indent=2)
    return (
        "// Auto-generated by rote.adapters.cloudflare. Hand-edit at your own risk\n"
        "// (re-running `rote emit --runtime cloudflare` will overwrite).\n"
        f"{body}\n"
    )


def emit_package_json(pipeline: Pipeline) -> str:
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
            "@anthropic-ai/sdk": "^0.91.0",
            "openai": "^6.0.0",
            "zod": "^4.0.0",
        },
        "devDependencies": {
            "@cloudflare/workers-types": "^4.20260426.0",
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


def _llm_clients(pipeline: Pipeline) -> set[str]:
    """Vendor clients used by the pipeline's llm_judge signature specs."""
    return {
        node.signature_spec.client
        for node in pipeline.nodes
        if node.kind is NodeKind.LLM_JUDGE and node.signature_spec is not None
    }


def _secret_names(pipeline: Pipeline) -> list[str]:
    """Secrets the emitted worker reads, in emission order.

    ``ANTHROPIC_API_KEY`` is always present (the emitted ``Env``
    interface requires it); ``OPENAI_API_KEY`` joins when any signature
    targets the OpenAI client.
    """
    secrets = ["ANTHROPIC_API_KEY"]
    if "openai" in _llm_clients(pipeline):
        secrets.append("OPENAI_API_KEY")
    return secrets


def emit_dev_vars_example(pipeline: Pipeline) -> str:
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

## Trigger from Claude (MCP)

`rote` can expose this deployed pipeline as an MCP tool so any MCP
client triggers the deterministic workflow instead of re-running
the fuzzy skill:

```sh
rote register <graduated-output-dir> \\
    --runtime cloudflare \\
    --url https://{pipeline.name}.<your-subdomain>.workers.dev
rote serve
```

See `docs/mcp-trigger.md` in the rote repository for the full flow.
"""


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

    def emit_workflow(self, pipeline: Pipeline) -> str:
        return emit_workflow(pipeline, self.config)

    def emit_index(self, pipeline: Pipeline) -> str:
        return emit_index(pipeline, self.config)

    def emit(self, pipeline: Pipeline, output_dir: str | Path) -> dict[str, Path]:
        out = Path(output_dir)
        src = out / "src"
        sigs = src / "signatures"
        extracted = src / "extracted"
        for d in (out, src, sigs, extracted):
            d.mkdir(parents=True, exist_ok=True)

        written: dict[str, Path] = {}

        wf_path = src / "workflow.ts"
        wf_path.write_text(self.emit_workflow(pipeline), encoding="utf-8")
        written["workflow"] = wf_path

        idx_path = src / "index.ts"
        idx_path.write_text(self.emit_index(pipeline), encoding="utf-8")
        written["index"] = idx_path

        for node in pipeline.nodes:
            if node.kind is NodeKind.HITL_GATE:
                continue
            if node.kind is NodeKind.LLM_JUDGE:
                p = sigs / f"{node.id}.ts"
                p.write_text(emit_signature_module(node, self.config), encoding="utf-8")
                written[f"signatures/{node.id}"] = p
            else:
                p = extracted / f"{node.id}.ts"
                p.write_text(emit_extracted_module(node), encoding="utf-8")
                written[f"extracted/{node.id}"] = p

        wrangler_path = out / "wrangler.jsonc"
        wrangler_path.write_text(emit_wrangler(pipeline, self.config), encoding="utf-8")
        written["wrangler"] = wrangler_path

        pkg_path = out / "package.json"
        pkg_path.write_text(emit_package_json(pipeline), encoding="utf-8")
        written["package.json"] = pkg_path

        ts_path = out / "tsconfig.json"
        ts_path.write_text(emit_tsconfig(), encoding="utf-8")
        written["tsconfig.json"] = ts_path

        readme_path = out / "README.md"
        readme_path.write_text(emit_readme(pipeline, self.config), encoding="utf-8")
        written["README"] = readme_path

        dev_vars_path = out / ".dev.vars.example"
        dev_vars_path.write_text(emit_dev_vars_example(pipeline), encoding="utf-8")
        written[".dev.vars.example"] = dev_vars_path

        return written
