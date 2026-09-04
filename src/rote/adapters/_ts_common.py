"""Helpers shared by every TypeScript-emitting runtime adapter.

Extracted from ``rote.adapters.cloudflare`` once the DBOS TypeScript
adapter became the second TS consumer — same rule as ``_common``: a
helper moves here only after two adapters prove it's genuinely shared.

What lives here:

* the JSON-Schema-to-Zod converter (``json_schema_to_zod`` and its
  internals ``_resolve_refs`` / ``_convert_zod``),
* the typed LLM signature module emitters (``emit_ts_signature_module``
  dispatching to ``emit_signature_anthropic`` / ``emit_signature_openai``)
  plus the runtime prompt-interpolation helper they embed
  (``_INTERPOLATE_HELPER``),
* data-flow rendering (``ref_to_ts_expr`` / ``payload_ts_literal``) and
  the workflow-file import block (``module_imports``),
* the Node-process helpers shared by the non-Workers TS runtimes
  (``judge_env_arg``, ``REQUIRE_ENV_HELPER``, ``emit_node_tsconfig``).

What deliberately does *not* live here: anything encoding a runtime's
execution semantics (Cloudflare ``step.do`` configs, DBOS step retry
options, …) — that stays in the adapter that owns it. Case conversion
(``_to_camel_case`` etc.) already lives in ``rote.adapters._common``
because the Python adapters use it too.

The emitted signature modules are runtime-agnostic TypeScript: they
import only Zod and the vendor SDK, and take API keys via an explicit
``env`` parameter so they work identically inside a Workers isolate
(bindings) and a Node process (``process.env``).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from typing import Any

from rote.adapters._common import (
    DEFAULT_AGENT_MAX_ITERATIONS,
    _to_camel_case,
    _to_pascal_case,
    fan_out_element_param,
    safe_block_comment_line,
)
from rote.ir import LLMSignature, Node, NodeKind, Pipeline, parse_input_ref

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


# ───────── signatures/<id>.ts emission ─────────


def override_env_vars(node_id: str) -> tuple[str, str]:
    """The per-node operator-override env var names for a judge.

    ``ROTE_MODEL_<ID>`` swaps the model; ``ROTE_BASE_URL_<ID>`` points the
    vendor SDK at a different endpoint (proxy, gateway, OpenAI-compatible
    server). Same convention as the Python (DBOS) emitter, so operators
    learn one knob regardless of target runtime. ``node_id`` is a
    validated identifier, so uppercasing it yields a safe env var name.
    """
    suffix = node_id.upper()
    return f"ROTE_MODEL_{suffix}", f"ROTE_BASE_URL_{suffix}"


def temperature_env_var(node_id: str) -> str:
    """The per-node ``temperature`` override env var name.

    Kept separate from :func:`override_env_vars` (whose two-tuple shape
    several emitters unpack) because it is only emitted for judges whose
    IR actually sets a temperature. Setting it to an empty string drops
    the parameter from the request entirely — current Anthropic models
    reject ``temperature`` with a 400, so an operator retargeting
    ``ROTE_MODEL_<ID>`` at a newer model needs a way out that does not
    require re-emitting.
    """
    return f"ROTE_TEMPERATURE_{node_id.upper()}"


def _temperature_block(node_id: str, temperature: float | None) -> tuple[str, str, str]:
    """``(env_type_line, const_line, spread_line)`` for the temperature knob.

    All three are empty when the IR sets no temperature — the judge then
    sends no sampling parameter at all, which is the right default given
    the output shape is already pinned by schema-locked decoding.
    """
    if temperature is None:
        return "", "", ""
    var = temperature_env_var(node_id)
    return (
        f"        {var}?: string;",
        f"    const TEMPERATURE = env.{var} ?? {json.dumps(str(temperature))};",
        "        ...(TEMPERATURE.trim() ? { temperature: Number(TEMPERATURE) } : {}),",
    )


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
        if (value === undefined) {
            // A hole in a judge prompt produces confident garbage that is
            // far harder to debug than an error naming the missing field.
            throw new Error(
                `prompt template references {{ ${key} }} but the input has no ` +
                    `such field; available top-level keys: ${Object.keys(vars).sort().join(", ")}`,
            );
        }
        if (value === null) return "";
        return typeof value === "string" ? value : JSON.stringify(value);
    });
}
"""


def emit_signature_anthropic(
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
    base_url: str | None,
    temperature: float | None,
    generated_by: str,
) -> str:
    """Emit a signatures/<id>.ts module calling Anthropic with tool-use output.

    ``generated_by`` names the emitting adapter module in the header
    JSDoc (e.g. ``rote.adapters.cloudflare``) so regeneration
    instructions point at the right runtime.
    """
    desc_first = safe_block_comment_line(description, fallback=node_id)
    temp_env_line, temp_const_line, temp_line = _temperature_block(node_id, temperature)
    quoted_id = json.dumps(node_id)
    quoted_desc = json.dumps(desc_first)
    quoted_model = json.dumps(model)
    quoted_prompt = json.dumps(prompt_template)
    model_var, base_var = override_env_vars(node_id)
    base_default = f" ?? {json.dumps(base_url)}" if base_url else ""
    parts = [
        "/**",
        f" * Typed LLM signature: {node_id}",
        " *",
        f" * {desc_first}",
        " *",
        f" * Auto-generated by {generated_by} from the IR's",
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
        "    env: {",
        "        ANTHROPIC_API_KEY: string;",
        "        // Operator overrides — swap the model or endpoint without re-emitting.",
        f"        {model_var}?: string;",
        f"        {base_var}?: string;",
        *([temp_env_line] if temp_env_line else []),
        "    },",
        f"): Promise<{pascal}Output> {{",
        f"    const input = {pascal}Input.parse(rawInput);",
        *([temp_const_line] if temp_const_line else []),
        "    const client = new Anthropic({",
        "        apiKey: env.ANTHROPIC_API_KEY,",
        f"        baseURL: env.{base_var}{base_default},",
        "    });",
        "",
        "    const response = await client.messages.create({",
        f"        model: env.{model_var} ?? {quoted_model},",
        "        max_tokens: 4096,",
    ]
    if temp_line:
        parts.append(temp_line)
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


def emit_signature_openai(
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
    base_url: str | None,
    temperature: float | None,
    generated_by: str,
) -> str:
    """Emit a signatures/<id>.ts module using OpenAI structured outputs."""
    desc_first = safe_block_comment_line(description, fallback=node_id)
    temp_env_line, temp_const_line, temp_line = _temperature_block(node_id, temperature)
    quoted_id = json.dumps(node_id)
    quoted_model = json.dumps(model)
    quoted_prompt = json.dumps(prompt_template)
    model_var, base_var = override_env_vars(node_id)
    base_default = f" ?? {json.dumps(base_url)}" if base_url else ""
    parts = [
        "/**",
        f" * Typed LLM signature: {node_id}",
        " *",
        f" * {desc_first}",
        " *",
        f" * Auto-generated by {generated_by} from the IR's",
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
        "    env: {",
        "        OPENAI_API_KEY: string;",
        "        // Operator overrides — swap the model or endpoint without re-emitting.",
        f"        {model_var}?: string;",
        f"        {base_var}?: string;",
        *([temp_env_line] if temp_env_line else []),
        "    },",
        f"): Promise<{pascal}Output> {{",
        f"    const input = {pascal}Input.parse(rawInput);",
        *([temp_const_line] if temp_const_line else []),
        "    const client = new OpenAI({",
        "        apiKey: env.OPENAI_API_KEY,",
        f"        baseURL: env.{base_var}{base_default},",
        "    });",
        "",
        "    const response = await client.chat.completions.create({",
        f"        model: env.{model_var} ?? {quoted_model},",
    ]
    if temp_line:
        parts.append(temp_line)
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


def emit_signature_workers_ai(
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
    base_url: str | None,  # unused: Workers AI routes via the AI binding, not a base URL
    temperature: float | None,
    generated_by: str,
) -> str:
    """Emit a signatures/<id>.ts module backed by the Cloudflare Workers AI
    binding (``env.AI.run``) with constrained JSON-Schema decoding.

    Unlike the anthropic/openai emitters there is no vendor SDK and no API
    key: inference runs on Cloudflare and is authenticated by the ``AI``
    binding. The call is routed through an AI Gateway (id from
    ``ROTE_GATEWAY_ID``, else ``"default"``) and tagged with run-attribution
    metadata (tenant / pipeline / run / node) so the platform can read cost
    and tokens back from the gateway logs. Output is schema-locked via
    ``response_format: json_schema``; ``temperature`` (when given) and a fixed
    ``seed`` maximise reproducibility — the determinism the pipeline promises.
    """
    del base_url
    desc_first = safe_block_comment_line(description, fallback=node_id)
    quoted_prompt = json.dumps(prompt_template)
    quoted_model = json.dumps(model)
    quoted_node = json.dumps(node_id)
    model_var, _base_var = override_env_vars(node_id)
    temp_env_line, temp_const_line, temp_line = _temperature_block(node_id, temperature)
    parts = [
        "/**",
        f" * Typed LLM signature: {node_id}",
        " *",
        f" * {desc_first}",
        " *",
        f" * Auto-generated by {generated_by} from the IR's",
        " * `signature_spec`. Runs on Cloudflare Workers AI via the `env.AI`",
        " * binding, routed through an AI Gateway with schema-locked output.",
        " */",
        "",
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
        "    env: {",
        "        AI: Ai;",
        "        // Operator override — swap the Workers AI model without re-emitting.",
        f"        {model_var}?: string;",
        *([temp_env_line] if temp_env_line else []),
        "        // Injected by the host at load time; harmless when unset.",
        "        ROTE_GATEWAY_ID?: string;",
        "        ROTE_TENANT_ID?: string;",
        "        ROTE_PIPELINE?: string;",
        "        ROTE_RUN_ID?: string;",
        "    },",
        f"): Promise<{pascal}Output> {{",
        f"    const input = {pascal}Input.parse(rawInput);",
        *([temp_const_line] if temp_const_line else []),
        "",
        "    // Attribution tags (max 5, primitive values) for gateway cost logs.",
        "    const metadata: Record<string, string> = { node: " + quoted_node + " };",
        '    if (env.ROTE_TENANT_ID) metadata["tenant_id"] = env.ROTE_TENANT_ID;',
        '    if (env.ROTE_PIPELINE) metadata["pipeline"] = env.ROTE_PIPELINE;',
        '    if (env.ROTE_RUN_ID) metadata["run_id"] = env.ROTE_RUN_ID;',
        "",
        "    const run = env.AI.run as unknown as (",
        "        model: string,",
        "        input: Record<string, unknown>,",
        "        options?: Record<string, unknown>,",
        "    ) => Promise<{ response?: unknown }>;",
        "",
        f"    const resp = await run(env.{model_var} ?? {quoted_model}, {{",
        "        messages: [",
        (
            '            { role: "user", '
            "content: interpolate(PROMPT, input as Record<string, unknown>) },"
        ),
        "        ],",
        '        response_format: { type: "json_schema", json_schema: OUTPUT_JSON_SCHEMA },',
        temp_line if temp_line else "        // temperature: vendor default",
        "        seed: 1,",
        "        max_tokens: 2048,",
        "    }, {",
        '        gateway: { id: env.ROTE_GATEWAY_ID ?? "default", skipCache: true, metadata },',
        "    });",
        "",
        "    // JSON-schema mode returns `response` already parsed (object); the",
        "    // json_object / string path returns a string. Handle both.",
        "    const raw = resp.response;",
        "    if (raw === undefined || raw === null) {",
        f'        throw new Error("Workers AI returned no response for {node_id}");',
        "    }",
        '    const parsed = typeof raw === "string" ? JSON.parse(raw) : raw;',
        f"    return {pascal}Output.parse(parsed);",
        "}",
        "",
    ]
    return "\n".join(parts)


# ───────── Data-flow references → TypeScript source ─────────

# Conservative ASCII identifier shape for emitted object keys. Anything
# that doesn't match is JSON-quoted, which is always-valid TS — so being
# conservative here (no `$`, no Unicode) costs nothing but quoting.
_TS_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def ref_to_ts_expr(ref: str) -> str:
    """Render an ``inputs:`` source reference as a TypeScript expression.

    Every TS-emitting adapter binds the pipeline input to
    ``pipelineInput`` and each node's result to ``<node_id>_result``:

    | Reference                  | Expression                                          |
    |----------------------------|-----------------------------------------------------|
    | ``pipeline.input``         | ``pipelineInput``                                   |
    | ``pipeline.input.f``       | ``pipelineInput["f"]``                              |
    | ``foo.output``             | ``foo_result``                                      |
    | ``foo.output.f``           | ``(foo_result as Record<string, unknown>)["f"]``    |

    Node-output field access goes through a Record cast: on Cloudflare,
    ``step.do``'s ``Rpc.Serializable`` constraint widens inferred result
    types so direct indexing doesn't compile, and on every runtime the
    cast stays valid for whatever concrete return type the user gives a
    stub later.
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


def payload_ts_literal(node: Node, indent: str) -> str:
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
        lines.append(f"{inner}{key}: {ref_to_ts_expr(ref)},")
    lines.append(indent + "}")
    return "\n".join(lines)


FAN_OUT_LIST_HELPER_TS = """\
/** The list a fan_out node dispatches over.
 *
 * A bare `.map()` on a missing key fails with "Cannot read properties
 * of undefined (reading 'map')", which names neither the node nor the
 * reference — in generated code that is close to undebuggable, and the
 * cause is almost always an upstream node that didn't return the key
 * the IR says it does. One call turns that into an actionable message.
 * Found by running it: the first live fan_out run hit exactly this. */
function fanOutList(value: unknown, nodeId: string, ref: string): unknown[] {
    if (!Array.isArray(value)) {
        const got = value === null ? "null" : typeof value;
        throw new Error(
            `fan_out node '${nodeId}' expected an array from '${ref}', got ${got}. ` +
                `The upstream node must return that key as a list.`,
        );
    }
    return value;
}
"""


def fan_out_ts_binding(node: Node, pipeline: Pipeline, indent: str = "") -> tuple[str, str]:
    """``(payload_literal, list_expr)`` for a ``fan_out`` node in TypeScript.

    The payload is a one-line object binding the element param to the
    loop variable ``_item`` and every other input to its shared
    expression — the TS rendering of
    :func:`rote.adapters._common.fan_out_element_param`, which decides
    *which* input is the list.

    The list expression goes through ``fanOutList``
    (:data:`FAN_OUT_LIST_HELPER_TS`), which both narrows the type — node
    results are ``Record<string, unknown>`` or ``never``, so neither
    ``.map`` nor ``for…of`` compiles against them — and reports a
    missing upstream key by node id instead of as a bare TypeError.

    ``indent`` is the column the ``fanOutList(`` token sits at. The call
    wraps against it instead of running to ~180 characters on one line —
    emitted code is meant to be read and reviewed.
    """
    element_param = fan_out_element_param(node, pipeline)
    entries = []
    for param, input_ref in node.inputs.items() if node.inputs else []:
        key = param if _TS_IDENT_RE.fullmatch(param) else json.dumps(param)
        value = "_item" if param == element_param else ref_to_ts_expr(input_ref)
        entries.append(f"{key}: {value}")
    payload = "{ " + ", ".join(entries) + " }"
    assert node.inputs is not None
    ref = node.inputs[element_param]
    inner = indent + "    "
    list_expr = (
        "fanOutList(\n"
        f"{inner}{ref_to_ts_expr(ref)},\n"
        f"{inner}{json.dumps(node.id)},\n"
        f"{inner}{json.dumps(ref)},\n"
        f"{indent})"
    )
    return payload, list_expr


# ───────── Pipeline queries ─────────


def llm_clients(pipeline: Pipeline) -> set[str]:
    """Vendor clients used by the pipeline's llm_judge signature specs."""
    return {
        node.signature_spec.client
        for node in pipeline.nodes
        if node.kind is NodeKind.LLM_JUDGE and node.signature_spec is not None
    }


def module_imports(pipeline: Pipeline, prefix: str = "./") -> str:
    """Build the static-import block for the emitted workflow file.

    One import per non-HITL node (including loop_body sub-nodes): the
    function name is camelCase of the node id; the path is
    ``<prefix>signatures/<id>`` for llm_judge and
    ``<prefix>extracted/<id>`` otherwise. ``prefix`` accommodates
    workflow files that don't sit next to those directories (Inngest's
    ``src/inngest/pipeline.ts`` passes ``"../"``).
    """
    lines: list[str] = []
    for node in pipeline.nodes:
        if node.kind is NodeKind.HITL_GATE:
            continue
        fn = _to_camel_case(node.id)
        if node.kind is NodeKind.LLM_JUDGE:
            lines.append(f'import {{ {fn} }} from "{prefix}signatures/{node.id}";')
        else:
            lines.append(f'import {{ {fn} }} from "{prefix}extracted/{node.id}";')
    return "\n".join(lines)


# ───────── Typed signature module (client dispatch) ─────────


def require_signature_spec(node: Node) -> LLMSignature:
    """Return the node's ``signature_spec`` or fail with the shared message.

    Every TS adapter has the same requirement: the legacy
    ``signature: path.py:Class`` form points at a Python module and
    cannot be emitted to TypeScript.
    """
    if node.signature_spec is None:
        raise ValueError(
            f"TypeScript emission: llm_judge {node.id!r} requires signature_spec "
            f"(structured form). The legacy path form signature: {node.signature!r} "
            f"points at a Python module and cannot be emitted to TypeScript."
        )
    return node.signature_spec


def emit_ts_signature_module(
    node: Node,
    *,
    anthropic_default_model: str,
    openai_default_model: str,
    generated_by: str,
    workers_ai_default_model: str = "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
) -> str:
    """Emit signatures/<node_id>.ts for an llm_judge node.

    The shared front door for every TS adapter: requires
    ``signature_spec``, converts the JSON Schemas to Zod, resolves the
    model default for the spec's client, and dispatches to the
    per-vendor emitter. ``generated_by`` is the calling adapter's module
    name for the header JSDoc.
    """
    spec = require_signature_spec(node)
    if spec.client == "anthropic":
        emitter = emit_signature_anthropic
        model = spec.model or anthropic_default_model
    elif spec.client == "openai":
        emitter = emit_signature_openai
        model = spec.model or openai_default_model
    elif spec.client == "workers-ai":
        emitter = emit_signature_workers_ai
        model = spec.model or workers_ai_default_model
    else:  # pragma: no cover — LLMSignature validates the client field
        raise ValueError(f"Unsupported LLM client: {spec.client!r}")
    return emitter(
        node_id=node.id,
        fn_name=_to_camel_case(node.id),
        pascal=_to_pascal_case(node.id),
        description=node.description,
        input_zod=json_schema_to_zod(spec.input_schema, indent=0),
        output_zod=json_schema_to_zod(spec.output_schema, indent=0),
        output_schema_json=json.dumps(spec.output_schema, indent=2),
        prompt_template=spec.prompt,
        model=model,
        base_url=spec.base_url,
        temperature=spec.temperature,
        generated_by=generated_by,
    )


# ───────── Node-process runtime helpers ─────────
#
# Shared by the TS adapters whose emitted code runs in a plain Node
# process (DBOS-TS, Inngest). Cloudflare Workers pass `this.env` instead,
# so these don't apply there.

REQUIRE_ENV_HELPER = """\
function requireEnv(name: string): string {
    const value = process.env[name];
    if (!value) {
        throw new Error(`missing required environment variable ${name}`);
    }
    return value;
}
"""


def judge_env_arg(node: Node) -> str:
    """The env object a judge step passes to its signature function.

    Besides the required API key, the per-node operator overrides
    (``ROTE_MODEL_<ID>`` / ``ROTE_BASE_URL_<ID>``) are threaded through
    from ``process.env`` so the signature honors them at runtime.
    """
    spec = require_signature_spec(node)
    key = "OPENAI_API_KEY" if spec.client == "openai" else "ANTHROPIC_API_KEY"
    model_var, base_var = override_env_vars(node.id)
    return (
        "{\n"
        f'        {key}: requireEnv("{key}"),\n'
        f"        {model_var}: process.env.{model_var},\n"
        f"        {base_var}: process.env.{base_var},\n"
        "    }"
    )


def emit_node_tsconfig() -> str:
    """tsconfig.json for the Node-process TS runtimes (DBOS-TS, Inngest)."""
    obj = {
        "compilerOptions": {
            "target": "ES2022",
            "module": "NodeNext",
            "moduleResolution": "NodeNext",
            "strict": True,
            "esModuleInterop": True,
            "skipLibCheck": True,
            "outDir": "dist",
            "rootDir": "src",
            "types": ["node"],
        },
        "include": ["src/**/*.ts"],
    }
    return json.dumps(obj, indent=2) + "\n"


# ───────── MCP-backed external_call emission (Node runtimes) ─────────

#: The npm package pin for the official MCP TypeScript SDK. v1.x is the
#: supported production line (the repo's main branch is v2 beta with
#: renamed packages); verified against npm 2026-07-10.
MCP_SDK_NPM_VERSION = "^1.29.0"

#: Source of ``src/extracted/_roteMcp.ts`` — the Node-side twin of
#: ``rote.mcp._runtime_helper``. It reads the SAME registry and token
#: files the rote CLI writes (the documented cross-language contract in
#: docs/mcp-client.md), refreshes stale access tokens with a plain
#: refresh-token grant against the stored token_endpoint (no discovery,
#: no browser — deliberately NOT the SDK's authProvider, which insists
#: on running discovery and can silently fall through to a new
#: authorization flow), and persists rotated refresh tokens atomically
#: so the CLI and other processes see them. Node-only (fs); the
#: Cloudflare adapter must NOT emit this file.
ROTE_MCP_HELPER_TS = """/**
 * MCP connection helper (generated by rote — do not edit; the source
 * of truth is rote.adapters._ts_common.ROTE_MCP_HELPER_TS).
 *
 * Mirrors the rote CLI's behavior exactly:
 * - Endpoint: ROTE_MCP_<SERVER>_URL env var > the rote registry
 *   (~/.config/rote/mcp.json) > the endpoint recorded in the pipeline.
 * - Auth: the rote token store (~/.local/share/rote/mcp-tokens/<server>.json,
 *   written by `rote mcp login`), refreshing stale access tokens via the
 *   OAuth refresh grant and writing rotated tokens back atomically; else
 *   static headers from the registry entry; else unauthenticated.
 * - Never interactive: anything that needs a human (no refresh path, a
 *   401 the refresh cannot fix) throws RoteMcpAuthNeeded — the emitted
 *   workflow parks durably on it until `rote mcp login <server>` (or a
 *   re-provisioned credential) releases it.
 */

import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import {
    StreamableHTTPClientTransport,
    StreamableHTTPError,
} from "@modelcontextprotocol/sdk/client/streamableHttp.js";

/**
 * The MCP server needs a human to (re)authenticate.
 *
 * Detect with isRoteMcpAuthNeeded(), never `instanceof`: durable runtimes
 * (DBOS, Inngest) serialize step errors across replay/process boundaries
 * and reconstruct plain Errors — class identity is lost, but `name` and
 * own enumerable properties survive.
 */
export class RoteMcpAuthNeeded extends Error {
    readonly server: string;
    readonly reason: string;

    constructor(server: string, reason: string) {
        super(
            `MCP server '${server}' needs (re)authentication — ${reason}. ` +
                `Run: rote mcp login ${server}`,
        );
        this.name = "RoteMcpAuthNeeded";
        this.server = server;
        this.reason = reason;
    }
}

/** True if a RoteMcpAuthNeeded hides anywhere in the error tree
 * (cause chains from step wrappers, `errors` arrays from retry
 * exhaustion / AggregateError). */
export function isRoteMcpAuthNeeded(err: unknown): boolean {
    let node: unknown = err;
    for (let depth = 0; node != null && depth < 16; depth++) {
        const e = node as { name?: unknown; errors?: unknown; cause?: unknown };
        if (e.name === "RoteMcpAuthNeeded") return true;
        if (Array.isArray(e.errors) && e.errors.some((sub) => isRoteMcpAuthNeeded(sub))) {
            return true;
        }
        node = e.cause;
    }
    return false;
}

interface TokenDoc {
    version?: number;
    server_url?: string;
    tokens: {
        access_token?: string;
        token_type?: string;
        expires_in?: number;
        refresh_token?: string;
        scope?: string;
    } | null;
    expires_at: number | null;
    client_info: { client_id?: string; client_secret?: string } | null;
    token_endpoint: string | null;
}

interface RegistryEntry {
    url?: string;
    client_id?: string;
    client_secret?: string;
    headers?: Record<string, string>;
}

function registryServers(): Record<string, RegistryEntry> {
    const override = process.env.ROTE_MCP_CONFIG;
    const base = process.env.XDG_CONFIG_HOME ?? path.join(os.homedir(), ".config");
    const file = override ?? path.join(base, "rote", "mcp.json");
    if (!fs.existsSync(file)) return {};
    const data = JSON.parse(fs.readFileSync(file, "utf-8")) as {
        servers?: Record<string, RegistryEntry>;
    };
    return data.servers ?? {};
}

function tokenDir(): string {
    const override = process.env.ROTE_MCP_TOKEN_DIR;
    if (override) return override;
    const base = process.env.XDG_DATA_HOME ?? path.join(os.homedir(), ".local", "share");
    return path.join(base, "rote", "mcp-tokens");
}

export function resolveUrl(server: string, pipelineUrl: string | null): string {
    const envUrl = process.env[`ROTE_MCP_${server.toUpperCase()}_URL`];
    if (envUrl) return envUrl;
    const entry = registryServers()[server];
    if (entry?.url) return entry.url;
    if (pipelineUrl) return pipelineUrl;
    throw new Error(
        `no endpoint for MCP server '${server}' — register it ` +
            `(rote mcp add ${server} <url>) or set ROTE_MCP_${server.toUpperCase()}_URL`,
    );
}

function readTokenDoc(server: string): TokenDoc | null {
    const file = path.join(tokenDir(), `${server}.json`);
    if (!fs.existsSync(file)) return null;
    return JSON.parse(fs.readFileSync(file, "utf-8")) as TokenDoc;
}

function writeTokenDoc(server: string, doc: TokenDoc): void {
    // Same discipline as the Python side: atomic replace, 0600. The CLI
    // and other workflows read this file concurrently.
    const dir = tokenDir();
    fs.mkdirSync(dir, { recursive: true });
    doc.version = doc.version ?? 1;
    const tmp = path.join(dir, `.${server}-${process.pid}-${Date.now()}.tmp`);
    fs.writeFileSync(tmp, JSON.stringify(doc, null, 2) + "\\n", { mode: 0o600 });
    fs.renameSync(tmp, path.join(dir, `${server}.json`));
}

async function refreshAccessToken(server: string, doc: TokenDoc): Promise<string> {
    const refreshToken = doc.tokens?.refresh_token;
    if (!refreshToken) {
        throw new RoteMcpAuthNeeded(
            server,
            "the access token is expired and no refresh token is stored",
        );
    }
    if (!doc.token_endpoint) {
        throw new RoteMcpAuthNeeded(
            server,
            "the token store has no token_endpoint (written by rote mcp login)",
        );
    }
    const entry = registryServers()[server];
    const clientId = doc.client_info?.client_id ?? entry?.client_id;
    const clientSecret = doc.client_info?.client_secret ?? entry?.client_secret;
    if (!clientId) {
        throw new RoteMcpAuthNeeded(server, "no client_id is stored for the refresh grant");
    }
    const body = new URLSearchParams({
        grant_type: "refresh_token",
        refresh_token: refreshToken,
        client_id: clientId,
    });
    if (clientSecret) body.set("client_secret", clientSecret);
    const response = await fetch(doc.token_endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: body.toString(),
    });
    if (!response.ok) {
        throw new RoteMcpAuthNeeded(server, `the refresh grant failed (${response.status})`);
    }
    const granted = (await response.json()) as {
        access_token: string;
        expires_in?: number;
        refresh_token?: string;
        token_type?: string;
        scope?: string;
    };
    // Persist the rotated credentials: some servers rotate the refresh
    // token on every use — losing the new one strands every consumer.
    doc.tokens = {
        ...doc.tokens,
        access_token: granted.access_token,
        token_type: granted.token_type ?? doc.tokens?.token_type ?? "Bearer",
        expires_in: granted.expires_in,
        refresh_token: granted.refresh_token ?? refreshToken,
        scope: granted.scope ?? doc.tokens?.scope,
    };
    doc.expires_at =
        granted.expires_in != null ? Date.now() / 1000 + granted.expires_in : null;
    writeTokenDoc(server, doc);
    return granted.access_token;
}

/** Currently-valid auth headers for `server`, or null (unauthenticated). */
export async function freshHeaders(
    server: string,
    forceRefresh = false,
): Promise<Record<string, string> | null> {
    const entry = registryServers()[server];
    if (entry?.headers && Object.keys(entry.headers).length > 0) {
        return { ...entry.headers };
    }
    const doc = readTokenDoc(server);
    const access = doc?.tokens?.access_token;
    if (!doc || !access) return null;
    const staleAt = doc.expires_at != null ? doc.expires_at - 60 : null;
    const isStale = forceRefresh || (staleAt != null && Date.now() / 1000 >= staleAt);
    const token = isStale ? await refreshAccessToken(server, doc) : access;
    return { Authorization: `Bearer ${token}` };
}

/**
 * Run `fn` against a connected, authenticated MCP session.
 *
 * Every operation shares one auth story: try with current credentials,
 * and on a 401 (revoked or expired-early token) refresh once and retry
 * on a fresh transport. Anything a refresh cannot fix becomes
 * RoteMcpAuthNeeded so the workflow parks instead of failing.
 */
async function withMcpSession<T>(
    server: string,
    pipelineUrl: string | null,
    fn: (client: Client) => Promise<T>,
): Promise<T> {
    const url = resolveUrl(server, pipelineUrl);

    const attempt = async (headers: Record<string, string> | null) => {
        const client = new Client({ name: "rote-pipeline", version: "1.0.0" });
        const transport = new StreamableHTTPClientTransport(new URL(url), {
            requestInit: headers ? { headers } : undefined,
        });
        try {
            await client.connect(transport);
            return await fn(client);
        } finally {
            await client.close().catch(() => undefined);
        }
    };

    try {
        return await attempt(await freshHeaders(server));
    } catch (err) {
        if (!isUnauthorized(err)) throw err;
        if (!readTokenDoc(server)?.tokens?.refresh_token) {
            throw new RoteMcpAuthNeeded(server, "the server returned 401 Unauthorized");
        }
        try {
            return await attempt(await freshHeaders(server, true));
        } catch (retryErr) {
            if (isUnauthorized(retryErr)) {
                throw new RoteMcpAuthNeeded(
                    server,
                    "the server returned 401 even after a token refresh",
                );
            }
            throw retryErr;
        }
    }
}

/**
 * Call one tool on an MCP server, authenticated from the rote credential
 * store. On a 401 (revoked/expired-early token) the call refreshes once
 * and retries with a fresh transport.
 */
export async function callMcpTool(
    server: string,
    pipelineUrl: string | null,
    tool: string,
    args: Record<string, unknown>,
): Promise<Record<string, unknown>> {
    return withMcpSession(server, pipelineUrl, async (client) => {
        const result = await client.callTool({ name: tool, arguments: args });
        if (result.isError) {
            throw new Error(
                `MCP tool '${tool}' on '${server}' returned an error: ` +
                    JSON.stringify(result.content),
            );
        }
        if (result.structuredContent) {
            return result.structuredContent as Record<string, unknown>;
        }
        const text = (result.content as Array<{ type: string; text?: string }>).find(
            (block) => block.type === "text",
        )?.text;
        return (text ? JSON.parse(text) : {}) as Record<string, unknown>;
    });
}

/** One tool as the server advertises it. */
export interface McpToolSpec {
    name: string;
    description: string;
    inputSchema: Record<string, unknown>;
}

/**
 * Every tool `server` advertises, with the JSON Schema it declares.
 *
 * An agent_loop names bare tools; the schema the model needs to call one
 * correctly lives on the server, not in the IR. Fetching it here means
 * the agent's tool contract is always the server's current contract —
 * it cannot go stale the way a schema copied into the pipeline would.
 */
export async function listMcpTools(
    server: string,
    pipelineUrl: string | null,
): Promise<McpToolSpec[]> {
    return withMcpSession(server, pipelineUrl, async (client) => {
        const { tools } = await client.listTools();
        return tools.map((tool) => ({
            name: tool.name,
            description: tool.description ?? "",
            inputSchema: (tool.inputSchema ?? {
                type: "object",
                properties: {},
            }) as Record<string, unknown>,
        }));
    });
}

/** A tool bound to a callable — structurally a BoundTool for the agent loop. */
export interface BoundMcpTool {
    name: string;
    description: string;
    parameters: Record<string, unknown>;
    run: (args: Record<string, unknown>) => Promise<Record<string, unknown>>;
}

/**
 * Candidate servers for tool discovery: every server the pipeline knows
 * about, plus every server registered locally.
 *
 * Over-wiring is safe *because* the declared tool list is the real
 * constraint — the agent can only call what the IR named. ROTE_MCP_SERVERS
 * replaces the whole list when an operator needs to point a loop somewhere
 * the pipeline never mentioned.
 */
function agentServerNames(declared: string[]): string[] {
    const override = process.env.ROTE_MCP_SERVERS;
    if (override) {
        return override
            .split(",")
            .map((name) => name.trim())
            .filter(Boolean);
    }
    const names = new Set(declared);
    for (const name of Object.keys(registryServers())) names.add(name);
    return [...names].sort();
}

/**
 * Bind the tools an agent_loop declared to callables the loop can invoke.
 *
 * The IR names tools without servers (one loop may reach several, and an
 * MCPBinding is a single server/tool pair), so discovery walks every
 * candidate server and keeps what matches the declared names.
 *
 * Discovery is best-effort per server: a server that is down or
 * unauthenticated is not necessarily one this loop needed, so it is
 * recorded rather than thrown. A declared tool that no reachable server
 * provides IS fatal — and when an auth failure is the plausible reason,
 * that failure is rethrown so the workflow parks durably and
 * `rote mcp login` can release it, rather than dying with a misleading
 * "no server provides this tool".
 */
export async function bindAgentTools(
    allowed: string[],
    declaredServers: string[],
    serverUrls: Record<string, string | null> = {},
    toolServers: Record<string, string> = {},
): Promise<BoundMcpTool[]> {
    const wanted = new Set(allowed);
    const bound = new Map<string, BoundMcpTool>();
    const failures: unknown[] = [];

    for (const server of agentServerNames(declaredServers)) {
        const url = serverUrls[server] ?? null;
        let specs: McpToolSpec[];
        try {
            specs = await listMcpTools(server, url);
        } catch (err) {
            failures.push(err);
            continue;
        }
        for (const spec of specs) {
            if (!wanted.has(spec.name) || bound.has(spec.name)) continue;
            // A tool the IR resolved to a server binds ONLY from that
            // server: two servers exporting the same tool name is exactly
            // the case an allowlist cannot disambiguate, and silently
            // taking whichever answered first is how a loop ends up
            // talking to the wrong endpoint. Unresolved tools keep the
            // first-wins fallback — there is nothing better to go on.
            const pinned = toolServers[spec.name];
            if (pinned !== undefined && pinned !== server) continue;
            bound.set(spec.name, {
                name: spec.name,
                description: spec.description,
                parameters: spec.inputSchema,
                run: (args) => callMcpTool(server, url, spec.name, args),
            });
        }
    }

    const missing = allowed.filter((name) => !bound.has(name));
    if (missing.length > 0) {
        const authFailure = failures.find((err) => isRoteMcpAuthNeeded(err));
        if (authFailure) throw authFailure;
        const detail = failures
            .map((err) => (err instanceof Error ? err.message : String(err)))
            .join("; ");
        throw new Error(
            `agent tools not provided by any reachable MCP server: ${missing.join(", ")}` +
                (detail ? ` (unreachable: ${detail})` : ""),
        );
    }
    return [...bound.values()];
}

function isUnauthorized(err: unknown): boolean {
    return err instanceof StreamableHTTPError
        ? err.code === 401
        : /\\b401\\b|[Uu]nauthorized/.test(String(err));
}
"""


def mcp_backed_nodes(pipeline: Pipeline, external_backend: str) -> list[Node]:
    """The nodes that emit working MCP calls instead of stubs."""
    if external_backend != "mcp":
        return []
    return [n for n in pipeline.nodes if n.mcp is not None]


def emit_mcp_call_module(node: Node, *, generated_by: str) -> str:
    """Emit ``src/extracted/<id>.ts`` as a WORKING MCP-backed call.

    Same module path and exported function name as the stub it replaces,
    so the workflow's imports and step registrations are unchanged — only
    the body differs. Auth and endpoint resolution live in the emitted
    ``_roteMcp`` helper (see :data:`ROTE_MCP_HELPER_TS`).
    """
    binding = node.mcp
    assert binding is not None
    fn_name = _to_camel_case(node.id)
    desc_first = safe_block_comment_line(node.description, fallback=node.id)
    url_literal = json.dumps(binding.url) if binding.url is not None else "null"
    if binding.args:
        arg_items = ", ".join(
            f"{json.dumps(tool_arg)}: payload[{json.dumps(payload_key)}]"
            for tool_arg, payload_key in binding.args.items()
        )
        args_line = f"    const args = {{ {arg_items} }};"
    else:
        args_line = "    const args = payload; // tool arg names match the threaded payload keys"
    mandatory_lines = (
        [
            " *",
            " * MANDATORY: this node was marked mandatory in the source skill.",
            " * The workflow always calls it; do not make it conditional.",
        ]
        if node.mandatory
        else []
    )
    lines = [
        "/**",
        f" * MCP-backed {node.kind.value}: {node.id}",
        " *",
        f" * {desc_first}",
        " *",
        f" * Auto-generated by {generated_by}: invokes tool {binding.tool!r} on MCP",
        f" * server {binding.server!r} over Streamable HTTP, authenticated from the",
        f" * rote credential store (`rote mcp login {binding.server}`). Swap to a",
        " * direct vendor-SDK call with `rote emit --backend api`.",
        *mandatory_lines,
        " */",
        "",
        'import { callMcpTool } from "./_roteMcp";',
        "",
        f"export async function {fn_name}(input: unknown): Promise<Record<string, unknown>> {{",
        "    const payload = (input ?? {}) as Record<string, unknown>;",
        args_line,
        f"    return callMcpTool({json.dumps(binding.server)}, {url_literal}, "
        f"{json.dumps(binding.tool)}, args);",
        "}",
        "",
    ]
    return "\n".join(lines)


#: Source of the Cloudflare Workers MCP helper. Workers have no
#: filesystem and no subprocess, so the rote token store cannot be
#: read at runtime — credentials are PROVISIONED instead:
#: ``rote mcp export <server>`` turns a completed ``rote mcp login``
#: into Worker secrets (refresh token, client id/secret, token
#: endpoint), the emitted code refreshes access tokens at runtime, and
#: rotated refresh tokens persist to the ``ROTE_MCP_TOKENS`` KV binding
#: (in-memory fallback per isolate when KV is absent — fine for
#: servers that don't rotate refresh tokens; the README says which).
ROTE_MCP_WORKERS_HELPER_TS = """/**
 * MCP connection helper for Cloudflare Workers (generated by rote — do
 * not edit; the source of truth is
 * rote.adapters._ts_common.ROTE_MCP_WORKERS_HELPER_TS).
 *
 * Credentials come from Worker secrets provisioned by
 * `rote mcp export <server>` (see README):
 *   ROTE_MCP_<SERVER>_REFRESH_TOKEN / _CLIENT_ID / _CLIENT_SECRET? /
 *   _TOKEN_ENDPOINT / _URL?
 * Access tokens are minted at runtime via the OAuth refresh grant and
 * cached — with rotated refresh tokens — in the ROTE_MCP_TOKENS KV
 * namespace, so every isolate sees the latest credentials.
 */

import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import {
    StreamableHTTPClientTransport,
    StreamableHTTPError,
} from "@modelcontextprotocol/sdk/client/streamableHttp.js";

export interface RoteMcpEnv {
    ROTE_MCP_TOKENS?: KVNamespace;
    [key: string]: unknown;
}

/**
 * The MCP server needs re-provisioned credentials (Workers cannot run an
 * interactive login — the fix is `rote mcp login` + `rote mcp export` +
 * re-uploading the secrets). Detect with isRoteMcpAuthNeeded(), never
 * `instanceof`: Workflows serialize errors across hibernation and replay
 * boundaries — class identity is lost, `name` survives.
 */
export class RoteMcpAuthNeeded extends Error {
    readonly server: string;
    readonly reason: string;

    constructor(server: string, reason: string) {
        super(
            `MCP server '${server}' needs (re)authentication — ${reason}. ` +
                `Re-provision with: rote mcp login ${server} && rote mcp export ${server}`,
        );
        this.name = "RoteMcpAuthNeeded";
        this.server = server;
        this.reason = reason;
    }
}

/** True if a RoteMcpAuthNeeded hides anywhere in the error tree. */
export function isRoteMcpAuthNeeded(err: unknown): boolean {
    let node: unknown = err;
    for (let depth = 0; node != null && depth < 16; depth++) {
        const e = node as { name?: unknown; errors?: unknown; cause?: unknown };
        if (e.name === "RoteMcpAuthNeeded") return true;
        if (Array.isArray(e.errors) && e.errors.some((sub) => isRoteMcpAuthNeeded(sub))) {
            return true;
        }
        node = e.cause;
    }
    return false;
}

// Concrete JSON types: Cloudflare's step.do constrains callback returns to
// Rpc.Serializable, which `unknown` members do not satisfy — a recursive
// concrete JSON shape does (same reason emitted stubs are Promise<never>).
export type JsonValue = string | number | boolean | null | JsonValue[] | JsonObject;
export type JsonObject = { [key: string]: JsonValue };

interface TokenState {
    access_token: string;
    expires_at: number | null;
    refresh_token?: string;
}

/** Per-isolate fallback cache when no KV binding is configured. */
const memoryCache = new Map<string, TokenState>();

function secret(env: RoteMcpEnv, server: string, suffix: string): string | undefined {
    const value = env[`ROTE_MCP_${server.toUpperCase()}_${suffix}`];
    return typeof value === "string" && value.length > 0 ? value : undefined;
}

export function resolveUrl(
    env: RoteMcpEnv,
    server: string,
    pipelineUrl: string | null,
): string {
    const bound = secret(env, server, "URL") ?? pipelineUrl;
    if (bound) return bound;
    throw new Error(
        `no endpoint for MCP server '${server}' — set the ` +
            `ROTE_MCP_${server.toUpperCase()}_URL variable (wrangler.jsonc vars or secret)`,
    );
}

async function readState(env: RoteMcpEnv, server: string): Promise<TokenState | null> {
    if (env.ROTE_MCP_TOKENS) {
        return (await env.ROTE_MCP_TOKENS.get(`mcp-token:${server}`, "json")) as
            | TokenState
            | null;
    }
    return memoryCache.get(server) ?? null;
}

async function writeState(env: RoteMcpEnv, server: string, state: TokenState): Promise<void> {
    if (env.ROTE_MCP_TOKENS) {
        await env.ROTE_MCP_TOKENS.put(`mcp-token:${server}`, JSON.stringify(state));
    } else {
        memoryCache.set(server, state);
    }
}

async function refreshAccessToken(env: RoteMcpEnv, server: string): Promise<string> {
    const state = await readState(env, server);
    // A rotated refresh token in KV supersedes the provisioned secret.
    const refreshToken = state?.refresh_token ?? secret(env, server, "REFRESH_TOKEN");
    const tokenEndpoint = secret(env, server, "TOKEN_ENDPOINT");
    const clientId = secret(env, server, "CLIENT_ID");
    const upper = server.toUpperCase();
    if (!refreshToken || !tokenEndpoint || !clientId) {
        throw new RoteMcpAuthNeeded(
            server,
            `the ROTE_MCP_${upper}_* secrets are not provisioned (wrangler secret bulk)`,
        );
    }
    const body = new URLSearchParams({
        grant_type: "refresh_token",
        refresh_token: refreshToken,
        client_id: clientId,
    });
    const clientSecret = secret(env, server, "CLIENT_SECRET");
    if (clientSecret) body.set("client_secret", clientSecret);
    const response = await fetch(tokenEndpoint, {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: body.toString(),
    });
    if (!response.ok) {
        throw new RoteMcpAuthNeeded(
            server,
            `the refresh grant failed (${response.status}) — the provisioned ` +
                `refresh token may be revoked`,
        );
    }
    const granted = (await response.json()) as {
        access_token: string;
        expires_in?: number;
        refresh_token?: string;
    };
    await writeState(env, server, {
        access_token: granted.access_token,
        expires_at:
            granted.expires_in != null ? Date.now() / 1000 + granted.expires_in : null,
        // Persist rotation — losing a rotated refresh token strands the Worker.
        refresh_token: granted.refresh_token ?? refreshToken,
    });
    return granted.access_token;
}

async function accessToken(
    env: RoteMcpEnv,
    server: string,
    forceRefresh: boolean,
): Promise<string> {
    if (!forceRefresh) {
        const state = await readState(env, server);
        if (
            state?.access_token &&
            (state.expires_at == null || Date.now() / 1000 < state.expires_at - 60)
        ) {
            return state.access_token;
        }
    }
    return refreshAccessToken(env, server);
}

/**
 * Run `fn` against a connected, authenticated MCP session.
 *
 * Every operation shares one auth story: try with the cached token, and
 * on a 401 refresh once and retry on a fresh transport. Anything a
 * refresh cannot fix becomes RoteMcpAuthNeeded so the instance parks.
 */
async function withMcpSession<T>(
    env: RoteMcpEnv,
    server: string,
    pipelineUrl: string | null,
    fn: (client: Client) => Promise<T>,
): Promise<T> {
    const url = resolveUrl(env, server, pipelineUrl);

    const attempt = async (token: string) => {
        const client = new Client({ name: "rote-pipeline", version: "1.0.0" });
        const transport = new StreamableHTTPClientTransport(new URL(url), {
            requestInit: { headers: { Authorization: `Bearer ${token}` } },
        });
        try {
            await client.connect(transport);
            return await fn(client);
        } finally {
            await client.close().catch(() => undefined);
        }
    };

    try {
        return await attempt(await accessToken(env, server, false));
    } catch (err) {
        if (!isUnauthorizedWorkers(err)) throw err;
        try {
            return await attempt(await accessToken(env, server, true));
        } catch (retryErr) {
            if (isUnauthorizedWorkers(retryErr)) {
                throw new RoteMcpAuthNeeded(
                    server,
                    "the server returned 401 even after a token refresh",
                );
            }
            throw retryErr;
        }
    }
}

/**
 * Call one tool on an MCP server, authenticated from provisioned Worker
 * secrets + KV-cached tokens. Retries once with a forced refresh on 401.
 */
export async function callMcpTool(
    env: RoteMcpEnv,
    server: string,
    pipelineUrl: string | null,
    tool: string,
    args: Record<string, unknown>,
): Promise<JsonObject> {
    return withMcpSession(env, server, pipelineUrl, async (client) => {
        const result = await client.callTool({ name: tool, arguments: args });
        if (result.isError) {
            throw new Error(
                `MCP tool '${tool}' on '${server}' returned an error: ` +
                    JSON.stringify(result.content),
            );
        }
        if (result.structuredContent) {
            return result.structuredContent as JsonObject;
        }
        const text = (result.content as Array<{ type: string; text?: string }>).find(
            (block) => block.type === "text",
        )?.text;
        return (text ? JSON.parse(text) : {}) as JsonObject;
    });
}

/** One tool as the server advertises it. */
export interface McpToolSpec {
    name: string;
    description: string;
    inputSchema: Record<string, unknown>;
}

/**
 * Every tool `server` advertises, with the JSON Schema it declares.
 *
 * An agent_loop names bare tools; the schema the model needs to call one
 * correctly lives on the server, not in the IR. Fetching it here means
 * the agent's tool contract is always the server's current contract —
 * it cannot go stale the way a schema copied into the pipeline would.
 */
export async function listMcpTools(
    env: RoteMcpEnv,
    server: string,
    pipelineUrl: string | null,
): Promise<McpToolSpec[]> {
    return withMcpSession(env, server, pipelineUrl, async (client) => {
        const { tools } = await client.listTools();
        return tools.map((tool) => ({
            name: tool.name,
            description: tool.description ?? "",
            inputSchema: (tool.inputSchema ?? {
                type: "object",
                properties: {},
            }) as Record<string, unknown>,
        }));
    });
}

/** A tool bound to a callable — structurally a BoundTool for the agent loop. */
export interface BoundMcpTool {
    name: string;
    description: string;
    parameters: Record<string, unknown>;
    run: (args: Record<string, unknown>) => Promise<JsonObject>;
}

/**
 * Candidate servers for tool discovery.
 *
 * Unlike the Node runtimes there is no registry file to enumerate here —
 * a Worker's servers are whatever the pipeline declared at emit time,
 * with their credentials provisioned as secrets under those names. The
 * ROTE_MCP_SERVERS variable replaces the list wholesale when an operator
 * needs to point a loop somewhere the pipeline never mentioned.
 */
function agentServerNames(env: RoteMcpEnv, declared: string[]): string[] {
    const override = env.ROTE_MCP_SERVERS;
    if (typeof override === "string" && override.length > 0) {
        return override
            .split(",")
            .map((name) => name.trim())
            .filter(Boolean);
    }
    return [...new Set(declared)].sort();
}

/**
 * Bind the tools an agent_loop declared to callables the loop can invoke.
 *
 * Same contract as the Node helper's: discovery is best-effort per server
 * (a server that is down may not be one this loop needed), a declared tool
 * no reachable server provides is fatal, and an auth failure that plausibly
 * explains the gap is rethrown so the Workflow parks on it instead of
 * failing with a misleading "no server provides this tool".
 */
export async function bindAgentTools(
    env: RoteMcpEnv,
    allowed: string[],
    declaredServers: string[],
    serverUrls: Record<string, string | null> = {},
    toolServers: Record<string, string> = {},
): Promise<BoundMcpTool[]> {
    const wanted = new Set(allowed);
    const bound = new Map<string, BoundMcpTool>();
    const failures: unknown[] = [];

    for (const server of agentServerNames(env, declaredServers)) {
        const url = serverUrls[server] ?? null;
        let specs: McpToolSpec[];
        try {
            specs = await listMcpTools(env, server, url);
        } catch (err) {
            failures.push(err);
            continue;
        }
        for (const spec of specs) {
            if (!wanted.has(spec.name) || bound.has(spec.name)) continue;
            // Resolved tools bind only from their own server — see the
            // Node helper's twin for why first-wins is not good enough.
            const pinned = toolServers[spec.name];
            if (pinned !== undefined && pinned !== server) continue;
            bound.set(spec.name, {
                name: spec.name,
                description: spec.description,
                parameters: spec.inputSchema,
                run: (args) => callMcpTool(env, server, url, spec.name, args),
            });
        }
    }

    const missing = allowed.filter((name) => !bound.has(name));
    if (missing.length > 0) {
        const authFailure = failures.find((err) => isRoteMcpAuthNeeded(err));
        if (authFailure) throw authFailure;
        const detail = failures
            .map((err) => (err instanceof Error ? err.message : String(err)))
            .join("; ");
        throw new Error(
            `agent tools not provided by any reachable MCP server: ${missing.join(", ")}` +
                (detail ? ` (unreachable: ${detail})` : ""),
        );
    }
    return [...bound.values()];
}

function isUnauthorizedWorkers(err: unknown): boolean {
    return err instanceof StreamableHTTPError
        ? err.code === 401
        : /\\b401\\b|[Uu]nauthorized/.test(String(err));
}
"""


def emit_workers_mcp_call_module(
    node: Node, *, generated_by: str, mcp_client: str = "direct"
) -> str:
    """Emit ``src/extracted/<id>.ts`` as a working Workers MCP call.

    Same module path and function name as the stub, but the function
    takes ``env`` (the workflow passes ``this.env``) — Workers
    credentials live on the environment, not a filesystem.
    """
    binding = node.mcp
    assert binding is not None
    fn_name = _to_camel_case(node.id)
    desc_first = safe_block_comment_line(node.description, fallback=node.id)
    url_literal = json.dumps(binding.url) if binding.url is not None else "null"
    if binding.args:
        arg_items = ", ".join(
            f"{json.dumps(tool_arg)}: payload[{json.dumps(payload_key)}]"
            for tool_arg, payload_key in binding.args.items()
        )
        args_line = f"    const args = {{ {arg_items} }};"
    else:
        args_line = "    const args = payload; // tool arg names match the threaded payload keys"
    mandatory_lines = (
        [
            " *",
            " * MANDATORY: this node was marked mandatory in the source skill.",
            " * The workflow always calls it; do not make it conditional.",
        ]
        if node.mandatory
        else []
    )
    lines = [
        "/**",
        f" * MCP-backed {node.kind.value}: {node.id}",
        " *",
        f" * {desc_first}",
        " *",
        f" * Auto-generated by {generated_by}: invokes tool {binding.tool!r} on MCP",
        # The provisioning sentence has to match the helper this build
        # actually emitted. Binding mode reads no per-server secrets at
        # all, and this header is the file an operator opens when a call
        # fails, so naming the wrong mechanism sends them to configure
        # secrets nothing will ever read.
        *(
            [
                f" * server {binding.server!r} through the platform's ROTE_MCP service",
                " * binding. The platform provisions the connection; this build reads",
                " * no per-server secrets.",
            ]
            if mcp_client == "binding"
            else [
                f" * server {binding.server!r} over Streamable HTTP, authenticated from",
                f" * Worker secrets provisioned by `rote mcp export {binding.server}`.",
            ]
        ),
        " * Swap to a direct vendor-SDK call with `rote emit --backend api`.",
        *mandatory_lines,
        " */",
        "",
        'import { callMcpTool, type RoteMcpEnv } from "./_roteMcp";',
        "",
        "// Declared Promise<never> for the same reason emitted stubs are:",
        "// step.do constrains callback returns to Rpc.Serializable, which",
        "// arbitrary JSON cannot express without blowing up type",
        "// instantiation. The runtime value is the tool's JSON result; the",
        "// workflow reads fields through its usual `as Record<...>` casts.",
        f"export async function {fn_name}(",
        "    input: unknown,",
        "    env: unknown, // the workflow's Env; cast below (RoteMcpEnv needs an index signature)",
        "): Promise<never> {",
        "    const payload = (input ?? {}) as Record<string, unknown>;",
        args_line,
        f"    return (await callMcpTool(env as RoteMcpEnv, {json.dumps(binding.server)}, "
        f"{url_literal}, {json.dumps(binding.tool)}, args)) as never;",
        "}",
        "",
    ]
    return "\n".join(lines)


#: Source of the BINDING variant of ``src/extracted/_roteMcp.ts`` —
#: emitted when the Cloudflare adapter runs with ``mcp_client="binding"``.
#: Same exported surface as :data:`ROTE_MCP_WORKERS_HELPER_TS` (the node
#: modules and agent loops import both identically), but every operation
#: is delegated to a platform-provisioned ``ROTE_MCP`` service binding:
#: no token store, no OAuth refresh, no per-server secrets, and no MCP
#: SDK dependency. Every proxy call carries a signed caller-auth object
#: built from dispatcher-injected env vars (``ROTE_TENANT_ID`` /
#: ``ROTE_PIPELINE`` / ``ROTE_RUN_ID`` / ``ROTE_MCP_SIG``) so tenant
#: isolates cannot impersonate each other. The platform proxy signals a
#: credential problem with a plain Error whose message starts with
#: ``ROTE_MCP_AUTH_NEEDED:`` (RPC boundaries strip class identity, so
#: the signal rides in the message); the helper rethrows that as
#: RoteMcpAuthNeeded so the emitted park-on-auth loop works unchanged.
ROTE_MCP_BINDING_HELPER_TS = """/**
 * MCP helper for platform-managed connections (generated by rote — do
 * not edit; the source of truth is
 * rote.adapters._ts_common.ROTE_MCP_BINDING_HELPER_TS).
 *
 * Every MCP operation is delegated to the `ROTE_MCP` service binding —
 * an RPC stub the hosting platform provisions on this Worker. The
 * platform owns endpoint resolution, credentials, and token refresh, so
 * this variant carries no token store, no OAuth, and no per-server
 * secrets. Every call carries a signed caller-auth object built from
 * the dispatcher-injected ROTE_TENANT_ID / ROTE_PIPELINE / ROTE_RUN_ID /
 * ROTE_MCP_SIG variables, so tenant isolates cannot impersonate each
 * other. When the platform proxy reports a missing or dead credential
 * (a plain Error whose message starts with "ROTE_MCP_AUTH_NEEDED:"),
 * the helper rethrows RoteMcpAuthNeeded so the workflow's park-on-auth
 * loop suspends the instance until the connection is re-authorized.
 */

/** One tool as the server advertises it. */
export interface McpToolSpec {
    name: string;
    description: string;
    inputSchema: Record<string, unknown>;
}

/** ROTE_MCP_SIG authenticates tenant + pipeline; run_id is attribution. */
export interface RoteMcpAuth {
    tenant_id: string;
    pipeline: string;
    run_id?: string;
    sig: string;
}

/** One call's outcome, as the proxy reports it. */
export interface RoteMcpCallResult {
    /** Text string, or the tool's structured object. */
    content: unknown;
    is_error: boolean;
}

/** One tool as the proxy advertises it (snake_case wire shape). */
export interface RoteMcpToolInfo {
    name: string;
    description: string | null;
    input_schema: unknown;
}

/** The platform's MCP proxy, provisioned as an RPC service binding. */
export interface RoteMcpBinding {
    call(
        auth: RoteMcpAuth,
        server: string,
        tool: string,
        args: Record<string, unknown>,
    ): Promise<RoteMcpCallResult>;
    listTools(auth: RoteMcpAuth, server: string): Promise<RoteMcpToolInfo[]>;
}

export interface RoteMcpEnv {
    ROTE_MCP: RoteMcpBinding;
    [key: string]: unknown;
}

/**
 * The MCP server's platform connection needs to be re-authorized.
 * Detect with isRoteMcpAuthNeeded(), never `instanceof`: Workflows
 * serialize errors across hibernation and replay boundaries — class
 * identity is lost, `name` survives.
 */
export class RoteMcpAuthNeeded extends Error {
    readonly server: string;
    readonly reason: string;

    constructor(server: string, reason: string) {
        super(
            `MCP server '${server}' needs (re)authentication — ${reason}. ` +
                `Re-authorize the connection on the platform hosting this worker.`,
        );
        this.name = "RoteMcpAuthNeeded";
        this.server = server;
        this.reason = reason;
    }
}

/** True if a RoteMcpAuthNeeded hides anywhere in the error tree. */
export function isRoteMcpAuthNeeded(err: unknown): boolean {
    let node: unknown = err;
    for (let depth = 0; node != null && depth < 16; depth++) {
        const e = node as { name?: unknown; errors?: unknown; cause?: unknown };
        if (e.name === "RoteMcpAuthNeeded") return true;
        if (Array.isArray(e.errors) && e.errors.some((sub) => isRoteMcpAuthNeeded(sub))) {
            return true;
        }
        node = e.cause;
    }
    return false;
}

// Concrete JSON types — same rationale as the direct helper: step.do
// constrains callback returns to Rpc.Serializable, which `unknown`
// members do not satisfy.
export type JsonValue = string | number | boolean | null | JsonValue[] | JsonObject;
export type JsonObject = { [key: string]: JsonValue };

/** The platform proxy's auth-failure contract: errors arrive as plain
 * Error (the RPC boundary strips class identity), so the signal rides
 * in the message prefix. */
const AUTH_PREFIX = "ROTE_MCP_AUTH_NEEDED:";

function translate(err: unknown, server: string): unknown {
    const msg = err instanceof Error ? err.message : String(err);
    if (msg.startsWith(AUTH_PREFIX)) {
        let reason = msg.slice(AUTH_PREFIX.length).trim();
        // The proxy's detail leads with the server name (its parse token);
        // RoteMcpAuthNeeded's own message already names the server, so drop
        // the leading token rather than saying it twice.
        if (reason === server) {
            reason = "";
        } else if (reason.startsWith(`${server} `)) {
            reason = reason.slice(server.length + 1).trim();
        }
        return new RoteMcpAuthNeeded(
            server,
            reason || "the platform reports the connection needs re-authorization",
        );
    }
    return err;
}

function requireBinding(env: RoteMcpEnv): void {
    if (!env.ROTE_MCP) {
        throw new Error(
            "the ROTE_MCP service binding is not configured — this build " +
                "delegates MCP calls to the hosting platform, which must " +
                "provision the binding",
        );
    }
}

function requireVar(env: RoteMcpEnv, name: string): string {
    const value = env[name];
    if (typeof value !== "string" || value.length === 0) {
        throw new Error(
            `the ${name} variable is not set — binding-mode MCP runs only on ` +
                "the platform, whose dispatcher injects it into the isolate",
        );
    }
    return value;
}

/** The signed caller identity sent on every proxy call. */
function buildAuth(env: RoteMcpEnv): RoteMcpAuth {
    // ROTE_MCP_SIG first: its absence is the clearest sign this build is
    // running off-platform, and the config error should say so.
    const sig = requireVar(env, "ROTE_MCP_SIG");
    const auth: RoteMcpAuth = {
        tenant_id: requireVar(env, "ROTE_TENANT_ID"),
        pipeline: requireVar(env, "ROTE_PIPELINE"),
        sig,
    };
    const runId = env.ROTE_RUN_ID;
    if (typeof runId === "string" && runId.length > 0) {
        auth.run_id = runId;
    }
    return auth;
}

/**
 * Call one tool on an MCP server through the platform proxy.
 *
 * `is_error` results throw exactly the way the direct helper treats
 * `result.isError`, and string content is JSON-parsed the way the
 * direct helper parses text blocks — so a workflow step behaves the
 * same under either client mode.
 */
export async function callMcpTool(
    env: RoteMcpEnv,
    server: string,
    _pipelineUrl: string | null,
    tool: string,
    args: Record<string, unknown>,
): Promise<JsonObject> {
    requireBinding(env);
    const auth = buildAuth(env);
    let result: RoteMcpCallResult;
    try {
        result = await env.ROTE_MCP.call(auth, server, tool, args);
    } catch (err) {
        throw translate(err, server);
    }
    if (result.is_error) {
        throw new Error(
            `MCP tool '${tool}' on '${server}' returned an error: ` +
                JSON.stringify(result.content),
        );
    }
    const content = result.content;
    if (typeof content === "string") {
        return (content ? JSON.parse(content) : {}) as JsonObject;
    }
    return (content ?? {}) as JsonObject;
}

/**
 * Every tool `server` advertises, fetched through the platform proxy.
 *
 * The proxy's snake_case wire shape (`input_schema`, nullable
 * description) is normalized into McpToolSpec so bindAgentTools and the
 * agent loops see the same shape under either client mode.
 */
export async function listMcpTools(
    env: RoteMcpEnv,
    server: string,
    _pipelineUrl: string | null,
): Promise<McpToolSpec[]> {
    requireBinding(env);
    const auth = buildAuth(env);
    let tools: RoteMcpToolInfo[];
    try {
        tools = await env.ROTE_MCP.listTools(auth, server);
    } catch (err) {
        throw translate(err, server);
    }
    return tools.map((tool) => ({
        name: tool.name,
        description: tool.description ?? "",
        inputSchema: (tool.input_schema ?? {
            type: "object",
            properties: {},
        }) as Record<string, unknown>,
    }));
}

/** A tool bound to a callable — structurally a BoundTool for the agent loop. */
export interface BoundMcpTool {
    name: string;
    description: string;
    parameters: Record<string, unknown>;
    run: (args: Record<string, unknown>) => Promise<JsonObject>;
}

/**
 * Candidate servers for tool discovery — same contract as the direct
 * helper: the pipeline's declared servers, with ROTE_MCP_SERVERS
 * replacing the list wholesale when an operator overrides it.
 */
function agentServerNames(env: RoteMcpEnv, declared: string[]): string[] {
    const override = env.ROTE_MCP_SERVERS;
    if (typeof override === "string" && override.length > 0) {
        return override
            .split(",")
            .map((name) => name.trim())
            .filter(Boolean);
    }
    return [...new Set(declared)].sort();
}

/**
 * Bind the tools an agent_loop declared to callables the loop can invoke.
 *
 * Same contract as the direct helper: discovery is best-effort per server
 * (a server that is down may not be one this loop needed), a declared
 * tool no reachable server provides is fatal, and an auth failure that
 * plausibly explains the gap is rethrown so the Workflow parks on it.
 */
export async function bindAgentTools(
    env: RoteMcpEnv,
    allowed: string[],
    declaredServers: string[],
    serverUrls: Record<string, string | null> = {},
    toolServers: Record<string, string> = {},
): Promise<BoundMcpTool[]> {
    const wanted = new Set(allowed);
    const bound = new Map<string, BoundMcpTool>();
    const failures: unknown[] = [];

    for (const server of agentServerNames(env, declaredServers)) {
        const url = serverUrls[server] ?? null;
        let specs: McpToolSpec[];
        try {
            specs = await listMcpTools(env, server, url);
        } catch (err) {
            failures.push(err);
            continue;
        }
        for (const spec of specs) {
            if (!wanted.has(spec.name) || bound.has(spec.name)) continue;
            // Resolved tools bind only from their own server — see the
            // direct helper's twin for why first-wins is not good enough.
            const pinned = toolServers[spec.name];
            if (pinned !== undefined && pinned !== server) continue;
            bound.set(spec.name, {
                name: spec.name,
                description: spec.description,
                parameters: spec.inputSchema,
                run: (args) => callMcpTool(env, server, url, spec.name, args),
            });
        }
    }

    const missing = allowed.filter((name) => !bound.has(name));
    if (missing.length > 0) {
        const authFailure = failures.find((err) => isRoteMcpAuthNeeded(err));
        if (authFailure) throw authFailure;
        const detail = failures
            .map((err) => (err instanceof Error ? err.message : String(err)))
            .join("; ");
        throw new Error(
            `agent tools not provided by any reachable MCP server: ${missing.join(", ")}` +
                (detail ? ` (unreachable: ${detail})` : ""),
        );
    }
    return [...bound.values()];
}
"""


# ───────── signatures/_roteInference.ts (agent loops) ─────────

#: The default Workers AI model for an agent loop. Tool-capable and the
#: same family the workers-ai *judge* lane already defaults to, so an
#: operator who picks the Cloudflare lane gets one model story.
WORKERS_AI_AGENT_MODEL = "@cf/meta/llama-3.3-70b-instruct-fp8-fast"

#: The Anthropic TypeScript SDK version the agent loop needs.
#: ``beta.messages.toolRunner`` plus the ``BetaRunnableTool`` shape used
#: below; older pins (0.91 / 0.110) predate it.
ANTHROPIC_SDK_NPM_VERSION = "^0.115.0"

#: Cloudflare's embedded function-calling toolkit — the workers-ai lane.
AI_UTILS_NPM_VERSION = "^1.0.1"

#: Source of ``src/signatures/_roteInference.ts`` — the TypeScript twin
#: of :mod:`rote.inference._runtime_helper`. Emitted verbatim, same
#: contract as ``_roteMcp.ts``: never hand-edit the emitted copy, fix
#: this constant and re-emit.
ROTE_INFERENCE_HELPER_TS = """/**
 * Agent-loop runtime (generated by rote — do not edit; the source of
 * truth is rote.adapters._ts_common.ROTE_INFERENCE_HELPER_TS).
 *
 * The TypeScript twin of rote.inference._runtime_helper: it decides who
 * pays for an agent_loop's inference, then runs the bounded loop.
 * Three lanes, in the order they are preferred:
 *
 *   api         the operator's own ANTHROPIC_API_KEY. ROTE_BASE_URL_<ID>
 *               points it at an AI Gateway or any compatible endpoint.
 *   workers-ai  the operator's own Cloudflare account, through the AI
 *               binding — no external key and no external bill. Only
 *               the Cloudflare runtime supplies this lane.
 *   rote-cloud  a rote tenant token, billed to the rote account.
 *
 * ROTE_INFERENCE pins one lane explicitly and fails loudly if that lane
 * is unavailable, so "use my own infrastructure" is a decision the
 * operator can actually enforce rather than a preference the resolver
 * may quietly override.
 *
 * There is deliberately no claude-cli lane. workerd cannot spawn a
 * subprocess, and a lane that worked on two of the three TypeScript
 * runtimes would be a portability trap — the subscription lane lives on
 * the Python runtimes, where it always works.
 *
 * This module never imports MCP. Tools arrive already bound: the call
 * site resolves them through whichever MCP helper its own runtime
 * emits, so an auth failure inside a tool is an ordinary throw that
 * propagates out of the loop and parks the workflow exactly as it does
 * for a plain MCP-backed step.
 */

import Anthropic from "@anthropic-ai/sdk";

/** A tool the agent may call, in the one currency all lanes accept. */
export interface BoundTool {
    name: string;
    description: string;
    /** JSON Schema for the arguments — from the MCP server or the IR. */
    parameters: Record<string, unknown>;
    run: (args: Record<string, unknown>) => Promise<unknown>;
}

/**
 * The Cloudflare lane, injected by the call site.
 *
 * Passing the binding AND the runner keeps this module free of any
 * Cloudflare import, so the Node runtimes can typecheck and bundle it
 * without @cloudflare/ai-utils on their dependency list.
 */
export interface WorkersAiLane {
    binding: unknown;
    model: string;
    runWithTools: (
        ai: unknown,
        model: string,
        input: { messages: Array<{ role: string; content: string }>; tools: unknown[] },
        config?: { maxRecursiveToolRuns?: number; strictValidation?: boolean },
    ) => Promise<unknown>;
}

export type AgentEnv = Record<string, string | undefined>;

export type Provider = "api" | "workers-ai" | "rote-cloud";

const PROVIDERS: Provider[] = ["api", "workers-ai", "rote-cloud"];

export interface AgentResult {
    result: string;
    provider: Provider;
    /**
     * Turns actually run, or null when the lane does not report it.
     * Workers AI's runWithTools returns only the final text, so null
     * there is the honest answer — better than echoing back the cap and
     * making every loop look like it saturated.
     */
    iterations: number | null;
}

const SYSTEM_PROMPT =
    "You are a step inside a compiled pipeline. Use the tools you have been " +
    "given to complete the task, then state the final answer plainly. Be " +
    "concise and do not ask questions — nobody is available to answer them.";

/** Why a lane is unusable, or null when it is ready. */
function laneBlocker(
    provider: Provider,
    env: AgentEnv,
    workersAi: WorkersAiLane | null,
): string | null {
    if (provider === "api") {
        return env.ANTHROPIC_API_KEY ? null : "ANTHROPIC_API_KEY is not set";
    }
    if (provider === "workers-ai") {
        return workersAi
            ? null
            : "this runtime has no Workers AI binding (Cloudflare only)";
    }
    return env.ROTE_CLOUD_TOKEN
        ? null
        : "ROTE_CLOUD_TOKEN is not set — run `rote login`";
}

function selectProvider(
    nodeId: string,
    env: AgentEnv,
    workersAi: WorkersAiLane | null,
): Provider {
    const pinned = env.ROTE_INFERENCE as Provider | undefined;
    if (pinned) {
        if (!PROVIDERS.includes(pinned)) {
            throw new Error(
                `ROTE_INFERENCE=${pinned} is not a lane this runtime has; ` +
                    `expected one of: ${PROVIDERS.join(", ")}`,
            );
        }
        const blocker = laneBlocker(pinned, env, workersAi);
        if (blocker) {
            throw new Error(
                `ROTE_INFERENCE pins the '${pinned}' lane for ${nodeId}, but ${blocker}`,
            );
        }
        return pinned;
    }
    for (const provider of PROVIDERS) {
        if (!laneBlocker(provider, env, workersAi)) return provider;
    }
    throw new Error(
        `no inference lane is available for agent_loop ${nodeId}: set ` +
            `ANTHROPIC_API_KEY, bind Workers AI, or run \\`rote login\\` for ` +
            `the rote cloud lane`,
    );
}

/** The task the agent is being asked to do, as one prompt. */
function taskPrompt(
    description: string,
    task: unknown,
    termination: string | null,
): string {
    const parts = [description.trim(), "", JSON.stringify(task, null, 2)];
    if (termination) parts.push("", `Stop when: ${termination}`);
    return parts.join("\\n");
}

/** Endpoint + credential for the SDK lanes. */
function sdkTarget(
    provider: Provider,
    env: AgentEnv,
    baseUrl: string | null,
): { baseURL: string | undefined; apiKey: string; authToken: string | null } {
    if (provider === "rote-cloud") {
        const root = (env.ROTE_CLOUD_URL ?? "https://app.roteskills.com").replace(
            /\\/+$/,
            "",
        );
        return {
            baseURL: `${root}/v1/inference/anthropic`,
            // The tenant token is the credential; an ambient ANTHROPIC_API_KEY
            // must not ride along to a third-party endpoint.
            apiKey: "",
            authToken: env.ROTE_CLOUD_TOKEN ?? "",
        };
    }
    return {
        baseURL: baseUrl ?? undefined,
        apiKey: env.ANTHROPIC_API_KEY ?? "",
        authToken: null,
    };
}

async function runViaAnthropic(
    nodeId: string,
    provider: Provider,
    model: string,
    prompt: string,
    tools: BoundTool[],
    maxIterations: number,
    env: AgentEnv,
    baseUrl: string | null,
): Promise<AgentResult> {
    const target = sdkTarget(provider, env, baseUrl);
    const client = new Anthropic({
        baseURL: target.baseURL,
        apiKey: target.apiKey,
        ...(target.authToken ? { authToken: target.authToken } : {}),
    });

    // A runnable tool is just the wire tool plus run/parse — raw JSON
    // Schema, no Zod conversion, so an MCP server's advertised schema
    // reaches the model exactly as the server wrote it.
    const runnable = tools.map((tool) => ({
        name: tool.name,
        description: tool.description,
        input_schema: tool.parameters as { type: "object"; [k: string]: unknown },
        run: async (args: Record<string, unknown>) => {
            const result = await tool.run(args);
            return typeof result === "string" ? result : JSON.stringify(result);
        },
        parse: (content: unknown) => content as Record<string, unknown>,
    }));

    const runner = client.beta.messages.toolRunner({
        model,
        max_tokens: 4096,
        system: SYSTEM_PROMPT,
        tools: runnable,
        max_iterations: maxIterations,
        messages: [{ role: "user", content: prompt }],
    });
    // runUntilDone(), never done(): done() waits for the async iterator to
    // finish but does not start it, so a caller that never iterates hangs
    // forever. runUntilDone() consumes the iterator itself.
    const final = await runner.runUntilDone();

    const text = final.content
        .filter((block) => block.type === "text")
        .map((block) => (block as { text: string }).text)
        .join("\\n");
    if (!text) {
        throw new Error(
            `agent_loop ${nodeId} finished without a text answer (stop reason: ` +
                `${final.stop_reason ?? "unknown"})`,
        );
    }
    // Iterations ACTUALLY run, not the cap. The runner accumulates the
    // conversation in its params, so one assistant message is one turn —
    // reporting the ceiling here would make every loop look saturated.
    const iterations = runner.params.messages.filter((m) => m.role === "assistant").length;
    return { result: text, provider, iterations };
}

async function runViaWorkersAi(
    nodeId: string,
    lane: WorkersAiLane,
    prompt: string,
    tools: BoundTool[],
    maxIterations: number,
): Promise<AgentResult> {
    const output = (await lane.runWithTools(
        lane.binding,
        lane.model,
        {
            messages: [
                { role: "system", content: SYSTEM_PROMPT },
                { role: "user", content: prompt },
            ],
            tools: tools.map((tool) => ({
                name: tool.name,
                description: tool.description,
                parameters: tool.parameters,
                function: async (args: Record<string, unknown>) => {
                    const result = await tool.run(args);
                    return typeof result === "string" ? result : JSON.stringify(result);
                },
            })),
        },
        { maxRecursiveToolRuns: maxIterations, strictValidation: true },
    )) as { response?: string };

    const text = output?.response;
    if (!text) {
        throw new Error(`agent_loop ${nodeId} returned no response from Workers AI`);
    }
    return { result: text, provider: "workers-ai", iterations: null };
}

/**
 * Run one bounded, tool-restricted agent loop and return its result.
 *
 * This is what an agent_loop node executes. It is a real loop on every
 * lane, never a stub — a pipeline that hands its one genuinely agentic
 * step back to the user has not compiled that step.
 */
export async function runAgentLoop(opts: {
    nodeId: string;
    description: string;
    task: unknown;
    model: string;
    tools: BoundTool[];
    maxIterations: number;
    termination?: string | null;
    baseUrl?: string | null;
    env: AgentEnv;
    workersAi?: WorkersAiLane | null;
}): Promise<AgentResult> {
    const workersAi = opts.workersAi ?? null;
    const provider = selectProvider(opts.nodeId, opts.env, workersAi);
    const prompt = taskPrompt(opts.description, opts.task, opts.termination ?? null);

    if (provider === "workers-ai") {
        // Non-null by construction: selectProvider only returns this lane
        // when laneBlocker found the binding present.
        return runViaWorkersAi(
            opts.nodeId,
            workersAi as WorkersAiLane,
            prompt,
            opts.tools,
            opts.maxIterations,
        );
    }
    return runViaAnthropic(
        opts.nodeId,
        provider,
        opts.model,
        prompt,
        opts.tools,
        opts.maxIterations,
        opts.env,
        opts.baseUrl ?? null,
    );
}
"""


# ───────── extracted/<id>.ts emission for agent_loop nodes ─────────


def agent_tool_servers(pipeline: Pipeline) -> dict[str, str | None]:
    """Candidate MCP servers for agent_loop tool discovery: name → recorded URL.

    ``Node.tools`` names tools without a server (one loop may reach
    several, and an :class:`MCPBinding` is a single server/tool pair), so
    the emitted loop is handed the servers this pipeline is already known
    to talk to and searches them for the declared names.

    Three sources, in descending order of authority:

    1. A loop's own ``tool_servers`` — the IR resolved this tool to this
       server (written by the compiler, or by ``rote compile`` from what
       the baseline actually called). No URL travels with it, so the
       endpoint still resolves from env/registry at run time.
    2. Any node's ``mcp:`` binding, which carries a URL. A loop's bare
       tools are searched here on the theory that a pipeline's loop
       usually reaches the servers the rest of the pipeline already does.
    3. At run time only: the local registry (Node runtimes) and
       ``ROTE_MCP_SERVERS`` (everywhere), which overrides the lot.

    (2) is a guess and (3) is an escape hatch; only (1) is knowledge, which
    is why it also feeds :attr:`Pipeline.required_mcp_servers` and the
    ``rote mcp login`` advisories.
    """
    servers: dict[str, str | None] = {}
    for node in pipeline.nodes:
        if node.mcp is not None and node.mcp.server not in servers:
            servers[node.mcp.server] = node.mcp.url
    for node in pipeline.nodes:
        for server in (node.tool_servers or {}).values():
            servers.setdefault(server, None)
    return dict(sorted(servers.items()))


#: Permissive argument schema for a loop_body sub-node with no declared
#: ``input_schema``. TypeScript cannot derive a schema from a function
#: signature the way Python's ``@beta_tool`` does, so the honest fallback
#: is "an object" — the sub-node still validates its own payload.
_OPEN_TOOL_SCHEMA: dict[str, Any] = {"type": "object", "additionalProperties": True}


def _local_tool_literal(
    sub: Node, call_expr: str, indent: str, *, ts_type: str = "BoundTool"
) -> list[str]:
    schema = sub.input_schema or _OPEN_TOOL_SCHEMA
    desc = sub.description.strip() or f"Run the {sub.id} step."
    return [
        f"{indent}{{",
        f"{indent}    name: {json.dumps(sub.id)},",
        f"{indent}    description: {json.dumps(desc)},",
        f"{indent}    parameters: {json.dumps(schema)},",
        f"{indent}    run: async (args) => {call_expr},",
        f"{indent}}},",
    ]


def emit_agent_loop_module(
    node: Node,
    pipeline: Pipeline,
    *,
    default_model: str,
    generated_by: str,
    workers: bool,
    sub_node_env_arg: Callable[[Node], str | None] | None = None,
    helpers: Sequence[str] = (),
    workers_ai_model: str | None = None,
) -> str:
    """Emit ``src/extracted/<id>.ts`` as a REAL bounded agent loop.

    This is what replaced the ``throw new Error("… implement me")`` stub.
    A pipeline that hands its one genuinely agentic step back to the user
    has not graduated that step, so the emitted module runs the loop: MCP
    tools bound through the runtime's own MCP helper, ``loop_body``
    sub-nodes bound as callables (already-emitted, already-working steps,
    so the agent drives them per iteration exactly as the IR describes),
    and the inference lane resolved at runtime by ``_roteInference.ts``.

    ``sub_node_env_arg`` lets each adapter supply its own calling
    convention for those sub-nodes — a judge takes an env object on the
    Node runtimes and ``env`` on Workers — because the adapter owns that
    convention and the module shape is all that is shared. ``helpers``
    carries any function those call expressions depend on (the Node
    runtimes' ``requireEnv``), since this module cannot import from the
    workflow file that normally defines them.

    ``workers_ai_model`` opts the module into the Cloudflare lane: the
    binding and its runner are passed IN, so the inference helper itself
    stays free of any Cloudflare import and the Node runtimes can compile
    it without ``@cloudflare/ai-utils`` on their dependency list.
    """
    assert node.kind is NodeKind.AGENT_LOOP
    fn_name = _to_camel_case(node.id)
    model_var, base_var = override_env_vars(node.id)
    max_iterations = (
        node.termination.max_iterations
        if node.termination is not None
        else DEFAULT_AGENT_MAX_ITERATIONS
    )
    tools = list(node.tools or [])
    servers = agent_tool_servers(pipeline) if tools else {}
    pins = dict(sorted((node.tool_servers or {}).items()))
    unresolved = [tool for tool in tools if tool not in pins]
    sub_nodes = [pipeline.node_by_id(sub) for sub in (node.loop_body or [])]

    doc = [
        "/**",
        f" * Agent loop: {node.id}",
        " *",
        f" * {safe_block_comment_line(node.description, fallback=node.id)}",
        " *",
        f" * Auto-generated by {generated_by}. Bounded at {max_iterations} iteration(s)",
        " * and restricted to the tools below — the agent sees nothing else. Which",
        " * lane pays for the inference (your Anthropic key, your Cloudflare account,",
        " * or rote cloud) is resolved at runtime by signatures/_roteInference.ts;",
        f" * {model_var} / {base_var} override the model and endpoint.",
    ]
    if node.mandatory:
        doc.extend(
            [
                " *",
                " * MANDATORY: this node was marked mandatory in the source skill.",
                " * The workflow always calls it; do not make it conditional.",
            ]
        )
    doc.append(" */")

    inference_names = ["runAgentLoop", "type BoundTool"]
    if workers and workers_ai_model is not None:
        inference_names.append("type WorkersAiLane")
    imports = [f'import {{ {", ".join(inference_names)} }} from "../signatures/_roteInference";']
    if tools:
        mcp_types = ", type RoteMcpEnv" if workers else ""
        imports.append(f'import {{ bindAgentTools{mcp_types} }} from "./_roteMcp";')
    for sub in sub_nodes:
        sub_fn = _to_camel_case(sub.id)
        where = "../signatures" if sub.kind is NodeKind.LLM_JUDGE else "."
        imports.append(f'import {{ {sub_fn} }} from "{where}/{sub.id}";')

    consts: list[str] = []
    if tools:
        consts.extend(
            [
                "",
                "/** Tools the IR declared for this loop — the allowlist IS the boundary. */",
                f"const TOOLS: string[] = {json.dumps(tools)};",
                "",
                "/** Servers searched for those tools, with the endpoints the IR recorded. */",
                f"const SERVERS: string[] = {json.dumps(list(servers))};",
                f"const SERVER_URLS: Record<string, string | null> = {json.dumps(servers)};",
                "",
                "/** Tools the IR resolved to a specific server — these bind from that",
                " * server and no other, so two servers exporting one tool name cannot",
                " * silently swap endpoints. Tools absent here fall back to first-wins",
                " * across SERVERS. */",
                f"const TOOL_SERVERS: Record<string, string> = {json.dumps(pins)};",
            ]
        )
        if pins and unresolved:
            consts.extend(
                [
                    "",
                    "// Resolved from the IR: "
                    + ", ".join(f"{tool} -> {server}" for tool, server in pins.items()),
                    "// Still unresolved (searched across SERVERS): " + ", ".join(unresolved),
                ]
            )
        if not servers:
            # Say so in the emitted source rather than letting the loop
            # fail at runtime with "no server provides this tool" and no
            # hint about why nothing was searched.
            consts[-3:-3] = [
                "// NOTE: no MCP server could be resolved at emit time — this loop's",
                "// tools are bare names and no node in the pipeline carries an `mcp:`",
                "// binding to search. The Node runtimes fall back to every server in",
                "// the local rote registry; set ROTE_MCP_SERVERS to name them",
                "// explicitly (required on Cloudflare, which has no registry).",
            ]

    if workers:
        # `unknown` + casts, the same convention the Workers MCP modules
        # use: the workflow's Env is an interface, and an interface has no
        # implicit index signature to satisfy RoteMcpEnv or a string map.
        params = "    input: unknown,\n    env: unknown, // the workflow's Env; cast below"
        bind_call = (
            "await bindAgentTools(\n"
            "        env as RoteMcpEnv,\n"
            "        TOOLS,\n"
            "        SERVERS,\n"
            "        SERVER_URLS,\n"
            "        TOOL_SERVERS,\n"
            "    )"
        )
        env_source = "env as Record<string, string | undefined>"
        if workers_ai_model is not None:
            imports.append('import { runWithTools } from "@cloudflare/ai-utils";')
    else:
        params = "    input: unknown,"
        bind_call = "await bindAgentTools(TOOLS, SERVERS, SERVER_URLS, TOOL_SERVERS)"
        env_source = "process.env"

    local_tools: list[str] = []
    for sub in sub_nodes:
        sub_fn = _to_camel_case(sub.id)
        extra = sub_node_env_arg(sub) if sub_node_env_arg is not None else None
        call = f"{sub_fn}(args)" if extra is None else f"{sub_fn}(args, {extra})"
        local_tools.extend(_local_tool_literal(sub, call, "        "))

    # The return type is an anonymous object type, never an interface:
    # TypeScript grants implicit index signatures to the former only, and
    # the workflow's data-flow references cast node results to
    # Record<string, unknown> (see ref_to_ts_expr).
    body = [
        "",
        f"export async function {fn_name}(",
        params,
        "): Promise<{ result: string; provider: string; iterations: number | null }> {",
        f"    const vars = {env_source};",
    ]
    if tools:
        body.append(f"    const tools: BoundTool[] = {bind_call};")
    else:
        body.append("    const tools: BoundTool[] = [];")
    if workers_ai_model is not None:
        # Offered, never forced: the lane is only *selected* when the
        # operator pins ROTE_INFERENCE=workers-ai or has no other lane, so
        # binding `ai` in wrangler.jsonc costs nothing until it is used.
        body.extend(
            [
                "    const ai = (env as { AI?: unknown }).AI;",
                "    const workersAi: WorkersAiLane | null = ai",
                "        ? {",
                "              binding: ai,",
                "              model: vars.ROTE_WORKERS_AI_MODEL ?? "
                f"{json.dumps(workers_ai_model)},",
                '              runWithTools: runWithTools as WorkersAiLane["runWithTools"],',
                "          }",
                "        : null;",
            ]
        )
    if local_tools:
        body.append("    // loop_body sub-nodes: real pipeline steps, driven per iteration.")
        body.append("    tools.push(")
        body.extend(local_tools)
        body.append("    );")
    body.extend(
        [
            "    return runAgentLoop({",
            f"        nodeId: {json.dumps(node.id)},",
            f"        description: {json.dumps(node.description)},",
            "        task: input,",
            f"        model: vars.{model_var} ?? {json.dumps(default_model)},",
            "        tools,",
            f"        maxIterations: {max_iterations},",
        ]
    )
    if node.termination is not None:
        body.append(f"        termination: {json.dumps(node.termination.condition)},")
    body.extend(
        [
            f"        baseUrl: vars.{base_var} ?? null,",
            "        env: vars,",
            *(["        workersAi,"] if workers_ai_model is not None else []),
            "    });",
            "}",
            "",
        ]
    )
    helper_block = ["", *(h.rstrip("\n") for h in helpers)] if helpers else []
    return "\n".join(doc + [""] + imports + consts + helper_block + body)


def emit_node_agent_loop_module(
    node: Node, pipeline: Pipeline, *, default_model: str, generated_by: str
) -> str:
    """An agent_loop module for the Node-process runtimes (DBOS-TS, Inngest).

    Both call their steps identically — a judge sub-node takes the same
    explicit env object the workflow's own judge steps build, so
    ``requireEnv`` rides along (the agent module cannot import it from the
    workflow file). Shared rather than copied: two runtimes that emitted
    *different* agent loops would be two implementations of the loop.
    """
    sub_kinds = {pipeline.node_by_id(sub).kind for sub in (node.loop_body or [])}
    return emit_agent_loop_module(
        node,
        pipeline,
        default_model=default_model,
        generated_by=generated_by,
        workers=False,
        sub_node_env_arg=lambda sub: judge_env_arg(sub) if sub.kind is NodeKind.LLM_JUDGE else None,
        helpers=[REQUIRE_ENV_HELPER] if NodeKind.LLM_JUDGE in sub_kinds else [],
    )
