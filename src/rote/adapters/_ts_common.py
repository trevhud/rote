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
from typing import Any

from rote.adapters._common import _to_camel_case, _to_pascal_case, safe_block_comment_line
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
    temp_line = f"        temperature: {temperature},\n" if temperature is not None else ""
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
        "    },",
        f"): Promise<{pascal}Output> {{",
        f"    const input = {pascal}Input.parse(rawInput);",
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
    temp_line = f"        temperature: {temperature},\n" if temperature is not None else ""
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
        "    },",
        f"): Promise<{pascal}Output> {{",
        f"    const input = {pascal}Input.parse(rawInput);",
        "    const client = new OpenAI({",
        "        apiKey: env.OPENAI_API_KEY,",
        f"        baseURL: env.{base_var}{base_default},",
        "    });",
        "",
        "    const response = await client.chat.completions.create({",
        f"        model: env.{model_var} ?? {quoted_model},",
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
    temp_line = f"        temperature: {temperature},\n" if temperature is not None else ""
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
        "        // Injected by the host at load time; harmless when unset.",
        "        ROTE_GATEWAY_ID?: string;",
        "        ROTE_TENANT_ID?: string;",
        "        ROTE_PIPELINE?: string;",
        "        ROTE_RUN_ID?: string;",
        "    },",
        f"): Promise<{pascal}Output> {{",
        f"    const input = {pascal}Input.parse(rawInput);",
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
        temp_line.rstrip("\n") if temp_line else "        // temperature: vendor default",
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
 */

import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import {
    StreamableHTTPClientTransport,
    StreamableHTTPError,
} from "@modelcontextprotocol/sdk/client/streamableHttp.js";

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
        throw new Error(
            `access token for MCP server '${server}' is expired and no refresh ` +
                `token is stored — re-authenticate with: rote mcp login ${server}`,
        );
    }
    if (!doc.token_endpoint) {
        throw new Error(
            `token store for '${server}' has no token_endpoint (written by ` +
                `rote mcp login) — re-authenticate with: rote mcp login ${server}`,
        );
    }
    const entry = registryServers()[server];
    const clientId = doc.client_info?.client_id ?? entry?.client_id;
    const clientSecret = doc.client_info?.client_secret ?? entry?.client_secret;
    if (!clientId) {
        throw new Error(
            `no client_id stored for '${server}' — re-authenticate with: rote mcp login ${server}`,
        );
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
        throw new Error(
            `refresh grant for '${server}' failed (${response.status}) — ` +
                `re-authenticate with: rote mcp login ${server}`,
        );
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
    const url = resolveUrl(server, pipelineUrl);

    const attempt = async (headers: Record<string, string> | null) => {
        const client = new Client({ name: "rote-pipeline", version: "1.0.0" });
        const transport = new StreamableHTTPClientTransport(new URL(url), {
            requestInit: headers ? { headers } : undefined,
        });
        try {
            await client.connect(transport);
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
        } finally {
            await client.close().catch(() => undefined);
        }
    };

    const headers = await freshHeaders(server);
    try {
        return await attempt(headers);
    } catch (err) {
        const unauthorized =
            err instanceof StreamableHTTPError
                ? err.code === 401
                : /\\b401\\b|[Uu]nauthorized/.test(String(err));
        if (unauthorized && readTokenDoc(server)?.tokens?.refresh_token) {
            return attempt(await freshHeaders(server, true));
        }
        throw err;
    }
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
