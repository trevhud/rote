"""Helpers shared by every TypeScript-emitting runtime adapter.

Extracted from ``rote.adapters.cloudflare`` once the DBOS TypeScript
adapter became the second TS consumer — same rule as ``_common``: a
helper moves here only after two adapters prove it's genuinely shared.

What lives here:

* the JSON-Schema-to-Zod converter (``json_schema_to_zod`` and its
  internals ``_resolve_refs`` / ``_convert_zod``),
* the typed LLM signature module emitters
  (``emit_signature_anthropic`` / ``emit_signature_openai``) plus the
  runtime prompt-interpolation helper they embed
  (``_INTERPOLATE_HELPER``).

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
from typing import Any

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
    temperature: float | None,
    generated_by: str,
) -> str:
    """Emit a signatures/<id>.ts module calling Anthropic with tool-use output.

    ``generated_by`` names the emitting adapter module in the header
    JSDoc (e.g. ``rote.adapters.cloudflare``) so regeneration
    instructions point at the right runtime.
    """
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
    temperature: float | None,
    generated_by: str,
) -> str:
    """Emit a signatures/<id>.ts module using OpenAI structured outputs."""
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
