"""Emission machinery shared by the Python-emitting adapters.

The DBOS adapter and the raw Python adapter emit the same logical
artifacts around their runtime-specific ``main.py``:

* ``signatures/<id>.py`` — Pydantic models generated from an llm_judge
  node's ``signature_spec`` JSON Schemas, plus a typed judge class that
  calls the vendor SDK with structured output.
* ``extracted/<module>.py`` — ``NotImplementedError`` stubs for
  pure_function / external_call / agent_loop nodes.

Both were born in ``rote.adapters.dbos`` and factored here when the raw
Python adapter needed them verbatim. Only the *identity strings* differ
per adapter (who generated the file, how to regenerate it, and one line
of context prose), so those are parameters rather than copies. Anything
that encodes a runtime's semantics (DBOS step decorators, retry loops)
stays in the adapter that owns it.
"""

from __future__ import annotations

import json
import keyword
import re
import textwrap
from typing import Any

from rote.adapters._common import _to_pascal_case, safe_docstring_line
from rote.ir import LLMSignature, Node, NodeKind, Pipeline, parse_input_ref

# ───────── impl / signature path references ─────────


def _impl_path_parts(impl: str) -> tuple[str, str]:
    """Parse 'extracted/foo.py:bar' into ('foo', 'bar')."""
    if ":" not in impl:
        raise ValueError(f"impl path missing ':function_name': {impl!r}")
    file_part, func = impl.split(":", 1)
    # Strip leading directories and .py suffix
    module_name = file_part.rsplit("/", 1)[-1].removesuffix(".py")
    return module_name, func


def _signature_path_parts(signature: str) -> tuple[str, str]:
    """Parse 'signatures/foo.py:Foo' into ('foo', 'Foo')."""
    return _impl_path_parts(signature)


# ───────── Data-flow references → Python source ─────────


def _ref_to_python_expr(ref: str) -> str:
    """Render an ``inputs:`` source reference as a Python expression.

    Every Python-emitting adapter binds the pipeline input to
    ``pipeline_input`` and each node's result to ``<node_id>_result``,
    so references map directly:

    | Reference                  | Expression                      |
    |----------------------------|---------------------------------|
    | ``pipeline.input``         | ``pipeline_input``              |
    | ``pipeline.input.f``       | ``pipeline_input["f"]``         |
    | ``foo.output``             | ``foo_result``                  |
    | ``foo.output.f``           | ``foo_result["f"]``             |
    """
    parsed = parse_input_ref(ref)
    base = "pipeline_input" if parsed.node_id is None else f"{parsed.node_id}_result"
    if parsed.field is None:
        return base
    return f'{base}["{parsed.field}"]'


def _payload_literal(node: Node, indent: str) -> str:
    """Render the step/activity payload dict for a node's data-flow bindings.

    Nodes without ``inputs`` keep the empty payload (back-compat).
    ``indent`` is the indentation of the line the literal starts on;
    continuation lines are indented one level deeper.
    """
    if not node.inputs:
        return "{}"
    inner = indent + "    "
    lines = ["{"]
    for param, ref in node.inputs.items():
        lines.append(f'{inner}"{param}": {_ref_to_python_expr(ref)},')
    lines.append(indent + "}")
    return "\n".join(lines)


# ───────── JSON Schema → Pydantic source ─────────


def _pascal_ident(s: str) -> str:
    """Sanitize an arbitrary schema title into a PascalCase identifier.

    Preserves interior capitalization ('EmploymentEntry' stays
    'EmploymentEntry'; ``str.capitalize`` would mangle it to
    'Employmententry').
    """
    cleaned = re.sub(r"[^0-9a-zA-Z_]", "_", s)
    parts = [p for p in cleaned.split("_") if p]
    if not parts:
        return "Model"
    ident = "".join(p[0].upper() + p[1:] for p in parts)
    if ident[0].isdigit():
        ident = f"Model{ident}"
    return ident


def _py_literal(value: Any) -> str:
    """Render a JSON-compatible value as a Python literal expression.

    Strings use double quotes (matching the repo's ruff-format style);
    containers recurse so nested strings get the same treatment.
    """
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, bool) or value is None:
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_py_literal(v) for v in value) + "]"
    if isinstance(value, dict):
        items = ", ".join(f"{_py_literal(k)}: {_py_literal(v)}" for k, v in value.items())
        return "{" + items + "}"
    return repr(value)


def _literal_alias(name: str, values: list[Any]) -> str:
    """Render ``Name = Literal[...]``, wrapping when the one-liner is long."""
    rendered = [_py_literal(v) for v in values]
    one_line = f"{name} = Literal[{', '.join(rendered)}]"
    if len(one_line) <= 96:
        return one_line
    body = "\n".join(f"    {r}," for r in rendered)
    return f"{name} = Literal[\n{body}\n]"


class _SchemaToPydantic:
    """Convert JSON Schemas (with ``$defs``/``$ref``) to Pydantic source.

    The Python analog of the Cloudflare adapter's ``json_schema_to_zod``.
    Named ``$defs`` become named classes / ``Literal`` aliases; inline
    object schemas get synthesized names. Blocks are accumulated in
    dependency order (a class is emitted only after everything it
    references), so the generated module imports cleanly top-to-bottom.
    """

    def __init__(self) -> None:
        self._defs: dict[str, dict[str, Any]] = {}
        self._blocks: list[str] = []
        self._emitted: dict[str, str] = {}  # $def name -> python name
        self._in_progress: set[str] = set()
        self._used_names: set[str] = set()
        self.uses_literal = False
        self.uses_config_dict = False

    # ── public API ──

    def add_defs(self, defs: dict[str, Any]) -> None:
        for name, schema in defs.items():
            if name in self._defs and self._defs[name] != schema:
                raise ValueError(
                    f"signature emission: $defs entry {name!r} appears in both input "
                    f"and output schemas with different definitions"
                )
            self._defs[name] = schema

    def emit_root(self, class_name: str, schema: dict[str, Any]) -> None:
        if schema.get("type") != "object":
            raise ValueError(
                f"signature emission: root signature schema for {class_name} must be "
                f"an object schema, got type={schema.get('type')!r}"
            )
        self._emit_object_class(class_name, schema)

    @property
    def blocks(self) -> list[str]:
        return list(self._blocks)

    # ── internals ──

    def _unique_name(self, base: str) -> str:
        name = base
        n = 2
        while name in self._used_names:
            name = f"{base}{n}"
            n += 1
        self._used_names.add(name)
        return name

    def _ensure_def(self, ref: str) -> str:
        if not ref.startswith("#/$defs/"):
            raise ValueError(f"signature emission: unsupported $ref form: {ref!r}")
        def_name = ref[len("#/$defs/") :]
        if def_name in self._emitted:
            return self._emitted[def_name]
        if def_name in self._in_progress:
            raise ValueError(
                f"signature emission: cyclic $ref through {def_name!r} is not supported"
            )
        if def_name not in self._defs:
            raise ValueError(f"signature emission: unknown $ref target: {def_name!r}")

        schema = self._defs[def_name]
        self._in_progress.add(def_name)
        try:
            py_name = self._unique_name(_pascal_ident(schema.get("title", def_name)))
            self._emitted[def_name] = py_name
            if "enum" in schema:
                self.uses_literal = True
                self._blocks.append(_literal_alias(py_name, schema["enum"]))
            elif schema.get("type") == "object":
                self._emit_object_class(py_name, schema)
            else:
                self._blocks.append(f"{py_name} = {self._type_expr(schema, py_name)}")
        finally:
            self._in_progress.discard(def_name)
        return py_name

    def _type_expr(self, schema: dict[str, Any], hint: str) -> str:
        if "$ref" in schema:
            return self._ensure_def(schema["$ref"])

        for union_key in ("anyOf", "oneOf"):
            if union_key in schema:
                variants = schema[union_key]
                non_null = [
                    v for v in variants if not (isinstance(v, dict) and v.get("type") == "null")
                ]
                has_null = len(non_null) < len(variants)
                parts = [self._type_expr(v, f"{hint}Variant{i}") for i, v in enumerate(non_null)]
                expr = " | ".join(parts) if parts else "None"
                if has_null and parts:
                    expr += " | None"
                return expr

        if "enum" in schema:
            self.uses_literal = True
            values = ", ".join(_py_literal(v) for v in schema["enum"])
            return f"Literal[{values}]"

        schema_type = schema.get("type")
        if schema_type == "object":
            if not schema.get("properties"):
                return "dict[str, Any]"
            name = self._unique_name(_pascal_ident(schema.get("title", hint)))
            self._emit_object_class(name, schema)
            return name
        if schema_type == "array":
            items = schema.get("items")
            if not isinstance(items, dict):
                return "list[Any]"
            return f"list[{self._type_expr(items, f'{hint}Item')}]"
        if schema_type == "string":
            return "str"
        if schema_type == "integer":
            return "int"
        if schema_type == "number":
            return "float"
        if schema_type == "boolean":
            return "bool"
        if schema_type == "null":
            return "None"
        return "Any"

    def _emit_object_class(self, class_name: str, schema: dict[str, Any]) -> None:
        required = set(schema.get("required", []))
        properties: dict[str, Any] = schema.get("properties", {})

        field_lines: list[str] = []
        for field_name, field_schema in properties.items():
            if not field_name.isidentifier() or keyword.iskeyword(field_name):
                raise ValueError(
                    f"signature emission: schema property {field_name!r} in "
                    f"{class_name} is not a valid Python identifier"
                )
            expr = self._type_expr(field_schema, f"{class_name}{_pascal_ident(field_name)}")
            if field_name in required and "default" not in field_schema:
                field_lines.append(f"    {field_name}: {expr}")
            elif "default" in field_schema:
                field_lines.append(
                    f"    {field_name}: {expr} = {_py_literal(field_schema['default'])}"
                )
            else:
                if not expr.endswith("| None") and expr != "None":
                    expr += " | None"
                field_lines.append(f"    {field_name}: {expr} = None")

        lines = [f"class {class_name}(BaseModel):"]
        description = schema.get("description")
        if description:
            first = safe_docstring_line(str(description))
            lines.append(f'    """{first}"""')
            lines.append("")
        if schema.get("additionalProperties") is False:
            self.uses_config_dict = True
            lines.append('    model_config = ConfigDict(extra="forbid")')
            lines.append("")
        if field_lines:
            lines.extend(field_lines)
        elif len(lines) == 1:
            lines.append("    pass")
        self._blocks.append("\n".join(lines))


# ───────── Long-string chunking (ruff line-length safety) ─────────


def _str_chunks(text: str, indent: str) -> list[str]:
    """Split a string into escaped literal chunks that fit the line budget.

    Emitted snapshots are committed to the repo and linted with ruff
    (line-length 100), so single-line ``json.dumps`` literals for prompts
    and schemas are not an option. The budget is measured against the
    *escaped* form at the given indent depth so no emitted line exceeds
    the limit.
    """
    width = max(40, 98 - len(indent) - 8)
    chunks: list[str] = []
    current = ""
    for ch in text:
        current += ch
        if len(json.dumps(current)) >= width:
            chunks.append(current)
            current = ""
    if current or not chunks:
        chunks.append(current)
    return chunks


def _chunked_str_literal(text: str, indent: str = "    ") -> str:
    """Render a long string as a parenthesized adjacent-literal expression."""
    chunks = _str_chunks(text, indent)
    if len(chunks) == 1:
        return json.dumps(chunks[0])
    body = "\n".join(f"{indent}{json.dumps(c)}" for c in chunks)
    outer = indent[:-4] if len(indent) >= 4 else ""
    return f"(\n{body}\n{outer})"


def _chunked_call_arg(text: str, indent: str) -> str:
    """Render a long string as adjacent literals for direct use as a call arg.

    Unlike :func:`_chunked_str_literal` this emits no wrapping parentheses
    (the call's own parentheses group the literals), which keeps ruff's
    UP034 (extraneous parentheses) quiet.
    """
    chunks = _str_chunks(text, indent)
    return "\n".join(f"{indent}{json.dumps(c)}" for c in chunks)


# ───────── signatures/<id>.py emission ─────────


def emit_signature_module(
    node: Node,
    *,
    anthropic_default_model: str,
    openai_default_model: str,
    generated_by: str,
    regen_command: str,
    context_note: str,
) -> str:
    """Emit signatures/<node_id>.py for an llm_judge node with a spec.

    The emitted module:

    1. Declares Pydantic models for input + output, generated from the
       IR's ``signature_spec`` JSON Schemas.
    2. Exports a ``<Pascal>`` judge class whose synchronous ``forward``
       interpolates the prompt, calls the vendor SDK with structured
       output (Anthropic tool-use / OpenAI json_schema), validates the
       response with Pydantic, and returns the typed output.

    The prompt template is interpolated with a minimal ``{{ key }}``
    substitution at runtime — same contract as the Cloudflare adapter's
    TS emission, so all runtimes share prompt semantics.

    ``generated_by`` / ``regen_command`` / ``context_note`` are the
    calling adapter's identity strings for the module docstring.
    """
    if node.signature_spec is None:
        raise ValueError(
            f"signature emission: emit_signature_module requires signature_spec on node {node.id!r}"
        )
    spec = node.signature_spec
    pascal = _to_pascal_case(node.id)

    converter = _SchemaToPydantic()
    converter.add_defs(spec.input_schema.get("$defs", {}))
    converter.add_defs(spec.output_schema.get("$defs", {}))
    converter.emit_root(f"{pascal}Input", spec.input_schema)
    converter.emit_root(f"{pascal}Output", spec.output_schema)

    model = spec.model or _default_model_for(spec, anthropic_default_model, openai_default_model)
    desc_first = safe_docstring_line(node.description, fallback=node.id)

    typing_names = "Any, Literal" if converter.uses_literal else "Any"
    pydantic_names = "BaseModel, ConfigDict" if converter.uses_config_dict else "BaseModel"
    schema_json = json.dumps(spec.output_schema, separators=(",", ":"))

    header = (
        f'"""Typed LLM signature: {node.id}\n'
        f"\n"
        f"{desc_first}\n"
        f"\n"
        f"Auto-generated by {generated_by} from the IR's ``signature_spec``.\n"
        f"{context_note}\n"
        f"\n"
        f"DO NOT EDIT BY HAND. Re-run ``{regen_command}`` to regenerate.\n"
        f'"""\n'
        f"\n"
        f"from __future__ import annotations\n"
        f"\n"
        f"import json\n"
        f"import os\n"
        f"import re\n"
        f"from typing import {typing_names}\n"
        f"\n"
        f"from pydantic import {pydantic_names}\n"
    )

    models_block = "\n\n\n".join(converter.blocks)

    prompt_block = f"PROMPT = {_chunked_str_literal(spec.prompt)}"
    description_block = (
        f"TOOL_DESCRIPTION = {_chunked_str_literal(desc_first)}\n\n"
        if spec.client == "anthropic"
        else ""
    )
    schema_block = (
        "OUTPUT_JSON_SCHEMA: dict[str, Any] = json.loads(\n"
        f"{_chunked_call_arg(schema_json, indent='    ')}\n"
        ")"
    )

    # Operator knobs: the model and endpoint bake in defaults from the IR
    # but stay overridable per-node at runtime, so switching models (or
    # pointing at an OpenAI-compatible server / gateway) never requires a
    # re-emit.
    env_suffix = node.id.upper()
    base_url_default = f", {json.dumps(spec.base_url)}" if spec.base_url else ""
    overrides_block = (
        "# Operator overrides: change the model or point at a different\n"
        "# endpoint (proxy, gateway, OpenAI-compatible server) without\n"
        "# re-emitting. Unset means the default below / the vendor's endpoint.\n"
        f'MODEL = os.environ.get("ROTE_MODEL_{env_suffix}", {json.dumps(model)})\n'
        f'BASE_URL = os.environ.get("ROTE_BASE_URL_{env_suffix}"{base_url_default})'
    )

    interpolate_block = textwrap.dedent(
        '''\
        def _interpolate(template: str, variables: dict[str, Any]) -> str:
            """Resolve ``{{ dotted.path }}`` placeholders against the input dict.

            An unresolvable placeholder raises instead of interpolating "" —
            a hole in a judge prompt produces confident garbage that is far
            harder to debug than a KeyError naming the missing variable.
            """

            def _resolve(match: re.Match[str]) -> str:
                path = match.group(1)
                value: Any = variables
                for part in path.split("."):
                    if not isinstance(value, dict) or part not in value:
                        raise KeyError(
                            f"prompt template references {{{{ {path} }}}} but the "
                            f"input has no such field; available top-level keys: "
                            f"{sorted(variables)}"
                        )
                    value = value[part]
                if value is None:
                    return ""
                return value if isinstance(value, str) else json.dumps(value, default=str)

            return re.sub(r"\\{\\{\\s*([\\w.]+)\\s*\\}\\}", _resolve, template)
        '''
    )

    if spec.client == "anthropic":
        call_block = _emit_forward_anthropic(node, pascal, spec)
    elif spec.client == "openai":
        call_block = _emit_forward_openai(node, pascal, spec)
    else:  # pragma: no cover — LLMSignature validates the client field
        raise ValueError(f"Unsupported LLM client: {spec.client!r}")

    usage_block = _emit_usage_logger(node, spec)

    return (
        header
        + "\n\n"
        + models_block
        + "\n\n\n"
        + prompt_block
        + "\n\n"
        + description_block
        + schema_block
        + "\n\n"
        + overrides_block
        + "\n\n\n"
        + interpolate_block
        + "\n\n"
        + usage_block
        + "\n\n"
        + call_block
    )


def _emit_usage_logger(node: Node, spec: LLMSignature) -> str:
    """Emit the opt-in usage log hook.

    When ``ROTE_USAGE_LOG`` names a file, every judge call appends one
    JSONL record with its real token usage. This is what lets
    ``rote eval --run`` measure the graduated pipeline's actual LLM
    footprint instead of estimating it — and it doubles as a zero-setup
    observability tap in production. No env var → no-op, and a logging
    failure never breaks the judge call itself.

    Records the module-level ``MODEL`` constant, not the emit-time
    default — operators can swap models at runtime via
    ``ROTE_MODEL_<NODE>``, and measured usage must be priced against
    the model that actually served the call.
    """
    if spec.client == "openai":
        in_attr, out_attr = "prompt_tokens", "completion_tokens"
    else:
        in_attr, out_attr = "input_tokens", "output_tokens"
    return (
        f"def _log_usage(response: Any) -> None:\n"
        f'    """Append token usage as JSONL to $ROTE_USAGE_LOG, if set."""\n'
        f'    path = os.environ.get("ROTE_USAGE_LOG")\n'
        f"    if not path:\n"
        f"        return\n"
        f"    try:\n"
        f'        usage = getattr(response, "usage", None)\n'
        f"        record = {{\n"
        f'            "node": {json.dumps(node.id)},\n'
        f'            "model": MODEL,\n'
        f'            "input_tokens": getattr(usage, {json.dumps(in_attr)}, None),\n'
        f'            "output_tokens": getattr(usage, {json.dumps(out_attr)}, None),\n'
        f"        }}\n"
        f'        with open(path, "a", encoding="utf-8") as f:\n'
        f'            f.write(json.dumps(record) + "\\n")\n'
        f"    except OSError:\n"
        f"        pass\n"
    )


def _default_model_for(
    spec: LLMSignature, anthropic_default_model: str, openai_default_model: str
) -> str:
    if spec.client == "openai":
        return openai_default_model
    return anthropic_default_model


def _temperature_line(spec: LLMSignature) -> str:
    if spec.temperature is None:
        return ""
    return f"            temperature={spec.temperature},\n"


def _emit_forward_anthropic(node: Node, pascal: str, spec: LLMSignature) -> str:
    temp = _temperature_line(spec)
    return (
        f"class {pascal}:\n"
        f'    """Typed LLM judge for {node.id} (Anthropic structured output)."""\n'
        f"\n"
        f"    def forward(self, inputs: {pascal}Input) -> {pascal}Output:\n"
        f"        # Lazy import: the SDK is only needed at call time, and the\n"
        f"        # module stays importable in environments without it.\n"
        f"        import anthropic\n"
        f"\n"
        f"        client = anthropic.Anthropic(base_url=BASE_URL)\n"
        f"        response = client.messages.create(\n"
        f"            model=MODEL,\n"
        f"            max_tokens=4096,\n"
        f"{temp}"
        f"            tools=[\n"
        f"                {{\n"
        f'                    "name": {json.dumps(node.id)},\n'
        f'                    "description": TOOL_DESCRIPTION,\n'
        f'                    "input_schema": OUTPUT_JSON_SCHEMA,\n'
        f"                }}\n"
        f"            ],\n"
        f'            tool_choice={{"type": "tool", "name": {json.dumps(node.id)}}},\n'
        f"            messages=[\n"
        f"                {{\n"
        f'                    "role": "user",\n'
        f'                    "content": _interpolate(PROMPT, inputs.model_dump()),\n'
        f"                }}\n"
        f"            ],\n"
        f"        )\n"
        f"        _log_usage(response)\n"
        f"        for block in response.content:\n"
        f'            if block.type == "tool_use":\n'
        f"                return {pascal}Output.model_validate(block.input)\n"
        f"        raise RuntimeError(\n"
        f'            "LLM did not return a tool_use block for {node.id}"\n'
        f"        )\n"
    )


def _emit_forward_openai(node: Node, pascal: str, spec: LLMSignature) -> str:
    temp = _temperature_line(spec)
    return (
        f"class {pascal}:\n"
        f'    """Typed LLM judge for {node.id} (OpenAI structured output)."""\n'
        f"\n"
        f"    def forward(self, inputs: {pascal}Input) -> {pascal}Output:\n"
        f"        # Lazy import: the SDK is only needed at call time, and the\n"
        f"        # module stays importable in environments without it.\n"
        f"        import openai\n"
        f"\n"
        f"        client = openai.OpenAI(base_url=BASE_URL)\n"
        f"        response = client.chat.completions.create(\n"
        f"            model=MODEL,\n"
        f"{temp}"
        f"            response_format={{\n"
        f'                "type": "json_schema",\n'
        f'                "json_schema": {{\n'
        f'                    "name": {json.dumps(node.id)},\n'
        f'                    "schema": OUTPUT_JSON_SCHEMA,\n'
        f'                    "strict": True,\n'
        f"                }},\n"
        f"            }},\n"
        f"            messages=[\n"
        f"                {{\n"
        f'                    "role": "user",\n'
        f'                    "content": _interpolate(PROMPT, inputs.model_dump()),\n'
        f"                }}\n"
        f"            ],\n"
        f"        )\n"
        f"        _log_usage(response)\n"
        f"        content = response.choices[0].message.content\n"
        f"        if not content:\n"
        f'            raise RuntimeError("OpenAI returned no content for {node.id}")\n'
        f"        return {pascal}Output.model_validate(json.loads(content))\n"
    )


# ───────── extracted/<module>.py emission ─────────


def _extracted_layout(pipeline: Pipeline) -> dict[str, list[Node]]:
    """Group impl-bearing and agent_loop nodes into extracted modules.

    ``pure_function`` / ``external_call`` nodes carry an
    ``extracted/<module>.py:<func>`` impl path; several nodes may share
    a module (BDR's hubspot.py has three functions). ``agent_loop``
    nodes have no impl path — each gets its own ``<node_id>.py`` module
    with a same-named stub function.
    """
    modules: dict[str, list[Node]] = {}
    for node in pipeline.nodes:
        if node.kind in (NodeKind.PURE_FUNCTION, NodeKind.EXTERNAL_CALL):
            assert node.impl is not None
            module_name, _ = _impl_path_parts(node.impl)
            modules.setdefault(module_name, []).append(node)
        elif node.kind is NodeKind.AGENT_LOOP:
            modules.setdefault(node.id, []).append(node)
    return modules


def _extracted_func_name(node: Node) -> str:
    if node.kind is NodeKind.AGENT_LOOP:
        return node.id
    assert node.impl is not None
    _, func_name = _impl_path_parts(node.impl)
    return func_name


def emit_extracted_module(
    module_name: str,
    nodes: list[Node],
    *,
    generated_by: str,
    caller_note: str,
) -> str:
    """Emit extracted/<module_name>.py with one stub function per node.

    Same scaffolding convention as the Temporal example package: the
    functions raise ``NotImplementedError`` until the user fills them in
    with direct vendor API calls. The graduation history (MCP origin) is
    documented in docstrings only — never in executable code.

    ``caller_note`` finishes the "Keep the signatures: …" sentence with
    the calling adapter's description of who invokes these stubs.
    """
    header_template = textwrap.dedent(
        '''\
        """Extracted module: {module_name}

        Auto-generated stubs by {generated_by}. Replace each body
        with the real implementation (direct vendor API calls — the MCP
        tool calls from the source skill were graduated away at emit
        time). Keep the signatures: {caller_note}
        """

        from __future__ import annotations

        from typing import Any
        '''
    )
    parts: list[str] = [
        header_template.format(
            module_name=module_name,
            generated_by=generated_by,
            caller_note=caller_note,
        )
    ]

    seen: set[str] = set()
    for node in nodes:
        func_name = _extracted_func_name(node)
        if func_name in seen:
            continue
        seen.add(func_name)
        desc_first = safe_docstring_line(node.description, fallback=node.id)

        doc_lines = [f"    {desc_first}", ""]
        if node.kind is NodeKind.AGENT_LOOP:
            doc_lines.append("    STUB — agent loops require an LLM agent runtime. Implement")
            doc_lines.append("    against the project's preferred agent harness (Anthropic")
            doc_lines.append("    Agent SDK, OpenAI Agents SDK, LangGraph, etc.).")
            if node.tools:
                doc_lines.append("")
                doc_lines.append("    Tools the agent should be allowed to call:")
                doc_lines.extend(f"      - {t}" for t in node.tools)
            if node.loop_body:
                doc_lines.append("")
                doc_lines.append("    Loop body sub-nodes (call once per iteration):")
                doc_lines.extend(f"      - {sn}" for sn in node.loop_body)
        else:
            doc_lines.append("    STUB — replace with the deterministic API call.")
        if node.mandatory:
            doc_lines.append("")
            doc_lines.append("    MANDATORY: this node was marked mandatory in the source")
            doc_lines.append("    skill. The workflow always calls it; do not make it")
            doc_lines.append("    conditional.")
        if node.constants:
            doc_lines.append("")
            doc_lines.append("    Constants from the IR (lifted from the source skill):")
            for k, v in node.constants.items():
                doc_lines.extend(
                    textwrap.wrap(
                        f"{k} = {v!r}",
                        width=88,
                        initial_indent="      ",
                        subsequent_indent="          ",
                    )
                )

        doc = "\n".join(doc_lines)
        parts.append(
            f"\n\ndef {func_name}(**payload: Any) -> Any:\n"
            f'    """\n{doc}\n    """\n'
            f"    raise NotImplementedError(\n"
            f'        "{module_name}.{func_name}: implement against the vendor API"\n'
            f"    )\n"
        )

    return "".join(parts)


# ───────── Emitted runtime helpers ─────────

_SERIALIZE_TEMPLATE = '''\
def _serialize(obj: Any) -> Any:
    """Convert pydantic models / tuples to plain JSON-safe values.

    {purpose}
    """
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, (list, tuple)):
        return [_serialize(x) for x in obj]
    if isinstance(obj, dict):
        return {{k: _serialize(v) for k, v in obj.items()}}
    return obj
'''


def serialize_helper(purpose: str) -> str:
    """The ``_serialize`` function emitted into every Python runtime's main.

    Behavioral emitted code: if the runtimes' serializers drift, the
    same pipeline produces differently-shaped results per target.
    ``purpose`` is the runtime-specific docstring line explaining why
    payloads must be JSON-safe (continuation lines indented 4 spaces).
    """
    return _SERIALIZE_TEMPLATE.format(purpose=purpose)
