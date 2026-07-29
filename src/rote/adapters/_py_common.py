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
from pathlib import Path
from typing import Any

from rote.adapters._common import (
    DEFAULT_AGENT_MAX_ITERATIONS,
    EmitWriter,
    _to_pascal_case,
    safe_docstring_line,
)
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


def fan_out_binding(node: Node, pipeline: Pipeline) -> tuple[str, str, dict[str, str]]:
    """``(element_param, list_expr, scalar_exprs)`` for a ``fan_out`` node.

    The element parameter is the one bound to the upstream *list* the
    node fans over (ir-schema.md); every other input is shared verbatim
    by all invocations. Which param that is, in precedence order:

    1. the param bound to the source of an incoming ``fan_out: true``
       edge (the IR's explicit marker);
    2. else the param bound to a node with any incoming edge — inputs
       may also reference nodes *without* an edge (shared context, e.g.
       BDR's ``intel``), which is what makes "the only node-bound
       param" too naive;
    3. else the only node-bound param.

    Ambiguity after all three is an emit-time error, never a guess:
    dispatching over the wrong list would silently judge the wrong
    things.
    """
    if not node.inputs:
        raise ValueError(f"fan_out node {node.id!r} has no inputs: nothing to fan over")
    node_bound = {
        param: parsed.node_id
        for param, ref in node.inputs.items()
        if (parsed := parse_input_ref(ref)).node_id is not None
    }
    edge_sources = {e.from_ for e in pipeline.edges if e.to == node.id}
    fan_edge_sources = {e.from_ for e in pipeline.edges if e.to == node.id and e.fan_out}

    for sources in (fan_edge_sources, edge_sources):
        matching = sorted(p for p, src in node_bound.items() if src in sources)
        if len(matching) == 1:
            element_param = matching[0]
            break
    else:
        if len(node_bound) != 1:
            raise ValueError(
                f"fan_out node {node.id!r}: cannot identify the element param — "
                f"node-bound inputs {sorted(node_bound)} and incoming edges "
                f"{sorted(edge_sources)} don't single one out; mark the list edge "
                f"with `fan_out: true`"
            )
        element_param = next(iter(node_bound))

    scalars = {p: _ref_to_python_expr(r) for p, r in node.inputs.items() if p != element_param}
    return element_param, _ref_to_python_expr(node.inputs[element_param]), scalars


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


def _safe_field_ident(name: str, taken: set[str]) -> str:
    """Map a JSON property name to a usable Pydantic attribute name.

    Valid non-keyword identifiers pass through unchanged (no alias needed).
    Everything else is sanitized — non-word characters become ``_``, a
    leading digit gets an ``f_`` prefix, keywords get a trailing ``_`` —
    and deduplicated against ``taken`` so two JSON keys can't collapse
    onto one attribute (e.g. ``from`` and ``from_`` in the same object).
    """
    if name.isidentifier() and not keyword.iskeyword(name) and name not in taken:
        return name
    ident = re.sub(r"\W", "_", name)
    if not ident or ident[0].isdigit():
        ident = f"f_{ident}"
    if keyword.iskeyword(ident):
        ident += "_"
    base = ident
    counter = 2
    while ident in taken:
        ident = f"{base}_{counter}"
        counter += 1
    return ident


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
        self.uses_field = False

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
        has_alias = False
        taken: set[str] = set()
        for field_name, field_schema in properties.items():
            # Real-world schemas legitimately carry keys Python can't use as
            # attribute names ("from" on an email, "message-id"): those get a
            # sanitized attribute plus a Field(alias=...) back to the JSON
            # key. The alias is rendered via _py_literal (json escaping), so
            # an adversarial key can't break out of the string literal.
            py_name = _safe_field_ident(field_name, taken)
            taken.add(py_name)
            aliased = py_name != field_name
            if aliased:
                has_alias = True
                self.uses_field = True
            alias_arg = f"alias={_py_literal(field_name)}" if aliased else None

            expr = self._type_expr(field_schema, f"{class_name}{_pascal_ident(field_name)}")
            if field_name in required and "default" not in field_schema:
                suffix = f" = Field({alias_arg})" if alias_arg else ""
                field_lines.append(f"    {py_name}: {expr}{suffix}")
            elif "default" in field_schema:
                default = _py_literal(field_schema["default"])
                value = f"Field(default={default}, {alias_arg})" if alias_arg else default
                field_lines.append(f"    {py_name}: {expr} = {value}")
            else:
                if not expr.endswith("| None") and expr != "None":
                    expr += " | None"
                value = f"Field(default=None, {alias_arg})" if alias_arg else "None"
                field_lines.append(f"    {py_name}: {expr} = {value}")

        lines = [f"class {class_name}(BaseModel):"]
        description = schema.get("description")
        if description:
            first = safe_docstring_line(str(description))
            lines.append(f'    """{first}"""')
            lines.append("")
        config_args: list[str] = []
        if schema.get("additionalProperties") is False:
            config_args.append('extra="forbid"')
        if has_alias:
            # Aliased models must also accept their python attribute names so
            # user code can construct them without knowing the JSON spelling.
            config_args.append("populate_by_name=True")
        if config_args:
            self.uses_config_dict = True
            lines.append(f"    model_config = ConfigDict({', '.join(config_args)})")
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
    pydantic_parts = ["BaseModel"]
    if converter.uses_config_dict:
        pydantic_parts.append("ConfigDict")
    if converter.uses_field:
        pydantic_parts.append("Field")
    pydantic_names = ", ".join(pydantic_parts)
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
    if spec.temperature is not None:
        overrides_block += (
            f"\nTEMPERATURE = os.environ.get(\n"
            f'    "ROTE_TEMPERATURE_{env_suffix}", {json.dumps(str(spec.temperature))}\n'
            f")"
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

    if spec.client not in ("anthropic", "openai"):  # pragma: no cover — IR-validated
        raise ValueError(f"Unsupported LLM client: {spec.client!r}")

    sampling_block = _emit_sampling_helper(spec)

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
        + sampling_block
        + _emit_forward(node, pascal, spec)
    )


def _emit_sampling_helper(spec: LLMSignature) -> str:
    """Emit ``temperature`` as a runtime knob rather than a baked constant.

    Current Anthropic models **reject** ``temperature`` outright with a
    400 ("`temperature` is not supported on this model"), so a constant
    compiled into the judge makes it unrunnable the moment an operator
    points ``ROTE_MODEL_<ID>`` at a newer model — which is the whole
    point of that knob. An empty ``ROTE_TEMPERATURE_<ID>`` omits the
    parameter entirely; unset keeps the IR's value.

    No IR temperature → no knob and no keyword argument, which is the
    right default: the forced tool call already pins the output shape.
    """
    if spec.temperature is None:
        return ""
    return (
        "def _sampling_kwargs() -> dict[str, Any]:\n"
        '    """``temperature`` if configured; empty env value omits it."""\n'
        "    if not TEMPERATURE.strip():\n"
        "        return {}\n"
        '    return {"temperature": float(TEMPERATURE)}\n'
        "\n\n"
    )


def _default_model_for(
    spec: LLMSignature, anthropic_default_model: str, openai_default_model: str
) -> str:
    if spec.client == "openai":
        return openai_default_model
    return anthropic_default_model


def _emit_forward(node: Node, pascal: str, spec: LLMSignature) -> str:
    """Emit the judge class: interpolate, delegate, validate.

    The vendor call itself is deliberately *not* emitted per node. It
    lives in ``signatures/_rote_inference.py`` — the verbatim copy of
    :mod:`rote.inference._runtime_helper` — because the same call has to
    choose between three billing lanes (the user's Claude subscription
    via ``claude -p``, their API key, or their rote cloud tenant) and
    that decision is a fact about the machine, not about the pipeline.
    Emitting it once per judge would mean N copies of the resolver
    drifting apart inside a single generated app.

    What stays here is what genuinely differs per node: the typed
    models, the prompt, the schema, and the operator knobs.
    """
    kind = "Anthropic" if spec.client == "anthropic" else "OpenAI"
    description_arg = (
        "            tool_description=TOOL_DESCRIPTION,\n" if spec.client == "anthropic" else ""
    )
    sampling_arg = (
        "            sampling=_sampling_kwargs(),\n" if spec.temperature is not None else ""
    )
    return (
        f"class {pascal}:\n"
        f'    """Typed LLM judge for {node.id} ({kind} structured output)."""\n'
        f"\n"
        f"    def forward(self, inputs: {pascal}Input) -> {pascal}Output:\n"
        f"        # Which provider serves this call — and who is billed for it —\n"
        f"        # is resolved at runtime; see signatures/_rote_inference.py.\n"
        f"        from signatures._rote_inference import call_judge\n"
        f"\n"
        f"        payload = call_judge(\n"
        f"            node_id={json.dumps(node.id)},\n"
        f"            client={json.dumps(spec.client)},\n"
        f"            model=MODEL,\n"
        f"            base_url=BASE_URL,\n"
        f"            prompt=_interpolate(PROMPT, inputs.model_dump(by_alias=True)),\n"
        f"            output_schema=OUTPUT_JSON_SCHEMA,\n"
        f"{description_arg}"
        f"{sampling_arg}"
        f"        )\n"
        f"        return {pascal}Output.model_validate(payload)\n"
    )


def agent_loop_call(
    node: Node, *, default_model: str, indent: str = "    ", include_local_tools: bool = True
) -> str:
    """Emit the ``run_agent_loop(...)`` call for an ``agent_loop`` node.

    This is what replaced the ``NotImplementedError`` stub. A pipeline
    that hands its one genuinely agentic step back to the user has not
    graduated that step, so the emitted code runs a real bounded loop:
    the subscription CLI when one is available and the tools are MCP,
    otherwise the vendor SDK's own tool runner.

    ``loop_body`` sub-nodes are passed as ``local_tools`` — they are
    already-emitted, already-working steps in the same module, so the
    agent drives them per iteration exactly as the IR describes rather
    than the adapter guessing an order for them.
    """
    max_iterations = (
        node.termination.max_iterations
        if node.termination is not None
        else DEFAULT_AGENT_MAX_ITERATIONS
    )
    inner = indent + "    "
    deep = inner + "    "
    model_expr = f'os.environ.get("ROTE_MODEL_{node.id.upper()}", {json.dumps(default_model)})'
    lines = [
        f"{indent}from signatures._rote_inference import run_agent_loop",
        "",
        f"{indent}return run_agent_loop(",
        f"{inner}node_id={json.dumps(node.id)},",
        f"{inner}description={_chunked_str_literal(node.description, deep)},",
        f"{inner}task=json.dumps(payload, default=str),",
        f"{inner}model={model_expr},",
    ]
    if node.tools:
        one_line = f"{inner}tools={_py_literal(list(node.tools))},"
        if len(one_line) <= 98:
            lines.append(one_line)
        else:
            # Emitted output is committed and ruff-linted at line-length
            # 100, so a long tool list wraps rather than overflowing.
            lines.append(f"{inner}tools=[")
            lines.extend(f"{deep}{_py_literal(t)}," for t in node.tools)
            lines.append(f"{inner}],")
    if node.tool_servers:
        # Resolved tool → server pins narrow the allowlist to real pairs.
        lines.append(f"{inner}tool_servers={_py_literal(dict(sorted(node.tool_servers.items())))},")
    if node.loop_body and include_local_tools:
        lines.append(f"{inner}local_tools={{")
        for sub in node.loop_body:
            lines.append(f"{inner}    {json.dumps(sub)}: {sub},")
        lines.append(f"{inner}}},")
    lines.append(f"{inner}max_iterations={max_iterations},")
    if node.termination is not None:
        condition = _chunked_str_literal(node.termination.condition, deep)
        lines.append(f"{inner}termination={condition},")
    lines.append(f"{indent})")
    return "\n".join(lines) + "\n"


def inference_helper_source() -> str:
    """The source text emitted as ``signatures/_rote_inference.py``.

    Verbatim, byte for byte — the same contract
    ``extracted/_rote_mcp.py`` has with :mod:`rote.mcp._runtime_helper`.
    **Never hand-edit the emitted copy; fix the module and re-emit.**
    """
    from rote.inference import _runtime_helper

    return Path(_runtime_helper.__file__).read_text(encoding="utf-8")


def write_signature_package(
    writer: EmitWriter,
    pipeline: Pipeline,
    *,
    anthropic_default_model: str,
    openai_default_model: str,
    generated_by: str,
    regen_command: str,
    context_note: str,
) -> dict[str, Path]:
    """Write the whole ``signatures/`` package for a pipeline's judges.

    Every Python-emitting adapter needs the identical tree — the package
    marker, one typed module per judge, and the verbatim inference
    helper they all import — differing only in the identity strings
    stamped into each module's docstring. Those are parameters; the
    layout is not, because a runtime that emitted a *different* judge
    package would be a second implementation of the judge.

    Returns the manifest entries to merge into the adapter's ``written``
    map. Empty only when nothing in the pipeline needs inference — an
    ``agent_loop`` pulls the helper in even with no judges, since the
    loop runs through the same provider resolver.
    """
    judges = spec_judges(pipeline)
    agent_loops = pipeline.nodes_by_kind(NodeKind.AGENT_LOOP)
    if not judges and not agent_loops:
        return {}
    pipeline_name = pipeline.name
    written: dict[str, Path] = {
        "signatures/__init__": writer.write(
            "signatures",
            "__init__.py",
            content=f'"""Generated LLM signatures for {pipeline_name}."""\n',
        ),
        # The provider resolver is the *source text* of
        # rote.inference._runtime_helper — one tested implementation,
        # emitted verbatim so the app stays standalone (no rote import
        # at runtime) while every judge shares one billing decision.
        "signatures/_rote_inference": writer.write(
            "signatures",
            "_rote_inference.py",
            content=inference_helper_source(),
        ),
    }
    for node in judges:
        written[f"signatures/{node.id}"] = writer.write(
            "signatures",
            f"{node.id}.py",
            content=emit_signature_module(
                node,
                anthropic_default_model=anthropic_default_model,
                openai_default_model=openai_default_model,
                generated_by=generated_by,
                regen_command=regen_command,
                context_note=context_note,
            ),
        )
    return written


def spec_judges(pipeline: Pipeline) -> list[Node]:
    """The llm_judge nodes carrying a structured ``signature_spec``."""
    return [
        n for n in pipeline.nodes if n.kind is NodeKind.LLM_JUDGE and n.signature_spec is not None
    ]


# ───────── extracted/<module>.py emission ─────────


def resolve_extracted_source(source_dir: Path | None, module_name: str) -> str | None:
    """The agent-written ``extracted/<module>.py`` beside the pipeline, if any.

    A graduation writes its real (test-verified) implementations into
    ``<graduated>/extracted/`` next to ``pipeline.yaml``. Emission
    prefers those files verbatim over IR-derived stubs so the runtime
    dir runs without anyone hand-copying modules across — the gap that
    made the first real-server pipeline raise ``NotImplementedError``
    on step one despite the agent having written the code.
    """
    if source_dir is None:
        return None
    candidate = Path(source_dir) / "extracted" / f"{module_name}.py"
    if not candidate.is_file():
        return None
    return candidate.read_text(encoding="utf-8")


def _extracted_layout(pipeline: Pipeline) -> dict[str, list[Node]]:
    """Group impl-bearing and agent_loop nodes into extracted modules.

    ``pure_function`` / ``external_call`` nodes carry an
    ``extracted/<module>.py:<func>`` impl path; several nodes may share
    a module (BDR's hubspot.py has three functions). ``agent_loop``
    nodes have no impl path — each gets its own ``<node_id>.py`` module
    with a same-named stub function.

    An ``external_call`` with only an ``mcp`` binding (and no ``impl``)
    has no stub to emit — its step body calls the MCP tool directly — so
    it is skipped here.
    """
    modules: dict[str, list[Node]] = {}
    for node in pipeline.nodes:
        if node.kind in (NodeKind.PURE_FUNCTION, NodeKind.EXTERNAL_CALL):
            if node.impl is None:
                # MCP-backed external_call with no direct-API impl: nothing
                # to scaffold in extracted/.
                continue
            module_name, _ = _impl_path_parts(node.impl)
            modules.setdefault(module_name, []).append(node)
        # agent_loop nodes deliberately get no extracted/ module: since
        # the inference helper runs them as real bounded loops, a stub
        # would be dead code that only invites someone to fill it in.
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
                # k is an identifier (IR-validated); v is arbitrary, so escape
                # its repr so a value containing \"\"\" can't close this docstring.
                doc_lines.extend(
                    textwrap.wrap(
                        f"{k} = {safe_docstring_line(repr(v))}",
                        width=88,
                        initial_indent="      ",
                        subsequent_indent="          ",
                    )
                )
        for label, schema in (
            ("Input contract", node.input_schema),
            ("Output contract", node.output_schema),
        ):
            if not schema:
                continue
            doc_lines.append("")
            doc_lines.append(f"    {label} (JSON Schema, from observed real payloads):")
            # json.dumps escapes every quote inside strings, so the rendered
            # block cannot contain an unescaped triple-quote; a trailing
            # backslash is impossible before the newline-terminated fence.
            doc_lines.extend(f"      {line}" for line in json.dumps(schema, indent=2).splitlines())

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
        return obj.model_dump(by_alias=True)
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
