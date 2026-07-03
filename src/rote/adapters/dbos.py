"""DBOS adapter — emits a durable Python app from a Pipeline IR.

Layer 3 of rote: takes a validated :class:`rote.ir.Pipeline` and emits a
runnable DBOS Transact application. DBOS is the "no workflow runner"
target — durable execution as a Python library checkpointing to Postgres
(or SQLite for local dev). There is no orchestrator process to deploy;
``python main.py`` *is* the runtime.

Output layout::

    out/
        main.py                     # @DBOS.workflow + one @DBOS.step per node
        dbos-config.yaml            # CLI/Cloud tooling config (`dbos start`)
        README.md                   # how to run, signal HITL gates, deploy
        extracted/<module>.py       # stubs for pure_function / external_call
        extracted/<agent_loop>.py   # stubs for agent_loop nodes
        signatures/<llm_judge>.py   # typed Pydantic + vendor-SDK signatures

Key design choices vs. the other adapters:

* **Single process, single file orchestration.** DBOS has no
  workflow/activity split (Temporal) and no platform entrypoint class
  (Cloudflare). Steps and the workflow live together in ``main.py``;
  durability comes from the library checkpointing every step result to
  the system database.

* **Parallel waves via ``Queue``.** DBOS's documented concurrency
  primitive is enqueueing work onto a :class:`dbos.Queue`, which returns
  a ``WorkflowHandle`` per enqueued function. Multi-node waves enqueue
  every node then join with ``get_result()``; single-node waves call the
  step directly. This keeps the emitted code synchronous and the
  step-checkpoint ordering deterministic (asyncio.gather over steps has
  subtler replay-ordering semantics).

* **HITL gates via durable messages.** ``DBOS.recv(topic=...)`` parks
  the workflow in the system database until ``DBOS.send(workflow_id,
  payload, topic=...)`` delivers the message — the officially documented
  human-in-the-loop pattern. The IR ``signal`` name maps to the topic.

* **Data-flow threading matches the Temporal adapter.** The workflow
  takes the pipeline input dict as ``pipeline_input``, binds every
  node's result (including HITL gate resume payloads) as
  ``<id>_result``, and builds each step's payload from the node's
  ``inputs:`` bindings via the shared reference grammar
  (:func:`rote.ir.parse_input_ref`). Forward references are rejected at
  emit time by :func:`rote.adapters._common.check_input_refs_available`.

* **Typed LLM signatures as generated Pydantic modules.** When the IR
  carries ``signature_spec`` (preferred), the adapter converts the JSON
  Schemas to Pydantic model source and emits a direct vendor-SDK call
  with structured (tool-use / JSON-schema) output — the Python analog of
  the Cloudflare adapter's Zod emission. The legacy ``signature`` path
  form falls back to importing the user's module, matching Temporal.

The emitted code never imports MCP runtime — same architectural invariant
as the other adapters, enforced by AST tests.
"""

from __future__ import annotations

import json
import keyword
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rote.adapters._common import (
    _execution_waves,
    _pipeline_hash,
    _to_pascal_case,
    check_input_refs_available,
    resolve_within,
    safe_docstring_line,
)
from rote.adapters.temporal import (
    _impl_path_parts,
    _payload_literal,
    _signature_path_parts,
)
from rote.ir import LLMSignature, Node, NodeKind, Pipeline

# ───────── Adapter configuration ─────────


@dataclass(frozen=True)
class DbosAdapterConfig:
    """Per-emission knobs for the DBOS adapter.

    Defaults work out-of-the-box for the BDR example: SQLite system
    database for local dev (overridable via ``DBOS_SYSTEM_DATABASE_URL``
    for Postgres in production), admin server off.
    """

    anthropic_default_model: str = "claude-sonnet-4-6"
    openai_default_model: str = "gpt-4.1"
    # Module path used for *legacy* `signature: path.py:Class` judges,
    # which import a user-maintained Python module instead of a
    # generated one. Mirrors TemporalAdapterConfig.signatures_module.
    legacy_signatures_module: str = "signatures"
    # Steps enqueued for parallel waves land on this queue. None means
    # derive "<pipeline-name>-queue".
    queue_name: str | None = None


# ───────── Duration handling ─────────

_DURATION_RE = re.compile(r"^(\d+(?:\.\d+)?)(ms|s|m|h|d)$")

_UNIT_TO_SECONDS = {
    "ms": 0.001,
    "s": 1.0,
    "m": 60.0,
    "h": 3600.0,
    "d": 86400.0,
}


def _duration_to_seconds(s: str) -> float:
    """Convert an IR duration string ('5m', '30s', '7d') to seconds.

    DBOS APIs take plain ``timeout_seconds`` floats, so unlike the
    Temporal adapter (which emits a runtime parser) we convert at
    emission time and emit a literal.
    """
    m = _DURATION_RE.fullmatch(s.strip())
    if not m:
        raise ValueError(
            f"DBOS adapter: cannot parse IR duration {s!r}. Expected forms like '30s', '5m', '7d'."
        )
    return float(m.group(1)) * _UNIT_TO_SECONDS[m.group(2)]


def _seconds_literal(seconds: float) -> str:
    """Render a seconds value as a compact Python literal."""
    if seconds == int(seconds):
        return str(int(seconds))
    return repr(seconds)


# ───────── Retry mapping ─────────


def _step_decorator(node: Node) -> str:
    """Render the ``@DBOS.step(...)`` decorator line for a node.

    Maps the IR's ``RetryPolicy`` onto DBOS step retry parameters:

    | IR field      | DBOS parameter                                |
    |---------------|-----------------------------------------------|
    | ``max``       | ``max_attempts`` (= max + 1; DBOS counts the  |
    |               | initial attempt, like Temporal)               |
    | ``backoff``   | ``backoff_rate`` (exponential → 2.0;          |
    |               | constant → 1.0; linear → 1.0, the closest     |
    |               | available approximation — DBOS delay is       |
    |               | ``interval_seconds * backoff_rate**attempt``) |
    | ``retry_on``  | not mapped (DBOS retries any exception; its   |
    |               | ``should_retry`` predicate takes a callable,  |
    |               | not categories — surfaced as a comment)       |
    """
    args = [f'name="{node.id}"']
    if node.retry and node.retry.max > 0:
        args.append("retries_allowed=True")
        args.append(f"max_attempts={node.retry.max + 1}")
        backoff_rate = 2.0 if node.retry.backoff == "exponential" else 1.0
        args.append(f"backoff_rate={backoff_rate}")
    return f"@DBOS.step({', '.join(args)})"


def _retry_on_comment(node: Node) -> str:
    """Emit a comment documenting retry_on categories DBOS can't express."""
    if node.retry and node.retry.retry_on:
        cats = ", ".join(node.retry.retry_on)
        return (
            f"    # retry_on categories from the IR: {cats}. DBOS retries any\n"
            f"    # exception; narrow with @DBOS.step(should_retry=...) if needed.\n"
        )
    return ""


def _timeout_comment(node: Node) -> str:
    """DBOS steps have no per-step timeout primitive; document the IR value."""
    if node.timeout:
        return (
            f"    # IR timeout {node.timeout!r}: DBOS has no per-step timeout\n"
            f"    # primitive; enforce inside the implementation if required.\n"
        )
    return ""


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

    # ── public API ──

    def add_defs(self, defs: dict[str, Any]) -> None:
        for name, schema in defs.items():
            if name in self._defs and self._defs[name] != schema:
                raise ValueError(
                    f"DBOS adapter: $defs entry {name!r} appears in both input "
                    f"and output schemas with different definitions"
                )
            self._defs[name] = schema

    def emit_root(self, class_name: str, schema: dict[str, Any]) -> None:
        if schema.get("type") != "object":
            raise ValueError(
                f"DBOS adapter: root signature schema for {class_name} must be "
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
            raise ValueError(f"DBOS adapter: unsupported $ref form: {ref!r}")
        def_name = ref[len("#/$defs/") :]
        if def_name in self._emitted:
            return self._emitted[def_name]
        if def_name in self._in_progress:
            raise ValueError(f"DBOS adapter: cyclic $ref through {def_name!r} is not supported")
        if def_name not in self._defs:
            raise ValueError(f"DBOS adapter: unknown $ref target: {def_name!r}")

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
                    f"DBOS adapter: schema property {field_name!r} in "
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


def emit_signature_module(node: Node, cfg: DbosAdapterConfig) -> str:
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
    TS emission, so the two runtimes share prompt semantics.
    """
    if node.signature_spec is None:
        raise ValueError(
            f"DBOS adapter: emit_signature_module requires signature_spec on node {node.id!r}"
        )
    spec = node.signature_spec
    pascal = _to_pascal_case(node.id)

    converter = _SchemaToPydantic()
    converter.add_defs(spec.input_schema.get("$defs", {}))
    converter.add_defs(spec.output_schema.get("$defs", {}))
    converter.emit_root(f"{pascal}Input", spec.input_schema)
    converter.emit_root(f"{pascal}Output", spec.output_schema)

    model = spec.model or _default_model_for(spec, cfg)
    desc_first = safe_docstring_line(node.description, fallback=node.id)

    typing_names = "Any, Literal" if converter.uses_literal else "Any"
    schema_json = json.dumps(spec.output_schema, separators=(",", ":"))

    header = textwrap.dedent(
        f'''\
        """Typed LLM signature: {node.id}

        {desc_first}

        Auto-generated by rote.adapters.dbos from the IR's ``signature_spec``.
        The non-determinism lives inside this module; the workflow step that
        calls it stays a checkpointed, retryable unit.

        DO NOT EDIT BY HAND. Re-run ``rote emit --runtime dbos`` to regenerate.
        """

        from __future__ import annotations

        import json
        import re
        from typing import {typing_names}

        from pydantic import BaseModel, ConfigDict
        '''
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
        call_block = _emit_forward_anthropic(node, pascal, model, spec)
    elif spec.client == "openai":
        call_block = _emit_forward_openai(node, pascal, model, spec)
    else:  # pragma: no cover — LLMSignature validates the client field
        raise ValueError(f"Unsupported LLM client: {spec.client!r}")

    return (
        header
        + "\n\n"
        + models_block
        + "\n\n\n"
        + prompt_block
        + "\n\n"
        + description_block
        + schema_block
        + "\n\n\n"
        + interpolate_block
        + "\n\n"
        + call_block
    )


def _default_model_for(spec: LLMSignature, cfg: DbosAdapterConfig) -> str:
    if spec.client == "openai":
        return cfg.openai_default_model
    return cfg.anthropic_default_model


def _temperature_line(spec: LLMSignature) -> str:
    if spec.temperature is None:
        return ""
    return f"            temperature={spec.temperature},\n"


def _emit_forward_anthropic(node: Node, pascal: str, model: str, spec: LLMSignature) -> str:
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
        f"        client = anthropic.Anthropic()\n"
        f"        response = client.messages.create(\n"
        f"            model={json.dumps(model)},\n"
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
        f"        for block in response.content:\n"
        f'            if block.type == "tool_use":\n'
        f"                return {pascal}Output.model_validate(block.input)\n"
        f"        raise RuntimeError(\n"
        f'            "LLM did not return a tool_use block for {node.id}"\n'
        f"        )\n"
    )


def _emit_forward_openai(node: Node, pascal: str, model: str, spec: LLMSignature) -> str:
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
        f"        client = openai.OpenAI()\n"
        f"        response = client.chat.completions.create(\n"
        f"            model={json.dumps(model)},\n"
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


def emit_extracted_module(module_name: str, nodes: list[Node]) -> str:
    """Emit extracted/<module_name>.py with one stub function per node.

    Same scaffolding convention as the Temporal example package: the
    functions raise ``NotImplementedError`` until the user fills them in
    with direct vendor API calls. The graduation history (MCP origin) is
    documented in docstrings only — never in executable code.
    """
    parts: list[str] = [
        textwrap.dedent(
            f'''\
            """Extracted module: {module_name}

            Auto-generated stubs by rote.adapters.dbos. Replace each body
            with the real implementation (direct vendor API calls — the MCP
            tool calls from the source skill were graduated away at emit
            time). Keep the signatures: the DBOS steps in main.py call these
            with the step payload as keyword arguments.
            """

            from __future__ import annotations

            from typing import Any
            '''
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


# ───────── main.py emission ─────────


def _emit_step_pure_or_external(node: Node) -> str:
    assert node.impl is not None
    module_name, func_name = _impl_path_parts(node.impl)
    mandatory_marker = ""
    if node.mandatory:
        mandatory_marker = (
            "    # MANDATORY: this node was marked mandatory in the source\n"
            "    # skill. The workflow always calls it; do not make it conditional.\n"
        )
    return (
        f"{_step_decorator(node)}\n"
        f"def {node.id}(payload: dict) -> dict:\n"
        f'    """{safe_docstring_line(node.description)}\n'
        f"\n"
        f"    Graduated from MCP tool call → deterministic API call. See\n"
        f"    ``extracted.{module_name}`` for the implementation.\n"
        f'    """\n'
        f"{mandatory_marker}"
        f"{_retry_on_comment(node)}"
        f"{_timeout_comment(node)}"
        f"    from extracted.{module_name} import {func_name}\n"
        f"\n"
        f"    return _serialize({func_name}(**payload))\n"
    )


def _emit_step_llm_judge(node: Node, cfg: DbosAdapterConfig) -> str:
    pascal_parts: tuple[str, str]
    if node.signature_spec is not None:
        # Preferred: generated signature module (signatures/<node_id>.py).
        module_name = node.id
        class_name = _to_pascal_case(node.id)
        import_line = f"    from signatures.{module_name} import {class_name}, {class_name}Input\n"
        call_lines = (
            f"    judge = {class_name}()\n"
            f"    return judge.forward({class_name}Input(**payload)).model_dump()\n"
        )
    else:
        # Legacy Temporal-style path: user-maintained module with an
        # async ``forward`` (the Temporal adapter's convention).
        assert node.signature is not None
        pascal_parts = _signature_path_parts(node.signature)
        module_name, class_name = pascal_parts
        import_line = (
            f"    import asyncio\n"
            f"\n"
            f"    from {cfg.legacy_signatures_module}.{module_name} import (\n"
            f"        {class_name},\n"
            f"        {class_name}Input,\n"
            f"    )\n"
        )
        call_lines = (
            f"    judge = {class_name}()\n"
            f"    result = asyncio.run(judge.forward({class_name}Input(**payload)))\n"
            f"    return result.model_dump()\n"
        )
    return (
        f"{_step_decorator(node)}\n"
        f"def {node.id}(payload: dict) -> dict:\n"
        f'    """{safe_docstring_line(node.description)}\n'
        f"\n"
        f"    LLM judge — typed input/output, bounded decision space. The\n"
        f"    non-determinism lives inside this step, not the workflow.\n"
        f'    """\n'
        f"{_retry_on_comment(node)}"
        f"{_timeout_comment(node)}"
        f"{import_line}"
        f"\n"
        f"{call_lines}"
    )


def _emit_step_agent_loop(node: Node) -> str:
    return (
        f"{_step_decorator(node)}\n"
        f"def {node.id}(payload: dict) -> dict:\n"
        f'    """{safe_docstring_line(node.description)}\n'
        f"\n"
        f"    Agent loop — bounded, tool-restricted. The stub in\n"
        f"    ``extracted.{node.id}`` raises until implemented against an\n"
        f"    agent harness.\n"
        f'    """\n'
        f"{_retry_on_comment(node)}"
        f"{_timeout_comment(node)}"
        f"    from extracted.{node.id} import {node.id} as _impl\n"
        f"\n"
        f"    return _serialize(_impl(**payload))\n"
    )


def _emit_hitl_wait(node: Node, pipeline: Pipeline) -> str:
    assert node.signal is not None
    timeout_str = node.timeout or pipeline.config.hitl.default_timeout
    timeout_seconds = _duration_to_seconds(timeout_str)
    return (
        f"    # ─── HITL gate: {node.id} ───\n"
        f"    # Durable receive: the workflow parks in the system database until\n"
        f"    # DBOS.send(workflow_id, payload, topic={node.signal!r})\n"
        f"    # delivers the resume message. Survives process restarts.\n"
        f"    {node.id}_result = DBOS.recv(\n"
        f'        topic="{node.signal}",\n'
        f"        timeout_seconds={_seconds_literal(timeout_seconds)},  # {timeout_str}\n"
        f"    )\n"
        f"    if {node.id}_result is None:\n"
        f"        raise TimeoutError(\n"
        f'            "HITL gate {node.id!r} timed out after {timeout_str} waiting "\n'
        f'            "for signal {node.signal!r}"\n'
        f"        )\n"
    )


def _emit_workflow_body(pipeline: Pipeline) -> str:
    waves = _execution_waves(pipeline)
    lines: list[str] = []

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
        lines.append(f"    # ─── Wave {wave_idx} ───")

        if len(non_hitl) == 1:
            node = non_hitl[0]
            payload = _payload_literal(node, indent=" " * 8)
            if payload == "{}":
                lines.append(f"    {node.id}_result = {node.id}({{}})")
            else:
                lines.append(f"    {node.id}_result = {node.id}(")
                lines.append(f"        {payload}")
                lines.append("    )")
        elif len(non_hitl) > 1:
            lines.append("    # Parallel fan-out: enqueue every node in the wave, then")
            lines.append("    # join. Each enqueued step runs as its own one-step")
            lines.append("    # workflow; get_result() blocks durably.")
            for node in non_hitl:
                payload = _payload_literal(node, indent=" " * 8)
                if payload == "{}":
                    lines.append(f"    {node.id}_handle = queue.enqueue({node.id}, {{}})")
                else:
                    lines.append(f"    {node.id}_handle = queue.enqueue(")
                    lines.append(f"        {node.id},")
                    lines.append(f"        {payload},")
                    lines.append("    )")
            for node in non_hitl:
                lines.append(f"    {node.id}_result = {node.id}_handle.get_result()")

        for gate in hitl:
            lines.append(_emit_hitl_wait(gate, pipeline).rstrip("\n"))

        available.update(n.id for n in wave)

    lines.append("")
    lines.append("    return {")
    for exit_id in pipeline.exit_nodes:
        lines.append(f'        "{exit_id}": {exit_id}_result,')
    lines.append("    }")
    return "\n".join(lines)


def emit_main(pipeline: Pipeline, cfg: DbosAdapterConfig | None = None) -> str:
    """Render the main.py source for a pipeline."""
    cfg = cfg or DbosAdapterConfig()

    pascal = _to_pascal_case(pipeline.name)
    pipeline_h = _pipeline_hash(pipeline)
    workflow_name = f"{pascal}_{pipeline_h}"
    queue_name = cfg.queue_name or f"{pipeline.name}-queue"
    sqlite_file = f"{pipeline.name}.dbos.sqlite"
    desc_first = safe_docstring_line(pipeline.description, fallback=pipeline.name)

    header = textwrap.dedent(
        f'''\
        """Auto-generated by rote.adapters.dbos.

        Pipeline: {pipeline.name} v{pipeline.version}
        Source skill: {pipeline.source_skill or "unknown"}
        Pipeline hash: {pipeline_h}

        DO NOT EDIT BY HAND. Re-run ``rote emit --runtime dbos`` to regenerate.

        Workflow versioning: the registered workflow name includes a hash of
        the pipeline so regenerated pipelines become a new workflow type.
        In-flight workflows recover onto the code they started with (DBOS
        recovers by workflow name + application_version); new starts use the
        new code.

        Architecture note: every step in this file wraps a deterministic
        function from ``extracted/`` or a typed LLM signature from
        ``signatures/``. None of them call MCP tools at runtime — the MCP
        tool calls from the source skill were graduated into direct API
        calls during the rote emission step.
        """

        from __future__ import annotations

        import json
        import os
        import sys
        import threading
        from pathlib import Path
        from typing import Any

        from dbos import DBOS, DBOSConfig, Queue

        # ───────── DBOS configuration ─────────
        #
        # Local dev runs on SQLite (zero infrastructure). Production should
        # point DBOS_SYSTEM_DATABASE_URL at Postgres — SQLite is
        # single-process only.
        _APP_DIR = Path(__file__).resolve().parent

        config: DBOSConfig = {{
            "name": "{pipeline.name}",
            "system_database_url": os.environ.get(
                "DBOS_SYSTEM_DATABASE_URL",
                f"sqlite:///{{_APP_DIR / '{sqlite_file}'}}",
            ),
            # The admin server (port 3001) is opt-in; enable it when you
            # want the DBOS console / management API.
            "run_admin_server": False,
        }}
        DBOS(config=config)

        # Parallel waves fan out onto this queue; each enqueued step runs as
        # a one-step workflow with its own durable handle.
        queue = Queue("{queue_name}")


        def _serialize(obj: Any) -> Any:
            """Convert pydantic models / tuples to plain JSON-safe values.

            Steps return JSON-serializable payloads so workflow state stays
            portable across the system database.
            """
            if hasattr(obj, "model_dump"):
                return obj.model_dump()
            if isinstance(obj, (list, tuple)):
                return [_serialize(x) for x in obj]
            if isinstance(obj, dict):
                return {{k: _serialize(v) for k, v in obj.items()}}
            return obj
        '''
    )

    step_parts: list[str] = []
    for node in pipeline.nodes:
        if node.kind is NodeKind.HITL_GATE:
            continue
        step_parts.append("\n\n")
        if node.kind in (NodeKind.PURE_FUNCTION, NodeKind.EXTERNAL_CALL):
            step_parts.append(_emit_step_pure_or_external(node))
        elif node.kind is NodeKind.LLM_JUDGE:
            step_parts.append(_emit_step_llm_judge(node, cfg))
        elif node.kind is NodeKind.AGENT_LOOP:
            step_parts.append(_emit_step_agent_loop(node))

    workflow_block = (
        f"\n\n@DBOS.workflow(name={json.dumps(workflow_name)})\n"
        f"def run_pipeline(pipeline_input: dict) -> dict:\n"
        f'    """{desc_first}"""\n'
        f"{_emit_workflow_body(pipeline)}\n"
    )

    main_block = textwrap.dedent(
        """\


        if __name__ == "__main__":
            # Launch DBOS (connects to the system database, starts queue
            # workers and recovery). Two modes:
            #
            #   python main.py --serve
            #       Run as a long-lived worker: execute runs enqueued
            #       externally (`rote serve` MCP tools / DBOSClient.enqueue)
            #       against the same system database. This is what
            #       `dbos start` and production deployments run.
            #
            #   python main.py '{"your": "input"}'
            #       Start one pipeline run with the input from argv[1]
            #       (JSON) and block until it completes.
            #
            # HITL gates are resumed from another process either way — see
            # README.md.
            DBOS.launch()
            try:
                if "--serve" in sys.argv[1:]:
                    print("serving: waiting for enqueued runs (Ctrl-C to stop)", file=sys.stderr)
                    threading.Event().wait()
                else:
                    pipeline_input = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
                    handle = DBOS.start_workflow(run_pipeline, pipeline_input)
                    print(f"workflow started: {handle.workflow_id}", file=sys.stderr)
                    print(json.dumps(handle.get_result(), indent=2, default=str))
            finally:
                DBOS.destroy()
        """
    )

    return header + "".join(step_parts) + workflow_block + main_block


# ───────── dbos-config.yaml + README emission ─────────


def emit_dbos_config(pipeline: Pipeline) -> str:
    """Emit dbos-config.yaml so ``dbos start`` (and DBOS Cloud) work.

    Runtime configuration is programmatic (the ``DBOSConfig`` dict in
    main.py); this file only feeds the CLI/Cloud tooling.
    """
    return (
        "# Auto-generated by rote.adapters.dbos. Used by the `dbos` CLI and\n"
        "# DBOS Cloud; runtime config lives in main.py's DBOSConfig.\n"
        f"name: {pipeline.name}\n"
        "language: python\n"
        "runtimeConfig:\n"
        "  start:\n"
        "    # Long-lived worker mode: executes runs enqueued externally\n"
        "    # (rote serve / DBOSClient). One-shot runs: python3 main.py '{...}'.\n"
        "    - python3 main.py --serve\n"
    )


def emit_readme(pipeline: Pipeline, cfg: DbosAdapterConfig) -> str:
    gates = [n for n in pipeline.nodes if n.kind is NodeKind.HITL_GATE]
    gate_lines = "\n".join(
        f"| `{g.id}` | `{g.signal}` | {g.timeout or pipeline.config.hitl.default_timeout} |"
        for g in gates
    )
    first_signal = gates[0].signal if gates else "example_signal"
    # Dedent BEFORE interpolating: a multi-line gate_lines value would put
    # column-0 lines inside the template, making dedent a no-op and shipping
    # the whole README indented (Markdown renders that as one code block).
    template = textwrap.dedent(
        """\
        # {pipeline_name} — DBOS runtime

        Auto-generated by `rote emit --runtime dbos`. Do not edit generated
        files by hand; re-run the emitter to regenerate.

        DBOS is durable execution as a library: there is no orchestrator to
        deploy. `main.py` checkpoints every step to a system database and
        resumes from the last completed step after a crash.

        ## Layout

        - `main.py` — the `@DBOS.workflow` DAG plus one `@DBOS.step` per node
        - `extracted/` — stubs for deterministic nodes; fill these in with
          direct vendor API calls (`NotImplementedError` until you do)
        - `signatures/` — typed LLM judges generated from the pipeline IR
          (Pydantic models + direct vendor SDK calls)
        - `dbos-config.yaml` — CLI/Cloud tooling config for `dbos start`

        ## Run locally

        ```sh
        pip install dbos
        python main.py '{{"your": "input"}}'   # one run, blocks until done
        python main.py --serve                 # long-lived worker (see below)
        ```

        `--serve` keeps the process alive executing runs enqueued externally
        — `rote register --runtime dbos` + `rote serve` expose this app as an
        MCP tool whose trigger enqueues onto this app's queue. `dbos start`
        runs the same mode.

        By default the system database is a SQLite file next to `main.py` —
        zero infrastructure, ideal for development. For production, point
        DBOS at Postgres (SQLite is single-process only):

        ```sh
        export DBOS_SYSTEM_DATABASE_URL="postgresql://user:pass@host:5432/db"
        python main.py '{{...}}'   # or: dbos start
        ```

        LLM judge steps call the vendor SDK directly and read the standard
        `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` environment variables.

        ## HITL gates

        The workflow parks durably at each gate until a message arrives on
        the gate's topic:

        | Gate | Signal (topic) | Timeout |
        | --- | --- | --- |
        {gate_lines}

        Resume a parked workflow from any process that can reach the system
        database:

        ```python
        from dbos import DBOSClient

        client = DBOSClient(system_database_url="...")
        client.send(workflow_id, {{"approved": True}}, topic="{first_signal}")
        ```

        (In-process, `DBOS.send(workflow_id, payload, topic=...)` does the
        same.) A gate that times out raises `TimeoutError` and fails the
        run — silence is not approval.
        """
    )
    return template.format(
        pipeline_name=pipeline.name,
        gate_lines=gate_lines,
        first_signal=first_signal,
    )


# ───────── Adapter facade ─────────


class DbosAdapter:
    """Facade that emits a DBOS application from a Pipeline IR.

    Output layout::

        out/
            main.py
            dbos-config.yaml
            README.md
            extracted/__init__.py
            extracted/<module>.py
            signatures/__init__.py
            signatures/<llm_judge_id>.py

    The directory runs with ``python main.py`` once the user fills in the
    extracted/ stubs (SQLite system DB by default; Postgres via
    ``DBOS_SYSTEM_DATABASE_URL``).
    """

    def __init__(self, config: DbosAdapterConfig | None = None) -> None:
        self.config = config or DbosAdapterConfig()

    def emit_main(self, pipeline: Pipeline) -> str:
        return emit_main(pipeline, self.config)

    def emit(self, pipeline: Pipeline, output_dir: str | Path) -> dict[str, Path]:
        out = Path(output_dir)
        extracted_dir = out / "extracted"
        signatures_dir = out / "signatures"
        out.mkdir(parents=True, exist_ok=True)

        written: dict[str, Path] = {}

        main_path = out / "main.py"
        main_path.write_text(self.emit_main(pipeline), encoding="utf-8")
        written["main"] = main_path

        extracted_modules = _extracted_layout(pipeline)
        if extracted_modules:
            extracted_dir.mkdir(exist_ok=True)
            init = extracted_dir / "__init__.py"
            init.write_text(
                f'"""Extracted deterministic modules for {pipeline.name}."""\n',
                encoding="utf-8",
            )
            written["extracted/__init__"] = init
            for module_name, nodes in sorted(extracted_modules.items()):
                p = resolve_within(extracted_dir, f"{module_name}.py")
                p.write_text(emit_extracted_module(module_name, nodes), encoding="utf-8")
                written[f"extracted/{module_name}"] = p

        spec_judges = [
            n
            for n in pipeline.nodes
            if n.kind is NodeKind.LLM_JUDGE and n.signature_spec is not None
        ]
        if spec_judges:
            signatures_dir.mkdir(exist_ok=True)
            init = signatures_dir / "__init__.py"
            init.write_text(
                f'"""Generated LLM signatures for {pipeline.name}."""\n',
                encoding="utf-8",
            )
            written["signatures/__init__"] = init
            for node in spec_judges:
                p = resolve_within(signatures_dir, f"{node.id}.py")
                p.write_text(emit_signature_module(node, self.config), encoding="utf-8")
                written[f"signatures/{node.id}"] = p

        config_path = out / "dbos-config.yaml"
        config_path.write_text(emit_dbos_config(pipeline), encoding="utf-8")
        written["dbos-config"] = config_path

        readme_path = out / "README.md"
        readme_path.write_text(emit_readme(pipeline, self.config), encoding="utf-8")
        written["README"] = readme_path

        return written
