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
import math
import re
import textwrap
from pathlib import Path
from typing import Any

from rote.adapters._common import (
    DEFAULT_AGENT_MAX_ITERATIONS,
    STUB_SENTINEL,
    EmitWriter,
    _to_pascal_case,
    agent_loop_output,
    mcp_tool_arguments,
    py_str_literal,
    safe_docstring_line,
)
from rote.inference import structured_output_schema
from rote.ir import (
    CONSTANT_ARGUMENT_KINDS,
    LLMSignature,
    Node,
    NodeKind,
    Pipeline,
    fan_out_element_param,
    parse_input_ref,
)

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
    | ``pipeline.input.f``       | ``pipeline_input.get("f")``     |
    | ``foo.output``             | ``foo_result``                  |
    | ``foo.output.f``           | ``foo_result["f"]``             |

    A pipeline input field is read with ``.get`` so an omitted optional
    field is None rather than a KeyError. Required fields are checked on
    entry by :func:`required_inputs_guard`, so ``.get`` never hides one.
    """
    parsed = parse_input_ref(ref)
    if parsed.node_id is None:
        if parsed.field is None:
            return "pipeline_input"
        return f'pipeline_input.get("{parsed.field}")'
    base = f"{parsed.node_id}_result"
    if parsed.field is None:
        return base
    return f'{base}["{parsed.field}"]'


def _constant_literal(node: Node, key: str, value: Any) -> str:
    """A constant as a Python literal, refusing values that have none.

    YAML can produce dates, NaN and infinity, none of which ``repr`` turns
    into an expression that evaluates back to the value. Failing at emit
    time names the node; emitting ``datetime.date(...)`` would fail with a
    bare NameError inside generated code.
    """
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"node {node.id!r}: constant {key!r} is {value!r}, not a finite number")
    if isinstance(value, list):
        for item in value:
            _constant_literal(node, key, item)
    elif isinstance(value, dict):
        for item_key, item in value.items():
            if not isinstance(item_key, str):
                raise ValueError(f"node {node.id!r}: constant {key!r} has a non-string key")
            _constant_literal(node, key, item)
    elif not isinstance(value, str | int | float | bool) and value is not None:
        raise ValueError(
            f"node {node.id!r}: constant {key!r} is a {type(value).__name__}; "
            "constants must be JSON values (quote dates in pipeline.yaml)"
        )
    return _py_literal(value)


def impl_kwargs_expr(node: Node) -> str:
    """The expression a step body unpacks into its ``impl`` call.

    Several nodes may share one ``impl`` with different constants (three
    fetch nodes, one ``fetch_file_content``, three ``file_id`` values), so
    each step merges its own node's constants into the payload it was
    called with: ``{**payload, "file_id": "..."}``. Merging in the step
    body, not the workflow's payload literal, is what reaches every
    caller: the DAG, a fan_out element, and an agent loop calling a
    ``loop_body`` sub-node as a tool with a payload the model built. The
    constants never appear in that tool's schema, and they come last so a
    model-built payload cannot override a pinned value.
    """
    if node.kind not in CONSTANT_ARGUMENT_KINDS or not node.constants:
        return "payload"
    pinned = ", ".join(
        f"{json.dumps(key)}: {_constant_literal(node, key, value)}"
        for key, value in node.constants.items()
    )
    return f"{{**payload, {pinned}}}"


def mcp_arguments_line(node: Node, indent: str = "    ") -> str:
    """``arguments = ...`` for an MCP-bound step (see ``mcp_tool_arguments``)."""
    sources = mcp_tool_arguments(node)
    if sources is None:
        return f"{indent}arguments = payload  # tool arg names match the threaded payload keys\n"
    items = ", ".join(
        f"{json.dumps(tool_arg)}: "
        + (
            # A constant and a literal are both fixed values: an escaped
            # Python literal, never spliced (invariant #7).
            _constant_literal(node, tool_arg, value)
            if origin in ("constant", "literal")
            else f"payload[{json.dumps(value)}]"
        )
        for tool_arg, origin, value in sources
    )
    return f"{indent}arguments = {{{items}}}\n"


def _payload_literal(node: Node, indent: str) -> str:
    """Render the step/activity payload dict for a node's data-flow bindings.

    Nodes without ``inputs`` keep the empty payload (back-compat).
    ``indent`` is the indentation of the line the literal starts on;
    continuation lines are indented one level deeper. Constants are not
    here: the step body merges them (:func:`impl_kwargs_expr`).
    """
    if not node.inputs:
        return "{}"
    inner = indent + "    "
    lines = ["{"]
    for param, ref in node.inputs.items():
        lines.append(f"{inner}{_payload_entry(param, ref)},")
    lines.append(indent + "}")
    return "\n".join(lines)


def _payload_entry(param: str, ref: str) -> str:
    """One entry of a payload dict literal.

    A pipeline input field is spread in through ``_input_field`` so that
    an omitted one is left out rather than set to None. Required fields go
    the same way; the entry check has already proven they are there.
    """
    parsed = parse_input_ref(ref)
    if parsed.node_id is None and parsed.field is not None:
        return f"**_input_field(pipeline_input, {json.dumps(param)}, {json.dumps(parsed.field)})"
    return f"{json.dumps(param)}: {_ref_to_python_expr(ref)}"


_FAN_OUT_LIST_HELPER_PY = '''\
def _fan_out_list(value: object, node_id: str, ref: str) -> list:
    """The list a fan_out node dispatches over.

    Iterating a missing key raises "'NoneType' object is not iterable",
    which names neither the node nor the reference — in generated code
    that is close to undebuggable, and the cause is almost always an
    upstream node that didn't return the key the IR says it does. The
    TypeScript runtimes emit the same guard (`fanOutList`), so a missing
    key reports identically on every target.
    """
    if not isinstance(value, list):
        raise TypeError(
            f"fan_out node {node_id!r} expected a list from {ref!r}, got "
            f"{type(value).__name__}. The upstream node must return that "
            f"key as a list."
        )
    return value
'''


def fan_out_list_helper() -> str:
    """The ``_fan_out_list`` guard emitted into Python runtimes that fan out.

    Behavioral emitted code, like ``_serialize``: if the runtimes' guards
    drift, the same broken pipeline reports differently per target.
    """
    return _FAN_OUT_LIST_HELPER_PY


_ASYNC_RUNNER_HELPER_PY = '''\
_ASYNC_LOOP: asyncio.AbstractEventLoop | None = None
_ASYNC_LOOP_THREAD: threading.Thread | None = None
_ASYNC_LOOP_LOCK = threading.Lock()


def _reset_async_loop_after_fork() -> None:
    """Forget the parent's loop in a forked child.

    A child inherits these globals but not the loop's thread, so the
    first ``_run_async`` there would submit to a loop nothing runs and
    block on ``.result()`` forever. Pre-fork servers (gunicorn, uvicorn
    workers) fork after the parent may already have run a step. The lock
    is replaced too: another parent thread may have held it at the fork.
    """
    global _ASYNC_LOOP, _ASYNC_LOOP_THREAD, _ASYNC_LOOP_LOCK
    _ASYNC_LOOP = None
    _ASYNC_LOOP_THREAD = None
    _ASYNC_LOOP_LOCK = threading.Lock()


if hasattr(os, "register_at_fork"):  # absent on platforms without fork()
    os.register_at_fork(after_in_child=_reset_async_loop_after_fork)


def _run_async(awaitable: Any) -> Any:
    """Run ``awaitable`` to completion on this process's background event loop.

    Steps here are synchronous, so async work (an ``async def`` extracted
    function, an MCP call, a legacy async judge) needs a loop to run on.
    It gets one loop for the life of the process, on a daemon thread,
    rather than ``asyncio.run`` per call: an async client that a module
    keeps at module level (an ``httpx.AsyncClient``, say) binds to the
    loop it first ran on, and ``asyncio.run`` closes that loop, so every
    later call (each fan_out element, each retry) failed with "Event loop
    is closed". ``asyncio.run`` also refuses to start inside a running
    loop, which is where ``run_pipeline`` lands when async code imports it.
    """
    global _ASYNC_LOOP, _ASYNC_LOOP_THREAD
    with _ASYNC_LOOP_LOCK:
        if _ASYNC_LOOP is None:
            loop = asyncio.new_event_loop()
            thread = threading.Thread(target=loop.run_forever, name="rote-async", daemon=True)
            thread.start()
            _ASYNC_LOOP, _ASYNC_LOOP_THREAD = loop, thread
    if threading.current_thread() is _ASYNC_LOOP_THREAD:
        close = getattr(awaitable, "close", None)
        if callable(close):
            close()  # never awaited; closing it skips the "never awaited" warning
        raise RuntimeError(
            "a step blocked on async work from inside the background event loop, "
            "which would deadlock; await it instead"
        )

    async def _await() -> Any:
        return await awaitable

    return asyncio.run_coroutine_threadsafe(_await(), _ASYNC_LOOP).result()
'''


_CALL_IMPL_HELPER_PY = '''\
def _call_impl(fn: Any, payload: dict) -> Any:
    """Call an extracted function with the step payload.

    Extracted modules may be plain or ``async def``: the IR-derived stubs
    are plain, while hand-written clients are often async. The check is
    on the result, not the function, so a plain wrapper that returns a
    coroutine is driven to completion too. Awaitables run on the one
    background loop (``_run_async``).
    """
    result = fn(**payload)
    if inspect.isawaitable(result):
        result = _run_async(result)
    return result
'''


_CALL_IMPL_HELPER_ASYNC_PY = '''\
async def _call_impl(fn: Any, payload: dict) -> Any:
    """Call an extracted function with the step payload.

    A coroutine function is awaited on this event loop. A plain one runs
    in a worker thread so it cannot block the loop, and if what it
    returns is awaitable (a plain wrapper around an async client), that
    is awaited here too. The python and dbos runtimes emit the
    synchronous twin of this helper and accept exactly the same forms.
    """
    if inspect.iscoroutinefunction(fn):
        result = await fn(**payload)
    else:
        result = await asyncio.to_thread(fn, **payload)
    if inspect.isawaitable(result):
        result = await result
    return result
'''


def async_runner_helper() -> str:
    """The ``_run_async`` function (and its loop state) for sync Python runtimes.

    Needs ``asyncio``, ``os``, ``threading`` and ``Any`` imported at
    module level.
    """
    return _ASYNC_RUNNER_HELPER_PY


def call_impl_helper(*, is_async: bool = False) -> str:
    """The ``_call_impl`` function emitted into every Python runtime.

    python and dbos steps are plain functions and use the synchronous
    form, which needs :func:`async_runner_helper` and ``inspect``.
    Temporal's activities are coroutines and use the async form, which
    needs ``asyncio`` and ``inspect``. One definition of which shapes an
    extracted function may take, so the runtimes cannot drift.
    """
    return _CALL_IMPL_HELPER_ASYNC_PY if is_async else _CALL_IMPL_HELPER_PY


def needs_call_impl(pipeline: Pipeline) -> bool:
    """Whether any node in ``pipeline`` calls an extracted ``impl``."""
    return any(
        n.kind in (NodeKind.PURE_FUNCTION, NodeKind.EXTERNAL_CALL) and n.impl is not None
        for n in pipeline.nodes
    )


def has_legacy_judges(pipeline: Pipeline) -> bool:
    """Whether any judge uses the legacy ``signature:`` path (async ``forward``)."""
    return any(n.kind is NodeKind.LLM_JUDGE and n.signature_spec is None for n in pipeline.nodes)


_MISSING_INPUTS_HELPER_PY = '''\
def _missing_inputs_error(pipeline_input: object, required: tuple[str, ...]) -> str | None:
    """Why ``pipeline_input`` cannot start a run, or None when it can.

    A step's payload simply leaves out an omitted optional field, so the
    impl's own default applies. A missing required field would be left out
    the same way and fail later, somewhere unrelated, so the run checks
    those first and fails naming them. A run started with no payload at
    all is reported the same way. The TypeScript runtimes emit the same
    check (`missingInputsError`), so a bad input reports identically on
    every target.
    """
    if not isinstance(pipeline_input, dict):
        return (
            "pipeline input must be a JSON object, got " + type(pipeline_input).__name__
        )
    missing = [name for name in required if name not in pipeline_input]
    if not missing:
        return None
    return "pipeline input is missing required field(s): " + ", ".join(missing)


def _input_field(pipeline_input: dict, key: str, field: str) -> dict:
    """``{key: value}`` when the run supplied ``field``, else nothing.

    Spread into a step's payload, so an omitted optional field is absent,
    not None: a Python default (``def f(d="today")``) or a schema default
    on a judge's input then applies, the same as on the TS runtimes, where
    an undefined field is dropped from the JSON payload.
    """
    return {key: pipeline_input[field]} if field in pipeline_input else {}
'''


def missing_inputs_helper(pipeline: Pipeline) -> str:
    """The input-check helpers, or "" when no node reads the pipeline input."""
    return _MISSING_INPUTS_HELPER_PY if pipeline.needs_input_check else ""


def required_inputs_guard(pipeline: Pipeline, raise_stmt: str, indent: str = "    ") -> str:
    """The entry check that fails a run missing a required input field.

    ``raise_stmt`` is the runtime's way to fail the run with the message in
    ``_input_error`` (Temporal needs a non-retryable ``ApplicationError``;
    a plain exception there would retry the workflow task forever).
    Returns "" when no node reads the pipeline input.
    """
    if not pipeline.needs_input_check:
        return ""
    names = "".join(f"{json.dumps(name)}, " for name in pipeline.required_input_fields)
    return (
        f"{indent}_input_error = _missing_inputs_error(pipeline_input, ({names.rstrip()}))\n"
        f"{indent}if _input_error is not None:\n"
        f"{indent}    {raise_stmt}\n"
    )


def cli_input_check(pipeline: Pipeline, indent: str) -> str:
    """The ``__main__`` check that rejects a bad input before anything starts.

    The same check as :func:`required_inputs_guard`, run where the script
    reads argv: a missing field is a usage error, so it is one line on
    stderr and exit status 2, not a traceback (DBOS's ran to ~5 KB). The
    guard inside ``run_pipeline`` stays for callers that import it.
    """
    if not pipeline.needs_input_check:
        return ""
    names = "".join(f"{json.dumps(name)}, " for name in pipeline.required_input_fields)
    return (
        f"{indent}_input_error = _missing_inputs_error(pipeline_input, ({names.rstrip()}))\n"
        f"{indent}if _input_error is not None:\n"
        f'{indent}    print(f"error: {{_input_error}}", file=sys.stderr)\n'
        f"{indent}    sys.exit(2)\n"
    )


def fan_out_binding(
    node: Node, pipeline: Pipeline, indent: str = ""
) -> tuple[str, str, dict[str, str]]:
    """``(element_param, list_expr, scalar_exprs)`` for a ``fan_out`` node.

    ``scalar_exprs`` maps every payload key but the element to its
    complete dict-literal entry (see :func:`_payload_entry`).

    The Python rendering of :func:`fan_out_element_param` — which input
    is the list is a language-neutral property of the IR, so it is
    resolved there and only the expressions are built here.

    ``indent`` is the column the ``_fan_out_list(`` token sits at; the
    guard call wraps against it rather than running long on one line.
    """
    element_param = fan_out_element_param(node, pipeline)
    # Narrowing only: fan_out_element_param raises on a node with no
    # inputs, so reaching here means there is something to fan over.
    assert node.inputs is not None
    scalars = {p: _payload_entry(p, r) for p, r in node.inputs.items() if p != element_param}
    ref = node.inputs[element_param]
    inner = indent + "    "
    # json.dumps, not !r: emitted Python uses double quotes throughout
    # (the repo's ruff-format style), same as _py_literal.
    list_expr = (
        "_fan_out_list(\n"
        f"{inner}{_ref_to_python_expr(ref)},\n"
        f"{inner}{json.dumps(node.id)},\n"
        f"{inner}{json.dumps(ref)},\n"
        f"{indent})"
    )
    return element_param, list_expr, scalars


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
        return py_str_literal(value)
    if isinstance(value, bool) or value is None:
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_py_literal(v) for v in value) + "]"
    if isinstance(value, dict):
        items = ", ".join(f"{_py_literal(k)}: {_py_literal(v)}" for k, v in value.items())
        return "{" + items + "}"
    return repr(value)


def _py_literal_block(value: Any, indent: str, width: int = 100) -> str:
    """:func:`_py_literal`, wrapped one item per line when it overflows.

    Emitted code is committed and ruff-linted at line-length 100. The first
    line continues whatever precedes it (``output_schema=``), so the block
    is fitted against ``width`` minus ``indent`` plus a margin for that.
    """
    flat = _py_literal(value)
    if len(indent) + len(flat) + 24 <= width or not isinstance(value, (dict, list)) or not value:
        return flat
    inner = indent + "    "
    if isinstance(value, dict):
        items = [
            f"{inner}{_py_literal(k)}: {_py_literal_block(v, inner, width)},"
            for k, v in value.items()
        ]
        return "{\n" + "\n".join(items) + f"\n{indent}}}"
    items = [f"{inner}{_py_literal_block(v, inner, width)}," for v in value]
    return "[\n" + "\n".join(items) + f"\n{indent}]"


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
    object schemas get synthesized names. Each root schema resolves its
    own definitions; generated names and dependency-ordered blocks are
    shared across the module.
    """

    def __init__(self, *, reserved_names: tuple[str, ...] = ()) -> None:
        self._defs: dict[str, dict[str, Any]] = {}
        self._blocks: list[str] = []
        self._emitted: dict[str, str] = {}  # $def name -> python name
        self._in_progress: set[str] = set()
        self._used_names: set[str] = {
            *keyword.kwlist,
            *reserved_names,
            "Any",
            "Literal",
            "BaseModel",
            "ConfigDict",
            "Field",
            "PROMPT",
            "MODEL",
            "TEMPERATURE",
        }
        self.uses_literal = False
        self.uses_config_dict = False
        self.uses_field = False

    # ── public API ──

    def emit_root(self, class_name: str, schema: dict[str, Any]) -> None:
        if schema.get("type") != "object":
            raise ValueError(
                f"signature emission: root signature schema for {class_name} must be "
                f"an object schema, got type={schema.get('type')!r}"
            )
        # A #/$defs reference belongs to its root document. Sharing this
        # cache across input/output can bind even identical wrappers to the
        # wrong nested definition. Only Python symbol allocation is global.
        self._defs = schema.get("$defs", {})
        self._emitted.clear()
        self._in_progress.clear()
        self._used_names.add(class_name)
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
        if len(py_str_literal(current)) >= width:
            chunks.append(current)
            current = ""
    if current or not chunks:
        chunks.append(current)
    return chunks


def _chunked_str_literal(text: str, indent: str = "    ") -> str:
    """Render a long string as a parenthesized adjacent-literal expression."""
    chunks = _str_chunks(text, indent)
    if len(chunks) == 1:
        return py_str_literal(chunks[0])
    body = "\n".join(f"{indent}{py_str_literal(c)}" for c in chunks)
    outer = indent[:-4] if len(indent) >= 4 else ""
    return f"(\n{body}\n{outer})"


def _chunked_call_arg(text: str, indent: str) -> str:
    """Render a long string as adjacent literals for direct use as a call arg.

    Unlike :func:`_chunked_str_literal` this emits no wrapping parentheses
    (the call's own parentheses group the literals), which keeps ruff's
    UP034 (extraneous parentheses) quiet.
    """
    chunks = _str_chunks(text, indent)
    return "\n".join(f"{indent}{py_str_literal(c)}" for c in chunks)


# ───────── vendor SDK requirements ─────────

#: The Anthropic SDK every emitted Python app needs for its judges and
#: agent loops. The floor is load-bearing: judges send
#: ``output_config.format`` (structured outputs), and an older SDK fails
#: every call with ``unexpected keyword argument 'output_config'``. 1.15.0
#: has it (verified by installing it) and is the version uv.lock pins.
ANTHROPIC_PY_REQUIREMENT = "anthropic>=1.15"

#: The OpenAI SDK for ``client: openai`` judges (``response_format``
#: json_schema).
OPENAI_PY_REQUIREMENT = "openai>=1.40"


def vendor_sdk_requirements(pipeline: Pipeline) -> list[str]:
    """The vendor SDK requirement lines an emitted Python app needs.

    One definition for every Python runtime's requirements.txt and README.
    An agent loop runs on the Anthropic SDK's own tool runner when it is
    not on the subscription CLI, so it needs the SDK even with no judges.
    """
    clients = {
        n.signature_spec.client
        for n in pipeline.nodes_by_kind(NodeKind.LLM_JUDGE)
        if n.signature_spec is not None
    }
    lines = []
    if "anthropic" in clients or pipeline.nodes_by_kind(NodeKind.AGENT_LOOP):
        lines.append(ANTHROPIC_PY_REQUIREMENT)
    if "openai" in clients:
        lines.append(OPENAI_PY_REQUIREMENT)
    return lines


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
       output (``output_config.format`` / OpenAI json_schema), validates the
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
    if spec.client == "anthropic":
        # Raises at emit time, naming the node and the schema path, for an
        # output schema Anthropic structured outputs cannot express. The
        # same call runs again per request inside the emitted helper.
        structured_output_schema(spec.output_schema, node_id=node.id)

    converter = _SchemaToPydantic(reserved_names=(pascal, f"{pascal}Input", f"{pascal}Output"))
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
    right default: schema-locked decoding already pins the output shape.
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
    # The same per-node endpoint override judges and the TS loops honor.
    lines.append(f'{inner}base_url=os.environ.get("ROTE_BASE_URL_{node.id.upper()}"),')
    contract = agent_loop_output(node)
    if contract.schema is not None:
        # The declared output, which downstream nodes index: the loop must
        # hand back exactly these keys (rote.adapters._common).
        lines.append(f"{inner}output_schema={_py_literal_block(contract.schema, inner)},")
    if contract.unwrap_key is not None:
        lines.append(f"{inner}unwrap_key={json.dumps(contract.unwrap_key)},")
    if contract.strict:
        lines.append(f"{inner}output_strict=True,")
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


#: Modules rote itself writes into extracted/; never sourced from the agent.
_ROTE_OWNED_MODULES = frozenset({"__init__", "_rote_mcp"})


def helper_module_names(source_dir: Path | None, node_modules: set[str]) -> list[str]:
    """Modules beside pipeline.yaml that no node names but node code imports.

    A shared helper the agent wrote (``extracted/_common.py``), or the rule
    tables System-1 generates, is imported by impl modules without being an
    impl itself. Emitting only the impl modules shipped code that crashed on
    its first import (#150 fidelity run), so every module in the source's
    ``extracted/`` ships, verbatim.
    """
    if source_dir is None or not (Path(source_dir) / "extracted").is_dir():
        return []
    # Top level only, two ways: this glob, and resolve_extracted_source,
    # which reads extracted/<name>.py and nothing deeper. Widening the glob
    # alone is an equivalent mutant (a subdirectory or .txt name resolves to
    # no top-level file and ships nothing), confirmed by
    # tests/test_emit_helper_modules.py::test_only_top_level_python_modules_ship.
    return sorted(
        p.stem
        for p in (Path(source_dir) / "extracted").glob("*.py")
        if p.stem not in node_modules and p.stem not in _ROTE_OWNED_MODULES
    )


def write_helper_modules(
    writer: EmitWriter, source_dir: Path | None, names: list[str]
) -> dict[str, Path]:
    written: dict[str, Path] = {}
    for name in names:
        content = resolve_extracted_source(source_dir, name)
        if content is not None:
            written[f"extracted/{name}"] = writer.write("extracted", f"{name}.py", content=content)
    return written


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
        # Every node that calls this function, not just the first: shared
        # impls exist precisely because nodes differ only in constants.
        sharers = [n for n in nodes if _extracted_func_name(n) == func_name]
        if any(n.constants for n in sharers):
            doc_lines.append("")
            doc_lines.append("    Constants from the IR (lifted from the source skill), passed")
            doc_lines.append("    as keyword arguments alongside the node's inputs:")
            for sharer in sharers:
                indent = "      "
                if len(sharers) > 1:
                    doc_lines.append(f"      from node {sharer.id}:")
                    indent = "        "
                    if not sharer.constants:
                        doc_lines.append(f"{indent}(none)")
                for k, v in (sharer.constants or {}).items():
                    # k is an identifier (IR-validated); v is arbitrary, so escape
                    # its repr so a value containing \"\"\" can't close this docstring.
                    doc_lines.extend(
                        textwrap.wrap(
                            f"{k} = {safe_docstring_line(repr(v))}",
                            width=88,
                            initial_indent=indent,
                            subsequent_indent=indent + "    ",
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
            f'        "{STUB_SENTINEL} {module_name}.{func_name}: "\n'
            f'        "implement against the vendor API"\n'
            f"    )\n"
        )

    return "".join(parts)


def write_extracted_package(
    writer: EmitWriter,
    pipeline: Pipeline,
    *,
    source_dir: Path | None,
    generated_by: str,
    caller_note: str,
) -> dict[str, Path]:
    """Write ``extracted/`` for an adapter whose steps import plain modules.

    Each module is the agent's real implementation from ``source_dir``
    when one exists (:func:`resolve_extracted_source`), otherwise an
    IR-derived stub. The ``extracted/<module>`` labels in the returned
    mapping are what ``rote emit``'s ``unimplemented_stubs`` scans, so an
    adapter that writes through here reports its TODO list for free.

    An ``agent_loop`` that declares tools also gets ``extracted/_rote_mcp.py``
    (the verbatim MCP helper): its tools are MCP tool names, and the
    inference helper resolves their endpoints through that module. Node
    code still never references MCP; only the agent does, and only for
    tools the IR declared.
    """
    written: dict[str, Path] = {}
    modules = _extracted_layout(pipeline)
    needs_mcp_helper = any(n.kind is NodeKind.AGENT_LOOP and n.tools for n in pipeline.nodes)
    helpers = helper_module_names(source_dir, set(modules))
    if not modules and not needs_mcp_helper and not helpers:
        return written

    written["extracted/__init__"] = writer.write(
        "extracted",
        "__init__.py",
        content=f'"""Extracted deterministic modules for {pipeline.name}."""\n',
    )
    for module_name, nodes in sorted(modules.items()):
        written[f"extracted/{module_name}"] = writer.write(
            "extracted",
            f"{module_name}.py",
            content=resolve_extracted_source(source_dir, module_name)
            or emit_extracted_module(
                module_name, nodes, generated_by=generated_by, caller_note=caller_note
            ),
        )
    written.update(write_helper_modules(writer, source_dir, helpers))
    if needs_mcp_helper:
        from rote.mcp import _runtime_helper as _mcp_helper

        written["extracted/_rote_mcp"] = writer.write(
            "extracted",
            "_rote_mcp.py",
            content=Path(_mcp_helper.__file__).read_text(encoding="utf-8"),
        )
    return written


class ConstantsNotAccepted(ValueError):
    """An emit refused because an impl cannot take its node's constants.

    Raised before any file is written (:func:`require_constant_parameters`),
    so the runtime dir is exactly as it was. A subclass of ``ValueError``
    so existing handlers still catch it; its own type is what lets ``rote
    compile`` say that nothing was written, which it cannot promise for
    any other emit failure.
    """


def require_constant_parameters(
    pipeline: Pipeline,
    writer: EmitWriter,
    source_dir: Path | None,
) -> None:
    """Refuse an emit whose impls cannot take their nodes' constants.

    Every step passes its node's constants as keyword arguments, so an
    ``extracted/`` function that neither names them nor takes ``**kwargs``
    raises TypeError on its first call, possibly after earlier steps have
    already had side effects. The modules checked are the ones the runtime
    dir will hold after this emit: an agent-written module sourced from
    beside pipeline.yaml, or a user-filled file the writer would preserve.
    A module with no source becomes a stub, which takes ``**payload`` and
    always passes, so only its preserved user edit (if any) is read. Call
    it before writing
    anything, so a refused emit leaves no new main.py behind.
    """
    from rote.contracts import constants_not_accepted

    landed: dict[str, str] = {}
    origins: dict[str, Path] = {}
    for module in _extracted_layout(pipeline):
        fresh = resolve_extracted_source(source_dir, module) or ""
        landed[module] = writer.landed_content("extracted", f"{module}.py", content=fresh)
        # The file to fix: the agent's module beside pipeline.yaml, unless
        # the writer would keep the user's own edit in the runtime dir.
        if landed[module] != fresh or source_dir is None:
            origins[module] = writer.output_dir / "extracted" / f"{module}.py"
        else:
            origins[module] = Path(source_dir) / "extracted" / f"{module}.py"
    problems = constants_not_accepted(pipeline, landed, origins)
    if problems:
        detail = "; ".join(problems)
        raise ConstantsNotAccepted(
            f"extracted/ functions do not accept their node's constants, which every step "
            f"passes as keyword arguments: {detail}. In each file named, add each constant "
            f"as a keyword parameter (or **kwargs), then re-emit"
        )


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
