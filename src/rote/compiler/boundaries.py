"""The keys a step reads across a boundary, against what the other side declares.

A generated module and its tests can agree with each other and still read a
key nothing sends. A real compile (#291) passed 38 generated tests while two
gate consumers indexed `opendock_appointment_completion` and
`datadog_metrics` where their gates declared `appointment_completion` and
`provided_datadog_metrics`, and a renderer required a pipeline-input field
the input contract did not declare. The tests built their fixtures with the
module's own keys, so nothing crossed the real boundary.

These checks read the deliverable statically, so they cost nothing and need
no sample:

- **A gate's result.** A person's answer has exactly the keys the gate
  declares (``output``). A step whose code reads a key it does not declare
  fails at that read on every run. (An ``inputs:`` reference to an
  undeclared key is already refused when the IR loads.)
- **The pipeline input.** A step that reads a pipeline-input field by a
  literal key in its code needs the field declared, so a caller learns of it
  from the contract and the entry check, not from the last step's error.
- **Another step's result.** A step that reads a key the value bound to it
  cannot hold (a judge's output, a fanned item of it, a result whose schema
  is closed) fails at that read on every run (#329).
- **A value nothing reads.** A judge field copied from its input
  (``passthrough``) exists only to carry that value on. When no later step
  reads it, the value was dropped on the way: #329's signed-in account id
  reached the judge's output and never the ticket-creation fan-out.

A key read in code counts only when it is a literal: ``value["key"]``,
``value.get("key")`` or ``value.pop("key")`` on the parameter bound to that
value, or the same reads inside a function of the same module that the
value is passed to, with the key as a literal argument
(``payload(value, "key")`` where ``payload`` does ``value[name]``). The
0.18.0 recompile of #291's skill read all three wrong gate keys that way, so
the checks saw none of them. A key built at run time is not seen.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from rote.contracts import ContractFinding
from rote.ir import (
    Node,
    NodeKind,
    Pipeline,
    fan_out_element_param,
    item_passthrough,
    parse_input_ref,
)

#: A step reads a gate key the gate does not declare.
GATE_KEY_UNDECLARED = "gate_key_undeclared"
#: A step's code reads a pipeline-input field the input contract does not declare.
INPUT_FIELD_UNDECLARED = "input_field_undeclared"
#: A step's code reads a key the value bound from another step cannot hold.
RESULT_KEY_UNDELIVERED = "result_key_undelivered"
#: A judge copies an input value into its output, and no later step reads it.
PASSTHROUGH_UNREAD = "passthrough_unread"


def boundary_findings(pipeline: Pipeline, root: Path) -> list[ContractFinding]:
    """Every read of a gate key or input field its producer does not declare."""
    findings: list[ContractFinding] = []
    by_id = {n.id: n for n in pipeline.nodes}
    declared_input = declared_input_fields(pipeline)
    # A loop-body node is called by its parent loop with arguments the model
    # chose, so its inputs describe rather than bind.
    in_loops = {sub for n in pipeline.nodes for sub in (n.loop_body or ())}
    for node in pipeline.nodes:
        if node.id in in_loops:
            continue
        found = _impl_function(node, root)
        fn, module = found if found is not None else (None, None)
        for param, source in sorted((node.inputs or {}).items()):
            ref = parse_input_ref(source)
            producer = by_id.get(ref.node_id) if ref.node_id is not None else None
            if producer is not None and producer.kind is NodeKind.HITL_GATE:
                keys = producer.output_keys
                if not keys or ref.field is not None or fn is None:
                    continue  # nothing declared, or a field the IR already checked
                for key in sorted(literal_keys_read(fn, param, module) - keys):
                    findings.append(_gate_finding(node, producer, param, key, keys))
            elif ref.node_id is None and ref.field is None and fn is not None:
                for key in sorted(literal_keys_read(fn, param, module) - declared_input):
                    findings.append(_input_finding(node, param, key, declared_input))
            elif producer is not None and fn is not None:
                delivered = delivered_keys(pipeline, node, param)
                if delivered is None:
                    continue
                for key in sorted(literal_keys_read(fn, param, module) - delivered):
                    findings.append(_result_finding(pipeline, node, param, key, delivered))
    return findings + passthrough_unread_findings(pipeline, root)


def delivered_keys(pipeline: Pipeline, node: Node, param: str) -> frozenset[str] | None:
    """The keys the value bound to ``node``'s ``param`` can hold, when its schema
    says so exactly, else None.

    Exactly means a closed object: a judge's output (structured outputs
    close every object, so a judge never answers a key its schema lacks), or
    a result whose schema sets ``additionalProperties: false``. A key a module
    reads beyond an open schema may well be there, and a false error makes
    the agent rewrite correct code, so those are not checked.
    """
    ref = parse_input_ref((node.inputs or {})[param])
    producer = pipeline.node_by_id(ref.node_id) if ref.node_id is not None else None
    if producer is None or producer.kind is NodeKind.HITL_GATE:
        return None
    judge = producer.kind is NodeKind.LLM_JUDGE
    spec = producer.signature_spec
    root = (spec.output_schema if judge and spec is not None else producer.output_schema) or {}
    if not root:
        return None
    schema: dict[str, Any] | None = root
    if ref.field is None and producer.fan_out:
        schema = {"type": "array", "items": root}
    if ref.field is not None:
        schema = _properties(root, root).get(ref.field)
    if schema is not None and node.fan_out and param == fan_out_element_param(node, pipeline):
        schema = _resolve(schema, root).get("items")
    if schema is None:
        return None
    schema = _resolve(schema, root)
    props = _properties(schema, root)
    if not props or schema.get("anyOf") or schema.get("oneOf"):
        return None
    extra = schema.get("additionalProperties")
    if extra not in (None, False) or (extra is None and not judge):
        return None
    return frozenset(props)


def result_keys(node: Node) -> frozenset[str]:
    """The top-level keys ``node``'s result declares, a judge's schema included."""
    spec = node.signature_spec
    judged = (
        _properties(spec.output_schema, spec.output_schema)
        if node.kind is NodeKind.LLM_JUDGE and spec is not None
        else {}
    )
    return node.output_keys | frozenset(judged)


def _resolve(schema: Any, root: dict[str, Any]) -> dict[str, Any]:
    """``schema`` with a local ``$ref`` (``#/$defs/X``) followed, a few deep."""
    for _ in range(8):
        ref = schema.get("$ref") if isinstance(schema, dict) else None
        if not isinstance(ref, str) or not ref.startswith("#/"):
            break
        target: Any = root
        for part in ref[2:].split("/"):
            target = target.get(part) if isinstance(target, dict) else None
        schema = target
    return schema if isinstance(schema, dict) else {}


def _properties(schema: Any, root: dict[str, Any]) -> dict[str, Any]:
    props = _resolve(schema, root).get("properties")
    return props if isinstance(props, dict) else {}


def passthrough_unread_findings(pipeline: Pipeline, root: Path) -> list[ContractFinding]:
    """Each judge field copied from the input that no later step reads.

    A copied field exists only to carry its value on. A step reads it when
    it binds ``<judge>.output.<field>``, or binds the whole output and either
    its code reads the field or it has no code rote can read (a judge, a gate,
    an agent loop, an MCP call or a stub take the whole value). The run's
    result counts as a reader. A per-item copy (``items[].f``) rides inside
    its list, which is read with it.
    """
    exits = set(pipeline.exit_nodes or ())
    findings: list[ContractFinding] = []
    for judge in pipeline.nodes:
        spec = judge.signature_spec
        if judge.kind is not NodeKind.LLM_JUDGE or spec is None or not spec.passthrough:
            continue
        if judge.id in exits:
            continue
        fields = sorted(
            t for t, src in spec.passthrough.items() if item_passthrough(t, src) is None
        )
        unread = set(fields)
        siblings: set[str] = set()
        for reader in pipeline.nodes:
            found: tuple[_Function, ast.Module] | None = None
            looked = False
            for param, source in (reader.inputs or {}).items():
                ref = parse_input_ref(source)
                if ref.node_id != judge.id:
                    continue
                if ref.field is not None:
                    unread.discard(ref.field)
                    siblings.add(reader.id)
                    continue
                if not looked:
                    found, looked = _impl_function(reader, root), True
                if found is None:
                    unread.clear()
                    continue
                fn, module = found
                read = literal_keys_read(fn, param, module)
                unread -= read
                if read:
                    siblings.add(reader.id)
        for name in sorted(unread):
            source = spec.passthrough[name]
            readers = ", ".join(f"`{r}`" for r in sorted(siblings)) or "no step"
            origin = _origin(judge, source)
            also = f", or `{origin}` where it came from" if origin else ""
            findings.append(
                ContractFinding(
                    node_id=judge.id,
                    kind=PASSTHROUGH_UNREAD,
                    severity="warning",
                    message=(
                        f"{judge.id} copies `{source}` into its output as `{name}`, and no "
                        f"later step reads `{judge.id}.output.{name}`, so the value is "
                        f"dropped there. {readers} read(s) {judge.id}'s other fields. If a "
                        f"step needs this value (an id, an account, a target the skill "
                        f"names), bind it to that step: `{judge.id}.output.{name}`{also}. "
                        f"Never let "
                        f"the step supply a default or another value in its place. If no "
                        f"step needs it, drop the copy"
                    ),
                    path=None,
                )
            )
    return findings


def declared_input_fields(pipeline: Pipeline) -> set[str]:
    """The pipeline-input fields the contract names: required, optional, or a
    property of its schema."""
    spec = pipeline.input
    fields = set(spec.required) | set(spec.optional)
    schema = spec.input_schema or {}
    fields |= set((schema.get("properties") or {}).keys())
    return fields


_Function = ast.FunctionDef | ast.AsyncFunctionDef

#: How many helper calls deep a read is followed. A helper that calls a
#: helper is common (``payload`` calling ``require``); deeper is not, and the
#: limit keeps a recursive helper from looping.
_HELPER_DEPTH = 3


def literal_keys_read(fn: _Function, name: str, module: ast.Module | None = None) -> set[str]:
    """The literal string keys ``fn`` reads from its parameter ``name``:
    ``name["k"]``, ``name.get("k", ...)``, ``name.pop("k", ...)``, and, given
    the ``module`` ``fn`` lives in, the same reads made by a top-level
    function of that module that ``name`` is passed to (``h(name, "k")``
    where ``h`` reads ``value[key]``). Nothing when ``name`` is not a
    parameter, or is reassigned in the body (it then holds something else)."""
    helpers = _module_functions(module) if module is not None else {}
    return _keys_read(fn, name, {}, helpers, _HELPER_DEPTH)


def _keys_read(
    fn: _Function,
    name: str,
    bound: dict[str, str],
    helpers: dict[str, _Function],
    depth: int,
) -> set[str]:
    """Keys ``fn`` reads from ``name``. ``bound`` maps ``fn``'s parameters
    that a caller passed a literal string to that string, so ``name[key]``
    with ``key`` one of them is a read of the literal."""
    params = _params(fn)
    if name not in params or _reassigned(fn, name):
        return set()
    # A helper name the function binds itself (a parameter, a local def or
    # import) is not the module's function at its calls.
    shadowed = set(params) | _bound_names(fn.body)
    helpers = {h: f for h, f in helpers.items() if h not in shadowed}
    literal_params = {p: v for p, v in bound.items() if p in params and not _reassigned(fn, p)}
    keys: set[str] = set()
    for sub in ast.walk(fn):
        if isinstance(sub, ast.Subscript) and _is_name(sub.value, name):
            if isinstance(sub.ctx, ast.Load):
                key = _literal(sub.slice, literal_params)
                if key is not None:
                    keys.add(key)
        elif not isinstance(sub, ast.Call):
            continue
        elif (
            isinstance(sub.func, ast.Attribute)
            and sub.func.attr in ("get", "pop")
            and _is_name(sub.func.value, name)
            and sub.args
        ):
            key = _literal(sub.args[0], literal_params)
            if key is not None:
                keys.add(key)
        elif depth > 0 and isinstance(sub.func, ast.Name) and sub.func.id in helpers:
            helper = helpers[sub.func.id]
            passed = _bind_call(helper, sub)
            for param, arg in passed.items():
                if not _is_name(arg, name):
                    continue
                literals = {
                    p: v
                    for p, a in passed.items()
                    if (v := _literal(a, literal_params)) is not None
                }
                keys |= _keys_read(helper, param, literals, helpers, depth - 1)
    return keys


def _literal(node: ast.AST, literal_params: dict[str, str]) -> str | None:
    """A literal string, or a parameter a caller bound to one."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return literal_params.get(node.id)
    return None


def _params(fn: _Function) -> list[str]:
    return [a.arg for a in [*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs]]


def _bind_call(fn: _Function, call: ast.Call) -> dict[str, ast.AST]:
    """Which argument expression each of ``fn``'s parameters gets at ``call``.
    A starred argument or ``**`` mapping makes the positions unknowable, so
    nothing is bound."""
    if any(isinstance(a, ast.Starred) for a in call.args) or any(
        k.arg is None for k in call.keywords
    ):
        return {}
    positional = [a.arg for a in [*fn.args.posonlyargs, *fn.args.args]]
    passed: dict[str, ast.AST] = dict(zip(positional, call.args, strict=False))
    for keyword in call.keywords:
        assert keyword.arg is not None
        passed[keyword.arg] = keyword.value
    return passed


def _module_functions(module: ast.Module) -> dict[str, _Function]:
    """The module's top-level functions by name. A name the module binds any
    other way too (defined twice, imported, assigned) is left out: which one
    a call reaches depends on order, not on reading it."""
    defs: dict[str, list[_Function]] = {}
    others: list[ast.stmt] = []
    for stmt in module.body:
        if isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef):
            defs.setdefault(stmt.name, []).append(stmt)
        else:
            others.append(stmt)
    rebound = _bound_names(others)
    return {
        name: found[0] for name, found in defs.items() if len(found) == 1 and name not in rebound
    }


def _is_name(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Name) and node.id == name


def _reassigned(fn: _Function, name: str) -> bool:
    """Whether ``name`` is bound anywhere in ``fn``'s body, so a read of it
    may not be the parameter: an assignment, a loop or comprehension target,
    a ``with``/``except``/``match`` capture, an import, a nested def or class,
    or a lambda's or nested def's own parameter. ``name["k"] = v`` and
    ``name.a = v`` change what ``name`` holds, not which object it is."""
    return name in _bound_names(fn.body)


def _bound_names(statements: list[ast.stmt]) -> set[str]:
    """Every name ``statements`` bind, at any depth."""
    names: set[str] = set()
    for stmt in statements:
        for sub in ast.walk(stmt):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store | ast.Del):
                names.add(sub.id)
            elif isinstance(sub, ast.arg):
                names.add(sub.arg)
            elif isinstance(sub, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                names.add(sub.name)
            elif isinstance(sub, ast.ExceptHandler):
                names.update([sub.name] if sub.name else [])
            elif isinstance(sub, ast.alias):
                names.add((sub.asname or sub.name).split(".")[0])
            elif isinstance(sub, ast.MatchAs | ast.MatchStar) and sub.name:
                names.add(sub.name)
            elif isinstance(sub, ast.MatchMapping) and sub.rest:
                names.add(sub.rest)
            elif isinstance(sub, ast.Global | ast.Nonlocal):
                names.update(sub.names)
    return names


def _impl_function(node: Node, root: Path) -> tuple[_Function, ast.Module] | None:
    """The top-level function ``node.impl`` names and its module, read from the deliverable.
    None for a node with no impl, or a module that is missing or unreadable
    (other checks report those). An MCP-bound call's step calls its tool, not
    this function, so its code reads nothing at run time."""
    if node.kind not in (NodeKind.PURE_FUNCTION, NodeKind.EXTERNAL_CALL) or not node.impl:
        return None
    if node.mcp is not None:
        return None
    rel, _, symbol = node.impl.partition(":")
    try:
        tree = ast.parse((root / rel).read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None
    for stmt in tree.body:
        if isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef) and stmt.name == symbol:
            return stmt, tree
    return None


def _gate_finding(
    node: Node, gate: Node, param: str, key: str, keys: frozenset[str]
) -> ContractFinding:
    declared = ", ".join(f"`{k}`" for k in sorted(keys))
    return ContractFinding(
        node_id=node.id,
        kind=GATE_KEY_UNDECLARED,
        severity="error",
        message=(
            f"{node.id} reads `{key}` from gate {gate.id}'s answer (its code reads "
            f"`{param}[...]`), but the "
            f"gate declares only {declared}, so the person's answer never has it: that "
            f"read fails, or gets nothing, on every run. Read the key the gate declares "
            f"and make the tests build the answer from the gate's declared keys. (Renaming "
            f"the gate's key "
            f"instead means changing the gate and every reader; on a System-1 compile the "
            f"gate's keys are decided, so read the declared one.)"
        ),
        path=node.impl,
    )


def _origin(judge: Node, source: str) -> str | None:
    """The reference a judge's input path ``source`` was bound from, when it is
    one step's output field (``me.account_id`` bound ``me: get_identity.output``
    is ``get_identity.output.account_id``)."""
    head, _, rest = source.partition(".")
    bound = (judge.inputs or {}).get(head)
    if bound is None:
        return None
    ref = parse_input_ref(bound)
    if ref.node_id is None:
        return None
    if ref.field is None:
        return f"{ref.node_id}.output.{rest}" if rest and "." not in rest else None
    return bound if not rest else None


def _result_finding(
    pipeline: Pipeline, node: Node, param: str, key: str, delivered: frozenset[str]
) -> ContractFinding:
    source = (node.inputs or {})[param]
    listed = ", ".join(f"`{k}`" for k in sorted(delivered))
    # Where else the key is: another step's output the node could bind.
    elsewhere = sorted(
        f"`{n.id}.output.{key}`"
        for n in pipeline.nodes
        if n.id != node.id and key in result_keys(n)
    )
    hint = (
        f" {' and '.join(elsewhere)} {'holds' if len(elsewhere) == 1 else 'hold'} it: if that "
        f"is the value meant, the step needs that binding"
        if elsewhere
        else ""
    )
    return ContractFinding(
        node_id=node.id,
        kind=RESULT_KEY_UNDELIVERED,
        severity="error",
        message=(
            f"{node.id}'s code reads `{key}` from `{param}` (bound to `{source}`), but that "
            f"value can hold only {listed}, so the read fails on every run.{hint}. Read a "
            f"key the value holds, and build the tests from the producer's schema. Never "
            f"replace the missing value with a default"
        ),
        path=node.impl,
    )


def _input_finding(node: Node, param: str, key: str, declared: set[str]) -> ContractFinding:
    listed = ", ".join(f"`{k}`" for k in sorted(declared)) or "none"
    return ContractFinding(
        node_id=node.id,
        kind=INPUT_FIELD_UNDECLARED,
        severity="error",
        message=(
            f"{node.id}'s code reads `{key}` from the pipeline input (`{param}`), but the "
            f"input contract declares no such field (it declares {listed}), so a caller "
            f"cannot know to send it. Declare `{key}` in the pipeline's input (required "
            f"if the step cannot run without it), or stop reading it. On a System-1 "
            f"compile the input schema is decided: add the field as an entry in "
            f"`system1/input_fields.yaml` (name, type, required, description, and the "
            f"skill's own words that ask for it), which rote checks and adds"
        ),
        path=node.impl,
    )
