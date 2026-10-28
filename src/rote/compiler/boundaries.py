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

from rote.contracts import ContractFinding
from rote.ir import Node, NodeKind, Pipeline, parse_input_ref

#: A step reads a gate key the gate does not declare.
GATE_KEY_UNDECLARED = "gate_key_undeclared"
#: A step's code reads a pipeline-input field the input contract does not declare.
INPUT_FIELD_UNDECLARED = "input_field_undeclared"


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
    """The module's top-level functions by name. A name defined twice is left
    out: which definition runs depends on order, not on reading it."""
    found: dict[str, _Function] = {}
    seen: set[str] = set()
    for stmt in module.body:
        if isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef):
            if stmt.name in seen:
                found.pop(stmt.name, None)
            else:
                found[stmt.name] = stmt
            seen.add(stmt.name)
    return found


def _is_name(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Name) and node.id == name


def _reassigned(fn: _Function, name: str) -> bool:
    for sub in ast.walk(fn):
        targets: list[ast.AST] = []
        if isinstance(sub, ast.Assign):
            targets = list(sub.targets)
        elif isinstance(
            sub, ast.AnnAssign | ast.AugAssign | ast.For | ast.AsyncFor | ast.NamedExpr
        ):
            targets = [sub.target]
        if any(_binds(target, name) for target in targets):
            return True
    return False


def _binds(target: ast.AST, name: str) -> bool:
    """Whether an assignment target rebinds ``name``: the name itself, or a
    name inside a tuple or list target. ``name["k"] = v`` and ``name.a = v``
    change what ``name`` holds, not which object it is."""
    if isinstance(target, ast.Tuple | ast.List):
        return any(_binds(t, name) for t in target.elts)
    if isinstance(target, ast.Starred):
        return _binds(target.value, name)
    return _is_name(target, name)


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
            f"gate declares only {declared}, so the person's answer never has it and every "
            f"run fails at that read. Read the key the gate declares and make the tests "
            f"build the answer from the gate's declared keys. (Renaming the gate's key "
            f"instead means changing the gate and every reader; on a System-1 compile the "
            f"gate's keys are decided, so read the declared one.)"
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
