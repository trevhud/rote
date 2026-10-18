"""Static contract checks between the IR and the artifacts the agent wrote.

``impl: extracted/hubspot.py:batch_upsert_contacts`` is a promise the IR
makes and nothing verified. Adapters emit ``batch_upsert_contacts(**payload)``
and the first time anyone learns the symbol is missing — or that it takes
``contact_list`` when the node binds ``contacts`` — is a ``NameError`` or
``TypeError`` at 3am in a background job, long after the compile run that
introduced it exited 0.

Everything here is decidable without running anything:

* the module the IR points at parses, and defines the symbol
* the symbol's parameters can actually receive ``**payload``, where the
  payload keys are the node's own ``inputs``
* a judge's prompt template only references fields its input schema
  declares

That last one matters more than it looks. The emitted ``_interpolate``
raises on an unresolvable ``{{ path }}`` — deliberately, because a hole in
a prompt produces confident garbage — but it raises *at judge-call time*,
in production, on a pipeline that compiled and deployed cleanly.

**AST only, never import.** These modules are LLM-written, may import
vendor SDKs that aren't installed here, and may run arbitrary code at
import time. Parsing answers every question above without any of that.

**Absent is not wrong.** A node whose ``extracted/<module>.py`` doesn't
exist is the documented stub path — emission synthesises a stub with the
right symbol. Only files the agent actually wrote are checked, which is
also what keeps this quiet enough to be worth reading.
"""

from __future__ import annotations

import ast
import io
import re
import sys
import tokenize
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from rote.ir import CONSTANT_ARGUMENT_KINDS, Node, NodeKind, Pipeline

#: The oldest Python rote supports (``requires-python``). Emitted modules
#: run wherever rote does, so an agent-written module must parse there.
OLDEST_SUPPORTED_PYTHON = (3, 11)

#: ``{{ dotted.path }}`` — must stay in step with the ``_interpolate``
#: helper emitted by ``_py_common`` / ``_ts_common``.
_PROMPT_VAR_RE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_.]*)\s*\}\}")


@dataclass(frozen=True)
class ContractFinding:
    """One violated promise between the IR and an artifact on disk."""

    node_id: str
    kind: str
    severity: str
    message: str
    path: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        return {
            "node": self.node_id,
            "kind": self.kind,
            "severity": self.severity,
            "message": self.message,
            "path": self.path,
        }


@dataclass
class _FString:
    """An f-string being tokenized: its closing quote and brace depth."""

    quote: str
    depth: int = 0


def newer_syntax_problems(source: str) -> list[tuple[int, str]]:
    """``(line, message)`` for syntax that parses here but not on Python 3.11.

    A real compile wrote ``f"{'\\n'.join(lines)}"``: valid on 3.12+, a
    SyntaxError on 3.11, so a compile run on a newer interpreter shipped
    a module that could not even be imported where rote is supported.
    ``ast.parse(feature_version=...)`` does not catch it, because PEP 701
    changed the *tokenizer* (f-strings became token streams) and
    ``feature_version`` only gates grammar rules. So two checks:

    * the grammar, through ``feature_version`` (``type`` aliases, PEP 695
      generics);
    * the PEP 701 relaxations, by walking the 3.12+ token stream: inside
      a replacement field, 3.11 allows no backslash, no comment, no
      newline under a single-quoted f-string, and no reuse of an
      enclosing f-string's quote.

    On 3.11 itself the interpreter's own parse is authoritative and the
    caller has already done it, so this returns nothing there.
    """
    if sys.version_info[:2] <= OLDEST_SUPPORTED_PYTHON:
        return []
    problems: list[tuple[int, str]] = []
    try:
        ast.parse(source, feature_version=OLDEST_SUPPORTED_PYTHON)
    except SyntaxError as exc:
        problems.append((exc.lineno or 0, exc.msg))

    stack: list[_FString] = []
    seen: set[tuple[int, str]] = set()

    def report(line: int, message: str) -> None:
        if (line, message) not in seen:
            seen.add((line, message))
            problems.append((line, message))

    # Token types that only exist on 3.12+ (3.14+ for t-strings); mypy checks
    # against 3.11, where the module has no such attributes.
    fstring_start: int = getattr(tokenize, "FSTRING_START")  # noqa: B009
    fstring_end: int = getattr(tokenize, "FSTRING_END")  # noqa: B009
    tstring_start = getattr(tokenize, "TSTRING_START", None)
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        line = tok.start[0]
        if tstring_start is not None and tok.type == tstring_start:
            report(line, "template strings (t-strings) need Python 3.14")
            continue
        if tok.type == fstring_start:
            # On 3.11 the outer f-string is ONE string token, so the
            # nested one may not contain the outer's closing quote.
            if any(outer.quote in tok.string for outer in stack):
                report(line, "reusing the enclosing quote inside an f-string needs Python 3.12")
            stack.append(_FString(quote=tok.string.lstrip("rRfFbBuU")))
            continue
        if tok.type == fstring_end:
            stack.pop()
            continue
        if not stack:
            continue
        top = stack[-1]
        if tok.type == tokenize.OP and tok.string in "{}" and len(tok.string) == 1:
            top.depth += 1 if tok.string == "{" else -1
            continue
        # Every f-string whose replacement field this token sits in. The
        # innermost one only counts when the token is inside ITS field;
        # its own literal text (FSTRING_MIDDLE at depth 0) is fine.
        enclosing = stack[:-1] + ([top] if top.depth > 0 else [])
        if not enclosing:
            continue
        if "\\" in tok.string:
            report(line, "a backslash inside an f-string's {expression} needs Python 3.12")
        if tok.type == tokenize.COMMENT:
            report(line, "a comment inside an f-string's {expression} needs Python 3.12")
        if tok.type == tokenize.NL and any(len(f.quote) == 1 for f in enclosing):
            report(
                line,
                "a line break inside a single-quoted f-string's {expression} needs Python 3.12",
            )
        if any(f.quote in tok.string for f in enclosing):
            report(line, "reusing the enclosing quote inside an f-string needs Python 3.12")
    return problems


def _split_ref(ref: str) -> tuple[str, str]:
    """``"extracted/hubspot.py:upsert"`` → ``("extracted/hubspot.py", "upsert")``."""
    path, _, symbol = ref.partition(":")
    return path, symbol


def _toplevel_defs(tree: ast.Module) -> dict[str, ast.stmt]:
    """Module-level names, by the statement that binds them.

    Assignments count: a node may legitimately point at a
    ``functools.partial`` or a module-level callable object rather than a
    ``def``. Those get the existence check but not the signature check.
    """
    defs: dict[str, ast.stmt] = {}
    for stmt in tree.body:
        if isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            defs[stmt.name] = stmt
        elif isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    defs[target.id] = stmt
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            defs[stmt.target.id] = stmt
    return defs


def _check_call_signature(
    fn: ast.FunctionDef | ast.AsyncFunctionDef, payload_keys: set[str]
) -> list[tuple[str, str]]:
    """Can ``fn(**payload)`` succeed? Returns ``(kind, message)`` pairs.

    Adapters always call through ``**payload``, so this is pure keyword
    dispatch. Three rules, each verified against the interpreter by
    ``test_contracts.py``'s differential test rather than reasoned about:

    * A **required** positional-only parameter can never be filled.
    * A payload key *naming* a positional-only parameter is rejected —
      even when that parameter has a default — UNLESS the function also
      declares ``**kwargs``, in which case the key lands there and the
      parameter quietly keeps its default.
    * Everything else is ordinary: required parameters must be present,
      and unknown keys need a ``**kwargs`` to land in.
    """
    findings: list[tuple[str, str]] = []
    a = fn.args

    # Defaults bind to the RIGHTMOST positional params (posonly + args).
    positional = list(a.posonlyargs) + list(a.args)
    cut = len(positional) - len(a.defaults)
    n_posonly = len(a.posonlyargs)

    required_posonly = {p.arg for i, p in enumerate(positional[:cut]) if i < n_posonly}
    required_named = {p.arg for i, p in enumerate(positional[:cut]) if i >= n_posonly}
    # kw_defaults is index-aligned with kwonlyargs; None means required.
    required_named |= {p.arg for p, d in zip(a.kwonlyargs, a.kw_defaults, strict=True) if d is None}

    posonly_names = {p.arg for p in a.posonlyargs}
    # Only `args` and `kwonlyargs` are reachable by keyword.
    accepted = {p.arg for p in a.args} | {p.arg for p in a.kwonlyargs}

    # A `**kwargs` gives a posonly-named key somewhere to land, so only the
    # unfillable-required case survives it.
    bad_posonly = required_posonly | ((posonly_names & payload_keys) if a.kwarg is None else set())
    if bad_posonly:
        findings.append(
            (
                "positional_only_param",
                f"{fn.name}() declares positional-only parameter(s) "
                f"[{', '.join(sorted(bad_posonly))}], which a `**payload` call "
                f"can never fill — remove the `/` marker",
            )
        )

    missing = required_named - payload_keys
    if missing:
        findings.append(
            (
                "missing_argument",
                f"{fn.name}() requires parameter(s) [{', '.join(sorted(missing))}] that the "
                f"node's inputs do not provide (payload keys: "
                f"[{', '.join(sorted(payload_keys)) or 'none'}]) — the call raises TypeError",
            )
        )

    if a.kwarg is None:
        # posonly collisions are already reported above; don't double-count.
        unexpected = payload_keys - accepted - posonly_names
        if unexpected:
            findings.append(
                (
                    "unexpected_argument",
                    f"the node binds input(s) [{', '.join(sorted(unexpected))}] that "
                    f"{fn.name}() does not accept — the call raises TypeError",
                )
            )
    return findings


def _prompt_paths_in_schema(prompt: str, input_schema: dict[str, Any]) -> list[str]:
    """Template paths the input schema cannot satisfy.

    Walks only as far as the schema actually describes. A path that
    reaches a level with no ``properties`` (a free-form object) is
    accepted — the schema stopped making claims, so neither can we.
    """
    unresolved: list[str] = []
    for path in _PROMPT_VAR_RE.findall(prompt):
        node: Any = input_schema
        for part in path.split("."):
            props = node.get("properties") if isinstance(node, dict) else None
            if not isinstance(props, dict):
                break  # schema stops describing here; accept the rest
            if part not in props:
                unresolved.append(path)
                break
            node = props[part]
    return unresolved


def _module_for(node: Node) -> tuple[str, str] | None:
    """The ``(relative path, symbol)`` an emitted step will import, if any."""
    if node.kind in (NodeKind.PURE_FUNCTION, NodeKind.EXTERNAL_CALL):
        if node.impl is None:
            return None  # mcp-bound: the step is a bare tool call
        return _split_ref(node.impl)
    if node.kind is NodeKind.AGENT_LOOP:
        # _extracted_layout gives each agent_loop its own same-named module.
        return f"extracted/{node.id}.py", node.id
    if node.kind is NodeKind.LLM_JUDGE and node.signature:
        return _split_ref(node.signature)
    return None


def check_contracts(pipeline: Pipeline, artifact_dir: Path | str) -> list[ContractFinding]:
    """Check every IR promise against the artifacts actually on disk.

    ``artifact_dir`` is the directory holding ``pipeline.yaml`` — i.e. the
    compiler's output dir, the same root ``resolve_extracted_source``
    reads from.
    """
    root = Path(artifact_dir)
    findings: list[ContractFinding] = []
    parsed: dict[Path, ast.Module | None] = {}

    # Data-flow threading (`inputs:`) postdates the first IRs. In one of
    # those every emitted call is `func()`, so every impl taking a required
    # argument would report — a wall of findings saying one thing. Say the
    # one thing once, and skip per-node signature checks.
    dataflow_in_use = any(n.inputs for n in pipeline.nodes)

    # Sub-nodes of an agent_loop are invoked by hand-written loop code, not
    # dispatched from the DAG with a generated payload, so their `inputs`
    # say nothing about how they get called. Existence still applies.
    loop_body_ids: set[str] = set()
    for n in pipeline.nodes:
        if n.loop_body:
            loop_body_ids.update(n.loop_body)

    for node in pipeline.nodes:
        target = _module_for(node)
        if target is not None:
            rel, symbol = target
            path = root / rel
            if path.is_file():
                if path not in parsed:
                    source = path.read_text(encoding="utf-8")
                    try:
                        parsed[path] = ast.parse(source)
                        findings.extend(
                            ContractFinding(
                                node_id=node.id,
                                kind="needs_newer_python",
                                severity="error",
                                message=(
                                    f"{rel} line {line}: {message}, but rote supports "
                                    f"Python {'.'.join(map(str, OLDEST_SUPPORTED_PYTHON))}"
                                    f" — the module would not import there"
                                ),
                                path=rel,
                            )
                            for line, message in newer_syntax_problems(source)
                        )
                    except SyntaxError as exc:
                        parsed[path] = None
                        findings.append(
                            ContractFinding(
                                node_id=node.id,
                                kind="syntax_error",
                                severity="error",
                                message=f"{rel} does not parse: {exc.msg} (line {exc.lineno})",
                                path=rel,
                            )
                        )
                tree = parsed.get(path)
                if tree is not None:
                    findings.extend(
                        _check_symbol(
                            node,
                            tree,
                            rel,
                            symbol,
                            loop_body_ids,
                            check_signature=dataflow_in_use,
                        ),
                    )

        if node.kind is NodeKind.LLM_JUDGE and node.signature_spec is not None:
            spec = node.signature_spec
            for bad in _prompt_paths_in_schema(spec.prompt, spec.input_schema):
                findings.append(
                    ContractFinding(
                        node_id=node.id,
                        kind="unresolved_prompt_var",
                        severity="error",
                        message=(
                            f"prompt references {{{{ {bad} }}}} but the judge's input "
                            f"schema declares no such field — emitted judges raise on an "
                            f"unresolvable placeholder, so this fails on the first real call"
                        ),
                    )
                )

    if not dataflow_in_use and any(n.impl for n in pipeline.nodes):
        findings.append(
            ContractFinding(
                node_id="(pipeline)",
                kind="no_dataflow",
                severity="warning",
                message=(
                    "no node declares `inputs:`, so every emitted call is `func()` "
                    "with an empty payload — per-node signature checks skipped. This "
                    "IR predates data-flow threading; recompile to thread real payloads"
                ),
            )
        )

    return findings


def _check_symbol(
    node: Node,
    tree: ast.Module,
    rel: str,
    symbol: str,
    loop_body_ids: set[str],
    *,
    check_signature: bool,
) -> list[ContractFinding]:
    defs = _toplevel_defs(tree)
    stmt = defs.get(symbol)
    if stmt is None:
        return [
            ContractFinding(
                node_id=node.id,
                kind="missing_symbol",
                severity="error",
                message=(
                    f"{rel} defines no {symbol!r} "
                    f"(found: [{', '.join(sorted(defs)) or 'nothing'}]) — emitted code "
                    f"imports it by name"
                ),
                path=rel,
            )
        ]

    if not isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef):
        return []
    constants = set(node.constants or {}) if node.kind in CONSTANT_ARGUMENT_KINDS else set()
    # Every caller, loop-body tool calls included, gets the node's constants
    # merged into its keyword arguments, so this holds even where the rest
    # of the signature check does not apply.
    findings: list[ContractFinding] = []
    rejected = _rejected_constants(stmt, constants)
    if rejected:
        findings.append(
            ContractFinding(
                node_id=node.id,
                kind="constant_not_accepted",
                severity="error",
                message=(
                    f"{symbol}() does not accept the node's constant(s) "
                    f"[{', '.join(sorted(rejected))}], which every call passes as keyword "
                    f"arguments; the call raises TypeError"
                ),
                path=rel,
            )
        )

    # Signature checking only applies to a generated `**payload` call site.
    # Judges are instantiated and driven through forward(); loop-body
    # sub-nodes are called by hand-written loop code.
    if not check_signature or node.kind is NodeKind.LLM_JUDGE or node.id in loop_body_ids:
        return findings

    # Rejected constants are reported above, not again as unexpected inputs.
    payload_keys = set(node.inputs or {}) | (constants - rejected)
    return findings + [
        ContractFinding(node_id=node.id, kind=kind, severity="error", message=message, path=rel)
        for kind, message in _check_call_signature(stmt, payload_keys)
    ]


def _rejected_constants(
    fn: ast.FunctionDef | ast.AsyncFunctionDef, constants: set[str]
) -> set[str]:
    """Constants ``fn`` cannot receive by keyword (none, if it takes ``**kwargs``)."""
    if fn.args.kwarg is not None:
        return set()
    accepted = {p.arg for p in fn.args.args} | {p.arg for p in fn.args.kwonlyargs}
    return constants - accepted


def constants_not_accepted(
    pipeline: Pipeline,
    sources: Mapping[str, str],
    origins: Mapping[str, Path] | None = None,
) -> list[str]:
    """``"node X: <file> f() takes no a, b"`` for every impl that would reject its constants.

    ``sources`` maps an ``extracted/`` module name to the text an emitted
    Python runtime will import; ``origins`` maps it to the file that text
    came from, so the message names the file to fix. Every step merges its node's constants into
    the impl's keyword arguments, loop_body sub-nodes included, so this
    holds for every caller.
    """
    parsed: dict[str, ast.Module | None] = {}
    problems: list[str] = []
    for node in pipeline.nodes:
        if node.kind not in CONSTANT_ARGUMENT_KINDS or not node.constants or node.impl is None:
            continue
        rel, symbol = _split_ref(node.impl)
        module = Path(rel).stem
        if module not in sources:
            continue
        if module not in parsed:
            try:
                parsed[module] = ast.parse(sources[module])
            except SyntaxError:
                parsed[module] = None  # check_contracts reports the syntax error
        tree = parsed[module]
        stmt = _toplevel_defs(tree).get(symbol) if tree is not None else None
        if not isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        rejected = _rejected_constants(stmt, set(node.constants))
        if rejected:
            where = (
                str(origins[module])
                if origins and module in origins
                else (f"extracted/{module}.py")
            )
            problems.append(
                f"node {node.id!r}: {where} {symbol}() takes no {', '.join(sorted(rejected))}"
            )
    return problems


# ───────── Implemented vs stub, per node ─────────

ImplStatus = Literal["implemented", "stub", "absent"]


@dataclass(frozen=True)
class NodeImplementation:
    """Whether the function an impl-bearing node points at does real work.

    ``absent`` means the agent wrote no such module or function, so
    emission synthesises a stub. ``reason`` is a stub's
    ``NotImplementedError`` message: the rubric has the agent say there
    why it could not implement the node (an external system, a business
    fact the skill does not state).
    """

    node_id: str
    kind: str
    impl: str
    status: ImplStatus
    reason: str | None = None
    #: The default backend calls the MCP tool instead of this function, so
    #: a stub here does not stop a run.
    mcp_bound: bool = False

    def as_dict(self) -> dict[str, str | bool | None]:
        return {
            "node": self.node_id,
            "kind": self.kind,
            "impl": self.impl,
            "status": self.status,
            "reason": self.reason,
            "mcp_bound": self.mcp_bound,
        }


def _raise_names_not_implemented(stmt: ast.stmt) -> bool:
    if not isinstance(stmt, ast.Raise) or stmt.exc is None:
        return False
    exc = stmt.exc.func if isinstance(stmt.exc, ast.Call) else stmt.exc
    return isinstance(exc, ast.Name) and exc.id == "NotImplementedError"


def stub_reason(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[bool, str | None]:
    """``(is_stub, reason)`` for one function definition.

    A function is a stub when its body raises ``NotImplementedError`` at
    the top level: nothing after that line runs and nothing before it
    returns. A ``raise`` inside a branch (an unsupported variant of a
    real implementation) is real code, not a stub. The reason is the
    exception's message, rendered from source when it is not a plain
    string (an f-string naming a constant, say).
    """
    for stmt in fn.body:
        if not _raise_names_not_implemented(stmt):
            continue
        assert isinstance(stmt, ast.Raise)
        if not isinstance(stmt.exc, ast.Call) or not stmt.exc.args:
            return True, None
        arg = stmt.exc.args[0]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            text = arg.value
        else:
            text = ast.unparse(arg)
        return True, " ".join(text.split()) or None
    return False, None


def implementation_status(pipeline: Pipeline, artifact_dir: Path | str) -> list[NodeImplementation]:
    """One entry per ``pure_function`` / ``external_call`` node with an ``impl``.

    Read by AST, like everything in this module: the modules are never
    imported. This is the per-node answer to "will the compiled pipeline
    run without hand-written code", which a per-file "contains
    NotImplementedError somewhere" check cannot give once a module mixes
    real functions with stubs.
    """
    root = Path(artifact_dir)
    trees: dict[Path, ast.Module | None] = {}
    statuses: list[NodeImplementation] = []
    for node in pipeline.nodes:
        if node.kind not in (NodeKind.PURE_FUNCTION, NodeKind.EXTERNAL_CALL) or not node.impl:
            continue
        rel, symbol = _split_ref(node.impl)
        path = root / rel
        if path not in trees:
            try:
                trees[path] = ast.parse(path.read_text(encoding="utf-8"))
            except (OSError, SyntaxError):
                trees[path] = None
        tree = trees[path]
        stmt = _toplevel_defs(tree).get(symbol) if tree is not None else None
        status: ImplStatus
        reason: str | None = None
        if stmt is None:
            status = "absent"
        elif isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef):
            is_stub, reason = stub_reason(stmt)
            status = "stub" if is_stub else "implemented"
        else:
            # A partial or callable object: something real is bound there.
            status = "implemented"
        statuses.append(
            NodeImplementation(
                node_id=node.id,
                kind=node.kind.value,
                impl=node.impl,
                status=status,
                reason=reason,
                mcp_bound=node.mcp is not None,
            )
        )
    return statuses
