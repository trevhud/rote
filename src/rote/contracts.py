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
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rote.ir import Node, NodeKind, Pipeline

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
                    try:
                        parsed[path] = ast.parse(path.read_text(encoding="utf-8"))
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

    # Signature checking only applies to a generated `**payload` call site.
    # Judges are instantiated and driven through forward(); loop-body
    # sub-nodes are called by hand-written loop code.
    if not check_signature or node.kind is NodeKind.LLM_JUDGE or node.id in loop_body_ids:
        return []
    if not isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef):
        return []

    payload_keys = set(node.inputs or {})
    return [
        ContractFinding(node_id=node.id, kind=kind, severity="error", message=message, path=rel)
        for kind, message in _check_call_signature(stmt, payload_keys)
    ]
