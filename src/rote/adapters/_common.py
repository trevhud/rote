"""Helpers shared by every runtime adapter.

These functions are runtime-agnostic: they operate purely on the
:class:`rote.ir.Pipeline` IR and on plain strings. Anything that encodes
a *runtime's* semantics (Temporal retry policies, Cloudflare step
configs, …) stays in the adapter that owns it.

Extracted from ``rote.adapters.temporal`` / ``rote.adapters.cloudflare``
once the second adapter proved the helpers were genuinely shared.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rote.ir import Node, NodeKind, Pipeline

# ───────── Default LLM models ─────────
#
# Single source of truth for the model ids adapters bake into emitted
# judges when the IR omits ``signature_spec.model``. Bump these when a
# new default ships — every adapter config inherits them.

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"
DEFAULT_OPENAI_MODEL = "gpt-4.1"

# ───────── Judge operator knobs ─────────
#
# One naming convention for every runtime, Python and TypeScript alike,
# so operators learn one set of knobs. The emitted code and the emitted
# README both derive the names from here.


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


def temperature_env_var(node_id: str) -> str:
    """The per-node ``temperature`` override env var name.

    Kept separate from :func:`override_env_vars` (whose two-tuple shape
    several emitters unpack) because it is only emitted for judges whose
    IR actually sets a temperature. Setting it to an empty string drops
    the parameter from the request entirely — current Anthropic models
    reject ``temperature`` with a 400, so an operator retargeting
    ``ROTE_MODEL_<ID>`` at a newer model needs a way out that does not
    require re-emitting.
    """
    return f"ROTE_TEMPERATURE_{node_id.upper()}"


def judge_override_readme(pipeline: Pipeline) -> str:
    """The emitted README's paragraph on judge credentials and env knobs.

    Derived from the pipeline's actual judges: each one with a
    ``signature_spec`` gets its real ``ROTE_MODEL_<ID>`` /
    ``ROTE_BASE_URL_<ID>`` names (plus ``ROTE_TEMPERATURE_<ID>`` when the
    IR sets a temperature). A pipeline with no judges says so instead of
    documenting knobs for a node that does not exist. Legacy
    ``signature: path:Class`` judges are user code and read no knobs.
    """
    judges = pipeline.nodes_by_kind(NodeKind.LLM_JUDGE)
    if not judges:
        return "This pipeline has no LLM-judge nodes, so it needs no model API keys.\n"
    lines = [
        "LLM judge steps call the vendor SDK directly and read the standard",
        "`ANTHROPIC_API_KEY` / `OPENAI_API_KEY` environment variables.",
    ]
    knobbed = [n for n in judges if n.signature_spec is not None]
    if not knobbed:
        return "\n".join(lines) + "\n"
    lines += [
        "Each judge also honors per-node overrides, so operators can change",
        "the model or point at an OpenAI-compatible endpoint (Ollama, vLLM,",
        "a gateway) without re-emitting:",
        "",
        "```sh",
    ]
    rows: list[tuple[str, str]] = []
    for node in knobbed:
        model_var, base_var = override_env_vars(node.id)
        rows.append((f"export {model_var}=...", f"`{node.id}`: model id"))
        rows.append((f"export {base_var}=...", f"`{node.id}`: custom endpoint"))
        assert node.signature_spec is not None
        if node.signature_spec.temperature is not None:
            rows.append(
                (
                    f"export {temperature_env_var(node.id)}=...",
                    f"`{node.id}`: empty value drops the parameter",
                )
            )
    width = max(len(cmd) for cmd, _ in rows)
    lines += [f"{cmd.ljust(width)}  # {note}" for cmd, note in rows]
    lines.append("```")
    return "\n".join(lines) + "\n"


# ───────── Prose → safe code comment/docstring ─────────


def mcp_tool_arguments(node: Node) -> list[tuple[str, str, Any]] | None:
    """How an MCP-bound step builds its tool arguments, on every runtime.

    ``None`` means the threaded payload is the arguments as-is: the node's
    inputs, never its constants, because a tool rejects arguments it does
    not declare (FastMCP raises "Unexpected keyword argument"). With
    ``mcp.args`` each entry is ``(tool_arg, "payload", key)`` or
    ``(tool_arg, "constant", value)``: a key that names one of the node's
    constants forwards that constant, which is the only way a constant
    reaches a tool. Language-neutral on purpose, so every runtime sends a
    tool the identical arguments.
    """
    assert node.mcp is not None
    if not node.mcp.args:
        return None
    constants = node.constants or {}
    return [
        (tool_arg, "constant", constants[key]) if key in constants else (tool_arg, "payload", key)
        for tool_arg, key in node.mcp.args.items()
    ]


def safe_docstring_line(text: str, fallback: str = "") -> str:
    """First line of ``text``, escaped so it cannot break out of a docstring.

    Node ``description`` is prose (it legitimately contains quotes and
    backslashes), yet adapters splice its first line straight into a
    ``\"\"\"…\"\"\"`` Python docstring / block comment. An unescaped
    ``\"\"\"`` — or a trailing backslash that escapes the closing quote —
    lets a crafted pipeline.yaml close the docstring early and inject
    code. Escaping backslashes and double-quotes neutralizes both without
    charset-restricting prose at the IR layer.
    """
    stripped = text.strip() if text else ""
    first = stripped.splitlines()[0] if stripped else fallback
    return first.replace("\\", "\\\\").replace('"', '\\"')


def safe_block_comment_line(text: str, fallback: str = "") -> str:
    """First line of ``text``, safe to splice into a ``/* … */`` block comment.

    The TypeScript adapter emits ``description`` first-lines into JSDoc
    comments. An unescaped ``*/`` would close the comment early and let a
    crafted pipeline.yaml inject code after it; neutralize the sequence by
    inserting a space (``*/`` → ``* /``). Newlines are already dropped by
    taking the first line.
    """
    stripped = text.strip() if text else ""
    first = stripped.splitlines()[0] if stripped else fallback
    return first.replace("*/", "* /")


# ───────── Output-path containment ─────────


def resolve_within(base: Path, *parts: str) -> Path:
    """Join ``parts`` onto ``base`` and assert the result stays inside it.

    Defense-in-depth for emitted-file paths. IR validation already pins
    ``node.id`` to an identifier, but this guard means any future field
    that reaches a filename cannot escape the output directory via an
    absolute segment or ``..`` traversal — it fails loudly at emit time.
    """
    base_resolved = base.resolve()
    target = base_resolved.joinpath(*parts).resolve()
    if target != base_resolved and base_resolved not in target.parents:
        raise ValueError(
            f"Refusing to write outside output directory: {target} is not within {base_resolved}"
        )
    return target


# ───────── Hash-guarded emission (safe re-emit) ─────────

MANIFEST_NAME = ".rote-manifest.json"


class EmitWriter:
    """Hash-guarded file writer shared by every adapter's ``emit()``.

    An emitted output directory is a working directory, not a build
    artifact: users fill in the ``extracted/`` stubs, implement agent-loop
    harnesses, and (on Temporal) write judge classes. A naive
    ``write_text`` re-emit clobbers that work, so every write goes
    through this guard instead.

    Each emit records a manifest (:data:`MANIFEST_NAME`) mapping every
    emitted file to the sha256 of the content rote wrote. On re-emit,
    per file:

    * target missing → write it
    * on-disk content identical to the fresh content → leave it alone
    * on-disk content matches the manifest hash (pristine since the last
      emit) → overwrite with the fresh content
    * anything else (edited by the user, or emitted before manifests
      existed) → **preserve the on-disk file** and park the fresh
      content in a ``<name>.new`` sibling

    :meth:`write` returns the path the fresh content actually landed at,
    so callers' ``written`` mappings stay honest; :attr:`preserved`
    collects the files that were protected, for user-facing reporting.
    """

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._root = self.output_dir.resolve()
        self._manifest_path = self._root / MANIFEST_NAME
        self._previous = self._load_previous()
        self._current: dict[str, str] = {}
        #: relative path → the ``.new`` sibling holding the fresh content
        self.preserved: dict[str, Path] = {}

    def _load_previous(self) -> dict[str, str]:
        if not self._manifest_path.is_file():
            return {}
        try:
            data = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Corrupt emit manifest {self._manifest_path}: {e}. "
                f"Delete the file to re-emit (existing files that differ from "
                f"the fresh output will then be preserved with .new siblings)."
            ) from e
        files = data.get("files")
        if not isinstance(files, dict):
            raise ValueError(
                f"Emit manifest {self._manifest_path} has no 'files' mapping. "
                f"Delete the file and re-emit."
            )
        return {str(k): str(v) for k, v in files.items()}

    @staticmethod
    def _hash(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def landed_content(self, *parts: str, content: str) -> str:
        """What ``write(*parts, content=content)`` would leave at the target.

        The fresh ``content``, unless the guard would preserve a file the
        user edited, in which case that file's text. Writes nothing, so a
        check can refuse an emit before any file changes.
        """
        target = resolve_within(self.output_dir, *parts)
        if not target.exists():
            return content
        rel = target.relative_to(self._root).as_posix()
        on_disk = target.read_bytes()
        if self._hash(on_disk) in (self._hash(content.encode("utf-8")), self._previous.get(rel)):
            return content
        return on_disk.decode("utf-8")

    def write(self, *parts: str, content: str) -> Path:
        """Write ``content`` at ``output_dir/<parts...>`` under the hash guard.

        Returns the path the content landed at — the target itself, or
        its ``.new`` sibling when the target was preserved.
        """
        target = resolve_within(self.output_dir, *parts)
        rel = target.relative_to(self._root).as_posix()
        new_hash = self._hash(content.encode("utf-8"))

        if target.exists():
            disk_hash = self._hash(target.read_bytes())
            if disk_hash == new_hash:
                # Already exactly the fresh content; skip the write so
                # mtimes (and any watching build tools) stay quiet.
                self._current[rel] = new_hash
                return target
            if disk_hash != self._previous.get(rel):
                # Edited since the last emit, or emitted before manifests
                # existed: the user owns this file now. Never record its
                # hash as ours — the manifest only ever attests to content
                # rote itself wrote.
                new_path = target.with_name(target.name + ".new")
                new_path.write_text(content, encoding="utf-8")
                self.preserved[rel] = new_path
                if rel in self._previous:
                    self._current[rel] = self._previous[rel]
                return new_path

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        self._current[rel] = new_hash
        return target

    def finalize(self) -> None:
        """Persist the manifest. Call once, after all writes.

        Entries for files this emit didn't produce (e.g. a node was
        renamed) are carried forward: they still attest to what rote
        last wrote at those paths, which keeps the guard correct if a
        later emit produces them again.
        """
        files = {**self._previous, **self._current}
        payload = {"version": 1, "files": dict(sorted(files.items()))}
        self._manifest_path.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )


#: Carried in the error message of every stub function rote emits, Python
#: and TypeScript alike. A module is a stub while any function in it still
#: raises with this sentinel, which survives the user filling in some of a
#: module's functions and not others.
STUB_SENTINEL = "[rote stub]"


#: A JS/TS string literal (any quote style, template literals included), a
#: block comment, or a line comment. One alternation, scanned left to
#: right, so a ``//`` inside a string (a URL) is never read as a comment
#: and a quote inside a comment never opens a string.
_TS_TOKEN = re.compile(
    r'(?P<string>"(?:[^"\\\n]|\\.)*"|\'(?:[^\'\\\n]|\\.)*\'|`(?:[^`\\]|\\.)*`)'
    r"|(?P<comment>/\*[\s\S]*?\*/|//[^\n]*)"
)


def strip_ts_comments(src: str) -> str:
    """``src`` with every TypeScript comment replaced by a space.

    String literals are kept intact. Shared by stub detection (a comment
    that mentions :data:`STUB_SENTINEL` must not keep a filled-in module
    listed) and the tests' MCP scan (comments may explain MCP history).
    """
    return _TS_TOKEN.sub(lambda m: m.group("string") or " ", src)


def _raises_not_implemented(stmt: ast.stmt) -> bool:
    if not isinstance(stmt, ast.Raise) or stmt.exc is None:
        return False
    exc = stmt.exc.func if isinstance(stmt.exc, ast.Call) else stmt.exc
    return isinstance(exc, ast.Name) and exc.id == "NotImplementedError"


def _raises_with_sentinel(node: ast.AST) -> bool:
    """A ``raise X("... [rote stub] ...")``: the marker as a raise argument,
    not in a comment, docstring, or unrelated string."""
    if not isinstance(node, ast.Raise) or not isinstance(node.exc, ast.Call):
        return False
    return any(
        isinstance(arg, ast.Constant) and isinstance(arg.value, str) and STUB_SENTINEL in arg.value
        for arg in node.exc.args
    )


def _is_abstract(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for dec in func.decorator_list:
        name = dec.attr if isinstance(dec, ast.Attribute) else getattr(dec, "id", None)
        if name == "abstractmethod":
            return True
    return False


def _python_module_is_stub(text: str) -> bool:
    """Whether a Python module still has a stub function anywhere in it.

    Two shapes count, at any depth (module functions and class methods
    alike, since agent-written modules wrap vendor clients in classes):

    * a ``raise`` whose argument carries :data:`STUB_SENTINEL` (rote's own
      templates). A comment or docstring mentioning the marker does not
      count, so a filled-in module that keeps a note about it is not
      refused;
    * a function :func:`stub_raise` calls a stub, the one definition the
      compile's per-node accounting uses too.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if _raises_with_sentinel(node):
            return True
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and stub_raise(node):
            return True
    return False


def stub_raise(func: ast.FunctionDef | ast.AsyncFunctionDef) -> ast.Raise | None:
    """The ``raise NotImplementedError`` that makes ``func`` a stub, if it is one.

    The ONE definition of a compile-agent stub, shared by ``rote run``'s
    preflight (through :func:`_python_module_is_stub`) and the compile's
    per-node accounting (``rote.contracts.implementation_status``). Two
    predicates once disagreed about the same function, so there is one.

    A stub is a function that can only end in ``raise NotImplementedError``:
    its last top-level statement is that raise, unconditionally, and it
    never returns a value. A preamble before the raise is fine (BDR's
    ``zoominfo.enrich_batch`` validates the batch size first, and a stub
    like ``total = len(rows); raise NotImplementedError(...)`` computes
    something it never returns). A raise on one branch of an ``if`` is a
    deliberate unsupported case, and a function that returns on its other
    paths is working code. An ``@abstractmethod`` is not a stub.
    """
    if _is_abstract(func) or not func.body:
        return None
    last = func.body[-1]
    if not _raises_not_implemented(last) or _returns_value(func):
        return None
    assert isinstance(last, ast.Raise)
    return last


def _returns_value(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Whether ``func`` itself (not a nested def or lambda) returns a value."""
    pending: list[ast.AST] = list(func.body)
    while pending:
        node = pending.pop()
        if isinstance(node, ast.Return) and node.value is not None:
            return True
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda | ast.ClassDef):
            continue
        pending.extend(ast.iter_child_nodes(node))
    return False


def _is_stub_module(path: Path, text: str) -> bool:
    if path.suffix == ".py":
        return _python_module_is_stub(text)
    return STUB_SENTINEL in strip_ts_comments(text)


def unimplemented_stubs(output_dir: str | Path) -> list[Path]:
    """The emitted ``extracted/`` modules in ``output_dir`` that are still stubs.

    The file list comes from the emit manifest, so this works the same on
    every runtime's layout (``extracted/x.py``, ``src/extracted/x.ts``)
    and for any later reader of the directory (``rote emit``'s report,
    ``rote run``'s preflight). "Still a stub" is read off disk: a module
    drops off the list once none of its functions is still a stub, either
    a function raising with :data:`STUB_SENTINEL` or (Python) one whose
    whole body is ``raise NotImplementedError``. A directory with no
    manifest (not emitted by rote) has no known stubs; see
    :func:`stubs_tracked`.
    """
    root = Path(output_dir).resolve()
    manifest = root / MANIFEST_NAME
    if not manifest.is_file():
        return []
    files = json.loads(manifest.read_text(encoding="utf-8")).get("files") or {}
    stubs: list[Path] = []
    for rel in files:
        parts = rel.split("/")
        if "extracted" not in parts[:-1] or parts[-1].split(".", 1)[0] == "__init__":
            continue
        path = root / rel
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if _is_stub_module(path, text):
            stubs.append(path)
    return sorted(stubs)


def stubs_tracked(output_dir: str | Path, pipeline: Pipeline) -> bool:
    """Whether :func:`unimplemented_stubs` can vouch for ``output_dir``.

    True when the pipeline needs no implementation modules at all, or when
    the emit put its ``extracted/`` modules in this directory. False for an
    output whose code imports implementations from somewhere else (the
    temporal adapter before it emitted ``extracted/``), where an empty
    stub list would wrongly read as "nothing left to implement".
    """
    if not any(n.impl for n in pipeline.nodes):
        return True
    manifest = Path(output_dir).resolve() / MANIFEST_NAME
    if not manifest.is_file():
        return False
    files = json.loads(manifest.read_text(encoding="utf-8")).get("files") or {}
    return any("extracted" in rel.split("/")[:-1] for rel in files)


class EmitResult(dict[str, Path]):
    """An adapter emit mapping that also carries preserved-conflict paths.

    It *is* the ``label -> written path`` dict every ``emit()`` has always
    returned (the path is the ``.new`` sibling when a user-edited file was
    preserved), so existing consumers that iterate it keep working
    unchanged. The extra :attr:`preserved` attribute exposes the
    :class:`EmitWriter`'s ``rel -> .new path`` map directly, for callers
    (the cloud runner) that want the conflict set without sniffing ``.new``
    suffixes off the values.
    """

    def __init__(
        self,
        written: dict[str, Path] | None = None,
        preserved: dict[str, Path] | None = None,
    ) -> None:
        super().__init__(written or {})
        self.preserved: dict[str, Path] = dict(preserved or {})


# ───────── Case conversion ─────────


def _to_pascal_case(s: str) -> str:
    """Convert kebab-case or snake_case to PascalCase."""
    parts = s.replace("-", "_").split("_")
    return "".join(p.capitalize() for p in parts if p)


def _to_camel_case(s: str) -> str:
    pascal = _to_pascal_case(s)
    if not pascal:
        return ""
    return pascal[0].lower() + pascal[1:]


# ───────── Pipeline identity ─────────


def _pipeline_hash(pipeline: Pipeline) -> str:
    """Stable 8-char hash of the pipeline's identity.

    Used to version emitted artifacts (e.g. the Temporal workflow type
    name) so a regenerated pipeline becomes a new type. Old in-flight
    workflows continue on the old code; new workflows use the new code.
    """
    # Hash the full validated contents: node wiring, inputs, retries, and
    # edges all participate, so any regeneration that changes behavior gets
    # a new workflow type — name/version/counts alone miss rewires.
    # Provenance is excluded — it's tooling metadata, and annotating where
    # a pipeline came from must not re-version in-flight workflows whose
    # behavior didn't change. That covers ``Node.source`` (which SKILL.md
    # section produced a node) and ``source_skill`` (where the skill bundle
    # lived): the latter is a filesystem path the compiler re-points per
    # output location, so hashing it would mint a new workflow type on
    # every re-compilation to a different directory.
    payload = pipeline.model_dump_json(
        by_alias=True,
        exclude_none=True,
        exclude={"source_skill": True, "nodes": {"__all__": {"source"}}},
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:8]


def workflow_class_name(pipeline: Pipeline) -> str:
    """The generated workflow class name: ``<PascalName>Workflow``.

    The Cloudflare and Temporal adapters both name their emitted
    workflow class this way; centralizing it here keeps the manifest the
    cloud consumes and the emitted TypeScript from drifting apart.
    """
    return f"{_to_pascal_case(pipeline.name)}Workflow"


def pipeline_identity(pipeline: Pipeline) -> dict[str, str]:
    """Stable identity fields for a pipeline, shared by emitters and manifests.

    Returns ``{name, version, pipeline_hash, class_name}`` — the same four
    values the Cloudflare adapter bakes into the workflow.ts header and
    writes into ``manifest.json`` for the cloud runner. Single source of
    truth so the two can't disagree.
    """
    return {
        "name": pipeline.name,
        "version": pipeline.version,
        "pipeline_hash": _pipeline_hash(pipeline),
        "class_name": workflow_class_name(pipeline),
    }


# ───────── Duration strings ─────────

_IR_DURATION_RE = re.compile(r"^(\d+(?:\.\d+)?)(ms|s|m|h|d)$")

_UNIT_TO_HUMAN = {
    "ms": "milliseconds",
    "s": "seconds",
    "m": "minutes",
    "h": "hours",
    "d": "days",
}


def ir_duration_to_human(s: str) -> str:
    """Convert an IR duration string ('5m', '30s', '7d') to a human form.

    Returns e.g. '5 minutes', '30 seconds', '7 days' — the format
    Cloudflare's step config accepts directly. Strings that don't match
    the IR shorthand pattern are passed through unchanged (they're
    assumed to already be in an acceptable form).

    A value of exactly 1 is singularized ('1 hour', not '1 hours').
    Cloudflare's ``WorkflowDuration`` accepts either spelling, so this
    is legibility rather than correctness — but emitted code is read by
    humans, and '1 hours' in a reviewed artifact looks like a bug.
    """
    s = s.strip()
    m = _IR_DURATION_RE.fullmatch(s)
    if not m:
        return s
    value, unit = m.group(1), _UNIT_TO_HUMAN[m.group(2)]
    if float(value) == 1:
        unit = unit.removesuffix("s")
    return f"{value} {unit}"


_UNIT_TO_SECONDS = {
    "ms": 0.001,
    "s": 1.0,
    "m": 60.0,
    "h": 3600.0,
    "d": 86400.0,
}


def _duration_to_seconds(s: str) -> float:
    """Convert an IR duration string ('5m', '30s', '7d') to seconds.

    For runtimes whose APIs take plain second/millisecond numbers
    (DBOS's ``timeout_seconds``, DBOS-TS's ``timeoutMS``), the adapter
    converts at emission time and emits a literal — unlike Temporal,
    which emits a runtime parser.
    """
    m = _IR_DURATION_RE.fullmatch(s.strip())
    if not m:
        raise ValueError(f"cannot parse IR duration {s!r}. Expected forms like '30s', '5m', '7d'.")
    return float(m.group(1)) * _UNIT_TO_SECONDS[m.group(2)]


def _seconds_literal(seconds: float) -> str:
    """Render a seconds value as a compact numeric literal.

    Whole values drop the trailing ``.0`` — valid and idiomatic in both
    Python and TypeScript source, the two languages adapters emit.
    """
    if seconds == int(seconds):
        return str(int(seconds))
    return repr(seconds)


# ───────── Topological waves ─────────


def _execution_waves(pipeline: Pipeline) -> list[list[Node]]:
    """The pipeline's parallel-execution waves (:meth:`Pipeline.execution_waves`).

    Every adapter emits from these. The IR owns the definition because its
    input-reachability validator is defined by it: a pipeline that loads
    is one whose every node-output input is bound before its node runs,
    so adapters need no emit-time check of their own.
    """
    return pipeline.execution_waves()


# ───────── Data-flow threading ─────────


def fan_out_nodes(wave: Sequence[Node]) -> tuple[list[Node], list[Node]]:
    """Split a wave into ``(fan_out_nodes, plain_nodes)``, order preserved.

    A ``fan_out`` node dispatches once per element of its bound list, so
    it never shares the single/parallel payload shapes an adapter emits
    for plain nodes.
    """
    return [n for n in wave if n.fan_out], [n for n in wave if not n.fan_out]


# ───────── MCP-capability refusal ─────────


def refuse_mcp_only_nodes(pipeline: Pipeline, adapter: str) -> None:
    """Refuse pipelines an MCP-incapable adapter cannot emit, loudly and early.

    ``ir.Node`` accepts ``impl`` *or* ``mcp`` on an ``external_call`` — a
    node carrying only an ``mcp`` binding is perfectly valid IR. Adapters
    with no MCP backend have no module to import for such a node, so they
    must refuse it rather than emit a call to nothing.

    Reported for the whole pipeline at once, before any file is written,
    so the user fixes every offending node in one pass instead of
    rediscovering them one re-emit at a time.
    """
    orphans = [
        n.id
        for n in pipeline.nodes
        if n.kind is NodeKind.EXTERNAL_CALL and n.impl is None and n.mcp is not None
    ]
    if not orphans:
        return
    raise ValueError(
        f"{adapter} adapter: pipeline {pipeline.name!r} contains external_call "
        f"node(s) [{', '.join(orphans)}] bound to an MCP server with no `impl:` — "
        f"the {adapter} adapter has no MCP backend, so there is nothing to call. "
        f"Emit onto a runtime that speaks MCP "
        f"(`--runtime dbos`, `--runtime dbos-ts`, `--runtime cloudflare`, "
        f"`--runtime inngest`), or give each node an `impl:` pointing at an "
        f"extracted module."
    )


#: Fallback iteration bound when an ``agent_loop`` declares no
#: ``termination``. An agent loop must always be bounded — an unbounded
#: one in a durable workflow burns budget until a human notices.
#:
#: Lives here rather than beside either language's emitter because it is
#: an IR policy, not a language detail: a loop compiled to Python and the
#: same loop compiled to TypeScript must agree on how many turns "no
#: declared bound" means.
DEFAULT_AGENT_MAX_ITERATIONS = 10


# ───────── an agent loop's declared output ─────────

#: The tool an agent loop calls once, at the end, to hand back its
#: declared output. Its arguments are the declared output, so the keys
#: downstream nodes read exist on every runtime and every lane.
AGENT_SUBMIT_TOOL = "submit_result"

#: The one key a single-type output (``output: IntelBrief``) is submitted
#: under. The step returns the value itself, never the wrapper.
AGENT_RESULT_KEY = "result"

#: Where the TS runtimes declare the IR's undefined named types, relative
#: to the app root. User-editable: the user narrows each alias once, and
#: EmitWriter keeps that edit on re-emit.
TS_TYPES_MODULE = "src/types.ts"

_JSON_OBJECT_TS = "{ [key: string]: JsonValue }"

_IR_SCALARS: dict[str, tuple[dict[str, Any], str]] = {
    "str": ({"type": "string"}, "string"),
    "string": ({"type": "string"}, "string"),
    "int": ({"type": "integer"}, "number"),
    "integer": ({"type": "integer"}, "number"),
    "float": ({"type": "number"}, "number"),
    "number": ({"type": "number"}, "number"),
    "bool": ({"type": "boolean"}, "boolean"),
    "boolean": ({"type": "boolean"}, "boolean"),
    "dict": ({"type": "object"}, _JSON_OBJECT_TS),
    "object": ({"type": "object"}, _JSON_OBJECT_TS),
    "list": ({"type": "array"}, "JsonValue[]"),
    "array": ({"type": "array"}, "JsonValue[]"),
}

_NULL_NAMES = ("null", "None")

#: A named type becomes a TS alias only when its name is a plain
#: identifier; anything else stays ``JsonValue``.
_TS_TYPE_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _split_top_level(text: str, sep: str) -> list[str]:
    """Split on ``sep`` outside brackets: ``list[a | b] | null`` is 2 parts."""
    parts: list[str] = []
    depth, start = 0, 0
    for i, ch in enumerate(text):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        elif ch == sep and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return [p.strip() for p in parts]


def ir_type(type_name: str, named: set[str] | None = None) -> tuple[dict[str, Any], str]:
    """``(JSON Schema, TS type)`` for one IR type name.

    IR types are free-form names (``list[InvoiceResult]``, ``str | null``).
    Scalars, lists, optionals and plain dicts map exactly. A named type the
    IR does not define (``InvoiceResult``) accepts any JSON value: the
    schema cannot invent its fields. Its TS type is the name itself, an
    alias the TS runtimes declare in :data:`TS_TYPES_MODULE` as
    ``JsonValue`` for the user to narrow; ``named`` collects those names.
    """
    name = type_name.strip().strip("\"'")
    alternatives = _split_top_level(name, "|")
    if len(alternatives) > 1:
        nullable = any(a in _NULL_NAMES for a in alternatives)
        typed = [ir_type(a, named) for a in alternatives if a not in _NULL_NAMES]
        schemas = [s for s, _ in typed] + ([{"type": "null"}] if nullable else [])
        ts = " | ".join([t for _, t in typed] + (["null"] if nullable else []))
        return {"anyOf": schemas}, ts
    match = re.fullmatch(r"Optional\[(.*)\]", name)
    if match:
        return ir_type(f"{match.group(1)} | null", named)
    match = re.fullmatch(r"(?:list|List|Sequence|set|tuple)\[(.*)\]", name)
    if match:
        item_schema, item_ts = ir_type(match.group(1), named)
        wrapped = f"({item_ts})" if "|" in item_ts else item_ts
        return {"type": "array", "items": item_schema}, f"{wrapped}[]"
    if re.fullmatch(r"(?:dict|Dict|Mapping)\[.*\]", name):
        return {"type": "object"}, _JSON_OBJECT_TS
    if name in _IR_SCALARS:
        schema, ts = _IR_SCALARS[name]
        return dict(schema), ts
    schema = {"description": f"Value of type {name}."}
    if _TS_TYPE_NAME_RE.fullmatch(name) and name != "JsonValue":
        if named is not None:
            named.add(name)
        return schema, name
    return schema, "JsonValue"


#: Anthropic caps a strict request at this many ``anyOf`` and type-array
#: parameters; every nullable field (``str | null``) is one. Past it the
#: request is a 400, so such a loop stays non-strict.
STRICT_UNION_LIMIT = 16


def _union_count(schema: Any) -> int:
    """``anyOf`` and type-array parameters in ``schema``, nested included."""
    if isinstance(schema, list):
        return sum(_union_count(item) for item in schema)
    if not isinstance(schema, dict):
        return 0
    own = int("anyOf" in schema) + int(isinstance(schema.get("type"), list))
    return own + sum(_union_count(value) for value in schema.values())


def strict_compatible(schema: Any) -> bool:
    """True when ``schema`` fits Anthropic's strict tool-input subset.

    Strict tool use constrains sampling to the schema, so every leaf needs
    a type and every object ``additionalProperties: false``. A named type
    the IR does not define is a description-only schema, which that
    subset does not cover, and a schema past :data:`STRICT_UNION_LIMIT`
    unions would 400. Either way the submit tool stays non-strict and
    rote's own validation (every lane) is what holds the line.
    """
    return _union_count(schema) <= STRICT_UNION_LIMIT and _strict_shape(schema)


def _strict_shape(schema: Any) -> bool:
    if not isinstance(schema, dict):
        return False
    if "anyOf" in schema:
        return all(_strict_shape(s) for s in schema["anyOf"])
    kind = schema.get("type")
    if kind is None:
        return False
    if kind == "object":
        properties = schema.get("properties")
        return (
            schema.get("additionalProperties") is False
            and isinstance(properties, dict)
            and all(_strict_shape(s) for s in properties.values())
        )
    if kind == "array":
        return "items" in schema and _strict_shape(schema["items"])
    return True


class AgentLoopOutput:
    """What an agent loop must hand back, derived once from its IR node.

    ``schema`` is the object schema of the ``submit_result`` arguments, or
    None when the node declares no output (the step then returns
    ``{"result": <final text>}``). ``unwrap_key`` is set for a single-type
    output, whose value is the step's whole result. ``ts_type`` is the TS
    return type of the emitted loop function. ``named_types`` maps each
    declared key (``"result"`` for a single type) to the IR names in it
    that the IR does not define, whose shape nothing checks. ``strict`` says
    whether the submit tool can be a strict tool.
    """

    def __init__(
        self,
        schema: dict[str, Any] | None,
        unwrap_key: str | None,
        ts_type: str,
        named_types: dict[str, list[str]],
    ):
        self.schema = schema
        self.unwrap_key = unwrap_key
        self.ts_type = ts_type
        self.named_types = named_types
        self.strict = schema is not None and strict_compatible(schema)


def agent_loop_output(node: Node) -> AgentLoopOutput:
    """The declared-output contract of an ``agent_loop`` node.

    Downstream nodes read the loop's declared ``output`` keys
    (``process_invoices_loop.output.invoice_results``), so the loop must
    produce exactly those keys, on every runtime. A node's
    ``output_schema`` wins over its type names: it is evidence from real
    payloads, a name is a guess.
    """
    assert node.kind is NodeKind.AGENT_LOOP
    output = node.output
    evidence = node.output_schema or {}
    if isinstance(output, dict) and output:
        observed = evidence.get("properties") if isinstance(evidence, dict) else None
        typed: dict[str, tuple[dict[str, Any], str]] = {}
        named: dict[str, list[str]] = {}
        for key, name in output.items():
            found: set[str] = set()
            typed[key] = ir_type(name, found)
            if found and key not in (observed or {}):
                named[key] = sorted(found)
        properties = {key: (observed or {}).get(key, schema) for key, (schema, _) in typed.items()}
        schema = {
            "type": "object",
            "properties": properties,
            "required": list(output),
            "additionalProperties": False,
        }
        ts = "{ " + "; ".join(f"{json.dumps(k)}: {t}" for k, (_, t) in typed.items()) + " }"
        return AgentLoopOutput(schema, None, ts, named)
    if isinstance(output, str) and output.strip():
        found = set()
        value_schema, ts = ir_type(output, found)
        schema = {
            "type": "object",
            "properties": {AGENT_RESULT_KEY: evidence or value_schema},
            "required": [AGENT_RESULT_KEY],
            "additionalProperties": False,
        }
        named = {AGENT_RESULT_KEY: sorted(found)} if found and not evidence else {}
        return AgentLoopOutput(schema, AGENT_RESULT_KEY, ts, named)
    return AgentLoopOutput(None, None, "{ result: string }", {})


def agent_loop_named_types(pipeline: Pipeline) -> list[str]:
    """Every undefined named type the pipeline's agent loops declare, sorted."""
    names: set[str] = set()
    for node in pipeline.nodes:
        if node.kind is NodeKind.AGENT_LOOP:
            for found in agent_loop_output(node).named_types.values():
                names.update(found)
    return sorted(names)


def unchecked_output_notes(pipeline: Pipeline, *, types_module: str | None) -> list[str]:
    """One line per agent-loop output key whose type the IR never defines.

    Its value reaches the next node unchecked, so emit says where to fix
    that: ``output_schema`` in the IR, and on TS the alias in
    ``types_module`` the user can narrow.
    """
    notes: list[str] = []
    for node in pipeline.nodes:
        if node.kind is not NodeKind.AGENT_LOOP:
            continue
        for key, names in agent_loop_output(node).named_types.items():
            for name in names:
                where = (
                    f"narrow it in {types_module}"
                    if types_module
                    else "declare its fields in the node's output_schema"
                )
                notes.append(
                    f"{node.id}.{key}: type {name} is not defined in the IR; "
                    f"shape unchecked, {where}"
                )
    return notes
