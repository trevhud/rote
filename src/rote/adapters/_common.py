"""Helpers shared by every runtime adapter.

These functions are runtime-agnostic: they operate purely on the
:class:`rote.ir.Pipeline` IR and on plain strings. Anything that encodes
a *runtime's* semantics (Temporal retry policies, Cloudflare step
configs, …) stays in the adapter that owns it.

Extracted from ``rote.adapters.temporal`` / ``rote.adapters.cloudflare``
once the second adapter proved the helpers were genuinely shared.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from rote.ir import Node, Pipeline, parse_input_ref

# ───────── Prose → safe code comment/docstring ─────────


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
    payload = pipeline.model_dump_json(by_alias=True, exclude_none=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:8]


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
    """
    s = s.strip()
    m = _IR_DURATION_RE.fullmatch(s)
    if not m:
        return s
    return f"{m.group(1)} {_UNIT_TO_HUMAN[m.group(2)]}"


# ───────── Topological waves ─────────


def _execution_waves(pipeline: Pipeline) -> list[list[Node]]:
    """Topologically sort the pipeline into parallel-execution waves.

    A "wave" is a set of nodes whose dependencies have all been satisfied
    by the time the wave starts; nodes within a wave can run in parallel.

    Nodes that appear inside another node's ``loop_body`` are excluded
    from the top-level waves — they're orchestrated from inside the
    parent loop activity, not by the workflow.
    """
    nested_ids: set[str] = set()
    for n in pipeline.nodes:
        if n.loop_body:
            nested_ids.update(n.loop_body)

    eligible = [n for n in pipeline.nodes if n.id not in nested_ids]
    eligible_ids = {n.id for n in eligible}

    in_degree: dict[str, int] = {n.id: 0 for n in eligible}
    for edge in pipeline.edges:
        if edge.from_ in eligible_ids and edge.to in eligible_ids:
            in_degree[edge.to] += 1

    waves: list[list[Node]] = []
    remaining: set[str] = set(in_degree.keys())

    while remaining:
        # Sort by id for stable, deterministic emission order.
        wave_ids = sorted(nid for nid in remaining if in_degree[nid] == 0)
        if not wave_ids:
            raise ValueError(
                f"Cycle detected in pipeline {pipeline.name!r}; "
                f"remaining nodes: {sorted(remaining)}"
            )
        waves.append([pipeline.node_by_id(nid) for nid in wave_ids])
        for nid in wave_ids:
            remaining.remove(nid)
            for edge in pipeline.edges:
                if edge.from_ == nid and edge.to in eligible_ids:
                    in_degree[edge.to] -= 1

    return waves


# ───────── Data-flow threading ─────────


def check_input_refs_available(node: Node, available: set[str]) -> None:
    """Emit-time guard: every node-output reference must already be bound.

    Adapters emit nodes wave-by-wave, binding each node's result to a
    local as they go. A node whose ``inputs`` reference a node in a
    *later* wave (or a loop-body sub-node, which never binds a top-level
    result) would produce code that references an undefined variable —
    fail loudly at emit time instead.

    IR validation already guarantees the referenced node *exists*; this
    check is about emission ordering, which is an adapter concern.
    """
    if not node.inputs:
        return
    for param, ref in node.inputs.items():
        parsed = parse_input_ref(ref)
        if parsed.node_id is not None and parsed.node_id not in available:
            raise ValueError(
                f"Node {node.id!r} input {param!r} references {ref!r}, but node "
                f"{parsed.node_id!r} has no result available at this point in the "
                f"workflow (it runs in a later wave, or is a loop-body sub-node "
                f"with no top-level result)."
            )
