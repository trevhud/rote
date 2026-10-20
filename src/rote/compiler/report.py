"""compile-report.md, written by rote from the deliverable, not by the agent.

The report used to be the agent's last file. Profiled compiles spent 17
to 66 seconds of output on it (the CLI streams roughly 75 to 150 tokens
a second), rewrote it again in repair rounds, and were sent back to
repair when its Open questions count fell short of the code's
``# rote: assumption:`` markers. None of it was a decision: every fact in
it already lived in ``pipeline.yaml`` (kinds, gates, each node's
one-sentence justification) or in the extracted code (``# skill:``
citations, assumption markers, stub messages). So rote renders it from
those, and the agent writes no report at all.

The Open questions section keeps the format
:func:`rote.compiler.provenance.open_questions` parses (one ``###``
sub-heading per entry), so ``rote bench``'s ``OQ`` column still counts
the questions a reviewer must answer: one per distinct assumption and
one per stub.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from rote.compiler.provenance import Assumption, _normalized, assumptions, citations
from rote.contracts import NodeImplementation, implementation_status
from rote.ir import NodeKind, Pipeline

#: Kinds in the order the rubric introduces them, so every report reads
#: the breakdown the same way.
_KIND_ORDER = (
    NodeKind.PURE_FUNCTION,
    NodeKind.EXTERNAL_CALL,
    NodeKind.LLM_JUDGE,
    NodeKind.AGENT_LOOP,
    NodeKind.HITL_GATE,
)


@dataclass(frozen=True)
class ReportExtras:
    """What a compile mode adds to the one shared report.

    The default compile passes none. ``--system1`` passes its preamble
    (rote decided the structure), its review items (low-confidence
    decisions, the agent's proposals, decisions rote restored) and its
    next steps. Every mode renders through :func:`render_compile_report`,
    so the summary, node table, crystallization log and the assumption
    and stub entries are computed one way.
    """

    #: A paragraph under the title, after rote's own provenance line.
    preamble: str | None = None
    #: ``(title, detail)`` pairs, each its own Open questions entry after
    #: the assumptions and stubs.
    questions: Sequence[tuple[str, str]] = field(default_factory=tuple)
    #: Bullets for a "Suggested next steps" section, omitted when empty.
    next_steps: Sequence[str] = field(default_factory=tuple)


def _cell(text: str) -> str:
    """A table cell: one line, pipes escaped."""
    return " ".join(text.split()).replace("|", "\\|")


def _module_summary(source: str) -> str | None:
    """The first line of a module's docstring, if it has one."""
    try:
        doc = ast.get_docstring(ast.parse(source))
    except SyntaxError:
        return None
    if not doc:
        return None
    first = doc.strip().splitlines()[0].strip()
    return first or None


def _summary(pipeline: Pipeline, impls: list[NodeImplementation]) -> list[str]:
    counts = {kind: 0 for kind in _KIND_ORDER}
    for node in pipeline.nodes:
        counts[node.kind] += 1
    total = len(pipeline.nodes)
    breakdown = ", ".join(f"{counts[k]} `{k.value}`" for k in _KIND_ORDER)
    lines = [f"- {total} node(s): {breakdown}."]
    non_gate = total - counts[NodeKind.HITL_GATE]
    if non_gate:
        codified = counts[NodeKind.PURE_FUNCTION] + counts[NodeKind.EXTERNAL_CALL]
        lines.append(
            f"- Codified (`pure_function` + `external_call`): {codified} of {non_gate} "
            f"non-gate node(s), {round(100 * codified / non_gate)}%."
        )
    if impls:
        done = sum(1 for i in impls if i.status == "implemented")
        lines.append(f"- Implemented impl node(s): {done} of {len(impls)}.")
    if pipeline.entry_nodes:
        lines.append(f"- Entry: {', '.join(f'`{n}`' for n in pipeline.entry_nodes)}.")
    if pipeline.exit_nodes:
        lines.append(f"- Exit: {', '.join(f'`{n}`' for n in pipeline.exit_nodes)}.")
    mandatory = [n.id for n in pipeline.nodes if n.mandatory]
    if mandatory:
        lines.append(f"- Mandatory: {', '.join(f'`{n}`' for n in mandatory)}.")
    fanned = [n.id for n in pipeline.nodes if n.fan_out]
    if fanned:
        lines.append(f"- Fan-out: {', '.join(f'`{n}`' for n in fanned)}.")
    gates = [n for n in pipeline.nodes if n.kind is NodeKind.HITL_GATE]
    for gate in gates:
        timeout = gate.timeout or pipeline.config.hitl.default_timeout
        lines.append(
            f"- HITL gate `{gate.id}` (signal `{gate.signal}`, timeout {timeout}): "
            f"{' '.join(gate.description.split())}"
        )
    servers = pipeline.required_mcp_servers
    if servers:
        named = ", ".join(f"`{s}` ({', '.join(nodes)})" for s, nodes in servers.items())
        lines.append(f"- MCP servers: {named}.")
    return lines


def _nodes_table(pipeline: Pipeline, impls: list[NodeImplementation]) -> list[str]:
    status = {i.node_id: i.status for i in impls}
    lines = [
        "| Node | Kind | Implementation | Skill section | Description |",
        "|---|---|---|---|---|",
    ]
    for node in pipeline.nodes:
        if node.impl:
            where = f"`{node.impl}` ({status.get(node.id, 'unchecked')})"
        elif node.signature_spec is not None or node.signature:
            where = f"`{node.signature}`" if node.signature else "signature_spec"
        elif node.tools:
            where = f"tools: {', '.join(node.tools)}"
        elif node.signal:
            where = f"signal `{node.signal}`"
        else:
            where = ""
        section = node.source.section if node.source is not None else ""
        lines.append(
            f"| `{node.id}` | {node.kind.value} | {_cell(where)} | {_cell(section)} "
            f"| {_cell(node.description)} |"
        )
    return lines


def _crystallization_log(root: Path) -> list[str]:
    """Each extracted module, with the skill text it cites."""
    extracted = root / "extracted"
    modules = sorted(extracted.rglob("*.py")) if extracted.is_dir() else []
    lines: list[str] = []
    for path in modules:
        rel = str(path.relative_to(root))
        source = path.read_text(encoding="utf-8", errors="replace")
        quotes = [c for c in citations(source) if c.quote is not None]
        summary = _module_summary(source)
        head = f"- `{rel}`" + (f": {summary}" if summary else "")
        lines.append(f"{head} ({len(quotes)} skill citation(s))")
        lines += [f'  - line {c.line}: "{" ".join(str(c.quote).split())}"' for c in quotes]
    return lines or ["- No extracted modules."]


def _open_questions(
    found: list[Assumption],
    impls: list[NodeImplementation],
    extra: Sequence[tuple[str, str]],
) -> list[str]:
    """One ``###`` entry per distinct assumption, one per stub, one per extra."""
    distinct: dict[str, list[Assumption]] = {}
    for a in found:
        distinct.setdefault(_normalized(a.text), []).append(a)
    lines: list[str] = []
    for number, group in enumerate(distinct.values(), start=1):
        owner = group[0].node_id or "module helper"
        lines += [
            f"### OQ-{number}: {owner}",
            "",
            group[0].text,
            "",
            "Marked `# rote: assumption:` at "
            + ", ".join(f"`{a.path}:{a.line}`" for a in group)
            + ".",
            "",
        ]
    for impl in impls:
        if impl.status == "implemented" or impl.mcp_bound:
            continue
        reason = impl.reason or "no reason given"
        lines += [f"### Stub: {impl.node_id} (`{impl.impl}`, {impl.status})", "", reason, ""]
    for title, detail in extra:
        lines += [f"### {' '.join(title.split())}", "", detail, ""]
    if not lines:
        return ["None: the code records no assumption and leaves no stub.", ""]
    return lines


def render_compile_report(
    pipeline: Pipeline, root: Path, extras: ReportExtras | None = None
) -> str:
    """compile-report.md for the deliverable in ``root``.

    A pure function of ``pipeline.yaml``, the files under ``root`` and
    ``extras``: re-rendering an unchanged deliverable yields the same text.
    """
    extras = extras or ReportExtras()
    impls = implementation_status(pipeline, root)
    found = assumptions(pipeline, root)
    preamble = ["", extras.preamble] if extras.preamble else []
    steps = ["", "## Suggested next steps", "", *(f"- {s}" for s in extras.next_steps)]
    parts = [
        f"# Compile report: {pipeline.name}",
        "",
        "rote wrote this report from `pipeline.yaml` and the extracted code. Each "
        "node's description is the compiler's reason for its kind; each "
        '`# skill: "..."` citation is the rule a line of code implements; each '
        "`# rote: assumption:` marker is a decision the skill does not state.",
        *preamble,
        "",
        "## Summary",
        "",
        *_summary(pipeline, impls),
        "",
        "## Nodes",
        "",
        *_nodes_table(pipeline, impls),
        "",
        "## Crystallization log",
        "",
        *_crystallization_log(root),
        "",
        "## Open questions",
        "",
        *_open_questions(found, impls, extras.questions),
        *(steps if extras.next_steps else []),
    ]
    return "\n".join(parts).rstrip("\n") + "\n"
