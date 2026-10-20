"""Every outcome the compiled code decides traces to the skill, or is declared.

A real compile decided that an expense over its category cap is a
``block``. The skill lists the rule and never says block or review, the
hand-written reference says review, and the agent's own tests encoded
its choice, so they passed. The rubric (implementation.md, "Every
outcome cites the skill") makes the agent mark each decided outcome one
of two ways:

* ``# skill: "<verbatim text>"`` quoting the skill sentence that decides
  it, or
* ``# rote: assumption: <what was chosen>`` for an outcome the skill does
  not state, which rote lists under the compile report's Open questions.

These checks keep the markers honest. They are deliberately mechanical:
a citation has to be a real, non-trivial quote of the skill's prose, and
an outcome literal with no marker at all is reported rather than trusted.
A review showed every looser version could be satisfied without telling
a reviewer anything (one-word quotes, quoting a heading, fragments
matched out of order, no comment at all), so each of those is a test.

What a finding costs the compile depends on what it means:

* A citation that is not in the skill (``citation_not_in_skill``,
  ``malformed_citation``) is a factual error: the code claims the skill
  says something it does not. It fails the compile (``severity="error"``).
* An assumption is a declared decision, and declaring it is the point:
  however many there are, they never fail the compile. They are listed
  for review instead (:func:`assumptions`).
* A *review gap* is a declaration that is missing, not false: an outcome
  with no marker (``uncited_outcome``). The repair loop asks
  the agent to close it, and one that survives the repair rounds is a
  warning, not a failure. A reviewer can close it from the listed
  assumptions and the warning, while failing the compile over it would
  discard a paid run for a bookkeeping gap.

``strict=True`` (``--strict-provenance``, for CI) makes every finding an
error, review gaps included.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

from rote.contracts import ContractFinding
from rote.ir import NodeKind, Pipeline

REPORT_FILENAME = "compile-report.md"

#: A quoted fragment shorter than this says nothing about an outcome:
#: "Review" or "must" can be found in any skill.
MIN_FRAGMENT_WORDS = 4


#: Name tokens that make a slot an outcome: the key, keyword, or variable
#: a pure_function writes a decided label into.
OUTCOME_WORDS = frozenset(
    {
        "action",
        "alert",
        "bucket",
        "category",
        "classification",
        "color",
        "colour",
        "decision",
        "disposition",
        "flag",
        "grade",
        "label",
        "level",
        "outcome",
        "priority",
        "rating",
        "result",
        "risk",
        "route",
        "routing",
        "severity",
        "status",
        "tier",
        "verdict",
    }
)

_CITATION_START = re.compile(r"#[ \t]*skill:[ \t]*")
_ASSUMPTION_RE = re.compile(r"#\s*rote:\s*assumption:\s*(?P<text>.+)")
_MARKER_RE = re.compile(r"#\s*(skill:|rote:\s*assumption:)")


def _normalized(text: str) -> str:
    """Lowercase, markdown links reduced to their text, emphasis and code
    marks dropped, quote marks made one, whitespace collapsed.

    A real compile quoted "rates in references/fx-rates.md", which the
    skill writes as a markdown link. Another quoted the skill's `search for
    "BDR do not contact" list` as `search for 'BDR do not contact' list`,
    because its own citation was in double quotes, and the compile failed
    on a "fabricated" citation that was verbatim but for the quote style.
    Which quote mark encloses a phrase is not part of what the skill says.

    Dashes likewise: a BDR compile at low effort quoted the skill's
    ``list \u2014 search for`` as ``list -- search for`` in an ASCII comment, and
    four verbatim citations failed the compile. An em dash, an en dash and
    a run of hyphens are one mark here. A single hyphen stays as it is, so
    ``do-not-contact`` still has to match ``do-not-contact``. The same
    citation also escaped the skill's inner quotes (``\\"BDR do not
    contact\\"``), a code habit that is quote style too, so a backslash
    before a quote mark is dropped.
    """
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\\([\"'])", r"\1", text)
    text = re.sub(r"[\"'\u2018\u2019\u201c\u201d]", "'", text)
    text = re.sub(r"[\u2013\u2014]|-{2,}", "\u2014", text)
    return " ".join(re.sub(r"[*_`]", "", text).lower().split())


# ───────── The skill's prose, as citable paragraphs ─────────


def _is_heading(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("#") or bool(re.fullmatch(r"\*\*[^*]+\*\*:?", stripped))


def skill_paragraphs(skill_dir: Path) -> list[str]:
    """Normalized prose paragraphs from every markdown file in the skill.

    Headings are not prose and are left out, so a heading cannot be
    cited. A paragraph is a run of non-blank lines (a table or a list
    counts as one), which bounds how far apart cited fragments may be.

    Inside a fenced block nothing is a heading: a skill's required output
    template (``# Expense audit: <period>`` in a markdown fence) is exactly
    the text a renderer should cite, and a real compile was sent back for
    citing it.
    """
    paragraphs: list[str] = []
    for path in sorted(skill_dir.rglob("*.md")):
        current: list[str] = []
        in_fence = False
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip().startswith(("```", "~~~")):
                in_fence = not in_fence
                continue
            if not line.strip() or (not in_fence and _is_heading(line)):
                if current:
                    paragraphs.append(_normalized(" ".join(current)))
                current = []
                continue
            current.append(line)
        if current:
            paragraphs.append(_normalized(" ".join(current)))
    return paragraphs


# ───────── Citations in the code ─────────


@dataclass(frozen=True)
class Citation:
    line: int
    quote: str | None  # None: opened with `"` and never closed


def citations(source: str) -> list[Citation]:
    """Every ``# skill: "..."`` in ``source``.

    The quote opens at the first ``"`` after ``skill:`` and closes at the
    ``"`` that ends a comment line, so text inside it may contain ``"``
    (skill prose quotes labels: "Damaged in Transit"). A long quote may
    continue on the following comment lines; their ``#`` is not part of
    it.
    """
    found: list[Citation] = []
    lines = source.splitlines()
    i = 0
    while i < len(lines):
        m = _CITATION_START.search(lines[i])
        if m is None or not lines[i][m.end() :].startswith('"'):
            i += 1
            continue
        start = i
        text = lines[i][m.end() + 1 :].rstrip()
        parts: list[str] = []
        while True:
            if text.endswith('"'):
                parts.append(text[:-1])
                found.append(Citation(start + 1, " ".join(parts)))
                break
            parts.append(text)
            i += 1
            if i >= len(lines) or not lines[i].lstrip().startswith("#"):
                found.append(Citation(start + 1, None))
                i -= 1
                break
            text = lines[i].lstrip()[1:].strip()
        i += 1
    return found


def citation_problem(quote: str, paragraphs: list[str]) -> str | None:
    """Why ``quote`` is not a citation of the skill, or None if it is."""
    fragments = [_normalized(f) for f in quote.replace("…", "...").split("...")]
    fragments = [f for f in fragments if f]
    if not fragments:
        return "it is empty"
    short = [f for f in fragments if len(f.split()) < MIN_FRAGMENT_WORDS]
    if short:
        return (
            f"{short[0]!r} is under {MIN_FRAGMENT_WORDS} words, too short to say which "
            f"rule decides anything"
        )
    for paragraph in paragraphs:
        at = 0
        for fragment in fragments:
            found = paragraph.find(fragment, at)
            if found < 0:
                break
            at = found + len(fragment)
        else:
            return None
    if len(fragments) > 1:
        return "no paragraph of the skill's prose contains these fragments in this order"
    return "the skill's prose contains no such text (headings do not count)"


# ───────── Outcome literals with no marker ─────────


def _is_outcome_name(name: str | None) -> bool:
    if not name:
        return False
    tokens = re.split(r"[^a-z0-9]+", re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower())
    return any(t in OUTCOME_WORDS for t in tokens)


def _label(value: ast.expr) -> ast.Constant | None:
    if (
        isinstance(value, ast.Constant)
        and isinstance(value.value, str)
        # A placeholder with no letters or digits ("—", "-") is display
        # punctuation, not a decided label.
        and any(ch.isalnum() for ch in value.value)
        and "\n" not in value.value
        and len(value.value) <= 80
    ):
        return value
    return None


#: Module-level names whose string values are labels a slot can receive by
#: reference: ``BLOCK = "block"`` (``"BLOCK"``), a table
#: ``SEVERITY_BY_RULE = {"over_cap": "block"}`` (its values), and Enum
#: members (``"Severity.BLOCK"``).
Named = dict[str, list[ast.Constant]]


def _labels(value: ast.expr | None, named: Named | None = None) -> list[ast.Constant]:
    """The string labels a value can evaluate to.

    Through ``a if c else b``, inside a returned or assigned tuple, list,
    set or dict (its values), and by reference: a module constant, a
    lookup in a module-level table (``TABLE[k]``, ``TABLE.get(k, d)``),
    or an Enum member. A label reached by reference is reported at its
    definition, which is where its marker belongs.
    """
    named = named or {}
    if value is None:
        return []
    if isinstance(value, ast.IfExp):
        return _labels(value.body, named) + _labels(value.orelse, named)
    if isinstance(value, ast.Tuple | ast.List | ast.Set):
        return [lit for elt in value.elts for lit in _labels(elt, named)]
    if isinstance(value, ast.Dict):
        return [lit for v in value.values for lit in _labels(v, named)]
    if isinstance(value, ast.Name):
        return list(named.get(value.id, []))
    if isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name):
        return list(named.get(f"{value.value.id}.{value.attr}", []))
    if isinstance(value, ast.Subscript) and isinstance(value.value, ast.Name):
        return list(named.get(value.value.id, []))
    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == "get"
        and isinstance(value.func.value, ast.Name)
    ):
        defaults = [lit for arg in value.args[1:] for lit in _labels(arg, named)]
        return list(named.get(value.func.value.id, [])) + defaults
    label = _label(value)
    return [label] if label is not None else []


def _is_enum_class(node: ast.ClassDef) -> bool:
    names = {b.id if isinstance(b, ast.Name) else getattr(b, "attr", "") for b in node.bases}
    return any(n.endswith("Enum") for n in names)


def _named_labels(tree: ast.Module) -> Named:
    """Module constants, tables, and Enum members holding string labels."""
    named: Named = {}
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign | ast.AnnAssign):
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            labels = _labels(stmt.value)
            for target in targets:
                if isinstance(target, ast.Name) and labels:
                    named[target.id] = labels
        elif isinstance(stmt, ast.ClassDef) and _is_enum_class(stmt):
            for member in stmt.body:
                if isinstance(member, ast.Assign):
                    labels = _labels(member.value)
                    for target in member.targets:
                        if isinstance(target, ast.Name) and labels:
                            named[f"{stmt.name}.{target.id}"] = labels
    return named


def _target_name(target: ast.expr) -> str | None:
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    if isinstance(target, ast.Subscript):
        key = target.slice
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            return key.value
    return None


def outcome_literals(tree: ast.Module) -> list[ast.Constant]:
    """The labels the code decides: returned, or written into an outcome slot.

    * ``return "block"`` or ``return ("block", reason)`` (a function that
      returns a label is a classifier);
    * a dict value under an outcome key (``{"severity": "block"}``);
    * a keyword argument with an outcome name (``Finding(severity="block")``);
    * an assignment to an outcome name (``status = "flagged"``,
      ``row["alert_level"] = "missort_alert"``), including a whole table
      (``SEVERITY_BY_RULE = {"over_cap": "block"}``: every value);
    * any of those reached by reference: ``{"severity": BLOCK}``,
      ``SEVERITY_BY_RULE[rule]``, ``Severity.BLOCK``, reported where the
      constant or member is defined.

    A slot is recognized by its name (:data:`OUTCOME_WORDS`). This catches
    the accidental unmarked label, not a deliberate evasion: a label stored
    under an unrecognized name, or built at run time, is not seen.
    """
    named = _named_labels(tree)
    found: dict[int, ast.Constant] = {}

    def add(labels: list[ast.Constant]) -> None:
        for label in labels:
            found.setdefault(id(label), label)

    for node in ast.walk(tree):
        if isinstance(node, ast.Return):
            add(_labels(node.value, named))
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if (
                    isinstance(key, ast.Constant)
                    and isinstance(key.value, str)
                    and _is_outcome_name(key.value)
                ):
                    add(_labels(value, named))
        elif (
            (isinstance(node, ast.keyword) and _is_outcome_name(node.arg))
            or (
                isinstance(node, ast.Assign)
                and any(_is_outcome_name(_target_name(t)) for t in node.targets)
            )
            or (isinstance(node, ast.AnnAssign) and _is_outcome_name(_target_name(node.target)))
        ):
            add(_labels(node.value, named))
    return sorted(found.values(), key=lambda c: (c.lineno, c.col_offset))


def _parents(tree: ast.Module) -> dict[ast.AST, ast.AST]:
    return {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}


def _anchors(stmt: ast.stmt, parents: dict[ast.AST, ast.AST]) -> list[int]:
    """The lines whose introducing comment may vouch for ``stmt``.

    The statement itself, every compound statement enclosing it (up to its
    function), and for each of those, the first of a run of sibling ``if``
    statements that each return or raise: one comment above such a ladder
    (``if n >= 41: return ...`` then ``if n >= 25: return ...``) describes
    all of it. Independent checks that fall through are not a ladder, so a
    citation for one does not cover the next (a real compile's cap check
    was "covered" by the entertainment check's citation above it).
    ``elif``/``else`` arms are already inside their ``if``.
    """
    anchors: list[int] = []
    node: ast.AST | None = stmt
    while node is not None and not isinstance(node, ast.Module):
        if isinstance(node, ast.stmt):
            anchors.append(node.lineno)
            parent = parents.get(node)
            for field in ("body", "orelse", "finalbody"):
                siblings = getattr(parent, field, None)
                if isinstance(siblings, list) and node in siblings:
                    i = siblings.index(node)
                    while isinstance(node, ast.If) and i > 0 and _returns(siblings[i - 1]):
                        i -= 1
                    anchors.append(siblings[i].lineno)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            break
        node = parents.get(node)
    return anchors


def _returns(stmt: ast.stmt) -> bool:
    """An ``if`` whose body ends by leaving: a rung of an early-return ladder."""
    return (
        isinstance(stmt, ast.If)
        and bool(stmt.body)
        and isinstance(stmt.body[-1], ast.Return | ast.Raise)
    )


def _comment_block_above(lines: list[str], line: int) -> range:
    """Line numbers of the comment lines directly above ``line``."""
    top = line
    while top > 1 and lines[top - 2].lstrip().startswith("#"):
        top -= 1
    return range(top, line)


def _span(node: ast.stmt) -> int:
    return (node.end_lineno or node.lineno) - node.lineno


def _innermost_statement(tree: ast.Module, lineno: int) -> ast.stmt | None:
    best: ast.stmt | None = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.stmt)
            and not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
            and node.lineno <= lineno <= (node.end_lineno or node.lineno)
            and (best is None or _span(node) < _span(best))
        ):
            best = node
    return best


def unmarked_outcomes(source: str) -> list[ast.Constant]:
    """Outcome literals no citation or assumption marker vouches for.

    A marker vouches for a literal when it sits on the literal's own
    statement, or in the comment block directly above that statement, or
    directly above a compound statement enclosing it (the ``if`` that
    decides it, or the first ``if`` of a ladder of them). Proximity is
    not enough: a real compile's ``severity: "review"` was "covered" by
    an unrelated lodging-nights comment six lines up, so a looser rule
    would let the very decision this exists for pass unexamined.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []  # check_contracts reports it
    lines = source.splitlines()
    marked = {i + 1 for i, line in enumerate(lines) if _MARKER_RE.search(line)}
    parents = _parents(tree)
    missing: list[ast.Constant] = []
    for literal in outcome_literals(tree):
        stmt = _innermost_statement(tree, literal.lineno)
        if stmt is None:
            own = range(literal.lineno, literal.lineno + 1)
            anchors = [literal.lineno]
        else:
            own = range(stmt.lineno, (stmt.end_lineno or stmt.lineno) + 1)
            anchors = _anchors(stmt, parents)
        vouching = set(own)
        for anchor in anchors:
            vouching.update(_comment_block_above(lines, anchor))
        if not vouching & marked:
            missing.append(literal)
    return missing


# ───────── The compile report's Open questions ─────────

#: A bold line standing in for a heading (``**Open questions:**``).
_BOLD_HEADING_LEVEL = 99


def open_questions(report: str) -> list[str]:
    """The entries under the report's Open questions heading.

    If the section uses sub-headings (``### OQ-1: ...``, the shape a real
    report used), each sub-heading is one entry and the bullets beneath
    it are its detail. Otherwise each top-level list item is one. The
    section ends at the next heading of the same or a higher level.

    Fenced code blocks are skipped: a real report quoted the
    ``# rote: assumption:`` comment in a fence, and that ``#`` line read as
    a top-level heading that ended the section halfway through. Bold
    labels inside a sub-headed entry (``**What we chose:**``) are its
    detail, not more entries.
    """
    subheadings: list[str] = []
    bold_entries: list[str] = []
    items: list[str] = []
    level: int | None = None
    in_fence = False
    for line in report.splitlines():
        stripped = line.strip()
        if stripped.startswith(("```", "~~~")):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        hashes = re.match(r"^(#+)\s+(.*)", stripped)
        bold = bool(re.fullmatch(r"\*\*[^*]+\*\*:?", stripped))
        if hashes or bold:
            this = len(hashes.group(1)) if hashes else _BOLD_HEADING_LEVEL
            if level is not None and this > level:
                (subheadings if hashes else bold_entries).append(stripped)
                continue
            level = this if "open question" in stripped.lower() else None
            continue
        if level is not None and re.match(r"^([-*+]|\d+[.)])\s+\S", line):
            items.append(stripped)
    return subheadings or bold_entries or items


# ───────── The check ─────────


#: The node id every provenance finding carries.
PROVENANCE_NODE = "(provenance)"

#: Findings that state something false about the skill. Always errors.
FACTUAL_KINDS = frozenset({"citation_not_in_skill", "malformed_citation"})

#: Findings about a missing declaration. Repaired, then warnings unless strict.
REVIEW_KINDS = frozenset({"uncited_outcome"})


@dataclass(frozen=True)
class Assumption:
    """One ``# rote: assumption:`` marker: a decision the skill does not state."""

    path: str
    line: int
    text: str
    #: The node whose implementation holds it, or None for a module helper
    #: that no single node owns.
    node_id: str | None

    def as_dict(self) -> dict[str, object]:
        return {"node": self.node_id, "path": self.path, "line": self.line, "text": self.text}


def _owners(pipeline: Pipeline) -> dict[str, dict[str, str]]:
    """``{module path: {symbol: node id}}`` for every node's ``impl`` and
    legacy ``signature`` module."""
    owners: dict[str, dict[str, str]] = {}
    for node in pipeline.nodes:
        for ref in (node.impl, node.signature):
            if ref:
                rel, _, symbol = ref.partition(":")
                owners.setdefault(rel, {})[symbol] = node.id
    return owners


def _node_at(tree: ast.Module | None, line: int, symbols: dict[str, str]) -> str | None:
    """The node owning ``line``: its enclosing top-level function's (or
    signature class's) node, or the module's only node."""
    if tree is not None:
        for stmt in tree.body:
            if (
                isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
                and stmt.name in symbols
            ):
                first = min([stmt.lineno, *(d.lineno for d in stmt.decorator_list)])
                if first <= line <= (stmt.end_lineno or stmt.lineno):
                    return symbols[stmt.name]
    owners = set(symbols.values())
    return owners.pop() if len(owners) == 1 else None


#: Where a pipeline.yaml node entry starts, and a top-level key that ends
#: the ``nodes:`` list.
_YAML_NODE_ID = re.compile(r"^\s*-\s*id:\s*['\"]?(?P<id>[A-Za-z_][A-Za-z0-9_]*)")
_YAML_TOP_KEY = re.compile(r"^[A-Za-z_]")


def _pipeline_yaml_assumptions(root: Path, node_ids: set[str]) -> list[Assumption]:
    """Markers in pipeline.yaml: a YAML comment, or text inside a judge's
    ``signature_spec.prompt``. Each belongs to the node entry it sits in."""
    path = root / "pipeline.yaml"
    if not path.is_file():
        return []
    found: list[Assumption] = []
    current: str | None = None
    for number, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
    ):
        node = _YAML_NODE_ID.match(line)
        if node is not None and node.group("id") in node_ids:
            current = node.group("id")
        elif _YAML_TOP_KEY.match(line):
            current = None
        m = _ASSUMPTION_RE.search(line)
        if m is not None:
            found.append(Assumption("pipeline.yaml", number, m.group("text").strip(), current))
    return found


#: The deliverable's code directories whose markers reach the report:
#: implementations, and judge signatures (their prompts are judgment).
_MARKED_DIRS = ("extracted", "signatures")


def assumptions(pipeline: Pipeline, root: Path) -> list[Assumption]:
    """Every assumption marker in the deliverable, by file and line.

    Python under ``extracted/`` and ``signatures/``, plus ``pipeline.yaml``:
    a structural guess (a kind, a gate, a mandatory flag) is marked in a
    YAML comment on its node, and a judge prompt's guess in the prompt or
    its signature module. The compile report lists every one, so this
    scan is what keeps a judgment call from going unseen.
    """
    found: list[Assumption] = []
    owners = _owners(pipeline)
    node_ids = {n.id for n in pipeline.nodes}
    for sub in _MARKED_DIRS:
        base = root / sub
        for path in sorted(base.rglob("*.py")) if base.is_dir() else []:
            rel = str(path.relative_to(root))
            source = path.read_text(encoding="utf-8", errors="replace")
            try:
                tree: ast.Module | None = ast.parse(source)
            except SyntaxError:
                tree = None
            symbols = owners.get(rel, {})
            for number, line in enumerate(source.splitlines(), start=1):
                m = _ASSUMPTION_RE.search(line)
                if m is not None:
                    owner = _node_at(tree, number, symbols)
                    if owner is None and not symbols and path.stem in node_ids:
                        # signatures/<node id>.py for a signature_spec judge.
                        owner = path.stem
                    found.append(Assumption(rel, number, m.group("text").strip(), owner))
    found += _pipeline_yaml_assumptions(root, node_ids)
    return found


def provenance_findings(
    pipeline: Pipeline, root: Path, skill_dir: Path | None, *, strict: bool = False
) -> list[ContractFinding]:
    """Citations quote the skill and outcomes are marked.

    Assumptions need no check here: rote lists every one in
    compile-report.md itself (:mod:`rote.compiler.report`).
    """

    def _finding(kind: str, message: str, path: str) -> ContractFinding:
        severity = "error" if strict or kind in FACTUAL_KINDS else "warning"
        return ContractFinding(
            node_id=PROVENANCE_NODE, kind=kind, severity=severity, message=message, path=path
        )

    findings: list[ContractFinding] = []
    paragraphs = (
        skill_paragraphs(skill_dir) if skill_dir is not None and skill_dir.is_dir() else None
    )
    pure_modules = {
        node.impl.partition(":")[0]
        for node in pipeline.nodes
        if node.kind is NodeKind.PURE_FUNCTION and node.impl
    }
    extracted = root / "extracted"
    for path in sorted(extracted.rglob("*.py")) if extracted.is_dir() else []:
        rel = str(path.relative_to(root))
        source = path.read_text(encoding="utf-8", errors="replace")
        if paragraphs is not None:
            for cite in citations(source):
                if cite.quote is None:
                    findings.append(
                        _finding(
                            "malformed_citation",
                            f'{rel} line {cite.line}: a # skill: "..." quote is never closed. '
                            f'End it with a " at the end of a comment line',
                            rel,
                        )
                    )
                    continue
                problem = citation_problem(cite.quote, paragraphs)
                if problem is not None:
                    findings.append(
                        _finding(
                            "citation_not_in_skill",
                            f'{rel} line {cite.line} cites the skill as "{cite.quote}", but '
                            f"{problem}. Quote the sentence that decides the outcome "
                            f"verbatim, or if the skill does not state it, mark it "
                            f"'# rote: assumption: ...' (rote lists it under the "
                            f"report's Open questions)",
                            rel,
                        )
                    )
        if rel in pure_modules:
            for literal in unmarked_outcomes(source):
                findings.append(
                    _finding(
                        "uncited_outcome",
                        f"{rel} line {literal.lineno}: {literal.value!r} is an outcome the "
                        f'code decides, with no # skill: "..." citation or '
                        f"# rote: assumption: marker on its statement, or in the comment "
                        f"directly above it or above the if that decides it",
                        rel,
                    )
                )
    return findings
