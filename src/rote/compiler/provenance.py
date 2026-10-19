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
  not state, listed under the compile report's Open questions.

These checks keep the markers honest. They are deliberately mechanical:
a citation has to be a real, non-trivial quote of the skill's prose, and
an outcome literal with no marker at all is reported rather than trusted.
A review showed every looser version could be satisfied without telling
a reviewer anything (one-word quotes, quoting a heading, fragments
matched out of order, no comment at all, one Open question for many
assumptions), so each of those is a test.
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
        "alert",
        "category",
        "classification",
        "color",
        "colour",
        "decision",
        "disposition",
        "flag",
        "label",
        "level",
        "outcome",
        "priority",
        "result",
        "route",
        "routing",
        "severity",
        "status",
        "tier",
        "verdict",
    }
)

#: How far above an outcome's statement a covering marker may sit.
MARKER_REACH_LINES = 10

_CITATION_START = re.compile(r"#[ \t]*skill:[ \t]*")
_ASSUMPTION_RE = re.compile(r"#\s*rote:\s*assumption:\s*(?P<text>.+)")
_MARKER_RE = re.compile(r"#\s*(skill:|rote:\s*assumption:)")


def _normalized(text: str) -> str:
    """Lowercase, markdown emphasis and code marks dropped, whitespace collapsed."""
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
    """
    paragraphs: list[str] = []
    for path in sorted(skill_dir.rglob("*.md")):
        current: list[str] = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip() or _is_heading(line):
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


def _labels(value: ast.expr | None) -> list[ast.Constant]:
    """String constants a value can evaluate to (through ``a if c else b``)."""
    if isinstance(value, ast.IfExp):
        return _labels(value.body) + _labels(value.orelse)
    if (
        isinstance(value, ast.Constant)
        and isinstance(value.value, str)
        and value.value.strip()
        and "\n" not in value.value
        and len(value.value) <= 80
    ):
        return [value]
    return []


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

    * ``return "block"`` (a function that returns a label is a classifier);
    * a dict value under an outcome key (``{"severity": "block"}``);
    * a keyword argument with an outcome name (``Finding(severity="block")``);
    * an assignment to an outcome name (``status = "flagged"``,
      ``row["alert_level"] = "missort_alert"``).
    """
    found: list[ast.Constant] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Return):
            found += _labels(node.value)
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if (
                    isinstance(key, ast.Constant)
                    and isinstance(key.value, str)
                    and _is_outcome_name(key.value)
                ):
                    found += _labels(value)
        elif (
            (isinstance(node, ast.keyword) and _is_outcome_name(node.arg))
            or (
                isinstance(node, ast.Assign)
                and any(_is_outcome_name(_target_name(t)) for t in node.targets)
            )
            or (isinstance(node, ast.AnnAssign) and _is_outcome_name(_target_name(node.target)))
        ):
            found += _labels(node.value)
    return found


def _statement_spans(tree: ast.Module) -> list[tuple[int, int]]:
    return [
        (n.lineno, n.end_lineno or n.lineno)
        for n in ast.walk(tree)
        if isinstance(n, ast.stmt) and not isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
    ]


def unmarked_outcomes(source: str) -> list[ast.Constant]:
    """Outcome literals with no citation or assumption marker covering them.

    A marker covers the code below it up to the next blank line, the way a
    comment above an ``if``/``elif`` chain describes the whole chain, and
    the statement it trails, but reaches at most :data:`MARKER_REACH_LINES`
    lines: one marker at the top of a long function does not vouch for
    every label in it. So for each outcome literal: take the
    innermost statement holding it, and look for a marker on its lines or
    on any line above it with no blank line in between.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []  # check_contracts reports it
    lines = source.splitlines()
    marked = {i + 1 for i, line in enumerate(lines) if _MARKER_RE.search(line)}
    spans = _statement_spans(tree)
    missing: list[ast.Constant] = []
    for literal in outcome_literals(tree):
        holding = [s for s in spans if s[0] <= literal.lineno <= s[1]]
        start, end = (
            min(holding, key=lambda s: s[1] - s[0]) if holding else (literal.lineno, literal.lineno)
        )
        top = start
        while top > 1 and start - top < MARKER_REACH_LINES and lines[top - 2].strip():
            top -= 1
        if not any(top <= line <= end for line in marked):
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
    """
    subheadings: list[str] = []
    items: list[str] = []
    level: int | None = None
    for line in report.splitlines():
        stripped = line.strip()
        hashes = re.match(r"^(#+)\s+(.*)", stripped)
        bold = bool(re.fullmatch(r"\*\*[^*]+\*\*:?", stripped))
        if hashes or bold:
            this = len(hashes.group(1)) if hashes else _BOLD_HEADING_LEVEL
            if level is not None and this > level:
                subheadings.append(stripped)
                continue
            level = this if "open question" in stripped.lower() else None
            continue
        if level is not None and re.match(r"^([-*+]|\d+[.)])\s+\S", line):
            items.append(stripped)
    return subheadings or items


# ───────── The check ─────────


def _finding(kind: str, message: str, path: str) -> ContractFinding:
    return ContractFinding(
        node_id="(provenance)", kind=kind, severity="error", message=message, path=path
    )


def provenance_findings(
    pipeline: Pipeline, root: Path, skill_dir: Path | None
) -> list[ContractFinding]:
    """Citations quote the skill, outcomes are marked, assumptions are listed."""
    findings: list[ContractFinding] = []
    paragraphs = (
        skill_paragraphs(skill_dir) if skill_dir is not None and skill_dir.is_dir() else None
    )
    pure_modules = {
        node.impl.partition(":")[0]
        for node in pipeline.nodes
        if node.kind is NodeKind.PURE_FUNCTION and node.impl
    }
    assumptions: list[tuple[str, str]] = []
    extracted = root / "extracted"
    for path in sorted(extracted.rglob("*.py")) if extracted.is_dir() else []:
        rel = str(path.relative_to(root))
        source = path.read_text(encoding="utf-8", errors="replace")
        assumptions += [(rel, m.group("text").strip()) for m in _ASSUMPTION_RE.finditer(source)]
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
                            f"'# rote: assumption: ...' and list it under Open questions",
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
                        f"# rote: assumption: marker on its statement or in the comment "
                        f"and code directly above it (up to a blank line, "
                        f"{MARKER_REACH_LINES} lines "
                        f"at most)",
                        rel,
                    )
                )
    if assumptions:
        report = root / REPORT_FILENAME
        listed = (
            open_questions(report.read_text(encoding="utf-8", errors="replace"))
            if report.is_file()
            else []
        )
        if len(listed) < len(assumptions):
            sample = "; ".join(f"{rel}: {text}" for rel, text in assumptions[:3])
            findings.append(
                _finding(
                    "unlisted_assumption",
                    f"the extracted code records {len(assumptions)} assumption(s) the skill "
                    f"does not state ({sample}), but {REPORT_FILENAME} lists "
                    f"{len(listed)} Open question(s). Give each assumption its own entry",
                    REPORT_FILENAME,
                )
            )
    return findings
