"""System-1 prototype: a zero-inference SKILL.md step extractor.

EXPERIMENTAL. Not wired into `rote compile`. It reads a skill bundle and
produces, without any LLM call:

* the section tree (headings, line spans, body text),
* step units: leaf procedural sections, split further where the skill
  itself enumerates sub-steps (bold-numbered paragraphs, numbered lists
  with bold titles),
* per-unit node proposals with a kind and a confidence, using the signals
  the rubric's node-kinds.md names (tool calls, human gates, rubrics,
  loops, fixed transformations), including the rubric's split rule for
  "call a tool, then post-process",
* MANDATORY markers, MCP server/tool mentions, cited reference files,
* every literal threshold / constant the prose states (comparators,
  "batch of N", "last N days", "$25.00 or more", markdown rule tables),
  with source file and line.

Signals were written from skills/rote-compile/references/node-kinds.md and
generic markdown structure. Four of the five example skills were read while
writing them (a dev set); invoice-push was held out (only its headings were
seen) and is reported separately by score.py.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

# ── section tree ─────────────────────────────────────────────────────────

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")


@dataclass
class Section:
    level: int
    title: str
    start: int  # 1-based line of the heading
    end: int  # last line of the body (exclusive of children headings)
    body: str
    children: list[Section] = field(default_factory=list)
    parent: Section | None = field(default=None, repr=False)

    def path(self) -> list[str]:
        out, s = [], self
        while s is not None and s.level > 0:
            out.append(s.title)
            s = s.parent
        return list(reversed(out))


def parse_frontmatter(text: str) -> tuple[dict, str, int]:
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end != -1:
            fm = yaml.safe_load(text[4:end]) or {}
            rest = text[end + 4 :]
            offset = text[: end + 4].count("\n")
            return fm, rest, offset
    return {}, text, 0


def parse_sections(body: str, line_offset: int) -> Section:
    root = Section(0, "<root>", 0, 0, "")
    stack = [root]
    lines = body.splitlines()
    current = root
    buf: list[str] = []
    in_fence = False

    def flush(upto: int) -> None:
        current.body = "\n".join(buf).strip()
        current.end = upto

    for i, line in enumerate(lines, start=1 + line_offset):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        m = None if in_fence else _HEADING.match(line)
        if m:
            flush(i - 1)
            buf = []
            level = len(m.group(1))
            while stack[-1].level >= level:
                stack.pop()
            sec = Section(level, m.group(2).strip(), i, i, "", parent=stack[-1])
            stack[-1].children.append(sec)
            stack.append(sec)
            current = sec
        else:
            buf.append(line)
    flush(len(lines) + line_offset)
    return root


def walk(sec: Section):
    for c in sec.children:
        yield c
        yield from walk(c)


# ── signals (from node-kinds.md) ─────────────────────────────────────────

_META = re.compile(
    r"\b(trigger|routing|available tools|tools?$|limits|gotchas|related|objective|overview|"
    r"when to use|background|notes?$|examples?$|glossary|faq|reference|failure code)\b",
    re.I,
)


def _rx(*parts: str) -> re.Pattern[str]:
    return re.compile("|".join(parts), re.I)


SIGNALS: dict[str, re.Pattern[str]] = {
    "hitl_gate": _rx(
        r"\bask (the |a )?(user|duty manager|controller|bdr|human|operator|manager"
        r"|reviewer|approver)",
        r"\bpresent (the |a )?[\w\s-]{0,40}\b(to the (user|bdr|reviewer)|for (approval|review))",
        r"\bwait for (a |the )?(decision|approval|response|confirmation|sign-?off)",
        r"\bconfirm with the user",
        r"\bfor (user |human )?approval\b",
        r"\bmust be done manually\b",
        r"\breview gate\b",
        r"\bhandoff\b",
        r"\brequest from (the )?user\b",
        r"\bpost [\w\s]{0,30}\bto the (controller|user|reviewer|approver)",
    ),
    "agent_loop": _rx(
        r"\bautonomous loop\b",
        r"\biterative\b",
        r"\bbackfill",
        r"\buntil [\w\s']{0,60}\b(quota|is met|are met|enough|found|exhausted)",
        r"\bexplor(e|atory)\b",
        r"\b(external|internal|web) research\b",
        r"\bfor each (row|item|page|record) on the page\b",
    ),
    "llm_judge": _rx(
        r"\bclassif(y|ies|ication)\b",
        r"\bscor(e|ing|es)\b",
        r"\bvet(ting|ted)?\b",
        r"\brubric\b",
        r"\bpersonali[sz]",
        r"\bdraft\b",
        r"\bjudg(e|ment)\b",
        r"\bassess\b",
        r"\bsentiment\b",
        r"\bwould this person\b",
        r"\bread the entire thread\b",
        r"\bextract for each\b",
    ),
    "fetch": _rx(
        r"\bmcp\b",
        r"\b(read|pull|fetch|search|query|look up|retrieve|list)\b[^.\n]{0,60}"
        r"\b(slack|gmail|drive|sheet|hubspot|zoominfo|salesforce|api|channel|inbox"
        r"|threads?|messages?)\b",
        r"\bsearch (query|gmail|for)\b",
        r"\bnavigate to\b",
        r"\bfile id\b",
    ),
    "write": _rx(
        r"\bsave (the|it|to|as)\b",
        r"\bupload\b",
        r"\bupsert\b",
        r"\bcreate (a |an |the )?[\w\s-]{0,30}"
        r"\b(list|template|record|folder|sheet|contact|ticket)s?\b",
        r"\bfile the\b",
        r"\bsend (the|an?|it)\b",
        r"\bclick\b",
        r"\bpush to\b",
        r"\bwrite [\w\s]{0,20}\bto (the )?[\w\s]{0,20}\b(folder|drive|sheet|crm|hubspot)\b",
    ),
    "pure_function": _rx(
        r"\bconvert\b",
        r"\bcount\b",
        r"\bgroup\b",
        r"\bsort\b",
        r"\btotal\b",
        r"\bround\b",
        r"\bcalculat",
        r"\bcompute\b",
        r"\bfilter\b",
        r"\bdedup",
        r"\bduplicates?\b",
        r"\bmatch(es|ed)?\b",
        r"\bformat\b",
        r"\brender\b",
        r"\bgenerate (a |an |the )?[\w\s-]{0,30}\b(html|report|table|summary|file|sheet)\b",
        r"\bparse\b",
        r"\bflag\b",
        r"\bcheck\b",
        r"\bbreak (it )?down\b",
        r"\bbuild the\b",
        r"\bexecutive[- ]level summary\b",
        r"\bkpi\b",
    ),
}

_MANDATORY = re.compile(
    r"\bMANDATORY\b|\bmust (always )?run\b"
    r"|\bthis (check|phase|step) is (mandatory|required)\b|\brequired before\b",
    re.I,
)
_DASHES = r"\u2014\u2013-"  # em dash, en dash, hyphen: skills separate titles with any of them
_MCP_PAREN = re.compile(
    rf"\(([A-Z][\w ]{{1,30}}?) MCP\s*(?:[{_DASHES}]\s*`?([a-z_][a-z0-9_]+)`?)?\)"
)
_MCP_USING = re.compile(r"\b(?:using|via|with) the ([A-Z][\w ]{1,30}?) MCP(?: tool)?\b")
_BACKTICK_TOOL = re.compile(r"`([a-z][a-z0-9]*(?:_[a-z0-9]+){1,6})`")
_REF = re.compile(r"\(?(references/[\w./-]+\.md)\)?")


def slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return s or "step"


# ── constants / thresholds ───────────────────────────────────────────────

_NUM = r"\$?\d[\d,]*(?:\.\d+)?%?"
_CONST_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("comparator", re.compile(rf"(>=|<=|≥|≤|>|<)\s*({_NUM})")),
    (
        "phrase_more_than",
        re.compile(rf"\b(more than|less than|over|under|at least|up to|within)\s+({_NUM})", re.I),
    ),
    ("or_more", re.compile(rf"({_NUM})\s+or (more|less|fewer|higher|lower)\b", re.I)),
    (
        "batch",
        re.compile(
            rf"\b(?:batch(?:es)? of|in batches of)\s+({_NUM})"
            rf"|({_NUM})\s+(?:contacts |rows |items |records )?per (?:call|batch|page|request)",
            re.I,
        ),
    ),
    (
        "window",
        re.compile(
            rf"\b(?:last|past|within|newer_than:)\s*({_NUM})\s*(d\b|days?|hours?|weeks?|months?|minutes?|seconds?)",
            re.I,
        ),
    ),
    ("count_recent", re.compile(rf"\b({_NUM})\s+most recent\b", re.I)),
    ("window_days", re.compile(rf"\b({_NUM})[- ](day|hour|minute|second)s?\b", re.I)),
    ("range", re.compile(rf"\b({_NUM})\s*[-\u2013]\s*({_NUM})\s*=", re.I)),
    ("plus", re.compile(rf"\b({_NUM})\+")),
    ("page_size", re.compile(rf"\b(?:page size|size) (?:to|of)\s+({_NUM})", re.I)),
    (
        "id_literal",
        re.compile(
            r"\b(?:file id|channel id|folder id|id)[:\s(`]+\s*`?([A-Za-z0-9_-]{8,})`?", re.I
        ),
    ),
]


def to_number(tok: str) -> float | None:
    t = tok.replace("$", "").replace(",", "").rstrip("%")
    try:
        return float(t)
    except ValueError:
        return None


@dataclass
class Constant:
    kind: str
    raw: str
    value: float | str | None
    file: str
    line: int


def extract_constants(text: str, file: str, first_line: int) -> list[Constant]:
    out: list[Constant] = []
    in_fence = False
    for i, line in enumerate(text.splitlines(), start=first_line):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        for kind, pat in _CONST_PATTERNS:
            for m in pat.finditer(line):
                groups = [g for g in m.groups() if g]
                if kind == "id_literal":
                    out.append(Constant(kind, m.group(0), groups[0], file, i))
                    continue
                for g in groups:
                    n = to_number(g)
                    if n is not None:
                        out.append(Constant(kind, m.group(0).strip(), n, file, i))
        # markdown table rows with numeric cells are rule tables
        if line.strip().startswith("|") and not re.match(r"^\s*\|[\s:|-]+\|\s*$", line):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            for c in cells[1:]:
                for tok in re.findall(_NUM, c):
                    n = to_number(tok)
                    if n is not None:
                        out.append(Constant("table_cell", f"{cells[0]} | {c}", n, file, i))
    return out


# ── step units and node proposals ────────────────────────────────────────


@dataclass
class Proposal:
    id: str
    kind: str
    confidence: float
    reasons: list[str]


@dataclass
class Unit:
    section: str  # the SKILL.md heading this unit came from (source.section)
    title: str
    path: list[str]
    line: int
    text: str
    mandatory: bool
    mcp: list[dict]
    tools: list[str]
    references: list[str]
    proposals: list[Proposal]


def _signal_hits(text: str, heading: str) -> dict[str, list[str]]:
    hits: dict[str, list[str]] = {}
    for name, pat in SIGNALS.items():
        found = [m.group(0) for m in pat.finditer(text)]
        found += [f"[heading] {m.group(0)}" for m in pat.finditer(heading)] * 2
        if found:
            hits[name] = found
    return hits


def propose(title: str, text: str, tool_names: set[str]) -> list[Proposal]:
    """Map one unit's signals onto node proposals, applying node-kinds.md's rules.

    * a human gate wins outright (the step exists to park);
    * an exploratory loop wins over its own tool calls (tools belong to the loop);
    * otherwise a tool call is a bare external_call, and anything the step
      does to the result is a separate judge or pure_function (the
      "mcp-bound node is a BARE tool call" split);
    * a side-effecting write is its own external_call.
    """
    hits = _signal_hits(text, title)
    backticked = {t for t in _BACKTICK_TOOL.findall(text) if t in tool_names}
    if backticked:
        hits.setdefault("fetch", []).extend(f"`{t}`" for t in sorted(backticked))
    score = {k: len(v) for k, v in hits.items()}
    total = sum(score.values()) or 1
    lead = rf"^(step|phase)\s*[\w.-]*\s*[:{_DASHES}]\s*"
    base = slug(re.sub(lead, "", title, flags=re.I))[:40]

    def conf(k: str) -> float:
        # share of signal mass for this kind, damped when there is little signal
        share = score.get(k, 0) / total
        mass = min(1.0, total / 4)
        return round(share * 0.6 + mass * 0.4, 2)

    if score.get("hitl_gate", 0) >= 1 and score["hitl_gate"] >= score.get("llm_judge", 0):
        return [Proposal(f"{base}_gate", "hitl_gate", conf("hitl_gate"), hits["hitl_gate"][:3])]
    if score.get("agent_loop", 0) >= 1:
        return [Proposal(f"{base}_loop", "agent_loop", conf("agent_loop"), hits["agent_loop"][:3])]

    out: list[Proposal] = []
    if score.get("fetch"):
        out.append(Proposal(f"fetch_{base}", "external_call", conf("fetch"), hits["fetch"][:3]))
    if score.get("llm_judge"):
        out.append(Proposal(f"{base}_judge", "llm_judge", conf("llm_judge"), hits["llm_judge"][:3]))
    if score.get("pure_function"):
        out.append(
            Proposal(f"{base}", "pure_function", conf("pure_function"), hits["pure_function"][:3])
        )
    if score.get("write"):
        out.append(Proposal(f"{base}_write", "external_call", conf("write"), hits["write"][:3]))
    if not out:
        out.append(Proposal(base, "pure_function", 0.1, ["no signal: default"]))
    return out


def _subunits(sec: Section) -> list[tuple[str, int, str]]:
    """Split a leaf section where the skill itself enumerates sub-steps."""
    lines = sec.body.splitlines()
    starts: list[tuple[int, str]] = []
    for idx, line in enumerate(lines):
        m = re.match(r"^\*\*\d+[.)]\s*(.+?)\*\*", line) or re.match(
            r"^\d+[.)]\s+\*\*(.+?)\*\*", line
        )
        if m:
            starts.append((idx, re.sub(r"\s*\(.*$", "", m.group(1)).strip(" :\u2014-")))
    if len(starts) < 2:
        return [(sec.title, sec.start, sec.body)]
    out = []
    preamble = "\n".join(lines[: starts[0][0]])
    for n, (idx, title) in enumerate(starts):
        stop = starts[n + 1][0] if n + 1 < len(starts) else len(lines)
        chunk = "\n".join(lines[idx:stop])
        out.append(
            (title, sec.start + 1 + idx, chunk + ("\n" + preamble if preamble.strip() else ""))
        )
    return out


def _is_meta(sec: Section) -> bool:
    return bool(_META.search(sec.title)) or sec.level == 1


def _tool_registry(root: Section) -> dict[str, str]:
    """Tool names declared in tool tables, mapped to the heading they sit under."""
    reg: dict[str, str] = {}
    for s in walk(root):
        for line in s.body.splitlines():
            if line.strip().startswith("|"):
                for t in _BACKTICK_TOOL.findall(line):
                    reg[t] = slug(s.title.split("(")[0])
    return reg


@dataclass
class Extraction:
    skill: str
    name: str
    description: str
    sections: list[dict]
    units: list[Unit]
    constants: list[Constant]
    tool_registry: dict[str, str]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


def extract(skill_dir: Path) -> Extraction:
    text = (skill_dir / "SKILL.md").read_text()
    fm, body, offset = parse_frontmatter(text)
    root = parse_sections(body, offset)
    registry = _tool_registry(root)
    units: list[Unit] = []
    for sec in walk(root):
        if _is_meta(sec) or any(_is_meta(a) and a.level > 1 for a in _ancestors(sec)):
            continue
        procedural_children = [c for c in sec.children if not _is_meta(c)]
        if procedural_children and len(sec.body.split()) < 25:
            continue  # a container ("## Steps", "## Pipeline"); its children are the steps
        hitl_section = bool(SIGNALS["hitl_gate"].search(sec.title)) or bool(
            SIGNALS["hitl_gate"].search(sec.body.split("\n\n")[0] if sec.body else "")
        )
        parts = [(sec.title, sec.start, sec.body)] if hitl_section else _subunits(sec)
        for title, line, chunk in parts:
            mcp = [
                {"server": slug(m.group(1)), "tool": m.group(2) if m.re.groups > 1 else None}
                for m in list(_MCP_PAREN.finditer(chunk)) + list(_MCP_USING.finditer(chunk))
            ]
            for t in sorted(set(_BACKTICK_TOOL.findall(chunk)) & registry.keys()):
                mcp.append({"server": registry[t], "tool": t})
            units.append(
                Unit(
                    section=sec.title,
                    title=title,
                    path=sec.path(),
                    line=line,
                    text=chunk,
                    mandatory=bool(_MANDATORY.search(sec.title) or _MANDATORY.search(chunk)),
                    mcp=mcp,
                    tools=sorted(set(_BACKTICK_TOOL.findall(chunk)) & registry.keys()),
                    references=sorted(set(_REF.findall(chunk))),
                    proposals=propose(title, chunk, set(registry)),
                )
            )
    constants = extract_constants(body, "SKILL.md", offset + 1)
    ref_dir = skill_dir / "references"
    if ref_dir.is_dir():
        for ref in sorted(ref_dir.glob("*.md")):
            constants += extract_constants(ref.read_text(), f"references/{ref.name}", 1)
    sections = [
        {"level": s.level, "title": s.title, "line": s.start, "meta": _is_meta(s)}
        for s in walk(root)
    ]
    return Extraction(
        skill=str(skill_dir),
        name=str(fm.get("name", skill_dir.parent.name)),
        description=str(fm.get("description", "")).strip(),
        sections=sections,
        units=units,
        constants=constants,
        tool_registry=registry,
    )


def _ancestors(sec: Section):
    p = sec.parent
    while p is not None and p.level > 0:
        yield p
        p = p.parent


if __name__ == "__main__":
    ex = extract(Path(sys.argv[1]))
    if "--json" in sys.argv:
        print(ex.to_json())
    else:
        for u in ex.units:
            kinds = ", ".join(f"{p.kind}({p.confidence})" for p in u.proposals)
            flags = (" MANDATORY" if u.mandatory else "") + (f" mcp={u.mcp}" if u.mcp else "")
            print(f"L{u.line:<4} {u.title[:55]:<55} -> {kinds}{flags}")
        print(
            f"{len(ex.units)} units, {sum(len(u.proposals) for u in ex.units)} proposals, "
            f"{len(ex.constants)} constants"
        )
