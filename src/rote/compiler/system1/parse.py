"""Read a skill bundle into steps, signals, constants and rule tables. No inference.

This is the measured extractor from the System-1 research
(``docs/design/system1-compile.md``; first measured in PR #145): the
signal patterns are unchanged from the version that was scored, so the
numbers in that document describe this code.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import yaml

# ── section tree ──────────────────────────────────────────────────────────

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")


@dataclass
class Section:
    level: int
    title: str
    start: int  # 1-based line of the heading in SKILL.md
    body: str = ""
    children: list[Section] = field(default_factory=list)
    parent: Section | None = field(default=None, repr=False)

    def ancestors(self) -> Iterator[Section]:
        p = self.parent
        while p is not None and p.level > 0:
            yield p
            p = p.parent


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str, int]:
    """(frontmatter, body, number of lines the frontmatter took)."""
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end != -1:
            loaded = yaml.safe_load(text[4:end]) or {}
            fm = loaded if isinstance(loaded, dict) else {}
            return fm, text[end + 4 :], text[: end + 4].count("\n")
    return {}, text, 0


def parse_sections(body: str, line_offset: int) -> Section:
    root = Section(0, "<root>", 0)
    stack = [root]
    current = root
    buf: list[str] = []
    in_fence = False
    for i, line in enumerate(body.splitlines(), start=1 + line_offset):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        m = None if in_fence else _HEADING.match(line)
        if m is None:
            buf.append(line)
            continue
        current.body = "\n".join(buf).strip()
        buf = []
        level = len(m.group(1))
        while stack[-1].level >= level:
            stack.pop()
        sec = Section(level, m.group(2).strip(), i, parent=stack[-1])
        stack[-1].children.append(sec)
        stack.append(sec)
        current = sec
    current.body = "\n".join(buf).strip()
    return root


def walk(sec: Section) -> Iterator[Section]:
    for c in sec.children:
        yield c
        yield from walk(c)


# ── signals (written from node-kinds.md) ──────────────────────────────────

#: Headings that are about the skill, not a step of it.
_META = re.compile(
    r"\b(trigger phrases?|triggers|routing|available tools|limits|gotchas|related skills?"
    r"|glossary|faq|when to use)\b",
    re.I,
)
#: Single-word meta headings, matched only as the whole heading: "Notes" is
#: meta, "Step 2: Tidy the notes" is a step.
_META_WHOLE = re.compile(
    r"^\W*(?:tools|notes|examples?|objective|overview|background|references?)\W*$", re.I
)


def _rx(*parts: str) -> re.Pattern[str]:
    return re.compile("|".join(parts), re.I)


#: Not a person: asking one of these is a call, not a human gate.
_PERSON = (
    r"(?!(?:the |a |an )?(?:api|tool|model|system|server|service|agent|script|mcp|llm"
    r"|endpoint|bot)\b)(?:the |a |an )?"
)
#: Where a retrieval goes: a named thing (#channel, @handle, a capitalized
#: name such as Zendesk) or a word for a service. "From the table" or "in the
#: row" is data the step already has, and the skill's own references are not
#: a service either.
_SOURCE = (
    r"(?!(?:the\s+)?\[?references/|\w+\.md\b)(?:the\s+|a\s+|an\s+|our\s+|your\s+)?"
    r"(?:[#@][\w-]+|(?-i:[A-Z])[\w.-]*"
    r"|(?:[\w-]+\s+){0,2}(?:portal|api|database|db|server|endpoint|inbox|channel|crm|system"
    r"|service|site|website|account|bucket|drive|feed|spreadsheet|sheet|folder|warehouse"
    r"|repository|repo|queue|mailbox|calendar|store)s?\b)"
)

SIGNALS: dict[str, re.Pattern[str]] = {
    "hitl_gate": _rx(
        # Anyone asked to act on the run: "ask the duty manager to provide",
        # "ask the finance lead to sign off". Asking someone FOR a fact
        # ("ask finance for the rate") is an error path, not a gate.
        rf"\bask {_PERSON}[\w -]{{1,40}}? to (?:confirm|approve|review|sign[ -]?off"
        r"|provide|decide|choose|check)\b",
        rf"\bask {_PERSON}[\w -]{{1,40}}? for (?:approval|sign[ -]?off|a decision"
        r"|confirmation|review)\b",
        r"\bpresent (?:the |a )?[\w\s-]{0,40}\bto (?:the )?[\w -]{1,30}?"
        r"(?: for (?:approval|review|sign[ -]?off)|$|[.,])",
        r"\bwait for (a |the )?(decision|approval|response|confirmation|sign[ -]?off)",
        r"\bconfirm with (?:the |a |an )?\w+",
        r"\bsigns? off\b|\bsign-off\b",
        r"\b(?:needs?|requires?|get|obtain) (?:\w+ ){0,2}approval\b",
        r"\bfor (\w+ )?approval\b",
        # Work the skill says a person does, not the run.
        r"\b(?:must|has to|needs to|should) be done (?:manually|by hand|by (?:a|the) \w+)\b",
        r"\bhand(?:s|ing)? (?:it |this |them )?(?:off|over)\b|\bhandoff\b",
        r"\bgate\b",
        r"\brequest from (the )?user\b",
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
    ),
    "fetch": _rx(
        r"\bmcp\b",
        # Any retrieval verb with a source, no vendor list: "pull open tickets
        # from Zendesk", "read the 50 most recent messages from #deal-intake".
        # A lookup in the skill's own reference files is not a call.
        r"\b(?:read|pull|fetch|query|look up|retrieve|download|export)\b[^.\n]{0,60}"
        rf"\b(?:from|in|via|using)\s+{_SOURCE}",
        r"\bsearch\b[^.\n]{0,40}\bfor\b|\bsearch query\b",
        r"\bnavigate to\b",
    ),
    "write": _rx(
        r"\bsave (the|it|to|as)\b",
        r"\bupload\b",
        r"\bupsert\b",
        r"\bpost(?:s|ed)?\b[^.\n]{0,40}\bto\b|\bpublish\b|\bnotify\b",
        r"\bcreate (a |an |the )?[\w\s-]{0,30}"
        r"\b(list|template|record|folder|sheet|contact|ticket)s?\b",
        r"\bfile the\b",
        r"\bsend (the|an?|it)\b",
        r"\bclick\b",
        r"\bpush to\b",
        rf"\bwrite [\w\s]{{0,20}}\bto {_SOURCE}",
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
        r"\bkpi\b",
    ),
}

MANDATORY = re.compile(
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
_REF = re.compile(r"(references/[\w./-]+\.md)")
_STEP_NO = re.compile(r"^(?:step|phase)\s+(\d+)\b", re.I)


#: A container heading whose sub-steps run once per item.
_PER_ITEM = re.compile(r"\bfor (?:each|every)\b|\beach (?!other\b|time\b)[a-z]+\b", re.I)


def slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return s or "step"


# ── constants and rule tables ─────────────────────────────────────────────

_NUM = r"\$?\d[\d,]*(?:\.\d+)?%?"
_CONST_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("threshold", re.compile(rf"(>=|<=|≥|≤|>|<)\s*({_NUM})")),
    (
        "threshold",
        re.compile(rf"\b(more than|less than|over|under|at least|up to|within)\s+({_NUM})", re.I),
    ),
    ("threshold", re.compile(rf"({_NUM})\s+or (more|less|fewer|higher|lower)\b", re.I)),
    (
        "batch_size",
        re.compile(
            rf"\b(?:batch(?:es)? of|in batches of)\s+({_NUM})"
            rf"|({_NUM})\s+(?:contacts |rows |items |records )?per (?:call|batch|page|request)",
            re.I,
        ),
    ),
    (
        "window",
        re.compile(
            rf"\b(?:last|past|within)\s*({_NUM})\s*"
            r"(d\b|days?|hours?|weeks?|months?|minutes?|seconds?)",
            re.I,
        ),
    ),
    ("limit", re.compile(rf"\b({_NUM})\s+most recent\b", re.I)),
    ("window", re.compile(rf"\b({_NUM})[- ](day|hour|minute|second)s?\b", re.I)),
    ("page_size", re.compile(rf"\b(?:page size|size) (?:to|of)\s+({_NUM})", re.I)),
]
_ID_LITERAL = re.compile(
    r"\b(file id|channel id|folder id)[:\s(`]+\s*`?([A-Za-z0-9_-]{8,})`?", re.I
)


#: A `Label: value` line, the way a skill writes a call's fixed inputs.
_STATED = re.compile(
    r"^[ \t]*(?:[-*][ \t]+)?(?:\*\*)?([A-Z][A-Za-z]*(?:[ /][A-Za-z]+){0,3}?)(?:\*\*)?[ \t]*:"
    r"[ \t]*(?:\*\*)?[ \t]*`?([^`\n]{1,120}?)`?[ \t]*$",
    re.M,
)
#: The label must name a parameter, not introduce prose ("Include:", "Note:").
_PARAMETER_NOUN = re.compile(
    r"\b(query|id|ids|tab|label|labels|channel|folder|file|sheet|range|filter|url|path"
    r"|name|format|window|list|table|database|project|board|team|calendar|mailbox|repo)\b",
    re.I,
)


def stated_parameters(text: str, first_line: int) -> list[Constant]:
    """Every `Label: value` line whose label names a parameter, in order."""
    out: list[Constant] = []
    for m in _STATED.finditer(text):
        label, value = m.group(1).strip(), m.group(2).strip().strip("*").strip()
        if not value or not _PARAMETER_NOUN.search(label):
            continue
        name = slug(label)
        if any(c.name == name for c in out):
            continue
        line = first_line + text[: m.start()].count("\n")
        out.append(Constant(name, value, m.group(0).strip(), line))
    return out


def to_number(token: str) -> int | float | None:
    t = token.replace("$", "").replace(",", "").rstrip("%")
    try:
        value = float(t)
    except ValueError:
        return None
    return int(value) if value.is_integer() and "." not in t else value


@dataclass(frozen=True)
class Constant:
    """One literal the prose states, named by what kind of literal it is."""

    name: str
    value: int | float | str
    phrase: str
    line: int


def constants_in(text: str, first_line: int) -> list[Constant]:
    """Every literal threshold, window, batch size and id in ``text``, in order, deduplicated."""
    out: list[Constant] = []
    seen: set[tuple[str, object]] = set()
    for i, line in enumerate(text.splitlines(), start=first_line):
        for name, pat in _CONST_PATTERNS:
            for m in pat.finditer(line):
                for g in (g for g in m.groups() if g):
                    n = to_number(g)
                    if n is None or (name, n) in seen:
                        continue
                    seen.add((name, n))
                    out.append(Constant(name, n, m.group(0).strip(), i))
        for m in _ID_LITERAL.finditer(line):
            name = slug(m.group(1))
            if (name, m.group(2)) not in seen:
                seen.add((name, m.group(2)))
                out.append(Constant(name, m.group(2), m.group(0).strip(), i))
    return out


_TABLE_SEP = re.compile(r"^\s*\|[\s:|-]+\|\s*$")
_MONEY = re.compile(r"^\$?(-?\d[\d,]*(?:\.\d+)?)%?\b")
_CODE = re.compile(r"^`([A-Za-z0-9_.-]+)`$")


@dataclass
class RuleTable:
    """A markdown table whose numeric columns become ``{key: Decimal}`` maps."""

    file: str
    heading: str
    line: int
    header: list[str]
    rows: list[list[str]]
    raw_rows: list[str]

    def columns(self) -> dict[str, dict[str, Decimal]]:
        code_cols = [
            c
            for c in range(len(self.header))
            if self.rows and all(c < len(r) and _CODE.match(r[c]) for r in self.rows)
        ]
        key_col = code_cols[0] if code_cols else 0
        out: dict[str, dict[str, Decimal]] = {}
        for c in range(len(self.header)):
            if c == key_col:
                continue
            col: dict[str, Decimal] = {}
            for r in self.rows:
                if c >= len(r) or key_col >= len(r):
                    continue
                m = _MONEY.match(r[c].strip())
                if m is None:
                    continue
                try:
                    value = Decimal(m.group(1).replace(",", ""))
                except InvalidOperation:
                    continue
                code = _CODE.match(r[key_col])
                col[code.group(1) if code else r[key_col]] = value
            if col and len(col) >= max(1, len(self.rows) // 2):
                out[self.header[c]] = col
        return out


#: What a reference file may be called for its name to reach generated code.
#: A skill is untrusted input (invariant #7): a newline in a file name
#: spliced into a comment would become a line of code.
_SAFE_REFERENCE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,120}\.md")


class UnsafeSkillText(ValueError):
    """A skill name or string that cannot safely reach generated code."""


def safe_reference_name(name: str) -> str:
    if not _SAFE_REFERENCE_NAME.fullmatch(name):
        raise UnsafeSkillText(
            f"reference file name {name!r} is not allowed in a --system1 compile: use "
            "letters, digits, '.', '_' and '-' only"
        )
    return name


def tables_in(path: Path, label: str) -> list[RuleTable]:
    out: list[RuleTable] = []
    heading = ""
    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("#"):
            heading = line.lstrip("#").strip()
        if line.strip().startswith("|") and i + 1 < len(lines) and _TABLE_SEP.match(lines[i + 1]):
            header = [c.strip() for c in line.strip().strip("|").split("|")]
            rows: list[list[str]] = []
            raw: list[str] = []
            j = i + 2
            while j < len(lines) and lines[j].strip().startswith("|"):
                raw.append(lines[j].strip())
                rows.append([c.strip() for c in lines[j].strip().strip("|").split("|")])
                j += 1
            out.append(RuleTable(label, heading, i + 1, header, rows, raw))
            i = j
            continue
        i += 1
    return out


def _is_sample(t: RuleTable) -> bool:
    """An illustration, not a rule: rows numbered 1, 2, 3, or cells elided with "...".

    A sample output table in a skill ("| 1 | ... | 98 | Ideal |") looks numeric
    but states no rule, and its elided cells cannot be cited."""
    keys = [r[0] for r in t.rows if r]
    numbered = bool(keys) and all(k.strip().isdigit() for k in keys)
    elided = any("..." in cell or "\u2026" in cell for r in t.rows for cell in r)
    return numbered or elided


# ── step units ────────────────────────────────────────────────────────────


@dataclass
class Unit:
    """One step of the skill: a leaf procedural section, or a sub-step the skill enumerates."""

    index: int
    section: str  # the SKILL.md heading this came from: source.section
    title: str
    step_number: int | None  # "Step 4" / "Phase 4" in the heading, if any
    line: int
    text: str
    sibling_group: int | None  # units split out of one section share a group id
    mandatory: bool
    mcp: list[tuple[str, str | None]]  # (server, tool)
    tools: list[str]
    references: list[str]
    signals: dict[str, list[str]]
    constants: list[Constant]
    #: Parameters the step states as `Label: value` lines ("Search query:
    #: label:dock-activity", "File ID: ...", "Tab: ..."): a call's arguments.
    stated: list[Constant] = field(default_factory=list)
    #: A heading whose sub-steps run once per item ("Process each invoice row").
    loop_container: bool = False
    #: The loop container unit this sub-step runs inside, if any.
    loop_parent: int | None = None


@dataclass
class SkillParse:
    name: str
    description: str
    preamble: str  # the H1 body: where cross-step ordering is usually stated
    units: list[Unit]
    tool_registry: dict[str, str]  # tool name -> server (from tool tables)
    tables: list[RuleTable]


def _is_meta(sec: Section) -> bool:
    return bool(_META.search(sec.title) or _META_WHOLE.match(sec.title)) or sec.level == 1


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
    chunks = [
        "\n".join(lines[idx : starts[n + 1][0] if n + 1 < len(starts) else len(lines)])
        for n, (idx, _) in enumerate(starts)
    ]
    if sum(1 for c in chunks if signal_hits(c, "")) * 2 < len(chunks):
        # Items without step signals ("Browser-automation plugin", "User is
        # logged in") are a checklist of one step, not steps.
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


def signal_hits(text: str, heading: str) -> dict[str, list[str]]:
    """Signal matches per kind; a heading match counts twice (headings name the step)."""
    hits: dict[str, list[str]] = {}
    for name, pat in SIGNALS.items():
        found = [m.group(0) for m in pat.finditer(text)]
        found += [f"[heading] {m.group(0)}" for m in pat.finditer(heading)] * 2
        if found:
            hits[name] = found
    return hits


def parse_skill(skill_dir: Path) -> SkillParse:
    text = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
    fm, body, offset = parse_frontmatter(text)
    root = parse_sections(body, offset)
    registry: dict[str, str] = {}
    for s in walk(root):
        for line in s.body.splitlines():
            if line.strip().startswith("|"):
                for t in _BACKTICK_TOOL.findall(line):
                    registry[t] = slug(s.title.split("(")[0])
    ref_dir = skill_dir / "references"
    if ref_dir.is_dir():
        # A reference file's tool tables name tools as surely as SKILL.md's do.
        for ref in sorted(ref_dir.glob("*.md")):
            _, ref_body, ref_offset = parse_frontmatter(ref.read_text(encoding="utf-8"))
            for s in walk(parse_sections(ref_body, ref_offset)):
                for line in s.body.splitlines():
                    if line.strip().startswith("|"):
                        for t in _BACKTICK_TOOL.findall(line):
                            registry.setdefault(t, slug(s.title.split("(")[0]))
    h1 = next((s for s in walk(root) if s.level == 1), None)
    units: list[Unit] = []
    group = 0
    loops: dict[int, int] = {}  # id(section) -> its loop container unit
    by_section: dict[int, Unit] = {}  # id(section) -> the unit made from it, if one
    for sec in walk(root):
        if _is_meta(sec) or any(_is_meta(a) and a.level > 1 for a in sec.ancestors()):
            continue
        parent_loop = next((loops[id(a)] for a in sec.ancestors() if id(a) in loops), None)
        if [c for c in sec.children if not _is_meta(c)] and len(sec.body.split()) < 25:
            # A container ("## Steps"): its children are the steps. One that
            # says its sub-steps run per item is a loop, and becomes a unit.
            if _PER_ITEM.search(f"{sec.title}\n{sec.body}") and parent_loop is None:
                step_no = _STEP_NO.match(sec.title)
                loops[id(sec)] = len(units)
                units.append(
                    Unit(
                        index=len(units),
                        section=sec.title,
                        title=sec.title,
                        step_number=int(step_no.group(1)) if step_no else None,
                        line=sec.start,
                        text=sec.body,
                        sibling_group=None,
                        mandatory=bool(MANDATORY.search(sec.title)),
                        mcp=[],
                        tools=[],
                        references=[],
                        signals={},
                        constants=[],
                        loop_container=True,
                    )
                )
            continue
        owner = next((by_section[id(a)] for a in sec.ancestors() if id(a) in by_section), None)
        if owner is not None and not signal_hits("", sec.title):
            # A sub-heading that names no action, under a step ("Tab 1: Invoice
            # Detail" under "Generate the report"), describes that step.
            owner.text += f"\n\n{sec.title}\n{sec.body}"
            owner.signals = signal_hits(owner.text, owner.title)
            owner.constants = constants_in(owner.text, owner.line)
            continue
        first_para = sec.body.split("\n\n")[0] if sec.body else ""
        is_gate = bool(SIGNALS["hitl_gate"].search(sec.title)) or bool(
            SIGNALS["hitl_gate"].search(first_para)
        )
        parts = [(sec.title, sec.start, sec.body)] if is_gate else _subunits(sec)
        group += 1
        step_no = _STEP_NO.match(sec.title)
        for title, start, chunk in parts:
            mcp: list[tuple[str, str | None]] = [
                (slug(m.group(1)), m.group(2)) for m in _MCP_PAREN.finditer(chunk)
            ]
            mcp += [(slug(m.group(1)), None) for m in _MCP_USING.finditer(chunk)]
            tools = sorted(set(_BACKTICK_TOOL.findall(chunk)) & registry.keys())
            mcp += [(registry[t], t) for t in tools]
            units.append(
                Unit(
                    index=len(units),
                    section=sec.title,
                    title=title,
                    step_number=int(step_no.group(1)) if step_no else None,
                    line=start,
                    text=chunk,
                    sibling_group=group if len(parts) > 1 else None,
                    mandatory=bool(MANDATORY.search(sec.title) or MANDATORY.search(chunk)),
                    mcp=mcp,
                    tools=tools,
                    references=sorted(set(_REF.findall(chunk))),
                    signals=signal_hits(chunk, title),
                    constants=constants_in(chunk, start),
                    stated=stated_parameters(chunk, start),
                    loop_parent=parent_loop,
                )
            )
        if len(parts) == 1:
            by_section[id(sec)] = units[-1]
    tables = tables_in(skill_dir / "SKILL.md", "SKILL.md")
    ref_dir = skill_dir / "references"
    if ref_dir.is_dir():
        for ref in sorted(ref_dir.glob("*.md")):
            tables += tables_in(ref, f"references/{safe_reference_name(ref.name)}")
    return SkillParse(
        name=str(fm.get("name") or skill_dir.name),
        description=" ".join(str(fm.get("description") or "").split()),
        preamble=h1.body if h1 is not None else "",
        units=units,
        tool_registry=registry,
        tables=[t for t in tables if t.columns() and not _is_sample(t)],
    )
