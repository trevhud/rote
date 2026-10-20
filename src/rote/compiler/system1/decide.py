"""Every compile decision, made by rules over the parsed skill. No inference.

A decision is a field of the IR that changes what the pipeline DOES: which
nodes exist and of which kind, how they are wired, which run once per
element, which are mandatory, where they park, which constants they carry.
Each carries a confidence and the rule that produced it, so a reader (and
the report) can see which ones a person should check.

Confidence is a property of the rule, not a probability: an explicit
statement in the skill ("Steps 1 to 3 do not depend on each other",
"MANDATORY", a named MCP tool) is high; a default applied because the skill
says nothing is low. Low-confidence decisions are never handed to a model.
They take the conservative default named in each rule below, and they are
listed for review in the compile report.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from typing import Any

from rote.compiler.system1.decision_model import DecisionModel, Verdict, Verdicts
from rote.compiler.system1.parse import SIGNALS, Constant, SkillParse, Unit, slug
from rote.compiler.tool_grounding import parameters
from rote.tool_schema import checkable

#: At or above this, a decision is reported as settled; below, it is listed
#: in the report for a person to check.
REVIEW_BELOW = 0.7

_NUMBER_WORDS = {
    "a": 1,
    "an": 1,
    **{
        w: n
        for n, w in enumerate(
            [
                "one",
                "two",
                "three",
                "four",
                "five",
                "six",
                "seven",
                "eight",
                "nine",
                "ten",
                "eleven",
                "twelve",
            ],
            start=1,
        )
    },
    "fourteen": 14,
    "fifteen": 15,
    "twenty": 20,
    "thirty": 30,
}
_GATE_TIMEOUT = re.compile(
    r"\b(?:up to|within|for up to|hold [\w ]{0,20}? for up to)\s+(\w+)\s+(day|hour|week)s?\b", re.I
)
_PARALLEL = re.compile(
    r"\bsteps?\s+(\d+)\s+(?:to|through|-|and)\s+(\d+)\s+do not depend on each other", re.I
)
_STEP_REF = re.compile(r"\bsteps?\s+(\d+)((?:\s*(?:,|and|or)\s*\d+)*)", re.I)
_FOR_EACH = re.compile(
    r"\bfor (?:each|every) ([a-z]+)\b|\beach (?!other\b|time\b|step\b)([a-z]+)\b"
    r"|\bper ([a-z]+)\b|\bone (?:at a time|by one)\b",
    re.I,
)
#: Units of measure "per" also takes ("$75.00 per line" is a cap, not a loop).
_NOT_ITEMS = frozenset(
    [
        "call",
        "batch",
        "request",
        "page",
        "minute",
        "second",
        "hour",
        "day",
        "week",
        "month",
        "year",
        "night",
        "ticket",
        "cent",
    ]
)
#: Pure-function signals that describe computing something, as opposed to
#: words like "check", "filter", "match" that UI and tool steps use too.
_TRANSFORM = re.compile(
    r"\b(convert|group|sort|total|round|calculat\w*|compute|dedup\w*|duplicates?|format"
    r"|render|parse|break (?:it )?down|count|summari[sz]e|aggregate)\b"
    r"|\bgenerate (?:a |an |the )?[\w\s-]{0,30}\b(?:html|report|table|summary|file|sheet)\b",
    re.I,
)
_UI_ACTION = re.compile(
    r"\b(click|navigate|select the|open the|scroll|type into|enter the|dropdown)\b", re.I
)
_GROUP_BY = re.compile(r"\bgroup\b[^.\n]{0,60}\bby ([a-z]+)", re.I)
#: Wording that makes a step conditional, so it may legitimately not run.
_OPTIONAL = re.compile(
    r"\boptional(ly)?\b|\bif (available|needed|necessary|requested|present|any)\b"
    r"|\bonly (if|when)\b|\bwhen (needed|asked|requested)\b|\bskip\b|\bmay\b",
    re.I,
)
#: Constants that parameterize the call itself (which file, how far back),
#: as opposed to the work done on its result.
_FETCH_CONSTANTS = frozenset({"file_id", "channel_id", "folder_id", "window", "limit", "page_size"})


@dataclass
class Decision:
    field: str
    value: Any
    confidence: float
    rule: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "value": self.value,
            "confidence": self.confidence,
            "rule": self.rule,
        }


@dataclass
class NodePlan:
    id: str
    kind: str
    role: str  # fetch | judge | process | write | gate | loop | group | each
    unit: Unit
    decisions: list[Decision] = field(default_factory=list)
    inputs: dict[str, str] = field(default_factory=dict)
    fan_out: bool = False
    element_source: str | None = None
    mandatory: bool = False
    constants: dict[str, Any] = field(default_factory=dict)
    mcp: dict[str, Any] | None = None
    tools: list[str] = field(default_factory=list)
    tool_servers: dict[str, str] = field(default_factory=dict)
    signal: str | None = None
    timeout: str | None = None
    lists_output: bool = False  # another node fans out over this node's `items`
    payload: list[str] = field(default_factory=list)  # a gate's resume payload keys
    loop_body: list[str] = field(default_factory=list)  # an agent_loop's per-item sub-nodes
    termination: dict[str, Any] | None = None  # an agent_loop's stated stop condition
    in_loop: str | None = None  # the loop that calls this node per item

    def decide(self, name: str, value: Any, confidence: float, rule: str) -> None:
        self.decisions.append(Decision(name, value, round(confidence, 2), rule))


@dataclass
class Plan:
    skill: SkillParse
    nodes: list[NodePlan]
    review: list[str]  # human-readable lines for the report
    #: A sample run (:class:`rote.compiler.grounding.Sample`), when there is
    #: one. Data outranks the skill's wording: its keys become the input
    #: schema and the gates' payload keys.
    sample: Any = None
    #: The tie-break's answers and cost (:class:`rote.compiler.system1.tiebreak.TieBreak`),
    #: when one ran.
    tiebreak: Any = None


# ── kinds and segmentation ────────────────────────────────────────────────


def _kind_confidence(signals: dict[str, list[str]], kind_key: str) -> float:
    """High when one kind's signals stand alone, lower when kinds compete."""
    mine = len(signals.get(kind_key, []))
    if mine == 0:
        return 0.3
    rivals = sum(len(v) for k, v in signals.items() if k != kind_key)
    if rivals == 0:
        return 0.9
    return 0.8 if mine >= rivals else 0.6


def _segment(
    u: Unit, registry: Mapping[str, str] | None = None
) -> list[tuple[str, str, float, str]]:
    """(role, kind, confidence, rule) for each node a unit becomes, in data-flow order.

    Rules, in precedence order, from node-kinds.md:
    1. A human gate wins outright: the step exists to park.
    2. An exploratory loop wins over its own tool calls: the tools belong to it.
    3. Otherwise a tool call is a bare external_call, and anything the step
       does to the result is its own node ("an mcp-bound node is a BARE
       tool call"): a judge for fuzzy work, a pure_function for fixed work.
    4. "Group ... by X" followed by "for each X" is two nodes: the grouping,
       then the per-group work fanned out over it.
    5. A side-effecting write is its own external_call, last.
    6. No signal at all: one pure_function (the rubric's "prefer the more
       deterministic kind"), at low confidence.
    """
    s = u.signals
    gate, judge = len(s.get("hitl_gate", [])), len(s.get("llm_judge", []))
    if gate and gate >= judge:
        return [("gate", "hitl_gate", _kind_confidence(s, "hitl_gate"), "human-gate signal")]
    if s.get("agent_loop"):
        return [
            ("loop", "agent_loop", _kind_confidence(s, "agent_loop"), "exploratory-loop signal")
        ]
    out: list[tuple[str, str, float, str]] = []
    if s.get("fetch") or any(tool for _, tool in u.mcp):
        out.append(("fetch", "external_call", _kind_confidence(s, "fetch"), "tool-call signal"))
    grouped = _GROUP_BY.search(u.text) and _FOR_EACH.search(u.text)
    if grouped:
        out.append(("group", "pure_function", 0.8, "'group ... by' before 'for each'"))
    if judge:
        out.append(
            ("judge", "llm_judge", _kind_confidence(s, "llm_judge"), "rubric/judgment signal")
        )
    ui = bool(_UI_ACTION.search(f"{u.title}\n{u.text}"))
    transforms = bool(_TRANSFORM.search(f"{u.title}\n{u.text}"))
    # A UI step (click, navigate, select) gets a pure_function only when it
    # also describes a transformation of data. "Clear all existing filter
    # pills" says "filter", but nothing is computed. A tool step that checks
    # what came back ("check each contact's memberships") keeps its check.
    if s.get("pure_function") and not (grouped and judge) and (transforms or not ui):
        role = "each" if grouped else "process"
        out.append(
            (role, "pure_function", _kind_confidence(s, "pure_function"), "fixed-logic signal")
        )
    if s.get("write") and _names_a_system(u, registry):
        out.append(("write", "external_call", _kind_confidence(s, "write"), "side-effect signal"))
    if not out:
        out.append(("process", "pure_function", 0.3, "no signal: most deterministic kind"))
    if s.get("write") and not _names_a_system(u, registry):
        role, kind, conf, rule = out[-1]
        out[-1] = (
            role,
            kind,
            min(conf, 0.6),
            f"{rule}; it writes to no named system, so its result is the pipeline's output",
        )
    return out


#: Words that carry no step identity: they make ids long and match nothing.
_ID_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "the",
        "of",
        "to",
        "for",
        "from",
        "in",
        "on",
        "into",
        "with",
        "by",
        "each",
        "every",
        "all",
        "any",
        "its",
        "it",
        "this",
        "that",
        "these",
        "those",
        "your",
        "our",
        "their",
        "be",
        "is",
        "are",
        "then",
        "step",
        "steps",
        "phase",
    ]
)


def _per_item(text: str) -> str | None:
    """The per-item wording a step uses ("for each employee", "each thread",
    "per contact"), if any. "Per" followed by a unit of measure does not count."""
    for m in _FOR_EACH.finditer(" ".join(text.split())):
        noun = next((g for g in m.groups() if g), "")
        if noun.lower().rstrip("s") in _NOT_ITEMS:
            continue
        return m.group(0)
    return None


#: A destination with a name: a #channel, an @handle, a URL, or a proper noun
#: (HubSpot, Google Sheet). Formats and units are not systems.
_NAMED = re.compile(r"[#@][\w-]+|https?://\S+|(?<=\s)(?-i:[A-Z][A-Za-z0-9]+)")
_NOT_SYSTEMS = frozenset(
    [
        "HTML",
        "CSV",
        "PDF",
        "JSON",
        "XLSX",
        "XML",
        "TXT",
        "MD",
        "YAML",
        "USD",
        "EUR",
        "GBP",
        "ID",
        "IDS",
        "UTC",
        "ISO",
    ]
)


#: Tool-name words that say what a tool does, not what it acts on.
_TOOL_VERBS = frozenset(
    [
        "create",
        "update",
        "get",
        "list",
        "search",
        "add",
        "remove",
        "delete",
        "read",
        "write",
        "batch",
        "upsert",
        "send",
        "set",
        "edit",
        "find",
        "fetch",
        "post",
        "put",
        "patch",
    ]
)


#: The verbs that make a tool a write.
_WRITE_VERBS = frozenset(["create", "update", "upsert", "post", "send", "add", "save", "write"])


def _registry_tool_for(sentence: str, registry: Mapping[str, str]) -> str | None:
    """A write tool the skill lists (its tool tables, its reference files)
    whose object the sentence names: "create or edit sales email templates"
    names ``hubspot_create_sales_template`` through "template"."""
    words = {w.rstrip("s") for w in slug(sentence).split("_") if w}
    for tool in sorted(registry):
        parts = tool.split("_")
        if not set(parts) & _WRITE_VERBS:
            continue
        # The object is the name's last non-verb word, its head noun: "template"
        # in hubspot_create_sales_template. A modifier ("sales") is not enough,
        # or "post the sales summary" would bind a template tool.
        nouns = [w for w in parts[1:] if w not in _TOOL_VERBS]
        if nouns and nouns[-1].rstrip("s") in words:
            return tool
    return None


def _names_a_system(u: Unit, registry: Mapping[str, str] | None = None) -> bool:
    """Does the step's write go somewhere named, outside the pipeline?

    "Save it to the finance shared folder" names no system: that step is the
    pipeline's output, returned to the caller, not an external_call stub
    that raises "target system not specified" (a real expense-audit run).
    A tool, an MCP mention or a UI action is always a named target."""
    if u.mcp or u.tools or _UI_ACTION.search(u.text):
        return True
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", u.text):
        if not SIGNALS["write"].search(sentence):
            continue
        if registry and _registry_tool_for(sentence, registry):
            return True
        for m in _NAMED.finditer(sentence.strip()):
            if m.group(0).strip("#@").upper() not in _NOT_SYSTEMS or m.group(0)[0] in "#@h":
                return True
    return False


def _base(title: str, section: str | None = None) -> str:
    """A short id from the heading's content words (at most four).

    A sub-step enumerated under a heading takes the heading's first content
    word as a prefix, so "Do-not-contact list" under "Exclusion Checks"
    reads as one of the exclusion checks.
    """

    def words(text: str) -> list[str]:
        text = re.sub(r"^(step|phase)\s*[\w.-]*\s*[:\u2014\u2013-]\s*", "", text, flags=re.I)
        text = re.sub(r"\([^)]*\)", "", text)  # "(MANDATORY)", "(Iterative)"
        return [w for w in slug(text).split("_") if w and w not in _ID_STOPWORDS]

    mine = words(title)
    if section is not None and section != title:
        head = words(section)[:1]
        mine = [*head, *[w for w in mine if w not in head]]
    base = "_".join(mine[:4]) or "step"
    # "7a: Read all column values" must still yield an identifier.
    return f"step_{base}" if base[0].isdigit() else base


_ROLE_IDS = {
    "fetch": "fetch_{b}",
    "judge": "{b}_judge",
    "process": "{b}",
    "write": "{b}_write",
    "gate": "{b}_gate",
    "loop": "{b}_loop",
    "group": "group_{b}",
    "each": "{b}_each",
}


# ── data flow ─────────────────────────────────────────────────────────────


def _unit_dependencies(skill: SkillParse) -> dict[int, tuple[list[int], float, str]]:
    """Which earlier units each unit reads from, with the rule that decided it.

    1. Explicit references ("the duplicate and late flags from steps 2 and
       3") are dependencies, at high confidence.
    2. "Steps N to M do not depend on each other" makes those steps
       siblings: each reads what came before the range.
    3. Sub-steps the skill enumerates under one heading (data sources 1 to
       4) are siblings too; the next step reads all of them.
    4. Otherwise a step reads the step before it (low confidence: the skill
       did not say, and a sequential chain is the conservative reading, since
       it can only over-order, never drop data).
    """
    by_step: dict[int, list[int]] = {}
    for u in skill.units:
        if u.step_number is not None:
            by_step.setdefault(u.step_number, []).append(u.index)
    parallel: list[tuple[int, int]] = []
    for text in [skill.preamble, *(u.text for u in skill.units)]:
        flat = " ".join(text.split())
        parallel += [(int(a), int(b)) for a, b in _PARALLEL.findall(flat)]

    def range_of(u: Unit) -> tuple[int, int] | None:
        n = u.step_number
        return next((r for r in parallel if n is not None and r[0] <= n <= r[1]), None)

    def siblings(i: int) -> list[int]:
        """The units that run alongside unit i: its enumerated group, or its parallel range."""
        u = skill.units[i]
        if u.sibling_group is not None:
            return [x.index for x in skill.units if x.sibling_group == u.sibling_group]
        rng = range_of(u)
        if rng is not None:
            return [x.index for x in skill.units if range_of(x) == rng]
        return [i]

    deps: dict[int, tuple[list[int], float, str]] = {}
    for u in skill.units:
        i = u.index
        flat = " ".join(u.text.split())
        explicit: list[int] = []
        for m in _STEP_REF.finditer(flat):
            nums = [int(m.group(1)), *(int(n) for n in re.findall(r"\d+", m.group(2) or ""))]
            for n in nums:
                if n != u.step_number:
                    explicit += [j for j in by_step.get(n, []) if j < i]
        mine = siblings(i)
        before = [j for j in range(i) if j not in mine]
        default = siblings(before[-1]) if before else []
        if len(mine) > 1:
            rule = "runs alongside its sibling steps"
            conf = 0.8 if range_of(u) else 0.7
        elif not before:
            rule, conf = "first step reads the pipeline input", 0.9
        elif len(default) > 1:
            rule, conf = "reads every sibling step before it", 0.7
        else:
            rule, conf = "default: the step before", 0.5
        if explicit:
            rule, conf = f"explicit step reference, plus {rule}", 0.9
        deps[i] = (sorted(set(default + explicit)), conf, rule)
    return deps


# ── constants, tools, gates ───────────────────────────────────────────────


def _constant_names(cs: list[Constant]) -> dict[str, Any]:
    """One parameter per distinct value: "more than 60" and "60 days" are one constant."""
    out: dict[str, Any] = {}
    for c in cs:
        if c.value in out.values():
            continue
        name, n = c.name, 2
        while name in out:
            name, n = f"{c.name}_{n}", n + 1
        out[name] = c.value
    return out


def _gate_timeout(text: str) -> str | None:
    """The longest wait the step states ("usually within a day ... hold for up to
    three days" parks for three): a gate that times out early fails the run."""
    hours: list[int] = []
    for m in _GATE_TIMEOUT.finditer(" ".join(text.split())):
        word = m.group(1).lower()
        n = int(word) if word.isdigit() else _NUMBER_WORDS.get(word)
        if n is not None:
            hours.append(n * {"hour": 1, "day": 24, "week": 168}[m.group(2).lower()])
    if not hours:
        return None
    longest = max(hours)
    return f"{longest // 24}d" if longest % 24 == 0 else f"{longest}h"


_DEFAULT_PAYLOAD_RULE = "no payload stated: the reviewer's approval and notes"
_BOLD_ITEM = re.compile(r"^\s*(?:\d+[.)]|[-*])\s+\*\*(.+?)\*\*", re.M)
_DECISION_VERBS = (
    ("approve", "approved"),
    ("reject", "rejected"),
    ("add", "added"),
    ("remove", "removed"),
    ("reprioritiz", "reprioritized"),
)


def _gate_payload(u: Unit) -> tuple[list[str], str]:
    """What the human sends back, in the step's own words.

    1. The items the step asks for, when it enumerates them in bold
       ("1. **Dock Appointment Completion** ... 2. **Sorter Metrics**").
    2. Otherwise the decisions it offers ("approve or reject", "add,
       remove, or reprioritize").
    3. Otherwise an approval flag and notes.
    """
    items = [slug(re.sub(r"\([^)]*\)", "", m)) for m in _BOLD_ITEM.findall(u.text)]
    items = [i for i in items if i]
    if items:
        return items, "the items the step asks the person for"
    low = u.text.lower()
    verbs = [key for stem, key in _DECISION_VERBS if re.search(rf"\b{stem}", low)]
    if verbs:
        return verbs, "the decisions the step offers the person"
    return ["approved", "notes"], _DEFAULT_PAYLOAD_RULE


_STOP = re.compile(
    r"\b(?:until|stop (?:when|once|after|if)|end (?:when|once))\b\s*([^.\n]{3,160})", re.I
)
_BOUND = re.compile(
    r"\b(?:at most|up to|no more than|a maximum of|max(?:imum)?(?: of)?)\s+(\d+)\s+"
    r"(?:iterations?|rounds?|times|passes|pages|attempts|searches|tries|loops|batches)\b",
    re.I,
)


def _termination(text: str) -> dict[str, Any] | None:
    """A loop's stop condition and hard bound, both in the skill's words, or None.

    Both are needed: the IR's termination carries a condition and a
    max_iterations, and a bound the skill does not state would be invented."""
    flat = " ".join(text.split())
    stop, bound = _STOP.search(flat), _BOUND.search(flat)
    if stop is None or bound is None:
        return None
    return {"condition": stop.group(0).strip(" ,;"), "max_iterations": int(bound.group(1))}


def _schema_parameters(
    tools: Mapping[tuple[str, str], Any], server: str, tool: str
) -> tuple[list[str], int] | None:
    """The tool's parameter names from grounding, and how many were withheld.

    A schema comes from the server, so it is untrusted: names pass through
    ``tool_grounding.parameters``, the same filter the agent's brief uses, and
    a name that is not identifier-shaped never becomes an ``mcp.args`` key.
    ``None`` when no schema is known or it cannot be read as a flat object.
    """
    schema = tools.get((server, tool))
    if schema is None or checkable(schema.input_schema) is None:
        return None
    kept, withheld = parameters(schema)
    return [p.name for p in kept], withheld


def _match_property(name: str, props: Collection[str], taken: set[str]) -> str | None:
    """The tool parameter a stated label names: the same name, or the property
    whose words are all in the label ("search_query" names ``query``)."""
    if name in props and name not in taken:
        return name
    words = set(name.split("_"))
    fits = [p for p in props if p not in taken and set(slug(p).split("_")) <= words]
    # Two properties both fitting the label is ambiguous: bind neither, and
    # the unbound value goes to review, rather than guessing the first.
    return fits[0] if len(fits) == 1 else None


def _bind_arguments(
    n: NodePlan,
    u: Unit,
    bound: tuple[str, str | None],
    tools: Mapping[tuple[str, str], Any] | None = None,
) -> None:
    """The MCP binding and its arguments, never the whole pipeline input.

    Without ``mcp.args`` an emitted step sends its entire payload as the
    tool's arguments, and the request went out as Gmail search arguments in
    a real ops-report run. Arguments are the step's own fixed values: the
    parameters it states ("Search query: label:dock-activity newer_than:1d")
    and the ids and windows it names, each a constant. A step that states
    none calls the tool with no arguments, listed for review.
    """
    for c in u.stated:
        if c.name not in n.constants and c.value not in n.constants.values():
            n.constants[c.name] = c.value
    server, tool = bound[0], bound[1] or ""
    found = _schema_parameters(tools or {}, server, tool)
    if found is not None:
        props, withheld = found
        args: dict[str, str] = {}
        for const in n.constants:
            prop = _match_property(const, props, set(args))
            if prop is not None:
                args[prop] = const
        n.mcp = {"server": server, "tool": tool, "args": args}
        n.decide("mcp", dict(n.mcp), 0.9, "MCP server and tool named in the step")
        unmatched = [c for c in n.constants if c not in args.values()]
        n.decide(
            "mcp.args",
            dict(args),
            0.9 if not unmatched and not withheld else 0.6,
            "argument names from the tool's input schema"
            + (f"; no parameter for {unmatched}" if unmatched else "")
            + (
                f"; {withheld} parameter name(s) withheld as unsafe, never bound"
                if withheld
                else ""
            ),
        )
        return
    n.mcp = {"server": server, "tool": tool, "args": {k: k for k in n.constants}}
    n.decide("mcp", dict(n.mcp), 0.9, "MCP server and tool named in the step")
    if n.constants:
        n.decide(
            "mcp.args",
            dict(n.mcp["args"]),
            0.6,
            "argument names are the step's own labels; check them against the tool's schema",
        )
    else:
        n.decide("mcp.args", {}, 0.5, "the step states no argument: the tool is called with none")


def _loop_tools(u: Unit, skill: SkillParse) -> tuple[list[str], dict[str, str], float, str]:
    if u.tools:
        return u.tools, {t: skill.tool_registry[t] for t in u.tools}, 0.9, "tools named in the step"
    words = set(slug(u.text).split("_"))
    servers = {s for s in skill.tool_registry.values() if s in words}
    tools = sorted(t for t, s in skill.tool_registry.items() if s in servers and "search" in t)
    if tools:
        return (
            tools,
            {t: skill.tool_registry[t] for t in tools},
            0.6,
            "search tools of servers the step names",
        )
    return [f"{_base(u.title)}_tool"], {}, 0.2, "no tool named: placeholder tool, needs review"


# ── the plan ──────────────────────────────────────────────────────────────


_ROLE_OF_KIND = {
    "pure_function": "process",
    "external_call": "fetch",
    "llm_judge": "judge",
    "agent_loop": "loop",
    "hitl_gate": "gate",
}


def _route(
    segments: list[tuple[str, str, float, str]], verdict: Verdict, model: DecisionModel
) -> list[tuple[str, str, float, str]]:
    """The model's answers over the rules' segmentation, only where it is confident.

    * The model says a person must act (p >= its human threshold): one gate.
    * The model's kind at or above its kind threshold that the rules did not
      find: one node of that kind (the rules' split is kept when they already
      include it).
    * Otherwise the rules decide. Where the model disagrees below its
      threshold, the rules' decision drops below REVIEW_BELOW so the report
      lists it.
    """
    rule_kinds = [k for _, k, _, _ in segments]
    who = model.label
    if verdict.human >= model.human_threshold:
        if rule_kinds == ["hitl_gate"]:
            return segments
        return [("gate", "hitl_gate", 0.9, f"{who}: a person must act (p={verdict.human:.2f})")]
    if verdict.kind_confidence >= model.kind_threshold:
        if verdict.kind in rule_kinds and "hitl_gate" not in rule_kinds:
            return segments
        return [
            (
                _ROLE_OF_KIND[verdict.kind],
                verdict.kind,
                round(verdict.kind_confidence, 2),
                f"{who} kind (p={verdict.kind_confidence:.2f}) over the rules' {rule_kinds}",
            )
        ]
    if verdict.kind not in rule_kinds or (rule_kinds == ["hitl_gate"]):
        return [
            (
                role,
                kind,
                min(conf, 0.6),
                f"{rule}; {who} disagrees below its threshold "
                f"({verdict.kind} p={verdict.kind_confidence:.2f}, person p={verdict.human:.2f})",
            )
            for role, kind, conf, rule in segments
        ]
    return segments


#: (role, kind, confidence, rule) for one node a step becomes.
Segment = tuple[str, str, float, str]


def unit_segments(skill: SkillParse, verdicts: Verdicts | None = None) -> dict[int, list[Segment]]:
    """The nodes each step unit becomes, by unit index: the rules' segmentation,
    then (with ``verdicts``) the decision model's confident answers over it."""
    out: dict[int, list[Segment]] = {}
    for u in skill.units:
        segments = (
            [("loop", "agent_loop", 0.7, "a heading whose sub-steps run per item")]
            if u.loop_container
            else _segment(u, skill.tool_registry)
        )
        if verdicts is not None and not u.loop_container and u.index in verdicts.by_unit:
            segments = _route(segments, verdicts.by_unit[u.index], verdicts.model)
        out[u.index] = segments
    return out


def unsure_units(skill: SkillParse, segments: Mapping[int, list[Segment]]) -> list[Unit]:
    """The step units whose kind decision is below REVIEW_BELOW: the rules had
    no clear signal, or a decision model disagreed below its threshold.

    A loop container is never unsure: its kind comes from the skill's shape
    (a heading whose sub-steps run per item), not from wording."""
    return [
        u
        for u in skill.units
        if not u.loop_container and min(conf for _, _, conf, _ in segments[u.index]) < REVIEW_BELOW
    ]


def infeasible_kinds(
    u: Unit,
    registry: Mapping[str, str] | None = None,
    tools: Collection[tuple[str, str]] = (),
) -> dict[str, str]:
    """Kinds this step cannot be, whoever proposes them, with the rule why.

    These are the rules that are constraints, not guesses: a model deciding
    an unsure step (the tie-break) may pick any other kind, never these. A
    real expense-audit compile shows why: the tie-break made "save it to the
    finance shared folder" an external_call, the agent could only write a
    stub for a system the skill never declares, and every run died there.

    Only a write is constrained: a step that writes, reads nothing, and that
    no declared system could perform (:func:`declares_a_system`). A read
    ("load the last 7 days of tickets") is never forbidden here, because its
    data has to come from somewhere and the agent's code can still be
    pointed at it. Within writes it is grounded in what is declared, not in
    phrasing: "email the summary" and "notify the owner by email" are
    treated alike, as calls when an email tool is declared and as output
    when nothing is.
    """
    writes = any(SIGNALS["write"].search(x) or _DELIVERS.search(x) for x in (u.title, u.text))
    reads = bool(u.signals.get("fetch")) or any(_READS.search(x) for x in (u.title, u.text))
    if not writes or reads or declares_a_system(u, registry, tools):
        return {}
    return {
        "external_call": (
            "it writes to no declared or named system (no MCP server or tool, no named "
            "product, API, interface or path), so it keeps the rules' decision: what it "
            "would write is returned in the pipeline's output for the caller to deliver, "
            "and the step is listed in Open questions"
        )
    }


_CODE_SPAN = re.compile(r"`([a-z][a-z0-9]*_[a-z0-9_]+)`")
#: The first word of a tool name that makes a code span a tool call.
_CALL_VERBS = _TOOL_VERBS | _WRITE_VERBS | {"append", "insert"}
#: A named API: a URL, or an HTTP verb with a path.
_NAMED_API = re.compile(r"https?://\S+|\b(?:GET|POST|PUT|PATCH|DELETE)\s+/\S*")
#: Verbs that deliver a result somewhere, beyond the parser's write signal
#: ("email the summary", "notify the owner", "write it up", "run upload_x").
#: Words that are also common nouns in reads (file, log, record) are left out.
_DELIVERS = re.compile(
    r"\b(?:e-?mail|notify|send|post|upload|publish|share|store|save|write|export|"
    r"append|insert)(?:s|ed|ing)?\b|\b\w*upload_\w+",
    re.I,
)
#: Verbs that bring data in. A step that also reads is not constrained.
_READS = re.compile(
    r"\b(?:fetch|read|load|pull|retrieve|download|query|look(?:s|ed|ing)? up)(?:s|ed|ing)?\b",
    re.I,
)
#: A local path (``~/reports/``, ``/var/data/``): the file system is the system.
_LOCAL_PATH = re.compile(r"(?<![\w.])[~/][\w.-]*/")


def declares_a_system(
    u: Unit,
    registry: Mapping[str, str] | None = None,
    tools: Collection[tuple[str, str]] = (),
) -> bool:
    """Whether a declared system could perform this step.

    Declared means one of: an MCP server or tool the step names (the parser's
    ``mcp`` and ``tools``); a tool from the skill's tool tables or a
    registered server's ``tools/list`` (``registry``, ``tools``) whose name
    the step uses, or whose server it names; a write tool whose object the
    step names (:func:`_registry_tool_for`); a named API (a URL, an HTTP
    verb with a path); a local path (the file system); a named product or
    service (Google Sheets, Slack, PagerDuty, a #channel or @handle); or a
    user interface it operates. A named product has an API the compile can
    bind or research, so the step can do what the skill does. Tool names
    match on word boundaries: a server's ``search`` tool does not make "do
    the research" a call.
    """
    if u.mcp or u.tools or _NAMED_API.search(u.text) or _names_a_product(u.text):
        return True
    if _LOCAL_PATH.search(u.text):
        return True
    # A step that operates a user interface (click, navigate, select) acts on
    # the application the skill drives, which is its system.
    if _UI_ACTION.search(f"{u.title}\n{u.text}"):
        return True
    # A tool the step names in code, verb first (`update_incident`), is a
    # tool the skill declares even outside a tool table.
    for span in _CODE_SPAN.findall(u.text):
        if span.split("_", 1)[0] in _CALL_VERBS:
            return True
    text = f"{u.title}\n{u.text}"
    words = set(slug(text).split("_"))
    reg = dict(registry or {})
    names = set(reg) | {t for _, t in tools}
    if any(re.search(rf"\b{re.escape(name)}\b", text, re.I) for name in names if name):
        return True
    servers = set(reg.values()) | {s for s, _ in tools}
    if any(set(slug(server).split("_")) <= words for server in servers if server):
        return True
    sentences = re.split(r"(?<=[.!?])\s+|\n+", u.text)
    return bool(reg) and any(_registry_tool_for(x, reg) for x in sentences)


#: A capitalized word inside a sentence: a product or service name. A
#: sentence's first word, and a title's words, say nothing.
_MID_SENTENCE_NAME = re.compile(r"(?<=[a-z0-9,] )(?-i:[A-Z][A-Za-z0-9]+)")
_HANDLE = re.compile(r"(?<![\w])[#@][A-Za-z][\w-]+")


def _names_a_product(text: str) -> bool:
    if _HANDLE.search(text):
        return True
    return any(m.group(0).upper() not in _NOT_SYSTEMS for m in _MID_SENTENCE_NAME.finditer(text))


def role_of_kind(kind: str) -> str:
    """The node role a kind decided outside the rules gets."""
    return _ROLE_OF_KIND[kind]


def plan(
    skill: SkillParse,
    verdicts: Verdicts | None = None,
    tools: Mapping[tuple[str, str], Any] | None = None,
    segments: Mapping[int, list[Segment]] | None = None,
) -> Plan:
    """Every decision for ``skill``. With ``verdicts`` (``--decider clef`` or
    ``jev``), the model's confident kind and gate answers override the rules
    per step. ``segments`` (from :func:`unit_segments`, possibly revised by
    the tie-break) replaces that step: the nodes each unit becomes. With
    ``tools`` (grounding's ``ToolSchema`` per ``(server, tool)``), MCP argument
    names come from each tool's input schema."""
    by_unit = segments if segments is not None else unit_segments(skill, verdicts)
    nodes: list[NodePlan] = []
    used: set[str] = set()
    last_of: dict[int, str] = {}
    loop_nodes: dict[int, NodePlan] = {}
    review: list[str] = []
    deps = _unit_dependencies(skill)
    for u in skill.units:
        unit_nodes: list[NodePlan] = []
        for role, kind, conf, rule in by_unit[u.index]:
            nid = _ROLE_IDS[role].format(b=_base(u.title, u.section if u.sibling_group else None))
            base_id, k = nid, 2
            while nid in used:
                nid, k = f"{base_id}_{k}", k + 1
            used.add(nid)
            node = NodePlan(nid, kind, role, u)
            node.decide("kind", kind, conf, rule)
            unit_nodes.append(node)
        node_deps, dconf, drule = deps[u.index]
        upstream = [last_of[j] for j in node_deps if j in last_of]
        loop = loop_nodes.get(u.loop_parent) if u.loop_parent is not None else None
        if loop is not None:
            # A per-item sub-step: its loop calls it, after the sub-step before it.
            prev = loop.loop_body[-1:] if loop.loop_body else []
            upstream, dconf, drule = prev, 0.8, f"runs inside {loop.id}, after the sub-step before"
            for n in unit_nodes:
                n.in_loop = loop.id
        for pos, node in enumerate(unit_nodes):
            node.inputs = {"request": "pipeline.input"}
            sources = upstream if pos == 0 else [unit_nodes[pos - 1].id]
            for src in sources:
                node.inputs[src] = f"{src}.output"
            node.decide(
                "inputs",
                dict(node.inputs),
                dconf if pos == 0 else 0.9,
                drule if pos == 0 else "inside one step, each node reads the one before",
            )
            each = None if loop is not None else _per_item(f"{u.title}\n{u.text}")
            # Only judges and "for each X" work fan out. A pure_function over a
            # list loops in code with the same result; a judge called once per
            # item is a different call than one judge over the whole list.
            if node.role in ("each", "judge") and each and sources:
                src = sources[-1]
                node.fan_out = True
                node.element_source = src
                node.inputs[src] = f"{src}.output.items"
                node.decide("fan_out", True, 0.7, f"per-item wording {each!r} over {src}")
            if node.kind not in ("agent_loop", "hitl_gate"):  # a gate is mandatory implicitly
                optional = _OPTIONAL.search(f"{u.title}\n{u.text}")
                if u.mandatory:
                    node.mandatory = True
                    node.decide("mandatory", True, 0.95, "MANDATORY / required in the step")
                elif optional:
                    node.decide(
                        "mandatory", False, 0.7, f"the step is conditional: {optional.group(0)!r}"
                    )
                else:
                    node.mandatory = True
                    node.decide(
                        "mandatory",
                        True,
                        0.6,
                        "the skill runs this step every time (no optional or conditional wording)",
                    )
        # Constants: ids, windows and limits parameterize the fetch; thresholds
        # and batch sizes the work on the result.
        fetch = next((n for n in unit_nodes if n.role == "fetch"), None)
        work = next(
            (n for n in unit_nodes if n.role in ("process", "each", "judge", "group")), None
        )
        per_node: dict[str, list[Constant]] = {}
        for c in u.constants:
            target = fetch if fetch is not None and c.name in _FETCH_CONSTANTS else None
            target = target or work or unit_nodes[0]
            per_node.setdefault(target.id, []).append(c)
        for n in unit_nodes:
            n.constants = _constant_names(per_node.get(n.id, []))
        for n in unit_nodes:
            if n.constants:
                n.decide("constants", dict(n.constants), 0.8, "literal values stated in the step")
            if n.role == "fetch":
                bound = next(((s, t) for s, t in u.mcp if t), None)
                if bound is not None:
                    _bind_arguments(n, u, bound, tools)
                elif u.mcp:
                    n.decide("mcp", None, 0.3, f"MCP server '{u.mcp[0][0]}' named, but no tool")
            if n.kind == "agent_loop":
                names, servers, tconf, trule = _loop_tools(u, skill)
                n.tools, n.tool_servers = names, servers
                n.decide("tools", names, tconf, trule)
                n.termination = _termination(u.text)
                if n.termination is not None:
                    n.decide(
                        "termination", dict(n.termination), 0.8, "stop wording and bound stated"
                    )
                else:
                    n.decide(
                        "termination",
                        None,
                        0.5,
                        "no stated stop condition and bound: the text agent writes it",
                    )
            if n.kind == "hitl_gate":
                keys, krule = _gate_payload(u)
                n.payload = keys
                n.decide("payload", keys, 0.8 if krule != _DEFAULT_PAYLOAD_RULE else 0.5, krule)
                suffix = "provided" if krule.startswith("the items") else "approved"
                n.signal = f"{n.id.removesuffix('_gate')}_{suffix}"
                n.decide("signal", n.signal, 0.9, "one signal per gate, named for what resumes it")
                n.timeout = _gate_timeout(u.text)
                if n.timeout:
                    n.decide("timeout", n.timeout, 0.8, "duration stated in the step")
        for n in unit_nodes:
            if n.element_source is not None:
                source = next(x for x in [*nodes, *unit_nodes] if x.id == n.element_source)
                source.lists_output = True
        nodes += unit_nodes
        if u.loop_container and unit_nodes:
            loop_nodes[u.index] = unit_nodes[0]
        if loop is not None:
            loop.loop_body += [n.id for n in unit_nodes]
            loop.tools = sorted({*loop.tools, *u.tools} - {f"{loop.id}_tool"}) or loop.tools
            last_of[u.index] = loop.id
        elif unit_nodes:
            last_of[u.index] = unit_nodes[-1].id
    for n in nodes:
        for d in n.decisions:
            if d.confidence < REVIEW_BELOW:
                review.append(
                    f"`{n.id}` {d.field} = {d.value!r} (confidence {d.confidence:.2f}): {d.rule}"
                )
    return Plan(skill=skill, nodes=nodes, review=review)
