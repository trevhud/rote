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
from dataclasses import dataclass, field
from typing import Any

from rote.compiler.system1.parse import Constant, SkillParse, Unit, slug

#: At or above this, a decision is reported as settled; below, it is listed
#: in the report for a person to check.
REVIEW_BELOW = 0.7

_NUMBER_WORDS = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "ten": 10,
    "fourteen": 14,
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
    r"\bfor each ([a-z]+)\b|\b(?:classify|score|vet|personali[sz]e|draft) each\b", re.I
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
    in_loop: str | None = None  # the loop that calls this node per item

    def decide(self, name: str, value: Any, confidence: float, rule: str) -> None:
        self.decisions.append(Decision(name, value, round(confidence, 2), rule))


@dataclass
class Plan:
    skill: SkillParse
    nodes: list[NodePlan]
    review: list[str]  # human-readable lines for the report


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


def _segment(u: Unit) -> list[tuple[str, str, float, str]]:
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
    if s.get("pure_function") and not (grouped and judge):
        role = "each" if grouped else "process"
        out.append(
            (role, "pure_function", _kind_confidence(s, "pure_function"), "fixed-logic signal")
        )
    if s.get("write"):
        out.append(("write", "external_call", _kind_confidence(s, "write"), "side-effect signal"))
    if not out:
        out.append(("process", "pure_function", 0.3, "no signal: most deterministic kind"))
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


_JUDGE_VERB = re.compile(
    r"^(score|classify|vet|rank|rate|grade|personali[sz]e|draft|assess|review)$"
)


def _plural_object(title: str) -> str | None:
    """The plural noun a judging heading acts on ("Score New Opportunities"), if any.

    A rubric applied to a plural is applied to each item: one judge call
    per item is how the skill's agent works through the list."""
    words = [w for w in slug(re.sub(r"\([^)]*\)", "", title)).split("_") if w]
    words = [w for w in words if not w.isdigit() and w not in ("step", "phase")]
    if not words or not _JUDGE_VERB.match(words[0]):
        return None  # "Pull Slack messages" fetches a plural; it does not judge one by one
    last = words[-1]
    if len(last) > 3 and last.endswith("s") and not last.endswith(("ss", "us", "is")):
        return last
    return None


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


def plan(skill: SkillParse) -> Plan:
    nodes: list[NodePlan] = []
    used: set[str] = set()
    last_of: dict[int, str] = {}
    loop_nodes: dict[int, NodePlan] = {}
    review: list[str] = []
    deps = _unit_dependencies(skill)
    for u in skill.units:
        segments = (
            [("loop", "agent_loop", 0.7, "a heading whose sub-steps run per item")]
            if u.loop_container
            else _segment(u)
        )
        unit_nodes: list[NodePlan] = []
        for role, kind, conf, rule in segments:
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
            each = None if loop is not None else _FOR_EACH.search(f"{u.title}\n{u.text}")
            plural = loop is None and node.role == "judge" and _plural_object(u.title)
            if node.role in ("each", "judge") and (each or plural) and sources:
                src = sources[-1]
                node.fan_out = True
                node.element_source = src
                node.inputs[src] = f"{src}.output.items"
                why = "'for each' / 'each'" if each else f"a judgment over {plural!r}, one per item"
                node.decide("fan_out", True, 0.7 if each else 0.6, f"{why} over {src}")
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
                    n.mcp = {"server": bound[0], "tool": bound[1]}
                    if n.constants:
                        n.mcp["args"] = {k: k for k in n.constants}
                    n.decide("mcp", dict(n.mcp), 0.9, "MCP server and tool named in the step")
                elif u.mcp:
                    n.decide("mcp", None, 0.3, f"MCP server '{u.mcp[0][0]}' named, but no tool")
            if n.kind == "agent_loop":
                tools, servers, tconf, trule = _loop_tools(u, skill)
                n.tools, n.tool_servers = tools, servers
                n.decide("tools", tools, tconf, trule)
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
