"""Ground a compile's contracts in data, not in the model's reading of prose.

A skill run as an agent adapts to whatever input arrives. A compiled
pipeline has a fixed contract: the field names of its input, the row keys
its functions index, the keys of each gate's payload. When the skill only
describes its input in prose ("the month's export: one row per expense
line, plus the reporting period"), every compile invents names for it
(``expense_lines``, ``lines``, ``reporting_period``), and a pipeline that
is otherwise right cannot run on the skill's own input. Measured on the
expense-audit example: all 36 compiles on disk failed at the input.

So rote, not the agent, decides the contract wherever data exists:

* :func:`find_sample` locates a sample input, in order: an explicit
  ``--sample-input`` file, a sample bundled in the skill dir
  (:data:`SAMPLE_INPUT_NAMES`), then the payload a ``rote baseline`` run
  actually ran the skill on. A sample may be the ``{"input": ...,
  "signals": {...}}`` envelope ``rote eval --run`` takes, which grounds
  the gate payloads too.
* :func:`input_contract` extracts field names and shapes from it with
  :func:`rote.probe.infer_schema`, row keys inside lists included. No
  model is involved.
* :func:`contract_brief` hands that contract to the compiler agent before
  it starts, and :func:`grounding_findings` checks the deliverable
  against it afterwards. A finding is an error the repair loop sends back
  verbatim.
* With no sample, the contract the agent chose is listed for review
  (:func:`contract_questions`) with the skill's own wording it came from,
  and so is every MCP result shape a node assumes without a recorded
  response.

The deterministic prefix run (:mod:`rote.compiler.prefix`) is the other
half: it executes the pipeline's pure functions on the sample, which is
what catches a row key the sample does not have.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rote.contracts import ContractFinding
from rote.ir import Node, NodeKind, Pipeline, is_field_name, parse_input_ref
from rote.skill_source import parse_sections

#: Where a skill bundles a sample input, in the order they are tried. A
#: JSON object at any of these is the sample; the first one found wins.
SAMPLE_INPUT_NAMES = (
    "input.json",
    "sample-input.json",
    "sample_input.json",
    "example-input.json",
    "examples/input.json",
    "examples/sample-input.json",
    "fixtures/input.json",
)

#: The work-dir directory the sample and its contract are copied into for
#: the agent. Reference material, removed before the output is written.
GROUNDING_DIRNAME = "grounding"
SAMPLE_FILENAME = "sample-input.json"
CONTRACT_FILENAME = "input-contract.json"

#: Pseudo node id for findings about the pipeline's input as a whole.
INPUT_NODE = "(pipeline input)"

#: How much of the skill's wording an open question quotes.
_WORDING_CHARS = 300


@dataclass(frozen=True)
class Sample:
    """A sample run of the skill: its input, and any gate payloads."""

    payload: dict[str, Any]
    #: Where it came from, for messages: a path, or "baseline input".
    source: str
    #: ``{signal name: payload}`` in the order the gates ask, from an envelope's
    #: ``signals`` (``{signal name: payload}``). Names are the sample's,
    #: not necessarily the compile's, so only the order and keys count.
    gates: dict[str, dict[str, Any]] = field(default_factory=dict)


def parse_sample(raw: Any, source: str) -> Sample:
    """A :class:`Sample` from a decoded JSON document, or ``ValueError``.

    The envelope ``{"input": {...}, "signals": {...}}`` is recognized by an
    explicit ``signals`` key, the same rule ``rote eval --run --input``
    applies, so a skill whose own input has an ``input`` field is not
    misread.
    """
    if not isinstance(raw, dict):
        raise ValueError(
            f"{source}: a sample input must be a JSON object, got {type(raw).__name__}"
        )
    if "signals" in raw and set(raw) <= {"input", "signals"}:
        payload = raw.get("input") or {}
        signals = raw.get("signals") or {}
        if not isinstance(payload, dict) or not isinstance(signals, dict):
            raise ValueError(f"{source}: envelope `input` and `signals` must be JSON objects")
        gates = {str(k): v for k, v in signals.items() if isinstance(v, dict)}
        return Sample(payload=payload, source=source, gates=gates)
    return Sample(payload=raw, source=source)


def load_sample(path: Path) -> Sample:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"{path}: not valid JSON: {e}") from e
    return parse_sample(raw, str(path))


def find_sample(
    skill_dir: Path, explicit: Path | None = None, probe_dir: Path | None = None
) -> Sample | None:
    """The sample input to ground a compile in, or ``None`` when there is none.

    An explicit file that does not exist or does not parse is an error
    (``ValueError``), never a silent fallback to a weaker source. A
    bundled file that is not a JSON object is skipped, since a skill may
    ship ``input.json`` for some other purpose.
    """
    if explicit is not None:
        if not explicit.is_file():
            raise ValueError(f"--sample-input file not found: {explicit}")
        return load_sample(explicit)
    for name in SAMPLE_INPUT_NAMES:
        path = skill_dir / name
        if path.is_file():
            try:
                return load_sample(path)
            except ValueError:
                continue
    if probe_dir is not None:
        return _baseline_sample(probe_dir)
    return None


def _baseline_sample(probe_dir: Path) -> Sample | None:
    """The payload a ``rote baseline`` run actually ran the skill on."""
    from rote.eval.baseline import METRICS_FILENAME

    path = probe_dir / METRICS_FILENAME
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8")).get("input")
    except (json.JSONDecodeError, AttributeError):
        return None
    if not isinstance(payload, dict) or not payload:
        return None
    return Sample(payload=payload, source=f"the input the baseline ran on ({path})")


# ───────── the contract ─────────


def input_contract(sample: Sample) -> dict[str, Any]:
    """The JSON Schema of the sample input, extracted by code.

    :func:`rote.probe.infer_schema` over the one payload: a key is required
    when every row of a list carries it, optional when only some do, and
    nested objects and lists are described the same way.
    """
    from rote.probe import infer_schema

    return infer_schema([sample.payload])


def _describe(schema: dict[str, Any], indent: str = "") -> list[str]:
    """``schema``'s properties as indented ``- name (type)`` lines, rows included."""
    lines: list[str] = []
    required = set(schema.get("required") or [])
    for name, prop in (schema.get("properties") or {}).items():
        kind = prop.get("type") or ("|".join(v.get("type", "?") for v in prop.get("anyOf", [])))
        note = "" if name in required else ", optional"
        if kind == "array" and (prop.get("items") or {}).get("type") == "object":
            lines.append(f"{indent}- `{name}` (list of objects{note}), each row with:")
            lines += _describe(prop["items"], indent + "  ")
        else:
            lines.append(f"{indent}- `{name}` ({kind}{note})")
    return lines


def materialize(work_dir: Path, sample: Sample) -> Path:
    """Copy the sample and its contract into the agent's work dir."""
    ctx = work_dir / GROUNDING_DIRNAME
    ctx.mkdir(parents=True, exist_ok=True)
    (ctx / SAMPLE_FILENAME).write_text(json.dumps(sample.payload, indent=2) + "\n", "utf-8")
    (ctx / CONTRACT_FILENAME).write_text(
        json.dumps(input_contract(sample), indent=2) + "\n", "utf-8"
    )
    return ctx


def remove_materialized(work_dir: Path) -> None:
    shutil.rmtree(work_dir / GROUNDING_DIRNAME, ignore_errors=True)


def contract_brief(sample: Sample | None, skill_dir: Path, ctx_dir: Path | None) -> str:
    """The prompt section that fixes the input contract before the agent starts."""
    parts: list[str] = []
    if sample is not None and ctx_dir is not None:
        schema = input_contract(sample)
        parts.append(
            "INPUT CONTRACT (decided by rote from a sample input, not open to "
            f"interpretation). The skill's sample input is {ctx_dir / SAMPLE_FILENAME} "
            f"(from {sample.source}); its JSON Schema is {ctx_dir / CONTRACT_FILENAME}. "
            "The pipeline must run on that file exactly as it is:\n"
            "- `input.required` lists only fields the sample has, with these exact "
            "names; `input.input_schema` is the contract schema (you may drop "
            "fields no node reads);\n"
            "- every function that reads a row of a list uses the row keys below, "
            "exactly as spelled, never a more descriptive name for one;\n"
            "- no other pipeline input is required. A value the sample lacks is a "
            "constant, a file the skill bundles, or a node's output.\n"
            "Fields:\n" + "\n".join(_describe(schema)) + "\n"
            "rote runs the pure functions on this sample after you finish; a "
            "KeyError or a missing input goes back to you as a failure. Do not "
            f"read {ctx_dir} from tests: copy the values a test needs into the test."
        )
        odd = unbindable_fields(sample)
        if odd:
            parts.append(
                "These sample fields are not identifiers, so no `pipeline.input.<field>` "
                f"reference can bind them: {', '.join(f'`{k}`' for k in odd)}. Leave them "
                "out of `input.required`, bind `pipeline.input` whole to the node that "
                "reads them, and read them by key there."
            )
        for signal, gate in sample.gates.items():
            keys = ", ".join(f"`{k}`" for k in gate)
            parts.append(
                f"GATE PAYLOAD `{signal}` (from the same sample): the person's answer "
                f"has the keys {keys}. The hitl_gate that collects it uses exactly "
                "these keys in its `output`."
            )
    refs = reference_files(skill_dir)
    if refs:
        listed = ", ".join(f"`references/{p}`" for p in refs)
        parts.append(
            f"Files the skill bundles ({listed}) are read by the pipeline itself: "
            "their content is a node's constants or a file read, never a pipeline "
            "input field."
        )
    named = named_human_items(skill_dir)
    if named:
        lines = [
            f'- section "{heading}": ' + ", ".join(f"**{term}** -> `{key}`" for term, key in items)
            for heading, items in named.items()
        ]
        parts.append(
            "The skill names what a person provides. A hitl_gate that collects "
            "these items uses these payload keys, the skill's own nouns in "
            "snake_case:\n" + "\n".join(lines)
        )
    return "\n\n".join(parts)


# ───────── the skill's own words ─────────


def reference_files(skill_dir: Path) -> list[str]:
    """Every file under ``references/``, relative to it, sorted."""
    root = skill_dir / "references"
    if not root.is_dir():
        return []
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())


def _words(text: str) -> list[str]:
    return [w for w in re.split(r"[^a-z0-9]+", text.casefold()) if w]


def snake(term: str) -> str:
    """``"Dock Appointment Completion (from the scheduler)"`` -> ``dock_appointment_completion``."""
    return "_".join(_words(re.sub(r"\([^)]*\)", " ", term)))


#: A section asks a person for something when its heading or prose says so.
_HUMAN_ASK_RE = re.compile(
    r"\b(ask|asks|request|provide|provides|approve|approves|approval|confirm|"
    r"decision|decide|enter|supply)\b",
    re.IGNORECASE,
)
#: A bold list item that is one of the person's options, not a thing they
#: provide: ``- **Approve**: ...`` is a value of the payload, not a key.
CHOICE_WORDS = frozenset(
    {
        "accept",
        "accepted",
        "approve",
        "approved",
        "cancel",
        "cancelled",
        "confirm",
        "confirmed",
        "decline",
        "declined",
        "defer",
        "deny",
        "denied",
        "escalate",
        "hold",
        "keep",
        "no",
        "pass",
        "reject",
        "rejected",
        "renegotiate",
        "retry",
        "skip",
        "yes",
    }
)
#: ``1. **Name**`` or ``- **Name**``: a list item that names what it is.
_NAMED_ITEM_RE = re.compile(r"^\s*(?:\d+[.)]|[-*+])\s+\*\*(?P<term>[^*\n]+?)\*\*")


def named_human_items(skill_dir: Path) -> dict[str, list[tuple[str, str]]]:
    """``{section heading: [(term, snake key)]}`` for lists a person is asked to fill.

    A list counts when what introduces it asks a person for something: its
    lead-in line (``After pulling the above, ask the duty manager to
    provide:``), or the section's heading when the list opens the section.
    Its bold item names are then the skill's nouns for what the person
    provides. Reading that from words is a guess, so a gate that does not
    use them is a review gap (:func:`gate_findings`), never an error.

    A section that only *mentions* asking somewhere used to qualify whole, so
    a workflow whose step 2 said "ask the user whether to move or skip" made
    its four step titles required answer keys (#319).
    """
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.is_file():
        return {}
    found: dict[str, list[tuple[str, str]]] = {}
    for heading, text in parse_sections(skill_md.read_text(encoding="utf-8")).items():
        if not heading:
            continue
        items: list[tuple[str, str]] = []
        # What introduces the next list: the heading until prose appears.
        lead: str | None = heading
        asking = False
        in_list = False
        # A list nested under an item ("2. Ask the operator to provide:" and
        # its sub-bullets) is introduced by that item: (its indent, asking).
        parent: tuple[int, bool] | None = None
        previous_item = ""
        for line in text.splitlines():
            if not line.strip():
                continue
            if line.lstrip().startswith("#"):
                lead, in_list, parent = heading, False, None  # the section's own heading line
                continue
            item = _NAMED_ITEM_RE.match(line)
            listed = item is not None or _LIST_ITEM_RE.match(line) is not None
            indent = len(line) - len(line.lstrip())
            if listed and not in_list:
                asking = lead is not None and _asks_for_what_follows(lead)
                parent = (indent, False)
            elif listed and parent is not None and indent > parent[0]:
                if not parent[1] and _asks_for_what_follows(previous_item):
                    parent = (parent[0], True)
            elif listed and parent is not None:
                parent = (indent, False)
            in_list = listed
            if not listed:
                if not line.startswith((" ", "\t")):
                    lead = line
                continue
            nested = parent is not None and indent > parent[0]
            if item is not None and (asking or (nested and parent is not None and parent[1])):
                term = item.group("term")
                items.append((term.strip().rstrip(":"), snake(term)))
            if not nested:
                previous_item = _LIST_ITEM_RE.sub("", line).replace("**", "")
        items = [(t, k) for t, k in items if k and k not in CHOICE_WORDS]
        if items:
            found[heading] = items
    return found


#: Any list item, named or not: it continues a list rather than introducing one.
_LIST_ITEM_RE = re.compile(r"^\s*(?:\d+[.)]|[-*+])\s+")


def _asks_for_what_follows(lead: str) -> bool:
    """Whether ``lead`` (a sentence or a heading) introduces a list of things
    a person is asked for: it asks, and it ends in a colon, the way a lead-in
    does. A heading such as "Approval gates" names a topic, not a request."""
    return bool(_HUMAN_ASK_RE.search(lead)) and lead.rstrip().endswith(":")


def _input_wording(skill_dir: Path) -> str | None:
    """The first sentence of SKILL.md that says what the input is."""
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.is_file():
        return None
    text = re.sub(r"(?s)^---\n.*?\n---\n", "", skill_md.read_text(encoding="utf-8"))
    flat = re.sub(r"\s+", " ", text)
    for sentence in re.split(r"(?<=[.!?])\s+", flat):
        if re.search(r"\binputs?\b", sentence, re.IGNORECASE):
            return sentence.strip()[:_WORDING_CHARS]
    return None


# ───────── checks ─────────


def _declared_inputs(pipeline: Pipeline) -> set[str]:
    schema = pipeline.input.input_schema or {}
    return (
        set(pipeline.input.required)
        | set(pipeline.input.optional)
        | set((schema.get("properties") or {}).keys())
    )


def _type_ok(value: Any, kind: Any) -> bool:
    kinds = kind if isinstance(kind, list) else [kind]
    checks = {
        "object": lambda v: isinstance(v, dict),
        "array": lambda v: isinstance(v, list),
        "string": lambda v: isinstance(v, str),
        "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        "boolean": lambda v: isinstance(v, bool),
        "null": lambda v: v is None,
    }
    return any(checks[k](value) for k in kinds if k in checks) or not any(
        k in checks for k in kinds
    )


def schema_mismatches(schema: dict[str, Any], value: Any, path: str = "input") -> list[str]:
    """Where ``value`` breaks ``schema``: type, required keys, list rows.

    A deliberately small reading of JSON Schema (``type``, ``properties``,
    ``required``, ``items``), the parts a pipeline input schema uses to
    say what it reads. Every row of a list is checked, so a key only some
    rows carry is named with the first row that lacks it.
    """
    problems: list[str] = []
    kind = schema.get("type")
    if kind is not None and not _type_ok(value, kind):
        return [f"{path} is {type(value).__name__}, the schema says {kind}"]
    if isinstance(value, dict):
        for key in schema.get("required") or []:
            if key not in value:
                problems.append(
                    f"{path} has no `{key}` (has: {', '.join(sorted(value)) or 'none'})"
                )
        for key, sub in (schema.get("properties") or {}).items():
            if key in value and isinstance(sub, dict):
                problems += schema_mismatches(sub, value[key], f"{path}.{key}")
    elif isinstance(value, list) and isinstance(schema.get("items"), dict):
        for i, row in enumerate(value):
            row_problems = schema_mismatches(schema["items"], row, f"{path}[{i}]")
            if row_problems:
                problems += row_problems
                break  # one row names the problem; the rest repeat it
    return problems


def input_findings(pipeline: Pipeline, sample: Sample) -> list[ContractFinding]:
    """The pipeline's input contract against the sample input."""
    findings: list[ContractFinding] = []
    have = sorted(sample.payload)
    missing = [f for f in pipeline.required_input_fields if f not in sample.payload]
    if missing:
        findings.append(
            ContractFinding(
                node_id=INPUT_NODE,
                kind="input_not_in_sample",
                severity="error",
                message=(
                    f"the pipeline requires input field(s) {', '.join(missing)}, which the "
                    f"skill's sample input ({sample.source}) does not have; it has "
                    f"{', '.join(have) or 'no fields'}. Rename each to the sample's field, "
                    f"or make it a constant or a node's output: the pipeline must run on "
                    f"the sample as it is"
                ),
                path="pipeline.yaml",
            )
        )
    if pipeline.input.input_schema:
        problems = schema_mismatches(pipeline.input.input_schema, sample.payload)
        if problems:
            findings.append(
                ContractFinding(
                    node_id=INPUT_NODE,
                    kind="sample_rejected_by_schema",
                    severity="error",
                    message=(
                        f"input.input_schema rejects the skill's sample input "
                        f"({sample.source}): {'; '.join(problems[:5])}. Make the schema "
                        f"describe the sample's real field and row names"
                    ),
                    path="pipeline.yaml",
                )
            )
    return findings


def reference_input_findings(pipeline: Pipeline, skill_dir: Path) -> list[ContractFinding]:
    """A pipeline input that is really a file the skill bundles.

    The field's words contain every word of a ``references/`` file's
    name (``fx_rates_markdown`` and ``fx-rates.md``): the skill reads that
    file itself, so the pipeline must too. A one-word name must match the
    whole field (``policy`` for ``policy.md``), since ``policy_id`` is a
    real input. Only fields the pipeline requires count; an optional
    override is the user's choice.
    """
    findings: list[ContractFinding] = []
    stems = {
        ref: set(_words(Path(ref).stem))
        for ref in reference_files(skill_dir)
        if _words(Path(ref).stem)
    }
    for name in pipeline.required_input_fields:
        words = set(_words(name))
        for ref, stem in stems.items():
            if (len(stem) >= 2 and stem <= words) or stem == words:
                findings.append(
                    ContractFinding(
                        node_id=INPUT_NODE,
                        kind="reference_as_input",
                        severity="error",
                        message=(
                            f"pipeline input `{name}` is the content of references/{ref}, "
                            f"which the skill bundles and reads itself. Make it a constant "
                            f"of the node that uses it (or a file read), not an input "
                            f"the caller must supply"
                        ),
                        path="pipeline.yaml",
                    )
                )
                break
    return findings


def _heading_key(heading: str) -> str:
    """A heading as a node cites it: case, markup and trailing colon do not count."""
    return " ".join(_words(heading))


def _gates_in_order(pipeline: Pipeline) -> list[Any]:
    return [n for wave in pipeline.execution_waves() for n in wave if n.kind is NodeKind.HITL_GATE]


@dataclass(frozen=True)
class GateView:
    """What matching a sample's gate answer to a gate needs to know about it."""

    id: str
    signal: str | None
    #: The gate's own payload keys: the compile's ``output``, or for a
    #: System-1 skeleton the step's wording (``approved``, ``rejected``).
    keys: frozenset[str]


def match_sample_gates(gates: list[GateView], sample: Sample) -> dict[str, str]:
    """``{sample signal: gate id}``, one gate per answer, in a fixed order of tries.

    The one matcher for both places a sample's gate answer meets a gate:
    System-1 assigning payload keys to its skeleton, and the check of a
    compiled IR. ``gates`` are in execution-wave order.

    1. A gate whose signal is the sample's signal name.
    2. Else the unmatched gate sharing the most words with the answer's
       keys (``approved_ids`` and ``approved`` share ``approved``); the
       earlier gate wins a tie.
    3. Else the next unmatched gate in execution order, which is the case
       where every key was renamed, so the check fails on that gate.
    """
    matched: dict[str, str] = {}
    taken: set[str] = set()
    pending = list(sample.gates.items())
    for signal, _ in pending:
        for gate in gates:
            if gate.id not in taken and gate.signal == signal:
                matched[signal] = gate.id
                taken.add(gate.id)
                break
    for signal, answer in pending:
        if signal in matched:
            continue
        want = {w for k in answer for w in _words(k)}
        best: tuple[int, GateView] | None = None
        for gate in gates:
            if gate.id in taken:
                continue
            score = len(want & {w for k in gate.keys for w in _words(k)})
            if score > 0 and (best is None or score > best[0]):
                best = (score, gate)
        if best is not None:
            matched[signal] = best[1].id
            taken.add(best[1].id)
    for signal, _ in pending:
        if signal in matched:
            continue
        free = next((g for g in gates if g.id not in taken), None)
        if free is None:
            break
        matched[signal] = free.id
        taken.add(free.id)
    return matched


#: A gate whose payload keys are not the skill's names for what it collects.
GATE_PAYLOAD_NOT_SKILL_NOUNS = "gate_payload_not_skill_nouns"


def gate_findings(
    pipeline: Pipeline, skill_dir: Path | None, sample: Sample | None
) -> list[ContractFinding]:
    """Gate payload keys against the sample's gate payloads and the skill's nouns."""
    findings: list[ContractFinding] = []
    gates = _gates_in_order(pipeline)
    if sample is not None:
        by_id = {g.id: g for g in gates}
        views = [GateView(g.id, g.signal, g.output_keys) for g in gates]
        for signal, gate_id in match_sample_gates(views, sample).items():
            want = sample.gates[signal]
            gate = by_id[gate_id]
            have = gate.output_keys
            absent = sorted(set(want) - have)
            if absent:
                findings.append(
                    ContractFinding(
                        node_id=gate.id,
                        kind="gate_payload_not_sample",
                        severity="error",
                        message=(
                            f"gate {gate.id} declares payload keys "
                            f"{', '.join(sorted(have)) or 'none'}, but the sample's answer at "
                            f"this gate ({sample.source}) has {', '.join(want)}. Declare "
                            f"{', '.join(absent)} in the gate's `output` with these names"
                        ),
                        path="pipeline.yaml",
                    )
                )
    named = {
        _heading_key(h): (h, items)
        for h, items in (named_human_items(skill_dir) if skill_dir is not None else {}).items()
    }
    # Several gates may split one section's items between them (one gate per
    # item the person supplies, #291), so a section is held to what all of
    # its gates collect together; each gate need not collect its siblings'.
    by_section: dict[str, list[Node]] = {}
    for gate in gates:
        cited = gate.source.section if gate.source is not None else None
        if cited:
            by_section.setdefault(_heading_key(cited), []).append(gate)
    for key, citing in by_section.items():
        section, items = named.get(key, (None, None))
        if not items:
            continue
        gate = citing[0]
        have = frozenset().union(*(g.output_keys for g in citing))
        absent = [(t, k) for t, k in items if k not in have]
        if absent:
            renames = ", ".join(f"**{t}** -> `{k}`" for t, k in absent)
            declared = ", ".join(sorted(have)) or "none"
            who = (
                f"gate {gate.id} collects"
                if len(citing) == 1
                else f"gates {', '.join(g.id for g in citing)} together collect"
            )
            findings.append(
                ContractFinding(
                    node_id=gate.id,
                    kind=GATE_PAYLOAD_NOT_SKILL_NOUNS,
                    # A guess about meaning, from words (#319): repaired once,
                    # listed if it survives, never a failed compile. A step
                    # reading a key the gate lacks is the boundary check's
                    # error (rote.compiler.boundaries).
                    severity="warning",
                    message=(
                        f'{who} what section "{section}" asks the person '
                        f"for, but their payload keys ({declared}) do not use "
                        f"the skill's names for those items: {renames}. Name each key after "
                        f"the item, in snake_case, and update the code that reads them"
                    ),
                    path="pipeline.yaml",
                )
            )
    return findings


def grounding_findings(
    pipeline: Pipeline, skill_dir: Path | None, sample: Sample | None
) -> list[ContractFinding]:
    """Every static contract check: the sample, the bundled files, the gates."""
    findings: list[ContractFinding] = []
    if sample is not None:
        findings += input_findings(pipeline, sample)
    if skill_dir is not None:
        findings += reference_input_findings(pipeline, skill_dir)
    return findings + gate_findings(pipeline, skill_dir, sample)


# ───────── open questions ─────────


@dataclass(frozen=True)
class ContractQuestion:
    """A contract rote could not ground in data, listed for a reviewer."""

    node_id: str
    text: str

    def as_dict(self) -> dict[str, str]:
        return {"node": self.node_id, "text": self.text}


@dataclass(frozen=True)
class Grounding:
    """What a compile is grounded in: the sample, if any, and recorded tool results."""

    sample: Sample | None = None
    #: ``rote baseline`` observations, for the prefix run's MCP stubs.
    observations: tuple[Any, ...] = field(default_factory=tuple)
    #: ``{(server, tool): ToolSchema}`` (:mod:`rote.compiler.tool_grounding`):
    #: each MCP tool's input schema, declared by its server or inferred from
    #: recorded calls.
    tools: dict[tuple[str, str], Any] = field(default_factory=dict)
    #: ``{server: why}`` for each registered server whose tools rote did
    #: not list: not named in the skill, or its ``tools/list`` failed.
    unlisted: dict[str, str] = field(default_factory=dict)
    #: The skill's bundled tools/list snapshot, as read, whether or not a
    #: live listing replaced it: kept to tell when it is stale.
    bundled: dict[tuple[str, str], Any] = field(default_factory=dict)


def unbindable_fields(sample: Sample) -> list[str]:
    """Top-level sample fields no ``pipeline.input.<field>`` reference can bind.

    Not a reason to reject the sample: the data is what it is, and a node
    can take the whole input and read ``request["Invoice #"]``. They are
    named for the agent and listed as an open question instead.
    """
    return sorted(k for k in sample.payload if not is_field_name(k))


def contract_questions(
    pipeline: Pipeline, skill_dir: Path | None, grounding: Grounding
) -> list[ContractQuestion]:
    """The input contract and MCP result shapes nothing recorded, for review."""
    questions: list[ContractQuestion] = []
    if grounding.sample is not None and unbindable_fields(grounding.sample):
        odd = ", ".join(f"`{k}`" for k in unbindable_fields(grounding.sample))
        questions.append(
            ContractQuestion(
                INPUT_NODE,
                f"sample input field(s) {odd} ({grounding.sample.source}) are not "
                f"identifiers, so they cannot be declared in input.required or bound one "
                f"by one; a node must take the whole input to read them. Renaming them "
                f"where the data comes from would let the entry check name them",
            )
        )
    if grounding.sample is None:
        fields = sorted(_declared_inputs(pipeline) | set(pipeline.required_input_fields))
        wording = _input_wording(skill_dir) if skill_dir is not None else None
        source = (
            f'from the skill\'s wording: "{wording}"'
            if wording
            else "the skill does not describe its input"
        )
        questions.append(
            ContractQuestion(
                INPUT_NODE,
                f"input contract chosen without a sample input: "
                f"{', '.join(f'`{f}`' for f in fields) or 'no fields'} ({source}). "
                f"Bundle a sample (input.json in the skill dir, or --sample-input) so "
                f"rote can check it",
            )
        )
    observed = {(o.server, o.tool) for o in grounding.observations if o.result is not None}
    consumers: dict[str, set[str]] = {}
    for node in pipeline.nodes:
        for ref in (node.inputs or {}).values():
            parsed = parse_input_ref(ref)
            if parsed.node_id is not None:
                consumers.setdefault(parsed.node_id, set()).add(
                    "the whole result" if parsed.field is None else f"key `{parsed.field}`"
                )
    for node in pipeline.nodes:
        if node.mcp is None or (node.mcp.server, node.mcp.tool) in observed:
            continue
        read = consumers.get(node.id)
        reads = f"; downstream nodes read {', '.join(sorted(read))}" if read else ""
        questions.append(
            ContractQuestion(
                node.id,
                f"result shape of {node.mcp.server}.{node.mcp.tool} is assumed, not "
                f"recorded{reads}. Run `rote baseline` against the real server to record it",
            )
        )
    return questions


def render_questions(questions: list[ContractQuestion]) -> str:
    """Markdown bullets for the compile report."""
    return "\n".join(f"- {q.node_id}: {q.text}" for q in questions)
