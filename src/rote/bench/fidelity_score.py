"""Behavioral fidelity scoring: does the compiled pipeline do what the skill did?

Pure functions over JSON data, no processes and no inference, so every
rule here is unit tested (``tests/test_bench_fidelity.py``). The runs
that produce the data live in :mod:`rote.bench.fidelity`.

The reference is the skill itself, run as an agent several times on the
same case. Its result.json follows a fixed output contract (the case
file names the fields), and the fixture MCP server logs every tool call.
Agents are not deterministic, so only what is **stable** across the
reference runs counts against a pipeline: a field whose normalized value
differs between two agent runs of the same skill on the same input says
nothing about the pipeline.

The pipeline's names are the compiler's choice, not the contract's, so
outputs are matched by *value*, not by key: a stable reference value
counts as reproduced when it appears anywhere in the pipeline's trace
(its final output and every DBOS step output, gate payloads included).
Lists compare as sets; a list of records compares per field (the set of
``expense_id`` values in a review queue). Values that would match by
coincidence (small integers, booleans, short strings, empty lists) also
need a key whose name shares a word with the contract field.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from rote.bench.score import _tokens

#: A reference value: a normalized scalar, or ``{"set": [...]}`` for a
#: list (sorted by its JSON text, so it is comparable and serializable).
Value = Any

_NUMBER_RE = re.compile(r"^[\s$€£]*([-+]?\d[\d,]*(?:\.\d+)?)\s*%?\s*$")
_WS_RE = re.compile(r"\s+")


def normalize(value: Any) -> Any:
    """One canonical form per scalar.

    Numbers round to the cent, so ``3304.150000001`` and ``3304.15`` agree,
    and an int and a float of the same amount are one value. Strings are
    case-folded with whitespace collapsed, and a string that is only a
    number (``"$1,234.50"``, ``"75%"``) becomes that number: an agent
    writes amounts as text as often as not. A ``Decimal`` (money code
    returns them, and DBOS step outputs keep them) is a number like any
    other, and any other non-JSON scalar (a date) is compared as its text.
    """
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float, Decimal)):
        return round(float(value), 2)
    if isinstance(value, str):
        m = _NUMBER_RE.match(value)
        if m:
            return round(float(m.group(1).replace(",", "")), 2)
        return _WS_RE.sub(" ", value.strip()).casefold()
    return normalize(str(value))


def _is_scalar(value: Any) -> bool:
    return not isinstance(value, (dict, list, tuple, set))


def _as_set(values: Iterable[Any]) -> dict[str, list[Any]]:
    unique = {json.dumps(normalize(v), sort_keys=True): normalize(v) for v in values}
    return {"set": [unique[k] for k in sorted(unique)]}


def flatten(obj: Any, prefix: str = "") -> dict[str, Value]:
    """Leaf paths of a result object, normalized.

    ``{"a": {"b": 1}}`` gives ``a.b``. A list of scalars is one leaf (a
    set). A list of records gives one set per scalar field, at
    ``path[].field``, because a list's order and its records' other
    fields are presentation, while *which* ids landed in the review queue
    is behavior.
    """
    out: dict[str, Value] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            out.update(flatten(value, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(obj, list):
        if all(_is_scalar(v) for v in obj):
            out[prefix] = _as_set(obj)
        elif all(isinstance(v, dict) for v in obj):
            keys = {k for v in obj for k in v}
            for key in sorted(keys):
                column = [v.get(key) for v in obj]
                if all(_is_scalar(c) for c in column):
                    out[f"{prefix}[].{key}"] = _as_set(column)
    else:
        out[prefix] = normalize(obj)
    return out


def layout_path(path: str) -> str:
    """``path`` with the nesting between its contract field and its leaf removed.

    ``report.kpi.active_sites`` and ``report.header.active_sites`` are the
    same number laid out two ways; ``report.active_sites`` is what both
    say. A record-list marker survives (``queue[].expense_id``), because a
    column of a list is not the same thing as a single value.
    """
    parts = path.split(".")
    if len(parts) <= 2:
        return path
    marker = "[]" if any(p.endswith("[]") for p in parts[:-1]) else ""
    return f"{parts[0].removesuffix('[]')}{marker}.{parts[-1]}"


def flatten_layout(obj: Any) -> dict[str, Value]:
    """:func:`flatten`, keyed by :func:`layout_path` where that is safe.

    A leaf collapses onto its layout path only when nothing else in the run
    collapses onto the same path with a different value. Two leaves that
    differ (``decisions.approved.ids`` and ``decisions.rejected.ids`` both
    collapse to ``decisions.ids``) keep their full paths: merging them
    would make a run that swapped approved and rejected agree with one
    that did not. Two leaves with the same value (a total repeated in two
    sections) are one fact and collapse.
    """
    groups: dict[str, dict[str, Value]] = {}
    for path, value in flatten(obj).items():
        groups.setdefault(layout_path(path), {})[path] = value
    out: dict[str, Value] = {}
    for layout, members in groups.items():
        if len({_key(v) for v in members.values()}) == 1:
            out[layout] = next(iter(members.values()))
        else:
            out.update(members)
    return out


def _key(value: Value) -> str:
    return json.dumps(value, sort_keys=True)


@dataclass
class TraceIndex:
    """Every value a pipeline run produced, with the keys it appeared under."""

    values: dict[str, set[str]] = field(default_factory=dict)
    strings: set[str] = field(default_factory=set)

    def add(self, value: Value, key: str) -> None:
        self.values.setdefault(_key(value), set()).add(key)
        if isinstance(value, str):
            self.strings.add(value)


def index_trace(objects: Iterable[Any]) -> TraceIndex:
    """Index scalars, scalar lists (as sets) and record-list columns."""
    index = TraceIndex()

    def walk(obj: Any, key: str) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, str(k))
        elif isinstance(obj, list):
            if all(_is_scalar(v) for v in obj):
                index.add(_as_set(obj), key)
            elif all(isinstance(v, dict) for v in obj):
                for col in sorted({k for v in obj for k in v}):
                    column = [v.get(col) for v in obj]
                    if all(_is_scalar(c) for c in column):
                        index.add(_as_set(column), col)
            for v in obj:
                walk(v, key)
        else:
            index.add(normalize(obj), key)

    for obj in objects:
        walk(obj, "")
    return index


def _informative(value: Value) -> bool:
    """Whether a value is unlikely to appear in a trace by coincidence."""
    if isinstance(value, dict):
        items = value.get("set", [])
        return bool(items) and (len(items) > 1 or _informative(items[0]))
    if value is None or isinstance(value, bool):
        return False
    if isinstance(value, float):
        return not (value.is_integer() and abs(value) < 10)
    if isinstance(value, str):
        return len(value) >= 3
    return True


def _leaf_tokens(path: str) -> set[str]:
    return _tokens(path.rsplit(".", 1)[-1].replace("[]", ""))


#: Shortest reference string matched inside a longer trace string. Short
#: strings (a status like ``final``) must match whole, or they would be
#: found inside unrelated text.
MIN_SUBSTRING = 7


def field_reproduced(path: str, value: Value, index: TraceIndex) -> bool:
    """Whether a stable reference leaf appears in the pipeline's trace.

    A string of at least :data:`MIN_SUBSTRING` characters also counts when
    it is part of a longer trace string: an agent reports the file name
    (``2026-09-expense-audit.md``) where a pipeline reports the full path.
    """
    keys = index.values.get(_key(value))
    if not keys:
        return (
            isinstance(value, str)
            and len(value) >= MIN_SUBSTRING
            and any(value in s for s in index.strings)
        )
    if _informative(value):
        return True
    want = _leaf_tokens(path)
    return any(want & _tokens(k) for k in keys)


# ───────── tool calls ─────────


def call_key(call: dict[str, Any]) -> str:
    """A call's identity.

    A read is identified by the fixture response that answered it: what
    matters is which data the run looked at, not how it spelled the query.
    A side effect, and a read no fixture answered, by server, tool and
    normalized arguments: for a write the arguments are the behavior.
    """
    if not call.get("side_effect") and call.get("response") is not None:
        return f"{call['server']}.{call['tool']}->response[{call['response']}]"
    args = {k: normalize(v) if _is_scalar(v) else v for k, v in call["arguments"].items()}
    return f"{call['server']}.{call['tool']}({json.dumps(args, sort_keys=True)})"


def _arg_value(value: Any) -> Any:
    return normalize(value) if _is_scalar(value) else value


def write_arguments(runs: list[dict[str, Any]]) -> dict[str, dict[str, list[str]]]:
    """``{"server.tool": {"kept": [...], "dropped": [...]}}`` for every written tool.

    An argument is kept, and identifies the write, when every run sent the
    same values for it to that tool (as a multiset, so two runs writing the
    same records in a different order still agree on them). Prose arguments
    differ in every agent run (a drafted brief, a message body): kept in the
    key, no two runs would ever agree on a write, and a pipeline that made
    the same write would count as an extra one. They are dropped, and the
    scorecard says so. A tool none of whose arguments agree is identified by
    the tool alone.
    """
    tools: dict[str, set[str]] = {}
    for run in runs:
        for c in run.get("calls", []):
            if c.get("side_effect"):
                tools.setdefault(f"{c['server']}.{c['tool']}", set()).update(c["arguments"])
    out: dict[str, dict[str, list[str]]] = {}
    for tool, names in sorted(tools.items()):
        kept: list[str] = []
        dropped: list[str] = []
        for name in sorted(names):
            per_run = {
                json.dumps(
                    sorted(
                        json.dumps(_arg_value(c["arguments"].get(name)), sort_keys=True)
                        for c in run.get("calls", [])
                        if c.get("side_effect") and f"{c['server']}.{c['tool']}" == tool
                    )
                )
                for run in runs
            }
            (kept if len(per_run) == 1 else dropped).append(name)
        out[tool] = {"kept": kept, "dropped": dropped}
    return out


def write_fields(runs: list[dict[str, Any]]) -> dict[str, list[str]]:
    """``{"server.tool": argument names}`` a write is identified by (:func:`write_arguments`)."""
    return {tool: a["kept"] for tool, a in write_arguments(runs).items()}


def effect_key(call: dict[str, Any], fields: dict[str, list[str]] | None) -> str:
    """A write's identity: server, tool and the arguments the reference agrees on.

    ``fields`` from :func:`write_fields`; ``None`` keys on every argument
    (:func:`call_key`). A tool no reference run wrote keeps every argument.
    """
    tool = f"{call['server']}.{call['tool']}"
    if fields is None or tool not in fields:
        return call_key(call)
    args = {k: _arg_value(call["arguments"][k]) for k in fields[tool] if k in call["arguments"]}
    return f"{tool}({json.dumps(args, sort_keys=True)})"


def _lcs(a: list[str], b: list[str]) -> int:
    table = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            table[i + 1][j + 1] = (
                table[i][j] + 1 if x == y else max(table[i][j + 1], table[i + 1][j])
            )
    return table[-1][-1]


# ───────── the reference ─────────


#: Fewest successful agent runs a case needs before anything counts as
#: stable. With one run, every field "agrees with itself", so the
#: reference would look complete while measuring nothing about variance.
MIN_REFERENCE_RUNS = 2

#: A case whose stable contract coverage is below this is flagged: most of
#: what the skill computes there is not being checked.
LOW_COVERAGE = 0.5


def _head(path: str) -> str:
    return re.split(r"[.\[]", path, maxsplit=1)[0]


def stable_reference(
    runs: list[dict[str, Any]],
    contract: Iterable[str] | None = None,
    min_runs: int = MIN_REFERENCE_RUNS,
) -> dict[str, Any]:
    """What every successful reference run agrees on.

    ``runs`` are ``{"output": {...} | None, "calls": [...], "error": ...}``.
    Fields: a leaf counts when every successful run has it with one value,
    compared under :func:`layout_path`, so the agent nesting the same
    number differently in two runs is not a disagreement. Reads: calls (by
    :func:`call_key`) made in every run. Side effects: the ordered
    sequence, when every run made the same one; otherwise only the calls
    every run made, unordered. A write is keyed by the arguments every run
    agreed on (``write_fields``, :func:`write_fields`). ``seen_calls`` is every call any run made:
    a pipeline call outside it is one no agent run ever made.

    ``contract`` limits fields to the output contract's top-level names: an
    agent that adds its own extra sections to result.json in one run
    should not make the reference look unstable. Fewer than ``min_runs``
    successful runs make nothing stable (``insufficient_runs``).

    ``coverage`` says how much of the contract is checked: contract fields
    with at least one stable leaf, out of all of them, and for the rest
    whether the runs disagreed on the values, only on where they put them,
    or left the field out.
    """
    ok = [r for r in runs if r.get("output") is not None and not r.get("error")]
    keep = None if contract is None else list(dict.fromkeys(contract))
    flats = [
        {p: v for p, v in flatten_layout(r["output"]).items() if keep is None or _head(p) in keep}
        for r in ok
    ]
    insufficient = len(ok) < min_runs
    stable: dict[str, Value] = {}
    unstable: list[str] = []
    for path in sorted({p for f in flats for p in f}):
        values = {_key(f.get(path, "<missing>")) for f in flats}
        if len(values) == 1 and not insufficient:  # a missing field adds "<missing>"
            stable[path] = flats[0][path]
        else:
            unstable.append(path)

    fields = keep if keep is not None else sorted({_head(p) for f in flats for p in f})
    covered = sorted({_head(p) for p in stable})
    layout: list[str] = []
    value: list[str] = []
    missing: list[str] = []
    for name in fields:
        if name in covered or not flats:
            continue
        per_run = [{p: v for p, v in f.items() if _head(p) == name} for f in flats]
        if any(not r for r in per_run):
            missing.append(name)
        elif len({tuple(sorted(_key(v) for v in r.values())) for r in per_run}) == 1:
            layout.append(name)
        else:
            value.append(name)
    rate = round(len(covered) / len(fields), 4) if fields else None
    coverage = {
        "fields": len(fields),
        "covered": len(covered),
        "rate": rate,
        "low": rate is not None and rate < LOW_COVERAGE,
        # "covered" means at least one stable leaf; a field with one stable
        # leaf of twenty is covered. The leaf ratio says how much of it.
        "leaves": len(stable) + len(unstable),
        "stable_leaves": len(stable),
        "leaf_rate": (
            round(len(stable) / (len(stable) + len(unstable)), 4) if stable or unstable else None
        ),
        "uncovered_value_disagreements": value,
        "uncovered_layout_disagreements": layout,
        "uncovered_missing_in_some_runs": missing,
    }

    counted = [] if insufficient else ok
    written = write_arguments(counted)
    effect_fields = {tool: a["kept"] for tool, a in written.items()}

    def keys(run: dict[str, Any], side_effect: bool) -> list[str]:
        return [
            effect_key(c, effect_fields) if side_effect else call_key(c)
            for c in run.get("calls", [])
            if bool(c.get("side_effect")) == side_effect
        ]

    read_sets = [set(keys(r, False)) for r in counted]
    effect_seqs = [keys(r, True) for r in counted]
    ordered = len({tuple(s) for s in effect_seqs}) == 1
    effect_sets = [set(s) for s in effect_seqs]
    return {
        "runs": len(runs),
        "successful_runs": len(ok),
        "insufficient_runs": insufficient,
        "coverage": coverage,
        "fields": stable,
        "unstable_fields": unstable,
        "reads": sorted(set.intersection(*read_sets)) if read_sets else [],
        "side_effects": effect_seqs[0] if counted and ordered else None,
        "side_effects_unordered": sorted(set.intersection(*effect_sets)) if effect_sets else [],
        "write_fields": effect_fields,
        "write_fields_dropped": {t: a["dropped"] for t, a in written.items() if a["dropped"]},
        "seen_calls": sorted({call_key(c) for r in ok for c in r.get("calls", [])}),
    }


# ───────── one case ─────────


def _mean(values: Iterator[float | None] | list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def score_case(reference: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    """One case's fidelity: each component in ``[0, 1]``, and their mean.

    ``trace`` is a pipeline run: ``{"completed": bool, "objects": [every
    output and step output], "calls": [...], "gate_fit": [...] }``.
    Components with nothing to compare are ``None`` and left out of the
    mean, so a case with no tool calls is not scored on tool calls.
    """
    # A stable null is the agent agreeing it never computed the field (the
    # skill stopped first). Absence cannot be found by value, and the
    # outcome component already scores the stop, so it is not an output.
    fields = {p: v for p, v in reference["fields"].items() if p != "outcome" and v is not None}
    calls = trace.get("calls", [])
    # What the pipeline wrote is output it produced, wherever the value came
    # from: a reviewer known only from the gate's answer that the pipeline
    # sends on in its publish call carried the value through. The gate's
    # received payload itself stays out (crediting it would reward being
    # told the answer); only the pipeline's own writes count, and only ones
    # the server accepted: a write a real server would refuse produced nothing.
    written = [
        c.get("arguments") or {}
        for c in calls
        if c.get("side_effect") and c.get("matched") is not False and not c.get("invalid_arguments")
    ]
    index = index_trace([*trace.get("objects", []), *written])
    reproduced = {p: field_reproduced(p, v, index) for p, v in fields.items()}
    outputs = round(sum(reproduced.values()) / len(fields), 4) if fields else None

    ref_outcome = reference["fields"].get("outcome")
    run_outcome = "completed" if trace.get("completed") else "stopped"
    outcome = None if ref_outcome is None else float(ref_outcome == run_outcome)

    pipe_reads = {call_key(c) for c in calls if not c.get("side_effect")}
    ref_reads = set(reference["reads"])
    seen = set(reference["seen_calls"])
    if ref_reads or pipe_reads:
        recall = len(ref_reads & pipe_reads) / len(ref_reads) if ref_reads else 1.0
        precision = len(pipe_reads & seen) / len(pipe_reads) if pipe_reads else 0.0
        reads = round((recall + precision) / 2, 4)
    else:
        reads = None

    write_ids = reference.get("write_fields")
    pipe_effects = [effect_key(c, write_ids) for c in calls if c.get("side_effect")]
    ordered = reference.get("side_effects")
    if ordered is not None and (ordered or pipe_effects):
        effects: float | None = round(
            _lcs(ordered, pipe_effects) / max(len(ordered), len(pipe_effects)), 4
        )
    elif ordered is None and (reference["side_effects_unordered"] or pipe_effects):
        want = set(reference["side_effects_unordered"])
        got = set(pipe_effects)
        effects = round(len(want & got) / len(want | got), 4) if want | got else None
    else:
        effects = None

    gate_fit = trace.get("gate_fit") or []
    gates = _mean([g["fit"] for g in gate_fit]) if gate_fit else None

    components = {
        "outputs": outputs,
        "outcome": outcome,
        "reads": reads,
        "side_effects": effects,
        "gates": gates,
    }
    return {
        "score": _mean(list(components.values())),
        **components,
        "fields_checked": len(fields),
        "fields_missing": sorted(p for p, ok in reproduced.items() if not ok),
        "reads_missing": sorted(ref_reads - pipe_reads),
        "reads_unexpected": sorted(pipe_reads - seen),
        "side_effects_pipeline": pipe_effects,
    }


def fidelity_summary(cases: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The top-level ``fidelity`` number: the mean case score."""
    return {
        "score": _mean([c.get("score") for c in cases.values()]),
        "cases": cases,
    }
