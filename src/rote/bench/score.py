"""Structural scoring of a compiled pipeline against a hand-written one.

Pure functions over two :class:`~rote.ir.Pipeline` values: no files, no
subprocesses, no inference. Everything the benchmark reports about
*quality of the IR* comes from here, which is why it is the part with
unit tests (``tests/test_bench_score.py``).

Matching
--------
Every metric below compares nodes, so the first step pairs each
hand-written (expected) node with at most one compiled node:

1. Same ``id``: similarity 1.0.
2. Otherwise the token overlap of ``id`` and ``description``
   (:func:`node_similarity`), kept when it reaches
   :data:`MATCH_THRESHOLD`.

Pairs are taken greedily, highest similarity first, one-to-one. Greedy
is deliberate: it is deterministic, explainable in one sentence, and on
pipelines of 8 to 15 nodes it agrees with an optimal assignment on every
fixture tried. An unmatched expected node is *missing*; an unmatched
compiled node is *extra*.

Every rate is ``None`` when its denominator is zero (no expected
fan_out nodes, say), never 0 or 1: a skill with nothing to find must
not read as a perfect or a failed score, and aggregates skip ``None``.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Any

from rote.ir import Node, NodeKind, Pipeline, parse_input_ref

#: Minimum :func:`node_similarity` for two differently named nodes to
#: count as the same step. Chosen on the committed examples: real
#: renames (``fetch_dock_pending`` vs ``get_pending_dock_appointments``)
#: score 0.3 to 0.6, unrelated nodes in one pipeline stay under 0.2.
MATCH_THRESHOLD = 0.25

#: Words that carry no step identity. Removing them keeps two unrelated
#: descriptions from matching on "the", "each", "and".
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "each",
        "every",
        "for",
        "from",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "then",
        "this",
        "to",
        "with",
        "which",
        "when",
        "where",
        "who",
        "will",
        "via",
        "per",
        "all",
        "any",
        "not",
        "no",
        "if",
        "so",
        "than",
        "but",
        "one",
        "two",
        "only",
        "also",
        "over",
        "under",
        "node",
        "step",
    ]
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    """Lower-case word tokens of ``text`` (``_`` splits), minus stopwords."""
    words = _TOKEN_RE.findall(text.lower().replace("_", " "))
    return {w for w in words if w not in _STOPWORDS and len(w) > 1}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def node_similarity(expected: Node, compiled: Node) -> float:
    """How likely two nodes are the same step, in ``[0, 1]``.

    1.0 for an identical id. Otherwise the mean of the id-token Jaccard
    and the description-token Jaccard: ids carry the verb and object
    (``find_duplicates``), descriptions carry the domain words that
    survive a rename.
    """
    if expected.id == compiled.id:
        return 1.0
    id_score = _jaccard(_tokens(expected.id), _tokens(compiled.id))
    desc_score = _jaccard(
        _tokens(expected.id + " " + expected.description),
        _tokens(compiled.id + " " + compiled.description),
    )
    return (id_score + desc_score) / 2


@dataclass(frozen=True)
class NodeMatch:
    """One expected node paired with the compiled node judged to be it."""

    expected_id: str
    compiled_id: str
    similarity: float
    expected_kind: str
    compiled_kind: str

    @property
    def kind_equal(self) -> bool:
        return self.expected_kind == self.compiled_kind


def match_nodes(
    expected: Pipeline, compiled: Pipeline, threshold: float = MATCH_THRESHOLD
) -> list[NodeMatch]:
    """Greedy one-to-one pairing, highest similarity first.

    Ties break on (expected id, compiled id) so the result does not
    depend on node order in either file.
    """
    candidates: list[tuple[float, str, str, Node, Node]] = []
    for e in expected.nodes:
        for c in compiled.nodes:
            s = node_similarity(e, c)
            if s >= threshold:
                candidates.append((s, e.id, c.id, e, c))
    candidates.sort(key=lambda t: (-t[0], t[1], t[2]))
    used_e: set[str] = set()
    used_c: set[str] = set()
    matches: list[NodeMatch] = []
    for s, eid, cid, e, c in candidates:
        if eid in used_e or cid in used_c:
            continue
        used_e.add(eid)
        used_c.add(cid)
        matches.append(NodeMatch(eid, cid, round(s, 4), e.kind.value, c.kind.value))
    matches.sort(key=lambda m: m.expected_id)
    return matches


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def _f1(p: float | None, r: float | None) -> float | None:
    if p is None or r is None:
        return None
    return round(2 * p * r / (p + r), 4) if p + r else 0.0


def precision_recall(
    expected_ids: set[str], compiled_ids: set[str], to_expected: dict[str, str]
) -> dict[str, Any]:
    """P/R of a node flag (fan_out, mandatory) in the expected id space.

    ``compiled_ids`` are compiled node ids carrying the flag; each maps
    through ``to_expected`` when it was matched. An unmatched flagged
    compiled node is a false positive: it claims a flag on a step the
    hand-written pipeline does not have.
    """
    mapped = {to_expected.get(cid, f"<extra:{cid}>") for cid in compiled_ids}
    tp = len(mapped & expected_ids)
    precision = _rate(tp, len(compiled_ids))
    recall = _rate(tp, len(expected_ids))
    return {
        "expected": len(expected_ids),
        "compiled": len(compiled_ids),
        "true_positive": tp,
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "missed": sorted(expected_ids - mapped),
        "spurious": sorted(m.removeprefix("<extra:").rstrip(">") for m in mapped - expected_ids),
    }


def dependency_edges(pipeline: Pipeline) -> set[tuple[str, str]]:
    """``(from, to)`` pairs: declared edges plus node-output ``inputs`` refs.

    Both, because a compile may express a dependency either way and the
    adapters honor both (edges order the waves, inputs carry the data).
    Self-references are dropped.
    """
    deps = {(e.from_, e.to) for e in pipeline.edges}
    for node in pipeline.nodes:
        for ref in (node.inputs or {}).values():
            try:
                parsed = parse_input_ref(ref)
            except ValueError:
                continue
            if parsed.node_id is not None:
                deps.add((parsed.node_id, node.id))
    return {(a, b) for a, b in deps if a != b}


def _wave_index(pipeline: Pipeline) -> dict[str, int]:
    try:
        waves = pipeline.execution_waves()
    except ValueError:
        return {}
    return {n.id: i for i, wave in enumerate(waves) for n in wave}


def _order(a: int, b: int) -> int:
    return (a > b) - (a < b)


def _payload_keys(node: Node) -> set[str]:
    """A gate's resume payload field names: ``output`` keys or schema properties."""
    keys: set[str] = set()
    if isinstance(node.output, dict):
        keys |= set(node.output)
    if node.output_schema:
        keys |= set((node.output_schema.get("properties") or {}).keys())
    return keys


def input_fields(pipeline: Pipeline) -> set[str]:
    """Every pipeline input field name the IR declares, any way it declares it."""
    schema = pipeline.input.input_schema or {}
    return (
        set(pipeline.input.required)
        | set(pipeline.input.optional)
        | set((schema.get("properties") or {}).keys())
    )


def score_pipeline(expected: Pipeline, compiled: Pipeline) -> dict[str, Any]:
    """Every structural metric of ``compiled`` against ``expected``.

    Returns plain JSON-able data. Keys are stable: the benchmark's
    ``--compare`` diffs them across runs.
    """
    matches = match_nodes(expected, compiled)
    to_expected = {m.compiled_id: m.expected_id for m in matches}
    matched_e = {m.expected_id for m in matches}
    matched_c = {m.compiled_id for m in matches}

    confusion: Counter[str] = Counter(f"{m.expected_kind}->{m.compiled_kind}" for m in matches)
    for n in expected.nodes:
        if n.id not in matched_e:
            confusion[f"{n.kind.value}->(missing)"] += 1
    for n in compiled.nodes:
        if n.id not in matched_c:
            confusion[f"(extra)->{n.kind.value}"] += 1

    kind_equal = sum(1 for m in matches if m.kind_equal)
    structure = {
        "expected_nodes": len(expected.nodes),
        "compiled_nodes": len(compiled.nodes),
        "matched": len(matches),
        "matched_by_id": sum(1 for m in matches if m.expected_id == m.compiled_id),
        "node_recall": _rate(len(matches), len(expected.nodes)),
        "node_precision": _rate(len(matches), len(compiled.nodes)),
        # Kind accuracy is over EXPECTED nodes: a missing node is a miss,
        # not a free pass.
        "kind_accuracy": _rate(kind_equal, len(expected.nodes)),
        "kind_accuracy_matched": _rate(kind_equal, len(matches)),
        "kind_counts_expected": dict(sorted(Counter(n.kind.value for n in expected.nodes).items())),
        "kind_counts_compiled": dict(sorted(Counter(n.kind.value for n in compiled.nodes).items())),
        "confusion": dict(sorted(confusion.items())),
        "matches": [
            {
                "expected": m.expected_id,
                "compiled": m.compiled_id,
                "similarity": m.similarity,
                "expected_kind": m.expected_kind,
                "compiled_kind": m.compiled_kind,
            }
            for m in matches
        ],
        "missing": sorted(n.id for n in expected.nodes if n.id not in matched_e),
        "extra": sorted(n.id for n in compiled.nodes if n.id not in matched_c),
    }

    # ── Edges, in the expected id space, between matched nodes only ──
    # An edge touching an unmatched node is already counted as a missing
    # or extra node; counting it again here would double-charge it.
    exp_edges = {(a, b) for a, b in dependency_edges(expected) if a in matched_e and b in matched_e}
    cmp_edges = {
        (to_expected[a], to_expected[b])
        for a, b in dependency_edges(compiled)
        if a in to_expected and b in to_expected and to_expected[a] != to_expected[b]
    }
    tp_edges = len(exp_edges & cmp_edges)
    edge_p = _rate(tp_edges, len(cmp_edges))
    edge_r = _rate(tp_edges, len(exp_edges))

    # ── Waves: pairwise ordering agreement over matched top-level nodes ──
    # Absolute wave indexes shift by one for every extra upstream node; the
    # relative order of two steps (before / together / after) does not.
    e_wave = _wave_index(expected)
    c_wave = _wave_index(compiled)
    pairs = [m for m in matches if m.expected_id in e_wave and m.compiled_id in c_wave]
    agree = total = 0
    for x, y in combinations(pairs, 2):
        total += 1
        e_ord = _order(e_wave[x.expected_id], e_wave[y.expected_id])
        c_ord = _order(c_wave[x.compiled_id], c_wave[y.compiled_id])
        agree += e_ord == c_ord
    graph = {
        "expected_edges": len(exp_edges),
        "compiled_edges": len(cmp_edges),
        "edge_precision": edge_p,
        "edge_recall": edge_r,
        "edge_f1": _f1(edge_p, edge_r),
        "expected_waves": len(set(e_wave.values())),
        "compiled_waves": len(set(c_wave.values())),
        "wave_order_agreement": _rate(agree, total),
    }

    fan_out = precision_recall(
        {n.id for n in expected.nodes if n.fan_out},
        {n.id for n in compiled.nodes if n.fan_out},
        to_expected,
    )
    mandatory = precision_recall(
        {n.id for n in expected.nodes if n.mandatory},
        {n.id for n in compiled.nodes if n.mandatory},
        to_expected,
    )

    # ── Gates ──
    by_id_c = {n.id: n for n in compiled.nodes}
    exp_gates = [n for n in expected.nodes if n.kind is NodeKind.HITL_GATE]
    cmp_gates = [n for n in compiled.nodes if n.kind is NodeKind.HITL_GATE]
    exp_to_cmp = {m.expected_id: m.compiled_id for m in matches}
    gate_rows: list[dict[str, Any]] = []
    for g in exp_gates:
        cid = exp_to_cmp.get(g.id)
        c = by_id_c.get(cid) if cid else None
        is_gate = c is not None and c.kind is NodeKind.HITL_GATE
        e_keys = _payload_keys(g)
        c_keys = _payload_keys(c) if c is not None and is_gate else set()
        gate_rows.append(
            {
                "expected": g.id,
                "compiled": cid if is_gate else None,
                "signal_expected": g.signal,
                "signal_compiled": c.signal if c is not None and is_gate else None,
                "signal_match": bool(is_gate and c is not None and c.signal == g.signal),
                "payload_expected": sorted(e_keys),
                "payload_compiled": sorted(c_keys),
                "payload_jaccard": round(_jaccard(e_keys, c_keys), 4) if is_gate else 0.0,
            }
        )
    gates = {
        "expected": len(exp_gates),
        "compiled": len(cmp_gates),
        "found": sum(1 for r in gate_rows if r["compiled"] is not None),
        "signal_matches": sum(1 for r in gate_rows if r["signal_match"]),
        "payload_jaccard_mean": (
            round(sum(r["payload_jaccard"] for r in gate_rows) / len(gate_rows), 4)
            if gate_rows
            else None
        ),
        "gates": gate_rows,
    }

    e_fields = input_fields(expected)
    c_fields = input_fields(compiled)
    inputs = {
        "expected": sorted(e_fields),
        "compiled": sorted(c_fields),
        "precision": _rate(len(e_fields & c_fields), len(c_fields)),
        "recall": _rate(len(e_fields & c_fields), len(e_fields)),
        "jaccard": round(_jaccard(e_fields, c_fields), 4) if e_fields or c_fields else None,
    }

    return {
        "structure": structure,
        "graph": graph,
        "fan_out": fan_out,
        "mandatory": mandatory,
        "gates": gates,
        "inputs": inputs,
    }
