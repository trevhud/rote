"""Structural scoring of a compiled pipeline against a hand-written one.

Pure functions over two :class:`~rote.ir.Pipeline` values: no files, no
subprocesses, no inference. Everything the benchmark reports about
*quality of the IR* comes from here, which is why it is the part with
unit tests (``tests/test_bench_score.py``).

Matching
--------
Every metric below compares nodes, so the first step pairs each
hand-written (expected) node with at most one compiled node. Names are
the weakest evidence there is: a compile that calls the expected
``render_welcome_post`` ``format_announcement`` did the same step, and a
name-keyed matcher charges it a kind error, a mandatory miss and a
spurious mandatory for it. So matching is name-independent:

1. :func:`node_similarity` scores every (expected, compiled) pair from
   what the node *is* in its pipeline (:data:`SIGNAL_WEIGHTS`): its
   topological rank, its in- and out-degree, its neighbors' kinds, the
   skill section it cites, its input and output names, its MCP server
   and tool, and only weakly its id and description. A node's own kind,
   flags and gate fields are deliberately left out: they are what the
   metrics grade, so matching on them would grade the matcher's choice.
2. An optimal one-to-one assignment (:func:`_max_weight_assignment`, the
   Hungarian algorithm) maximizes the total similarity, so competing
   candidates are settled for the pipeline as a whole, not first-come.
3. A pair under :data:`MATCH_THRESHOLD` is dropped. An unmatched
   expected node is *missing*; an unmatched compiled node is *extra*.
   A compile that splits one step in two scores one match and one
   extra, and the matched half is the one that sits where the expected
   step sat.

Gate signal and payload names are reported twice: exact, and on
normalized tokens (:func:`_name_tokens`), so ``cs_lead_approved`` and
``cs_lead_approval`` count as the same signal there.

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

#: Minimum :func:`node_similarity` for an assigned pair to count as the
#: same step. This value and :data:`SIGNAL_WEIGHTS` were chosen by a
#: sweep scored against hand labels of 200 contested matches across 48
#: real compiles: the committed snapshots plus every bench run on disk
#: (described in the PR that introduced this matcher). Below 0.35,
#: leftovers pair with unrelated nodes. Above it, real counterparts in
#: restructured compiles start dropping out.
MATCH_THRESHOLD = 0.35

#: How much each signal counts. A signal joins the mean only when it
#: applies to the pair (both nodes cite a section, either has inputs,
#: both have MCP facts, ...), so a compile that omits provenance is not
#: penalized for it. The id is weight 1.5 of up to 17.5: enough to
#: break a tie between two structurally identical nodes, never enough to
#: carry a match on its own. Description is content, not a name, which
#: is why it counts for more.
SIGNAL_WEIGHTS: dict[str, float] = {
    "position": 2.0,
    "degree": 1.0,
    "neighbor_kinds": 1.5,
    "matched_neighbors": 2.0,
    "source": 2.0,
    "inputs": 1.5,
    "outputs": 1.0,
    "mcp": 2.0,
    "description": 2.0,
    "name": 1.5,
}

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


#: Suffixes :func:`_name_tokens` strips, longest first, so ``approved``,
#: ``approval`` and ``approver`` all reduce to ``approv``. Crude on
#: purpose: it only has to make two spellings of one name agree.
_SUFFIXES = ("ations", "ation", "ings", "ing", "als", "al", "ers", "er", "ed", "es", "s")


def _name_tokens(name: str) -> set[str]:
    """``name``'s tokens with inflection stripped, for normalized matching."""
    out: set[str] = set()
    for tok in _tokens(name):
        for suffix in _SUFFIXES:
            if tok.endswith(suffix) and len(tok) - len(suffix) >= 3:
                tok = tok[: -len(suffix)]
                break
        out.add(tok)
    return out


def _multiset_jaccard(a: Counter[str], b: Counter[str]) -> float:
    if not a and not b:
        return 1.0
    return sum((a & b).values()) / sum((a | b).values())


def _ref_field_tokens(ref: str) -> set[str]:
    """Tokens of an ``inputs:`` reference that survive a node rename.

    ``pipeline.input.<field>`` names a field both pipelines share, so it
    keeps a marker; ``<node>.output.<field>`` keeps only the field,
    because the upstream id is exactly what may have been renamed.
    """
    try:
        parsed = parse_input_ref(ref)
    except ValueError:
        return set()
    if parsed.node_id is None:
        return {f"input:{parsed.field or '*'}"}
    return _name_tokens(parsed.field) if parsed.field else set()


def _input_tokens(node: Node) -> set[str]:
    toks = {t for k in (node.inputs or {}) for t in _name_tokens(k)}
    toks |= {t for k in node.input for t in _name_tokens(k)}
    return toks | {t for r in (node.inputs or {}).values() for t in _ref_field_tokens(r)}


def _output_tokens(node: Node) -> set[str]:
    keys: set[str] = set()
    if isinstance(node.output, dict):
        keys |= set(node.output)
    elif node.output:
        keys.add(node.output)
    if node.output_schema:
        keys |= set((node.output_schema.get("properties") or {}).keys())
    return {t for k in keys for t in _name_tokens(k)}


def _mcp_facts(node: Node) -> set[str]:
    """The MCP servers and tools a node reaches, from either declaration."""
    facts: set[str] = set()
    if node.mcp:
        facts |= {f"server:{node.mcp.server}", f"tool:{node.mcp.tool}"}
    facts |= {f"tool:{t}" for t in node.tools or ()}
    facts |= {f"server:{s}" for s in (node.tool_servers or {}).values()}
    return facts


class _Context:
    """Per-pipeline graph facts the similarity reads, computed once."""

    def __init__(self, pipeline: Pipeline) -> None:
        nodes = {n.id: n for n in pipeline.nodes}
        self.preds: dict[str, set[str]] = {nid: set() for nid in nodes}
        self.succs: dict[str, set[str]] = {nid: set() for nid in nodes}
        for a, b in dependency_edges(pipeline):
            if a in nodes and b in nodes:
                self.preds[b].add(a)
                self.succs[a].add(b)
        self.kind = {nid: n.kind.value for nid, n in nodes.items()}
        self.rank = self._ranks(pipeline)

    def _ranks(self, pipeline: Pipeline) -> dict[str, float]:
        """Longest-path depth over dependencies, scaled to ``[0, 1]``.

        A loop-body sub-node with no dependencies of its own takes its
        loop's depth: it runs inside the loop, not at the pipeline start
        its edgeless declaration would place it.
        """
        depth: dict[str, int] = {}
        visiting: set[str] = set()

        def visit(nid: str) -> int:
            if nid in depth:
                return depth[nid]
            if nid in visiting:  # a cycle; the IR rejects these, never recurse forever
                return 0
            visiting.add(nid)
            depth[nid] = max((visit(p) + 1 for p in sorted(self.preds[nid])), default=0)
            visiting.discard(nid)
            return depth[nid]

        for nid in sorted(self.preds):
            visit(nid)
        for n in pipeline.nodes:
            for sub in n.loop_body or ():
                if sub in depth and not self.preds[sub]:
                    depth[sub] = depth[n.id]
        top = max(depth.values(), default=0)
        return {nid: (d / top if top else 0.0) for nid, d in depth.items()}

    def neighbor_kinds(self, nid: str, side: str) -> Counter[str]:
        ids = self.preds[nid] if side == "in" else self.succs[nid]
        return Counter(self.kind[i] for i in ids)


def _matched_neighbors(
    e: str, ec: _Context, c: str, cc: _Context, assignment: dict[str, str]
) -> float:
    """How many of ``e``'s neighbors are, under ``assignment``, ``c``'s.

    Directed: an upstream neighbor only agrees with an upstream one. The
    Jaccard is over neighbor slots, so an unmatched neighbor on either
    side counts against the pair. Two isolated nodes agree fully.
    """
    mine = {("in", p) for p in ec.preds[e]} | {("out", s) for s in ec.succs[e]}
    theirs = {("in", p) for p in cc.preds[c]} | {("out", s) for s in cc.succs[c]}
    if not mine and not theirs:
        return 1.0
    mapped = {(side, assignment[n]) for side, n in mine if n in assignment}
    agree = len(mapped & theirs)
    return agree / (len(mine) + len(theirs) - agree)


def similarity_signals(
    expected: Node,
    ec: _Context,
    compiled: Node,
    cc: _Context,
    assignment: dict[str, str] | None = None,
) -> dict[str, float]:
    """Every signal that applies to one pair, each in ``[0, 1]``.

    ``assignment`` (expected id -> compiled id, from a first matching
    pass) adds ``matched_neighbors``, the strongest structural signal:
    two nodes are the same step when their neighbors are the same steps.
    """
    e, c = expected, compiled
    signals: dict[str, float] = {
        "position": 1.0 - abs(ec.rank[e.id] - cc.rank[c.id]),
        "degree": (
            1 / (1 + abs(len(ec.preds[e.id]) - len(cc.preds[c.id])))
            + 1 / (1 + abs(len(ec.succs[e.id]) - len(cc.succs[c.id])))
        )
        / 2,
        "neighbor_kinds": (
            _multiset_jaccard(ec.neighbor_kinds(e.id, "in"), cc.neighbor_kinds(c.id, "in"))
            + _multiset_jaccard(ec.neighbor_kinds(e.id, "out"), cc.neighbor_kinds(c.id, "out"))
        )
        / 2,
    }
    if assignment is not None:
        signals["matched_neighbors"] = _matched_neighbors(e.id, ec, c.id, cc, assignment)
    if e.source and c.source:
        signals["source"] = (
            1.0
            if e.source.section == c.source.section
            else _jaccard(_tokens(e.source.section), _tokens(c.source.section))
        )
    ein, cin = _input_tokens(e), _input_tokens(c)
    if ein or cin:
        signals["inputs"] = _jaccard(ein, cin)
    eout, cout = _output_tokens(e), _output_tokens(c)
    if eout or cout:
        signals["outputs"] = _jaccard(eout, cout)
    # Both sides, or the signal does not apply: a compile that forgot a
    # binding made an MCP error, which the binding is not here to grade,
    # and the step is still the same step.
    emcp, cmcp = _mcp_facts(e), _mcp_facts(c)
    if emcp and cmcp:
        signals["mcp"] = _jaccard(emcp, cmcp)
    signals["description"] = _jaccard(_name_tokens(e.description), _name_tokens(c.description))
    signals["name"] = 1.0 if e.id == c.id else _jaccard(_name_tokens(e.id), _name_tokens(c.id))
    return signals


def _weighted(signals: dict[str, float]) -> float:
    total = sum(SIGNAL_WEIGHTS[k] for k in signals)
    return sum(SIGNAL_WEIGHTS[k] * v for k, v in signals.items()) / total


def node_similarity(
    expected: Node, compiled: Node, expected_pipeline: Pipeline, compiled_pipeline: Pipeline
) -> float:
    """How likely two nodes are the same step, in ``[0, 1]``.

    The :data:`SIGNAL_WEIGHTS`-weighted mean of the signals that apply.
    Each node is read in the context of its own pipeline.
    """
    return _weighted(
        similarity_signals(
            expected, _Context(expected_pipeline), compiled, _Context(compiled_pipeline)
        )
    )


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


def _max_weight_assignment(weights: list[list[float]]) -> list[tuple[int, int]]:
    """Rows paired with columns so the summed weight is maximal (Hungarian).

    ``weights`` is ``n x m``, entries ``>= 0``. Returns every assigned
    ``(row, col)``, zero-weight pairs included (callers drop them). The
    potentials form of the algorithm, O(size^3), on a square cost matrix
    of ``-weight`` padded with zeros. Equal-cost ties go to the lowest
    indexes, which the caller makes deterministic by sorting its inputs.
    """
    n = len(weights)
    m = len(weights[0]) if n else 0
    size = max(n, m)
    if size == 0:
        return []
    cost = [[-weights[i][j] if i < n and j < m else 0.0 for j in range(size)] for i in range(size)]
    inf = float("inf")
    u = [0.0] * (size + 1)  # row potentials
    v = [0.0] * (size + 1)  # column potentials
    owner = [0] * (size + 1)  # owner[j]: 1-based row assigned to column j (0 = none)
    back = [0] * (size + 1)  # back[j]: previous column on the augmenting path
    for row in range(1, size + 1):
        owner[0] = row
        col = 0
        best = [inf] * (size + 1)
        done = [False] * (size + 1)
        while True:
            done[col] = True
            r = owner[col]
            delta, nxt = inf, 0
            for j in range(1, size + 1):
                if done[j]:
                    continue
                reduced = cost[r - 1][j - 1] - u[r] - v[j]
                if reduced < best[j]:
                    best[j], back[j] = reduced, col
                if best[j] < delta:
                    delta, nxt = best[j], j
            for j in range(size + 1):
                if done[j]:
                    u[owner[j]] += delta
                    v[j] -= delta
                else:
                    best[j] -= delta
            col = nxt
            if owner[col] == 0:
                break
        while col:
            prev = back[col]
            owner[col] = owner[prev]
            col = prev
    return [(owner[j] - 1, j - 1) for j in range(1, size + 1) if 0 < owner[j] <= n and j <= m]


def match_nodes(
    expected: Pipeline, compiled: Pipeline, threshold: float = MATCH_THRESHOLD
) -> list[NodeMatch]:
    """The optimal one-to-one pairing over :func:`node_similarity`.

    A pair under ``threshold`` weighs zero in the assignment, so it never
    gives up a real match for one it would then drop. Nodes are sorted
    by id first, so the result does not depend on node order in either
    file.
    """
    ec, cc = _Context(expected), _Context(compiled)
    e_nodes = sorted(expected.nodes, key=lambda n: n.id)
    c_nodes = sorted(compiled.nodes, key=lambda n: n.id)

    def assign(
        assignment: dict[str, str] | None,
    ) -> tuple[list[list[float]], list[tuple[int, int]]]:
        sims = [
            [_weighted(similarity_signals(e, ec, c, cc, assignment)) for c in c_nodes]
            for e in e_nodes
        ]
        weights = [[s if s >= threshold else 0.0 for s in row] for row in sims]
        pairs = [(i, j) for i, j in _max_weight_assignment(weights) if weights[i][j] > 0]
        return sims, pairs

    # Pass 1 matches on each node alone; pass 2 adds whether the neighbors
    # matched too, which settles nodes that look alike in isolation (two
    # fetches at the pipeline start) by what they feed.
    _, first = assign(None)
    sims, pairs = assign({e_nodes[i].id: c_nodes[j].id for i, j in first})
    matches = [
        NodeMatch(
            e_nodes[i].id,
            c_nodes[j].id,
            round(sims[i][j], 4),
            e_nodes[i].kind.value,
            c_nodes[j].kind.value,
        )
        for i, j in pairs
    ]
    matches.sort(key=lambda m: m.expected_id)
    return matches


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


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
    e_kinds = Counter(n.kind.value for n in expected.nodes)
    c_kinds = Counter(n.kind.value for n in compiled.nodes)
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
        # Matching-free: how much of the kind mix the two pipelines share
        # (sum of per-kind minimums over the larger node count). Lexical
        # matching misfires when a compile splits or merges steps; this
        # number cannot, so read kind_accuracy beside it.
        "kind_mix_overlap": _rate(
            sum((e_kinds & c_kinds).values()), max(len(expected.nodes), len(compiled.nodes))
        ),
        "kind_counts_expected": dict(sorted(e_kinds.items())),
        "kind_counts_compiled": dict(sorted(c_kinds.items())),
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
        c_signal = c.signal if c is not None and is_gate else None
        e_norm = {t for k in e_keys for t in _name_tokens(k)}
        c_norm = {t for k in c_keys for t in _name_tokens(k)}
        gate_rows.append(
            {
                "expected": g.id,
                "compiled": cid if is_gate else None,
                "signal_expected": g.signal,
                "signal_compiled": c_signal,
                "signal_match": bool(is_gate and c_signal == g.signal),
                # Same name up to inflection and separators:
                # cs_lead_approved == cs_lead_approval.
                "signal_match_normalized": bool(
                    is_gate and c_signal and _name_tokens(c_signal) == _name_tokens(g.signal or "")
                ),
                "payload_expected": sorted(e_keys),
                "payload_compiled": sorted(c_keys),
                "payload_jaccard": round(_jaccard(e_keys, c_keys), 4) if is_gate else 0.0,
                # Over the pooled normalized tokens of every key, so
                # approved_plan / approved_seats vs plan / seats / approver
                # reads as the same payload.
                "payload_jaccard_normalized": (
                    round(_jaccard(e_norm, c_norm), 4) if is_gate else 0.0
                ),
            }
        )
    gates = {
        "expected": len(exp_gates),
        "compiled": len(cmp_gates),
        "found": sum(1 for r in gate_rows if r["compiled"] is not None),
        "signal_matches": sum(1 for r in gate_rows if r["signal_match"]),
        "signal_matches_normalized": sum(1 for r in gate_rows if r["signal_match_normalized"]),
        "payload_jaccard_mean": _mean([r["payload_jaccard"] for r in gate_rows]),
        "payload_jaccard_normalized_mean": _mean(
            [r["payload_jaccard_normalized"] for r in gate_rows]
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
