"""Flatten, aggregate, render and diff benchmark results.

A job's full result (``collect.collect_job``) is nested and detailed;
this module reduces it to one flat row of headline numbers
(:func:`headline`), which is what the tables print, what the aggregate
averages, and what ``--compare`` diffs. Pure functions over JSON data.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Sequence
from typing import Any

Number = int | float

#: (key, column header, format). Order is the table's column order.
COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("nodes", "nodes", "{}"),
    ("node_recall", "nodeR", "{:.2f}"),
    ("node_precision", "nodeP", "{:.2f}"),
    ("kind_accuracy", "kind", "{:.2f}"),
    ("edge_f1", "edgeF1", "{:.2f}"),
    ("wave_order", "waves", "{:.2f}"),
    ("fan_out_precision", "fanP", "{:.2f}"),
    ("fan_out_recall", "fanR", "{:.2f}"),
    ("mandatory_precision", "mandP", "{:.2f}"),
    ("mandatory_recall", "mandR", "{:.2f}"),
    ("gates", "gates", "{}"),
    ("gate_signal_matches", "sig", "{}"),
    ("gate_payload_jaccard", "gateJ", "{:.2f}"),
    ("input_recall", "inR", "{:.2f}"),
    ("input_precision", "inP", "{:.2f}"),
    ("implemented", "impl", "{}"),
    ("tests", "tests", "{}"),
    ("citation_errors", "citeE", "{}"),
    ("assumptions", "assum", "{}"),
    ("open_questions", "OQ", "{}"),
    ("verified", "ok", "{}"),
    ("exit_code", "exit", "{}"),
    ("repair_rounds", "rep", "{}"),
    ("turns", "turns", "{}"),
    ("cost_usd", "cost$", "{:.2f}"),
    ("input_tokens_total", "inTok", "{}"),
    ("output_tokens", "outTok", "{}"),
    ("wall_seconds", "wall_s", "{:.0f}"),
    ("emit_dbos", "dbos", "{}"),
    ("emit_cloudflare", "cf", "{}"),
    ("tsc", "tsc", "{}"),
)

#: Keys summed, not averaged, in the aggregate.
SUMMED = frozenset({"cost_usd", "wall_seconds"})

#: Headline keys where a larger number is worse. ``--compare`` marks a
#: delta as a regression by this, so the reader need not remember it.
LOWER_IS_BETTER = frozenset(
    {
        "citation_errors",
        "exit_code",
        "repair_rounds",
        "turns",
        "cost_usd",
        "input_tokens_total",
        "output_tokens",
        "wall_seconds",
    }
)


def _get(d: dict[str, Any], *path: str) -> Any:
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return cur


def _ratio(num: Any, den: Any) -> str | None:
    if num is None or den is None:
        return None
    return f"{num}/{den}"


def headline(job: dict[str, Any]) -> dict[str, Any]:
    """One flat row of the numbers a reader compares across runs."""
    run = job.get("run", {})
    cache = (run.get("cache_write_tokens") or 0) + (run.get("cache_read_tokens") or 0)
    in_total = (
        (run.get("input_tokens") or 0) + cache
        if run.get("input_tokens") is not None or cache
        else None
    )
    tests = _get(job, "verification", "tests")
    if tests is None:
        tests_cell = "none" if "verification" in job else None
    else:
        tests_cell = tests["status"]
        if "passed" in tests or "failed" in tests:
            tests_cell += (
                f"({tests.get('passed', 0)}/{tests.get('passed', 0) + tests.get('failed', 0)})"
            )
    emit = job.get("emit") or {}

    def _emit_cell(rt: str) -> str | None:
        r = emit.get(rt)
        return None if r is None else ("ok" if r["ok"] else "FAIL")

    impl = _get(job, "verification", "implementation") or {}
    return {
        "nodes": _ratio(
            _get(job, "structure", "compiled_nodes"), _get(job, "structure", "expected_nodes")
        ),
        "node_recall": _get(job, "structure", "node_recall"),
        "node_precision": _get(job, "structure", "node_precision"),
        "kind_accuracy": _get(job, "structure", "kind_accuracy"),
        "edge_f1": _get(job, "graph", "edge_f1"),
        "wave_order": _get(job, "graph", "wave_order_agreement"),
        "fan_out_precision": _get(job, "fan_out", "precision"),
        "fan_out_recall": _get(job, "fan_out", "recall"),
        "mandatory_precision": _get(job, "mandatory", "precision"),
        "mandatory_recall": _get(job, "mandatory", "recall"),
        "gates": _ratio(_get(job, "gates", "compiled"), _get(job, "gates", "expected")),
        "gate_signal_matches": _get(job, "gates", "signal_matches"),
        "gate_payload_jaccard": _get(job, "gates", "payload_jaccard_mean"),
        "input_recall": _get(job, "inputs", "recall"),
        "input_precision": _get(job, "inputs", "precision"),
        "implemented": _ratio(impl.get("implemented"), impl.get("pure_functions")),
        "implemented_rate": (
            round(impl["implemented"] / impl["pure_functions"], 4)
            if impl.get("pure_functions")
            else None
        ),
        "tests": tests_cell,
        "tests_passed": None if tests is None else int(tests["status"] == "passed"),
        "citation_errors": _get(job, "verification", "provenance", "citation_errors"),
        "assumptions": _get(job, "verification", "provenance", "assumptions"),
        "open_questions": _get(job, "verification", "provenance", "open_questions"),
        "verified": (None if "verification" not in job else int(bool(job["verification"]["ok"]))),
        "exit_code": run.get("exit_code"),
        "repair_rounds": run.get("repair_rounds"),
        "turns": run.get("turns"),
        "cost_usd": run.get("cost_usd"),
        "input_tokens_total": in_total,
        "output_tokens": run.get("output_tokens"),
        "cache_read_tokens": run.get("cache_read_tokens"),
        "cache_write_tokens": run.get("cache_write_tokens"),
        "wall_seconds": run.get("wall_seconds"),
        "emit_dbos": _emit_cell("dbos"),
        "emit_cloudflare": _emit_cell("cloudflare"),
        "emit_ok": (None if not emit else int(all(r["ok"] for r in emit.values()))),
        "tsc": (emit.get("cloudflare") or {}).get("tsc"),
        "tsc_passed": (
            None
            if (emit.get("cloudflare") or {}).get("tsc") in (None, "skipped", "no-npm")
            else int(emit["cloudflare"]["tsc"] == "passed")
        ),
    }


def _numeric(values: Iterable[Any]) -> list[Number]:
    return [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]


def aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Mean of every numeric headline key (sum for cost and wall time).

    ``None`` values are skipped, so a rate with no denominator in one
    skill does not drag the mean. ``n`` per key says how many rows
    contributed.
    """
    out: dict[str, Any] = {"jobs": len(rows)}
    keys = [k for k in (rows[0] if rows else {}) if _numeric(r.get(k) for r in rows)]
    for key in keys:
        vals = _numeric(r.get(key) for r in rows)
        out[key] = round(sum(vals), 4) if key in SUMMED else round(statistics.fmean(vals), 4)
    return out


def variance(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Number]]:
    """Per numeric key: min, max and population stdev across repeats."""
    out: dict[str, dict[str, Number]] = {}
    if len(rows) < 2:
        return out
    for key in rows[0]:
        vals = _numeric(r.get(key) for r in rows)
        if len(vals) >= 2:
            out[key] = {
                "min": min(vals),
                "max": max(vals),
                "stdev": round(statistics.pstdev(vals), 4),
            }
    return out


def _cell(value: Any, fmt: str) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return fmt.format(value)
        except (ValueError, TypeError):
            return str(value)
    return str(value)


def render_table(labelled_rows: Sequence[tuple[str, dict[str, Any]]]) -> str:
    """A fixed-width text table, one row per (label, headline row)."""
    headers = ["job", *(h for _, h, _ in COLUMNS)]
    body = [
        [label, *(_cell(row.get(k), fmt) for k, _, fmt in COLUMNS)] for label, row in labelled_rows
    ]
    widths = [max(len(r[i]) for r in [headers, *body]) for i in range(len(headers))]
    lines = ["  ".join(c.ljust(w) for c, w in zip(headers, widths, strict=True))]
    lines.append("  ".join("-" * w for w in widths))
    lines += ["  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)) for r in body]
    return "\n".join(lines)


def compare(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """Deltas of the aggregate and of each skill's mean, ``new - old``.

    Only keys numeric on both sides. Each delta carries ``better``:
    ``True`` / ``False`` by :data:`LOWER_IS_BETTER`, ``None`` for zero.
    """

    def _delta(a: dict[str, Any], b: dict[str, Any]) -> dict[str, dict[str, Any]]:
        d: dict[str, dict[str, Any]] = {}
        for key, nv in b.items():
            ov = a.get(key)
            if not _numeric([ov]) or not _numeric([nv]) or key == "jobs":
                continue
            diff = round(nv - ov, 4)
            better = None if diff == 0 else (diff < 0) == (key in LOWER_IS_BETTER)
            d[key] = {"old": ov, "new": nv, "delta": diff, "better": better}
        return d

    per_skill = {}
    for skill, agg in (new.get("per_skill") or {}).items():
        if skill in (old.get("per_skill") or {}):
            per_skill[skill] = _delta(old["per_skill"][skill]["mean"], agg["mean"])
    return {
        "aggregate": _delta(old.get("aggregate") or {}, new.get("aggregate") or {}),
        "per_skill": per_skill,
    }


def render_compare(diff: dict[str, Any]) -> str:
    lines: list[str] = []

    def _block(title: str, d: dict[str, dict[str, Any]]) -> None:
        lines.append(title)
        if not d:
            lines.append("  (no common numeric keys)")
        for key, v in d.items():
            mark = {True: "better", False: "WORSE", None: "same"}[v["better"]]
            lines.append(
                f"  {key:<22} {v['old']!s:>12} -> {v['new']!s:<12} {v['delta']:+.4g}  {mark}"
            )

    _block("aggregate (new - old):", diff["aggregate"])
    for skill, d in diff["per_skill"].items():
        _block(f"{skill} (mean, new - old):", d)
    return "\n".join(lines)
