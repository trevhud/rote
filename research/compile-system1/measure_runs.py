"""Break down where a real `rote compile` run spent its wall time and money.

usage: python measure_runs.py <compile --out dir> [...]

Reads only the run's own artifacts (progress.jsonl, compiled/compile-metrics.json,
compiled/**). Segments the timeline at the first Write (end of reading and
planning) and at the first repair warning (end of the first deliverable).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# claude-sonnet-4-6 first-party rates, $/MTok (claude-api skill, cached 2026-10-06).
RATES = {"input": 3.0, "output": 15.0, "cache_write": 3.75, "cache_read": 0.30}


def measure(out: Path) -> dict:
    lines = (out / "progress.jsonl").read_text().splitlines()
    # The CLI appends a summary line with no timestamp; the timeline ignores it.
    events = [e for e in (json.loads(x) for x in lines if x.strip()) if "ts" in e]
    t0 = events[0]["ts"]
    t_end = events[-1]["ts"]

    def is_work_write(e: dict) -> bool:
        return (
            e.get("type") == "tool"
            and e.get("tool_name") in ("Write", "Edit")
            and not str(e.get("path", "")).endswith("progress.ndjson")
        )

    first_write = next((e["ts"] for e in events if is_work_write(e)), t_end)
    first_repair = next(
        (
            e["ts"]
            for e in events
            if e.get("type") == "warning" and "repair" in e.get("message", "")
        ),
        t_end,
    )
    turns = [e for e in events if e.get("type") == "turn"]
    gaps = sorted(
        ((b["ts"] - a["ts"], a["turn"]) for a, b in zip(turns, turns[1:], strict=False)),
        reverse=True,
    )
    tools = [e for e in events if e.get("type") == "tool"]
    reads = sum(1 for e in tools if e.get("tool_name") in ("Read", "Glob", "Grep"))
    writes = sum(1 for e in tools if is_work_write(e))
    m = json.loads((out / "compiled" / "compile-metrics.json").read_text())["metadata"]
    cost_parts = {
        "output": (m.get("output_tokens") or 0) * RATES["output"] / 1e6,
        "cache_read": (m.get("cache_read_tokens") or 0) * RATES["cache_read"] / 1e6,
        "cache_write": (m.get("cache_write_tokens") or 0) * RATES["cache_write"] / 1e6,
    }
    sizes: dict[str, int] = {}
    for p in (out / "compiled").rglob("*"):
        if p.is_file() and p.suffix in (".py", ".yaml", ".md", ".jsonl"):
            top = p.relative_to(out / "compiled").parts[0]
            sizes[top] = sizes.get(top, 0) + len(p.read_text(errors="replace"))
    return {
        "run": out.name,
        "wall_s": round(t_end - t0),
        "read_plan_s": round(first_write - t0),
        "write_s": round(first_repair - first_write),
        "repair_s": round(t_end - first_repair),
        "turns": m.get("num_turns"),
        "repairs": m.get("repair_rounds", 0),
        "cost_usd": round(m.get("cost_usd") or 0, 2),
        "cost_split": {k: round(v, 2) for k, v in cost_parts.items()},
        "output_tokens": m.get("output_tokens"),
        "cache_read_tokens": m.get("cache_read_tokens"),
        "reads": reads,
        "writes": writes,
        "top_gaps_s": [(round(g), t) for g, t in gaps[:3]],
        "artifact_chars": dict(sorted(sizes.items())),
    }


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        print(json.dumps(measure(Path(arg))))
