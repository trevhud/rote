"""What a compile's token usage is known to be, stage by stage.

A usage number is only as complete as the fields read. A codex compile
whose driver reported nothing became four zeros and ``usage_complete:
true`` in the progress summary (issue #295): a run that certainly spent
tokens read as free. So every stage that made model calls says whether it
was measured, and nothing is inferred from an absent field:

* ``measured``: every call's input and output counts were reported.
* ``partial``: some were; the known amounts are kept and the stage says
  what is missing.
* ``unavailable``: none were. Its tokens are ``None``, never zero.

The stages are the System-1 calls recorded in decisions.json (the
structure pass, the tie-break, the decision model) and the agent: the
driver's run, which covers implementation and every repair round it ran.
An explicit zero from a driver is a measured zero.

The whole compile is ``complete`` when every stage is measured,
``unavailable`` when none reported anything, and ``partial`` otherwise.
One function builds it from what is on disk and in the driver's metadata,
so compile-metrics.json, ``--json``, the human summary and the final
progress.jsonl line cannot disagree.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rote.compiler.system1.skeleton import BRIEF_DIRNAME, DECISIONS_FILENAME

#: The four buckets every report uses, and the metadata key for each.
BUCKETS = {
    "input": "input_tokens",
    "output": "output_tokens",
    "cache_write": "cache_write_tokens",
    "cache_read": "cache_read_tokens",
}

MEASURED = "measured"
PARTIAL = "partial"
UNAVAILABLE = "unavailable"

#: The decisions.json calls that record usage rows, in the order they run.
_ROW_STAGES = ("structure", "tiebreak")


def _count(value: object) -> int | None:
    """A token count, or None for anything that is not one."""
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _row_tokens(row: dict[str, Any]) -> dict[str, int | None] | None:
    """One call's buckets, or None when its input or output went unreported.

    A cache bucket a lane does not report (the OpenAI lane has no cache
    writes) is zero once input and output are known: that lane cannot have
    spent it."""
    tokens = {bucket: _count(row.get(key)) for bucket, key in BUCKETS.items()}
    if tokens["input"] is None or tokens["output"] is None:
        return None
    return {b: v or 0 for b, v in tokens.items()}


@dataclass(frozen=True)
class StageUsage:
    """One stage's usage: its status, what was measured, and what was not."""

    stage: str
    status: str
    tokens: dict[str, int | None] | None
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"stage": self.stage, "status": self.status, "tokens": self.tokens}
        if self.reason is not None:
            out["reason"] = self.reason
        return out


@dataclass(frozen=True)
class CompileUsage:
    """Every stage of one compile, and what they add up to."""

    stages: list[StageUsage] = field(default_factory=list)

    @property
    def status(self) -> str:
        statuses = {s.status for s in self.stages}
        if statuses <= {MEASURED}:
            return "complete"
        if statuses == {UNAVAILABLE}:
            return UNAVAILABLE
        return PARTIAL

    @property
    def complete(self) -> bool:
        return self.status == "complete"

    @property
    def tokens(self) -> dict[str, int | None] | None:
        """Known amounts summed per bucket; None when no stage reported any.

        A bucket is None when a stage that reported something could not say
        it (claude's output before its final result): a sum that skipped it
        would read as a small number."""
        reported = [s.tokens for s in self.stages if s.tokens is not None]
        if not reported and self.stages:
            return None
        total: dict[str, int | None] = {}
        for bucket in BUCKETS:
            values = [t.get(bucket) for t in reported]
            known = [v for v in values if v is not None]
            total[bucket] = sum(known) if len(known) == len(values) else None
        return total

    @property
    def missing(self) -> list[str]:
        """``stage: why`` for each stage that is not fully measured."""
        return [f"{s.stage}: {s.reason}" for s in self.stages if s.status != MEASURED]

    @property
    def agent_tokens(self) -> dict[str, int | None] | None:
        """The agent stage's tokens when it was measured, else None.

        What a single-model price estimate may be applied to: System-1's
        calls are priced on their own, so pricing the total would count
        them twice."""
        for s in self.stages:
            if s.stage == "agent" and s.status == MEASURED:
                return s.tokens
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "complete": self.complete,
            "tokens": self.tokens,
            "stages": [s.as_dict() for s in self.stages],
            "missing": self.missing,
        }

    def summary_line(self) -> str:
        """One human line, e.g. ``complete: in=1200 (200 cached) out=50``."""
        tokens = self.tokens
        if tokens is None:
            return f"{self.status} ({'; '.join(self.missing)})"
        cached = (tokens.get("cache_write") or 0) + (tokens.get("cache_read") or 0)
        total_in = None if tokens.get("input") is None else (tokens["input"] or 0) + cached
        line = (
            f"{self.status}: tokens in={'unknown' if total_in is None else total_in}"
            f"{f' ({cached} cached)' if cached else ''} "
            f"out={'unknown' if tokens.get('output') is None else tokens['output']}"
        )
        if self.missing:
            line += f"; not counted: {'; '.join(self.missing)}"
        return line


def agent_stage(driver_name: str, metadata: dict[str, Any]) -> StageUsage:
    """The driver's run, from its metadata.

    ``usage_complete: false`` (a claude run recovered without its final
    result, a parallel compile with cancelled calls) keeps the known
    amounts and is partial."""
    tokens = {bucket: _count(metadata.get(key)) for bucket, key in BUCKETS.items()}
    if all(v is None for v in tokens.values()):
        return StageUsage(
            "agent", UNAVAILABLE, None, f"the {driver_name} driver reported no token usage"
        )
    unknown = [b for b, v in tokens.items() if v is None]
    if metadata.get("usage_complete") is False or "input" in unknown or "output" in unknown:
        # Not the driver's usage_note: compile-metrics.json never copies
        # driver prose, and this reason must read the same there.
        reason = f"the {driver_name} driver's usage is incomplete" + (
            f" (no {', '.join(unknown)} count)" if unknown else ""
        )
        return StageUsage("agent", PARTIAL, tokens, reason)
    # Input and output are known: a cache bucket the lane does not report is zero.
    return StageUsage("agent", MEASURED, {b: v or 0 for b, v in tokens.items()})


def _rows_stage(stage: str, rows: list[dict[str, Any]]) -> StageUsage:
    """A System-1 call's stage from its recorded usage rows."""
    measured = [t for t in (_row_tokens(r) for r in rows) if t is not None]
    if not measured:
        why = f"none of its {len(rows)} call(s) reported tokens"
        return StageUsage(stage, UNAVAILABLE, None, why)
    tokens: dict[str, int | None] = {b: sum(t[b] or 0 for t in measured) for b in BUCKETS}
    if len(measured) < len(rows):
        unreported = len(rows) - len(measured)
        return StageUsage(
            stage, PARTIAL, tokens, f"{unreported} of its {len(rows)} call(s) reported no tokens"
        )
    return StageUsage(stage, MEASURED, tokens)


def system1_stages(output_dir: Path) -> list[StageUsage]:
    """The System-1 calls decisions.json records, one stage per caller.

    A call with no rows made no model calls (a ``rules`` pass) and is not a
    stage. The decision model records its cost, not its tokens, so its
    stage is unavailable."""
    path = output_dir / BRIEF_DIRNAME / DECISIONS_FILENAME
    if not path.is_file():
        return []
    doc = json.loads(path.read_text(encoding="utf-8"))
    stages: list[StageUsage] = []
    decider = doc.get("decider")
    if isinstance(decider, dict):
        stages.append(
            StageUsage(
                "decider",
                UNAVAILABLE,
                None,
                f"the {decider.get('name') or 'decision'} model records its cost, not its tokens",
            )
        )
    for name in _ROW_STAGES:
        call = doc.get(name)
        rows = call.get("usage") if isinstance(call, dict) else None
        if isinstance(rows, list) and rows:
            stages.append(_rows_stage(name, [r for r in rows if isinstance(r, dict)]))
    return stages


def compile_usage(driver_name: str, metadata: dict[str, Any], output_dir: Path) -> CompileUsage:
    """Every stage of the compile whose deliverable is in ``output_dir``."""
    return CompileUsage([*system1_stages(output_dir), agent_stage(driver_name, metadata)])
