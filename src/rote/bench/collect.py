"""Read one finished ``rote compile --out`` directory into benchmark metrics.

No inference anywhere in here. Everything is either a file the compile
wrote (``compile-metrics.json``, ``progress.jsonl``, ``compile-report.md``)
or a deterministic check rote already owns, re-run on the final
deliverable so every job is judged the same way whatever exit path its
compile took: :func:`~rote.compiler.verify.verify_deliverable` for
implementations, generated tests and provenance, and the adapters for
the emit check.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from rote.bench.score import score_pipeline
from rote.compiler.provenance import REPORT_FILENAME, open_questions
from rote.ir import Pipeline, load_pipeline

#: The runtimes the emit check covers: the CLI's logged-out default
#: (Python) and the logged-in default (TypeScript, so tsc means something).
EMIT_RUNTIMES = ("dbos", "cloudflare")

#: npm install + tsc on one emitted Cloudflare app. Generous: a cold npm
#: cache downloads wrangler and its workerd binary.
TSC_TIMEOUT_SECONDS = 600


def deliverable_dir(out_dir: Path) -> Path | None:
    """Where the compile left its pipeline.yaml, or ``None`` if nowhere.

    ``compiled/`` on success; ``compiled.rejected/`` when the IR failed
    validation after the repair rounds (the orchestrator keeps a paid
    run's work there rather than replacing a previous good output).
    """
    for name in ("compiled", "compiled.rejected"):
        candidate = out_dir / name
        if (candidate / "pipeline.yaml").is_file():
            return candidate
    return None


def read_compile_metrics(compiled: Path | None) -> dict[str, Any]:
    """``compile-metrics.json``'s allowlisted driver metadata, or ``{}``."""
    if compiled is None:
        return {}
    path = compiled / "compile-metrics.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    meta = data.get("metadata")
    return dict(meta) if isinstance(meta, dict) else {}


def phase_durations(progress_jsonl: Path) -> dict[str, Any]:
    """Seconds spent in each announced phase, from ``progress.jsonl``.

    A phase lasts from its ``phase`` event to the next one, and the last
    phase to the final ``complete`` / ``error`` event (or the last event
    of any kind when neither arrived). Also reports the last ``turn`` and
    the number of repair warnings the orchestrator logged.
    """
    if not progress_jsonl.is_file():
        return {"phases": [], "events": 0}
    events: list[dict[str, Any]] = []
    for line in progress_jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("ts"), (int, float)):
            events.append(obj)
    if not events:
        return {"phases": [], "events": 0}
    start = events[0]["ts"]
    end_events = [e for e in events if e.get("type") in ("complete", "error")]
    end = (end_events[-1] if end_events else events[-1])["ts"]
    marks = [e for e in events if e.get("type") == "phase"]
    phases: list[dict[str, Any]] = []
    for i, mark in enumerate(marks):
        until = marks[i + 1]["ts"] if i + 1 < len(marks) else end
        phases.append(
            {
                "phase": mark.get("phase"),
                "name": mark.get("phase_name"),
                "seconds": round(max(0.0, until - mark["ts"]), 1),
            }
        )
    turns = [e["turn"] for e in events if isinstance(e.get("turn"), int)]
    return {
        "phases": phases,
        "before_first_phase_seconds": round((marks[0]["ts"] if marks else end) - start, 1),
        "agent_seconds": round(end - start, 1),
        "last_turn": max(turns) if turns else None,
        "repair_warnings": sum(
            1 for e in events if e.get("type") == "warning" and "repair" in str(e.get("message"))
        ),
        "events": len(events),
    }


def _test_counts(summary: str) -> dict[str, int]:
    """``{"passed": 12, "failed": 1}`` from a pytest summary line."""
    import re

    return {
        word: int(n)
        for n, word in re.findall(r"(\d+) (passed|failed|error|errors|skipped)", summary)
    }


def verification_metrics(pipeline: Pipeline, compiled: Path, skill_dir: Path) -> dict[str, Any]:
    """Implementation, generated tests and provenance, re-checked offline."""
    from rote.compiler.verify import verify_deliverable

    v = verify_deliverable(compiled, run_tests=True, skill_dir=skill_dir)
    impls = v.implementations
    pure = [i for i in impls if i.kind == "pure_function"]
    statuses = Counter(i.status for i in pure)
    tests = v.tests
    provenance_kinds = Counter(f.kind for f in v.findings if f.node_id == "(provenance)")
    report = compiled / REPORT_FILENAME
    oq = (
        len(open_questions(report.read_text(encoding="utf-8", errors="replace")))
        if report.is_file()
        else None
    )
    return {
        "ok": v.ok,
        "errors": len(v.errors),
        "error_kinds": dict(sorted(Counter(f.kind for f in v.errors).items())),
        "implementation": {
            "pure_functions": len(pure),
            "implemented": statuses.get("implemented", 0),
            "stub": statuses.get("stub", 0),
            "absent": statuses.get("absent", 0),
            "impl_nodes": len(impls),
            "impl_nodes_implemented": sum(1 for i in impls if i.status == "implemented"),
        },
        "tests": (
            None
            if tests is None
            else {"status": tests.status, "summary": tests.summary, **_test_counts(tests.summary)}
        ),
        "provenance": {
            "citation_errors": provenance_kinds.get("citation_not_in_skill", 0)
            + provenance_kinds.get("malformed_citation", 0),
            "uncited_outcomes": provenance_kinds.get("uncited_outcome", 0),
            "unlisted_assumption": provenance_kinds.get("unlisted_assumption", 0),
            "assumptions": len(v.assumptions),
            "open_questions": oq,
        },
    }


def emit_check(pipeline: Pipeline, compiled: Path, work: Path, *, tsc: bool) -> dict[str, Any]:
    """Emit on each of :data:`EMIT_RUNTIMES`; type-check the Cloudflare output.

    Emission is in-process (it is pure template code, and this is the
    same installed rote that ran the compile). ``tsc`` runs the emitted
    app's own toolchain: ``npm install`` then ``npx tsc --noEmit``.
    """
    from rote.adapters import get_adapter

    results: dict[str, Any] = {}
    for runtime in EMIT_RUNTIMES:
        out = work / runtime
        if out.exists():
            shutil.rmtree(out)
        try:
            get_adapter(runtime, extracted_source_dir=compiled).emit(pipeline, out)
            results[runtime] = {"ok": True}
        except Exception as e:  # noqa: BLE001 (any refusal is a benchmark result)
            results[runtime] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:500]}
    cf = results["cloudflare"]
    if not cf["ok"]:
        cf["tsc"] = None
    elif not tsc:
        cf["tsc"] = "skipped"
    elif shutil.which("npm") is None:
        cf["tsc"] = "no-npm"
    else:
        cf["tsc"], cf["tsc_output"] = _run_tsc(work / "cloudflare")
    return results


def _run_tsc(app: Path) -> tuple[str, str]:
    for argv in (
        ["npm", "install", "--no-audit", "--no-fund", "--prefer-offline"],
        ["npx", "--no-install", "tsc", "--noEmit"],
    ):
        try:
            proc = subprocess.run(
                argv,
                cwd=app,
                capture_output=True,
                text=True,
                timeout=TSC_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return "timeout", " ".join(argv)
        if proc.returncode != 0:
            tail = (proc.stdout + proc.stderr)[-1500:]
            return ("npm-install-failed" if argv[0] == "npm" else "failed"), tail
    return "passed", ""


def collect_job(
    out_dir: Path,
    skill_dir: Path,
    expected_yaml: Path,
    *,
    exit_code: int | None,
    wall_seconds: float | None,
    tsc: bool = True,
) -> dict[str, Any]:
    """Every metric for one compile job. Never raises on a bad compile:
    a missing or invalid pipeline is itself the result."""
    compiled = deliverable_dir(out_dir)
    metrics = read_compile_metrics(compiled)
    run: dict[str, Any] = {
        "exit_code": exit_code,
        "wall_seconds": round(wall_seconds, 1) if wall_seconds is not None else None,
        "deliverable": compiled.name if compiled is not None else None,
        "turns": metrics.get("num_turns"),
        "repair_rounds": metrics.get("repair_rounds"),
        "cost_usd": metrics.get("cost_usd"),
        "input_tokens": metrics.get("input_tokens"),
        "output_tokens": metrics.get("output_tokens"),
        "cache_write_tokens": metrics.get("cache_write_tokens"),
        "cache_read_tokens": metrics.get("cache_read_tokens"),
        "driver_duration_seconds": (
            round(metrics["duration_ms"] / 1000, 1)
            if isinstance(metrics.get("duration_ms"), int)
            else None
        ),
        "usage_complete": metrics.get("usage_complete"),
        **phase_durations(out_dir / "progress.jsonl"),
    }
    job: dict[str, Any] = {"run": run}
    if compiled is None:
        job["error"] = "no pipeline.yaml produced"
        return job
    try:
        pipeline = load_pipeline(compiled / "pipeline.yaml")
    except Exception as e:  # noqa: BLE001
        job["error"] = f"invalid pipeline.yaml: {e}"[:1000]
        return job
    job.update(score_pipeline(load_pipeline(expected_yaml), pipeline))
    job["verification"] = verification_metrics(pipeline, compiled, skill_dir)
    job["emit"] = emit_check(pipeline, compiled, out_dir / "bench-emit", tsc=tsc)
    return job
