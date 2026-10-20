"""Run the compile benchmark: ``rote bench`` (hidden from ``rote --help``).

Each job is one real ``rote compile`` subprocess of one example skill,
run through the same interpreter as this command (``sys.executable -m
rote``), so the benchmark measures exactly the installed rote it runs
from. Point it at an immutable wheel install of the branch under test.

Isolation is not optional. Every job gets the ``ROTE_*`` path overrides
the test suite uses, pointed at ``<out>/state/``: a developer's config
could pin a runtime or a model, a cloud login would move the compile
server-side, and an app registry would collect the emitted dirs. The
skill is copied next to the job first, so the compiler agent cannot
wander from the skill dir into ``expected/`` and read the answer.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rote.bench.collect import collect_job
from rote.bench.report import (
    aggregate,
    compare,
    headline,
    render_compare,
    render_table,
    variance,
)

#: The examples with a hand-written expected/pipeline.yaml, smallest
#: first. expense-audit alone is the cheap inner loop (about $2.5).
DEFAULT_SKILLS = ("expense-audit", "ops-report", "invoice-push", "deal-monitor", "bdr-outreach")

#: The ``ROTE_*`` overrides ``tests/conftest.py`` sets, relative to the
#: state dir. Keep the two lists in step.
STATE_ENV = {
    "ROTE_APPS_PATH": "rote-apps/apps.json",
    "ROTE_CONFIG_PATH": "rote-config/config.yaml",
    "ROTE_PROJECT_CONFIG_PATH": "rote-config/rote.yaml",
    "ROTE_CLOUD_CRED_PATH": "cloud-cred/cloud.json",
    "ROTE_MCP_CONFIG": "mcp-config/mcp.json",
    "ROTE_MCP_TOKEN_DIR": "mcp-tokens",
}

RESULTS_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Job:
    skill: str
    repeat: int
    out_dir: Path

    @property
    def label(self) -> str:
        return f"{self.skill}#{self.repeat}"


def isolated_env(state_dir: Path) -> dict[str, str]:
    """The child environment: the caller's, with every rote state path moved."""
    env = dict(os.environ)
    for key, rel in STATE_ENV.items():
        env[key] = str(state_dir / rel)
    # The bench passes --runtime/--model/--agent explicitly; an exported
    # ROTE_RUNTIME etc. would otherwise change what is being measured.
    for key in ("ROTE_RUNTIME", "ROTE_DEPLOY", "ROTE_AGENT", "ROTE_MODEL"):
        env.pop(key, None)
    return env


def compile_argv(skill_copy: Path, out_dir: Path, args: argparse.Namespace) -> list[str]:
    argv = [
        sys.executable,
        "-m",
        "rote",
        "compile",
        str(skill_copy),
        "--local",
        "--runtime",
        args.runtime,
        "--no-deploy",
        "--no-eval",
        "--json",
        "--out",
        str(out_dir),
    ]
    if args.model:
        argv += ["--model", args.model]
    if args.agent:
        argv += ["--agent", args.agent]
    return argv


def _skill_paths(examples: Path, skill: str) -> tuple[Path, Path]:
    skill_dir = examples / skill / "skill"
    expected = examples / skill / "expected" / "pipeline.yaml"
    if not (skill_dir / "SKILL.md").is_file():
        raise SystemExit(f"rote bench: {skill_dir} is not a skill (no SKILL.md)")
    if not expected.is_file():
        raise SystemExit(f"rote bench: {expected} does not exist")
    return skill_dir, expected


async def _run_job(
    job: Job,
    args: argparse.Namespace,
    env: dict[str, str],
    gate: asyncio.Semaphore,
    budget: _Budget,
) -> dict[str, Any]:
    skill_dir, expected = _skill_paths(args.examples, job.skill)
    async with gate:
        if budget.exhausted():
            print(f"rote bench: {job.label}: skipped, budget spent", file=sys.stderr)
            return {"skill": job.skill, "repeat": job.repeat, "skipped": "budget"}
        if job.out_dir.exists():
            shutil.rmtree(job.out_dir)
        job.out_dir.mkdir(parents=True)
        skill_copy = job.out_dir / "skill-src" / job.skill
        shutil.copytree(skill_dir, skill_copy)
        argv = compile_argv(skill_copy, job.out_dir, args)
        print(f"rote bench: {job.label}: compiling -> {job.out_dir}", file=sys.stderr)
        started = time.monotonic()
        with (
            (job.out_dir / "bench-compile.stdout.log").open("wb") as out,
            (job.out_dir / "bench-compile.stderr.log").open("wb") as err,
        ):
            proc = await asyncio.create_subprocess_exec(
                *argv, cwd=job.out_dir, env=env, stdout=out, stderr=err
            )
            code = await proc.wait()
        wall = time.monotonic() - started
    result = await asyncio.to_thread(
        collect_job,
        job.out_dir,
        skill_copy,
        expected,
        exit_code=code,
        wall_seconds=wall,
        tsc=not args.no_tsc,
    )
    spent = budget.add(result["run"].get("cost_usd"))
    row = headline(result)
    print(
        f"rote bench: {job.label}: exit {code}, {wall:.0f}s, turns {row['turns']}, "
        f"${row['cost_usd'] if row['cost_usd'] is not None else '?'} "
        f"(spent ${spent:.2f}), kind {row['kind_accuracy']}, verified {row['verified']}",
        file=sys.stderr,
    )
    return {
        "skill": job.skill,
        "repeat": job.repeat,
        "out_dir": str(job.out_dir),
        "argv": argv[1:],
        "headline": row,
        **result,
    }


class _Budget:
    """Running inference spend; new jobs stop launching once it is reached.

    Jobs already running finish (a killed compile is money spent for
    nothing), so the overshoot is bounded by ``concurrency - 1`` jobs.
    A job whose cost is unknown counts as zero and is flagged in its row.
    """

    def __init__(self, limit: float | None) -> None:
        self.limit = limit
        self.spent = 0.0

    def exhausted(self) -> bool:
        return self.limit is not None and self.spent >= self.limit

    def add(self, cost: Any) -> float:
        if isinstance(cost, (int, float)):
            self.spent += float(cost)
        return self.spent


def summarize(jobs: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(per_skill, aggregate)`` over the jobs that ran."""
    ran = [j for j in jobs if "headline" in j]
    per_skill: dict[str, Any] = {}
    for skill in dict.fromkeys(j["skill"] for j in ran):
        rows = [j["headline"] for j in ran if j["skill"] == skill]
        per_skill[skill] = {"jobs": len(rows), "mean": aggregate(rows), "variance": variance(rows)}
    return per_skill, aggregate([j["headline"] for j in ran])


def render_results(results: dict[str, Any]) -> str:
    jobs = [j for j in results["jobs"] if "headline" in j]
    rows: list[tuple[str, dict[str, Any]]] = [
        (f"{j['skill']}#{j['repeat']}", j["headline"]) for j in jobs
    ]
    multi = any(v["jobs"] > 1 for v in results["per_skill"].values())
    if multi:
        rows += [(f"{s} (mean)", v["mean"]) for s, v in results["per_skill"].items()]
    rows.append(("AGGREGATE", results["aggregate"]))
    text = [render_table(rows)]
    skipped = [j for j in results["jobs"] if "skipped" in j]
    if skipped:
        text.append(
            "skipped (budget): " + ", ".join(f"{j['skill']}#{j['repeat']}" for j in skipped)
        )
    text.append(
        f"spent ${results['spent_usd']:.2f} over {results['aggregate']['jobs']} job(s); "
        f"aggregate cost$ and wall_s are sums, everything else is a mean"
    )
    if multi:
        lines = ["run-to-run spread (min..max, stdev) per skill:"]
        keys = ("kind_accuracy", "node_recall", "edge_f1", "cost_usd", "turns", "wall_seconds")
        for skill, v in results["per_skill"].items():
            parts = [
                f"{k} {s['min']}..{s['max']} ({s['stdev']})"
                for k in keys
                if (s := v["variance"].get(k)) is not None
            ]
            lines.append(f"  {skill}: " + "; ".join(parts))
        text.append("\n".join(lines))
    return "\n\n".join(text)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rote bench",
        description=(
            "Compile the example skills with the real compiler (SPENDS TOKENS), score "
            "each pipeline.yaml against examples/<name>/expected/pipeline.yaml, and "
            "write a diffable results.json. See docs/compile-bench.md."
        ),
    )
    p.add_argument("--examples", type=Path, help="The repo's examples/ directory")
    p.add_argument("--out", type=Path, help="Run directory (compiles, state, results.json)")
    p.add_argument(
        "--skills",
        default=",".join(DEFAULT_SKILLS),
        help=f"Comma-separated subset (default: all: {','.join(DEFAULT_SKILLS)})",
    )
    p.add_argument("--repeat", type=int, default=1, help="Compiles per skill, for variance")
    p.add_argument("--concurrency", type=int, default=1, help="Compiles in flight at once")
    p.add_argument(
        "--budget",
        type=float,
        default=None,
        help="Stop launching compiles once total_cost_usd reaches this many dollars",
    )
    p.add_argument("--runtime", default="dbos", help="The compile's --runtime (default: dbos)")
    p.add_argument("--model", default=None, help="Passed to rote compile --model")
    p.add_argument("--agent", default=None, help="Passed to rote compile --agent")
    p.add_argument("--no-tsc", action="store_true", help="Skip npm install + tsc on cloudflare")
    p.add_argument("--label", default=None, help="Free text stored in results.json (a git sha)")
    p.add_argument(
        "--rescore",
        type=Path,
        default=None,
        metavar="RUN_DIR",
        help="Re-collect an existing run dir's compiles without compiling (no tokens)",
    )
    p.add_argument(
        "--report",
        type=Path,
        default=None,
        metavar="RESULTS_JSON",
        help="Print an existing results.json (with --compare: its deltas) and exit",
    )
    p.add_argument(
        "--compare",
        type=Path,
        default=None,
        metavar="OLD_JSON",
        help="Print deltas of this run against an older results.json",
    )
    return p


def _finish(results: dict[str, Any], args: argparse.Namespace, out: Path | None) -> int:
    print(render_results(results))
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        path = out / "results.json"
        path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (out / "results.txt").write_text(render_results(results) + "\n", encoding="utf-8")
        print(f"\nresults: {path}")
    if args.compare is not None:
        old = json.loads(args.compare.read_text(encoding="utf-8"))
        print()
        print(render_compare(compare(old, results)))
    return 0


def _rescore(args: argparse.Namespace) -> int:
    run_dir: Path = args.rescore
    prior = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    examples = args.examples or Path(prior["config"]["examples"])
    jobs: list[dict[str, Any]] = []
    for old in prior["jobs"]:
        if "out_dir" not in old:
            jobs.append(old)
            continue
        out_dir = Path(old["out_dir"])
        _, expected = _skill_paths(examples, old["skill"])
        result = collect_job(
            out_dir,
            out_dir / "skill-src" / old["skill"],
            expected,
            exit_code=old["run"]["exit_code"],
            wall_seconds=old["run"]["wall_seconds"],
            tsc=not args.no_tsc,
        )
        jobs.append(
            {
                **{k: old[k] for k in ("skill", "repeat", "out_dir", "argv") if k in old},
                "headline": headline(result),
                **result,
            }
        )
    per_skill, agg = summarize(jobs)
    results = {
        **prior,
        "jobs": jobs,
        "per_skill": per_skill,
        "aggregate": agg,
        "rescored_at": datetime.now(UTC).isoformat(),
    }
    return _finish(results, args, run_dir)


def main(argv: list[str]) -> int:
    args = _parser().parse_args(argv)
    if args.report is not None:
        results = json.loads(args.report.read_text(encoding="utf-8"))
        return _finish(results, args, None)
    if args.rescore is not None:
        return _rescore(args)
    if args.examples is None or args.out is None:
        print("rote bench: --examples and --out are required to run compiles", file=sys.stderr)
        return 2
    args.examples = args.examples.resolve()
    out: Path = args.out.resolve()
    skills = [s.strip() for s in args.skills.split(",") if s.strip()]
    for skill in skills:
        _skill_paths(args.examples, skill)
    if args.repeat < 1 or args.concurrency < 1:
        print("rote bench: --repeat and --concurrency must be >= 1", file=sys.stderr)
        return 2

    from rote import __version__

    env = isolated_env(out / "state")
    jobs = [Job(s, k, out / s / f"r{k}") for k in range(1, args.repeat + 1) for s in skills]
    budget = _Budget(args.budget)
    started = datetime.now(UTC).isoformat()

    async def _all() -> list[dict[str, Any]]:
        gate = asyncio.Semaphore(args.concurrency)
        return list(await asyncio.gather(*(_run_job(j, args, env, gate, budget) for j in jobs)))

    job_results = asyncio.run(_all())
    per_skill, agg = summarize(job_results)
    results = {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "rote_version": __version__,
        "label": args.label,
        "started_at": started,
        "finished_at": datetime.now(UTC).isoformat(),
        "config": {
            "examples": str(args.examples),
            "skills": skills,
            "repeat": args.repeat,
            "concurrency": args.concurrency,
            "runtime": args.runtime,
            "model": args.model,
            "agent": args.agent,
            "budget": args.budget,
            "python": sys.executable,
        },
        "spent_usd": round(budget.spent, 4),
        "jobs": job_results,
        "per_skill": per_skill,
        "aggregate": agg,
    }
    return _finish(results, args, out)
