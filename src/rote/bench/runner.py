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
import contextlib
import json
import os
import shutil
import sys
import tempfile
import time
from collections.abc import Iterator
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


#: The sample input an example ships beside its skill (examples/<name>/).
SAMPLE_INPUT_FILENAME = "sample.json"


def sample_input(examples: Path, skill: str) -> Path | None:
    """The example's sample run, which every compile is grounded in.

    ``sample.json`` is the ``{"input", "signals"}`` envelope: the skill's
    input and the answer a person gives at each gate. It has the same keys
    as the fidelity cases and none of their values (different people, ids
    and amounts; pinned by a test), the analog of the sample a real user
    supplies: the keys are the contract, the values are test data. It sits
    beside the skill, not in it, so the bench passes it with
    ``--sample-input``. ``input.json``, ``expected/`` and ``fidelity/``
    stay out of the compile, since the cases are built from them.
    """
    path = examples / skill / SAMPLE_INPUT_FILENAME
    return path if path.is_file() else None


#: The per-job MCP registry the compile reads, beside the job's output.
TOOL_REGISTRY_RELPATH = "bench-tool-servers/mcp.json"


@contextlib.contextmanager
def tool_servers(example: Path, job_dir: Path, env: dict[str, str]) -> Iterator[dict[str, str]]:
    """Register the example's fixture MCP servers for the length of one compile.

    A real compile lists each registered server's tools (``tools/list``)
    to ground its tool arguments (:mod:`rote.compiler.tool_grounding`).
    The bench's analog of "the user has these servers registered" is the
    fidelity fixtures, started here with no responses at all: the compile
    only ever lists their tools, and the schemas it reads are the very
    ``parameters`` the fixtures enforce when the pipeline later runs, with
    no second copy to drift. Yields the compile's environment, pointing
    ``ROTE_MCP_CONFIG`` at a registry of this job's own, so concurrent
    jobs never share one. An example without fixture servers changes
    nothing.
    """
    from rote.bench.fidelity import Case, fixture_servers, has_cases, load_cases

    if not has_cases(example):
        yield env
        return
    casefile = load_cases(example)
    if not casefile.servers:
        yield env
        return
    silent = Case(
        id="compile", description="", input={}, gates=[], gate_instructions=None, fixtures={}
    )
    registry = job_dir / TOOL_REGISTRY_RELPATH
    with fixture_servers(casefile, silent, registry.parent) as (urls, _log):
        registry.write_text(
            json.dumps(
                {
                    "version": 1,
                    "servers": {
                        name: {"url": url, "transport": "streamable-http"}
                        for name, url in urls.items()
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        yield {**env, "ROTE_MCP_CONFIG": str(registry)}


def compile_argv(
    skill_copy: Path, out_dir: Path, args: argparse.Namespace, sample: Path | None = None
) -> list[str]:
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
    if sample is not None:
        argv += ["--sample-input", str(sample)]
    if args.agent:
        argv += ["--agent", args.agent]
    argv += args.compile_flag
    return argv


def discover_skills(examples: Path) -> list[str]:
    """Every ``<name>/`` under ``examples`` with a skill and an expected IR.

    The five original examples keep their ``DEFAULT_SKILLS`` order; any
    other skill (the holdout set) follows, smallest skill first, so a
    ``--budget`` cut drops the expensive compiles rather than the cheap
    ones. Size is the bytes of SKILL.md plus its references.

    ``examples/holdout/`` is not itself a skill, so discovery on
    ``examples/`` never reaches the held-out set: benching it takes an
    explicit ``--examples examples/holdout``.
    """
    found = [
        d.name
        for d in examples.iterdir()
        if (d / "skill" / "SKILL.md").is_file() and (d / "expected" / "pipeline.yaml").is_file()
    ]
    if not found:
        raise SystemExit(
            f"rote bench: no <name>/skill + <name>/expected/pipeline.yaml in {examples}"
        )
    rank = {s: i for i, s in enumerate(DEFAULT_SKILLS)}

    def size(name: str) -> int:
        return sum(p.stat().st_size for p in (examples / name / "skill").rglob("*.md"))

    return sorted(found, key=lambda n: (rank.get(n, len(rank)), size(n), n))


def _fidelity_example(args: argparse.Namespace, skill: str) -> Path | None:
    """The example dir to run fidelity cases from, when ``--fidelity`` is on."""
    return args.examples / skill if args.fidelity else None


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
        argv = compile_argv(skill_copy, job.out_dir, args, sample_input(args.examples, job.skill))
        print(f"rote bench: {job.label}: compiling -> {job.out_dir}", file=sys.stderr)
        started = time.monotonic()
        with (
            tool_servers(args.examples / job.skill, job.out_dir, env) as job_env,
            (job.out_dir / "bench-compile.stdout.log").open("wb") as out,
            (job.out_dir / "bench-compile.stderr.log").open("wb") as err,
        ):
            proc = await asyncio.create_subprocess_exec(
                *argv, cwd=job.out_dir, env=job_env, stdout=out, stderr=err
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
        fidelity_example=_fidelity_example(args, job.skill),
    )
    budget.add((result.get("fidelity") or {}).get("cost_usd"))
    spent = budget.add(result["run"].get("cost_usd"))
    row = headline(result)
    print(
        f"rote bench: {job.label}: exit {code}, {wall:.0f}s, turns {row['turns']}, "
        f"${row['cost_usd'] if row['cost_usd'] is not None else '?'} "
        f"(spent ${spent:.2f}), kind {row['kind_accuracy']}, verified {row['verified']}, "
        f"fidelity {row['fidelity']}",
        file=sys.stderr,
    )
    for line in coverage_lines(result.get("fidelity") or {}):
        print(f"  {line}", file=sys.stderr)
    return {
        "skill": job.skill,
        "repeat": job.repeat,
        "out_dir": job.out_dir.relative_to(args.out).as_posix(),
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


def produced_pipeline(job: dict[str, Any]) -> bool:
    """Whether a job got as far as a pipeline.yaml (``--resume`` keeps these)."""
    return "headline" in job and not str(job.get("error", "")).startswith(
        "no pipeline.yaml produced"
    )


def total_spend(jobs: list[dict[str, Any]], replaced: list[dict[str, Any]]) -> float:
    """Every dollar the run dir cost: its jobs plus the failed jobs a
    ``--resume`` re-ran (their spend is real even though their row is gone)."""
    costs = [j.get("run", {}).get("cost_usd") for j in jobs] + [r.get("cost_usd") for r in replaced]
    return round(sum(c for c in costs if isinstance(c, (int, float))), 4)


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
    p.add_argument(
        "--examples",
        type=Path,
        help=(
            "The repo's examples/ directory, or examples/holdout for the held-out "
            "skills nobody tunes against"
        ),
    )
    p.add_argument("--out", type=Path, help="Run directory (compiles, state, results.json)")
    p.add_argument(
        "--skills",
        default=None,
        help=(
            "Comma-separated subset (default: every skill under --examples; for "
            f"examples/ that is {','.join(DEFAULT_SKILLS)})"
        ),
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
    p.add_argument(
        "--compile-flag",
        action="append",
        default=[],
        metavar="FLAG",
        help=(
            "Extra flag passed to every rote compile, repeatable "
            "(e.g. --compile-flag=--no-system1 for the full-rubric path)"
        ),
    )
    p.add_argument("--no-tsc", action="store_true", help="Skip npm install + tsc on cloudflare")
    p.add_argument("--label", default=None, help="Free text stored in results.json (a git sha)")
    p.add_argument(
        "--fidelity",
        action="store_true",
        help=(
            "Also run each compiled pipeline on the example's fidelity cases and "
            "score it against the recorded skill reference (judges spend tokens)"
        ),
    )
    p.add_argument(
        "--record-reference",
        action="store_true",
        help=(
            "Run each skill as an agent on its fidelity cases (--trials times) and "
            "write examples/<name>/fidelity/reference.json. SPENDS TOKENS; no compile"
        ),
    )
    p.add_argument("--trials", type=int, default=3, help="Agent runs per case (--record-reference)")
    p.add_argument(
        "--fidelity-of",
        type=Path,
        default=None,
        metavar="COMPILED_DIR",
        help=(
            "Score one compiled dir (holding pipeline.yaml and extracted/) on the "
            "fidelity cases of the single --skills example, print JSON, and exit"
        ),
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Keep the jobs already scored in <out>/results.json and run only the "
            "missing ones (their spend counts toward --budget)"
        ),
    )
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


def portable(obj: Any, replacements: dict[str, str]) -> Any:
    """``obj`` with every absolute local path prefix replaced by a placeholder.

    results.json is meant to be committed and diffed across machines, so it
    must not carry a run dir, a checkout path or a home directory. Job dirs
    are already relative to the run dir; this catches the rest (argv, error
    messages, tsc output). Longest prefix first, so a run dir inside the
    checkout is not half-replaced by the checkout's placeholder.
    """
    if isinstance(obj, dict):
        return {k: portable(v, replacements) for k, v in obj.items()}
    if isinstance(obj, list):
        return [portable(v, replacements) for v in obj]
    if isinstance(obj, str):
        for prefix in sorted(replacements, key=len, reverse=True):
            obj = obj.replace(prefix, replacements[prefix])
        return obj
    return obj


def _replacements(out: Path, examples: Path | None) -> dict[str, str]:
    reps = {
        str(out): "<run>",
        sys.executable: "<python>",
        sys.prefix: "<venv>",
        str(Path.home()): "~",
    }
    if out.resolve() != out:
        reps[str(out.resolve())] = "<run>"
    if examples is not None:
        reps[str(examples)] = "<examples>"
        reps[str(examples.resolve())] = "<examples>"
    return reps


def _finish(results: dict[str, Any], args: argparse.Namespace, out: Path | None) -> int:
    if out is not None:
        results = portable(results, _replacements(out, args.examples))
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
    run_dir: Path = args.rescore.resolve()
    prior = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    if args.examples is None:
        print("rote bench: --rescore needs --examples (results.json is portable)", file=sys.stderr)
        return 2
    examples = args.examples = args.examples.resolve()
    jobs: list[dict[str, Any]] = []
    for old in prior["jobs"]:
        if "out_dir" not in old:
            jobs.append(old)
            continue
        out_dir = run_dir / old["out_dir"]
        _, expected = _skill_paths(examples, old["skill"])
        result = collect_job(
            out_dir,
            out_dir / "skill-src" / old["skill"],
            expected,
            exit_code=old["run"]["exit_code"],
            wall_seconds=old["run"]["wall_seconds"],
            tsc=not args.no_tsc,
            fidelity_example=_fidelity_example(args, old["skill"]),
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
        "spent_usd": total_spend(jobs, prior.get("replaced_jobs", [])),
        "rescored_at": datetime.now(UTC).isoformat(),
    }
    return _finish(results, args, run_dir)


def _coverage_line(case_id: str, ref: dict[str, Any]) -> str:
    """One case's reference coverage, flagged when it checks too little."""
    cov = ref.get("coverage", ref)
    flag = ""
    if ref.get("insufficient_runs"):
        flag = f"  LOW: only {ref.get('successful_runs')} successful run(s), nothing is stable"
    elif cov.get("low"):
        parts = [
            f"{label} {', '.join(names)}"
            for label, names in (
                ("values differ:", cov.get("uncovered_value_disagreements") or []),
                ("layout differs:", cov.get("uncovered_layout_disagreements") or []),
                ("absent in some runs:", cov.get("uncovered_missing_in_some_runs") or []),
            )
            if names
        ]
        flag = "  LOW: " + "; ".join(parts)
    counts = (
        f"{cov.get('covered')}/{cov.get('fields')} contract fields with a stable leaf, "
        f"{cov.get('stable_leaves')}/{cov.get('leaves')} leaves stable"
    )
    return f"reference {case_id}: {counts}{flag}"


def coverage_lines(fidelity: dict[str, Any]) -> list[str]:
    """Per-case coverage lines for a ``pipeline_fidelity`` report."""
    return [
        _coverage_line(cid, c["reference"])
        for cid, c in (fidelity.get("cases") or {}).items()
        if "reference" in c
    ]


def _fidelity_command(args: argparse.Namespace) -> int:
    """``--record-reference`` and ``--fidelity-of``: no compile involved."""
    from rote.bench.fidelity import has_cases, load_cases, pipeline_fidelity, record_reference

    if args.examples is None:
        print("rote bench: --examples is required", file=sys.stderr)
        return 2
    examples = args.examples.resolve()
    skills = (
        discover_skills(examples)
        if args.skills is None
        else [s.strip() for s in args.skills.split(",") if s.strip()]
    )
    if args.fidelity_of is not None:
        if len(skills) != 1:
            print("rote bench: --fidelity-of needs exactly one --skills example", file=sys.stderr)
            return 2
        report = pipeline_fidelity(args.fidelity_of.resolve(), examples / skills[0])
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
        for line in coverage_lines(report):
            print(f"rote bench: {line}", file=sys.stderr)
        return 0
    from rote.compiler.drivers.claude import DEFAULT_MODEL

    work = Path(args.out).resolve() if args.out else Path(tempfile.mkdtemp(prefix="rote-ref-"))
    spent = 0.0
    for skill in skills:
        example = examples / skill
        if not has_cases(example):
            print(f"rote bench: {skill}: no fidelity/cases.yaml, skipped", file=sys.stderr)
            continue
        if args.budget is not None and spent >= args.budget:
            print(f"rote bench: {skill}: skipped, budget spent", file=sys.stderr)
            continue
        casefile = load_cases(example)
        reference = record_reference(
            casefile,
            trials=args.trials,
            model=args.model or DEFAULT_MODEL,
            work=work / skill,
            log=lambda line: print(line, file=sys.stderr),
            budget=None if args.budget is None else args.budget - spent,
        )
        spent += reference["cost_usd"]
        casefile.reference_path.write_text(
            json.dumps(portable(reference, _replacements(work, examples)), indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        print(f"rote bench: {skill}: reference ${reference['cost_usd']:.2f}")
        for cid, c in reference["cases"].items():
            print(f"  {_coverage_line(cid, c['stable'])}")
        print(f"  wrote {casefile.reference_path}")
    print(f"rote bench: reference traces cost ${spent:.2f} in all")
    return 0


def main(argv: list[str]) -> int:
    args = _parser().parse_args(argv)
    if args.report is not None:
        results = json.loads(args.report.read_text(encoding="utf-8"))
        return _finish(results, args, None)
    # Fidelity and reference runs happen in this process's children (the
    # skill's claude -p, the emitted DBOS app), not in the compile children
    # that get isolated_env, so isolate this process too: a developer's MCP
    # registry or config must not change what a fidelity run sees.
    run_dir = args.rescore or args.out
    state = (
        Path(run_dir).resolve() / "state"
        if run_dir is not None
        else Path(tempfile.mkdtemp(prefix="rote-bench-state-"))
    )
    os.environ.update(isolated_env(state))
    for key in ("ROTE_RUNTIME", "ROTE_DEPLOY", "ROTE_AGENT", "ROTE_MODEL"):
        os.environ.pop(key, None)
    if args.rescore is not None:
        return _rescore(args)
    if args.record_reference or args.fidelity_of is not None:
        return _fidelity_command(args)
    if args.examples is None or args.out is None:
        print("rote bench: --examples and --out are required to run compiles", file=sys.stderr)
        return 2
    args.examples = args.examples.resolve()
    out: Path = args.out.resolve()
    args.out = out
    if args.skills is None:
        skills = discover_skills(args.examples)
    else:
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
    kept: list[dict[str, Any]] = []
    replaced: list[dict[str, Any]] = []
    if args.resume and (out / "results.json").is_file():
        prior = json.loads((out / "results.json").read_text(encoding="utf-8"))
        wanted = {(j.skill, j.repeat) for j in jobs}
        mine = [j for j in prior["jobs"] if (j["skill"], j["repeat"]) in wanted]
        # Jobs this invocation does not name stay as they were, so resuming
        # one skill never drops another skill's row (or its spend).
        kept = [j for j in prior["jobs"] if (j["skill"], j["repeat"]) not in wanted]
        kept += [j for j in mine if produced_pipeline(j)]
        replaced = list(prior.get("replaced_jobs", [])) + [
            {
                "skill": j["skill"],
                "repeat": j["repeat"],
                "cost_usd": j.get("run", {}).get("cost_usd"),
                "error": j.get("error"),
            }
            for j in mine
            if not produced_pipeline(j) and "skipped" not in j
        ]
        done = {(j["skill"], j["repeat"]) for j in kept}
        skills = list(dict.fromkeys([*skills, *(j["skill"] for j in kept)]))
        jobs = [j for j in jobs if (j.skill, j.repeat) not in done]
        for j in kept:
            budget.add(j["run"].get("cost_usd"))
        for r in replaced:
            budget.add(r.get("cost_usd"))
        started = prior.get("started_at", started)
        print(
            f"rote bench: resuming: {len(kept)} job(s) kept (${budget.spent:.2f}), "
            f"{len(jobs)} to run",
            file=sys.stderr,
        )

    async def _all() -> list[dict[str, Any]]:
        gate = asyncio.Semaphore(args.concurrency)
        return list(await asyncio.gather(*(_run_job(j, args, env, gate, budget) for j in jobs)))

    rank = {s: i for i, s in enumerate([*skills, *DEFAULT_SKILLS])}
    job_results = sorted(
        [*kept, *asyncio.run(_all())],
        key=lambda j: (j["repeat"], rank.get(j["skill"], len(rank)), j["skill"]),
    )
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
        "spent_usd": total_spend(job_results, replaced),
        "replaced_jobs": replaced,
        "jobs": job_results,
        "per_skill": per_skill,
        "aggregate": agg,
    }
    return _finish(results, args, out)
