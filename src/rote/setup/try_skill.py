"""``rote try [<skill>]``: compile one of your skills, then show it does the same thing.

The first-run proof (docs/design/compiled-skill-target.md §7, PR 8). It is
three :class:`~rote.setup.Step` objects run by :func:`rote.setup.run_steps`,
each a plan the person confirms before anything happens:

1. **compile**: ``rote compile <skill> --runtime skill --local ...`` with the
   argv :func:`rote.eval.skill_ab.compile_argv` pins (the decider included),
   through :func:`rote.eval.skill_ab.compile_for_eval`. Paid.
2. **install**: :func:`rote.skill_target.commands.install_plan`, the same
   plan ``rote skill install`` shows, and the same person check.
3. **eval**: :func:`rote.eval.skill_ab.run_subject`, the original and the
   compiled skill as ``claude -p`` subagents on the same input, ``--trials``
   pairs. Paid.

Nothing here compares or measures on its own: the fields come from
:mod:`rote.proof` and the numbers from :mod:`rote.eval.skill_ab`. This module
estimates, plans, records and words the result.

**Resumable.** Each step's status is read from what is on disk: the compile
record in ``<out>/try.json`` (for this skill's files and this input), the
apps registry, and ``<out>/try-result.json`` (for this compile). An
interrupted run is resumed by running the same command; finished steps say
"done already" and cost nothing. Every paid run goes into
``<out>/ledger.json``, across runs.

**Spend.** Each paid action carries a :class:`~rote.interact.plan.Spend`
whose flag is ``--live``: a person is asked (default no), and without one
``--yes --live`` is required. The approved high estimate is the ledger's cap,
so the eval stops before a run that would cross what the person said yes to.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rote.interact.plan import Action, Contact, FileWrite, Launch, Plan, Spend

if TYPE_CHECKING:
    from rote.eval import skill_ab
    from rote.interact.prompter import Prompter
    from rote.setup import SetupContext, StepStatus

#: The flag that lets a run with nobody at the terminal spend money.
LIVE_FLAG = "--live"
#: Pairs run by default: two, so the original against itself gives a noise floor.
DEFAULT_TRIALS = 2
#: Who pays for every paid run here.
PAYER = "your Claude subscription (claude -p)"
#: What the person approved and what was compiled, beside the compile.
STATE_FILE = "try.json"
#: The eval's result, read back on a re-run.
RESULT_FILE = "try-result.json"
LEDGER_FILE = "ledger.json"
STATE_FORMAT = "rote-try/1"
#: The high estimate is this many times the low one: a compile can take a
#: repair round, and an agent more turns than the recorded run took.
HIGH_FACTOR = 3.0


class TryError(RuntimeError):
    """``rote try`` cannot go ahead with what it was given; nothing was spent."""


# ───────── Estimates ─────────


@dataclass(frozen=True)
class Estimate:
    """One kind of paid run: dollars low to high, and seconds."""

    low_usd: float
    high_usd: float
    seconds: float

    def times(self, n: int) -> Estimate:
        return Estimate(self.low_usd * n, self.high_usd * n, self.seconds * n)

    def plus(self, other: Estimate) -> Estimate:
        return Estimate(
            self.low_usd + other.low_usd,
            self.high_usd + other.high_usd,
            self.seconds + other.seconds,
        )

    def spend(self) -> Spend:
        return Spend(round(self.low_usd, 2), round(self.high_usd, 2), PAYER, LIVE_FLAG)


@dataclass(frozen=True)
class Estimates:
    compile: Estimate
    #: A control run ("reply ok"), one per side.
    control: Estimate
    #: One trial of one side.
    trial: Estimate
    #: Where the numbers come from, for the plan.
    basis: str

    def eval(self, trials: int) -> Estimate:
        return self.control.times(2).plus(self.trial.times(2 * trials))


def _skill_bytes(skill_dir: Path) -> int:
    from rote.skill_files import reference_paths

    paths = [skill_dir / "SKILL.md", *reference_paths(skill_dir, recursive=True)]
    return sum(p.stat().st_size for p in paths if p.is_file())


def estimate(skill_dir: Path) -> Estimates:
    """docs-digest's recorded runs (``reference.json``), scaled by how much
    skill text there is to read, never below them.

    Both a compile and an agent run read the skill, so its size is the one
    number that scales both without a model call. The high end is
    :data:`HIGH_FACTOR` times the low one. A control run's estimate is
    :data:`rote.eval.skill_ab.DEFAULT_CONTROL_ESTIMATE_USD` (one request)."""
    from rote import demo
    from rote.eval.skill_ab import DEFAULT_CONTROL_ESTIMATE_USD

    raw = demo.load_reference().raw
    ratio = max(1.0, _skill_bytes(skill_dir) / _skill_bytes(demo.skill_dir()))

    def scaled(run: Mapping[str, Any]) -> Estimate:
        low = float(run["cost_usd"]) * ratio
        return Estimate(low, low * HIGH_FACTOR, float(run["wall_seconds"]) * ratio)

    basis = f"docs-digest's recorded runs ({raw.get('recorded_on')})"
    if ratio >= 1.05:
        basis += f", scaled by this skill's size ({ratio:.1f}x docs-digest's)"
    return Estimates(
        compile=scaled(raw["compile"]),
        control=Estimate(DEFAULT_CONTROL_ESTIMATE_USD / 5, DEFAULT_CONTROL_ESTIMATE_USD, 5.0),
        trial=scaled(raw["skill_run"]),
        basis=basis,
    )


def _money(e: Estimate) -> str:
    return f"${e.low_usd:.2f} to ${e.high_usd:.2f}"


def _minutes(seconds: float) -> str:
    minutes = seconds / 60
    return "under a minute" if minutes < 1 else f"about {round(minutes)} min"


# ───────── The session: one skill, one input, one out dir ─────────


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def skill_digest(skill_dir: Path) -> str:
    """Every file in the source skill, so an edit since the compile recompiles."""
    from rote.skill_target.trust import scan

    return _digest(scan(skill_dir).files)


@dataclass
class Runners:
    """What ``rote try`` runs, injectable for tests. The defaults are
    :mod:`rote.eval.skill_ab`'s own."""

    compile: Callable[..., tuple[Path, dict[str, Any]]]
    run_subject: Callable[[skill_ab.Harness, skill_ab.Subject, int], skill_ab.SubjectReport]
    #: The ``claude`` executable the trials run.
    executable: Callable[[], str]

    @classmethod
    def default(cls) -> Runners:
        from rote.eval import skill_ab

        return cls(
            compile=skill_ab.compile_for_eval,
            run_subject=skill_ab.run_subject,
            executable=lambda: skill_ab.require_executable("claude"),
        )


#: Replaced by tests: the trials run a fake ``claude`` there.
RUNNERS: Callable[[], Runners] = Runners.default


@dataclass
class Session:
    skill_dir: Path
    name: str
    #: The task: an input payload, or ``{"input", "signals"}`` for gates.
    task: dict[str, Any]
    out: Path
    trials: int
    #: The skill side's model.
    model: str
    estimates: Estimates
    runners: Runners = field(default_factory=lambda: RUNNERS())
    log: Callable[[str], None] = lambda line: print(line, file=sys.stderr)

    @property
    def payload(self) -> dict[str, Any]:
        """The input the compile is grounded in (the envelope's ``input``)."""
        inner = self.task.get("input")
        if "signals" in self.task and isinstance(inner, dict):
            return inner
        return self.task

    @property
    def compile_dir(self) -> Path:
        return self.out / "compile"

    @property
    def pipeline_yaml(self) -> Path:
        return self.compile_dir / "compiled" / "pipeline.yaml"

    @property
    def compiled_skill(self) -> Path:
        return self.compile_dir / "runtime" / "skill" / self.name

    def compile_argv(self) -> list[str]:
        from rote.eval.skill_ab import compile_argv

        return ["rote", *compile_argv(self.skill_dir, self.out, model=None)]

    def ledger(self, approved: Estimate) -> skill_ab.Ledger:
        """The ledger, capped at what was spent so far plus ``approved``'s high end."""
        from rote.eval.skill_ab import Ledger

        path = self.out / LEDGER_FILE
        spent = Ledger(path, 0.0).spent
        # A hair over, so a run estimated at exactly the approved high end fits.
        return Ledger(path, spent + approved.high_usd + 1e-9)

    def spent(self) -> float:
        from rote.eval.skill_ab import Ledger

        return Ledger(self.out / LEDGER_FILE, 0.0).spent

    def read_state(self) -> dict[str, Any]:
        path = self.out / STATE_FILE
        if not path.is_file():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) and data.get("format") == STATE_FORMAT else {}

    def write_json(self, name: str, data: Mapping[str, Any]) -> Path:
        from rote.interact.plan import atomic_write_text

        path = self.out / name
        atomic_write_text(path, json.dumps(data, indent=2, default=str) + "\n")
        return path

    def compile_key(self) -> dict[str, str]:
        """What a recorded compile must match to be reused."""
        return {
            "skill": str(self.skill_dir),
            "skill_digest": skill_digest(self.skill_dir),
            "input_digest": _digest(self.payload),
        }

    def compiled(self) -> dict[str, Any] | None:
        """The recorded compile, when it is of this skill and input and still on disk."""
        state = self.read_state()
        record = state.get("compile")
        if not isinstance(record, dict) or state.get("key") != self.compile_key():
            return None
        if not self.pipeline_yaml.is_file() or not (self.compiled_skill / "SKILL.md").is_file():
            return None
        return record

    def result_key(self) -> dict[str, str]:
        """What a recorded eval must match to be reused: the original skill and
        the input (both sides), and the pipeline the compiled skill runs."""
        return {
            **self.compile_key(),
            "pipeline_digest": hashlib.sha256(self.pipeline_yaml.read_bytes()).hexdigest(),
        }

    def result(self) -> dict[str, Any] | None:
        """The recorded eval, when it measured this skill, input and compile."""
        path = self.out / RESULT_FILE
        if not path.is_file() or not self.pipeline_yaml.is_file():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("key") != self.result_key():
            return None
        return data


# ───────── Step 1: compile ─────────


@dataclass
class CompileStep:
    session: Session
    id: str = "compile"
    title: str = "Compile the skill"
    command: str = ""

    def __post_init__(self) -> None:
        self.title = f"Compile {self.session.name} to a compiled skill"
        self.command = " ".join(self.session.compile_argv())

    def status(self, ctx: SetupContext) -> StepStatus:
        from rote.setup import StepStatus

        record = self.session.compiled()
        if record is None:
            return StepStatus("pending", "not compiled for this skill and input")
        cost = record.get("cost_usd")
        cost_text = "cost unknown" if cost is None else f"${float(cost):.3f}"
        return StepStatus("done", f"compiled into {self.session.compile_dir} ({cost_text})")

    def plan(self, ctx: SetupContext, prompter: Prompter) -> Plan:
        from rote import proof

        s = self.session
        e = s.estimates
        whole = e.compile.plus(e.eval(s.trials))
        effects = (
            Launch("claude", f"rote compile runs claude -p to compile {s.name}", paid=True),
            e.compile.spend(),
            FileWrite(
                s.compile_dir,
                "the compiled pipeline and the compiled skill",
                existed=s.compile_dir.exists(),
                replaces=s.compile_dir.exists(),
            ),
        )
        notes = [
            "pinned: "
            + ", ".join(f"{p.flag[2:]} {p.value} ({p.why})" for p in proof.pinned_settings()),
            f"in all, rote try spends {_money(whole)} and takes {_minutes(whole.seconds)}: this "
            f"compile ({_money(e.compile)}), then {s.trials} paired trial(s) of the original and "
            f"the compiled skill plus one control run per side ({_money(e.eval(s.trials))}, "
            "asked separately)",
            f"estimates: {e.basis}; nothing runs past the high end you approve",
        ]
        if s.spent():
            notes.append(f"already spent by earlier rote try runs here: ${s.spent():.3f}")
        return Plan(
            f"rote try {s.name}: compile",
            (Action("compile", self.title, effects, self._run, self.command),),
            tuple(notes),
        )

    def _run(self) -> Path:
        s = self.session
        ledger = s.ledger(s.estimates.compile)
        s.out.mkdir(parents=True, exist_ok=True)
        s.log(f"  compiling; watch it with: tail -f {s.compile_dir / 'progress.jsonl'}")
        pipeline_yaml, provenance = s.runners.compile(
            s.skill_dir,
            s.payload,
            s.out,
            ledger,
            model=None,
            log=s.log,
            estimate_usd=s.estimates.compile.high_usd,
            caller="rote try",
        )
        s.write_json(
            STATE_FILE,
            {"format": STATE_FORMAT, "key": s.compile_key(), "compile": provenance},
        )
        return pipeline_yaml


# ───────── Step 2: install ─────────


@dataclass
class InstallStep:
    session: Session
    id: str = "install"
    title: str = "Install the compiled skill"
    command: str = ""

    def __post_init__(self) -> None:
        self.command = f"rote skill install {self.session.compiled_skill}"

    def _plan(self) -> tuple[Plan, str | None]:
        from rote.skill_target import commands, trust

        skill = trust.read_skill_dir(self.session.compiled_skill)
        now = trust.scan(skill.root)
        found = trust.against_manifest(skill, now)
        if not found.clean:
            raise TryError(f"{skill.root} is not what rote emitted ({found.describe()})")
        read, apps = commands._registry()
        return commands.install_plan(skill, read, apps, now)

    def status(self, ctx: SetupContext) -> StepStatus:
        from rote.setup import StepStatus

        plan, _why = self._plan()
        if not plan.actions:
            return StepStatus("done", "; ".join(plan.notes))
        return StepStatus("pending", "not installed")

    def plan(self, ctx: SetupContext, prompter: Prompter) -> Plan:
        from rote.skill_target.commands import _require_person

        plan, why = self._plan()
        _require_person(ctx.mode, prompter, plan, why, self.command)
        return plan


# ───────── Step 3: the paired eval ─────────


@dataclass
class EvalStep:
    session: Session
    id: str = "eval"
    title: str = "Run the original and the compiled skill on the same input"
    command: str = ""

    def __post_init__(self) -> None:
        s = self.session
        self.command = (
            f"rote eval {s.pipeline_yaml} --run --runtime skill --skill {s.skill_dir} "
            f"--trials {s.trials} --input <task.json> --budget <usd>"
        )

    def status(self, ctx: SetupContext) -> StepStatus:
        from rote.setup import StepStatus

        if self.session.result() is not None:
            where = self.session.out / RESULT_FILE
            return StepStatus("done", f"measured; the result is in {where}")
        return StepStatus("pending", "not measured for this compile")

    def subject(self) -> skill_ab.Subject:
        from rote.eval import skill_ab

        s = self.session
        record = s.compiled() or {}
        try:
            return skill_ab.subject_for(
                s.pipeline_yaml, skill_dir=s.skill_dir, task=s.task, provenance=record
            )
        except skill_ab.SkillAbError as e:
            raise TryError(str(e)) from None

    def plan(self, ctx: SetupContext, prompter: Prompter) -> Plan:
        from urllib.parse import urlsplit

        from rote.eval.empirical import default_corpus_path

        s = self.session
        subject = self.subject()
        cost = s.estimates.eval(s.trials)
        hosts = sorted({urlsplit(u).hostname or "" for u in subject.live_servers.values()} - {""})
        effects: list[Any] = [
            Launch(
                "claude",
                f"claude -p runs the original and the compiled skill, {s.trials} time(s) each, "
                "plus one control run per side",
                paid=True,
            ),
            cost.spend(),
            *(Contact(h, "the skill's MCP reads, read-only, on both sides") for h in hosts),
            FileWrite(
                s.out / RESULT_FILE,
                "the comparison",
                existed=(s.out / RESULT_FILE).exists(),
            ),
            FileWrite(
                default_corpus_path(),
                "append the measurements",
                existed=default_corpus_path().exists(),
            ),
        ]
        notes = [
            f"input: {json.dumps(subject.input)[:200]}",
            f"compared: {', '.join(f.name for f in subject.fields)}",
            f"takes {_minutes(cost.seconds)}",
        ]
        return Plan(
            f"rote try {s.name}: the paired eval",
            (Action("eval", self.title, tuple(effects), lambda: self._run(subject)),),
            tuple(notes),
        )

    def _run(self, subject: skill_ab.Subject) -> Path:
        from datetime import UTC, datetime

        from rote.eval import skill_ab

        s = self.session
        e = s.estimates
        work = s.out / "eval"
        work.mkdir(parents=True, exist_ok=True)
        harness = skill_ab.Harness(
            work=work,
            model=s.model,
            ledger=s.ledger(e.eval(s.trials)),
            executable=s.runners.executable(),
            log=s.log,
            trial_estimate_usd=e.trial.high_usd,
            control_estimate_usd=e.control.high_usd,
            caller="rote try",
        )
        spent_before = harness.ledger.spent
        report = s.runners.run_subject(harness, subject, s.trials)
        generated_at = datetime.now(UTC).isoformat(timespec="seconds")
        skill_ab.append_corpus(
            [skill_ab.corpus_record(report, model=s.model, generated_at=generated_at)]
        )
        summary = summarize(report)
        summary.update(
            {
                "key": s.result_key(),
                "generated_at": generated_at,
                "model": s.model,
                "spent_usd": harness.ledger.spent,
                "eval_spent_usd": round(harness.ledger.spent - spent_before, 6),
                "eval_approved_usd": round(e.eval(s.trials).high_usd, 2),
                "report": report.to_dict(),
            }
        )
        return s.write_json(RESULT_FILE, summary)


# ───────── The result, in words ─────────


def summarize(report: skill_ab.SubjectReport) -> dict[str, Any]:
    """Every number and verdict the result shows, as data (``--json`` and the
    resumed re-render read this, never the report)."""
    from rote import proof

    pairs = report.pairs
    fields = []
    for index, spec in enumerate(report.subject.fields):
        in_pairs = [c.fields[index] for c in pairs]
        in_noise = [c.fields[index] for c in report.noise]
        first_off = next((f for f in in_pairs if f.status != "matched"), None)
        fields.append(
            {
                "name": spec.name,
                "expect": spec.expect,
                "why": spec.why,
                "matched": sum(1 for f in in_pairs if f.status == "matched"),
                "pairs": len(in_pairs),
                "statuses": [f.status for f in in_pairs],
                "noise_matched": sum(1 for f in in_noise if f.status == "matched"),
                "noise_pairs": len(in_noise),
                "difference": None
                if first_off is None
                else {
                    "status": first_off.status,
                    "original": first_off.original_value,
                    "compiled": first_off.compiled_value,
                    "problem": first_off.problem,
                },
            }
        )
    compared = pairs[0].compared if pairs else 0
    problems = [
        f"{side} #{t.index}: {p}"
        for side, summary in report.sides.items()
        for t in summary.trials
        for p in t.problems
    ]
    if report.stopped:
        problems.append(f"stopped early: {report.stopped}")
    same = bool(pairs) and all(c.same_result for c in pairs) and not problems
    ok = bool(pairs) and all(c.ok for c in pairs) and not problems
    if not pairs:
        headline = "no pair finished, so nothing was compared"
    elif compared == 0:
        headline = "no fields to compare exactly"
    else:
        reproduced = min(c.reproduced for c in pairs)
        headline = f"reproduced {reproduced} of {compared} fields in every pair"
        if any(c.reproduced != reproduced for c in pairs):
            headline = (
                f"reproduced {', '.join(str(c.reproduced) for c in pairs)} of {compared} "
                "fields across the pairs"
            )
    verdict = proof.verdict(pairs[0]) if pairs else "Nothing was compared."
    if pairs and len(pairs) > 1 and not all(c.same_result for c in pairs):
        verdict = next(proof.verdict(c) for c in pairs if not c.same_result)
    return {
        "skill": report.subject.name,
        "headline": headline,
        "verdict": verdict,
        "ok": ok,
        "same_result": same,
        "compared": compared,
        "pairs": len(pairs),
        "noise_pairs": len(report.noise),
        "fields": fields,
        "sides": {side: summary.medians() for side, summary in report.sides.items()},
        "trials_ok": {
            side: f"{len(summary.ok_trials)}/{len(summary.trials)}"
            for side, summary in report.sides.items()
        },
        "problems": problems,
    }


def _flat(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return " ".join(text.split())


def difference(original: Any, compiled: Any, *, width: int = 60) -> str:
    """Both values, or for long ones the stretch of each where they first part."""
    a, b = _flat(original), _flat(compiled)
    if len(a) <= width and len(b) <= width:
        return f"original {a!r}, compiled {b!r}"
    pairs = zip(a, b, strict=False)
    at = next((i for i, (x, y) in enumerate(pairs) if x != y), min(len(a), len(b)))
    start = max(0, at - 15)

    def cut(text: str) -> str:
        piece = text[start : start + width]
        return ("..." if start else "") + piece + ("..." if start + width < len(text) else "")

    return f"they part at character {at}: original {cut(a)!r}, compiled {cut(b)!r}"


def _num(value: Any, unit: str = "") -> str:
    if value is None:
        return "n/a"
    if unit == "$":
        return f"${float(value):.3f}"
    if unit == "k":
        return f"{float(value) / 1000:.1f}k"
    if unit == "s":
        return f"{float(value):.1f} s"
    return f"{float(value):.0f}"


def render(summary: Mapping[str, Any], *, name: str, next_steps: Sequence[str]) -> list[str]:
    """The short result a person reads."""
    lines = [f"rote try {name}: {summary['headline']}", ""]
    width = max([len(f["name"]) for f in summary["fields"]] + [5])
    header = f"  {'field'.ljust(width)}  {'expect':<8}{'compiled = original':<22}"
    lines.append(header + "original = itself")
    for f in summary["fields"]:
        noise = (
            f"{f['noise_matched']} of {f['noise_pairs']}"
            if f["noise_pairs"]
            else "not measured (1 trial)"
        )
        matched = f"{f['matched']} of {f['pairs']} pairs"
        lines.append(f"  {f['name'].ljust(width)}  {f['expect']:<8}{matched:<22}{noise}")
    for f in summary["fields"]:
        d = f["difference"]
        if d is None:
            continue
        what = d["problem"] or difference(d["original"], d["compiled"])
        lines.append(f"  {d['status']}: {f['name']}: {what}")
        if f["noise_pairs"] and f["noise_matched"] < f["noise_pairs"]:
            lines.append("    the original differs from itself here too: noise, not the compile")
    sides = summary["sides"]
    rows = (
        ("wall time", "wall_seconds", "s"),
        ("turns", "num_turns", ""),
        ("input tokens", "input_tokens", "k"),
        ("output tokens", "output_tokens", "k"),
        ("cost", "cost_usd", "$"),
    )
    lines += ["", f"  {'':<16}{'original skill':<17}compiled skill"]
    for label, key, unit in rows:
        lines.append(
            f"  {label:<16}{_num(sides['original'].get(key), unit):<17}"
            f"{_num(sides['compiled'].get(key), unit)}"
        )
    lines.append(
        f"  (medians of the trials that passed every check: original "
        f"{summary['trials_ok']['original']}, compiled {summary['trials_ok']['compiled']})"
    )
    lines += ["", summary["verdict"]]
    for problem in summary["problems"]:
        lines.append(f"  problem: {problem}")
    if summary.get("spent_usd") is not None:
        lines.append(
            f"Spent ${summary['spent_usd']:.3f} in all: ${summary['eval_spent_usd']:.3f} on the "
            f"trials (approved up to ${summary['eval_approved_usd']:.2f}), the rest on the compile."
        )
    if next_steps:
        lines += ["", "Next:"]
        lines += [f"  {step}" for step in next_steps]
    return lines


def next_steps(session: Session, summary: Mapping[str, Any]) -> list[str]:
    s = session
    steps = []
    if summary.get("ok"):
        steps.append(
            f"an agent runs the compiled skill with: rote skill run {s.name} start --file "
            "<input.json>"
        )
        steps.append(
            f"let Claude Code run it without asking: rote skill install {s.compiled_skill} "
            "--save-allow-rule"
        )
    else:
        steps.append(f"read the trials' transcripts under {s.out / 'eval'}")
    steps.append(f"measure again: delete {s.out / RESULT_FILE} and run rote try again")
    return steps


# ───────── Choosing the skill and its input ─────────

#: How many discovered skills a person is offered.
OFFERED = 10


def choose_skill(target: str | None, prompter: Prompter) -> Path:
    """``target`` (a skill directory, or a name ``rote skills list`` shows), or
    a pick from the discovered skills, best ranked first and the default."""
    from rote.interact.prompter import Choice
    from rote.skills import discover

    if target is not None and (Path(target) / "SKILL.md").is_file():
        return Path(target).resolve()
    found = discover()
    if target is not None:
        named = [s for s in found.skills if s.name == target]
        if len(named) == 1:
            return named[0].path
        if named:
            paths = ", ".join(str(s.path) for s in named)
            raise TryError(f"{len(named)} skills are named {target}: pass one directory ({paths})")
        raise TryError(
            f"{target} is not a skill directory (no SKILL.md) and no skill of that name was "
            "found: rote skills list shows them"
        )
    offered = [s for s in found.skills if s.readiness.state != "unreadable"][:OFFERED]
    if not offered:
        raise TryError("no skills found: pass a skill directory (rote try <dir>)")
    choices = [Choice(str(s.path), s.name, hint=s.readiness.reason) for s in offered]
    picked = prompter.select(
        "Which skill should rote try?", choices, flag="--skill", default=choices[0].value
    )
    return Path(picked)


def read_task(skill_dir: Path, input_arg: str | None, prompter: Prompter) -> dict[str, Any]:
    """``--input``, else the skill's own ``input.json``, else asked for."""
    if input_arg is not None:
        path = Path(input_arg)
    elif (skill_dir / "input.json").is_file():
        path = skill_dir / "input.json"
    else:

        def check(text: str) -> str | None:
            return None if Path(text).is_file() else f"{text} is not a file"

        path = Path(
            prompter.text(
                f"{skill_dir.name} has no input.json. Which JSON file should both sides run on?",
                flag="--input",
                validate=check,
            )
        )
    try:
        task = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise TryError(f"cannot read the input {path}: {e}") from None
    if not isinstance(task, dict):
        raise TryError(f"the input {path} must hold a JSON object")
    return task


# ───────── The command ─────────


def cmd_try(args: Any) -> int:
    """``rote try [<skill>]``. Exit 0: the compiled skill matched on every field
    expected to match exactly, in every pair. 1: it did not, nothing could be
    compared, or a step failed. 2: an answer or a consent is missing
    (``action_required``)."""
    from rote.adapters.skill import read_identity
    from rote.compiler.drivers.claude import DEFAULT_MODEL
    from rote.interact.mode import Mode
    from rote.interact.output import Output
    from rote.interact.plan import FORCE_FLAG, ConfirmationRequired
    from rote.interact.prompter import PromptCancelled, PromptRequired, prompter_for
    from rote.interact.report import (
        EXIT_ACTION_REQUIRED,
        PersonRequired,
        action_required,
        person_required,
    )
    from rote.setup import SetupContext, Step, run_steps
    from rote.skill_target.trust import SkillTrustError

    try:
        mode = Mode.resolve(json=args.json, yes=args.yes, agent_mode=args.agent_mode)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    out = Output(mode)
    if args.trials < 1:
        out.error(f"--trials must be at least 1, got {args.trials}")
        return 2
    if args.skill is not None and args.skill_flag is not None:
        out.error("pass the skill once: as an argument or as --skill, not both")
        return 2
    prompter = prompter_for(mode)
    argv: list[str] = list(getattr(args, "raw_argv", sys.argv[1:]))
    granted = {f for f, given in ((LIVE_FLAG, args.live), (FORCE_FLAG, args.force)) if given}
    try:
        skill_dir = choose_skill(args.skill or args.skill_flag, prompter)
        task = read_task(skill_dir, args.input, prompter)
        name = read_identity(skill_dir).name
        session = Session(
            skill_dir=skill_dir,
            name=name,
            task=task,
            out=Path(args.out or f"rote-try-{name}").resolve(),
            trials=args.trials,
            model=DEFAULT_MODEL,
            estimates=estimate(skill_dir),
            log=out.say,
        )
        out.say(f"rote try: {name} ({skill_dir}), working in {session.out}")
        steps: list[Step] = [CompileStep(session), InstallStep(session), EvalStep(session)]
        reports = run_steps(steps, SetupContext(mode, out), prompter, granted=granted)
    except (PromptRequired, ConfirmationRequired) as e:
        out.error(str(e))
        out.result(action_required(e, argv))
        return EXIT_ACTION_REQUIRED
    except PersonRequired as e:
        out.error(str(e))
        out.result(person_required(e))
        return EXIT_ACTION_REQUIRED
    except (TryError, SkillTrustError, ValueError) as e:
        message = f"{e.cause} Fix: {e.fix}" if isinstance(e, SkillTrustError) else str(e)
        out.error(message)
        out.result({"ok": False, "error": message})
        return 1
    except PromptCancelled:
        out.say("rote try: cancelled; the steps already done stay done")
        return 130
    except KeyboardInterrupt:
        out.say("rote try: interrupted. Run the same command again to resume from this step")
        return 130
    steps_json = [r.to_dict() for r in reports]
    for report in reports:
        for outcome in report.outcomes:
            if outcome.status != "done":
                out.say(f"  {outcome.status}: {outcome.action_id} ({outcome.error})")
    last = reports[-1]
    if last.step_id != "eval" or last.state in ("declined", "failed"):
        why = "not applied" if last.state == "declined" else last.detail
        out.say(f"rote try: stopped at {last.step_id} ({why}). Run it again to resume")
        out.result(
            {"ok": False, "skill": session.name, "out": str(session.out), "steps": steps_json}
        )
        return 1
    summary = session.result()
    if summary is None:
        out.error("the eval finished but recorded no result")
        return 1
    follow = next_steps(session, summary)
    out.say()
    for line in render(summary, name=session.name, next_steps=follow):
        out.say(line)
    out.result(
        {
            "ok": bool(summary["ok"]),
            "out": str(session.out),
            "compile": session.compiled(),
            "steps": steps_json,
            "next": [{"command": step} for step in follow],
            **{k: v for k, v in summary.items() if k != "report"},
        }
    )
    return 0 if summary["ok"] else 1
