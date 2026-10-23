"""``rote try``: show a skill and its compiled pipeline doing the same thing.

Two paths, one comparison.

- **The precompiled proof** (``rote try`` or ``rote try docs-digest``, the
  default). The bundled docs-digest pipeline (:mod:`rote.demo`) runs live
  on the recorded input: two MCP calls to public, read-only servers and no
  model call, so it costs nothing. Its output is compared, field by field,
  against the skill run recorded in ``reference.json``.
- **The live path** (``--live``, and always for a skill directory). The
  skill runs read-only as an agent (``rote run <skill>``'s run), then
  ``rote compile`` runs with ``--sample-input`` set to the same input, then
  the compiled pipeline runs on that input, and the two fresh outputs are
  compared.

Every step is an :class:`~rote.interact.plan.Action` in one
:class:`~rote.interact.plan.Plan`, shown and confirmed before anything
runs. Each paid step carries a :class:`~rote.interact.plan.Spend` whose
flag is ``--live``, so ``--yes`` alone never spends and a person sees one
confirmation with the total estimate. Estimates are never constants:

- docs-digest's come from its recorded runs (``skill_run.cost_usd``,
  ``compile.cost_usd``);
- another skill's are the recorded ones scaled by rote's own estimate of
  the two skills (:func:`rote.eval.estimate.estimate_skill`, the same model
  ``rote eval`` uses, on the offline token heuristic): the agent run by its
  token footprint, the compile by the skill text it reads.

How fields compare. A field names a value in each output (JSON pointers,
with the reference's ``section`` / ``measure`` narrowing, read by
:func:`rote.demo.value_at`), and values are compared after
:func:`rote.bench.fidelity_score.normalize` (whitespace and case folded,
numbers to the cent). What a difference means depends on what the field
is expected to do:

======== ==================== ============================================
expect   a difference reads   where the expectation comes from
======== ==================== ============================================
equal    ``differs``          the reference, or the IR for a field only code
                              and tools produce
drifts   ``drifted``          the reference: the value is live data
wording  ``differs-judge``    the IR: an llm_judge or agent_loop node is
                              upstream, so the words can change
======== ==================== ============================================

A value that is missing from either output is ``differs`` whatever the
field expects. "Same result" is claimed only from ``equal`` fields, and the
headline reads "reproduced N of M fields" over those.
"""

from __future__ import annotations

import contextlib
import json
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlsplit

from rote import demo
from rote.interact.plan import (
    Action,
    Contact,
    Effect,
    FileWrite,
    Launch,
    Plan,
    Spend,
    StaleFile,
    file_state,
)

if TYPE_CHECKING:
    from rote.eval.empirical import MeasuredRun
    from rote.interact.prompter import Prompter
    from rote.ir import Pipeline
    from rote.setup import SetupContext, StepStatus

#: The flag every paid step needs when nobody is at the terminal.
LIVE_FLAG = "--live"
#: Who pays for ``claude -p`` (the skill run and the compile both use it,
#: with API keys scrubbed: rote.inference.build_subscription_env).
SUBSCRIPTION = "your Claude subscription (claude -p)"
#: The host ``claude -p`` sends model requests to.
ANTHROPIC_HOST = "api.anthropic.com"
#: The runtime the live path compiles for and runs: it speaks MCP and needs
#: no Node and no Docker (the README's reason for the demo, too).
RUNTIME = "dbos"

Expect = Literal["equal", "drifts", "wording"]
Status = Literal["matched", "differs", "drifted", "differs-judge"]

_ON_MISMATCH: dict[Expect, Status] = {
    "equal": "differs",
    "drifts": "drifted",
    "wording": "differs-judge",
}


class TryError(RuntimeError):
    """``rote try`` cannot build a plan for what it was given."""


# ───────── Fields ─────────


@dataclass(frozen=True)
class FieldSpec:
    """One value both outputs are compared on."""

    name: str
    #: JSON pointer into the skill's output.
    skill: str
    #: JSON pointer into the pipeline's output.
    pipeline: str
    section: str | None
    measure: demo.Measure | None
    expect: Expect
    why: str

    @classmethod
    def from_reference(cls, ref_field: demo.Field) -> FieldSpec:
        return cls(
            name=ref_field.name,
            skill=ref_field.skill,
            pipeline=ref_field.pipeline,
            section=ref_field.section,
            measure=ref_field.measure,
            expect=ref_field.expect,
            why=ref_field.why,
        )


def _pointer_token(key: str) -> str:
    return key.replace("~", "~0").replace("/", "~1")


_MODEL_KINDS = ("llm_judge", "agent_loop")


def model_nodes(pipeline: Pipeline) -> list[str]:
    """The nodes that call a model when the pipeline runs."""
    return [n.id for n in pipeline.nodes if n.kind.value in _MODEL_KINDS]


def _upstream(pipeline: Pipeline, node_id: str) -> set[str]:
    """``node_id`` and every node whose output can reach it (edges and ``inputs:``)."""
    from rote.ir import parse_input_ref

    parents: dict[str, set[str]] = {n.id: set() for n in pipeline.nodes}
    for edge in pipeline.edges:
        parents.setdefault(edge.to, set()).add(edge.from_)
    for node in pipeline.nodes:
        for ref in (node.inputs or {}).values():
            source = parse_input_ref(ref).node_id
            if source is not None:
                parents[node.id].add(source)
    seen: set[str] = set()
    stack = [node_id]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(parents.get(current, ()))
    return seen


def fields_from_ir(pipeline: Pipeline) -> tuple[FieldSpec, ...]:
    """The fields to compare when there is no reference: every declared key
    of every exit node (the pipeline returns ``{exit_id: result}``; the
    skill's result.json is read at the same key, unprefixed).

    A field is ``wording`` when an llm_judge or agent_loop node is the exit
    node or upstream of it, and ``equal`` otherwise: code and tool calls
    alone produce it, so the same input should give the same value.
    """
    model = set(model_nodes(pipeline))
    single_exit = len(pipeline.exit_nodes) == 1
    fields: list[FieldSpec] = []
    for exit_id in pipeline.exit_nodes:
        node = pipeline.node_by_id(exit_id)
        judged = sorted(_upstream(pipeline, exit_id) & model)
        expect: Expect = "wording" if judged else "equal"
        why = (
            f"produced downstream of {', '.join(judged)}, which call a model"
            if judged
            else "produced by code and tool calls only"
        )
        keys: list[str] = []
        if isinstance(node.output, dict) and node.output:
            keys = list(node.output)
        elif isinstance(node.output_schema, dict) and node.output_schema.get("type") == "object":
            keys = list((node.output_schema.get("properties") or {}).keys())
        if not keys:
            fields.append(
                FieldSpec(
                    name=exit_id,
                    skill=f"/{_pointer_token(exit_id)}",
                    pipeline=f"/{_pointer_token(exit_id)}",
                    section=None,
                    measure=None,
                    expect=expect,
                    why=why,
                )
            )
            continue
        for key in keys:
            fields.append(
                FieldSpec(
                    name=key if single_exit else f"{exit_id}.{key}",
                    skill=f"/{_pointer_token(key)}",
                    pipeline=f"/{_pointer_token(exit_id)}/{_pointer_token(key)}",
                    section=None,
                    measure=None,
                    expect=expect,
                    why=why,
                )
            )
    return tuple(fields)


@dataclass(frozen=True)
class FieldResult:
    spec: FieldSpec
    status: Status
    #: The compared values, after section and measure (None when missing).
    skill_value: Any
    pipeline_value: Any
    #: Why a value could not be read, when one could not.
    problem: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.spec.name,
            "expect": self.spec.expect,
            "status": self.status,
            "why": self.spec.why,
            "skill": self.skill_value,
            "pipeline": self.pipeline_value,
            "problem": self.problem,
        }


def _canonical(value: Any) -> Any:
    """``normalize`` applied to every leaf, order kept (a reading list's
    order is part of it)."""
    from rote.bench.fidelity_score import normalize

    if isinstance(value, Mapping):
        return {str(k): _canonical(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    return normalize(value)


def compare_outputs(
    fields: Sequence[FieldSpec], skill_output: Any, pipeline_output: Any
) -> tuple[FieldResult, ...]:
    results: list[FieldResult] = []
    for spec in fields:
        problems: list[str] = []
        values: list[Any] = []
        for side, document, pointer in (
            ("skill", skill_output, spec.skill),
            ("pipeline", pipeline_output, spec.pipeline),
        ):
            try:
                values.append(demo.value_at(document, pointer, spec.section, spec.measure))
            except demo.DemoError as e:
                values.append(None)
                problems.append(f"{side} output: {e}")
        if problems:
            status: Status = "differs"
        elif _canonical(values[0]) == _canonical(values[1]):
            status = "matched"
        else:
            status = _ON_MISMATCH[spec.expect]
        results.append(FieldResult(spec, status, values[0], values[1], "; ".join(problems) or None))
    return tuple(results)


@dataclass(frozen=True)
class Comparison:
    fields: tuple[FieldResult, ...]

    @property
    def compared(self) -> int:
        """The ``equal`` fields: the only ones a "same result" claim rests on."""
        return sum(1 for f in self.fields if f.spec.expect == "equal")

    @property
    def reproduced(self) -> int:
        return sum(1 for f in self.fields if f.spec.expect == "equal" and f.status == "matched")

    @property
    def ok(self) -> bool:
        """No field differs where it must not (a drift or a judge's wording is fine)."""
        return all(f.status != "differs" for f in self.fields)

    @property
    def same_result(self) -> bool:
        return self.ok and self.compared > 0 and self.reproduced == self.compared

    @property
    def headline(self) -> str:
        return f"reproduced {self.reproduced} of {self.compared} fields"

    def to_dict(self) -> dict[str, Any]:
        return {
            "headline": self.headline,
            "reproduced": self.reproduced,
            "compared": self.compared,
            "same_result": self.same_result,
            "fields": [f.to_dict() for f in self.fields],
        }


# ───────── Runs ─────────


@dataclass(frozen=True)
class SideMetrics:
    """One side of the before/after table."""

    label: str
    wall_seconds: float | None
    #: Agent turns (the skill side).
    turns: int | None
    #: Model calls (the pipeline side; a judge or an agent-loop turn).
    model_calls: int | None
    cost_usd: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "wall_seconds": self.wall_seconds,
            "turns": self.turns,
            "model_calls": self.model_calls,
            "cost_usd": self.cost_usd,
        }


def skill_metrics(run: MeasuredRun, label: str) -> SideMetrics:
    return SideMetrics(label, run.wall_seconds, run.turns, None, run.cost_usd)


def pipeline_metrics(run: MeasuredRun, label: str) -> SideMetrics:
    """A pipeline run's wall time, its model calls (the usage log's rows),
    and their cost: 0 with none, unknown when any row has no price."""
    costs = [row.get("cost_usd") for row in run.judge_usage]
    cost = None if any(c is None for c in costs) else float(sum(c for c in costs if c))
    return SideMetrics(label, run.wall_seconds, None, len(run.judge_usage), cost)


@dataclass(frozen=True)
class CompileRun:
    out_dir: Path
    wall_seconds: float
    turns: int | None
    cost_usd: float | None

    @property
    def pipeline_yaml(self) -> Path:
        return self.out_dir / "compiled" / "pipeline.yaml"

    @property
    def app_dir(self) -> Path:
        return self.out_dir / "runtime" / RUNTIME

    def to_dict(self) -> dict[str, Any]:
        return {
            "out_dir": str(self.out_dir),
            "wall_seconds": self.wall_seconds,
            "turns": self.turns,
            "cost_usd": self.cost_usd,
        }


@dataclass(frozen=True)
class Runners:
    """The three things ``rote try`` runs, injectable for tests.

    The defaults are rote's own commands: ``rote run``'s skill run,
    ``rote compile`` itself, and ``rote run``'s pipeline run."""

    run_skill: Callable[[Path, dict[str, Any]], MeasuredRun]
    compile: Callable[[Path, Path, Path], CompileRun]
    run_pipeline: Callable[[Path, dict[str, Any], Pipeline], MeasuredRun]

    @classmethod
    def default(cls) -> Runners:
        return cls(run_skill=_run_skill, compile=_compile, run_pipeline=_run_pipeline)


def _run_skill(skill_dir: Path, payload: dict[str, Any]) -> MeasuredRun:
    from rote.compiler.drivers.claude import DEFAULT_MODEL
    from rote.runners import run_skill

    run = run_skill(skill_dir, payload, model=DEFAULT_MODEL).run
    if not run.succeeded:
        raise RuntimeError(f"the skill run failed: {run.error}")
    if run.missing_mcp_servers:
        missing = ", ".join(f"{s} ({why})" for s, why in run.missing_mcp_servers.items())
        raise RuntimeError(
            f"the skill run never called {missing}, so its output may rest on "
            "substituted data and is not a fair reference"
        )
    return run


def compile_command(skill_dir: Path, input_path: Path, out_dir: Path) -> list[str]:
    """The ``rote compile`` argv the live path runs (without ``rote``)."""
    return [
        "compile",
        str(skill_dir),
        "--sample-input",
        str(input_path),
        "--runtime",
        RUNTIME,
        "--out",
        str(out_dir),
        "--agent",
        "claude",
        "--local",
        "--no-eval",
    ]


def _compile(skill_dir: Path, input_path: Path, out_dir: Path) -> CompileRun:
    """``rote compile`` in this process, its stdout sent to stderr so a
    ``--json`` run keeps one object on stdout."""
    from rote.bench.collect import decisions_cost_usd, read_compile_metrics
    from rote.cli import main as cli_main

    started = time.monotonic()
    with contextlib.redirect_stdout(sys.stderr):
        code = cli_main(compile_command(skill_dir, input_path, out_dir))
    wall = time.monotonic() - started
    if code != 0:
        raise RuntimeError(f"rote compile exited {code} (its output is above)")
    compiled = out_dir / "compiled"
    meta = read_compile_metrics(compiled)
    agent_cost = meta.get("cost_usd")
    decisions = decisions_cost_usd(compiled)
    cost = None if agent_cost is None or decisions is None else float(agent_cost) + float(decisions)
    turns = meta.get("num_turns")
    return CompileRun(out_dir, wall, int(turns) if isinstance(turns, int) else None, cost)


def _run_pipeline(app_dir: Path, payload: dict[str, Any], pipeline: Pipeline) -> MeasuredRun:
    from rote.runners import detect_target, preflight_pipeline, run_pipeline

    target = detect_target(app_dir, runtime=RUNTIME)
    preflight_pipeline(target)
    run = run_pipeline(target, payload, pipeline=pipeline, interactive=False).run
    if not run.succeeded:
        raise RuntimeError(f"the pipeline run failed: {run.error}")
    return run


# ───────── The plan ─────────


@dataclass(frozen=True)
class TryRequest:
    #: None for the bundled demo, else a skill directory.
    skill_dir: Path | None
    live: bool
    #: The input file (a skill directory's run; the demo has its own).
    input_path: Path | None = None
    #: Where the live path's compile writes (``--out``).
    out_dir: Path | None = None


@dataclass
class TryState:
    """What the plan's actions produced, read by the report afterwards."""

    skill: MeasuredRun | None = None
    compile: CompileRun | None = None
    pipeline: MeasuredRun | None = None


def _host(url: str) -> str:
    host = urlsplit(url).hostname
    if not host:
        raise TryError(f"cannot read a host from {url!r}")
    return host


def _total(spends: Sequence[Spend]) -> tuple[float, float]:
    return sum(s.low_usd for s in spends), sum(s.high_usd for s in spends)


def _money(low: float, high: float) -> str:
    return f"about ${low:.2f}" if low == high else f"${low:.2f} to ${high:.2f}"


@dataclass(frozen=True)
class Estimates:
    """The live path's spend, in dollars, and where each number came from."""

    skill_run: tuple[float, float]
    compile: tuple[float, float]
    #: None when the pipeline is known to make no model call.
    pipeline_run: tuple[float, float] | None
    basis: str


def estimate_spend(reference: demo.Reference, skill_dir: Path | None) -> Estimates:
    """The demo's recorded costs, or those scaled to another skill.

    Scaling uses :func:`rote.eval.estimate.estimate_skill` on both skills
    with the offline token heuristic, so no price list or tokenizer is
    fetched. The agent run scales by the estimated tokens it reads and
    writes (low and high against the demo's midpoint), the compile by the
    skill text it reads. A compiled pipeline's own model calls are unknown
    until it is compiled; they are bounded by the agent run's high
    estimate, since the pipeline makes at most the calls the agent did.
    """
    skill_cost = float(reference.raw["skill_run"]["cost_usd"])
    compile_cost = float(reference.raw["compile"]["cost_usd"])
    if skill_dir is None:
        pipeline_cost = float(reference.raw["pipeline_run"]["cost_usd"])
        return Estimates(
            skill_run=(skill_cost, skill_cost),
            compile=(compile_cost, compile_cost),
            pipeline_run=None if pipeline_cost == 0 else (pipeline_cost, pipeline_cost),
            basis=f"docs-digest's recorded runs ({reference.raw.get('recorded_on')})",
        )
    from rote.eval.estimate import estimate_skill
    from rote.eval.priors import Priors
    from rote.eval.tokens import HeuristicTokenCounter

    priors = Priors()
    counter = HeuristicTokenCounter(priors)
    demo_est = estimate_skill(demo.skill_dir(), counter, priors)
    user_est = estimate_skill(skill_dir, counter, priors)

    def footprint(est: Any) -> tuple[float, float]:
        low = est.fresh_input_tokens.low + est.cached_read_tokens.low + est.output_tokens.low
        high = est.fresh_input_tokens.high + est.cached_read_tokens.high + est.output_tokens.high
        return low, high

    demo_low, demo_high = footprint(demo_est)
    demo_mid = (demo_low + demo_high) / 2
    user_low, user_high = footprint(user_est)
    skill_run = (skill_cost * user_low / demo_mid, skill_cost * user_high / demo_mid)
    scaled_compile = compile_cost * user_est.context_tokens / demo_est.context_tokens
    return Estimates(
        skill_run=skill_run,
        compile=(scaled_compile, scaled_compile),
        pipeline_run=(0.0, skill_run[1]),
        basis=(
            "docs-digest's recorded runs, scaled by rote's estimate of this skill "
            f"({user_est.turn_method}; {user_est.token_count_method})"
        ),
    )


def _spend(cost: tuple[float, float]) -> Spend:
    return Spend(
        low_usd=round(cost[0], 4), high_usd=round(cost[1], 4), payer=SUBSCRIPTION, flag=LIVE_FLAG
    )


def _registered_hosts() -> list[str]:
    from rote.mcp.registry import load_registry

    return sorted({_host(c.url) for c in load_registry().servers.values()})


def _contacts(hosts: Sequence[str], purpose: str) -> tuple[Contact, ...]:
    return tuple(Contact(h, purpose) for h in dict.fromkeys(hosts))


def _register_action(reference: demo.Reference) -> tuple[Action | None, list[str]]:
    """Register the demo's servers read-only with no login, if they are not
    already: ``rote mcp add <name> <url> --auth none --read-only``, through
    the same function that command calls."""
    from rote.mcp._runtime_helper import normalize_server_url
    from rote.mcp.learned import learned_auth, learned_path
    from rote.mcp.registry import McpServerConfig, add_server, load_registry, registry_path

    registry = load_registry()
    todo: list[demo.Server] = []
    replaced: list[str] = []
    notes: list[str] = []
    for server in reference.servers:
        current = registry.servers.get(server.name)
        same_url = current is not None and normalize_server_url(
            current.url
        ) == normalize_server_url(server.url)
        learned = learned_auth(server.name, server.url)
        if (
            current is not None
            and same_url
            and current.read_only is True
            and learned is not None
            and learned.auth == "none"
        ):
            notes.append(f"{server.name}: already registered read-only, no login")
            continue
        if current is not None and not same_url:
            replaced.append(server.name)
        todo.append(server)
    if not todo:
        return None, notes
    reg_path, auth_path = registry_path(), learned_path()
    reg_state, auth_state = file_state(reg_path), file_state(auth_path)
    names = ", ".join(s.name for s in todo)
    summary = f"add {names}, read-only"
    if replaced:
        summary += f"; replaces the URL registered for {', '.join(replaced)}"

    def run() -> list[str]:
        for path, state in ((reg_path, reg_state), (auth_path, auth_state)):
            if file_state(path) != state:
                raise StaleFile(f"{path} changed since the plan was made; nothing written, re-run")
        for server in todo:
            add_server(server.name, McpServerConfig(url=server.url, read_only=True), auth="none")
        return [s.name for s in todo]

    effects: tuple[Effect, ...] = (
        FileWrite(
            path=reg_path,
            summary=summary,
            existed=reg_state is not None,
            replaces=bool(replaced),
            file_mode=0o600,
        ),
        FileWrite(
            path=auth_path,
            summary=f"record that {names} need no login",
            existed=auth_state is not None,
            file_mode=0o600,
        ),
    )
    command = " && ".join(f"rote mcp add {s.name} {s.url} --auth none --read-only" for s in todo)
    return (
        Action(
            id="register-servers",
            title=f"Register the demo's public MCP servers: {names}",
            effects=effects,
            run=run,
            command=command,
        ),
        notes,
    )


def _guard_model_calls(pipeline: Pipeline, approved: bool) -> None:
    calls = model_nodes(pipeline)
    if calls and not approved:
        raise RuntimeError(
            f"the compiled pipeline calls a model ({', '.join(calls)}), which this "
            "plan did not include; nothing ran. Run it on purpose with rote run"
        )


def _read_input(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise TryError(f"--input {path}: {e}") from e
    if not isinstance(payload, dict):
        raise TryError(f"--input {path}: expected a JSON object")
    return payload


def _skill_input(skill_dir: Path, given: Path | None, prompter: Prompter) -> Path:
    """``--input``, else the sample ``rote compile`` would pick up
    (``input.json`` or ``examples/input.json`` in the skill dir), else asked."""
    if given is not None:
        return given
    for candidate in (skill_dir / "input.json", skill_dir / "examples" / "input.json"):
        if candidate.is_file():
            return candidate
    answer = prompter.text(
        f"Which input should the skill and the pipeline run on? (a JSON file; {skill_dir} "
        "has no input.json)",
        flag="--input",
        validate=lambda v: None if Path(v).is_file() else f"no file at {v}",
    )
    return Path(answer)


class TryStep:
    """``rote try`` as a :class:`rote.setup.Step`, so ``rote setup`` can run it."""

    id = "try"
    title = "See a skill become a pipeline that does the same thing"
    command = "rote try"

    def __init__(self, request: TryRequest, runners: Runners | None = None) -> None:
        self.request = request
        self.runners = runners or Runners.default()
        self.state = TryState()
        self.reference = demo.load_reference()
        self.fields: tuple[FieldSpec, ...] = ()
        #: The pipeline-run spend the plan approved (None: no model calls).
        self._pipeline_spend: Spend | None = None
        self._live = request.live or request.skill_dir is not None

    @property
    def name(self) -> str:
        return demo.DEMO_NAME if self.request.skill_dir is None else self.request.skill_dir.name

    @property
    def live(self) -> bool:
        return self._live

    def status(self, ctx: SetupContext) -> StepStatus:
        from rote.setup import StepStatus

        # A try is a fresh run every time: nothing on disk says it is done.
        return StepStatus("pending", "runs fresh each time")

    def plan(self, ctx: SetupContext, prompter: Prompter) -> Plan:
        if not self._live:
            return self._precompiled_plan()
        return self._live_plan(prompter)

    # The precompiled proof: the bundled pipeline, live, no model calls.
    def _precompiled_plan(self) -> Plan:
        from rote.ir import load_pipeline

        reference = self.reference
        register, notes = _register_action(reference)
        pipeline_yaml = demo.pipeline_path()
        pipeline = load_pipeline(pipeline_yaml)
        _guard_model_calls(pipeline, approved=False)
        self.fields = tuple(FieldSpec.from_reference(f) for f in reference.fields)

        def run() -> MeasuredRun:
            from rote.adapters import get_adapter

            with tempfile.TemporaryDirectory(prefix="rote-try-") as tmp:
                app = Path(tmp) / "app"
                get_adapter(RUNTIME, extracted_source_dir=pipeline_yaml.parent).emit(pipeline, app)
                self.state.pipeline = self.runners.run_pipeline(app, reference.input, pipeline)
            return self.state.pipeline

        hosts = [_host(s.url) for s in reference.servers]
        action = Action(
            id="run-pipeline",
            title="Run the precompiled docs-digest pipeline on the recorded input",
            effects=(
                Launch(
                    "python",
                    f"the pipeline as a {RUNTIME} app on SQLite, in a temporary "
                    "directory removed afterwards",
                ),
                *_contacts(hosts, "the pipeline's MCP calls, read-only"),
            ),
            run=run,
            command=(
                f"rote emit <demo>/compiled/pipeline.yaml --runtime {RUNTIME} --out <app> && "
                "rote run <app> --input <demo>/skill/input.json"
            ),
        )
        actions = tuple(a for a in (register, action) if a is not None)
        return Plan(
            title=f"rote try {demo.DEMO_NAME}: the precompiled pipeline (no model calls)",
            actions=actions,
            notes=(
                *notes,
                f"compared with the skill run recorded on {reference.raw.get('recorded_on')}; "
                f"rote try {LIVE_FLAG} runs the skill and a compile fresh (paid)",
            ),
        )

    # The live path: skill, compile, pipeline, all fresh.
    def _live_plan(self, prompter: Prompter) -> Plan:
        from rote.app_registry import apps_path

        reference = self.reference
        skill_dir = self.request.skill_dir
        actions: list[Action] = []
        notes: list[str] = []
        if skill_dir is None:
            register, notes = _register_action(reference)
            if register is not None:
                actions.append(register)
            skill = demo.skill_dir()
            input_path = skill / "input.json"
            payload = reference.input
            hosts = [_host(s.url) for s in reference.servers]
            self.fields = tuple(FieldSpec.from_reference(f) for f in reference.fields)
        else:
            if not (skill_dir / "SKILL.md").is_file():
                raise TryError(f"{skill_dir} has no SKILL.md")
            skill = skill_dir
            input_path = _skill_input(skill_dir, self.request.input_path, prompter)
            payload = _read_input(input_path)
            # The skill run wires every registered server (rote run's rule),
            # and the compiled pipeline resolves its servers from the registry.
            hosts = _registered_hosts()
            notes.append("fields to compare come from the compiled IR (there is no reference)")
        estimates = estimate_spend(reference, skill_dir)
        out_dir = (self.request.out_dir or Path.cwd() / f"rote-try-{self.name}").resolve()

        skill_spend = _spend(estimates.skill_run)
        actions.append(
            Action(
                id="run-skill",
                title=f"Run the {self.name} skill as an agent, read-only",
                effects=(
                    Launch("claude", "claude -p runs the skill on the input", paid=True),
                    skill_spend,
                    Contact(ANTHROPIC_HOST, "the agent's model calls"),
                    *_contacts(hosts, "the agent's MCP calls, read-only"),
                ),
                run=lambda: self._run_skill(skill, payload),
                command=f"rote run {skill} --input {input_path}",
            )
        )
        compile_spend = _spend(estimates.compile)
        pipeline_yaml = out_dir / "compiled" / "pipeline.yaml"
        existed = pipeline_yaml.is_file()
        apps = apps_path()
        actions.append(
            Action(
                id="compile",
                title=f"Compile {self.name} for {RUNTIME}, grounded in the same input",
                effects=(
                    Launch("claude", "rote compile's compiler agent", paid=True),
                    compile_spend,
                    Contact(ANTHROPIC_HOST, "the compiler's model calls"),
                    *_contacts(hosts, "a read-only tools/list, to ground tool arguments"),
                    FileWrite(
                        pipeline_yaml,
                        f"the compiled pipeline, its code and the {RUNTIME} app under {out_dir}",
                        existed=existed,
                        replaces=existed,
                    ),
                    FileWrite(
                        apps,
                        "record the compiled app, so rote mcp login can find it",
                        existed=apps.is_file(),
                    ),
                ),
                run=lambda: self._compile(skill, input_path, out_dir),
                command="rote " + " ".join(compile_command(skill, input_path, out_dir)),
            )
        )
        pipeline_effects: list[Effect] = [
            Launch("python", f"the compiled pipeline as a {RUNTIME} app"),
            *_contacts(hosts, "the pipeline's MCP calls, read-only"),
        ]
        if estimates.pipeline_run is not None:
            self._pipeline_spend = _spend(estimates.pipeline_run)
            pipeline_effects.insert(1, self._pipeline_spend)
            pipeline_effects.insert(
                1, Contact(ANTHROPIC_HOST, "the pipeline's model calls, if any")
            )
        actions.append(
            Action(
                id="run-pipeline",
                title="Run the compiled pipeline on the same input",
                effects=tuple(pipeline_effects),
                run=lambda: self._run_compiled(payload),
                command=f"rote run {out_dir} --input {input_path}",
            )
        )
        spends = [e for a in actions for e in a.effects if isinstance(e, Spend)]
        low, high = _total(spends)
        notes.append(f"total estimate: {_money(low, high)}, billed to {SUBSCRIPTION}")
        notes.append(f"estimates from {estimates.basis}")
        return Plan(
            title=f"rote try {self.name} {LIVE_FLAG}: the skill, a compile, then the pipeline",
            actions=tuple(actions),
            notes=tuple(notes),
        )

    def _run_skill(self, skill: Path, payload: dict[str, Any]) -> MeasuredRun:
        self.state.skill = self.runners.run_skill(skill, payload)
        return self.state.skill

    def _compile(self, skill: Path, input_path: Path, out_dir: Path) -> CompileRun:
        self.state.compile = self.runners.compile(skill, input_path, out_dir)
        return self.state.compile

    def _run_compiled(self, payload: dict[str, Any]) -> MeasuredRun:
        from rote.ir import load_pipeline

        compiled = self.state.compile
        if compiled is None:
            raise RuntimeError("nothing was compiled")
        pipeline = load_pipeline(compiled.pipeline_yaml)
        _guard_model_calls(pipeline, approved=self._pipeline_spend is not None)
        if self.request.skill_dir is not None:
            self.fields = fields_from_ir(pipeline)
        self.state.pipeline = self.runners.run_pipeline(compiled.app_dir, payload, pipeline)
        return self.state.pipeline

    # ───────── The report ─────────

    def report(self) -> TryReport:
        """Compare what ran. Call after the plan applied."""
        pipeline = self.state.pipeline
        if pipeline is None:
            raise RuntimeError("the pipeline did not run")
        recorded = self.reference.raw["skill_run"]
        if self._live:
            skill = self.state.skill
            if skill is None:
                raise RuntimeError("the skill did not run")
            before = skill_metrics(skill, "skill, as an agent (this run)")
            skill_output: Any = skill.output
        else:
            before = SideMetrics(
                f"skill, as an agent (recorded {self.reference.raw.get('recorded_on')})",
                float(recorded["wall_seconds"]),
                int(recorded["turns"]),
                None,
                float(recorded["cost_usd"]),
            )
            skill_output = self.reference.skill_output
        after = pipeline_metrics(pipeline, "compiled pipeline (this run)")
        comparison = Comparison(compare_outputs(self.fields, skill_output, pipeline.output))
        return TryReport(self.name, self._live, before, after, self.state.compile, comparison)


@dataclass(frozen=True)
class TryReport:
    skill: str
    live: bool
    before: SideMetrics
    after: SideMetrics
    compile: CompileRun | None
    comparison: Comparison

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill": self.skill,
            "mode": "live" if self.live else "precompiled",
            "headline": self.comparison.headline,
            "same_result": self.comparison.same_result,
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
            "compile": None if self.compile is None else self.compile.to_dict(),
            "comparison": self.comparison.to_dict(),
        }


_STATUS_TEXT: dict[Status, str] = {
    "matched": "matched",
    "differs": "differs",
    "drifted": "drifted (live data)",
    "differs-judge": "differs-judge (a model's wording)",
}


def _seconds(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f} s"


def _cost(value: float | None) -> str:
    return "n/a" if value is None else f"${value:.3f}"


def _work(side: SideMetrics) -> str:
    if side.turns is not None:
        return f"{side.turns} turns"
    if side.model_calls is not None:
        return f"{side.model_calls} model calls"
    return "n/a"


def render_report(report: TryReport) -> list[str]:
    """The side-by-side for a person."""
    cmp = report.comparison
    lines = [f"rote try {report.skill}: {cmp.headline}", ""]
    rows = [
        ("", report.before.label, report.after.label),
        ("wall time", _seconds(report.before.wall_seconds), _seconds(report.after.wall_seconds)),
        ("turns", _work(report.before), _work(report.after)),
        ("cost", _cost(report.before.cost_usd), _cost(report.after.cost_usd)),
    ]
    first = max(len(r[0]) for r in rows)
    second = max(len(r[1]) for r in rows)
    for label, left, right in rows:
        lines.append(f"  {label.ljust(first)}  {left.ljust(second)}  {right}".rstrip())
    if report.compile is not None:
        c = report.compile
        turns = "n/a" if c.turns is None else f"{c.turns} turns"
        lines.append(
            f"  compile, once: {_seconds(c.wall_seconds)}, {turns}, {_cost(c.cost_usd)} "
            f"({c.out_dir})"
        )
    lines.append("")
    width = max((len(f.spec.name) for f in cmp.fields), default=0)
    for result in cmp.fields:
        line = f"  {result.spec.name.ljust(width)}  {_STATUS_TEXT[result.status]}"
        if result.problem:
            line += f": {result.problem}"
        elif result.status in ("drifted", "differs-judge"):
            line += f": {result.spec.why}"
        lines.append(line)
    lines.append("")
    if cmp.same_result:
        lines.append(f"Same result on all {cmp.compared} fields expected to match exactly.")
    elif cmp.compared == 0:
        lines.append("No field is expected to match exactly, so no same-result claim is made.")
    elif cmp.reproduced == cmp.compared:
        lines.append("Not the same result: a compared value is missing from one of the outputs.")
    else:
        lines.append(
            f"Not the same result: {cmp.compared - cmp.reproduced} of {cmp.compared} "
            "fields expected to match exactly did not."
        )
    return lines
