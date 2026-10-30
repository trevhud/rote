"""The paired eval: a source skill and its compiled skill, each run by an agent.

``rote eval <compiled> --run --runtime skill`` runs the original skill as a
``claude -p`` subagent, and the compiled skill (emitted with ``--runtime
skill``, installed into a throwaway state, driven through ``rote skill
run``) as another, on the same input, ``--trials`` times each. Then it puts
the two side by side. It is an eval, not a test: every trial costs tokens,
so nothing here runs in the automated suite except against a fake ``claude``.

What holds the comparison fair (docs/design/compiled-skill-target.md, §6-7):

- **One harness.** Both sides run under :func:`rote.eval.empirical.
  skill_run_isolation_args`, the same prompt builder and the same output
  contract. Both load Bash, because a real Claude Code session always has
  it: the original's is denied (``dontAsk`` and no rule), and the compiled
  side's only extra is the allow rule ``Bash(rote skill run <name> *)`` plus
  a read rule for its run folder, where the driver writes ``result.json``.
- **One person.** Both sides get the ``rote_gates`` server
  (:mod:`rote.eval.gate_server`), so a HUMAN block and the original's own
  review step get the case's fixed answers from the same simulated person.
- **One set of tools and data.** Fixture servers for a bench case (the same
  fixtures ``rote bench --fidelity`` uses), or servers registered read-only.
  The compiled side reaches them through a registry in its own throwaway
  state, because ``rote skill run`` honors nothing else.

What it checks, besides the numbers:

- **Fields** (:mod:`rote.proof`): original trial i against compiled trial i,
  field by field. The original against itself (every pair of its trials) is
  the noise floor.
- **Bypass**: a compiled trial with no finished run folder did the work
  itself, so it fails as "bypassed the driver".
- **Tiers**: every gate answer in the journal must say ``answered_by:
  "human"`` with the person's words, after an ``ask_person`` call; a gate
  the agent answered itself fails the trial. Every step the agent answers
  (a DECIDE judge, a DO step, a LOOP) must say ``answered_by: "agent"``, and
  the evidence counts them by block.
- **Misreport**: values the agent put in its result.json are checked against
  the run folder's result.json wherever both name the same field.
- **Context**: ``K = I - T*B`` per side, with ``B`` from one control run per
  side (the same flags, the prompt "reply ok").

Every trial's cost goes into a ledger, and the eval stops before a trial
whose estimate would cross the cap.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from rote import proof
from rote.eval.empirical import (
    GATE_INSTRUCTIONS,
    RESULT_FILENAME,
    EmpiricalError,
    _skill_prompt,
    default_corpus_path,
    failure_excerpt,
    gate_answers_server,
    permission_denials,
    skill_run_file_rules,
    skill_run_isolation_args,
    split_task,
)
from rote.eval.gate_server import GATE_SERVER
from rote.inference import build_subscription_env
from rote.ir import NodeKind, Pipeline
from rote.isolation import isolated_env

if TYPE_CHECKING:
    from rote import demo
    from rote.bench.fidelity import Case, CaseFile

Side = Literal["original", "compiled"]
SIDES: tuple[Side, Side] = ("original", "compiled")
Shape = Literal["compute-heavy", "tool-and-copy"]

#: The turn budget a trial gets: the earlier round's longest run took 7.
DEFAULT_MAX_TURNS = 40
#: A trial that outlives this is stopped and recorded as an error.
TRIAL_TIMEOUT_SECONDS = 900.0
#: What a trial is assumed to cost before one of its kind has run: the
#: earlier round's most expensive trial was $0.39 (expense-audit, original).
DEFAULT_TRIAL_ESTIMATE_USD = 0.40
#: A control run ("reply ok"): one request, mostly a cache write.
DEFAULT_CONTROL_ESTIMATE_USD = 0.10
#: A compile, when the eval compiles the skill itself: the top of the measured
#: range (CLAUDE.md: Sonnet 4.6 compiles ran $1.20 to $5.14), so a first compile
#: cannot overshoot --budget by a multiple.
DEFAULT_COMPILE_ESTIMATE_USD = 5.20
#: The control run's whole prompt.
CONTROL_PROMPT = "Reply with the single word ok. Do not use any tool."
#: The corpus record kind (the corpus is shared with ``rote eval --run``).
CORPUS_KIND = "skill_ab"


class SkillAbError(RuntimeError):
    """The eval cannot be set up for what it was given."""


class BudgetExhausted(RuntimeError):
    """The next paid run would cross the ledger's cap; nothing was started."""


# ───────── Budget ─────────


class Ledger:
    """Every paid run's cost, kept in one JSON file across invocations.

    A run whose cost is unknown is recorded at its estimate, so an unknown
    never reads as free."""

    def __init__(self, path: Path, cap_usd: float) -> None:
        self.path = path
        self.cap_usd = cap_usd
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            self.runs: list[dict[str, Any]] = list(data.get("runs") or [])
        else:
            self.runs = []

    @property
    def spent(self) -> float:
        return round(sum(float(r["cost_usd"]) for r in self.runs), 6)

    def estimate(self, kind: str, default: float) -> float:
        """The dearest run of this kind so far, else ``default``."""
        seen = [float(r["cost_usd"]) for r in self.runs if r.get("kind") == kind]
        return max(seen) if seen else default

    def check(self, kind: str, default: float) -> None:
        estimate = self.estimate(kind, default)
        if self.spent + estimate > self.cap_usd:
            raise BudgetExhausted(
                f"spent ${self.spent:.3f} of ${self.cap_usd:.2f}; the next run ({kind}) is "
                f"estimated at ${estimate:.2f}, which would cross the cap. Nothing was started"
            )

    def record(self, kind: str, label: str, cost_usd: float | None, default: float) -> None:
        estimated = cost_usd is None
        self.runs.append(
            {
                "kind": kind,
                "label": label,
                "cost_usd": self.estimate(kind, default) if estimated else cost_usd,
                "estimated": estimated,
                "at": datetime.now(UTC).isoformat(timespec="seconds"),
            }
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"cap_usd": self.cap_usd, "spent_usd": self.spent, "runs": self.runs}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
        os.replace(tmp, self.path)


# ───────── Usage, from the stream ─────────


@dataclass(frozen=True)
class Usage:
    """One ``claude -p`` run's tokens, cost and turns.

    Input is all three input buckets: a prompt-cached run reports almost
    nothing in ``input_tokens`` (CLAUDE.md, "A prompt-cached run reports
    almost nothing"). Totals come from the result envelope's ``modelUsage``
    (the whole query), else its ``usage``; per-request inputs come from the
    stream's assistant messages, counted once per ``message.id``, since
    Claude Code repeats a message's usage on every content block."""

    input: int
    cache_write: int
    cache_read: int
    #: None when no final usage arrived (stream placeholders are not output).
    output: int | None
    #: Model requests (distinct assistant message ids): ``T`` in §6's formula.
    requests: int
    #: Each request's input, all buckets, in order.
    per_request_input: tuple[int, ...]
    #: The CLI's ``num_turns``.
    num_turns: int | None
    #: ``total_cost_usd``: Claude Code's client-side estimate.
    cost_usd: float | None
    source: Literal["modelUsage", "usage", "stream"]
    #: The first request read the prompt cache: the run started warm.
    warm_start: bool
    #: Every tool the agent called, in order (``mcp__server__tool``, ``Bash``...).
    tools_used: tuple[str, ...] = ()

    @property
    def total_input(self) -> int:
        return self.input + self.cache_write + self.cache_read

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input,
            "cache_write_tokens": self.cache_write,
            "cache_read_tokens": self.cache_read,
            "total_input_tokens": self.total_input,
            "output_tokens": self.output,
            "requests": self.requests,
            "per_request_input": list(self.per_request_input),
            "num_turns": self.num_turns,
            "cost_usd": self.cost_usd,
            "usage_source": self.source,
            "cache": "warm" if self.warm_start else "cold",
            "tools_used": list(self.tools_used),
        }


def _int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _bucket_input(usage: Mapping[str, Any]) -> int:
    return (
        _int(usage.get("input_tokens"))
        + _int(usage.get("cache_creation_input_tokens"))
        + _int(usage.get("cache_read_input_tokens"))
    )


def parse_stream(lines: Sequence[str]) -> tuple[Usage, dict[str, Any]]:
    """``(usage, result envelope)`` from a ``--output-format stream-json`` run."""
    from rote.compiler.drivers.claude import _usage_delta

    seen: dict[str, Mapping[str, Any]] = {}
    order: list[str] = []
    tools: dict[str, str] = {}
    result: dict[str, Any] = {}
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") == "result":
            result = event
            continue
        message = event.get("message")
        if event.get("type") != "assistant" or not isinstance(message, dict):
            continue
        for block in message.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                tools.setdefault(str(block.get("id")), str(block.get("name")))
        message_id = message.get("id")
        usage = message.get("usage")
        if not isinstance(message_id, str) or not isinstance(usage, dict):
            continue
        if message_id not in seen:
            seen[message_id] = usage
            order.append(message_id)
    per_request = tuple(_bucket_input(seen[m]) for m in order)
    stream_totals = {"input": 0, "output": 0, "cache_write": 0, "cache_read": 0}
    for m in order:
        for key, value in _usage_delta({"usage": seen[m]}).items():
            stream_totals[key] += value
    totals = stream_totals
    source: Literal["modelUsage", "usage", "stream"] = "stream"
    output: int | None = None
    by_model = result.get("modelUsage")
    if isinstance(by_model, dict) and by_model:
        model_totals = {"input": 0, "output": 0, "cache_write": 0, "cache_read": 0}
        for entry in by_model.values():
            if not isinstance(entry, dict):
                continue
            model_totals["input"] += _int(entry.get("inputTokens"))
            model_totals["output"] += _int(entry.get("outputTokens"))
            model_totals["cache_write"] += _int(entry.get("cacheCreationInputTokens"))
            model_totals["cache_read"] += _int(entry.get("cacheReadInputTokens"))
        totals, source, output = model_totals, "modelUsage", model_totals["output"]
    elif isinstance(result.get("usage"), dict):
        totals = _usage_delta({"usage": result["usage"]})
        source, output = "usage", totals["output"]
    first = seen[order[0]] if order else {}
    cost = result.get("total_cost_usd")
    turns = result.get("num_turns")
    return (
        Usage(
            input=totals["input"],
            cache_write=totals["cache_write"],
            cache_read=totals["cache_read"],
            output=output,
            requests=len(order),
            per_request_input=per_request,
            num_turns=turns if isinstance(turns, int) else None,
            cost_usd=float(cost) if isinstance(cost, (int, float)) else None,
            source=source,
            warm_start=_int(first.get("cache_read_input_tokens")) > 0,
            tools_used=tuple(tools.values()),
        ),
        result,
    )


# ───────── What a side runs on ─────────


@dataclass(frozen=True)
class Subject:
    """One skill, its compiled pipeline, and the input both sides run on."""

    name: str
    skill_dir: Path
    pipeline_yaml: Path
    pipeline: Pipeline
    input: dict[str, Any]
    #: What the simulated person answers at each gate, in order.
    gate_answers: tuple[Any, ...]
    #: The result.json fields both agents are told to write.
    contract: tuple[str, ...]
    #: Text added to both prompts (the bench case's contract and gate rule).
    instructions: str | None
    #: Text added to the original's prompt only (use the connected tools).
    original_instructions: str | None
    fields: tuple[proof.FieldSpec, ...]
    shape: Shape
    shape_why: str
    #: A bench case: fixture servers and a recorded reference to score against.
    casefile: CaseFile | None = None
    case: Case | None = None
    #: The case is a replay of a recorded run (:mod:`rote.replay`): writes are
    #: captured, and a call with no recorded response makes a trial incomplete.
    replay: bool = False
    #: The bundled demo's reference (docs-digest), scored against the same way.
    reference: demo.Reference | None = None
    #: Live servers both sides call: ``{name: url}``, registered read-only,
    #: or gated per tool (``gated``).
    live_servers: Mapping[str, str] = field(default_factory=dict)
    #: The live servers not registered read-only, whose own ``readOnlyHint``
    #: annotations decide which tools both sides may call.
    gated: Mapping[str, GatedServer] = field(default_factory=dict)
    #: How the pipeline came to be (the compile argv, or "given").
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return self.name if self.case is None else f"{self.name}/{self.case.id}"


def shape_of(pipeline: Pipeline) -> tuple[Shape, str]:
    """compute-heavy when the pipeline calls no tool (all its data is in the
    input, and the original agent works the result out in its own output);
    tool-and-copy when it reads tools and mostly passes their results on.
    §6 measured the two shapes behaving differently, so results are grouped
    by it. Decided from the IR, never from the skill's words."""
    reads = [n.id for n in pipeline.nodes if n.mcp is not None]
    if reads:
        return "tool-and-copy", f"{len(reads)} MCP read(s): {', '.join(reads)}"
    return "compute-heavy", "no MCP reads: every value is computed from the input"


def _gate_order(pipeline: Pipeline) -> list[Any]:
    return [n for wave in pipeline.execution_waves() for n in wave if n.kind is NodeKind.HITL_GATE]


def gate_answers_from_signals(pipeline: Pipeline, signals: Mapping[str, Any]) -> tuple[Any, ...]:
    """An ``--input`` envelope's ``signals``, in the order the gates are reached."""
    answers: list[Any] = []
    for gate in _gate_order(pipeline):
        if gate.signal not in signals:
            raise SkillAbError(
                f"gate {gate.id} waits for signal {gate.signal!r}, and the --input envelope "
                'gives no answer for it: add it under "signals"'
            )
        answers.append(signals[gate.signal])
    return tuple(answers)


def _skill_name(skill_dir: Path) -> str:
    from rote.adapters.skill import read_identity

    return read_identity(skill_dir).name


@dataclass(frozen=True)
class GatedServer:
    """A live server the user registered but not read-only: both sides may
    call only the tools it marks ``readOnlyHint`` itself."""

    url: str
    #: ``mcp__<server>__<tool>`` ids of its read-only tools, and of the rest.
    allowed: tuple[str, ...]
    denied: tuple[str, ...]
    #: The user's login for it, copied into each trial's throwaway state.
    token: Path | None


def _bound_tools(pipeline: Pipeline) -> dict[str, set[str]]:
    """``{server: tools}`` the pipeline calls: ``mcp:`` bindings and the
    tools an agent loop reaches through ``tool_servers``."""
    out: dict[str, set[str]] = {}
    for node in pipeline.nodes:
        if node.mcp is not None:
            out.setdefault(node.mcp.server, set()).add(node.mcp.tool)
        for tool, server in (node.tool_servers or {}).items():
            out.setdefault(server, set()).add(tool)
    return out


def live_servers(
    servers: Sequence[str], bound: Mapping[str, set[str]] | None = None
) -> tuple[dict[str, str], dict[str, GatedServer]]:
    """``({server: url}, {server: GatedServer})`` for ``servers``, from the
    user's registry.

    A server registered read-only is called as is. One registered without
    it is gated per tool, by the server's own ``readOnlyHint`` annotations:
    every tool in ``bound`` (``{server: tools}`` the pipeline calls) must be
    one it marks read-only, or the eval stops before spending anything and
    names the tools that may write. With ``bound`` unknown (``rote try``,
    before the compile), the server needs at least one read-only tool.
    The bundled demo's servers (public, read-only, the ones ``rote try``
    registers) count as registered."""
    import asyncio

    from rote import demo
    from rote.eval.baseline import read_only_allowlist
    from rote.mcp import read_token_file, token_file
    from rote.mcp.live_tools import mcp_tool_id
    from rote.mcp.registry import load_registry

    registry = load_registry()
    published = {s.name: s.url for s in demo.load_reference().servers}
    found: dict[str, str] = {}
    gated: dict[str, GatedServer] = {}
    for server in servers:
        entry = registry.servers.get(server)
        if entry is not None and entry.read_only is True:
            found[server] = entry.url
        elif server in published and entry is None:
            found[server] = published[server]
        elif entry is not None:
            allowed, denied, skipped = asyncio.run(read_only_allowlist([server]))
            if server in skipped:
                raise SkillAbError(
                    f"the pipeline reads {server}, which is not registered read-only, and its "
                    f"tools could not be checked: {skipped[server]}. If none of its tools can "
                    f"change anything, say so: rote setup --read-only {server}"
                )
            writes = sorted(
                t for t in (bound or {}).get(server, set()) if mcp_tool_id(server, t) not in allowed
            )
            if writes:
                raise SkillAbError(
                    f"the pipeline calls {', '.join(writes)} on {server}, which {server} does "
                    "not mark read-only, and the eval only reads. If none of its tools can "
                    f"change anything, say so: rote setup --read-only {server}"
                )
            path = token_file(server)
            gated[server] = GatedServer(
                url=entry.url,
                allowed=tuple(allowed),
                denied=tuple(denied),
                token=path if read_token_file(server) is not None else None,
            )
            found[server] = entry.url
        else:
            raise SkillAbError(
                f"the pipeline reads {server}, and the eval only calls servers registered "
                f"with rote: rote mcp add {server} <url> (or use a bench case, whose "
                "fixtures stand in for it)"
            )
    return found, gated


def subject_for(
    pipeline_yaml: Path,
    *,
    skill_dir: Path,
    task: Any = None,
    case_id: str | None = None,
    provenance: Mapping[str, Any] | None = None,
    cases_dir: Path | None = None,
) -> Subject:
    """What both sides run on: a bench case (``case_id``), or ``task`` (an
    input payload, or the ``{"input", "signals"}`` envelope ``rote eval --run``
    takes)."""
    from rote import demo
    from rote.bench.fidelity import agent_instructions, map_servers
    from rote.ir import load_pipeline

    pipeline = load_pipeline(pipeline_yaml)
    shape, shape_why = shape_of(pipeline)
    casefile: CaseFile | None = None
    case: Case | None = None
    reference: demo.Reference | None = None
    live: dict[str, str] = {}
    gated: dict[str, GatedServer] = {}
    original_only: str | None = None
    if case_id is not None:
        casefile, case = bench_case(skill_dir, case_id, cases_dir)
        _mapped, unknown = map_servers(pipeline, casefile)
        if unknown:
            raise SkillAbError(
                f"the pipeline reads {', '.join(unknown)}, which no fixture of {case_id} serves"
            )
        payload = case.input
        answers: tuple[Any, ...] = tuple(case.gates)
        contract: tuple[str, ...] = (*casefile.output_fields, "outcome")
        full = agent_instructions(casefile, case)
        common = agent_instructions(casefile, case, tools_clause=False)
        original_only = full[len(common) :].strip() or None
        shared: str | None = common
        fields = proof.contract_fields(pipeline, contract)
    else:
        payload, signals = split_task(pipeline, task)
        if not isinstance(payload, dict):
            raise SkillAbError("--input must hold a JSON object (or pass a bench --case)")
        answers = gate_answers_from_signals(pipeline, signals)
        shared = GATE_INSTRUCTIONS if answers else None
        # The demo's reference knows which of its fields are live data
        # (drifts), whether its pipeline is the recorded one or a new compile.
        if (
            pipeline_yaml.resolve() == demo.pipeline_path().resolve()
            or skill_dir.resolve() == demo.skill_dir().resolve()
        ):
            reference = demo.load_reference()
            fields = tuple(
                proof.FieldSpec.from_reference(f, same_shape=True) for f in reference.fields
            )
        else:
            # Both agents write the skill's shape, so both are read at its pointer.
            fields = tuple(
                proof.FieldSpec(f.name, f.original, f.original, None, None, f.expect, f.why)
                for f in proof.fields_from_ir(pipeline)
            )
        contract = tuple(dict.fromkeys(f.original.split("/")[1] for f in fields))
        live, gated = live_servers(sorted(pipeline.required_mcp_servers), _bound_tools(pipeline))
    return Subject(
        name=_skill_name(skill_dir),
        skill_dir=skill_dir.resolve(),
        pipeline_yaml=pipeline_yaml.resolve(),
        pipeline=pipeline,
        input=payload,
        gate_answers=answers,
        contract=contract,
        instructions=shared,
        original_instructions=original_only,
        fields=fields,
        shape=shape,
        shape_why=shape_why,
        casefile=casefile,
        case=case,
        reference=reference,
        live_servers=live,
        gated=gated,
        provenance=dict(provenance or given_provenance(pipeline_yaml)),
        replay=cases_dir is not None,
    )


def given_provenance(pipeline_yaml: Path) -> dict[str, Any]:
    """What is known about how a given pipeline was made.

    A compile records its System-1 settings in ``system1/decisions.json``
    (the decider that checked the structure, the structure pass's model,
    the tie-break) and its cost in ``compile-metrics.json``; both are copied
    when present. A pipeline with neither was written by hand (an example's
    ``expected/``), and the record says so rather than calling it a compile."""
    from rote.bench.collect import read_compile_metrics

    compiled = pipeline_yaml.parent
    record: dict[str, Any] = {"pipeline": "given"}
    decisions_path = compiled / "system1" / "decisions.json"
    metrics = read_compile_metrics(compiled)
    if not decisions_path.is_file() and not metrics:
        record["pipeline"] = "given, with no compile record (hand-written)"
        return record
    if decisions_path.is_file():
        decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
        checker = decisions.get("checker")
        if isinstance(checker, dict):
            record["decider"] = checker.get("decider")
        structure = decisions.get("structure")
        if isinstance(structure, dict):
            record["structure"] = structure.get("model")
        tiebreak = decisions.get("tiebreak")
        record["tiebreak"] = tiebreak.get("model") if isinstance(tiebreak, dict) else "none"
    if metrics:
        record["compile_cost_usd"] = metrics.get("cost_usd")
        record["compile_turns"] = metrics.get("num_turns")
    return record


def bench_case(
    skill_dir: Path, case_id: str, cases_dir: Path | None = None
) -> tuple[CaseFile, Case]:
    """A bench case from the example the skill sits in (``<example>/skill``),
    or from a replay dir (``cases_dir``, :mod:`rote.replay`).

    An example's case is read-only. A replay may hold write tools: its
    fixture servers log the write each side proposes and answer it with an
    error, never forwarding it (:mod:`rote.bench.fixture_mcp`)."""
    from rote.bench.fidelity import load_cases

    example_dir = cases_dir.resolve() if cases_dir is not None else skill_dir.resolve().parent
    try:
        casefile = load_cases(example_dir)
    except FileNotFoundError:
        raise SkillAbError(f"--case {case_id}: {example_dir} has no fidelity/cases.yaml") from None
    case = next((c for c in casefile.cases if c.id == case_id), None)
    if case is None:
        ids = ", ".join(c.id for c in casefile.cases)
        raise SkillAbError(f"--case {case_id}: no such case in {example_dir} (it has: {ids})")
    if cases_dir is None:
        for server in casefile.servers.values():
            writes = [t for t, spec in server["tools"].items() if spec.get("side_effect")]
            if writes:
                raise SkillAbError(
                    f"--case {case_id}: fixture tools {', '.join(writes)} write, and the eval "
                    "is read-only"
                )
    return casefile, case


# ───────── Driver evidence (the compiled side) ─────────


@dataclass(frozen=True)
class DriverEvidence:
    """What the compiled skill's run folder says happened."""

    #: Run folders the trial left (exactly one is expected).
    runs: int
    run_id: str | None
    state: str | None
    #: The run folder's result.json (the exit nodes' results).
    result: Any
    #: Gates answered, each ``{step, answered_by, said, matches_person}``.
    gates: tuple[dict[str, Any], ...]
    #: Fields the agent's result.json shares with the driver's, compared.
    reported_checked: int
    problems: tuple[str, ...]
    #: Steps the agent answered, by block (``{"DECIDE": 1, "DO": 2}``).
    agent_steps: Mapping[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "runs": self.runs,
            "run": self.run_id,
            "state": self.state,
            "gates": list(self.gates),
            "agent_steps": dict(self.agent_steps),
            "reported_checked": self.reported_checked,
            "problems": list(self.problems),
        }


def driver_evidence(
    run_root: Path,
    *,
    skill: str,
    plan: Mapping[str, Any],
    expected_input: Mapping[str, Any],
    agent_output: Any,
    asks: int,
    gate_answers: Sequence[Any],
) -> DriverEvidence:
    """Check the compiled trial's run folder: it ran, on this input, through
    the driver; every gate was answered by the person; and the agent reported
    what the driver computed.

    ``asks`` is how many ``ask_person`` calls the trial made."""
    problems: list[str] = []
    folders = sorted(p.parent for p in (run_root / skill).glob("*/run.json"))
    if not folders:
        return DriverEvidence(0, None, None, None, (), 0, ("bypassed the driver: no run folder",))
    if len(folders) > 1:
        problems.append(f"{len(folders)} runs started; the last one is checked")
    folder = max(folders, key=lambda p: p.stat().st_mtime)
    journal = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    state = journal.get("state")
    if state != "done":
        problems.append(f"the run ended {state}, not done")
    if proof.canonical(journal.get("input")) != proof.canonical(dict(expected_input)):
        problems.append("the agent started the run on a different input than it was given")
    result_path = folder / RESULT_FILENAME
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else None
    if result is None and state == "done":
        problems.append("the run is done but has no result.json")

    gates: list[dict[str, Any]] = []
    gate_steps = [s["id"] for s in plan.get("steps") or [] if s.get("kind") == "gate"]
    for index, step in enumerate(gate_steps):
        node = folder / "nodes" / f"{step}.json"
        if not node.is_file():
            if state == "done":
                problems.append(f"gate {step} has no journal entry")
            continue
        record = json.loads(node.read_text(encoding="utf-8"))
        said = record.get("said")
        by = record.get("answered_by")
        person = gate_answers[index] if index < len(gate_answers) else None
        recorded = record.get("result")
        matches = (
            isinstance(recorded, Mapping)
            and isinstance(person, Mapping)
            and all(
                proof.canonical(recorded[k]) == proof.canonical(person.get(k)) for k in recorded
            )
        )
        gates.append({"step": step, "answered_by": by, "said": said, "matches_person": matches})
        if by != "human" or not isinstance(said, str) or not said.strip():
            problems.append(f"gate {step} was not answered by a person (answered_by={by!r})")
        elif not matches:
            problems.append(f"gate {step}'s recorded answer is not what the person said")
    from rote.skill_target._driver import AGENT_BLOCKS

    agent_steps: dict[str, int] = {}
    for step in plan.get("steps") or []:
        block = AGENT_BLOCKS.get(str(step.get("kind")))
        if block is None or (step.get("kind") == "judge" and plan.get("judges") == "helper"):
            continue
        node = folder / "nodes" / f"{step['id']}.json"
        if not node.is_file():
            continue
        by = json.loads(node.read_text(encoding="utf-8")).get("answered_by")
        if by != "agent":
            problems.append(f"{block} step {step['id']} was not answered by the agent ({by!r})")
            continue
        agent_steps[block] = agent_steps.get(block, 0) + 1
    if gates and asks < len(gates):
        problems.append(
            f"{len(gates)} gate(s) answered after {asks} ask_person call(s): the agent "
            "answered a gate without asking"
        )

    checked = 0
    if isinstance(agent_output, Mapping) and isinstance(result, Mapping):
        driver_values: dict[str, Any] = {}
        for node_result in result.values():
            if isinstance(node_result, Mapping):
                for key, value in node_result.items():
                    driver_values.setdefault(str(key), value)
        for key, value in agent_output.items():
            if key in driver_values:
                checked += 1
                if proof.canonical(value) != proof.canonical(driver_values[key]):
                    problems.append(
                        f"misreported {key}: the agent wrote {json.dumps(value)[:80]}, the "
                        f"driver's result has {json.dumps(driver_values[key])[:80]}"
                    )
    return DriverEvidence(
        len(folders),
        folder.name,
        state if isinstance(state, str) else None,
        result,
        tuple(gates),
        checked,
        tuple(problems),
        agent_steps,
    )


# ───────── One trial ─────────


@dataclass(frozen=True)
class ProcessResult:
    lines: tuple[str, ...]
    stderr: str
    returncode: int | None
    wall_seconds: float
    timed_out: bool = False


#: Runs ``claude``: ``(argv, cwd, env, timeout) -> ProcessResult``. Injected by tests.
ClaudeRunner = Callable[[list[str], Path, dict[str, str], float], ProcessResult]


def run_claude(argv: list[str], cwd: Path, env: dict[str, str], timeout: float) -> ProcessResult:
    started = time.monotonic()
    proc = subprocess.Popen(
        argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        timed_out = False
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        timed_out = True
    return ProcessResult(
        tuple(stdout.splitlines()), stderr, proc.returncode, time.monotonic() - started, timed_out
    )


@dataclass(frozen=True)
class Trial:
    side: Side
    index: int
    #: True for a control run ("reply ok"): measures B, does no work.
    control: bool
    wall_seconds: float
    usage: Usage
    output: Any
    error: str | None
    denied_tools: tuple[str, ...]
    #: ``server.tool`` for every tool call the fixture or gate servers logged.
    calls: tuple[str, ...]
    #: The calls as the bench scores them (fixture cases only).
    portable_calls: tuple[dict[str, Any], ...]
    driver: DriverEvidence | None
    work_dir: Path
    #: Writes the trial proposed, in order: ``{server, tool, arguments,
    #: answered}``. A fixture server never forwards one.
    writes: tuple[dict[str, Any], ...] = ()
    #: Replay only: what the recording could not answer, so the trial did
    #: not run as recorded from that call on.
    incomplete: tuple[str, ...] = ()

    @property
    def problems(self) -> tuple[str, ...]:
        found = list(self.driver.problems) if self.driver is not None else []
        if self.error:
            found.insert(0, self.error)
        found += [f"incomplete: {why}" for why in self.incomplete]
        return tuple(found)

    @property
    def ok(self) -> bool:
        return not self.problems

    def to_dict(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "trial": self.index,
            "control": self.control,
            "wall_seconds": round(self.wall_seconds, 1),
            **self.usage.to_dict(),
            "error": self.error,
            "denied_tools": list(self.denied_tools),
            "calls": list(self.calls),
            "writes": list(self.writes),
            "incomplete": list(self.incomplete),
            "driver": None if self.driver is None else self.driver.to_dict(),
            "problems": list(self.problems),
            "output": self.output,
            "work_dir": str(self.work_dir),
        }


def _shim_dir(work: Path) -> Path:
    """A PATH entry holding only ``rote``: this interpreter's rote, so the
    compiled skill's ``rote skill run`` is the one under test."""
    bin_dir = work / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    shim = bin_dir / "rote"
    shim.write_text(f'#!/bin/sh\nexec "{sys.executable}" -I -m rote "$@"\n', encoding="utf-8")
    shim.chmod(0o755)
    return bin_dir


def _trial_env(state: Path, shim: Path) -> dict[str, str]:
    """The subscription env (API keys scrubbed), every rote state path in
    ``state``, ``rote`` from ``shim`` first on PATH, and no endpoint override
    (``rote skill run`` drops those anyway; the original never needs one)."""
    env = isolated_env(state, build_subscription_env())
    for key in [k for k in env if k.startswith("ROTE_MCP_") and k.endswith("_URL")]:
        del env[key]
    env["PATH"] = f"{shim}{os.pathsep}{env.get('PATH', '')}"
    return env


@contextlib.contextmanager
def _environ(env: Mapping[str, str]) -> Iterator[None]:
    saved = dict(os.environ)
    os.environ.clear()
    os.environ.update(env)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


def _mcp_entry(name: str, url: str, gated: Mapping[str, GatedServer]) -> dict[str, Any]:
    """The original side's ``--mcp-config`` entry for one server. A gated
    server with a login authenticates through ``rote mcp headers``, which
    reads the trial's copy of it, so no token lands in a file the run sees."""
    import shlex

    entry: dict[str, Any] = {"type": "http", "url": url}
    if name in gated and gated[name].token is not None:
        entry["headersHelper"] = (
            f"{shlex.quote(sys.executable)} -m rote mcp headers {shlex.quote(name)}"
        )
    return entry


def _copy_logins(gated: Mapping[str, GatedServer], env: Mapping[str, str]) -> None:
    """Copy each gated server's login into the trial's token dir, owner-only.
    The trial state is a throwaway dir removed with the work dir."""
    tokens = Path(env["ROTE_MCP_TOKEN_DIR"])
    for server in gated.values():
        if server.token is None:
            continue
        tokens.mkdir(parents=True, exist_ok=True, mode=0o700)
        dest = tokens / server.token.name
        dest.write_bytes(server.token.read_bytes())
        dest.chmod(0o600)


def install_compiled(
    skill_dir: Path,
    env: Mapping[str, str],
    servers: Mapping[str, str],
    gated: Collection[str] = (),
) -> None:
    """Register ``servers`` and install the compiled skill, in the trial's own
    state (``env``'s rote paths, a dir this eval created).

    Each server is registered read-only, except the ``gated`` ones: those
    stay unmarked, so the driver calls only the tools each one marks
    ``readOnlyHint`` itself.

    This is ``rote skill install``'s plan, applied without its person check.
    That check guards the user's registry against an agent widening trust
    there; here the registry, the servers in it (local fixtures, ones the
    user registered read-only, or gated ones) and the install record all live
    in a throwaway dir, and nothing of the user's is touched."""
    from rote.interact.plan import apply_plan
    from rote.mcp.registry import McpRegistry, McpServerConfig, save_registry
    from rote.skill_target import commands, trust

    state_paths = {env[k] for k in ("ROTE_APPS_PATH", "ROTE_MCP_CONFIG")}
    with _environ(env):
        if {os.environ["ROTE_APPS_PATH"], os.environ["ROTE_MCP_CONFIG"]} != state_paths:
            raise SkillAbError("the trial state was not applied")
        save_registry(
            McpRegistry(
                servers={
                    n: McpServerConfig(url=u, read_only=n not in gated) for n, u in servers.items()
                }
            )
        )
        skill = trust.read_skill_dir(skill_dir)
        now = trust.scan(skill.root)
        found = trust.against_manifest(skill, now)
        if not found.clean:
            raise SkillAbError(f"{skill.root} is not what rote emitted ({found.describe()})")
        read, apps = commands._registry()
        plan, _why = commands.install_plan(skill, read, apps, now)
        failed = [o for o in apply_plan(plan) if o.status != "done"]
        if failed:
            raise SkillAbError(f"installing {skill.name} failed: {failed[0].error}")


def emit_compiled(subject: Subject, out: Path) -> Path:
    """The compiled skill, emitted with ``--runtime skill`` into ``out/<name>``."""
    from rote.adapters.skill import SkillAdapter, SkillAdapterConfig

    target = out / subject.name
    if target.exists():
        shutil.rmtree(target)
    adapter = SkillAdapter(
        SkillAdapterConfig(
            skill_dir=subject.skill_dir, extracted_source_dir=subject.pipeline_yaml.parent
        )
    )
    adapter.emit(subject.pipeline, target)
    return target


@dataclass(frozen=True)
class Harness:
    """Where trials run and how: shared by every trial of one eval."""

    work: Path
    model: str
    ledger: Ledger
    executable: str = "claude"
    max_turns: int = DEFAULT_MAX_TURNS
    timeout_seconds: float = TRIAL_TIMEOUT_SECONDS
    runner: ClaudeRunner = run_claude
    log: Callable[[str], None] = lambda line: print(line, file=sys.stderr)
    #: What a trial and a control run are assumed to cost before one of
    #: their kind has run: the ledger checks these against its cap.
    trial_estimate_usd: float = DEFAULT_TRIAL_ESTIMATE_USD
    control_estimate_usd: float = DEFAULT_CONTROL_ESTIMATE_USD
    #: The command the log lines are prefixed with.
    caller: str = "rote eval"


@contextlib.contextmanager
def _servers(subject: Subject, side: Side, work: Path) -> Iterator[tuple[dict[str, str], Path]]:
    """``({server name as this side calls it: url}, call log)`` for one trial.

    The original calls a bench case's fixtures by their own names; the
    compiled side by the names its pipeline binds (``map_servers``)."""
    log = work / "calls" / "calls.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.touch()
    if subject.casefile is None or subject.case is None or not subject.casefile.servers:
        yield dict(subject.live_servers), log
        return
    from rote.bench.fidelity import fixture_servers, map_servers

    with fixture_servers(subject.casefile, subject.case, log.parent) as (urls, fixture_log):
        if side == "original":
            yield dict(urls), fixture_log
        else:
            mapped, _unknown = map_servers(subject.pipeline, subject.casefile)
            yield {name: urls[target] for name, target in mapped.items()}, fixture_log


def _read_calls(log: Path) -> list[dict[str, Any]]:
    if not log.is_file():
        return []
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line]


def run_trial(
    harness: Harness,
    subject: Subject,
    side: Side,
    index: int,
    *,
    compiled_dir: Path | None,
    control: bool = False,
) -> Trial:
    """One side, once (or its control run). The ledger is checked first."""
    kind = f"{subject.name}:{side}:{'control' if control else 'trial'}"
    default = harness.control_estimate_usd if control else harness.trial_estimate_usd
    harness.ledger.check(kind, default)
    label = f"{subject.label} {side} {'control' if control else f'#{index}'}"
    work = Path(tempfile.mkdtemp(prefix=f"{subject.name}-{side}-", dir=harness.work))
    cwd = work / "cwd"
    cwd.mkdir()
    state = work / "state"
    env = _trial_env(state, _shim_dir(harness.work))
    _copy_logins(subject.gated, env)
    skill_dir = subject.skill_dir if side == "original" else compiled_dir
    if skill_dir is None:
        raise SkillAbError("the compiled side needs the compiled skill's directory")
    with _servers(subject, side, work) as (servers, log):
        mcp: dict[str, dict[str, Any]] = {}
        allowed: list[str] = []
        denied: list[str] = []
        if side == "original":
            mcp = {n: _mcp_entry(n, u, subject.gated) for n, u in servers.items()}
            for n in sorted(servers):
                if n in subject.gated:
                    allowed += subject.gated[n].allowed
                    denied += subject.gated[n].denied
                else:
                    allowed.append(f"mcp__{n}__*")
        else:
            install_compiled(skill_dir, env, servers, gated=set(subject.gated))
            run_root = Path(env["ROTE_SKILL_RUN_DIR"])
            run_root.mkdir(parents=True, exist_ok=True)
            # The driver writes result.json into the run folder; the agent may
            # read it there (skill_run_file_rules' second rule is that dir's).
            allowed = [
                f"Bash(rote skill run {subject.name} *)",
                skill_run_file_rules(run_root)[1],
            ]
        if subject.gate_answers:
            mcp[GATE_SERVER], gate_tool = gate_answers_server(
                list(subject.gate_answers), log.parent
            )
            allowed.append(gate_tool)
        if control:
            prompt = CONTROL_PROMPT
        else:
            extra = "\n\n".join(
                part
                for part in (
                    subject.instructions,
                    subject.original_instructions if side == "original" else None,
                )
                if part
            )
            prompt = _skill_prompt(skill_dir, subject.input, list(subject.contract), extra or None)
        argv = [
            harness.executable,
            "-p",
            prompt,
            "--model",
            harness.model,
            "--add-dir",
            str(skill_dir),
            *skill_run_isolation_args(
                work, skill_dir, mcp, allowed, denied_mcp_tools=denied, extra_tools=("Bash",)
            ),
            "--output-format",
            "stream-json",
            "--verbose",
            "--max-turns",
            str(1 if control else harness.max_turns),
        ]
        (work / "argv.json").write_text(json.dumps(argv[3:], indent=1), encoding="utf-8")
        harness.log(
            f"{harness.caller}: {label}: running (spent ${harness.ledger.spent:.3f} so far)"
        )
        proc = harness.runner(argv, cwd, env, harness.timeout_seconds)
        calls = _read_calls(log)
    (work / "stream.jsonl").write_text("\n".join(proc.lines) + "\n", encoding="utf-8")
    (work / "stderr.txt").write_text(proc.stderr, encoding="utf-8")
    usage, envelope = parse_stream(proc.lines)
    harness.ledger.record(kind, label, usage.cost_usd, default)

    output: Any = None
    error: str | None = None
    result_path = cwd / RESULT_FILENAME
    if result_path.is_file():
        try:
            output = json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            error = f"{RESULT_FILENAME} is not valid JSON: {e}"
    if proc.timed_out:
        error = f"timed out after {harness.timeout_seconds:g}s"
    elif envelope.get("is_error"):
        error = f"claude reported an error: {str(envelope.get('result'))[:200]}"
    elif not control and output is None and error is None:
        error = (
            f"the agent wrote no {RESULT_FILENAME}"
            if proc.returncode == 0
            else f"claude exited {proc.returncode}: {failure_excerpt(proc.stderr)}"
        )
    driver = None
    if side == "compiled" and not control:
        plan = json.loads((skill_dir / "scripts" / "plan.json").read_text(encoding="utf-8"))
        driver = driver_evidence(
            Path(env["ROTE_SKILL_RUN_DIR"]),
            skill=subject.name,
            plan=plan,
            expected_input=subject.input,
            agent_output=output,
            asks=sum(1 for c in calls if c.get("server") == GATE_SERVER),
            gate_answers=subject.gate_answers,
        )
    from rote.bench.fidelity import _portable_call

    writes = tuple(
        {
            "server": c.get("server"),
            "tool": c.get("tool"),
            "arguments": c.get("arguments"),
            "answered": bool(c.get("matched")),
        }
        for c in calls
        if c.get("side_effect")
    )
    incomplete: tuple[str, ...] = ()
    if subject.replay:
        incomplete = tuple(
            dict.fromkeys(
                (
                    f"{c.get('server')}.{c.get('tool')} proposed a write the recording has no "
                    "response for"
                    if c.get("side_effect")
                    else f"{c.get('server')}.{c.get('tool')} was called with arguments the "
                    "recording never saw"
                )
                for c in calls
                if not c.get("matched")
            )
        )
    trial = Trial(
        side=side,
        index=index,
        control=control,
        wall_seconds=proc.wall_seconds,
        usage=usage,
        output=output,
        error=error,
        denied_tools=permission_denials(envelope),
        calls=tuple(f"{c.get('server')}.{c.get('tool')}" for c in calls),
        portable_calls=tuple(_portable_call(c) for c in calls),
        driver=driver,
        work_dir=work,
        writes=writes,
        incomplete=incomplete,
    )
    cost = "n/a" if usage.cost_usd is None else f"${usage.cost_usd:.3f}"
    status = "ok" if trial.ok else "; ".join(trial.problems)[:160]
    harness.log(
        f"{harness.caller}: {label}: {status}, {proc.wall_seconds:.1f} s, "
        f"{usage.requests} request(s), {cost} (spent ${harness.ledger.spent:.3f})"
    )
    return trial


# ───────── Scoring and the report ─────────


def bench_score(subject: Subject, trial: Trial) -> float | None:
    """The trial's ``rote bench --fidelity`` score against the case's
    committed reference, the number the earlier round reported. A replay has
    no committed reference: the recording it replays is its reference, and
    the pairs compare both sides directly."""
    if (
        subject.replay
        or subject.casefile is None
        or subject.case is None
        or not isinstance(trial.output, dict)
    ):
        return None
    from rote.bench.fidelity import load_reference
    from rote.bench.fidelity_score import score_case, stable_reference

    reference = load_reference(subject.casefile)
    runs = reference["cases"][subject.case.id]["runs"]
    stable = stable_reference(runs, contract=list(subject.contract))
    trace = {
        "completed": trial.output.get("outcome") == "completed",
        "objects": [trial.output],
        "calls": list(trial.portable_calls),
    }
    score = score_case(stable, trace).get("score")
    return float(score) if isinstance(score, (int, float)) else None


def vs_reference(subject: Subject, trial: Trial) -> proof.Comparison | None:
    """The bundled demo's recorded skill output against this trial's."""
    if subject.reference is None:
        return None
    return proof.Comparison(
        proof.compare_outputs(subject.fields, subject.reference.skill_output, trial.output)
    )


def _median(values: Sequence[float | int | None]) -> float | None:
    present = [float(v) for v in values if v is not None]
    return statistics.median(present) if present else None


@dataclass(frozen=True)
class SideSummary:
    side: Side
    trials: tuple[Trial, ...]
    #: ``B``: the control run's one request's input, all buckets.
    base_per_request: int | None
    bench_scores: tuple[float | None, ...]

    @property
    def ok_trials(self) -> tuple[Trial, ...]:
        return tuple(t for t in self.trials if t.ok)

    def context(self, trial: Trial) -> int | None:
        """``K = I - T*B``: what the skill itself put in context, over all turns."""
        if self.base_per_request is None:
            return None
        return trial.usage.total_input - trial.usage.requests * self.base_per_request

    def medians(self) -> dict[str, float | None]:
        ts = self.ok_trials
        return {
            "wall_seconds": _median([t.wall_seconds for t in ts]),
            "num_turns": _median([t.usage.num_turns for t in ts]),
            "requests": _median([t.usage.requests for t in ts]),
            "input_tokens": _median([t.usage.total_input for t in ts]),
            "cache_write_tokens": _median([t.usage.cache_write for t in ts]),
            "cache_read_tokens": _median([t.usage.cache_read for t in ts]),
            "output_tokens": _median([t.usage.output for t in ts]),
            "cost_usd": _median([t.usage.cost_usd for t in ts]),
            "context_tokens": _median([self.context(t) for t in ts]),
            "bench_score": _median(
                [s for t, s in zip(self.trials, self.bench_scores, strict=True) if t.ok]
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "base_per_request_tokens": self.base_per_request,
            "medians": self.medians(),
            "trials": [
                {**t.to_dict(), "context_tokens": self.context(t), "bench_score": s}
                for t, s in zip(self.trials, self.bench_scores, strict=True)
            ],
        }


@dataclass(frozen=True)
class SubjectReport:
    subject: Subject
    sides: dict[Side, SideSummary]
    #: Original trial i against compiled trial i.
    pairs: tuple[proof.Comparison, ...]
    #: Every pair of original trials: the noise floor.
    noise: tuple[proof.Comparison, ...]
    #: Each side's trials against the bundled demo's reference.
    references: dict[Side, tuple[proof.Comparison | None, ...]]
    #: Why the eval stopped early, if it did.
    stopped: str | None = None

    def to_dict(self) -> dict[str, Any]:
        s = self.subject
        return {
            "skill": s.name,
            "case": None if s.case is None else s.case.id,
            "shape": s.shape,
            "shape_why": s.shape_why,
            "pipeline": str(s.pipeline_yaml),
            "provenance": dict(s.provenance),
            "sides": {side: summary.to_dict() for side, summary in self.sides.items()},
            "pairs": [c.to_dict() for c in self.pairs],
            "noise": [c.to_dict() for c in self.noise],
            "vs_reference": {
                side: [None if c is None else c.to_dict() for c in cs]
                for side, cs in self.references.items()
            },
            "stopped": self.stopped,
        }


def _fraction(comparisons: Sequence[proof.Comparison]) -> str:
    if not comparisons:
        return "n/a"
    if all(c.compared == 0 for c in comparisons):
        return "no exact fields"
    reproduced = sum(c.reproduced for c in comparisons)
    compared = sum(c.compared for c in comparisons)
    return f"{reproduced}/{compared}"


def run_subject(harness: Harness, subject: Subject, trials: int) -> SubjectReport:
    """Both sides' controls, then ``trials`` interleaved pairs (original 1,
    compiled 1, original 2, ...), so neither side always runs on a warmer
    cache. A budget stop keeps what already ran."""
    compiled_dir = emit_compiled(subject, harness.work / "compiled")
    done: dict[Side, list[Trial]] = {"original": [], "compiled": []}
    base: dict[Side, int | None] = {"original": None, "compiled": None}
    stopped: str | None = None
    try:
        for side in SIDES:
            control = run_trial(harness, subject, side, 0, compiled_dir=compiled_dir, control=True)
            if control.usage.requests == 1 and control.usage.per_request_input:
                base[side] = control.usage.per_request_input[0]
        for index in range(1, trials + 1):
            for side in SIDES:
                done[side].append(
                    run_trial(harness, subject, side, index, compiled_dir=compiled_dir)
                )
    except BudgetExhausted as e:
        stopped = str(e)
        harness.log(f"{harness.caller}: {subject.label}: stopped: {e}")
    sides = {
        side: SideSummary(
            side, tuple(done[side]), base[side], tuple(bench_score(subject, t) for t in done[side])
        )
        for side in SIDES
    }
    pairs = tuple(
        proof.Comparison(proof.compare_outputs(subject.fields, o.output, c.output))
        for o, c in zip(done["original"], done["compiled"], strict=False)
    )
    noise = tuple(
        proof.Comparison(proof.compare_outputs(subject.fields, a.output, b.output))
        for a, b in combinations(done["original"], 2)
    )
    references = {side: tuple(vs_reference(subject, t) for t in done[side]) for side in SIDES}
    return SubjectReport(subject, sides, pairs, noise, references, stopped)


def _num(value: float | None, unit: str = "", digits: int = 0) -> str:
    if value is None:
        return "n/a"
    if unit == "$":
        return f"${value:.3f}"
    if unit == "k":
        return f"{value / 1000:.1f}k"
    return f"{value:.{digits}f}{unit}"


def render_markdown(reports: Sequence[SubjectReport], *, model: str, spent: float) -> str:
    """The results table (medians per side), grouped by shape, and the checks."""
    lines = [
        f"Model: {model}. Medians over the trials that passed every check. Input counts "
        "every cache bucket. K is what the skill put in context (I - T*B, B from a "
        "control run per side). Fields: original trial i against compiled trial i; "
        f"noise: the original against itself. Spent: ${spent:.3f}.",
        "",
    ]
    for shape in ("compute-heavy", "tool-and-copy"):
        group = [r for r in reports if r.subject.shape == shape]
        if not group:
            continue
        lines += [
            f"### {shape}",
            "",
            "| Skill | Side | Trials ok | Turns | Input | Output | K | Wall | Cost | Fields "
            "| Bench |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for report in group:
            for side in SIDES:
                summary = report.sides[side]
                m = summary.medians()
                fields = _fraction(report.pairs) if side == "compiled" else _fraction(report.noise)
                lines.append(
                    f"| {report.subject.label if side == 'original' else ''} | {side} | "
                    f"{len(summary.ok_trials)}/{len(summary.trials)} | "
                    f"{_num(m['num_turns'])} | {_num(m['input_tokens'], 'k')} | "
                    f"{_num(m['output_tokens'], 'k')} | {_num(m['context_tokens'], 'k')} | "
                    f"{_num(m['wall_seconds'], ' s')} | {_num(m['cost_usd'], '$')} | {fields} | "
                    f"{_num(m['bench_score'], digits=2)} |"
                )
        lines.append("")
    lines.append("Checks:")
    clean = True
    for report in reports:
        for side in SIDES:
            for trial in report.sides[side].trials:
                if trial.problems:
                    clean = False
                    lines.append(
                        f"- {report.subject.label} {side} #{trial.index}: "
                        + "; ".join(trial.problems)
                    )
        for i, cmp in enumerate(report.pairs, 1):
            for f in cmp.fields:
                if f.status != "matched":
                    lines.append(
                        f"- {report.subject.label} pair {i}: {f.spec.name} {f.status}"
                        + (f" ({f.problem})" if f.problem else "")
                    )
        if report.stopped:
            lines.append(f"- {report.subject.label}: stopped early: {report.stopped}")
        if report.subject.replay:
            lines += _replay_lines(report)
    if clean:
        lines.append(
            "- every compiled trial ran through the driver, and every gate was answered "
            "by the person"
        )
    return "\n".join(lines) + "\n"


def _write_key(write: Mapping[str, Any]) -> str:
    return json.dumps([write["server"], write["tool"], write["arguments"]], sort_keys=True)


def _replay_lines(report: SubjectReport) -> list[str]:
    """A replay's proposed writes, side by side per pair, and its timing caveat."""
    lines = [
        f"- {report.subject.label}: replayed against recorded responses; wall times are "
        "against local fixture servers, not live service latency"
    ]
    originals = report.sides["original"].trials
    compiled = report.sides["compiled"].trials
    for i, (o, c) in enumerate(zip(originals, compiled, strict=False), 1):
        if not o.writes and not c.writes:
            lines.append(f"- {report.subject.label} pair {i}: neither side proposed a write")
            continue
        # What each side asked to write, and how often: a compiled step that
        # retries a failed write proposes the same call again, which is a
        # difference in how often, not in what.
        same_calls = set(map(_write_key, o.writes)) == set(map(_write_key, c.writes))
        same_counts = sorted(map(_write_key, o.writes)) == sorted(map(_write_key, c.writes))
        verdict = (
            "agree"
            if same_counts
            else "agree on what to write, not how often"
            if same_calls
            else "differ"
        )
        lines.append(
            f"- {report.subject.label} pair {i}: proposed writes {verdict} "
            f"(original {len(o.writes)}, compiled {len(c.writes)}; none sent)"
        )
        if same_counts:
            continue
        for side, trial in (("original", o), ("compiled", c)):
            counts: dict[str, int] = {}
            for w in trial.writes:
                counts[_write_key(w)] = counts.get(_write_key(w), 0) + 1
            for key, n in counts.items():
                server, tool, arguments = json.loads(key)
                times = f" x{n}" if n > 1 else ""
                lines.append(
                    f"  - {side}: {server}.{tool} {json.dumps(arguments, sort_keys=True)}{times}"
                )
    return lines


def corpus_record(report: SubjectReport, *, model: str, generated_at: str) -> dict[str, Any]:
    return {
        "kind": CORPUS_KIND,
        "generated_at": generated_at,
        "skill_model": model,
        **{k: v for k, v in report.to_dict().items() if k != "sides"},
        "sides": {
            side: {
                "base_per_request_tokens": summary.base_per_request,
                "medians": summary.medians(),
                "trials": [
                    {k: v for k, v in t.to_dict().items() if k not in ("output", "work_dir")}
                    for t in summary.trials
                ],
            }
            for side, summary in report.sides.items()
        },
    }


def append_corpus(records: Sequence[dict[str, Any]], path: Path | None = None) -> Path:
    corpus = path or default_corpus_path()
    corpus.parent.mkdir(parents=True, exist_ok=True)
    with corpus.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, default=str) + "\n")
    return corpus


def compile_argv(skill_dir: Path, work: Path, *, model: str | None) -> list[str]:
    """The ``rote compile`` argv (without ``rote``) :func:`compile_for_eval`
    runs in ``work``: :func:`rote.proof.compile_command`, pinned, into
    ``work/compile``, with the sample input at ``work/compile-input.json``."""
    return proof.compile_command(
        skill_dir, work / "compile-input.json", work / "compile", runtime="skill", model=model
    )


def compile_for_eval(
    skill_dir: Path,
    input_payload: dict[str, Any],
    work: Path,
    ledger: Ledger,
    *,
    model: str | None,
    log: Callable[[str], None],
    estimate_usd: float = DEFAULT_COMPILE_ESTIMATE_USD,
    caller: str = "rote eval",
) -> tuple[Path, dict[str, Any]]:
    """Compile ``skill_dir`` with :func:`compile_argv`, in its own isolated
    state. Returns the pipeline.yaml and what to record about the compile
    (the argv, the pinned settings, its cost). ``estimate_usd`` is what the
    ledger assumes it costs before any compile has run."""
    from rote.bench.collect import decisions_cost_usd, read_compile_metrics

    ledger.check("compile", estimate_usd)
    input_path = work / "compile-input.json"
    input_path.write_text(json.dumps(input_payload, indent=2), encoding="utf-8")
    out_dir = work / "compile"
    argv = compile_argv(skill_dir, work, model=model)
    log(f"{caller}: compiling: rote " + " ".join(argv))
    for setting in proof.pinned_settings():
        log(f"  {setting.flag[2:]}: {setting.value}, pinned by {caller} ({setting.why})")
    env = isolated_env(work / "compile-state")
    started = time.monotonic()
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "rote", *argv], env=env, capture_output=True, text=True
        )
    except BaseException:
        # Interrupted (Ctrl-C, a kill): the compile may have spent anything up
        # to now, so it is recorded at its estimate, never as free.
        ledger.record(
            "compile", f"compile {_skill_name(skill_dir)} (interrupted)", None, estimate_usd
        )
        raise
    wall = time.monotonic() - started
    compiled = out_dir / "compiled"
    meta = read_compile_metrics(compiled)
    agent_cost = meta.get("cost_usd")
    decisions = decisions_cost_usd(compiled)
    cost = (
        None
        if not isinstance(agent_cost, (int, float)) or decisions is None
        else float(agent_cost) + decisions
    )
    ledger.record("compile", f"compile {_skill_name(skill_dir)}", cost, estimate_usd)
    if proc.returncode != 0 or not (compiled / "pipeline.yaml").is_file():
        raise SkillAbError(
            f"rote compile exited {proc.returncode}: {failure_excerpt(proc.stderr, proc.stdout)}"
        )
    return compiled / "pipeline.yaml", {
        "pipeline": "compiled by this eval",
        "argv": ["rote", *argv],
        "pinned": {s.flag[2:]: s.value for s in proof.pinned_settings()},
        "wall_seconds": round(wall, 1),
        "cost_usd": cost,
    }


def require_executable(executable: str) -> str:
    found = shutil.which(executable)
    if found is None:
        raise EmpiricalError(
            f"`{executable}` CLI not found: install Claude Code or pass a different executable"
        )
    return found
