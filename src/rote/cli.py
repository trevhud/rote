"""rote CLI — entry point.

The north star for this CLI:

    rote compile ./path/to/skill --runtime temporal --out ./compiled/

One command, skill in, runnable workflow out. Every other command is a
building block for that flow:

    rote emit    <pipeline.yaml> --runtime temporal  # IR → code only
    rote analyze <skill-path>                        # compiler dry run
    rote run     <skill-or-compiled-dir>            # one-off local execution
    rote eval    <compiled-dir>                     # before/after scorecard
    rote login                                       # connect to rote cloud
    rote init                                        # save defaults (one-time wizard)
    rote config                                      # show effective defaults + sources

With a stored login, ``rote compile`` defaults to emitting the
cloudflare runtime and hosting the result on rote cloud in one step.
Defaults resolve as flag > ROTE_* env > project rote.yaml > user
config > built-in (see ``rote.config``).

Internally ``rote compile`` runs the ``rote-compile`` skill in an agent
loop, which reads the source skill, produces a ``pipeline.yaml``, stubs
the extracted/signature modules, then invokes the chosen adapter to emit
the runtime code.

This module is deliberately thin — subcommands delegate to real
implementations in sibling modules so the CLI itself stays boring and
testable.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import json
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rote import __version__
from rote.adapters import ADAPTERS, get_adapter
from rote.adapters._common import stubs_tracked
from rote.compiler import Compiler, CompilerError
from rote.compiler.drivers import DRIVERS, available_drivers
from rote.compiler.events import CompilationEvent
from rote.compiler.provenance import PROVENANCE_NODE
from rote.compiler.system1 import DECIDERS, TIEBREAKS
from rote.compiler.system1.structure import DEFAULT as DEFAULT_STRUCTURE
from rote.compiler.system1.structure import MODELS as STRUCTURE_MODELS
from rote.compiler.system1.tiebreak import DEFAULT_TIEBREAK
from rote.compiler.verify import EmitTarget
from rote.contracts import check_contracts
from rote.extras import (
    BASE_MODULES,
    EXTRAS,
    dependency_hint,
    install_hint,
    is_installed,
    reinstall_hint,
)
from rote.inference import PROVIDERS as INFERENCE_PROVIDERS
from rote.ir import Pipeline, load_pipeline

if TYPE_CHECKING:
    from rote.config import Resolved
    from rote.eval.pricing import PricingCatalog
    from rote.mcp.learned import LearnedAuth

# ───────── Subcommand: emit ─────────


def _print_written(written: dict[str, Path], indent: str = "  ", stream: Any = None) -> None:
    """Print an adapter's written-files mapping, flagging preserved files.

    When the emit writer finds a file the user edited since the last
    emit, it leaves the file alone and parks the fresh content in a
    ``<name>.new`` sibling — surface those so the preservation is a
    visible event, not a silent one.

    ``stream`` defaults to stdout; ``--json`` callers pass ``sys.stderr``
    so stdout carries only the JSON object.
    """
    out = stream if stream is not None else sys.stdout
    for label, path in written.items():
        print(f"{indent}{label}: {path}", file=out)
    preserved = [path for path in written.values() if path.name.endswith(".new")]
    if preserved:
        print(file=out)
        print(
            f"{indent}note: {len(preserved)} file(s) were edited since the last emit "
            f"and were left untouched.",
            file=out,
        )
        print(
            f"{indent}Fresh output landed alongside as '.new' files — merge or delete them:",
            file=out,
        )
        for path in preserved:
            print(f"{indent}  {path}", file=out)


def _preserved_new_files(written: dict[str, Path]) -> list[str]:
    """The ``.new`` siblings the emit writer parked (user-edited files it
    refused to clobber). Detected the same way :func:`_print_written` reports
    them, so the JSON and the human output never disagree."""
    return sorted(str(path) for path in written.values() if path.name.endswith(".new"))


def _unimplemented_stubs(runtime_dir: Path) -> list[str]:
    """The emitted ``extracted/*`` modules still carrying a stub marker: the
    fill-in TODO list. One source for the human report, ``--json``, the
    compile progress summary, and ``rote run``'s preflight
    (:func:`rote.adapters._common.unimplemented_stubs`), so they never
    disagree."""
    from rote.adapters._common import unimplemented_stubs

    return [str(p) for p in unimplemented_stubs(runtime_dir)]


def _run_command(runtime_dir: Path, pipeline: Pipeline | None, pipeline_path: Path | None) -> str:
    """The ``rote run`` command for an emitted app, ready to paste.

    An app emitted from a bare pipeline.yaml has no skill to derive an
    input from, so a pipeline with required inputs needs ``--input``. The
    file is the ``input.json`` beside the pipeline or one level up (the
    examples' layout); without one, the placeholder names the fields.
    """
    import shlex

    from rote.adapters.skill import emitted_skill

    candidates = []
    if pipeline_path is not None:
        candidates = [
            pipeline_path.parent / "input.json",
            pipeline_path.parent.parent / "input.json",
        ]
    skill = emitted_skill(runtime_dir)
    if skill is not None:
        # A compiled skill runs through `rote skill run <name>`, which runs
        # only what `rote skill install` recorded; `start` takes the input.
        command = (
            f"rote skill install {shlex.quote(str(runtime_dir.resolve()))} && "
            f"rote skill run {skill['name']} start"
        )
        file_flag, inline = "--file", ""
        if pipeline_path is not None:
            # A compiled skill's own sample input (the examples keep one there).
            candidates += [p / "skill" / "input.json" for p in pipeline_path.parents[:2]]
    else:
        command = f"rote run {shlex.quote(str(runtime_dir))}"
        file_flag, inline = "--input", "--input "
    if pipeline is None or pipeline_path is None or not pipeline.required_input_fields:
        return command if inline else f"{command} '{{}}'"
    for candidate in candidates:
        if candidate.is_file():
            return f"{command} {file_flag} {shlex.quote(str(candidate))}"
    fields = ", ".join(pipeline.required_input_fields)
    return f"{command} {inline}'<JSON object with {fields}>'"


def _print_stub_todo(
    stubs: list[str],
    runtime_dir: Path,
    *,
    tracked: bool,
    indent: str = "  ",
    stream: Any = None,
    run_command: str | None = None,
) -> None:
    """Tell a human which emitted modules they still have to fill in.

    ``tracked`` is :func:`rote.adapters._common.stubs_tracked`: when the
    emitted code imports its implementations from outside the directory,
    an empty list proves nothing, so this says so instead of "none".
    """
    out = stream if stream is not None else sys.stdout
    run = run_command or f"rote run {runtime_dir}"
    if not tracked:
        print(
            f"{indent}stubs: not tracked for this output (its implementation "
            "modules are not emitted here); implement what the emitted code "
            "imports before running",
            file=out,
        )
        return
    if not stubs:
        print(f"{indent}stubs: none, every extracted module is implemented", file=out)
        print(f"{indent}next: {run}", file=out)
        return
    print(
        f"{indent}stubs: {len(stubs)} module(s) still raise until you implement them:",
        file=out,
    )
    for stub in stubs:
        print(f"{indent}  {stub}", file=out)
    print(f"{indent}next: fill those in, then {run}", file=out)


def _add_skill_runtime_args(parser: argparse.ArgumentParser, *, with_skill_dir: bool) -> None:
    """The ``--runtime skill`` options, shared by emit and compile."""
    from rote.plan import JUDGE_MODES

    group = parser.add_argument_group("--runtime skill")
    if with_skill_dir:
        group.add_argument(
            "--skill",
            default=None,
            metavar="DIR",
            dest="skill_dir",
            help=(
                "The source skill's directory, whose name and description the compiled "
                "skill keeps (default: the pipeline's source_skill, when the "
                "provenance.json beside pipeline.yaml vouches for it)"
            ),
        )
    group.add_argument(
        "--judges",
        default=None,
        choices=JUDGE_MODES,
        help=(
            "Who answers the compiled skill's judges: 'session' (default), the agent "
            "running the skill, in a DECIDE block; or 'helper', rote's inference helper "
            "on the IR's model, one call per element, which bills the Claude "
            "subscription, an API key or rote cloud. Recorded in the manifest"
        ),
    )
    group.add_argument(
        "--allow-stubs",
        action="store_true",
        help=(
            "Emit the skill although an extracted/ module is still a stub (a run stops "
            "with an ERROR at its first call)"
        ),
    )


def _skill_runtime_options(args: argparse.Namespace) -> dict[str, Any]:
    """The adapter options ``--runtime skill`` takes, from either command's flags."""
    if args.runtime != "skill":
        return {}
    return {
        "skill_dir": getattr(args, "skill_dir", None),
        "judges": args.judges,
        "allow_stubs": args.allow_stubs,
    }


def _reject_skill_flags_off_skill(args: argparse.Namespace, command: str) -> bool:
    """Print an error and return True when a skill-only flag targets another runtime."""
    given = [
        flag
        for flag, value in (
            ("--skill", getattr(args, "skill_dir", None)),
            ("--judges", args.judges),
            ("--allow-stubs", args.allow_stubs),
        )
        if value
    ]
    if given and args.runtime != "skill":
        print(
            f"error: rote {command}: {', '.join(given)} only apply to --runtime skill "
            f"(the runtime is {args.runtime})",
            file=sys.stderr,
        )
        return True
    return False


def _record_emitted_app(out_dir: Path, runtime: str, pipeline: Pipeline) -> None:
    """Record an emitted app in the apps registry; a compiled skill under its skill name."""
    from rote.adapters.skill import emitted_skill
    from rote.app_registry import record_app

    skill = emitted_skill(out_dir) if runtime == "skill" else None
    record_app(out_dir, runtime, pipeline.name, name=skill["name"] if skill else None)


def _note_skill_dir_name(out_dir: Path) -> None:
    """One stderr note when a compiled skill's directory is not named after it.

    The agentskills.io spec has the folder match ``name``, and some clients
    name the skill by its folder; the skill still works, so it is a note."""
    from rote.adapters.skill import emitted_skill

    skill = emitted_skill(out_dir)
    if skill and out_dir.resolve().name != skill["name"]:
        print(
            f"rote emit: note: clients expect a skill's folder to be named after it; this one "
            f"is {out_dir.resolve().name!r}, and the skill is {skill['name']!r}",
            file=sys.stderr,
        )


def _cmd_emit(args: argparse.Namespace) -> int:
    """Render a runtime adapter from a pipeline.yaml.

    Pure IR → code. No agent involved. This is the lowest-level
    subcommand and the simplest to reason about — it should be used
    when the user already has a hand-written or previously-compiled
    ``pipeline.yaml`` and just wants to re-render the runtime code.
    """
    pipeline_path = Path(args.pipeline_yaml)
    if not pipeline_path.exists():
        print(f"error: pipeline file not found: {pipeline_path}", file=sys.stderr)
        return 2

    # Layered runtime default (flag > ROTE_RUNTIME > project > user),
    # falling back to emit's historical built-in: dbos.
    from rote import config as rote_config

    try:
        runtime_rv = rote_config.resolve("runtime", args.runtime, layers=rote_config.load_layers())
    except rote_config.ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    args.runtime = runtime_rv.value or "dbos"
    if _reject_skill_flags_off_skill(args, "emit"):
        return 2
    _note_missing_runtime_extra("emit", args.runtime)

    try:
        pipeline = load_pipeline(pipeline_path)
    except Exception as e:
        print(f"error: failed to load pipeline: {e}", file=sys.stderr)
        return 1

    backend = _resolve_backend("emit", args.backend, pipeline_path.parent)
    if backend is None:
        return 2
    args.backend = backend

    try:
        adapter = get_adapter(
            args.runtime,
            external_backend=args.backend,
            # The agent's real extracted/ modules live beside the
            # pipeline.yaml; Python-emitting adapters use them verbatim
            # instead of stubs.
            extracted_source_dir=pipeline_path.parent,
            **_skill_runtime_options(args),
        )
    except KeyError as e:
        print(f"error: {e.args[0]}", file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    try:
        written = adapter.emit(pipeline, out_dir)
    except ValueError as e:
        # Emit-time rejections are UX, not crashes: e.g. the python
        # adapter refusing a hitl_gate pipeline (with a pointer at a
        # durable runtime), or a forward data-flow reference.
        print(f"error: {e}", file=sys.stderr)
        return 1

    # Record the app so later commands (e.g. `rote mcp login` releasing
    # parked workflows) can find it without being told where it lives.
    _record_emitted_app(out_dir, args.runtime, pipeline)
    if args.runtime == "skill":
        _note_skill_dir_name(out_dir)

    mcp_servers = _mcp_requirements(pipeline, args.backend, args.runtime)

    # The IR's `impl` refs point at modules beside the pipeline.yaml, which
    # is exactly what was just sourced into the runtime dir — so this is
    # also the fix-and-re-emit loop's checkpoint. Free, and it needs no
    # runtime; reported, never fatal (the emitted dir is still useful).
    contract_findings = check_contracts(pipeline, pipeline_path.parent)
    stubs = _unimplemented_stubs(out_dir)
    tracked = stubs_tracked(out_dir, pipeline)
    from rote.adapters._common import TS_TYPES_MODULE, unchecked_output_notes
    from rote.runners import _TS_RUNTIMES

    unchecked = unchecked_output_notes(
        pipeline, types_module=TS_TYPES_MODULE if args.runtime in _TS_RUNTIMES else None
    )

    if args.json:
        import json

        payload = {
            "pipeline": {"name": pipeline.name, "version": pipeline.version},
            "runtime": args.runtime,
            "out_dir": str(out_dir.resolve()),
            "written": {label: str(path) for label, path in written.items()},
            "preserved_new_files": _preserved_new_files(written),
            "unimplemented_stubs": stubs,
            "stubs_tracked": tracked,
            "mcp_servers": mcp_servers,
            "contract_findings": [f.as_dict() for f in contract_findings],
            "unchecked_output_types": unchecked,
        }
        print(json.dumps(payload, indent=2))
        return 0

    print(f"rote: emitted {pipeline.name} v{pipeline.version} → {out_dir}")
    _print_written(written)
    for finding in contract_findings:
        label = "ERROR" if finding.severity == "error" else "warning"
        print(f"  contract: {label} {finding.node_id}: {finding.message}")
    for note in unchecked:
        print(f"  note: {note}")
    if mcp_servers:
        rendered = ", ".join(f"{e['server']} [{e['auth']}]" for e in mcp_servers)
        print(f"  required MCP servers: {rendered}")
        for line in _mcp_recommendation_lines(mcp_servers):
            print(line)
    _print_stub_todo(
        stubs,
        out_dir,
        tracked=tracked,
        run_command=_run_command(out_dir, pipeline, pipeline_path),
    )
    return 0


# ───────── Subcommand stubs (compile / analyze / eval) ─────────


#: The rote-compile skill runs seven numbered phases; used to render
#: ``[phase N/7]`` progress lines. Kept in sync with skills/rote-compile.
_COMPILE_TOTAL_PHASES = 7


def _format_token_note(tokens: dict[str, int] | None) -> str:
    """Compact ``(in 40.2k / out 8.1k tok, 39.9k cached)`` annotation.

    Counts >=1000 are abbreviated to one decimal of ``k``; smaller ones
    print raw. Empty string when there's nothing to show, so callers can
    append it unconditionally.

    The ``in`` figure is the run's whole input volume: plain input plus
    cache writes plus cache reads. Reporting only the plain field made a
    fully prompt-cached compilation read as ``in 113`` when it had in
    fact moved six figures of input, so the cached share is broken out
    rather than dropped.
    """
    if not tokens:
        return ""

    def _h(n: int) -> str:
        return f"{n / 1000:.1f}k" if n >= 1000 else str(n)

    cached = tokens.get("cache_write", 0) + tokens.get("cache_read", 0)
    total_in = tokens.get("input", 0) + cached
    output = _h(tokens["output"]) if "output" in tokens else "pending"
    note = f" (in {_h(total_in)} / out {output} tok"
    if cached:
        note += f", {_h(cached)} cached"
    return note + ")"


def _compile_progress_printer() -> Callable[[CompilationEvent], None]:
    """One-line live progress to stderr for ``rote compile``.

    Plain ``print(file=sys.stderr)`` — no rich, no spinner, so it composes
    with piping and CI logs. Renders the event types a human watching a run
    cares about: phase transitions, the agent's tool calls (keyed by turn),
    warnings/errors, and the orchestrator's log + completion lines. Bare
    ``turn`` events (assistant reasoning with no action) are skipped to keep
    the stream readable — the tool lines already carry the turn number, and
    now also carry the run's cumulative token spend (the printer remembers
    the latest tally a ``turn`` event reported and annotates the tool lines
    that follow it, since the token figures ride on the turn events).
    """
    last_tokens: dict[str, int] | None = None

    def printer(event: CompilationEvent) -> None:
        nonlocal last_tokens
        line: str | None
        if event.type == "phase":
            name = event.phase_name or ""
            line = f"[phase {event.phase}/{_COMPILE_TOTAL_PHASES}] {name}".rstrip()
        elif event.type == "tool":
            target = event.path or event.url or (repr(event.query) if event.query else None)
            loc = f" {target}" if target else ""
            line = f"[turn {event.turn}] {event.tool_name}{loc}{_format_token_note(last_tokens)}"
        elif event.type == "warning":
            line = f"warning: {event.message}"
        elif event.type == "error":
            line = f"error: {event.message}"
        elif event.type == "turn":
            if event.tokens:
                last_tokens = event.tokens
            line = None  # too noisy; tool lines already show the turn
        else:  # log, artifact, complete
            line = event.message or None
        if line:
            print(line, file=sys.stderr)

    return printer


@dataclasses.dataclass(frozen=True)
class _Rates:
    """Per-Mtok USD rates for one model, cache buckets included.

    ``cache_read_per_mtok`` / ``cache_write_per_mtok`` are ``None`` when
    the catalog does not publish them for this model. That is a distinct
    state from zero, and :meth:`_JsonlProgressSink._cost_usd` treats it as
    "cannot price this run" rather than "these tokens were free".
    """

    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float | None = None
    cache_write_per_mtok: float | None = None


class _JsonlProgressSink:
    """Append-per-event JSONL progress sink for ``rote compile``.

    A compilation is a long, real-money agent loop; an agent (or a cloud
    runner) that launched it wants to tail its progress live. This sink
    writes one JSON object per :class:`CompilationEvent` to a file —
    ``json.dumps`` of the event's fields with the ``None``-valued ones
    dropped for compactness — flushed per event so a tailer sees each
    line the instant it lands. It runs *alongside* the stderr printer, not
    instead of it.

    Two enrichments live only in the serialized line, never on the wire
    :class:`CompilationEvent` (whose schema is locked across a network
    boundary): a per-event ``cost_usd`` priced from the model's current
    published rates for any event carrying cumulative ``tokens``, and a
    final ``type: "summary"`` digest written last (see
    :meth:`write_summary`).

    Deliberately defensive: a broken progress file must never kill a paid
    run, so every method swallows its own exceptions on top of the
    ``emit_safely`` guard the driver already applies. Pricing is
    best-effort — an offline price fetch or an unknown model simply omits
    ``cost_usd`` and the run continues.
    """

    def __init__(
        self,
        path: Path,
        model_id: str | None,
        model_of: Callable[[], str | None] | None = None,
    ) -> None:
        """``model_id`` is what a token-carrying event is priced as until
        ``model_of`` (the model the selected driver actually runs, asked at
        each event) names one. A compile that ran several models prices its
        summary per call instead (:meth:`write_summary`)."""
        self._path = path
        self._model_id = model_id
        self._model_of = model_of
        self._rates: dict[str, _Rates | None] = {}
        self._file: Any = None
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            # Truncate at start: one run owns the file for its lifetime.
            self._file = self._path.open("w", encoding="utf-8")
        except OSError:
            self._file = None

    @staticmethod
    def _resolve_prices(model_id: str) -> _Rates | None:
        """Per-Mtok rates for ``model_id``, or ``None``.

        One live catalog fetch at construction. Offline (``PricingError``)
        or an unpriced model yields ``None``, and cost enrichment is then
        silently skipped rather than failing the run.

        Cache rates come from :meth:`PricingCatalog.rates_for`, which
        spans the whole fetch rather than the three tier representatives.
        A model whose fetch published no cache rate keeps ``None`` there,
        which is deliberately distinct from zero: a run billed at the
        wrong cache rate is worse than a run with no reported cost.
        """
        from rote.eval.pricing import PricingError, fetch_catalog

        try:
            catalog = fetch_catalog(provider="anthropic")
        except PricingError:
            return None
        rates = catalog.rates_for(model_id)
        if rates is None:
            return None
        return _Rates(
            input_per_mtok=rates[0],
            output_per_mtok=rates[1],
            cache_read_per_mtok=rates[2],
            cache_write_per_mtok=rates[3],
        )

    def _model(self) -> str | None:
        ran = self._model_of() if self._model_of is not None else None
        return ran or self._model_id

    def _prices_for(self, model_id: str | None) -> _Rates | None:
        """Rates for ``model_id``, fetched once per model."""
        if model_id is None:
            return None
        if model_id not in self._rates:
            self._rates[model_id] = self._resolve_prices(model_id)
        return self._rates[model_id]

    def _cost_usd(self, tokens: dict[str, int], model_id: str | None = None) -> float | None:
        """Price the four token buckets, or return ``None`` honestly.

        A ``claude -p`` run is prompt-cached, so ``cache_write`` and
        ``cache_read`` carry nearly all of its input volume. Pricing only
        the plain input/output pair reported a 22-minute, 62-turn
        compilation at $0.029.

        When a run used the cache but the catalog has no cache rates for
        the model, this returns ``None`` rather than a number. Falling
        back to the plain input rate would overstate cache reads roughly
        tenfold, and dropping the buckets understates the run by orders of
        magnitude. No figure beats a wrong figure, and the claude lane
        reports its own whole-query ``total_cost_usd`` estimate.
        """
        rates = self._prices_for(model_id or self._model())
        if rates is None or tokens.get("output") is None:
            return None
        cache_write = tokens.get("cache_write", 0)
        cache_read = tokens.get("cache_read", 0)
        if (cache_write and rates.cache_write_per_mtok is None) or (
            cache_read and rates.cache_read_per_mtok is None
        ):
            return None
        cost = (
            tokens.get("input", 0) / 1e6 * rates.input_per_mtok
            + tokens.get("output", 0) / 1e6 * rates.output_per_mtok
            + cache_write / 1e6 * (rates.cache_write_per_mtok or 0.0)
            + cache_read / 1e6 * (rates.cache_read_per_mtok or 0.0)
        )
        return round(cost, 6)

    def _write(self, obj: dict[str, Any]) -> None:
        if self._file is None:
            return
        try:
            self._file.write(json.dumps(obj) + "\n")
            self._file.flush()
        except (OSError, ValueError, TypeError):
            pass

    def emit(self, event: CompilationEvent) -> None:
        """Serialize one event as an NDJSON line (never raises)."""
        try:
            obj = {k: v for k, v in dataclasses.asdict(event).items() if v is not None}
            if event.tokens and event.usage_complete is not False:
                cost = self._cost_usd(event.tokens)
                if cost is not None:
                    obj["cost_usd"] = cost
            self._write(obj)
        except Exception:
            # A progress-file bug must never sink a paid run.
            pass

    def _call_cost(self, call: dict[str, Any]) -> float | None:
        """One recorded call's cost: its own (the runtime's) when it has one,
        else its tokens at its model's rates, else None."""
        own = call.get("cost_usd")
        if isinstance(own, int | float) and not isinstance(own, bool):
            return float(own)
        if call.get("output_tokens") is None:
            return None
        tokens = {
            "input": int(call.get("input_tokens") or 0),
            "output": int(call["output_tokens"]),
            "cache_write": int(call.get("cache_write_tokens") or 0),
            "cache_read": int(call.get("cache_read_tokens") or 0),
        }
        return self._cost_usd(tokens, call.get("model"))

    def write_summary(
        self,
        summary: dict[str, Any],
        reported_cost: float | None = None,
        calls: list[dict[str, Any]] | None = None,
        system1_cost: float | None = 0.0,
    ) -> None:
        """Write the final ``type: "summary"`` digest as the last line.

        Prices the run's ``total_tokens`` into a ``cost_usd`` field the
        same way per-event lines are priced. Never raises.

        ``reported_cost`` is the agent runtime's whole-query estimate
        (e.g. from ``claude -p``), not authoritative billing data. It wins
        when present because it includes the per-model usage split that
        this single-model catalog estimate cannot represent.

        ``calls`` are a parallel compile's per-call records, each with its
        own ``model``: priced one by one at that model's rates, since its
        calls run different models (Sonnet 5.5 text, Haiku structure). One
        unpriceable call leaves the cost out rather than understating it.

        ``system1_cost`` is what System-1's own calls cost (decider,
        tie-break, structure pass; ``None`` when one could not be priced).
        It is added in, so ``cost_usd`` is every model call the compile made.
        """
        try:
            agent: float | None = None
            if reported_cost is not None:
                agent = reported_cost
            elif calls:
                priced = [self._call_cost(c) for c in calls]
                if all(x is not None for x in priced):
                    agent = sum(x or 0.0 for x in priced)
            else:
                total = summary.get("total_tokens")
                if isinstance(total, dict):
                    agent = self._cost_usd(total)
            summary = {**summary, "system1_cost_usd": system1_cost}
            if agent is not None and system1_cost is not None:
                summary["cost_usd"] = round(agent + system1_cost, 6)
            self._write(summary)
        except Exception:
            pass

    def close(self) -> None:
        if self._file is not None:
            with contextlib.suppress(OSError):
                self._file.close()
            self._file = None


def _live_mcp_specs_for_driver(agent: str | None) -> list[dict[str, Any]] | None:
    """Registry-backed MCP server specs for the in-process drivers.

    The api / openai-api drivers run the compiler agent in-process and
    can expose the local registry's read-only MCP tools to it — the
    compile-time analog of the baseline's registry wiring. The
    subprocess drivers (claude / codex) don't take the kwarg, so
    resolution only happens when the run will actually use an in-process
    driver. Never fatal: a missing fastmcp or an unauthenticated
    server prints a note to stderr and is skipped.
    """
    name = agent
    if name is None:
        from rote.compiler.drivers import auto_detect

        probe = auto_detect()
        name = probe.name if probe is not None else None
    if name not in ("api", "openai-api"):
        return None

    from rote.mcp import load_registry

    if not load_registry().servers:
        return None
    try:
        import fastmcp  # noqa: F401
    except ImportError:
        print(
            "rote compile: mcp — registered servers not wired into the "
            f"compiler ({dependency_hint('fastmcp')})",
            file=sys.stderr,
        )
        return None

    from rote.mcp.live_tools import registry_server_specs

    def _skip(server: str, reason: str) -> None:
        print(f"rote compile: mcp — skipping server {server!r}: {reason}", file=sys.stderr)

    specs = asyncio.run(registry_server_specs(on_skip=_skip))
    return specs or None


def _no_mcp_driver(agent: str | None) -> str:
    """Resolve and pin a driver whose live MCP tools we can actually disable."""
    if agent is None:
        from rote.compiler.drivers import auto_detect

        driver = auto_detect()
        agent = driver.name if driver is not None else None
    if agent not in ("api", "openai-api"):
        raise ValueError(
            "--no-mcp requires an api or openai-api compiler driver "
            f"(selected: {agent or 'none'}); subprocess drivers manage their own MCP "
            "configuration. Select --agent api or --agent openai-api."
        )
    return agent


def _note_missing_runtime_extra(command: str, runtime: str | None) -> None:
    """One stderr line when ``runtime``'s extra is missing, for a command
    that only writes the runtime's code (compile, emit, init): it does its
    work without the extra, so it neither asks nor stops."""
    from rote.setup.install_extra import missing_extra_note

    note = missing_extra_note(runtime)
    if note is not None:
        print(f"rote {command}: note: {note}", file=sys.stderr)


def _ensure_runtime_extra(
    args: argparse.Namespace, runtime: str | None, purpose: str
) -> int | None:
    """Offer to install ``runtime``'s extra when it is missing, for a
    command that runs the runtime (:func:`rote.setup.install_extra.ensure_extra`).

    None means go on; an int is the exit code to return. Reads the
    command's ``--json`` / ``--yes`` / ``--agent-mode`` when it has them."""
    from rote.extras import RUNTIME_EXTRAS, is_installed

    extra = RUNTIME_EXTRAS.get(runtime or "")
    if extra is None or is_installed(extra):
        return None
    from rote.interact.mode import Mode
    from rote.interact.output import Output
    from rote.setup.install_extra import ensure_extra

    try:
        mode = Mode.resolve(
            json=bool(getattr(args, "json", False)),
            yes=bool(getattr(args, "yes", False)),
            agent_mode=getattr(args, "agent_mode", None),
        )
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    argv: list[str] = getattr(args, "raw_argv", sys.argv[1:])
    return ensure_extra(extra, purpose, mode=mode, out=Output(mode), argv=argv)


def _cmd_compile(args: argparse.Namespace) -> int:
    """Run the full one-shot compilation flow.

    Steps:

    1. Resolve the source skill directory; reject if it's not a skill bundle.
    2. Run the compiler agent (via ``Compiler.compile``) to produce
       ``<out>/compiled/pipeline.yaml`` plus extracted modules and stubs.
    3. Hand the IR to the chosen runtime adapter to emit
       ``<out>/runtime/<target>/`` with workflow.py + activities.py.
    4. Print a one-screen summary of what was produced.

    Failure modes are reported with EX_USAGE (2) for user errors and
    EX_SOFTWARE (70) for everything else.
    """
    skill_path = Path(args.skill_path)
    if not skill_path.is_dir():
        print(
            f"error: skill path is not a directory: {skill_path}",
            file=sys.stderr,
        )
        return 2

    # ── Layered defaults ── flag > ROTE_* env > project rote.yaml >
    # user config > built-in. Resolved up front because the logged-in
    # cloud default (below) must honor a config-pinned local runtime or a
    # disabled deploy — those are explicit user choices, not the cloud path.
    from rote import config as rote_config
    from rote.cloud_auth import load_credential

    if args.cloud and args.no_deploy:
        print(
            "error: --cloud always deploys on rote cloud — drop --no-deploy "
            "(or use --local for a local run)",
            file=sys.stderr,
        )
        return 2

    cloud_cred = load_credential()
    try:
        layers = rote_config.load_layers()
        runtime_rv = rote_config.resolve("runtime", args.runtime, layers=layers)
        deploy_rv = rote_config.resolve("deploy", "none" if args.no_deploy else None, layers=layers)
        agent_rv = rote_config.resolve("agent", args.agent, layers=layers)
        model_rv = rote_config.resolve("model", args.model, layers=layers)
        effort_rv = rote_config.resolve("effort", args.effort, layers=layers)
        decider_rv = rote_config.resolve("decider", args.decider, layers=layers)
        tiebreak_rv = rote_config.resolve("tiebreak", args.tiebreak, layers=layers)
        structure_rv = rote_config.resolve("structure", args.structure, layers=layers)
    except rote_config.ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    # ── Cloud vs local mode ── with a login the compilation runs server-side
    # by default; --local / --no-deploy opt out, --cloud forces it. A
    # resolved runtime that isn't cloudflare, or a disabled deploy, is an
    # explicit choice for a local run — honor it silently (a config pin
    # earns a one-line reason). --cloud overrides (it always emits
    # cloudflare; an explicit --runtime FLAG + --cloud is rejected inside
    # _cmd_compile_cloud).
    config_prefers_local = runtime_rv.value not in (None, "cloudflare") or deploy_rv.value == "none"
    if args.cloud:
        use_cloud = True
    elif args.local or args.no_deploy or cloud_cred is None:
        # `args.no_deploy` here is redundant but deliberate: it already
        # forced `deploy_rv.value == "none"` above (it is that resolve's
        # flag argument), so `config_prefers_local` covers it. Dropping
        # it is an equivalent mutation — unobservable, not untested — and
        # it stays because this is where a reader looks to learn what
        # keeps a run local.
        use_cloud = False
    else:
        use_cloud = not config_prefers_local
    if use_cloud:
        # Cloud mode sees only per-run FLAG values — args is NOT yet
        # overwritten with config-resolved agent/model. Config `agent` /
        # `model` are local-execution preferences (the server picks the
        # compiler driver and has its own model default); leaking them here
        # would spuriously trip the local-only flag rejection or fail
        # resolve_model against the server lineup.
        return _cmd_compile_cloud(args)

    # Local path: fold the config-resolved agent/model into args now.
    args.agent, args.model = agent_rv.value, model_rv.value
    args.effort = effort_rv.value
    try:
        args.decider = _local_decider(args, decider_rv, "compile")
        args.tiebreak = _local_system1_choice(args, tiebreak_rv, "--tiebreak", DEFAULT_TIEBREAK)
        args.structure = _local_system1_choice(args, structure_rv, "--structure", DEFAULT_STRUCTURE)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if args.no_mcp:
        try:
            if args.baseline:
                raise ValueError(
                    "--no-mcp cannot be combined with --baseline: the raw skill baseline "
                    "uses its own MCP tools. Run the baseline separately."
                )
            args.agent = _no_mcp_driver(args.agent)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
    if cloud_cred is not None and config_prefers_local and not args.local and not args.no_deploy:
        # A config file (not a per-run flag/env) opted out of the cloud
        # default — say why once, the way the logged-out path hints login.
        pinned_runtime = runtime_rv.value not in (None, "cloudflare")
        pin_rv = runtime_rv if pinned_runtime else deploy_rv
        if pin_rv.source not in ("flag",) and not pin_rv.source.startswith("env "):
            setting = "runtime" if pinned_runtime else "deploy"
            print(
                f"rote compile: your {setting} config opts out of the rote-cloud "
                "default — running locally (pass --cloud to override)",
                file=sys.stderr,
            )

    if runtime_rv.value is not None:
        args.runtime = runtime_rv.value
    elif cloud_cred is not None:
        args.runtime = "cloudflare"
    else:
        args.runtime = "dbos"
        if layers.user or layers.project:
            print(
                "rote compile: tip — `rote login` makes rote cloud the default "
                "host (compile + deploy in one step)",
                file=sys.stderr,
            )
        else:
            print(
                "rote compile: tip — `rote init` saves your defaults; `rote login` "
                "makes rote cloud the default host (compile + deploy in one step)",
                file=sys.stderr,
            )

    if deploy_rv.value == "rote-cloud":
        if args.runtime != "cloudflare":
            if runtime_rv.source == "flag" or runtime_rv.source.startswith("env "):
                # An explicit runtime override beats the configured cloud
                # destination for this run — downgrade loudly, don't block.
                print(
                    f"rote compile: skipping the configured rote-cloud deploy — "
                    f"--runtime {args.runtime} is not cloud-hostable",
                    file=sys.stderr,
                )
                auto_deploy = False
            else:
                print(
                    f"error: {deploy_rv.source} sets `deploy: rote-cloud`, which requires "
                    f"the cloudflare runtime — but runtime resolves to {args.runtime} "
                    f"({runtime_rv.source})",
                    file=sys.stderr,
                )
                return 2
        elif cloud_cred is None:
            # An explicit cloud destination fails fast BEFORE the agent
            # spends money, not after the paid run.
            print(
                f"error: {deploy_rv.source} sets `deploy: rote-cloud` but you are not "
                "logged in — run `rote login`, or pass --no-deploy for a local-only run",
                file=sys.stderr,
            )
            return 2
        else:
            auto_deploy = True
    elif deploy_rv.value == "none":
        auto_deploy = False
    else:
        auto_deploy = cloud_cred is not None and args.runtime == "cloudflare"

    # An update keeps the backend the previous compile recorded unless the
    # flag says otherwise; a fresh compile defaults to mcp.
    if args.update:
        backend = _resolve_backend("compile", args.backend, Path(args.out) / "compiled")
        if backend is None:
            return 2
        args.backend = backend
    else:
        args.backend = args.backend or "mcp"

    from rote.runners import _TS_RUNTIMES

    if args.backend == "api" and args.runtime in _TS_RUNTIMES:
        # The api backend's vendor calls are the agent's Python modules, and
        # a TypeScript runtime emits a stub for every external_call instead:
        # the paid docs research would be thrown away. Refused before the
        # agent runs.
        why = runtime_rv.source if runtime_rv.value else "the logged-in default"
        print(
            f"error: --backend api writes the vendor calls as Python modules, and the "
            f"{args.runtime} runtime ({why}) emits TypeScript, with a stub for every "
            f"external_call. Pass --runtime dbos, python or temporal, or keep "
            f"--backend mcp for {args.runtime}",
            file=sys.stderr,
        )
        return 2

    # Emitting needs none of the runtime's extra, so a missing one is a
    # note, never a stop: `rote run` on the result offers the install.
    _note_missing_runtime_extra("compile", args.runtime)

    if _reject_skill_flags_off_skill(args, "compile"):
        return 2
    out_dir = Path(args.out)
    compiled_dir = out_dir / "compiled"
    runtime_dir = out_dir / "runtime" / args.runtime
    if args.runtime == "skill":
        # The compiled skill keeps this skill's name and description, so a
        # skill rote cannot name is refused before the agent spends money.
        # Its folder is named after it, as clients expect.
        from rote.adapters.skill import SkillIdentityError, read_identity

        try:
            runtime_dir = runtime_dir / read_identity(skill_path).name
        except SkillIdentityError as e:
            print(f"error: --runtime skill: {e}", file=sys.stderr)
            return 2

    # ── Optional measured baseline (before the compiler spends money) ──
    baseline_result = None
    if args.baseline:
        from rote.compiler.drivers.claude import DEFAULT_MODEL as _BASELINE_MODEL
        from rote.eval.baseline import run_baseline
        from rote.eval.empirical import EmpiricalError

        try:
            baseline_payload = _resolve_baseline_input(
                skill_path, args.baseline_input, assume_yes=args.yes, out_dir=out_dir
            )
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        except EmpiricalError as e:
            print(f"rote compile: baseline: {e}", file=sys.stderr)
            return 1
        if baseline_payload is None:
            return 0
        print("rote compile: running measured baseline of the raw skill…", file=sys.stderr)
        try:
            baseline_result = run_baseline(
                skill_path,
                baseline_payload,
                out_dir,
                model=_BASELINE_MODEL,
                allow_writes=args.allow_writes,
            )
        except EmpiricalError as e:
            print(f"rote compile: baseline: {e}", file=sys.stderr)
            return 1
        for r in baseline_result.runs:
            cost = f"${r.cost_usd:.2f}" if r.cost_usd is not None else "cost n/a"
            note = f" [{', '.join(r.flags)}]" if r.flags else ""
            print(
                f"  baseline: {r.wall_seconds:.0f}s, {r.turns} turns, {cost}{note}",
                file=sys.stderr,
            )
            # A degraded baseline still informs the compile (what it did
            # record is real), but its numbers are not the skill's.
            _print_skill_run_gaps(r, sys.stderr)
    else:
        print(
            "rote compile: tip — add --baseline to measure the raw skill first "
            "(the scorecard's before-side becomes data, not an estimate)",
            file=sys.stderr,
        )

    # A JSONL progress sink runs alongside the stderr printer: both fire
    # for every event. The sink writes one machine-readable line per
    # event; the printer keeps the human view. It always writes to
    # <out>/progress.jsonl (--progress-file overrides the path) so that
    # a run driven by an agent stays observable from the user's own
    # terminal — the watch hint below is printed first thing for exactly
    # that tail. A sink failure must never take down a paid run, so its
    # body is self-guarded (and emit_safely wraps the whole callback
    # anyway).
    printer = _compile_progress_printer()
    # Price against the model the run actually uses: the selected driver's
    # own model once the Compiler has chosen it (`compiler` is bound below,
    # before any event fires), else the --model override. A parallel
    # compile's summary is priced per call (write_summary's `calls`).
    progress_path = Path(args.progress_file) if args.progress_file else out_dir / "progress.jsonl"
    progress_sink: _JsonlProgressSink | None = _JsonlProgressSink(
        progress_path,
        model_id=args.model,
        model_of=lambda: getattr(compiler, "driver_model", None),
    )
    print(f"rote compile: watch progress with: tail -f {progress_path}", file=sys.stderr)

    def _on_event(event: CompilationEvent) -> None:
        printer(event)
        if progress_sink is not None:
            progress_sink.emit(event)

    # The backend reaches the Compiler through emit_target below; the
    # Compiler gives the driver web tools and the agent its brief for
    # --backend api (rote.compiler.api_docs).
    driver_kwargs: dict[str, Any] = {}
    # The claude and openai-api drivers size thinking by effort; the others
    # absorb the kwarg (driver-registry convention).
    if args.effort is not None:
        driver_kwargs["effort"] = args.effort
    # In-process drivers (api / openai-api) can call MCP tools live during
    # the compilation; wire in the local registry's authenticated servers
    # the same way the baseline does. Read-only tools only.
    if args.no_mcp:
        driver_kwargs["mcp_servers"] = []
    else:
        mcp_specs = _live_mcp_specs_for_driver(args.agent)
        if mcp_specs:
            driver_kwargs["mcp_servers"] = mcp_specs
    compiler = Compiler(
        agent=args.agent,
        model=args.model,
        on_event=_on_event,
        driver_kwargs=driver_kwargs or None,
        run_generated_tests=not args.no_verify_tests,
        trust_skill=args.trust_skill,
        strict_provenance=args.strict_provenance,
        system1=args.system1,
        system1_decider=args.decider,
        system1_tiebreak=args.tiebreak,
        max_parallel=args.max_parallel,
        system1_structure=args.structure,
        # Tool schemas come from a read-only tools/list; --no-mcp means the
        # compile touches no MCP server at all.
        tool_discovery="none" if args.no_mcp else args.tool_discovery,
        # Verification dry-emits for the runtime emitted below, so an IR
        # this adapter refuses is repaired inside the paid run, not
        # reported after it.
        emit_target=EmitTarget(
            args.runtime,
            args.backend,
            skill_dir=skill_path if args.runtime == "skill" else None,
            judges=args.judges,
        ),
    )

    try:
        from rote.eval.baseline import BASELINE_DIRNAME as _PROBE_BASELINE_DIRNAME

        result = asyncio.run(
            compiler.compile(
                skill_path,
                compiled_dir,
                update=args.update,
                # Baseline artifacts (from --baseline just now, or a prior
                # `rote baseline --out` here) are ground truth the agent
                # reads during compilation — observed MCP traffic + schemas.
                probe_dir=out_dir / _PROBE_BASELINE_DIRNAME,
                sample_input=args.sample_input,
            )
        )
    except CompilerError as e:
        if progress_sink is not None:
            progress_sink.close()
        print(f"rote compile: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        if progress_sink is not None:
            progress_sink.close()
        print("rote compile: interrupted", file=sys.stderr)
        return 130

    try:
        adapter = get_adapter(
            args.runtime,
            external_backend=args.backend,
            # Use the just-compiled extracted/ implementations verbatim
            # in the runtime dir (Python-emitting adapters only).
            extracted_source_dir=compiled_dir,
            **{**_skill_runtime_options(args), "skill_dir": skill_path},
        )
    except KeyError as e:
        if progress_sink is not None:
            progress_sink.close()
        print(f"error: {e.args[0]}", file=sys.stderr)
        return 2

    # ── Observed-contract enrichment ── baseline observations (from
    # --baseline just now, or a prior `rote baseline --out` here) carry
    # real payloads per MCP tool; nodes bound to observed tools gain
    # input/output JSON-Schema contracts the agent didn't set. Must happen
    # BEFORE emission so stubs carry the contracts, and the rewritten
    # pipeline.yaml stays the single source of truth for what was emitted.
    if baseline_result is not None:
        observations = list(baseline_result.observations)
    else:
        from rote.eval.baseline import BASELINE_DIRNAME as _BASELINE_DIRNAME
        from rote.eval.baseline import load_observations

        observations = load_observations(out_dir / _BASELINE_DIRNAME)

    pipeline = result.pipeline
    if observations:
        from rote.compiler.tool_grounding import declared_output_schemas
        from rote.ir import save_pipeline
        from rote.probe import enrich_pipeline, infer_tool_schemas

        pipeline, enriched_ids = enrich_pipeline(
            pipeline, infer_tool_schemas(observations, declared_output_schemas(compiled_dir))
        )
        if enriched_ids:
            save_pipeline(pipeline, compiled_dir / "pipeline.yaml")
            print(
                f"rote compile: typed {len(enriched_ids)} node contract(s) from "
                f"observed baseline traffic: {', '.join(enriched_ids)}",
                file=sys.stderr,
            )

    # The agent has been paid and the IR is on disk in compiled_dir, so
    # an emission failure must never surface as a raw traceback that
    # reads like the whole run was lost. Any exception: emission is
    # template code over an LLM-written IR, and an unanticipated shape
    # can fail in more ways than the adapters' own ValueErrors.
    try:
        written = adapter.emit(pipeline, runtime_dir)
    except Exception as e:  # noqa: BLE001 (reported with the retry command)
        if progress_sink is not None:
            progress_sink.close()
        _print_emit_failure_after_compile(e, compiled_dir, runtime_dir, args)
        return 1

    _record_emitted_app(runtime_dir, args.runtime, pipeline)

    # ── Deliverable verification ── the compiler already ran it (and fed
    # failures back to the agent for repair); this reports the final
    # verdict. A compiler that predates it (or a no-op update) leaves it
    # unset, and then only the generated tests are run, as before.
    from rote.compiler.verify import run_generated_tests

    deliverable = result.verification
    if args.no_verify_tests:
        tests_run = None
        print(
            f"rote compile: generated tests NOT RUN (--no-verify-tests); review "
            f"{compiled_dir / 'tests'} before running them",
            file=sys.stderr,
        )
    elif deliverable is not None:
        tests_run = deliverable.tests
    else:
        tests_run = run_generated_tests(compiled_dir, trust_skill=args.trust_skill)
    verification = tests_run.as_dict() if tests_run is not None else None
    if tests_run is not None:
        status = {
            "passed": "passed",
            "failed": "FAILED",
            "unverified": "NOT RUN, deliverable UNVERIFIED",
        }[tests_run.status]
        print(
            f"rote compile: generated tests {status} ({tests_run.summary})",
            file=sys.stderr,
        )
        if tests_run.status == "failed":
            print(
                f"  review the agent's implementations against {compiled_dir / 'tests'}: "
                "a red test means an implementation disagrees with the skill's rules "
                "or the observed payloads",
                file=sys.stderr,
            )
    if deliverable is not None and deliverable.implementations:
        impls = deliverable.implementations
        done = [i for i in impls if i.status == "implemented"]
        via_mcp = [i for i in impls if i.status != "implemented" and not i.blocks_run]
        print(
            f"rote compile: {len(done)} of {len(impls)} impl node(s) implemented"
            + (f", {len(via_mcp)} more run via their mcp binding" if via_mcp else ""),
            file=sys.stderr,
        )
        for impl in impls:
            if impl.status == "implemented":
                continue
            via = "" if impl.blocks_run else " (runs via its mcp binding)"
            reason = f": {impl.reason}" if impl.reason else ""
            print(f"  {impl.status}: {impl.node_id}{via}{reason}", file=sys.stderr)
    if deliverable is not None:
        for line in _provenance_lines(deliverable, "compile"):
            print(line, file=sys.stderr)
    if deliverable is not None or tests_run is not None or args.no_verify_tests:
        _record_verification_in_report(compiled_dir, deliverable, tests_run, args)

    # ── Scorecard (auxiliary: a price-fetch outage must not sink a
    # just-completed, real-money compilation run) ──
    scorecard_path: Path | None = None
    # An eval.yaml still invalid after the repair rounds is a verification
    # error, reported with the others below. Building the scorecard would
    # raise on it after the paid run, so skip it and say why.
    from rote.compiler.verify import SIDECAR_INVALID
    from rote.eval.sidecar import EVAL_SIDECAR_FILENAME

    sidecar_invalid = deliverable is not None and any(
        f.kind == SIDECAR_INVALID for f in deliverable.errors
    )
    if sidecar_invalid and not args.no_eval:
        print(
            f"rote compile: warning: skipped scorecard ({EVAL_SIDECAR_FILENAME} fails "
            f"its schema; see the verification errors). Generate it once fixed with: "
            f"rote eval {out_dir}",
            file=sys.stderr,
        )
    elif not args.no_eval:
        from rote.eval import build_scorecard_for
        from rote.eval.pricing import PricingError

        try:
            scorecard = build_scorecard_for(
                pipeline,
                compiled_dir / "pipeline.yaml",
                skill_path,
                provider="anthropic",
                # Infrastructure is priced for where this run is headed:
                # rote cloud when it auto-deploys there, else the emit.
                runtime="rote-cloud" if auto_deploy else args.runtime,
            )
            scorecard_path = compiled_dir / "scorecard.md"
            scorecard_md = scorecard.to_markdown() + "\n"
            if baseline_result is not None:
                from rote.eval.baseline import render_baseline_markdown

                scorecard_md += "\n" + render_baseline_markdown(baseline_result)
            scorecard_path.write_text(scorecard_md, encoding="utf-8")
        except PricingError as e:
            print(
                f"rote compile: warning: skipped scorecard (live price fetch "
                f"failed: {e}). Generate it later with: rote eval {out_dir}",
                file=sys.stderr,
            )

    # ── Summary ── (human text to stderr under --json, so stdout is the
    # JSON object only — the same stdout-contract split as `rote mcp headers`)
    summary_stream = sys.stderr if args.json else sys.stdout
    print(
        f"rote compile: ✓ {pipeline.name} v{pipeline.version}",
        file=summary_stream,
    )
    print(f"  driver: {result.driver_name}", file=summary_stream)
    if result.driver_metadata:
        meta_str = ", ".join(f"{k}={v}" for k, v in result.driver_metadata.items())
        print(f"  metadata: {meta_str}", file=summary_stream)
    print(f"  compiled artifacts: {compiled_dir}", file=summary_stream)
    if scorecard_path is not None:
        print(f"  eval scorecard: {scorecard_path}", file=summary_stream)
    print(f"  emitted runtime ({args.runtime}): {runtime_dir}", file=summary_stream)
    _print_written(written, indent="    ", stream=summary_stream)
    _print_stub_todo(
        _unimplemented_stubs(runtime_dir),
        runtime_dir,
        tracked=stubs_tracked(runtime_dir, pipeline),
        stream=summary_stream,
    )

    # ── Required MCP servers ── advisory only: the run parks-on-auth as
    # the backstop, so a dead credential is never fatal — but telling the
    # user now beats discovering it mid-flight.
    mcp_servers = _mcp_requirements(pipeline, args.backend, args.runtime)
    if mcp_servers:
        rendered = ", ".join(f"{e['server']} [{e['auth']}]" for e in mcp_servers)
        print(f"  required MCP servers: {rendered}", file=summary_stream)
        for line in _mcp_recommendation_lines(mcp_servers):
            print(line, file=summary_stream)
    if deliverable is not None:
        for line in _unchecked_tool_lines(
            deliverable.unchecked_tools, mcp_servers, deliverable.unlisted_servers
        ):
            print(line, file=summary_stream)
        for line in _tool_source_lines(deliverable):
            print(f"  {line}", file=summary_stream)

    # ── Static contract check ── `impl: extracted/x.py:f` is a promise the
    # IR makes and nothing verified: emitted steps call `f(**payload)` and
    # the first anyone hears of a missing symbol or a parameter mismatch is
    # a TypeError in a background job. Decidable by parsing, so it costs
    # nothing and needs no runtime. Reported, never fatal — the artifacts
    # are still the valuable, fixable output of a real-money run.
    contract_findings = check_contracts(pipeline, compiled_dir)
    for finding in contract_findings:
        label = "ERROR" if finding.severity == "error" else "warning"
        print(
            f"  contract: {label} {finding.node_id}: {finding.message}",
            file=summary_stream,
        )
    if any(f.severity == "error" for f in contract_findings):
        print(
            f"  contract: fix the above in {compiled_dir}, then re-run "
            f"`rote emit {compiled_dir / 'pipeline.yaml'}`",
            file=summary_stream,
        )

    # ── Static vs observed cross-check ── runs whenever baseline
    # observations exist for this out dir (from --baseline just now, or a
    # prior `rote baseline --out` here). observed_only entries are the
    # loud ones: the skill demonstrably calls a tool the pipeline missed.
    cross: dict[str, object] | None = None
    if observations:
        from rote.compiler.tool_grounding import declared_output_schemas
        from rote.probe import cross_check

        cross = cross_check(pipeline, observations, declared_output_schemas(compiled_dir))
        observed_only = cross["observed_only"]
        static_only = cross["static_only"]
        confirmed = cross["confirmed"]
        assert isinstance(observed_only, list)
        assert isinstance(static_only, list)
        assert isinstance(confirmed, list)
        if confirmed:
            print(
                f"  cross-check: {len(confirmed)} MCP binding(s) confirmed by observed traffic",
                file=summary_stream,
            )
        for entry in observed_only:
            print(
                f"  cross-check: skill called {entry['server']}.{entry['tool']} "
                f"({entry['observed_calls']}×) but no pipeline node binds it — "
                "likely a missed requirement; review the compiled IR",
                file=summary_stream,
            )
        for entry in static_only:
            print(
                f"  cross-check: {entry['server']}.{entry['tool']} is bound "
                f"({', '.join(entry['nodes'])}) but was not observed in the "
                "baseline (different task path, or held back by the read-only gate)",
                file=summary_stream,
            )
        output_mismatch = cross["output_mismatch"]
        assert isinstance(output_mismatch, list)
        for entry in output_mismatch:
            print(
                f"  cross-check: WARNING node {entry['node']} declares output "
                f"key(s) [{', '.join(entry['missing_keys'])}] that the observed "
                f"{entry['server']}.{entry['tool']} result "
                f"(a JSON {entry['observed_shape']}) cannot provide — an "
                "mcp-bound node emits a bare tool call, so post-processing "
                "folded into it is dropped; split the transformation into a "
                "pure_function node (node-kinds.md, external_call)",
                file=summary_stream,
            )

    # ── The upload leg of the logged-in default ── runs last so a deploy
    # failure can never cost the local artifacts; those are already on
    # disk and re-deployable without another agent run.
    deploy_entry: dict[str, object] | None = None
    exit_code = 0
    # A deliverable that still fails verification after the repair rounds
    # is kept (it is the output of a paid run, usually a small fix away)
    # but is not a success: exit non-zero, and don't ship it to the cloud.
    if deliverable is not None and not deliverable.ok:
        exit_code = 1
        rounds = result.driver_metadata.get("repair_rounds", 0)
        print(
            f"rote compile: the deliverable still fails verification after {rounds} "
            f"repair round(s); see the lines above. Everything is kept in "
            f"{compiled_dir}. Fix it there, then re-emit without an agent run: "
            f"rote emit {compiled_dir / 'pipeline.yaml'} --runtime {args.runtime} "
            f"--out {runtime_dir}",
            file=sys.stderr,
        )
        if auto_deploy:
            auto_deploy = False
            print(
                f"  skipped the rote cloud deploy; once fixed: "
                f"rote deploy {out_dir} --target rote-cloud",
                file=sys.stderr,
            )
    if auto_deploy:
        from rote.deploy import DeployError
        from rote.deploy_rote_cloud import deploy_rote_cloud
        from rote.runners import RunTarget

        cloud_target = RunTarget(
            kind="pipeline",
            path=runtime_dir,
            runtime="cloudflare",
            pipeline_yaml=compiled_dir / "pipeline.yaml",
        )
        try:
            report = deploy_rote_cloud(cloud_target)
            deploy_entry = {"target": "rote-cloud", "ok": True, "detail": report.detail}
            assert cloud_cred is not None
            print(f"  hosted on rote cloud: {cloud_cred.url}", file=summary_stream)
        except DeployError as e:
            deploy_entry = {"target": "rote-cloud", "ok": False, "error": str(e)}
            exit_code = 1
            print(f"rote compile: deploy to rote cloud failed: {e}", file=sys.stderr)
            print(
                f"  the compilation itself succeeded — retry the upload with: "
                f"rote deploy {out_dir} --target rote-cloud",
                file=sys.stderr,
            )

    if args.json:
        import json

        payload = {
            "pipeline": {"name": pipeline.name, "version": pipeline.version},
            "runtime": args.runtime,
            "out_dir": str(out_dir.resolve()),
            "compiled_dir": str(compiled_dir.resolve()),
            "runtime_dir": str(runtime_dir.resolve()),
            "scorecard": str(scorecard_path.resolve()) if scorecard_path is not None else None,
            "driver": result.driver_name,
            "driver_metadata": result.driver_metadata,
            "written": {label: str(path) for label, path in written.items()},
            "preserved_new_files": _preserved_new_files(written),
            "unimplemented_stubs": _unimplemented_stubs(runtime_dir),
            "stubs_tracked": stubs_tracked(runtime_dir, pipeline),
            "mcp_servers": mcp_servers,
        }
        if baseline_result is not None:
            from rote.eval.baseline import BASELINE_DIRNAME
            from rote.eval.empirical import measured_run_record

            payload["baseline"] = {
                "read_only": baseline_result.read_only,
                "runs": [measured_run_record(r) for r in baseline_result.runs],
                "observed_servers": baseline_result.observed_servers,
                "baseline_dir": str((out_dir / BASELINE_DIRNAME).resolve()),
            }
        if cross is not None:
            payload["mcp_cross_check"] = cross
        if verification is not None:
            payload["generated_tests"] = verification
        payload["contract_findings"] = [f.as_dict() for f in contract_findings]
        if deliverable is not None:
            payload["verification"] = {
                "ok": deliverable.ok,
                "errors": [f.as_dict() for f in deliverable.errors],
                "implementations": [i.as_dict() for i in deliverable.implementations],
            }
            payload["provenance"] = deliverable.provenance_report()
            payload["grounding"] = {
                "prefix": deliverable.prefix.as_dict() if deliverable.prefix else None,
                "questions": [q.as_dict() for q in deliverable.questions],
            }
        if deploy_entry is not None:
            payload["deploy"] = deploy_entry
        print(json.dumps(payload, indent=2))

    # ── Machine end-of-run digest ── The per-event `complete` line is for
    # a human tailer; this is the structured summary an agent reads: the
    # compiled pipeline's shape (roteness / node-kind counts, reusing the
    # same pure analysis `rote analyze` prints), where the artifacts landed,
    # the remaining stub TODOs, and the run's total token spend + cost. It's
    # the LAST NDJSON line, written after everything else is on disk.
    if progress_sink is not None:
        analysis = _build_analysis(pipeline, result.driver_name, result.driver_metadata, skill_path)
        analysis_nodes = analysis["nodes"]
        assert isinstance(analysis_nodes, dict)
        meta = result.driver_metadata
        total_tokens = {
            "input": int(meta.get("input_tokens") or 0),
            "output": (
                None if meta.get("usage_complete") is False else int(meta.get("output_tokens") or 0)
            ),
            "cache_write": int(meta.get("cache_write_tokens") or 0),
            "cache_read": int(meta.get("cache_read_tokens") or 0),
        }
        progress_sink.write_summary(
            {
                "type": "summary",
                "roteness": analysis["roteness"],
                "node_kinds": analysis_nodes["by_kind"],
                "nodes": analysis_nodes["total"],
                "compiled_dir": str(compiled_dir.resolve()),
                "runtime_dir": str(runtime_dir.resolve()),
                "unimplemented_stubs": _unimplemented_stubs(runtime_dir),
                "mcp_servers": analysis["mcp_servers"],
                "contract_errors": sum(1 for f in contract_findings if f.severity == "error"),
                **(
                    {"provenance": result.verification.provenance_report()}
                    if result.verification is not None
                    else {}
                ),
                "total_tokens": total_tokens,
                "usage_complete": meta.get("usage_complete", True),
            },
            reported_cost=meta.get("cost_usd"),
            # Failed calls were billed too: a parallel compile's cost is
            # every call it sent, answered or not.
            calls=[*(meta.get("generations") or []), *(meta.get("failed_calls") or [])],
            system1_cost=meta.get("system1_cost_usd", 0.0),
        )
        progress_sink.close()
    return exit_code


def _assumptions_by_node(found: list[Any]) -> list[tuple[str, list[Any]]]:
    """Assumptions grouped by owning node, in first-seen order."""
    groups: dict[str, list[Any]] = {}
    for a in found:
        groups.setdefault(a.node_id or "(module helpers)", []).append(a)
    return list(groups.items())


def _provenance_lines(deliverable: Any, command: str) -> list[str]:
    """The compile output's provenance block: every assumption, grouped by
    node, then the gaps the repair rounds left. Assumptions are listed for
    review and never fail the compile; the gaps are warnings unless
    ``--strict-provenance`` made them errors (then they are reported with
    the other unresolved problems)."""
    lines: list[str] = []
    found = deliverable.assumptions
    if found:
        lines.append(
            f"rote {command}: {len(found)} assumption(s) the skill does not state; "
            f"review each one (also in compile-report.md):"
        )
        for node, entries in _assumptions_by_node(found):
            lines.append(f"  {node}:")
            lines += [f"    {a.path}:{a.line}  {a.text}" for a in entries]
    for finding in deliverable.review_gaps:
        if finding in deliverable.provenance_gaps:
            what, where = "provenance warning", ""
        else:
            what, where = "warning", f"{finding.node_id}: "
        lines.append(f"rote {command}: {what} ({finding.kind}): {where}{finding.message}")
    if deliverable.prefix is not None:
        lines.append(
            f"rote {command}: deterministic prefix on the sample input: "
            f"{deliverable.prefix.summary}"
        )
    questions = deliverable.questions
    if questions:
        lines.append(
            f"rote {command}: {len(questions)} contract(s) nothing grounded; review each one "
            f"(also in compile-report.md):"
        )
        lines += [f"  {q.node_id}: {q.text}" for q in questions]
    return lines


#: The heading rote owns at the end of compile-report.md. Everything above
#: it is the agent's; the section is replaced, not appended, on a re-run.
_VERIFICATION_HEADING = "## rote verification"


def _tool_source_lines(deliverable: Any) -> list[str]:
    """What grounded each bound server's tool arguments, and a stale snapshot.

    Shared by the compile summary and compile-report.md, so the two say the
    same thing.
    """
    from rote.tool_schema import BUNDLED_TOOLS_FILENAME

    lines: list[str] = []
    sources = deliverable.tool_sources
    if sources:
        lines.append(
            "tool schemas: " + ", ".join(f"{server} from {how}" for server, how in sources.items())
        )
    stale = deliverable.stale_bundled_tools
    if stale:
        lines.append(
            f"WARNING: the skill's bundled {BUNDLED_TOOLS_FILENAME} is stale for "
            f"{', '.join(stale)}: the live server's schema differs, and the live one was "
            "used. Update the snapshot, or a compile without the server registered "
            "grounds in the old schema."
        )
    return lines


def _record_verification_in_report(
    compiled_dir: Path, deliverable: Any, tests_run: Any, args: argparse.Namespace
) -> None:
    """State in compile-report.md what rote checked, so the report cannot
    read as verified when the generated tests never ran."""
    report = compiled_dir / "compile-report.md"
    if not report.is_file():
        return
    if args.no_verify_tests:
        tests_line = "NOT RUN (--no-verify-tests). The deliverable is UNVERIFIED."
    elif tests_run is None:
        tests_line = "none were written."
    elif tests_run.status == "unverified":
        tests_line = f"NOT RUN. The deliverable is UNVERIFIED: {tests_run.summary}."
    else:
        where = "without a sandbox (--trust-skill)" if args.trust_skill else "in an OS sandbox"
        tests_line = f"{tests_run.status} ({tests_run.summary}), run {where}."
    lines = [_VERIFICATION_HEADING, "", f"- Generated tests: {tests_line}"]
    if deliverable is not None:
        impls = deliverable.implementations
        done = sum(1 for i in impls if i.status == "implemented")
        lines.append(f"- Implemented impl nodes: {done} of {len(impls)}.")
        errors = deliverable.errors
        lines.append(
            f"- Unresolved problems: {len(errors)}"
            + (": " + "; ".join(f"{f.node_id}: {f.kind}" for f in errors[:10]) if errors else ".")
        )
        gaps = deliverable.review_gaps
        if gaps:
            lines.append(f"- Review gaps ({len(gaps)}, not failures):")
            lines += [f"  - {f.kind}: {f.message}" for f in gaps]
        found = deliverable.assumptions
        lines.append(
            f"- Assumptions the skill does not state: {len(found)}"
            + (". Review each one:" if found else ".")
        )
        for node, entries in _assumptions_by_node(found):
            lines.append(f"  - {node}:")
            lines += [f"    - `{a.path}:{a.line}` {a.text}" for a in entries]
        if deliverable.prefix is not None:
            lines.append(
                f"- Deterministic prefix on the sample input: {deliverable.prefix.summary}."
            )
        lines += [f"- {line[0].upper()}{line[1:]}." for line in _tool_source_lines(deliverable)]
    text = report.read_text(encoding="utf-8")
    at = text.find("\n" + _VERIFICATION_HEADING)
    if at >= 0:
        text = text[:at]
    report.write_text(text.rstrip("\n") + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8")


def _print_emit_failure_after_compile(
    error: Exception, compiled_dir: Path, runtime_dir: Path, args: argparse.Namespace
) -> None:
    """Explain an emission failure that happened after a paid agent run.

    The point is the second line: the compiled IR survived, and
    ``rote emit`` retries the emission without spending on the agent
    again.
    """
    import shlex

    pipeline_yaml = compiled_dir / "pipeline.yaml"
    retry = ["rote", "emit", str(pipeline_yaml), "--runtime", args.runtime]
    if args.backend != "mcp":
        retry += ["--backend", args.backend]  # backend.json records it too
    if args.runtime == "skill":
        retry += ["--skill", str(Path(args.skill_path))]
        if args.judges:
            retry += ["--judges", args.judges]
        if args.allow_stubs:
            retry.append("--allow-stubs")
    retry += ["--out", str(runtime_dir)]
    from rote.adapters._py_common import ConstantsNotAccepted

    print(
        f"rote compile: the compilation succeeded, but emitting the {args.runtime} "
        f"runtime failed: {type(error).__name__}: {error}",
        file=sys.stderr,
    )
    if isinstance(error, ConstantsNotAccepted):
        # The one emit failure known to refuse before writing anything.
        where = (
            f"Nothing in {runtime_dir} was written. Fix the file the error names (an "
            f"agent-written module under {compiled_dir / 'extracted'}, or your own edit "
            f"under {runtime_dir / 'extracted'})"
        )
    else:
        where = (
            f"{runtime_dir} may hold a partial emit, which the retry overwrites. Fix what "
            f"the error names"
        )
    print(
        f"  the compiled IR and extracted/ modules are kept in {compiled_dir}. {where}, "
        f"then retry the emission (no agent run, no cost):\n"
        f"    {shlex.join(retry)}",
        file=sys.stderr,
    )


def _reject_local_only_cloud_flags(args: argparse.Namespace) -> str | None:
    """Message for the first local-only flag set under cloud mode, else None.

    Cloud compilation emits the cloudflare runtime and runs the compiler
    server-side, so the flags that only mean something on the local path
    (a chosen driver/backend, the local baseline pass, the local eval
    scorecard) are rejected with a pointer to ``--local``. ``--agent``
    defaults to ``None`` (auto-detect) so it only counts as set when the
    user passed a value; so does ``--backend``.
    """
    if args.runtime not in (None, "cloudflare"):
        return (
            f"rote compile: cloud compilation emits the cloudflare runtime — "
            f"add --local for --runtime {args.runtime}"
        )
    for flag, is_set in (
        ("--agent", args.agent is not None),
        ("--effort", args.effort is not None),
        ("--backend", args.backend is not None),
        ("--baseline", args.baseline),
        ("--baseline-input", args.baseline_input is not None),
        ("--sample-input", args.sample_input is not None),
        ("--tool-discovery", args.tool_discovery != "named"),
        ("--yes", args.yes),
        ("--no-eval", args.no_eval),
        ("--no-mcp", args.no_mcp),
        ("--no-verify-tests", args.no_verify_tests),
        ("--trust-skill", args.trust_skill),
        ("--strict-provenance", args.strict_provenance),
        ("--no-system1", args.system1 is False),
        ("--decider", args.decider is not None),
        ("--tiebreak", args.tiebreak is not None),
        ("--structure", args.structure is not None),
        ("--judges", args.judges is not None),
        ("--allow-stubs", args.allow_stubs),
    ):
        if is_set:
            return f"rote compile: {flag} runs locally — add --local to use it"
    return None


def _cloud_manifest_identity(written: dict[str, Path], runtime_dir: Path) -> tuple[str, str]:
    """``(name, version)`` from the downloaded manifest.json, or ``("?", "?")``."""
    manifest_path = runtime_dir / "manifest.json"
    for rel, dest in written.items():
        if Path(rel).name == "manifest.json":
            manifest_path = dest
            break
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "?", "?"
    return str(data.get("name") or "?"), str(data.get("version") or "?")


def _cmd_compile_cloud(args: argparse.Namespace) -> int:
    """Run the compilation on rote cloud and download the artifacts.

    Mirrors ``_cmd_compile``'s output conventions (the stderr progress
    printer, the optional ``--progress-file`` JSONL sink, the stdout/stderr
    split under ``--json``) but the work happens server-side: sync the
    skill bundle, start (or attach to) a compilation, stream its events,
    then download the finished tree. Full-mode compilations auto-deploy on
    the server, so there is no local deploy leg here.
    """
    from rote.cloud_compile import (
        ActiveCompilationExists,
        CloudCompileError,
        NoChanges,
        cancel_compilation,
        download_artifacts,
        poll_compilation,
        resolve_cloud_endpoint,
        resolve_model,
        start_compilation,
        sync_skill,
    )

    rejection = _reject_local_only_cloud_flags(args)
    if rejection is not None:
        print(rejection, file=sys.stderr)
        return 2

    skill_path = Path(args.skill_path)
    try:
        ep = resolve_cloud_endpoint()
        model = resolve_model(ep, args.model)
    except CloudCompileError as e:
        print(f"rote compile: {e}", file=sys.stderr)
        return e.exit_code

    print(
        f"rote compile: running on rote cloud ({ep.url}), model {model}",
        file=sys.stderr,
    )

    out_dir = Path(args.out)
    compiled_dir = out_dir / "compiled"
    runtime_dir = out_dir / "runtime" / "cloudflare"

    printer = _compile_progress_printer()
    progress_sink: _JsonlProgressSink | None = None
    if args.progress_file:
        progress_sink = _JsonlProgressSink(Path(args.progress_file), model)

    def _on_event(event: CompilationEvent) -> None:
        printer(event)
        if progress_sink is not None:
            progress_sink.emit(event)

    def _close_sink() -> None:
        if progress_sink is not None:
            progress_sink.close()

    # ── Sync the skill, or attach to a compilation already running ──
    attached = False
    compilation_id: str | None = None
    try:
        skill_id = sync_skill(ep, skill_path)
    except ActiveCompilationExists as e:
        print(
            "rote compile: a compilation is already running for this skill — attaching "
            "(local changes NOT uploaded; Ctrl-C detaches)",
            file=sys.stderr,
        )
        skill_id, compilation_id, attached = e.skill_id or "", e.compilation_id, True
    except CloudCompileError as e:
        _close_sink()
        print(f"rote compile: {e}", file=sys.stderr)
        return e.exit_code

    if not attached:
        mode = "update" if args.update else "full"
        try:
            record = start_compilation(ep, skill_id, mode=mode, model=model)
            compilation_id = str(record.get("id") or "")
        except NoChanges:
            _close_sink()
            print(
                "rote compile: no changes since the last compilation — nothing to do",
                file=sys.stderr,
            )
            return 0
        except ActiveCompilationExists as e:
            print(
                "rote compile: a compilation is already running for this skill — attaching "
                "(Ctrl-C detaches)",
                file=sys.stderr,
            )
            compilation_id, attached = e.compilation_id, True
        except CloudCompileError as e:
            _close_sink()
            print(f"rote compile: {e}", file=sys.stderr)
            return e.exit_code

    if not compilation_id:
        _close_sink()
        print("rote compile: rote cloud did not return a compilation id", file=sys.stderr)
        return 1

    # ── Stream events to a terminal state ──
    try:
        final = poll_compilation(ep, compilation_id, _on_event)
    except KeyboardInterrupt:
        if attached:
            print(
                f"rote compile: detached — the compilation keeps running (id {compilation_id})",
                file=sys.stderr,
            )
        else:
            cancel_compilation(ep, compilation_id)
            print(
                f"rote compile: interrupted — canceled compilation {compilation_id}",
                file=sys.stderr,
            )
        _close_sink()
        return 130
    except CloudCompileError as e:
        _close_sink()
        print(f"rote compile: {e}", file=sys.stderr)
        return e.exit_code

    status = final.get("status")
    summary_stream = sys.stderr if args.json else sys.stdout

    if status == "canceled":
        print(f"rote compile: compilation {compilation_id} was canceled", file=sys.stderr)
        _close_sink()
        return 130

    if status == "error":
        server_error = final.get("error") or "the compilation failed on rote cloud"
        print(f"rote compile: {server_error}", file=sys.stderr)
        # Best-effort partial download so the user keeps whatever landed.
        with contextlib.suppress(CloudCompileError):
            download_artifacts(ep, compilation_id, out_dir)
        _close_sink()
        return 1

    # ── status == complete: download the artifacts, then report ──
    try:
        written = download_artifacts(ep, compilation_id, out_dir)
    except CloudCompileError as e:
        _close_sink()
        print(f"rote compile: {e}", file=sys.stderr)
        print(
            f"  the compilation itself succeeded — artifacts remain on {ep.url}",
            file=sys.stderr,
        )
        return 1

    name, version = _cloud_manifest_identity(written, runtime_dir)
    scorecard_path = next(
        (dest for rel, dest in written.items() if Path(rel).name == "scorecard.md"), None
    )
    tokens = final.get("tokens") if isinstance(final.get("tokens"), dict) else None
    cost_usd = final.get("cost_usd")
    pipeline_id = final.get("pipeline_id")
    deployed = bool(final.get("deployed"))

    print(f"rote compile: ✓ {name} v{version}", file=summary_stream)
    print(f"  compiled artifacts: {compiled_dir}", file=summary_stream)
    print(f"  emitted runtime (cloudflare): {runtime_dir}", file=summary_stream)
    if scorecard_path is not None:
        print(f"  eval scorecard: {scorecard_path}", file=summary_stream)
    print(f"  hosted on rote cloud: {ep.url}", file=summary_stream)
    if pipeline_id:
        print(f"  pipeline id: {pipeline_id}", file=summary_stream)
    if tokens is not None:
        note = _format_token_note(tokens).strip()
        cost = f", cost ${cost_usd:.4f}" if isinstance(cost_usd, (int, float)) else ""
        print(f"  {note}{cost}".rstrip(), file=summary_stream)

    cloud_entry: dict[str, Any] = {
        "compilation_id": compilation_id,
        "skill_id": skill_id,
        "status": status,
        "mode": final.get("mode"),
        "model": final.get("model") or model,
        "tokens": tokens,
        "cost_usd": cost_usd,
        "pipeline_id": pipeline_id,
        "url": ep.url,
    }

    if args.json:
        payload = {
            "pipeline": {"name": name, "version": version},
            "runtime": "cloudflare",
            "out_dir": str(out_dir.resolve()),
            "compiled_dir": str(compiled_dir.resolve()),
            "runtime_dir": str(runtime_dir.resolve()),
            "scorecard": str(scorecard_path.resolve()) if scorecard_path is not None else None,
            "written": {rel: str(dest.resolve()) for rel, dest in written.items()},
            "cloud": cloud_entry,
            "deploy": {"target": "rote-cloud", "ok": deployed, "pipeline_id": pipeline_id},
        }
        print(json.dumps(payload, indent=2))

    if progress_sink is not None:
        progress_sink.write_summary(
            {
                "type": "summary",
                "compiled_dir": str(compiled_dir.resolve()),
                "runtime_dir": str(runtime_dir.resolve()),
                "cloud": cloud_entry,
                "total_tokens": tokens or {"input": 0, "output": 0},
            }
        )
        progress_sink.close()
    return 0


# ───────── Subcommand: register ─────────


def _resolve_pipeline_yaml(path: Path) -> Path | None:
    """Find the pipeline.yaml behind a user-supplied path.

    Accepts the file itself, a directory containing one, or a
    ``rote compile --out`` directory (which nests it under compiled/).
    """
    if path.is_file():
        return path
    for candidate in (path / "pipeline.yaml", path / "compiled" / "pipeline.yaml"):
        if candidate.is_file():
            return candidate
    return None


def _derive_dbos_sqlite_url(compiled_arg: Path, pipeline_name: str) -> str | None:
    """Locate the emitted DBOS app dir and derive its default SQLite URL.

    The emitted main.py defaults its system database to
    ``sqlite:///<app dir>/<pipeline.name>.dbos.sqlite``. Given what the
    user passed to ``rote register`` (the pipeline.yaml, its directory,
    or a ``rote compile --out`` directory), the app dir is wherever a
    ``main.py`` is found in the known layouts.
    """
    base = compiled_arg if compiled_arg.is_dir() else compiled_arg.parent
    candidates = (
        base,  # the emitted runtime dir itself
        base / "runtime" / "dbos",  # a compile --out dir
        base.parent / "runtime" / "dbos",  # the compiled/ dir inside one
    )
    for app_dir in candidates:
        if (app_dir / "main.py").is_file():
            return f"sqlite:///{(app_dir / f'{pipeline_name}.dbos.sqlite').resolve()}"
    return None


def _cmd_register(args: argparse.Namespace) -> int:
    """Add or update a registry entry for a compiled pipeline.

    Reads the pipeline.yaml from a compile run, derives the MCP tool
    name / description / inputSchema from it, attaches the runtime
    trigger config from the flags, and upserts ``~/.local/share/rote/serve-registry.json``
    (or ``--registry``). ``rote serve`` picks the change up live.
    """
    import os

    from rote.ir import NodeKind
    from rote.serve.registry import (
        CloudflareTrigger,
        DbosTrigger,
        Registry,
        TemporalTrigger,
        default_registry_path,
        entry_from_pipeline,
    )

    pipeline_yaml = _resolve_pipeline_yaml(Path(args.compiled_dir))
    if pipeline_yaml is None:
        print(
            f"error: no pipeline.yaml found at or under: {args.compiled_dir}",
            file=sys.stderr,
        )
        return 2

    try:
        pipeline = load_pipeline(pipeline_yaml)
    except Exception as e:
        print(f"error: failed to load pipeline: {e}", file=sys.stderr)
        return 1

    # Both the DBOS and Temporal adapters emit the same versioned
    # workflow name (PascalCase pipeline name + pipeline hash); the
    # trigger must match it exactly or starting the workflow fails.
    workflow_name = args.workflow_name
    if workflow_name is None:
        from rote.adapters._common import _pipeline_hash, _to_pascal_case

        workflow_name = f"{_to_pascal_case(pipeline.name)}_{_pipeline_hash(pipeline)}"

    trigger: TemporalTrigger | CloudflareTrigger | DbosTrigger
    if args.runtime == "dbos":
        system_database_url = (
            args.system_database_url
            or os.environ.get("DBOS_SYSTEM_DATABASE_URL")
            or _derive_dbos_sqlite_url(Path(args.compiled_dir), pipeline.name)
        )
        if not system_database_url:
            print(
                "error: --system-database-url is required for --runtime dbos "
                "(no DBOS_SYSTEM_DATABASE_URL in the environment, and no "
                "emitted main.py found near the pipeline to derive the "
                "default SQLite path from)",
                file=sys.stderr,
            )
            return 2
        trigger = DbosTrigger(
            system_database_url=system_database_url,
            workflow_name=workflow_name,
            # Must match the Queue the emitted main.py declares.
            queue_name=args.queue_name or f"{pipeline.name}-queue",
            gate_signals=[
                node.signal
                for node in pipeline.nodes
                if node.kind is NodeKind.HITL_GATE and node.signal is not None
            ],
        )
    elif args.runtime == "temporal":
        trigger = TemporalTrigger(
            address=args.temporal_address,
            namespace=args.temporal_namespace,
            task_queue=args.task_queue or pipeline.name,
            workflow_name=workflow_name,
        )
    else:  # cloudflare
        if not args.url:
            print(
                "error: --url is required for --runtime cloudflare "
                "(the deployed worker's trigger endpoint)",
                file=sys.stderr,
            )
            return 2
        trigger = CloudflareTrigger(url=args.url, status_url=args.status_url)

    try:
        entry = entry_from_pipeline(pipeline, pipeline_yaml, trigger, name=args.name)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    registry_path = Path(args.registry) if args.registry else default_registry_path()
    try:
        registry = Registry.load(registry_path)
    except Exception as e:
        print(
            f"error: failed to load registry {registry_path}: {e}\n"
            f"Fix or delete the file and re-run.",
            file=sys.stderr,
        )
        return 1
    replaced = registry.upsert(entry)
    registry.save(registry_path)

    verb = "updated" if replaced else "registered"
    print(f"rote register: {verb} tool '{entry.name}' ({args.runtime}) → {registry_path}")
    print(f"  pipeline: {entry.pipeline_yaml}")
    print(f"  serve it: rote serve --registry {registry_path}")
    return 0


# ───────── Subcommand: serve ─────────


def _cmd_serve(args: argparse.Namespace) -> int:
    """Launch the MCP server exposing every registered pipeline as a tool.

    stdio by default (what ``claude mcp add`` expects); ``--http`` for
    Streamable HTTP. Nothing may be printed to stdout in stdio mode —
    stdout is the MCP transport.
    """
    from rote.serve.registry import default_registry_path

    # The base install's fastmcp is the slim client; the server half is
    # the serve extra. Never offered as an install here: in stdio mode
    # stdin and stdout are the MCP transport, so there is nobody to ask.
    if not is_installed("serve"):
        print(
            "error: rote serve needs the serve extra (fastmcp's server half): "
            f"{install_hint('serve')}",
            file=sys.stderr,
        )
        return 2
    try:
        from rote.serve.server import build_server
    except ModuleNotFoundError as e:
        if e.name is None or e.name.split(".")[0] not in BASE_MODULES:
            raise
        print(
            f"error: rote serve needs {dependency_hint(e.name.split('.')[0])}",
            file=sys.stderr,
        )
        return 2

    registry_path = Path(args.registry) if args.registry else default_registry_path()
    if not registry_path.exists():
        print(
            f"rote serve: registry {registry_path} does not exist yet. Serving only "
            f"the built-in rote_list_pipelines and rote_describe_pipeline tools. "
            f"Register a pipeline with `rote register`.",
            file=sys.stderr,
        )

    server = build_server(registry_path)
    if args.http:
        server.run(transport="http", host=args.host, port=args.port, show_banner=False)
    else:
        server.run(show_banner=False)  # stdio
    return 0


# ───────── Subcommand group: mcp ─────────

_MCP_SERVER_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
"""Same constraint as the IR's ``mcp.server`` field — the registry key
IS the binding's logical server name, so they must share a charset."""


def _cmd_mcp_add(args: argparse.Namespace) -> int:
    from rote.mcp import McpServerConfig, load_registry, save_registry

    if not _MCP_SERVER_NAME_RE.fullmatch(args.name):
        print(
            f"error: server name {args.name!r} must be a valid identifier "
            f"(letters, digits, underscores; the same rule as the IR's mcp.server)",
            file=sys.stderr,
        )
        return 2
    headers: dict[str, str] = {}
    for item in args.header or []:
        key, sep, value = item.partition(":")
        if not sep or not key.strip():
            print(f"error: --header expects 'Name: value', got {item!r}", file=sys.stderr)
            return 2
        headers[key.strip()] = value.strip()

    issuer = args.issuer
    if args.issuer:
        from rote.mcp.auth import is_trusted_transport

        if not is_trusted_transport(args.issuer):
            print(
                f"error: --issuer {args.issuer!r} must be an https URL "
                "(http is accepted only for loopback hosts)",
                file=sys.stderr,
            )
            return 2
    if args.issuer and not args.client_id:
        print(
            "error: --issuer binds a pre-registered client, so it needs --client-id",
            file=sys.stderr,
        )
        return 2
    if args.client_id and not issuer:
        # Bind the client from its first login: record the authorization
        # server the resource names right now.
        import asyncio

        from rote.mcp.auth import discover_issuer

        discovery = asyncio.run(discover_issuer(args.url))
        issuer = discovery.issuer
        if issuer is None:
            print(
                f"  note: no authorization server recorded ({discovery.problem}). The "
                "client binds to whichever one its first login uses, or pass --issuer",
                file=sys.stderr,
            )

    registry = load_registry()
    replacing = args.name in registry.servers
    registry.servers[args.name] = McpServerConfig(
        url=args.url,
        transport=args.transport,
        client_id=args.client_id,
        client_secret=args.client_secret,
        issuer=issuer,
        scopes=args.scope or None,
        headers=headers or None,
        read_only=True if args.read_only else None,
    )
    path = save_registry(registry)
    if args.auth is not None:
        from rote.mcp.learned import record_auth

        record_auth(args.name, args.url, args.auth, source="user")
    verb = "updated" if replacing else "added"
    print(f"rote mcp: {verb} {args.name!r} → {args.url} ({path})")
    if args.read_only:
        print("  read-only: every tool on it may be called by read-only runs")
    if issuer:
        print(f"  client {args.client_id!r} bound to authorization server {issuer}")
    if not headers and args.auth is None:
        print(
            f"  next: rote mcp login {args.name} (logs in, or records that the "
            f"server needs no auth)"
        )
    elif not headers and args.auth == "oauth":
        print(f"  authenticate with: rote mcp login {args.name}")
    return 0


def _cmd_mcp_import(args: argparse.Namespace) -> int:
    """Copy MCP servers from the user's AI clients into rote's registry.

    Reads the clients' configs, asks what to import, shows the plan (the
    registry write, the work-domain answer, any logins) and applies it only
    on a yes (``rote.setup.mcp_import``). Nobody at the terminal: every
    question takes its flag, a missing one stops naming it (exit 2,
    ``action_required`` under ``--json``), and the plan needs ``--yes``.
    """
    from rote import config as rote_config
    from rote.clients import ClientContext
    from rote.interact.mode import Mode
    from rote.interact.output import Output
    from rote.interact.plan import FORCE_FLAG, ConfirmationRequired, apply_plan, confirm_plan
    from rote.interact.prompter import PromptCancelled, PromptRequired, prompter_for
    from rote.interact.report import EXIT_ACTION_REQUIRED, action_required
    from rote.mcp.registry import registry_path
    from rote.setup.mcp_import import (
        ImportFlags,
        ImportPlan,
        ImportPlanner,
        McpImportError,
        render_rows,
        split_names,
    )

    try:
        mode = Mode.resolve(json=args.json, yes=args.yes, agent_mode=args.agent_mode)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    renames: list[tuple[str, str]] = []
    for item in args.rename or []:
        old, sep, new = item.partition("=")
        if not sep or not old.strip() or not new.strip():
            print(f"error: --rename expects old=new, got {item!r}", file=sys.stderr)
            return 2
        renames.append((old.strip(), new.strip()))
    login: tuple[str, ...] | None = None
    if args.no_login:
        login = ()
    elif args.login:
        login = split_names(args.login)
    flags = ImportFlags(
        from_client=args.from_client,
        only=split_names(args.only) if args.only else None,
        servers_for=Path(args.servers_for) if args.servers_for else None,
        allow_url_secret=split_names(args.allow_url_secret or []),
        allow_untrusted=split_names(args.allow_untrusted or []),
        rename=tuple(renames),
        replace=split_names(args.replace or []),
        work_domains=split_names(args.work_domain or []),
        login=login,
    )
    try:
        layers = rote_config.load_layers()
    except rote_config.ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    out = Output(mode)
    prompter = prompter_for(mode)
    planner = ImportPlanner(
        flags=flags,
        prompter=prompter,
        out=out,
        ctx=ClientContext.current(),
        layers=layers,
        registry_path=registry_path(),
        user_config_path=rote_config.user_config_path(),
        detected_client=mode.detected.client if mode.detected is not None else None,
    )
    argv: list[str] = getattr(args, "raw_argv", sys.argv[1:])
    planned: ImportPlan | None = None
    try:
        planned = planner.build()
        for line in render_rows(planned.rows):
            out.say(line)
        out.plan(planned.plan)
        granted = {FORCE_FLAG} if args.force else set()
        if not confirm_plan(planned.plan, prompter, yes=mode.yes, granted=granted):
            out.say("rote mcp import: nothing was written")
            out.result(
                {"ok": False, "applied": False, "plan": planned.plan.to_dict(), **planned.to_dict()}
            )
            return 1
    except (PromptRequired, ConfirmationRequired) as e:
        out.error(str(e))
        payload = action_required(e, argv)
        # What was found, so an agent can relay the choice with hosts and
        # flags: the same display-safe shape (hosts, never URLs).
        if planned is not None:
            payload.update(planned.to_dict())
        else:
            payload["servers"] = [c.to_dict() for c in planner.found]
        out.result(payload)
        return EXIT_ACTION_REQUIRED
    except McpImportError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except PromptCancelled:
        out.say("rote mcp import: cancelled; nothing was written")
        return 130
    outcomes = apply_plan(planned.plan)
    for outcome in outcomes:
        line = f"  {outcome.status}: {outcome.action_id}"
        if outcome.error:
            line += f" ({outcome.error})"
        out.say(line)
    ok = all(o.status == "done" for o in outcomes)
    if ok and planned.plan.actions:
        out.say("  see them with: rote mcp list")
    out.result(
        {
            "ok": ok,
            "applied": True,
            "plan": planned.plan.to_dict(),
            "outcomes": [o.to_dict() for o in outcomes],
            **planned.to_dict(),
        }
    )
    return 0 if ok else 1


def _cmd_mcp_list(args: argparse.Namespace) -> int:
    import json as _json

    from rote.mcp import auth_status, load_registry, read_token_file
    from rote.mcp.learned import learned_auth

    registry = load_registry()
    entries: list[dict[str, object]] = []
    for name, config in sorted(registry.servers.items()):
        learned = learned_auth(name, config.url)
        entries.append(
            {
                "name": name,
                "url": config.url,
                "transport": config.transport,
                "auth": auth_status(
                    config, read_token_file(name), learned.auth if learned else None
                ),
                **_learned_fields(learned),
            }
        )

    if args.json:
        print(_json.dumps({"servers": entries}, indent=2))
        return 0
    if not entries:
        print("rote mcp: no servers registered — add one with: rote mcp add <name> <url>")
        return 0
    width = max(len(str(e["name"])) for e in entries)
    for e in entries:
        print(f"  {str(e['name']).ljust(width)}  {_auth_label(e):<34}  {e['url']}")
    return 0


def _cmd_mcp_remove(args: argparse.Namespace) -> int:
    from rote.mcp import clear_token_file, load_registry, save_registry

    registry = load_registry()
    if args.name not in registry.servers:
        print(f"error: no MCP server named {args.name!r}", file=sys.stderr)
        return 2
    del registry.servers[args.name]
    save_registry(registry)
    cleared = clear_token_file(args.name)
    print(f"rote mcp: removed {args.name!r}" + (" (tokens cleared)" if cleared else ""))
    return 0


def _cmd_mcp_login(args: argparse.Namespace) -> int:
    """``rote mcp login``: blocking by default. ``--no-wait`` / ``--complete`` /
    ``--cancel`` split it for agents (rote.mcp.pending_login)."""
    import asyncio

    from rote.interact.mode import Mode
    from rote.interact.output import Output
    from rote.mcp import McpAuthError, load_registry
    from rote.mcp.auth import login
    from rote.mcp.pending_login import PendingLoginError, result_summary

    chosen = [
        flag
        for flag, value in (
            ("--no-wait", args.no_wait),
            ("--complete", args.complete is not None),
            ("--cancel", args.cancel is not None),
        )
        if value
    ]
    if len(chosen) > 1:
        print(f"error: {' and '.join(chosen)} cannot be combined", file=sys.stderr)
        return 2
    if args.callback_url is not None and args.complete is None:
        print("error: --callback-url goes with --complete", file=sys.stderr)
        return 2
    try:
        mode = Mode.resolve(json=args.json, yes=args.yes, agent_mode=args.agent_mode)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    out = Output(mode)
    if args.complete is not None or args.cancel is not None:
        try:
            if args.cancel is not None:
                return _mcp_login_cancel(args, out)
            return _mcp_login_complete(args, out)
        except PendingLoginError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1

    if args.name is None:
        print("error: name the server to log in to: rote mcp login <name>", file=sys.stderr)
        return 2
    registry = load_registry()
    config = registry.servers.get(args.name)
    if config is None:
        print(
            f"error: no MCP server named {args.name!r} — register it first: "
            f"rote mcp add {args.name} <url>",
            file=sys.stderr,
        )
        return 2
    if args.no_wait:
        return _mcp_login_no_wait(args, config, mode, out)
    try:
        doc = asyncio.run(
            login(
                args.name,
                config,
                no_browser=args.no_browser,
                callback_port=args.callback_port,
                on_rejected_callback=lambda: print(
                    "note: ignored a callback with the wrong state (not this login's "
                    "browser); still waiting",
                    file=sys.stderr,
                    flush=True,
                ),
            )
        )
    except McpAuthError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    _report_mcp_login(args.name, result_summary(doc, config), out)
    return 0


def _report_mcp_login(server: str, summary: dict[str, Any], out: Any) -> None:
    """The lines (and the ``--json`` object) for a finished login, then the release.

    ``summary`` is :func:`rote.mcp.pending_login.result_summary`: the
    blocking login and ``--complete`` report through this one function.
    """
    if not summary.get("auth_required"):
        out.say(
            f"rote mcp: {server!r} answered without asking for auth; recorded "
            "as no auth required (runs connect unauthenticated)"
        )
    else:
        refresh = " (refresh token stored)" if summary.get("refresh_token") else ""
        out.say(f"rote mcp: authenticated {server!r}{refresh}")
        client = {
            "pre-registered": "the pre-registered client from the registry",
            "cimd": "rote's client metadata document",
            "dcr": "a dynamically registered client",
        }[summary["registration"]]
        issuer = summary.get("issuer")
        out.say(f"  client: {client}" + (f"; authorization server: {issuer}" if issuer else ""))
        expires_at = summary.get("expires_at")
        if expires_at:
            out.say(f"  access token expires at epoch {expires_at:.0f}; refresh is automatic")
    released = _release_parked_after_login(server, say=out.say)
    out.result({"ok": True, "server": server, **summary, "released": released})


def _mcp_login_no_wait(args: argparse.Namespace, config: Any, mode: Any, out: Any) -> int:
    """Start a login an agent can hand to its user, then exit (rote.mcp.pending_login)."""
    import webbrowser
    from urllib.parse import urlsplit

    from rote.interact.plan import (
        Action,
        ConfirmationRequired,
        Contact,
        FileWrite,
        Launch,
        Plan,
        apply_plan,
        confirm_plan,
    )
    from rote.interact.prompter import PromptCancelled, prompter_for
    from rote.interact.report import (
        EXIT_ACTION_REQUIRED,
        action_required,
        authorization_pending,
    )
    from rote.mcp.auth import CALLBACK_PORTS
    from rote.mcp.pending_login import (
        PENDING_TTL_SECONDS,
        Started,
        pending_dir,
        start_pending_login,
    )

    server = args.name
    if args.callback_port is not None:
        print(
            "error: --no-wait uses rote's published callback ports; drop --callback-port",
            file=sys.stderr,
        )
        return 2
    if config.headers and not config.client_id:
        print(
            f"error: server {server!r} is configured with static headers — no OAuth login needed",
            file=sys.stderr,
        )
        return 2
    host = urlsplit(config.url).hostname or config.url
    minutes = PENDING_TTL_SECONDS // 60
    ports = f"{CALLBACK_PORTS[0]}-{CALLBACK_PORTS[-1]}"
    started: list[Started] = []

    def start() -> str:
        started.append(start_pending_login(server, config))
        return started[0].kind

    actions = [
        Action(
            id=f"start-login-{server}",
            title=f"Start an OAuth login for {server!r} that waits in the background",
            effects=(
                Contact(
                    host=host,
                    purpose="find its authorization server, and register rote as a "
                    "client there if it needs one",
                ),
                Launch(
                    program=f"a rote callback listener on localhost (one of ports {ports})",
                    purpose=f"receive the browser's return, for at most {minutes} minutes",
                ),
                FileWrite(
                    path=pending_dir() / "<id>.json",
                    summary="the pending login record, owner-only",
                    file_mode=0o600,
                ),
            ),
            run=start,
            command=f"rote mcp login {server} --no-wait",
        )
    ]
    if not args.no_browser:

        def open_browser() -> bool:
            if not started or started[0].authorization_url is None:
                return False
            return bool(webbrowser.open(started[0].authorization_url))

        actions.append(
            Action(
                id="open-browser",
                title="Open the authorization page",
                effects=(Launch(program="your web browser", purpose="approve the login"),),
                run=open_browser,
                command=f"rote mcp login {server} --no-wait --no-browser",
            )
        )
    plan = Plan(title=f"Log in to {server!r} without waiting", actions=tuple(actions))
    argv: list[str] = getattr(args, "raw_argv", sys.argv[1:])
    out.plan(plan)
    try:
        if not confirm_plan(plan, prompter_for(mode), yes=mode.yes):
            out.say("rote mcp login: nothing was started")
            out.result({"ok": False, "applied": False, "plan": plan.to_dict()})
            return 1
    except ConfirmationRequired as e:
        out.error(str(e))
        out.result(action_required(e, argv))
        return EXIT_ACTION_REQUIRED
    except PromptCancelled:
        out.say("rote mcp login: cancelled; nothing was started")
        return 130
    outcomes = apply_plan(plan)
    if not started:
        failed = next(o for o in outcomes if o.status == "failed")
        print(f"error: {(failed.error or '').split(': ', 1)[-1]}", file=sys.stderr)
        return 1
    begun = started[0]
    if begun.kind == "done":
        out.say(f"rote mcp: {server!r} needed no browser approval")
        _report_mcp_login(server, begun.result or {}, out)
        return 0
    pending = begun.pending
    url = begun.authorization_url or ""
    opened = any(o.action_id == "open-browser" and o.result is True for o in outcomes)
    out.say(f"Authorize {server!r} in a browser on this machine:")
    out.say("")
    out.say(f"  {url}")
    out.say("")
    if opened:
        out.say("(opened in your browser)")
    out.say(f"Then run: rote mcp login --complete {pending.id}")
    out.say(
        "A browser on another machine ends on a page that cannot load: copy that "
        f"page's URL, run: rote mcp login --complete {pending.id} --callback-url - "
        "and paste it on stdin"
    )
    out.say(f"This login expires in {minutes} minutes.")
    out.result(
        {
            **authorization_pending(
                server=server,
                pending_id=pending.id,
                authorization_url=url,
                expires_at=pending.expires_at,
            ),
            "browser_opened": opened,
            "applied": True,
            "plan": plan.to_dict(),
            "outcomes": [o.to_dict() for o in outcomes],
        }
    )
    return 0


def _mcp_login_complete(args: argparse.Namespace, out: Any) -> int:
    from rote.interact.report import EXIT_ACTION_REQUIRED, authorization_pending
    from rote.mcp.pending_login import replay_callback, resolve_pending, wait_for_outcome

    pending = resolve_pending(args.complete or None, args.name)
    if args.callback_url is not None:
        url = args.callback_url
        if url == "-":
            # Off argv, so the code and state never show in `ps`.
            url = sys.stdin.readline().strip()
        replay_callback(pending, url)
    outcome = wait_for_outcome(pending, wait=args.wait)
    current = outcome.pending
    if current.rejected_callbacks:
        out.warn(
            f"{current.rejected_callbacks} callback(s) to the login listener carried the "
            "wrong state and were ignored"
        )
    if outcome.kind == "done":
        _report_mcp_login(current.server, current.result or {}, out)
        return 0
    if outcome.kind == "waiting":
        out.error(
            f"the login for {current.server!r} is still waiting for the browser. Approve "
            f"it at the authorization URL, then run again: rote mcp login --complete "
            f"{current.id}"
        )
        out.result(
            authorization_pending(
                server=current.server,
                pending_id=current.id,
                authorization_url=None,
                expires_at=current.expires_at,
                still_waiting=True,
            )
        )
        return EXIT_ACTION_REQUIRED
    why = (
        "expired before the browser returned"
        if outcome.kind == "expired"
        else f"failed: {current.error}"
    )
    print(
        f"error: the login for {current.server!r} {why}. "
        f"Start again: rote mcp login {current.server} --no-wait",
        file=sys.stderr,
    )
    return 1


def _mcp_login_cancel(args: argparse.Namespace, out: Any) -> int:
    from rote.mcp.pending_login import delete_pending, resolve_pending

    pending = resolve_pending(args.cancel or None, args.name)
    delete_pending(pending.id)
    out.say(f"rote mcp: cancelled the pending login {pending.id} for {pending.server!r}")
    out.result({"ok": True, "cancelled": pending.id, "server": pending.server})
    return 0


def _release_parked_after_login(
    server: str, *, say: Callable[[str], None] | None = None
) -> list[str]:
    """Wake emitted workflows parked waiting for this server's auth.

    Emitted apps park durably on missing/dead MCP credentials (see
    rote.mcp.release); a successful login is the event they're waiting
    for, so discovery + release happens here rather than as a separate
    command the user has to know about. (`rote mcp release <server>`
    exists for when the credential was fixed some other way — e.g.
    re-provisioned Worker secrets.)
    """
    from rote.mcp.release import ReleaseUnavailable, release_parked_workflows

    def note(line: str) -> None:
        print(line, file=sys.stderr)

    # Under --json, `say` sends every line to stderr: stdout holds one object.
    tell = say or print
    try:
        report = release_parked_workflows(server)
    except ReleaseUnavailable as e:
        (say or note)(f"  note: {e}")
        return []
    for wf in report.released:
        tell(f"  released parked workflow {wf.workflow_id} ({wf.app})")
    for bc in report.broadcasts:
        tell(f"  broadcast release event {bc.event!r} → {bc.endpoint}")
    for app in report.skipped:
        (say or note)(f"  note: skipped registered app {app.app}: {app.reason}")
    return [wf.workflow_id for wf in report.released]


def _cmd_mcp_release(args: argparse.Namespace) -> int:
    """Release parked workflows without a login.

    For the paths where the credential is fixed out-of-band: Worker
    secrets re-provisioned via `rote mcp export` + `wrangler secret
    bulk`, a token file synced from another machine, or a server whose
    registry entry switched to static headers.
    """
    from rote.mcp.release import ReleaseUnavailable, release_parked_workflows

    try:
        report = release_parked_workflows(args.name)
    except ReleaseUnavailable as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    for wf in report.released:
        print(f"released parked workflow {wf.workflow_id} ({wf.app})")
    for bc in report.broadcasts:
        print(f"broadcast release event {bc.event!r} → {bc.endpoint}")
    for app in report.skipped:
        print(f"note: skipped registered app {app.app}: {app.reason}", file=sys.stderr)
    if not report.released and not report.broadcasts:
        print(f"rote mcp: no workflows parked waiting for {args.name!r}")
    return 0


def _cmd_mcp_parked(args: argparse.Namespace) -> int:
    """List workflows parked on MCP auth, without releasing them."""
    from rote.mcp.parked import list_parked_workflows
    from rote.mcp.release import ReleaseUnavailable

    try:
        report = list_parked_workflows(args.name)
    except ReleaseUnavailable as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if args.json:
        doc = {
            "parked": [
                {
                    "app": str(p.app),
                    "runtime": p.runtime,
                    "workflow_id": p.workflow_id,
                    "server": p.server,
                    "node": p.node,
                }
                for p in report.parked
            ],
            "skipped": [{"app": str(a.app), "reason": a.reason} for a in report.skipped],
        }
        print(json.dumps(doc, indent=2))
        return 0
    for p in report.parked:
        node = f" node {p.node!r}" if p.node else ""
        print(f"{p.server}: workflow {p.workflow_id}{node} ({p.runtime}, {p.app})")
    for app in report.skipped:
        print(f"note: skipped registered app {app.app}: {app.reason}", file=sys.stderr)
    if not report.parked:
        which = f" waiting for {args.name!r}" if args.name else ""
        print(f"rote mcp: no workflows parked{which}")
    else:
        servers = sorted({p.server for p in report.parked})
        print("release with: " + "; ".join(f"rote mcp login {s}" for s in servers))
    return 0


def _cmd_mcp_logout(args: argparse.Namespace) -> int:
    from rote.mcp import clear_token_file

    if clear_token_file(args.name):
        print(f"rote mcp: cleared stored credentials for {args.name!r}")
        return 0
    print(f"rote mcp: no stored credentials for {args.name!r}")
    return 0


def _cmd_mcp_export(args: argparse.Namespace) -> int:
    """Turn a completed login into deployable Worker secrets.

    Cloudflare Workers can't read the rote token store (no filesystem),
    so their emitted code refreshes access tokens at runtime from
    provisioned credentials. This prints them — refresh token, client
    id/secret, token endpoint, URL — as ``KEY=value`` lines (paste into
    ``.dev.vars`` for wrangler dev) or ``--json`` for
    ``npx wrangler secret bulk``. stdout carries secrets by design;
    pipe it, don't screenshot it.
    """
    import json as _json

    from rote.mcp import load_registry, read_token_file

    registry = load_registry()
    config = registry.servers.get(args.name)
    if config is None:
        print(f"error: no MCP server named {args.name!r}", file=sys.stderr)
        return 2
    doc = read_token_file(args.name) or {}
    tokens = doc.get("tokens") or {}
    client_info = doc.get("client_info") or {}
    refresh_token = tokens.get("refresh_token")
    token_endpoint = doc.get("token_endpoint")
    # id and secret from one source: the stored client, else the registry's.
    client_source = (
        client_info
        if client_info.get("client_id")
        else {
            "client_id": config.client_id,
            "client_secret": config.client_secret,
        }
    )
    client_id = client_source.get("client_id")
    if not refresh_token or not token_endpoint or not client_id:
        missing = [
            name
            for name, ok in [
                ("refresh token", refresh_token),
                ("token endpoint", token_endpoint),
                ("client id", client_id),
            ]
            if not ok
        ]
        print(
            f"error: cannot export {args.name!r} — missing {', '.join(missing)}. "
            f"Run: rote mcp login {args.name}",
            file=sys.stderr,
        )
        return 1
    upper = args.name.upper()
    secrets: dict[str, str] = {
        f"ROTE_MCP_{upper}_REFRESH_TOKEN": str(refresh_token),
        f"ROTE_MCP_{upper}_CLIENT_ID": str(client_id),
        f"ROTE_MCP_{upper}_TOKEN_ENDPOINT": str(token_endpoint),
        f"ROTE_MCP_{upper}_URL": str(doc.get("server_url") or config.url),
    }
    client_secret = client_source.get("client_secret")
    if client_secret:
        secrets[f"ROTE_MCP_{upper}_CLIENT_SECRET"] = str(client_secret)

    if args.json:
        print(_json.dumps(secrets, indent=2))
    else:
        for key, value in secrets.items():
            print(f"{key}={value}")
    print(
        f"rote mcp: exported {len(secrets)} secrets for {args.name!r} — deploy with "
        f"`rote mcp export {args.name} --json | npx wrangler secret bulk` "
        f"or paste into .dev.vars for wrangler dev. NOTE: the exported refresh "
        f"token is shared with this machine's store; if the server rotates "
        f"refresh tokens on use, dedicate a login to the Worker.",
        file=sys.stderr,
    )
    return 0


def _cmd_mcp_headers(args: argparse.Namespace) -> int:
    """Print fresh auth headers as JSON — the machine-facing token API.

    This is what Claude Code's ``headersHelper`` invokes (per
    connection, and again on a 401 retry), so it must always emit
    currently-valid credentials: static headers with each ``${NAME}``
    filled from the environment (the runtime helpers' rule), or the
    stored access token refreshed through the OAuth provider when
    stale. stdout is the contract; everything else goes to stderr.
    """
    import asyncio
    import json as _json

    from rote.mcp import McpAuthError, load_registry
    from rote.mcp._runtime_helper import RoteMcpConfigError, expand_headers
    from rote.mcp.auth import fresh_access_token

    registry = load_registry()
    config = registry.servers.get(args.name)
    if config is None:
        print(f"error: no MCP server named {args.name!r}", file=sys.stderr)
        return 2
    if config.headers:
        try:
            print(_json.dumps(expand_headers(args.name, config.headers)))
        except RoteMcpConfigError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        return 0
    try:
        token = asyncio.run(fresh_access_token(args.name, config))
    except McpAuthError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(_json.dumps({"Authorization": f"Bearer {token}"}))
    return 0


def _resolve_backend(command: str, flag: str | None, compiled_dir: Path) -> str | None:
    """``--backend`` for a command on an existing compiled IR, or ``None`` after an error.

    The flag wins; with none, the backend recorded at compile time
    (``backend.json``, :mod:`rote.compiler.backend`), else ``mcp``. A flag
    that contradicts the record is honored with a warning, since emitting
    the other backend drops code the compile wrote for this one.
    """
    from rote.compiler.backend import BACKEND_FILENAME, BackendRecordError, read_backend

    try:
        recorded = read_backend(compiled_dir)
    except BackendRecordError as e:
        print(f"error: {e}", file=sys.stderr)
        return None
    record = compiled_dir / BACKEND_FILENAME
    if flag is None:
        if recorded is not None:
            print(
                f"rote {command}: using --backend {recorded}, recorded when this pipeline "
                f"was compiled ({record})",
                file=sys.stderr,
            )
        return recorded or "mcp"
    if recorded is not None and flag != recorded:
        lost = (
            "the vendor-API modules the compile wrote and researched are not called for "
            "mcp-bound nodes"
            if recorded == "api"
            else "its mcp-bound nodes were compiled as tool calls, so their impl modules "
            "may be stubs"
        )
        print(
            f"warning: this pipeline was compiled for --backend {recorded} ({record}); "
            f"using --backend {flag} as asked: {lost}",
            file=sys.stderr,
        )
    return flag


def _mcp_requirements(
    pipeline: Pipeline, backend: str = "mcp", runtime: str | None = None
) -> list[dict[str, object]]:
    """Per required MCP server: the binding nodes, tools, and current auth state.

    ``backend="api"``, and the python and temporal runtimes, count no
    ``mcp:`` binding, since none is called
    (:func:`rote.compiler.api_docs.api_view`); an agent loop's servers
    still count.

    Requirement detection is pure IR (``Pipeline.required_mcp_servers``);
    the ``auth`` column reads the local registry + token store — no
    network. A server the registry doesn't know reports ``"not
    registered"``; registry/token read failures degrade to
    ``"registry unreadable"``
    rather than sinking the report (same posture as doctor). Nothing here
    gates anything — callers render recommendations, and park-on-auth
    remains the runtime backstop.
    """
    from rote.adapters._common import bindings_called
    from rote.mcp import auth_status, load_registry, read_token_file
    from rote.mcp.learned import learned_auth

    if not bindings_called(runtime or "", backend):
        from rote.compiler.api_docs import api_view

        pipeline = api_view(pipeline)
    required = pipeline.required_mcp_servers
    if not required:
        return []
    try:
        registry_servers = load_registry().servers
    except Exception:
        registry_servers = None

    report: list[dict[str, object]] = []
    for server, node_ids in required.items():
        tools = sorted(
            {n.mcp.tool for n in pipeline.nodes if n.mcp is not None and n.mcp.server == server}
        )
        if registry_servers is None:
            auth = "registry unreadable"
        elif server not in registry_servers:
            auth = "not registered"
        else:
            try:
                config = registry_servers[server]
                learned = learned_auth(server, config.url)
                auth = auth_status(
                    config, read_token_file(server), learned.auth if learned else None
                )
            except Exception:
                auth = "registry unreadable"
        report.append({"server": server, "nodes": node_ids, "tools": tools, "auth": auth})
    return report


#: Auth states that warrant a recommendation. "expired (refreshable)"
#: recovers unattended, and "static headers" and "no auth required" never
#: touch OAuth, so none of those do. "not registered" and "auth unknown"
#: are worded as unknown: a server that needs no auth runs fine there.
_MCP_AUTH_NEEDS_ACTION = frozenset(
    {"not registered", "auth unknown", "not authenticated", "expired"}
)


def _mcp_recommendation_lines(mcp_servers: list[dict[str, object]]) -> list[str]:
    """Non-blocking auth recommendations for servers that would park a run.

    Deliberately advisory: rote never hard-gates on auth (time-to-value),
    it tells you what to run so the pipeline won't park mid-flight.
    """
    lines: list[str] = []
    for entry in mcp_servers:
        auth = entry["auth"]
        if auth not in _MCP_AUTH_NEEDS_ACTION:
            continue
        server = entry["server"]
        if auth == "not registered":
            lines.append(
                f"  {server}: not in the rote registry, so whether it needs auth is "
                f"unknown. If it does, the run will park at its first {server} call. "
                f"Register it and check: rote mcp add {server} <url> && "
                f"rote mcp login {server}"
            )
        elif auth == "auth unknown":
            lines.append(
                f"  {server}: auth unknown (no token stored, and rote has not "
                f"recorded whether {server} needs one). If it does, the run will "
                f"park at its first {server} call. Check with: rote mcp login "
                f"{server} (records 'no auth required' if it needs none)"
            )
        else:
            lines.append(
                f"  {server}: {auth} — the run will park at its first {server} call; "
                f"recommended: rote mcp login {server}"
            )
    return lines


#: A ``tools/list`` failure that a login fixes, as the MCP client words it.
_AUTH_FAILURE = re.compile(r"RoteMcpAuthNeeded|Unauthorized|\b401\b")


def _unchecked_tool_lines(
    unchecked: dict[str, list[str]],
    mcp_servers: list[dict[str, object]],
    unlisted: dict[str, str] | None = None,
) -> list[str]:
    """The summary's warning for MCP calls whose arguments no schema checked.

    The parameter names are the compile's guess (a skill says "search
    Gmail", never that the argument is ``query``), and a real server
    refuses a wrong one, so this sits in the summary rather than only in
    the open questions. The fix depends on why there was no schema:
    the server is not registered, needs a login, was not listed, or
    listed tools that do not include this one.
    """
    auth = {entry["server"]: entry["auth"] for entry in mcp_servers}
    unlisted = unlisted or {}
    # A server the run does not call (python/temporal emit the impl) is not
    # this run's problem.
    unchecked = {server: tools for server, tools in unchecked.items() if server in auth}
    if not unchecked:
        return []
    calls = ", ".join(f"{server}.{tool}" for server, tools in unchecked.items() for tool in tools)
    lines = [
        f"  WARNING: tool arguments NOT checked against a schema: {calls}. The "
        "parameter names are a guess, and a real server refuses a wrong one."
    ]
    for server, tools in unchecked.items():
        why = unlisted.get(server)
        if auth[server] == "not registered":
            fix = (
                f"{server} is not registered, so rote could not read its tools. "
                f"Register it, then compile again: rote mcp add {server} <url>"
            )
        elif why is not None and _AUTH_FAILURE.search(why):
            fix = (
                f"{server} needs a login before rote can read its tools ({why}). "
                f"Log in, then compile again: rote mcp login {server}"
            )
        elif why is not None:
            fix = (
                f"rote did not read {server}'s tools: {why}. Fix that (or pass "
                "--tool-discovery all), then compile again"
            )
        else:
            fix = (
                f"{server}'s tools/list has no {', '.join(tools)}. Check the binding's "
                "tool name against the server"
            )
        lines.append(f"    {fix}")
    return lines


def _build_analysis(
    pipeline: Pipeline,
    driver_name: str,
    driver_metadata: dict[str, object],
    skill_path: Path,
) -> dict[str, object]:
    """Derive a structural report from a compiled pipeline's IR.

    Pure function of the IR plus the local MCP registry/token files — no
    model, no network. Step counting mirrors the eval scorecard exactly
    (top-level execution-wave nodes; loop_body sub-nodes are costed inside
    their parent loop, so they're excluded from the top-level count) so
    ``analyze``'s roteness equals ``eval``'s.
    """
    from rote.adapters import ADAPTERS
    from rote.adapters._common import _execution_waves
    from rote.ir import NodeKind

    top_level = [n for wave in _execution_waves(pipeline) for n in wave]
    total = len(top_level)

    by_kind = {k.value: 0 for k in NodeKind}
    for n in top_level:
        by_kind[n.kind.value] += 1

    sampled = by_kind[NodeKind.LLM_JUDGE.value] + by_kind[NodeKind.AGENT_LOOP.value]
    deterministic = total - sampled
    roteness = (deterministic / total) if total else 0.0

    loop_body_ids = sorted({b for n in pipeline.nodes if n.loop_body for b in n.loop_body})

    untargetable: dict[str, str] = {}
    if pipeline.requires_durable_execution:
        untargetable["python"] = (
            "pipeline parks on a human gate (hitl_gate); needs durable execution"
        )
    from rote.adapters.skill import refusal as skill_refusal

    skill_problem = skill_refusal(pipeline)
    if skill_problem:
        untargetable["skill"] = skill_problem
    targetable = [name for name in sorted(ADAPTERS) if name not in untargetable]

    return {
        "pipeline": pipeline.name,
        "version": pipeline.version,
        "description": pipeline.description,
        "driver": driver_name,
        "driver_metadata": driver_metadata,
        "source_skill": str(skill_path),
        "nodes": {
            "total": total,
            "by_kind": by_kind,
            "loop_body_subnodes": loop_body_ids,
        },
        "roteness": roteness,
        "deterministic_steps": deterministic,
        "sampled_steps": sampled,
        "mandatory": [n.id for n in top_level if n.mandatory],
        "hitl_gates": [
            {"id": n.id, "signal": n.signal} for n in top_level if n.kind is NodeKind.HITL_GATE
        ],
        "agent_loops": [
            {
                "id": n.id,
                "max_iterations": (n.termination.max_iterations if n.termination else None),
            }
            for n in top_level
            if n.kind is NodeKind.AGENT_LOOP
        ],
        "targetable_runtimes": targetable,
        "untargetable_runtimes": untargetable,
        "mcp_servers": _mcp_requirements(pipeline),
    }


def _render_analysis_text(report: dict[str, object]) -> str:
    """Render an :func:`_build_analysis` report as a scannable text block."""
    nodes = report["nodes"]
    assert isinstance(nodes, dict)
    by_kind: dict[str, int] = nodes["by_kind"]
    total = nodes["total"]

    lines = [
        f"rote analyze: {report['pipeline']} v{report['version']}",
        f"  driver: {report['driver']}",
        f"  source: {report['source_skill']}",
    ]
    description = report["description"]
    if isinstance(description, str) and description.strip():
        # Collapse internal whitespace/newlines to one line, then clip.
        flat = " ".join(description.split())
        lines.append(f"  {flat[:200] + '…' if len(flat) > 200 else flat}")

    lines.append("")
    lines.append(f"Nodes ({total} executed step{'s' if total != 1 else ''})")
    # Stable, readable kind order; only show kinds that occur.
    for kind in ("pure_function", "external_call", "llm_judge", "agent_loop", "hitl_gate"):
        count = by_kind.get(kind, 0)
        if count:
            lines.append(f"  {kind:<15} {count:>2}  {'█' * count}")
    subnodes = nodes["loop_body_subnodes"]
    assert isinstance(subnodes, list)
    if subnodes:
        lines.append(f"  (+{len(subnodes)} loop-body sub-node(s), costed inside their loop)")

    roteness = report["roteness"]
    assert isinstance(roteness, float)
    lines.append("")
    lines.append(
        f"Roteness: {roteness:.0%} deterministic "
        f"({report['deterministic_steps']} of {total} steps run as code, not inference)"
    )

    mandatory = report["mandatory"]
    assert isinstance(mandatory, list)
    if mandatory:
        lines.append(f"Mandatory checks (cannot be skipped): {', '.join(mandatory)}")

    hitl = report["hitl_gates"]
    assert isinstance(hitl, list)
    if hitl:
        gates = ", ".join(f"{g['id']} (signal: {g['signal']})" for g in hitl)
        lines.append(f"HITL gates: {gates}")

    loops = report["agent_loops"]
    assert isinstance(loops, list)
    if loops:
        rendered = ", ".join(
            f"{loop['id']} (max {loop['max_iterations']} iterations)"
            if loop["max_iterations"] is not None
            else str(loop["id"])
            for loop in loops
        )
        lines.append(f"Agent loops: {rendered}")

    mcp_servers = report["mcp_servers"]
    assert isinstance(mcp_servers, list)
    if mcp_servers:
        lines.append("")
        lines.append(f"Required MCP servers ({len(mcp_servers)})")
        for entry in mcp_servers:
            entry_nodes = entry["nodes"]
            assert isinstance(entry_nodes, list)
            lines.append(
                f"  {entry['server']:<12} {len(entry_nodes)} node(s): "
                f"{', '.join(entry_nodes)}  [{entry['auth']}]"
            )
        recommendations = _mcp_recommendation_lines(mcp_servers)
        if recommendations:
            lines.extend(recommendations)

    targetable = report["targetable_runtimes"]
    assert isinstance(targetable, list)
    lines.append("")
    lines.append(f"Targetable runtimes: {', '.join(targetable)}")
    untargetable = report["untargetable_runtimes"]
    assert isinstance(untargetable, dict)
    for name, reason in untargetable.items():
        lines.append(f"  {name}: unavailable — {reason}")

    return "\n".join(lines)


def _cmd_analyze(args: argparse.Namespace) -> int:
    """Dry-run the compiler: report a skill's compiled shape, emit nothing.

    The ``plan`` to ``compile``'s ``apply`` — runs the same compiler
    agent to produce a validated IR, then prints a structural report
    instead of emitting runtime code. With ``--out`` the compiled IR +
    stubs are kept (ready for a later ``rote emit``); without it, they're
    produced in a temp dir and discarded after reporting.
    """
    import json
    import tempfile
    from contextlib import nullcontext

    skill_path = Path(args.skill_path)
    if not skill_path.is_dir():
        print(f"error: skill path is not a directory: {skill_path}", file=sys.stderr)
        return 2

    # Same layered agent/model defaults as compile — analyze is its dry run.
    from rote import config as rote_config

    try:
        layers = rote_config.load_layers()
        agent = rote_config.resolve("agent", args.agent, layers=layers).value
        model = rote_config.resolve("model", args.model, layers=layers).value
        effort = rote_config.resolve("effort", args.effort, layers=layers).value
        decider = _local_decider(
            args, rote_config.resolve("decider", args.decider, layers=layers), "analyze"
        )
        tiebreak = _local_system1_choice(
            args,
            rote_config.resolve("tiebreak", args.tiebreak, layers=layers),
            "--tiebreak",
            DEFAULT_TIEBREAK,
        )
        structure = _local_system1_choice(
            args,
            rote_config.resolve("structure", args.structure, layers=layers),
            "--structure",
            DEFAULT_STRUCTURE,
        )
    except (rote_config.ConfigError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if args.no_mcp:
        try:
            agent = _no_mcp_driver(agent)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
    analyze_kwargs: dict[str, Any] = {}
    if args.no_mcp:
        analyze_kwargs["mcp_servers"] = []
    if effort is not None:
        analyze_kwargs["effort"] = effort
    compiler = Compiler(
        agent=agent,
        model=model,
        driver_kwargs=analyze_kwargs or None,
        run_generated_tests=not args.no_verify_tests,
        trust_skill=args.trust_skill,
        strict_provenance=args.strict_provenance,
        system1=args.system1,
        system1_decider=decider,
        system1_tiebreak=tiebreak,
        system1_structure=structure,
        max_parallel=args.max_parallel,
    )

    out_ctx: Any = (
        nullcontext(str(args.out))
        if args.out
        else tempfile.TemporaryDirectory(prefix="rote-analyze-")
    )
    with out_ctx as out_raw:
        compiled_dir = Path(out_raw)
        try:
            result = asyncio.run(compiler.compile(skill_path, compiled_dir))
        except CompilerError as e:
            print(f"rote analyze: {e}", file=sys.stderr)
            return 1
        except KeyboardInterrupt:
            print("rote analyze: interrupted", file=sys.stderr)
            return 130

        report = _build_analysis(
            result.pipeline, result.driver_name, result.driver_metadata, skill_path
        )
        # The compiled artifacts exist here even when they're about to be
        # discarded, so the contract check costs nothing and is exactly the
        # kind of thing a dry run should surface before you pay for `compile`.
        contract_findings = check_contracts(result.pipeline, compiled_dir)
        report["contract_findings"] = [f.as_dict() for f in contract_findings]
        deliverable = result.verification
        if deliverable is not None:
            report["provenance"] = deliverable.provenance_report()
        # analyze is a dry run and exits 0 on what it finds, except that
        # --strict-provenance (for CI) fails on any unresolved provenance
        # finding, which that flag has already made an error.
        strict_failed = (
            args.strict_provenance
            and deliverable is not None
            and any(f.node_id == PROVENANCE_NODE for f in deliverable.errors)
        )
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print(_render_analysis_text(report))
            for finding in contract_findings:
                label = "ERROR" if finding.severity == "error" else "warning"
                print(f"  contract: {label} {finding.node_id}: {finding.message}")
            if deliverable is not None:
                for line in _provenance_lines(deliverable, "analyze"):
                    print(line)
                if strict_failed:
                    for finding in deliverable.errors:
                        if finding.node_id == PROVENANCE_NODE:
                            print(f"  provenance: ERROR ({finding.kind}): {finding.message}")
            if args.out:
                print()
                print(f"  compiled IR kept at: {compiled_dir / 'pipeline.yaml'}")
                print(
                    f"  emit a runtime with:  rote emit {compiled_dir / 'pipeline.yaml'} "
                    f"--runtime dbos --out ./runtime"
                )
            else:
                print()
                print("  (report only — pass --out DIR to keep the pipeline.yaml for `rote emit`)")
        if strict_failed:
            print(
                "rote analyze: unresolved provenance findings (--strict-provenance)",
                file=sys.stderr,
            )
            return 1
    return 0


def _doctor_mcp_servers() -> list[dict[str, object]]:
    """Registered MCP servers with their auth status.

    Degrades to an empty list on any read failure — the registry and
    token store are documented as optionally-absent, so a missing or
    malformed store is informational, never a doctor traceback.
    """
    from rote.mcp import auth_status, load_registry, read_token_file
    from rote.mcp.learned import learned_auth

    try:
        registry = load_registry()
    except Exception:
        return []
    servers: list[dict[str, object]] = []
    for name, config in sorted(registry.servers.items()):
        try:
            doc = read_token_file(name)
        except Exception:
            doc = None
        try:
            learned = learned_auth(name, config.url)
        except Exception:
            learned = None
        servers.append(
            {
                "name": name,
                "url": config.url,
                "auth": auth_status(config, doc, learned.auth if learned else None),
                **_learned_fields(learned),
            }
        )
    return servers


def _learned_fields(learned: LearnedAuth | None) -> dict[str, object]:
    """The JSON fields describing what rote learned (or was told) about auth."""
    return {
        "auth_learned_at": learned.learned_at if learned else None,
        "auth_source": learned.source if learned else None,
    }


def _auth_label(server: dict[str, object]) -> str:
    """The auth column, with the date for a requirement rote learned.

    "no auth required" is a fact rote observed once, not a credential it
    can re-check, so it carries the date it was learned.
    """
    auth = str(server["auth"])
    learned = server.get("auth_learned_at")
    if auth == "no auth required" and learned:
        verb = "set" if server.get("auth_source") == "user" else "learned"
        return f"{auth} ({verb} {learned})"
    return auth


def _doctor_apps() -> list[dict[str, object]]:
    """Emitted apps rote has recorded, flagging any whose directory is gone.

    Degrades to an empty list on any read failure (missing/corrupt
    registry). Staleness is expected — a moved or deleted app dir shows
    ``exists: false`` rather than being dropped.
    """
    from rote.app_registry import registered_apps

    try:
        apps = registered_apps()
    except Exception:
        return []
    return [
        {
            "path": str(app.path),
            "runtime": app.runtime,
            "pipeline": app.pipeline,
            "exists": app.path.is_dir(),
        }
        for app in apps
    ]


def _build_doctor_report() -> dict[str, Any]:
    """Gather the full read-only preflight report (the ``--json`` payload)."""
    import importlib.util
    import platform

    drivers = [
        {"name": name, "available": available, "reason": reason}
        for name, available, reason in available_drivers()
    ]
    runtimes = [
        {"name": extra, "used_for": spec.used_for, "installed": is_installed(extra)}
        for extra, spec in EXTRAS.items()
    ]
    # Base dependencies a working install always has. Every MCP step and
    # login needs them, so one missing fails the check however many
    # drivers are available.
    dependencies = [
        {
            "module": module,
            "used_for": dep.used_for,
            "installed": importlib.util.find_spec(module) is not None,
        }
        for module, dep in BASE_MODULES.items()
    ]
    return {
        "version": __version__,
        "python": platform.python_version(),
        "drivers": drivers,
        "dependencies": dependencies,
        "runtimes": runtimes,
        "mcp_servers": _doctor_mcp_servers(),
        "apps": _doctor_apps(),
        "ok": any(bool(d["available"]) for d in drivers)
        and all(bool(d["installed"]) for d in dependencies),
    }


def _render_doctor_text(report: dict[str, Any]) -> str:
    """Render the doctor report as a scannable ✓/✗ checklist."""
    lines: list[str] = []
    lines.append("rote doctor — preflight (read-only; costs nothing)")
    lines.append("")
    lines.append(f"  rote {report['version']}  ·  Python {report['python']}")
    lines.append("")

    # The driver and the extra share the name "api"; the labels say which
    # is which ("api driver" uses your key, "[api] extra" is a package set).
    lines.append("Compiler drivers (at least one required):")
    for driver in report["drivers"]:
        mark = "✓" if driver["available"] else "✗"
        suffix = "" if driver["available"] else f" — {driver['reason']}"
        lines.append(f"  {mark} {driver['name']} driver{suffix}")
    lines.append("  note: CLI subscription auth (claude/codex) is only verified at run time.")
    lines.append("")

    lines.append("Dependencies (always installed with rote-cli):")
    for dep in report["dependencies"]:
        if dep["installed"]:
            lines.append(f"  ✓ {dep['module']} ({dep['used_for']})")
        else:
            lines.append(f"  ✗ {dep['module']} ({dep['used_for']}) — missing")
    lines.append("")

    lines.append("Optional extras (installed on demand; these do not fail this check):")
    for runtime in report["runtimes"]:
        used_for = f" ({runtime['used_for']})" if runtime.get("used_for") else ""
        if runtime["installed"]:
            lines.append(f"  ✓ [{runtime['name']}] extra{used_for}")
        else:
            lines.append(
                f"  ✗ [{runtime['name']}] extra{used_for} — {install_hint(runtime['name'])}"
            )
    lines.append("")

    lines.append("MCP servers:")
    servers = report["mcp_servers"]
    if not servers:
        lines.append("  (none registered)")
    else:
        width = max(len(str(s["name"])) for s in servers)
        for server in servers:
            healthy = server["auth"] in ("authenticated", "static headers", "no auth required")
            mark = "✓" if healthy else "✗"
            name = str(server["name"]).ljust(width)
            lines.append(f"  {mark} {name}  {_auth_label(server):<34}  {server['url']}")
    lines.append("")

    lines.append("Registered apps:")
    apps = report["apps"]
    if not apps:
        lines.append("  (none registered)")
    else:
        for app in apps:
            mark = "✓" if app["exists"] else "✗"
            suffix = "" if app["exists"] else "  (directory missing)"
            lines.append(f"  {mark} {app['path']}  [{app['runtime']}] {app['pipeline']}{suffix}")
    lines.append("  (MCP and app issues are informational — they do not fail this check.)")
    lines.append("")

    missing = [str(d["module"]) for d in report["dependencies"] if not d["installed"]]
    if missing:
        lines.append(f"✗ this install is missing {', '.join(missing)}: {reinstall_hint()}")
    elif report["ok"]:
        lines.append("✓ ready to compile")
    else:
        lines.append("✗ no compiler driver available")
    return "\n".join(lines)


def _resolve_baseline_input(
    skill_path: Path, input_arg: str | None, *, assume_yes: bool, out_dir: Path
) -> dict[str, Any] | None:
    """The baseline's task payload: from ``--input``, or derived + confirmed.

    Derivation is a guess about how the skill is invoked, so it is never
    spent against silently: the proposal is printed and must be accepted —
    interactively on a terminal, or via ``--yes`` for non-interactive
    callers. Either way the proposal is saved to
    ``baseline/derived-input.json`` so a declined run can be edited and
    re-invoked with ``--input``. Returns ``None`` when the user declines
    (callers exit 0); raises ``ValueError`` for usage errors.
    """
    import json

    from rote.eval.baseline import (
        BASELINE_DIRNAME,
        DERIVED_INPUT_FILENAME,
        derive_input_payload,
    )

    if input_arg is not None:
        input_path = Path(input_arg)
        if not input_path.is_file():
            raise ValueError(f"--input file not found: {input_path}")
        try:
            payload = json.loads(input_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(f"--input is not valid JSON: {e}") from e
        if not isinstance(payload, dict):
            raise ValueError("--input must be a JSON object")
        return payload

    print(
        "no --input given — deriving a representative task from SKILL.md (cheap single-shot call)…",
        file=sys.stderr,
    )
    derived = derive_input_payload(skill_path)
    derived_path = out_dir / BASELINE_DIRNAME / DERIVED_INPUT_FILENAME
    derived_path.parent.mkdir(parents=True, exist_ok=True)
    derived_path.write_text(json.dumps(derived, indent=2) + "\n", encoding="utf-8")

    print("derived input proposal:", file=sys.stderr)
    print(json.dumps(derived, indent=2), file=sys.stderr)
    if assume_yes:
        print("--yes: accepting derived input", file=sys.stderr)
        return derived
    if not sys.stdin.isatty():
        raise ValueError(
            "not a terminal and no --input given — pass --input FILE, or --yes "
            f"to accept the derived proposal (saved to {derived_path})"
        )
    answer = input(f"Run the baseline with this input? (saved to {derived_path}) [y/N] ")
    if answer.strip().lower() not in {"y", "yes"}:
        print(
            f"declined — edit {derived_path} and re-run with --input {derived_path}",
            file=sys.stderr,
        )
        return None
    return derived


def _cmd_baseline(args: argparse.Namespace) -> int:
    """Run the raw skill as an agent, measured and observed (pre-compilation).

    The three-artifacts-in-one-run command: measured "before" metrics,
    the full stream-json transcript, and the observed MCP tool traffic
    (real inputs and payloads). See :mod:`rote.eval.baseline`.
    """
    import json

    from rote.compiler.drivers.claude import DEFAULT_MODEL
    from rote.eval.baseline import BASELINE_DIRNAME, run_baseline
    from rote.eval.empirical import EmpiricalError, measured_run_record

    skill_path = Path(args.skill_path)
    if not skill_path.is_dir():
        print(f"error: skill path is not a directory: {skill_path}", file=sys.stderr)
        return 2
    try:
        maybe_payload = _resolve_baseline_input(
            skill_path, args.input, assume_yes=args.yes, out_dir=Path(args.out)
        )
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except EmpiricalError as e:
        print(f"rote baseline: {e}", file=sys.stderr)
        return 1
    if maybe_payload is None:
        return 0
    input_payload = maybe_payload

    try:
        result = run_baseline(
            skill_path,
            input_payload,
            args.out,
            model=args.model or DEFAULT_MODEL,
            trials=args.trials,
            allow_writes=args.allow_writes,
            max_turns=args.max_turns,
        )
    except EmpiricalError as e:
        print(f"rote baseline: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("rote baseline: interrupted", file=sys.stderr)
        return 130

    baseline_dir = Path(args.out) / BASELINE_DIRNAME
    ok_runs = [r for r in result.runs if r.succeeded]

    summary_stream = sys.stderr if args.json else sys.stdout
    print(
        f"rote baseline: {len(ok_runs)}/{len(result.runs)} trial(s) succeeded "
        f"({'read-only MCP gate' if result.read_only else 'writes allowed'})",
        file=summary_stream,
    )
    for r in result.runs:
        note = f" [{', '.join(r.flags)}]" if r.flags else ""
        cost = f"${r.cost_usd:.2f}" if r.cost_usd is not None else "cost n/a"
        turns = f"{r.turns} turns" if r.turns is not None else "turns n/a"
        print(f"  {r.wall_seconds:.0f}s, {turns}, {cost}{note}", file=summary_stream)
        _print_skill_run_gaps(r, summary_stream)
    if result.observations:
        print(
            f"  observed MCP traffic: {len(result.observations)} call(s) across "
            f"{', '.join(result.observed_servers)}",
            file=summary_stream,
        )
    else:
        print("  observed MCP traffic: none", file=summary_stream)
    for server, reason in result.servers_skipped.items():
        print(f"  {server}: {reason}", file=summary_stream)
    print(f"  artifacts: {baseline_dir}", file=summary_stream)

    if args.json:
        payload = {
            "skill_dir": str(result.skill_dir),
            "model": result.model,
            "read_only": result.read_only,
            "runs": [measured_run_record(r) for r in result.runs],
            "observed_servers": result.observed_servers,
            "observed_calls": len(result.observations),
            "servers_wired": list(result.servers_wired),
            "servers_skipped": result.servers_skipped,
            "baseline_dir": str(baseline_dir.resolve()),
        }
        print(json.dumps(payload, indent=2))
    return 0


def _export_inference_selection(flag_value: str | None) -> None:
    """Publish the resolved judge provider into the environment.

    ``ROTE_INFERENCE`` is the emitted app's own interface — a generated
    pipeline reads it whether rote launched it or a user did — so the
    CLI's job here is only to resolve the layered default and export it,
    never to pass a provider through some private channel. Runners that
    spawn a subprocess inherit it; the in-process ones read the same
    ``os.environ``.

    Nothing resolved means nothing exported: the app auto-detects.
    """
    from rote import config as rote_config

    resolved = rote_config.resolve("inference", flag_value, layers=rote_config.load_layers())
    if resolved.value is not None:
        os.environ["ROTE_INFERENCE"] = resolved.value


def _judge_price_catalog() -> PricingCatalog:
    """The live price catalog for judge usage (a seam tests patch offline)."""
    from rote.eval.pricing import fetch_catalog

    return fetch_catalog(provider="anthropic")


def _judge_cost_text(rows: Sequence[Mapping[str, Any]]) -> str:
    """``"$0.0060"``, or why there is no dollar figure.

    The price catalog is fetched only when a row lacks the CLI's own
    estimate. No figure is printed when any row is unpriced: a partial sum
    reads as the whole cost.
    """
    from rote.eval.empirical import judge_usage_cost_usd
    from rote.eval.pricing import PricingError

    catalog: PricingCatalog | None = None
    offline = False
    if any(r.get("cost_usd") is None for r in rows):
        try:
            catalog = _judge_price_catalog()
        except PricingError:
            offline = True
    total, unpriced = judge_usage_cost_usd(rows, catalog)
    if offline:
        return "cost unknown (price catalog unreachable)"
    if unpriced:
        return f"cost unknown (no published price for {', '.join(sorted(unpriced))})"
    return f"${total:.4f}"


def _cmd_live_check(args: argparse.Namespace) -> int:
    """``rote run <compile dir> --live-check --input ...`` (rote.vendor_specs.live)."""
    from rote.compiler.backend import read_backend
    from rote.vendor_specs.live import LiveCheckRefused, live_check

    root = Path(args.path)
    compiled = root / "compiled" if (root / "compiled" / "pipeline.yaml").is_file() else root
    if not (compiled / "pipeline.yaml").is_file():
        print(f"error: {root} holds no compiled pipeline.yaml", file=sys.stderr)
        return 2
    if read_backend(compiled) != "api":
        print(
            "error: --live-check checks the vendor calls of a `--backend api` compile, and "
            f"{compiled} was not compiled for it (backend.json)",
            file=sys.stderr,
        )
        return 2
    if args.input is None:
        print("error: --live-check needs --input (a JSON object or a file)", file=sys.stderr)
        return 2
    raw = Path(args.input).read_text(encoding="utf-8") if Path(args.input).is_file() else args.input
    payload = json.loads(raw)

    def confirm(plan: list[Any]) -> bool:
        print(f"rote run: the live check will send {len(plan)} request(s):", file=sys.stderr)
        for p in plan:
            carries = f", with {p.credential}" if p.credential else ", with no credential"
            print(f"  {p.method} {p.host}  {p.operation}{carries}", file=sys.stderr)
        if args.yes:
            return True
        if not sys.stdin.isatty():
            print("rote run: pass --yes to send them (no terminal to ask)", file=sys.stderr)
            return False
        return input("Send them? [y/N] ").strip().lower() in ("y", "yes")

    try:
        report = live_check(
            compiled.resolve(),
            payload,
            confirm=confirm,
            allow_hosts=frozenset(args.allow_host or ()),
            allow_credentials=frozenset(args.allow_credential or ()),
        )
    except LiveCheckRefused as e:
        print(f"rote run: live check refused: {e}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
    else:
        for c in report.checked:
            where = f"{c.method} {c.host}{c.path}"
            print(f"{c.node_id}: {where} -> {c.status} ({c.operation or 'unknown operation'})")
            for path in c.reads_missing:
                print(f"  the code reads {path}; the live response does not have it")
            for line in c.schema_drift:
                print(f"  spec drift: {line}")
            for path in c.spec_only:
                print(f"  the spec has {path}; the live response did not return it")
            for path in c.fixture_only:
                print(f"  fixtures have {path}; the live response did not")
            for path in c.live_only:
                print(f"  live response has {path}; no stored fixture does")
        for node, why in report.skipped.items():
            print(f"{node}: not called ({why})")
        for where, why in report.unchecked.items():
            print(f"{where}: not live-checked ({why})")
        for where, error in report.errors.items():
            print(f"{where}: failed ({error})")
    return 1 if report.drifted or report.errors else 0


def _print_skill_run_gaps(run: Any, stream: Any) -> None:
    """Say what a skill run did not get: servers it never called, tools refused.

    One wording for ``rote run``, ``rote baseline`` and ``compile
    --baseline``. A run without a server the skill names still writes a
    result, usually on data the agent substituted, so this is never silent.
    """
    for server, why in run.missing_mcp_servers.items():
        print(f"  missing MCP server {server}: {why}", file=stream)
    if run.denied_tools:
        refused = ", ".join(sorted(set(run.denied_tools)))
        print(f"  refused by the gate: {refused}", file=stream)


def _cmd_run(args: argparse.Namespace) -> int:
    """One-off local execution of a skill or an emitted pipeline.

    The skill side is a baseline-grade ``claude -p`` run without the
    artifact ceremony; the pipeline side executes the emitted app
    (python script, or DBOS app with HITL gate payloads delivered
    cross-process). The run's output JSON goes to stdout so it pipes;
    status lines go to stderr.
    """
    if args.live_check:
        return _cmd_live_check(args)
    from rote.compiler.drivers.claude import DEFAULT_MODEL
    from rote.eval.empirical import EmpiricalError, measured_run_record
    from rote.ir import NodeKind
    from rote.runners import (
        TargetError,
        detect_target,
        parse_signal_args,
        preflight_pipeline,
        resolve_gate_signals,
        resolve_input,
        run_pipeline,
        run_skill,
    )

    try:
        target = detect_target(Path(args.path), runtime=args.runtime)
    except TargetError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    try:
        if target.kind == "skill":
            maybe_payload = resolve_input(args.input, skill_dir=target.path, assume_yes=args.yes)
            if maybe_payload is None:
                return 0
            outcome = run_skill(
                target.path,
                maybe_payload,
                model=args.model or DEFAULT_MODEL,
                allow_writes=args.allow_writes,
                signals=parse_signal_args(args.signal),
                max_turns=args.max_turns,
                timeout_seconds=args.timeout or 1800.0,
            )
            run = outcome.run
            gate_note = "read-only MCP gate" if outcome.read_only else "writes allowed"
            if not run.succeeded:
                status = f"failed: {run.error}"
            elif run.missing_mcp_servers:
                status = "degraded: it never called MCP server(s) the skill names"
            else:
                status = "succeeded"
            print(f"rote run (skill): {status} ({gate_note})", file=sys.stderr)
            _print_skill_run_gaps(run, sys.stderr)
            cost = f"${run.cost_usd:.2f}" if run.cost_usd is not None else "cost n/a"
            turns = f"{run.turns} turns" if run.turns is not None else "turns n/a"
            print(f"  {run.wall_seconds:.0f}s, {turns}, {cost}", file=sys.stderr)
            if outcome.observations:
                print(
                    f"  observed MCP traffic: {len(outcome.observations)} call(s) across "
                    f"{', '.join(outcome.observed_servers)}",
                    file=sys.stderr,
                )
            for server, reason in outcome.servers_skipped.items():
                print(f"  {server}: {reason}", file=sys.stderr)
            if args.json:
                print(
                    json.dumps(
                        {
                            "kind": "skill",
                            "skill_dir": str(target.path),
                            "model": run.model,
                            "read_only": outcome.read_only,
                            "run": measured_run_record(run),
                            "output": run.output,
                            "observed_servers": outcome.observed_servers,
                        },
                        indent=2,
                    )
                )
            elif run.output is not None:
                print(json.dumps(run.output, indent=2))
            if not run.succeeded:
                return 1
            if run.missing_mcp_servers and not args.allow_missing_servers:
                print(
                    "rote run (skill): exiting 1 because the output may rest on "
                    "substituted data; pass --allow-missing-servers to accept it",
                    file=sys.stderr,
                )
                return 1
            return 0

        # Before input derivation (an LLM call) and gate prompts: a run
        # that cannot start should say so before asking for anything. A
        # missing runtime extra is offered as an install first; preflight
        # still refuses one that stayed missing.
        code = _ensure_runtime_extra(args, target.runtime, f"running a {target.runtime} app")
        if code is not None:
            return code
        preflight_pipeline(target, allow_stubs=args.allow_stubs)
        _export_inference_selection(args.inference)

        pipeline: Pipeline | None = None
        if target.pipeline_yaml is not None:
            pipeline = load_pipeline(target.pipeline_yaml)
        skill_dir = (
            _resolve_eval_skill_dir(None, target.pipeline_yaml, pipeline.source_skill)
            if pipeline is not None and target.pipeline_yaml is not None
            else None
        )
        maybe_payload = resolve_input(args.input, skill_dir=skill_dir, assume_yes=args.yes)
        if maybe_payload is None:
            return 0
        signals = parse_signal_args(args.signal)
        gate_order: list[str] = []
        if pipeline is not None:
            from rote.adapters._common import _execution_waves

            # Topological order — parked-then-send runtimes (cloudflare)
            # deliver gate payloads in this order. Any loop-body gate
            # falls back to node-list order at the end.
            gate_order = [
                n.signal
                for wave in _execution_waves(pipeline)
                for n in wave
                if n.kind is NodeKind.HITL_GATE and n.signal
            ]
            gate_order += [
                n.signal
                for n in pipeline.nodes
                if n.kind is NodeKind.HITL_GATE and n.signal and n.signal not in gate_order
            ]
            signals = resolve_gate_signals(gate_order, signals, interactive=sys.stdin.isatty())
        # No IR located (bare emitted dir): --signal payloads pass through
        # unvalidated — the app itself is the contract then.
        p_outcome = run_pipeline(
            target,
            maybe_payload,
            allow_writes=args.allow_writes,
            signals=signals or None,
            gate_order=gate_order or None,
            pipeline=pipeline,
            timeout_seconds=args.timeout or 600.0,
            # A browser login is only offered with a human on both ends.
            interactive=sys.stdin.isatty() and sys.stderr.isatty(),
        )
        run = p_outcome.run
        gate_note = "read-only MCP gate" if p_outcome.read_only else "writes allowed"
        # A failure's message stays the whole line (an app's one-line error).
        status = f"succeeded ({gate_note})" if run.succeeded else f"failed: {run.error}"
        print(f"rote run ({p_outcome.runtime}): {status}", file=sys.stderr)
        print(f"  {run.wall_seconds:.1f}s", file=sys.stderr)
        if run.judge_usage:
            # Cache buckets fold into "in": a prompt-cached claude -p judge
            # reports ~3 plain input tokens beside ~1k-30k cached ones.
            tokens_in = sum(
                (r.get("input_tokens") or 0)
                + (r.get("cache_write_tokens") or 0)
                + (r.get("cache_read_tokens") or 0)
                for r in run.judge_usage
            )
            tokens_out = sum(r.get("output_tokens") or 0 for r in run.judge_usage)
            print(
                f"  judge usage: {len(run.judge_usage)} call(s), "
                f"{tokens_in} in / {tokens_out} out tokens, {_judge_cost_text(run.judge_usage)}",
                file=sys.stderr,
            )
        if args.json:
            print(
                json.dumps(
                    {
                        "kind": "pipeline",
                        "runtime": p_outcome.runtime,
                        "app_dir": str(p_outcome.app_dir),
                        "read_only": p_outcome.read_only,
                        "wall_seconds": run.wall_seconds,
                        "output": run.output,
                        "error": run.error,
                        "judge_usage": list(run.judge_usage),
                    },
                    indent=2,
                )
            )
        elif run.output is not None:
            print(json.dumps(run.output, indent=2))
        return 0 if run.succeeded else 1
    except TargetError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except EmpiricalError as e:
        print(f"rote run: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("rote run: interrupted", file=sys.stderr)
        return 130


def _refuse_local_only_inference() -> str | None:
    """Refuse to deploy a pipeline configured to judge via ``claude -p``.

    The refusal belongs here, not at emit time: the provider is
    deliberately *not* in the IR, so an emitted artifact is supposed to
    run on a subscription locally and on a key or the tenant account in
    production, with no re-emit and no diff. Emit cannot know where the
    directory is headed. Deploy can, and it is the last moment before a
    personal Claude login would become a production dependency in a
    container that has no ``claude`` binary anyway.
    """
    from rote import config as rote_config
    from rote.inference import LOCAL_ONLY_PROVIDERS

    try:
        resolved = rote_config.resolve("inference", None, layers=rote_config.load_layers())
    except rote_config.ConfigError as e:
        return str(e)
    if resolved.value not in LOCAL_ONLY_PROVIDERS:
        return None
    return (
        f"inference is set to `{resolved.value}` ({resolved.source}), which only "
        f"works on a machine with your Claude login — a deployed pipeline has no "
        f"`claude` binary, and a personal subscription credential should not "
        f"become a production dependency. Set `inference: api` or "
        f"`inference: rote-cloud` (or ROTE_INFERENCE) before deploying; the "
        f"emitted code itself needs no change."
    )


def _cmd_deploy(args: argparse.Namespace) -> int:
    """Deploy an emitted pipeline to its hosting target.

    Push targets get a thin wrapper over the vendor CLI (wrangler /
    dbos-cloud own auth and output; rote adds detection + preflight).
    Runtimes with no push-deploy get honest, current guidance instead of
    a fake action — see :mod:`rote.deploy`.
    """
    from rote.deploy import (
        DeployError,
        DeployReport,
        deploy_cloudflare,
        deploy_dbos_cloud,
        guidance_for,
        resolve_deploy_target,
    )
    from rote.runners import TargetError, detect_target

    try:
        target = detect_target(Path(args.path), runtime=args.runtime)
        if target.kind != "pipeline":
            print(
                "error: deploy takes an emitted runtime dir (or a `rote compile "
                "--out` dir) — a skill isn't deployable, compile it first",
                file=sys.stderr,
            )
            return 2
        assert target.runtime is not None
        if (refusal := _refuse_local_only_inference()) is not None:
            print(f"error: {refusal}", file=sys.stderr)
            return 2
        deploy_target = resolve_deploy_target(args.target, target.runtime)

        if deploy_target is None:
            print(f"rote deploy ({target.runtime}): no push-deploy target exists —")
            print(guidance_for(target.runtime))
            if args.json:
                print(
                    json.dumps(
                        {
                            "target": None,
                            "runtime": target.runtime,
                            "app_dir": str(target.path),
                            "ok": True,
                            "action": "guidance",
                        },
                        indent=2,
                    )
                )
            return 0

        report: DeployReport
        if deploy_target == "cloudflare":
            report = deploy_cloudflare(target, dry_run=args.dry_run)
        elif deploy_target == "rote-cloud":
            from rote.deploy_rote_cloud import deploy_rote_cloud

            input_example = None
            if args.input_example:
                try:
                    input_example = json.loads(args.input_example)
                except json.JSONDecodeError as e:
                    print(f"error: --input-example is not valid JSON: {e}", file=sys.stderr)
                    return 2
            report = deploy_rote_cloud(
                target, url=args.url, token=args.token, input_example=input_example
            )
        else:  # dbos-cloud
            report = deploy_dbos_cloud(target)
    except TargetError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except DeployError as e:
        print(f"rote deploy: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("rote deploy: interrupted", file=sys.stderr)
        return 130

    status = "ok" if report.ok else f"failed: {report.detail}"
    print(f"rote deploy ({report.target}): {report.action} — {status}", file=sys.stderr)
    if args.json:
        print(
            json.dumps(
                {
                    "target": report.target,
                    "runtime": report.runtime,
                    "app_dir": str(report.app_dir),
                    "ok": report.ok,
                    "action": report.action,
                    "detail": report.detail,
                },
                indent=2,
            )
        )
    return 0 if report.ok else 1


def _cmd_login(args: argparse.Namespace) -> int:
    """Connect the CLI to a rote-cloud account (OAuth device flow).

    One grant path serves both UXes: with a browser we open the
    verification URL (code pre-filled, one Approve click); with
    ``--device`` or no browser the printed code + URL do the same job
    over SSH. Either way the CLI ends up holding a tenant ``rote_…``
    API key in the credential store — never a session token.
    """
    from rote.cloud_auth import LoginError, login

    try:
        login(args.url, open_browser=not args.device)
    except LoginError as e:
        print(f"rote login: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("rote login: interrupted", file=sys.stderr)
        return 130
    return 0


def _cmd_logout(args: argparse.Namespace) -> int:
    from rote.cloud_auth import LoginError, clear_credential, load_credential, revoke_key

    cred = load_credential()
    if cred is None:
        print("rote logout: not logged in", file=sys.stderr)
        return 0
    revoked = True
    try:
        revoke_key(cred)
    except LoginError as e:
        # The local credential still gets cleared: an unreachable
        # platform must not trap the user in a logged-in state. The
        # server-side key outlives this, so say so loudly.
        revoked = False
        print(
            f"rote logout: warning — {e}\n"
            "  the key is still active server-side; revoke it in the "
            "dashboard (profile → API keys)",
            file=sys.stderr,
        )
    clear_credential()
    print(
        f"rote logout: ✓ logged out of {cred.url}"
        + ("" if revoked else " (local credential cleared)"),
        file=sys.stderr,
    )
    return 0


def _cmd_whoami(args: argparse.Namespace) -> int:
    """Who the stored credential is, verified live against ``/v1/me``."""
    from rote.cloud_auth import LoginError, fetch_me, load_credential

    cred = load_credential()
    if cred is None:
        print("rote whoami: not logged in — run `rote login`", file=sys.stderr)
        return 1
    try:
        me = fetch_me(cred)
    except LoginError as e:
        print(f"rote whoami: {e}", file=sys.stderr)
        return 1
    user = str(me.get("user") or cred.user or "(unknown user)")
    tenant = str(me.get("tenant", ""))
    if args.json:
        print(json.dumps({"url": cred.url, "user": user, "tenant": tenant}, indent=2))
    else:
        line = f"rote whoami: {user} @ {cred.url}"
        if tenant:
            line += f" (tenant: {tenant})"
        print(line)
    return 0


def _cmd_init(args: argparse.Namespace) -> int:
    """Onboarding: choose the defaults once, save them.

    Interactive in a terminal. Otherwise every question takes its flag, or
    its default under ``--yes``, and a run missing one stops naming it
    (``rote.interact``): exit 2, nothing written.
    """
    from rote.init_wizard import InitError, InitFlags, build_plan
    from rote.interact.mode import Mode
    from rote.interact.output import Output
    from rote.interact.plan import (
        FORCE_FLAG,
        ConfirmationRequired,
        apply_plan,
        confirm_plan,
    )
    from rote.interact.prompter import PromptCancelled, PromptRequired, prompter_for
    from rote.interact.report import EXIT_ACTION_REQUIRED, action_required

    try:
        mode = Mode.resolve(json=args.json, yes=args.yes, agent_mode=args.agent_mode)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    out = Output(mode)
    prompter = prompter_for(mode)
    flags = InitFlags(
        deploy=args.deploy,
        runtime=args.runtime,
        agent=args.agent,
        model=args.model,
        structure=args.structure,
        decider=args.decider,
        tiebreak=args.tiebreak,
        login=args.login,
    )
    argv: list[str] = getattr(args, "raw_argv", sys.argv[1:])
    try:
        plan = build_plan(flags, prompter=prompter, out=out, project=args.project)
        out.plan(plan)
        granted = {FORCE_FLAG} if args.force else set()
        if not confirm_plan(plan, prompter, yes=mode.yes, granted=granted):
            out.say("rote init: nothing was written")
            out.result({"ok": False, "applied": False, "plan": plan.to_dict()})
            return 1
    except (PromptRequired, ConfirmationRequired) as e:
        out.error(str(e))
        out.result(action_required(e, argv))
        return EXIT_ACTION_REQUIRED
    except InitError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except PromptCancelled:
        out.say("rote init: cancelled; nothing was written")
        return 130
    outcomes = apply_plan(plan)
    for outcome in outcomes:
        line = f"  {outcome.status}: {outcome.action_id}"
        if outcome.error:
            line += f" ({outcome.error})"
        out.say(line)
    ok = all(o.status == "done" for o in outcomes)
    if ok:
        out.say("  see the effective setup any time with: rote config")
        out.say("  next: rote compile <skill-dir> --out ./compiled")
    out.result(
        {
            "ok": ok,
            "applied": True,
            "plan": plan.to_dict(),
            "outcomes": [o.to_dict() for o in outcomes],
        }
    )
    return 0 if ok else 1


def _cmd_config(args: argparse.Namespace) -> int:
    """Show every configurable default, its effective value, and its source."""
    from rote import config as rote_config
    from rote.cloud_auth import load_credential
    from rote.compiler.drivers.claude import DEFAULT_EFFORT as _COMPILER_DEFAULT_EFFORT

    try:
        layers = rote_config.load_layers()
        resolved = {
            key.name: rote_config.resolve(key.name, None, layers=layers)
            for key in rote_config.CONFIG_KEYS
            if key.kind == "str"
        }
        lists = {
            key.name: rote_config.resolve_list(key.name, layers=layers)
            for key in rote_config.CONFIG_KEYS
            if key.kind == "list"
        }
    except rote_config.ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    cred = load_credential()
    # Mirror compile's built-in resolution so this report shows what a
    # flagless `rote compile` would actually do, not just raw file
    # contents.
    runtime = resolved["runtime"].value or ("cloudflare" if cred is not None else "dbos")
    deploy = resolved["deploy"].value or (
        "rote-cloud" if cred is not None and runtime == "cloudflare" else "none"
    )
    login_note = "logged in" if cred is not None else "logged out"
    effective: dict[str, tuple[str, str]] = {
        "runtime": (runtime, resolved["runtime"].source)
        if resolved["runtime"].value
        else (runtime, f"built-in default ({login_note})"),
        "deploy": (deploy, resolved["deploy"].source)
        if resolved["deploy"].value
        else (deploy, f"built-in default ({login_note})"),
        "agent": (resolved["agent"].value or "auto-detect", resolved["agent"].source),
        "model": (resolved["model"].value or "driver default", resolved["model"].source),
        "effort": (
            resolved["effort"].value or f"{_COMPILER_DEFAULT_EFFORT} (claude driver)",
            resolved["effort"].source,
        ),
        "decider": (resolved["decider"].value or "auto", resolved["decider"].source),
        "tiebreak": (
            resolved["tiebreak"].value or DEFAULT_TIEBREAK,
            resolved["tiebreak"].source,
        ),
        "structure": (
            resolved["structure"].value or DEFAULT_STRUCTURE,
            resolved["structure"].source,
        ),
        "inference": (
            resolved["inference"].value or "auto-detect",
            resolved["inference"].source,
        ),
    }
    work = lists["work_domains"]
    effective["work_domains"] = (
        ", ".join(work.values)
        if work.values
        else ("none" if work.is_set else "not set (rote mcp import asks once)"),
        work.source,
    )

    if args.json:
        # Machine consumers get real values or null, never display prose:
        # runtime/deploy are always concrete (built-in resolution applies),
        # agent/model stay null when nothing configured them.
        json_values: dict[str, str | None] = {
            "runtime": runtime,
            "deploy": deploy,
            "agent": resolved["agent"].value,
            "model": resolved["model"].value,
            "effort": resolved["effort"].value,
            # Concrete like runtime: unset means rules, which is what runs.
            "decider": resolved["decider"].value or "auto",
            "tiebreak": resolved["tiebreak"].value or DEFAULT_TIEBREAK,
            "structure": resolved["structure"].value or DEFAULT_STRUCTURE,
            "inference": resolved["inference"].value,
        }
        list_values: dict[str, list[str] | None] = {
            name: list(r.values) if r.is_set else None for name, r in lists.items()
        }
        payload = {
            "user_config": {
                "path": str(layers.user_path),
                "present": layers.user_path.is_file(),
            },
            "project_config": {
                "path": str(layers.project_path) if layers.project_path else None,
                "present": layers.project_path is not None,
            },
            "settings": {
                name: {
                    "value": list_values[name] if name in list_values else json_values[name],
                    "source": source,
                }
                for name, (_value, source) in effective.items()
            },
            "cloud": {
                "logged_in": cred is not None,
                "url": cred.url if cred else None,
                "user": cred.user if cred else None,
            },
        }
        print(json.dumps(payload, indent=2))
        return 0

    home = str(Path.home())

    def _tilde(text: str) -> str:
        return text.replace(home, "~")

    print("rote config — effective defaults (flag > ROTE_* env > project > user > built-in)")
    print()
    user_state = "present" if layers.user_path.is_file() else "absent — run `rote init`"
    print(f"  user config:    {_tilde(str(layers.user_path))}  ({user_state})")
    project_state = _tilde(str(layers.project_path)) if layers.project_path else "(none found)"
    print(f"  project config: {project_state}")
    print()
    # Column widths come from the rows themselves, so a longer key
    # ("inference") or value never pushes its row out of line.
    name_width = max(len(name) for name in effective)
    value_width = max(len(value) for value, _source in effective.values())
    for name, (value, source) in effective.items():
        print(f"  {name:<{name_width}}  {value:<{value_width}}  {_tilde(source)}")
    print()
    if cred is not None:
        print(f"  rote cloud: logged in as {cred.user or '(unknown)'} ({cred.url})")
    else:
        print("  rote cloud: logged out (`rote login` to make it the default host)")
    return 0


def _cmd_skills_list(args: argparse.Namespace) -> int:
    """``rote skills list``: :func:`rote.skills.discover`, rendered."""
    import json as _json

    from rote.skills import discover

    found = discover()
    skills = found.skills if args.limit is None else found.skills[: max(args.limit, 0)]
    if args.json:
        payload = found.to_dict()
        payload["skills"] = [s.to_dict() for s in skills]
        print(_json.dumps({"ok": True, "total": len(found.skills), **payload}, indent=2))
        return 0
    for problem in found.problems:
        print(f"note: {problem}", file=sys.stderr)
    if not found.skills:
        print("rote skills: no skills found in any client's skill directory")
        return 0
    labels = {
        "bundled": "demo",
        "ready": "ready",
        "needs_login": "login",
        "needs_server": "add server",
        "unreadable": "unreadable",
    }
    width = max(len(s.name) for s in skills)
    for skill in skills:
        root = skill.found_in[0]
        where = root.client + (f" plugin {root.plugin}" if root.plugin else f" {root.scope}")
        state = labels[skill.readiness.state]
        print(f"  {state:<10}  {skill.name.ljust(width)}  {where:<24}  {skill.readiness.reason}")
    if len(skills) < len(found.skills):
        print(f"  ... {len(found.skills) - len(skills)} more (drop --limit to see them)")
    return 0


def _cmd_example(args: argparse.Namespace) -> int:
    """List the bundled examples, or copy one into ./<name> and say how to run it."""
    from rote.examples import BUNDLED, ExampleError, copy_example, render_commands

    if args.name is None:
        print("Bundled examples (copy one with: rote example <name>):")
        width = max(len(name) for name in BUNDLED)
        for name, example in BUNDLED.items():
            print(f"  {name.ljust(width)}  {example.summary}")
        return 0
    dest = Path.cwd() / args.name
    try:
        example = copy_example(args.name, dest, force=args.force)
    except ExampleError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"rote example: copied {example.name} to ./{example.name}")
    print("\nNext, from this directory:\n")
    for line in render_commands(example, example.name):
        print(f"  {line}" if line else "")
    from rote.extras import DISTRIBUTION

    print(
        f"\nNo rote on PATH? Prefix each command with `uvx --from {DISTRIBUTION}`, "
        f"or `uvx --from '{DISTRIBUTION}[dbos]'` for the dbos ones."
    )
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    """Read-only preflight: is a compilation set up to succeed before spending money?

    Reports the rote/Python versions, which compiler drivers are usable,
    whether the base dependencies are importable, which optional extras
    are installed, the auth state of every registered MCP server, and
    every emitted app rote knows about. Exits non-zero when NO compiler
    driver is available or a base dependency is missing (the two
    load-bearing gates), so it doubles as a scriptable gate; MCP and app
    findings are informational.
    """
    report = _build_doctor_report()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(_render_doctor_text(report))
    return 0 if report["ok"] else 1


def _resolve_eval_skill_dir(
    args_skill: str | None, pipeline_yaml: Path, source_skill: str | None
) -> Path | None:
    """Locate the source-skill directory for the 'before' baseline.

    Explicit ``--skill`` wins; otherwise try the pipeline's recorded
    ``source_skill`` relative to the current directory and to the
    pipeline.yaml's own location. None = no baseline (after-only card).
    """
    if args_skill:
        p = Path(args_skill)
        return p if (p / "SKILL.md").is_file() else None
    if source_skill:
        # source_skill was recorded relative to wherever the compilation
        # ran (often a temp work dir), so no single base is reliable —
        # try the cwd, then every ancestor of the pipeline.yaml.
        pipeline_dir = pipeline_yaml.resolve().parent
        bases = [Path.cwd(), pipeline_dir, *pipeline_dir.parents]
        for base in bases[:10]:
            candidate = (base / source_skill).resolve()
            if (candidate / "SKILL.md").is_file():
                return candidate
    return None


def _single_emitted_runtime(target: Path) -> str | None:
    """The runtime a ``rote compile --out`` directory was emitted for.

    That layout nests the emit under ``runtime/<name>/``. With exactly one
    there, the scorecard can price that runtime without a flag; with zero
    or several it has no basis to pick, and ``--runtime`` decides.
    """
    runtime_dir = target / "runtime"
    if not runtime_dir.is_dir():
        return None
    names = sorted(p.name for p in runtime_dir.iterdir() if p.is_dir())
    return names[0] if len(names) == 1 else None


def _cmd_eval_skill_ab(args: argparse.Namespace) -> int:
    """``rote eval --run --runtime skill``: the source skill and its compiled
    skill, each run by a ``claude -p`` subagent on the same input
    (:mod:`rote.eval.skill_ab`).

    The target is a compiled pipeline (as for every ``rote eval``), or a
    source skill dir, which is compiled first with the pinned argv
    :func:`rote.proof.compile_command` builds."""
    import json
    import tempfile
    from datetime import UTC, datetime

    from rote.compiler.drivers.claude import DEFAULT_MODEL
    from rote.eval import skill_ab
    from rote.eval.empirical import EmpiricalError

    if args.trials < 1:
        print(f"error: --trials must be >= 1, got {args.trials}", file=sys.stderr)
        return 2
    # Flags of the pipeline-trial mode that mean nothing here: refused, not ignored.
    pipeline_only = [
        flag
        for flag, value in (
            ("--inference", args.inference),
            ("--python", args.python),
            ("--runtime-dir", args.runtime_dir),
            ("--prior", args.prior),
        )
        if value is not None
    ]
    if pipeline_only:
        print(
            f"error: {', '.join(pipeline_only)} do not apply to --run --runtime skill: both "
            "sides run as claude -p agents",
            file=sys.stderr,
        )
        return 2
    if args.budget is None:
        print(
            "error: --run --runtime skill needs --budget USD: every trial is a paid "
            "claude -p run, and the eval stops before the one that would cross it",
            file=sys.stderr,
        )
        return 2
    if (args.input is None) == (args.case is None):
        print(
            "error: pass exactly one of --input <task.json> or --case <id> (a bench "
            "case from the example's fidelity/cases.yaml)",
            file=sys.stderr,
        )
        return 2
    model = args.model or DEFAULT_MODEL
    work = (
        Path(args.work_dir) if args.work_dir else Path(tempfile.mkdtemp(prefix="rote-skill-ab-"))
    ).resolve()
    work.mkdir(parents=True, exist_ok=True)
    ledger = skill_ab.Ledger(
        Path(args.ledger) if args.ledger else work / "ledger.json", float(args.budget)
    )
    print(f"rote eval: work dir {work}; ledger {ledger.path}", file=sys.stderr)
    log = lambda line: print(line, file=sys.stderr)  # noqa: E731
    target = Path(args.target)
    try:
        executable = skill_ab.require_executable("claude")
        task = json.loads(Path(args.input).read_text(encoding="utf-8")) if args.input else None
        provenance: dict[str, object] | None = None
        if (target / "SKILL.md").is_file():
            skill_dir = target
            sample: object = (
                skill_ab.bench_case(skill_dir, args.case)[1].input
                if args.case is not None
                else task["input"]
                if isinstance(task, dict) and "signals" in task
                else task
            )
            if not isinstance(sample, dict):
                raise skill_ab.SkillAbError("the input must be a JSON object")
            pipeline_yaml, provenance = skill_ab.compile_for_eval(
                skill_dir, sample, work, ledger, model=args.compile_model, log=log
            )
        else:
            found = _resolve_pipeline_yaml(target)
            if found is None:
                raise skill_ab.SkillAbError(
                    f"no pipeline.yaml or SKILL.md at {target}: pass a compiled dir, a "
                    "pipeline.yaml, or a source skill to compile first"
                )
            pipeline_yaml = found
            resolved = _resolve_eval_skill_dir(
                args.skill, pipeline_yaml, load_pipeline(pipeline_yaml).source_skill
            )
            if resolved is None:
                raise skill_ab.SkillAbError(
                    "the eval runs the source skill too: pass its directory with --skill <dir>"
                )
            skill_dir = resolved
        subject = skill_ab.subject_for(
            pipeline_yaml,
            skill_dir=skill_dir,
            task=task,
            case_id=args.case,
            provenance=provenance,
        )
        harness = skill_ab.Harness(
            work=work,
            model=model,
            ledger=ledger,
            executable=executable,
            max_turns=args.max_turns,
            log=log,
        )
        report = skill_ab.run_subject(harness, subject, args.trials)
    except skill_ab.BudgetExhausted as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except (skill_ab.SkillAbError, EmpiricalError, OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    generated_at = datetime.now(UTC).isoformat(timespec="seconds")
    corpus = skill_ab.append_corpus(
        [skill_ab.corpus_record(report, model=model, generated_at=generated_at)]
    )
    print(f"rote eval: measurements appended to {corpus}", file=sys.stderr)
    if args.json:
        rendered = json.dumps(
            {
                "generated_at": generated_at,
                "model": model,
                "spent_usd": ledger.spent,
                "budget_usd": ledger.cap_usd,
                "work_dir": str(work),
                **report.to_dict(),
            },
            indent=2,
            default=str,
        )
    else:
        rendered = skill_ab.render_markdown([report], model=model, spent=ledger.spent)
    if args.out:
        Path(args.out).write_text(rendered + ("\n" if args.json else ""), encoding="utf-8")
        print(f"rote eval: wrote {args.out}", file=sys.stderr)
    else:
        print(rendered)
    clean = report.stopped is None and all(
        t.ok for summary in report.sides.values() for t in summary.trials
    )
    return 0 if clean else 1


def _cmd_eval(args: argparse.Namespace) -> int:
    """Static before/after scorecard: speed, cost, determinism.

    'Before' models the source skill running as raw agent instructions;
    'after' is derived from the pipeline IR. Prices are fetched live —
    a network failure is a loud error, never a stale builtin table.
    """
    import json

    from rote.eval.pricing import PricingError

    if args.run and args.runtime == "skill":
        return _cmd_eval_skill_ab(args)
    skill_ab_only = [
        flag
        for flag, value in (
            ("--case", args.case),
            ("--budget", args.budget),
            ("--ledger", args.ledger),
            ("--work-dir", args.work_dir),
            ("--compile-model", args.compile_model),
        )
        if value is not None
    ]
    if skill_ab_only:
        print(
            f"error: {', '.join(skill_ab_only)} only apply to --run --runtime skill",
            file=sys.stderr,
        )
        return 2

    target = Path(args.target)
    pipeline_yaml = _resolve_pipeline_yaml(target)
    if pipeline_yaml is None:
        print(
            f"error: no pipeline.yaml found at or under {target} — "
            f"pass a compiled dir, a compile --out dir, or the file itself",
            file=sys.stderr,
        )
        return 2

    try:
        pipeline = load_pipeline(pipeline_yaml)
    except Exception as e:
        print(f"error: failed to load pipeline: {e}", file=sys.stderr)
        return 1

    skill_dir = _resolve_eval_skill_dir(args.skill, pipeline_yaml, pipeline.source_skill)
    if skill_dir is None:
        if args.skill:
            print(f"error: --skill {args.skill} has no SKILL.md", file=sys.stderr)
            return 2
        print(
            "rote eval: no source skill found (no --skill, and the pipeline's "
            "source_skill did not resolve) — emitting the after-side only",
            file=sys.stderr,
        )

    from rote.eval import build_scorecard_for
    from rote.eval.priors import priors_from_overrides

    try:
        priors = priors_from_overrides(args.prior or [])
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    try:
        scorecard = build_scorecard_for(
            pipeline,
            pipeline_yaml,
            skill_dir,
            args.provider,
            priors=priors,
            runtime=args.runtime or _single_emitted_runtime(target),
        )
    except PricingError as e:
        print(f"error: could not fetch live prices: {e}", file=sys.stderr)
        return 1

    measured_md = ""
    measured_json: dict[str, object] | None = None
    if args.run:
        try:
            # Empirical trials execute real judges, so they are billed to
            # a real lane; the same resolution `rote run` uses picks it.
            _export_inference_selection(args.inference)
            measured_md, measured_json = _run_empirical(args, pipeline, pipeline_yaml, skill_dir)
        except Exception as e:
            print(f"error: empirical run failed: {e}", file=sys.stderr)
            return 1

    if args.json:
        payload = scorecard.to_dict()
        if measured_json is not None:
            payload["measured"] = measured_json
        rendered = json.dumps(payload, indent=2)
    else:
        rendered = scorecard.to_markdown()
        if measured_md:
            rendered += "\n" + measured_md
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered + "\n", encoding="utf-8")
        print(f"rote eval: wrote {out_path}")
    else:
        print(rendered)
    return 0


def _resolve_runtime_dir(explicit: str | None, pipeline_yaml: Path) -> Path:
    """Locate the emitted runtime app for empirical pipeline trials.

    Explicit ``--runtime-dir`` wins; otherwise try the ``rote compile
    --out`` layout's siblings (runtime/dbos, runtime/python) and the
    pipeline's own directory.
    """
    if explicit:
        p = Path(explicit)
        if (p / "main.py").is_file():
            return p
        raise ValueError(f"--runtime-dir {explicit} has no main.py")
    base = pipeline_yaml.resolve().parent.parent
    candidates = (
        base / "runtime" / "dbos",
        base / "runtime" / "python",
        pipeline_yaml.resolve().parent,
    )
    for candidate in candidates:
        if (candidate / "main.py").is_file():
            return candidate
    raise ValueError(
        "no emitted runtime app found (looked for main.py in "
        + ", ".join(str(c) for c in candidates)
        + ") — emit one with `rote emit` or pass --runtime-dir"
    )


def _run_empirical(
    args: argparse.Namespace,
    pipeline: Pipeline,
    pipeline_yaml: Path,
    skill_dir: Path | None,
) -> tuple[str, dict[str, object]]:
    """Execute the --run trials and return (markdown section, JSON section).

    Pipeline trials run first: they're fast and cheap, so a
    misconfigured runtime dir fails before any agent money is spent.
    """
    import json
    from datetime import UTC, datetime

    from rote.eval.empirical import (
        EmpiricalResult,
        append_corpus,
        mcp_servers_for_pipeline,
        measured_to_dict,
        min_expected_turns_for,
        render_measured_markdown,
        run_pipeline_trial,
        run_skill_trial,
        split_task,
    )
    from rote.eval.pricing import fetch_catalog

    if not args.input:
        raise ValueError("--run requires --input <task.json> (the pipeline input payload)")
    if args.trials < 1:
        raise ValueError(f"--trials must be >= 1, got {args.trials}")
    task = json.loads(Path(args.input).read_text(encoding="utf-8"))
    input_payload, signals = split_task(pipeline, task)

    runtime_dir = _resolve_runtime_dir(args.runtime_dir, pipeline_yaml)
    catalog = fetch_catalog(provider=args.provider)
    sample = catalog.sample(provider=args.provider)
    # Default the agent to the mid tier — the realistic "run it as a
    # skill" model — resolved from the live lineup, never hardcoded.
    skill_model = args.model or (sample[1].model_id if len(sample) > 1 else sample[0].model_id)

    output_fields: set[str] = set()
    for exit_id in pipeline.exit_nodes:
        node_output = pipeline.node_by_id(exit_id).output
        if isinstance(node_output, dict):
            output_fields.update(node_output.keys())

    pipeline_runs = []
    for i in range(args.trials):
        print(f"rote eval: pipeline trial {i + 1}/{args.trials} ({runtime_dir})", file=sys.stderr)
        pipeline_runs.append(
            run_pipeline_trial(
                runtime_dir,
                input_payload,
                signals=signals or None,
                python_executable=args.python,
            )
        )

    skill_runs = []
    if skill_dir is not None:
        # Wire the pipeline's MCP bindings into the skill trial so the
        # agent runs over the same live tools the pipeline binds to —
        # the measurement is only representative when the skill can
        # actually pull its data.
        mcp_servers, missing_servers = mcp_servers_for_pipeline(pipeline)
        if mcp_servers:
            print(
                "rote eval: wiring MCP servers into the skill trial: "
                + ", ".join(sorted(mcp_servers)),
                file=sys.stderr,
            )
        if missing_servers:
            print(
                "rote eval: WARNING — pipeline binds MCP servers with no "
                "resolvable endpoint (set ROTE_MCP_<SERVER>_URL): "
                + ", ".join(missing_servers)
                + " — skill trials will be flagged unrepresentative",
                file=sys.stderr,
            )
        for i in range(args.trials):
            print(
                f"rote eval: skill trial {i + 1}/{args.trials} "
                f"(claude -p, {skill_model}, subscription auth) — this takes minutes",
                file=sys.stderr,
            )
            skill_runs.append(
                run_skill_trial(
                    skill_dir,
                    input_payload,
                    model=skill_model,
                    max_turns=args.max_turns,
                    output_fields=sorted(output_fields) or None,
                    mcp_servers=mcp_servers or None,
                    missing_mcp_servers=missing_servers or None,
                    min_expected_turns=min_expected_turns_for(pipeline),
                )
            )
    else:
        print("rote eval: no source skill — measuring the pipeline side only", file=sys.stderr)

    result = EmpiricalResult(
        trials=args.trials,
        skill_runs=tuple(skill_runs),
        pipeline_runs=tuple(pipeline_runs),
        skill_model=skill_model if skill_runs else None,
    )
    generated_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    corpus_path = append_corpus(result, generated_at=generated_at)
    print(f"rote eval: measurements appended to {corpus_path}", file=sys.stderr)
    return render_measured_markdown(result, catalog), measured_to_dict(result, catalog)


# ───────── Argument parsing ─────────


def _local_decider(args: argparse.Namespace, resolved: Resolved, command: str) -> str:
    """The System-1 decider a local compile (or analyze) uses.

    ``resolved`` is the layered ``decider`` setting. A model decider on a
    compile that does not run System-1 (``--no-system1``, or ``--update``,
    which always runs the full-rubric agent) is a usage error when it came
    from a flag, so a decider the user asked for never silently goes unused.
    From a config file or env var it is a standing preference for System-1
    compiles, so this one compile says it does not apply and uses none.
    """
    decider = resolved.value or "auto"
    if decider in ("rules", "auto") or _runs_system1(args):
        return decider
    why = "--no-system1" if args.system1 is False else "--update (always the full-rubric agent)"
    if resolved.source == "flag":
        raise ValueError(
            f"--decider {decider} decides System-1's structure, and {why} does not run System-1"
        )
    print(
        f"rote {command}: {resolved.source} sets `decider: {decider}`, which applies only to "
        f"System-1 compiles; {why} does not run System-1",
        file=sys.stderr,
    )
    return "rules"


def _runs_system1(args: argparse.Namespace) -> bool:
    """Whether this compile takes the System-1 path (the Compiler's own rule)."""
    return bool(args.system1) if args.system1 is not None else not getattr(args, "update", False)


def _local_system1_choice(
    args: argparse.Namespace, resolved: Resolved, flag: str, default: str
) -> str:
    """What a local compile (or analyze) uses for a System-1 call setting
    (``--tiebreak``, ``--structure``): the layered setting, else ``default``.
    The flag on a compile that does not run System-1 is a usage error, like
    ``--decider``. A configured value is a standing preference and needs no
    note: on that path nothing would ask."""
    if resolved.source == "flag" and not _runs_system1(args):
        why = "--no-system1" if args.system1 is False else "--update (always the full-rubric agent)"
        raise ValueError(f"{flag} applies to System-1 compiles, and {why} does not run one")
    return resolved.value or default


def _add_system1_flags(p: argparse.ArgumentParser) -> None:
    """The System-1 flags, shared by ``compile`` and its dry run ``analyze``."""
    from rote.compiler.system1.generate import DEFAULT_MAX_PARALLEL

    p.add_argument(
        "--no-system1",
        dest="system1",
        action="store_const",
        const=False,
        default=None,
        help=(
            "Run the full-rubric agent, which decides the pipeline structure itself. By "
            "default (System-1) rote decides the whole structure with rules (kinds, data "
            "flow, fan_out, mandatory, gates, constants, rule tables as code) before the "
            "agent runs, and the agent writes only text (implementations, judge prompts, "
            "the report). --update always runs the full-rubric agent, because System-1 "
            "does not support it yet. See docs/design/system1-compile.md"
        ),
    )
    p.add_argument(
        "--decider",
        dest="decider",
        choices=DECIDERS,
        default=None,
        help=(
            "The decision model that checks the structure pass, or decides step kinds and "
            "gates under --structure rules (default: the `decider` setting, else auto). "
            "'auto' takes Clef when ROTE_CLEF_ACCOUNT_ID and ROTE_CLEF_API_TOKEN are set "
            "(Workers AI, about $0.005 a compile), else Jev when TYPESAFE_API_KEY is set "
            "(Typesafe's API, about $0.001), else no check, and says which. CLOUDFLARE_* "
            "deploy credentials are never used for it. 'clef' or 'jev' by name fails "
            "before any spend without its credentials. 'rules' makes no call"
        ),
    )
    p.add_argument(
        "--structure",
        choices=["rules", *STRUCTURE_MODELS],
        default=None,
        help=(
            "Who reads the whole skill and decides the pipeline's skeleton (default: "
            f"the `structure` setting, else {DEFAULT_STRUCTURE}). A model answers in "
            "one schema-locked call, and rote checks every node against the skill's "
            "text, its tools' schemas and the sample input before using it, asking "
            "again up to twice. It costs about $0.015 a compile on Haiku. It runs on "
            "the compile driver's own client: claude -p under the claude driver, the "
            "api driver's client under --agent api, and the openai-api driver's "
            "client and model under --agent openai-api, and codex exec --output-schema "
            "under codex. "
            "'rules' makes no call and decides the skeleton step by step from the "
            "skill's text"
        ),
    )
    p.add_argument(
        "--tiebreak",
        choices=TIEBREAKS,
        default=None,
        help=(
            "Who decides the steps System-1 is still unsure of (default: the `tiebreak` "
            f"setting, else {DEFAULT_TIEBREAK}). 'haiku' asks Claude Haiku in one "
            "schema-locked call, "
            "on the compile driver's own client, like --structure: claude -p on your "
            "subscription, the api driver's client, the openai-api driver's client "
            "and model, or codex exec. 'none' makes no call: those steps take "
            "a conservative default and are listed for review"
        ),
    )
    p.add_argument(
        "--max-parallel",
        type=int,
        default=DEFAULT_MAX_PARALLEL,
        metavar="N",
        help=(
            "The most model calls in flight at once while System-1 writes the text "
            "(one call per node, all at once, with the claude and api drivers; "
            f"default {DEFAULT_MAX_PARALLEL}). See docs/design/parallel-generation.md"
        ),
    )


def _skill_command(name: str) -> Callable[[argparse.Namespace], int]:
    """A ``rote skill`` handler from :mod:`rote.skill_target.commands`, imported on use."""

    def handler(args: argparse.Namespace) -> int:
        from rote.skill_target import commands

        result: int = getattr(commands, name)(args)
        return result

    return handler


def _add_skill_parser(subparsers: Any) -> None:
    """``rote skill``: install, trust and run compiled skills (``--runtime skill``)."""
    skill_p = subparsers.add_parser(
        "skill",
        help="Install, trust and run compiled skills (--runtime skill)",
        description=(
            "A compiled skill runs only through rote skill run <name>, and only while its "
            "files are the ones the user installed. rote skill install <dir> records them "
            "(the first trust point); rote skill trust <name> shows what changed since and "
            "records it again, and with --allow-writes lets its tools write. Every write is "
            "a plan, shown and confirmed first."
        ),
    )
    sub = skill_p.add_subparsers(dest="skill_command", required=True)

    run_p = sub.add_parser(
        "run",
        help="Run one driver command of an installed skill (what its SKILL.md tells agents)",
        description=(
            "Pass a command through to the installed skill's driver: start '<input json>' | "
            "--file <path>; answer <run> <step> '<json>' | --file <path> [--said '<words>']; "
            "resume <run>; status [<run>]; cancel <run>; runs [--prune]. Before it runs, "
            "every file in the skill must match its install record (a changed, new, missing "
            "or symlinked file is refused, naming rote skill trust <name>). The driver runs "
            "with rote's interpreter in isolated mode, the skill's scripts dir and rote's "
            "site-packages on the path, and an environment built from an allowlist: MCP "
            "endpoints come from the registry only, and the state paths are the ones the "
            "install recorded. It prints exactly one ROTE/1 block on stdout."
        ),
    )
    run_p.add_argument("name", help="the skill's name (its SKILL.md name)")
    run_p.add_argument(
        "driver_args",
        nargs=argparse.REMAINDER,
        metavar="COMMAND ...",
        help="the driver command and its arguments",
    )
    run_p.set_defaults(func=_skill_command("cmd_run"))

    install_p = sub.add_parser(
        "install",
        help="Record a compiled skill and the hash of every file in it",
        description=(
            "Check that <dir> is exactly what rote emitted, then record it under its name "
            "with the hash of every file: rote skill run refuses anything else. Then offer to "
            "save the Claude Code allow rule Bash(rote skill run <name> *) in the project's "
            ".claude/settings.local.json, as a separate step that --yes never covers: a "
            "person is asked (default no), otherwise only --save-allow-rule saves it."
        ),
    )
    install_p.add_argument("dir", help="the compiled skill's directory")
    install_p.add_argument(
        "--save-allow-rule",
        action="store_true",
        help="save the Claude Code allow rule without asking (never implied by --yes)",
    )
    install_p.add_argument(
        "--claude-project",
        default=None,
        metavar="DIR",
        help=(
            "the project whose .claude/settings.local.json gets the rule (default: the "
            "project the skill lives in, else the current directory)"
        ),
    )
    _add_onboarding_mode_flags(install_p)
    install_p.set_defaults(func=_skill_command("cmd_install"))

    trust_p = sub.add_parser(
        "trust",
        help="Show what changed in an installed skill, and record it",
        description=(
            "List every file that changed, appeared or went missing since the skill was "
            "installed or last trusted (lines added and removed), and any state path that "
            "moved, then record them on a yes. --allow-writes also lists every tool the "
            "skill's plan calls that may write and records the user's consent for this plan: "
            "the driver lets a tool write only with it. That needs a person at a terminal; "
            "--yes cannot stand in for one."
        ),
    )
    trust_p.add_argument("name", help="the skill's name")
    trust_p.add_argument(
        "--allow-writes",
        action="store_true",
        help="let the skill's write tools run (asks a person; never accepted from --yes alone)",
    )
    trust_p.add_argument(
        "--diff", action="store_true", help="print each change as a unified diff first"
    )
    _add_onboarding_mode_flags(trust_p, force=False)
    trust_p.set_defaults(func=_skill_command("cmd_trust"))

    list_p = sub.add_parser("list", help="List compiled skills and whether each would run")
    list_p.add_argument("--json", action="store_true", help="One JSON object on stdout")
    list_p.set_defaults(func=_skill_command("cmd_list"))

    runs_p = sub.add_parser("runs", help="List every installed skill's runs")
    runs_p.add_argument(
        "--prune",
        action="store_true",
        help="remove expired runs first (finished after 7 days, waiting after 30 or past "
        "their deadline)",
    )
    runs_p.add_argument("--json", action="store_true", help="One JSON object on stdout")
    runs_p.set_defaults(func=_skill_command("cmd_runs"))


def _add_onboarding_mode_flags(parser: argparse.ArgumentParser, *, force: bool = True) -> None:
    """The flags every onboarding command shares (``rote.interact.mode``).

    ``force=False`` for a command whose plans never replace a file."""
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="take every default and apply the plan without asking",
    )
    if force:
        parser.add_argument(
            "--force",
            action="store_true",
            help="allow the plan to replace a file that exists",
        )
    parser.add_argument(
        "--agent-mode",
        choices=("auto", "yes", "no"),
        default=None,
        help=(
            "auto (default): an AI agent detected in the environment means no prompts; "
            "yes: never prompt; no: a person is here (also ROTE_AGENT_MODE)"
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the plan and the outcome as one JSON object on stdout; never prompts",
    )


def _build_parser() -> argparse.ArgumentParser:
    # Help text names the real defaults, read from the constants the code
    # uses, so `--help` cannot drift from behavior when a default moves.
    from rote.compiler.drivers.anthropic_api import DEFAULT_MODEL as API_DEFAULT_MODEL
    from rote.compiler.drivers.claude import DEFAULT_EFFORT, EFFORT_LEVELS
    from rote.compiler.drivers.claude import DEFAULT_MODEL as CLAUDE_DEFAULT_MODEL
    from rote.runners import RUNNABLE_RUNTIMES

    EFFORT_HELP = (
        "How much the compiler agent thinks, for the claude driver and the "
        "openai-api driver (sent as reasoning_effort; other drivers ignore it). "
        f"Default: the `effort` config, else {DEFAULT_EFFORT} on claude and the "
        "model's own default on openai-api. "
        "Higher costs more and runs longer: thinking is most of a compile's "
        "output tokens. Overrides CLAUDE_CODE_EFFORT_LEVEL, which rote "
        "clears for the compiler."
    )
    model_default_note = (
        f"{CLAUDE_DEFAULT_MODEL} for the claude and api drivers"
        if CLAUDE_DEFAULT_MODEL == API_DEFAULT_MODEL
        else f"{CLAUDE_DEFAULT_MODEL} for the claude driver, {API_DEFAULT_MODEL} for api"
    )
    runnable = ", ".join(sorted(RUNNABLE_RUNTIMES))

    parser = argparse.ArgumentParser(
        prog="rote",
        description=(
            "Compile fuzzy AI skills into deterministic, reliable workflows. "
            "The north-star command is `rote compile`; `rote emit` exposes "
            "the underlying IR → code step for users who already have a "
            "pipeline.yaml."
        ),
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"rote {__version__}",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        metavar="<command>",
    )

    available_runtimes = sorted(ADAPTERS)

    # rote emit
    emit = subparsers.add_parser(
        "emit",
        help="Render runtime code from a pipeline.yaml (IR → code only)",
        description=(
            "Render runtime code from a pipeline.yaml. This is the pure "
            "IR → adapter step — no compiler agent is invoked."
        ),
    )
    emit.add_argument(
        "pipeline_yaml",
        help="Path to a pipeline.yaml (IR file)",
    )
    emit.add_argument(
        "--runtime",
        default=None,
        choices=available_runtimes,
        help=(
            f"Target workflow runtime (default: the configured `runtime` — see "
            f"`rote config` — else dbos, durable execution as a library with no "
            f"orchestrator to run). Available: {', '.join(available_runtimes)}"
        ),
    )
    emit.add_argument(
        "--backend",
        default=None,
        choices=["mcp", "api"],
        help=(
            "For external_call nodes with an mcp binding (dbos, dbos-ts, "
            "cloudflare, inngest): 'mcp' (default) emits a working "
            "Streamable-HTTP call to the MCP tool the skill used; 'api' emits "
            "the node's impl, the direct vendor-API path (on a TypeScript "
            "runtime, a stub to fill in). No effect on nodes without an mcp "
            "binding, or on python and temporal, which always run the impl."
        ),
    )
    emit.add_argument(
        "--out",
        required=True,
        help=(
            "Output directory (created if missing). For --runtime skill this is the "
            "skill's own directory, which clients expect to be named after the skill"
        ),
    )
    _add_skill_runtime_args(emit, with_skill_dir=True)
    emit.add_argument(
        "--json",
        action="store_true",
        help=(
            "Emit a single JSON object (out_dir, written files, preserved "
            ".new files, unimplemented stubs) to stdout; keep human logs on "
            "stderr. For piping into another tool or agent."
        ),
    )
    emit.set_defaults(func=_cmd_emit)

    # rote compile (stub)
    compile_p = subparsers.add_parser(
        "compile",
        help="Compile a skill into a runnable pipeline (north-star command)",
        description=(
            "Read a skill bundle, run the rote-compile agent to produce "
            "a pipeline IR + extracted modules + signature stubs, then "
            "emit runtime code for the chosen adapter. Output layout: "
            "<out>/compiled/ (the IR + stubs) and <out>/runtime/<target>/ "
            "(the emitted workflow code)."
        ),
    )
    compile_p.add_argument("skill_path", help="Path to the source skill directory")
    compile_p.add_argument(
        "--runtime",
        default=None,
        choices=available_runtimes,
        help=(
            f"Target workflow runtime. Default: the configured `runtime` "
            f"(see `rote config`), else cloudflare + auto-deploy to rote "
            f"cloud when logged in (`rote login`), else dbos (durable "
            f"execution as a library, no orchestrator to run). "
            f"Available: {', '.join(available_runtimes)}"
        ),
    )
    compile_p.add_argument(
        "--no-deploy",
        action="store_true",
        help=(
            "When logged in to rote cloud: run the compilation locally and "
            "skip the auto-deploy (implies a local run; deploy later with "
            "`rote deploy --target rote-cloud`)"
        ),
    )
    cloud_mode = compile_p.add_mutually_exclusive_group()
    cloud_mode.add_argument(
        "--local",
        action="store_true",
        help="Run the compilation on this machine even when logged in to rote cloud",
    )
    cloud_mode.add_argument(
        "--cloud",
        action="store_true",
        help="Run the compilation on rote cloud; requires login or ROTE_CLOUD_* env",
    )
    compile_p.add_argument(
        "--backend",
        default=None,
        choices=["mcp", "api"],
        help=(
            "For external_call nodes with an mcp binding: 'mcp' (default) "
            "emits a working call to the MCP tool the skill used; 'api' has the "
            "compiler write a direct vendor-API call instead, researched in the "
            "vendor's current docs (the agent gets web search and fetch tools, "
            "and compile-report.md lists the pages it cited). 'api' needs a "
            "Python runtime: dbos, python or temporal."
        ),
    )
    compile_p.add_argument(
        "--out",
        required=True,
        help=(
            "Output directory for the compiled artifacts. --runtime skill writes the "
            "skill to <out>/runtime/skill/<its name>/"
        ),
    )
    _add_skill_runtime_args(compile_p, with_skill_dir=False)
    compile_p.add_argument(
        "--agent",
        choices=sorted(DRIVERS),
        default=None,
        help=(
            "Agent runtime to use for the compiler. Defaults to auto-detect "
            "(claude CLI first, then codex, then anthropic API). Users with "
            "a Claude Max/Pro or ChatGPT subscription should prefer claude/codex "
            "to avoid per-token API charges. openai-api is explicit-only and "
            "requires --model."
        ),
    )
    compile_p.add_argument(
        "--model",
        default=None,
        help=(
            "Override the LLM model used by the compiler agent (a larger "
            "model costs more per run). Defaults to the driver's default: "
            f"{model_default_note}; codex uses its CLI's configured model."
        ),
    )
    compile_p.add_argument(
        "--effort",
        choices=EFFORT_LEVELS,
        default=None,
        help=EFFORT_HELP,
    )
    compile_p.add_argument(
        "--no-eval",
        action="store_true",
        help=(
            "Skip the before/after eval scorecard (scorecard.md). The "
            "scorecard needs one network fetch for live model prices; "
            "everything else about it is static."
        ),
    )
    compile_p.add_argument(
        "--update",
        action="store_true",
        help=(
            "Incremental re-compilation: requires a previous compilation in "
            "--out. Diffs the skill against the previous run's provenance "
            "and re-derives only nodes whose source sections changed — "
            "unchanged nodes are preserved verbatim (ids and all), and a "
            "skill with no changes skips the agent entirely."
        ),
    )
    compile_p.add_argument(
        "--json",
        action="store_true",
        help=(
            "Emit a single JSON object (compiled_dir, runtime_dir, written "
            "files, scorecard path, unimplemented stubs) to stdout; keep the "
            "progress log and summary on stderr. For piping into another tool "
            "or agent."
        ),
    )
    compile_p.add_argument(
        "--progress-file",
        default=None,
        metavar="PATH",
        help=(
            "Stream structured progress to PATH as JSON Lines (one event per "
            "line, flushed live) for an agent tailing the run: phase / turn / "
            "tool / warning / complete events, each token-carrying line priced "
            "with a live cost_usd, and a final type:summary digest (roteness, "
            "node-kind counts, stub TODOs, total tokens + cost). Runs "
            "alongside the stderr progress log; composes with --json."
        ),
    )
    compile_p.add_argument(
        "--baseline",
        action="store_true",
        help=(
            "Run the raw skill once as an agent BEFORE compiling (measured "
            "wall/turns/cost + observed MCP traffic; see `rote baseline`). "
            "The scorecard then carries a measured before-side instead of an "
            "estimate. Adds minutes and tens of cents."
        ),
    )
    compile_p.add_argument(
        "--baseline-input",
        default=None,
        metavar="FILE",
        help=(
            "Task payload for --baseline as a JSON object file. Omit to have "
            "one derived from SKILL.md and confirmed before spending."
        ),
    )
    compile_p.add_argument(
        "--tool-discovery",
        choices=("named", "all", "none"),
        default="named",
        help=(
            "Which registered MCP servers rote sends a read-only tools/list to before "
            "the agent runs, to ground tool argument names in each tool's own schema. "
            "named (default): only servers whose registered name appears in the skill's "
            "text (google_drive matches 'Google Drive'). all: every registered server, "
            "with its stored credentials. none: no server. --no-mcp implies none. Works "
            "on every compiler driver."
        ),
    )
    compile_p.add_argument(
        "--sample-input",
        default=None,
        metavar="FILE",
        help=(
            "A sample input for the skill, as a JSON object file (or the "
            '{"input": ..., "signals": ...} envelope `rote eval --run` takes). '
            "rote extracts the input contract from it, field and row names "
            "included, and runs the pipeline's deterministic steps on it before "
            "the compile is done. Default: input.json (or examples/input.json) "
            "in the skill dir, then the input a baseline ran on."
        ),
    )
    compile_p.add_argument(
        "--yes",
        action="store_true",
        help="With --baseline: accept the derived input proposal without prompting",
    )
    compile_p.add_argument(
        "--allow-writes",
        action="store_true",
        help=(
            "With --baseline: lift the read-only MCP gate — the skill runs "
            "exactly as in production, side effects included"
        ),
    )
    compile_p.set_defaults(func=_cmd_compile)
    compile_p.add_argument(
        "--no-verify-tests",
        action="store_true",
        help=("Don't run the tests the compiler agent writes. Static checks still run."),
    )
    _add_system1_flags(compile_p)
    compile_p.add_argument(
        "--strict-provenance",
        action="store_true",
        help=(
            "Fail (exit 1) on every unresolved provenance finding, for CI. By default "
            "only a citation the skill does not contain fails; assumptions never do, "
            "and a provenance gap the repair rounds did not close (an unmarked outcome, "
            "an assumption missing from Open questions) is a warning"
        ),
    )
    compile_p.add_argument(
        "--trust-skill",
        action="store_true",
        help=(
            "Run the agent-written tests even without an OS sandbox. By default they "
            "run under one (sandbox-exec on macOS, bubblewrap on Linux): no network, "
            "no reads outside the compile's work dir and the Python install, no "
            "programs but Python, env limited to PATH and locale, an empty HOME. With "
            "no sandbox available they are skipped and the deliverable is reported "
            "unverified. With this flag they run with only the env limits and the "
            "empty HOME, so they can read your files and reach the network."
        ),
    )
    compile_p.add_argument(
        "--no-mcp",
        action="store_true",
        help=(
            "Disable live MCP discovery and calls during local api/openai-api "
            "compilation. Does not change emitted MCP bindings. Cannot be used "
            "with --baseline or cloud compilation."
        ),
    )

    # rote run
    run_p = subparsers.add_parser(
        "run",
        help="Run a skill (via claude -p) or an emitted pipeline locally, once",
        description=(
            "One-off local execution of either side of a compilation. A skill "
            "directory runs as an agent via `claude -p` — registered MCP "
            "servers injected, read-only gate unless --allow-writes, billed "
            "to your Claude subscription. An emitted runtime directory (or a "
            "`rote compile --out` directory) runs the pipeline itself on a "
            f"local dev setup for its runtime ({runnable}). Emitted modules "
            "that are still stubs are refused up front. The run's output "
            "JSON lands on stdout so it pipes; status goes to stderr."
        ),
    )
    run_p.add_argument(
        "path",
        help="Skill dir (SKILL.md), emitted runtime dir, or `rote compile --out` dir",
    )
    run_p.add_argument(
        "--input",
        default=None,
        help=(
            "Input payload: an inline JSON object, or a JSON file (PATH or "
            "@PATH). Omit to have one derived from the skill and confirmed with you."
        ),
    )
    run_p.add_argument(
        "--runtime",
        default=None,
        help="Which emitted runtime to run when several exist under <path>/runtime/",
    )
    run_p.add_argument(
        "--signal",
        action="append",
        default=[],
        metavar="NAME=JSON",
        help=(
            "Answer for a HITL gate (repeatable): inline JSON, or NAME=@FILE "
            "to read it from a JSON file. A skill run is given the answers in "
            "the order passed (a skill's gates have no names, so NAME is a "
            "label). For a pipeline, gates without one "
            "are prompted for interactively before the run starts — payloads "
            "are delivered when the workflow parks."
        ),
    )
    run_p.add_argument(
        "--model",
        default=None,
        help=f"Skill runs: the agent model (default: {CLAUDE_DEFAULT_MODEL})",
    )
    run_p.add_argument(
        "--inference",
        default=None,
        choices=INFERENCE_PROVIDERS,
        help=(
            "Pipeline runs: who serves (and pays for) the emitted judges — "
            "claude-cli bills your Claude subscription, api bills your own "
            "key, rote-cloud bills your rote tenant. Default: auto-detect, "
            "cheapest to you first."
        ),
    )
    run_p.add_argument(
        "--allow-writes",
        action="store_true",
        help=(
            "Lift the read-only MCP gate, on a skill or a pipeline run: tools "
            "that may write are called (side effects included). Without it, "
            "only tools the server marks readOnlyHint, or every tool on a "
            "server registered with `rote mcp add --read-only`, are called"
        ),
    )
    run_p.add_argument(
        "--allow-missing-servers",
        action="store_true",
        help=(
            "Skill runs: exit 0 even when the run never called an MCP server "
            "the skill names (not registered, refused by the read-only gate, "
            "or not reached). The run is still reported as degraded"
        ),
    )
    run_p.add_argument(
        "--allow-stubs",
        action="store_true",
        help=(
            "Pipeline runs: start even though emitted extracted/ modules are "
            "still stubs (for a run that will not reach them)"
        ),
    )
    run_p.add_argument(
        "--allow-host",
        action="append",
        metavar="HOST",
        help=(
            "With --live-check: allow a host from the compile's own docs spec that no "
            "fetched page it cites vouches for: one the compile did not fetch, or one on "
            "a different registrable domain from its docs (docs.slack.dev does not vouch "
            "for slack.com) (repeatable)"
        ),
    )
    run_p.add_argument(
        "--allow-credential",
        action="append",
        metavar="ENV",
        help=(
            "With --live-check and --allow-host: send the credential the compile's own "
            "docs spec declares in this environment variable (repeatable; --yes never "
            "covers it, and without it a docs-spec host is called with no credential)"
        ),
    )
    run_p.add_argument(
        "--live-check",
        action="store_true",
        help=(
            "For a `--backend api` compile: send each entry vendor read once, for "
            "real, built from the code without running it (GET/HEAD only), with "
            "your credentials, and compare the responses' shape with the spec and "
            "the stored fixtures (values are never kept)"
        ),
    )
    run_p.add_argument(
        "--max-turns",
        type=int,
        default=60,
        help="Skill runs: turn budget (default: 60)",
    )
    run_p.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Seconds before the run is killed (default: 1800 skill / 600 pipeline)",
    )
    run_p.add_argument(
        "--yes",
        action="store_true",
        help=(
            "Accept the derived input proposal without prompting (non-interactive runs); "
            "with --live-check, send the printed requests without asking"
        ),
    )
    run_p.add_argument(
        "--json",
        action="store_true",
        help="Print a single JSON result object on stdout",
    )
    run_p.set_defaults(func=_cmd_run)

    # rote deploy
    deploy_p = subparsers.add_parser(
        "deploy",
        help="Deploy an emitted pipeline to its hosting target",
        description=(
            "Push an emitted pipeline where it will actually run. Targets "
            "with a real push-deploy are wrapped thinly — the vendor CLI "
            "owns auth and output: cloudflare via `npx wrangler deploy`, "
            "dbos/dbos-ts via `npx dbos-cloud app deploy`. Runtimes with no "
            "push model (temporal, inngest, python) print honest, current "
            "hosting guidance with doc links instead of a fake action."
        ),
    )
    deploy_p.add_argument(
        "path",
        help="Emitted runtime dir, or a `rote compile --out` dir",
    )
    deploy_p.add_argument(
        "--target",
        default="auto",
        choices=["auto", "cloudflare", "dbos-cloud", "rote-cloud"],
        help=(
            "Deploy target (default: auto-detected from the emitted runtime). "
            "rote-cloud bundles a cloudflare-emitted app and uploads it to a "
            "hosted rote-cloud instance."
        ),
    )
    deploy_p.add_argument(
        "--url",
        default=None,
        help="rote-cloud: the platform base URL (default: $ROTE_CLOUD_URL)",
    )
    deploy_p.add_argument(
        "--token",
        default=None,
        help="rote-cloud: tenant API token (default: $ROTE_CLOUD_TOKEN)",
    )
    deploy_p.add_argument(
        "--input-example",
        default=None,
        metavar="JSON",
        help="rote-cloud: example input attached to the pipeline (dashboard Run buttons)",
    )
    deploy_p.add_argument(
        "--runtime",
        default=None,
        help="Which emitted runtime to deploy when several exist under <path>/runtime/",
    )
    deploy_p.add_argument(
        "--dry-run",
        action="store_true",
        help="cloudflare: build and validate without uploading (`wrangler deploy --dry-run`)",
    )
    deploy_p.add_argument(
        "--json",
        action="store_true",
        help="Print a machine-readable deploy report on stdout",
    )
    deploy_p.set_defaults(func=_cmd_deploy)

    # rote login / logout / whoami
    login_p = subparsers.add_parser(
        "login",
        help="Connect the CLI to a rote-cloud account",
        description=(
            "OAuth device-flow login: opens the browser to approve a "
            "one-time code (or prints code + URL over SSH with --device), "
            "then stores a tenant API key at ~/.local/share/rote/cloud.json. "
            "Once logged in, `rote compile` defaults to hosting on rote "
            "cloud and `rote deploy --target rote-cloud` needs no flags."
        ),
    )
    login_p.add_argument(
        "--url",
        default=None,
        help="Platform base URL (default: https://app.roteskills.com; use a dev instance URL here)",
    )
    login_p.add_argument(
        "--device",
        action="store_true",
        help="Don't open a browser — print the code + verification URL only (SSH/headless)",
    )
    login_p.set_defaults(func=_cmd_login)

    logout_p = subparsers.add_parser(
        "logout",
        help="Revoke the rote-cloud API key and clear the stored login",
    )
    logout_p.set_defaults(func=_cmd_logout)

    whoami_p = subparsers.add_parser(
        "whoami",
        help="Show the logged-in rote-cloud account (verified live)",
    )
    whoami_p.add_argument(
        "--json",
        action="store_true",
        help="Print {url, user, tenant} as JSON on stdout",
    )
    whoami_p.set_defaults(func=_cmd_whoami)

    # rote register
    register = subparsers.add_parser(
        "register",
        help="Register a compiled pipeline as an MCP tool for `rote serve`",
        description=(
            "Append or update an entry in the serve registry "
            "(~/.local/share/rote/serve-registry.json by default) from a compile run's "
            "pipeline.yaml. Tool name comes from pipeline.name, description "
            "from pipeline.description, inputSchema from the pipeline input "
            "contract. A running `rote serve` picks the change up live."
        ),
    )
    register.add_argument(
        "compiled_dir",
        help=(
            "A compile --out directory, a directory containing pipeline.yaml, "
            "or the pipeline.yaml itself"
        ),
    )
    register.add_argument(
        "--registry",
        default=None,
        help=(
            "Registry file (default: $ROTE_SERVE_REGISTRY_PATH, "
            "else ~/.local/share/rote/serve-registry.json)"
        ),
    )
    register.add_argument(
        "--runtime",
        default="dbos",
        choices=["dbos", "temporal", "cloudflare"],
        help=(
            "Runtime the compiled pipeline is deployed on (default: dbos, matching `rote compile`)"
        ),
    )
    register.add_argument(
        "--name",
        default=None,
        help="Override the MCP tool name (default: pipeline.name)",
    )
    register.add_argument(
        "--system-database-url",
        dest="system_database_url",
        default=None,
        help=(
            "DBOS: SQLAlchemy URL of the system database the emitted app "
            "checkpoints to (default: $DBOS_SYSTEM_DATABASE_URL, else the "
            "emitted app's SQLite file if a main.py is found near the "
            "pipeline)"
        ),
    )
    register.add_argument(
        "--queue-name",
        dest="queue_name",
        default=None,
        help=(
            "DBOS: queue to enqueue runs on; must match the emitted Queue "
            "(default: '<pipeline.name>-queue')"
        ),
    )
    register.add_argument(
        "--temporal-address",
        default="localhost:7233",
        help="Temporal frontend address (default: localhost:7233)",
    )
    register.add_argument(
        "--temporal-namespace",
        default="default",
        help="Temporal namespace (default: default)",
    )
    register.add_argument(
        "--task-queue",
        default=None,
        help="Temporal task queue the compiled worker polls (default: pipeline.name)",
    )
    register.add_argument(
        "--workflow-name",
        default=None,
        help=(
            "DBOS/Temporal: registered workflow name (default: the versioned "
            "name the adapter emits for this pipeline — PascalCase name + "
            "pipeline hash)"
        ),
    )
    register.add_argument(
        "--url",
        default=None,
        help="Cloudflare: the deployed worker's trigger endpoint (required for cloudflare)",
    )
    register.add_argument(
        "--status-url",
        dest="status_url",
        default=None,
        help=(
            "Cloudflare: optional status endpoint template containing "
            "'{workflow_id}' if the worker exposes a status route"
        ),
    )
    register.set_defaults(func=_cmd_register)

    # rote serve
    serve = subparsers.add_parser(
        "serve",
        help="Serve every registered pipeline as an MCP tool (stdio or HTTP)",
        description=(
            "Launch a single MCP server exposing one tool per registry entry "
            "(plus a <tool>_status companion for polling long-running "
            "workflows). stdio by default — add it to Claude Code with "
            "`claude mcp add rote -- rote serve`. Use --http for Streamable "
            "HTTP."
        ),
    )
    serve.add_argument(
        "--registry",
        default=None,
        help=(
            "Registry file (default: $ROTE_SERVE_REGISTRY_PATH, "
            "else ~/.local/share/rote/serve-registry.json)"
        ),
    )
    serve.add_argument(
        "--http",
        action="store_true",
        help="Serve Streamable HTTP instead of stdio",
    )
    serve.add_argument(
        "--host",
        default="127.0.0.1",
        help="HTTP bind host (default: 127.0.0.1; only with --http)",
    )
    serve.add_argument(
        "--port",
        type=int,
        default=8734,
        help="HTTP port (default: 8734; only with --http)",
    )
    serve.set_defaults(func=_cmd_serve)

    # rote mcp
    mcp = subparsers.add_parser(
        "mcp",
        help="Manage MCP servers: register endpoints, authenticate (OAuth), mint headers",
        description=(
            "The MCP client layer. Registered server names match the logical "
            "`mcp.server` names in compiled pipelines; `login` runs the full "
            "OAuth 2.1 dance (discovery, PKCE, a pre-registered client, else "
            "rote's client metadata document, else dynamic registration) and "
            "stores tokens durably so emitted workflows and eval trials "
            "authenticate without re-prompting."
        ),
    )
    mcp_sub = mcp.add_subparsers(dest="mcp_command", required=True)

    mcp_add = mcp_sub.add_parser("add", help="Register (or update) an MCP server")
    mcp_add.add_argument("name", help="Logical server name (matches the IR's mcp.server)")
    mcp_add.add_argument("url", help="Streamable HTTP endpoint URL")
    mcp_add.add_argument(
        "--transport",
        choices=["streamable-http", "sse"],
        default="streamable-http",
    )
    mcp_add.add_argument(
        "--client-id",
        default=None,
        help=(
            "Pre-registered OAuth client id. Used before anything else; required for "
            "servers whose authorization server offers neither a client metadata "
            "document nor dynamic registration (GitHub-class)"
        ),
    )
    mcp_add.add_argument(
        "--issuer",
        default=None,
        help=(
            "Authorization server (issuer) the --client-id was registered with. "
            "Default: discovered from the server's protected-resource metadata now"
        ),
    )
    mcp_add.add_argument(
        "--client-secret",
        default=None,
        help="Pre-registered client secret (stored 0600 in the registry)",
    )
    mcp_add.add_argument(
        "--scope",
        action="append",
        default=None,
        help="OAuth scope to request (repeatable)",
    )
    mcp_add.add_argument(
        "--header",
        action="append",
        default=None,
        metavar="'Name: value'",
        help="Static auth header (API-key schemes) instead of OAuth (repeatable)",
    )
    mcp_add.add_argument(
        "--auth",
        choices=["oauth", "none"],
        default=None,
        help=(
            "Whether the server requires auth. Omit to let `rote mcp login` find "
            "out; 'none' tells rote that runs connect unauthenticated and never park"
        ),
    )
    mcp_add.add_argument(
        "--read-only",
        action="store_true",
        help=(
            "Declare that no tool on this server writes. Read-only runs (rote run "
            "without --allow-writes, rote baseline) then call all of its tools, "
            "whether or not the server marks them readOnlyHint"
        ),
    )
    mcp_add.set_defaults(func=_cmd_mcp_add)

    mcp_import = mcp_sub.add_parser(
        "import",
        help="Copy MCP servers from your AI clients (Claude Code) into rote's registry",
        description=(
            "Read the MCP servers your AI clients use, pick the ones to import, and "
            "apply the plan only on a yes: one additive write of rote's registry "
            "(what is there is never changed without --replace), your work domains "
            "the first time, and an optional rote mcp login per server. Nothing is "
            "preselected but the servers --servers-for <skill> needs, and never one on "
            "a work domain or one whose URL may carry a key. In a terminal it asks; "
            "otherwise each question takes its flag, and the plan needs --yes."
        ),
    )
    from rote.clients import SOURCES as _CLIENT_SOURCES

    mcp_import.add_argument(
        "--from",
        dest="from_client",
        default=None,
        metavar="CLIENT",
        help=(
            "read one client only ("
            + ", ".join(s.id for s in _CLIENT_SOURCES)
            + "); default: every client found, the one you run in first"
        ),
    )
    picks = mcp_import.add_mutually_exclusive_group()
    picks.add_argument(
        "--only",
        action="append",
        default=None,
        metavar="NAME[,NAME]",
        help="import exactly these servers (names, or the client:scope:name keys shown)",
    )
    picks.add_argument(
        "--servers-for",
        default=None,
        metavar="SKILL_DIR",
        help="preselect the servers this skill names (work and risky servers excepted)",
    )
    mcp_import.add_argument(
        "--allow-url-secret",
        action="append",
        default=None,
        metavar="NAME",
        help="import NAME although its URL path looks like it carries a key (repeatable)",
    )
    mcp_import.add_argument(
        "--allow-untrusted",
        action="append",
        default=None,
        metavar="NAME",
        help=(
            "import NAME although rote could not check the client's trust in it "
            "(a project server in a client whose approvals rote cannot read; repeatable)"
        ),
    )
    mcp_import.add_argument(
        "--rename",
        action="append",
        default=None,
        metavar="OLD=NEW",
        help="register server OLD under the name NEW (repeatable)",
    )
    mcp_import.add_argument(
        "--replace",
        action="append",
        default=None,
        metavar="NAME",
        help="overwrite rote's entry NAME with the client's (needs --force without a person)",
    )
    mcp_import.add_argument(
        "--work-domain",
        action="append",
        default=None,
        metavar="DOMAIN",
        help=(
            "a work domain whose servers are never preselected, saved in your user config "
            "(repeatable; 'none' for none)"
        ),
    )
    logins = mcp_import.add_mutually_exclusive_group()
    logins.add_argument(
        "--login",
        action="append",
        default=None,
        metavar="NAME[,NAME]",
        help="log in to these imported servers after the write (needs a person: a browser)",
    )
    logins.add_argument("--no-login", action="store_true", help="do not offer logins")
    _add_onboarding_mode_flags(mcp_import)
    mcp_import.set_defaults(func=_cmd_mcp_import)

    mcp_list = mcp_sub.add_parser("list", help="List registered servers and auth status")
    mcp_list.add_argument("--json", action="store_true")
    mcp_list.set_defaults(func=_cmd_mcp_list)

    mcp_remove = mcp_sub.add_parser("remove", help="Remove a server (and its tokens)")
    mcp_remove.add_argument("name")
    mcp_remove.set_defaults(func=_cmd_mcp_remove)

    mcp_login = mcp_sub.add_parser(
        "login",
        help="Authenticate a server via OAuth (opens a browser)",
        description=(
            "Log in to a registered MCP server. Blocks until the browser returns, "
            "unless --no-wait: then a background listener waits for the browser, the "
            "command prints the authorization URL and a pending id and exits, and "
            "--complete <id> reports the outcome. That pair works from an agent's "
            "non-interactive shell."
        ),
    )
    mcp_login.add_argument("name", nargs="?", default=None)
    mcp_login.add_argument(
        "--no-browser",
        action="store_true",
        help="Print the authorization URL instead of opening a browser (SSH boxes)",
    )
    mcp_login.add_argument(
        "--callback-port",
        type=int,
        default=None,
        help=(
            "Fixed localhost callback port. Default: the first free one of the ports "
            "rote's client metadata document lists (33418-33421); any other port "
            "skips that document and uses DCR or a pre-registered client"
        ),
    )
    mcp_login.add_argument(
        "--no-wait",
        action="store_true",
        help=(
            "Start the login, print the authorization URL and a pending id, and exit; "
            "a listener on localhost waits for the browser for up to 10 minutes"
        ),
    )
    mcp_login.add_argument(
        "--complete",
        nargs="?",
        const="",
        default=None,
        metavar="ID",
        help="Finish a --no-wait login (the only pending one when no id is given)",
    )
    mcp_login.add_argument(
        "--callback-url",
        default=None,
        metavar="URL",
        help=(
            "With --complete: the http://localhost:<port>/callback?... URL a browser on "
            "another machine ended on; rote hands it to the waiting listener. '-' reads "
            "it from stdin, which keeps it out of the process list"
        ),
    )
    mcp_login.add_argument(
        "--wait",
        type=float,
        default=30.0,
        metavar="SECONDS",
        help="With --complete: how long to wait for the browser (default 30)",
    )
    mcp_login.add_argument(
        "--cancel",
        nargs="?",
        const="",
        default=None,
        metavar="ID",
        help="Cancel a --no-wait login; its listener stops",
    )
    _add_onboarding_mode_flags(mcp_login, force=False)
    mcp_login.set_defaults(func=_cmd_mcp_login)

    mcp_logout = mcp_sub.add_parser("logout", help="Clear a server's stored credentials")
    mcp_logout.add_argument("name")
    mcp_logout.set_defaults(func=_cmd_mcp_logout)

    mcp_release = mcp_sub.add_parser(
        "release",
        help="Wake workflows parked waiting for a server's auth "
        "(login does this automatically; use this when the credential "
        "was fixed another way, e.g. re-provisioned Worker secrets)",
    )
    mcp_release.add_argument("name")
    mcp_release.set_defaults(func=_cmd_mcp_release)

    mcp_parked = mcp_sub.add_parser(
        "parked",
        help="List workflows parked waiting for a server's auth, without releasing them",
    )
    mcp_parked.add_argument("name", nargs="?", default=None, help="Only this server")
    mcp_parked.add_argument("--json", action="store_true", help="Machine-readable output")
    mcp_parked.set_defaults(func=_cmd_mcp_parked)

    mcp_headers = mcp_sub.add_parser(
        "headers",
        help="Print fresh auth headers as JSON (for headersHelper and scripts)",
    )
    mcp_headers.add_argument("name")
    mcp_headers.set_defaults(func=_cmd_mcp_headers)

    mcp_export = mcp_sub.add_parser(
        "export",
        help="Export a login as Worker secrets (Cloudflare provisioning)",
    )
    mcp_export.add_argument("name")
    mcp_export.add_argument(
        "--json",
        action="store_true",
        help="JSON object form, pipeable to `npx wrangler secret bulk`",
    )
    mcp_export.set_defaults(func=_cmd_mcp_export)

    # rote analyze
    analyze = subparsers.add_parser(
        "analyze",
        help="Dry-run the compiler: report a skill's compiled shape, emit no runtime code",
        description=(
            "Run the rote-compile agent against a skill and print what "
            "compilation would produce — node-kind breakdown, roteness "
            "(deterministic vs. LLM-sampled steps), mandatory checks, HITL "
            "gates, agent loops, and which runtimes can target it — without "
            "emitting any runtime code. This is the `plan` to `compile`'s "
            "`apply`. Pass --out to keep the pipeline.yaml for a later `rote emit`."
        ),
    )
    analyze.add_argument("skill_path", help="Path to the source skill directory")
    analyze.add_argument(
        "--agent",
        choices=sorted(DRIVERS),
        default=None,
        help=(
            "Agent runtime for the compiler (default: auto-detect — claude "
            "CLI first, then codex, then the anthropic API). openai-api is "
            "explicit-only and requires --model."
        ),
    )
    analyze.add_argument(
        "--model",
        default=None,
        help="Override the LLM model the compiler uses (default: the driver's default).",
    )
    analyze.add_argument(
        "--effort",
        choices=EFFORT_LEVELS,
        default=None,
        help=EFFORT_HELP,
    )
    analyze.add_argument(
        "--out",
        default=None,
        help=(
            "Keep the compiled IR + stubs in this directory (ready for "
            "`rote emit`). Default: produce them in a temp dir and discard "
            "after reporting."
        ),
    )
    analyze.add_argument(
        "--json",
        action="store_true",
        help="Emit the analysis report as JSON instead of text.",
    )
    analyze.set_defaults(func=_cmd_analyze)
    _add_system1_flags(analyze)
    analyze.add_argument(
        "--no-verify-tests",
        action="store_true",
        help="Don't run the tests the compiler agent writes (see rote compile --help)",
    )
    analyze.add_argument(
        "--strict-provenance",
        action="store_true",
        help=(
            "Fail (exit 1) on every unresolved provenance finding, for CI. By default "
            "only a citation the skill does not contain fails; assumptions never do, "
            "and a provenance gap the repair rounds did not close (an unmarked outcome, "
            "an assumption missing from Open questions) is a warning"
        ),
    )
    analyze.add_argument(
        "--trust-skill",
        action="store_true",
        help="Run the agent-written tests even without an OS sandbox (see rote compile --help)",
    )
    analyze.add_argument(
        "--no-mcp",
        action="store_true",
        help=(
            "Explicitly disable live MCP tools for api/openai-api analysis. "
            "Does not change the pipeline's emitted MCP bindings."
        ),
    )

    # rote example
    example_p = subparsers.add_parser(
        "example",
        help="List the bundled examples, or copy one here to run for free",
        description=(
            "With no name, list the examples bundled with rote. With a name, "
            "copy that example (its skill, compiled pipeline, working "
            "extracted/ modules and sample input) into ./<name> and print the "
            "exact emit, run and eval commands for the copy. No clone, no "
            "credentials, no tokens."
        ),
    )
    example_p.add_argument("name", nargs="?", help="Example to copy (omit to list them)")
    example_p.add_argument(
        "--force",
        action="store_true",
        help="Copy into ./<name> even if it exists, overwriting the example's own files",
    )
    example_p.set_defaults(func=_cmd_example)

    # rote skills
    skills_p = subparsers.add_parser(
        "skills",
        help="The skills your AI clients load, ranked by how ready they are to compile",
        description=(
            "Find the skills Claude Code (user, project, enabled plugins), Codex, "
            "Gemini CLI, OpenCode and .agents/skills load, plus rote's bundled demo, "
            "and rank them: the demo, then skills whose MCP servers are all registered "
            "and logged in, then those a login away, then the rest. Reads files only; "
            "contacts nothing."
        ),
    )
    skills_sub = skills_p.add_subparsers(dest="skills_command", required=True)
    skills_list = skills_sub.add_parser("list", help="List skills, best first")
    skills_list.add_argument("--json", action="store_true", help="One JSON object on stdout")
    skills_list.add_argument(
        "--limit", type=int, default=None, metavar="N", help="Show only the first N"
    )
    skills_list.set_defaults(func=_cmd_skills_list)

    _add_skill_parser(subparsers)

    # rote doctor
    doctor = subparsers.add_parser(
        "doctor",
        help="Preflight check: compiler drivers, runtime deps, MCP auth, and registered apps",
        description=(
            "Read-only preflight that tells you (or your agent) whether a "
            "compilation will succeed before spending a cent: which compiler "
            "drivers are usable, which optional runtime dependencies are "
            "installed, the auth state of every registered MCP server, and "
            "every emitted app rote knows about. Nothing is executed and "
            "nothing is spent. Exits non-zero only when no compiler driver "
            "is available, so it works as a scriptable gate."
        ),
    )
    doctor.add_argument(
        "--json",
        action="store_true",
        help="Emit the report as a single JSON object to stdout instead of the checklist.",
    )
    doctor.set_defaults(func=_cmd_doctor)

    # rote init
    init_p = subparsers.add_parser(
        "init",
        help="Onboarding: pick your defaults once (hosting, runtime, driver, structure)",
        description=(
            "Choose the defaults that shape every later command (rote cloud or a "
            "local runtime, compiler driver and model, who decides a compile's "
            "structure) and save them to a config file. Writes the user-level "
            "config (~/.config/rote/config.yaml) by default; --project writes a "
            "./rote.yaml that overrides it for this repo. In a terminal it asks; "
            "otherwise each question takes its flag, or its default with --yes, "
            "and the plan is applied only with --yes (plus --force to replace a file)."
        ),
    )
    init_p.add_argument(
        "--project",
        action="store_true",
        help="Write ./rote.yaml (per-repo, beats the user config) instead",
    )
    init_p.add_argument(
        "--deploy", choices=("rote-cloud", "none"), default=None, help="rote cloud, or local"
    )
    init_p.add_argument("--runtime", default=None, help="default runtime (local hosting)")
    init_p.add_argument(
        "--agent", default=None, help="compiler driver, or auto to detect it at run time"
    )
    init_p.add_argument("--model", default=None, help="compiler model ('' for the driver default)")
    init_p.add_argument("--structure", default=None, help="haiku, sonnet or rules")
    init_p.add_argument("--decider", default=None, help="auto, rules, clef or jev")
    init_p.add_argument("--tiebreak", default=None, help="haiku or none")
    login_group = init_p.add_mutually_exclusive_group()
    login_group.add_argument(
        "--login", dest="login", action="store_true", default=None, help="log in to rote cloud"
    )
    login_group.add_argument(
        "--no-login", dest="login", action="store_false", help="do not log in to rote cloud"
    )
    _add_onboarding_mode_flags(init_p)
    init_p.set_defaults(func=_cmd_init)

    # rote config
    config_p = subparsers.add_parser(
        "config",
        help="Show the effective defaults and where each one comes from",
        description=(
            "Print every configurable default with its effective value and "
            "the layer that set it (flag > ROTE_* env > project rote.yaml > "
            "user config > built-in), plus the rote cloud login state."
        ),
    )
    config_p.add_argument("--json", action="store_true", help="Machine-readable report")
    config_p.set_defaults(func=_cmd_config)

    # rote eval
    eval_cmd = subparsers.add_parser(
        "eval",
        help="Before/after scorecard: speed, cost, determinism (static, no execution)",
        description=(
            "Estimate what a skill costs to run as raw agent instructions "
            "versus as its compiled pipeline — wall clock, dollars across "
            "a sampling of current models at live official prices, and "
            "determinism (LLM sampling surface). Static: nothing executes."
        ),
    )
    eval_cmd.add_argument(
        "target",
        help=("A compiled directory, a compile --out directory, or a pipeline.yaml path"),
    )
    eval_cmd.add_argument(
        "--skill",
        default=None,
        help=(
            "Source skill directory for the 'before' baseline (default: the "
            "pipeline's recorded source_skill, if it resolves)"
        ),
    )
    eval_cmd.add_argument(
        "--provider",
        default="anthropic",
        help="LLM provider whose current model lineup to price (default: anthropic)",
    )
    eval_cmd.add_argument(
        "--runtime",
        default=None,
        choices=[*sorted(ADAPTERS), "rote-cloud"],
        help=(
            "Where the pipeline runs, for runtime infrastructure cost: "
            "cloudflare and rote-cloud price Cloudflare Workflows steps, "
            "requests, CPU and storage at live rates (default: the runtime "
            "of a compile --out dir, if it holds exactly one)"
        ),
    )
    eval_cmd.add_argument(
        "--prior",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help=(
            "Override an estimator prior (repeatable), e.g. "
            "--prior transcript_growth_per_turn=5962 — feed back the "
            "re-fits a previous --run reported. Per-MCP-tool payload: "
            "--prior payload_tokens_per_tool.<tool>=12000"
        ),
    )
    eval_cmd.add_argument(
        "--json",
        action="store_true",
        help="Emit the scorecard as JSON instead of Markdown",
    )
    eval_cmd.add_argument(
        "--out",
        default=None,
        help="Write the scorecard to a file instead of stdout",
    )
    eval_cmd.add_argument(
        "--run",
        action="store_true",
        help=(
            "Empirical mode: actually execute both sides --trials times and "
            "append a Measured section. Runs the emitted pipeline (fast, "
            "cheap) and the raw skill via `claude -p` (minutes per trial, "
            "billed to your Claude subscription). Requires --input."
        ),
    )
    eval_cmd.add_argument(
        "--inference",
        default=None,
        choices=INFERENCE_PROVIDERS,
        help=(
            "--run only: who serves (and pays for) the pipeline's judges. "
            "Default: auto-detect, cheapest to you first. Recorded with the "
            "measurement — the same token count costs differently per lane."
        ),
    )
    eval_cmd.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Trials per side in --run mode (default: 3)",
    )
    eval_cmd.add_argument(
        "--input",
        default=None,
        help=(
            "Task file for --run: the pipeline input payload as JSON, or an "
            'envelope {"input": {...}, "signals": {"<signal>": <payload>}} '
            "for pipelines with HITL gates"
        ),
    )
    eval_cmd.add_argument(
        "--model",
        default=None,
        help=("Model for the skill-side agent trials (default: the live lineup's mid tier)"),
    )
    eval_cmd.add_argument(
        "--max-turns",
        type=int,
        default=60,
        help="Turn budget per skill-side trial (default: 60)",
    )
    eval_cmd.add_argument(
        "--runtime-dir",
        default=None,
        help=(
            "Emitted runtime app for the pipeline trials (default: the "
            "compile --out layout's runtime/dbos or runtime/python sibling)"
        ),
    )
    eval_cmd.add_argument(
        "--python",
        default=None,
        dest="python",
        help=(
            "Python executable for pipeline trials (default: the current "
            "interpreter; point at your app's venv if it has its own deps)"
        ),
    )
    skill_ab = eval_cmd.add_argument_group(
        "--run --runtime skill",
        "The paired eval: the source skill and its compiled skill (emitted with --runtime "
        "skill, installed into a throwaway state, run through rote skill run), each by a "
        "claude -p subagent with the same flags, tools and simulated person, on the same "
        "input. Reports wall time, all four token buckets, cost, turns, the context the "
        "skill left (from a control run per side), and field-by-field fidelity, with the "
        "original against itself as the noise floor; checks that the compiled side ran "
        "through the driver and that every gate was answered by the person. --model "
        "defaults to the compiler's default model. TARGET may also be a source skill, "
        "which is compiled first with a pinned argv (--local, --decider rules).",
    )
    skill_ab.add_argument(
        "--case",
        default=None,
        metavar="ID",
        help="run a bench case from <example>/fidelity/cases.yaml (its input, fixture "
        "servers and gate answers) instead of --input",
    )
    skill_ab.add_argument(
        "--budget",
        type=float,
        default=None,
        metavar="USD",
        help="required: the spend cap; the eval stops before a run whose estimate would "
        "cross it (counted in --ledger, so several invocations can share one cap)",
    )
    skill_ab.add_argument(
        "--ledger",
        default=None,
        metavar="FILE",
        help="the spend ledger, a JSON file (default: ledger.json in the work dir)",
    )
    skill_ab.add_argument(
        "--work-dir",
        default=None,
        metavar="DIR",
        help="where each trial's transcript, run folder and result are kept (default: a "
        "new temp dir, printed)",
    )
    skill_ab.add_argument(
        "--compile-model",
        default=None,
        metavar="MODEL",
        help="the compiler's model, when TARGET is a source skill (default: the compiler's)",
    )
    eval_cmd.set_defaults(func=_cmd_eval)

    # rote baseline
    baseline = subparsers.add_parser(
        "baseline",
        help="Run the raw skill once as an agent — measured and observed (pre-compilation)",
        description=(
            "Execute the source skill via `claude -p` before compiling it: "
            "capture measured before-metrics (wall clock, turns, tokens, "
            "cost), the full transcript, and every MCP tool call the agent "
            "actually made with its real payloads. rote's registered MCP "
            "servers are injected into the run (`rote mcp login` covers it); "
            "by default only tools whose server declares readOnlyHint are "
            "callable — pass --allow-writes if this skill's side effects are "
            "acceptable to fire once. Billed to your Claude subscription; "
            "minutes per trial."
        ),
    )
    baseline.add_argument("skill_path", help="Path to the source skill directory")
    baseline.add_argument(
        "--out",
        required=True,
        help="Output directory — artifacts land under <out>/baseline/",
    )
    baseline.add_argument(
        "--input",
        default=None,
        help=(
            "Task file: the skill's input payload as a JSON object. Omit to "
            "have a representative input derived from SKILL.md and confirmed "
            "with you before any money is spent."
        ),
    )
    baseline.add_argument(
        "--yes",
        action="store_true",
        help="Accept the derived input proposal without prompting (non-interactive runs)",
    )
    baseline.add_argument(
        "--trials",
        type=int,
        default=1,
        help="How many times to run the skill (default: 1)",
    )
    baseline.add_argument(
        "--model",
        default=None,
        help=f"Model for the agent run (default: {CLAUDE_DEFAULT_MODEL})",
    )
    baseline.add_argument(
        "--max-turns",
        type=int,
        default=60,
        help="Turn budget per trial (default: 60)",
    )
    baseline.add_argument(
        "--allow-writes",
        action="store_true",
        help=(
            "Lift the read-only MCP gate: allow every tool on every wired "
            "server, including destructive ones. The skill runs exactly as "
            "in production — sent emails send, pushed records push."
        ),
    )
    baseline.add_argument(
        "--json",
        action="store_true",
        help="Print the baseline summary as JSON on stdout",
    )
    baseline.set_defaults(func=_cmd_baseline)

    return parser


#: Retired subcommand names that still dispatch, mapped to their replacement.
#: ``rote graduate`` shipped through 0.10.x, so it lives in people's shell
#: history, scripts and CI. It keeps working, and it is absent from ``--help``
#: on purpose: the alias exists so old invocations do not break, not to offer a
#: second spelling anyone should learn.
_RETIRED_COMMANDS = {"graduate": "compile"}


def _resolve_retired_command(argv: list[str]) -> list[str]:
    """Rewrite a retired subcommand, telling the caller once on stderr.

    Only the first token that is not an option is considered, which is
    unambiguously the subcommand because no top-level option takes a value.
    That keeps ``rote compile ./graduate`` (a skill directory that happens to
    be named after the old verb) untouched.
    """
    for index, token in enumerate(argv):
        if token.startswith("-"):
            continue
        replacement = _RETIRED_COMMANDS.get(token)
        if replacement is None:
            return argv
        print(
            f"rote: `{token}` is now `{replacement}` — running `rote {replacement}`. "
            f"The old name still works; update your scripts when convenient.",
            file=sys.stderr,
        )
        return [*argv[:index], replacement, *argv[index + 1 :]]
    return argv


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(argv) if argv is not None else sys.argv[1:]
    if raw[:1] == ["bench"]:
        # A dev tool, dispatched before the user-facing parser so it stays
        # out of `rote --help` (argparse cannot hide a subcommand cleanly).
        # It spends tokens and needs the repo's examples/; it ships in the
        # wheel so it runs from an immutable install of the branch it judges.
        from rote.bench.runner import main as bench_main

        return bench_main(raw[1:])
    parser = _build_parser()
    args = parser.parse_args(_resolve_retired_command(raw))
    # The words as typed, so a command that stops for a missing flag can
    # print the exact command to run next (rote.interact.report).
    args.raw_argv = raw

    if not getattr(args, "command", None):
        # No subcommand — print a helpful banner and usage.
        print(f"rote {__version__} — compile fuzzy AI skills into deterministic workflows")
        print()
        parser.print_help()
        return 0

    func = args.func
    if not getattr(args, "json", False):
        return int(func(args))
    return _run_json_command(func, args)


class _RecordingStream:
    """A text stream proxy that remembers what passed through it."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.text: list[str] = []

    def write(self, data: str) -> int:
        self.text.append(data)
        return int(self._inner.write(data))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _run_json_command(func: Callable[[argparse.Namespace], int], args: argparse.Namespace) -> int:
    """Run a ``--json`` command and hold it to the stdout contract.

    ``--json`` promises one JSON object on stdout, so a pipe into ``jq``
    gets something to parse whatever happened. Handlers print their own
    payload on success, but many early-exit error paths print only to
    stderr, which left a consumer with empty stdout and a nonzero exit.
    Enforcing it here covers every ``--json`` command at once: when a
    handler fails without writing stdout, the error it printed is
    re-emitted as ``{"ok": false, "exit_code": N, "error": "..."}``. The
    exit code is unchanged.
    """
    out = _RecordingStream(sys.stdout)
    err = _RecordingStream(sys.stderr)
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = int(func(args))
    if code != 0 and not "".join(out.text).strip():
        print(json.dumps({"ok": False, "exit_code": code, "error": _last_error(err.text)}))
    return code


def _last_error(stderr_chunks: list[str]) -> str:
    """The error a failed command reported on stderr, for its JSON error object.

    Prefers the last ``error: ...`` line (the CLI's convention) plus the
    detail lines printed after it; falls back to the stderr tail.
    """
    lines = "".join(stderr_chunks).rstrip().splitlines()
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].startswith("error: "):
            return "\n".join([lines[i].removeprefix("error: "), *lines[i + 1 :]])
    tail = [ln for ln in lines if ln.strip()][-5:]
    return "\n".join(tail) if tail else "command failed without an error message"


if __name__ == "__main__":
    raise SystemExit(main())
