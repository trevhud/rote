"""rote CLI — entry point.

The north star for this CLI:

    rote graduate ./path/to/skill --runtime temporal --out ./graduated/

One command, skill in, runnable workflow out. Every other command is a
building block for that flow:

    rote emit    <pipeline.yaml> --runtime temporal  # IR → code only
    rote analyze <skill-path>                        # graduator dry run
    rote eval    <graduated-dir>                     # before/after scorecard

Internally ``rote graduate`` runs the ``rote-graduate`` skill in an agent
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
import re
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from rote import __version__
from rote.adapters import ADAPTERS, get_adapter
from rote.graduator import Graduator, GraduatorError
from rote.graduator.drivers import available_drivers
from rote.graduator.events import GraduationEvent
from rote.ir import Pipeline, load_pipeline

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


def _unimplemented_stubs(written: dict[str, Path]) -> list[str]:
    """The emitted ``extracted/*`` modules still carrying a stub marker — the
    agent's fill-in TODO list.

    "Still a stub" is read straight off disk: a Python stub raises
    ``NotImplementedError``; a TypeScript stub throws ``"… stub not
    implemented"`` (or, for agent loops, ``"… requires an agent runtime"``).
    A node the user has filled in, or an ``mcp``-backed external_call that
    emits a working call, no longer matches, so it drops off the list. On
    re-emit a preserved (user-edited) file's fresh content lands in a
    ``.new`` sibling; we read the authoritative on-disk target, not the
    ``.new``, so a filled-in stub isn't re-reported as a TODO.
    """
    markers = ("NotImplementedError", "stub not implemented", "requires an agent runtime")
    stubs: list[str] = []
    for label, path in written.items():
        if not label.startswith("extracted/") or label.rsplit("/", 1)[-1] == "__init__":
            continue
        target = path.with_name(path.name[:-4]) if path.name.endswith(".new") else path
        try:
            text = target.read_text(encoding="utf-8")
        except OSError:
            continue
        if any(marker in text for marker in markers):
            stubs.append(str(target))
    return sorted(stubs)


def _cmd_emit(args: argparse.Namespace) -> int:
    """Render a runtime adapter from a pipeline.yaml.

    Pure IR → code. No agent involved. This is the lowest-level
    subcommand and the simplest to reason about — it should be used
    when the user already has a hand-written or previously-graduated
    ``pipeline.yaml`` and just wants to re-render the runtime code.
    """
    pipeline_path = Path(args.pipeline_yaml)
    if not pipeline_path.exists():
        print(f"error: pipeline file not found: {pipeline_path}", file=sys.stderr)
        return 2

    try:
        pipeline = load_pipeline(pipeline_path)
    except Exception as e:
        print(f"error: failed to load pipeline: {e}", file=sys.stderr)
        return 1

    try:
        adapter = get_adapter(args.runtime, external_backend=args.backend)
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
    from rote.app_registry import record_app

    record_app(out_dir, args.runtime, pipeline.name)

    mcp_servers = _mcp_requirements(pipeline)

    if args.json:
        import json

        payload = {
            "pipeline": {"name": pipeline.name, "version": pipeline.version},
            "runtime": args.runtime,
            "out_dir": str(out_dir.resolve()),
            "written": {label: str(path) for label, path in written.items()},
            "preserved_new_files": _preserved_new_files(written),
            "unimplemented_stubs": _unimplemented_stubs(written),
            "mcp_servers": mcp_servers,
        }
        print(json.dumps(payload, indent=2))
        return 0

    print(f"rote: emitted {pipeline.name} v{pipeline.version} → {out_dir}")
    _print_written(written)
    if mcp_servers:
        rendered = ", ".join(f"{e['server']} [{e['auth']}]" for e in mcp_servers)
        print(f"  required MCP servers: {rendered}")
        for line in _mcp_recommendation_lines(mcp_servers):
            print(line)
    return 0


# ───────── Subcommand stubs (graduate / analyze / eval) ─────────


#: The rote-graduate skill runs seven numbered phases; used to render
#: ``[phase N/7]`` progress lines. Kept in sync with skills/rote-graduate.
_GRADUATE_TOTAL_PHASES = 7


def _format_token_note(tokens: dict[str, int] | None) -> str:
    """Compact ``(in 40.2k / out 8.1k tok)`` annotation, or ``""``.

    Counts ≥1000 are abbreviated to one decimal of ``k``; smaller ones
    print raw. Empty string when there's nothing to show, so callers can
    append it unconditionally.
    """
    if not tokens:
        return ""

    def _h(n: int) -> str:
        return f"{n / 1000:.1f}k" if n >= 1000 else str(n)

    return f" (in {_h(tokens.get('input', 0))} / out {_h(tokens.get('output', 0))} tok)"


def _graduate_progress_printer() -> Callable[[GraduationEvent], None]:
    """One-line live progress to stderr for ``rote graduate``.

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

    def printer(event: GraduationEvent) -> None:
        nonlocal last_tokens
        line: str | None
        if event.type == "phase":
            name = event.phase_name or ""
            line = f"[phase {event.phase}/{_GRADUATE_TOTAL_PHASES}] {name}".rstrip()
        elif event.type == "tool":
            loc = f" {event.path}" if event.path else ""
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


class _JsonlProgressSink:
    """Append-per-event JSONL progress sink for ``rote graduate``.

    A graduation is a long, real-money agent loop; an agent (or a cloud
    runner) that launched it wants to tail its progress live. This sink
    writes one JSON object per :class:`GraduationEvent` to a file —
    ``json.dumps`` of the event's fields with the ``None``-valued ones
    dropped for compactness — flushed per event so a tailer sees each
    line the instant it lands. It runs *alongside* the stderr printer, not
    instead of it.

    Two enrichments live only in the serialized line, never on the wire
    :class:`GraduationEvent` (whose schema is locked across a network
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

    def __init__(self, path: Path, model_id: str) -> None:
        self._path = path
        self._model_id = model_id
        self._prices = self._resolve_prices(model_id)
        self._file: Any = None
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            # Truncate at start: one run owns the file for its lifetime.
            self._file = self._path.open("w", encoding="utf-8")
        except OSError:
            self._file = None

    @staticmethod
    def _resolve_prices(model_id: str) -> tuple[float, float] | None:
        """``(input, output)`` USD/Mtok for ``model_id``, or ``None``.

        One live catalog fetch at construction. Offline (``PricingError``)
        or an unpriced model yields ``None`` — cost enrichment is then
        silently skipped rather than failing the run.
        """
        from rote.eval.pricing import PricingError, fetch_catalog

        try:
            catalog = fetch_catalog(provider="anthropic")
        except PricingError:
            return None
        return catalog.price_for(model_id)

    def _cost_usd(self, tokens: dict[str, int]) -> float | None:
        if self._prices is None:
            return None
        input_per_mtok, output_per_mtok = self._prices
        cost = (
            tokens.get("input", 0) / 1e6 * input_per_mtok
            + tokens.get("output", 0) / 1e6 * output_per_mtok
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

    def emit(self, event: GraduationEvent) -> None:
        """Serialize one event as an NDJSON line (never raises)."""
        try:
            obj = {k: v for k, v in dataclasses.asdict(event).items() if v is not None}
            if event.tokens:
                cost = self._cost_usd(event.tokens)
                if cost is not None:
                    obj["cost_usd"] = cost
            self._write(obj)
        except Exception:
            # A progress-file bug must never sink a paid run.
            pass

    def write_summary(self, summary: dict[str, Any]) -> None:
        """Write the final ``type: "summary"`` digest as the last line.

        Prices the run's ``total_tokens`` into a ``cost_usd`` field the
        same way per-event lines are priced. Never raises.
        """
        try:
            total = summary.get("total_tokens")
            if isinstance(total, dict):
                cost = self._cost_usd(total)
                if cost is not None:
                    summary = {**summary, "cost_usd": cost}
            self._write(summary)
        except Exception:
            pass

    def close(self) -> None:
        if self._file is not None:
            with contextlib.suppress(OSError):
                self._file.close()
            self._file = None


def _cmd_graduate(args: argparse.Namespace) -> int:
    """Run the full one-shot graduation flow.

    Steps:

    1. Resolve the source skill directory; reject if it's not a skill bundle.
    2. Run the graduator agent (via ``Graduator.graduate``) to produce
       ``<out>/graduated/pipeline.yaml`` plus extracted modules and stubs.
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

    out_dir = Path(args.out)
    graduated_dir = out_dir / "graduated"
    runtime_dir = out_dir / "runtime" / args.runtime

    # A JSONL progress sink (opt-in via --progress-file) runs alongside the
    # stderr printer: both fire for every event. The sink writes one machine-
    # readable line per event for an agent tailing the run; the printer keeps
    # the human view. A sink failure must never take down a paid run, so its
    # body is self-guarded (and emit_safely wraps the whole callback anyway).
    printer = _graduate_progress_printer()
    progress_sink: _JsonlProgressSink | None = None
    if args.progress_file:
        # Price against the model the run actually uses: the --model override
        # when given, else the graduator's default (the subscription path's
        # Sonnet). Both subprocess and api drivers share this default.
        from rote.graduator.drivers.claude import DEFAULT_MODEL as _GRADUATOR_DEFAULT_MODEL

        progress_sink = _JsonlProgressSink(
            Path(args.progress_file),
            model_id=args.model or _GRADUATOR_DEFAULT_MODEL,
        )

    def _on_event(event: GraduationEvent) -> None:
        printer(event)
        if progress_sink is not None:
            progress_sink.emit(event)

    graduator = Graduator(agent=args.agent, model=args.model, on_event=_on_event)

    try:
        result = asyncio.run(graduator.graduate(skill_path, graduated_dir, update=args.update))
    except GraduatorError as e:
        if progress_sink is not None:
            progress_sink.close()
        print(f"rote graduate: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        if progress_sink is not None:
            progress_sink.close()
        print("rote graduate: interrupted", file=sys.stderr)
        return 130

    try:
        adapter = get_adapter(args.runtime, external_backend=args.backend)
    except KeyError as e:
        if progress_sink is not None:
            progress_sink.close()
        print(f"error: {e.args[0]}", file=sys.stderr)
        return 2

    written = adapter.emit(result.pipeline, runtime_dir)

    from rote.app_registry import record_app

    record_app(runtime_dir, args.runtime, result.pipeline.name)

    # ── Scorecard (auxiliary: a price-fetch outage must not sink a
    # just-completed, real-money graduation run) ──
    scorecard_path: Path | None = None
    if not args.no_eval:
        from rote.eval import build_scorecard_for
        from rote.eval.pricing import PricingError

        try:
            scorecard = build_scorecard_for(
                result.pipeline,
                graduated_dir / "pipeline.yaml",
                skill_path,
                provider="anthropic",
            )
            scorecard_path = graduated_dir / "scorecard.md"
            scorecard_path.write_text(scorecard.to_markdown() + "\n", encoding="utf-8")
        except PricingError as e:
            print(
                f"rote graduate: warning: skipped scorecard (live price fetch "
                f"failed: {e}). Generate it later with: rote eval {out_dir}",
                file=sys.stderr,
            )

    # ── Summary ── (human text to stderr under --json, so stdout is the
    # JSON object only — the same stdout-contract split as `rote mcp headers`)
    summary_stream = sys.stderr if args.json else sys.stdout
    print(
        f"rote graduate: ✓ {result.pipeline.name} v{result.pipeline.version}",
        file=summary_stream,
    )
    print(f"  driver: {result.driver_name}", file=summary_stream)
    if result.driver_metadata:
        meta_str = ", ".join(f"{k}={v}" for k, v in result.driver_metadata.items())
        print(f"  metadata: {meta_str}", file=summary_stream)
    print(f"  graduated artifacts: {graduated_dir}", file=summary_stream)
    if scorecard_path is not None:
        print(f"  eval scorecard: {scorecard_path}", file=summary_stream)
    print(f"  emitted runtime ({args.runtime}): {runtime_dir}", file=summary_stream)
    _print_written(written, indent="    ", stream=summary_stream)

    # ── Required MCP servers ── advisory only: the run parks-on-auth as
    # the backstop, so a dead credential is never fatal — but telling the
    # user now beats discovering it mid-flight.
    mcp_servers = _mcp_requirements(result.pipeline)
    if mcp_servers:
        rendered = ", ".join(f"{e['server']} [{e['auth']}]" for e in mcp_servers)
        print(f"  required MCP servers: {rendered}", file=summary_stream)
        for line in _mcp_recommendation_lines(mcp_servers):
            print(line, file=summary_stream)

    if args.json:
        import json

        payload = {
            "pipeline": {"name": result.pipeline.name, "version": result.pipeline.version},
            "runtime": args.runtime,
            "out_dir": str(out_dir.resolve()),
            "graduated_dir": str(graduated_dir.resolve()),
            "runtime_dir": str(runtime_dir.resolve()),
            "scorecard": str(scorecard_path.resolve()) if scorecard_path is not None else None,
            "driver": result.driver_name,
            "driver_metadata": result.driver_metadata,
            "written": {label: str(path) for label, path in written.items()},
            "preserved_new_files": _preserved_new_files(written),
            "unimplemented_stubs": _unimplemented_stubs(written),
            "mcp_servers": mcp_servers,
        }
        print(json.dumps(payload, indent=2))

    # ── Machine end-of-run digest ── The per-event `complete` line is for
    # a human tailer; this is the structured summary an agent reads: the
    # graduated pipeline's shape (roteness / node-kind counts, reusing the
    # same pure analysis `rote analyze` prints), where the artifacts landed,
    # the remaining stub TODOs, and the run's total token spend + cost. It's
    # the LAST NDJSON line, written after everything else is on disk.
    if progress_sink is not None:
        analysis = _build_analysis(
            result.pipeline, result.driver_name, result.driver_metadata, skill_path
        )
        analysis_nodes = analysis["nodes"]
        assert isinstance(analysis_nodes, dict)
        meta = result.driver_metadata
        total_tokens = {
            "input": int(meta.get("input_tokens") or 0),
            "output": int(meta.get("output_tokens") or 0),
        }
        progress_sink.write_summary(
            {
                "type": "summary",
                "roteness": analysis["roteness"],
                "node_kinds": analysis_nodes["by_kind"],
                "nodes": analysis_nodes["total"],
                "graduated_dir": str(graduated_dir.resolve()),
                "runtime_dir": str(runtime_dir.resolve()),
                "unimplemented_stubs": _unimplemented_stubs(written),
                "mcp_servers": analysis["mcp_servers"],
                "total_tokens": total_tokens,
            }
        )
        progress_sink.close()
    return 0


# ───────── Subcommand: register ─────────


def _resolve_pipeline_yaml(path: Path) -> Path | None:
    """Find the pipeline.yaml behind a user-supplied path.

    Accepts the file itself, a directory containing one, or a
    ``rote graduate --out`` directory (which nests it under graduated/).
    """
    if path.is_file():
        return path
    for candidate in (path / "pipeline.yaml", path / "graduated" / "pipeline.yaml"):
        if candidate.is_file():
            return candidate
    return None


def _derive_dbos_sqlite_url(graduated_arg: Path, pipeline_name: str) -> str | None:
    """Locate the emitted DBOS app dir and derive its default SQLite URL.

    The emitted main.py defaults its system database to
    ``sqlite:///<app dir>/<pipeline.name>.dbos.sqlite``. Given what the
    user passed to ``rote register`` (the pipeline.yaml, its directory,
    or a ``rote graduate --out`` directory), the app dir is wherever a
    ``main.py`` is found in the known layouts.
    """
    base = graduated_arg if graduated_arg.is_dir() else graduated_arg.parent
    candidates = (
        base,  # the emitted runtime dir itself
        base / "runtime" / "dbos",  # a graduate --out dir
        base.parent / "runtime" / "dbos",  # the graduated/ dir inside one
    )
    for app_dir in candidates:
        if (app_dir / "main.py").is_file():
            return f"sqlite:///{(app_dir / f'{pipeline_name}.dbos.sqlite').resolve()}"
    return None


def _cmd_register(args: argparse.Namespace) -> int:
    """Add or update a registry entry for a graduated pipeline.

    Reads the pipeline.yaml from a graduate run, derives the MCP tool
    name / description / inputSchema from it, attaches the runtime
    trigger config from the flags, and upserts ``~/.rote/registry.json``
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

    pipeline_yaml = _resolve_pipeline_yaml(Path(args.graduated_dir))
    if pipeline_yaml is None:
        print(
            f"error: no pipeline.yaml found at or under: {args.graduated_dir}",
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
            or _derive_dbos_sqlite_url(Path(args.graduated_dir), pipeline.name)
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

    try:
        from rote.serve.server import build_server
    except ImportError as e:
        print(
            f"error: fastmcp is not installed ({e}). "
            "Install the serve extra: pip install 'rote[serve]'",
            file=sys.stderr,
        )
        return 2

    registry_path = Path(args.registry) if args.registry else default_registry_path()
    if not registry_path.exists():
        print(
            f"rote serve: registry {registry_path} does not exist yet — serving an "
            f"empty tool list. Register a pipeline with `rote register`.",
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

    registry = load_registry()
    replacing = args.name in registry.servers
    registry.servers[args.name] = McpServerConfig(
        url=args.url,
        transport=args.transport,
        client_id=args.client_id,
        client_secret=args.client_secret,
        scopes=args.scope or None,
        headers=headers or None,
    )
    path = save_registry(registry)
    verb = "updated" if replacing else "added"
    print(f"rote mcp: {verb} {args.name!r} → {args.url} ({path})")
    if not headers:
        print(f"  authenticate with: rote mcp login {args.name}")
    return 0


def _cmd_mcp_list(args: argparse.Namespace) -> int:
    import json as _json

    from rote.mcp import auth_status, load_registry, read_token_file

    registry = load_registry()
    entries: list[dict[str, object]] = []
    for name, config in sorted(registry.servers.items()):
        doc = read_token_file(name)
        status = auth_status(config, doc)
        entries.append(
            {"name": name, "url": config.url, "transport": config.transport, "auth": status}
        )

    if args.json:
        print(_json.dumps({"servers": entries}, indent=2))
        return 0
    if not entries:
        print("rote mcp: no servers registered — add one with: rote mcp add <name> <url>")
        return 0
    width = max(len(str(e["name"])) for e in entries)
    for e in entries:
        print(f"  {str(e['name']).ljust(width)}  {e['auth']:<22}  {e['url']}")
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
    import asyncio

    from rote.mcp import McpAuthError, load_registry
    from rote.mcp.auth import login

    registry = load_registry()
    config = registry.servers.get(args.name)
    if config is None:
        print(
            f"error: no MCP server named {args.name!r} — register it first: "
            f"rote mcp add {args.name} <url>",
            file=sys.stderr,
        )
        return 2
    try:
        doc = asyncio.run(
            login(args.name, config, no_browser=args.no_browser, callback_port=args.callback_port)
        )
    except McpAuthError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    expires_at = doc.get("expires_at")
    refresh = " (refresh token stored)" if (doc.get("tokens") or {}).get("refresh_token") else ""
    print(f"rote mcp: authenticated {args.name!r}{refresh}")
    if expires_at:
        print(f"  access token expires at epoch {expires_at:.0f}; refresh is automatic")
    _release_parked_after_login(args.name)
    return 0


def _release_parked_after_login(server: str) -> None:
    """Wake emitted workflows parked waiting for this server's auth.

    Emitted apps park durably on missing/dead MCP credentials (see
    rote.mcp.release); a successful login is the event they're waiting
    for, so discovery + release happens here rather than as a separate
    command the user has to know about. (`rote mcp release <server>`
    exists for when the credential was fixed some other way — e.g.
    re-provisioned Worker secrets.)
    """
    from rote.mcp.release import ReleaseUnavailable, release_parked_workflows

    try:
        report = release_parked_workflows(server)
    except ReleaseUnavailable as e:
        print(f"  note: {e}", file=sys.stderr)
        return
    for wf in report.released:
        print(f"  released parked workflow {wf.workflow_id} ({wf.app})")
    for bc in report.broadcasts:
        print(f"  broadcast release event {bc.event!r} → {bc.endpoint}")
    for app in report.skipped:
        print(f"  note: skipped registered app {app.app}: {app.reason}", file=sys.stderr)


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
    client_id = client_info.get("client_id") or config.client_id
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
    client_secret = client_info.get("client_secret") or config.client_secret
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
    currently-valid credentials: static headers verbatim, or the
    stored access token refreshed through the OAuth provider when
    stale. stdout is the contract; everything else goes to stderr.
    """
    import asyncio
    import json as _json

    from rote.mcp import McpAuthError, load_registry
    from rote.mcp.auth import fresh_access_token

    registry = load_registry()
    config = registry.servers.get(args.name)
    if config is None:
        print(f"error: no MCP server named {args.name!r}", file=sys.stderr)
        return 2
    if config.headers:
        print(_json.dumps(config.headers))
        return 0
    try:
        token = asyncio.run(fresh_access_token(args.name, config))
    except McpAuthError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(_json.dumps({"Authorization": f"Bearer {token}"}))
    return 0


def _mcp_requirements(pipeline: Pipeline) -> list[dict[str, object]]:
    """Per required MCP server: the binding nodes, tools, and current auth state.

    Requirement detection is pure IR (``Pipeline.required_mcp_servers``);
    the ``auth`` column reads the local registry + token store — no
    network. A server the registry doesn't know reports ``"not
    registered"``; registry/token read failures degrade to ``"unknown"``
    rather than sinking the report (same posture as doctor). Nothing here
    gates anything — callers render recommendations, and park-on-auth
    remains the runtime backstop.
    """
    from rote.mcp import auth_status, load_registry, read_token_file

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
            auth = "unknown"
        elif server not in registry_servers:
            auth = "not registered"
        else:
            try:
                auth = auth_status(registry_servers[server], read_token_file(server))
            except Exception:
                auth = "unknown"
        report.append({"server": server, "nodes": node_ids, "tools": tools, "auth": auth})
    return report


#: Auth states that mean a run would park on its first call to the server.
#: "expired (refreshable)" recovers unattended and "static headers" never
#: touches OAuth, so neither warrants a recommendation.
_MCP_AUTH_NEEDS_ACTION = frozenset({"not registered", "not authenticated", "expired"})


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
                f"  {server}: not in the rote registry — register and authenticate "
                f"before running: rote mcp add {server} <url> && rote mcp login {server}"
            )
        else:
            lines.append(
                f"  {server}: {auth} — the run will park at its first {server} call; "
                f"recommended: rote mcp login {server}"
            )
    return lines


def _build_analysis(
    pipeline: Pipeline,
    driver_name: str,
    driver_metadata: dict[str, object],
    skill_path: Path,
) -> dict[str, object]:
    """Derive a structural report from a graduated pipeline's IR.

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
    """Dry-run the graduator: report a skill's graduated shape, emit nothing.

    The ``plan`` to ``graduate``'s ``apply`` — runs the same graduator
    agent to produce a validated IR, then prints a structural report
    instead of emitting runtime code. With ``--out`` the graduated IR +
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

    graduator = Graduator(agent=args.agent, model=args.model)

    out_ctx: Any = (
        nullcontext(str(args.out))
        if args.out
        else tempfile.TemporaryDirectory(prefix="rote-analyze-")
    )
    with out_ctx as out_raw:
        graduated_dir = Path(out_raw)
        try:
            result = asyncio.run(graduator.graduate(skill_path, graduated_dir))
        except GraduatorError as e:
            print(f"rote analyze: {e}", file=sys.stderr)
            return 1
        except KeyboardInterrupt:
            print("rote analyze: interrupted", file=sys.stderr)
            return 130

        report = _build_analysis(
            result.pipeline, result.driver_name, result.driver_metadata, skill_path
        )
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print(_render_analysis_text(report))
            if args.out:
                print()
                print(f"  graduated IR kept at: {graduated_dir / 'pipeline.yaml'}")
                print(
                    f"  emit a runtime with:  rote emit {graduated_dir / 'pipeline.yaml'} "
                    f"--runtime dbos --out ./runtime"
                )
            else:
                print()
                print("  (report only — pass --out DIR to keep the pipeline.yaml for `rote emit`)")
    return 0


#: Optional package extras → the module whose importability proves the
#: extra is installed. ``find_spec`` is a pure lookup (no import side
#: effects), so this stays cheap and safe to run on every ``doctor``.
_DOCTOR_EXTRAS: dict[str, str] = {
    "temporal": "temporalio",
    "api": "anthropic",
    "openai-api": "openai",
    "dbos": "dbos",
    "serve": "fastmcp",
    "mcp": "fastmcp",
}


def _doctor_mcp_servers() -> list[dict[str, object]]:
    """Registered MCP servers with their five-state auth status.

    Degrades to an empty list on any read failure — the registry and
    token store are documented as optionally-absent, so a missing or
    malformed store is informational, never a doctor traceback.
    """
    from rote.mcp import auth_status, load_registry, read_token_file

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
        servers.append({"name": name, "url": config.url, "auth": auth_status(config, doc)})
    return servers


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
        {"name": extra, "installed": importlib.util.find_spec(module) is not None}
        for extra, module in _DOCTOR_EXTRAS.items()
    ]
    return {
        "version": __version__,
        "python": platform.python_version(),
        "drivers": drivers,
        "runtimes": runtimes,
        "mcp_servers": _doctor_mcp_servers(),
        "apps": _doctor_apps(),
        "ok": any(bool(d["available"]) for d in drivers),
    }


def _render_doctor_text(report: dict[str, Any]) -> str:
    """Render the doctor report as a scannable ✓/✗ checklist."""
    lines: list[str] = []
    lines.append("rote doctor — preflight (read-only; costs nothing)")
    lines.append("")
    lines.append(f"  rote {report['version']}  ·  Python {report['python']}")
    lines.append("")

    lines.append("Graduator drivers (at least one required):")
    for driver in report["drivers"]:
        mark = "✓" if driver["available"] else "✗"
        suffix = "" if driver["available"] else f" — {driver['reason']}"
        lines.append(f"  {mark} {driver['name']}{suffix}")
    lines.append("  note: CLI subscription auth (claude/codex) is only verified at run time.")
    lines.append("")

    lines.append("Runtime dependencies:")
    for runtime in report["runtimes"]:
        if runtime["installed"]:
            lines.append(f"  ✓ {runtime['name']}")
        else:
            lines.append(f"  ✗ {runtime['name']} — pip install 'rote-cli[{runtime['name']}]'")
    lines.append("")

    lines.append("MCP servers:")
    servers = report["mcp_servers"]
    if not servers:
        lines.append("  (none registered)")
    else:
        width = max(len(str(s["name"])) for s in servers)
        for server in servers:
            healthy = server["auth"] in ("authenticated", "static headers")
            mark = "✓" if healthy else "✗"
            name = str(server["name"]).ljust(width)
            lines.append(f"  {mark} {name}  {str(server['auth']):<22}  {server['url']}")
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

    lines.append("✓ ready to graduate" if report["ok"] else "✗ no graduator driver available")
    return "\n".join(lines)


def _cmd_doctor(args: argparse.Namespace) -> int:
    """Read-only preflight: is a graduation set up to succeed before spending money?

    Reports the rote/Python versions, which graduator drivers are usable
    (the load-bearing gate), which optional runtime deps are installed,
    the auth state of every registered MCP server, and every emitted app
    rote knows about. Exits non-zero only when NO graduator driver is
    available, so it doubles as a scriptable gate; MCP and app findings
    are informational.
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
        # source_skill was recorded relative to wherever the graduation
        # ran (often a temp work dir), so no single base is reliable —
        # try the cwd, then every ancestor of the pipeline.yaml.
        pipeline_dir = pipeline_yaml.resolve().parent
        bases = [Path.cwd(), pipeline_dir, *pipeline_dir.parents]
        for base in bases[:10]:
            candidate = (base / source_skill).resolve()
            if (candidate / "SKILL.md").is_file():
                return candidate
    return None


def _cmd_eval(args: argparse.Namespace) -> int:
    """Static before/after scorecard: speed, cost, determinism.

    'Before' models the source skill running as raw agent instructions;
    'after' is derived from the pipeline IR. Prices are fetched live —
    a network failure is a loud error, never a stale builtin table.
    """
    import json

    from rote.eval.pricing import PricingError

    target = Path(args.target)
    pipeline_yaml = _resolve_pipeline_yaml(target)
    if pipeline_yaml is None:
        print(
            f"error: no pipeline.yaml found at or under {target} — "
            f"pass a graduated dir, a graduate --out dir, or the file itself",
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
            pipeline, pipeline_yaml, skill_dir, args.provider, priors=priors
        )
    except PricingError as e:
        print(f"error: could not fetch live prices: {e}", file=sys.stderr)
        return 1

    measured_md = ""
    measured_json: dict[str, object] | None = None
    if args.run:
        try:
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

    Explicit ``--runtime-dir`` wins; otherwise try the ``rote graduate
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
    )
    from rote.eval.pricing import fetch_catalog

    if not args.input:
        raise ValueError("--run requires --input <task.json> (the pipeline input payload)")
    if args.trials < 1:
        raise ValueError(f"--trials must be >= 1, got {args.trials}")
    task = json.loads(Path(args.input).read_text(encoding="utf-8"))
    # Envelope form: {"input": {...}, "signals": {...}}. Disambiguate
    # against a pipeline whose *payload* legitimately has a top-level
    # "input" field: an explicit "signals" key always means envelope;
    # otherwise "input" only reads as envelope when the pipeline's
    # declared input contract has no field of that name.
    declared = set(pipeline.input.required) | set(pipeline.input.optional)
    if pipeline.input.input_schema is not None:
        props = pipeline.input.input_schema.get("properties")
        if isinstance(props, dict):
            declared |= set(props.keys())
    is_envelope = (
        isinstance(task, dict)
        and "input" in task
        and set(task) <= {"input", "signals"}
        and ("signals" in task or "input" not in declared)
    )
    if is_envelope:
        input_payload = task["input"]
        signals = task.get("signals") or {}
    else:
        input_payload = task
        signals = {}

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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rote",
        description=(
            "Graduate fuzzy AI skills into deterministic, reliable workflows. "
            "The north-star command is `rote graduate`; `rote emit` exposes "
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
            "IR → adapter step — no graduator agent is invoked."
        ),
    )
    emit.add_argument(
        "pipeline_yaml",
        help="Path to a pipeline.yaml (IR file)",
    )
    emit.add_argument(
        "--runtime",
        default="dbos",
        choices=available_runtimes,
        help=(
            f"Target workflow runtime (default: dbos — durable execution as a "
            f"library, no orchestrator to run). Available: {', '.join(available_runtimes)}"
        ),
    )
    emit.add_argument(
        "--backend",
        default="mcp",
        choices=["mcp", "api"],
        help=(
            "For external_call nodes with an mcp binding (dbos runtime): "
            "'mcp' (default) emits a working Streamable-HTTP call to the MCP "
            "tool the skill used; 'api' emits the direct vendor-SDK path "
            "(fill in the extracted/ stub; one key in .env). No effect on "
            "nodes without an mcp binding or on runtimes without mcp support."
        ),
    )
    emit.add_argument(
        "--out",
        required=True,
        help="Output directory (created if missing)",
    )
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

    # rote graduate (stub)
    graduate = subparsers.add_parser(
        "graduate",
        help="Graduate a skill into a runnable pipeline (north-star command)",
        description=(
            "Read a skill bundle, run the rote-graduate agent to produce "
            "a pipeline IR + extracted modules + signature stubs, then "
            "emit runtime code for the chosen adapter. Output layout: "
            "<out>/graduated/ (the IR + stubs) and <out>/runtime/<target>/ "
            "(the emitted workflow code)."
        ),
    )
    graduate.add_argument("skill_path", help="Path to the source skill directory")
    graduate.add_argument(
        "--runtime",
        default="dbos",
        choices=available_runtimes,
        help=(
            f"Target workflow runtime (default: dbos — durable execution as a "
            f"library, no orchestrator to run). Available: {', '.join(available_runtimes)}"
        ),
    )
    graduate.add_argument(
        "--backend",
        default="mcp",
        choices=["mcp", "api"],
        help=(
            "For external_call nodes with an mcp binding (dbos runtime): "
            "'mcp' (default) emits a working call to the MCP tool the skill "
            "used; 'api' emits the direct vendor-SDK path. See `rote emit`."
        ),
    )
    graduate.add_argument(
        "--out",
        required=True,
        help="Output directory for the graduated artifacts",
    )
    graduate.add_argument(
        "--agent",
        choices=["claude", "codex", "api"],
        default=None,
        help=(
            "Agent runtime to use for the graduator. Defaults to auto-detect "
            "(claude CLI first, then codex, then anthropic API). Users with "
            "a Claude Max/Pro or ChatGPT subscription should prefer claude/codex "
            "to avoid per-token API charges."
        ),
    )
    graduate.add_argument(
        "--model",
        default=None,
        help=(
            "Override the LLM model used by the graduator agent "
            "(e.g. 'claude-opus-4-6' for higher-quality / higher-cost "
            "runs on complex skills). Defaults to the driver's default, "
            "which is Sonnet 4.6 for the subscription path."
        ),
    )
    graduate.add_argument(
        "--no-eval",
        action="store_true",
        help=(
            "Skip the before/after eval scorecard (scorecard.md). The "
            "scorecard needs one network fetch for live model prices; "
            "everything else about it is static."
        ),
    )
    graduate.add_argument(
        "--update",
        action="store_true",
        help=(
            "Incremental re-graduation: requires a previous graduation in "
            "--out. Diffs the skill against the previous run's provenance "
            "and re-derives only nodes whose source sections changed — "
            "unchanged nodes are preserved verbatim (ids and all), and a "
            "skill with no changes skips the agent entirely."
        ),
    )
    graduate.add_argument(
        "--json",
        action="store_true",
        help=(
            "Emit a single JSON object (graduated_dir, runtime_dir, written "
            "files, scorecard path, unimplemented stubs) to stdout; keep the "
            "progress log and summary on stderr. For piping into another tool "
            "or agent."
        ),
    )
    graduate.add_argument(
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
    graduate.set_defaults(func=_cmd_graduate)

    # rote register
    register = subparsers.add_parser(
        "register",
        help="Register a graduated pipeline as an MCP tool for `rote serve`",
        description=(
            "Append or update an entry in the serve registry "
            "(~/.rote/registry.json by default) from a graduate run's "
            "pipeline.yaml. Tool name comes from pipeline.name, description "
            "from pipeline.description, inputSchema from the pipeline input "
            "contract. A running `rote serve` picks the change up live."
        ),
    )
    register.add_argument(
        "graduated_dir",
        help=(
            "A graduate --out directory, a directory containing pipeline.yaml, "
            "or the pipeline.yaml itself"
        ),
    )
    register.add_argument(
        "--registry",
        default=None,
        help="Registry file (default: ~/.rote/registry.json)",
    )
    register.add_argument(
        "--runtime",
        default="dbos",
        choices=["dbos", "temporal", "cloudflare"],
        help=(
            "Runtime the graduated pipeline is deployed on (default: dbos, "
            "matching `rote graduate`)"
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
        help="Temporal task queue the graduated worker polls (default: pipeline.name)",
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
        help="Registry file (default: ~/.rote/registry.json)",
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
            "`mcp.server` names in graduated pipelines; `login` runs the full "
            "OAuth 2.1 dance (discovery, PKCE, dynamic registration) and "
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
        help="Pre-registered OAuth client id (for servers without dynamic registration)",
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
    mcp_add.set_defaults(func=_cmd_mcp_add)

    mcp_list = mcp_sub.add_parser("list", help="List registered servers and auth status")
    mcp_list.add_argument("--json", action="store_true")
    mcp_list.set_defaults(func=_cmd_mcp_list)

    mcp_remove = mcp_sub.add_parser("remove", help="Remove a server (and its tokens)")
    mcp_remove.add_argument("name")
    mcp_remove.set_defaults(func=_cmd_mcp_remove)

    mcp_login = mcp_sub.add_parser(
        "login", help="Authenticate a server via OAuth (opens a browser)"
    )
    mcp_login.add_argument("name")
    mcp_login.add_argument(
        "--no-browser",
        action="store_true",
        help="Print the authorization URL instead of opening a browser (SSH boxes)",
    )
    mcp_login.add_argument(
        "--callback-port",
        type=int,
        default=None,
        help="Fixed localhost callback port (for servers with pinned redirect URIs)",
    )
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
        help="Dry-run the graduator: report a skill's graduated shape, emit no runtime code",
        description=(
            "Run the rote-graduate agent against a skill and print what "
            "graduation would produce — node-kind breakdown, roteness "
            "(deterministic vs. LLM-sampled steps), mandatory checks, HITL "
            "gates, agent loops, and which runtimes can target it — without "
            "emitting any runtime code. This is the `plan` to `graduate`'s "
            "`apply`. Pass --out to keep the pipeline.yaml for a later `rote emit`."
        ),
    )
    analyze.add_argument("skill_path", help="Path to the source skill directory")
    analyze.add_argument(
        "--agent",
        choices=["claude", "codex", "api"],
        default=None,
        help=(
            "Agent runtime for the graduator (default: auto-detect — claude "
            "CLI first, then codex, then the anthropic API)."
        ),
    )
    analyze.add_argument(
        "--model",
        default=None,
        help="Override the LLM model the graduator uses (default: the driver's default).",
    )
    analyze.add_argument(
        "--out",
        default=None,
        help=(
            "Keep the graduated IR + stubs in this directory (ready for "
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

    # rote doctor
    doctor = subparsers.add_parser(
        "doctor",
        help="Preflight check: graduator drivers, runtime deps, MCP auth, and registered apps",
        description=(
            "Read-only preflight that tells you (or your agent) whether a "
            "graduation will succeed before spending a cent: which graduator "
            "drivers are usable, which optional runtime dependencies are "
            "installed, the auth state of every registered MCP server, and "
            "every emitted app rote knows about. Nothing is executed and "
            "nothing is spent. Exits non-zero only when no graduator driver "
            "is available, so it works as a scriptable gate."
        ),
    )
    doctor.add_argument(
        "--json",
        action="store_true",
        help="Emit the report as a single JSON object to stdout instead of the checklist.",
    )
    doctor.set_defaults(func=_cmd_doctor)

    # rote eval
    eval_cmd = subparsers.add_parser(
        "eval",
        help="Before/after scorecard: speed, cost, determinism (static, no execution)",
        description=(
            "Estimate what a skill costs to run as raw agent instructions "
            "versus as its graduated pipeline — wall clock, dollars across "
            "a sampling of current models at live official prices, and "
            "determinism (LLM sampling surface). Static: nothing executes."
        ),
    )
    eval_cmd.add_argument(
        "target",
        help=("A graduated directory, a graduate --out directory, or a pipeline.yaml path"),
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
            "graduate --out layout's runtime/dbos or runtime/python sibling)"
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
    eval_cmd.set_defaults(func=_cmd_eval)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "command", None):
        # No subcommand — print a helpful banner and usage.
        print(f"rote {__version__} — graduate fuzzy AI skills into deterministic workflows")
        print()
        parser.print_help()
        return 0

    func = args.func
    return int(func(args))


if __name__ == "__main__":
    raise SystemExit(main())
