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
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rote import __version__
from rote.adapters import ADAPTERS, get_adapter
from rote.graduator import Graduator, GraduatorError
from rote.ir import Pipeline, load_pipeline

if TYPE_CHECKING:
    from rote.eval.priors import Priors
    from rote.eval.scorecard import Scorecard

# ───────── Subcommand: emit ─────────


def _print_written(written: dict[str, Path], indent: str = "  ") -> None:
    """Print an adapter's written-files mapping, flagging preserved files.

    When the emit writer finds a file the user edited since the last
    emit, it leaves the file alone and parks the fresh content in a
    ``<name>.new`` sibling — surface those so the preservation is a
    visible event, not a silent one.
    """
    for label, path in written.items():
        print(f"{indent}{label}: {path}")
    preserved = [path for path in written.values() if path.name.endswith(".new")]
    if preserved:
        print()
        print(
            f"{indent}note: {len(preserved)} file(s) were edited since the last emit "
            f"and were left untouched."
        )
        print(f"{indent}Fresh output landed alongside as '.new' files — merge or delete them:")
        for path in preserved:
            print(f"{indent}  {path}")


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

    print(f"rote: emitted {pipeline.name} v{pipeline.version} → {out_dir}")
    _print_written(written)
    return 0


# ───────── Subcommand stubs (graduate / analyze / eval) ─────────


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

    graduator = Graduator(agent=args.agent, model=args.model)

    try:
        result = asyncio.run(graduator.graduate(skill_path, graduated_dir, update=args.update))
    except GraduatorError as e:
        print(f"rote graduate: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("rote graduate: interrupted", file=sys.stderr)
        return 130

    try:
        adapter = get_adapter(args.runtime, external_backend=args.backend)
    except KeyError as e:
        print(f"error: {e.args[0]}", file=sys.stderr)
        return 2

    written = adapter.emit(result.pipeline, runtime_dir)

    # ── Scorecard (auxiliary: a price-fetch outage must not sink a
    # just-completed, real-money graduation run) ──
    scorecard_path: Path | None = None
    if not args.no_eval:
        from rote.eval.pricing import PricingError

        try:
            scorecard = _build_scorecard_for(
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

    # ── Summary ──
    print(f"rote graduate: ✓ {result.pipeline.name} v{result.pipeline.version}")
    print(f"  driver: {result.driver_name}")
    if result.driver_metadata:
        meta_str = ", ".join(f"{k}={v}" for k, v in result.driver_metadata.items())
        print(f"  metadata: {meta_str}")
    print(f"  graduated artifacts: {graduated_dir}")
    if scorecard_path is not None:
        print(f"  eval scorecard: {scorecard_path}")
    print(f"  emitted runtime ({args.runtime}): {runtime_dir}")
    _print_written(written, indent="    ")
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


def _build_analysis(
    pipeline: Pipeline,
    driver_name: str,
    driver_metadata: dict[str, object],
    skill_path: Path,
) -> dict[str, object]:
    """Derive a structural report from a graduated pipeline's IR.

    Pure function of the IR — no model, no network. Step counting mirrors
    the eval scorecard exactly (top-level execution-wave nodes; loop_body
    sub-nodes are costed inside their parent loop, so they're excluded
    from the top-level count) so ``analyze``'s roteness equals ``eval``'s.
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


def _build_scorecard_for(
    pipeline: Pipeline,
    pipeline_yaml: Path,
    skill_dir: Path | None,
    provider: str,
    priors: Priors | None = None,
) -> Scorecard:
    """Shared eval flow: estimate both sides, fetch live prices, assemble.

    Raises ``rote.eval.pricing.PricingError`` when the live price source
    is unreachable — callers decide whether that's fatal (``rote eval``)
    or a warning (``rote graduate``'s auxiliary scorecard).
    """
    from datetime import UTC, datetime

    from rote.eval import (
        Priors,
        estimate_pipeline,
        estimate_skill,
        external_call_payload_tokens,
        load_eval_estimates,
    )
    from rote.eval.pricing import fetch_catalog
    from rote.eval.scorecard import build_scorecard
    from rote.eval.sidecar import EVAL_SIDECAR_FILENAME
    from rote.eval.tokens import pick_token_counter

    priors = priors or Priors()
    prices = fetch_catalog(provider=provider).sample(provider=provider)
    # Exact counting needs a live model id (tokenizers are per-model);
    # the small tier's is as good as any and free either way. Only the
    # Anthropic endpoint exists, so other providers get the heuristic.
    count_model = prices[-1].model_id if provider == "anthropic" else None
    counter = pick_token_counter(priors, model=count_model)
    pipeline_estimate = estimate_pipeline(pipeline, counter, priors)

    skill_estimate = None
    if skill_dir is not None:
        sidecar_path = pipeline_yaml.parent / EVAL_SIDECAR_FILENAME
        sidecar = load_eval_estimates(sidecar_path) if sidecar_path.is_file() else None
        # The agent pulls the same sources the pipeline's external_call nodes
        # bind to; use that footprint to size the before-side context payload.
        data_payload = external_call_payload_tokens(pipeline, priors)
        skill_estimate = estimate_skill(
            skill_dir, counter, priors, sidecar=sidecar, data_payload_tokens=data_payload
        )

    return build_scorecard(
        pipeline_name=pipeline.name,
        pipeline_estimate=pipeline_estimate,
        skill_estimate=skill_estimate,
        prices=prices,
        priors=priors,
        generated_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


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

    from rote.eval.priors import priors_from_overrides

    try:
        priors = priors_from_overrides(args.prior or [])
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    try:
        scorecard = _build_scorecard_for(
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
        measured_to_dict,
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
