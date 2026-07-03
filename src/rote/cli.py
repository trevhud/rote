"""rote CLI — entry point.

The north star for this CLI:

    rote graduate ./path/to/skill --runtime temporal --out ./graduated/

One command, skill in, runnable workflow out. Every other command is a
building block for that flow:

    rote emit    <pipeline.yaml> --runtime temporal  # IR → code only
    rote analyze <skill-path>                        # graduator dry run
    rote eval    <skill-path>                        # regression against expected/

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

from rote import __version__
from rote.adapters import ADAPTERS, get_adapter
from rote.graduator import Graduator, GraduatorError
from rote.ir import load_pipeline

# ───────── Subcommand: emit ─────────


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
        adapter = get_adapter(args.runtime)
    except KeyError as e:
        print(f"error: {e.args[0]}", file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    written = adapter.emit(pipeline, out_dir)

    print(f"rote: emitted {pipeline.name} v{pipeline.version} → {out_dir}")
    for label, path in written.items():
        print(f"  {label}: {path}")
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
        result = asyncio.run(graduator.graduate(skill_path, graduated_dir))
    except GraduatorError as e:
        print(f"rote graduate: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("rote graduate: interrupted", file=sys.stderr)
        return 130

    try:
        adapter = get_adapter(args.runtime)
    except KeyError as e:
        print(f"error: {e.args[0]}", file=sys.stderr)
        return 2

    written = adapter.emit(result.pipeline, runtime_dir)

    # ── Summary ──
    print(f"rote graduate: ✓ {result.pipeline.name} v{result.pipeline.version}")
    print(f"  driver: {result.driver_name}")
    if result.driver_metadata:
        meta_str = ", ".join(f"{k}={v}" for k, v in result.driver_metadata.items())
        print(f"  metadata: {meta_str}")
    print(f"  graduated artifacts: {graduated_dir}")
    print(f"  emitted runtime ({args.runtime}): {runtime_dir}")
    for label, path in written.items():
        print(f"    {label}: {path}")
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    print("rote analyze: not yet implemented.", file=sys.stderr)
    return 70


def _cmd_eval(args: argparse.Namespace) -> int:
    print("rote eval: not yet implemented.", file=sys.stderr)
    return 70


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
        default="temporal",
        choices=available_runtimes,
        help="Target workflow runtime (default: temporal). "
        f"Available: {', '.join(available_runtimes)}",
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
        default="temporal",
        choices=available_runtimes,
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
    graduate.set_defaults(func=_cmd_graduate)

    # rote analyze (stub)
    analyze = subparsers.add_parser(
        "analyze",
        help="Run the graduator against a skill and print a report only",
    )
    analyze.add_argument("skill_path")
    analyze.set_defaults(func=_cmd_analyze)

    # rote eval (stub)
    eval_cmd = subparsers.add_parser(
        "eval",
        help="Compare graduator output against an expected/ baseline",
    )
    eval_cmd.add_argument("skill_path")
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
