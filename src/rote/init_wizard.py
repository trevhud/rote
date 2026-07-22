"""The interactive ``rote init`` wizard.

Walks the choices that shape every later command — where graduated
pipelines run (rote cloud vs a local runtime), which graduator driver
does the work, and optionally which model — then writes them to a
config file (:mod:`rote.config`). It is the only interactive entry
point in the CLI besides login itself; everything else must stay
prompt-free for CI, which is why this lives behind an explicit command
rather than a first-run hook.

All I/O is injected (``input_fn`` / ``echo``) so the flow is testable
without a TTY.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

from rote.config import user_config_path, write_config

#: Runtime menu: (name, one-line pitch). Order = recommendation order,
#: matching the graduate default (dbos) first for the local path.
RUNTIME_MENU: tuple[tuple[str, str], ...] = (
    ("dbos", "durable Python, no orchestrator process to run (local default)"),
    ("temporal", "workflow.py + activities.py for a Temporal cluster"),
    ("cloudflare", "TypeScript Workflows on Cloudflare Workers"),
    ("dbos-ts", "durable TypeScript for DBOS Transact (Postgres)"),
    ("inngest", "TypeScript function app that mounts into Node/Next.js"),
    ("python", "plain script, maximum legibility (no HITL gates)"),
)


class WizardAborted(RuntimeError):
    """The user backed out (Ctrl-C or declined an overwrite)."""


def _ask(
    prompt: str,
    input_fn: Callable[[str], str],
    *,
    default: str,
    valid: set[str],
) -> str:
    while True:
        answer = input_fn(prompt).strip().lower() or default
        if answer in valid:
            return answer


def run_wizard(
    *,
    project: bool,
    input_fn: Callable[[str], str] | None = None,
    echo: Callable[[str], None] | None = None,
    cwd: Path | None = None,
) -> Path:
    """Run the wizard and return the path of the written config file."""
    # Resolve at call time, not def time — a def-time `= input` default
    # would freeze the original builtin and dodge test monkeypatching.
    if input_fn is None:
        input_fn = input
    if echo is None:
        echo = lambda line: print(line, file=sys.stderr)  # noqa: E731
    from rote.cloud_auth import LoginError, load_credential, login
    from rote.graduator.drivers import available_drivers

    echo("rote init — set your defaults once; every command reads them.")
    echo("")

    # ── Compact preflight: just the parts onboarding can act on ──
    # (drivers + runtime deps; the full MCP/app inventory is `rote doctor`)
    from rote.cli import _build_doctor_report

    report = _build_doctor_report()
    echo("Graduator drivers (at least one required — see `rote doctor` for the full preflight):")
    for driver in report["drivers"]:
        mark = "✓" if driver["available"] else "✗"
        suffix = "" if driver["available"] else f" — {driver['reason']}"
        echo(f"  {mark} {driver['name']}{suffix}")
    missing = [r["name"] for r in report["runtimes"] if not r["installed"]]
    if missing:
        echo(f"Runtime extras not installed (pip install 'rote-cli[<name>]'): {', '.join(missing)}")
    echo("")

    values: dict[str, str] = {}

    # ── 1. Hosting: rote cloud, or a local runtime ──
    echo("Where should graduated pipelines run?")
    echo("  1. rote cloud (recommended) — hosted; graduate + deploy in one step")
    echo("  2. locally — pick the workflow runtime rote emits")
    hosting = _ask("choice [1]: ", input_fn, default="1", valid={"1", "2"})
    if hosting == "1":
        values["runtime"] = "cloudflare"
        values["deploy"] = "rote-cloud"
        cred = load_credential()
        if cred is not None:
            echo(f"  already logged in as {cred.user or cred.url}")
        else:
            do_login = _ask(
                "  not logged in — log in to rote cloud now? [Y/n]: ",
                input_fn,
                default="y",
                valid={"y", "n"},
            )
            if do_login == "y":
                try:
                    login()
                except LoginError as e:
                    echo(f"  login failed: {e}")
                    echo("  (kept the cloud default — run `rote login` before graduating)")
            else:
                echo("  ok — `rote graduate` will ask you to `rote login` first")
    else:
        echo("")
        echo("Which runtime should rote emit by default?")
        for index, (name, pitch) in enumerate(RUNTIME_MENU, start=1):
            echo(f"  {index}. {name:<10} — {pitch}")
        pick = _ask(
            "choice [1]: ",
            input_fn,
            default="1",
            valid={str(i) for i in range(1, len(RUNTIME_MENU) + 1)},
        )
        values["runtime"] = RUNTIME_MENU[int(pick) - 1][0]
        values["deploy"] = "none"

    # ── 2. Graduator driver ──
    echo("")
    echo("Which agent should run graduations? (auto-detect probes in this order)")
    drivers = available_drivers()
    for index, (name, available, reason) in enumerate(drivers, start=1):
        mark = "✓" if available else "✗"
        suffix = "" if available else f" — {reason}"
        echo(f"  {index}. {mark} {name}{suffix}")
    echo(f"  {len(drivers) + 1}. auto — first available at run time (recommended)")
    pick = _ask(
        f"choice [{len(drivers) + 1}]: ",
        input_fn,
        default=str(len(drivers) + 1),
        valid={str(i) for i in range(1, len(drivers) + 2)},
    )
    if pick != str(len(drivers) + 1):
        values["agent"] = drivers[int(pick) - 1][0]

    # ── 3. Model override ──
    model = input_fn("graduator model [driver default — Sonnet]: ").strip()
    if model:
        values["model"] = model

    # ── Write ──
    target = (cwd or Path.cwd()) / "rote.yaml" if project else user_config_path()
    if target.is_file():
        overwrite = _ask(
            f"{target} exists — overwrite? [y/N]: ", input_fn, default="n", valid={"y", "n"}
        )
        if overwrite == "n":
            raise WizardAborted(f"kept existing {target}")
    write_config(target, values)

    echo("")
    echo(f"✓ wrote {target}")
    for name, value in values.items():
        echo(f"    {name}: {value}")
    echo("  see the effective setup any time with: rote config")
    echo("  next: rote graduate <skill-dir> --out ./graduated")
    return target
