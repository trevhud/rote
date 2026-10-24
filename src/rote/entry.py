"""The ``rote`` command: ``rote skill run`` straight to its launcher, the rest to :mod:`rote.cli`.

``rote skill run <name> ...`` is what a compiled skill's agent types on every
step, so its start-up is paid on every block. :mod:`rote.cli` builds the whole
parser and imports the compiler, the adapters and the IR to do it (measured
170 ms of a 2.5 s ``start``), and the launcher needs none of that. Every other
command, and every ``skill run`` form this does not take whole (``-h``, a
missing name, ``--`` before the name), goes to :func:`rote.cli.main`.
:func:`skill_run_args` must read the words exactly as argparse's ``name`` then
``argparse.REMAINDER`` does; ``tests/test_skill_commands.py`` holds the two
to the same answer over every form.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence


def skill_run_args(raw: Sequence[str]) -> tuple[str, list[str]] | None:
    """``(name, driver args)`` for a ``skill run`` this takes whole, else None."""
    if list(raw[:2]) != ["skill", "run"] or len(raw) < 3 or raw[2].startswith("-"):
        return None
    rest = list(raw[3:])
    # argparse drops one `--` that leads the remainder, and only that one.
    if rest[:1] == ["--"]:
        rest = rest[1:]
    return raw[2], rest


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(argv) if argv is not None else sys.argv[1:]
    fast = skill_run_args(raw)
    if fast is not None:
        from rote.skill_target.launch import run

        return run(*fast)
    from rote.cli import main as cli_main

    return cli_main(raw)


if __name__ == "__main__":
    sys.exit(main())
