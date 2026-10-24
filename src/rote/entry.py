"""The ``rote`` command: ``rote skill run`` straight to its launcher, the rest to :mod:`rote.cli`.

``rote skill run <name> ...`` is what a compiled skill's agent types on every
step, so its start-up is paid on every block. :mod:`rote.cli` builds the whole
parser and imports the compiler, the adapters and the IR to do it (measured
170 ms of a 2.5 s ``start``), and the launcher needs none of that. Every other
command, and every ``skill run`` form this does not take whole (``-h``, a
missing name), goes to :func:`rote.cli.main`, which parses the same words the
same way (``name`` then ``argparse.REMAINDER``).
"""

from __future__ import annotations

import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(argv) if argv is not None else sys.argv[1:]
    if raw[:2] == ["skill", "run"] and len(raw) > 2 and not raw[2].startswith("-"):
        from rote.skill_target.launch import run

        return run(raw[2], raw[3:])
    from rote.cli import main as cli_main

    return cli_main(raw)


if __name__ == "__main__":
    sys.exit(main())
