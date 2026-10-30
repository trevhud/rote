"""A stdio MCP server whose one tool runs ``rote skill run <name> ...`` (#334).

On the Claude lane a compiled skill's agent may run exactly one command,
``Bash(rote skill run <name> *)``. Codex cannot express that next to scoped
reads: on codex-cli 0.165.0 a custom permissions profile (what scopes reads to
the skill and the workspace) disables every sandbox escalation, so an
``allow`` prefix rule no longer lets a command out, and without the profile
the whole disk is readable. An MCP server runs outside Codex's sandbox and is
gated by ``enabled_tools``, so the driver goes through this server instead:
its one tool takes the arguments after the skill name as a list (no shell
quoting) and runs the driver with this skill's name fixed. Nothing else can
leave the sandbox.

Started by Codex as ``python -I -m rote.eval.skill_run_server <name> <rote>``,
where ``<rote>`` is the ``rote`` executable to run (the trial's shim).
"""

from __future__ import annotations

import subprocess
import sys
from typing import Any

#: The server name the agent sees, and its one tool.
SERVER = "rote_skill"
TOOL = "run"

#: How long one driver invocation may take: a block can run several tool calls.
TIMEOUT_SECONDS = 600.0


def run(name: str, rote: str, args: list[str]) -> dict[str, Any]:
    """``rote skill run <name> *args``: its exit code, stdout and stderr."""
    if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
        return {"exit_code": 2, "stdout": "", "stderr": "args must be a list of strings"}
    try:
        proc = subprocess.run(
            [rote, "skill", "run", name, *args],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return {"exit_code": None, "stdout": "", "stderr": f"timed out after {TIMEOUT_SECONDS:g}s"}
    return {"exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}


def main(argv: list[str]) -> None:
    from fastmcp import FastMCP

    name, rote = argv
    server = FastMCP(SERVER)

    @server.tool(
        name=TOOL, description=f"Run `rote skill run {name} <args...>` and return its output."
    )
    def tool(args: list[str]) -> dict[str, Any]:
        """``args``: everything after ``rote skill run {name}``, one list item per argument."""
        return run(name, rote, args)

    server.run(transport="stdio", show_banner=False)


if __name__ == "__main__":
    main(sys.argv[1:])
