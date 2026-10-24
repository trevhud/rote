"""The compiled-skill target (``--runtime skill``): what a compiled skill's scripts are.

A compiled skill runs its plan with a stdlib driver, :mod:`rote.skill_target._driver`.
The driver and the helpers it loads by name are emitted verbatim into the
skill's ``scripts/`` dir, next to ``plan.json`` and ``extracted/``.
:data:`HELPER_FILES` is the one list of them, and :func:`script_files` the
one rule for which a given plan ships, for the emitter
(:mod:`rote.adapters.skill`) and the tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rote.plan import plan_text

_HERE = Path(__file__).resolve().parent

DRIVER_PATH = _HERE / "_driver.py"

#: ``{name in scripts/: the module whose source it is}``. The driver imports
#: ``_rote_schema`` always, ``_rote_mcp`` only when a step calls MCP, and
#: ``_rote_inference`` only for a plan whose judges the helper answers, by
#: these names, from the scripts dir it puts first on ``sys.path``.
HELPER_FILES: dict[str, Path] = {
    "_rote_skill.py": DRIVER_PATH,
    "_rote_schema.py": _HERE.parent / "plan" / "_schema.py",
    "_rote_mcp.py": _HERE.parent / "mcp" / "_runtime_helper.py",
    "_rote_inference.py": _HERE.parent / "inference" / "_runtime_helper.py",
}


def calls_mcp(plan: dict[str, Any]) -> bool:
    """Whether running ``plan`` can reach an MCP server.

    An ``mcp`` step does, and so does an agent loop that declares tools
    (they are MCP tool names) or has an ``mcp`` body step. A plan that
    cannot ships no MCP helper and names no MCP dependency (invariant #3).
    """
    for step in plan.get("steps") or []:
        if step.get("kind") == "mcp":
            return True
        if step.get("kind") == "loop" and (
            step.get("tools") or any(b.get("kind") == "mcp" for b in step.get("body") or [])
        ):
            return True
    return False


def helper_judges(plan: dict[str, Any]) -> bool:
    """Whether the inference helper, not the agent, answers ``plan``'s judges,
    an agent loop's judge sub-nodes included (the loop calls them as tools)."""
    if plan.get("judges") != "helper":
        return False
    for step in plan.get("steps") or []:
        if step.get("kind") == "judge":
            return True
        if step.get("kind") == "loop" and any(b.get("kind") == "judge" for b in step["body"]):
            return True
    return False


def script_files(plan: dict[str, Any]) -> dict[str, str]:
    """``{name in scripts/: text}``: ``plan.json`` and the verbatim helpers ``plan`` needs."""
    needed = {
        "_rote_skill.py": True,
        "_rote_schema.py": True,
        "_rote_mcp.py": calls_mcp(plan),
        "_rote_inference.py": helper_judges(plan),
    }
    files = {
        name: source.read_text(encoding="utf-8")
        for name, source in HELPER_FILES.items()
        if needed[name]
    }
    files["plan.json"] = plan_text(plan)
    return files


def write_scripts(scripts: Path, plan: dict[str, Any]) -> Path:
    """Write ``plan.json`` and the verbatim helpers into ``scripts``; return the driver.

    Not the whole emitter (that writes ``extracted/``, SKILL.md and the
    manifest too, through rote's hash-guarded writer): the part every
    compiled skill shares, for the driver tests.
    """
    scripts.mkdir(parents=True, exist_ok=True)
    for name, text in script_files(plan).items():
        (scripts / name).write_text(text, encoding="utf-8")
    return scripts / "_rote_skill.py"
