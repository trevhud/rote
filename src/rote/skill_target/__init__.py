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

_HERE = Path(__file__).resolve().parent

DRIVER_PATH = _HERE / "_driver.py"

#: ``{name in scripts/: the module whose source it is}``. The driver imports
#: ``_rote_inference`` always (its schema validator checks every answer, and
#: it answers the judges of a ``--judges helper`` plan), ``_rote_gate`` when a gate
#: has a message to show (the workflow runtimes' renderer, so a message reads
#: the same everywhere), and ``_rote_mcp`` only when a step calls MCP, by these
#: names, from the scripts dir it puts first on ``sys.path``.
HELPER_FILES: dict[str, Path] = {
    "_rote_skill.py": DRIVER_PATH,
    "_rote_mcp.py": _HERE.parent / "mcp" / "_runtime_helper.py",
    "_rote_inference.py": _HERE.parent / "inference" / "_runtime_helper.py",
    "_rote_gate.py": _HERE.parent / "adapters" / "_gate_runtime_helper.py",
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


def shows_gate_message(plan: dict[str, Any]) -> bool:
    """Whether a gate in ``plan`` has a message to render (``_rote_gate``)."""
    return any(
        step.get("message") for step in plan.get("steps") or [] if step.get("kind") == "gate"
    )


def helper_judges(plan: dict[str, Any]) -> bool:
    """Whether the inference helper, not the agent, answers ``plan``'s judges,
    an agent loop's judge sub-nodes included (the loop calls them as tools)."""
    return bool(helper_judge_ids(plan))


def helper_judge_ids(plan: dict[str, Any]) -> list[str]:
    """The ids of the judges the inference helper answers in ``plan``: its judge
    steps and each agent loop's judge sub-nodes, in plan order. Empty unless
    the plan was emitted with ``--judges helper``."""
    if plan.get("judges") != "helper":
        return []
    found: list[str] = []
    for step in plan.get("steps") or []:
        if step.get("kind") == "judge":
            found.append(str(step["id"]))
        if step.get("kind") == "loop":
            found += [str(b["id"]) for b in step["body"] if b.get("kind") == "judge"]
    return found


def script_files(plan: dict[str, Any]) -> dict[str, str]:
    """``{name in scripts/: text}``: ``plan.json`` and the verbatim helpers ``plan`` needs."""
    # Imported here, not at the top: rote.plan pulls in the IR (pydantic), and
    # `rote skill run` imports this package on every block without needing it.
    from rote.plan import plan_text

    needed = {
        "_rote_skill.py": True,
        "_rote_mcp.py": calls_mcp(plan),
        "_rote_inference.py": True,
        "_rote_gate.py": shows_gate_message(plan),
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
