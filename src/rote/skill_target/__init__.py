"""The compiled-skill target (``--runtime skill``): what a compiled skill's scripts are.

A compiled skill runs its plan with a stdlib driver, :mod:`rote.skill_target._driver`.
The driver and the two helpers it loads by name are emitted verbatim into
the skill's ``scripts/`` dir, next to ``plan.json`` and ``extracted/``.
:data:`HELPER_FILES` is the one list of them, for the emitter and the tests.
"""

from __future__ import annotations

from pathlib import Path

from rote.plan import plan_text

_HERE = Path(__file__).resolve().parent

DRIVER_PATH = _HERE / "_driver.py"

#: ``{name in scripts/: the module whose source it is}``. The driver imports
#: ``_rote_schema`` and (only when a step calls MCP) ``_rote_mcp`` by these
#: names, from the scripts dir it puts first on ``sys.path``.
HELPER_FILES: dict[str, Path] = {
    "_rote_skill.py": DRIVER_PATH,
    "_rote_schema.py": _HERE.parent / "plan" / "_schema.py",
    "_rote_mcp.py": _HERE.parent / "mcp" / "_runtime_helper.py",
}


def write_scripts(scripts: Path, plan: dict[str, object]) -> Path:
    """Write ``plan.json`` and the verbatim helpers into ``scripts``; return the driver.

    Not the whole emitter (that writes ``extracted/``, SKILL.md and the
    manifest too): the part every compiled skill shares.
    """
    scripts.mkdir(parents=True, exist_ok=True)
    for name, source in HELPER_FILES.items():
        (scripts / name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    (scripts / "plan.json").write_text(plan_text(plan), encoding="utf-8")
    return scripts / "_rote_skill.py"
