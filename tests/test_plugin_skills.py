"""The plugin skills tell the agent the same thing rote checks.

Both /rote:compile and /rote:setup carry the rule for when compiling pays
off, so the agent judges before it spends. A skill cannot include another,
so the section is copied, and these keep the copies, and the line they
state, equal to what ``rote.compiler.worth`` enforces.
"""

from __future__ import annotations

import re
from pathlib import Path

from rote.compiler.worth import MAX_MODEL_STOPS

SKILLS = Path(__file__).resolve().parent.parent / "plugin" / "skills"
WORTH = re.compile(r"<!-- rote:worth -->\n(.*?)<!-- /rote:worth -->", re.DOTALL)
NUMBERS = {1: "one", 2: "two", 3: "three", 4: "four"}


def _section(skill: str) -> str:
    found = WORTH.findall((SKILLS / skill / "SKILL.md").read_text(encoding="utf-8"))
    assert len(found) == 1, f"{skill}/SKILL.md must hold the rote:worth section once"
    return str(found[0])


def test_compile_and_setup_carry_the_same_rule() -> None:
    assert _section("compile") == _section("setup")


def test_the_rule_states_the_line_rote_enforces() -> None:
    text = " ".join(_section("compile").split())
    assert f"more than {NUMBERS[MAX_MODEL_STOPS]}," in text
    assert "--compile-anyway" in text
