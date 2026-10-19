"""The deal-scoring rubric the score_new_opportunities judge applies.

The source skill says: "Read the deal-scoring skill at
``~/.claude/skills/deal-scoring/SKILL.md`` and apply its scoring rules."
The pipeline keeps that as the default and lets a caller pass the rubric
text instead (``scoring_rubric``), for a scheduled run on a machine
without the skill installed.
"""

from __future__ import annotations

from pathlib import Path


def load_scoring_rubric(*, default_rubric_path: str, scoring_rubric: str | None = None) -> str:
    """The rubric text: the input when given, else the skill file.

    ``scoring_rubric`` is an optional pipeline input, so an omitted one is
    left out of the call (None here). ``default_rubric_path`` is the node's
    constant and every call sends it, so it has no default.
    """
    if scoring_rubric is not None:
        if not scoring_rubric.strip():
            raise ValueError("load_scoring_rubric: the scoring_rubric input is empty")
        return scoring_rubric
    path = Path(default_rubric_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"load_scoring_rubric: no scoring_rubric input was passed and the "
            f"deal-scoring skill is not at {path}. Install the skill there or "
            "pass the rubric text as scoring_rubric."
        )
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"load_scoring_rubric: the deal-scoring skill at {path} is empty")
    return text
