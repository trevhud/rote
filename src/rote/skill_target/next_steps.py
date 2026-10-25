"""The next steps rote prints after compiling a skill, shared by
``rote compile`` and ``rote setup``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NextSteps:
    """``swap`` reads on its own; ``rote compile`` prints it after the
    install command with a leading "then"."""

    install: str
    swap: list[str]
    proof: str


def compiled_skill_next_steps(runtime_dir: Path, skill_path: Path) -> NextSteps:
    """What to do after a skill compile: the install command, the lines on
    how an agent gets it, and the line pointing at the proof (``rote try``).

    A Claude Code skill folder may be a symlink to a directory elsewhere
    (https://code.claude.com/docs/en/skills), so the link points the agent
    at the installed directory itself and ``rote skill run`` runs what the
    install recorded. The compiled skill keeps the original's name, so it
    goes in place of the original, never beside it.

    The path is checked before the command is printed: with an entry already
    at ``~/.claude/skills/<name>``, BSD ``ln -s`` exits 0 and puts the link
    inside that folder, so the compiled skill would never load. Then the
    lines move the original aside first (nothing deleted) and say how to
    undo it. rote prints these commands and never runs them."""
    import shlex

    from rote.adapters.skill import emitted_skill
    from rote.clients.base import ClientContext
    from rote.clients.claude_code import user_skills_dir

    skill = emitted_skill(runtime_dir)
    name = skill["name"] if skill else runtime_dir.name
    target = shlex.quote(str(runtime_dir.resolve()))
    ctx = ClientContext.current()
    slot = user_skills_dir(ctx) / name

    # Printed as ~/... when under the home (the skill name is a safe
    # identifier, so only an outside CLAUDE_CONFIG_DIR needs quoting).
    shown = (
        f"~/{slot.relative_to(ctx.home)}"
        if slot.is_relative_to(ctx.home)
        else shlex.quote(str(slot))
    )
    link = f"ln -s {target} {shown}"
    proof = f"or see it give the original's answer first: rote try {shlex.quote(str(skill_path))}"
    if slot.is_symlink() and slot.resolve() == runtime_dir.resolve():
        swap = [f"your agent has it: {shown} already links to it"]
    elif slot.exists() or slot.is_symlink():
        # A free name to move it to: mv onto an existing dir nests inside it.
        aside = f"{name}.original"
        n = 2
        while (ctx.home / aside).exists() or (ctx.home / aside).is_symlink():
            aside, n = f"{name}.original-{n}", n + 1
        swap = [
            f"for Claude Code: your original {name} skill is at {shown}. To use the "
            f"compiled one instead, move the original out of the skills folder (nothing is "
            f"deleted): mv {shown} ~/{aside} && {link}",
            f"  undo: rm {shown} && mv ~/{aside} {shown}",
        ]
    else:
        swap = [f"swap it in for the original {name} skill (Claude Code: {link})"]
    return NextSteps(install=f"rote skill install {target}", swap=swap, proof=proof)
