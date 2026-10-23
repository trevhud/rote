"""Reading a skill's files with bounds: SKILL.md, its references, its tool snapshot.

A skill is untrusted input. It may be third-party, or sit in a cloned
repository's ``.claude/skills`` that ``rote skills list`` scans without
being asked to. Measured on hostile skills: a 1.5 GB SKILL.md cost 7.3 GB
of RSS to read, and ``references -> /`` walked the whole filesystem. So
every reader that takes a skill's text whole goes through here:

- ``SKILL.md`` is refused above :data:`SKILL_MD_MAX_BYTES`, checked with
  ``stat`` before a byte is read.
- A file under ``references/`` is read only when its resolved path stays
  inside the resolved skill directory (a symlink out is skipped), the
  walk never follows a symlinked directory, and all of them together are
  refused above :data:`REFERENCES_MAX_BYTES`, again before reading.
- The bundled tool snapshot is refused above :data:`BUNDLED_TOOLS_MAX_BYTES`.

The limits are far above any real skill (the largest example is under
100 KB), so a normal compile reads exactly what it read before.
"""

from __future__ import annotations

import os
from pathlib import Path

SKILL_MD = "SKILL.md"
REFERENCES_DIR = "references"

MIB = 1024 * 1024
SKILL_MD_MAX_BYTES = 4 * MIB
REFERENCES_MAX_BYTES = 16 * MIB
BUNDLED_TOOLS_MAX_BYTES = 16 * MIB


class SkillTooLarge(ValueError):
    """A skill file, or the references together, exceed what rote will read."""


def check_size(path: Path, limit: int, what: str) -> int:
    size = path.stat().st_size
    if size > limit:
        raise SkillTooLarge(
            f"{what} is {size / MIB:.1f} MB; rote reads at most {limit // MIB} MB of it"
        )
    return size


def read_skill_md(skill_dir: Path) -> str:
    path = skill_dir / SKILL_MD
    check_size(path, SKILL_MD_MAX_BYTES, SKILL_MD)
    return path.read_text(encoding="utf-8")


def reference_paths(skill_dir: Path, *, recursive: bool) -> list[Path]:
    """The ``*.md`` files under ``references/`` that stay inside the skill, sorted.

    ``recursive`` walks subdirectories (never through a symlinked one);
    otherwise only the top level, as the System-1 parser reads it.
    """
    refs = skill_dir / REFERENCES_DIR
    if not refs.is_dir():
        return []
    root = skill_dir.resolve()
    if not refs.resolve().is_relative_to(root):
        return []
    if recursive:
        found = [
            Path(directory) / name
            for directory, _, files in os.walk(refs, followlinks=False)
            for name in files
            if name.endswith(".md")
        ]
    else:
        found = list(refs.glob("*.md"))
    return sorted(p for p in found if p.is_file() and p.resolve().is_relative_to(root))


def read_references(skill_dir: Path, *, recursive: bool) -> list[tuple[Path, str]]:
    """``(path, text)`` for :func:`reference_paths`, refused as a whole above
    :data:`REFERENCES_MAX_BYTES` before any of them is read."""
    paths = reference_paths(skill_dir, recursive=recursive)
    total = sum(p.stat().st_size for p in paths)
    if total > REFERENCES_MAX_BYTES:
        raise SkillTooLarge(
            f"{REFERENCES_DIR}/ holds {total / MIB:.1f} MB of markdown; rote reads at most "
            f"{REFERENCES_MAX_BYTES // MIB} MB"
        )
    return [(p, p.read_text(encoding="utf-8")) for p in paths]
