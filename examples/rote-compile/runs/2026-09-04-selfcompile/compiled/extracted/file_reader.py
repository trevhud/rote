"""
Read a skill bundle from disk: SKILL.md + all files under references/.

Contract:
  Input:  skill_dir — absolute or relative path to a directory containing SKILL.md
  Output: dict with keys:
            skill_md      — full text of SKILL.md
            reference_files — {filename: full_text} for every file in references/
            file_paths    — sorted list of all paths read (SKILL.md first, then refs)

Raises:
  FileNotFoundError  — if skill_dir does not exist or SKILL.md is missing
  IOError            — on any read failure (the durable runtime retries this)
"""

from __future__ import annotations

import os
from pathlib import Path


SKILL_MD_FILENAME = "SKILL.md"
REFERENCES_SUBDIR = "references"


def read_skill_bundle(skill_dir: str) -> dict:
    """Read SKILL.md and all references/*.md files from skill_dir."""
    root = Path(skill_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"skill_dir does not exist: {skill_dir}")

    skill_md_path = root / SKILL_MD_FILENAME
    if not skill_md_path.is_file():
        raise FileNotFoundError(
            f"SKILL.md not found in {skill_dir}. "
            f"Expected: {skill_md_path}"
        )

    skill_md = skill_md_path.read_text(encoding="utf-8")

    references_dir = root / REFERENCES_SUBDIR
    reference_files: dict[str, str] = {}
    file_paths: list[str] = [str(skill_md_path)]

    if references_dir.is_dir():
        for ref_path in sorted(references_dir.iterdir()):
            if ref_path.is_file():
                content = ref_path.read_text(encoding="utf-8")
                reference_files[ref_path.name] = content
                file_paths.append(str(ref_path))

    return {
        "skill_md": skill_md,
        "reference_files": reference_files,
        "file_paths": file_paths,
    }
