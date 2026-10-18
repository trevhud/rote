"""Structural checks for the Claude Code plugin's skills.

The plugin's skills are prose an agent follows, so the failure mode is
drift: a skill names a file rote no longer writes, links a reference
that moved, or lists stub markers the CLI stopped using. Nothing else in
the suite reads these files, so these tests are the only thing that
notices.
"""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path

import pytest
import yaml

from rote.adapters import get_adapter
from rote.cli import STUB_MARKERS
from rote.ir import load_pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILLS_DIR = REPO_ROOT / "plugin" / "skills"
IMPLEMENT = SKILLS_DIR / "implement"

SKILL_FILES = sorted(SKILLS_DIR.glob("*/SKILL.md"))

# Claude Code truncates description + when_to_use at 1,536 characters in
# the skill listing (code.claude.com/docs/en/skills).
_DESCRIPTION_LIMIT = 1536

_LINK = re.compile(r"\]\(([^)#\s]+)\)")
_BACKTICKED = re.compile(r"`([^`\s]+)`")


def _split_frontmatter(path: Path) -> tuple[dict[str, object], str]:
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n"), f"{path} has no YAML frontmatter"
    head, _, body = text[4:].partition("\n---\n")
    data = yaml.safe_load(head)
    assert isinstance(data, dict), f"{path} frontmatter is not a mapping"
    return data, body


def test_every_skill_is_discovered() -> None:
    names = {p.parent.name for p in SKILL_FILES}
    assert {"compile", "implement", "serve"} <= names


@pytest.mark.parametrize("skill_md", SKILL_FILES, ids=lambda p: p.parent.name)
def test_frontmatter_is_valid(skill_md: Path) -> None:
    data, body = _split_frontmatter(skill_md)
    # A frontmatter name replaces the directory segment of /rote:<name>,
    # so the two must agree or the documented command stops existing.
    assert data.get("name") == skill_md.parent.name
    description = data.get("description")
    assert isinstance(description, str) and description.strip()
    assert len(description) <= _DESCRIPTION_LIMIT
    assert body.strip(), f"{skill_md} has no instructions"


@pytest.mark.parametrize("skill_md", SKILL_FILES, ids=lambda p: p.parent.name)
def test_relative_links_resolve(skill_md: Path) -> None:
    md_files = [skill_md, *sorted(skill_md.parent.glob("references/*.md"))]
    for md in md_files:
        for target in _LINK.findall(md.read_text(encoding="utf-8")):
            if "://" in target:
                continue
            assert (md.parent / target).is_file(), f"{md}: broken link {target}"


def test_implement_lists_the_cli_stub_markers() -> None:
    """The skill tells the agent which strings keep a file on the TODO
    list. If the CLI changes its markers, the agent would leave a marker
    in finished code, or chase one that no longer means anything."""
    text = (IMPLEMENT / "references" / "runtimes.md").read_text(encoding="utf-8")
    section = text.split("## Stub markers", 1)[1]
    listed = set(re.findall(r"^- `([^`]+)`", section, flags=re.MULTILINE))
    assert listed == set(STUB_MARKERS)


# Files rote writes that the skill deliberately lets the user amend:
# dependency manifests (the skill says to add new dependencies there).
_USER_AMENDABLE = {"package.json", "requirements.txt"}


def _owned_patterns() -> list[str]:
    """Every backticked path in the skill's "Never edit" section."""
    text = (IMPLEMENT / "SKILL.md").read_text(encoding="utf-8")
    section = text.split("## 5. Never edit rote-owned files", 1)[1].split("\n## ", 1)[0]
    return _BACKTICKED.findall(section)


def _is_user_stub(rel: str) -> bool:
    parts = rel.split("/")
    if "extracted" not in parts:
        return False
    name = parts[-1]
    return not name.startswith("_") and name not in {"__init__.py"}


@pytest.mark.parametrize(
    ("example", "runtime"),
    [
        ("bdr-outreach", "dbos"),
        ("bdr-outreach", "temporal"),
        ("bdr-outreach", "cloudflare"),
        ("bdr-outreach", "dbos-ts"),
        ("bdr-outreach", "inngest"),
        ("invoice-push", "python"),
    ],
)
def test_implement_names_every_rote_owned_file(example: str, runtime: str, tmp_path: Path) -> None:
    """Every file rote emits outside the user's stub modules must be
    covered by the skill's "Never edit" list, so a new emitted helper or
    entrypoint cannot quietly become something the agent edits.

    BDR exercises judges, agent loops, MCP bindings and gates, which pulls
    in every verbatim helper; invoice-push is HITL-free, which the python
    adapter requires.
    """
    pipeline = load_pipeline(REPO_ROOT / "examples" / example / "expected" / "pipeline.yaml")
    written = get_adapter(runtime).emit(pipeline, tmp_path / runtime)
    root = (tmp_path / runtime).resolve()
    patterns = _owned_patterns()

    owned = set()
    for path in written.values():
        rel = path.resolve().relative_to(root).as_posix()
        if _is_user_stub(rel) or rel.rsplit("/", 1)[-1] in _USER_AMENDABLE:
            continue
        owned.add(rel)
    owned.add(".rote-manifest.json")
    assert owned, "emit produced no rote-owned files; the check would be vacuous"

    def covered(rel: str) -> bool:
        name = rel.rsplit("/", 1)[-1]
        return any(fnmatch.fnmatch(rel, p) or p == name for p in patterns)

    uncovered = sorted(rel for rel in owned if not covered(rel))
    assert not uncovered, f"implement skill's Never-edit list misses: {uncovered}"
