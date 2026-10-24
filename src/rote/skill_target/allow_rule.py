"""The Claude Code allow rule that lets an agent run one compiled skill without a prompt.

``rote skill install`` offers to save ``Bash(rote skill run <name> *)`` as
its own confirmed Action, never implied by ``--yes``: it changes another
program's permissions. Where it goes is what was measured, and nothing
else (Claude Code 2.1.309, ``claude -p --permission-mode dontAsk``,
2026-10-23):

- ``<project>/.claude/settings.local.json`` with ``{"permissions": {"allow":
  ["Bash(rote skill run docs-digest *)"]}}`` let ``rote skill run docs-digest
  status`` and ``rote skill run docs-digest start '{"topic": ...}'`` run, and
  denied ``rote skill run other-skill status``. That file is the one Claude
  Code's own "Yes, and don't ask again" writes (design doc section 1).
- The same rule did not let ``ROTE_MCP_CONFIG=... rote skill run
  docs-digest status``, ``env ROTE_MCP_CONFIG=... rote ...`` or ``export
  ...; rote ...`` run: an environment prefix does not match the prefix rule.
- The user-level ``~/.claude/settings.json`` could not be measured: moving
  it (``CLAUDE_CONFIG_DIR`` or ``HOME``) also moves Claude Code's login, and
  the probe must not edit the real file. So rote does not write it.

The project is the one the skill lives in (``<project>/.claude/skills/<name>``),
else the root of the git work tree the current directory is in (Claude Code
2.1.211+ keeps the local file at the repository root, per its settings docs),
else the current directory; ``--claude-project`` picks another. The file is merged, never
replaced: every key and rule already there is kept, and the write is refused
if the file changed since it was read (:func:`rote.interact.plan.write_file_action`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rote.interact.plan import Action, Snapshot, snapshot, write_file_action

SETTINGS_FILE = Path(".claude") / "settings.local.json"


def rule_for(name: str) -> str:
    """The rule the SKILL.md's ``allowed-tools`` names too (``rote.adapters.skill``)."""
    return f"Bash(rote skill run {name} *)"


def default_project(skill_dir: Path, cwd: Path) -> Path:
    """The project whose ``.claude/skills`` holds the skill; else the git work tree
    ``cwd`` is in, whose root is where Claude Code 2.1.211+ reads and writes the
    local file (not the home dir, which it treats as no repository); else ``cwd``."""
    parent = skill_dir.parent
    if parent.name == "skills" and parent.parent.name == ".claude":
        return parent.parent.parent
    import shutil
    import subprocess

    git = shutil.which("git")
    if git is not None:
        proc = subprocess.run(
            [git, "-C", str(cwd), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=False,
        )
        root = Path(proc.stdout.strip()) if proc.returncode == 0 and proc.stdout.strip() else None
        from rote.clients import ClientContext

        # The client's home (ROTE_CLIENT_HOME_DIR moves it): Claude Code
        # treats a repository rooted there as none.
        if root is not None and root != ClientContext.current().home:
            return root
    return cwd


def _merged(read: Snapshot, rule: str) -> str | None:
    """The settings text with ``rule`` allowed; None when it already is."""
    text = read.text()
    try:
        data: Any = json.loads(text) if text is not None and text.strip() else {}
    except ValueError as e:
        raise ValueError(f"{read.path} is not JSON ({e}); fix it, then install again") from e
    if not isinstance(data, dict):
        raise ValueError(f"{read.path} does not hold a JSON object")
    permissions = data.setdefault("permissions", {})
    if not isinstance(permissions, dict):
        raise ValueError(f"{read.path}: permissions is not an object")
    allow = permissions.setdefault("allow", [])
    if not isinstance(allow, list):
        raise ValueError(f"{read.path}: permissions.allow is not a list")
    if rule in allow:
        return None
    allow.append(rule)
    return json.dumps(data, indent=2) + "\n"


def tracked_by_git(path: Path) -> bool:
    """Whether ``path`` is in a git work tree and not ignored there.

    Claude Code keeps the file out of git only when it writes it itself (it
    adds ``**/.claude/settings.local.json`` to the global git excludes; the
    settings docs say a hand-made file is not covered). A tracked file also
    needs workspace trust before its allow rules apply. So rote asks git
    (``git check-ignore``), says so in the Action when the file would be
    committed, and changes nothing in the user's ignore files."""
    import shutil
    import subprocess

    git = shutil.which("git")
    if git is None or not path.parent.parent.is_dir():
        return False
    proc = subprocess.run(
        [git, "-C", str(path.parent.parent), "check-ignore", "-q", str(path)],
        capture_output=True,
        check=False,
    )
    # 0: ignored. 1: not ignored. 128: not a work tree (or another git error).
    return proc.returncode == 1


def allow_rule_action(name: str, project: Path, command: str) -> Action | None:
    """The Action that saves ``name``'s rule in ``project``; None when it is saved already."""
    path = project / SETTINGS_FILE
    read = snapshot(path)
    rule = rule_for(name)
    content = _merged(read, rule)
    if content is None:
        return None
    title = f"Let Claude Code run {name} without asking, in {project}"
    if tracked_by_git(path):
        title += (
            f". Warning: git does not ignore {SETTINGS_FILE} here, so it would be committed "
            "with the rule; add it to .gitignore if that is not what you want"
        )
    return write_file_action(
        id=f"claude-allow-{name}",
        title=title,
        read=read,
        content=content,
        summary=f"allow {rule}",
        replaces=False,
        command=command,
    )
