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
else the current directory, which is where Claude Code is started for the
rule to apply; ``--claude-project`` picks another. The file is merged, never
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
    """The project whose ``.claude/skills`` holds the skill, else ``cwd``."""
    parent = skill_dir.parent
    if parent.name == "skills" and parent.parent.name == ".claude":
        return parent.parent.parent
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


def allow_rule_action(name: str, project: Path, command: str) -> Action | None:
    """The Action that saves ``name``'s rule in ``project``; None when it is saved already."""
    path = project / SETTINGS_FILE
    read = snapshot(path)
    rule = rule_for(name)
    content = _merged(read, rule)
    if content is None:
        return None
    return write_file_action(
        id=f"claude-allow-{name}",
        title=f"Let Claude Code run {name} without asking, in {project}",
        read=read,
        content=content,
        summary=f"allow {rule}",
        replaces=False,
        command=command,
    )
