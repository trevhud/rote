"""The demo skill that ``rote try`` and ``rote setup`` run: docs-digest.

docs-digest searches Microsoft Learn and reads a repository's DeepWiki
outline, two public MCP servers that need no login, so a first run can
show a real skill becoming a real pipeline with nothing to configure but
two ``rote mcp add`` lines. It ships in the wheel the same way as the
``rote example`` bundles (one home, ``examples/docs-digest/``, listed in
:data:`rote.examples.BUNDLED` and force-included by ``pyproject.toml``).

Next to the skill and its compiled pipeline sits ``reference.json``: one
recorded run of each side on the skill's own ``input.json``. A fresh
pipeline run is compared against it field by field. The format
(:data:`REFERENCE_FORMAT`) is::

    {
      "format": "rote-demo-reference/1",
      "skill": "docs-digest",
      "recorded_on": "YYYY-MM-DD",
      "rote_version": "...", "claude_cli_version": "...",
      "input": {...},                      # the payload both sides ran on
      "servers": {name: {"url", "tool"}},  # what to `rote mcp add --read-only`
      "skill_run": {"model", "wall_seconds", "turns", "cost_usd",
                    "tokens": {"input", "output", "cache_read", "cache_write"},
                    "output": {...}},      # what the agent returned
      "compile": {"runtime", "models", "wall_seconds", "turns", "cost_usd",
                  "tokens": {...}},
      "pipeline_run": {"runtime", "wall_seconds", "model_calls", "cost_usd",
                       "output": {...}},   # what the pipeline returned
      "fields": [{"name", "pipeline", "skill", "section", "expect", "why"}],
      "drift": {...}                       # what changed between recorded runs
    }

Each entry of ``fields`` names one comparable value. ``pipeline`` and
``skill`` are JSON pointers (RFC 6901) into the two ``output`` objects,
since the pipeline returns its exit node's result under the node id.
``section``, when set, narrows a markdown string to the body under the
``## <section>`` heading. ``expect`` is ``"equal"`` (a fresh run must
match the reference) or ``"drifts"`` (live data: a difference is
reported, never a failure). Only values that are the same on every run
of the same input are ``"equal"``; everything a server can change between
runs is ``"drifts"``, with ``why`` saying which server and how.

Nothing in the file is secret or machine-specific: no paths, hosts other
than the two servers, or account names.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

#: The ``format`` value this module reads. Bump the suffix on any change a
#: reader of the old format would misread.
REFERENCE_FORMAT = "rote-demo-reference/1"

#: The demo's directory name under the bundled examples.
DEMO_NAME = "docs-digest"

Expect = Literal["equal", "drifts"]


class DemoError(RuntimeError):
    """The bundled demo or its reference is missing or malformed."""


@dataclass(frozen=True)
class Field:
    """One value a fresh run is compared on (see the module docstring)."""

    name: str
    pipeline: str
    skill: str
    section: str | None
    expect: Expect
    why: str


@dataclass(frozen=True)
class Reference:
    """``reference.json``, checked on load. ``raw`` keeps the whole record."""

    raw: dict[str, Any]
    input: dict[str, Any]
    servers: dict[str, dict[str, str]]
    skill_output: dict[str, Any]
    pipeline_output: dict[str, Any]
    fields: tuple[Field, ...]


def docs_digest_dir() -> Path:
    """``examples/docs-digest/`` in this install: ``skill/``, ``compiled/``,
    ``reference.json`` and a README."""
    from rote.examples import examples_root

    return examples_root() / DEMO_NAME


def skill_dir() -> Path:
    """The demo skill (``SKILL.md``, ``input.json``, ``mcp-tools.json``)."""
    return docs_digest_dir() / "skill"


def pipeline_path() -> Path:
    """The recorded compile's ``pipeline.yaml``; ``rote emit`` reads its
    ``extracted/`` beside it."""
    return docs_digest_dir() / "compiled" / "pipeline.yaml"


def reference_path() -> Path:
    return docs_digest_dir() / "reference.json"


def load_reference(path: Path | None = None) -> Reference:
    """Read and check ``reference.json``. Every field must resolve in both
    recorded outputs, so a reference that cannot be compared never loads."""
    path = path or reference_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise DemoError(f"{path}: cannot read the demo reference: {e}") from e
    if not isinstance(raw, dict) or raw.get("format") != REFERENCE_FORMAT:
        found = raw.get("format") if isinstance(raw, dict) else None
        raise DemoError(f"{path}: expected format {REFERENCE_FORMAT!r}, found {found!r}")
    try:
        skill_output = raw["skill_run"]["output"]
        pipeline_output = raw["pipeline_run"]["output"]
        reference = Reference(
            raw=raw,
            input=dict(raw["input"]),
            servers={str(k): dict(v) for k, v in raw["servers"].items()},
            skill_output=skill_output,
            pipeline_output=pipeline_output,
            fields=tuple(_field(entry) for entry in raw["fields"]),
        )
    except (KeyError, TypeError, ValueError) as e:
        raise DemoError(f"{path}: malformed demo reference: {e!r}") from e
    for field in reference.fields:
        value_at(pipeline_output, field.pipeline, field.section)
        value_at(skill_output, field.skill, field.section)
    return reference


def _field(entry: dict[str, Any]) -> Field:
    expect = entry["expect"]
    if expect not in ("equal", "drifts"):
        raise ValueError(f"field {entry.get('name')!r}: expect must be equal or drifts")
    section = entry.get("section")
    return Field(
        name=str(entry["name"]),
        pipeline=str(entry["pipeline"]),
        skill=str(entry["skill"]),
        section=None if section is None else str(section),
        expect=expect,
        why=str(entry["why"]),
    )


def value_at(document: Any, pointer: str, section: str | None = None) -> Any:
    """The value at JSON ``pointer`` in ``document``, narrowed to the body
    of markdown heading ``## <section>`` when ``section`` is given.

    Raises :class:`DemoError` naming what is missing, because a run whose
    output lacks a compared field is a finding, not a blank.
    """
    value = document
    if pointer:
        if not pointer.startswith("/"):
            raise DemoError(f"{pointer!r} is not a JSON pointer")
        for token in pointer[1:].split("/"):
            key = token.replace("~1", "/").replace("~0", "~")
            if isinstance(value, dict) and key in value:
                value = value[key]
            elif isinstance(value, list) and key.isdigit() and int(key) < len(value):
                value = value[int(key)]
            else:
                raise DemoError(f"no value at {pointer!r} (missing {key!r})")
    if section is None:
        return value
    if not isinstance(value, str):
        raise DemoError(f"{pointer!r} is not markdown text, so it has no section {section!r}")
    return markdown_section(value, section)


def markdown_section(text: str, heading: str) -> str:
    """The body under ``## <heading>``, up to the next ``#``/``##`` heading,
    with surrounding blank lines removed."""
    lines = text.splitlines()
    marker = f"## {heading}"
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == marker)
    except StopIteration:
        raise DemoError(f"no '{marker}' heading") from None
    body: list[str] = []
    for line in lines[start + 1 :]:
        if line.startswith("# ") or line.startswith("## "):
            break
        body.append(line)
    return "\n".join(body).strip("\n")
