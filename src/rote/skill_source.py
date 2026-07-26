"""Provenance between a SKILL.md and its compiled pipeline.

A compiled ``pipeline.yaml`` records, per node, which SKILL.md section
the node was derived from (``Node.source.section`` — written by the
compiler agent, which knows the mapping but cannot compute hashes).
This module supplies the deterministic half: it splits a SKILL.md into
heading-delimited sections, hashes each one, and writes a
``provenance.json`` sidecar next to the pipeline.

The sidecar is what makes incremental re-compilation possible: when the
skill changes, comparing fresh section hashes against the stamped ones
identifies exactly which nodes' source material moved — unchanged
sections keep their nodes verbatim, so the agent only re-derives what
actually changed.

Section hashes live in the sidecar rather than in ``pipeline.yaml``
because stamping them into the YAML would mean rewriting the
agent-authored file (losing its comments) — and because provenance is
tooling metadata, not pipeline behavior.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from rote.ir import Pipeline

PROVENANCE_FILENAME = "provenance.json"

# ATX headings only (``## Heading``). Setext headings (underlined with
# === / ---) are rare in skills and ambiguous to detect cheaply; a skill
# using them parses as one big preamble section, which degrades to
# "everything changed" — safe, just not incremental.
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")

#: Key under which text before the first heading is recorded.
PREAMBLE_KEY = ""


def parse_sections(markdown: str) -> dict[str, str]:
    """Split a markdown document into heading-delimited sections.

    Returns a mapping of heading text (without the ``#`` markers, any
    level) → section text (the heading line through the line before the
    next heading). Text before the first heading is keyed by
    :data:`PREAMBLE_KEY`. A duplicated heading gets its occurrences
    concatenated — a change in any of them changes the shared hash,
    which errs toward re-deriving too much rather than too little.
    """
    sections: dict[str, list[str]] = {}
    current = PREAMBLE_KEY
    sections[current] = []
    in_fence = False
    for line in markdown.splitlines():
        # Headings inside fenced code blocks are content, not structure.
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
        m = None if in_fence else _HEADING_RE.match(line)
        if m:
            current = m.group(2)
            sections.setdefault(current, [])
        sections[current].append(line)
    return {
        heading: "\n".join(lines)
        for heading, lines in sections.items()
        if heading != PREAMBLE_KEY or any(ln.strip() for ln in lines)
    }


def section_hash(text: str) -> str:
    """sha256 of a section's text, insensitive to leading/trailing blanks."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def compute_section_hashes(markdown: str) -> dict[str, str]:
    return {heading: section_hash(text) for heading, text in parse_sections(markdown).items()}


def build_provenance(pipeline: Pipeline, skill_md_text: str) -> dict[str, Any]:
    """Assemble the provenance sidecar payload.

    ``sections`` records the hash of *every* section at compilation time
    (not just node-mapped ones) so a later diff can also detect sections
    that were added to the skill. A node whose ``source.section`` names
    a heading that doesn't exist gets ``content_hash: null`` — treated
    as "always changed" by the incremental path, which degrades to
    re-deriving that node rather than failing a completed run.
    """
    hashes = compute_section_hashes(skill_md_text)
    nodes: dict[str, Any] = {}
    for node in pipeline.nodes:
        if node.source is None:
            continue
        nodes[node.id] = {
            "section": node.source.section,
            "content_hash": hashes.get(node.source.section),
        }
    return {
        "version": 1,
        "skill_md_sha256": hashlib.sha256(skill_md_text.encode("utf-8")).hexdigest(),
        "sections": dict(sorted(hashes.items())),
        "nodes": dict(sorted(nodes.items())),
    }


def write_provenance(path: Path, pipeline: Pipeline, skill_md_text: str) -> dict[str, Any]:
    payload = build_provenance(pipeline, skill_md_text)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def load_provenance(path: Path) -> dict[str, Any]:
    """Load a provenance sidecar, validating the minimal shape."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("sections"), dict):
        raise ValueError(f"Provenance file {path} has no 'sections' mapping")
    if not isinstance(data.get("nodes"), dict):
        raise ValueError(f"Provenance file {path} has no 'nodes' mapping")
    return data


__all__ = [
    "PROVENANCE_FILENAME",
    "PREAMBLE_KEY",
    "build_provenance",
    "compute_section_hashes",
    "load_provenance",
    "parse_sections",
    "section_hash",
    "write_provenance",
]
