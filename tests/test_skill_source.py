"""Tests for SKILL.md section parsing and provenance hashing.

Provenance is the foundation of incremental re-graduation: per-node
``source.section`` refs (agent-written) plus tool-computed section
hashes let ``rote graduate --update`` re-derive only the nodes whose
source material actually changed. These tests pin the section grammar,
the hash normalization, the sidecar shape — and the invariant that
provenance never perturbs the pipeline identity hash.
"""

from __future__ import annotations

import pytest

from rote.adapters._common import _pipeline_hash
from rote.ir import Node, NodeKind, Pipeline, SourceRef
from rote.skill_source import (
    PREAMBLE_KEY,
    build_provenance,
    compute_section_hashes,
    parse_sections,
    section_hash,
)

SKILL_MD = """\
---
name: demo
---

Intro prose before any heading.

# Demo skill

Top-level description.

## Phase 1: Gather

Step one instructions.

```md
# not a heading — inside a code fence
```

## Phase 2: Judge

Step two instructions.
"""


# ───────── parse_sections ─────────


def test_parse_sections_splits_on_atx_headings() -> None:
    sections = parse_sections(SKILL_MD)
    assert set(sections) == {PREAMBLE_KEY, "Demo skill", "Phase 1: Gather", "Phase 2: Judge"}
    assert "Step one instructions." in sections["Phase 1: Gather"]
    assert "Step two instructions." in sections["Phase 2: Judge"]
    assert "Intro prose" in sections[PREAMBLE_KEY]


def test_parse_sections_ignores_headings_inside_code_fences() -> None:
    sections = parse_sections(SKILL_MD)
    assert "not a heading" not in set(sections)
    assert "# not a heading — inside a code fence" in sections["Phase 1: Gather"]


def test_parse_sections_concatenates_duplicate_headings() -> None:
    md = "# A\n\nfirst\n\n# B\n\nmiddle\n\n# A\n\nsecond\n"
    sections = parse_sections(md)
    assert "first" in sections["A"]
    assert "second" in sections["A"]


def test_parse_sections_omits_empty_preamble() -> None:
    sections = parse_sections("# Only\n\nbody\n")
    assert PREAMBLE_KEY not in sections


def test_section_hash_is_insensitive_to_surrounding_blank_lines() -> None:
    assert section_hash("## H\n\nbody\n\n\n") == section_hash("\n## H\n\nbody")
    assert section_hash("## H\n\nbody") != section_hash("## H\n\nbody changed")


# ───────── build_provenance ─────────


def _node(node_id: str, section: str | None) -> Node:
    return Node(
        id=node_id,
        kind=NodeKind.PURE_FUNCTION,
        description="x",
        impl=f"extracted/{node_id}.py:{node_id}",
        source=SourceRef(section=section) if section is not None else None,
    )


def _pipeline(*nodes: Node) -> Pipeline:
    return Pipeline(
        name="demo",
        input={"type": "In", "required": [], "optional": []},
        nodes=list(nodes),
        edges=[],
        entry_nodes=[nodes[0].id],
        exit_nodes=[nodes[-1].id],
    )


def test_build_provenance_resolves_section_hashes() -> None:
    pipeline = _pipeline(_node("gather", "Phase 1: Gather"), _node("judge", "Phase 2: Judge"))
    prov = build_provenance(pipeline, SKILL_MD)

    expected = compute_section_hashes(SKILL_MD)
    assert prov["nodes"]["gather"] == {
        "section": "Phase 1: Gather",
        "content_hash": expected["Phase 1: Gather"],
    }
    # Every section is recorded, node-mapped or not — added-section
    # detection needs the full census.
    assert set(prov["sections"]) == set(expected)
    assert prov["version"] == 1


def test_build_provenance_unknown_section_gets_null_hash() -> None:
    """An agent typo in source.section degrades to 'always re-derive
    this node', never to a failed run."""
    pipeline = _pipeline(_node("gather", "No Such Heading"))
    prov = build_provenance(pipeline, SKILL_MD)
    assert prov["nodes"]["gather"]["content_hash"] is None


def test_build_provenance_omits_nodes_without_source() -> None:
    pipeline = _pipeline(_node("gather", None))
    prov = build_provenance(pipeline, SKILL_MD)
    assert prov["nodes"] == {}


# ───────── provenance never re-versions the workflow ─────────


def test_source_ref_does_not_change_pipeline_hash() -> None:
    """The pipeline hash versions emitted workflow types. Annotating
    provenance is a metadata change, not a behavior change — it must
    not orphan in-flight durable workflows."""
    bare = _pipeline(_node("gather", None))
    annotated = _pipeline(_node("gather", "Phase 1: Gather"))
    assert _pipeline_hash(bare) == _pipeline_hash(annotated)


# ───────── SourceRef validation ─────────


def test_source_ref_rejects_malformed_content_hash() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="sha256"):
        SourceRef(section="A", content_hash="not-a-hash")


def test_source_ref_accepts_valid_content_hash() -> None:
    ref = SourceRef(section="A", content_hash="a" * 64)
    assert ref.content_hash == "a" * 64
