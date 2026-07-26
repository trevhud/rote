"""Incremental re-compilation planning.

A full compilation costs ~13 minutes and real tokens. When only part of
a SKILL.md changed, most of that spend re-derives nodes whose source
material is untouched. This module turns the provenance sidecar
(:mod:`rote.skill_source`) into an :class:`UpdatePlan`: which sections
changed, which nodes are provenance-verified unchanged (and must
survive verbatim), and which need re-deriving.

The plan drives two things:

1. The ``UPDATE.md`` brief materialized into the driver's work dir —
   the agent's instructions for a minimal-diff run.
2. Post-run enforcement — the orchestrator refuses an update that
   dropped or renamed a preserved node, because node ids are function
   names in emitted code and feed the pipeline hash that versions
   in-flight durable workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rote.ir import Pipeline
from rote.skill_source import compute_section_hashes

#: Directory inside the driver work dir holding read-only update context
#: (previous pipeline.yaml, previous stubs, UPDATE.md). Removed before
#: the work dir is merged into the user's output.
UPDATE_CONTEXT_DIRNAME = "update-context"


@dataclass(frozen=True)
class UpdatePlan:
    """The section diff and its consequences for the previous pipeline."""

    changed_sections: tuple[str, ...]
    added_sections: tuple[str, ...]
    removed_sections: tuple[str, ...]
    #: Nodes whose source section's hash is unchanged — preserved verbatim.
    preserved_node_ids: tuple[str, ...]
    #: Nodes that must be re-examined: their section changed or was
    #: removed, or they carry no verifiable provenance.
    stale_node_ids: tuple[str, ...]

    @property
    def is_noop(self) -> bool:
        """True when no section changed — nothing for the agent to do."""
        return not (self.changed_sections or self.added_sections or self.removed_sections)


def build_update_plan(
    prev_pipeline: Pipeline,
    prev_provenance: dict[str, Any],
    skill_md_text: str,
) -> UpdatePlan:
    """Diff the current SKILL.md against the stamped provenance.

    A node counts as preserved only when its provenance entry carries a
    hash *and* that hash matches the same-named section in the current
    skill. Everything else — changed section, removed section, agent
    typo'd section name (null hash), node with no ``source`` at all —
    is stale, which errs toward re-deriving too much rather than
    silently keeping a node whose source material moved.
    """
    new_hashes = compute_section_hashes(skill_md_text)
    old_hashes: dict[str, Any] = prev_provenance["sections"]

    changed = tuple(
        sorted(s for s in new_hashes if s in old_hashes and old_hashes[s] != new_hashes[s])
    )
    added = tuple(sorted(set(new_hashes) - set(old_hashes)))
    removed = tuple(sorted(set(old_hashes) - set(new_hashes)))

    node_prov: dict[str, Any] = prev_provenance["nodes"]
    preserved: list[str] = []
    stale: list[str] = []
    for node in prev_pipeline.nodes:
        entry = node_prov.get(node.id)
        content_hash = entry.get("content_hash") if entry else None
        section = str(entry.get("section", "")) if entry else ""
        if content_hash is not None and new_hashes.get(section) == content_hash:
            preserved.append(node.id)
        else:
            stale.append(node.id)

    return UpdatePlan(
        changed_sections=changed,
        added_sections=added,
        removed_sections=removed,
        preserved_node_ids=tuple(preserved),
        stale_node_ids=tuple(stale),
    )


def _bullet_list(items: tuple[str, ...], empty: str = "(none)") -> str:
    if not items:
        return f"- {empty}"
    return "\n".join(f"- {item}" for item in items)


def render_update_brief(plan: UpdatePlan) -> str:
    """The UPDATE.md content the agent reads before an incremental run."""
    return f"""\
# Incremental update brief

This is an incremental re-compilation, not a from-scratch run. A
previous compilation of this skill exists; the source SKILL.md has
changed since. Everything in this `{UPDATE_CONTEXT_DIRNAME}/` directory
is read-only reference material:

- `pipeline.yaml` — the previous compiled pipeline. Start from it.
- `extracted/`, `signatures/` — the previous modules, possibly filled
  in by the user since compilation.

## What changed in the source skill

Changed sections:
{_bullet_list(plan.changed_sections)}

Added sections:
{_bullet_list(plan.added_sections)}

Removed sections:
{_bullet_list(plan.removed_sections)}

## Nodes

Preserve these nodes VERBATIM — their source sections did not change.
Copy them into the updated pipeline unmodified (same id, same fields),
unless a change in a neighboring node forces a wiring update to their
`inputs:`/edges; explain any such deviation in the compilation report:
{_bullet_list(plan.preserved_node_ids, empty="(none — every node needs re-examination)")}

Re-derive (or re-examine) only these nodes, plus any new nodes the
added sections call for:
{_bullet_list(plan.stale_node_ids)}

## Ground rules

1. Write the COMPLETE updated pipeline.yaml to the work dir root, as
   usual — it is the whole IR, not a diff.
2. Keep node ids stable wherever the underlying step survives.
   Renaming an id re-versions the emitted workflow and orphans
   in-flight runs.
3. Only write extracted/signature files that are NEW or CHANGED. Files
   you do not write are kept from the previous run — do not rewrite an
   unchanged module (the user may have filled in its implementation).
4. Keep every node's `source.section` accurate, including nodes whose
   sections were renamed.
"""
