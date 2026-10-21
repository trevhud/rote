"""A decision model double-checks the structure pass (``--decider clef|jev``).

The structure pass reads the whole skill and proposes the skeleton. Then the
configured decision model, reading one step at a time (which measured
better on kinds than giving it the whole skill), answers its two questions
for every SKILL.md heading: the main work's kind, and whether a person has
to act. The phrasings, the state and the thresholds are the frozen ones of
:mod:`rote.compiler.system1.decision_model`, so each heading is asked
exactly as the rules path asks a step.

Where the model is confident (at or above its frozen threshold) and the
skeleton says otherwise, that is a second opinion, never an override. The
disagreements go back to the structure pass as one more repair round, and
the pass may change the node or keep it. One it keeps is listed in the
report's Open questions with both answers. Mandatory and fan_out are not
checked, as on the rules path.

A heading's verdict is compared with every node that cites it: a step the
pass split into a call and the work on its result is two nodes under one
heading, so the kind check asks whether the model's kind is among them.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rote.compiler.system1.decision_model import Verdicts
from rote.compiler.system1.parse import Unit

#: How a kept disagreement starts in ``Plan.review``; the report titles it by it.
CHECKER_NOTE_PREFIX = "decision model disagrees: "
GATE = "hitl_gate"


def disagreements(verdicts: Verdicts, units: list[Unit], answer: Mapping[str, Any]) -> list[str]:
    """Where ``verdicts`` (one per heading, by unit index) are confident and
    ``answer`` (the structure pass's skeleton) says otherwise, one line each."""
    model = verdicts.model
    by_section: dict[int, list[Mapping[str, Any]]] = {}
    for node in answer.get("nodes") or []:
        section = node.get("section")
        if isinstance(section, int):
            by_section.setdefault(section, []).append(node)
    out: list[str] = []
    for unit in units:
        verdict = verdicts.by_unit.get(unit.index)
        if verdict is None:
            continue
        nodes = by_section.get(unit.index, [])
        ids = ", ".join(f"`{n.get('id')}`" for n in nodes)
        kinds = sorted({str(n.get("kind")) for n in nodes} - {GATE})
        where = f"section {unit.index} {unit.title!r}" + (f" ({ids})" if ids else "")
        if (
            kinds
            and verdict.kind != GATE
            and verdict.kind_confidence >= model.kind_threshold
            and verdict.kind not in kinds
        ):
            out.append(
                f"{where}: {model.label} reads this step's main work as {verdict.kind} "
                f"(confidence {verdict.kind_confidence:.2f}); the skeleton gives it "
                f"{' and '.join(kinds)}"
            )
        has_gate = any(n.get("kind") == GATE for n in nodes)
        person = verdict.human >= model.human_threshold
        if person and not has_gate:
            out.append(
                f"{where}: {model.label} says a person must act or answer here before the "
                f"steps after it run (probability {verdict.human:.2f}); the skeleton has no "
                "hitl_gate for it"
            )
        elif has_gate and not person:
            out.append(
                f"{where}: {model.label} says no person has to act here (probability a "
                f"person must: {verdict.human:.2f}); the skeleton pauses for one"
            )
    return out
