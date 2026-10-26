"""Whether compiling a skill to the skill runtime will pay off.

A compiled skill runs its code itself and stops for the agent only at a
model step: a judge (a DECIDE block) or an agent loop (a LOOP block). Each
stop is a round trip in the agent's conversation, which re-reads everything
said so far, so a skill whose work is mostly judgment gets slower and
costlier compiled. Measured on Sonnet 4.6 (2026-10-26):

==========================  ===========  ===========================
skill                       model stops  compiled vs the original
==========================  ===========  ===========================
expense-audit               0            3.6x faster, 62% cheaper
docs-digest                 1            as fast, 47% cheaper
ops-report                  1-2          a little faster, even cost
search-company-knowledge    4            4x slower, 3x costlier
==========================  ===========  ===========================

A human gate is not counted: the original skill stops for the person too.
Judges answered by the inference helper (``--judges helper``) are API calls,
not round trips, so they are not counted either.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

#: The most model stops a compiled skill may have and still be compiled
#: by default. The table above puts the line between 2 and 4.
MAX_MODEL_STOPS = 2


@dataclass(frozen=True)
class Worth:
    #: ``(node id, kind)`` of each step that stops for the agent.
    stops: tuple[tuple[str, str], ...]

    @property
    def pays_off(self) -> bool:
        return len(self.stops) <= MAX_MODEL_STOPS

    def explain(self) -> str:
        names = ", ".join(f"{node} ({kind})" for node, kind in self.stops)
        return (
            f"compiling this skill would not pay off: each run would stop for the model "
            f"{len(self.stops)} times ({names}), and each stop is a round trip that "
            f"re-reads the conversation. Compiling helps a skill whose work is mostly code "
            f"(at most {MAX_MODEL_STOPS} model steps); one that is mostly judgment runs "
            "slower and costs more compiled, so keep it as a skill. Pass --compile-anyway "
            "to compile it regardless"
        )


def worth(nodes: Iterable[Any], judges: str | None) -> Worth:
    """The model stops in a plan's ``nodes`` (System-1's NodePlans, each with
    an ``id`` and a ``kind``), with judges counted unless ``judges`` is
    ``"helper"``. Loop-body sub-nodes run inside their loop's stop."""
    counted = {"agent_loop"} | (set() if judges == "helper" else {"llm_judge"})
    stops = tuple(
        (n.id, _kind(n)) for n in nodes if _kind(n) in counted and not getattr(n, "in_loop", False)
    )
    return Worth(stops)


def _kind(node: Any) -> str:
    kind = node.kind
    return str(getattr(kind, "value", kind))
