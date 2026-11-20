"""Whether compiling a skill to the skill runtime will pay off.

A compiled skill runs its code itself and stops for the agent at a model
step: a judge (a DECIDE block), an agent loop (a LOOP block), or a call no
bound tool makes (a DO block, which the agent does with its own tools). Each
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

A run of decisions with nothing between them is asked in one DECIDE block
(#344), but each judge still counts here. Measured 2026-10-31 (Sonnet 4.6,
2 trials a side) on contact-note-triage, three judges and one small pure
function: batching cut the compiled side from 6 requests and 72.5k input
tokens to 5 and 60.6k, and the original still did the work in 2 requests and
22.5k, so the compiled skill cost 2.7x the tokens. Counting a batch as one
stop would have let that skill compile: the line moves only on a measured
eval that says so.

Stops after batching (``_driver.stops``, listed in compile-report.md) do
not predict cost better. Broken down per request (2026-11-03), each request
re-read about 11k tokens of fixed context and the driver's blocks added
under 1k, so the compiled side cost about one 11k read per stop. Whether a
stop pays off depends on how much work the original did between stops,
which the plan does not show. And in the committed examples a code or tool
step sits between every two judges (deal-monitor, bdr-outreach), so stops
equal judges there; the one skill where they differ is the one above.

A call no bound tool makes counts like a judge. Measured 2026-11-19 (Sonnet
4.6, 3 trials a side, ops-report against its fixture servers), one pipeline
run two ways, its four reads bound to MCP tools and then unbound:

==========================  ========  ======  ======  =====  ======
arm                         requests  input   output  wall   cost
==========================  ========  ======  ======  =====  ======
the original skill          4         59k     3.5k    47 s   $0.113
compiled, reads bound       5         70k     3.4k    43 s   $0.118
compiled, reads as DO       7 to 9    189k    12.7k   142 s  $0.367
==========================  ========  ======  ======  =====  ======

The four DO steps shared one block (``_driver.do_run``) and still made the
run 3.1x costlier and 3x slower: the agent makes each call, then types every
result back as its answer, with 7k to 9k thinking tokens spent on that. So,
as with judges, each call counts and a shared block does not lower the
count. The measurement is of reads; the IR does not say which calls write,
so a write counts the same. A call is unbound for several reasons (no server
rote listed offers the tool, a required argument has no source, the step is
a file or a web page and not a tool), and all count. On ``--backend api``
the compile is expected to write these calls as code, so none is counted,
though one it leaves a stub still becomes a DO step.

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
#: What a stop is called when it is an external call with no tool bound.
UNBOUND_CALL = "call with no tool bound"


@dataclass(frozen=True)
class Worth:
    #: ``(node id, kind)`` of each step that stops for the agent.
    stops: tuple[tuple[str, str], ...]

    @property
    def pays_off(self) -> bool:
        return len(self.stops) <= MAX_MODEL_STOPS

    def explain(self) -> str:
        names = ", ".join(f"{node} ({kind})" for node, kind in self.stops)
        unbound = [node for node, kind in self.stops if kind == UNBOUND_CALL]
        calls = (
            ""
            if not unbound
            else f" {len(unbound)} of them ({', '.join(unbound)}) are calls no bound MCP tool "
            "makes: the agent would make each one with its own tools and type the result "
            "back, which measured 3x slower and costlier than the same calls made in code. "
            "Where a call goes to an MCP server, register it (rote mcp import, or rote mcp "
            "add), log in to it (rote mcp login) and compile again: a call rote can bind to "
            "a tool is made by the compiled skill itself."
        )
        return (
            f"compiling this skill would not pay off: each run would stop for the model "
            f"{len(self.stops)} times ({names}), and each stop is a round trip that "
            f"re-reads the conversation.{calls} Compiling helps a skill whose work is mostly "
            f"code (at most {MAX_MODEL_STOPS} model steps); one that is mostly judgment runs "
            "slower and costs more compiled, so keep it as a skill. Pass --compile-anyway "
            "to compile it regardless"
        )


def worth(nodes: Iterable[Any], judges: str | None, backend: str | None) -> Worth:
    """The model stops in a plan's ``nodes`` (System-1's NodePlans, each with
    an ``id``, a ``kind`` and its ``mcp`` binding), with judges counted unless
    ``judges`` is ``"helper"``, and an external call with no tool bound
    counted unless ``backend`` is ``"api"`` (there the compile is expected to
    write the call as code). Loop-body sub-nodes run inside their loop's stop."""
    counted = {"agent_loop"} | (set() if judges == "helper" else {"llm_judge"})
    stops: list[tuple[str, str]] = []
    for n in nodes:
        if getattr(n, "in_loop", False):
            continue
        kind = _kind(n)
        if kind in counted:
            stops.append((n.id, kind))
        elif kind == "external_call" and backend != "api" and n.mcp is None:
            stops.append((n.id, UNBOUND_CALL))
    return Worth(tuple(stops))


def _kind(node: Any) -> str:
    kind = node.kind
    return str(getattr(kind, "value", kind))
