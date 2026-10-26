"""Give a baseline's recorded tool results the shape a compiled step receives.

A baseline records what the agent saw: the text blocks of each MCP result,
read from the ``claude -p`` transcript. A compiled step reads the result
through rote's client, which returns the server's structured result when it
sends one (:func:`rote.mcp._runtime_helper._result_payload`). Most servers
send the same value both ways. Atlassian's does not: ``getConfluencePage``'s
text block is ``{"content": {"nodes": [page]}, "context": ...}`` and its
structured result is the page itself, with no outputSchema to say so. A
compile that learned the text shape wrote a judge whose input schema
rejected every page the compiled skill fetched.

So before the compile reads the recordings, each recorded call on a tool its
server marks ``readOnlyHint`` is sent once more through rote's own client,
with the recorded arguments, and the result that comes back replaces the
recorded one. Only servers the baseline called are contacted, only read-only
tools are called, and a call that fails keeps its recording.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import replace
from typing import Any


async def runtime_shapes(
    observations: Iterable[Any], log: Callable[[str], None] = lambda _line: None
) -> tuple[Any, ...]:
    """``observations`` with each read-only call's result replaced by what
    rote's client returns for the same arguments, as JSON text (the form
    :func:`rote.probe.parse_tool_result` decodes)."""
    from rote.eval.baseline import read_only_allowlist
    from rote.mcp._runtime_helper import call_mcp_tool
    from rote.mcp.live_tools import mcp_tool_id

    calls = tuple(observations)
    servers = sorted({c.server for c in calls if not c.is_error})
    if not servers:
        return calls
    allowed, _denied, skipped = await read_only_allowlist(servers)
    for server, why in sorted(skipped.items()):
        log(f"recorded results: {server!r} kept as recorded ({why})")
    read_only = set(allowed)
    seen: dict[tuple[str, str, str], str | None] = {}
    out: list[Any] = []
    for call in calls:
        if call.is_error or mcp_tool_id(call.server, call.tool) not in read_only:
            out.append(call)
            continue
        key = (call.server, call.tool, json.dumps(call.input, sort_keys=True, default=str))
        if key not in seen:
            try:
                value = await call_mcp_tool(call.server, None, call.tool, dict(call.input))
                seen[key] = json.dumps(value, default=str)
            except Exception as e:  # noqa: BLE001 (a failed replay keeps the recording)
                log(
                    f"recorded results: {call.server}.{call.tool} kept as recorded "
                    f"({type(e).__name__})"
                )
                seen[key] = None
        replayed = seen[key]
        out.append(call if replayed is None else replace(call, result=replayed))
    changed = sum(1 for a, b in zip(calls, out, strict=True) if a is not b)
    if changed:
        log(
            f"recorded results: {changed} read-only call(s) replayed through rote's client, "
            "so the compile sees the shape a compiled step receives"
        )
    return tuple(out)
