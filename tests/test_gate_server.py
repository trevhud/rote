"""ask_person: the person at a skill's gates, as a stdio MCP tool.

Run through a real MCP client (fastmcp over stdio), as the agent's client
launches it: the i-th ask gets the i-th answer, an ask past the end is an
error, and every ask lands in the call log in order.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from rote.eval.empirical import gate_answers_server
from rote.eval.gate_server import GATE_SERVER, GATE_TOOL

fastmcp = pytest.importorskip("fastmcp")


def test_each_ask_gets_the_next_answer_and_lands_in_the_log(tmp_path: Path) -> None:
    from fastmcp import Client
    from fastmcp.client.transports import StdioTransport

    answers = [{"approved": True, "reviewer": "Dana Lee"}, {"n": 2}]
    entry, tool_id = gate_answers_server(answers, tmp_path)
    assert (entry["type"], entry["command"], tool_id) == (
        "stdio",
        sys.executable,
        f"mcp__{GATE_SERVER}__{GATE_TOOL}",
    )

    async def session() -> tuple[list[object], list[object]]:
        transport = StdioTransport(command=entry["command"], args=entry["args"])
        async with Client(transport) as client:
            tools = await client.list_tools()
            results = [
                await client.call_tool(GATE_TOOL, {"question": f"ask {i}"}, raise_on_error=False)
                for i in range(3)
            ]
        return tools, results

    tools, results = asyncio.run(session())
    (tool,) = tools
    assert tool.name == GATE_TOOL  # type: ignore[attr-defined]
    assert tool.annotations.read_only_hint is True  # type: ignore[attr-defined]
    texts = [r.content[0].text for r in results]  # type: ignore[attr-defined]
    assert [json.loads(t) for t in texts[:2]] == answers
    assert [r.is_error for r in results] == [False, False, True]  # type: ignore[attr-defined]
    log = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert [(e["server"], e["gate"], e["matched"]) for e in log] == [
        (GATE_SERVER, 0, True),
        (GATE_SERVER, 1, True),
        (GATE_SERVER, 2, False),
    ]
    assert log[0]["arguments"] == {"question": "ask 0"}
