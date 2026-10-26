"""A recorded result takes the shape rote's own client returns.

Atlassian's getConfluencePage returns ``{"content": {"nodes": [page]}}`` as
text and the page itself as its structured result; a compiled step gets
the structured one. These pin the replay that makes the compile see it.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from rote.compiler.replay import runtime_shapes
from rote.eval.baseline import ObservedToolCall
from rote.probe import parse_tool_result

TEXT = [{"type": "text", "text": json.dumps({"content": {"nodes": [{"id": "1"}]}})}]


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str, dict[str, Any]]]:
    """atlassian marks getConfluencePage read-only and createJiraIssue not;
    each call through rote's client returns the bare page."""
    sent: list[tuple[str, str, dict[str, Any]]] = []

    async def _allowlist(names: list[str]) -> tuple[list[str], list[str], dict[str, str]]:
        return (
            ["mcp__atlassian__getConfluencePage"],
            ["mcp__atlassian__createJiraIssue"],
            {},
        )

    async def _call(server: str, url: Any, tool: str, args: dict[str, Any]) -> Any:
        sent.append((server, tool, args))
        if args.get("pageId") == "broken":
            raise RuntimeError("server error")
        return {"id": args["pageId"], "title": "AI traces"}

    monkeypatch.setattr("rote.eval.baseline.read_only_allowlist", _allowlist)
    monkeypatch.setattr("rote.mcp._runtime_helper.call_mcp_tool", _call)
    return sent


def _call(tool: str, page: str, **extra: Any) -> ObservedToolCall:
    return ObservedToolCall("atlassian", tool, {"pageId": page}, TEXT, **extra)


def test_a_read_only_call_takes_the_shape_rotes_client_returns(server: Any) -> None:
    (replayed,) = asyncio.run(runtime_shapes([_call("getConfluencePage", "5208211471")]))
    assert parse_tool_result(replayed.result) == {"id": "5208211471", "title": "AI traces"}
    assert server == [("atlassian", "getConfluencePage", {"pageId": "5208211471"})]


def test_a_write_tool_is_never_called_and_keeps_its_recording(server: Any) -> None:
    write = _call("createJiraIssue", "x")
    assert asyncio.run(runtime_shapes([write])) == (write,)
    assert server == []


def test_a_failed_call_or_a_failed_replay_keeps_its_recording(server: Any) -> None:
    failed = _call("getConfluencePage", "1", is_error=True)
    broken = _call("getConfluencePage", "broken")
    assert asyncio.run(runtime_shapes([failed, broken])) == (failed, broken)
    assert [args for _s, _t, args in server] == [{"pageId": "broken"}]


def test_the_same_call_recorded_twice_is_replayed_once(server: Any) -> None:
    lines: list[str] = []
    out = asyncio.run(
        runtime_shapes(
            [_call("getConfluencePage", "7"), _call("getConfluencePage", "7")], lines.append
        )
    )
    assert len(server) == 1
    assert all(parse_tool_result(c.result) == {"id": "7", "title": "AI traces"} for c in out)
    assert any("2 read-only call(s) replayed" in line for line in lines)
