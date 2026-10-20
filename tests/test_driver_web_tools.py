"""``web_tools=True`` (``--backend api``) on the drivers other than claude.

The claude driver's WebSearch/WebFetch are covered in test_claude_driver.
Here: the Messages API's server-side web_search / web_fetch (tool types,
reported lookups, ``pause_turn``, usage), Codex's live web search (the
config override, and the searches and page opens read off ``--json``),
and the openai-api driver, which has no web tool and says so.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from rote.compiler.drivers.anthropic_api import WEB_TOOL_SCHEMAS, AnthropicApiDriver
from rote.compiler.drivers.codex import WEB_SEARCH_CONFIG, CodexDriver, web_events
from rote.compiler.events import CompilationEvent
from tests.test_anthropic_driver import (
    VALID_PIPELINE_YAML,
    _msg,
    _text,
    _tool_use,
    fake_anthropic,  # noqa: F401 (fixture)
    fake_skills,  # noqa: F401 (fixture)
)
from tests.test_codex_driver import fake_codex_subprocess  # noqa: F401 (fixture)

GMAIL = "https://developers.google.com/workspace/gmail/api/reference/rest/v1/users.threads/list"

# ───────── Anthropic Messages API ─────────


def _server_tool_use(name: str, input_: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(type="server_tool_use", id=f"srvtoolu_{name}", name=name, input=input_)


def _usage_msg(content: list[Any], *, stop_reason: str, searches: int, fetches: int) -> Any:
    msg = _msg(content, stop_reason=stop_reason)
    msg.usage.server_tool_use = SimpleNamespace(
        web_search_requests=searches, web_fetch_requests=fetches
    )
    return msg


def test_the_documented_server_tool_versions() -> None:
    """Exact strings from the web search / web fetch tool docs. The basic
    versions: 20260209 and later provision code execution, which a real
    compile used as a second workspace at ~$14 of extra input tokens. A
    typo is a 400 on every api compile."""
    assert WEB_TOOL_SCHEMAS == (
        {"type": "web_search_20250305", "name": "web_search"},
        {"type": "web_fetch_20250910", "name": "web_fetch", "max_content_tokens": 32000},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("web_tools", [True, False])
async def test_api_driver_sends_server_web_tools_only_when_asked(
    fake_skills: tuple[Path, Path, Path],  # noqa: F811
    fake_anthropic: Any,  # noqa: F811
    web_tools: bool,
) -> None:
    skill_dir, compiler_dir, work_dir = fake_skills
    work_dir.mkdir()
    client = fake_anthropic(
        [
            _msg(
                [
                    _tool_use(
                        "w",
                        "write_file",
                        {"path": str(work_dir / "pipeline.yaml"), "content": VALID_PIPELINE_YAML},
                    )
                ],
                stop_reason="tool_use",
            ),
            _msg([_text("done")], stop_reason="end_turn"),
        ]
    )
    result = await AnthropicApiDriver(web_tools=web_tools).run(skill_dir, compiler_dir, work_dir)
    types = [t.get("type") for t in client.messages.calls[0]["tools"]]
    server = [t for t in types if t and t.startswith("web_")]
    assert server == (["web_search_20250305", "web_fetch_20250910"] if web_tools else [])
    assert ("web_search_requests" in result.metadata) is web_tools


@pytest.mark.asyncio
async def test_api_driver_reports_lookups_and_resumes_a_paused_turn(
    fake_skills: tuple[Path, Path, Path],  # noqa: F811
    fake_anthropic: Any,  # noqa: F811
) -> None:
    """Server tool blocks become tool events (the report's evidence), and a
    ``pause_turn`` is resent as-is: no user turn after it, not a stop."""
    skill_dir, compiler_dir, work_dir = fake_skills
    work_dir.mkdir()
    paused_content = [
        _server_tool_use("web_search", {"query": "gmail api threads list"}),
        _server_tool_use("web_fetch", {"url": GMAIL}),
    ]
    client = fake_anthropic(
        [
            _usage_msg(paused_content, stop_reason="pause_turn", searches=1, fetches=1),
            _usage_msg(
                [
                    _tool_use(
                        "w",
                        "write_file",
                        {"path": str(work_dir / "pipeline.yaml"), "content": VALID_PIPELINE_YAML},
                    )
                ],
                stop_reason="tool_use",
                searches=2,
                fetches=0,
            ),
            _msg([_text("done")], stop_reason="end_turn"),
        ]
    )
    events: list[CompilationEvent] = []
    result = await AnthropicApiDriver(web_tools=True).run(
        skill_dir, compiler_dir, work_dir, on_event=events.append
    )

    tools = [(e.tool_name, e.url, e.query) for e in events if e.type == "tool"]
    assert tools[:2] == [
        ("web_search", None, "gmail api threads list"),
        ("web_fetch", GMAIL, None),
    ]
    # The second request ends with the paused assistant turn, unchanged.
    resumed = client.messages.calls[1]["messages"]
    assert resumed[-1] == {"role": "assistant", "content": paused_content}
    assert len(client.messages.calls) == 3
    assert (result.metadata["web_search_requests"], result.metadata["web_fetch_requests"]) == (
        3,
        1,
    )


# ───────── Codex ─────────

#: Trimmed from a real `codex exec --json -c web_search="live"` run
#: (codex-cli 0.163.0): one search, then a page open, which Codex reports
#: as action type "other" with the opened page under a `turnNviewM` ref.
CODEX_STREAM = (
    "\n".join(
        json.dumps(e)
        for e in [
            {"type": "thread.started", "thread_id": "t"},
            {
                "type": "item.started",
                "item": {"type": "web_search", "query": "", "action": {"type": "other"}},
            },
            {
                "type": "item.completed",
                "item": {
                    "type": "web_search",
                    "query": "gmail api users.threads.list",
                    "action": {"type": "search", "query": "gmail api users.threads.list"},
                    "results": [{"type": "text_result", "ref_id": "turn0search0", "url": GMAIL}],
                },
            },
            {
                "type": "item.completed",
                "item": {
                    "type": "web_search",
                    "query": "",
                    "action": {"type": "other"},
                    "results": [
                        {"type": "text_result", "ref_id": "turn1view0", "url": GMAIL},
                        {"type": "text_result", "ref_id": "turn1view1", "url": GMAIL + "s"},
                        {
                            "type": "text_result",
                            "ref_id": "turn1search3",
                            "url": "https://hit.example",
                        },
                    ],
                },
            },
            {
                "type": "item.completed",
                "item": {
                    "type": "web_search",
                    "action": {"type": "open_page", "url": "https://opened.example"},
                },
            },
            {"type": "item.completed", "item": {"type": "agent_message", "text": "done"}},
        ]
    )
    + "\nnot json\n[1, 2]\n"
)


def test_codex_web_events_reads_searches_and_page_opens() -> None:
    got = [(e.tool_name, e.url, e.query) for e in web_events(CODEX_STREAM)]
    assert got == [
        ("web_search", None, "gmail api users.threads.list"),
        ("web_fetch", GMAIL, None),
        ("web_fetch", GMAIL + "s", None),
        ("web_fetch", "https://opened.example", None),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("web_tools", [True, False])
async def test_codex_enables_live_search_and_reports_it_only_when_asked(
    fake_skills: tuple[Path, Path, Path],  # noqa: F811
    fake_codex_subprocess: Any,  # noqa: F811
    web_tools: bool,
) -> None:
    skill_dir, compiler_dir, work_dir = fake_skills
    recorder = fake_codex_subprocess(
        stdout_text=CODEX_STREAM, write_files={"pipeline.yaml": VALID_PIPELINE_YAML}
    )
    events: list[CompilationEvent] = []
    await CodexDriver(web_tools=web_tools).run(
        skill_dir, compiler_dir, work_dir, on_event=events.append
    )
    args = list(recorder.calls[0]["args"])
    if web_tools:
        assert args[args.index("-c") + 1] == WEB_SEARCH_CONFIG == 'web_search="live"'
        assert "--json" in args
        assert any(e.tool_name == "web_fetch" and e.url == GMAIL for e in events)
    else:
        assert "-c" not in args and "--json" not in args
        assert not [e for e in events if e.type == "tool"]


# ───────── openai-api ─────────


def test_openai_driver_refuses_web_tools_instead_of_compiling_from_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rote.compiler.drivers.openai_api import OpenAIApiDriver

    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    available, reason = OpenAIApiDriver(model="m", web_tools=True).is_available()
    assert not available and "--agent claude, codex or api" in reason
    assert OpenAIApiDriver(model="m").is_available() == (True, "")
