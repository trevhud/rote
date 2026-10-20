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
        {"type": "web_search_20250305", "name": "web_search", "max_uses": 5},
        {
            "type": "web_fetch_20250910",
            "name": "web_fetch",
            "max_uses": 10,
            "max_content_tokens": 32000,
        },
    )


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "web_search_20260318", "name": "web_search"},
        {"type": "web_fetch_20260318", "name": "web_fetch"},
        {"type": "code_execution_20260120", "name": "code_execution"},
        # The pinned type with its caps removed is not the pinned tool.
        {"type": "web_search_20250305", "name": "web_search"},
    ],
)
def test_the_driver_refuses_any_server_tool_but_the_pinned_ones(schema: dict[str, Any]) -> None:
    from rote.compiler.drivers import DriverError
    from rote.compiler.drivers.anthropic_api import _check_server_tools

    _check_server_tools([{"name": "read_file", "input_schema": {"type": "object"}}])
    _check_server_tools([dict(t) for t in WEB_TOOL_SCHEMAS])
    with pytest.raises(DriverError, match="refusing to send server tool"):
        _check_server_tools([schema])


def _write_turn(work_dir: Path) -> Any:
    return _tool_use(
        "w", "write_file", {"path": str(work_dir / "pipeline.yaml"), "content": VALID_PIPELINE_YAML}
    )


@pytest.mark.asyncio
async def test_an_unrequested_server_tool_stops_the_run(
    fake_skills: tuple[Path, Path, Path],  # noqa: F811
    fake_anthropic: Any,  # noqa: F811
) -> None:
    from rote.compiler.drivers import DriverError

    skill_dir, compiler_dir, work_dir = fake_skills
    work_dir.mkdir()
    client = fake_anthropic(
        [
            _msg(
                [_server_tool_use("code_execution", {"code": "print(1)"})], stop_reason="pause_turn"
            ),
            _msg([_write_turn(work_dir)], stop_reason="tool_use"),
        ]
    )
    with pytest.raises(DriverError, match="'code_execution', which the api driver did not ask for"):
        await AnthropicApiDriver(web_tools=True).run(skill_dir, compiler_dir, work_dir)
    assert len(client.messages.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(("billed", "stops"), [(400_001, True), (400_000, False)])
async def test_a_request_over_the_input_ceiling_stops_the_run(
    fake_skills: tuple[Path, Path, Path],  # noqa: F811
    fake_anthropic: Any,  # noqa: F811
    billed: int,
    stops: bool,
) -> None:
    """Cache buckets count: the ceiling is on what the request billed."""
    from rote.compiler.drivers import DriverError

    skill_dir, compiler_dir, work_dir = fake_skills
    work_dir.mkdir()
    heavy = _msg([_write_turn(work_dir)], stop_reason="tool_use", input_tokens=1)
    heavy.usage.cache_read_input_tokens = billed - 1
    client = fake_anthropic([heavy, _msg([_text("done")], stop_reason="end_turn")])
    driver = AnthropicApiDriver(web_tools=True)
    if stops:
        with pytest.raises(DriverError, match=f"billed {billed} input tokens"):
            await driver.run(skill_dir, compiler_dir, work_dir)
        assert len(client.messages.calls) == 1
    else:
        await driver.run(skill_dir, compiler_dir, work_dir)
        assert len(client.messages.calls) == 2


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


@pytest.mark.asyncio
async def test_run_checks_the_tool_list_before_the_first_request(
    fake_skills: tuple[Path, Path, Path],  # noqa: F811
    fake_anthropic: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whatever assembles the tool list, an unpinned server tool never leaves."""
    from rote.compiler.drivers import DriverError, anthropic_api

    skill_dir, compiler_dir, work_dir = fake_skills
    real = anthropic_api.anthropic_tool_schemas
    monkeypatch.setattr(
        anthropic_api,
        "anthropic_tool_schemas",
        lambda: [*real(), {"type": "web_search_20260318", "name": "web_search"}],
    )
    client = fake_anthropic([])
    with pytest.raises(DriverError, match="refusing to send server tool"):
        await AnthropicApiDriver().run(skill_dir, compiler_dir, work_dir)
    assert client.messages.calls == []


# ───────── Pricing and error size ─────────


def test_a_web_search_is_priced_at_the_documented_rate() -> None:
    """ "$10 per 1,000 searches" (web search tool page, checked 2026-10-20)."""
    from rote.compiler.drivers.anthropic_api import WEB_SEARCH_USD_PER_REQUEST

    assert pytest.approx(0.01) == WEB_SEARCH_USD_PER_REQUEST


@pytest.mark.parametrize(("searches", "reported"), [(30, None), (0, None), (30, 2.0)])
def test_the_summary_prices_searches_on_top_of_tokens(
    tmp_path: Path, searches: int, reported: float | None
) -> None:
    from rote.cli import _JsonlProgressSink, _Rates

    path = tmp_path / "p.ndjson"
    sink = _JsonlProgressSink.__new__(_JsonlProgressSink)
    sink._path = path  # type: ignore[attr-defined]
    sink._model_id = "m"  # type: ignore[attr-defined]
    sink._prices = _Rates(input_per_mtok=3.0, output_per_mtok=15.0)  # type: ignore[attr-defined]
    sink._file = path.open("w", encoding="utf-8")  # type: ignore[attr-defined]
    digest = {"type": "summary", "total_tokens": {"input": 1000, "output": 500}}
    sink.write_summary(digest, reported_cost=reported, web_searches=searches)
    sink.close()
    written = json.loads(path.read_text(encoding="utf-8").strip())
    if reported is not None:
        # The runtime's own estimate already covers what it billed.
        assert written["cost_usd"] == reported and "web_search_usd" not in written
    else:
        assert written["cost_usd"] == pytest.approx(0.0105 + searches * 0.01)
        assert written.get("web_search_usd") == (pytest.approx(0.3) if searches else None)


@pytest.mark.asyncio
async def test_a_failed_codex_run_carries_the_tail_of_its_event_stream(
    fake_skills: tuple[Path, Path, Path],  # noqa: F811
    fake_codex_subprocess: Any,  # noqa: F811
) -> None:
    from rote.compiler.drivers import DriverError
    from rote.compiler.drivers.codex import DETAILS_MAX_CHARS

    skill_dir, compiler_dir, work_dir = fake_skills
    stream = "x" * 50_000 + "THE LAST LINE SAYS WHY"
    fake_codex_subprocess(stdout_text=stream, returncode=1)
    with pytest.raises(DriverError) as caught:
        await CodexDriver(web_tools=True).run(skill_dir, compiler_dir, work_dir)
    details = caught.value.details or ""
    assert details.endswith("THE LAST LINE SAYS WHY")
    assert len(details) < DETAILS_MAX_CHARS + 100
    assert "earlier characters cut" in details


def test_compile_hands_the_drivers_search_count_to_the_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    from rote.cli import main
    from tests.test_cli import _install_mock_compiler, _patch_pricing, _read_ndjson

    _patch_pricing(monkeypatch, (3.0, 15.0))
    _install_mock_compiler(
        monkeypatch,
        driver_metadata={"input_tokens": 1000, "output_tokens": 500, "web_search_requests": 7},
    )
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: t\n---\n", encoding="utf-8")
    out = tmp_path / "out"
    assert (
        main(
            ["compile", str(skill), "--local", "--runtime", "dbos", "--no-eval", "--out", str(out)]
        )
        == 0
    )
    capsys.readouterr()
    [summary] = [o for o in _read_ndjson(out / "progress.jsonl") if o.get("type") == "summary"]
    assert summary["web_search_usd"] == pytest.approx(0.07)
