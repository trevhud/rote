"""``web_tools=True`` (``--backend api``) on the drivers other than claude.

The claude driver's WebSearch/WebFetch are covered in test_claude_driver.
Here: the Messages API's server-side web_search / web_fetch (tool types,
reported lookups, ``pause_turn``, usage), Codex's live web search (the
config override, and the searches and page opens read off ``--json``),
and the openai-api driver, which has no web tool and says so.
"""

from __future__ import annotations

import json
import time
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


#: Trimmed from https://openrouter.ai/api/v1/models (2026-10-20): native
#: search on an Anthropic model is passed through at Anthropic's price.
_OPENROUTER = {
    "data": [
        {
            "id": "anthropic/claude-sonnet-4.6",
            "pricing": {"prompt": "0.000003", "completion": "0.000015", "web_search": "0.01"},
        },
        {
            "id": "anthropic/claude-sonnet-4.6:batch",
            "pricing": {"prompt": "0.0000015", "completion": "0.0000075", "web_search": "0.02"},
        },
        {"id": "anthropic/claude-haiku-5.5", "pricing": {"prompt": "0.0000001", "completion": "0"}},
    ]
}


def _catalog(web_search: dict[str, float]) -> Any:
    from rote.eval.pricing import PricingCatalog

    return PricingCatalog(
        prices=(),
        reference_prices={"claude-sonnet-4-6": (3.0, 15.0)},
        reference_cache_rates={"claude-sonnet-4-6": (0.3, 3.75)},
        web_search_rates=web_search,
    )


def test_the_web_search_price_is_read_live_from_openrouter() -> None:
    from rote.eval.pricing import PricingError, _parse_openrouter_web_search

    rates = _parse_openrouter_web_search(_OPENROUTER)
    # The base model's listing wins over its :batch variant.
    assert rates == {"claude-sonnet-4-6": 0.01}
    catalog = _catalog(rates)
    assert catalog.web_search_usd_per_request("anthropic/claude-sonnet-4.6") == 0.01
    with pytest.raises(PricingError, match="no live web search price"):
        catalog.web_search_usd_per_request("claude-haiku-5-5")


def test_the_web_search_price_survives_the_disk_cache(tmp_path: Path) -> None:
    from dataclasses import replace

    from rote.eval.pricing import ModelPrice, ModelTier, _load_cache, _store_cache

    # _load_cache refuses a catalog with no tier representative.
    price = ModelPrice(
        model_id="claude-sonnet-4-6",
        provider="anthropic",
        display_name="Sonnet 4.6",
        tier=ModelTier.MID,
        input_per_mtok=3.0,
        output_per_mtok=15.0,
        cache_read_per_mtok=0.3,
        cache_write_per_mtok=3.75,
        source="t",
        fetched_at="t",
    )
    catalog = replace(_catalog({"claude-sonnet-4-6": 0.01}), prices=(price,))
    path = tmp_path / "pricing.json"
    _store_cache(path, catalog, "anthropic")
    loaded = _load_cache(path, 3600, "anthropic")
    assert loaded is not None and loaded.web_search_rates == {"claude-sonnet-4-6": 0.01}


@pytest.mark.asyncio
@pytest.mark.parametrize("priced", [True, False])
async def test_searches_are_priced_into_the_runs_cost_or_the_cost_is_unknown(
    tmp_path: Path,
    fake_skills: tuple[Path, Path, Path],  # noqa: F811
    fake_anthropic: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    priced: bool,
) -> None:
    from rote.compiler import _write_compile_metrics

    catalog = _catalog({"claude-sonnet-4-6": 0.01} if priced else {})
    monkeypatch.setattr("rote.eval.pricing.fetch_catalog", lambda provider: catalog)
    skill_dir, compiler_dir, work_dir = fake_skills
    work_dir.mkdir()
    fake_anthropic(
        [
            _usage_msg(
                [
                    _tool_use(
                        "w",
                        "write_file",
                        {"path": str(work_dir / "pipeline.yaml"), "content": VALID_PIPELINE_YAML},
                    )
                ],
                stop_reason="tool_use",
                searches=7,
                fetches=2,
            ),
            _msg([_text("done")], stop_reason="end_turn"),
        ]
    )
    result = await AnthropicApiDriver(model="claude-sonnet-4-6", web_tools=True).run(
        skill_dir, compiler_dir, work_dir
    )
    meta = result.metadata
    tokens = (
        meta["input_tokens"] * 3.0
        + meta["output_tokens"] * 15.0
        + meta["cache_read_tokens"] * 0.3
        + meta["cache_write_tokens"] * 3.75
    ) / 1e6
    if priced:
        assert meta["web_search_usd"] == pytest.approx(0.07)
        assert meta["cost_usd"] == pytest.approx(tokens + 0.07)
    else:
        # A search with no live price is not billed at a remembered rate.
        assert meta["web_search_usd"] is None and meta["cost_usd"] is None
    out = tmp_path / "compiled"
    out.mkdir()
    _write_compile_metrics(result, out)
    written = json.loads((out / "compile-metrics.json").read_text())["metadata"]
    assert written["web_search_requests"] == 7 and written["web_fetch_requests"] == 2
    assert written.get("web_search_usd") == (pytest.approx(0.07) if priced else None)


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


def test_a_cache_from_before_web_search_rates_is_refetched(tmp_path: Path) -> None:
    from rote.eval.pricing import _load_cache

    path = tmp_path / "pricing.json"
    # Inside its TTL, otherwise valid, but written before the key existed.
    path.write_text(
        json.dumps(
            {
                "cached_at_epoch": time.time(),
                "provider": "anthropic",
                "prices": [
                    {
                        "model_id": "claude-sonnet-4-6",
                        "provider": "anthropic",
                        "display_name": "Sonnet 4.6",
                        "tier": "mid",
                        "input_per_mtok": 3.0,
                        "output_per_mtok": 15.0,
                        "cache_read_per_mtok": 0.3,
                        "cache_write_per_mtok": 3.75,
                        "source": "t",
                        "fetched_at": "t",
                    }
                ],
                "reference_prices": {"claude-sonnet-4-6": [3.0, 15.0]},
                "reference_cache_rates": {"claude-sonnet-4-6": [0.3, 3.75]},
            }
        ),
        encoding="utf-8",
    )
    assert _load_cache(path, 3600, "anthropic") is None
    data = json.loads(path.read_text(encoding="utf-8"))
    data["web_search_rates"] = {"claude-sonnet-4-6": 0.01}
    path.write_text(json.dumps(data), encoding="utf-8")
    loaded = _load_cache(path, 3600, "anthropic")
    assert loaded is not None and loaded.web_search_rates == {"claude-sonnet-4-6": 0.01}
