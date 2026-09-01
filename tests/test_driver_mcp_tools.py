"""Tests for live MCP tools in the in-process compiler drivers.

Two layers:

* ``rote.mcp.live_tools`` unit behavior — the shared readOnlyHint gate,
  tool ids, result flattening, the connect-per-call tool manager
  (fastmcp's ``Client`` mocked; no network), and the CLI's registry
  resolution.
* Driver integration — the api / openai-api drivers expose only
  read-only tools (adapted to each wire shape with the server's input
  schema passed through), dispatch each call over a fresh connection,
  report tool failures as error tool results, warn and continue when a
  server is unreachable, and fail loudly when the ``mcp`` extra is
  missing but servers were requested.

The fake-SDK builders are imported from the two driver test modules so
the canned-response shape cannot drift between suites.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from rote.compiler.drivers import DriverError
from rote.compiler.drivers.anthropic_api import AnthropicApiDriver
from rote.compiler.drivers.openai_api import OpenAIApiDriver
from rote.mcp.live_tools import (
    LiveMcpTools,
    mcp_tool_id,
    read_only_tools,
    registry_server_specs,
    result_text,
)
from tests.test_anthropic_driver import (
    VALID_PIPELINE_YAML,
    _FakeAsyncAnthropic,
    _msg,
    _text,
    _tool_use,
)
from tests.test_openai_driver import (
    _FakeAsyncOpenAI,
    _message,
    _response,
    _tool_call,
)

# ───────── Builders ─────────


def _tool(
    name: str,
    read_only: bool | None,
    *,
    schema: dict[str, Any] | None = None,
    description: str = "",
) -> SimpleNamespace:
    """A fake MCP tool listing entry. ``read_only=None`` = no annotations."""
    return SimpleNamespace(
        name=name,
        description=description,
        inputSchema=schema or {"type": "object", "properties": {"q": {"type": "string"}}},
        annotations=None if read_only is None else SimpleNamespace(readOnlyHint=read_only),
    )


def _text_result(text: str) -> SimpleNamespace:
    return SimpleNamespace(structured_content=None, content=[SimpleNamespace(text=text)])


class _FakeMcpClient:
    """Stands in for a fastmcp ``Client``: async context manager +
    list_tools/call_tool, with canned tools and per-tool results."""

    def __init__(
        self,
        tools: list[SimpleNamespace] | None = None,
        results: dict[str, Any] | None = None,
        fail_connect: bool = False,
    ) -> None:
        self._tools = tools or []
        self._results = results or {}
        self._fail_connect = fail_connect
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.closed = False

    async def __aenter__(self) -> _FakeMcpClient:
        if self._fail_connect:
            raise ConnectionError("connection refused")
        return self

    async def __aexit__(self, *args: Any) -> None:
        self.closed = True

    async def list_tools(self) -> list[SimpleNamespace]:
        return list(self._tools)

    async def call_tool(self, name: str, args: dict[str, Any]) -> Any:
        self.calls.append((name, args))
        value = self._results.get(name)
        if isinstance(value, Exception):
            raise value
        return value


@pytest.fixture
def fake_fastmcp(monkeypatch: pytest.MonkeyPatch):  # noqa: ANN201
    """Patch ``fastmcp.Client`` to hand out fakes in construction order.

    Connect-per-call means one fake per CONNECTION: the startup listing
    consumes one client per server, and every subsequent tool call
    consumes another.
    """

    def _patch(clients: list[_FakeMcpClient]) -> list[_FakeMcpClient]:
        remaining = list(clients)

        def _client(*args: Any, **kwargs: Any) -> _FakeMcpClient:
            if not remaining:
                raise RuntimeError("fake fastmcp.Client out of canned clients")
            return remaining.pop(0)

        import fastmcp

        monkeypatch.setattr(fastmcp, "Client", _client)
        return clients

    return _patch


@pytest.fixture
def fake_skills(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A fake source skill, fake compiler skill, and (created) work dir."""
    skill_dir = tmp_path / "fake-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: fake\n---\n\n# Fake skill\n", encoding="utf-8")

    compiler_dir = tmp_path / "fake-compiler-skill"
    compiler_dir.mkdir()
    (compiler_dir / "SKILL.md").write_text("# Fake rote-compile skill\n", encoding="utf-8")
    (compiler_dir / "references").mkdir()

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    return skill_dir, compiler_dir, work_dir


def _wordbank_servers() -> list[dict[str, Any]]:
    return [{"name": "wordbank", "url": "http://localhost:9/mcp", "headers": None}]


# ───────── Shared helper units ─────────


def test_mcp_tool_id_matches_the_claude_p_convention() -> None:
    assert mcp_tool_id("wordbank", "lookup") == "mcp__wordbank__lookup"


def test_read_only_tools_requires_a_true_server_hint() -> None:
    ro = _tool("lookup", True)
    writer = _tool("delete", False)
    unannotated = _tool("mystery", None)
    assert read_only_tools([ro, writer, unannotated]) == [ro]


def test_result_text_stringifies_structured_content() -> None:
    result = SimpleNamespace(structured_content={"a": 1}, content=[SimpleNamespace(text="x")])
    assert result_text(result) == json.dumps({"a": 1})


def test_result_text_concatenates_text_parts() -> None:
    result = SimpleNamespace(
        structured_content=None,
        content=[
            SimpleNamespace(text="part one"),
            SimpleNamespace(data=b"img"),  # non-text block skipped
            SimpleNamespace(text="part two"),
        ],
    )
    assert result_text(result) == "part one\npart two"


@pytest.mark.asyncio
async def test_live_tools_expose_only_read_only_with_schema_passthrough(
    fake_fastmcp,  # noqa: ANN001
) -> None:
    schema = {
        "type": "object",
        "properties": {"word": {"type": "string"}},
        "required": ["word"],
    }
    client = _FakeMcpClient(
        [
            _tool("lookup", True, schema=schema, description="Look up a word."),
            _tool("delete_word", False),
            _tool("mystery", None),
        ]
    )
    fake_fastmcp([client])

    async with LiveMcpTools(_wordbank_servers()) as live:
        # The listing session was closed at startup, not held open.
        assert client.closed is True
        anthropic_schemas = live.anthropic_tool_schemas()
        openai_schemas = live.openai_tool_schemas()
        assert live.owns("mcp__wordbank__lookup")
        assert not live.owns("mcp__wordbank__delete_word")
        assert not live.owns("read_file")

    # Anthropic shape: name/description/input_schema, schema verbatim.
    assert anthropic_schemas == [
        {
            "name": "mcp__wordbank__lookup",
            "description": "Look up a word.",
            "input_schema": schema,
        }
    ]
    # OpenAI shape: function wrapper with `parameters`.
    assert openai_schemas == [
        {
            "type": "function",
            "function": {
                "name": "mcp__wordbank__lookup",
                "description": "Look up a word.",
                "parameters": schema,
            },
        }
    ]


@pytest.mark.asyncio
async def test_live_tools_open_a_fresh_connection_per_call(
    fake_fastmcp,  # noqa: ANN001
) -> None:
    """Nothing is held open between calls: the startup listing session
    closes immediately, and each call() gets its own client, closed after
    the one tools/call — long-lived sessions froze the loop under the
    hosted container runtime."""
    lister = _FakeMcpClient([_tool("lookup", True)])
    call_one = _FakeMcpClient(results={"lookup": _text_result("one")})
    call_two = _FakeMcpClient(results={"lookup": _text_result("two")})
    fake_fastmcp([lister, call_one, call_two])

    async with LiveMcpTools(_wordbank_servers()) as live:
        assert lister.closed is True
        assert await live.call("mcp__wordbank__lookup", {"q": "a"}) == "one"
        assert call_one.closed is True
        assert await live.call("mcp__wordbank__lookup", {"q": "b"}) == "two"
        assert call_two.closed is True

    assert call_one.calls == [("lookup", {"q": "a"})]
    assert call_two.calls == [("lookup", {"q": "b"})]
    # The listing client never served a tools/call.
    assert lister.calls == []


@pytest.mark.asyncio
async def test_live_tools_unreachable_server_warns_and_continues(
    fake_fastmcp,  # noqa: ANN001
) -> None:
    dead = _FakeMcpClient(fail_connect=True)
    alive = _FakeMcpClient([_tool("lookup", True)])
    fake_fastmcp([dead, alive])

    warnings: list[str] = []
    servers = [
        {"name": "deadsrv", "url": "http://localhost:1/mcp", "headers": None},
        {"name": "wordbank", "url": "http://localhost:9/mcp", "headers": None},
    ]
    async with LiveMcpTools(servers, on_warning=warnings.append) as live:
        assert live.owns("mcp__wordbank__lookup")
        assert not any(name.startswith("mcp__deadsrv__") for name, _, _ in live._defs)
    assert len(warnings) == 1
    assert "deadsrv" in warnings[0]
    assert "unavailable" in warnings[0]


@pytest.mark.asyncio
async def test_live_tools_warn_when_no_read_only_tools(
    fake_fastmcp,  # noqa: ANN001
) -> None:
    client = _FakeMcpClient([_tool("delete", False)])
    fake_fastmcp([client])
    warnings: list[str] = []
    async with LiveMcpTools(_wordbank_servers(), on_warning=warnings.append) as live:
        assert live.anthropic_tool_schemas() == []
    assert len(warnings) == 1
    assert "no read-only tools" in warnings[0]


@pytest.mark.asyncio
async def test_live_tools_missing_fastmcp_names_the_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "fastmcp", None)
    with pytest.raises(RuntimeError, match=r"rote-cli\[mcp\]"):
        async with LiveMcpTools(_wordbank_servers()):
            pass


# ───────── Registry resolution (the CLI parity path) ─────────


@pytest.mark.asyncio
async def test_registry_specs_static_headers_and_skip_reasons() -> None:
    from rote.mcp import McpRegistry, McpServerConfig, save_registry

    save_registry(
        McpRegistry(
            servers={
                "keyed": McpServerConfig(url="https://k.example/mcp", headers={"X-Api-Key": "k"}),
                "loggedout": McpServerConfig(url="https://o.example/mcp"),
                "ssely": McpServerConfig(url="https://s.example/sse", transport="sse"),
            }
        )
    )
    skips: dict[str, str] = {}
    specs = await registry_server_specs(on_skip=lambda s, r: skips.__setitem__(s, r))

    assert specs == [
        {"name": "keyed", "url": "https://k.example/mcp", "headers": {"X-Api-Key": "k"}}
    ]
    assert "rote mcp login loggedout" in skips["loggedout"]
    assert "transport" in skips["ssely"]


@pytest.mark.asyncio
async def test_registry_specs_refresh_logged_in_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rote.mcp
    from rote.mcp import McpRegistry, McpServerConfig, save_registry
    from rote.mcp.tokens import write_token_file

    url = "https://o.example/mcp"
    save_registry(McpRegistry(servers={"oauthy": McpServerConfig(url=url)}))
    write_token_file(
        "oauthy",
        {
            "server_url": url,
            "tokens": {"access_token": "stale"},
            "expires_at": None,
            "client_info": None,
            "token_endpoint": None,
        },
    )

    async def _fake_fresh(server: str, config: Any) -> str:
        assert server == "oauthy"
        return "fresh-token"

    monkeypatch.setattr(rote.mcp, "fresh_access_token", _fake_fresh)
    specs = await registry_server_specs()
    assert specs == [
        {"name": "oauthy", "url": url, "headers": {"Authorization": "Bearer fresh-token"}}
    ]


@pytest.mark.asyncio
async def test_registry_specs_failed_refresh_is_a_skip_not_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rote.mcp
    from rote.mcp import McpAuthError, McpRegistry, McpServerConfig, save_registry
    from rote.mcp.tokens import write_token_file

    url = "https://o.example/mcp"
    save_registry(McpRegistry(servers={"oauthy": McpServerConfig(url=url)}))
    write_token_file(
        "oauthy",
        {
            "server_url": url,
            "tokens": {"access_token": "stale"},
            "expires_at": None,
            "client_info": None,
            "token_endpoint": None,
        },
    )

    async def _fail_fresh(server: str, config: Any) -> str:
        raise McpAuthError("refresh failed")

    monkeypatch.setattr(rote.mcp, "fresh_access_token", _fail_fresh)
    skips: dict[str, str] = {}
    specs = await registry_server_specs(on_skip=lambda s, r: skips.__setitem__(s, r))
    assert specs == []
    assert "refresh failed" in skips["oauthy"]


# ───────── Anthropic driver integration ─────────


def _lookup_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"word": {"type": "string"}},
        "required": ["word"],
    }


@pytest.mark.asyncio
async def test_anthropic_driver_exposes_and_dispatches_mcp_tools(
    fake_skills: tuple[Path, Path, Path],
    fake_fastmcp,  # noqa: ANN001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill_dir, compiler_dir, work_dir = fake_skills
    lister = _FakeMcpClient(
        [
            _tool("lookup", True, schema=_lookup_schema(), description="Look up a word."),
            _tool("delete_word", False),
        ]
    )
    caller = _FakeMcpClient(results={"lookup": _text_result("definition: by memory")})
    fake_fastmcp([lister, caller])

    responses = [
        _msg(
            [_tool_use("t1", "mcp__wordbank__lookup", {"word": "rote"})],
            stop_reason="tool_use",
        ),
        _msg(
            [
                _tool_use(
                    "t2",
                    "write_file",
                    {
                        "path": str((work_dir / "pipeline.yaml").resolve()),
                        "content": VALID_PIPELINE_YAML,
                    },
                )
            ],
            stop_reason="tool_use",
        ),
        _msg([_text("done")], stop_reason="end_turn"),
    ]
    client = _FakeAsyncAnthropic(responses)
    monkeypatch.setattr(
        "rote.compiler.drivers.anthropic_api.anthropic.AsyncAnthropic",
        lambda *a, **k: client,
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")

    driver = AnthropicApiDriver(mcp_servers=_wordbank_servers())
    result = await driver.run(skill_dir, compiler_dir, work_dir)
    assert result.pipeline_yaml_path.is_file()

    # Only the read-only tool was declared, in Anthropic shape, with the
    # server's schema passed through verbatim — beside the fs tools.
    tools_sent = client.messages.calls[0]["tools"]
    names = [t["name"] for t in tools_sent]
    assert "read_file" in names and "write_file" in names
    assert "mcp__wordbank__lookup" in names
    assert "mcp__wordbank__delete_word" not in names
    spec = next(t for t in tools_sent if t["name"] == "mcp__wordbank__lookup")
    assert spec["input_schema"] == _lookup_schema()
    assert spec["description"] == "Look up a word."

    # The call was dispatched with the model's args; the text came back
    # as a non-error tool result.
    assert caller.calls == [("lookup", {"word": "rote"})]
    tool_results = client.messages.calls[1]["messages"][-1]["content"]
    assert tool_results[0]["content"] == "definition: by memory"
    assert tool_results[0]["is_error"] is False

    # Connect-per-call: the listing session closed at startup, the call's
    # own session closed after the dispatch, and the two never mix.
    assert lister.closed is True
    assert caller.closed is True
    assert lister.calls == []


@pytest.mark.asyncio
async def test_anthropic_driver_reports_mcp_failure_as_tool_error(
    fake_skills: tuple[Path, Path, Path],
    fake_fastmcp,  # noqa: ANN001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill_dir, compiler_dir, work_dir = fake_skills
    lister = _FakeMcpClient([_tool("lookup", True)])
    caller = _FakeMcpClient(results={"lookup": RuntimeError("server exploded")})
    fake_fastmcp([lister, caller])

    responses = [
        _msg([_tool_use("t1", "mcp__wordbank__lookup", {"q": "x"})], stop_reason="tool_use"),
        _msg(
            [
                _tool_use(
                    "t2",
                    "write_file",
                    {
                        "path": str((work_dir / "pipeline.yaml").resolve()),
                        "content": VALID_PIPELINE_YAML,
                    },
                )
            ],
            stop_reason="tool_use",
        ),
        _msg([_text("done")], stop_reason="end_turn"),
    ]
    client = _FakeAsyncAnthropic(responses)
    monkeypatch.setattr(
        "rote.compiler.drivers.anthropic_api.anthropic.AsyncAnthropic",
        lambda *a, **k: client,
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")

    result = await AnthropicApiDriver(mcp_servers=_wordbank_servers()).run(
        skill_dir, compiler_dir, work_dir
    )
    assert result.pipeline_yaml_path.is_file()

    tool_results = client.messages.calls[1]["messages"][-1]["content"]
    assert tool_results[0]["is_error"] is True
    assert "server exploded" in tool_results[0]["content"]
    # Even a failed call's connection is closed.
    assert caller.closed is True


@pytest.mark.asyncio
async def test_anthropic_driver_warns_and_continues_without_dead_server(
    fake_skills: tuple[Path, Path, Path],
    fake_fastmcp,  # noqa: ANN001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill_dir, compiler_dir, work_dir = fake_skills
    fake_fastmcp([_FakeMcpClient(fail_connect=True)])

    responses = [
        _msg(
            [
                _tool_use(
                    "t1",
                    "write_file",
                    {
                        "path": str((work_dir / "pipeline.yaml").resolve()),
                        "content": VALID_PIPELINE_YAML,
                    },
                )
            ],
            stop_reason="tool_use",
        ),
        _msg([_text("done")], stop_reason="end_turn"),
    ]
    client = _FakeAsyncAnthropic(responses)
    monkeypatch.setattr(
        "rote.compiler.drivers.anthropic_api.anthropic.AsyncAnthropic",
        lambda *a, **k: client,
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")

    events: list[Any] = []
    result = await AnthropicApiDriver(mcp_servers=_wordbank_servers()).run(
        skill_dir, compiler_dir, work_dir, on_event=events.append
    )
    assert result.pipeline_yaml_path.is_file()

    warnings = [e for e in events if e.type == "warning"]
    assert any("wordbank" in w.message and "unavailable" in w.message for w in warnings)
    # No MCP tools were declared to the model.
    assert all(not t["name"].startswith("mcp__") for t in client.messages.calls[0]["tools"])


@pytest.mark.asyncio
async def test_anthropic_driver_missing_mcp_extra_is_a_driver_error(
    fake_skills: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """mcp_servers requested but the extra is absent: fail loudly (a
    compile silently missing its tools would be wrong, not degraded)."""
    skill_dir, compiler_dir, work_dir = fake_skills
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    monkeypatch.setitem(sys.modules, "fastmcp", None)

    driver = AnthropicApiDriver(mcp_servers=_wordbank_servers())
    with pytest.raises(DriverError, match=r"rote-cli\[mcp\]"):
        await driver.run(skill_dir, compiler_dir, work_dir)


@pytest.mark.asyncio
async def test_anthropic_driver_stringifies_structured_content(
    fake_skills: tuple[Path, Path, Path],
    fake_fastmcp,  # noqa: ANN001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill_dir, compiler_dir, work_dir = fake_skills
    lister = _FakeMcpClient([_tool("lookup", True)])
    caller = _FakeMcpClient(
        results={
            "lookup": SimpleNamespace(structured_content={"definition": "by memory"}, content=[])
        }
    )
    fake_fastmcp([lister, caller])

    responses = [
        _msg([_tool_use("t1", "mcp__wordbank__lookup", {"q": "x"})], stop_reason="tool_use"),
        _msg(
            [
                _tool_use(
                    "t2",
                    "write_file",
                    {
                        "path": str((work_dir / "pipeline.yaml").resolve()),
                        "content": VALID_PIPELINE_YAML,
                    },
                )
            ],
            stop_reason="tool_use",
        ),
        _msg([_text("done")], stop_reason="end_turn"),
    ]
    client = _FakeAsyncAnthropic(responses)
    monkeypatch.setattr(
        "rote.compiler.drivers.anthropic_api.anthropic.AsyncAnthropic",
        lambda *a, **k: client,
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")

    await AnthropicApiDriver(mcp_servers=_wordbank_servers()).run(skill_dir, compiler_dir, work_dir)
    tool_results = client.messages.calls[1]["messages"][-1]["content"]
    assert tool_results[0]["content"] == json.dumps({"definition": "by memory"})


# ───────── OpenAI driver integration ─────────


@pytest.mark.asyncio
async def test_openai_driver_exposes_and_dispatches_mcp_tools(
    fake_skills: tuple[Path, Path, Path],
    fake_fastmcp,  # noqa: ANN001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill_dir, compiler_dir, work_dir = fake_skills
    lister = _FakeMcpClient(
        [
            _tool("lookup", True, schema=_lookup_schema(), description="Look up a word."),
            _tool("delete_word", False),
        ]
    )
    caller = _FakeMcpClient(results={"lookup": _text_result("definition: by memory")})
    fake_fastmcp([lister, caller])

    responses = [
        _response(
            _message(
                tool_calls=[_tool_call("c1", "mcp__wordbank__lookup", json.dumps({"word": "rote"}))]
            )
        ),
        _response(
            _message(
                tool_calls=[
                    _tool_call(
                        "c2",
                        "write_file",
                        json.dumps(
                            {
                                "path": str((work_dir / "pipeline.yaml").resolve()),
                                "content": VALID_PIPELINE_YAML,
                            }
                        ),
                    )
                ]
            )
        ),
        _response(_message(content="Done.")),
    ]
    client = _FakeAsyncOpenAI(responses)
    monkeypatch.setattr(
        "rote.compiler.drivers.openai_api.openai.AsyncOpenAI",
        lambda *a, **k: client,
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")

    driver = OpenAIApiDriver(model="openai/gpt-5.5", mcp_servers=_wordbank_servers())
    result = await driver.run(skill_dir, compiler_dir, work_dir)
    assert result.pipeline_yaml_path.is_file()

    # Function-calling shape, read-only gate, schema passthrough.
    tools_sent = client.chat.completions.calls[0]["tools"]
    names = [t["function"]["name"] for t in tools_sent]
    assert "mcp__wordbank__lookup" in names
    assert "mcp__wordbank__delete_word" not in names
    spec = next(t for t in tools_sent if t["function"]["name"] == "mcp__wordbank__lookup")
    assert spec["type"] == "function"
    assert spec["function"]["parameters"] == _lookup_schema()

    # Dispatch + text result as a role:tool message, over a fresh
    # connection that closed after the one call.
    assert caller.calls == [("lookup", {"word": "rote"})]
    tool_msgs = [m for m in client.chat.completions.calls[1]["messages"] if m["role"] == "tool"]
    assert tool_msgs[0]["tool_call_id"] == "c1"
    assert tool_msgs[0]["content"] == "definition: by memory"
    assert lister.closed is True
    assert caller.closed is True


@pytest.mark.asyncio
async def test_openai_driver_reports_mcp_failure_as_tool_error(
    fake_skills: tuple[Path, Path, Path],
    fake_fastmcp,  # noqa: ANN001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill_dir, compiler_dir, work_dir = fake_skills
    lister = _FakeMcpClient([_tool("lookup", True)])
    caller = _FakeMcpClient(results={"lookup": RuntimeError("server exploded")})
    fake_fastmcp([lister, caller])

    responses = [
        _response(_message(tool_calls=[_tool_call("c1", "mcp__wordbank__lookup", json.dumps({}))])),
        _response(
            _message(
                tool_calls=[
                    _tool_call(
                        "c2",
                        "write_file",
                        json.dumps(
                            {
                                "path": str((work_dir / "pipeline.yaml").resolve()),
                                "content": VALID_PIPELINE_YAML,
                            }
                        ),
                    )
                ]
            )
        ),
        _response(_message(content="Done.")),
    ]
    client = _FakeAsyncOpenAI(responses)
    monkeypatch.setattr(
        "rote.compiler.drivers.openai_api.openai.AsyncOpenAI",
        lambda *a, **k: client,
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")

    result = await OpenAIApiDriver(model="openai/gpt-5.5", mcp_servers=_wordbank_servers()).run(
        skill_dir, compiler_dir, work_dir
    )
    assert result.pipeline_yaml_path.is_file()

    tool_msgs = [m for m in client.chat.completions.calls[1]["messages"] if m["role"] == "tool"]
    assert "server exploded" in tool_msgs[0]["content"]
