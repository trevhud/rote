"""A read-only run hands an agent loop only the MCP tools that cannot write.

An agent loop calls its tools through its own runner (``claude -p``, the
Anthropic MCP connector, the SDK tool runner), so the run holds it to
read-only calls by what it hands over, by the rule every pipeline step's
own call meets (``_rote_mcp.write_refusal``). The server here lists the
same two tools the live parity suite's does: ``lookup``, marked
``readOnlyHint``, and ``publish``, not marked.
"""

from __future__ import annotations

import contextlib
import json
import sys
import types
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest

from rote.inference import _runtime_helper as helper
from rote.mcp import _runtime_helper as mcp
from tests.test_inference import _agent_envelope, _capture_cli

URL = "https://ro.example/mcp"


def _tool(name: str, read_only: bool) -> Any:
    annotations = types.SimpleNamespace(read_only_hint=True) if read_only else None
    return types.SimpleNamespace(name=name, annotations=annotations, output_schema=None)


LISTED = [_tool("lookup", True), _tool("publish", False)]
#: Servers that list something else; any other server lists LISTED.
LISTED_BY = {"mirror": [_tool("publish", True)]}


@pytest.fixture()
def server(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """The `ro` server, listed through a fake client; yields each listing's server."""
    listings: list[str] = []

    @contextlib.asynccontextmanager
    async def client(server: str, url: str | None, **_kwargs: Any) -> AsyncIterator[Any]:
        listings.append(server)

        async def list_tools() -> list[Any]:
            return LISTED_BY.get(server, LISTED)

        yield types.SimpleNamespace(list_tools=list_tools)

    monkeypatch.setattr(mcp, "mcp_client", client)
    monkeypatch.setattr(mcp, "_registry_servers", lambda: {})
    monkeypatch.setenv("ROTE_MCP_RO_URL", URL)
    # The inference helper imports the MCP helper as the emitted app does.
    monkeypatch.setitem(sys.modules, "extracted", types.ModuleType("extracted"))
    monkeypatch.setitem(sys.modules, "extracted._rote_mcp", mcp)
    yield listings


def _read_only(monkeypatch: pytest.MonkeyPatch, declared: dict[str, str] | None = None) -> None:
    monkeypatch.setenv(mcp.READ_ONLY_ENV, "1")
    monkeypatch.setenv(mcp.READ_ONLY_SERVERS_ENV, json.dumps(declared or {}))


PINS = {"lookup": "ro", "publish": "ro"}


def test_a_read_only_run_withholds_a_tool_that_may_write(
    server: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _read_only(monkeypatch)
    servers, allowed, removed, withheld = helper._mcp_config_for(["lookup", "publish"], PINS)
    assert allowed == [("ro", "lookup")]
    assert removed == [("ro", "publish")]
    assert list(withheld) == ["publish"]
    assert "ro does not mark it readOnlyHint and is not registered read-only" in withheld["publish"]
    assert servers == {"ro": {"type": "http", "url": URL}}


def test_a_run_that_may_write_lists_nothing_and_withholds_nothing(
    server: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(mcp.READ_ONLY_ENV, raising=False)
    _servers, allowed, removed, withheld = helper._mcp_config_for(["lookup", "publish"], PINS)
    assert (allowed, removed, withheld) == ([("ro", "lookup"), ("ro", "publish")], [], {})
    assert server == []  # no tools/list round trip at all


@pytest.mark.parametrize(
    ("declared", "kept"),
    [
        ({"ro": URL + "/"}, [("ro", "lookup"), ("ro", "publish")]),  # this endpoint: trusted
        ({"ro": "https://elsewhere.example/mcp"}, [("ro", "lookup")]),  # another one: not
    ],
)
def test_a_read_only_registration_covers_its_own_endpoint_only(
    server: list[str],
    monkeypatch: pytest.MonkeyPatch,
    declared: dict[str, str],
    kept: list[tuple[str, str]],
) -> None:
    _read_only(monkeypatch, declared)
    assert helper._mcp_config_for(["lookup", "publish"], PINS).allowed == kept


def test_a_loop_left_with_no_tool_is_refused(
    server: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _read_only(monkeypatch)
    with pytest.raises(mcp.RoteMcpConfigError, match="no MCP tool left to call"):
        helper._mcp_config_for(["publish"], PINS)


def test_the_cli_loop_is_never_offered_a_withheld_tool(
    server: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _read_only(monkeypatch)
    seen = _capture_cli(monkeypatch, _agent_envelope())
    monkeypatch.setattr(helper, "select_provider", lambda *a, **k: "claude-cli")
    helper.run_agent_loop(
        node_id="research",
        description="Research.",
        task="{}",
        model="claude-sonnet-5-5",
        tools=["lookup", "publish"],
        tool_servers=PINS,
        max_iterations=3,
    )
    command = seen["command"]
    assert command[command.index("--allowedTools") + 1] == "mcp__ro__lookup"
    # Removed, not just unlisted: in print mode an unlisted tool is still
    # offered to the model, and only refused when it calls it.
    assert command[command.index("--disallowedTools") + 1] == "mcp__ro__publish"


def test_a_cli_loop_that_may_write_gets_no_deny_list(
    server: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(mcp.READ_ONLY_ENV, raising=False)
    seen = _capture_cli(monkeypatch, _agent_envelope())
    monkeypatch.setattr(helper, "select_provider", lambda *a, **k: "claude-cli")
    helper.run_agent_loop(
        node_id="research",
        description="Research.",
        task="{}",
        model="claude-sonnet-5-5",
        tools=["lookup", "publish"],
        tool_servers=PINS,
        max_iterations=3,
    )
    command = seen["command"]
    assert command[command.index("--allowedTools") + 1] == "mcp__ro__lookup,mcp__ro__publish"
    assert "--disallowedTools" not in command


class _Runner:
    def __iter__(self) -> Iterator[Any]:
        return iter([])

    def until_done(self) -> Any:
        return types.SimpleNamespace(
            content=[types.SimpleNamespace(type="text", text="done")], usage=None
        )


def _sdk_request(monkeypatch: pytest.MonkeyPatch, tools: list[str]) -> dict[str, Any]:
    import anthropic

    seen: dict[str, Any] = {}

    class Client:
        def __init__(self, **_kwargs: Any) -> None:
            def tool_runner(**kwargs: Any) -> _Runner:
                seen.update(kwargs)
                return _Runner()

            self.beta = types.SimpleNamespace(
                messages=types.SimpleNamespace(tool_runner=tool_runner)
            )

    monkeypatch.setattr(anthropic, "Anthropic", Client)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    helper._run_agent_via_sdk(
        node_id="research",
        provider="api",
        model="claude-sonnet-5-5",
        base_url=None,
        prompt="Research.",
        local_tools={},
        tools=tools,
        tool_servers=PINS,
        max_iterations=3,
        timeout=30,
    )
    return seen


def test_the_connector_enables_only_the_tools_that_pass(
    server: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _read_only(monkeypatch)
    request = _sdk_request(monkeypatch, ["lookup", "publish"])
    assert request["mcp_servers"] == [{"type": "url", "url": URL, "name": "ro"}]
    assert request["tools"] == [
        {
            "type": "mcp_toolset",
            "mcp_server_name": "ro",
            "default_config": {"enabled": False},
            "configs": {"lookup": {"enabled": True}},
        }
    ]
    # Without the beta, `mcp_servers` is a 400 (measured against the API).
    assert request["betas"] == [helper.MCP_CONNECTOR_BETA]


def test_the_connector_holds_any_loop_to_its_declared_tools(
    server: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Writes allowed: still only what the loop declared, never the whole server."""
    monkeypatch.delenv(mcp.READ_ONLY_ENV, raising=False)
    request = _sdk_request(monkeypatch, ["lookup"])
    [toolset] = request["tools"]
    assert toolset["default_config"] == {"enabled": False}
    assert toolset["configs"] == {"lookup": {"enabled": True}}


def test_the_refusal_rule_lists_nothing_when_the_run_may_write(
    server: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio

    monkeypatch.delenv(mcp.READ_ONLY_ENV, raising=False)
    assert asyncio.run(mcp.agent_tool_refusals("ro", None, ["publish"])) == {}
    assert server == []
    _read_only(monkeypatch)
    assert list(asyncio.run(mcp.agent_tool_refusals("ro", None, ["lookup", "publish"]))) == [
        "publish"
    ]


# ───────── the loop is told what it was not given ─────────

NOTE = "these tools are not available to you because they may write: publish_brief"


def _cli_prompt(monkeypatch: pytest.MonkeyPatch) -> str:
    seen = _capture_cli(monkeypatch, _agent_envelope())
    monkeypatch.setattr(helper, "select_provider", lambda *a, **k: "claude-cli")
    helper.run_agent_loop(
        node_id="research",
        description="Research, then publish a brief.",
        task="{}",
        model="claude-sonnet-5-5",
        tools=["lookup", "publish"],
        tool_servers=PINS,
        max_iterations=3,
    )
    return str(seen["input"])


def test_a_read_only_loop_is_told_which_tools_it_was_not_given(
    server: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without it a real claude -p loop asked to publish reported it had."""
    _read_only(monkeypatch)
    prompt = _cli_prompt(monkeypatch)
    assert NOTE.replace("publish_brief", "publish") in prompt
    assert "Do not say you used them" in prompt


def test_a_loop_that_may_write_is_told_nothing(
    server: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(mcp.READ_ONLY_ENV, raising=False)
    assert "not available to you" not in _cli_prompt(monkeypatch)


def test_the_connector_loop_is_told_too(server: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    _read_only(monkeypatch)
    [message] = _sdk_request(monkeypatch, ["lookup", "publish"])["messages"]
    assert NOTE.replace("publish_brief", "publish") in message["content"]


def test_both_languages_tell_a_loop_the_same_thing() -> None:
    import re

    from rote.adapters._ts_common import ROTE_INFERENCE_HELPER_TS

    body = re.search(
        r"function withheldNote\(names: string\[\]\): string \{(.*?)\n\}",
        ROTE_INFERENCE_HELPER_TS,
        re.S,
    )
    assert body is not None
    source = body.group(1).replace('${names.join(", ")}', "publish_brief")
    ts = "".join(re.findall(r'[`"]([^`"]*)[`"]', source))
    assert ts == helper._withheld_note({"publish_brief": "why"})


def test_the_emitted_ts_loop_passes_the_withheld_tools_on() -> None:
    from rote.adapters import get_adapter
    from rote.ir import Pipeline

    pipeline = Pipeline.model_validate(
        {
            "name": "p",
            "version": "0.1.0",
            "input": {"type": "In"},
            "nodes": [
                {
                    "id": "loop",
                    "kind": "agent_loop",
                    "description": "d",
                    "tools": ["lookup"],
                    "tool_servers": {"lookup": "ro"},
                    "termination": {"condition": "done", "max_iterations": 2},
                }
            ],
            "edges": [],
        }
    )
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as out:
        get_adapter("dbos-ts").emit(pipeline, Path(out))
        src = (Path(out) / "src" / "extracted" / "loop.ts").read_text()
    assert "const tools: BoundTool[] = [...bound.tools];" in src
    assert "withheld: bound.withheld," in src


# ───────── ids are built from (server, tool) pairs, never parsed ─────────


def test_every_removed_id_is_denied_even_when_another_server_keeps_the_name(
    server: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`publish` may write on `ro` and is read-only on `mirror`: the loop keeps
    mirror's and must be denied ro's, though the name is still allowed."""
    _read_only(monkeypatch)
    monkeypatch.setenv("ROTE_MCP_MIRROR_URL", "https://mirror.example/mcp")
    monkeypatch.setattr(
        mcp, "_registry_servers", lambda: {"ro": {"url": URL}, "mirror": {"url": "x"}}
    )
    seen = _capture_cli(monkeypatch, _agent_envelope())
    monkeypatch.setattr(helper, "select_provider", lambda *a, **k: "claude-cli")
    helper.run_agent_loop(
        node_id="research",
        description="Research.",
        task="{}",
        model="claude-sonnet-5-5",
        tools=["publish"],  # unpinned: offered on every wired server
        max_iterations=3,
    )
    command = seen["command"]
    assert command[command.index("--allowedTools") + 1] == "mcp__mirror__publish"
    assert command[command.index("--disallowedTools") + 1] == "mcp__ro__publish"
    # The name is still available, so nothing is reported withheld.
    assert "not available to you" not in seen["input"]


def test_a_server_name_with_a_double_underscore_binds_as_itself(
    server: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`mcp__my__srv__lookup` split on `__` reads as server `my`, tool
    `srv__lookup`: the connector lane raised KeyError on it."""
    _read_only(monkeypatch)
    monkeypatch.setenv("ROTE_MCP_MY__SRV_URL", URL)
    pins = {"lookup": "my__srv", "publish": "my__srv"}
    wiring = helper._mcp_config_for(["lookup", "publish"], pins)
    assert (wiring.allowed, wiring.removed) == ([("my__srv", "lookup")], [("my__srv", "publish")])
    import anthropic

    seen: dict[str, Any] = {}

    class Client:
        def __init__(self, **_kwargs: Any) -> None:
            def tool_runner(**kwargs: Any) -> _Runner:
                seen.update(kwargs)
                return _Runner()

            self.beta = types.SimpleNamespace(
                messages=types.SimpleNamespace(tool_runner=tool_runner)
            )

    monkeypatch.setattr(anthropic, "Anthropic", Client)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    helper._run_agent_via_sdk(
        node_id="research",
        provider="api",
        model="claude-sonnet-5-5",
        base_url=None,
        prompt="Research.",
        local_tools={},
        tools=["lookup", "publish"],
        tool_servers=pins,
        max_iterations=3,
        timeout=30,
    )
    assert seen["mcp_servers"] == [{"type": "url", "url": URL, "name": "my__srv"}]
    [toolset] = seen["tools"]
    assert (toolset["mcp_server_name"], toolset["configs"]) == (
        "my__srv",
        {"lookup": {"enabled": True}},
    )


def test_each_server_is_asked_only_about_the_tools_it_is_offered_for(
    server: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`lookup` is pinned to ro; `publish` reaches ro and mirror. Mirror lists
    no `lookup`, and must not be asked to judge it."""
    _read_only(monkeypatch)
    monkeypatch.setenv("ROTE_MCP_MIRROR_URL", "https://mirror.example/mcp")
    monkeypatch.setattr(
        mcp, "_registry_servers", lambda: {"ro": {"url": URL}, "mirror": {"url": "x"}}
    )
    wiring = helper._mcp_config_for(["lookup", "publish"], {"lookup": "ro"})
    assert wiring.allowed == [("mirror", "publish"), ("ro", "lookup")]
    assert wiring.removed == [("ro", "publish")]
    assert wiring.withheld == {}
