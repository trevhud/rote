"""Tests for the `rote serve` MCP server (rote.serve.server).

Fast tests use FastMCP's in-memory client (pass the server object
straight to Client — no transport, no subprocess) with the runtime
backends monkeypatched, so the suite stays fast and offline.

The slow test at the bottom launches `rote serve` as a real subprocess
over stdio and drives it with a real MCP client, with the Cloudflare
backend hitting a local http.server stub — proving the server speaks
actual MCP over an actual transport and the backend does an actual
HTTP POST.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
import threading
from collections.abc import AsyncIterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import mcp.types
import pytest
from fastmcp import Client
from fastmcp.client.messages import MessageHandler
from fastmcp.exceptions import ToolError
from mcp.client.subscriptions import ToolsListChanged, listen
from mcp_types.version import MODERN_PROTOCOL_VERSIONS

from rote.serve import backends
from rote.serve.registry import (
    CloudflareTrigger,
    DbosTrigger,
    Registry,
    RegistryEntry,
    TemporalTrigger,
)
from rote.serve.server import (
    BUILTIN_TOOL_NAMES,
    DESCRIBE_PIPELINE_TOOL,
    LIST_PIPELINES_TOOL,
    build_server,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
BDR_PIPELINE_YAML = REPO_ROOT / "examples" / "bdr-outreach" / "expected" / "pipeline.yaml"

BDR_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"drug_brand": {"type": "string"}, "target_quota": {"type": "integer"}},
    "required": ["drug_brand", "target_quota"],
    "additionalProperties": True,
}


def _bdr_entry() -> RegistryEntry:
    return RegistryEntry(
        name="bdr-campaign",
        description="BDR outreach campaign pipeline",
        pipeline_yaml=str(BDR_PIPELINE_YAML),
        input_schema=BDR_INPUT_SCHEMA,
        trigger=TemporalTrigger(task_queue="bdr-campaign", workflow_name="BdrCampaign_abc123"),
    )


def _cf_entry(url: str = "https://wf.example.workers.dev") -> RegistryEntry:
    return RegistryEntry(
        name="cf-pipeline",
        description="Cloudflare-deployed pipeline",
        pipeline_yaml=str(BDR_PIPELINE_YAML),
        input_schema={"type": "object", "properties": {"x": {"type": "string"}}},
        trigger=CloudflareTrigger(url=url),
    )


def _dbos_entry(gate_signals: list[str] | None = None) -> RegistryEntry:
    return RegistryEntry(
        name="dbos-pipeline",
        description="DBOS-deployed pipeline",
        pipeline_yaml=str(BDR_PIPELINE_YAML),
        input_schema={"type": "object", "properties": {"x": {"type": "string"}}},
        trigger=DbosTrigger(
            system_database_url="sqlite:////tmp/app/bdr.dbos.sqlite",
            workflow_name="BdrCampaign_abc123",
            queue_name="bdr-campaign-queue",
            gate_signals=(
                gate_signals if gate_signals is not None else ["contact_review_approved"]
            ),
        ),
    )


@pytest.fixture()
def registry_path(tmp_path: Path) -> Path:
    path = tmp_path / "registry.json"
    registry = Registry()
    registry.upsert(_bdr_entry())
    registry.upsert(_cf_entry())
    registry.upsert(_dbos_entry())
    registry.save(path)
    return path


# ───────── Tool synthesis + listing ─────────


@pytest.mark.asyncio
async def test_list_tools_one_trigger_plus_one_status_per_entry(registry_path: Path) -> None:
    server = build_server(registry_path)
    async with Client(server) as client:
        tools = {t.name: t for t in await client.list_tools()}

    assert set(tools) == {
        *BUILTIN_TOOL_NAMES,
        "bdr-campaign",
        "bdr-campaign_status",
        "cf-pipeline",
        "cf-pipeline_status",
        "dbos-pipeline",
        "dbos-pipeline_status",
        # Only the DBOS entry gets a _signal companion (HITL resume).
        "dbos-pipeline_signal",
    }
    # The trigger tool's inputSchema is exactly the registry's stored schema.
    assert tools["bdr-campaign"].input_schema == BDR_INPUT_SCHEMA
    # The status tool takes only workflow_id.
    assert tools["bdr-campaign_status"].input_schema["required"] == ["workflow_id"]
    assert "workflow_id" in tools["bdr-campaign_status"].input_schema["properties"]
    # Descriptions guide the LLM toward the polling pattern.
    assert "workflow_id" in (tools["bdr-campaign"].description or "")
    # The signal tool requires workflow_id + signal, and the register-time
    # gate list becomes an enum so the LLM can't invent topic names.
    signal_schema = tools["dbos-pipeline_signal"].input_schema
    assert set(signal_schema["required"]) == {"workflow_id", "signal"}
    assert signal_schema["properties"]["signal"]["enum"] == ["contact_review_approved"]


@pytest.mark.asyncio
async def test_empty_registry_serves_only_the_builtins(tmp_path: Path) -> None:
    """No registry file: no pipeline tools, but the two built-ins are
    there, so a client (or a directory crawler) still finds tools."""
    server = build_server(tmp_path / "missing.json")
    async with Client(server) as client:
        assert {t.name for t in await client.list_tools()} == BUILTIN_TOOL_NAMES


# ───────── Built-in tools ─────────


@pytest.mark.asyncio
async def test_builtin_tool_definitions(tmp_path: Path) -> None:
    """The built-ins carry what a client needs to pick and call them
    without trial and error: a description, a described parameter, a
    structured output schema, and read-only annotations."""
    server = build_server(tmp_path / "missing.json")
    async with Client(server) as client:
        tools = {t.name: t for t in await client.list_tools()}

    listing = tools[LIST_PIPELINES_TOOL]
    assert listing.description and "registry" in listing.description
    runtime_param = listing.input_schema["properties"]["runtime"]
    assert runtime_param["description"]
    assert runtime_param["anyOf"][0]["enum"] == ["temporal", "cloudflare", "dbos"]
    assert listing.input_schema.get("required", []) == []
    assert listing.output_schema is not None
    assert set(listing.output_schema["required"]) == {
        "registry_path",
        "registry_exists",
        "pipelines",
    }

    describe = tools[DESCRIBE_PIPELINE_TOOL]
    assert describe.description and "JSON Schema" in describe.description
    assert describe.input_schema["required"] == ["name"]
    assert describe.input_schema["properties"]["name"]["description"]
    assert describe.output_schema is not None
    assert {"input_schema", "trigger", "tools"} <= set(describe.output_schema["required"])

    for tool in (listing, describe):
        assert tool.annotations is not None
        assert tool.annotations.read_only_hint is True
        assert tool.annotations.idempotent_hint is True
        assert tool.annotations.open_world_hint is False
        # No newlines or em dashes: directories render the first sentence.
        assert "\n" not in (tool.description or "")


@pytest.mark.asyncio
async def test_list_pipelines_with_empty_registry(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    server = build_server(missing)
    async with Client(server) as client:
        result = await client.call_tool(LIST_PIPELINES_TOOL, {})

    assert result.structured_content == {
        "registry_path": str(missing),
        "registry_exists": False,
        "pipelines": [],
    }


@pytest.mark.asyncio
async def test_list_pipelines_reports_every_entry(registry_path: Path) -> None:
    server = build_server(registry_path)
    async with Client(server) as client:
        result = await client.call_tool(LIST_PIPELINES_TOOL, {})
        only_dbos = await client.call_tool(LIST_PIPELINES_TOOL, {"runtime": "dbos"})

    data = result.structured_content
    assert data is not None
    assert data["registry_path"] == str(registry_path)
    assert data["registry_exists"] is True
    by_name = {p["name"]: p for p in data["pipelines"]}
    assert list(by_name) == ["bdr-campaign", "cf-pipeline", "dbos-pipeline"]

    bdr = by_name["bdr-campaign"]
    assert bdr["description"] == "BDR outreach campaign pipeline"
    assert bdr["runtime"] == "temporal"
    assert bdr["tools"] == {
        "trigger": "bdr-campaign",
        "status": "bdr-campaign_status",
        "signal": None,
    }
    assert "localhost:7233" in bdr["trigger_summary"]
    assert "BdrCampaign_abc123" in bdr["trigger_summary"]
    assert bdr["pipeline_yaml"] == str(BDR_PIPELINE_YAML)
    assert bdr["registered_at"]

    # Only the DBOS entry has a signal tool, and its gate names are listed.
    dbos = by_name["dbos-pipeline"]
    assert dbos["tools"]["signal"] == "dbos-pipeline_signal"
    assert "contact_review_approved" in dbos["trigger_summary"]

    assert "no status endpoint" in by_name["cf-pipeline"]["trigger_summary"]

    # The runtime filter narrows the list, and the envelope stays the same.
    assert only_dbos.structured_content is not None
    assert [p["name"] for p in only_dbos.structured_content["pipelines"]] == ["dbos-pipeline"]


@pytest.mark.asyncio
async def test_list_pipelines_rejects_unknown_runtime(registry_path: Path) -> None:
    server = build_server(registry_path)
    async with Client(server) as client:
        with pytest.raises(ToolError, match="runtime"):
            await client.call_tool(LIST_PIPELINES_TOOL, {"runtime": "inngest"})


@pytest.mark.asyncio
async def test_describe_pipeline_returns_schema_and_redacted_trigger(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.json"
    registry = Registry()
    registry.upsert(_bdr_entry())
    secret_dbos = _dbos_entry()
    secret_dbos.trigger = DbosTrigger(
        system_database_url="postgresql://rote:hunter2@db.internal:5432/rote",
        workflow_name="BdrCampaign_abc123",
        queue_name="bdr-campaign-queue",
        gate_signals=["contact_review_approved"],
    )
    registry.upsert(secret_dbos)
    registry.save(registry_path)

    server = build_server(registry_path)
    async with Client(server) as client:
        bdr = await client.call_tool(DESCRIBE_PIPELINE_TOOL, {"name": "bdr-campaign"})
        dbos = await client.call_tool(DESCRIBE_PIPELINE_TOOL, {"name": "dbos-pipeline"})

    assert bdr.structured_content is not None
    assert (
        bdr.structured_content["registered_at"]
        == Registry.load(registry_path).entries[0].registered_at
    )
    assert {k: v for k, v in bdr.structured_content.items() if k != "registered_at"} == {
        "name": "bdr-campaign",
        "description": "BDR outreach campaign pipeline",
        "runtime": "temporal",
        "tools": {
            "trigger": "bdr-campaign",
            "status": "bdr-campaign_status",
            "signal": None,
        },
        # The served contract, verbatim.
        "input_schema": BDR_INPUT_SCHEMA,
        "trigger": {
            "runtime": "temporal",
            "address": "localhost:7233",
            "namespace": "default",
            "task_queue": "bdr-campaign",
            "workflow_name": "BdrCampaign_abc123",
        },
        "pipeline_yaml": str(BDR_PIPELINE_YAML),
    }

    # The full DBOS trigger comes back, minus the database password.
    assert dbos.structured_content is not None
    trigger = dbos.structured_content["trigger"]
    assert trigger["system_database_url"] == "postgresql://rote:***@db.internal:5432/rote"
    assert trigger["gate_signals"] == ["contact_review_approved"]
    assert "hunter2" not in json.dumps(dbos.structured_content)


@pytest.mark.asyncio
async def test_describe_unknown_pipeline_is_clear_error(registry_path: Path) -> None:
    server = build_server(registry_path)
    async with Client(server) as client:
        with pytest.raises(ToolError, match="No registered pipeline named 'nope'") as exc_info:
            await client.call_tool(DESCRIBE_PIPELINE_TOOL, {"name": "nope"})
    # The error names what is registered so the caller can self-correct.
    assert "bdr-campaign" in str(exc_info.value)
    assert LIST_PIPELINES_TOOL in str(exc_info.value)


@pytest.mark.asyncio
async def test_builtins_survive_registry_reload(registry_path: Path) -> None:
    """A registry change reloads the pipeline tools. The built-ins are
    registered once and stay, and they read the new registry."""
    server = build_server(registry_path, poll_interval=0.05)
    async with Client(server) as client:
        assert {t.name for t in await client.list_tools()} >= BUILTIN_TOOL_NAMES

        registry = Registry.load(registry_path)
        registry.upsert(
            RegistryEntry(
                name="late-pipeline",
                description="Registered mid-session",
                pipeline_yaml=str(BDR_PIPELINE_YAML),
                input_schema={"type": "object"},
                trigger=TemporalTrigger(task_queue="q", workflow_name="Late_abc123"),
            )
        )
        registry.save(registry_path)
        # Long enough for the watcher to reload (the per-request re-read
        # would also catch it. This proves the watcher path keeps them).
        await asyncio.sleep(0.3)

        names = {t.name for t in await client.list_tools()}
        assert names >= BUILTIN_TOOL_NAMES
        assert {"late-pipeline", "late-pipeline_status"} <= names
        listed = await client.call_tool(LIST_PIPELINES_TOOL, {})
        assert listed.structured_content is not None
        assert "late-pipeline" in [p["name"] for p in listed.structured_content["pipelines"]]


@pytest.mark.asyncio
async def test_pipeline_named_like_a_builtin_is_skipped_not_served(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """`rote register` refuses the rote_ prefix, so this is the hand-edited
    registry case. The built-in keeps its name and keeps working. The
    colliding entry is left out of tools/list and of the built-ins'
    output, with one warning naming the fix."""
    registry_path = tmp_path / "registry.json"
    registry = Registry()
    registry.upsert(_bdr_entry())
    registry.upsert(
        RegistryEntry(
            name=LIST_PIPELINES_TOOL,
            description="Impostor",
            pipeline_yaml=str(BDR_PIPELINE_YAML),
            input_schema={"type": "object"},
            trigger=TemporalTrigger(task_queue="q", workflow_name="Impostor_abc123"),
        )
    )
    registry.save(registry_path)

    server = build_server(registry_path)
    with caplog.at_level("WARNING", logger="rote.serve.server"):
        async with Client(server) as client:
            tools = {t.name: t for t in await client.list_tools()}
            # Listing twice must not warn twice.
            await client.list_tools()
            listed = await client.call_tool(LIST_PIPELINES_TOOL, {})
            with pytest.raises(ToolError, match="No registered pipeline named"):
                await client.call_tool(DESCRIBE_PIPELINE_TOOL, {"name": LIST_PIPELINES_TOOL})

    assert set(tools) == {*BUILTIN_TOOL_NAMES, "bdr-campaign", "bdr-campaign_status"}
    assert "Impostor" not in (tools[LIST_PIPELINES_TOOL].description or "")
    assert f"{LIST_PIPELINES_TOOL}_status" not in tools
    assert listed.structured_content is not None
    assert [p["name"] for p in listed.structured_content["pipelines"]] == ["bdr-campaign"]

    warnings = [
        r.getMessage() for r in caplog.records if "collides with a built-in" in r.getMessage()
    ]
    assert len(warnings) == 1
    assert f"'{LIST_PIPELINES_TOOL}'" in warnings[0]
    assert "rote register --name" in warnings[0]


@pytest.mark.asyncio
async def test_corrupt_registry_keeps_previous_tool_list_on_request(
    registry_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A tools/list (or built-in call) that lands while the registry file
    is corrupt serves the previous entries plus the built-ins, instead of
    the provider raising and FastMCP answering with no tools at all."""
    server = build_server(registry_path, poll_interval=3600.0)
    async with Client(server) as client:
        before = {t.name for t in await client.list_tools()}
        registry_path.write_text("{ not json", encoding="utf-8")
        with caplog.at_level("WARNING", logger="rote.serve.server"):
            during = {t.name for t in await client.list_tools()}
            listed = await client.call_tool(LIST_PIPELINES_TOOL, {})

    assert during == before
    assert during >= BUILTIN_TOOL_NAMES
    assert listed.structured_content is not None
    assert [p["name"] for p in listed.structured_content["pipelines"]] == [
        "bdr-campaign",
        "cf-pipeline",
        "dbos-pipeline",
    ]
    assert any("registry reload failed" in r.getMessage() for r in caplog.records)


# ───────── Tool calls (mocked backends) ─────────


@pytest.mark.asyncio
async def test_call_trigger_tool_starts_workflow(
    registry_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[RegistryEntry, dict[str, Any]]] = []

    async def fake_start(entry: RegistryEntry, payload: dict[str, Any]) -> dict[str, Any]:
        calls.append((entry, payload))
        return {"workflow_id": "bdr-campaign-deadbeef", "status": "started", "runtime": "temporal"}

    monkeypatch.setattr(backends, "start_workflow", fake_start)

    server = build_server(registry_path)
    async with Client(server) as client:
        result = await client.call_tool(
            "bdr-campaign", {"drug_brand": "ExampleDrug", "target_quota": 25}
        )

    assert result.data == {
        "workflow_id": "bdr-campaign-deadbeef",
        "status": "started",
        "runtime": "temporal",
    }
    (entry, payload) = calls[0]
    assert entry.name == "bdr-campaign"
    assert payload == {"drug_brand": "ExampleDrug", "target_quota": 25}


@pytest.mark.asyncio
async def test_call_status_tool_polls_workflow(
    registry_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_status(entry: RegistryEntry, workflow_id: str) -> dict[str, Any]:
        return {"workflow_id": workflow_id, "status": "running", "runtime": "temporal"}

    monkeypatch.setattr(backends, "workflow_status", fake_status)

    server = build_server(registry_path)
    async with Client(server) as client:
        result = await client.call_tool(
            "bdr-campaign_status", {"workflow_id": "bdr-campaign-deadbeef"}
        )

    assert result.data == {
        "workflow_id": "bdr-campaign-deadbeef",
        "status": "running",
        "runtime": "temporal",
    }


@pytest.mark.asyncio
async def test_backend_error_surfaces_as_tool_error(
    registry_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def failing_start(entry: RegistryEntry, payload: dict[str, Any]) -> dict[str, Any]:
        raise backends.BackendError("Temporal at localhost:7233 is unreachable: boom")

    monkeypatch.setattr(backends, "start_workflow", failing_start)

    server = build_server(registry_path)
    async with Client(server) as client:
        with pytest.raises(ToolError, match="localhost:7233 is unreachable"):
            await client.call_tool("bdr-campaign", {"drug_brand": "X", "target_quota": 1})


@pytest.mark.asyncio
async def test_cloudflare_status_without_status_url_is_clear_error(
    registry_path: Path,
) -> None:
    # Real backend code path — no mocking. The trigger has no status_url,
    # so the status tool must fail with guidance, not a stack trace.
    server = build_server(registry_path)
    async with Client(server) as client:
        with pytest.raises(ToolError, match="wrangler workflows instances describe"):
            await client.call_tool("cf-pipeline_status", {"workflow_id": "abc"})


# ───────── Live registry updates ─────────


class _ToolListChangedRecorder(MessageHandler):
    def __init__(self) -> None:
        super().__init__()
        self.changed = asyncio.Event()

    async def on_tool_list_changed(
        self, notification: mcp.types.ToolListChangedNotification
    ) -> None:
        self.changed.set()


@contextlib.asynccontextmanager
async def _tool_change_watch(server: Any, era: str) -> AsyncIterator[tuple[Client, asyncio.Event]]:
    """A connected client plus an event that fires on a tool-list change.

    The two protocol eras deliver the change on different channels, and
    the server has to serve both: a handshake-era session gets a bare
    ``notifications/tools/list_changed``; a 2026-07-28 connection has no
    session and only hears about changes on a ``subscriptions/listen``
    stream it opened.
    """
    if era == "legacy":
        handler = _ToolListChangedRecorder()
        async with Client(server, message_handler=handler, mode="legacy") as client:
            assert client.protocol_version not in MODERN_PROTOCOL_VERSIONS
            yield client, handler.changed
        return

    changed = asyncio.Event()
    async with Client(server) as client:
        assert client.protocol_version in MODERN_PROTOCOL_VERSIONS
        # Without a listen handler the SDK advertises listChanged: false on
        # this era, and a well-behaved client would never even ask.
        assert client.server_capabilities.tools is not None
        assert client.server_capabilities.tools.list_changed is True
        async with listen(client.session, tools_list_changed=True) as subscription:

            async def _drain() -> None:
                async for event in subscription:
                    if isinstance(event, ToolsListChanged):
                        changed.set()

            drain = asyncio.create_task(_drain())
            try:
                yield client, changed
            finally:
                drain.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await drain


@pytest.mark.asyncio
@pytest.mark.parametrize("era", ["legacy", "modern"])
async def test_register_mid_session_notifies_and_updates_tool_list(
    registry_path: Path, era: str
) -> None:
    """Writing a new entry to the registry while a client is connected must
    (a) tell that client the tool list changed, on whichever channel its
    protocol era uses, and (b) show the new tool on the next tools/list."""
    server = build_server(registry_path, poll_interval=0.05)

    async with _tool_change_watch(server, era) as (client, changed):
        before = {t.name for t in await client.list_tools()}
        assert "new-pipeline" not in before

        # Simulate `rote register` from another process: rewrite the file.
        registry = Registry.load(registry_path)
        registry.upsert(
            RegistryEntry(
                name="new-pipeline",
                description="Registered mid-session",
                pipeline_yaml=str(BDR_PIPELINE_YAML),
                input_schema={"type": "object"},
                trigger=TemporalTrigger(task_queue="q", workflow_name="New_abc123"),
            )
        )
        registry.save(registry_path)

        await asyncio.wait_for(changed.wait(), timeout=5.0)

        after = {t.name for t in await client.list_tools()}
        assert {"new-pipeline", "new-pipeline_status"} <= after


@pytest.mark.asyncio
async def test_sdk_moving_the_connection_degrades_only_notifications(
    registry_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Connection tracking reads a private MCP SDK attribute. If a release
    moves it, legacy-era tool calls must keep working and the server must
    say why notifications stopped, once, instead of failing every
    request in the middleware."""
    from rote.serve import server as server_module

    async def fake_start(entry: RegistryEntry, payload: dict[str, Any]) -> dict[str, Any]:
        return {"workflow_id": "wf-1", "status": "started", "runtime": "temporal"}

    monkeypatch.setattr(backends, "start_workflow", fake_start)
    monkeypatch.setattr(server_module, "_SESSION_CONNECTION_ATTR", "_moved_in_a_future_sdk")
    server = build_server(registry_path, poll_interval=3600.0)

    with caplog.at_level("WARNING", logger="rote.serve.server"):
        async with Client(server, mode="legacy") as client:
            assert "bdr-campaign" in {t.name for t in await client.list_tools()}
            result = await client.call_tool(
                "bdr-campaign", {"drug_brand": "ExampleDrug", "target_quota": 25}
            )
            assert result.structured_content == {
                "workflow_id": "wf-1",
                "status": "started",
                "runtime": "temporal",
            }

    warnings = [r for r in caplog.records if "cannot track handshake-era" in r.getMessage()]
    assert len(warnings) == 1


@pytest.mark.asyncio
async def test_tool_list_fresh_without_notification(registry_path: Path) -> None:
    """Even with the watcher effectively disabled, the provider re-reads the
    registry per request — clients that re-list always see current tools."""
    server = build_server(registry_path, poll_interval=3600.0)

    async with Client(server) as client:
        registry = Registry.load(registry_path)
        registry.upsert(
            RegistryEntry(
                name="another-pipeline",
                description="",
                pipeline_yaml=str(BDR_PIPELINE_YAML),
                input_schema={"type": "object"},
                trigger=TemporalTrigger(task_queue="q", workflow_name="A_abc123"),
            )
        )
        registry.save(registry_path)

        names = {t.name for t in await client.list_tools()}
        assert "another-pipeline" in names


@pytest.mark.asyncio
@pytest.mark.parametrize("era", ["legacy", "modern"])
async def test_corrupt_registry_survives_and_recovers(registry_path: Path, era: str) -> None:
    """A corrupt registry file must not kill the watcher or pin the stale
    digest: the server keeps serving the previous tool list, and once the
    file is fixed the next poll picks it up and notifies."""
    server = build_server(registry_path, poll_interval=0.05)

    async with _tool_change_watch(server, era) as (client, changed):
        before = {t.name for t in await client.list_tools()}
        assert "bdr-campaign" in before

        # Corrupt the file (hand-edit typo). Give the watcher several polls:
        # it must neither die nor notify.
        registry_path.write_text("{ not json", encoding="utf-8")
        await asyncio.sleep(0.3)
        assert not changed.is_set()

        # Fix the file with a new entry. The watcher must still be alive to
        # see it — this fails if the corrupt poll killed the task or if the
        # corrupt digest was committed as "seen".
        registry = Registry()
        registry.upsert(_bdr_entry())
        registry.upsert(_cf_entry())
        registry.upsert(
            RegistryEntry(
                name="post-recovery",
                description="Registered after the corrupt write was fixed",
                pipeline_yaml=str(BDR_PIPELINE_YAML),
                input_schema={"type": "object"},
                trigger=TemporalTrigger(task_queue="q", workflow_name="R_abc123"),
            )
        )
        registry.save(registry_path)

        await asyncio.wait_for(changed.wait(), timeout=5.0)
        after = {t.name for t in await client.list_tools()}
        assert "post-recovery" in after


# ───────── Backend units (no server) ─────────


@pytest.mark.asyncio
async def test_cloudflare_status_url_template_is_hit(monkeypatch: pytest.MonkeyPatch) -> None:
    entry = _cf_entry()
    entry.trigger = CloudflareTrigger(
        url="https://wf.example.workers.dev",
        status_url="https://wf.example.workers.dev/status/{workflow_id}",
    )

    seen: list[str] = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, Any]:
            return {"status": "complete"}

    class FakeAsyncClient:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> FakeAsyncClient:
            return self

        async def __aexit__(self, *exc: Any) -> None:
            pass

        async def get(self, url: str) -> FakeResponse:
            seen.append(url)
            return FakeResponse()

    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    result = await backends.workflow_status(entry, "wf-42")
    assert seen == ["https://wf.example.workers.dev/status/wf-42"]
    assert result == {"workflow_id": "wf-42", "status": "complete", "runtime": "cloudflare"}


@pytest.mark.asyncio
async def test_cloudflare_start_without_id_is_backend_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 2xx JSON response without an `id` means the URL is not the emitted
    worker — that must be a BackendError, not a fabricated success with an
    empty workflow_id."""
    entry = _cf_entry()

    class FakeResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, Any]:
            return {"message": "welcome to some unrelated service"}

    class FakeAsyncClient:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> FakeAsyncClient:
            return self

        async def __aexit__(self, *exc: Any) -> None:
            pass

        async def post(self, url: str, json: Any = None) -> FakeResponse:
            return FakeResponse()

    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    with pytest.raises(backends.BackendError, match="did not return a workflow id"):
        await backends.start_workflow(entry, {"x": "y"})


# ───────── DBOS tool calls (mocked backend) ─────────


@pytest.mark.asyncio
async def test_call_signal_tool_resumes_gate(
    registry_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, str, str, dict[str, Any]]] = []

    async def fake_signal(
        entry: RegistryEntry, workflow_id: str, signal: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        calls.append((entry.name, workflow_id, signal, payload))
        return {
            "workflow_id": workflow_id,
            "signal": signal,
            "status": "signaled",
            "runtime": "dbos",
        }

    monkeypatch.setattr(backends, "signal_workflow", fake_signal)

    server = build_server(registry_path)
    async with Client(server) as client:
        result = await client.call_tool(
            "dbos-pipeline_signal",
            {
                "workflow_id": "dbos-pipeline-deadbeef",
                "signal": "contact_review_approved",
                "payload": {"approved": True},
            },
        )

    assert result.data["status"] == "signaled"
    assert calls == [
        (
            "dbos-pipeline",
            "dbos-pipeline-deadbeef",
            "contact_review_approved",
            {"approved": True},
        )
    ]


# ───────── DBOS backend units (fake DBOSClient) ─────────


class _FakeDbosHandle:
    def __init__(self, status: str) -> None:
        self._status = status

    async def get_status(self) -> Any:
        from types import SimpleNamespace

        return SimpleNamespace(status=self._status)


class _FakeDbosClient:
    """Mimics the DBOSClient surface the backend touches. Class attributes
    configure behavior per test; instances record calls."""

    construct_error: Exception | None = None
    retrieve_error: Exception | None = None
    status: str = "PENDING"
    enqueued: list[tuple[dict[str, Any], tuple[Any, ...]]] = []
    sent: list[tuple[str, Any, str | None]] = []
    constructed_with: list[str] = []
    destroyed: int = 0

    def __init__(self, *, system_database_url: str) -> None:
        type(self).constructed_with.append(system_database_url)
        if type(self).construct_error is not None:
            raise type(self).construct_error

    async def enqueue_async(self, options: dict[str, Any], *args: Any) -> Any:
        type(self).enqueued.append((options, args))
        return _FakeDbosHandle(type(self).status)

    async def retrieve_workflow_async(self, workflow_id: str) -> _FakeDbosHandle:
        if type(self).retrieve_error is not None:
            raise type(self).retrieve_error
        return _FakeDbosHandle(type(self).status)

    async def send_async(self, destination_id: str, message: Any, topic: str | None = None) -> None:
        type(self).sent.append((destination_id, message, topic))

    def destroy(self) -> None:
        type(self).destroyed += 1


@pytest.fixture()
def fake_dbos_client(monkeypatch: pytest.MonkeyPatch) -> type[_FakeDbosClient]:
    import dbos as dbos_module

    _FakeDbosClient.construct_error = None
    _FakeDbosClient.retrieve_error = None
    _FakeDbosClient.status = "PENDING"
    _FakeDbosClient.enqueued = []
    _FakeDbosClient.sent = []
    _FakeDbosClient.constructed_with = []
    _FakeDbosClient.destroyed = 0
    monkeypatch.setattr(dbos_module, "DBOSClient", _FakeDbosClient)
    return _FakeDbosClient


@pytest.mark.asyncio
async def test_dbos_start_enqueues_on_registered_coordinates(
    fake_dbos_client: type[_FakeDbosClient],
) -> None:
    entry = _dbos_entry()
    result = await backends.start_workflow(entry, {"x": "hello"})

    assert result["status"] == "started"
    assert result["runtime"] == "dbos"
    assert result["workflow_id"].startswith("dbos-pipeline-")

    assert fake_dbos_client.constructed_with == ["sqlite:////tmp/app/bdr.dbos.sqlite"]
    ((options, args),) = fake_dbos_client.enqueued
    assert options["workflow_name"] == "BdrCampaign_abc123"
    assert options["queue_name"] == "bdr-campaign-queue"
    assert options["workflow_id"] == result["workflow_id"]
    # The tool arguments travel as the workflow's single positional input.
    assert args == ({"x": "hello"},)
    # The connection pool is released per call (lazy-per-call contract).
    assert fake_dbos_client.destroyed == 1


@pytest.mark.asyncio
async def test_dbos_status_is_lowercased(fake_dbos_client: type[_FakeDbosClient]) -> None:
    fake_dbos_client.status = "ENQUEUED"
    result = await backends.workflow_status(_dbos_entry(), "wf-1")
    assert result == {"workflow_id": "wf-1", "status": "enqueued", "runtime": "dbos"}


@pytest.mark.asyncio
async def test_dbos_signal_sends_on_topic(fake_dbos_client: type[_FakeDbosClient]) -> None:
    entry = _dbos_entry()
    result = await backends.signal_workflow(
        entry, "wf-1", "contact_review_approved", {"approved": True}
    )
    assert result["status"] == "signaled"
    assert fake_dbos_client.sent == [("wf-1", {"approved": True}, "contact_review_approved")]


@pytest.mark.asyncio
async def test_dbos_unreachable_database_is_backend_error(
    fake_dbos_client: type[_FakeDbosClient],
) -> None:
    fake_dbos_client.construct_error = ConnectionError("connection refused")
    entry = _dbos_entry()
    entry.trigger = DbosTrigger(
        system_database_url="postgresql://rote:hunter2@db.internal:5432/rote",
        workflow_name="BdrCampaign_abc123",
        queue_name="bdr-campaign-queue",
    )
    with pytest.raises(backends.BackendError, match="is unreachable") as exc_info:
        await backends.start_workflow(entry, {})
    # The error names the database but never leaks the password.
    assert "db.internal" in str(exc_info.value)
    assert "hunter2" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_dbos_unknown_workflow_is_backend_error(
    fake_dbos_client: type[_FakeDbosClient],
) -> None:
    fake_dbos_client.retrieve_error = KeyError("no such workflow")
    with pytest.raises(backends.BackendError, match="could not retrieve workflow 'wf-missing'"):
        await backends.workflow_status(_dbos_entry(), "wf-missing")


@pytest.mark.asyncio
async def test_dbos_unknown_signal_is_rejected_before_sending(
    fake_dbos_client: type[_FakeDbosClient],
) -> None:
    """A message on a topic no gate recv's on is silently swallowed by
    DBOS, so the backend rejects signals outside the registered gate list."""
    with pytest.raises(backends.BackendError, match="Unknown signal 'not_a_gate'"):
        await backends.signal_workflow(_dbos_entry(), "wf-1", "not_a_gate", {})
    assert fake_dbos_client.sent == []
    # Rejected before any database connection was made.
    assert fake_dbos_client.constructed_with == []


@pytest.mark.asyncio
async def test_signal_on_non_dbos_entry_is_backend_error() -> None:
    with pytest.raises(backends.BackendError, match="only supported for the dbos runtime"):
        await backends.signal_workflow(_bdr_entry(), "wf-1", "anything", {})


# ───────── Temporal backend against a real (test) server ─────────

from temporalio import workflow  # noqa: E402 - grouped with the test that needs it


@workflow.defn(name="ServeSmokeWorkflow")
class ServeSmokeWorkflow:
    """Minimal workflow for the backend smoke test (temporalio forbids
    local classes under @workflow.defn)."""

    @workflow.run
    async def run(self, brief: dict[str, Any]) -> dict[str, Any]:
        return {"echo": brief}


@pytest.mark.asyncio
async def test_temporal_backend_starts_and_polls_real_workflow() -> None:
    """Drive the real Temporal backend (its own Client.connect, real
    start_workflow / describe calls) against Temporal's time-skipping test
    server — the same environment test_temporal_e2e.py uses. This is the
    empirical proof that the backend's temporalio usage is correct, not
    just the mocked dispatch."""
    from uuid import uuid4

    from temporalio.testing import WorkflowEnvironment
    from temporalio.worker import UnsandboxedWorkflowRunner, Worker

    async with await WorkflowEnvironment.start_time_skipping() as env:
        address = env.client.service_client.config.target_host
        task_queue = f"rote-serve-smoke-{uuid4()}"
        entry = RegistryEntry(
            name="serve-smoke",
            description="",
            pipeline_yaml=str(BDR_PIPELINE_YAML),
            input_schema={"type": "object"},
            trigger=TemporalTrigger(
                address=address,
                task_queue=task_queue,
                workflow_name="ServeSmokeWorkflow",
            ),
        )

        async with Worker(
            env.client,
            task_queue=task_queue,
            workflows=[ServeSmokeWorkflow],
            # The workflow is pure; the sandbox would re-import this test
            # module and trip on fastmcp's import-hook dependencies. Same
            # approach as test_temporal_e2e.py.
            workflow_runner=UnsandboxedWorkflowRunner(),
        ):
            started = await backends.start_workflow(entry, {"drug_brand": "X"})
            assert started["status"] == "started"
            assert started["runtime"] == "temporal"
            assert started["workflow_id"].startswith("serve-smoke-")

            handle = env.client.get_workflow_handle(started["workflow_id"])
            assert await handle.result() == {"echo": {"drug_brand": "X"}}

            polled = await backends.workflow_status(entry, started["workflow_id"])
            assert polled == {
                "workflow_id": started["workflow_id"],
                "status": "completed",
                "runtime": "temporal",
            }


# ───────── Real transport smoke test ─────────


class _WorkerStub(BaseHTTPRequestHandler):
    """Mimics the emitted Cloudflare src/index.ts fetch handler:
    JSON POST body in, {id, status} out."""

    last_body: dict[str, Any] | None = None

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        length = int(self.headers.get("Content-Length", "0"))
        type(self).last_body = json.loads(self.rfile.read(length) or b"{}")
        payload = json.dumps(
            {"id": "stub-instance-42", "status": {"status": "running", "output": None}}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass  # keep pytest output clean


@pytest.mark.slow
@pytest.mark.asyncio
async def test_serve_over_real_stdio_transport(tmp_path: Path) -> None:
    """Launch `rote serve` as a subprocess over stdio and drive it with a
    real MCP client. The registered pipeline's Cloudflare trigger points at
    a local http.server stub, so the tool call exercises the real backend
    HTTP path with no cloud dependency. Slow: spawns a Python subprocess.
    """
    from fastmcp.client.transports import StdioTransport

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _WorkerStub)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    registry_path = tmp_path / "registry.json"
    registry = Registry()
    registry.upsert(_cf_entry(url=f"http://127.0.0.1:{port}/"))
    registry.save(registry_path)

    transport = StdioTransport(
        command=sys.executable,
        args=["-m", "rote.cli", "serve", "--registry", str(registry_path)],
    )
    try:
        async with Client(transport) as client:
            tools = {t.name for t in await client.list_tools()}
            assert tools == {*BUILTIN_TOOL_NAMES, "cf-pipeline", "cf-pipeline_status"}

            listed = await client.call_tool(LIST_PIPELINES_TOOL, {})
            assert listed.structured_content is not None
            assert [p["name"] for p in listed.structured_content["pipelines"]] == ["cf-pipeline"]

            result = await client.call_tool("cf-pipeline", {"x": "hello"})
            assert result.data["workflow_id"] == "stub-instance-42"
            assert result.data["status"] == "started"
            assert result.data["runtime"] == "cloudflare"
            # The stub received the tool arguments as the workflow params.
            assert _WorkerStub.last_body == {"x": "hello"}
    finally:
        httpd.shutdown()
        thread.join(timeout=5)
