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
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import mcp.types
import pytest
from fastmcp import Client
from fastmcp.client.messages import MessageHandler
from fastmcp.exceptions import ToolError

from rote.serve import backends
from rote.serve.registry import (
    CloudflareTrigger,
    Registry,
    RegistryEntry,
    TemporalTrigger,
)
from rote.serve.server import build_server

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


@pytest.fixture()
def registry_path(tmp_path: Path) -> Path:
    path = tmp_path / "registry.json"
    registry = Registry()
    registry.upsert(_bdr_entry())
    registry.upsert(_cf_entry())
    registry.save(path)
    return path


# ───────── Tool synthesis + listing ─────────


@pytest.mark.asyncio
async def test_list_tools_one_trigger_plus_one_status_per_entry(registry_path: Path) -> None:
    server = build_server(registry_path)
    async with Client(server) as client:
        tools = {t.name: t for t in await client.list_tools()}

    assert set(tools) == {
        "bdr-campaign",
        "bdr-campaign_status",
        "cf-pipeline",
        "cf-pipeline_status",
    }
    # The trigger tool's inputSchema is exactly the registry's stored schema.
    assert tools["bdr-campaign"].inputSchema == BDR_INPUT_SCHEMA
    # The status tool takes only workflow_id.
    assert tools["bdr-campaign_status"].inputSchema["required"] == ["workflow_id"]
    assert "workflow_id" in tools["bdr-campaign_status"].inputSchema["properties"]
    # Descriptions guide the LLM toward the polling pattern.
    assert "workflow_id" in (tools["bdr-campaign"].description or "")


@pytest.mark.asyncio
async def test_empty_registry_serves_no_tools(tmp_path: Path) -> None:
    server = build_server(tmp_path / "missing.json")
    async with Client(server) as client:
        assert await client.list_tools() == []


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


@pytest.mark.asyncio
async def test_register_mid_session_notifies_and_updates_tool_list(
    registry_path: Path,
) -> None:
    """Writing a new entry to the registry while a client is connected must
    (a) push notifications/tools/list_changed to that client and
    (b) show the new tool on the next tools/list."""
    handler = _ToolListChangedRecorder()
    server = build_server(registry_path, poll_interval=0.05)

    async with Client(server, message_handler=handler) as client:
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

        await asyncio.wait_for(handler.changed.wait(), timeout=5.0)

        after = {t.name for t in await client.list_tools()}
        assert {"new-pipeline", "new-pipeline_status"} <= after


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
            assert tools == {"cf-pipeline", "cf-pipeline_status"}

            result = await client.call_tool("cf-pipeline", {"x": "hello"})
            assert result.data["workflow_id"] == "stub-instance-42"
            assert result.data["status"] == "started"
            assert result.data["runtime"] == "cloudflare"
            # The stub received the tool arguments as the workflow params.
            assert _WorkerStub.last_body == {"x": "hello"}
    finally:
        httpd.shutdown()
        thread.join(timeout=5)
