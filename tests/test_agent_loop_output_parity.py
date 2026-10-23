"""An agent loop produces its declared output keys, on every runtime.

Downstream nodes read an agent loop's declared ``output`` keys
(``process_invoices_loop.output.invoice_results``), but the loop helper
returned ``{result, provider, iterations}`` on every runtime, so none of
those keys existed: a KeyError on the Python runtimes, ``undefined`` on
Cloudflare, and a type error (TS7053) on DBOS-TS and Inngest. The loop now
hands its output back through a ``submit_result`` tool whose schema is the
declared output, and the step returns exactly those keys.

This file holds the fixture every runtime shares and runs the Python
runtimes for real: a fastmcp server for the loop's tool, and a local
Messages endpoint that plays the model (it calls ``submit_result``, then
ends its turn). A downstream node consumes two declared keys, and the
pipeline's output is what that node made of them. No inference: the
endpoint is a stub. The TS runtimes run in
``test_agent_loop_output_parity_ts_e2e.py`` (and the DBOS-TS one in its
Docker job).

Slow: real fastmcp server, DBOS SQLite app, Temporal dev server.
"""

from __future__ import annotations

import json
import sys
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

from rote.adapters import get_adapter
from rote.eval.empirical import run_pipeline_trial
from rote.ir import Pipeline
from tests._messages_api import CURRENT_SONNET, reject_like_the_api, send_json, structured_message
from tests.test_ts_agent_loop_e2e import _MOCK_MCP, _serve_mcp

pytestmark = pytest.mark.slow

PIPELINE = Pipeline.model_validate(
    {
        "name": "loop-output",
        "version": "0.1.0",
        "input": {"type": "In", "required": ["account"], "optional": []},
        "nodes": [
            {
                "id": "research",
                "kind": "agent_loop",
                "description": "Research the account and score it.",
                "tools": ["web_search"],
                "termination": {"condition": "account scored", "max_iterations": 4},
                "inputs": {"account": "pipeline.input.account"},
                "output": {"summary": "str", "score": "int"},
            },
            {
                "id": "report",
                "kind": "pure_function",
                "description": "Write the one-line report.",
                "impl": "extracted/report.py:report",
                "inputs": {
                    "summary": "research.output.summary",
                    "score": "research.output.score",
                },
                "output": {"line": "str"},
            },
        ],
        "edges": [{"from": "research", "to": "report"}],
        "entry_nodes": ["research"],
        "exit_nodes": ["report"],
    }
)

#: What the stub model submits, and what the downstream node makes of it.
SUBMITTED = {"summary": "acme is gold", "score": 7}
EXPECTED = {"report": {"line": "acme is gold / 7"}}

REPORT_PY = "def report(summary, score):\n    return {'line': f'{summary} / {score}'}\n"

SERVER = "research_tools"


class MessagesStub(BaseHTTPRequestHandler):
    """Plays the model: turn 1 calls ``submit_result``, turn 2 ends.

    ``mode`` varies that: ``"forced"`` ends turn 1 with text and answers
    only the schema-locked (``output_config``) turn that follows; ``"bad"``
    submits a score of the wrong type. A request the real API would refuse
    (forced ``tool_choice`` on the current Sonnet, which ``loop_env`` runs
    the loop on) gets the real API's 400. ``POST /token`` answers a refresh grant, which
    the Workers MCP helper needs before it will call even an
    unauthenticated server.
    """

    requests: list[dict[str, Any]] = []
    mode = "submit"

    def do_POST(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler's API
        body = self.rfile.read(int(self.headers.get("content-length", 0)))
        if self.path.rstrip("/").endswith("/token"):
            self._send({"access_token": "local", "token_type": "Bearer", "expires_in": 3600})
            return
        payload = json.loads(body or b"{}")
        type(self).requests.append(payload)
        if reject_like_the_api(self, payload):
            return
        if type(self).mode == "forced" and "format" in payload.get("output_config", {}):
            send_json(self, structured_message(payload, SUBMITTED))
            return
        submitted = any(
            isinstance(block, dict) and block.get("type") == "tool_result"
            for message in payload.get("messages", [])
            if isinstance(message.get("content"), list)
            for block in message["content"]
        )
        answer = {**SUBMITTED, "score": "7"} if type(self).mode == "bad" else SUBMITTED
        submit = [
            {"type": "tool_use", "id": "toolu_submit", "name": "submit_result", "input": answer}
        ]
        content: list[dict[str, Any]]
        if type(self).mode == "forced":
            content = [{"type": "text", "text": "Done, I think."}]
            submitted = True
        else:
            content = [{"type": "text", "text": "Submitted."}] if submitted else submit
        self._send(
            {
                "id": f"msg_{len(type(self).requests)}",
                "type": "message",
                "role": "assistant",
                "model": payload.get("model", "stub"),
                "stop_reason": "end_turn" if submitted else "tool_use",
                "content": content,
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
        )

    def _send(self, message: dict[str, Any]) -> None:
        raw = json.dumps(message).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args: object) -> None:
        return


@pytest.fixture()
def messages_stub() -> Iterator[str]:
    MessagesStub.requests = []
    MessagesStub.mode = "submit"
    server = HTTPServer(("127.0.0.1", 0), MessagesStub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture(scope="module")
def mcp_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    with _serve_mcp(_MOCK_MCP, tmp_path_factory.mktemp("loop-output-mcp"), "research") as url:
        yield url


def loop_env(messages_stub: str, mcp_server: str, tmp_path: Path) -> dict[str, str]:
    """What every runtime needs to run the loop against the two stubs."""
    upper = SERVER.upper()
    return {
        "ROTE_INFERENCE": "api",
        "ANTHROPIC_API_KEY": "local-stub-key",
        "ROTE_BASE_URL_RESEARCH": messages_stub,
        # The model that refuses forced tool choice, so the stub catches it.
        "ROTE_MODEL_RESEARCH": CURRENT_SONNET,
        # Belt and braces: anything that ignored the per-node override would
        # reach a closed port, never the real API.
        "ANTHROPIC_BASE_URL": "http://127.0.0.1:9",
        "ROTE_MCP_SERVERS": SERVER,
        f"ROTE_MCP_{upper}_URL": mcp_server,
        "ROTE_MCP_TOKEN_DIR": str(tmp_path / "tokens"),
        # The Workers helper mints its token by refresh grant.
        f"ROTE_MCP_{upper}_REFRESH_TOKEN": "local",
        f"ROTE_MCP_{upper}_CLIENT_ID": "local",
        f"ROTE_MCP_{upper}_TOKEN_ENDPOINT": f"{messages_stub}/token",
    }


def assert_the_model_was_offered_the_declared_output() -> None:
    """The submit tool's schema is the declared output, keys and types."""
    first = MessagesStub.requests[0]
    tools = {tool["name"]: tool for tool in first["tools"]}
    schema = tools["submit_result"]["input_schema"]
    assert schema["required"] == ["summary", "score"]
    assert schema["properties"] == {"summary": {"type": "string"}, "score": {"type": "integer"}}
    assert schema["additionalProperties"] is False


def _emit(runtime: str, tmp_path: Path) -> Path:
    source = tmp_path / "compiled"
    (source / "extracted").mkdir(parents=True)
    (source / "extracted" / "report.py").write_text(REPORT_PY, encoding="utf-8")
    out = tmp_path / runtime
    get_adapter(runtime, extracted_source_dir=source).emit(PIPELINE, out)
    return out


@pytest.mark.parametrize("runtime", ["dbos", "python"])
def test_downstream_reads_the_loops_declared_keys(
    runtime: str,
    messages_stub: str,
    mcp_server: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key, value in loop_env(messages_stub, mcp_server, tmp_path).items():
        monkeypatch.setenv(key, value)
    run = run_pipeline_trial(_emit(runtime, tmp_path), {"account": "acme"}, timeout_seconds=120)
    assert run.error is None, run.error
    assert run.output == EXPECTED
    assert_the_model_was_offered_the_declared_output()


def test_temporal_downstream_reads_the_loops_declared_keys(
    messages_stub: str, mcp_server: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rote.runners.temporal import run_temporal

    for key, value in loop_env(messages_stub, mcp_server, tmp_path).items():
        monkeypatch.setenv(key, value)
    app = _emit("temporal", tmp_path)
    try:
        run = run_temporal(app, {"account": "acme"}, signals={}, timeout_seconds=120)
    finally:
        sys.path.remove(str(app))
        for name in [
            m
            for m in sys.modules
            if m.split(".")[0] in ("workflow", "activities", "extracted", "signatures")
        ]:
            del sys.modules[name]
    assert run.error is None, run.error
    assert run.output == EXPECTED
    assert_the_model_was_offered_the_declared_output()
