"""An emitted TypeScript judge, compiled and run against a Messages stub.

The stub refuses what the real API refuses (``tests/_messages_api.py``):
forced ``tool_choice`` on claude-sonnet-5-5, and an ``output_config``
schema outside the structured-outputs subset. So a judge that went back
to forcing a tool call, or sent its IR schema without the rewrite, fails
here the way it would fail on every call in production.

Slow: real npm install + tsc + node. No inference: the endpoint is local.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

from rote.adapters import get_adapter
from rote.inference import structured_output_schema
from rote.ir import Pipeline
from tests._messages_api import CURRENT_SONNET, reject_like_the_api, send_json, structured_message

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        shutil.which("node") is None or shutil.which("npm") is None,
        reason="requires node + npm",
    ),
]

#: Two things structured outputs 400s on as written: a nested object
#: without additionalProperties, and integer bounds. Plus an enum and a
#: const, which the TS SDK's own transformJSONSchema would move into
#: descriptions: the emitted judge must not run it.
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "grade": {"type": "integer", "minimum": 1, "maximum": 10},
        "verdict": {"type": "string", "enum": ["pass", "fail"]},
        "rubric": {"type": "string", "const": "v1"},
        "notes": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    "required": ["grade", "verdict", "rubric", "notes"],
    "additionalProperties": False,
}

ANSWER = {"grade": 9, "verdict": "pass", "rubric": "v1", "notes": {"text": "clear thesis"}}


class _JudgeStub(BaseHTTPRequestHandler):
    requests: list[dict[str, Any]] = []
    stop = "end_turn"

    def do_POST(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        payload = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
        type(self).requests.append(payload)
        if reject_like_the_api(self, payload):
            return
        send_json(self, structured_message(payload, ANSWER, stop_reason=type(self).stop))

    def log_message(self, *args: object) -> None:
        return


@pytest.fixture()
def judge_stub() -> Iterator[str]:
    _JudgeStub.requests = []
    _JudgeStub.stop = "end_turn"
    server = HTTPServer(("127.0.0.1", 0), _JudgeStub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


PIPELINE = Pipeline.model_validate(
    {
        "name": "ts-judge-e2e",
        "version": "0.1.0",
        "source_skill": "tests/fixtures/ts-judge-e2e",
        "description": "One judge.",
        "input": {"type": "Essay", "required": ["essay"], "optional": []},
        "nodes": [
            {
                "id": "grade_essay",
                "kind": "llm_judge",
                "description": "Grade an essay.",
                "inputs": {"essay": "pipeline.input.essay"},
                "signature_spec": {
                    "input_schema": {
                        "type": "object",
                        "properties": {"essay": {"type": "string"}},
                        "required": ["essay"],
                    },
                    "output_schema": OUTPUT_SCHEMA,
                    "prompt": "Grade this essay: {{ essay }}",
                },
            }
        ],
        "edges": [],
        "entry_nodes": ["grade_essay"],
        "exit_nodes": ["grade_essay"],
    }
)

DRIVER_TS = """\
import { gradeEssay } from "./signatures/grade_essay";

async function main(): Promise<void> {
    try {
        const result = await gradeEssay({ essay: "An essay." }, {
            ANTHROPIC_API_KEY: "local-stub-key",
            ROTE_MODEL_GRADE_ESSAY: process.env.ROTE_MODEL_GRADE_ESSAY,
            ROTE_BASE_URL_GRADE_ESSAY: process.env.ROTE_BASE_URL_GRADE_ESSAY,
            ROTE_EFFORT_GRADE_ESSAY: process.env.ROTE_EFFORT_GRADE_ESSAY,
            ROTE_JUDGE_EFFORT: process.env.ROTE_JUDGE_EFFORT,
        });
        console.log(JSON.stringify({ output: result }));
    } catch (err) {
        console.log(JSON.stringify({ error: (err as Error).message }));
    }
}

void main();
"""


@pytest.fixture(scope="module")
def judge_app(tmp_path_factory: pytest.TempPathFactory) -> Path:
    app = tmp_path_factory.mktemp("ts-judge") / "app"
    get_adapter("dbos-ts").emit(PIPELINE, app)
    (app / "src" / "driver.ts").write_text(DRIVER_TS, encoding="utf-8")
    install = subprocess.run(
        ["npm", "install", "--no-audit", "--no-fund"],
        cwd=app,
        capture_output=True,
        text=True,
        timeout=600,
        env={**os.environ, "npm_config_progress": "false"},
    )
    assert install.returncode == 0, f"npm install failed:\n{install.stdout}\n{install.stderr}"
    build = subprocess.run(["npx", "tsc"], cwd=app, capture_output=True, text=True, timeout=300)
    assert build.returncode == 0, f"tsc failed:\n{build.stdout}\n{build.stderr}"
    return app


def _run(app: Path, stub: str, **knobs: str) -> dict[str, Any]:
    proc = subprocess.run(
        ["node", "dist/driver.js"],
        cwd=app,
        capture_output=True,
        text=True,
        timeout=120,
        env={
            # An ambient effort knob must not decide what these tests see.
            **{
                k: v
                for k, v in os.environ.items()
                if not k.startswith(("ROTE_JUDGE_EFFORT", "ROTE_JUDGE_EFFORT"))
            },
            "ROTE_MODEL_GRADE_ESSAY": CURRENT_SONNET,
            "ROTE_BASE_URL_GRADE_ESSAY": stub,
            # Anything that ignored the override would reach a closed port.
            "ANTHROPIC_BASE_URL": "http://127.0.0.1:9",
            **knobs,
        },
    )
    assert proc.returncode == 0, f"driver failed:\n{proc.stdout}\n{proc.stderr}"
    result: dict[str, Any] = json.loads(proc.stdout.strip().splitlines()[-1])
    return result


def test_ts_judge_on_the_current_sonnet_is_schema_locked(judge_app: Path, judge_stub: str) -> None:
    assert _run(judge_app, judge_stub) == {"output": ANSWER}
    (request,) = _JudgeStub.requests
    assert request["model"] == CURRENT_SONNET
    assert "tools" not in request
    assert "tool_choice" not in request
    # The rewrite computed at emit time, by the same function the Python
    # helper runs per request.
    sent = request["output_config"]["format"]["schema"]["properties"]
    assert sent["verdict"]["enum"] == ["pass", "fail"]
    assert sent["rubric"]["const"] == "v1"
    assert request["output_config"] == {
        "format": {
            "type": "json_schema",
            "schema": structured_output_schema(OUTPUT_SCHEMA, node_id="grade_essay"),
        }
    }


@pytest.mark.parametrize("stop", ["refusal", "max_tokens"])
def test_ts_judge_rejects_an_incomplete_answer(judge_app: Path, judge_stub: str, stop: str) -> None:
    _JudgeStub.stop = stop
    assert _run(judge_app, judge_stub) == {
        "error": f"structured output for grade_essay ended with stop_reason {stop}, "
        "not end_turn, so it is not a complete answer"
    }


@pytest.mark.parametrize(
    ("knobs", "sent"),
    [
        ({}, None),
        ({"ROTE_JUDGE_EFFORT": "low"}, "low"),
        ({"ROTE_JUDGE_EFFORT": "low", "ROTE_EFFORT_GRADE_ESSAY": "medium"}, "medium"),
        ({"ROTE_JUDGE_EFFORT": "low", "ROTE_EFFORT_GRADE_ESSAY": ""}, None),
    ],
)
def test_ts_judge_effort_is_the_same_operator_knob(
    judge_app: Path, judge_stub: str, knobs: dict[str, str], sent: str | None
) -> None:
    assert _run(judge_app, judge_stub, **knobs) == {"output": ANSWER}
    config = _JudgeStub.requests[0]["output_config"]
    assert config.get("effort") == sent
    assert ("effort" in config) is (sent is not None)
