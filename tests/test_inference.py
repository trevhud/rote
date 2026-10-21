"""The judge inference helper: provider selection and both transports.

This is where the *vendor call* is tested now. Emission tests assert
that judges delegate here; these assert what happens when they do.
"""

from __future__ import annotations

import json
import pickle
import sys
import threading
import types
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

from rote.adapters import ADAPTERS
from rote.inference import _runtime_helper as helper
from rote.ir import Node, NodeKind, Pipeline
from tests._helpers import mini_pipeline
from tests._messages_api import CURRENT_SONNET, reject_like_the_api, send_json, structured_message

# ───────── fixtures / doubles ─────────


class _FakeAPIError(Exception):
    """Shaped like the vendor SDKs' error classes: reconstructible only
    with keyword-only ``response`` / ``body``, so pickle cannot rebuild it
    from ``BaseException.__reduce__``'s ``(cls, args)``. Module-level so
    it is picklable at all — the failure under test is on *load*."""

    def __init__(self, message: str, *, response: object, body: object) -> None:
        super().__init__(message)
        self.response = response
        self.body = body
        self.status_code = 400


def _fake_anthropic(create: Any) -> types.ModuleType:
    module = types.ModuleType("anthropic")
    module.APIError = _FakeAPIError  # type: ignore[attr-defined]
    module.Anthropic = lambda **kwargs: types.SimpleNamespace(  # type: ignore[attr-defined]
        _kwargs=kwargs, messages=types.SimpleNamespace(create=create)
    )
    return module


#: A closed output schema: what structured outputs accepts as is.
_GRADE = {
    "type": "object",
    "properties": {"grade": {"type": "integer"}},
    "required": ["grade"],
    "additionalProperties": False,
}


def _structured_response(payload: dict[str, Any]) -> Any:
    return types.SimpleNamespace(
        stop_reason="end_turn",
        content=[types.SimpleNamespace(type="text", text=json.dumps(payload))],
        usage=types.SimpleNamespace(
            input_tokens=11,
            output_tokens=7,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
    )


@pytest.fixture(autouse=True)
def _no_ambient_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralize the developer's own machine: a real ``claude`` binary or
    a real API key in the environment must not decide these tests."""
    for var in (
        "ROTE_INFERENCE",
        "ROTE_INFERENCE_GRADE_ESSAY",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "OPENAI_API_KEY",
        "ROTE_CLOUD_URL",
        "ROTE_CLOUD_TOKEN",
        "ROTE_USAGE_LOG",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(helper.shutil, "which", lambda _name: None)


# ───────── provider selection ─────────


def test_workload_shape_decides_the_lane_not_just_price(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The orders are inverses, and that is the design.

    A judge is one structured call, so the CLI's process spawn plus agent
    harness is pure overhead (measured 3.6-5.0s vs ~1.3s) — a key serves
    it. An agent loop amortizes that same overhead over every turn and is
    the expensive part of a pipeline, so the subscription serves it.
    """
    monkeypatch.setattr(helper.shutil, "which", lambda _name: "/usr/bin/claude")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-whatever")
    assert helper.select_provider("grade_essay", "anthropic") == "api"
    assert helper.select_provider("research", "anthropic", workload="agent") == "claude-cli"


def test_subscription_serves_judges_when_there_is_no_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cliff this whole feature exists to remove: a subscriber with no
    API key must still be able to run what they graduated."""
    monkeypatch.setattr(helper.shutil, "which", lambda _name: "/usr/bin/claude")
    assert helper.select_provider("grade_essay", "anthropic") == "claude-cli"


def test_an_agent_loop_with_local_tools_cannot_use_the_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`claude -p` reaches tools only over MCP, so loop_body sub-nodes
    bound as in-process callables push the loop to the SDK runner."""
    monkeypatch.setattr(helper.shutil, "which", lambda _name: "/usr/bin/claude")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-whatever")
    provider = helper.select_provider("lead_gen", "anthropic", workload="agent", local_tools=True)
    assert provider == "api"


def test_api_wins_when_no_cli_is_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A deployed image has no `claude` binary, so the first candidate
    drops out on its own — no separate 'deployed' branch needed."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-whatever")
    assert helper.select_provider("grade_essay", "anthropic") == "api"


def test_rote_cloud_is_the_last_resort(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cred = tmp_path / "cloud.json"
    cred.write_text(json.dumps({"base_url": "https://cloud.example", "token": "rote_x"}))
    monkeypatch.setenv("ROTE_CLOUD_CRED_PATH", str(cred))
    assert helper.select_provider("grade_essay", "anthropic") == "rote-cloud"


def test_an_explicit_endpoint_rules_out_the_subscription(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The proven gateway config (ROTE_BASE_URL_<ID> + a bearer token)
    must not be silently ignored just because `claude` is on PATH — the
    operator pointed this judge somewhere on purpose."""
    monkeypatch.setattr(helper.shutil, "which", lambda _name: "/usr/bin/claude")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "cf-token")
    provider = helper.select_provider(
        "grade_essay", "anthropic", base_url="https://gateway.example/ai"
    )
    assert provider == "api"


def test_a_claude_subscription_cannot_serve_an_openai_judge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(helper.shutil, "which", lambda _name: "/usr/bin/claude")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    assert helper.select_provider("grade_essay", "openai") == "api"


def test_explicit_choice_beats_auto_detect(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(helper.shutil, "which", lambda _name: "/usr/bin/claude")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-whatever")
    monkeypatch.setenv("ROTE_INFERENCE", "api")
    assert helper.select_provider("grade_essay", "anthropic") == "api"


def test_per_node_choice_beats_the_global_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(helper.shutil, "which", lambda _name: "/usr/bin/claude")
    monkeypatch.setenv("ROTE_INFERENCE", "api")
    monkeypatch.setenv("ROTE_INFERENCE_GRADE_ESSAY", "claude-cli")
    assert helper.select_provider("grade_essay", "anthropic") == "claude-cli"


def test_an_unavailable_explicit_choice_is_an_error_not_a_downgrade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Never silently bill a lane the operator did not pick. Falling back
    from `claude-cli` to `api` here would move the cost from a flat
    subscription onto a metered key without anyone saying so."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-whatever")
    monkeypatch.setenv("ROTE_INFERENCE", "claude-cli")  # not on PATH
    with pytest.raises(RuntimeError, match="cannot serve grade_essay"):
        helper.select_provider("grade_essay", "anthropic")


def test_an_unknown_provider_name_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROTE_INFERENCE", "free-lunch")
    with pytest.raises(RuntimeError, match="unknown inference provider"):
        helper.select_provider("grade_essay", "anthropic")


def test_nothing_available_lists_every_setup_step() -> None:
    with pytest.raises(RuntimeError) as excinfo:
        helper.select_provider("grade_essay", "anthropic")
    message = str(excinfo.value)
    assert "claude login" in message
    assert "ANTHROPIC_API_KEY" in message
    assert "rote login" in message


# ───────── SDK transport ─────────


def test_judge_translates_sdk_errors_so_they_survive_a_durable_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A durable runtime persists a failed step by pickling the exception,
    and the vendor SDKs' error classes take keyword-only ``response`` /
    ``body`` arguments — pickle cannot rebuild them. Raising one straight
    out of a judge replaced a real 400 (rejecting ``temperature``) with a
    bare ``TypeError`` at the fan-out join, hiding the cause entirely.
    """
    raised = _FakeAPIError("temperature is not supported on this model", response=object(), body={})
    with pytest.raises(TypeError):  # the bug this translation exists to avoid
        pickle.loads(pickle.dumps(raised))

    def _boom(**_kwargs: object) -> object:
        raise raised

    monkeypatch.setitem(sys.modules, "anthropic", _fake_anthropic(_boom))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-whatever")

    with pytest.raises(RuntimeError) as excinfo:
        helper.call_judge(
            node_id="grade_essay",
            client="anthropic",
            model="claude-sonnet-5",
            prompt="Grade it.",
            output_schema=_GRADE,
        )

    message = str(excinfo.value)
    assert "APIError" in message
    assert "HTTP 400" in message
    assert "temperature is not supported" in message
    assert "via api" in message  # which lane was billed for the failure
    # The whole point: this one crosses the boundary intact.
    assert str(pickle.loads(pickle.dumps(excinfo.value))) == message


def test_api_provider_leaves_credentials_to_the_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """The proven Cloudflare AI Gateway config works because the SDK does
    its own env resolution — passing a credential explicitly here would
    suppress that and break it."""
    seen: dict[str, Any] = {}

    def _create(**kwargs: Any) -> Any:
        return _structured_response({"grade": 9})

    module = _fake_anthropic(_create)

    def _ctor(**kwargs: Any) -> Any:
        seen.update(kwargs)
        return types.SimpleNamespace(messages=types.SimpleNamespace(create=_create))

    module.Anthropic = _ctor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", module)
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "cf-token")

    helper.call_judge(
        node_id="grade_essay",
        client="anthropic",
        model="anthropic/claude-sonnet-5",
        base_url="https://gateway.example/ai",
        prompt="Grade it.",
        output_schema=_GRADE,
    )
    assert seen == {"base_url": "https://gateway.example/ai"}


def test_rote_cloud_passes_its_token_explicitly_so_no_key_rides_along(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An explicit ``auth_token`` also suppresses the SDK's credential env
    reads ("explicit ctor args are total"), which is what keeps a user's
    ambient ANTHROPIC_API_KEY out of a request to our proxy."""
    cred = tmp_path / "cloud.json"
    cred.write_text(json.dumps({"base_url": "https://cloud.example/", "token": "rote_secret"}))
    monkeypatch.setenv("ROTE_CLOUD_CRED_PATH", str(cred))
    monkeypatch.setenv("ROTE_INFERENCE", "rote-cloud")

    seen: dict[str, Any] = {}

    def _create(**kwargs: Any) -> Any:
        return _structured_response({"grade": 9})

    module = _fake_anthropic(_create)

    def _ctor(**kwargs: Any) -> Any:
        seen.update(kwargs)
        return types.SimpleNamespace(messages=types.SimpleNamespace(create=_create))

    module.Anthropic = _ctor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", module)

    result = helper.call_judge(
        node_id="grade_essay",
        client="anthropic",
        model="claude-sonnet-5",
        prompt="Grade it.",
        output_schema=_GRADE,
    )
    assert result == {"grade": 9}
    assert seen == {
        "base_url": "https://cloud.example/v1/inference/anthropic",
        "auth_token": "rote_secret",
    }


def test_usage_log_records_the_provider_that_served_the_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The same token count costs differently per billing lane, so a
    subscription trial and a key trial are not interchangeable rows."""
    log = tmp_path / "usage.jsonl"
    monkeypatch.setenv("ROTE_USAGE_LOG", str(log))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-whatever")
    monkeypatch.setitem(
        sys.modules,
        "anthropic",
        _fake_anthropic(lambda **_kwargs: _structured_response({"grade": 9})),
    )

    helper.call_judge(
        node_id="grade_essay",
        client="anthropic",
        model="claude-sonnet-5",
        prompt="Grade it.",
        output_schema=_GRADE,
    )
    record = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
    assert record == {
        "node": "grade_essay",
        "provider": "api",
        "model": "claude-sonnet-5",
        "input_tokens": 11,
        "output_tokens": 7,
        "cache_write_tokens": 0,
        "cache_read_tokens": 0,
    }


# ───────── CLI transport ─────────


def _cli_envelope(**overrides: Any) -> str:
    envelope = {
        "is_error": False,
        "structured_output": {"grade": 9},
        "result": '{"grade": 9}',
        "usage": {"input_tokens": 740, "output_tokens": 638},
    }
    envelope.update(overrides)
    return json.dumps(envelope)


def test_cli_transport_asks_for_schema_locked_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``--json-schema`` makes ``claude -p`` do a forced tool call, so the
    subscription path has the same structural guarantee as the SDK path.
    The prompt goes on stdin: judge prompts carry whole documents, and
    argv is both size-limited and visible in ``ps``."""
    monkeypatch.setattr(helper.shutil, "which", lambda _name: "/usr/bin/claude")
    seen: dict[str, Any] = {}

    def _run(command: list[str], **kwargs: Any) -> Any:
        seen["command"] = command
        seen["input"] = kwargs["input"]
        seen["env"] = kwargs["env"]
        return types.SimpleNamespace(stdout=_cli_envelope(), stderr="", returncode=0)

    monkeypatch.setattr(helper.subprocess, "run", _run)
    schema = {"type": "object", "properties": {"grade": {"type": "integer"}}}

    result = helper.call_judge(
        node_id="grade_essay",
        client="anthropic",
        model="anthropic/claude-sonnet-5",
        prompt="Grade it.",
        output_schema=schema,
    )

    assert result == {"grade": 9}
    assert seen["input"] == "Grade it."
    command = seen["command"]
    assert "--json-schema" in command
    assert json.loads(command[command.index("--json-schema") + 1]) == schema
    # A gateway-qualified id means nothing to the CLI.
    assert command[command.index("--model") + 1] == "claude-sonnet-5"
    # A judge reasons and answers; it must not read files or run commands.
    # The empty value is load-bearing: the flag is variadic, so a bare
    # `--tools` makes the CLI exit 1 before it ever runs the judge.
    assert command[command.index("--tools") + 1] == ""
    assert command[command.index("--setting-sources") + 1] == ""


def test_cli_transport_forces_subscription_billing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole reason this provider exists: in ``claude -p`` these env
    vars beat an active OAuth session, so a stray key would silently move
    the bill from the subscription to per-token billing."""
    monkeypatch.setattr(helper.shutil, "which", lambda _name: "/usr/bin/claude")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-whatever")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "bearer-whatever")
    monkeypatch.setenv("ROTE_INFERENCE", "claude-cli")
    seen: dict[str, Any] = {}

    def _run(command: list[str], **kwargs: Any) -> Any:
        seen["env"] = kwargs["env"]
        return types.SimpleNamespace(stdout=_cli_envelope(), stderr="", returncode=0)

    monkeypatch.setattr(helper.subprocess, "run", _run)
    helper.call_judge(
        node_id="grade_essay",
        client="anthropic",
        model="claude-sonnet-5",
        prompt="Grade it.",
        output_schema={"type": "object", "properties": {}},
    )
    assert "ANTHROPIC_API_KEY" not in seen["env"]
    assert "ANTHROPIC_AUTH_TOKEN" not in seen["env"]


def test_cli_transport_surfaces_the_envelopes_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """``claude -p`` reports API failures inside a successful exit — the
    returncode alone would call this a win."""
    monkeypatch.setattr(helper.shutil, "which", lambda _name: "/usr/bin/claude")
    monkeypatch.setattr(
        helper.subprocess,
        "run",
        lambda command, **kwargs: types.SimpleNamespace(
            stdout=_cli_envelope(is_error=True, result="model not found"),
            stderr="",
            returncode=0,
        ),
    )
    with pytest.raises(RuntimeError, match="model not found"):
        helper.call_judge(
            node_id="grade_essay",
            client="anthropic",
            model="claude-sonnet-5",
            prompt="Grade it.",
            output_schema={"type": "object", "properties": {}},
        )


def test_cli_transport_falls_back_to_the_result_string(monkeypatch: pytest.MonkeyPatch) -> None:
    """``structured_output`` is the documented field, but the same JSON is
    also in ``result``; a missing key should not lose a paid answer."""
    monkeypatch.setattr(helper.shutil, "which", lambda _name: "/usr/bin/claude")
    envelope = json.loads(_cli_envelope())
    del envelope["structured_output"]
    monkeypatch.setattr(
        helper.subprocess,
        "run",
        lambda command, **kwargs: types.SimpleNamespace(
            stdout=json.dumps(envelope), stderr="", returncode=0
        ),
    )
    assert helper.call_judge(
        node_id="grade_essay",
        client="anthropic",
        model="claude-sonnet-5",
        prompt="Grade it.",
        output_schema={"type": "object", "properties": {}},
    ) == {"grade": 9}


# ───────── schema-locked judges: output_config.format ─────────


class _JudgeStub(BaseHTTPRequestHandler):
    """A Messages endpoint that refuses what the real API refuses, then
    answers ``answer`` as structured output, ending on ``stop``."""

    requests: list[dict[str, Any]] = []
    answer: Any = {"grade": 9}
    stop = "end_turn"

    def do_POST(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        payload = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
        type(self).requests.append(payload)
        if reject_like_the_api(self, payload):
            return
        send_json(self, structured_message(payload, type(self).answer, stop_reason=type(self).stop))

    def log_message(self, *args: object) -> None:
        return


@pytest.fixture()
def judge_stub(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    monkeypatch.setenv("ROTE_INFERENCE", "api")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    _JudgeStub.requests = []
    _JudgeStub.answer = {"grade": 9}
    _JudgeStub.stop = "end_turn"
    server = HTTPServer(("127.0.0.1", 0), _JudgeStub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


def _judge(stub: str, schema: dict[str, Any] = _GRADE) -> dict[str, Any]:
    return helper.call_judge(
        node_id="grade_essay",
        client="anthropic",
        model=CURRENT_SONNET,
        base_url=stub,
        prompt="Grade it.",
        output_schema=schema,
    )


def test_a_judge_on_the_current_sonnet_is_schema_locked_without_forcing_a_tool(
    judge_stub: str,
) -> None:
    """claude-sonnet-5-5 400s on tool_choice "tool"/"any", so a forced
    tool call made every judge dead on arrival there. The stub refuses
    forcing the way the API does; output_config.format is the mechanism."""
    assert _judge(judge_stub) == {"grade": 9}
    (request,) = _JudgeStub.requests
    assert request["model"] == CURRENT_SONNET
    assert request["output_config"] == {"format": {"type": "json_schema", "schema": _GRADE}}
    assert "tools" not in request
    assert "tool_choice" not in request


@pytest.mark.parametrize(
    ("env", "sent"),
    [
        ({}, None),
        ({"ROTE_JUDGE_EFFORT": "low"}, "low"),
        ({"ROTE_JUDGE_EFFORT": "low", "ROTE_EFFORT_GRADE_ESSAY": "medium"}, "medium"),
        # Set but empty omits the parameter, even over a pipeline-wide value.
        ({"ROTE_JUDGE_EFFORT": "low", "ROTE_EFFORT_GRADE_ESSAY": ""}, None),
        ({"ROTE_EFFORT_OTHER_NODE": "low"}, None),
        # The compiler's knob (rote compile --effort) never reaches a judge.
        ({"ROTE_EFFORT": "low"}, None),
    ],
)
def test_judge_effort_is_an_operator_knob(
    judge_stub: str, monkeypatch: pytest.MonkeyPatch, env: dict[str, str], sent: str | None
) -> None:
    """Unset sends no effort: measured 2026-10-20 on 8 example judges, low
    effort changed decisions the default made consistently (it dropped an
    ARR stated in the message and missed a deploy as the cause), so the
    API's own default (high) stays the default."""
    for var in (
        "ROTE_JUDGE_EFFORT",
        "ROTE_EFFORT",
        "ROTE_EFFORT_GRADE_ESSAY",
        "ROTE_EFFORT_OTHER_NODE",
    ):
        monkeypatch.delenv(var, raising=False)
    for var, value in env.items():
        monkeypatch.setenv(var, value)
    _judge(judge_stub)
    config = _JudgeStub.requests[0]["output_config"]
    assert config.get("effort") == sent
    assert ("effort" in config) is (sent is not None)


def test_a_judge_sends_the_rewritten_schema(judge_stub: str) -> None:
    """A nested object without additionalProperties and a bounded integer
    are both 400s as written; the request carries the rewrite."""
    schema = {
        "type": "object",
        "properties": {
            "grade": {"type": "integer", "minimum": 1, "maximum": 10},
            "notes": {"type": "object", "properties": {"text": {"type": "string"}}},
        },
        "required": ["grade"],
    }
    _JudgeStub.answer = {"grade": 9, "notes": {"text": "fine"}}
    assert _judge(judge_stub, schema) == {"grade": 9, "notes": {"text": "fine"}}
    sent = _JudgeStub.requests[0]["output_config"]["format"]["schema"]
    assert sent == helper.structured_output_schema(schema, node_id="grade_essay")
    assert sent["properties"]["notes"]["additionalProperties"] is False
    assert "minimum" not in sent["properties"]["grade"]


@pytest.mark.parametrize("stop", ["refusal", "max_tokens"])
def test_an_incomplete_judge_answer_is_a_durable_error_never_data(
    judge_stub: str, stop: str
) -> None:
    _JudgeStub.stop = stop
    with pytest.raises(RuntimeError) as excinfo:
        _judge(judge_stub)
    message = str(excinfo.value)
    assert message == (
        f"structured output for grade_essay ended with stop_reason {stop!r}, not end_turn, "
        "so it is not a complete answer"
    )
    assert str(pickle.loads(pickle.dumps(excinfo.value))) == message


def test_a_judge_answer_that_is_not_an_object_is_an_error(judge_stub: str) -> None:
    _JudgeStub.answer = ["grade", 9]
    with pytest.raises(
        RuntimeError, match="structured output for grade_essay is not a JSON object"
    ):
        _judge(judge_stub)


def test_an_api_refusal_of_the_schema_crosses_a_durable_step(
    judge_stub: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What the API 400s on reaches the user with its message, as a
    RuntimeError that survives pickling. The normalizer is bypassed so
    the stub's schema check, standing in for the API, sees the original."""
    monkeypatch.setattr(helper, "structured_output_schema", lambda schema, *, node_id: schema)
    with pytest.raises(RuntimeError) as excinfo:
        _judge(judge_stub, {**_GRADE, "additionalProperties": True})
    message = str(excinfo.value)
    assert "HTTP 400" in message
    assert "'additionalProperties' must be explicitly set to false" in message
    assert str(pickle.loads(pickle.dumps(excinfo.value))) == message


def test_a_judge_schema_structured_outputs_cannot_express_fails_before_any_request(
    judge_stub: str,
) -> None:
    with pytest.raises(RuntimeError, match=r"grade_essay: output schema at #/properties/extra"):
        _judge(
            judge_stub,
            {**_GRADE, "properties": {**_GRADE["properties"], "extra": {"type": "object"}}},
        )
    assert _JudgeStub.requests == []


# ───────── the verbatim-copy contract ─────────


def _judge_pipeline() -> Pipeline:
    return mini_pipeline(
        Node(
            id="grade_essay",
            kind=NodeKind.LLM_JUDGE,
            description="Grade an essay.",
            inputs={"essay": "pipeline.input.essay"},
            signature_spec={
                "input_schema": {
                    "type": "object",
                    "properties": {"essay": {"type": "string"}},
                    "required": ["essay"],
                },
                "output_schema": {
                    "type": "object",
                    "properties": {"grade": {"type": "integer"}},
                    "required": ["grade"],
                },
                "prompt": "Grade: {{ essay }}",
            },
        )
    )


@pytest.mark.parametrize("runtime", ["dbos", "python", "temporal"])
def test_emitted_helper_is_the_source_module_byte_for_byte(runtime: str, tmp_path: Path) -> None:
    """Same contract ``extracted/_rote_mcp.py`` has: the emitted file IS
    the module's source. A hand-edited copy would be a second
    implementation of the billing decision, drifting silently."""
    out = tmp_path / runtime
    ADAPTERS[runtime]().emit(_judge_pipeline(), out)
    emitted = (out / "signatures" / "_rote_inference.py").read_text(encoding="utf-8")
    assert emitted == Path(helper.__file__).read_text(encoding="utf-8")


def test_the_helper_imports_nothing_but_stdlib_at_module_level() -> None:
    """It executes inside emitted apps where rote is not installed, and
    where the vendor SDKs may not be installed either — that constraint
    is what makes the verbatim copy safe rather than a fork."""
    import ast

    tree = ast.parse(Path(helper.__file__).read_text(encoding="utf-8"))
    roots: list[str] = []
    for node in tree.body:  # module level only; lazy imports live in functions
        if isinstance(node, ast.Import):
            roots += [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            roots.append(node.module.split(".")[0])
    assert set(roots) <= set(sys.stdlib_module_names) | {"__future__"}


# ───────── CLI transport: agent loops ─────────


def _agent_envelope(**overrides: Any) -> str:
    envelope = {
        "is_error": False,
        "result": "Researched 3 accounts.",
        "num_turns": 4,
        "usage": {"input_tokens": 900, "output_tokens": 120},
    }
    envelope.update(overrides)
    return json.dumps(envelope)


def _capture_cli(
    monkeypatch: pytest.MonkeyPatch,
    envelope: str,
    stderr: str = "",
    seen: dict[str, Any] | None = None,
) -> dict[str, Any]:
    monkeypatch.setattr(helper.shutil, "which", lambda _name: "/usr/bin/claude")
    seen = {} if seen is None else seen

    def _run(command: list[str], **kwargs: Any) -> Any:
        if "--version" in command:
            # The helper's zero-cost effort check; answered the way Claude
            # Code 2.1.304 does (see _stub_cli for the real-process version).
            level = command[command.index("--effort") + 1]
            seen.setdefault("probes", []).append(level)
            warning = "" if level in _CLI_LEVELS else _cli_effort_warning(level)
            return types.SimpleNamespace(stdout="2.1.304 (Claude Code)\n", stderr=warning)
        seen["command"] = command
        seen["input"] = kwargs["input"]
        return types.SimpleNamespace(stdout=envelope, stderr=stderr, returncode=0)

    monkeypatch.setattr(helper.subprocess, "run", _run)
    return seen


_CLI_LEVELS = ("low", "medium", "high", "xhigh", "max")


def _cli_effort_warning(level: str) -> str:
    return (
        f"Warning: Unknown --effort value '{level}' — ignoring it and using the "
        "default effort. Valid values: low, medium, high, xhigh, max.\n"
    )


@pytest.fixture(autouse=True)
def _fresh_effort_checks() -> Iterator[None]:
    """The helper caches effort checks per process; each test starts clean."""
    helper._CHECKED_EFFORTS.clear()
    yield
    helper._CHECKED_EFFORTS.clear()


def test_agent_loop_cli_never_passes_a_bare_tools_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--tools`` is variadic, so a bare flag exits 1 before the loop runs.

    This shipped broken and no test caught it: every emitted agent loop on
    the subscription lane died with "option '--tools <tools...>' argument
    missing" the moment no MCP server resolved — the common case on a
    machine with no registry. Asserting the flag's *value* is the point;
    asserting its presence is what let the bug through.
    """
    monkeypatch.setattr(helper, "_mcp_config_for", lambda *_a: helper.AgentMcp({}, [], [], {}))
    seen = _capture_cli(monkeypatch, _agent_envelope())

    result = helper.run_agent_loop(
        node_id="target_research",
        description="Research the account.",
        task='{"account": "acme"}',
        model="claude-sonnet-4-6",
        tools=["bright_data_search"],
        max_iterations=6,
    )

    assert result["result"] == "Researched 3 accounts."
    command = seen["command"]
    assert command[command.index("--tools") + 1] == ""
    # No servers resolved means no allowlist and no mcp-config to pass.
    assert "--allowedTools" not in command
    assert "--mcp-config" not in command
    # Bounded by the caller's cap, not the CLI's default.
    assert command[command.index("--max-turns") + 1] == "6"


def test_agent_loop_cli_wires_resolved_servers_and_allowlists_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With servers resolved, the allowlist — not the wiring — is the boundary.

    Over-wiring is safe precisely because ``--allowedTools`` constrains
    what the agent may call; the IR named the tools, so a server offering
    anything else is unreachable.
    """
    servers = {"vendor": {"type": "http", "url": "https://mcp.example.com/mcp"}}
    allowed = [("vendor", "bright_data_search")]
    monkeypatch.setattr(
        helper, "_mcp_config_for", lambda *_a: helper.AgentMcp(servers, allowed, [], {})
    )
    seen = _capture_cli(monkeypatch, _agent_envelope())

    helper.run_agent_loop(
        node_id="target_research",
        description="Research the account.",
        task='{"account": "acme"}',
        model="claude-sonnet-4-6",
        tools=["bright_data_search"],
        max_iterations=6,
    )

    command = seen["command"]
    assert command[command.index("--allowedTools") + 1] == "mcp__vendor__bright_data_search"
    assert json.loads(command[command.index("--mcp-config") + 1]) == {"mcpServers": servers}
    # `--tools ""` removes only the built-in coding tools; MCP tools are
    # unaffected by it, which is why the loop can carry both. Measured on
    # Claude Code 2.1.302: the declared MCP tool still bound and was
    # called, and the 31 built-in tools (31k tokens a turn) were gone.
    assert command[command.index("--tools") + 1] == ""
    # Only the servers passed above, never the user's own connectors.
    assert "--strict-mcp-config" in command


def test_resolved_tool_servers_narrow_the_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    """A tool resolved to one server must not be allowlisted on the others.

    Without the narrowing the allowlist is a full cross product: mostly
    phantom entries, and a real hazard when two servers export the same
    tool name — the agent may then call whichever it likes, which is the
    endpoint-swap this field exists to prevent.
    """
    registry = {
        "brightdata": {"url": "https://bd.example/mcp"},
        "internal": {"url": "https://int.example/mcp"},
    }

    class _FakeMcp:
        READ_ONLY_ENV = "ROTE_MCP_READ_ONLY"

        @staticmethod
        def _registry_servers() -> dict[str, dict[str, str]]:
            return registry

        @staticmethod
        def resolve_url(name: str, _pipeline_url: None) -> str:
            return registry[name]["url"]

    monkeypatch.setitem(sys.modules, "extracted", types.SimpleNamespace(_rote_mcp=_FakeMcp))
    monkeypatch.setitem(sys.modules, "extracted._rote_mcp", _FakeMcp)

    tools = ["web_search", "lookup_account"]
    servers, allowed, _removed, _withheld = helper._mcp_config_for(
        tools, {"web_search": "brightdata", "lookup_account": "internal"}
    )

    assert set(servers) == {"brightdata", "internal"}
    assert allowed == [("brightdata", "web_search"), ("internal", "lookup_account")]

    # Unresolved names still reach every wired server: there is nothing
    # better to go on, and a name no server provides never resolves anyway.
    unpinned = helper._mcp_config_for(tools, None).allowed
    assert len(unpinned) == 4


# ───────── The overhead-control flags, on BOTH CLI paths ─────────
#
# `claude -p` defaults to shipping Claude Code's entire coding-agent
# system prompt: ~37k cache-creation tokens and 11.5s for a one-sentence
# judge. `--system-prompt <one line>` + `--tools ""` + `--setting-sources
# ""` bring the identical call to ~740 tokens and ~3.5s.
#
# A mutation sweep found `--system-prompt` untested on BOTH paths and
# `--setting-sources` / `is_error` untested on the agent-loop path — the
# same asymmetry that let the bare `--tools` flag ship broken, since the
# agent-loop CLI path had no coverage at all.


def _cli_flag(command: list[str], flag: str) -> str:
    assert flag in command, f"{flag} missing from the invocation"
    return command[command.index(flag) + 1]


def test_judge_cli_replaces_the_default_coding_agent_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A judge must carry its own short system prompt.

    Asserting presence would pass with an empty value; asserting the
    exact text would break on any wording change. The load-bearing
    property is that it is *short and custom* — dropping the flag is
    what restores the 37k-token default.
    """
    monkeypatch.setattr(helper, "_mcp_config_for", lambda *_a: helper.AgentMcp({}, [], [], {}))
    seen = _capture_cli(monkeypatch, _cli_envelope(structured_output={"grade": 9}))

    helper.call_judge(
        node_id="grade_essay",
        client="anthropic",
        prompt="Grade it.",
        output_schema={"type": "object", "properties": {"grade": {"type": "integer"}}},
        model="claude-sonnet-4-6",
    )

    prompt = _cli_flag(seen["command"], "--system-prompt")
    assert prompt.strip(), "an empty system prompt restores Claude Code's default"
    assert len(prompt) < 400, (
        f"the judge system prompt is {len(prompt)} chars — it is meant to be a "
        f"one-liner; the whole point of the flag is not shipping a large prompt"
    )


def test_agent_loop_cli_replaces_the_default_coding_agent_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same contract on the agent-loop path, which had no coverage of it."""
    monkeypatch.setattr(helper, "_mcp_config_for", lambda *_a: helper.AgentMcp({}, [], [], {}))
    seen = _capture_cli(monkeypatch, _agent_envelope())

    helper.run_agent_loop(
        node_id="target_research",
        description="Research the account.",
        task='{"account": "acme"}',
        model="claude-sonnet-4-6",
        tools=["bright_data_search"],
        max_iterations=6,
    )

    prompt = _cli_flag(seen["command"], "--system-prompt")
    assert prompt.strip(), "an empty system prompt restores Claude Code's default"
    assert len(prompt) < 400


def test_agent_loop_cli_does_not_inherit_user_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--setting-sources ""` on the agent loop, as on the judge.

    Without it the loop picks up whatever hooks, MCP servers and
    permissions the *developer's* machine has configured — so an emitted
    pipeline behaves differently per laptop, which is the opposite of
    what compiling a skill is for.
    """
    monkeypatch.setattr(helper, "_mcp_config_for", lambda *_a: helper.AgentMcp({}, [], [], {}))
    seen = _capture_cli(monkeypatch, _agent_envelope())

    helper.run_agent_loop(
        node_id="target_research",
        description="Research the account.",
        task='{"account": "acme"}',
        model="claude-sonnet-4-6",
        tools=["bright_data_search"],
        max_iterations=6,
    )

    assert _cli_flag(seen["command"], "--setting-sources") == ""


def test_agent_loop_cli_surfaces_the_envelopes_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed agent loop must not read as a success.

    `claude -p` reports API failures *inside a zero exit* — `is_error`
    in the envelope — so returncode alone calls a dead loop a win and
    the workflow proceeds on a garbage result. Covered on the judge
    path; this is the loop half.
    """
    monkeypatch.setattr(helper, "_mcp_config_for", lambda *_a: helper.AgentMcp({}, [], [], {}))
    _capture_cli(monkeypatch, _agent_envelope(is_error=True, result="rate limited"))

    with pytest.raises(RuntimeError, match="rate limited"):
        helper.run_agent_loop(
            node_id="target_research",
            description="Research the account.",
            task='{"account": "acme"}',
            model="claude-sonnet-4-6",
            tools=["bright_data_search"],
            max_iterations=6,
        )


# ───────── iterations: the turns that ran, not the cap ─────────


def _logged(log: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]


def test_agent_loop_cli_reports_the_envelopes_turn_count(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`num_turns` is the CLI's own count, in the unit `--max-turns` caps.
    The loop's result is its declared output, so the count is logged."""
    log = tmp_path / "usage.jsonl"
    monkeypatch.setenv("ROTE_USAGE_LOG", str(log))
    monkeypatch.setattr(helper, "_mcp_config_for", lambda *_a: helper.AgentMcp({}, [], [], {}))
    _capture_cli(monkeypatch, _agent_envelope(num_turns=4))

    result = helper.run_agent_loop(
        node_id="target_research",
        description="Research the account.",
        task='{"account": "acme"}',
        model="claude-sonnet-4-6",
        max_iterations=9,
    )

    assert result == {"result": "Researched 3 accounts."}
    (record,) = _logged(log)
    assert (record["provider"], record["iterations"]) == ("claude-cli", 4)


def test_agent_loop_cli_without_a_turn_count_reports_none_not_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Unknown is left out, the TS Workers AI lane's answer. A 0 would read
    as a loop that never ran."""
    log = tmp_path / "usage.jsonl"
    monkeypatch.setenv("ROTE_USAGE_LOG", str(log))
    monkeypatch.setattr(helper, "_mcp_config_for", lambda *_a: helper.AgentMcp({}, [], [], {}))
    envelope = json.loads(_agent_envelope())
    del envelope["num_turns"]
    _capture_cli(monkeypatch, json.dumps(envelope))

    helper.run_agent_loop(
        node_id="target_research",
        description="Research the account.",
        task='{"account": "acme"}',
        model="claude-sonnet-4-6",
        max_iterations=9,
    )

    (record,) = _logged(log)
    assert "iterations" not in record


class _ToolLoopStub(BaseHTTPRequestHandler):
    """A Messages endpoint that asks for ``lookup`` twice, then answers."""

    requests: list[dict[str, Any]] = []

    def do_POST(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler's API
        payload = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
        type(self).requests.append(payload)
        if reject_like_the_api(self, payload):
            return
        turn = len(type(self).requests)
        message: dict[str, Any] = {
            "id": f"msg_{turn}",
            "type": "message",
            "role": "assistant",
            "model": payload["model"],
            "usage": {
                "input_tokens": 100 * turn,
                "output_tokens": 10 * turn,
                "cache_creation_input_tokens": 1000 * turn,
                "cache_read_input_tokens": 5000 * turn,
            },
        }
        if turn <= 2:
            message["stop_reason"] = "tool_use"
            message["content"] = [
                {
                    "type": "tool_use",
                    "id": f"toolu_{turn}",
                    "name": "lookup",
                    "input": {"account": f"acct-{turn}"},
                }
            ]
        else:
            message["stop_reason"] = "end_turn"
            message["content"] = [{"type": "text", "text": "Looked up 2 accounts."}]
        raw = json.dumps(message).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args: object) -> None:
        return


@pytest.fixture()
def tool_loop_stub() -> Iterator[str]:
    _ToolLoopStub.requests = []
    server = HTTPServer(("127.0.0.1", 0), _ToolLoopStub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


def test_agent_loop_sdk_reports_turns_run_and_usage_of_every_turn(
    monkeypatch: pytest.MonkeyPatch, tool_loop_stub: str, tmp_path: Path
) -> None:
    """The SDK lane used to return ``max_iterations`` as the turn count, so
    every loop looked saturated, and logged only the last request's usage.

    The real tool runner drives three turns against a local endpoint under
    a cap of ten, so the cap and the count cannot coincide.
    """
    monkeypatch.setenv("ROTE_INFERENCE", "api")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    usage_log = tmp_path / "usage.jsonl"
    monkeypatch.setenv("ROTE_USAGE_LOG", str(usage_log))
    monkeypatch.setattr(helper, "_mcp_config_for", lambda *_a: helper.AgentMcp({}, [], [], {}))
    looked_up: list[str] = []

    def lookup(account: str) -> str:
        """Look up one account."""
        looked_up.append(account)
        return json.dumps({"account": account, "tier": "gold"})

    result = helper.run_agent_loop(
        node_id="lead_gen",
        description="Look up the accounts.",
        task='{"accounts": ["acct-1", "acct-2"]}',
        model="claude-sonnet-4-6",
        local_tools={"lookup": lookup},
        max_iterations=10,
        base_url=tool_loop_stub,
    )

    assert result == {"result": "Looked up 2 accounts."}
    assert looked_up == ["acct-1", "acct-2"]
    assert len(_ToolLoopStub.requests) == 3
    record = json.loads(usage_log.read_text(encoding="utf-8"))
    assert (record["provider"], record["iterations"]) == ("api", 3)
    assert record["input_tokens"] == 100 + 200 + 300
    assert record["output_tokens"] == 10 + 20 + 30
    assert record["cache_write_tokens"] == 1000 + 2000 + 3000
    assert record["cache_read_tokens"] == 5000 + 10000 + 15000


# ───────── Context isolation and the four usage buckets, on BOTH paths ─────────
#
# `--tools "" --setting-sources ""` was not enough. Measured on Claude Code
# 2.1.302 with a real "say hi" judge through this helper: 30.3k tokens and
# $0.183 a cold call, because the user's claude.ai connectors and plugin MCP
# servers, plus the auto-memory index of the repository the pipeline ran
# from, were all still in the prompt. The full isolation set brought it to
# 1.25k. An agent loop carried 60k a turn; isolated, 1.4k with its declared
# MCP tool still bound.


def _run_cli_path(
    path: str,
    monkeypatch: pytest.MonkeyPatch,
    envelope: str,
    stderr: str = "",
    seen: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Drive one CLI path through the public entry point and capture argv."""
    if path == "agent_loop_with_servers":
        servers = {"vendor": {"type": "http", "url": "https://mcp.example.com/mcp"}}
        allowed = [("vendor", "bright_data_search")]
        monkeypatch.setattr(
            helper, "_mcp_config_for", lambda *_a: helper.AgentMcp(servers, allowed, [], {})
        )
    else:
        monkeypatch.setattr(helper, "_mcp_config_for", lambda *_a: helper.AgentMcp({}, [], [], {}))
    seen = _capture_cli(monkeypatch, envelope, stderr, seen)
    if path == "judge":
        helper.call_judge(
            node_id="grade_essay",
            client="anthropic",
            prompt="Grade it.",
            output_schema={"type": "object", "properties": {"grade": {"type": "integer"}}},
            model="claude-sonnet-4-6",
        )
    else:
        helper.run_agent_loop(
            node_id="target_research",
            description="Research the account.",
            task='{"account": "acme"}',
            model="claude-sonnet-4-6",
            tools=["bright_data_search"],
            max_iterations=6,
        )
    return seen


_CLI_PATHS = ["judge", "agent_loop_no_servers", "agent_loop_with_servers"]


@pytest.mark.parametrize("path", _CLI_PATHS)
def test_every_cli_path_isolates_the_child_from_the_users_setup(
    path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each flag here was measured by removing it; each one costs real tokens.

    Values, not presence: `--settings` must actually turn auto memory off,
    and `--tools` must be the empty string (a bare flag exits 1).
    """
    envelope = _cli_envelope() if path == "judge" else _agent_envelope()
    command = _run_cli_path(path, monkeypatch, envelope)["command"]

    assert "--strict-mcp-config" in command
    assert json.loads(_cli_flag(command, "--settings")) == {"autoMemoryEnabled": False}
    assert _cli_flag(command, "--setting-sources") == ""
    assert _cli_flag(command, "--tools") == ""
    # Strict MCP with a config means only the declared servers; without
    # one it means no MCP at all, which is right when nothing resolved.
    if path == "agent_loop_with_servers":
        assert set(json.loads(_cli_flag(command, "--mcp-config"))["mcpServers"]) == {"vendor"}
    else:
        assert "--mcp-config" not in command


_CACHED_USAGE = {
    # Shape of a real prompt-cached `claude -p` envelope: almost nothing in
    # input_tokens, the volume in the two cache buckets.
    "input_tokens": 3,
    "output_tokens": 92,
    "cache_creation_input_tokens": 1220,
    "cache_read_input_tokens": 28474,
}


_EFFORT_VARS = (
    "ROTE_JUDGE_EFFORT",
    "ROTE_EFFORT",
    "ROTE_EFFORT_GRADE_ESSAY",
    "ROTE_EFFORT_TARGET_RESEARCH",
    "ROTE_EFFORT_RESEARCH",
)


def _set_effort(monkeypatch: pytest.MonkeyPatch, env: dict[str, str]) -> None:
    for var in _EFFORT_VARS:
        monkeypatch.delenv(var, raising=False)
    for var, value in env.items():
        monkeypatch.setenv(var, value)


@pytest.mark.parametrize("path", _CLI_PATHS)
@pytest.mark.parametrize(
    ("env", "sent"),
    [
        ({}, None),
        ({"ROTE_JUDGE_EFFORT": "low"}, "low"),
        ({"ROTE_JUDGE_EFFORT": "low", "ROTE_EFFORT_{NODE}": "max"}, "max"),
        ({"ROTE_JUDGE_EFFORT": "low", "ROTE_EFFORT_{NODE}": ""}, None),
        ({"ROTE_EFFORT": "low"}, None),
    ],
)
def test_every_cli_path_passes_the_effort_knob(
    path: str, env: dict[str, str], sent: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On a subscription the knob is ``claude -p --effort``. The judge and
    agent-loop command lines are separate copies, so both are checked: the
    loop copy is the one that drifted every time before."""
    node = "GRADE_ESSAY" if path == "judge" else "TARGET_RESEARCH"
    _set_effort(monkeypatch, {var.replace("{NODE}", node): value for var, value in env.items()})
    envelope = _cli_envelope() if path == "judge" else _agent_envelope()
    command = _run_cli_path(path, monkeypatch, envelope)["command"]
    if sent is None:
        assert "--effort" not in command
    else:
        assert _cli_flag(command, "--effort") == sent
        assert command.count("--effort") == 1


@pytest.mark.parametrize("path", _CLI_PATHS)
def test_every_cli_path_refuses_an_unknown_level_before_paying_for_a_call(
    path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``claude -p`` warns and runs at its default on an unknown level,
    exit 0 (Claude Code 2.1.304), while the SDK lanes 400 on the same
    value. The level is checked with a free ``--version`` call first."""
    _set_effort(monkeypatch, {"ROTE_JUDGE_EFFORT": "bogus"})
    envelope = _cli_envelope() if path == "judge" else _agent_envelope()
    seen: dict[str, Any] = {}
    with pytest.raises(RuntimeError, match=r"Unknown --effort value 'bogus'"):
        _run_cli_path(path, monkeypatch, envelope, seen=seen)
    assert seen["probes"] == [helper._EFFORT_PROBE, "bogus"]
    assert "command" not in seen  # no paid call was made


@pytest.mark.parametrize("path", _CLI_PATHS)
def test_every_cli_path_refuses_a_run_that_says_it_ignored_its_effort(
    path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The run's own stderr is the last word, whatever the pre-check said."""
    _set_effort(monkeypatch, {"ROTE_JUDGE_EFFORT": "low"})
    envelope = _cli_envelope() if path == "judge" else _agent_envelope()
    with pytest.raises(RuntimeError, match=r"invalid effort 'low'"):
        _run_cli_path(path, monkeypatch, envelope, stderr="error: invalid effort 'low'\n")


@pytest.mark.parametrize("path", _CLI_PATHS)
def test_every_cli_path_logs_all_four_usage_buckets(
    path: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Logging `input_tokens` alone recorded a ~30k-token call as 3."""
    log = tmp_path / "usage.jsonl"
    monkeypatch.setenv("ROTE_USAGE_LOG", str(log))
    envelope = (
        _cli_envelope(usage=_CACHED_USAGE)
        if path == "judge"
        else _agent_envelope(usage=_CACHED_USAGE)
    )
    _run_cli_path(path, monkeypatch, envelope)

    record = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
    assert record["provider"] == "claude-cli"
    assert record["input_tokens"] == 3
    assert record["output_tokens"] == 92
    assert record["cache_write_tokens"] == 1220
    assert record["cache_read_tokens"] == 28474


def test_openai_usage_splits_cached_tokens_out_of_prompt_tokens() -> None:
    """OpenAI counts cached tokens inside prompt_tokens; the log must not
    count them twice once they also appear as cache_read."""
    usage = types.SimpleNamespace(
        prompt_tokens=1000,
        completion_tokens=50,
        prompt_tokens_details=types.SimpleNamespace(cached_tokens=800),
    )
    assert helper._openai_usage(usage) == {
        "input_tokens": 200,
        "output_tokens": 50,
        "cache_write_tokens": 0,
        "cache_read_tokens": 800,
    }


#: The ``modelUsage`` of a real ``claude -p --model sonnet`` judge call
#: (2026-10-18): the alias resolves to a concrete id, with the CLI's own
#: cost estimate. Its cache write is a 1h write, which bills at twice the
#: 5m rate a catalog publishes.
_REAL_MODEL_USAGE = {
    "claude-sonnet-5-5": {
        "inputTokens": 2,
        "outputTokens": 53,
        "cacheReadInputTokens": 0,
        "cacheCreationInputTokens": 1375,
        "costUSD": 0.006034,
        "canonicalModel": "claude-sonnet-5-5",
    }
}


@pytest.mark.parametrize("path", _CLI_PATHS)
def test_every_cli_path_logs_the_model_the_cli_actually_ran(
    path: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A row logged under an alias (``sonnet``) can never be priced. The
    envelope names the concrete model; log that, with the CLI's cost."""
    log = tmp_path / "usage.jsonl"
    monkeypatch.setenv("ROTE_USAGE_LOG", str(log))
    make = _cli_envelope if path == "judge" else _agent_envelope
    _run_cli_path(path, monkeypatch, make(modelUsage=_REAL_MODEL_USAGE))

    (record,) = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert record["model"] == "claude-sonnet-5-5"
    assert record["input_tokens"] == 2
    assert record["output_tokens"] == 53
    assert record["cache_write_tokens"] == 1375
    assert record["cache_read_tokens"] == 0
    assert record["cost_usd"] == 0.006034


def test_cli_logs_one_row_per_model_it_ran(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Each model is priced at its own rate, so a call that used two keeps
    two rows rather than one summed under the requested name."""
    log = tmp_path / "usage.jsonl"
    monkeypatch.setenv("ROTE_USAGE_LOG", str(log))
    usage = {
        **_REAL_MODEL_USAGE,
        "claude-haiku-5": {"inputTokens": 40, "outputTokens": 7, "costUSD": 0.0001},
    }
    _run_cli_path("judge", monkeypatch, _cli_envelope(modelUsage=usage))

    records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert [(r["model"], r["cost_usd"]) for r in records] == [
        ("claude-sonnet-5-5", 0.006034),
        ("claude-haiku-5", 0.0001),
    ]


def test_cli_without_model_usage_keeps_the_requested_id_and_no_cost(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No ``modelUsage``: nothing to resolve the alias from, so the row
    keeps the requested id and carries no cost for pricing to trust."""
    log = tmp_path / "usage.jsonl"
    monkeypatch.setenv("ROTE_USAGE_LOG", str(log))
    _run_cli_path("judge", monkeypatch, _cli_envelope())

    (record,) = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert record["model"] == "claude-sonnet-4-6"
    assert "cost_usd" not in record


# ───────── an agent loop returns its declared output ─────────

_DECLARED = {
    "type": "object",
    "properties": {"summary": {"type": "string"}, "score": {"type": "integer"}},
    "required": ["summary", "score"],
    "additionalProperties": False,
}


class _SubmitStub(BaseHTTPRequestHandler):
    """Turn 1 calls ``submit_result`` with ``answer``; turn 2 ends.

    With no ``answer`` the model never submits, and refuses the final
    schema-locked turn too.
    """

    requests: list[dict[str, Any]] = []
    answer: dict[str, Any] | None = None

    def do_POST(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler's API
        payload = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
        type(self).requests.append(payload)
        if reject_like_the_api(self, payload):
            return
        if "format" in payload.get("output_config", {}):
            send_json(self, structured_message(payload, None, stop_reason="refusal", content=[]))
            return
        first = len(type(self).requests) == 1
        if first and type(self).answer is not None:
            content: list[dict[str, Any]] = [
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "submit_result",
                    "input": type(self).answer,
                }
            ]
        else:
            content = [{"type": "text", "text": "All done."}]
        message = {
            "id": f"msg_{len(type(self).requests)}",
            "type": "message",
            "role": "assistant",
            "model": payload["model"],
            "stop_reason": "tool_use" if content[0]["type"] == "tool_use" else "end_turn",
            "content": content,
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        raw = json.dumps(message).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args: object) -> None:
        return


@pytest.fixture()
def submit_stub(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    monkeypatch.setenv("ROTE_INFERENCE", "api")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    monkeypatch.setattr(helper, "_mcp_config_for", lambda *_a: helper.AgentMcp({}, [], [], {}))
    _SubmitStub.requests = []
    server = HTTPServer(("127.0.0.1", 0), _SubmitStub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


def _loop(stub: str, **kwargs: Any) -> Any:
    kwargs.setdefault("output_schema", _DECLARED)
    return helper.run_agent_loop(
        node_id="research",
        description="Research the account.",
        task='{"account": "acme"}',
        model=CURRENT_SONNET,
        max_iterations=4,
        base_url=stub,
        **kwargs,
    )


def test_sdk_loop_returns_exactly_the_submitted_declared_keys(submit_stub: str) -> None:
    _SubmitStub.answer = {"summary": "acme is gold", "score": 7}
    assert _loop(submit_stub) == {"summary": "acme is gold", "score": 7}
    sent = {tool["name"]: tool for tool in _SubmitStub.requests[0]["tools"]}
    assert sent["submit_result"]["input_schema"] == _DECLARED


def test_sdk_loop_unwraps_a_single_type_output(submit_stub: str) -> None:
    schema = {
        "type": "object",
        "properties": {"result": {"type": "string"}},
        "required": ["result"],
        "additionalProperties": False,
    }
    _SubmitStub.answer = {"result": "the brief"}
    out = helper.run_agent_loop(
        node_id="research",
        description="Research.",
        task="{}",
        model="claude-sonnet-4-6",
        max_iterations=4,
        base_url=submit_stub,
        output_schema=schema,
        unwrap_key="result",
    )
    assert out == "the brief"


def test_sdk_loop_that_never_submits_fails_naming_the_node(submit_stub: str) -> None:
    _SubmitStub.answer = None
    with pytest.raises(RuntimeError, match="research ended with stop_reason 'refusal'"):
        _loop(submit_stub)


def test_sdk_loop_missing_a_declared_key_fails_naming_it(submit_stub: str) -> None:
    _SubmitStub.answer = {"summary": "acme is gold"}
    with pytest.raises(
        RuntimeError,
        match="agent loop research \\(api\\) did not hand back its declared output: "
        "output: missing key 'score'",
    ):
        _loop(submit_stub)


def test_cli_loop_asks_for_the_declared_output_as_structured_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The subscription lane gets the same contract through --json-schema."""
    monkeypatch.setattr(helper, "_mcp_config_for", lambda *_a: helper.AgentMcp({}, [], [], {}))
    seen = _capture_cli(
        monkeypatch,
        _agent_envelope(structured_output={"summary": "acme is gold", "score": 7}),
    )
    out = helper.run_agent_loop(
        node_id="research",
        description="Research the account.",
        task='{"account": "acme"}',
        model="claude-sonnet-4-6",
        max_iterations=4,
        output_schema=_DECLARED,
    )
    assert out == {"summary": "acme is gold", "score": 7}
    assert json.loads(_cli_flag(seen["command"], "--json-schema")) == _DECLARED


def test_a_loop_without_declared_output_returns_its_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(helper, "_mcp_config_for", lambda *_a: helper.AgentMcp({}, [], [], {}))
    seen = _capture_cli(monkeypatch, _agent_envelope())
    out = helper.run_agent_loop(
        node_id="research",
        description="Research.",
        task="{}",
        model="claude-sonnet-4-6",
        max_iterations=4,
    )
    assert out == {"result": "Researched 3 accounts."}
    assert "--json-schema" not in seen["command"]


class _ForceStub(BaseHTTPRequestHandler):
    """A model that does not submit on its own.

    ``first`` is the loop's first reply: ``"text"`` ends the turn, and
    ``"tool"`` calls ``lookup`` (which hits a cap of 1). A schema-locked
    request (``output_config``) gets ``forced_answer`` as JSON text, or
    ends on ``forced_stop`` with no answer when that is None.
    """

    requests: list[dict[str, Any]] = []
    first = "text"
    forced_answer: dict[str, Any] | None = None
    forced_stop = "refusal"

    def do_POST(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler's API
        payload = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
        type(self).requests.append(payload)
        if reject_like_the_api(self, payload):
            return
        last = payload["messages"][-1]["content"]
        asked = isinstance(last, list) and str(last[-1].get("text", "")).startswith(
            "Call submit_result now"
        )
        if asked and type(self).forced_answer is not None:
            submit = {
                "type": "tool_use",
                "id": "t_submit",
                "name": "submit_result",
                "input": type(self).forced_answer,
            }
            send_json(
                self, structured_message(payload, None, stop_reason="tool_use", content=[submit])
            )
            return
        if "format" in payload.get("output_config", {}):
            answer = type(self).forced_answer
            if answer is not None:
                reply = structured_message(payload, answer)
            else:
                partial = [{"type": "text", "text": '{"summary": "acme'}]
                reply = structured_message(
                    payload, None, stop_reason=type(self).forced_stop, content=partial
                )
            send_json(self, reply)
            return
        content: list[dict[str, Any]]
        if type(self).first == "tool":
            content = [{"type": "tool_use", "id": "t_lookup", "name": "lookup", "input": {}}]
        else:
            content = [{"type": "text", "text": "Done, I think."}]
        message = {
            "id": f"msg_{len(type(self).requests)}",
            "type": "message",
            "role": "assistant",
            "model": payload["model"],
            "stop_reason": "tool_use" if content[0]["type"] == "tool_use" else "end_turn",
            "content": content,
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        raw = json.dumps(message).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args: object) -> None:
        return


@pytest.fixture()
def force_stub(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[str]:
    monkeypatch.setenv("ROTE_INFERENCE", "api")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    monkeypatch.setenv("ROTE_USAGE_LOG", str(tmp_path / "usage.jsonl"))
    monkeypatch.setattr(helper, "_mcp_config_for", lambda *_a: helper.AgentMcp({}, [], [], {}))
    _ForceStub.requests = []
    _ForceStub.first = "text"
    _ForceStub.forced_answer = None
    _ForceStub.forced_stop = "refusal"
    server = HTTPServer(("127.0.0.1", 0), _ForceStub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


def test_a_loop_that_ends_without_submitting_gets_one_forced_turn(
    force_stub: str, tmp_path: Path
) -> None:
    """The judge mechanism, once: the answer schema-locked by
    ``output_config.format``. Forcing ``tool_choice`` instead is a 400 on
    the current Sonnet, which the stub enforces."""
    _ForceStub.forced_answer = {"summary": "acme is gold", "score": 7}
    assert _loop(force_stub) == {"summary": "acme is gold", "score": 7}

    loop_turn, forced = _ForceStub.requests
    assert "tool_choice" not in loop_turn
    assert "output_config" not in loop_turn
    assert forced["model"] == CURRENT_SONNET
    assert forced["tool_choice"] == {"type": "none"}
    assert forced["output_config"] == {"format": {"type": "json_schema", "schema": _DECLARED}}
    # The tools stay defined: the history holds tool_use blocks.
    assert {tool["name"] for tool in forced["tools"]} == {"submit_result"}
    # The turn the loop ended on is in the forced request, then the nudge.
    assert forced["messages"][-2]["content"][0]["text"] == "Done, I think."
    assert (
        forced["messages"][-1]["content"][-1]["text"] == "Reply now with the step's final answer."
    )
    (record,) = _logged(tmp_path / "usage.jsonl")
    assert record["iterations"] == 2  # the forced turn counts


def test_a_loop_at_its_cap_is_forced_without_rerunning_its_tools(force_stub: str) -> None:
    """At the cap the runner already ran the last turn's tool calls; the
    forced request carries their results and adds the nudge to that same
    user message, so no call runs twice."""
    _ForceStub.first = "tool"
    _ForceStub.forced_answer = {"summary": "s", "score": 1}
    calls: list[str] = []

    def lookup() -> str:
        """Look something up."""
        calls.append("lookup")
        return "found"

    out = helper.run_agent_loop(
        node_id="research",
        description="Research.",
        task="{}",
        model=CURRENT_SONNET,
        max_iterations=1,
        base_url=force_stub,
        local_tools={"lookup": lookup},
        output_schema=_DECLARED,
    )
    assert out == {"summary": "s", "score": 1}
    assert calls == ["lookup"]
    forced = _ForceStub.requests[-1]
    roles = [m["role"] for m in forced["messages"]]
    assert roles == ["user", "assistant", "user"]
    last = forced["messages"][-1]["content"]
    assert [block["type"] for block in last] == ["tool_result", "text"]


@pytest.mark.parametrize("stop", ["refusal", "max_tokens"])
def test_an_incomplete_forced_answer_fails_naming_the_stop_reason(
    force_stub: str, stop: str
) -> None:
    """A refusal or a truncation is not an answer, even when its text
    happens to start like one; the failure crosses a durable step."""
    _ForceStub.forced_stop = stop
    with pytest.raises(RuntimeError) as excinfo:
        _loop(force_stub)
    message = str(excinfo.value)
    assert message == (
        f"structured output for research ended with stop_reason {stop!r}, not end_turn, "
        "so it is not a complete answer"
    )
    assert str(pickle.loads(pickle.dumps(excinfo.value))) == message
    assert len(_ForceStub.requests) == 2


_UNTYPED = {
    "type": "object",
    "properties": {"result": {"description": "Value of type Account."}},
    "required": ["result"],
    "additionalProperties": False,
}


def test_an_output_no_schema_can_lock_gets_asked_once_more_never_forced(
    force_stub: str,
) -> None:
    """An output naming a type the IR never defines has no JSON type, so
    structured outputs cannot lock it. The last turn only asks for the
    submit call: no output_config the API would 400, and no forced
    tool_choice, which the current Sonnet would 400."""
    _ForceStub.forced_answer = {"result": {"name": "acme"}}
    assert _loop(force_stub, output_schema=_UNTYPED) == {"result": {"name": "acme"}}
    loop_turn, asked = _ForceStub.requests
    assert "output_config" not in asked
    assert "tool_choice" not in asked
    assert asked["messages"][-1]["content"][-1]["text"].startswith("Call submit_result now")


def test_a_malformed_output_schema_fails_rather_than_asking_for_submit_result(
    force_stub: str,
) -> None:
    """Only an inexpressible schema switches the last turn to asking for
    the submit tool. A malformed one is a bug in the emitted call and
    surfaces as one, before the extra request is sent."""
    malformed = {"type": "object", "properties": {"summary": 3}, "additionalProperties": False}
    with pytest.raises(RuntimeError, match=r"#/properties/summary is not a JSON Schema object"):
        _loop(force_stub, output_schema=malformed)
    assert len(_ForceStub.requests) == 1


def test_an_unlockable_output_never_submitted_fails_with_the_shared_message(
    force_stub: str,
) -> None:
    with pytest.raises(
        RuntimeError,
        match=(
            r"agent loop research \(api\) did not hand back its declared output: "
            r"it never called submit_result, even when asked once more"
        ),
    ):
        _loop(force_stub, output_schema=_UNTYPED)
    assert len(_ForceStub.requests) == 2


@pytest.mark.parametrize(
    ("answer", "reason"),
    [
        ({"summary": "s", "score": "7"}, "output.score: expected integer, got str"),
        ({"summary": "s", "score": True}, "output.score: expected integer, got bool"),
        ({"summary": "s", "score": 7, "extra": 1}, "output: unexpected key 'extra'"),
        ({"summary": None, "score": 7}, "output.summary: expected string, got NoneType"),
    ],
)
def test_a_submission_is_validated_against_the_declared_schema(
    submit_stub: str, answer: dict[str, Any], reason: str
) -> None:
    _SubmitStub.answer = answer
    with pytest.raises(RuntimeError) as excinfo:
        _loop(submit_stub)
    assert str(excinfo.value) == (
        f"agent loop research (api) did not hand back its declared output: {reason}"
    )


def test_an_sdk_loop_runs_at_the_configured_effort(
    submit_stub: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_effort(monkeypatch, {"ROTE_EFFORT_RESEARCH": "low"})
    _SubmitStub.answer = {"summary": "s", "score": 1}
    _loop(submit_stub)
    assert all(r["output_config"] == {"effort": "low"} for r in _SubmitStub.requests)


def test_an_sdk_loop_sends_no_effort_by_default(
    submit_stub: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_effort(monkeypatch, {})
    _SubmitStub.answer = {"summary": "s", "score": 1}
    _loop(submit_stub)
    assert all("output_config" not in r for r in _SubmitStub.requests)


def test_a_loops_last_turn_keeps_its_effort(
    force_stub: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_effort(monkeypatch, {"ROTE_JUDGE_EFFORT": "medium"})
    _ForceStub.forced_answer = {"summary": "s", "score": 1}
    _loop(force_stub)
    loop_turn, forced = _ForceStub.requests
    assert loop_turn["output_config"] == {"effort": "medium"}
    assert forced["output_config"] == {
        "effort": "medium",
        "format": {"type": "json_schema", "schema": _DECLARED},
    }


def test_strict_marks_the_submit_tool_only_when_asked(submit_stub: str) -> None:
    _SubmitStub.answer = {"summary": "s", "score": 1}
    _loop(submit_stub, output_strict=True)
    strict_tool = {t["name"]: t for t in _SubmitStub.requests[0]["tools"]}["submit_result"]
    assert strict_tool["strict"] is True

    _SubmitStub.requests = []
    _loop(submit_stub)
    open_tool = {t["name"]: t for t in _SubmitStub.requests[0]["tools"]}["submit_result"]
    assert "strict" not in open_tool


def test_cli_lane_without_structured_output_fails_with_the_shared_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(helper, "_mcp_config_for", lambda *_a: helper.AgentMcp({}, [], [], {}))
    _capture_cli(monkeypatch, _agent_envelope())
    with pytest.raises(
        RuntimeError,
        match=(
            r"agent loop research \(claude-cli\) did not hand back its declared output: "
            r"the CLI returned no structured output"
        ),
    ):
        helper.run_agent_loop(
            node_id="research",
            description="Research.",
            task="{}",
            model="claude-sonnet-4-6",
            max_iterations=4,
            output_schema=_DECLARED,
        )
