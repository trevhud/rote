"""The drivers' single schema-locked call (``generate``), and the prefix's output check.

No inference: ``claude -p`` is a fake process and the Anthropic SDK a fake
client. What is pinned is what each lane sends (flag values, not their
presence) and what it reports back (all four usage buckets, the CLI's own
cost, ``is_error`` inside a zero exit).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from rote.compiler.drivers import DriverError, GenerationRequest
from rote.compiler.drivers.anthropic_api import AnthropicApiDriver
from rote.compiler.drivers.claude import ClaudeDriver
from rote.compiler.grounding import Sample
from rote.compiler.prefix import output_findings
from rote.ir import Pipeline

SCHEMA = {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}
REQUEST = GenerationRequest(
    system="SHARED PREFIX", prompt="the task", schema=SCHEMA, model="claude-sonnet-5-5", label="n1"
)


def _envelope(**fields: Any) -> dict[str, Any]:
    return {
        "type": "result",
        "is_error": False,
        "structured_output": {"x": "y"},
        "total_cost_usd": 0.0123,
        "modelUsage": {
            "claude-sonnet-5-5": {
                "inputTokens": 2,
                "outputTokens": 40,
                "cacheReadInputTokens": 900,
                "cacheCreationInputTokens": 30,
            }
        },
        **fields,
    }


class _FakeProc:
    def __init__(self, stdout: str, seen: dict[str, Any]) -> None:
        self._stdout = stdout
        self._seen = seen
        self.returncode = 0

    async def communicate(self, data: bytes) -> tuple[bytes, bytes]:
        args = self._seen["args"]
        # Read the system prompt file while it exists (the driver deletes it after).
        path = Path(args[args.index("--system-prompt-file") + 1])
        self._seen["system"] = path.read_text(encoding="utf-8")
        self._seen["stdin"] = data.decode()
        return self._stdout.encode(), b""


@pytest.fixture
def fake_claude(monkeypatch: pytest.MonkeyPatch):  # noqa: ANN201
    def _install(envelope: dict[str, Any]) -> dict[str, Any]:
        seen: dict[str, Any] = {}

        async def _exec(*args: str, **kwargs: Any) -> _FakeProc:
            seen["args"] = list(args)
            seen["env"] = kwargs.get("env") or {}
            return _FakeProc(json.dumps(envelope), seen)

        monkeypatch.setattr("asyncio.create_subprocess_exec", _exec)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-reach-the-child")
        return seen

    return _install


def _value(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


@pytest.mark.asyncio
async def test_claude_generate_sends_a_schema_locked_isolated_call(fake_claude: Any) -> None:
    seen = fake_claude(_envelope())
    g = await ClaudeDriver(effort="low").generate(REQUEST)
    args = seen["args"]
    assert json.loads(_value(args, "--json-schema")) == SCHEMA
    assert _value(args, "--model") == "claude-sonnet-5-5"
    assert _value(args, "--effort") == "low"
    assert _value(args, "--output-format") == "json"
    assert _value(args, "--tools") == ""  # no tools: a value, never a bare flag
    assert _value(args, "--setting-sources") == ""
    assert "--strict-mcp-config" in args
    assert seen["system"] == "SHARED PREFIX"
    assert seen["stdin"] == "the task"
    assert "the task" not in args and "SHARED PREFIX" not in args
    assert "ANTHROPIC_API_KEY" not in seen["env"]
    assert seen["env"]["CLAUDE_CODE_PROMPT_CACHE_TTL"] == "5m"

    assert g.payload == {"x": "y"}
    assert g.model == "claude-sonnet-5-5"
    assert (g.input_tokens, g.output_tokens) == (2, 40)
    assert (g.cache_write_tokens, g.cache_read_tokens) == (30, 900)
    assert g.cost_usd == 0.0123


@pytest.mark.asyncio
async def test_claude_generate_fails_on_is_error_inside_a_zero_exit(fake_claude: Any) -> None:
    fake_claude(_envelope(is_error=True, result="API Error: 529 overloaded"))
    with pytest.raises(DriverError, match="529 overloaded") as err:
        await ClaudeDriver().generate(REQUEST)
    assert err.value.metadata is not None
    assert err.value.metadata["cost_usd"] == 0.0123  # a failed call is still a paid one
    assert err.value.metadata["output_tokens"] == 40


@pytest.mark.asyncio
async def test_claude_generate_fails_without_structured_output(fake_claude: Any) -> None:
    fake_claude(_envelope(structured_output=None, result="structured_output_retry_exhausted"))
    with pytest.raises(DriverError, match="structured_output_retry_exhausted"):
        await ClaudeDriver().generate(REQUEST)


class _Stream:
    def __init__(self, message: Any) -> None:
        self.message = message

    async def __aenter__(self) -> _Stream:
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    def __aiter__(self) -> _Stream:
        return self

    async def __anext__(self) -> Any:
        raise StopAsyncIteration

    async def get_final_message(self) -> Any:
        return self.message


class _Client:
    def __init__(self, message: Any) -> None:
        self.calls: list[dict[str, Any]] = []
        self.closed = False
        outer = self

        class _Messages:
            def stream(self, **kwargs: Any) -> _Stream:
                outer.calls.append(kwargs)
                return _Stream(message)

        self.messages = _Messages()

    async def close(self) -> None:
        self.closed = True


def _message(content: list[Any]) -> Any:
    return SimpleNamespace(
        content=content,
        model="claude-sonnet-5-5",
        stop_reason="tool_use",
        usage=SimpleNamespace(
            input_tokens=5,
            output_tokens=60,
            cache_creation_input_tokens=700,
            cache_read_input_tokens=8000,
        ),
    )


@pytest.mark.asyncio
async def test_api_generate_locks_the_answer_and_caches_the_shared_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _Client(_message([SimpleNamespace(type="text", text='{"x": "z"}')]))
    monkeypatch.setattr("anthropic.AsyncAnthropic", lambda **k: client)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    g = await AnthropicApiDriver().generate(REQUEST)
    (call,) = client.calls
    # Structured outputs, never a forced tool: current models 400 on tool_choice "tool".
    assert call["output_config"] == {
        "format": {"type": "json_schema", "schema": SCHEMA},
        "effort": "low",  # unset, the API's default cost 5x the output tokens
    }
    assert "tool_choice" not in call and "tools" not in call
    assert call["system"] == [
        {"type": "text", "text": "SHARED PREFIX", "cache_control": {"type": "ephemeral"}}
    ]
    assert call["messages"] == [{"role": "user", "content": "the task"}]
    assert "temperature" not in call
    assert client.closed
    assert g.payload == {"x": "z"}
    assert (g.input_tokens, g.output_tokens) == (5, 60)
    assert (g.cache_write_tokens, g.cache_read_tokens) == (700, 8000)
    assert g.cost_usd is None  # the lane reports none; never a guess


@pytest.mark.asyncio
async def test_api_generate_without_a_json_answer_fails_with_its_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _Client(_message([SimpleNamespace(type="text", text="no")]))
    monkeypatch.setattr("anthropic.AsyncAnthropic", lambda **k: client)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    with pytest.raises(DriverError, match="no answer") as err:
        await AnthropicApiDriver().generate(REQUEST)
    assert err.value.metadata == {
        "input_tokens": 5,
        "output_tokens": 60,
        "cache_write_tokens": 700,
        "cache_read_tokens": 8000,
    }


# ───────── the prefix checks results against output_schema ─────────


def _pipeline(fan_out: bool) -> Pipeline:
    schema = {
        "type": "object",
        "properties": {"ids": {"type": "array"}, "total": {"type": "number"}},
        "required": ["ids", "total"],
    }
    nodes: list[dict[str, Any]] = [
        {
            "id": "src",
            "kind": "pure_function",
            "description": "rows",
            "impl": "extracted/src.py:src",
            "inputs": {"request": "pipeline.input"},
            "output": {"items": "list"},
        },
        {
            "id": "audit",
            "kind": "pure_function",
            "description": "audit one row",
            "impl": "extracted/audit.py:audit",
            "inputs": {"row": "src.output.items"},
            "output_schema": schema,
            "fan_out": fan_out,
        },
    ]
    return Pipeline.model_validate(
        {
            "name": "p",
            "version": "0.1.0",
            "input": {"type": "PInput"},
            "nodes": nodes,
            "edges": [{"from": "src", "to": "audit", "fan_out": fan_out}],
        }
    )


def test_a_result_that_breaks_its_output_schema_is_an_error() -> None:
    sample = Sample(payload={}, source="sample.json")
    report = {"audit": {"status": "ok", "output": {"ids": ["a"], "total": "12"}}}
    (finding,) = output_findings(_pipeline(fan_out=False), report, sample)
    assert finding.node_id == "audit" and finding.kind == "output_schema_mismatch"
    assert "audit.total is str, the schema says number" in finding.message
    assert finding.path == "extracted/audit.py"
    ok = {"audit": {"status": "ok", "output": {"ids": [], "total": 1.5}}}
    assert output_findings(_pipeline(fan_out=False), ok, sample) == []


def test_each_fanned_result_is_checked() -> None:
    sample = Sample(payload={}, source="sample.json")
    report = {
        "audit": {
            "status": "ok",
            "output": [{"ids": [], "total": 1}, {"ids": "x", "total": 2}],
        }
    }
    (finding,) = output_findings(_pipeline(fan_out=True), report, sample)
    assert "audit[1].ids is str" in finding.message


def test_the_prefix_run_reports_schema_breaks_on_the_sample_and_returns_no_payloads(
    tmp_path: Path,
) -> None:
    """End to end through the sandboxed runner: the runner hands rote each result,
    rote checks it against output_schema, and the verdict carries no payloads."""
    from tests.test_grounding import _prefix_root, _run

    root = _prefix_root(tmp_path)
    ir = (root / "pipeline.yaml").read_text()
    ir = ir.replace(
        "    impl: extracted/checks.py:summary\n",
        "    impl: extracted/checks.py:summary\n"
        "    output_schema:\n"
        "      type: object\n"
        "      properties: {count: {type: string}}\n"
        "      required: [count, total]\n",
    )
    (root / "pipeline.yaml").write_text(ir)
    run = _run(root)
    (finding,) = run.findings
    assert finding.kind == "output_schema_mismatch" and finding.node_id == "summary"
    assert "summary has no `total`" in finding.message
    assert all("output" not in step for step in run.steps.values())


def test_the_prefix_runs_past_a_gate_the_sample_answers(tmp_path: Path) -> None:
    """A node after a gate runs on the sample's answer, so its crash is found at
    compile time (expense-audit's file_audit_report raised on every case)."""
    from rote.contracts import implementation_status
    from tests.test_grounding import _ROWS, _pipeline, _prefix_root, _sample

    root = _prefix_root(tmp_path)
    (root / "extracted" / "checks.py").write_text(
        (root / "extracted" / "checks.py")
        .read_text()
        .replace(
            "def after(decision):\n    return decision",
            "def after(decision):\n    return {'ok': decision['approved_ids'][5]}",
        )
    )
    pipeline = _pipeline(root)
    from rote.compiler.prefix import run_prefix

    run = run_prefix(
        pipeline,
        root,
        _sample(_ROWS, gates={"controller_review": {"approved_ids": ["E1"]}}),
        implementation_status(pipeline, root),
        trust_skill=True,
    )
    assert run.steps["review"]["status"] == "ok"
    (finding,) = run.findings
    assert finding.node_id == "after_gate" and "IndexError" in finding.message
