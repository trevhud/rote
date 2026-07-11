"""Tests for the OpenAIApiDriver run() implementation.

We mock ``openai.AsyncOpenAI`` to return canned chat/completions
responses so the tool-use loop runs end-to-end without an API call. The
driver only inspects duck-typed attributes (``choices[0].message``
``.content`` / ``.tool_calls``, ``usage.prompt_tokens`` /
``.completion_tokens``, and each tool call's ``id`` / ``function.name`` /
``function.arguments``), so a SimpleNamespace fake is enough.

Coverage:

* Happy path: read_file → write_file → stop; result, file, metadata
* Live events: turn (cumulative tokens) / tool / phase ordering,
  progress.ndjson interception on write_file
* max_completion_tokens (not max_tokens) is sent
* Assistant turn carrying BOTH content and tool_calls; tool results
  appended as role:tool with tool_call_id; reasoning fields ignored
* Malformed tool-arguments JSON → is_error tool result, loop survives
* max_iterations exceeded → DriverError (for/else)
* is_available paths (env key / gateway headers / neither / no SDK) and
  base_url/default_headers/api_key plumbing
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from rote.graduator.drivers import DriverError
from rote.graduator.drivers.openai_api import OpenAIApiDriver

# ───────── Fake OpenAI SDK ─────────


class _FakeChatCompletions:
    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        snapshot = dict(kwargs)
        if "messages" in snapshot:
            # Deep-ish copy: the driver mutates its local messages list.
            snapshot["messages"] = [dict(m) for m in snapshot["messages"]]
        self.calls.append(snapshot)
        if not self._responses:
            raise RuntimeError("FakeChatCompletions out of canned responses; too few turns")
        return self._responses.pop(0)


class _FakeChat:
    def __init__(self, responses: list[Any]) -> None:
        self.completions = _FakeChatCompletions(responses)


class _FakeAsyncOpenAI:
    def __init__(self, responses: list[Any]) -> None:
        self.chat = _FakeChat(responses)


@pytest.fixture
def fake_openai(monkeypatch: pytest.MonkeyPatch):  # noqa: ANN201
    """Patch ``openai.AsyncOpenAI`` to return a fake client and set a key."""

    def _patch(responses: list[Any]) -> _FakeAsyncOpenAI:
        client = _FakeAsyncOpenAI(responses)
        monkeypatch.setattr(
            "rote.graduator.drivers.openai_api.openai.AsyncOpenAI",
            lambda *a, **k: client,
        )
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
        return client

    return _patch


# ───────── Canned-response builders ─────────


def _tool_call(call_id: str, name: str, arguments: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _message(
    content: str | None = None,
    tool_calls: list[SimpleNamespace] | None = None,
    **extra: Any,
) -> SimpleNamespace:
    # ``extra`` lets tests attach reasoning/thinking fields the driver
    # must ignore.
    return SimpleNamespace(content=content, tool_calls=tool_calls, **extra)


def _response(
    message: SimpleNamespace,
    *,
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    finish_reason: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


VALID_PIPELINE_YAML = """\
name: fake-pipeline
version: "0.1.0"
description: |
  A minimal pipeline used for driver testing.

input:
  type: FakeInput
  required: [foo]

nodes:
  - id: only_node
    kind: pure_function
    description: The one and only node.
    impl: extracted/foo.py:only_node

edges: []

entry_nodes: [only_node]
exit_nodes: [only_node]
"""


@pytest.fixture
def fake_skills(tmp_path: Path) -> tuple[Path, Path, Path]:
    skill_dir = tmp_path / "fake-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: fake\n---\n\n# Fake skill\n\nThis is the source skill.\n",
        encoding="utf-8",
    )

    graduator_dir = tmp_path / "fake-graduator-skill"
    graduator_dir.mkdir()
    (graduator_dir / "SKILL.md").write_text(
        "# Fake rote-graduate skill\n\nDo the graduation thing.\n",
        encoding="utf-8",
    )
    (graduator_dir / "references").mkdir()

    work_dir = tmp_path / "work"
    return skill_dir, graduator_dir, work_dir


def _write_pipeline_call(call_id: str, work_dir: Path) -> SimpleNamespace:
    import json

    return _tool_call(
        call_id,
        "write_file",
        json.dumps(
            {"path": str((work_dir / "pipeline.yaml").resolve()), "content": VALID_PIPELINE_YAML}
        ),
    )


# ───────── Happy path ─────────


@pytest.mark.asyncio
async def test_happy_path_read_then_write_then_stop(
    fake_skills: tuple[Path, Path, Path],
    fake_openai,  # noqa: ANN001
) -> None:
    import json

    skill_dir, graduator_dir, work_dir = fake_skills
    work_dir.mkdir()
    skill_md_abs = (skill_dir / "SKILL.md").resolve()

    responses = [
        _response(
            _message(
                content="Reading the skill.",
                tool_calls=[_tool_call("c1", "read_file", json.dumps({"path": str(skill_md_abs)}))],
            ),
            prompt_tokens=200,
            completion_tokens=40,
        ),
        _response(
            _message(tool_calls=[_write_pipeline_call("c2", work_dir)]),
            prompt_tokens=300,
            completion_tokens=500,
        ),
        _response(_message(content="Done."), prompt_tokens=100, completion_tokens=20),
    ]
    client = fake_openai(responses)

    driver = OpenAIApiDriver(model="openai/gpt-5.5")
    result = await driver.run(
        skill_dir=skill_dir, graduator_skill_dir=graduator_dir, work_dir=work_dir
    )

    assert result.driver_name == "openai-api"
    assert result.pipeline_yaml_path == (work_dir / "pipeline.yaml").resolve()
    assert result.pipeline_yaml_path.read_text() == VALID_PIPELINE_YAML
    assert result.metadata == {
        "model": "openai/gpt-5.5",
        "input_tokens": 600,
        "output_tokens": 560,
        "iterations": 3,
    }

    # First message is the system prompt; model + max_completion_tokens sent.
    first = client.chat.completions.calls[0]
    assert first["model"] == "openai/gpt-5.5"
    assert first["max_completion_tokens"] == driver.max_tokens_per_turn
    assert "max_tokens" not in first
    assert first["messages"][0]["role"] == "system"
    assert "ROTE GRADUATE SKILL" in first["messages"][0]["content"]


# ───────── Live events ─────────


@pytest.mark.asyncio
async def test_emits_turn_tool_and_phase_events(
    fake_skills: tuple[Path, Path, Path],
    fake_openai,  # noqa: ANN001
) -> None:
    import json

    skill_dir, graduator_dir, work_dir = fake_skills
    work_dir.mkdir()
    progress_abs = (work_dir / "progress.ndjson").resolve()
    skill_md_abs = (skill_dir / "SKILL.md").resolve()

    responses = [
        _response(
            _message(
                content="Reading.",
                tool_calls=[_tool_call("c1", "read_file", json.dumps({"path": str(skill_md_abs)}))],
            ),
            prompt_tokens=200,
            completion_tokens=40,
        ),
        _response(
            _message(
                tool_calls=[
                    _tool_call(
                        "c2",
                        "write_file",
                        json.dumps(
                            {
                                "path": str(progress_abs),
                                "content": '{"phase": 1, "name": "Intake"}\n'
                                '{"phase": 2, "name": "Classify"}\n',
                            }
                        ),
                    )
                ]
            ),
            prompt_tokens=100,
            completion_tokens=30,
        ),
        _response(
            _message(tool_calls=[_write_pipeline_call("c3", work_dir)]),
            prompt_tokens=100,
            completion_tokens=200,
        ),
        _response(_message(content="Done."), prompt_tokens=50, completion_tokens=10),
    ]
    fake_openai(responses)

    events: list[Any] = []
    driver = OpenAIApiDriver(model="glm-5.2")
    await driver.run(
        skill_dir=skill_dir,
        graduator_skill_dir=graduator_dir,
        work_dir=work_dir,
        on_event=events.append,
    )

    turns = [e for e in events if e.type == "turn"]
    tools = [e for e in events if e.type == "tool"]
    phases = [e for e in events if e.type == "phase"]

    assert [e.turn for e in turns] == [1, 2, 3, 4]
    assert turns[0].tokens == {"input": 200, "output": 40}
    assert turns[-1].tokens == {"input": 450, "output": 280}
    assert turns[0].message.startswith("turn 1: Reading.")

    assert [e.tool_name for e in tools] == ["read_file", "write_file", "write_file"]
    assert any(e.path == "progress.ndjson" for e in tools)
    assert any(e.path == "pipeline.yaml" for e in tools)

    assert [e.phase for e in phases] == [1, 2]
    assert phases[0].phase_name == "Intake"


# ───────── Message shape / robustness ─────────


@pytest.mark.asyncio
async def test_assistant_with_content_and_tools_and_reasoning(
    fake_skills: tuple[Path, Path, Path],
    fake_openai,  # noqa: ANN001
) -> None:
    """A turn with BOTH content and tool_calls dispatches the tools and
    replays a clean assistant message; reasoning fields are dropped."""
    skill_dir, graduator_dir, work_dir = fake_skills
    work_dir.mkdir()

    responses = [
        _response(
            _message(
                content="I will write the pipeline now.",
                tool_calls=[_write_pipeline_call("c1", work_dir)],
                reasoning="internal chain of thought",  # must be ignored
            )
        ),
        _response(_message(content="Done.")),
    ]
    client = fake_openai(responses)

    driver = OpenAIApiDriver(model="moonshotai/kimi-k2.6")
    result = await driver.run(
        skill_dir=skill_dir, graduator_skill_dir=graduator_dir, work_dir=work_dir
    )
    assert result.pipeline_yaml_path.is_file()

    # The 2nd create call's messages carry the replayed assistant turn and
    # the tool result — the assistant turn keeps content + tool_calls but
    # not the reasoning field, and the tool result is role:tool.
    replayed = client.chat.completions.calls[1]["messages"]
    assistant = next(m for m in replayed if m["role"] == "assistant")
    assert assistant["content"] == "I will write the pipeline now."
    assert "reasoning" not in assistant
    assert assistant["tool_calls"][0]["function"]["name"] == "write_file"
    tool_msg = next(m for m in replayed if m["role"] == "tool")
    assert tool_msg["tool_call_id"] == "c1"
    assert "Wrote" in tool_msg["content"]


@pytest.mark.asyncio
async def test_malformed_tool_arguments_do_not_crash_loop(
    fake_skills: tuple[Path, Path, Path],
    fake_openai,  # noqa: ANN001
) -> None:
    skill_dir, graduator_dir, work_dir = fake_skills
    work_dir.mkdir()

    responses = [
        # Invalid JSON in the tool arguments.
        _response(_message(tool_calls=[_tool_call("bad", "write_file", '{"path": ')])),
        # Recover on the next turn with a good write.
        _response(_message(tool_calls=[_write_pipeline_call("c2", work_dir)])),
        _response(_message(content="Done.")),
    ]
    client = fake_openai(responses)

    driver = OpenAIApiDriver(model="openai/gpt-5.5")
    result = await driver.run(
        skill_dir=skill_dir, graduator_skill_dir=graduator_dir, work_dir=work_dir
    )
    assert result.pipeline_yaml_path.is_file()

    # The error was reported back as a tool result, not raised.
    tool_msgs = [m for m in client.chat.completions.calls[1]["messages"] if m["role"] == "tool"]
    assert any("parse" in m["content"].lower() and m["tool_call_id"] == "bad" for m in tool_msgs)


@pytest.mark.asyncio
async def test_path_jail_error_returned_to_model(
    fake_skills: tuple[Path, Path, Path],
    fake_openai,  # noqa: ANN001
    tmp_path: Path,
) -> None:
    import json

    skill_dir, graduator_dir, work_dir = fake_skills
    work_dir.mkdir()
    outside = tmp_path / "outside" / "evil.py"

    responses = [
        _response(
            _message(
                tool_calls=[
                    _tool_call(
                        "c1", "write_file", json.dumps({"path": str(outside), "content": "x"})
                    )
                ]
            )
        ),
        _response(_message(tool_calls=[_write_pipeline_call("c2", work_dir)])),
        _response(_message(content="Done.")),
    ]
    client = fake_openai(responses)

    driver = OpenAIApiDriver(model="openai/gpt-5.5")
    await driver.run(skill_dir=skill_dir, graduator_skill_dir=graduator_dir, work_dir=work_dir)

    tool_msgs = [m for m in client.chat.completions.calls[1]["messages"] if m["role"] == "tool"]
    assert any("not within allowed write root" in m["content"] for m in tool_msgs)
    assert not outside.exists()


# ───────── Failure modes ─────────


@pytest.mark.asyncio
async def test_max_iterations_exceeded_raises(
    fake_skills: tuple[Path, Path, Path],
    fake_openai,  # noqa: ANN001
) -> None:
    import json

    skill_dir, graduator_dir, work_dir = fake_skills
    work_dir.mkdir()
    skill_md_abs = (skill_dir / "SKILL.md").resolve()

    # Every turn just reads a file — the agent never writes pipeline.yaml.
    looping = _response(
        _message(tool_calls=[_tool_call("c", "read_file", json.dumps({"path": str(skill_md_abs)}))])
    )
    fake_openai([looping] * 5)

    driver = OpenAIApiDriver(model="openai/gpt-5.5", max_iterations=3)
    with pytest.raises(DriverError, match="did not complete within 3 iterations"):
        await driver.run(skill_dir=skill_dir, graduator_skill_dir=graduator_dir, work_dir=work_dir)


@pytest.mark.asyncio
async def test_finishes_without_pipeline_raises(
    fake_skills: tuple[Path, Path, Path],
    fake_openai,  # noqa: ANN001
) -> None:
    skill_dir, graduator_dir, work_dir = fake_skills
    work_dir.mkdir()
    fake_openai([_response(_message(content="I give up."))])

    driver = OpenAIApiDriver(model="openai/gpt-5.5")
    with pytest.raises(DriverError, match="did not produce"):
        await driver.run(skill_dir=skill_dir, graduator_skill_dir=graduator_dir, work_dir=work_dir)


# ───────── is_available + client plumbing ─────────


def test_is_available_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    assert OpenAIApiDriver(model="x").is_available() == (True, "")


def test_is_available_gateway_cf_header(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    d = OpenAIApiDriver(model="x", default_headers={"cf-aig-authorization": "Bearer t"})
    assert d.is_available()[0] is True


def test_is_available_gateway_authorization_header(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    d = OpenAIApiDriver(model="x", default_headers={"Authorization": "Bearer t"})
    assert d.is_available()[0] is True


def test_is_available_neither(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    available, reason = OpenAIApiDriver(model="x").is_available()
    assert available is False
    assert "OPENAI_API_KEY" in reason


def test_is_available_sdk_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("rote.graduator.drivers.openai_api._OPENAI_AVAILABLE", False)
    available, reason = OpenAIApiDriver(model="x").is_available()
    assert available is False
    assert "rote[openai-api]" in reason


def test_client_kwargs_omitted_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "real")
    assert OpenAIApiDriver(model="x")._client_kwargs() == {}


def test_client_kwargs_plumbs_base_url_and_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "real")
    headers = {"cf-aig-authorization": "Bearer g"}
    d = OpenAIApiDriver(model="x", base_url="https://gw.example/v1", default_headers=headers)
    kwargs = d._client_kwargs()
    assert kwargs["base_url"] == "https://gw.example/v1"
    assert kwargs["default_headers"] == headers
    assert "api_key" not in kwargs  # real env key present


def test_client_kwargs_gateway_placeholder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    d = OpenAIApiDriver(model="x", default_headers={"Authorization": "Bearer g"})
    assert d._client_kwargs()["api_key"] == "rote-gateway"


@pytest.mark.asyncio
async def test_run_constructs_client_with_plumbed_kwargs(
    fake_skills: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill_dir, graduator_dir, work_dir = fake_skills
    work_dir.mkdir()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    captured: dict[str, Any] = {}
    client = _FakeAsyncOpenAI(
        [
            _response(_message(tool_calls=[_write_pipeline_call("c1", work_dir)])),
            _response(_message(content="done")),
        ]
    )

    def _ctor(*args: Any, **kwargs: Any) -> _FakeAsyncOpenAI:
        captured.update(kwargs)
        return client

    monkeypatch.setattr("rote.graduator.drivers.openai_api.openai.AsyncOpenAI", _ctor)

    driver = OpenAIApiDriver(
        model="openai/gpt-5.5",
        base_url="https://gw.example/v1",
        default_headers={"cf-aig-authorization": "Bearer g"},
    )
    await driver.run(skill_dir=skill_dir, graduator_skill_dir=graduator_dir, work_dir=work_dir)

    assert captured["base_url"] == "https://gw.example/v1"
    assert captured["default_headers"] == {"cf-aig-authorization": "Bearer g"}
    assert captured["api_key"] == "rote-gateway"


# ───────── Turn truncation (finish_reason "length") ─────────


@pytest.mark.asyncio
async def test_length_truncation_continues_and_warns(
    fake_skills: tuple[Path, Path, Path],
    fake_openai,  # noqa: ANN001
) -> None:
    """A turn cut off at the output-token limit (finish_reason "length",
    no tool_calls) must continue rather than complete: warn, nudge, and
    finish normally on a later turn."""
    skill_dir, graduator_dir, work_dir = fake_skills
    work_dir.mkdir()

    responses = [
        # Turn 1: reasoning model spent the whole budget thinking; cut off.
        _response(_message(content=None), finish_reason="length"),
        # Turn 2: writes the pipeline.
        _response(_message(tool_calls=[_write_pipeline_call("c2", work_dir)])),
        # Turn 3: done.
        _response(_message(content="Done."), finish_reason="stop"),
    ]
    client = fake_openai(responses)

    events: list[Any] = []
    driver = OpenAIApiDriver(model="openai/gpt-5.5")
    result = await driver.run(
        skill_dir=skill_dir,
        graduator_skill_dir=graduator_dir,
        work_dir=work_dir,
        on_event=events.append,
    )

    assert result.pipeline_yaml_path.is_file()
    warnings = [e for e in events if e.type == "warning"]
    assert len(warnings) == 1
    assert warnings[0].turn == 1
    assert "truncated" in warnings[0].message

    # The truncated assistant turn was replayed with a non-null content
    # (some endpoints reject null content + no tool_calls), followed by a
    # user "continue" nudge.
    turn2_messages = client.chat.completions.calls[1]["messages"]
    assistant = [m for m in turn2_messages if m["role"] == "assistant"][-1]
    assert assistant["content"] == ""
    assert turn2_messages[-1]["role"] == "user"
    assert "output-token limit" in turn2_messages[-1]["content"]


def test_default_max_tokens_per_turn_is_generous() -> None:
    from rote.graduator.drivers.openai_api import DEFAULT_MAX_TOKENS_PER_TURN

    assert DEFAULT_MAX_TOKENS_PER_TURN == 32768
