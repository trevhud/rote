"""Tests for the ClaudeDriver subprocess implementation.

We mock ``asyncio.create_subprocess_exec`` to simulate Claude Code
spawning without ever invoking the real CLI. The fake process streams
canned ``stream-json`` NDJSON lines on stdout (one JSON object per
line) and canned stderr, so the driver's incremental reader, event
mapping, and metadata extraction are all exercised.

Coverage:

* Happy path: args are correct (stream-json + verbose), env is
  scrubbed, pipeline.yaml is produced, metadata comes from the final
  ``result`` object
* Env scrubbing + OAuth passthrough + animation-silence flag
* Live events: assistant lines → turn events, tool_use blocks → tool
  events (turn-numbered, work-dir-relative paths); progress.ndjson →
  phase events via the concurrent watcher
* ``_map_stream_line`` pure-function edge cases
* Nonzero exit → DriverError with stderr; nonzero exit after
  pipeline.yaml written → recovered; exit 0 without pipeline.yaml →
  DriverError; missing result object → minimal metadata
* Missing compiler SKILL.md → DriverError before subprocess
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rote.compiler.drivers import DriverError
from rote.compiler.drivers.claude import (
    DEFAULT_ALLOWED_TOOLS,
    DEFAULT_MAX_TURNS,
    DEFAULT_MODEL,
    ClaudeDriver,
    _map_stream_line,
)
from rote.compiler.events import CompilationEvent

# ───────── stream-json line builders ─────────


def _assistant_line(
    text: str | None = None,
    tools: list[tuple[str, dict[str, Any]]] | None = None,
    usage: dict[str, int] | None = None,
    message_id: str | None = None,
) -> str:
    """A ``{"type":"assistant"}`` NDJSON line with text + tool_use blocks.

    ``usage`` mirrors Claude Code's per-message ``message.usage`` block
    (``input_tokens`` / ``output_tokens`` for the single model response).
    """
    content: list[dict[str, Any]] = []
    if text is not None:
        content.append({"type": "text", "text": text})
    for name, inp in tools or []:
        content.append({"type": "tool_use", "name": name, "input": inp})
    message: dict[str, Any] = {"content": content}
    if message_id is not None:
        message["id"] = message_id
    if usage is not None:
        message["usage"] = usage
    return json.dumps({"type": "assistant", "message": message})


def _result_line(**fields: Any) -> str:
    """A final ``{"type":"result", ...}`` NDJSON line."""
    return json.dumps({"type": "result", **fields})


def _stream(*lines: str) -> str:
    return "\n".join(lines) + "\n"


# ───────── Fake streaming subprocess ─────────


class _FakeStreamReader:
    """Async-iterable / readable stand-in for a subprocess pipe."""

    def __init__(self, data: bytes = b"") -> None:
        self._buf = data

    def __aiter__(self):  # noqa: ANN204
        async def gen():  # noqa: ANN202
            for line in self._buf.splitlines(keepends=True):
                yield line

        return gen()

    async def read(self) -> bytes:
        data = self._buf
        self._buf = b""
        return data


class _FakeProcess:
    def __init__(self, stdout_bytes: bytes, stderr_bytes: bytes, returncode: int) -> None:
        self.stdout = _FakeStreamReader(stdout_bytes)
        self.stderr = _FakeStreamReader(stderr_bytes)
        self.returncode = returncode

    async def wait(self) -> int:
        return self.returncode


class _FakeSubprocessRecorder:
    """Captures every ``create_subprocess_exec`` call and returns a fake proc."""

    def __init__(
        self,
        *,
        stdout_text: str = "",
        stderr_text: str = "",
        returncode: int = 0,
        write_files: dict[str, str] | None = None,
    ) -> None:
        self.stdout_text = stdout_text
        self.stderr_text = stderr_text
        self.returncode = returncode
        self.write_files = write_files or {}
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, *args: str, **kwargs: Any) -> _FakeProcess:
        env = kwargs.get("env") or {}
        self.calls.append({"args": args, "env": dict(env), "kwargs": kwargs})

        # A real ``claude -p`` run writes files into the work dir via its
        # Write tool; simulate that so the driver's on-disk check passes
        # and the progress watcher has something to read.
        if self.write_files:
            work_dir = self._extract_work_dir(args)
            if work_dir is not None:
                for rel_path, content in self.write_files.items():
                    target = work_dir / rel_path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(content, encoding="utf-8")

        return _FakeProcess(
            self.stdout_text.encode("utf-8"),
            self.stderr_text.encode("utf-8"),
            self.returncode,
        )

    @staticmethod
    def _extract_work_dir(args: tuple[str, ...]) -> Path | None:
        add_dir_values: list[str] = []
        for i, a in enumerate(args):
            if a == "--add-dir" and i + 1 < len(args):
                add_dir_values.append(args[i + 1])
        if len(add_dir_values) >= 3:
            return Path(add_dir_values[-1])
        return None


@pytest.fixture
def fake_claude_subprocess(monkeypatch: pytest.MonkeyPatch):  # noqa: ANN201
    """Install a fake ``create_subprocess_exec`` + ``which`` and return the recorder."""

    def _install(
        *,
        stdout_text: str = "",
        stderr_text: str = "",
        returncode: int = 0,
        write_files: dict[str, str] | None = None,
    ) -> _FakeSubprocessRecorder:
        recorder = _FakeSubprocessRecorder(
            stdout_text=stdout_text,
            stderr_text=stderr_text,
            returncode=returncode,
            write_files=write_files,
        )
        monkeypatch.setattr(
            "rote.compiler.drivers.claude.asyncio.create_subprocess_exec",
            recorder,
        )
        monkeypatch.setattr(
            "rote.compiler.drivers.claude.which",
            lambda name: "/usr/local/bin/claude" if name == "claude" else None,
        )
        return recorder

    return _install


# ───────── Skill bundle fixture ─────────


@pytest.fixture
def fake_skills(tmp_path: Path) -> tuple[Path, Path, Path]:
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: fake\n---\n\n# Fake skill\n", encoding="utf-8")

    compiler_dir = tmp_path / "compiler"
    compiler_dir.mkdir()
    (compiler_dir / "SKILL.md").write_text(
        "# Fake compiler rubric\n\nDo the compilation thing.\n",
        encoding="utf-8",
    )
    (compiler_dir / "references").mkdir()

    work_dir = tmp_path / "work"
    return skill_dir, compiler_dir, work_dir


VALID_PIPELINE_YAML = """\
name: fake-pipeline
version: "0.1.0"
description: |
  Fake pipeline used for ClaudeDriver testing.

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


# ───────── Happy path ─────────


@pytest.mark.asyncio
async def test_happy_path_args_env_and_metadata(
    fake_skills: tuple[Path, Path, Path],
    fake_claude_subprocess,  # noqa: ANN001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill_dir, compiler_dir, work_dir = fake_skills

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-be-scrubbed")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "auth-should-be-scrubbed")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-should-pass-through")

    stdout = _stream(
        _assistant_line(text="Reading the rubric."),
        _result_line(
            result="Compiled successfully.",
            total_cost_usd=0.42,
            duration_ms=58000,
            num_turns=18,
            session_id="sess-abc-123",
        ),
    )
    recorder = fake_claude_subprocess(
        stdout_text=stdout,
        returncode=0,
        write_files={"pipeline.yaml": VALID_PIPELINE_YAML},
    )

    driver = ClaudeDriver()
    result = await driver.run(skill_dir, compiler_dir, work_dir)

    assert result.driver_name == "claude"
    assert result.pipeline_yaml_path == (work_dir / "pipeline.yaml").resolve()
    assert result.pipeline_yaml_path.read_text() == VALID_PIPELINE_YAML

    # Metadata parsed from the final result object
    assert result.metadata["driver"] == "claude"
    assert result.metadata["cost_usd"] == 0.42
    assert result.metadata["duration_ms"] == 58000
    assert result.metadata["num_turns"] == 18
    assert result.metadata["session_id"] == "sess-abc-123"

    assert len(recorder.calls) == 1
    args = list(recorder.calls[0]["args"])
    assert args[0] == "claude"
    assert "-p" in args
    assert args[args.index("--model") + 1] == DEFAULT_MODEL
    assert "--append-system-prompt" in args
    assert args.count("--add-dir") == 3
    assert DEFAULT_ALLOWED_TOOLS in args
    assert args[args.index("--output-format") + 1] == "stream-json"
    assert "--verbose" in args
    assert str(DEFAULT_MAX_TURNS) in args
    assert recorder.calls[0]["kwargs"].get("limit")

    env = recorder.calls[0]["env"]
    assert "ANTHROPIC_API_KEY" not in env
    assert "ANTHROPIC_AUTH_TOKEN" not in env
    assert env.get("CLAUDE_CODE_DISABLE_NONINTERACTIVE_ANIMATIONS") == "1"
    assert env.get("CLAUDE_CODE_OAUTH_TOKEN") == "sk-ant-oat01-should-pass-through"


def test_model_usage_includes_all_models_and_overrides_main_loop_usage() -> None:
    result = {
        "usage": {"input_tokens": 1, "output_tokens": 2},
        "modelUsage": {
            "main": {
                "inputTokens": 11,
                "outputTokens": 22,
                "cacheCreationInputTokens": 33,
                "cacheReadInputTokens": 44,
            },
            "child": {
                "inputTokens": 100,
                "outputTokens": 200,
                "cacheCreationInputTokens": 300,
                "cacheReadInputTokens": 400,
            },
        },
        "num_turns": 9,
        "total_cost_usd": 1.23,
    }
    metadata = ClaudeDriver()._parse_metadata(result, {"input": 999, "output": 999})
    assert metadata["input_tokens"] == 111
    assert metadata["output_tokens"] == 222
    assert metadata["cache_write_tokens"] == 333
    assert metadata["cache_read_tokens"] == 444
    assert metadata["usage_source"] == "modelUsage"
    assert metadata["usage_complete"] is True
    assert metadata["num_turns"] == 9  # provider count, not progress response count


@pytest.mark.parametrize(
    "result",
    [
        {"usage": {}},
        {"usage": {"input_tokens": 1, "output_tokens": None}},
        {"usage": {"input_tokens": 1, "output_tokens": True}},
        {"modelUsage": {"main": {"inputTokens": 1}}},
        {"modelUsage": {"main": None}},
        {"modelUsage": {"main": {"outputTokens": 10}, "child": {}}},
    ],
)
def test_missing_final_output_is_incomplete_not_zero(result: dict[str, Any]) -> None:
    metadata = ClaudeDriver()._parse_metadata(result, {"input": 3, "cache_write": 61263})
    assert metadata["usage_complete"] is False
    assert metadata["output_tokens"] is None
    assert metadata["input_tokens"] == 3
    assert metadata["cache_write_tokens"] == 61263


def test_zero_final_usage_is_not_replaced_with_stream_totals() -> None:
    metadata = ClaudeDriver()._parse_metadata(
        {
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            }
        },
        {"input": 999, "output": 999, "cache_write": 999, "cache_read": 999},
    )
    assert metadata["input_tokens"] == metadata["output_tokens"] == 0
    assert metadata["cache_write_tokens"] == metadata["cache_read_tokens"] == 0
    assert metadata["usage_complete"] is True


@pytest.mark.asyncio
async def test_prompt_and_system_prompt_content(
    fake_skills: tuple[Path, Path, Path],
    fake_claude_subprocess,  # noqa: ANN001
) -> None:
    skill_dir, compiler_dir, work_dir = fake_skills
    recorder = fake_claude_subprocess(
        stdout_text=_stream(_result_line(result="ok")),
        write_files={"pipeline.yaml": VALID_PIPELINE_YAML},
    )

    driver = ClaudeDriver()
    await driver.run(skill_dir, compiler_dir, work_dir)

    args = list(recorder.calls[0]["args"])
    user_prompt = args[args.index("-p") + 1]
    assert str(skill_dir.resolve()) in user_prompt
    assert str(compiler_dir.resolve()) in user_prompt
    assert str(work_dir.resolve()) in user_prompt
    assert "pipeline.yaml" in user_prompt

    system_prompt = args[args.index("--append-system-prompt") + 1]
    assert "ROTE COMPILE SKILL" in system_prompt
    assert "Fake compiler rubric" in system_prompt


# ───────── Live events ─────────


@pytest.mark.asyncio
async def test_stream_emits_turn_tool_and_phase_events(
    fake_skills: tuple[Path, Path, Path],
    fake_claude_subprocess,  # noqa: ANN001
) -> None:
    """assistant lines → turn events, tool_use → tool events (work-dir
    relative path), and progress.ndjson → phase events via the watcher."""
    skill_dir, compiler_dir, work_dir = fake_skills
    sig_path = str((work_dir / "signatures" / "qualify.ts").resolve())

    stdout = _stream(
        _assistant_line(text="Classifying nodes."),
        _assistant_line(
            text="Writing the signature.",
            tools=[("Write", {"file_path": sig_path, "content": "x"})],
        ),
        _result_line(result="done", num_turns=2),
    )
    fake_claude_subprocess(
        stdout_text=stdout,
        returncode=0,
        write_files={
            "pipeline.yaml": VALID_PIPELINE_YAML,
            "progress.ndjson": '{"phase": 1, "name": "Intake"}\n'
            '{"phase": 2, "name": "Node Classification"}\n',
        },
    )

    events: list[CompilationEvent] = []
    driver = ClaudeDriver()
    await driver.run(skill_dir, compiler_dir, work_dir, on_event=events.append)

    turns = [e for e in events if e.type == "turn"]
    tools = [e for e in events if e.type == "tool"]
    phases = [e for e in events if e.type == "phase"]

    assert [e.turn for e in turns] == [1, 2]
    assert len(tools) == 1
    assert tools[0].turn == 2
    assert tools[0].tool_name == "Write"
    # Path is rendered relative to the work dir
    assert tools[0].path == str(Path("signatures") / "qualify.ts")
    assert {e.phase for e in phases} == {1, 2}
    assert phases[0].phase_name == "Intake"


@pytest.mark.asyncio
async def test_no_events_when_on_event_none_still_parses_metadata(
    fake_skills: tuple[Path, Path, Path],
    fake_claude_subprocess,  # noqa: ANN001
) -> None:
    """The stream is parsed the same way without a sink — metadata still lands."""
    skill_dir, compiler_dir, work_dir = fake_skills
    fake_claude_subprocess(
        stdout_text=_stream(
            _assistant_line(text="working"),
            _result_line(result="ok", num_turns=5, total_cost_usd=0.1),
        ),
        write_files={"pipeline.yaml": VALID_PIPELINE_YAML},
    )

    driver = ClaudeDriver()
    result = await driver.run(skill_dir, compiler_dir, work_dir)
    assert result.metadata["num_turns"] == 5
    assert result.metadata["cost_usd"] == 0.1


# ───────── _map_stream_line pure function ─────────


def test_map_stream_line_assistant_increments_turn_and_maps_tools(tmp_path: Path) -> None:
    line = _assistant_line(
        text="Here is my plan for the pipeline.",
        tools=[("Read", {"file_path": str(tmp_path / "SKILL.md")})],
    )
    events, new_turn, obj = _map_stream_line(line, 4, tmp_path)

    assert new_turn == 5
    assert obj is not None and obj["type"] == "assistant"
    assert [e.type for e in events] == ["turn", "tool"]
    assert events[0].turn == 5
    assert events[0].message.startswith("turn 5: Here is my plan")
    assert events[1].tool_name == "Read"
    assert events[1].path == "SKILL.md"


def test_map_stream_line_text_only_turn_has_no_tool_events() -> None:
    events, new_turn, _ = _map_stream_line(_assistant_line(text="just thinking"), 0)
    assert new_turn == 1
    assert [e.type for e in events] == ["turn"]


def test_map_stream_line_tool_only_turn_reports_thinking() -> None:
    events, _, _ = _map_stream_line(_assistant_line(tools=[("Grep", {"pattern": "foo"})]), 0)
    turn_events = [e for e in events if e.type == "turn"]
    assert turn_events[0].message == "turn 1: thinking…"


def test_map_stream_line_accumulates_cumulative_tokens() -> None:
    """Per-message ``usage`` is summed into a running total, and each turn
    event carries the cumulative per-bucket figures."""
    totals: dict[str, int] = {"input": 0, "output": 0, "cache_write": 0, "cache_read": 0}

    events1, turn1, _ = _map_stream_line(
        _assistant_line(text="one", usage={"input_tokens": 100, "output_tokens": 20}),
        0,
        None,
        totals,
    )
    assert events1[0].tokens == {
        "input": 100,
        "cache_write": 0,
        "cache_read": 0,
    }

    events2, _turn2, _ = _map_stream_line(
        _assistant_line(text="two", usage={"input_tokens": 250, "output_tokens": 30}),
        turn1,
        None,
        totals,
    )
    # Cumulative across turns, not just this message's usage.
    expected = {"input": 350, "cache_write": 0, "cache_read": 0}
    assert events2[0].tokens == expected
    assert totals == {**expected, "output": 0}


def test_map_stream_line_missing_usage_keeps_prior_total() -> None:
    """A message without ``usage`` leaves the running total unchanged; the
    turn event still reports the cumulative figure carried so far."""
    totals = {"input": 40, "output": 5, "cache_write": 7, "cache_read": 9}
    events, _turn, _ = _map_stream_line(_assistant_line(text="no usage"), 3, None, totals)
    assert events[0].tokens == {"input": 40, "cache_write": 7, "cache_read": 9}
    assert events[0].usage_complete is False


def test_map_stream_line_result_line_yields_no_events_but_returns_obj() -> None:
    events, new_turn, obj = _map_stream_line(_result_line(result="ok", num_turns=3), 7)
    assert events == []
    assert new_turn == 7
    assert obj is not None and obj["type"] == "result"


def test_map_stream_line_garbage_is_tolerated() -> None:
    events, new_turn, obj = _map_stream_line("not json at all", 2)
    assert events == []
    assert new_turn == 2
    assert obj is None


# ───────── Failure modes ─────────


@pytest.mark.asyncio
async def test_nonzero_exit_raises_driver_error_with_stderr(
    fake_skills: tuple[Path, Path, Path],
    fake_claude_subprocess,  # noqa: ANN001
) -> None:
    skill_dir, compiler_dir, work_dir = fake_skills
    fake_claude_subprocess(
        stdout_text="",
        stderr_text="Error: rate limit exceeded",
        returncode=1,
    )

    driver = ClaudeDriver()
    with pytest.raises(DriverError) as excinfo:
        await driver.run(skill_dir, compiler_dir, work_dir)
    assert "exited with code 1" in str(excinfo.value)
    assert "rate limit exceeded" in (excinfo.value.details or "")


@pytest.mark.asyncio
async def test_nonzero_exit_with_pipeline_yaml_recovers_run(
    fake_skills: tuple[Path, Path, Path],
    fake_claude_subprocess,  # noqa: ANN001
) -> None:
    """A nonzero exit after pipeline.yaml was written is treated as success."""
    skill_dir, compiler_dir, work_dir = fake_skills
    fake_claude_subprocess(
        stdout_text=_stream(_result_line(result="API Error: ECONNRESET", is_error=True)),
        stderr_text="",
        returncode=1,
        write_files={"pipeline.yaml": VALID_PIPELINE_YAML},
    )

    driver = ClaudeDriver()
    result = await driver.run(skill_dir, compiler_dir, work_dir)
    assert result.pipeline_yaml_path.is_file()
    assert "subprocess_warning" in result.metadata
    assert "exited with code 1" in result.metadata["subprocess_warning"]


@pytest.mark.asyncio
async def test_exit_zero_without_pipeline_yaml_raises_driver_error(
    fake_skills: tuple[Path, Path, Path],
    fake_claude_subprocess,  # noqa: ANN001
) -> None:
    skill_dir, compiler_dir, work_dir = fake_skills
    fake_claude_subprocess(
        stdout_text=_stream(_result_line(result="I gave up")),
        returncode=0,
        write_files=None,
    )

    driver = ClaudeDriver()
    with pytest.raises(DriverError, match="did not produce"):
        await driver.run(skill_dir, compiler_dir, work_dir)


@pytest.mark.asyncio
async def test_missing_result_object_returns_minimal_metadata(
    fake_skills: tuple[Path, Path, Path],
    fake_claude_subprocess,  # noqa: ANN001
) -> None:
    """A stream that ends without a result line still yields a result;
    metadata degrades to the driver name plus the (here zero) token totals."""
    skill_dir, compiler_dir, work_dir = fake_skills
    fake_claude_subprocess(
        stdout_text=_stream(_assistant_line(text="done but no summary line")),
        returncode=0,
        write_files={"pipeline.yaml": VALID_PIPELINE_YAML},
    )

    driver = ClaudeDriver()
    result = await driver.run(skill_dir, compiler_dir, work_dir)
    assert result.metadata == {
        "driver": "claude",
        "input_tokens": 0,
        "output_tokens": None,
        "cache_write_tokens": 0,
        "cache_read_tokens": 0,
        "usage_source": "stream",
        "usage_complete": False,
    }


@pytest.mark.asyncio
async def test_metadata_carries_cumulative_stream_tokens(
    fake_skills: tuple[Path, Path, Path],
    fake_claude_subprocess,  # noqa: ANN001
) -> None:
    """The final metadata sums the per-message usage across the stream,
    mirroring the api driver's input_tokens / output_tokens final fields."""
    skill_dir, compiler_dir, work_dir = fake_skills
    fake_claude_subprocess(
        stdout_text=_stream(
            _assistant_line(text="one", usage={"input_tokens": 100, "output_tokens": 20}),
            _assistant_line(text="two", usage={"input_tokens": 250, "output_tokens": 30}),
            _result_line(result="done", num_turns=2),
        ),
        write_files={"pipeline.yaml": VALID_PIPELINE_YAML},
    )

    driver = ClaudeDriver()
    result = await driver.run(skill_dir, compiler_dir, work_dir)
    assert result.metadata["input_tokens"] == 350
    assert result.metadata["output_tokens"] is None
    assert result.metadata["usage_complete"] is False
    assert result.metadata["num_turns"] == 2


@pytest.mark.asyncio
async def test_custom_model_override(
    fake_skills: tuple[Path, Path, Path],
    fake_claude_subprocess,  # noqa: ANN001
) -> None:
    skill_dir, compiler_dir, work_dir = fake_skills
    recorder = fake_claude_subprocess(
        stdout_text=_stream(_result_line(result="ok")),
        write_files={"pipeline.yaml": VALID_PIPELINE_YAML},
    )

    driver = ClaudeDriver(model="claude-opus-4-6")
    await driver.run(skill_dir, compiler_dir, work_dir)

    args = list(recorder.calls[0]["args"])
    assert args[args.index("--model") + 1] == "claude-opus-4-6"


@pytest.mark.asyncio
async def test_missing_compiler_skill_md_fails_before_subprocess(
    tmp_path: Path,
    fake_claude_subprocess,  # noqa: ANN001
) -> None:
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("hi")

    bad_compiler = tmp_path / "no-skill-md"
    bad_compiler.mkdir()

    work_dir = tmp_path / "work"

    recorder = fake_claude_subprocess(
        stdout_text=_stream(_result_line(result="ok")),
        write_files={"pipeline.yaml": VALID_PIPELINE_YAML},
    )

    driver = ClaudeDriver()
    with pytest.raises(DriverError, match="rote-compile SKILL.md not found"):
        await driver.run(skill_dir, bad_compiler, work_dir)

    assert len(recorder.calls) == 0


def test_web_tools_flag_appends_research_tools() -> None:
    """--backend api compilations get WebSearch/WebFetch so vendor code is
    written against current docs, not training-data memory."""
    from rote.compiler.drivers.claude import DEFAULT_ALLOWED_TOOLS, WEB_TOOLS, ClaudeDriver

    assert ClaudeDriver().allowed_tools == DEFAULT_ALLOWED_TOOLS  # off by default
    driver = ClaudeDriver(web_tools=True)
    assert driver.allowed_tools == f"{DEFAULT_ALLOWED_TOOLS},{WEB_TOOLS}"
    assert "WebSearch" in driver.allowed_tools
    assert "WebFetch" in driver.allowed_tools


def test_other_drivers_swallow_web_tools_kwarg() -> None:
    """The registry contract: cross-driver kwargs never hard-fail a
    driver that doesn't implement them."""
    from rote.compiler.drivers.anthropic_api import AnthropicApiDriver
    from rote.compiler.drivers.codex import CodexDriver

    CodexDriver(web_tools=True)
    AnthropicApiDriver(web_tools=True)


# ───────── Prompt-cache accounting ─────────


@pytest.mark.asyncio
async def test_split_assistant_blocks_count_one_response_but_keep_each_tool(
    fake_skills: tuple[Path, Path, Path],
    fake_claude_subprocess,
) -> None:
    """Parallel tool blocks share the response ID and its input/cache usage.

    Claude's documented stream contract matches the live self-compile's
    repeated 3 input / 61263 cache-write snapshots. These are separate
    blocks from one response, not additional model requests.
    """
    skill_dir, compiler_dir, work_dir = fake_skills
    usage = {
        "input_tokens": 3,
        "output_tokens": 7,
        "cache_creation_input_tokens": 61263,
        "cache_read_input_tokens": 0,
    }
    fake_claude_subprocess(
        stdout_text=_stream(
            _assistant_line(text="Reading the skill.", usage=usage, message_id="msg_one"),
            _assistant_line(
                tools=[("Read", {"file_path": "SKILL.md"})], usage=usage, message_id="msg_one"
            ),
            _assistant_line(
                tools=[("Read", {"file_path": "references/a.md"})],
                usage=usage,
                message_id="msg_one",
            ),
            _assistant_line(text="Next response", usage=usage, message_id="msg_two"),
        ),
        write_files={"pipeline.yaml": VALID_PIPELINE_YAML},
    )
    events: list[CompilationEvent] = []
    result = await ClaudeDriver().run(skill_dir, compiler_dir, work_dir, on_event=events.append)
    assert result.metadata["input_tokens"] == 6
    assert result.metadata["cache_write_tokens"] == 122526
    assert [event.turn for event in events if event.type == "turn"] == [1, 2]
    assert [(event.tool_name, event.turn) for event in events if event.type == "tool"] == [
        ("Read", 1),
        ("Read", 1),
    ]


@pytest.mark.asyncio
async def test_final_result_usage_replaces_stream_placeholders(
    fake_skills: tuple[Path, Path, Path],
    fake_claude_subprocess,
) -> None:
    """The result carries final output usage, even on error results."""
    skill_dir, compiler_dir, work_dir = fake_skills
    fake_claude_subprocess(
        stdout_text=_stream(
            _assistant_line(usage={"input_tokens": 3, "output_tokens": 7}, message_id="msg_one"),
            _result_line(
                subtype="error_max_turns",
                num_turns=9,
                total_cost_usd=0.42,
                usage={
                    "input_tokens": 11,
                    "output_tokens": 2345,
                    "cache_creation_input_tokens": 61263,
                    "cache_read_input_tokens": 98765,
                },
            ),
        ),
        write_files={"pipeline.yaml": VALID_PIPELINE_YAML},
    )
    result = await ClaudeDriver().run(skill_dir, compiler_dir, work_dir)
    assert result.metadata["input_tokens"] == 11
    assert result.metadata["output_tokens"] == 2345
    assert result.metadata["cache_write_tokens"] == 61263
    assert result.metadata["cache_read_tokens"] == 98765
    assert result.metadata["num_turns"] == 9
    assert result.metadata["cost_usd"] == 0.42


def test_cache_buckets_are_counted_not_dropped() -> None:
    """A prompt-cached message's input lands in the cache buckets.

    ``claude -p`` always runs prompt-cached, so ``input_tokens`` holds
    only the few tokens that were neither written to nor read from the
    cache. The usage dict below is the shape a real CLI call returns: a
    one-word reply reported ``input_tokens: 3`` beside
    ``cache_creation_input_tokens: 122956``.

    Summing the plain pair alone reports that run as 3 input tokens, and
    every cost derived from it is wrong by the same factor.
    """
    totals: dict[str, int] = {"input": 0, "output": 0, "cache_write": 0, "cache_read": 0}
    events, _turn, _ = _map_stream_line(
        _assistant_line(
            text="ok",
            usage={
                "input_tokens": 3,
                "cache_creation_input_tokens": 122956,
                "cache_read_input_tokens": 41000,
                "output_tokens": 2,
            },
        ),
        0,
        None,
        totals,
    )
    assert events[0].tokens == {
        "input": 3,
        "cache_write": 122956,
        "cache_read": 41000,
    }
    # The negative half: the plain field must not absorb the cache volume
    # either, or the buckets become unpriceable at their own rates.
    assert totals["input"] == 3


@pytest.mark.asyncio
async def test_metadata_carries_cache_buckets(
    fake_skills: tuple[Path, Path, Path],
    fake_claude_subprocess,  # noqa: ANN001
) -> None:
    """The cache buckets survive all the way into driver metadata.

    The accumulator counting them is not enough; ``_parse_metadata`` has
    to forward them, since that dict is what the CLI prices and prints.
    """
    skill_dir, compiler_dir, work_dir = fake_skills
    fake_claude_subprocess(
        stdout_text=_stream(
            _assistant_line(
                text="one",
                usage={
                    "input_tokens": 3,
                    "cache_creation_input_tokens": 100_000,
                    "cache_read_input_tokens": 0,
                    "output_tokens": 20,
                },
            ),
            _assistant_line(
                text="two",
                usage={
                    "input_tokens": 4,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 100_000,
                    "output_tokens": 30,
                },
            ),
            _result_line(result="done", num_turns=2),
        ),
        write_files={"pipeline.yaml": VALID_PIPELINE_YAML},
    )

    driver = ClaudeDriver()
    result = await driver.run(skill_dir, compiler_dir, work_dir)
    assert result.metadata["input_tokens"] == 7
    assert result.metadata["output_tokens"] is None
    assert result.metadata["usage_complete"] is False
    assert result.metadata["cache_write_tokens"] == 100_000
    assert result.metadata["cache_read_tokens"] == 100_000


@pytest.mark.asyncio
async def test_cost_reads_total_cost_usd_and_ignores_an_invented_key(
    fake_skills: tuple[Path, Path, Path],
    fake_claude_subprocess,  # noqa: ANN001
) -> None:
    """The envelope's cost key is ``total_cost_usd``.

    There is no ``cost_usd`` key in a real ``claude -p`` result line, so
    reading that name returned ``None`` on every run. Both halves matter:
    the real key must be read, and a payload carrying only the invented
    name must not satisfy the assertion, or a revert would pass.
    """
    skill_dir, compiler_dir, work_dir = fake_skills

    fake_claude_subprocess(
        stdout_text=_stream(
            _assistant_line(text="working"),
            _result_line(result="ok", total_cost_usd=0.7378),
        ),
        write_files={"pipeline.yaml": VALID_PIPELINE_YAML},
    )
    result = await ClaudeDriver().run(skill_dir, compiler_dir, work_dir)
    assert result.metadata["cost_usd"] == 0.7378

    fake_claude_subprocess(
        stdout_text=_stream(
            _assistant_line(text="working"),
            _result_line(result="ok", cost_usd=0.7378),
        ),
        write_files={"pipeline.yaml": VALID_PIPELINE_YAML},
    )
    result = await ClaudeDriver().run(skill_dir, compiler_dir, work_dir)
    assert result.metadata["cost_usd"] is None
