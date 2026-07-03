"""Tests for the ClaudeDriver subprocess implementation.

We mock ``asyncio.create_subprocess_exec`` to simulate Claude Code
spawning without ever invoking the real CLI. The fake process:

1. Captures the args and env it was invoked with
2. Optionally performs side effects (writing files to work_dir) to
   mimic what a real agent run would produce
3. Returns canned stdout/stderr/returncode so the driver's output
   parser is exercised

Coverage:

* Happy path: args are correct, env is scrubbed, metadata parsed
* Env scrubbing: ANTHROPIC_API_KEY and ANTHROPIC_AUTH_TOKEN removed
* CLAUDE_CODE_OAUTH_TOKEN passed through if set
* Animation-silence env var is set
* Nonzero exit → DriverError with stderr in details
* Exit 0 but no pipeline.yaml → DriverError
* Malformed JSON → metadata["raw_output"]
* Parseable JSON on the last line only → recovered
* Missing graduator SKILL.md → DriverError before subprocess
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rote.graduator.drivers import DriverError
from rote.graduator.drivers.claude import (
    DEFAULT_ALLOWED_TOOLS,
    DEFAULT_MAX_TURNS,
    DEFAULT_MODEL,
    ClaudeDriver,
)

# ───────── Fake subprocess ─────────


class _FakeProcess:
    def __init__(
        self,
        stdout_bytes: bytes,
        stderr_bytes: bytes,
        returncode: int,
    ) -> None:
        self._stdout = stdout_bytes
        self._stderr = stderr_bytes
        self.returncode = returncode

    async def communicate(self) -> tuple[bytes, bytes]:
        return self._stdout, self._stderr


class _FakeSubprocessRecorder:
    """Captures every ``create_subprocess_exec`` call and returns a fake proc.

    Test code instantiates this, configures the canned response(s), and
    installs it via monkeypatch. The recorder exposes ``calls`` for
    post-hoc assertions.
    """

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
        # Snapshot env so later test modifications don't pollute it
        self.calls.append({"args": args, "env": dict(env)})

        # Simulate the side effect: a real ``claude -p`` run would
        # write files into the work dir via its Write tool.
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
        """Pull the third --add-dir value out of the argv.

        The ClaudeDriver passes three ``--add-dir`` flags in the order
        skill_dir, graduator_skill_dir, work_dir. We find all three
        and return the last one, which is the work dir.
        """
        add_dir_values: list[str] = []
        for i, a in enumerate(args):
            if a == "--add-dir" and i + 1 < len(args):
                add_dir_values.append(args[i + 1])
        if len(add_dir_values) >= 3:
            return Path(add_dir_values[-1])
        return None


@pytest.fixture
def fake_claude_subprocess(monkeypatch: pytest.MonkeyPatch):  # noqa: ANN201
    """Fixture that installs a fake ``create_subprocess_exec`` and
    ``which`` and returns the recorder for post-hoc assertions."""

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
            "rote.graduator.drivers.claude.asyncio.create_subprocess_exec",
            recorder,
        )
        # Pretend `claude` is on PATH so is_available() would pass
        monkeypatch.setattr(
            "rote.graduator.drivers.claude.which",
            lambda name: "/usr/local/bin/claude" if name == "claude" else None,
        )
        return recorder

    return _install


# ───────── Skill bundle fixture ─────────


@pytest.fixture
def fake_skills(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create a fake source skill, fake graduator skill, and work dir."""
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: fake\n---\n\n# Fake skill\n",
        encoding="utf-8",
    )

    graduator_dir = tmp_path / "graduator"
    graduator_dir.mkdir()
    (graduator_dir / "SKILL.md").write_text(
        "# Fake graduator rubric\n\nDo the graduation thing.\n",
        encoding="utf-8",
    )
    (graduator_dir / "references").mkdir()

    work_dir = tmp_path / "work"
    return skill_dir, graduator_dir, work_dir


# A small valid pipeline.yaml the fake agent "writes" during the run.
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
    """End-to-end: driver spawns claude with the expected args, env is
    scrubbed, pipeline.yaml is produced, metadata is parsed."""
    skill_dir, graduator_dir, work_dir = fake_skills

    # Set both scrubbable vars in the parent env to prove they're
    # removed from the child env.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-be-scrubbed")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "auth-should-be-scrubbed")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-should-pass-through")

    metadata_json = json.dumps(
        {
            "result": "Graduated successfully.",
            "cost_usd": 0.42,
            "duration_ms": 58000,
            "num_turns": 18,
            "session_id": "sess-abc-123",
        }
    )

    recorder = fake_claude_subprocess(
        stdout_text=metadata_json,
        returncode=0,
        write_files={"pipeline.yaml": VALID_PIPELINE_YAML},
    )

    driver = ClaudeDriver()
    result = await driver.run(skill_dir, graduator_dir, work_dir)

    # Result shape
    assert result.driver_name == "claude"
    assert result.pipeline_yaml_path == (work_dir / "pipeline.yaml").resolve()
    assert result.pipeline_yaml_path.read_text() == VALID_PIPELINE_YAML

    # Metadata parsed from stdout
    assert result.metadata["driver"] == "claude"
    assert result.metadata["cost_usd"] == 0.42
    assert result.metadata["duration_ms"] == 58000
    assert result.metadata["num_turns"] == 18
    assert result.metadata["session_id"] == "sess-abc-123"

    # One subprocess call
    assert len(recorder.calls) == 1
    call = recorder.calls[0]

    # Args include the expected flags
    args = list(call["args"])
    assert args[0] == "claude"
    assert "-p" in args
    assert "--model" in args
    # Model flag's value is immediately after --model
    assert args[args.index("--model") + 1] == DEFAULT_MODEL
    assert "--append-system-prompt" in args
    assert "--add-dir" in args
    assert args.count("--add-dir") == 3  # skill, graduator, work
    assert "--allowedTools" in args
    assert DEFAULT_ALLOWED_TOOLS in args
    assert "--output-format" in args
    assert "json" in args
    assert "--max-turns" in args
    assert str(DEFAULT_MAX_TURNS) in args

    # Env was scrubbed of API key vars
    env = call["env"]
    assert "ANTHROPIC_API_KEY" not in env
    assert "ANTHROPIC_AUTH_TOKEN" not in env

    # Animation silence flag set
    assert env.get("CLAUDE_CODE_DISABLE_NONINTERACTIVE_ANIMATIONS") == "1"

    # OAuth token passed through
    assert env.get("CLAUDE_CODE_OAUTH_TOKEN") == "sk-ant-oat01-should-pass-through"


@pytest.mark.asyncio
async def test_prompt_mentions_all_three_paths(
    fake_skills: tuple[Path, Path, Path],
    fake_claude_subprocess,  # noqa: ANN001
) -> None:
    """The user prompt should reference the source skill, graduator,
    and work directories by absolute path so the agent knows where
    to read from and write to."""
    skill_dir, graduator_dir, work_dir = fake_skills
    recorder = fake_claude_subprocess(
        stdout_text='{"result": "ok"}',
        write_files={"pipeline.yaml": VALID_PIPELINE_YAML},
    )

    driver = ClaudeDriver()
    await driver.run(skill_dir, graduator_dir, work_dir)

    args = list(recorder.calls[0]["args"])
    # The prompt immediately follows -p
    prompt_idx = args.index("-p") + 1
    user_prompt = args[prompt_idx]

    assert str(skill_dir.resolve()) in user_prompt
    assert str(graduator_dir.resolve()) in user_prompt
    assert str(work_dir.resolve()) in user_prompt
    assert "pipeline.yaml" in user_prompt


@pytest.mark.asyncio
async def test_system_prompt_contains_graduator_skill(
    fake_skills: tuple[Path, Path, Path],
    fake_claude_subprocess,  # noqa: ANN001
) -> None:
    """The system prompt should include the rote-graduate SKILL.md
    content verbatim so the agent has the rubric loaded."""
    skill_dir, graduator_dir, work_dir = fake_skills
    recorder = fake_claude_subprocess(
        stdout_text='{"result": "ok"}',
        write_files={"pipeline.yaml": VALID_PIPELINE_YAML},
    )

    driver = ClaudeDriver()
    await driver.run(skill_dir, graduator_dir, work_dir)

    args = list(recorder.calls[0]["args"])
    sys_idx = args.index("--append-system-prompt") + 1
    system_prompt = args[sys_idx]

    assert "ROTE GRADUATE SKILL" in system_prompt
    assert "Fake graduator rubric" in system_prompt


# ───────── Failure modes ─────────


@pytest.mark.asyncio
async def test_nonzero_exit_raises_driver_error_with_stderr(
    fake_skills: tuple[Path, Path, Path],
    fake_claude_subprocess,  # noqa: ANN001
) -> None:
    skill_dir, graduator_dir, work_dir = fake_skills
    fake_claude_subprocess(
        stdout_text="",
        stderr_text="Error: rate limit exceeded",
        returncode=1,
    )

    driver = ClaudeDriver()
    with pytest.raises(DriverError) as excinfo:
        await driver.run(skill_dir, graduator_dir, work_dir)
    assert "exited with code 1" in str(excinfo.value)
    assert "rate limit exceeded" in (excinfo.value.details or "")


@pytest.mark.asyncio
async def test_nonzero_exit_with_pipeline_yaml_recovers_run(
    fake_skills: tuple[Path, Path, Path],
    fake_claude_subprocess,  # noqa: ANN001
) -> None:
    """If the agent wrote pipeline.yaml before the subprocess errored
    out (e.g. transient ECONNRESET on a final validation turn), we
    treat the run as success and surface the error in metadata.

    The deliverable is a file on disk, not a clean exit code — losing
    a $2+ run because of a network blip after the work was done is
    unacceptable."""
    skill_dir, graduator_dir, work_dir = fake_skills
    fake_claude_subprocess(
        stdout_text=(
            '{"result": "API Error: ECONNRESET", "is_error": true, "total_cost_usd": 2.5}'
        ),
        stderr_text="",
        returncode=1,
        write_files={"pipeline.yaml": VALID_PIPELINE_YAML},
    )

    driver = ClaudeDriver()
    result = await driver.run(skill_dir, graduator_dir, work_dir)
    assert result.pipeline_yaml_path.is_file()
    assert "subprocess_warning" in result.metadata
    assert "exited with code 1" in result.metadata["subprocess_warning"]


@pytest.mark.asyncio
async def test_exit_zero_without_pipeline_yaml_raises_driver_error(
    fake_skills: tuple[Path, Path, Path],
    fake_claude_subprocess,  # noqa: ANN001
) -> None:
    skill_dir, graduator_dir, work_dir = fake_skills
    fake_claude_subprocess(
        stdout_text='{"result": "I gave up"}',
        returncode=0,
        write_files=None,  # agent writes nothing
    )

    driver = ClaudeDriver()
    with pytest.raises(DriverError, match="did not produce"):
        await driver.run(skill_dir, graduator_dir, work_dir)


@pytest.mark.asyncio
async def test_missing_graduator_skill_md_fails_before_subprocess(
    tmp_path: Path,
    fake_claude_subprocess,  # noqa: ANN001
) -> None:
    """If the graduator skill dir doesn't have a SKILL.md, we should
    bail before even spawning claude."""
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("hi")

    bad_graduator = tmp_path / "no-skill-md"
    bad_graduator.mkdir()

    work_dir = tmp_path / "work"

    recorder = fake_claude_subprocess(
        stdout_text='{"result": "ok"}',
        write_files={"pipeline.yaml": VALID_PIPELINE_YAML},
    )

    driver = ClaudeDriver()
    with pytest.raises(DriverError, match="rote-graduate SKILL.md not found"):
        await driver.run(skill_dir, bad_graduator, work_dir)

    # No subprocess was spawned
    assert len(recorder.calls) == 0


# ───────── Metadata parsing edge cases ─────────


@pytest.mark.asyncio
async def test_malformed_json_falls_back_to_raw_output(
    fake_skills: tuple[Path, Path, Path],
    fake_claude_subprocess,  # noqa: ANN001
) -> None:
    """If stdout isn't valid JSON, metadata should still carry a
    truncated raw_output for debugging."""
    skill_dir, graduator_dir, work_dir = fake_skills
    fake_claude_subprocess(
        stdout_text="this is definitely not json",
        returncode=0,
        write_files={"pipeline.yaml": VALID_PIPELINE_YAML},
    )

    driver = ClaudeDriver()
    result = await driver.run(skill_dir, graduator_dir, work_dir)

    assert result.metadata["driver"] == "claude"
    assert "raw_output" in result.metadata
    assert "not json" in result.metadata["raw_output"]


@pytest.mark.asyncio
async def test_json_on_last_line_after_preamble(
    fake_skills: tuple[Path, Path, Path],
    fake_claude_subprocess,  # noqa: ANN001
) -> None:
    """Some claude versions may print a banner before the JSON.
    The parser should still find the JSON on a later line."""
    skill_dir, graduator_dir, work_dir = fake_skills
    stdout = (
        "Loading skills...\n"
        "Loaded 2 skills.\n"
        '{"result": "ok", "cost_usd": 0.25, "num_turns": 10, '
        '"session_id": "s1", "duration_ms": 1000}\n'
    )
    fake_claude_subprocess(
        stdout_text=stdout,
        returncode=0,
        write_files={"pipeline.yaml": VALID_PIPELINE_YAML},
    )

    driver = ClaudeDriver()
    result = await driver.run(skill_dir, graduator_dir, work_dir)

    assert result.metadata["cost_usd"] == 0.25
    assert result.metadata["num_turns"] == 10


@pytest.mark.asyncio
async def test_custom_model_override(
    fake_skills: tuple[Path, Path, Path],
    fake_claude_subprocess,  # noqa: ANN001
) -> None:
    """Passing ``model=...`` to ClaudeDriver should flow through to
    the ``--model`` flag on the subprocess."""
    skill_dir, graduator_dir, work_dir = fake_skills
    recorder = fake_claude_subprocess(
        stdout_text='{"result": "ok"}',
        write_files={"pipeline.yaml": VALID_PIPELINE_YAML},
    )

    driver = ClaudeDriver(model="claude-opus-4-6")
    await driver.run(skill_dir, graduator_dir, work_dir)

    args = list(recorder.calls[0]["args"])
    assert args[args.index("--model") + 1] == "claude-opus-4-6"


@pytest.mark.asyncio
async def test_empty_stdout_returns_minimal_metadata(
    fake_skills: tuple[Path, Path, Path],
    fake_claude_subprocess,  # noqa: ANN001
) -> None:
    skill_dir, graduator_dir, work_dir = fake_skills
    fake_claude_subprocess(
        stdout_text="",
        returncode=0,
        write_files={"pipeline.yaml": VALID_PIPELINE_YAML},
    )

    driver = ClaudeDriver()
    result = await driver.run(skill_dir, graduator_dir, work_dir)

    # Should still return a result; metadata just has the driver name
    assert result.metadata == {"driver": "claude"}
