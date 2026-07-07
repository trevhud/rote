"""Tests for the CodexDriver subprocess implementation.

Mirrors ``test_claude_driver.py``: we mock ``asyncio.create_subprocess_exec``
to simulate ``codex exec`` spawning without ever invoking the real CLI.
The fake process captures the args/env it was invoked with, optionally
writes files into work_dir (mimicking the agent), and returns canned
stdout/stderr/returncode.

Coverage:

* Happy path: exec args are correct (sandbox, cd, skip-git, no approval
  flag, no --add-dir), env is passed through, metadata carries the
  captured last message
* Sandbox must be workspace-write (exec defaults to read-only)
* No ``--ask-for-approval`` / ``--add-dir`` flags are emitted
* Model override flows to ``--model``; omitted when None
* OPENAI_API_KEY / CODEX_API_KEY are NOT scrubbed (pass-through auth)
* Nonzero exit without pipeline.yaml → DriverError with stderr in details
* Nonzero exit WITH pipeline.yaml → recovered, warning in metadata
* Exit 0 but no pipeline.yaml → DriverError
* Missing graduator SKILL.md → DriverError before subprocess
* The last-message temp file is cleaned up and never lands in work_dir
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from rote.graduator.drivers import DriverError
from rote.graduator.drivers.codex import DEFAULT_SANDBOX, CodexDriver

# ───────── Fake subprocess ─────────


class _FakeProcess:
    def __init__(self, stdout_bytes: bytes, stderr_bytes: bytes, returncode: int) -> None:
        self._stdout = stdout_bytes
        self._stderr = stderr_bytes
        self.returncode = returncode

    async def communicate(self) -> tuple[bytes, bytes]:
        return self._stdout, self._stderr


class _FakeSubprocessRecorder:
    """Captures every ``create_subprocess_exec`` call and returns a fake proc.

    Writes ``write_files`` into work_dir (pulled from the ``--cd`` argv
    value) and, when configured, writes ``last_message`` into the file
    named by ``--output-last-message`` to mimic Codex's behavior.
    """

    def __init__(
        self,
        *,
        stdout_text: str = "",
        stderr_text: str = "",
        returncode: int = 0,
        write_files: dict[str, str] | None = None,
        last_message: str | None = None,
    ) -> None:
        self.stdout_text = stdout_text
        self.stderr_text = stderr_text
        self.returncode = returncode
        self.write_files = write_files or {}
        self.last_message = last_message
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, *args: str, **kwargs: Any) -> _FakeProcess:
        env = kwargs.get("env") or {}
        self.calls.append({"args": args, "env": dict(env)})

        work_dir = self._flag_value(args, "--cd")
        if self.write_files and work_dir is not None:
            for rel_path, content in self.write_files.items():
                target = Path(work_dir) / rel_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")

        if self.last_message is not None:
            last_path = self._flag_value(args, "--output-last-message")
            if last_path is not None:
                Path(last_path).write_text(self.last_message, encoding="utf-8")

        return _FakeProcess(
            self.stdout_text.encode("utf-8"),
            self.stderr_text.encode("utf-8"),
            self.returncode,
        )

    @staticmethod
    def _flag_value(args: tuple[str, ...], flag: str) -> str | None:
        for i, a in enumerate(args):
            if a == flag and i + 1 < len(args):
                return args[i + 1]
        return None


@pytest.fixture
def fake_codex_subprocess(monkeypatch: pytest.MonkeyPatch):  # noqa: ANN201
    def _install(
        *,
        stdout_text: str = "",
        stderr_text: str = "",
        returncode: int = 0,
        write_files: dict[str, str] | None = None,
        last_message: str | None = None,
    ) -> _FakeSubprocessRecorder:
        recorder = _FakeSubprocessRecorder(
            stdout_text=stdout_text,
            stderr_text=stderr_text,
            returncode=returncode,
            write_files=write_files,
            last_message=last_message,
        )
        monkeypatch.setattr(
            "rote.graduator.drivers.codex.asyncio.create_subprocess_exec",
            recorder,
        )
        monkeypatch.setattr(
            "rote.graduator.drivers.codex.which",
            lambda name: "/usr/local/bin/codex" if name == "codex" else None,
        )
        return recorder

    return _install


# ───────── Skill bundle fixture ─────────


@pytest.fixture
def fake_skills(tmp_path: Path) -> tuple[Path, Path, Path]:
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: fake\n---\n\n# Fake skill\n", encoding="utf-8")

    graduator_dir = tmp_path / "graduator"
    graduator_dir.mkdir()
    (graduator_dir / "SKILL.md").write_text(
        "# Fake graduator rubric\n\nDo the graduation thing.\n", encoding="utf-8"
    )
    (graduator_dir / "references").mkdir()

    work_dir = tmp_path / "work"
    return skill_dir, graduator_dir, work_dir


VALID_PIPELINE_YAML = """\
name: fake-pipeline
version: "0.1.0"
description: |
  Fake pipeline used for CodexDriver testing.

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
    fake_codex_subprocess,  # noqa: ANN001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill_dir, graduator_dir, work_dir = fake_skills

    # Both key vars set in the parent env — prove they pass THROUGH
    # (Codex driver does not scrub, unlike the Claude driver).
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-pass-through")
    monkeypatch.setenv("CODEX_API_KEY", "codex-should-pass-through")

    recorder = fake_codex_subprocess(
        stdout_text="Graduated.",
        returncode=0,
        write_files={"pipeline.yaml": VALID_PIPELINE_YAML},
        last_message="Wrote pipeline.yaml with 1 node.",
    )

    driver = CodexDriver()
    result = await driver.run(skill_dir, graduator_dir, work_dir)

    assert result.driver_name == "codex"
    assert result.pipeline_yaml_path == (work_dir / "pipeline.yaml").resolve()
    assert result.pipeline_yaml_path.read_text() == VALID_PIPELINE_YAML
    assert result.metadata["driver"] == "codex"
    assert result.metadata["last_message"] == "Wrote pipeline.yaml with 1 node."

    assert len(recorder.calls) == 1
    args = list(recorder.calls[0]["args"])

    assert args[0] == "codex"
    assert args[1] == "exec"
    # cd points at the (resolved) work dir
    assert args[args.index("--cd") + 1] == str(work_dir.resolve())
    # sandbox is workspace-write (exec defaults to read-only, which can't write)
    assert args[args.index("--sandbox") + 1] == DEFAULT_SANDBOX == "workspace-write"
    assert "--skip-git-repo-check" in args
    assert "--output-last-message" in args
    # The prompt is the final positional arg.
    prompt = args[-1]
    assert str(skill_dir.resolve()) in prompt
    assert str(work_dir.resolve()) in prompt
    assert "ROTE GRADUATE SKILL" in prompt
    assert "Fake graduator rubric" in prompt

    # Env passed through untouched.
    env = recorder.calls[0]["env"]
    assert env.get("OPENAI_API_KEY") == "sk-should-pass-through"
    assert env.get("CODEX_API_KEY") == "codex-should-pass-through"


@pytest.mark.asyncio
async def test_does_not_emit_removed_or_wrong_flags(
    fake_skills: tuple[Path, Path, Path],
    fake_codex_subprocess,  # noqa: ANN001
) -> None:
    """``--ask-for-approval`` hard-errors on exec, and ``--add-dir`` would
    wrongly grant write access to the read-only source skill. Neither must
    appear."""
    skill_dir, graduator_dir, work_dir = fake_skills
    recorder = fake_codex_subprocess(
        write_files={"pipeline.yaml": VALID_PIPELINE_YAML},
    )

    driver = CodexDriver()
    await driver.run(skill_dir, graduator_dir, work_dir)

    args = list(recorder.calls[0]["args"])
    assert "--ask-for-approval" not in args
    assert "--add-dir" not in args


@pytest.mark.asyncio
async def test_model_override_flows_to_flag(
    fake_skills: tuple[Path, Path, Path],
    fake_codex_subprocess,  # noqa: ANN001
) -> None:
    skill_dir, graduator_dir, work_dir = fake_skills
    recorder = fake_codex_subprocess(write_files={"pipeline.yaml": VALID_PIPELINE_YAML})

    driver = CodexDriver(model="gpt-5.1-codex")
    await driver.run(skill_dir, graduator_dir, work_dir)

    args = list(recorder.calls[0]["args"])
    assert args[args.index("--model") + 1] == "gpt-5.1-codex"


@pytest.mark.asyncio
async def test_no_model_flag_when_default(
    fake_skills: tuple[Path, Path, Path],
    fake_codex_subprocess,  # noqa: ANN001
) -> None:
    """With no model set we omit ``--model`` and let Codex use its
    configured default."""
    skill_dir, graduator_dir, work_dir = fake_skills
    recorder = fake_codex_subprocess(write_files={"pipeline.yaml": VALID_PIPELINE_YAML})

    driver = CodexDriver()
    await driver.run(skill_dir, graduator_dir, work_dir)

    assert "--model" not in list(recorder.calls[0]["args"])


@pytest.mark.asyncio
async def test_absorbs_unknown_kwargs(
    fake_skills: tuple[Path, Path, Path],
    fake_codex_subprocess,  # noqa: ANN001
) -> None:
    """The registry forwards kwargs like max_turns; Codex has no such
    concept and must swallow them, not crash."""
    skill_dir, graduator_dir, work_dir = fake_skills
    fake_codex_subprocess(write_files={"pipeline.yaml": VALID_PIPELINE_YAML})

    driver = CodexDriver(model=None, max_turns=99)  # type: ignore[call-arg]
    result = await driver.run(skill_dir, graduator_dir, work_dir)
    assert result.pipeline_yaml_path.is_file()


# ───────── Failure modes ─────────


@pytest.mark.asyncio
async def test_nonzero_exit_raises_driver_error_with_stderr(
    fake_skills: tuple[Path, Path, Path],
    fake_codex_subprocess,  # noqa: ANN001
) -> None:
    skill_dir, graduator_dir, work_dir = fake_skills
    fake_codex_subprocess(stderr_text="Error: not authenticated", returncode=1)

    driver = CodexDriver()
    with pytest.raises(DriverError) as excinfo:
        await driver.run(skill_dir, graduator_dir, work_dir)
    assert "exited with code 1" in str(excinfo.value)
    assert "not authenticated" in (excinfo.value.details or "")


@pytest.mark.asyncio
async def test_nonzero_exit_with_pipeline_yaml_recovers_run(
    fake_skills: tuple[Path, Path, Path],
    fake_codex_subprocess,  # noqa: ANN001
) -> None:
    """A nonzero exit after pipeline.yaml was written (transient blip on a
    final self-check turn) is treated as success with a warning."""
    skill_dir, graduator_dir, work_dir = fake_skills
    fake_codex_subprocess(
        stderr_text="stream error",
        returncode=1,
        write_files={"pipeline.yaml": VALID_PIPELINE_YAML},
    )

    driver = CodexDriver()
    result = await driver.run(skill_dir, graduator_dir, work_dir)
    assert result.pipeline_yaml_path.is_file()
    assert "subprocess_warning" in result.metadata
    assert "exited with code 1" in result.metadata["subprocess_warning"]


@pytest.mark.asyncio
async def test_exit_zero_without_pipeline_yaml_raises(
    fake_skills: tuple[Path, Path, Path],
    fake_codex_subprocess,  # noqa: ANN001
) -> None:
    skill_dir, graduator_dir, work_dir = fake_skills
    fake_codex_subprocess(stdout_text="I could not finish.", returncode=0, write_files=None)

    driver = CodexDriver()
    with pytest.raises(DriverError, match="did not produce"):
        await driver.run(skill_dir, graduator_dir, work_dir)


@pytest.mark.asyncio
async def test_missing_graduator_skill_md_fails_before_subprocess(
    tmp_path: Path,
    fake_codex_subprocess,  # noqa: ANN001
) -> None:
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("hi")
    bad_graduator = tmp_path / "no-skill-md"
    bad_graduator.mkdir()
    work_dir = tmp_path / "work"

    recorder = fake_codex_subprocess(write_files={"pipeline.yaml": VALID_PIPELINE_YAML})

    driver = CodexDriver()
    with pytest.raises(DriverError, match="rote-graduate SKILL.md not found"):
        await driver.run(skill_dir, bad_graduator, work_dir)
    assert len(recorder.calls) == 0


# ───────── Hygiene ─────────


@pytest.mark.asyncio
async def test_last_message_tempfile_is_cleaned_up_and_out_of_tree(
    fake_skills: tuple[Path, Path, Path],
    fake_codex_subprocess,  # noqa: ANN001
) -> None:
    """The --output-last-message file must live outside work_dir (so it's
    not swept into the user's output) and be unlinked after the run."""
    skill_dir, graduator_dir, work_dir = fake_skills
    recorder = fake_codex_subprocess(
        write_files={"pipeline.yaml": VALID_PIPELINE_YAML},
        last_message="done",
    )

    driver = CodexDriver()
    await driver.run(skill_dir, graduator_dir, work_dir)

    last_value = _FakeSubprocessRecorder._flag_value(
        recorder.calls[0]["args"], "--output-last-message"
    )
    assert last_value is not None
    last_path = Path(last_value)
    # Outside the work dir…
    assert work_dir.resolve() not in last_path.resolve().parents
    # …and cleaned up.
    assert not last_path.exists()
    # work_dir contains only the agent's real output, no stray temp file.
    assert {p.name for p in work_dir.iterdir()} == {"pipeline.yaml"}


@pytest.mark.asyncio
async def test_empty_last_message_yields_minimal_metadata(
    fake_skills: tuple[Path, Path, Path],
    fake_codex_subprocess,  # noqa: ANN001
) -> None:
    skill_dir, graduator_dir, work_dir = fake_skills
    fake_codex_subprocess(
        write_files={"pipeline.yaml": VALID_PIPELINE_YAML},
        last_message="",
    )

    driver = CodexDriver()
    result = await driver.run(skill_dir, graduator_dir, work_dir)
    assert result.metadata == {"driver": "codex"}
