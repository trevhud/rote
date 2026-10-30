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
* Missing compiler SKILL.md → DriverError before subprocess
* The last-message temp file is cleaned up and never lands in work_dir
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from rote.compiler.drivers import DriverError
from rote.compiler.drivers.codex import DEFAULT_SANDBOX, CodexDriver

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
            "rote.compiler.drivers.codex.asyncio.create_subprocess_exec",
            recorder,
        )
        monkeypatch.setattr(
            "rote.compiler.drivers.codex.which",
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

    compiler_dir = tmp_path / "compiler"
    compiler_dir.mkdir()
    (compiler_dir / "SKILL.md").write_text(
        "# Fake compiler rubric\n\nDo the compilation thing.\n", encoding="utf-8"
    )
    (compiler_dir / "references").mkdir()

    work_dir = tmp_path / "work"
    return skill_dir, compiler_dir, work_dir


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
    skill_dir, compiler_dir, work_dir = fake_skills

    # Both key vars set in the parent env — prove they pass THROUGH
    # (Codex driver does not scrub, unlike the Claude driver).
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-pass-through")
    monkeypatch.setenv("CODEX_API_KEY", "codex-should-pass-through")

    recorder = fake_codex_subprocess(
        stdout_text="Compiled.",
        returncode=0,
        write_files={"pipeline.yaml": VALID_PIPELINE_YAML},
        last_message="Wrote pipeline.yaml with 1 node.",
    )

    driver = CodexDriver()
    result = await driver.run(skill_dir, compiler_dir, work_dir)

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
    assert "ROTE COMPILE SKILL" in prompt
    assert "Fake compiler rubric" in prompt

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
    skill_dir, compiler_dir, work_dir = fake_skills
    recorder = fake_codex_subprocess(
        write_files={"pipeline.yaml": VALID_PIPELINE_YAML},
    )

    driver = CodexDriver()
    await driver.run(skill_dir, compiler_dir, work_dir)

    args = list(recorder.calls[0]["args"])
    assert "--ask-for-approval" not in args
    assert "--add-dir" not in args


@pytest.mark.asyncio
async def test_model_override_flows_to_flag(
    fake_skills: tuple[Path, Path, Path],
    fake_codex_subprocess,  # noqa: ANN001
) -> None:
    skill_dir, compiler_dir, work_dir = fake_skills
    recorder = fake_codex_subprocess(write_files={"pipeline.yaml": VALID_PIPELINE_YAML})

    driver = CodexDriver(model="gpt-5.1-codex")
    await driver.run(skill_dir, compiler_dir, work_dir)

    args = list(recorder.calls[0]["args"])
    assert args[args.index("--model") + 1] == "gpt-5.1-codex"


@pytest.mark.asyncio
async def test_no_model_flag_when_default(
    fake_skills: tuple[Path, Path, Path],
    fake_codex_subprocess,  # noqa: ANN001
) -> None:
    """With no model set we omit ``--model`` and let Codex use its
    configured default."""
    skill_dir, compiler_dir, work_dir = fake_skills
    recorder = fake_codex_subprocess(write_files={"pipeline.yaml": VALID_PIPELINE_YAML})

    driver = CodexDriver()
    await driver.run(skill_dir, compiler_dir, work_dir)

    assert "--model" not in list(recorder.calls[0]["args"])


@pytest.mark.asyncio
async def test_absorbs_unknown_kwargs(
    fake_skills: tuple[Path, Path, Path],
    fake_codex_subprocess,  # noqa: ANN001
) -> None:
    """The registry forwards kwargs like max_turns; Codex has no such
    concept and must swallow them, not crash."""
    skill_dir, compiler_dir, work_dir = fake_skills
    fake_codex_subprocess(write_files={"pipeline.yaml": VALID_PIPELINE_YAML})

    driver = CodexDriver(model=None, max_turns=99)  # type: ignore[call-arg]
    result = await driver.run(skill_dir, compiler_dir, work_dir)
    assert result.pipeline_yaml_path.is_file()


# ───────── Failure modes ─────────


@pytest.mark.asyncio
async def test_nonzero_exit_raises_driver_error_with_stderr(
    fake_skills: tuple[Path, Path, Path],
    fake_codex_subprocess,  # noqa: ANN001
) -> None:
    skill_dir, compiler_dir, work_dir = fake_skills
    fake_codex_subprocess(stderr_text="Error: not authenticated", returncode=1)

    driver = CodexDriver()
    with pytest.raises(DriverError) as excinfo:
        await driver.run(skill_dir, compiler_dir, work_dir)
    assert "exited with code 1" in str(excinfo.value)
    assert "not authenticated" in (excinfo.value.details or "")


@pytest.mark.asyncio
async def test_nonzero_exit_with_pipeline_yaml_recovers_run(
    fake_skills: tuple[Path, Path, Path],
    fake_codex_subprocess,  # noqa: ANN001
) -> None:
    """A nonzero exit after pipeline.yaml was written (transient blip on a
    final self-check turn) is treated as success with a warning."""
    skill_dir, compiler_dir, work_dir = fake_skills
    fake_codex_subprocess(
        stderr_text="stream error",
        returncode=1,
        write_files={"pipeline.yaml": VALID_PIPELINE_YAML},
    )

    driver = CodexDriver()
    result = await driver.run(skill_dir, compiler_dir, work_dir)
    assert result.pipeline_yaml_path.is_file()
    assert "subprocess_warning" in result.metadata
    assert "exited with code 1" in result.metadata["subprocess_warning"]


@pytest.mark.asyncio
async def test_exit_zero_without_pipeline_yaml_raises(
    fake_skills: tuple[Path, Path, Path],
    fake_codex_subprocess,  # noqa: ANN001
) -> None:
    skill_dir, compiler_dir, work_dir = fake_skills
    fake_codex_subprocess(stdout_text="I could not finish.", returncode=0, write_files=None)

    driver = CodexDriver()
    with pytest.raises(DriverError, match="did not produce"):
        await driver.run(skill_dir, compiler_dir, work_dir)


@pytest.mark.asyncio
async def test_missing_compiler_skill_md_fails_before_subprocess(
    tmp_path: Path,
    fake_codex_subprocess,  # noqa: ANN001
) -> None:
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("hi")
    bad_compiler = tmp_path / "no-skill-md"
    bad_compiler.mkdir()
    work_dir = tmp_path / "work"

    recorder = fake_codex_subprocess(write_files={"pipeline.yaml": VALID_PIPELINE_YAML})

    driver = CodexDriver()
    with pytest.raises(DriverError, match="rote-compile SKILL.md not found"):
        await driver.run(skill_dir, bad_compiler, work_dir)
    assert len(recorder.calls) == 0


# ───────── Hygiene ─────────


@pytest.mark.asyncio
async def test_last_message_tempfile_is_cleaned_up_and_out_of_tree(
    fake_skills: tuple[Path, Path, Path],
    fake_codex_subprocess,  # noqa: ANN001
) -> None:
    """The --output-last-message file must live outside work_dir (so it's
    not swept into the user's output) and be unlinked after the run."""
    skill_dir, compiler_dir, work_dir = fake_skills
    recorder = fake_codex_subprocess(
        write_files={"pipeline.yaml": VALID_PIPELINE_YAML},
        last_message="done",
    )

    driver = CodexDriver()
    await driver.run(skill_dir, compiler_dir, work_dir)

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
    skill_dir, compiler_dir, work_dir = fake_skills
    fake_codex_subprocess(
        write_files={"pipeline.yaml": VALID_PIPELINE_YAML},
        last_message="",
    )

    driver = CodexDriver()
    result = await driver.run(skill_dir, compiler_dir, work_dir)
    assert result.metadata == {"driver": "codex", "repair_rounds": 0}


# ───────── Usage (#295) ─────────


def _event(obj: dict[str, Any]) -> str:
    import json

    return json.dumps(obj)


@pytest.mark.asyncio
async def test_usage_is_the_last_turn_completed_split_into_buckets(
    fake_skills: tuple[Path, Path, Path],
    fake_codex_subprocess,  # noqa: ANN001
) -> None:
    """``turn.completed`` carries the thread's running total (codex-rs fills
    it from ``ThreadTokenUsageUpdated.total``), so the last one is the run:
    summing would count the first turn twice. Its ``input_tokens`` holds the
    cached reads and cache writes, which are split out, never left inside
    the uncached input."""
    skill_dir, compiler_dir, work_dir = fake_skills
    stdout = "\n".join(
        [
            _event({"type": "thread.started", "thread_id": "t"}),
            _event({"type": "turn.started"}),
            _event(
                {
                    "type": "turn.completed",
                    "usage": {
                        "input_tokens": 100,
                        "cached_input_tokens": 40,
                        "cache_write_input_tokens": 10,
                        "output_tokens": 5,
                        "reasoning_output_tokens": 0,
                    },
                }
            ),
            "not json",
            _event({"type": "item.completed", "item": {"type": "agent_message", "text": "x"}}),
            _event(
                {
                    "type": "turn.completed",
                    "usage": {
                        "input_tokens": 300,
                        "cached_input_tokens": 120,
                        "cache_write_input_tokens": 30,
                        "output_tokens": 20,
                        "reasoning_output_tokens": 8,
                    },
                }
            ),
        ]
    )
    recorder = fake_codex_subprocess(
        stdout_text=stdout, write_files={"pipeline.yaml": VALID_PIPELINE_YAML}
    )
    result = await CodexDriver().run(skill_dir, compiler_dir, work_dir)
    assert "--json" in list(recorder.calls[0]["args"])
    meta = result.metadata
    assert meta["input_tokens"] == 150
    assert meta["cache_read_tokens"] == 120
    assert meta["cache_write_tokens"] == 30
    # Reasoning is already inside output_tokens: not added again.
    assert meta["output_tokens"] == 20
    assert meta["usage_source"] == "turn.completed"
    assert meta["usage_complete"] is True


@pytest.mark.asyncio
async def test_a_failed_turn_reports_no_usage_rather_than_zero(
    fake_skills: tuple[Path, Path, Path],
    fake_codex_subprocess,  # noqa: ANN001
) -> None:
    """A run that wrote the pipeline but never completed a turn has no
    usage report; no token key may appear, or it would read as zero."""
    skill_dir, compiler_dir, work_dir = fake_skills
    stdout = "\n".join(
        [
            _event({"type": "turn.started"}),
            _event({"type": "turn.failed", "error": {"message": "stream disconnected"}}),
        ]
    )
    fake_codex_subprocess(
        stdout_text=stdout, returncode=1, write_files={"pipeline.yaml": VALID_PIPELINE_YAML}
    )
    result = await CodexDriver().run(skill_dir, compiler_dir, work_dir)
    assert not {
        "input_tokens",
        "output_tokens",
        "cache_write_tokens",
        "cache_read_tokens",
        "usage_complete",
    } & set(result.metadata)


# ───────── Repair by resuming the session ─────────


def _usage(input_tokens: int, output_tokens: int) -> str:
    return _event(
        {
            "type": "turn.completed",
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        }
    )


class _Rounds:
    """A fake ``codex exec`` that answers each invocation with its own round:
    ``(files to write, stdout, returncode)``."""

    def __init__(self, rounds: list[tuple[dict[str, str], str, int]]) -> None:
        self.rounds = list(rounds)
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, *args: str, **kwargs: Any) -> _FakeProcess:
        self.calls.append(
            {"args": list(args), "cwd": kwargs.get("cwd"), "stdin": kwargs.get("stdin")}
        )
        files, stdout, returncode = self.rounds.pop(0)
        work_dir = Path(args[list(args).index("--cd") + 1])
        for rel, content in files.items():
            (work_dir / rel).parent.mkdir(parents=True, exist_ok=True)
            (work_dir / rel).write_text(content, encoding="utf-8")
        return _FakeProcess(stdout.encode(), b"", returncode)


@pytest.fixture
def codex_rounds(monkeypatch: pytest.MonkeyPatch):  # noqa: ANN201
    def _install(rounds: list[tuple[dict[str, str], str, int]]) -> _Rounds:
        fake = _Rounds(rounds)
        monkeypatch.setattr("rote.compiler.drivers.codex.asyncio.create_subprocess_exec", fake)
        return fake

    return _install


THREAD = _event({"type": "thread.started", "thread_id": "0199-thread"})


@pytest.mark.asyncio
async def test_a_repair_resumes_the_same_thread_until_the_repairer_accepts(
    fake_skills: tuple[Path, Path, Path],
    codex_rounds,  # noqa: ANN001
) -> None:
    """The repairer's instruction goes back into the thread that wrote the
    deliverable (``codex exec ... resume <thread id> <instruction>``), with
    the first run's sandbox, work dir and output flags, from the work dir,
    and the loop stops once the repairer returns None (#308)."""
    skill_dir, compiler_dir, work_dir = fake_skills
    fake = codex_rounds(
        [
            ({"pipeline.yaml": "first"}, f"{THREAD}\n{_usage(100, 10)}", 0),
            ({"pipeline.yaml": "second"}, f"{THREAD}\n{_usage(250, 30)}", 0),
        ]
    )
    seen: list[str] = []

    def repair(path: Path) -> str | None:
        seen.append(path.read_text())
        return "fix the gate key" if len(seen) == 1 else None

    driver = CodexDriver(model="gpt-test", web_tools=True)
    result = await driver.run(skill_dir, compiler_dir, work_dir, repair=repair)

    assert seen == ["first", "second"]
    first, second = (call["args"] for call in fake.calls)
    assert "resume" not in first
    assert second[second.index("resume") :] == ["resume", "0199-thread", "fix the gate key"]
    for args in (first, second):
        # Resuming needs the session on disk.
        assert "--ephemeral" not in args
        assert args[args.index("--sandbox") + 1] == "workspace-write"
        assert args[args.index("--cd") + 1] == str(work_dir.resolve())
        assert args[args.index("--model") + 1] == "gpt-test"
        assert args[args.index("-c") + 1] == 'web_search="live"'
    assert [call["cwd"] for call in fake.calls] == [work_dir.resolve()] * 2
    # codex exec appends a piped stdin to the prompt and waits for it to
    # close; an inherited one that never closes hung a real run.
    assert [call["stdin"] for call in fake.calls] == [asyncio.subprocess.DEVNULL] * 2
    meta = result.metadata
    assert meta["repair_rounds"] == 1
    assert "repair_warning" not in meta
    # The resumed round reports the thread's running total: the last one is
    # the session, not the sum.
    assert (meta["input_tokens"], meta["output_tokens"]) == (250, 30)
    assert meta["usage_basis"] == "cumulative"
    assert meta["usage_complete"] is True


@pytest.mark.asyncio
async def test_a_run_nothing_will_resume_stays_ephemeral(
    fake_skills: tuple[Path, Path, Path],
    codex_rounds,  # noqa: ANN001
) -> None:
    skill_dir, compiler_dir, work_dir = fake_skills
    fake = codex_rounds([({"pipeline.yaml": VALID_PIPELINE_YAML}, THREAD, 0)])
    await CodexDriver().run(skill_dir, compiler_dir, work_dir)
    assert "--ephemeral" in fake.calls[0]["args"]


@pytest.mark.asyncio
async def test_a_stream_with_no_thread_id_cannot_be_repaired_and_says_so(
    fake_skills: tuple[Path, Path, Path],
    codex_rounds,  # noqa: ANN001
) -> None:
    skill_dir, compiler_dir, work_dir = fake_skills
    fake = codex_rounds([({"pipeline.yaml": "x"}, _usage(1, 1), 0)])
    result = await CodexDriver().run(
        skill_dir, compiler_dir, work_dir, repair=lambda _path: "fix it"
    )
    assert len(fake.calls) == 1
    assert result.metadata["repair_rounds"] == 0
    assert "no thread id to resume" in result.metadata["repair_warning"]


@pytest.mark.asyncio
async def test_a_failed_repair_round_ends_the_loop_with_a_warning(
    fake_skills: tuple[Path, Path, Path],
    codex_rounds,  # noqa: ANN001
) -> None:
    skill_dir, compiler_dir, work_dir = fake_skills
    fake = codex_rounds(
        [
            ({"pipeline.yaml": "first"}, f"{THREAD}\n{_usage(100, 10)}", 0),
            ({}, "", 1),
        ]
    )
    result = await CodexDriver().run(
        skill_dir, compiler_dir, work_dir, repair=lambda _path: "fix it"
    )
    assert len(fake.calls) == 2
    meta = result.metadata
    assert meta["repair_rounds"] == 1
    assert "a repair round exited with code 1" in meta["repair_warning"]
    # The failed round reported no usage, so the figure is incomplete.
    assert meta["usage_complete"] is False
    assert meta["input_tokens"] == 100


@pytest.mark.asyncio
async def test_totals_that_go_down_are_summed_per_round(
    fake_skills: tuple[Path, Path, Path],
    codex_rounds,  # noqa: ANN001
) -> None:
    """Cumulative totals are observed, not documented: a resumed round that
    reports less than the round before is counted on its own."""
    skill_dir, compiler_dir, work_dir = fake_skills
    codex_rounds(
        [
            ({"pipeline.yaml": "first"}, f"{THREAD}\n{_usage(100, 10)}", 0),
            ({}, f"{THREAD}\n{_usage(40, 5)}", 0),
        ]
    )
    answers = iter(["fix it", None])
    result = await CodexDriver().run(
        skill_dir, compiler_dir, work_dir, repair=lambda _path: next(answers)
    )
    meta = result.metadata
    assert (meta["input_tokens"], meta["output_tokens"]) == (140, 15)
    assert meta["usage_basis"] == "summed"


def test_a_structure_pass_on_codexs_default_model_names_it() -> None:
    """A codex lane with no --model has an empty model id; the summary line
    read "structure pass by  via codex exec" on a live 0.18.0 compile."""
    from rote.compiler.system1.structure import Structure

    assert Structure(model="", lane="codex exec").model_label == "the lane's default model"
    assert Structure(model="gpt-5.5", lane="codex exec").model_label == "gpt-5.5"
