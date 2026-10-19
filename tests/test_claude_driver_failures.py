"""A failed ``claude -p`` compile still reports what it cost, and the output cap resumes.

A real invoice-push compile died after 28 minutes and 8 turns with "Claude's
response exceeded the 32000 output token maximum", and no cost was reported
anywhere: the driver raised without reading the result envelope. These tests
run the real driver against a fake ``claude`` executable (a real subprocess,
real environment, real stream-json) that emits exactly that envelope.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from rote.compiler import Compiler, CompilerError
from rote.compiler.drivers import DriverError
from rote.compiler.drivers.claude import MAX_OUTPUT_TOKENS, OUTPUT_CAP_RESUME, ClaudeDriver

CAP_MESSAGE = (
    "API Error: Claude's response exceeded the 32000 output token maximum. To configure "
    "this behavior, set the CLAUDE_CODE_MAX_OUTPUT_TOKENS environment variable."
)

#: The fake CLI. Each invocation appends {argv, prompt, env knob} to a log,
#: then plays the next scripted step: an error envelope (exit 1) or a
#: success that writes pipeline.yaml into its cwd (the work dir). Cost is
#: cumulative across a resumed session, as the real CLI reports it.
FAKE_CLI = """\
import json, os, sys
log = os.environ["FAKE_CLAUDE_LOG"]
steps = json.loads(os.environ["FAKE_CLAUDE_STEPS"])
n = sum(1 for _ in open(log))
argv = sys.argv[1:]
with open(log, "a") as f:
    f.write(json.dumps({
        "argv": argv,
        "prompt": argv[argv.index("-p") + 1],
        "max_output": os.environ.get("CLAUDE_CODE_MAX_OUTPUT_TOKENS"),
    }) + "\\n")
step = steps[n]
usage = {"claude-sonnet-4-6": {"inputTokens": 12, "outputTokens": step["out"],
         "cacheCreationInputTokens": 90000, "cacheReadInputTokens": 400000}}
envelope = {"type": "result", "subtype": "success", "session_id": "sess-cap",
            "total_cost_usd": step["cost"], "num_turns": step["turns"],
            "duration_ms": 1000, "modelUsage": usage}
if step["kind"] == "ok":
    open("pipeline.yaml", "w").write("name: fake\\n")
    envelope.update(is_error=False, result="done")
    print(json.dumps(envelope), flush=True)
    sys.exit(0)
envelope.update(is_error=True, result=step["message"])
print(json.dumps(envelope), flush=True)
sys.exit(1)
"""


def _fake_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: list[dict[str, Any]]) -> Path:
    exe = tmp_path / "bin" / "claude"
    exe.parent.mkdir()
    exe.write_text(f"#!{sys.executable}\n{FAKE_CLI}", encoding="utf-8")
    exe.chmod(0o755)
    log = tmp_path / "calls.ndjson"
    log.write_text("")
    monkeypatch.setenv("FAKE_CLAUDE_LOG", str(log))
    monkeypatch.setenv("FAKE_CLAUDE_STEPS", json.dumps(steps))
    return exe


def _calls(tmp_path: Path) -> list[dict[str, Any]]:
    lines = (tmp_path / "calls.ndjson").read_text().splitlines()
    return [json.loads(line) for line in lines]


def _cap(cost: float, turns: int = 8) -> dict[str, Any]:
    return {"kind": "error", "message": CAP_MESSAGE, "cost": cost, "turns": turns, "out": 32000}


def _ok(cost: float, turns: int = 3) -> dict[str, Any]:
    return {"kind": "ok", "cost": cost, "turns": turns, "out": 4000}


@pytest.fixture
def dirs(tmp_path: Path) -> tuple[Path, Path, Path]:
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: fake\n---\n\n# Fake\n", encoding="utf-8")
    compiler = tmp_path / "compiler"
    (compiler / "references").mkdir(parents=True)
    (compiler / "SKILL.md").write_text("# rubric\n", encoding="utf-8")
    return skill, compiler, tmp_path / "work"


@pytest.mark.asyncio
async def test_the_output_cap_resumes_the_session_once_and_the_compile_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dirs: tuple[Path, Path, Path]
) -> None:
    exe = _fake_cli(tmp_path, monkeypatch, [_cap(3.10), _ok(3.55)])
    events: list[Any] = []
    result = await ClaudeDriver(claude_executable=str(exe)).run(*dirs, on_event=events.append)
    first, resumed = _calls(tmp_path)
    assert "--resume" not in first["argv"]
    assert resumed["argv"][resumed["argv"].index("--resume") + 1] == "sess-cap"
    assert resumed["prompt"] == OUTPUT_CAP_RESUME
    assert "in pieces" in OUTPUT_CAP_RESUME
    assert result.pipeline_yaml_path.read_text() == "name: fake\n"
    # The whole session's cost, read from the resume's cumulative envelope.
    assert result.metadata["cost_usd"] == 3.55
    assert result.metadata["num_turns"] == 11
    assert any("output-token cap" in e.message for e in events if e.type == "warning")


@pytest.mark.asyncio
async def test_a_repair_round_that_hits_the_cap_is_resumed_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dirs: tuple[Path, Path, Path]
) -> None:
    exe = _fake_cli(tmp_path, monkeypatch, [_ok(2.0), _cap(2.6, turns=1), _ok(2.9)])
    asked: list[str] = []

    def repair(_path: Path) -> str | None:
        asked.append("x")
        return "FIX: a test fails" if len(asked) == 1 else None

    result = await ClaudeDriver(claude_executable=str(exe)).run(*dirs, repair=repair)
    prompts = [c["prompt"] for c in _calls(tmp_path)]
    assert prompts[1:] == ["FIX: a test fails", OUTPUT_CAP_RESUME]
    assert "repair_warning" not in result.metadata
    assert result.metadata["cost_usd"] == 2.9


@pytest.mark.asyncio
async def test_the_one_cap_resume_is_per_compile_not_per_round(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dirs: tuple[Path, Path, Path]
) -> None:
    """A second cap, in a repair round, is not resumed: the deliverable is
    what the agent had written, reported as a repair_warning."""
    exe = _fake_cli(tmp_path, monkeypatch, [_cap(1.0), _ok(1.5), _cap(2.0), _ok(9.0)])
    asked: list[str] = []

    def repair(_path: Path) -> str | None:
        asked.append("x")
        return "FIX" if len(asked) == 1 else None

    result = await ClaudeDriver(claude_executable=str(exe)).run(*dirs, repair=repair)
    assert len(_calls(tmp_path)) == 3
    assert "exited with code 1" in result.metadata["repair_warning"]


@pytest.mark.asyncio
async def test_the_cap_is_resumed_once_not_forever_and_the_failure_carries_its_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dirs: tuple[Path, Path, Path]
) -> None:
    exe = _fake_cli(tmp_path, monkeypatch, [_cap(3.10), _cap(4.20, turns=2), _ok(9.99)])
    with pytest.raises(DriverError) as exc:
        await ClaudeDriver(claude_executable=str(exe)).run(*dirs)
    assert len(_calls(tmp_path)) == 2
    assert exc.value.metadata is not None
    assert exc.value.metadata["cost_usd"] == 4.20
    assert exc.value.metadata["num_turns"] == 10
    assert "output token maximum" in (exc.value.details or "")


@pytest.mark.asyncio
async def test_any_other_failure_is_not_resumed_but_still_reports_its_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dirs: tuple[Path, Path, Path]
) -> None:
    other = {"kind": "error", "message": "API Error: 529 overloaded", "cost": 0.42}
    exe = _fake_cli(tmp_path, monkeypatch, [{**other, "turns": 4, "out": 900}, _ok(1.0)])
    with pytest.raises(DriverError) as exc:
        await ClaudeDriver(claude_executable=str(exe)).run(*dirs)
    assert len(_calls(tmp_path)) == 1
    assert exc.value.metadata is not None
    assert exc.value.metadata["cost_usd"] == 0.42
    assert exc.value.metadata["cache_read_tokens"] == 400000


@pytest.mark.asyncio
async def test_the_child_gets_a_raised_output_cap_unless_the_caller_set_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dirs: tuple[Path, Path, Path]
) -> None:
    monkeypatch.delenv("CLAUDE_CODE_MAX_OUTPUT_TOKENS", raising=False)
    exe = _fake_cli(tmp_path, monkeypatch, [_ok(1.0), _ok(1.0)])
    await ClaudeDriver(claude_executable=str(exe)).run(*dirs)
    assert MAX_OUTPUT_TOKENS == "64000"
    assert _calls(tmp_path)[0]["max_output"] == "64000"

    (dirs[2] / "pipeline.yaml").unlink()
    monkeypatch.setenv("CLAUDE_CODE_MAX_OUTPUT_TOKENS", "100000")
    await ClaudeDriver(claude_executable=str(exe)).run(*dirs)
    assert _calls(tmp_path)[1]["max_output"] == "100000"


@pytest.mark.asyncio
async def test_a_failed_compile_says_what_it_cost_in_the_error_and_the_feed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dirs: tuple[Path, Path, Path]
) -> None:
    exe = _fake_cli(tmp_path, monkeypatch, [_cap(3.10), _cap(4.20, turns=2)])
    skill, compiler_dir, _ = dirs
    events: list[Any] = []
    compiler = Compiler(compiler_skill_dir=compiler_dir, on_event=events.append)
    compiler.select_driver = lambda: ClaudeDriver(claude_executable=str(exe))  # type: ignore[method-assign]
    with pytest.raises(CompilerError) as exc:
        await compiler.compile(skill, tmp_path / "out")
    assert "cost=$4.2" in str(exc.value)
    assert "turns=10" in str(exc.value)
    logs = [e.message for e in events if e.type == "log"]
    assert any(m.startswith("usage before the failure:") and "cost=$4.2" in m for m in logs)
    errors = [e.message for e in events if e.type == "error"]
    assert errors and "cost=$4.2" in errors[-1]


def test_rote_compile_writes_the_failed_runs_cost_to_progress_jsonl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dirs: tuple[Path, Path, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    from rote.cli import main

    exe = _fake_cli(tmp_path, monkeypatch, [_cap(3.10), _cap(4.20, turns=2)])
    monkeypatch.setattr(
        Compiler, "select_driver", lambda self: ClaudeDriver(claude_executable=str(exe))
    )
    out = tmp_path / "run"
    rc = main(
        ["compile", str(dirs[0]), "--local", "--runtime", "dbos", "--no-eval"] + ["--out", str(out)]
    )
    assert rc == 1
    assert "cost=$4.2" in capsys.readouterr().err
    progress = [json.loads(line) for line in (out / "progress.jsonl").read_text().splitlines()]
    assert any("cost=$4.2" in (p.get("message") or "") for p in progress)
