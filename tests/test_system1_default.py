"""System-1 is the default compile path; ``--no-system1`` and ``--update`` opt out.

What is pinned: a flagless compile and a flagless analyze run System-1, the
opt-out runs the full rubric, ``--update`` runs the full rubric and says so,
a model decider never runs silently as rules, and cloud mode rejects the
local-only System-1 flags.
"""

from __future__ import annotations

import argparse
import contextlib
import shutil
from pathlib import Path
from typing import Any

import pytest

from rote import cli
from rote.compiler import Compiler, CompilerError
from rote.compiler.drivers import CompilerDriver, DriverResult
from rote.ir import load_pipeline

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "examples"


class _RubricSpy(CompilerDriver):
    """Records which rubric the agent got, then stops the compile."""

    name = "rubric-spy"

    def __init__(self) -> None:
        self.rubric: Path | None = None

    def is_available(self) -> tuple[bool, str]:
        return (True, "")

    async def run(
        self,
        skill_dir: Path,
        compiler_skill_dir: Path,
        work_dir: Path,
        extra_instructions: str | None = None,
        on_event: object = None,
    ) -> DriverResult:
        self.rubric = compiler_skill_dir
        raise _Stop


class _Stop(Exception):
    pass


def _skill(tmp_path: Path) -> Path:
    skill = tmp_path / "skill"
    shutil.copytree(EXAMPLES / "expense-audit" / "skill", skill)
    return skill


async def _rubric_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **kwargs: Any
) -> tuple[Path, list[str]]:
    spy = _RubricSpy()
    monkeypatch.setattr(Compiler, "select_driver", lambda self: spy)
    logs: list[str] = []
    compiler = Compiler(on_event=lambda e: logs.append(e.message), **kwargs)
    with pytest.raises(_Stop):
        await compiler.compile(_skill(tmp_path), tmp_path / "out")
    assert spy.rubric is not None
    return spy.rubric, logs


@pytest.mark.asyncio
async def test_a_flagless_compiler_runs_system1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rubric, _ = await _rubric_for(tmp_path, monkeypatch)
    assert rubric.name.startswith("rote-system1-rubric-")


@pytest.mark.asyncio
async def test_no_system1_runs_the_full_rubric(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rubric, _ = await _rubric_for(tmp_path, monkeypatch, system1=False)
    assert not rubric.name.startswith("rote-system1-rubric-")
    assert (rubric / "SKILL.md").read_text().startswith("---\nname: rote-compile\n")


@pytest.mark.asyncio
async def test_update_runs_the_full_rubric_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--update on the default path is the full-rubric agent, announced."""
    seen: dict[str, Any] = {}

    def plan_update(self: Compiler, output_dir: Path, text: str) -> Any:
        seen["planned"] = True
        raise CompilerError("stop after the mode decision")

    monkeypatch.setattr(Compiler, "_plan_update", plan_update)
    logs: list[str] = []
    compiler = Compiler(on_event=lambda e: logs.append(e.message))
    with pytest.raises(CompilerError, match="stop after the mode decision"):
        await compiler.compile(_skill(tmp_path), tmp_path / "out", update=True)
    assert seen == {"planned": True}
    assert any("--update runs the full-rubric agent" in m for m in logs)


@pytest.mark.asyncio
async def test_a_model_decider_never_runs_as_rules(tmp_path: Path) -> None:
    """--update cannot run System-1, so a model decider there is an error, not ignored."""
    skill = _skill(tmp_path)
    with pytest.raises(CompilerError, match="does not run System-1"):
        await Compiler(system1_decider="clef").compile(skill, tmp_path / "out", update=True)
    with pytest.raises(CompilerError, match="does not run System-1"):
        await Compiler(system1=False, system1_decider="clef").compile(skill, tmp_path / "out")


def _parsed(argv: list[str]) -> argparse.Namespace:
    out = ["--out", "o"] if argv[0] == "compile" else []
    return cli._build_parser().parse_args([*argv, *out])


@pytest.mark.parametrize("command", ["compile", "analyze"])
def test_the_flag_is_tri_state(command: str) -> None:
    assert _parsed([command, "s"]).system1 is None
    assert _parsed([command, "s", "--system1"]).system1 is True
    assert _parsed([command, "s", "--no-system1"]).system1 is False


@pytest.mark.parametrize("command", ["compile", "analyze"])
def test_the_cli_passes_the_flag_to_the_compiler(
    command: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Any] = []
    real_init = Compiler.__init__

    def init(self: Compiler, *a: Any, **kw: Any) -> None:
        seen.append(kw.get("system1", "absent"))
        real_init(self, *a, **kw)
        raise _Stop

    monkeypatch.setattr(Compiler, "__init__", init)
    skill = _skill(tmp_path)
    for flags, expected in (([], None), (["--no-system1"], False), (["--system1"], True)):
        argv = [command, str(skill), *flags]
        if command == "compile":
            argv += ["--local", "--runtime", "dbos", "--no-eval", "--out", str(tmp_path / "o")]
        with pytest.raises(_Stop):
            cli.main(argv)
        assert seen[-1] == expected


def test_a_decider_with_no_system1_is_a_usage_error(tmp_path: Path) -> None:
    for command in ("compile", "analyze"):
        argv = [command, str(_skill(tmp_path / command)), "--no-system1"]
        if command == "compile":
            argv += ["--out", str(tmp_path / "o")]
        assert cli.main([*argv, "--system1-decider", "clef"]) == 2


def test_cloud_mode_rejects_the_local_system1_flags() -> None:
    assert cli._reject_local_only_cloud_flags(_parsed(["compile", "s"])) is None
    for flags, named in (
        (["--no-system1"], "--system1/--no-system1"),
        (["--system1"], "--system1/--no-system1"),
        (["--system1-decider", "clef"], "--system1-decider"),
    ):
        message = cli._reject_local_only_cloud_flags(_parsed(["compile", "s", *flags]))
        assert message is not None and message.startswith(f"rote compile: {named} runs locally")


def test_system1_requires_a_signature_spec_on_every_judge(tmp_path: Path) -> None:
    """The legacy `signature` path emits only on Python runtimes; System-1's
    deliverable must emit everywhere, so a judge without a spec is an error."""
    from rote.compiler import system1
    from rote.compiler.verify import verify_deliverable

    work = tmp_path / "work"
    work.mkdir()
    system1.prepare(EXAMPLES / "deal-monitor" / "skill", work)
    judges = [n for n in load_pipeline(work / "pipeline.yaml").nodes if n.kind.value == "llm_judge"]
    assert judges and all(j.signature_spec is None for j in judges)

    def missing(portable: bool) -> set[str]:
        v = verify_deliverable(work, run_tests=False, portable_judges=portable)
        return {f.node_id for f in v.findings if f.kind == "missing_signature_spec"}

    assert missing(True) == {j.id for j in judges}
    assert missing(False) == set()


@pytest.mark.asyncio
@pytest.mark.parametrize(("system1", "flagged"), [(None, True), (False, False)])
async def test_the_compiler_checks_judge_specs_on_the_system1_path_only(
    system1: bool | None, flagged: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[bool] = []
    import rote.compiler as compiler_mod

    real = compiler_mod.verify_deliverable

    def spy(*a: Any, **kw: Any) -> Any:
        seen.append(kw.get("portable_judges", False))
        return real(*a, **kw)

    monkeypatch.setattr(compiler_mod, "verify_deliverable", spy)

    class _Writes(_RubricSpy):
        async def run(
            self, skill_dir: Path, compiler_skill_dir: Path, work_dir: Path, **kw: Any
        ) -> DriverResult:  # noqa: E501
            if not (work_dir / "pipeline.yaml").is_file():
                shutil.copy(EXAMPLES / "expense-audit" / "expected" / "pipeline.yaml", work_dir)
            return DriverResult(work_dir / "pipeline.yaml", work_dir, self.name, {})

    monkeypatch.setattr(Compiler, "select_driver", lambda self: _Writes())
    with contextlib.suppress(CompilerError):
        await Compiler(system1=system1, run_generated_tests=False).compile(
            _skill(tmp_path), tmp_path / "out"
        )
    assert seen and set(seen) == {flagged}
