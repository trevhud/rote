"""The runtime a ``rote compile`` with no ``--runtime`` emits.

The compiled skill (``--runtime skill``) is the built-in default, logged in
or not (docs/design/compiled-skill-target.md §9, PR 10). Every layer above
the built-in keeps working as before: flag > ``ROTE_RUNTIME`` > project
``rote.yaml`` > user config > built-in. ``deploy: rote-cloud`` is the one
setting that changes the built-in, to cloudflare, because rote cloud hosts
only that runtime.

The resolution tests stop at the Compiler (it records the runtime the
compile would emit and raises), so no agent runs. The end-to-end tests run
the real emit on docs-digest's committed compile, whose modules are
implemented, so the skill adapter accepts it.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from rote.cli import DEFAULT_COMPILE_RUNTIME, builtin_compile_runtime
from rote.cli import main as cli_main
from rote.cloud_auth import CloudCredential, save_credential

DOCS_DIGEST = Path(__file__).resolve().parent.parent / "examples" / "docs-digest"


class _Reached(Exception):
    def __init__(self, runtime: str) -> None:
        super().__init__(runtime)
        self.runtime = runtime


@pytest.fixture
def stop_at_compiler(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Cloud dispatch records "cloud"; the local Compiler raises the runtime."""
    cloud: list[str] = []

    def to_cloud(_args: Any) -> int:
        cloud.append("cloud")
        return 77

    class _Compiler:
        def __init__(self, **kw: Any) -> None:
            raise _Reached(kw["emit_target"].runtime)

    monkeypatch.setattr("rote.cli._cmd_compile_cloud", to_cloud)
    monkeypatch.setattr("rote.cli.Compiler", _Compiler)
    return cloud


@pytest.fixture
def layers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    user = tmp_path / "user" / "config.yaml"
    project = tmp_path / "proj" / "rote.yaml"
    user.parent.mkdir()
    project.parent.mkdir()
    monkeypatch.setenv("ROTE_CONFIG_PATH", str(user))
    monkeypatch.setenv("ROTE_PROJECT_CONFIG_PATH", str(project))
    monkeypatch.delenv("ROTE_RUNTIME", raising=False)
    monkeypatch.delenv("ROTE_DEPLOY", raising=False)
    return {"user": user, "project": project}


def _skill(tmp_path: Path) -> Path:
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        "---\nname: word-lookup\ndescription: Look a word up.\n---\n# Look a word up\n",
        encoding="utf-8",
    )
    return skill


def _login() -> None:
    save_credential(CloudCredential(url="http://cloud", token="rote_t", user="t@x"))


def _resolved(tmp_path: Path, *flags: str) -> str:
    with pytest.raises(_Reached) as reached:
        cli_main(["compile", str(_skill(tmp_path)), "--out", str(tmp_path / "o"), *flags])
    return reached.value.runtime


# ───────── the built-in layer ─────────


def test_the_built_in_is_the_compiled_skill() -> None:
    assert DEFAULT_COMPILE_RUNTIME == "skill"
    assert builtin_compile_runtime(None) == "skill"
    assert builtin_compile_runtime("none") == "skill"
    # rote cloud hosts only cloudflare, so asking for it changes the built-in.
    assert builtin_compile_runtime("rote-cloud") == "cloudflare"


@pytest.mark.parametrize("logged_in", [False, True])
def test_no_flag_no_config_compiles_a_skill_locally(
    tmp_path: Path,
    layers: dict[str, Path],
    stop_at_compiler: list[str],
    capsys: pytest.CaptureFixture[str],
    logged_in: bool,
) -> None:
    """Logged out this was dbos, logged in it went to rote cloud (cloudflare).
    Now both are a local skill compile, and stderr says why once."""
    if logged_in:
        _login()
    assert _resolved(tmp_path) == "skill"
    assert stop_at_compiler == []  # never sent to rote cloud
    err = capsys.readouterr().err
    assert "rote compile: runtime skill (the default:" in err
    # The cloud pointer is for someone who can use it.
    assert ("--cloud compiles and hosts it on rote cloud" in err) is logged_in
    # The old tips described defaults that no longer exist.
    assert "makes rote cloud the default host" not in err


def test_local_flag_logged_in_is_a_skill_too(
    tmp_path: Path, layers: dict[str, Path], stop_at_compiler: list[str]
) -> None:
    _login()
    assert _resolved(tmp_path, "--local") == "skill"
    assert stop_at_compiler == []


# ───────── every layer above the built-in still wins ─────────


@pytest.mark.parametrize("logged_in", [False, True])
def test_user_config_beats_the_built_in(
    tmp_path: Path, layers: dict[str, Path], stop_at_compiler: list[str], logged_in: bool
) -> None:
    if logged_in:
        _login()
    layers["user"].write_text("runtime: dbos\n", encoding="utf-8")
    assert _resolved(tmp_path) == "dbos"


@pytest.mark.parametrize("logged_in", [False, True])
def test_project_beats_user(
    tmp_path: Path, layers: dict[str, Path], stop_at_compiler: list[str], logged_in: bool
) -> None:
    if logged_in:
        _login()
    layers["user"].write_text("runtime: dbos\n", encoding="utf-8")
    layers["project"].write_text("runtime: temporal\n", encoding="utf-8")
    assert _resolved(tmp_path) == "temporal"


@pytest.mark.parametrize("logged_in", [False, True])
def test_env_beats_project(
    tmp_path: Path,
    layers: dict[str, Path],
    stop_at_compiler: list[str],
    monkeypatch: pytest.MonkeyPatch,
    logged_in: bool,
) -> None:
    if logged_in:
        _login()
    layers["project"].write_text("runtime: temporal\n", encoding="utf-8")
    monkeypatch.setenv("ROTE_RUNTIME", "python")
    assert _resolved(tmp_path) == "python"


@pytest.mark.parametrize("logged_in", [False, True])
def test_flag_beats_env(
    tmp_path: Path,
    layers: dict[str, Path],
    stop_at_compiler: list[str],
    monkeypatch: pytest.MonkeyPatch,
    logged_in: bool,
) -> None:
    if logged_in:
        _login()
    monkeypatch.setenv("ROTE_RUNTIME", "python")
    assert _resolved(tmp_path, "--runtime", "dbos") == "dbos"


def test_a_configured_runtime_prints_no_default_line(
    tmp_path: Path,
    layers: dict[str, Path],
    stop_at_compiler: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    layers["user"].write_text("runtime: skill\n", encoding="utf-8")
    assert _resolved(tmp_path) == "skill"
    assert "(the default:" not in capsys.readouterr().err


# ───────── deploy: rote-cloud, the one setting that moves the built-in ─────────


def test_deploy_rote_cloud_logged_in_runs_on_rote_cloud(
    tmp_path: Path, layers: dict[str, Path], stop_at_compiler: list[str]
) -> None:
    _login()
    layers["user"].write_text("deploy: rote-cloud\n", encoding="utf-8")
    rc = cli_main(["compile", str(_skill(tmp_path)), "--out", str(tmp_path / "o")])
    assert (rc, stop_at_compiler) == (77, ["cloud"])


def test_deploy_rote_cloud_local_emits_cloudflare(
    tmp_path: Path, layers: dict[str, Path], stop_at_compiler: list[str]
) -> None:
    _login()
    layers["user"].write_text("deploy: rote-cloud\n", encoding="utf-8")
    assert _resolved(tmp_path, "--local") == "cloudflare"


def test_deploy_rote_cloud_logged_out_stops_before_the_agent(
    tmp_path: Path,
    layers: dict[str, Path],
    stop_at_compiler: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    layers["user"].write_text("deploy: rote-cloud\n", encoding="utf-8")
    rc = cli_main(["compile", str(_skill(tmp_path)), "--out", str(tmp_path / "o")])
    assert rc == 2
    assert "you are not logged in" in capsys.readouterr().err
    assert stop_at_compiler == []


@pytest.mark.parametrize("logged_in", [True, False])
def test_no_deploy_keeps_the_cloudflare_built_in_of_a_configured_rote_cloud_deploy(
    tmp_path: Path,
    layers: dict[str, Path],
    stop_at_compiler: list[str],
    capsys: pytest.CaptureFixture[str],
    logged_in: bool,
) -> None:
    """--no-deploy skips the upload, never the runtime the config implies:
    `deploy: rote-cloud` + --no-deploy is a local cloudflare compile, as on
    main (logged in). Resolving the built-in after --no-deploy forced deploy
    to none made it a skill, and the not-logged-in hint then named a
    runtime nobody asked for."""
    if logged_in:
        _login()
    layers["user"].write_text("deploy: rote-cloud\n", encoding="utf-8")
    assert _resolved(tmp_path, "--no-deploy") == "cloudflare"
    assert stop_at_compiler == []
    assert "(the default:" not in capsys.readouterr().err


# ───────── a default the skill cannot meet says how to pick another ─────────


def test_a_skill_without_a_description_names_the_way_out(
    tmp_path: Path,
    layers: dict[str, Path],
    stop_at_compiler: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    skill = tmp_path / "bare"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: bare\n---\n# Bare\n", encoding="utf-8")
    rc = cli_main(["compile", str(skill), "--out", str(tmp_path / "o")])
    assert rc == 2
    err = capsys.readouterr().err
    assert "has no `description:`" in err
    assert "skill is the default runtime; --runtime dbos" in err
    # An explicit --runtime skill is a choice, so the hint is not repeated.
    rc = cli_main(["compile", str(skill), "--out", str(tmp_path / "o"), "--runtime", "skill"])
    assert rc == 2
    assert "skill is the default runtime" not in capsys.readouterr().err


# ───────── end to end: the emitted skill and the next step ─────────


def _docs_digest_compiler(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Compiler that returns docs-digest's committed compile (implemented
    modules, so the skill adapter emits it)."""
    from rote.compiler import CompilationResult
    from rote.ir import load_pipeline

    class _Compiler:
        def __init__(self, **_kw: Any) -> None:
            pass

        async def compile(self, skill_path, output_dir, update=False, **_kw):  # noqa: ANN001
            output_dir = Path(output_dir)
            shutil.copytree(DOCS_DIGEST / "compiled", output_dir, dirs_exist_ok=True)
            return CompilationResult(
                pipeline=load_pipeline(output_dir / "pipeline.yaml"),
                output_dir=output_dir,
                driver_name="mock",
                driver_metadata={},
            )

    monkeypatch.setattr("rote.cli.Compiler", _Compiler)


@pytest.mark.parametrize("logged_in", [False, True])
def test_a_flagless_compile_emits_the_skill_and_prints_the_next_step(
    tmp_path: Path,
    layers: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    logged_in: bool,
) -> None:
    if logged_in:
        _login()
    monkeypatch.setenv("ROTE_CLIENT_HOME_DIR", str(tmp_path / "home"))  # a free skills slot
    deploys: list[Any] = []
    monkeypatch.setattr(
        "rote.deploy_rote_cloud.deploy_rote_cloud", lambda *a, **k: deploys.append(a)
    )
    monkeypatch.setattr("rote.cli._cmd_compile_cloud", lambda _a: pytest.fail("went to cloud"))
    _docs_digest_compiler(monkeypatch)
    skill = tmp_path / "docs-digest"
    shutil.copytree(DOCS_DIGEST / "skill", skill)
    out = tmp_path / "out"

    rc = cli_main(["compile", str(skill), "--out", str(out), "--no-eval"])

    captured = capsys.readouterr()
    assert rc == 0, captured.err
    compiled_skill = out / "runtime" / "skill" / "docs-digest"
    assert (compiled_skill / "SKILL.md").is_file()
    assert (compiled_skill / "scripts" / "run.py").is_file()
    assert not (out / "runtime" / "dbos").exists()
    assert not (out / "runtime" / "cloudflare").exists()
    assert deploys == []
    lines = captured.out.splitlines()
    nxt = next(i for i, line in enumerate(lines) if line.strip().startswith("next:"))
    assert lines[nxt].strip() == f"next: rote skill install {compiled_skill.resolve()}"
    assert lines[nxt + 1].strip() == (
        "then swap it in for the original docs-digest skill (Claude Code: "
        f"ln -s {compiled_skill.resolve()} ~/.claude/skills/docs-digest)"
    )
    assert lines[nxt + 2].strip() == (
        f"or see it give the original's answer first: rote try {skill}"
    )
    # The workflow hint is not printed for a skill.
    assert "rote run" not in captured.out


def test_a_workflow_compile_keeps_its_rote_run_next_step(
    tmp_path: Path,
    layers: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _docs_digest_compiler(monkeypatch)
    skill = tmp_path / "docs-digest"
    shutil.copytree(DOCS_DIGEST / "skill", skill)
    out = tmp_path / "out"
    rc = cli_main(["compile", str(skill), "--out", str(out), "--no-eval", "--runtime", "python"])
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    # python stubs the MCP-bound steps, so the next step fills them in first.
    assert f"next: fill those in, then rote run {out / 'runtime' / 'python'}" in captured.out
    assert "rote skill install" not in captured.out
    assert "rote try" not in captured.out


# ───────── the swap command checks ~/.claude/skills/<name> first ─────────


def _next_lines(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, list[str]]:
    from rote.skill_target.next_steps import compiled_skill_next_steps

    home = tmp_path / "home"
    (home / ".claude" / "skills").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("ROTE_CLIENT_HOME_DIR", str(home))
    compiled = tmp_path / "out" / "runtime" / "skill" / "word-lookup"
    compiled.mkdir(parents=True, exist_ok=True)
    steps = compiled_skill_next_steps(compiled, Path("my-skill"))
    assert steps.install == f"rote skill install {compiled.resolve()}"
    assert steps.proof == "or see it give the original's answer first: rote try my-skill"
    return compiled, steps.swap


def test_a_free_skills_slot_gets_the_plain_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    compiled, lines = _next_lines(tmp_path, monkeypatch)
    assert lines == [
        "swap it in for the original word-lookup skill (Claude Code: "
        f"ln -s {compiled.resolve()} ~/.claude/skills/word-lookup)"
    ]


@pytest.mark.parametrize("kind", ["dir", "foreign_link"])
def test_an_occupied_slot_moves_the_original_aside_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """BSD ln -s into an existing dir exits 0 and nests the link inside the
    original, so the compiled skill never loads: the plain command must not
    be offered alone."""
    slot = tmp_path / "home" / ".claude" / "skills" / "word-lookup"
    slot.parent.mkdir(parents=True)
    if kind == "dir":
        slot.mkdir()
        (slot / "SKILL.md").write_text("original\n", encoding="utf-8")
    else:
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        slot.symlink_to(elsewhere, target_is_directory=True)
    compiled, lines = _next_lines(tmp_path, monkeypatch)
    target = compiled.resolve()
    assert lines == [
        "for Claude Code: your original word-lookup skill is at "
        "~/.claude/skills/word-lookup. To use the compiled one instead, move the original "
        "out of the skills folder (nothing is deleted): mv ~/.claude/skills/word-lookup "
        f"~/word-lookup.original && ln -s {target} ~/.claude/skills/word-lookup",
        "  undo: rm ~/.claude/skills/word-lookup && mv ~/word-lookup.original "
        "~/.claude/skills/word-lookup",
    ]
    assert not any(line.startswith("swap it in") for line in lines)
    # Printing never touched the original.
    assert slot.exists() or slot.is_symlink()


def test_a_slot_already_linked_to_this_skill_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    compiled = tmp_path / "out" / "runtime" / "skill" / "word-lookup"
    compiled.mkdir(parents=True)
    slot = tmp_path / "home" / ".claude" / "skills" / "word-lookup"
    slot.parent.mkdir(parents=True)
    slot.symlink_to(compiled, target_is_directory=True)
    _, lines = _next_lines(tmp_path, monkeypatch)
    assert lines == ["your agent has it: ~/.claude/skills/word-lookup already links to it"]


def test_a_moved_claude_config_dir_is_where_the_slot_is_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLAUDE_CONFIG_DIR moves ~/.claude, skills included; the slot checked
    and printed is the moved one, as Claude Code loads it."""
    from rote.skill_target.next_steps import compiled_skill_next_steps

    config = tmp_path / "claude config"
    (config / "skills" / "word-lookup").mkdir(parents=True)
    monkeypatch.delenv("ROTE_CLIENT_HOME_DIR", raising=False)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config))
    compiled = tmp_path / "out" / "runtime" / "skill" / "word-lookup"
    compiled.mkdir(parents=True)
    lines = compiled_skill_next_steps(compiled, Path("my-skill")).swap
    quoted = f"'{config / 'skills' / 'word-lookup'}'"
    assert lines[0].startswith(f"for Claude Code: your original word-lookup skill is at {quoted}.")
    assert f"mv {quoted} ~/word-lookup.original && ln -s " in lines[0]


def test_the_move_aside_name_is_one_nothing_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mv onto an existing ~/<name>.original nests the original inside it, so
    the printed name skips every taken one (a file or link counts too)."""
    home = tmp_path / "home"
    slot = home / ".claude" / "skills" / "word-lookup"
    slot.mkdir(parents=True)
    (home / "word-lookup.original").mkdir()
    (home / "word-lookup.original-2").write_text("taken\n", encoding="utf-8")
    compiled, lines = _next_lines(tmp_path, monkeypatch)
    assert (
        f"mv ~/.claude/skills/word-lookup ~/word-lookup.original-3 && ln -s {compiled.resolve()} "
        in lines[0]
    )
    assert lines[1] == (
        "  undo: rm ~/.claude/skills/word-lookup && mv ~/word-lookup.original-3 "
        "~/.claude/skills/word-lookup"
    )
    assert "word-lookup.original " not in lines[0]
