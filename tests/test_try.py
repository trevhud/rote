"""``rote try``: the consent points, resume, and the honest-result rules. No inference.

The compile is a fake that lays down docs-digest's committed compile (the
real compile is :mod:`rote.eval.skill_ab`'s, tested there). Everything after
it is real: the install is ``rote skill install``'s plan, and the paired eval
is :func:`rote.eval.skill_ab.run_subject` with ``tests/fixtures/skill_ab/
fake_claude.py`` as ``claude``, which runs the real ``rote skill run`` against
the fixture MCP server (``tests/_skill_mcp_server.py``). A person at a
terminal is a :class:`ScriptedPrompter`; every other mode is the real
non-interactive path.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from rote import demo
from rote.cli import main
from rote.eval import skill_ab
from rote.interact.prompter import DEFAULT, Answer, ScriptedPrompter
from rote.mcp.registry import McpRegistry, McpServerConfig, save_registry
from rote.setup import try_skill
from tests._skill_driver import FixtureServer, fixture_server

pytest.importorskip("fastmcp")

REPO = Path(__file__).resolve().parent.parent
FAKE_CLAUDE = REPO / "tests" / "fixtures" / "skill_ab" / "fake_claude.py"
DOCS_COMPILED = REPO / "examples" / "docs-digest" / "compiled"
COMPILE_COST = 0.05


@pytest.fixture(scope="module")
def fixtures(tmp_path_factory: pytest.TempPathFactory) -> Iterator[FixtureServer]:
    with fixture_server(tmp_path_factory.mktemp("try-mcp") / "hits.jsonl") as server:
        yield server


class World:
    def __init__(self, tmp_path: Path) -> None:
        self.tmp = tmp_path
        self.compiles: list[list[str]] = []
        self.evals: list[skill_ab.Harness] = []
        self.out = tmp_path / "rote-try-docs-digest"

    def compile(
        self,
        skill_dir: Path,
        payload: dict[str, Any],
        work: Path,
        ledger: skill_ab.Ledger,
        *,
        model: str | None,
        log: Any,
        estimate_usd: float,
        caller: str,
    ) -> tuple[Path, dict[str, Any]]:
        """docs-digest's committed compile, emitted for ``--runtime skill``,
        recorded in the ledger as a paid run."""
        from rote.adapters.skill import SkillAdapter, SkillAdapterConfig
        from rote.ir import load_pipeline

        ledger.check("compile", estimate_usd)
        argv = skill_ab.compile_argv(skill_dir, work, model=model)
        self.compiles.append(argv)
        compiled = work / "compile" / "compiled"
        if compiled.exists():
            shutil.rmtree(compiled)
        shutil.copytree(DOCS_COMPILED, compiled)
        pipeline_yaml = compiled / "pipeline.yaml"
        SkillAdapter(SkillAdapterConfig(skill_dir=skill_dir, extracted_source_dir=compiled)).emit(
            load_pipeline(pipeline_yaml), work / "compile" / "runtime" / "skill" / "docs-digest"
        )
        ledger.record("compile", "compile", COMPILE_COST, estimate_usd)
        return pipeline_yaml, {"argv": ["rote", *argv], "cost_usd": COMPILE_COST}

    def run_subject(
        self, harness: skill_ab.Harness, subject: skill_ab.Subject, trials: int
    ) -> skill_ab.SubjectReport:
        self.evals.append(harness)
        return skill_ab.run_subject(harness, subject, trials)

    def ledger(self) -> list[dict[str, Any]]:
        path = self.out / try_skill.LEDGER_FILE
        return list(json.loads(path.read_text())["runs"]) if path.is_file() else []


def _no_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    from rote.interact.agents import SIGNALS

    for var in ("AI_AGENT", "ROTE_AGENT_MODE", *(s.var for s in SIGNALS)):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fixtures: FixtureServer) -> World:
    from rote.skill_target.trust import CA_VARS, PROXY_VARS

    for var in (*PROXY_VARS, *CA_VARS):
        monkeypatch.delenv(var, raising=False)
    _no_agent(monkeypatch)
    fixtures.reset()
    # Both sides call the fixture server, registered read-only (the eval calls
    # nothing else).
    save_registry(
        McpRegistry(
            servers={
                s: McpServerConfig(url=fixtures.url, read_only=True)
                for s in ("mslearn", "deepwiki")
            }
        )
    )
    original = tmp_path / "original-output.json"
    original.write_text(json.dumps(demo.load_reference().skill_output))
    monkeypatch.setenv("FAKE_ORIGINAL_OUTPUT", str(original))
    monkeypatch.chdir(tmp_path)
    w = World(tmp_path)
    monkeypatch.setattr(
        try_skill,
        "RUNNERS",
        lambda: try_skill.Runners(
            compile=w.compile, run_subject=w.run_subject, executable=lambda: str(FAKE_CLAUDE)
        ),
    )
    return w


def _cli(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    status = main(["try", *argv])
    captured = capsys.readouterr()
    return status, captured.out, captured.err


def _person(monkeypatch: pytest.MonkeyPatch, *answers: Answer) -> ScriptedPrompter:
    prompter = ScriptedPrompter(list(answers))
    monkeypatch.setattr("rote.interact.prompter.prompter_for", lambda mode: prompter)
    return prompter


YES = Answer("confirm", "--yes", True)


def _confirmed(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    *flags: str,
    yeses: int = 3,
) -> tuple[int, str, str]:
    """``rote try <flags>`` with a person who says yes ``yeses`` times. The
    install needs a person here (the world's servers are registered
    read-only), so a run that gets past it has one."""
    prompter = _person(monkeypatch, *([YES] * yeses))
    result = _cli(capsys, *flags)
    assert prompter.answers == [], f"{len(prompter.answers)} answer(s) left over: {result[2]}"
    return result


# ───────── consent ─────────


def test_json_stops_at_the_compile_plan_with_the_pinned_argv_and_spends_nothing(
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    status, out, err = _cli(capsys, "docs-digest", "--json")
    assert status == 2, err
    payload = json.loads(out)
    assert (payload["status"], payload["reason"]) == ("action_required", "confirmation_required")
    assert payload["missing"] == ["--yes", "--live"]
    assert payload["next"] == [{"command": "rote try docs-digest --json --yes --live"}]
    (action,) = payload["plan"]["actions"]
    argv = action["command"].split()
    # review6 fix 1: the exact argv, --local and the runtime pinned, printed in the plan.
    assert argv == ["rote", *skill_ab.compile_argv(demo.skill_dir(), world.out, model=None)]
    assert "--local" in argv and argv[argv.index("--runtime") + 1] == "skill"
    # review6 fix 2: the decider is pinned on the argv and named in the plan.
    assert argv[argv.index("--decider") + 1] == "rules"
    assert any("decider rules" in n for n in payload["plan"]["notes"])
    spend = [e for e in action["effects"] if e["kind"] == "spend"]
    assert spend and spend[0]["flag"] == "--live" and spend[0]["high_usd"] > 0
    assert world.compiles == [] and not world.out.exists()


def test_yes_alone_never_spends(world: World, capsys: pytest.CaptureFixture[str]) -> None:
    status, out, _ = _cli(capsys, "docs-digest", "--json", "--yes")
    assert status == 2
    assert json.loads(out)["missing"] == ["--live"]
    assert world.compiles == [] and world.ledger() == []


def test_without_a_terminal_the_plan_names_the_flags_and_nothing_runs(
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    status, out, err = _cli(capsys, "docs-digest")
    assert status == 2
    assert out == ""
    assert "Pass --yes --live to apply it" in err
    assert world.compiles == []


def test_an_agent_without_a_skill_is_asked_which_one(
    world: World, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CLAUDE_CODE_CHILD_SESSION", "1")
    status, out, _ = _cli(capsys, "--json")
    payload = json.loads(out)
    assert (status, payload["reason"]) == (2, "missing_answer")
    assert payload["missing"] == ["--skill"]
    assert str(demo.skill_dir().resolve()) in payload["choices"]
    assert world.compiles == []


def test_a_person_confirms_each_step_and_sees_the_result(
    world: World, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    prompter = _person(monkeypatch, Answer("select", "--skill", DEFAULT), YES, YES, YES)
    status, _, err = _cli(capsys)
    assert status == 0, err
    # Three plans, each confirmed by the person: compile, install, eval.
    assert [q[:2] for q in prompter.asked] == [
        ("select", "--skill"),
        ("confirm", "--yes"),
        ("confirm", "--yes"),
        ("confirm", "--yes"),
    ]
    assert len(world.compiles) == 1 and len(world.evals) == 1
    lines = err.splitlines()
    start = lines.index(
        "rote try docs-digest: Same answer: the compiled skill matched sections and "
        "reading_list_size in both runs."
    )
    # Two trials: the original against itself is the noise floor.
    assert lines[start + 1] == (
        "For scale, the original skill matched its own other run(s) on 4 of 4 fields."
    )
    # The live-data fields, each with the reason it is not expected to match.
    why = {f.name: f.why for f in demo.load_reference().fields}
    assert f"  reading_list: matched in 2 of 2 runs. {why['reading_list']}" in lines
    assert f"  repo_outline: matched in 2 of 2 runs. {why['repo_outline']}" in lines
    # The metrics come last, before the spend and the next steps.
    metrics = next(i for i, line in enumerate(lines) if line.startswith("How each ran"))
    assert metrics > lines.index("Not expected to match (live data or model wording):")
    assert any(line.startswith("  input tokens") for line in lines[metrics:])
    assert "Spent $" in err
    result = json.loads((world.out / try_skill.RESULT_FILE).read_text())
    assert result["same_result"] is True and result["noise_pairs"] == 1
    # The installed skill is the compiled one, in the user's own registry.
    from rote.app_registry import registered_apps

    assert any(
        a.name == "docs-digest" and a.trust is not None
        for a in registered_apps()
        if a.path == (world.out / "compile" / "runtime" / "skill" / "docs-digest").resolve()
    )


def test_a_person_who_declines_the_eval_spends_only_on_the_compile(
    world: World, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _person(monkeypatch, YES, YES, Answer("confirm", "--yes", False))
    status, _, err = _cli(capsys, "docs-digest")
    assert status == 1
    assert "stopped at eval (not applied)" in err
    assert world.evals == []
    assert [r["kind"] for r in world.ledger()] == ["compile"]


def test_an_agent_stops_at_each_consent_point_and_resumes(
    world: World, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CLAUDE_CODE_CHILD_SESSION", "1")
    flags = ("docs-digest", "--json", "--trials", "1")
    status, out, _ = _cli(capsys, *flags)
    assert (status, json.loads(out)["missing"]) == (2, ["--yes", "--live"])
    # With the spend flags: the compile runs, then the install stops for the
    # person, because the skill relies on servers registered read-only.
    status, out, _ = _cli(capsys, *flags, "--yes", "--live")
    payload = json.loads(out)
    assert (status, payload["reason"]) == (2, "person_required")
    compiled_skill = world.out / "compile" / "runtime" / "skill" / "docs-digest"
    assert payload["next"][0]["command"] == f"rote skill install {compiled_skill}"
    assert len(world.compiles) == 1 and world.evals == []
    # The person installs it in their own terminal.
    from rote.interact import prompter as prompting

    real_prompter_for = prompting.prompter_for
    monkeypatch.delenv("CLAUDE_CODE_CHILD_SESSION")
    _person(monkeypatch, YES, Answer("confirm", "--save-allow-rule", False))
    assert main(["skill", "install", str(compiled_skill)]) == 0
    capsys.readouterr()
    # Back to the agent.
    monkeypatch.setattr(prompting, "prompter_for", real_prompter_for)
    monkeypatch.setenv("CLAUDE_CODE_CHILD_SESSION", "1")
    status, out, err = _cli(capsys, *flags, "--yes", "--live")
    payload = json.loads(out)
    assert status == 0, err
    # Resumed: no second compile, the install read as done, the eval ran.
    assert len(world.compiles) == 1 and len(world.evals) == 1
    assert [s["state"] for s in payload["steps"]] == ["skipped", "skipped", "applied"]
    assert payload["same_result"] is True and payload["noise_pairs"] == 0
    assert "Each skill ran once, so run-to-run variation was not measured." in err


# ───────── resume ─────────


def test_an_interrupted_eval_resumes_without_recompiling_and_a_done_run_costs_nothing(
    world: World, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupted(*_: Any) -> Any:
        raise KeyboardInterrupt

    real = world.run_subject
    monkeypatch.setattr(world, "run_subject", interrupted)
    flags = ("docs-digest", "--trials", "1")
    status, _, err = _confirmed(capsys, monkeypatch, *flags)
    assert status == 130 and "Run the same command again to resume" in err
    monkeypatch.setattr(world, "run_subject", real)
    # Resumed: only the eval is asked about again.
    status, _, err = _confirmed(capsys, monkeypatch, *flags, yeses=1)
    assert status == 0, err
    assert "done already: compiled into" in err and len(world.compiles) == 1
    spent = world.ledger()
    # Done: a re-run asks nothing, shows the recorded result and runs nothing.
    status, _, err = _confirmed(capsys, monkeypatch, *flags, yeses=0)
    assert status == 0
    assert err.count("done already") == 3
    assert "Same answer: the compiled skill matched sections and reading_list_size" in err
    assert world.ledger() == spent and len(world.evals) == 1


def test_an_edited_skill_is_compiled_again(
    world: World,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    skill = tmp_path / "skill"
    shutil.copytree(demo.skill_dir(), skill)
    flags = (str(skill), "--trials", "1")
    assert _confirmed(capsys, monkeypatch, *flags)[0] == 0
    # A new file (SKILL.md itself is digested by the emit of the fake's
    # committed compile, which an edit there would refuse).
    (skill / "notes.txt").write_text("a note the skill now ships\n")
    # The compile and the eval again (the original changed); the compiled
    # skill came out byte for byte the same, so its install stands.
    status, _, err = _confirmed(capsys, monkeypatch, *flags, yeses=2)
    assert status == 0, err
    assert len(world.compiles) == 2 and len(world.evals) == 2


# ───────── the honest result ─────────


def test_no_same_result_claim_from_zero_compared_fields(
    world: World, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """review6 fix 3: every field drifts, so nothing is compared exactly."""
    from dataclasses import replace

    reference = demo.load_reference()
    drifting = replace(
        reference, fields=tuple(replace(f, expect="drifts") for f in reference.fields)
    )
    monkeypatch.setattr(demo, "load_reference", lambda path=None: drifting)
    status, out, err = _confirmed(capsys, monkeypatch, "docs-digest", "--trials", "1", "--json")
    payload = json.loads(out)
    assert status == 1
    assert payload["headline"] == "no fields to compare exactly"
    assert (payload["same_result"], payload["ok"], payload["compared"]) == (False, False, 0)
    assert "rote try docs-digest: No same-answer claim: none of the fields" in err
    assert "Same answer" not in err


def test_a_field_that_differs_is_named_with_both_values(
    world: World, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    output = dict(demo.load_reference().skill_output)
    lines = output["digest"].splitlines()
    first_bullet = next(i for i, line in enumerate(lines) if line.startswith("- ["))
    del lines[first_bullet]
    output["digest"] = "\n".join(lines)
    path = world.tmp / "short.json"
    path.write_text(json.dumps(output))
    monkeypatch.setenv("FAKE_ORIGINAL_OUTPUT", str(path))
    status, out, err = _confirmed(capsys, monkeypatch, "docs-digest", "--trials", "1", "--json")
    assert status == 1
    payload = json.loads(out)
    assert payload["headline"] == "reproduced 1 of 2 fields in every pair"
    assert "  reading_list_size: original '4', compiled '5'" in err
    assert (
        "Different answer: reading_list_size matched in 0 of 1 runs. sections matched in the "
        "one run." in err
    )


def test_the_eval_is_capped_at_what_was_approved(
    world: World, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    status, _, err = _confirmed(capsys, monkeypatch, "docs-digest", "--trials", "1")
    assert status == 0, err
    (harness,) = world.evals
    estimates = try_skill.estimate(demo.skill_dir())
    assert harness.ledger.cap_usd == pytest.approx(
        COMPILE_COST + estimates.eval(1).high_usd, abs=1e-6
    )
    assert harness.trial_estimate_usd == estimates.trial.high_usd


def test_the_parser_spells_the_modules_constants() -> None:
    from rote.cli import _build_parser

    args = _build_parser().parse_args(["try", try_skill.LIVE_FLAG])
    assert args.trials == try_skill.DEFAULT_TRIALS and args.live is True


def test_a_long_difference_shows_where_the_two_part() -> None:
    shared = "Available pages for a repo: " + "- 1 Overview " * 10
    text = try_skill.difference(shared + "- 2 Setup", shared + "- 2 Getting Started")
    assert text.startswith(f"they part at character {len(shared) + 4}:")
    assert (
        "original '...1 Overview - 2 Setup', compiled '...1 Overview - 2 Getting Started'" in text
    )
    assert try_skill.difference(4, 5) == "original '4', compiled '5'"


def test_yes_live_with_nobody_at_the_terminal_stops_at_the_install(
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    """No terminal and no agent: --yes --live pays for the compile, but the
    install relies on servers registered read-only, and only a person may
    approve that."""
    status, out, err = _cli(capsys, "docs-digest", "--yes", "--live", "--json")
    payload = json.loads(out)
    assert (status, payload["reason"]) == (2, "person_required")
    assert "needs the user at a terminal" in err
    assert len(world.compiles) == 1 and world.evals == []


def test_small_token_counts_print_whole() -> None:
    assert try_skill._num(640, "k") == "640"
    assert try_skill._num(47400, "k") == "47.4k"


def test_the_noise_floor_is_the_original_against_its_own_other_run(
    world: World, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The second original run's repo outline differs: that field drifts in one
    pair and in the original against itself, and the rest hold."""
    reference = demo.load_reference().skill_output
    changed = dict(reference)
    changed["digest"] = reference["digest"].replace("- 1 Overview", "- 1 Introduction")
    paths = []
    for index, output in enumerate((reference, changed)):
        path = world.tmp / f"original-{index}.json"
        path.write_text(json.dumps(output))
        paths.append(str(path))
    monkeypatch.setenv("FAKE_ORIGINAL_OUTPUTS", os.pathsep.join(paths))
    monkeypatch.setenv("FAKE_COUNTER", str(world.tmp / "counter"))
    status, _, err = _confirmed(capsys, monkeypatch, "docs-digest", "--trials", "2")
    assert status == 0, err
    lines = err.splitlines()
    assert "For scale, the original skill matched its own other run(s) on 3 of 4 fields." in lines
    why = {f.name: f.why for f in demo.load_reference().fields}
    at = lines.index(f"  repo_outline: matched in 1 of 2 runs. {why['repo_outline']}")
    assert lines[at + 1].strip().startswith("The original skill also differed from itself here")
    assert f"  reading_list: matched in 2 of 2 runs. {why['reading_list']}" in lines
    # One pair of original runs: the outline matched in neither.
    result = json.loads((world.out / try_skill.RESULT_FILE).read_text())
    noise = {f["name"]: (f["noise_matched"], f["noise_pairs"]) for f in result["fields"]}
    assert noise == {
        "sections": (1, 1),
        "reading_list_size": (1, 1),
        "reading_list": (1, 1),
        "repo_outline": (0, 1),
    }


def test_a_run_that_failed_its_checks_is_not_a_same_answer() -> None:
    fields = [{"name": "total", "expect": "equal", "matched": 1, "pairs": 1}]
    assert try_skill.answer(fields, 1, ()).startswith("Same answer:")
    assert try_skill.answer(fields, 1, ("compiled #1: bypassed the driver",)).startswith(
        "Not confirmed: the compiled skill matched total in the one run, but a run failed"
    )
