"""``rote bench`` skill discovery: which skills a run benches by default.

No compiles here (those spend tokens). The holdout split is the point:
discovery on ``examples/`` must find exactly the five tuned-against
skills, and ``examples/holdout`` must find the held-out ones, so a
default bench never mixes the two sets.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rote.bench.runner import DEFAULT_SKILLS, discover_skills

REPO_ROOT = Path(__file__).resolve().parent.parent


def _skill(root: Path, name: str, body: str = "x", expected: bool = True) -> None:
    (root / name / "skill").mkdir(parents=True)
    (root / name / "skill" / "SKILL.md").write_text(body, encoding="utf-8")
    if expected:
        (root / name / "expected").mkdir()
        (root / name / "expected" / "pipeline.yaml").write_text("name: x\n", encoding="utf-8")


def test_repo_examples_discover_exactly_the_default_skills_in_order() -> None:
    assert discover_skills(REPO_ROOT / "examples") == list(DEFAULT_SKILLS)


def test_holdout_is_a_separate_set() -> None:
    holdout = discover_skills(REPO_ROOT / "examples" / "holdout")
    assert len(holdout) >= 3
    assert not set(holdout) & set(DEFAULT_SKILLS)


def test_unknown_skills_sort_smallest_first(tmp_path: Path) -> None:
    _skill(tmp_path, "aaa-big", "x" * 500)
    _skill(tmp_path, "zzz-small", "x" * 10)
    _skill(tmp_path, "mid", "x" * 100)
    (tmp_path / "mid" / "skill" / "references").mkdir()
    (tmp_path / "mid" / "skill" / "references" / "r.md").write_text("x" * 1000, encoding="utf-8")
    # references count toward size: mid (1100 bytes) is now the biggest.
    assert discover_skills(tmp_path) == ["zzz-small", "aaa-big", "mid"]


def test_dirs_without_a_skill_or_an_expected_ir_are_skipped(tmp_path: Path) -> None:
    _skill(tmp_path, "scored")
    _skill(tmp_path, "no-answer-key", expected=False)
    (tmp_path / "bench").mkdir()
    assert discover_skills(tmp_path) == ["scored"]


def test_an_empty_examples_dir_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="no <name>/skill"):
        discover_skills(tmp_path)


def test_bench_can_pin_the_checker_and_reports_which_ran(tmp_path: Path) -> None:
    """The default decider (auto) depends on the machine's credentials, so
    a bench row says which ran, and `--decider` pins it."""
    import json

    from rote.bench.collect import _checker
    from rote.bench.runner import _parser, compile_argv

    pinned = compile_argv(
        tmp_path / "s", tmp_path / "o", _parser().parse_args(["--decider", "jev"])
    )
    assert pinned[pinned.index("--decider") + 1] == "jev"
    assert "--decider" not in compile_argv(tmp_path / "s", tmp_path / "o", _parser().parse_args([]))

    compiled = tmp_path / "compiled"
    (compiled / "system1").mkdir(parents=True)
    (compiled / "system1" / "decisions.json").write_text(
        json.dumps({"checker": {"decider": "clef", "why": "the default"}})
    )
    assert _checker(compiled) == "clef"
    assert _checker(tmp_path / "none") is None


def test_a_bench_job_records_its_checker(tmp_path: Path) -> None:
    import json
    import shutil

    from rote.bench.collect import collect_job

    repo = Path(__file__).resolve().parents[1] / "examples" / "ops-report"
    compiled = tmp_path / "out" / "compiled"
    (compiled / "system1").mkdir(parents=True)
    shutil.copy(repo / "expected" / "pipeline.yaml", compiled / "pipeline.yaml")
    (compiled / "system1" / "decisions.json").write_text(
        json.dumps({"checker": {"decider": "jev", "why": "the default"}})
    )
    job = collect_job(
        tmp_path / "out",
        repo / "skill",
        repo / "expected" / "pipeline.yaml",
        exit_code=0,
        wall_seconds=1.0,
        tsc=False,
    )
    assert job["run"]["checker"] == "jev"
