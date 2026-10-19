"""deal-monitor's scoring rubric reaches the judge whether or not it is passed.

The IR promised that an omitted ``scoring_rubric`` loads the deal-scoring
SKILL.md from its default path, but nothing did: the judge read
``pipeline.input.scoring_rubric`` directly, so an omitted rubric left a
required judge field empty. ``load_scoring_rubric`` now resolves it, and
these tests run that step of the emitted app on both paths, plus the
failure when neither source exists.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from rote.adapters import get_adapter
from rote.ir import load_pipeline

PIPELINE = (
    Path(__file__).resolve().parent.parent
    / "examples"
    / "deal-monitor"
    / "expected"
    / "pipeline.yaml"
)

_DRIVER = """\
import json, sys
from main import load_scoring_rubric
try:
    print(json.dumps({"rubric": load_scoring_rubric(json.loads(sys.argv[1]))}))
except Exception as exc:
    print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
"""


@pytest.fixture(scope="module")
def app(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("deal-monitor")
    get_adapter("python", extracted_source_dir=PIPELINE.parent).emit(load_pipeline(PIPELINE), out)
    (out / "driver.py").write_text(_DRIVER, encoding="utf-8")
    return out


def _step(app: Path, home: Path, payload: dict[str, str]) -> dict[str, str]:
    run = subprocess.run(
        [sys.executable, "driver.py", json.dumps(payload)],
        cwd=app,
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "HOME": str(home)},
    )
    assert run.returncode == 0, run.stderr
    result: dict[str, str] = json.loads(run.stdout.strip().splitlines()[-1])
    return result


def test_the_judge_reads_the_resolved_rubric_not_the_raw_input() -> None:
    pipeline = load_pipeline(PIPELINE)
    judge = next(n for n in pipeline.nodes if n.id == "score_new_opportunities")
    assert judge.inputs["scoring_rubric"] == "load_scoring_rubric.output"
    loader = next(n for n in pipeline.nodes if n.id == "load_scoring_rubric")
    assert loader.inputs == {"scoring_rubric": "pipeline.input.scoring_rubric"}
    assert "scoring_rubric" in pipeline.optional_input_fields


def test_a_passed_rubric_is_used_as_is(app: Path, tmp_path: Path) -> None:
    skill = tmp_path / ".claude" / "skills" / "deal-scoring" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("the installed rubric", encoding="utf-8")
    assert _step(app, tmp_path, {"scoring_rubric": "High: ARR over $1M."}) == {
        "rubric": "High: ARR over $1M."
    }


def test_an_omitted_rubric_loads_the_deal_scoring_skill(app: Path, tmp_path: Path) -> None:
    skill = tmp_path / ".claude" / "skills" / "deal-scoring" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("# Deal scoring\nHigh: ARR over $1M.\n", encoding="utf-8")
    assert _step(app, tmp_path, {}) == {"rubric": "# Deal scoring\nHigh: ARR over $1M.\n"}


def test_no_rubric_anywhere_fails_naming_the_path(app: Path, tmp_path: Path) -> None:
    result = _step(app, tmp_path, {})
    expected = tmp_path / ".claude" / "skills" / "deal-scoring" / "SKILL.md"
    assert result["error"].startswith("FileNotFoundError: load_scoring_rubric:")
    assert str(expected) in result["error"]
