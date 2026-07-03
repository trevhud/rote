"""`rote eval` CLI tests — pricing fetch monkeypatched, no network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rote.cli import main
from rote.eval.pricing import ModelPrice, ModelTier, PricingCatalog, PricingError

REPO_ROOT = Path(__file__).resolve().parent.parent
BDR_RUN = REPO_ROOT / "examples" / "bdr-outreach" / "runs" / "2026-07-03-data-flow"
BDR_SKILL = REPO_ROOT / "examples" / "bdr-outreach" / "skill"

FAKE_CATALOG = PricingCatalog(
    prices=(
        ModelPrice(
            model_id="fake-flagship",
            provider="anthropic",
            display_name="Fake Flagship",
            tier=ModelTier.FLAGSHIP,
            input_per_mtok=10.0,
            output_per_mtok=50.0,
            cache_read_per_mtok=1.0,
            cache_write_per_mtok=12.5,
            source="https://example.test/prices",
            fetched_at="2026-07-03T00:00:00Z",
        ),
    )
)


@pytest.fixture()
def fake_prices(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    monkeypatch.setattr("rote.eval.pricing.fetch_catalog", lambda **kwargs: FAKE_CATALOG)


def test_eval_markdown_to_stdout(fake_prices: None, capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["eval", str(BDR_RUN), "--skill", str(BDR_SKILL)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "# Eval scorecard — bdr-campaign" in out
    assert "fake-flagship" in out
    assert "## Assumptions (audit me)" in out


def test_eval_json_output(fake_prices: None, capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["eval", str(BDR_RUN), "--skill", str(BDR_SKILL), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["pipeline"] == "bdr-campaign"
    assert payload["before"] is not None
    assert payload["costs"][0]["model"] == "fake-flagship"


def test_eval_writes_out_file(fake_prices: None, tmp_path: Path) -> None:
    out = tmp_path / "card" / "scorecard.md"
    rc = main(["eval", str(BDR_RUN), "--skill", str(BDR_SKILL), "--out", str(out)])
    assert rc == 0
    assert "# Eval scorecard" in out.read_text(encoding="utf-8")


def test_eval_source_skill_resolves_automatically(
    fake_prices: None, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The committed run records source_skill relative to the repo root.
    monkeypatch.chdir(REPO_ROOT)
    rc = main(["eval", str(BDR_RUN)])
    assert rc == 0
    out = capsys.readouterr().out
    # A resolved baseline means the before column is populated.
    assert "As agent instructions" in out
    assert "_no skill baseline_" not in out


def test_eval_missing_pipeline_yaml_is_usage_error(
    fake_prices: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main(["eval", str(tmp_path)])
    assert rc == 2
    assert "no pipeline.yaml" in capsys.readouterr().err


def test_eval_price_outage_is_loud_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _down(**kwargs: object) -> PricingCatalog:
        raise PricingError("connection refused")

    monkeypatch.setattr("rote.eval.pricing.fetch_catalog", _down)
    rc = main(["eval", str(BDR_RUN), "--skill", str(BDR_SKILL)])
    assert rc == 1
    assert "could not fetch live prices" in capsys.readouterr().err


# ───────── --run mode (trial functions monkeypatched) ─────────


def test_eval_run_mode_appends_measured_section(
    fake_prices: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from rote.eval.empirical import MeasuredRun

    def fake_skill_trial(*args: object, **kwargs: object) -> MeasuredRun:
        return MeasuredRun(
            wall_seconds=90.0, output={"ok": True}, turns=9, cost_usd=0.4, output_tokens=1800
        )

    def fake_pipeline_trial(*args: object, **kwargs: object) -> MeasuredRun:
        return MeasuredRun(
            wall_seconds=3.0,
            output={"ok": True},
            judge_usage=({"model": "fake-flagship", "input_tokens": 100, "output_tokens": 10},),
        )

    monkeypatch.setattr("rote.eval.empirical.run_skill_trial", fake_skill_trial)
    monkeypatch.setattr("rote.eval.empirical.run_pipeline_trial", fake_pipeline_trial)
    monkeypatch.setattr("rote.eval.empirical.DEFAULT_CORPUS_PATH", tmp_path / "corpus.jsonl")

    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    (runtime_dir / "main.py").write_text("# app\n", encoding="utf-8")
    task = tmp_path / "task.json"
    task.write_text(json.dumps({"numbers": [1, 2]}), encoding="utf-8")

    rc = main(
        [
            "eval",
            str(BDR_RUN),
            "--skill",
            str(BDR_SKILL),
            "--run",
            "--trials",
            "2",
            "--input",
            str(task),
            "--runtime-dir",
            str(runtime_dir),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "## Measured (2 trials per side)" in out
    assert "list-price notional" in out
    corpus = (tmp_path / "corpus.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(corpus) == 1


def test_eval_run_requires_input(fake_prices: None, capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["eval", str(BDR_RUN), "--skill", str(BDR_SKILL), "--run"])
    assert rc == 1
    assert "--input" in capsys.readouterr().err


def test_eval_run_rejects_nonpositive_trials(
    fake_prices: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    task = tmp_path / "task.json"
    task.write_text("{}", encoding="utf-8")
    rc = main(
        [
            "eval",
            str(BDR_RUN),
            "--skill",
            str(BDR_SKILL),
            "--run",
            "--trials",
            "0",
            "--input",
            str(task),
        ]
    )
    assert rc == 1
    assert "--trials" in capsys.readouterr().err
