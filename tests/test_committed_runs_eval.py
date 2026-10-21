"""``rote eval`` works on every committed compile output.

Two committed runs carried ``eval.yaml`` sidecars the schema never
accepted (a per-path ``totals`` block, per-step ``notes``), and one
recorded a ``source_skill`` that resolved nowhere, so ``rote eval`` either
crashed on them or silently dropped the before side. Nothing ran eval over
the committed dirs, so no test noticed. This one does, for every
pipeline.yaml under examples/.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rote.cli import main
from rote.eval.sidecar import EVAL_SIDECAR_FILENAME, load_eval_estimates
from tests.test_eval_cli import FAKE_CATALOG

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
PIPELINE_DIRS = sorted(p.parent for p in EXAMPLES.rglob("pipeline.yaml"))
SIDECARS = sorted(EXAMPLES.rglob(EVAL_SIDECAR_FILENAME))


def _id(path: Path) -> str:
    return str(path.relative_to(EXAMPLES))


def test_the_sweep_is_not_degenerate() -> None:
    assert len(PIPELINE_DIRS) >= 20
    assert len(SIDECARS) >= 11


@pytest.mark.parametrize("sidecar", SIDECARS, ids=_id)
def test_every_committed_sidecar_validates(sidecar: Path) -> None:
    load_eval_estimates(sidecar)


@pytest.mark.parametrize("pipeline_dir", PIPELINE_DIRS, ids=_id)
def test_rote_eval_scores_every_committed_pipeline_with_its_skill(
    pipeline_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("rote.eval.pricing.fetch_catalog", lambda **kwargs: FAKE_CATALOG)
    assert main(["eval", str(pipeline_dir)]) == 0
    captured = capsys.readouterr()
    assert "# Eval scorecard" in captured.out
    # Every example ships its skill, so a before side is always possible.
    # A source_skill that resolves nowhere degrades to an after-only card.
    assert "no source skill found" not in captured.err
