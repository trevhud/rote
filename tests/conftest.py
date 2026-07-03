"""Shared fixtures for the rote test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from rote.ir import Pipeline, load_pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
BDR_PIPELINE_YAML = REPO_ROOT / "examples" / "bdr-outreach" / "expected" / "pipeline.yaml"


@pytest.fixture(scope="session")
def bdr_pipeline() -> Pipeline:
    """The canonical BDR pipeline — the IR that exercises all five node kinds.

    Session-scoped: one parse serves the whole suite. Tests must treat
    it as immutable; anything that needs to mutate should deep-copy (see
    test_dbos_adapter's copy.deepcopy usage) or build its own pipeline
    via tests._helpers.mini_pipeline.
    """
    assert BDR_PIPELINE_YAML.exists(), f"Missing fixture: {BDR_PIPELINE_YAML}"
    return load_pipeline(BDR_PIPELINE_YAML)
