"""Shared fixtures for the rote test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from rote.ir import Pipeline, load_pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
BDR_PIPELINE_YAML = REPO_ROOT / "examples" / "bdr-outreach" / "expected" / "pipeline.yaml"


@pytest.fixture(autouse=True)
def _isolated_mcp_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No test may read (or write!) the developer's real MCP registry,
    token store, or app registry — several code paths (eval MCP wiring,
    emitted-helper resolution, the `rote mcp` commands, and `rote emit`
    recording the output dir) consult them by default."""
    monkeypatch.setenv("ROTE_MCP_CONFIG", str(tmp_path / "mcp-config" / "mcp.json"))
    monkeypatch.setenv("ROTE_MCP_TOKEN_DIR", str(tmp_path / "mcp-tokens"))
    monkeypatch.setenv("ROTE_APPS_PATH", str(tmp_path / "rote-apps" / "apps.json"))
    # Same rule for the rote-cloud login: a developer's real credential
    # must never satisfy (or be clobbered by) a test's resolution chain.
    monkeypatch.setenv("ROTE_CLOUD_CRED_PATH", str(tmp_path / "cloud-cred" / "cloud.json"))
    # And for the layered config: a developer's ~/.config/rote/config.yaml
    # (or a stray rote.yaml above the test cwd) must not change what any
    # command resolves. Both paths point at files that don't exist.
    monkeypatch.setenv("ROTE_CONFIG_PATH", str(tmp_path / "rote-config" / "config.yaml"))
    monkeypatch.setenv("ROTE_PROJECT_CONFIG_PATH", str(tmp_path / "rote-config" / "rote.yaml"))
    # `rote compile` now always constructs its JSONL progress sink (the
    # <out>/progress.jsonl sidecar), whose price enrichment does a live
    # catalog fetch. No unit test may touch the network for it — tests
    # that exercise pricing re-patch this same seam with real numbers.
    monkeypatch.setattr(
        "rote.cli._JsonlProgressSink._resolve_prices",
        staticmethod(lambda _model_id: None),
    )


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
