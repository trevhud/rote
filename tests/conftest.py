"""Shared fixtures for the rote test suite."""

from __future__ import annotations

import os
import shutil
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


#: Credentials that would let a test reach a real model. Scrubbed from
#: every test's environment — see :func:`_no_inference_in_tests`.
_INFERENCE_CREDENTIAL_VARS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "ROTE_INFERENCE",
    "ROTE_CLOUD_TOKEN",
)

#: Subscription-auth CLIs. These are the dangerous ones: `claude -p`
#: needs no API key at all, so a test that merely *finds* the binary
#: spends the developer's subscription with nothing in the environment
#: to scrub. Exactly how two DBOS e2e suites quietly burned real
#: inference on every run until it was caught by timing.
_SUBSCRIPTION_CLIS = frozenset({"claude", "codex"})


@pytest.fixture(autouse=True)
def _no_inference_in_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    """A test may never spend tokens.

    The project's split: anything that does not require inference is a
    **test** — automated, fast, run on every push. Anything that costs
    tokens is an **eval** — run deliberately and periodically, never
    from CI. This fixture makes the first half impossible to violate by
    accident instead of merely documented.

    Two independent leaks, both closed here:

    1. **Credentials in the environment.** A developer running the suite
       locally has real keys exported; a vendor SDK constructed without
       an explicit key picks them up silently.
    2. **A subscription CLI on PATH.** `claude -p` authenticates from an
       OAuth session, so no env var gates it — the only defense is not
       finding the binary. `shutil.which` is wrapped rather than
       replaced so that lookups the e2e suites depend on (node, npm,
       docker, wrangler) still resolve normally.

    Only a *bare name* lookup is blocked, because only that consults
    PATH and can reach the developer's real install. Several tests build
    a fake `claude` script under tmp_path and pass its absolute path as
    `executable=`; that is a stub, not a subscription, and it still
    resolves.

    A test that legitimately needs one of these monkeypatches it back:
    this fixture runs first, so a test-local patch wins.
    """
    for var in _INFERENCE_CREDENTIAL_VARS:
        monkeypatch.delenv(var, raising=False)

    real_which = shutil.which

    def _which_without_subscription_clis(cmd: str, *args: object, **kwargs: object):  # noqa: ANN202
        if isinstance(cmd, str) and os.sep not in cmd and cmd in _SUBSCRIPTION_CLIS:
            return None
        return real_which(cmd, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(shutil, "which", _which_without_subscription_clis)


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
