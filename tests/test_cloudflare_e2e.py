"""Slow integration tests for the Cloudflare adapter.

These tests verify that the emitted TypeScript is *real, deployable code*
— not just structurally plausible. They run ``npm install`` and
``tsc --noEmit`` against the emitted output. That requires Node.js + npm
on the host and a network connection for the install step.

Tests are gated:

* ``@pytest.mark.slow`` so they're easy to skip during fast iteration:
  ``pytest -m 'not slow'``.
* The Node toolchain is detected at runtime; if missing, tests skip with
  a clear message. CI should have Node available.

A future v0.3 milestone may add miniflare-based execution tests that
exercise ``step.do`` and ``waitForEvent`` against a real Workers
runtime; for now the tsc compile is the strongest mechanical signal we
can get without that dependency.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from rote.adapters.cloudflare import CloudflareAdapter
from rote.ir import Pipeline, load_pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
BDR_PIPELINE_YAML = REPO_ROOT / "examples" / "bdr-outreach" / "expected" / "pipeline.yaml"


pytestmark = pytest.mark.slow


def _node_available() -> bool:
    return shutil.which("node") is not None and shutil.which("npm") is not None


@pytest.fixture(scope="module")
def bdr_pipeline() -> Pipeline:
    return load_pipeline(BDR_PIPELINE_YAML)


@pytest.fixture(scope="module")
def emitted_dir(bdr_pipeline: Pipeline, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Emit the BDR pipeline + run npm install once per module."""
    if not _node_available():
        pytest.skip("Node / npm not available — skipping cloudflare e2e tests")

    out = tmp_path_factory.mktemp("cf-e2e")
    CloudflareAdapter().emit(bdr_pipeline, out)

    # Install dependencies. Use a fresh cache to avoid stale lockfile drift,
    # but keep silent unless something fails (the install is noisy).
    proc = subprocess.run(
        ["npm", "install", "--no-audit", "--no-fund"],
        cwd=out,
        capture_output=True,
        text=True,
        timeout=180,
        env={**os.environ, "npm_config_progress": "false"},
    )
    if proc.returncode != 0:
        pytest.fail(
            f"npm install failed in {out}:\n--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}"
        )
    return out


def test_emitted_typescript_compiles(emitted_dir: Path) -> None:
    """Run `tsc --noEmit` over the emitted output. Zero diagnostics expected.

    This is the strongest mechanical signal short of running the
    workflow. If this passes, the emitted code at least typechecks
    against the @cloudflare/workers-types definitions, the Anthropic
    SDK, and Zod.
    """
    proc = subprocess.run(
        ["npx", "--no-install", "tsc", "--noEmit"],
        cwd=emitted_dir,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"tsc --noEmit reported errors in emitted Cloudflare code:\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )


def test_node_modules_contains_expected_packages(emitted_dir: Path) -> None:
    """Sanity check that the emitted package.json's deps are real and resolvable."""
    node_modules = emitted_dir / "node_modules"
    assert (node_modules / "@anthropic-ai" / "sdk").exists()
    assert (node_modules / "zod").exists()
    assert (node_modules / "@cloudflare" / "workers-types").exists()
    assert (node_modules / "typescript").exists()
    assert (node_modules / "wrangler").exists()
