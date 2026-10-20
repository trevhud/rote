"""The held-out gold pipelines type-check on every TypeScript runtime.

``tests/test_examples.py`` already emits every holdout gold on every
adapter, but for TS it only proves emission does not crash. A
TypeScript-only regression (an emitted type that tsc rejects) would pass
there and surface only when a bench run type-checks a compile. These
golds are a good net for that class: between them they carry MCP
``args`` literals and constants, an MCP-bound fan_out, a fanned judge
with shared context, gates with inputs, and an agent loop with a
partial ``tool_servers`` map.

Each case emits one gold, runs ``npm install`` and ``tsc --noEmit``.
Gated behind ``@pytest.mark.slow``: needs Node/npm and the npm registry.
No inference: tsc only.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from rote.adapters import get_adapter
from rote.ir import load_pipeline
from tests import _node_trees as node_trees

pytestmark = pytest.mark.slow

REPO_ROOT = Path(__file__).resolve().parent.parent
HOLDOUT_PIPELINES = sorted((REPO_ROOT / "examples" / "holdout").glob("*/expected/pipeline.yaml"))
TS_RUNTIMES = ("cloudflare", "dbos-ts", "inngest")


def test_the_holdout_sweep_is_not_degenerate() -> None:
    assert len(HOLDOUT_PIPELINES) >= 3


@pytest.mark.parametrize("runtime", TS_RUNTIMES)
@pytest.mark.parametrize("pipeline_yaml", HOLDOUT_PIPELINES, ids=lambda p: p.parent.parent.name)
def test_holdout_gold_type_checks(pipeline_yaml: Path, runtime: str, tmp_path: Path) -> None:
    if shutil.which("node") is None or shutil.which("npm") is None:
        pytest.skip("Node / npm not available")
    out = tmp_path / runtime
    get_adapter(runtime).emit(load_pipeline(pipeline_yaml), out)
    node_trees.install(out)
    tsc = subprocess.run(
        ["npx", "--no-install", "tsc", "--noEmit"],
        cwd=out,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert tsc.returncode == 0, f"tsc rejected the emitted {runtime} app:\n{tsc.stdout}{tsc.stderr}"
