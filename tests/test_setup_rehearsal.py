"""``scripts/rehearse-setup``: the whole first run, both modes, with no inference.

The rehearsal is what proves the walkthrough before a person sees it: a
fixture Claude Code config with two localhost MCP servers, a fixture skill
that calls them, a stubbed compile and a fake ``claude``, from ``rote
setup`` to the verdict, once as an agent relaying each stop and once on a
pseudo-terminal answering the prompts.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastmcp")

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "rehearse-setup"


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["agent", "tty"])
def test_the_rehearsal_reaches_the_same_answer_verdict(mode: str) -> None:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--mode", mode],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    assert f"===== {mode} mode (exit 0) =====" in proc.stdout
    assert "Same answer: the compiled skill matched" in proc.stdout
