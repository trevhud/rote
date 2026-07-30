"""Every slow test suite must actually run somewhere in CI.

`@pytest.mark.slow` excludes a suite from the default run, which is the
point — they need Node, Docker, or minutes. The failure mode is that
excluding them locally also excluded them from CI: for a long stretch
only ``test_mcp_e2e.py`` ran there, so 24 of 27 slow tests existed
solely on a maintainer's laptop. Three of the six supported runtimes
had no automated proof at all, and a fan_out regression that only the
Cloudflare e2e catches would have reached main unnoticed.

Marking a suite slow is a statement about the *toolchain it needs*, not
about whether it should run. This test makes adding a slow suite without
wiring it into a CI job a failure rather than an omission.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"


def _slow_test_files() -> set[str]:
    """Collect the slow suite by asking pytest, not by globbing.

    A file is only slow if pytest actually collects a slow test from it,
    so this can't drift from the markers.
    """
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/",
            "-m",
            "slow",
            "-q",
            "--collect-only",
            "--no-header",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, f"collection failed:\n{proc.stdout}\n{proc.stderr}"
    return {
        line.split("::")[0].rsplit("/", 1)[-1]
        for line in proc.stdout.splitlines()
        if line.startswith("tests/") and "::" in line
    }


def test_every_slow_suite_is_wired_into_a_ci_job() -> None:
    collected = _slow_test_files()
    # Sanity: if collection returns nothing the assertion below is vacuous.
    assert len(collected) >= 10, f"expected a substantial slow suite, got {sorted(collected)}"

    referenced = set(re.findall(r"tests/(test_\w+\.py)", CI_WORKFLOW.read_text(encoding="utf-8")))
    missing = sorted(collected - referenced)

    assert not missing, (
        f"These slow suites run nowhere in CI: {missing}. Add them to a job in "
        f"{CI_WORKFLOW.relative_to(REPO_ROOT)} — 'Python e2e' for pure-Python "
        f"suites, 'TypeScript e2e' for Node ones, 'DBOS-TS e2e' if they also "
        f"need Docker. Marking a suite slow says what toolchain it needs, not "
        f"that it should go unrun."
    )


def test_ci_does_not_reference_deleted_test_files() -> None:
    """A renamed suite must not leave CI silently running nothing.

    `pytest path/that/does/not/exist.py` exits 4, so this would surface
    as a red job — but only on the next push that happens to touch CI.
    Failing here names the stale path directly.
    """
    referenced = set(re.findall(r"tests/(test_\w+\.py)", CI_WORKFLOW.read_text(encoding="utf-8")))
    absent = sorted(name for name in referenced if not (REPO_ROOT / "tests" / name).is_file())
    assert not absent, f"ci.yml references test files that no longer exist: {absent}"
