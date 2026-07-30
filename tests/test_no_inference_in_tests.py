"""The automated suite must not be able to spend inference tokens.

The project's split:

* **test** — needs no inference. Automated, fast, runs on every push.
* **eval** — costs tokens. Run deliberately and periodically
  (``rote eval --run``), never from CI.

`conftest._no_inference_in_tests` enforces the first half. These tests
verify the enforcement itself, because a guard nobody checks is a guard
that quietly stops working — and the failure mode is silent spending,
which no assertion elsewhere would ever notice.

This is not hypothetical: two DBOS e2e suites burned real subscription
inference on every run until it was caught by wall-clock timing, and a
`claude` binary is on PATH on the maintainer's machine right now.
"""

from __future__ import annotations

import os
import shutil

import pytest


def test_the_subscription_cli_is_unreachable_from_a_test() -> None:
    """`shutil.which("claude")` is how the subscription lane finds its CLI.

    This is the leak with no environment variable to scrub: `claude -p`
    authenticates from an OAuth session, so possessing no API key is not
    protection. Not finding the binary is the only defense.
    """
    assert shutil.which("claude") is None, (
        "a test can reach the real Claude CLI — an emitted judge or agent "
        "loop reaching the subscription lane would spend real tokens with "
        "nothing in the environment to stop it"
    )
    assert shutil.which("codex") is None


def test_real_vendor_credentials_are_invisible_to_a_test() -> None:
    """A vendor SDK built without an explicit key reads these from the env."""
    for var in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "OPENAI_API_KEY",
        "ROTE_CLOUD_TOKEN",
    ):
        assert os.environ.get(var) is None, (
            f"{var} is visible to tests — a developer running the suite with "
            f"real credentials exported would bill them"
        )


def test_the_guard_does_not_break_toolchain_probes() -> None:
    """The e2e suites gate on node/npm/docker; those must still resolve.

    A guard that blocked every `which` would turn the TypeScript e2e
    suites into silent skips — trading a spending bug for a coverage
    bug, which is exactly the trade this project keeps having to undo.
    """
    assert shutil.which("python3") is not None


def test_a_test_supplied_stub_binary_still_resolves(tmp_path) -> None:  # noqa: ANN001
    """Only bare-name PATH lookups are blocked.

    Several driver tests write a fake `claude` script and pass its
    absolute path as ``executable=``. That is a stub, not a
    subscription, and blocking it would force those tests to work around
    the guard — the usual first step toward disabling it.
    """
    stub = tmp_path / "claude"
    stub.write_text("#!/bin/sh\necho stub\n", encoding="utf-8")
    stub.chmod(0o755)
    assert shutil.which(str(stub)) == str(stub)


def test_an_opt_in_test_can_restore_what_it_needs(monkeypatch: pytest.MonkeyPatch) -> None:
    """The guard is a default, not a cage.

    A test that genuinely needs the lookup patches it back; the autouse
    fixture runs first, so a test-local monkeypatch wins. Verifying this
    keeps the guard from being seen as something to route around.
    """
    monkeypatch.setattr(shutil, "which", lambda name, *a, **k: f"/fake/{name}")
    assert shutil.which("claude") == "/fake/claude"
