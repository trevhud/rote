"""The automated suite must not be able to spend inference tokens.

The project's split:

* **test** — needs no inference. Automated, fast, runs on every push.
* **eval** — costs tokens. Run deliberately and periodically
  (``rote eval --run``), never from CI.

`conftest.pytest_configure` enforces the first half for the whole session
(tests/_inference_guard.py), and `conftest.pytest_runtest_call` fails any
test during which a subscription CLI resolves. These tests verify the
enforcement itself, because a guard nobody checks is a guard that quietly
stops working, and the failure mode is silent spending, which no assertion
elsewhere would ever notice.

This is not hypothetical: two DBOS e2e suites burned real subscription
inference on every run until it was caught by wall-clock timing, and later
a test's subprocess compile found `claude` on the inherited PATH and billed
$0.29, because the guard of the day only patched `shutil.which` in-process.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from rote import isolation
from rote.app_registry import apps_path
from rote.serve.registry import default_registry_path
from tests import _inference_guard
from tests.conftest import REPO_ROOT, SESSION_STATE

pytest_plugins = ("pytester",)


def _shell_finds(name: str) -> str:
    return subprocess.run(
        ["/bin/sh", "-c", f"command -v {name}; true"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def test_the_subscription_cli_is_unreachable_from_a_test() -> None:
    """`shutil.which("claude")` is how the subscription lane finds its CLI.

    This is the leak with no environment variable to scrub: `claude -p`
    authenticates from an OAuth session, so possessing no API key is not
    protection. Not finding the binary is the only defense.
    """
    for name in _inference_guard.SUBSCRIPTION_CLIS:
        assert shutil.which(name) is None, (
            f"a test can reach the real {name} CLI: an emitted judge or agent "
            "loop reaching the subscription lane would spend real tokens with "
            "nothing in the environment to stop it"
        )


def test_a_child_process_cannot_find_the_subscription_cli() -> None:
    """A subprocess resolves bare names from the PATH it inherits.

    Patching `shutil.which` never reached it; `python -m rote compile` run
    by a test is exactly such a child.
    """
    for name in _inference_guard.SUBSCRIPTION_CLIS:
        assert _shell_finds(name) == ""


def test_monkeypatch_undo_cannot_remove_the_guard(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Undoing every monkeypatch leaves the session's isolation in place.

    The guard and the state paths used to be per-test monkeypatches, so this
    exact call put the real CLI back on reach and unset ROTE_APPS_PATH.
    """
    monkeypatch.undo()

    for name in _inference_guard.SUBSCRIPTION_CLIS:
        assert shutil.which(name) is None
        assert _shell_finds(name) == ""

    session = request.config.stash[SESSION_STATE]
    for key, rel in isolation.STATE_PATHS.items():
        value = os.environ.get(key)
        assert value is not None, f"{key} is unset after monkeypatch.undo()"
        assert Path(value) == session / rel, f"{key} left the session state dir: {value}"
    for key, rel in _inference_guard.XDG_DIRS.items():
        assert os.environ.get(key) == str(session / rel), f"{key} left the session state dir"


def test_a_deleted_override_falls_back_under_the_session(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a ROTE_* override gone, a default path is still not the real one.

    Every default goes through an XDG dir, and those point under the
    session, so neither a test that deletes an override nor a child process
    that drops the ROTE_* variables reaches ~/.local/share/rote.
    """
    session = request.config.stash[SESSION_STATE]
    monkeypatch.delenv("ROTE_APPS_PATH")
    monkeypatch.delenv("ROTE_SERVE_REGISTRY_PATH")
    assert apps_path().is_relative_to(session)
    assert default_registry_path().is_relative_to(session)
    assert not apps_path().is_relative_to(Path.home())


def test_a_path_mirror_is_read_only(tmp_path: Path) -> None:
    """A test cannot drop a `claude` link into a mirror for later tests to find."""
    shared = tmp_path / "shared-bin"
    shared.mkdir()
    for name in ("claude", "node"):
        (shared / name).write_text("#!/bin/sh\n", encoding="utf-8")
        (shared / name).chmod(0o755)
    state = tmp_path / "state"
    mirror = Path(_inference_guard.scrub_path(str(shared), state / "path-mirrors"))

    assert mirror.stat().st_mode & 0o777 == 0o555
    with pytest.raises(PermissionError):
        (mirror / "claude").symlink_to(shared / "claude")
    # And the session can still delete it afterwards.
    _inference_guard.release_mirrors(state)
    shutil.rmtree(state)
    assert not state.exists()


def test_the_per_test_check_fails_a_test_that_puts_a_cli_on_path(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The conftest hook, run for real on an inner session, fails such a test.

    Its negative half is the inner test that leaves PATH alone and passes,
    so the failure is the check and not the inner session breaking.
    """
    pytester.makepyfile(
        test_inner="""
        import os

        def test_leaves_path_alone():
            pass

        def test_puts_a_stub_claude_on_path(tmp_path, monkeypatch):
            stub = tmp_path / "claude"
            stub.write_text("#!/bin/sh\\n", encoding="utf-8")
            stub.chmod(0o755)
            monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
        """
    )
    monkeypatch.setenv("PYTHONPATH", str(REPO_ROOT))
    result = pytester.runpytest_subprocess("-p", "tests.conftest", "-p", "no:cacheprovider")
    result.assert_outcomes(passed=1, failed=1)
    result.stdout.fnmatch_lines(
        ["*test_puts_a_stub_claude_on_path*", "*a subscription CLI is reachable after*"]
    )


def test_real_vendor_credentials_are_invisible_to_a_test() -> None:
    """A vendor SDK built without an explicit key reads these from the env."""
    for var in _inference_guard.INFERENCE_CREDENTIAL_VARS:
        assert os.environ.get(var) is None, (
            f"{var} is visible to tests — a developer running the suite with "
            f"real credentials exported would bill them"
        )


def test_scrub_path_mirrors_a_shared_bin_dir_without_the_clis(tmp_path: Path) -> None:
    """A dir holding `claude` beside other tools keeps the other tools.

    On the maintainer's machine `~/.local/bin` holds `claude` and `codex`
    next to `rote`, `op` and a `python3.11`; dropping the whole entry would
    take those with it.
    """
    shared = tmp_path / "shared-bin"
    shared.mkdir()
    for name in ("claude", "codex", "node"):
        tool = shared / name
        tool.write_text("#!/bin/sh\necho real\n", encoding="utf-8")
        tool.chmod(0o755)
    plain = tmp_path / "plain-bin"
    plain.mkdir()

    path = os.pathsep.join([str(shared), str(plain), str(shared)])
    scrubbed = _inference_guard.scrub_path(path, tmp_path / "mirrors")

    entries = scrubbed.split(os.pathsep)
    assert entries[1] == str(plain)
    assert entries[0] == entries[2] != str(shared)
    assert shutil.which("node", path=scrubbed) == str(Path(entries[0]) / "node")
    assert Path(entries[0], "node").resolve() == (shared / "node").resolve()
    assert shutil.which("claude", path=scrubbed) is None
    assert shutil.which("codex", path=scrubbed) is None
    # The negative half: unscrubbed, the same path does find them.
    assert shutil.which("claude", path=path) == str(shared / "claude")


def test_a_cli_put_on_path_is_reported(tmp_path: Path) -> None:
    """The per-test check sees a CLI on PATH, in-process and in a child."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "claude"
    stub.write_text("#!/bin/sh\necho stub\n", encoding="utf-8")
    stub.chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}

    assert _inference_guard.reachable_clis(env) == [
        f"shutil.which: {stub}",
        f"subprocess command -v: {stub}",
    ]
    assert _inference_guard.reachable_clis(os.environ) == []


def test_the_guard_does_not_break_toolchain_probes() -> None:
    """The e2e suites gate on node/npm/docker; those must still resolve.

    A guard that blocked every `which` would turn the TypeScript e2e
    suites into silent skips — trading a spending bug for a coverage
    bug, which is exactly the trade this project keeps having to undo.
    """
    assert shutil.which("python3") is not None


def test_a_test_supplied_stub_binary_still_resolves(tmp_path: Path) -> None:
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
    assert subprocess.run([str(stub)], capture_output=True, text=True).stdout == "stub\n"


def test_an_opt_in_test_can_patch_the_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    """A test may stand in a fake `shutil.which` for the code under test.

    The per-test check asks the filesystem through the real lookup, so a
    patched one does not trip it; it is not a way to reach a real CLI.
    """
    monkeypatch.setattr(shutil, "which", lambda name, *a, **k: f"/fake/{name}")
    assert shutil.which("claude") == "/fake/claude"
