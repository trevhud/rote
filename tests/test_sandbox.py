"""Agent-written tests run in an OS sandbox, or not at all.

A reviewer showed that an environment allowlist alone stops nothing a
determined test wants: ``pwd`` names the real home, the network is open,
the ssh-agent socket and the Keychain CLI are a path away. So the
generated tests run under sandbox-exec (macOS) or bubblewrap (Linux).

The snooping test below runs as a *generated* test inside the sandbox:
each of its assertions passes only if the sandbox denied the attempt.
The same deliverable run with ``trust_skill=True`` (no sandbox) must
fail, which is what proves the assertions can fail at all.

CI must exercise this, not skip it: off CI a machine without a sandbox
skips; on CI (``CI`` set) the same condition fails the test.
"""

from __future__ import annotations

import os
import pwd
import sys
from pathlib import Path

import pytest

from rote.compiler import sandbox
from rote.compiler.verify import run_generated_tests

BOX, WHY = sandbox.detect()

needs_sandbox = pytest.mark.skipif(
    BOX is None and not os.environ.get("CI"),
    reason=f"no OS sandbox on this machine ({WHY})",
)

REAL_HOME = pwd.getpwuid(os.getuid()).pw_dir
if sys.platform == "darwin":
    AGENT_SOCKET_DIR = "/private/tmp"  # launchd puts SSH_AUTH_SOCK under it
    SECRET_CLI = "/usr/bin/security"  # the Keychain
else:
    AGENT_SOCKET_DIR = f"/run/user/{os.getuid()}"
    SECRET_CLI = "/bin/sh"

SNOOP = f"""\
import os
import pwd
import socket
import subprocess

import pytest

from extracted.logic import double


def test_the_work_dir_modules_import_and_run():
    assert double(21) == 42


def test_the_real_home_cannot_be_read():
    home = {REAL_HOME!r}
    try:
        home = pwd.getpwuid(os.getuid()).pw_dir
    except KeyError:
        pass
    with pytest.raises(OSError):
        os.listdir(home)


def test_the_network_is_unreachable():
    with pytest.raises(OSError):
        socket.create_connection(("1.1.1.1", 443), timeout=5)


def test_the_agent_socket_dir_cannot_be_listed():
    with pytest.raises(OSError):
        os.listdir({AGENT_SOCKET_DIR!r})


def test_no_other_program_can_run():
    with pytest.raises(OSError):
        subprocess.run([{SECRET_CLI!r}, "-c" if {SECRET_CLI!r} == "/bin/sh" else "help"])
"""


def _deliverable(root: Path) -> Path:
    (root / "extracted").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "extracted" / "logic.py").write_text("def double(x):\n    return x * 2\n")
    (root / "tests" / "test_snoop.py").write_text(SNOOP)
    return root


@needs_sandbox
def test_the_sandbox_denies_home_network_agent_socket_and_exec(tmp_path: Path) -> None:
    assert BOX is not None, f"CI must have a working sandbox: {WHY}"
    result = run_generated_tests(_deliverable(tmp_path / "work"))
    assert result is not None
    assert result.status == "passed", result.output
    assert "5 passed" in result.summary


@needs_sandbox
def test_without_the_sandbox_the_same_tests_fail(tmp_path: Path) -> None:
    """The negative half: these assertions are capable of failing."""
    result = run_generated_tests(_deliverable(tmp_path / "work"), trust_skill=True)
    assert result is not None and result.status == "failed", result.output
    # The legitimate test still passes; it is the denials that fail.
    assert "test_the_real_home_cannot_be_read" in result.output
    assert "test_no_other_program_can_run" in result.output


def test_no_sandbox_means_not_run_and_unverified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sandbox, "detect", lambda: (None, "nothing here"))
    result = run_generated_tests(_deliverable(tmp_path / "work"))
    assert result is not None
    assert result.status == "unverified" and not result.passed
    assert "nothing here" in result.summary and "--trust-skill" in result.summary


def test_the_macos_profile_denies_by_default_and_allows_no_network(tmp_path: Path) -> None:
    profile = sandbox.macos_profile(tmp_path / "w", tmp_path / "t", [tmp_path / "py"])
    assert "(deny default)" in profile
    assert "network" not in profile
    assert profile.count("(allow process-exec") == 1
    assert str(Path(os.path.realpath(sys.executable))) in profile


def test_the_bwrap_namespace_has_no_network_and_no_home(tmp_path: Path) -> None:
    argv = sandbox.bwrap_argv(tmp_path / "w", tmp_path / "t", [tmp_path / "py"])
    assert "--unshare-all" in argv
    bound = {argv[i + 2] for i, a in enumerate(argv) if a in ("--bind", "--ro-bind")}
    assert REAL_HOME not in bound
    assert not any(b.startswith("/home") or b == "/root" for b in bound)
