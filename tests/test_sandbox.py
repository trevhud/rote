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
import tempfile
from collections.abc import Iterator
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


def _home_skeleton() -> set[str]:
    """The names a bwrap namespace may show at the top of the real home.

    bwrap creates every parent of a bind destination, so binding a
    uv-managed Python under ``~/.local/share/uv/python`` makes an empty
    ``~/.local`` appear inside, and a venv under ``~/work`` an empty
    ``~/work``. Those names are paths to the dirs Python needs, not
    contents of the home. Nothing else may be listed.
    """
    layout = sandbox.current_layout()
    home = Path(REAL_HOME)
    return {
        path.relative_to(home).parts[0]
        for path in (*layout.read_dirs, *layout.read_files, *layout.exec_files)
        if path.is_relative_to(home) and path != home
    }


def _snoop(secret_files: list[str]) -> str:
    return f"""\
import os
import pwd
import socket
import subprocess

import pytest

from extracted.logic import double


def test_the_work_dir_modules_import_and_run():
    assert double(21) == 42


def test_the_real_home_shows_no_contents():
    home = {REAL_HOME!r}
    try:
        home = pwd.getpwuid(os.getuid()).pw_dir
    except KeyError:
        pass
    try:
        listed = set(os.listdir(home))
    except OSError:
        return
    assert listed <= {_home_skeleton()!r}, listed


def test_the_users_files_cannot_be_read():
    for path in {secret_files!r}:
        with pytest.raises(OSError):
            open(path).read()
    for path in {[f"{REAL_HOME}/.ssh", f"{REAL_HOME}/.local/share/rote"]!r}:
        with pytest.raises(OSError):
            os.listdir(path)


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


@pytest.fixture
def planted() -> Iterator[list[str]]:
    """Real secret files beside what the sandbox binds: one in the real
    home, one in the venv root next to the site-packages it binds. Each
    exists and is readable outside, so the denial is capable of failing."""
    dirs = [REAL_HOME]
    if sys.prefix != sys.base_prefix:
        dirs.append(sys.prefix)
    paths = []
    try:
        for d in dirs:
            fd, path = tempfile.mkstemp(prefix=".rote-planted-secret-", dir=d)
            os.write(fd, b"s3cret")
            os.close(fd)
            paths.append(path)
        yield paths
    finally:
        for path in paths:
            os.unlink(path)


def _deliverable(root: Path, secret_files: list[str]) -> Path:
    (root / "extracted").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "extracted" / "logic.py").write_text("def double(x):\n    return x * 2\n")
    (root / "tests" / "test_snoop.py").write_text(_snoop(secret_files))
    return root


@needs_sandbox
def test_the_sandbox_denies_home_network_agent_socket_and_exec(
    tmp_path: Path, planted: list[str]
) -> None:
    assert BOX is not None, f"CI must have a working sandbox: {WHY}"
    result = run_generated_tests(_deliverable(tmp_path / "work", planted))
    assert result is not None
    assert result.status == "passed", result.output
    assert "6 passed" in result.summary


@needs_sandbox
def test_without_the_sandbox_the_same_tests_fail(tmp_path: Path, planted: list[str]) -> None:
    """The negative half: these assertions are capable of failing."""
    result = run_generated_tests(_deliverable(tmp_path / "work", planted), trust_skill=True)
    assert result is not None and result.status == "failed", result.output
    # The legitimate test still passes; it is the denials that fail.
    assert "test_the_real_home_shows_no_contents" in result.output
    assert "test_the_users_files_cannot_be_read" in result.output
    assert "test_no_other_program_can_run" in result.output


def test_no_sandbox_means_not_run_and_unverified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sandbox, "detect", lambda: (None, "nothing here"))
    result = run_generated_tests(_deliverable(tmp_path / "work", []))
    assert result is not None
    assert result.status == "unverified" and not result.passed
    assert "nothing here" in result.summary and "--trust-skill" in result.summary


def test_the_macos_profile_denies_by_default_and_allows_no_network(tmp_path: Path) -> None:
    layout = sandbox.current_layout()
    profile = sandbox.macos_profile(tmp_path / "w", tmp_path / "t", layout)
    assert "(deny default)" in profile
    assert "network" not in profile
    [exec_rule] = [ln for ln in profile.splitlines() if ln.startswith("(allow process-exec")]
    # Literals only: no directory of binaries is executable.
    assert "subpath" not in exec_rule
    assert str(Path(os.path.realpath(sys.executable))) in exec_rule


def test_the_bwrap_namespace_has_no_network_and_no_home_contents(tmp_path: Path) -> None:
    """A bind under the home is allowed, and is on CI: a venv in the
    checkout, or a uv-managed Python under ~/.local/share/uv/python. What
    must hold is that only the dirs this Python needs are bound there,
    never the home, an ancestor of it, or anything holding ~/.ssh or
    rote's own data."""
    layout = sandbox.current_layout()
    argv = sandbox.bwrap_argv(tmp_path / "w", tmp_path / "t", layout)
    assert "--unshare-all" in argv
    bound = {
        Path(argv[i + 2])
        for i, a in enumerate(argv)
        if a in ("--bind", "--ro-bind", "--ro-bind-try")
    }
    home = Path(REAL_HOME)
    needed = {*layout.read_dirs, *layout.read_files, *layout.exec_files}
    for path in bound:
        assert not home.is_relative_to(path), f"{path} exposes the home"
        for secret in (
            home / ".ssh",
            home / ".local" / "share" / "rote",
            home / ".config" / "rote",
        ):
            assert not secret.is_relative_to(path), f"{path} exposes {secret}"
        if path.is_relative_to(home):
            assert path in needed, f"{path} is under the home and not part of this Python"
    # The work and tmp dirs are writable, and nothing else is.
    writable = {Path(argv[i + 2]) for i, a in enumerate(argv) if a == "--bind"}
    assert writable == {tmp_path / "w", tmp_path / "t"}


def test_a_uv_managed_python_under_home_is_bound_not_refused() -> None:
    """setup-uv's interpreter lives under ~/.local/share/uv/python. Binding
    that install read-only is fine; it holds nothing of the user's."""
    home = Path("/home/runner")
    install = home / ".local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu"
    layout = sandbox.Layout(
        executable=install / "bin/python3.11",
        read_dirs=(
            install / "lib/python3.11",
            home / "work/rote/.venv/lib/python3.11/site-packages",
        ),
        read_files=(install / "bin/python3.11",),
    )
    assert sandbox.refusal(layout, {home}) is None
    argv = sandbox.bwrap_argv(Path("/w"), Path("/t"), layout)
    bound = {argv[i + 1] for i, a in enumerate(argv) if a == "--ro-bind"}
    assert str(install / "lib/python3.11") in bound
    assert not {str(home), str(home / ".local"), str(home / ".local/share")} & bound


def test_the_child_is_launched_by_the_real_binary_not_the_venv() -> None:
    """Through a venv, Python finds its stdlib via pyvenv.cfg's ``home``,
    a path nothing binds; every uv-managed run died on 'encodings'."""
    assert sandbox.launch_executable() == os.path.realpath(sys.executable)
    assert Path(sandbox.launch_executable()) in sandbox.current_layout().read_files


# ───────── An install that cannot be confined is refused, not loosened ─────────

HOME = Path("/Users/alice")


def _layout(executable: str, *dirs: str) -> sandbox.Layout:
    return sandbox.Layout(
        executable=Path(executable),
        read_dirs=tuple(Path(d) for d in dirs),
        read_files=(Path(executable),),
    )


#: The shapes the review found: a Python under the home, and a system
#: Python whose install root is /usr. Each derived root is checked.
UNSAFE = {
    "home itself readable": _layout("/Users/alice/bin/python", "/Users/alice"),
    "an ancestor of home": _layout("/Users/python", "/Users"),
    "rote's data dir under a root": _layout(
        "/Users/alice/.local/bin/python", "/Users/alice/.local"
    ),
    "rote's config dir under a root": _layout("/Users/alice/py/python", "/Users/alice/.config"),
    "~/.ssh under a root": _layout("/Users/alice/py/python", "/Users/alice/.ssh/.."),
    "filesystem root": _layout("/python", "/"),
    "exec from the home dir": _layout("/Users/alice/python", "/opt/py/lib/python3.12"),
}


@pytest.mark.parametrize("name", sorted(UNSAFE))
def test_an_unconfinable_layout_is_refused(name: str) -> None:
    layout = UNSAFE[name]
    assert sandbox.refusal(layout, {HOME}) is not None, name


def test_a_usr_python_is_confined_to_its_own_dirs_not_usr() -> None:
    """/usr/bin/python3 used to make all of /usr readable and executable."""
    layout = _layout("/usr/bin/python3.12", "/usr/lib/python3.12", "/usr/lib/python3/dist-packages")
    assert sandbox.refusal(layout, {Path("/home/alice")}) is None
    argv = sandbox.bwrap_argv(Path("/w"), Path("/t"), layout)
    bound = {argv[i + 1] for i, a in enumerate(argv) if a in ("--bind", "--ro-bind")}
    assert "/usr" not in bound and "/usr/bin" not in bound
    assert "/usr/bin/python3.12" in bound  # the one binary, as a file
    profile = sandbox.macos_profile(Path("/w"), Path("/t"), layout)
    assert '(subpath "/usr")' not in profile and '(subpath "/usr/bin")' not in profile


def test_a_home_rooted_python_falls_back_to_unverified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home_python = _layout(REAL_HOME + "/bin/python", REAL_HOME)
    monkeypatch.setattr(sandbox, "current_layout", lambda: home_python)
    sandbox.detect.cache_clear()
    try:
        box, why = sandbox.detect()
        assert box is None and "cannot be confined" in why
        result = run_generated_tests(_deliverable(tmp_path / "work", []))
        assert result is not None and result.status == "unverified"
    finally:
        sandbox.detect.cache_clear()
