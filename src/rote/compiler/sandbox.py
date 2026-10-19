"""An OS sandbox for the agent-written tests ``rote compile`` runs.

The tests are code an LLM wrote from a skill that may be third-party
(invariant #7), and rote runs them automatically on the user's machine.
An environment allowlist keeps secrets out of ``os.environ``, but a
review showed how little that covers: ``pwd.getpwuid(os.getuid())``
still names the real home, so ``~/.ssh``, ``~/.netrc`` and rote's own
``cloud.json`` are one absolute path away, the network is open, and the
ssh-agent socket and the Keychain CLI are reachable. So the tests run
under the platform's sandbox, or not at all:

* **macOS**: ``sandbox-exec`` with a generated profile (:func:`macos_profile`).
  Everything is denied by default. Reads are allowed for the work dir,
  the interpreter's install and the import paths pytest needs, system
  libraries, and a few devices. Writes are allowed under the work dir and
  a private temp dir. Exec is allowed for the interpreter only. There is
  no network.
* **Linux**: ``bwrap`` (bubblewrap) with ``--unshare-all`` (network
  included), read-only binds for the system library dirs and the
  interpreter's paths, a bind for the work dir, a tmpfs ``/tmp``, and no
  home: the user's files are not in the namespace at all.

What is NOT isolated, on purpose or by limitation: file metadata
(``stat``) outside the allowed trees on macOS, since path resolution
needs it (file contents and directory listings stay denied); CPU and
memory (only the wall-clock timeout bounds them); and on Linux, the
system library dirs are readable (they hold no user data).

:func:`detect` probes for a working sandbox once per process. A
sandbox-exec deprecated past usefulness, or a kernel that refuses
unprivileged user namespaces, both read as "no sandbox", and the caller
then skips the tests unless the user passed ``--trust-skill``.
"""

from __future__ import annotations

import functools
import os
import shutil
import subprocess
import sys
import sysconfig
from dataclasses import dataclass
from pathlib import Path

#: macOS system trees a sandboxed interpreter must read to start at all:
#: the dyld shared cache, system frameworks, libc and friends.
_MACOS_SYSTEM_READ = (
    "/System",
    "/usr/lib",
    "/usr/share",
    "/private/var/db/dyld",
    "/Library/Apple/System",
)
_MACOS_DEVICES = ("/dev/null", "/dev/zero", "/dev/random", "/dev/urandom", "/dev/dtracehelper")

#: Linux system dirs bound read-only. User data lives elsewhere (/home,
#: /root, /run/user), and none of those are bound.
_LINUX_SYSTEM_READ = ("/usr/lib", "/usr/lib64", "/lib", "/lib64", "/etc/ld.so.cache")


@dataclass(frozen=True)
class Sandbox:
    """A sandbox that works on this machine."""

    kind: str  # "sandbox-exec" | "bwrap"
    executable: str


def interpreter_paths() -> list[Path]:
    """What a sandboxed ``python -m pytest`` must be able to read.

    The interpreter binary (through symlinks), its standard library, and
    the directories pytest and its dependencies import from, resolved from
    this process rather than guessed.
    """
    paths = {
        Path(os.path.realpath(sys.executable)).parent.parent,
        Path(sys.base_prefix).resolve(),
        Path(sys.prefix).resolve(),
        Path(sysconfig.get_paths()["stdlib"]).resolve(),
    }
    import pytest

    paths.add(Path(pytest.__file__).resolve().parent.parent)
    for entry in sys.path:
        if entry and Path(entry).is_dir() and "site-packages" in entry:
            paths.add(Path(entry).resolve())
    return sorted(paths)


def import_path() -> list[str]:
    """The ``sys.path`` entries the sandboxed run gets via PYTHONPATH.

    The child runs with ``-S`` (no site processing), so ``.pth`` files,
    which can point anywhere and run code, are never read. The
    site-packages dirs that hold pytest are passed explicitly instead.
    """
    return sorted(
        {
            str(Path(e).resolve())
            for e in sys.path
            if e and "site-packages" in e and Path(e).is_dir()
        }
    )


def _sbpl(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def macos_profile(work_dir: Path, tmp_dir: Path, readable: list[Path]) -> str:
    """The ``sandbox-exec`` profile (SBPL) for one generated-test run."""
    python = os.path.realpath(sys.executable)
    reads = [*map(str, readable), *_MACOS_SYSTEM_READ]
    lines = [
        "(version 1)",
        "(deny default)",
        "(allow process-fork)",
        f"(allow process-exec (literal {_sbpl(python)}))",
        # Path resolution stats every parent directory. Metadata only:
        # contents and listings of anything not allowed below stay denied.
        "(allow file-read-metadata)",
        "(allow file-read* "
        + " ".join(f"(subpath {_sbpl(str(Path(p).resolve()))})" for p in reads)
        + " "
        + " ".join(f"(literal {_sbpl(d)})" for d in ("/", *_MACOS_DEVICES))
        + ")",
        f"(allow file-read* file-write* (subpath {_sbpl(str(work_dir.resolve()))})"
        f" (subpath {_sbpl(str(tmp_dir.resolve()))}))",
        '(allow file-write-data (literal "/dev/null"))',
        "(allow sysctl-read)",
        "(allow signal (target self))",
    ]
    return "\n".join(lines) + "\n"


def bwrap_argv(work_dir: Path, tmp_dir: Path, readable: list[Path]) -> list[str]:
    """The ``bwrap`` prefix for one generated-test run."""
    argv = [
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--clearenv",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
    ]
    for system_dir in _LINUX_SYSTEM_READ:
        argv += ["--ro-bind-try", system_dir, system_dir]
    for path in readable:
        argv += ["--ro-bind", str(path), str(path)]
    argv += ["--bind", str(work_dir), str(work_dir)]
    argv += ["--bind", str(tmp_dir), str(tmp_dir), "--chdir", str(work_dir)]
    return argv


def wrap(
    sandbox: Sandbox, argv: list[str], work_dir: Path, tmp_dir: Path, env: dict[str, str]
) -> tuple[list[str], dict[str, str]]:
    """``(argv, env)`` that run ``argv`` inside ``sandbox``."""
    readable = interpreter_paths()
    if sandbox.kind == "sandbox-exec":
        profile = tmp_dir / "sandbox.sb"
        profile.write_text(macos_profile(work_dir, tmp_dir, readable), encoding="utf-8")
        return [sandbox.executable, "-f", str(profile), *argv], env
    # bwrap clears the environment; pass each variable explicitly.
    setenv: list[str] = []
    for key, value in env.items():
        setenv += ["--setenv", key, value]
    return [sandbox.executable, *bwrap_argv(work_dir, tmp_dir, readable), *setenv, "--", *argv], {}


def _probe(argv: list[str]) -> bool:
    try:
        return subprocess.run(argv, capture_output=True, timeout=20).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


@functools.cache
def detect() -> tuple[Sandbox | None, str]:
    """The sandbox to use here, or ``(None, why there is none)``."""
    if sys.platform == "darwin":
        exe = shutil.which("sandbox-exec") or "/usr/bin/sandbox-exec"
        if not Path(exe).is_file():
            return None, "sandbox-exec is not installed"
        # Probe with the real profile and the real interpreter: a profile
        # the OS no longer honors must read as "no sandbox", not as a pass.
        import tempfile

        with tempfile.TemporaryDirectory(prefix="rote-sandbox-probe-") as tmp:
            work, scratch = Path(tmp) / "work", Path(tmp) / "tmp"
            work.mkdir()
            scratch.mkdir()
            profile = macos_profile(work, scratch, interpreter_paths())
            ok = _probe([exe, "-p", profile, os.path.realpath(sys.executable), "-S", "-c", "pass"])
        if not ok:
            return (
                None,
                "sandbox-exec is present but cannot run the interpreter under rote's profile",
            )
        return Sandbox("sandbox-exec", exe), ""
    if sys.platform.startswith("linux"):
        exe = shutil.which("bwrap")
        if exe is None:
            return None, "bubblewrap (bwrap) is not installed"
        if not _probe([exe, "--unshare-all", "--ro-bind", "/", "/", "true"]):
            return None, (
                "bwrap is installed but cannot create namespaces here (unprivileged user "
                "namespaces may be disabled, e.g. kernel.apparmor_restrict_unprivileged_userns)"
            )
        return Sandbox("bwrap", exe), ""
    return None, f"no supported sandbox on {sys.platform}"
