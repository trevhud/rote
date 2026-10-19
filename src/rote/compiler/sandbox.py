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
  the interpreter's own files (:class:`Layout`: stdlib, site-packages,
  binary, libpython), system libraries, and a few devices. Writes are
  allowed under the work dir and a private temp dir. Exec is allowed for
  the interpreter binary only. There is no network. A layout that would
  expose a home or rote's own data is refused (:func:`refusal`) rather
  than loosened, so the tests are skipped as unverified.
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


@dataclass(frozen=True)
class Layout:
    """Exactly what a sandboxed interpreter may read and execute.

    Derived from the running interpreter (:func:`current_layout`), never
    from "the directory above its binary": a Python copied to ``~/bin``
    would make that the whole home, and ``/usr/bin/python3`` would make it
    all of ``/usr``. Directories are read recursively; files are single
    literals.
    """

    executable: Path
    read_dirs: tuple[Path, ...]
    read_files: tuple[Path, ...] = ()
    exec_files: tuple[Path, ...] = ()

    @property
    def exec_paths(self) -> tuple[Path, ...]:
        return (self.executable, *self.exec_files)


def _site_dirs() -> list[Path]:
    return sorted(
        {
            Path(e).resolve()
            for e in sys.path
            if e and ("site-packages" in e or "dist-packages" in e) and Path(e).is_dir()
        }
    )


def current_layout() -> Layout:
    """The running interpreter's layout: binary, stdlib, libpython, site dirs."""
    executable = Path(os.path.realpath(sys.executable))
    stdlib = Path(sysconfig.get_paths()["stdlib"]).resolve()
    read_dirs = {stdlib, *_site_dirs()}
    # The sandboxed child is launched by this resolved path (see
    # :func:`launch_executable`), never through a venv or symlink, so no
    # venv's pyvenv.cfg and no symlinked install path need to exist inside.
    read_files = {executable}
    exec_files: set[Path] = set()
    ldlibrary = sysconfig.get_config_var("LDLIBRARY") or ""
    framework_prefix = sysconfig.get_config_var("PYTHONFRAMEWORKPREFIX") or ""
    libdir = sysconfig.get_config_var("LIBDIR") or ""
    if framework_prefix:
        # macOS framework build (Homebrew, python.org): libpython is the
        # framework binary, and bin/python3.x is a stub that posix_spawns
        # Resources/Python.app/Contents/MacOS/Python.
        read_files.add(Path(framework_prefix, ldlibrary).resolve())
        app = Path(framework_prefix, ldlibrary).resolve().parent / "Resources" / "Python.app"
        if app.is_dir():
            read_dirs.add(app)
            exec_files.add(app / "Contents" / "MacOS" / "Python")
    elif libdir and ldlibrary and Path(libdir, ldlibrary).exists():
        # libpython3.x.so is a symlink to the soname the loader asks for.
        for name in {ldlibrary, sysconfig.get_config_var("INSTSONAME") or ldlibrary}:
            lib = Path(libdir, name)
            if lib.exists():
                read_files.update({Path(os.path.abspath(lib)), lib.resolve()})
    return Layout(
        executable=executable,
        read_dirs=tuple(sorted(read_dirs)),
        read_files=tuple(sorted(read_files)),
        exec_files=tuple(sorted(exec_files)),
    )


def launch_executable() -> str:
    """The interpreter path a sandboxed run is started with: the real binary.

    Not ``sys.executable``. Launched through a venv, Python reads the venv's
    ``pyvenv.cfg`` and finds its stdlib through the ``home`` recorded there,
    and uv records a symlinked install path (``cpython-3.11-linux-...``, not
    the versioned dir it resolves to). That path does not exist inside a bwrap
    namespace, so every sandboxed run on a uv-managed Python died with "No
    module named 'encodings'". The real binary finds its stdlib next to
    itself, and the child runs with ``-S`` and an explicit PYTHONPATH, so it
    needs nothing from the venv.
    """
    return str(current_layout().executable)


def _homes() -> set[Path]:
    import pwd

    homes = {Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()}
    if os.environ.get("HOME"):
        homes.add(Path(os.environ["HOME"]).resolve())
    return homes


def refusal(layout: Layout, homes: set[Path] | None = None) -> str | None:
    """Why ``layout`` must not be sandboxed as is, or None if it is safe.

    A directory rule grants everything under it, so a directory that is
    ``/``, a home, an ancestor of a home, or one that holds rote's data,
    rote's config, or ``~/.ssh`` would hand the sandboxed tests exactly
    what the sandbox exists to withhold. That is not narrowed silently; it
    is "no sandbox", so the tests are skipped as unverified.
    """
    homes = homes if homes is not None else _homes()
    sensitive = {
        sub
        for home in homes
        for sub in (home / ".local" / "share" / "rote", home / ".config" / "rote", home / ".ssh")
    }
    for root in (*layout.read_dirs, *(p.parent for p in layout.exec_paths)):
        # Lexically normalized: a "~/.ssh/.." root is the home, whatever it says.
        root = Path(os.path.normpath(root))
        if root == Path("/"):
            return f"{root} would be readable inside the sandbox"
        for home in homes:
            if home == root or home.is_relative_to(root):
                return f"{root} would expose the home directory {home} inside the sandbox"
        for path in sensitive:
            if path.is_relative_to(root):
                return f"{root} would expose {path} inside the sandbox"
    return None


def import_path(layout: Layout | None = None) -> list[str]:
    """The ``sys.path`` entries the sandboxed run gets via PYTHONPATH.

    The child runs with ``-S`` (no site processing), so ``.pth`` files,
    which can point anywhere and run code, are never read. The
    site-packages dirs that hold pytest are passed explicitly instead.
    """
    return [str(p) for p in _site_dirs()]


def _sbpl(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def macos_profile(work_dir: Path, tmp_dir: Path, layout: Layout) -> str:
    """The ``sandbox-exec`` profile (SBPL) for one generated-test run."""
    dirs = [*map(str, layout.read_dirs), *_MACOS_SYSTEM_READ]
    files = [*map(str, layout.read_files), "/", *_MACOS_DEVICES]
    lines = [
        "(version 1)",
        "(deny default)",
        "(allow process-fork)",
        "(allow process-exec "
        + " ".join(f"(literal {_sbpl(str(p))})" for p in layout.exec_paths)
        + ")",
        # Path resolution stats every parent directory. Metadata only:
        # contents and listings of anything not allowed below stay denied.
        "(allow file-read-metadata)",
        "(allow file-read* "
        + " ".join(f"(subpath {_sbpl(d)})" for d in dirs)
        + " "
        + " ".join(f"(literal {_sbpl(f)})" for f in files)
        + ")",
        f"(allow file-read* file-write* (subpath {_sbpl(str(work_dir.resolve()))})"
        f" (subpath {_sbpl(str(tmp_dir.resolve()))}))",
        '(allow file-write-data (literal "/dev/null"))',
        "(allow sysctl-read)",
        "(allow signal (target self))",
    ]
    return "\n".join(lines) + "\n"


def bwrap_argv(work_dir: Path, tmp_dir: Path, layout: Layout) -> list[str]:
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
    for path in (*layout.read_dirs, *layout.read_files, *layout.exec_files):
        argv += ["--ro-bind", str(path), str(path)]
    argv += ["--bind", str(work_dir), str(work_dir)]
    argv += ["--bind", str(tmp_dir), str(tmp_dir), "--chdir", str(work_dir)]
    return argv


def wrap(
    sandbox: Sandbox, argv: list[str], work_dir: Path, tmp_dir: Path, env: dict[str, str]
) -> tuple[list[str], dict[str, str]]:
    """``(argv, env)`` that run ``argv`` inside ``sandbox``."""
    layout = current_layout()
    if sandbox.kind == "sandbox-exec":
        profile = tmp_dir / "sandbox.sb"
        profile.write_text(macos_profile(work_dir, tmp_dir, layout), encoding="utf-8")
        return [sandbox.executable, "-f", str(profile), *argv], env
    # bwrap clears the environment; pass each variable explicitly.
    setenv: list[str] = []
    for key, value in env.items():
        setenv += ["--setenv", key, value]
    return [sandbox.executable, *bwrap_argv(work_dir, tmp_dir, layout), *setenv, "--", *argv], {}


def _probe(argv: list[str]) -> bool:
    try:
        return subprocess.run(argv, capture_output=True, timeout=20).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


@functools.cache
def detect() -> tuple[Sandbox | None, str]:
    """The sandbox to use here, or ``(None, why there is none)``."""
    unsafe = refusal(current_layout())
    if unsafe is not None:
        return None, f"this Python's install cannot be confined: {unsafe}"
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
            profile = macos_profile(work, scratch, current_layout())
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
