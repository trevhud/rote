"""The suite's process environment: no inference, no real per-user state.

Applied once, in ``pytest_configure``, to ``os.environ`` itself, before any
test is collected. It used to be two per-test monkeypatches (``shutil.which``
wrapped, state paths set per test), and both leaked:

* ``monkeypatch.undo()`` inside a test removed them, after which
  ``shutil.which("claude")`` found the real CLI and ``ROTE_APPS_PATH`` was
  unset again.
* A patched ``shutil.which`` never reached a child process. A test that ran
  ``python -m rote compile`` as a subprocess (``compile_for_eval``) inherited
  the real PATH, found ``claude``, compiled for real ($0.29) and wrote into
  the developer's real app registry.

So the PATH itself no longer leads to a subscription CLI. Each PATH entry
that holds ``claude`` or ``codex`` is replaced by a session-owned mirror of
that directory: a symlink to every other entry, and nothing for the two CLIs.
The mirrors are read-only, so a test cannot drop a ``claude`` link into one
and reach every test after it.
A mirror rather than dropping the entry, because on a developer machine the
same directory (``~/.local/bin``) holds other tools; and not a failing stub,
because a stub makes ``which("claude")`` succeed, and rote's own code reads
"found" as "available" (auto-detect picks the ``claude-cli`` lane, the
Claude driver reports itself ready), which changes the code path under test.

A test-written stub passed by absolute path still runs: only bare-name
lookups consult PATH.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Mapping, MutableMapping
from pathlib import Path

from rote import isolation

#: Subscription-auth CLIs. These are the dangerous ones: ``claude -p`` needs
#: no API key at all, so finding the binary is enough to spend the
#: developer's subscription, with nothing in the environment to scrub.
SUBSCRIPTION_CLIS = ("claude", "codex")

#: Credentials that would let a test reach a real model.
INFERENCE_CREDENTIAL_VARS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "ROTE_INFERENCE",
    "ROTE_CLOUD_TOKEN",
    # The decision models: with these set, the default decider (auto) would
    # call Clef or Jev from any compile test.
    "ROTE_CLEF_ACCOUNT_ID",
    "ROTE_CLEF_API_TOKEN",
    "TYPESAFE_API_KEY",
)

#: The XDG base dirs, relative to the session state dir. HOME stays: npm
#: and docker keep their caches and logins under it, and every rote path
#: that read the home directly now goes through an XDG dir.
XDG_DIRS = {
    "XDG_CONFIG_HOME": "xdg/config",
    "XDG_DATA_HOME": "xdg/data",
    "XDG_CACHE_HOME": "xdg/cache",
    "XDG_STATE_HOME": "xdg/state",
}

#: The real lookup, captured before any test can patch it. The per-test
#: check must ask the filesystem, not whatever a test put in its place.
_REAL_WHICH = shutil.which


def _holds_subscription_cli(directory: Path) -> bool:
    return any(
        (directory / name).is_file() and os.access(directory / name, os.X_OK)
        for name in SUBSCRIPTION_CLIS
    )


_READ_ONLY_DIR = 0o555
_MIRRORS = "path-mirrors"


def scrub_path(path: str, mirror_root: Path) -> str:
    """``path`` with every entry holding a subscription CLI replaced by a mirror.

    The mirror holds a symlink to each of the entry's other children, so
    node, uv or anything else installed beside ``claude`` still resolves.
    """
    mirrors: dict[Path, Path] = {}
    entries = []
    for entry in path.split(os.pathsep):
        directory = Path(entry)
        if not entry or not directory.is_dir() or not _holds_subscription_cli(directory):
            entries.append(entry)
            continue
        real = directory.resolve()
        if real not in mirrors:
            mirror = mirror_root / str(len(mirrors))
            mirror.mkdir(parents=True)
            for child in sorted(real.iterdir()):
                if child.name not in SUBSCRIPTION_CLIS:
                    (mirror / child.name).symlink_to(child)
            mirror.chmod(_READ_ONLY_DIR)
            mirrors[real] = mirror
        entries.append(str(mirrors[real]))
    return os.pathsep.join(entries)


def isolate(environ: MutableMapping[str, str], state_dir: Path) -> None:
    """Point every rote state path under ``state_dir``, drop every credential
    and config default, and take the subscription CLIs off PATH, in ``environ``."""
    from rote.config import CONFIG_KEYS

    environ.update(isolation.isolated_env(state_dir, base={}))
    for key in (
        *isolation.CLOUD_CREDENTIAL_ENV,
        *INFERENCE_CREDENTIAL_VARS,
        # The config layers' env twins (ROTE_RUNTIME, ROTE_WORK_DOMAINS, ...):
        # a developer's exported default must not change what a test resolves.
        *(config_key.env for config_key in CONFIG_KEYS),
    ):
        environ.pop(key, None)
    # The XDG dirs too: every rote default path (and the TS helpers' and a
    # compiled skill's driver's own copies of that rule) falls back to them
    # when a ROTE_* override is missing, as it is after a test deletes one
    # or in a child process that drops the ROTE_* variables.
    # uv keeps its Python installs and cache under XDG_DATA_HOME and
    # XDG_CACHE_HOME, so pin those to where uv finds them now, first: an
    # installed Python 3.10 vanishing turned test_helper_on_oldest_python
    # into a skip. Toolchain state, not rote state, so it stays shared.
    environ.update(_uv_dirs(environ))
    for var, rel in XDG_DIRS.items():
        environ[var] = str(state_dir / rel)
    # The Codex skill-run lane links the user's Codex login into each run: a
    # test must never find the real one, so Codex's home is an empty dir.
    environ["CODEX_HOME"] = str(state_dir / "codex-home")
    environ["PATH"] = scrub_path(environ.get("PATH", ""), state_dir / _MIRRORS)


#: uv's own overrides, and the ``uv`` subcommand that prints each one's value.
_UV_DIRS = {"UV_PYTHON_INSTALL_DIR": ("python", "dir"), "UV_CACHE_DIR": ("cache", "dir")}


def _uv_dirs(environ: Mapping[str, str]) -> dict[str, str]:
    """``{UV_*: dir}`` for each uv dir the environment does not already pin."""
    uv = _REAL_WHICH("uv", path=environ.get("PATH", ""))
    if uv is None:
        return {}
    return {
        var: subprocess.run(
            [uv, *argv], env=dict(environ), capture_output=True, text=True, check=True
        ).stdout.strip()
        for var, argv in _UV_DIRS.items()
        if not environ.get(var)
    }


def release_mirrors(state_dir: Path) -> None:
    """Make the read-only mirrors writable again, so the session dir can be removed."""
    root = state_dir / _MIRRORS
    if root.is_dir():
        for mirror in root.iterdir():
            mirror.chmod(0o755)


def found_in_process(env: Mapping[str, str]) -> list[str]:
    """Where an in-process bare-name lookup finds each subscription CLI."""
    path = env.get("PATH", "")
    return [hit for name in SUBSCRIPTION_CLIS if (hit := _REAL_WHICH(name, path=path))]


#: PATH values a child shell has already been shown to find nothing on. A
#: shell resolves a bare name from PATH and the filesystem, exactly as the
#: uncached in-process lookup does, so that lookup covers a file appearing
#: later on an unchanged PATH, and a shell (about 10 ms) runs only for a
#: PATH not seen before, instead of twice per test.
_CLEAN_SUBPROCESS_PATHS: set[str] = set()


def found_in_subprocess(env: Mapping[str, str]) -> list[str]:
    """Where a child shell, given ``env``, finds each subscription CLI."""
    path = env.get("PATH", "")
    if path in _CLEAN_SUBPROCESS_PATHS:
        return []
    script = "; ".join(f"command -v {name}" for name in SUBSCRIPTION_CLIS) + "; true"
    out = subprocess.run(
        ["/bin/sh", "-c", script], env=dict(env), capture_output=True, text=True, check=True
    ).stdout
    hits = out.split()
    if not hits:
        _CLEAN_SUBPROCESS_PATHS.add(path)
    return hits


def reachable_clis(env: Mapping[str, str]) -> list[str]:
    """Every way a bare ``claude`` / ``codex`` resolves under ``env``."""
    return [
        *(f"shutil.which: {hit}" for hit in found_in_process(env)),
        *(f"subprocess command -v: {hit}" for hit in found_in_subprocess(env)),
    ]
