"""The vendor libraries an emitted app imports, installed where its code can run.

An ``--backend api`` module imports the vendor's SDK
(``google-api-python-client``, ``httpx``...), which is not a dependency of
rote. Two places run that code: verification's generated tests and the
fidelity bench. Both get the same install: the distributions named by a
curated import map, into a target directory keyed by the requirement set
(``<spec cache>/../vendor-deps/<hash>``), added to ``PYTHONPATH``. An import
the map does not know is an error, never a guessed package name.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

from rote.vendor_specs.store import default_root


class VendorDepsError(RuntimeError):
    """The app's vendor libraries could not be resolved or installed."""


#: Import name -> the distribution that provides it, for vendor libraries an
#: emitted module may use that rote itself does not install.
DISTRIBUTIONS = {
    "googleapiclient": "google-api-python-client",
    "google_auth_httplib2": "google-auth-httplib2",
    "google.auth": "google-auth",
    "google.oauth2": "google-auth",
    "httplib2": "httplib2",
    "httpx": "httpx",
    "requests": "requests",
}


def third_party_imports(
    app: Path, dirs: tuple[str, ...] = ("extracted", "signatures")
) -> dict[str, set[Path]]:
    """Every non-stdlib absolute import in the app's ``dirs``, with the files that make it.

    Imports of the app's own packages (``extracted``, ``signatures``) are
    not third party. Imports inside functions count: a lazily imported
    vendor SDK is still needed when that function runs.
    """
    found: dict[str, set[Path]] = {}
    for d in dirs:
        for path in sorted((app / d).glob("*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    names = [node.module]
                for name in names:
                    top = name.split(".")[0]
                    if top in sys.stdlib_module_names or top in ("extracted", "signatures"):
                        continue
                    found.setdefault(name, set()).add(path)
    return found


def distribution_for(name: str) -> str | None:
    """The distribution providing import ``name``, from :data:`DISTRIBUTIONS`, else None."""
    return (
        DISTRIBUTIONS.get(name)
        or DISTRIBUTIONS.get(".".join(name.split(".")[:2]))
        or DISTRIBUTIONS.get(name.split(".")[0])
    )


def vendor_requirements(app: Path) -> list[str]:
    """Distributions the app imports that this interpreter cannot import yet.

    An import that is neither importable here nor in :data:`DISTRIBUTIONS`
    is an error: a guessed distribution name could install the wrong
    package.
    """
    needed: set[str] = set()
    unknown: set[str] = set()
    for name in third_party_imports(app):
        if _importable(name):
            continue
        dist = distribution_for(name)
        if dist is None:
            unknown.add(name)
        else:
            needed.add(dist)
    if unknown:
        raise VendorDepsError(
            f"the app imports {', '.join(sorted(unknown))}, which is not installed here and "
            f"has no known distribution (rote.vendor_specs.deps.DISTRIBUTIONS)"
        )
    return sorted(needed)


def _importable(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def install_vendor_deps(app: Path) -> Path | None:
    """Install the app's missing vendor libraries into a cached target dir.

    Two compiles often install the same set at once (the bench runs two, of
    the same skill). Each installs into its own temporary sibling, marks it
    ``.complete`` there, and renames it into place; the loser of the rename
    discards its copy. So a directory at the key is always a finished
    install, never one being written or one another run deleted.

    The key includes today's date: the newest compatible releases are
    installed again the next day instead of the first install sticking.
    """
    wanted = vendor_requirements(app)
    if not wanted:
        return None
    day = date.today()
    today = day.isoformat()
    key = "\n".join([sys.version, today, *wanted])
    target = default_root().parent / "vendor-deps" / hashlib.sha256(key.encode()).hexdigest()[:16]
    if (target / ".complete").is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f"{target.name}.", dir=target.parent))
    try:
        _uv_install(wanted, staging)
        record = {"python": sys.version, "date": today, "requirements": wanted}
        (staging / ".complete").write_text(json.dumps(record), encoding="utf-8")
        if target.exists() and not (target / ".complete").is_file():
            # Left by a run that died mid-install under the old in-place scheme.
            shutil.rmtree(target, ignore_errors=True)
        try:
            os.replace(staging, target)
        except OSError:
            if not (target / ".complete").is_file():
                raise  # not a lost race: the rename itself failed
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    _prune_older(target, wanted, (day - timedelta(days=1)).isoformat())
    return target


def _prune_older(target: Path, wanted: list[str], keep_from: str) -> None:
    """Remove installs of the same set on this Python dated before ``keep_from``.

    ``keep_from`` is yesterday: a run that started before midnight may still
    import a submodule it has not loaded yet from yesterday's install, and
    deleting the directory would break that import. Anything older belongs
    to no run still in progress.
    """
    for sibling in target.parent.iterdir():
        if sibling == target or not (sibling / ".complete").is_file():
            continue
        try:
            record = json.loads((sibling / ".complete").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (
            isinstance(record, dict)
            and record.get("python") == sys.version
            and record.get("requirements") == wanted
            and str(record.get("date", "")) < keep_from
        ):
            shutil.rmtree(sibling, ignore_errors=True)


def _uv_install(wanted: list[str], target: Path) -> None:
    """``uv pip install --target``: the one place packages are installed."""
    uv = shutil.which("uv")
    if uv is None:
        raise VendorDepsError(f"installing {', '.join(wanted)} for the app needs uv on PATH")
    result = _run_uv(
        [
            uv,
            "pip",
            "install",
            "--quiet",
            "--python",
            sys.executable,
            "--target",
            str(target),
            *wanted,
        ]
    )
    if result.returncode != 0:
        raise VendorDepsError(f"could not install {', '.join(wanted)}: {result.stderr[-2000:]}")


def _run_uv(args: list[str]) -> subprocess.CompletedProcess[str]:
    """The uv subprocess itself (tests replace it: they never install packages)."""
    return subprocess.run(args, capture_output=True, text=True, timeout=600)


__all__ = ["DISTRIBUTIONS", "VendorDepsError", "install_vendor_deps", "vendor_requirements"]
