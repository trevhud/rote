"""One installed ``node_modules`` per distinct dependency set, shared by the slow suites.

Every TypeScript e2e test emits an app and needs its dependencies. Installing
them into each test's own tmp dir cost about 9G of disk per CI run (69M to
302M per app, about 40 apps), which is what exhausted the per-user /tmp quota
on ubuntu-26.04. Most of those trees were identical, because most apps are
the same runtime's ``package.json`` with a different ``name``.

``install`` keys a tree on everything that decides what npm puts on disk: the
dependency fields of ``package.json``, its install lifecycle scripts, a
``package-lock.json`` if the app has one, the extra npm arguments, and the
node and npm versions. A miss runs the real installer
(``rote.runners._node.ensure_npm_install``, the one ``rote run`` uses) in a
directory holding a copy of the app's manifest. Every app with the same key
then gets ``node_modules`` as a symlink to that tree. The app still runs its
own toolchain: ``tsc``, ``wrangler``, ``inngest-cli`` and the SDKs are the
ones installed for its exact dependency set, resolved from its own directory.

Two consequences to know about:

- An extra package that npm saves (``npm install inngest-cli@...``) is
  recorded in the shared tree's ``package.json``, not the app's. Nothing reads
  it back from the app, and it keeps a hit and a miss indistinguishable.
- A tree is shared mutable state. A test must not ``npm install`` into an app
  whose ``node_modules`` came from here, which is why every install in the
  suites goes through this module (``tests/test_node_trees.py`` enforces it).

The trees live under pytest's basetemp (set by ``tests/conftest.py``), so
they last one session and pytest's retention cleans them up.
"""

from __future__ import annotations

import functools
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from rote.runners import _node

#: Captured at import, before conftest swaps the runners' copies for
#: ``install``, so a miss always reaches the real installer.
_real_install = _node.ensure_npm_install

#: package.json fields that do not change what npm installs.
_INERT_FIELDS = frozenset({"name", "version", "description", "private"})

#: Lifecycle scripts npm runs during an install. Every other script
#: (build, start, typecheck) is inert for the tree.
_INSTALL_SCRIPTS = frozenset(
    {"preinstall", "install", "postinstall", "prepare", "preprepare", "postprepare"}
)

#: Written last, so a tree whose install died part way is rebuilt.
_COMPLETE = ".complete"

_root: Path | None = None


def set_root(root: Path) -> None:
    global _root
    _root = root


@functools.cache
def _toolchain() -> str:
    versions = [
        subprocess.run([tool, "--version"], capture_output=True, text=True, check=True).stdout
        for tool in ("node", "npm")
    ]
    return "\0".join(v.strip() for v in versions)


def tree_key(app_dir: Path, extra_args: tuple[str, ...]) -> str:
    manifest = json.loads((app_dir / "package.json").read_text(encoding="utf-8"))
    for field in _INERT_FIELDS:
        manifest.pop(field, None)
    scripts = {k: v for k, v in manifest.pop("scripts", {}).items() if k in _INSTALL_SCRIPTS}
    if scripts:
        manifest["scripts"] = scripts
    lock = app_dir / "package-lock.json"
    digest = hashlib.sha256()
    for part in (
        json.dumps(manifest, sort_keys=True).encode(),
        lock.read_bytes() if lock.exists() else b"",
        "\0".join(extra_args).encode(),
        _toolchain().encode(),
    ):
        digest.update(part)
        digest.update(b"\0\0")
    return digest.hexdigest()[:16]


def install(app_dir: Path, *extra_args: str) -> None:
    """Give ``app_dir`` a ``node_modules`` installed from its own dependencies.

    Same signature as ``ensure_npm_install``, so conftest can hand it to the
    runners. An app that already has a real ``node_modules`` directory is
    left to the real installer, which is what ``rote run`` would do with it.
    """
    if _root is None:
        raise RuntimeError("tests/conftest.py sets the node-tree root for the session")
    modules = app_dir / "node_modules"
    if modules.exists() and not modules.is_symlink():
        _real_install(app_dir, *extra_args)
        return
    tree = _root / tree_key(app_dir, extra_args)
    if modules.is_symlink():
        modules.unlink()
    if not (tree / _COMPLETE).exists():
        _build(app_dir, tree, extra_args)
    modules.symlink_to(tree / "node_modules", target_is_directory=True)


def _build(app_dir: Path, tree: Path, extra_args: tuple[str, ...]) -> None:
    if tree.exists():
        shutil.rmtree(tree)
    tree.mkdir(parents=True)
    for name in ("package.json", "package-lock.json"):
        if (app_dir / name).exists():
            shutil.copy2(app_dir / name, tree / name)
    _real_install(tree, *extra_args)
    (tree / _COMPLETE).write_text(f"first installed for {app_dir}\n", encoding="utf-8")
