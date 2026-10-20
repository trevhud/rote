"""The shared node_modules trees the slow TS suites install through.

A fake ``npm`` on PATH stands in for the registry. It records each call and
writes a marker naming the dependencies it saw, so the tests can tell which
install a tree came from. ``ensure_npm_install`` itself is real.
"""

from __future__ import annotations

import json
import os
import re
import stat
from pathlib import Path

import pytest

from tests import _node_trees as node_trees

TESTS_DIR = Path(__file__).resolve().parent

_FAKE_NPM = """#!/bin/sh
echo "$PWD $*" >> "{log}"
mkdir -p node_modules/dep
cp package.json node_modules/dep/installed-from.json
"""


@pytest.fixture
def npm_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "npm.log"
    log.touch()
    npm = bin_dir / "npm"
    npm.write_text(_FAKE_NPM.format(log=log), encoding="utf-8")
    npm.chmod(npm.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr(node_trees, "_toolchain", lambda: "node v0\0npm v0")
    monkeypatch.setattr(node_trees, "_root", tmp_path / "trees")
    return log


def _app(root: Path, name: str, deps: dict[str, str]) -> Path:
    app = root / name
    app.mkdir()
    manifest = {
        "name": name,
        "version": "0.1.0",
        "private": True,
        "scripts": {"build": f"tsc -p {name}"},
        "dependencies": deps,
    }
    (app / "package.json").write_text(json.dumps(manifest), encoding="utf-8")
    return app


def _installs(log: Path) -> list[str]:
    return log.read_text(encoding="utf-8").splitlines()


def _installed_deps(app: Path) -> dict[str, str]:
    manifest = json.loads((app / "node_modules" / "dep" / "installed-from.json").read_text())
    deps: dict[str, str] = manifest["dependencies"]
    return deps


def test_apps_with_the_same_dependencies_share_one_install(tmp_path: Path, npm_log: Path) -> None:
    first = _app(tmp_path, "first", {"zod": "^4.0.0"})
    second = _app(tmp_path, "second", {"zod": "^4.0.0"})
    node_trees.install(first)
    node_trees.install(second)
    assert len(_installs(npm_log)) == 1
    assert (first / "node_modules").is_symlink() and (second / "node_modules").is_symlink()
    assert (first / "node_modules").resolve() == (second / "node_modules").resolve()
    assert _installed_deps(second) == {"zod": "^4.0.0"}
    # The install ran in the tree, never in an app.
    [line] = _installs(npm_log)
    assert line.split()[0].startswith(str(tmp_path / "trees"))


def test_a_different_dependency_set_gets_its_own_tree(tmp_path: Path, npm_log: Path) -> None:
    old = _app(tmp_path, "old", {"zod": "^3.0.0"})
    new = _app(tmp_path, "new", {"zod": "^4.0.0"})
    node_trees.install(old)
    node_trees.install(new)
    assert len(_installs(npm_log)) == 2
    assert (old / "node_modules").resolve() != (new / "node_modules").resolve()
    assert _installed_deps(old) == {"zod": "^3.0.0"}
    assert _installed_deps(new) == {"zod": "^4.0.0"}


def test_an_install_script_is_part_of_the_key(tmp_path: Path, npm_log: Path) -> None:
    plain = _app(tmp_path, "plain", {"zod": "^4.0.0"})
    hooked = _app(tmp_path, "hooked", {"zod": "^4.0.0"})
    manifest = json.loads((hooked / "package.json").read_text())
    manifest["scripts"]["postinstall"] = "node fetch-binary.js"
    (hooked / "package.json").write_text(json.dumps(manifest), encoding="utf-8")
    node_trees.install(plain)
    node_trees.install(hooked)
    assert len(_installs(npm_log)) == 2


def test_extra_npm_arguments_are_part_of_the_key(tmp_path: Path, npm_log: Path) -> None:
    plain = _app(tmp_path, "plain", {"inngest": "^4.0.0"})
    with_cli = _app(tmp_path, "with-cli", {"inngest": "^4.0.0"})
    node_trees.install(plain)
    node_trees.install(with_cli, "--ignore-scripts=false", "inngest-cli@^1.0.0")
    installs = _installs(npm_log)
    assert len(installs) == 2
    assert installs[1].endswith(
        "install --no-audit --no-fund --ignore-scripts=false inngest-cli@^1.0.0"
    )


def test_installing_twice_is_a_no_op(tmp_path: Path, npm_log: Path) -> None:
    app = _app(tmp_path, "app", {"zod": "^4.0.0"})
    node_trees.install(app)
    target = (app / "node_modules").readlink()
    node_trees.install(app)
    assert len(_installs(npm_log)) == 1
    assert (app / "node_modules").readlink() == target


def test_a_changed_manifest_relinks_the_app(tmp_path: Path, npm_log: Path) -> None:
    app = _app(tmp_path, "app", {"zod": "^3.0.0"})
    node_trees.install(app)
    manifest = json.loads((app / "package.json").read_text())
    manifest["dependencies"] = {"zod": "^4.0.0"}
    (app / "package.json").write_text(json.dumps(manifest), encoding="utf-8")
    node_trees.install(app)
    assert len(_installs(npm_log)) == 2
    assert _installed_deps(app) == {"zod": "^4.0.0"}


def test_an_interrupted_install_is_rebuilt(tmp_path: Path, npm_log: Path) -> None:
    app = _app(tmp_path, "app", {"zod": "^4.0.0"})
    half_built = tmp_path / "trees" / node_trees.tree_key(app, ())
    (half_built / "node_modules" / "dep").mkdir(parents=True)
    node_trees.install(app)
    assert len(_installs(npm_log)) == 1
    assert _installed_deps(app) == {"zod": "^4.0.0"}


def test_an_app_with_its_own_node_modules_keeps_it(tmp_path: Path, npm_log: Path) -> None:
    """That is ``ensure_npm_install``'s contract, which ``rote run`` relies on."""
    app = _app(tmp_path, "app", {"zod": "^4.0.0"})
    (app / "node_modules").mkdir()
    node_trees.install(app)
    assert _installs(npm_log) == []
    assert not (app / "node_modules").is_symlink()


#: An argv that writes a node_modules: npm install, npm ci, or a generic
#: npm wrapper that forwards ``*args``.
_RAW_NPM_INSTALL = re.compile(r"""["']npm["']\s*,\s*(?:["'](?:install|i|ci|add)["']|\*)""")


def test_every_npm_install_in_the_suites_goes_through_node_trees() -> None:
    """An app's node_modules may be a shared tree, so a raw install into it
    would change what every later test with the same dependencies sees."""
    offenders = [
        f"{path.name}:{number}"
        for path in sorted(TESTS_DIR.glob("*.py"))
        if path.name != "_node_trees.py"
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if _RAW_NPM_INSTALL.search(line)
    ]
    assert offenders == [], f"install through tests/_node_trees.py instead: {offenders}"
