"""Packaging invariants that only show up after a release.

`uvx <dist>` runs the console script named after the distribution, so a
wheel without a script called `rote-cli` makes `uvx rote-cli serve` fail
with "executable not found" even though `uvx --from rote-cli rote` works.
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import pytest

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _project() -> dict[str, object]:
    with PYPROJECT.open("rb") as f:
        project = tomllib.load(f)["project"]
    assert isinstance(project, dict)
    return project


def test_distribution_name_is_a_console_script() -> None:
    project = _project()
    scripts = project["scripts"]
    assert isinstance(scripts, dict)
    assert scripts[project["name"]] == "rote.cli:main"


def test_rote_command_still_exists() -> None:
    scripts = _project()["scripts"]
    assert isinstance(scripts, dict)
    assert scripts["rote"] == "rote.cli:main"


def test_no_source_file_compiles_with_a_warning() -> None:
    """An invalid escape in a template string (`'\\/'` inside a regex in the
    emitted TS) warns on every fresh import and becomes a SyntaxError in a
    future Python. Compile every module with warnings as errors."""
    import warnings

    src = PYPROJECT.parent / "src" / "rote"
    for path in sorted(src.rglob("*.py")):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            compile(path.read_text(encoding="utf-8"), str(path), "exec")


def test_the_package_requires_pytest_without_any_extra() -> None:
    """rote compile runs its agent's tests. With pytest only in the dev
    extra, every plain `pip install rote-cli` silently skipped them, and a
    real compile exited 0 while a generated test failed.

    Read the installed distribution's metadata, which the build backend
    generated from pyproject.toml exactly as it does for the wheel: the
    requirement must be unconditional, not gated on `extra == ...`.
    """
    from importlib.metadata import requires

    unconditional = [r for r in requires("rote-cli") or [] if "extra ==" not in r]
    assert any(re.match(r"pytest\s*[<>=!~]", r) for r in unconditional), unconditional
    deps = _project()["dependencies"]
    assert isinstance(deps, list) and any(d.startswith("pytest") for d in deps)


def test_dbos_and_fastmcp_are_base_requirements_and_their_extras_are_empty() -> None:
    """The compile default (dbos) and the first-run demo (dbos plus MCP)
    failed on a plain `pip install rote-cli` while both were extras.

    Read the installed metadata, as the wheel carries it: dbos and fastmcp
    must be unconditional, and `[dbos]`, `[mcp]`, `[serve]` must still be
    declared extras (an old command resolves) that add nothing."""
    from importlib.metadata import metadata, requires

    from rote.extras import ALIAS_EXTRAS

    reqs = requires("rote-cli") or []
    unconditional = {
        re.split(r"[<>=!~\[ ;]", r, maxsplit=1)[0] for r in reqs if "extra ==" not in r
    }
    assert {"dbos", "fastmcp", "httpx"} <= unconditional, sorted(unconditional)
    provided = set(metadata("rote-cli").get_all("Provides-Extra") or [])
    assert set(ALIAS_EXTRAS) <= provided
    for alias in ALIAS_EXTRAS:
        assert not [r for r in reqs if f'extra == "{alias}"' in r], alias


def test_base_dependencies_import() -> None:
    """A base dependency is never skipped around in this suite; it must import."""
    import importlib

    from rote.extras import BASE_MODULES

    for module in BASE_MODULES:
        importlib.import_module(module)


@pytest.mark.slow
def test_an_installed_wheel_resolves_the_demo_inside_itself(tmp_path: Path) -> None:
    """`rote try` reads the demo from the install, not a checkout. Build
    the real wheel, unpack it as site-packages would hold it, and resolve
    the demo from a fresh interpreter that imports rote from there."""
    import subprocess
    import sys
    import zipfile

    from tests.test_examples_command import _build_wheel

    _build_wheel(tmp_path / "dist")
    (wheel,) = (tmp_path / "dist").glob("rote_cli-*.whl")
    site = tmp_path / "site"
    zipfile.ZipFile(wheel).extractall(site)
    probe = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "import json, rote.demo as d; r = d.load_reference(); "
        "print(json.dumps([str(d.docs_digest_dir()), str(d.pipeline_path()), "
        "(d.skill_dir() / 'input.json').is_file(), len(r.fields)]))"
    )
    done = subprocess.run(
        [sys.executable, "-c", probe, str(site)],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=tmp_path,
    )
    assert done.returncode == 0, done.stderr
    demo_dir, pipeline, has_input, fields = json.loads(done.stdout)
    assert Path(demo_dir) == site / "rote" / "examples" / "docs-digest"
    assert Path(pipeline).is_file() and Path(pipeline).is_relative_to(site)
    assert has_input is True and fields == 4
