"""Packaging invariants that only show up after a release.

`uvx <dist>` runs the console script named after the distribution, so a
wheel without a script called `rote-cli` makes `uvx rote-cli serve` fail
with "executable not found" even though `uvx --from rote-cli rote` works.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

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
