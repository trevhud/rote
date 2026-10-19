"""Bundled examples: ``rote example [<name>]``.

The free first run should not need a clone. pip and uv installs get no
``examples/`` directory, so the examples that run end to end with no
credentials ship inside the wheel. The files have one home, the repo's
``examples/<name>/``: ``pyproject.toml`` force-includes the paths listed
in :data:`BUNDLED` into the wheel at ``rote/examples/<name>/`` (a test
pins that the two lists agree), the same arrangement as the compiler's
skill bundle.

Only examples whose extracted modules are real code belong here. The
others stop at ``NotImplementedError`` stubs until someone writes a
vendor client, which is not a first run.
"""

from __future__ import annotations

import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Example:
    """One bundled example: what it is, what ships, and how to run it."""

    name: str
    summary: str
    #: Paths under ``examples/<name>/`` that ship in the wheel and that
    #: ``rote example <name>`` copies. Directories are copied whole.
    files: tuple[str, ...]
    #: Commands to print after copying, with ``{dir}`` for the copy's path.
    #: Each is ``(comment, command)``.
    commands: tuple[tuple[str, str], ...]
    #: ``(repo text, copy text)`` replacements applied to README.md in the
    #: copy, in order, with ``{dir}`` for the copy's path. The README is
    #: written for the repo (``examples/<name>/`` paths, ``/tmp`` app
    #: dirs); the copy's version must name the copy's own paths. A test
    #: requires every repo text to occur, so a README edit cannot make a
    #: rewrite silently stop applying.
    readme_rewrites: tuple[tuple[str, str], ...] = ()


_REVIEW = (
    '\'{"reviewer": "Priya Shah", "approved_ids": ["EXP-2403", "EXP-2413", '
    '"EXP-2420"], "rejected_ids": ["EXP-2410"]}\''
)

BUNDLED: dict[str, Example] = {
    "expense-audit": Example(
        name="expense-audit",
        summary=(
            "Month-end expense audit: FX conversion, duplicate and late checks, "
            "policy caps, a fan_out per employee and a controller gate. Working "
            "code, no credentials, no tokens."
        ),
        files=(
            "README.md",
            "input.json",
            "eval-input.json",
            "skill",
            "expected/pipeline.yaml",
            "expected/pipeline-draft.yaml",
            "expected/eval.yaml",
            "expected/extracted",
            "scorecard.md",
            "runs",
        ),
        commands=(
            (
                "Plain Python, no gate: a draft report",
                "rote emit {dir}/expected/pipeline-draft.yaml --runtime python "
                "--out {dir}/app-python",
            ),
            ("", "rote run {dir}/app-python --input {dir}/input.json"),
            (
                "DBOS, with the controller's decision for the parked gate (needs the dbos extra)",
                "rote emit {dir}/expected/pipeline.yaml --runtime dbos --out {dir}/app-dbos",
            ),
            (
                "",
                "rote run {dir}/app-dbos --input {dir}/input.json "
                "--signal controller_review=" + _REVIEW,
            ),
            (
                "The before/after scorecard (fetches live model prices, no inference)",
                "rote eval {dir}/expected/pipeline.yaml",
            ),
        ),
        readme_rewrites=(
            (
                "From the repo root, after",
                "From the directory you ran `rote example` in, after",
            ),
            (
                "(`tests/test_expense_audit_example.py`)",
                "(`tests/test_expense_audit_example.py` in the rote repo)",
            ),
            ("/tmp/expense-audit-dbos", "{dir}/app-dbos"),
            ("/tmp/expense-audit-temporal", "{dir}/app-temporal"),
            ("/tmp/expense-audit", "{dir}/app-python"),
            ("examples/expense-audit/", "{dir}/"),
        ),
    ),
}


class ExampleError(RuntimeError):
    """A bundled example could not be found or copied."""


def examples_root() -> Path:
    """Where the bundled examples live in this install.

    A wheel carries them at ``rote/examples/``. An editable install or a
    source checkout reads the repo's ``examples/`` directly.
    """
    import rote

    package = Path(rote.__file__).resolve().parent
    for candidate in (package / "examples", package.parent.parent / "examples"):
        if all((candidate / name).is_dir() for name in BUNDLED):
            return candidate
    raise ExampleError(
        "the bundled examples are missing from this install; reinstall rote-cli, "
        "or clone https://github.com/trevhud/rote and use examples/ there"
    )


def copy_example(name: str, dest: Path, *, force: bool = False) -> Example:
    """Copy example ``name`` into ``dest`` and return it.

    Refuses an existing ``dest`` unless ``force``, which overwrites the
    example's own files and leaves anything else in ``dest`` alone.
    """
    example = BUNDLED.get(name)
    if example is None:
        raise ExampleError(f"no bundled example named {name!r}; available: {', '.join(BUNDLED)}")
    if dest.exists() and not force:
        raise ExampleError(f"{dest} already exists; pass --force to overwrite the example's files")
    source = examples_root() / name
    for rel in example.files:
        src = source / rel
        target = dest / rel
        if src.is_dir():
            shutil.copytree(
                src,
                target,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
        elif src.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
        else:
            raise ExampleError(f"bundled example {name!r} is missing {rel}")
    readme = dest / "README.md"
    if example.readme_rewrites and readme.is_file():
        readme.write_text(
            rewrite_readme(example, readme.read_text(encoding="utf-8"), dest.name),
            encoding="utf-8",
        )
    return example


def rewrite_readme(example: Example, text: str, directory: str) -> str:
    """``text`` (the repo's README) with the copy's paths in place."""
    for old, new in example.readme_rewrites:
        text = text.replace(old, new.replace("{dir}", directory))
    return text


def render_commands(example: Example, directory: str) -> list[str]:
    """The example's commands for a copy at ``directory``, comments included."""
    lines: list[str] = []
    for comment, command in example.commands:
        if comment:
            if lines:
                lines.append("")
            lines.append(f"# {comment}")
        # replace, not format: the commands carry literal JSON braces.
        lines.append(command.replace("{dir}", shlex.quote(directory)))
    return lines
