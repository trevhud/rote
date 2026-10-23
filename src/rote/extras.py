"""rote-cli's dependencies, optional and base, and the hint that installs each one.

The PyPI distribution is ``rote-cli``. The name ``rote`` on PyPI belongs
to an unrelated library, so a hint that names ``rote`` with an extra
installs the wrong package. Every install hint in rote comes from this
module so there is one place that spells the name, and
``tests/test_first_run_output.py`` fails on any hint in ``src/`` that
uses the bare name or types ``rote-cli[`` by hand.

Two kinds of dependency:

- **Optional extras** (:data:`EXTRAS`): temporal, api, openai-api. A
  missing one is normal, and :func:`install_hint` names the extra.
- **Base dependencies** (:data:`BASE_MODULES`): dbos and fastmcp/httpx
  were the ``dbos``, ``mcp`` and ``serve`` extras until the compile
  default (dbos) and the first-run demo needed them on every install.
  Those extras stay as empty aliases (:data:`ALIAS_EXTRAS`) so an older
  install command still resolves. A missing base module means a broken
  install, and :func:`reinstall_hint` restores it.

This module is stdlib-only and imports nothing from rote but its
version, so the CLI can use it before any dependency is importable.
"""

from __future__ import annotations

import importlib.util
import re

__all__ = [
    "ALIAS_EXTRAS",
    "BASE_MODULES",
    "DISTRIBUTION",
    "EXTRAS",
    "MissingDependencyError",
    "dependency_hint",
    "install_hint",
    "is_installed",
    "missing_module_in",
    "module_is_importable",
    "reinstall_hint",
    "require_module",
]

#: The PyPI distribution name. Never ``rote``: that is someone else's package.
DISTRIBUTION = "rote-cli"

#: Optional extra name -> the top-level module whose importability proves
#: the extra is installed. Kept in step with
#: ``[project.optional-dependencies]`` in pyproject.toml (tested).
EXTRAS: dict[str, str] = {
    "temporal": "temporalio",
    "api": "anthropic",
    "openai-api": "openai",
}

#: Extras that install nothing beyond the base dependencies. They exist so
#: ``rote-cli[dbos]`` and ``rote-cli[mcp]`` from an older README, script or
#: lock file still resolve. Each is empty in pyproject.toml (tested).
ALIAS_EXTRAS: tuple[str, ...] = ("dbos", "mcp", "serve")

#: Top-level modules of base dependencies that a check reports by name ->
#: what rote uses them for. Kept in step with ``[project.dependencies]``
#: (tested).
BASE_MODULES: dict[str, str] = {
    "dbos": "the dbos runtime (the compile default)",
    "fastmcp": "the MCP client and `rote serve`",
    "httpx": "MCP transports and run releases",
}

#: Top-level module -> the optional extra that provides it, for translating
#: a subprocess's ``ModuleNotFoundError`` into an install hint.
_MODULE_TO_EXTRA: dict[str, str] = {module: extra for extra, module in EXTRAS.items()}

_MISSING_MODULE_RE = re.compile(r"ModuleNotFoundError: No module named '([A-Za-z0-9_.]+)'")


class MissingDependencyError(RuntimeError):
    """A module rote needs is not importable; the message carries the fix."""


def install_hint(extra: str) -> str:
    """The exact command that installs optional ``extra`` into the running rote.

    A uv tool install (``uv tool install rote-cli``) has no pip, and
    pip-installing into it is not how uv manages it: re-running ``uv tool
    install`` with the extras replaces the tool's environment, so the hint
    keeps the extras it already has. Under ``uvx`` there is nothing to
    install into; the next ``uvx`` call names the extra. Otherwise pip.

    Raises ValueError for an alias extra: it installs nothing, so a hint
    naming it would send the user to a command that changes nothing.
    """
    if extra not in EXTRAS:
        if extra in ALIAS_EXTRAS:
            raise ValueError(
                f"the {extra} extra is an empty alias (its packages are base "
                "dependencies); use dependency_hint(<module>) or reinstall_hint()"
            )
        raise ValueError(f"unknown extra {extra!r}; known: {', '.join(EXTRAS)}")
    kind, existing = _install_context()
    if kind == "uv-tool":
        extras = ",".join(sorted({*existing, extra}))
        return f"uv tool install '{DISTRIBUTION}[{extras}]'"
    if kind == "uvx":
        return f"uvx --from '{DISTRIBUTION}[{extra}]' rote ..."
    return f"pip install '{DISTRIBUTION}[{extra}]'"


def reinstall_hint() -> str:
    """The command that restores rote-cli's base dependencies, at this version.

    A pinned ``pip install 'rote-cli==<version>'`` installs whatever
    dependency is missing and leaves rote at the version that is running
    (checked against pip 26.1). uv needs ``--reinstall``, since it treats an
    installed tool or a cached uvx environment as done.
    """
    from rote import __version__

    kind, existing = _install_context()
    pinned = f"{DISTRIBUTION}=={__version__}"
    if kind == "uv-tool":
        extras = f"[{','.join(sorted(existing))}]" if existing else ""
        return f"uv tool install --reinstall '{DISTRIBUTION}{extras}=={__version__}'"
    if kind == "uvx":
        return f"uvx --reinstall --from '{pinned}' rote ..."
    return f"pip install '{pinned}'"


def _install_context() -> tuple[str, tuple[str, ...]]:
    """``("uv-tool", extras)``, ``("uvx", ())`` or ``("pip", ())`` for this process.

    uv writes ``uv-receipt.toml`` into every ``uv tool install`` environment
    (its ``requirements`` carry the requested extras), and builds ``uvx``
    environments under its cache's ``archive-v*`` directories.
    """
    import sys
    import tomllib
    from pathlib import Path

    prefix = Path(sys.prefix)
    receipt = prefix / "uv-receipt.toml"
    if receipt.is_file():
        try:
            requirements = tomllib.loads(receipt.read_text(encoding="utf-8"))["tool"][
                "requirements"
            ]
        except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError):
            return ("uv-tool", ())
        for requirement in requirements:
            if isinstance(requirement, dict) and requirement.get("name") == DISTRIBUTION:
                extras = requirement.get("extras") or []
                return ("uv-tool", tuple(str(e) for e in extras if e not in ALIAS_EXTRAS))
        return ("uv-tool", ())
    if any(part.startswith("archive-v") for part in prefix.parts):
        return ("uvx", ())
    return ("pip", ())


def module_is_importable(module: str) -> bool:
    """Whether top-level ``module`` can be imported (no import side effects)."""
    return importlib.util.find_spec(module) is not None


def is_installed(extra: str) -> bool:
    """Whether optional ``extra``'s marker module is importable."""
    return module_is_importable(EXTRAS[extra])


def dependency_hint(module: str) -> str:
    """How to get top-level ``module``, as a noun phrase that reads after
    "needs" or "imports": which package set provides it and the command that
    installs it.

    A base module is reported as missing from this install (it should
    never be), an extra's module as coming with that extra."""
    if module in BASE_MODULES:
        return (
            f"`{module}`, a {DISTRIBUTION} dependency missing from this install "
            f"(restore it with: {reinstall_hint()})"
        )
    if module in _MODULE_TO_EXTRA:
        extra = _MODULE_TO_EXTRA[module]
        return f"`{module}`, from the {extra} extra ({install_hint(extra)})"
    raise ValueError(f"{module!r} is neither a base module nor an extra's module")


def require_module(module: str, purpose: str) -> None:
    """Raise :class:`MissingDependencyError` unless ``module`` is importable.

    ``purpose`` finishes the sentence "<purpose> needs ...", e.g.
    ``"running a dbos app"``.
    """
    if not module_is_importable(module):
        raise MissingDependencyError(f"{purpose} needs {dependency_hint(module)}")


def missing_module_in(output: str) -> str | None:
    """Find a ``ModuleNotFoundError`` for a module rote knows how to install
    in process output, and return that top-level module.

    Returns the last one found, or None when the output has none, or the
    missing module is neither a base dependency nor an extra's (a typo in
    user code, say). :func:`dependency_hint` turns it into the fix.
    """
    matches: list[str] = _MISSING_MODULE_RE.findall(output)
    for name in reversed(matches):
        top: str = name.split(".", 1)[0]
        if top in _MODULE_TO_EXTRA or top in BASE_MODULES:
            return top
    return None
