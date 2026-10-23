"""rote-cli's dependencies, optional and base, and how to install each one.

The PyPI distribution is ``rote-cli``. The name ``rote`` on PyPI belongs
to an unrelated library, so a hint that names ``rote`` with an extra
installs the wrong package. Every install hint in rote comes from this
module so there is one place that spells the name, and
``tests/test_first_run_output.py`` fails on any hint in ``src/`` that
uses the bare name or types ``rote-cli[`` by hand.

Two kinds of dependency:

- **Base dependencies** (:data:`BASE_MODULES`): the MCP client
  (``fastmcp-slim[client]``) and httpx. Every first run needs them: an
  MCP login, compile's tool grounding, every MCP-bound step. A missing
  one means a broken install, and :func:`reinstall_hint` restores it.
- **Optional extras** (:data:`EXTRAS`): the workflow runtimes rote runs in
  its own environment (dbos, temporal), ``rote serve`` (fastmcp's server
  half), and the API compiler drivers. A missing one is normal. A command
  that needs a runtime's extra offers to install it as a confirmed plan
  (:mod:`rote.setup.install_extra`), with the command :func:`installer`
  derives for this install; anything else prints :func:`install_hint`.

The ``mcp`` extra stays as an empty alias (:data:`ALIAS_EXTRAS`) so an
older install command still resolves.

This module is stdlib-only and imports nothing from rote but its
version, so the CLI can use it before any dependency is importable.
"""

from __future__ import annotations

import importlib.util
import re
import shlex
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple

__all__ = [
    "ALIAS_EXTRAS",
    "BASE_MODULES",
    "DISTRIBUTION",
    "EXTRAS",
    "RUNTIME_EXTRAS",
    "BaseDependency",
    "Extra",
    "Installer",
    "MissingDependencyError",
    "dependency_hint",
    "install_hint",
    "installer",
    "is_installed",
    "missing_module_in",
    "module_is_importable",
    "reinstall_hint",
    "require_module",
]

#: The PyPI distribution name. Never ``rote``: that is someone else's package.
DISTRIBUTION = "rote-cli"


@dataclass(frozen=True)
class Extra:
    """One optional extra of rote-cli."""

    #: What it adds, for a plan's title and ``rote doctor``.
    used_for: str
    #: The top-level module it brings. A ``ModuleNotFoundError`` naming it
    #: is translated into this extra's install hint.
    module: str
    #: Set when :attr:`module` is importable without the extra, so only the
    #: distribution proves it is installed. fastmcp: the slim client is a
    #: base dependency and imports as ``fastmcp`` too; the serve extra is
    #: the full ``fastmcp`` distribution, which adds the server half.
    distribution: str | None = None


#: Optional extra name -> what proves it is installed. Kept in step with
#: ``[project.optional-dependencies]`` in pyproject.toml (tested).
EXTRAS: dict[str, Extra] = {
    "dbos": Extra("the dbos runtime (rote run, rote serve's dbos backend)", "dbos"),
    "temporal": Extra("the temporal runtime (rote run's local worker)", "temporalio"),
    "serve": Extra("rote serve and the bench's fixture MCP servers", "fastmcp", "fastmcp"),
    "api": Extra("the api compiler driver", "anthropic"),
    "openai-api": Extra("the openai-api compiler driver", "openai"),
}

#: Runtime -> the extra rote needs in its own environment to run that
#: runtime's apps (dbos runs under rote's interpreter, temporal runs its
#: worker in-process). The TypeScript runtimes bring their own toolchain,
#: and python needs nothing.
RUNTIME_EXTRAS: dict[str, str] = {"dbos": "dbos", "temporal": "temporal"}

#: Extras that install nothing beyond the base dependencies. They exist so
#: ``rote-cli[mcp]`` from an older README, script or lock file still
#: resolves. Each is empty in pyproject.toml (tested).
ALIAS_EXTRAS: tuple[str, ...] = ("mcp",)


class BaseDependency(NamedTuple):
    #: The distribution pyproject names (the module's own name can differ).
    distribution: str
    used_for: str


#: Top-level modules of base dependencies that a check reports by name.
#: Kept in step with ``[project.dependencies]`` (tested).
BASE_MODULES: dict[str, BaseDependency] = {
    "fastmcp": BaseDependency("fastmcp-slim", "the MCP client"),
    "httpx": BaseDependency("httpx", "run releases and pending logins"),
}

#: Top-level module -> the optional extra that provides it, for translating
#: a subprocess's ``ModuleNotFoundError`` into an install hint. An extra
#: proven by its distribution shares its module with the base install, so a
#: missing module there is a broken install, not a missing extra.
_MODULE_TO_EXTRA: dict[str, str] = {
    extra.module: name for name, extra in EXTRAS.items() if extra.distribution is None
}

_MISSING_MODULE_RE = re.compile(r"ModuleNotFoundError: No module named '([A-Za-z0-9_.]+)'")


class MissingDependencyError(RuntimeError):
    """A module rote needs is not importable; the message carries the fix."""


# ───────── How this rote was installed ─────────

InstallKind = Literal["uv-tool", "uvx", "pip"]


@dataclass(frozen=True)
class _Context:
    kind: InstallKind
    #: Extras the install already has (a uv tool's receipt names them).
    existing: tuple[str, ...] = ()
    #: Other packages a uv tool carries (``uv tool install --with``), as
    #: requirement strings, so re-running ``uv tool install`` keeps them.
    with_requirements: tuple[str, ...] = ()
    #: The interpreter a uv tool was installed with (``--python``), if named.
    python: str | None = None
    #: Where rote-cli came from when not PyPI: an editable checkout's path
    #: (``editable``) or any other direct URL (``other``).
    source: tuple[Literal["editable", "other"], str] | None = None


def _install_context() -> _Context:
    """How the running rote was installed.

    uv writes ``uv-receipt.toml`` into every ``uv tool install`` environment
    (its ``requirements`` carry the requested extras and any ``--with``
    packages), and builds ``uvx`` environments under its cache's
    ``archive-v*`` directories. Anything else is a pip-style environment,
    where ``direct_url.json`` (PEP 610) says whether rote-cli came from a
    checkout instead of PyPI.
    """
    import sys
    import tomllib
    from pathlib import Path

    prefix = Path(sys.prefix)
    receipt = prefix / "uv-receipt.toml"
    if receipt.is_file():
        try:
            tool = tomllib.loads(receipt.read_text(encoding="utf-8"))["tool"]
            requirements = tool["requirements"]
        except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError):
            return _Context("uv-tool", source=("other", "an unreadable uv receipt"))
        existing: tuple[str, ...] = ()
        withs: list[str] = []
        source: tuple[Literal["editable", "other"], str] | None = None
        for requirement in requirements:
            if not isinstance(requirement, dict):
                source = ("other", "an unreadable uv receipt")
                continue
            simple = _simple_requirement(requirement)
            if requirement.get("name") == DISTRIBUTION:
                extras = requirement.get("extras") or []
                existing = tuple(str(e) for e in extras if e not in ALIAS_EXTRAS)
                if simple is None:
                    source = ("other", _receipt_source(requirement))
            elif simple is None:
                source = ("other", f"{requirement.get('name')} from {_receipt_source(requirement)}")
            else:
                withs.append(simple)
        python = tool.get("python") if isinstance(tool.get("python"), str) else None
        return _Context("uv-tool", existing, tuple(withs), python, source)
    if any(part.startswith("archive-v") for part in prefix.parts):
        return _Context("uvx")
    return _Context("pip", source=_direct_url_source())


def _simple_requirement(requirement: dict[str, Any]) -> str | None:
    """A receipt entry as a requirement string, or None when it names a
    source (a path, a git URL) that a requirement string cannot carry."""
    if set(requirement) - {"name", "extras", "specifier", "marker"}:
        return None
    extras = ",".join(str(e) for e in requirement.get("extras") or [])
    spec = str(requirement["name"]) + (f"[{extras}]" if extras else "")
    spec += str(requirement.get("specifier") or "")
    if requirement.get("marker"):
        spec += f"; {requirement['marker']}"
    return spec


def _receipt_source(requirement: dict[str, Any]) -> str:
    for key in ("editable", "directory", "path", "git", "url"):
        if key in requirement:
            return f"{key} {requirement[key]}"
    return "a source uv records outside PyPI"


def _direct_url_source() -> tuple[Literal["editable", "other"], str] | None:
    """Where rote-cli came from, per its ``direct_url.json``; None for an index."""
    import json
    from importlib.metadata import PackageNotFoundError, distribution
    from urllib.parse import unquote, urlsplit

    try:
        text = distribution(DISTRIBUTION).read_text("direct_url.json")
    except PackageNotFoundError:
        return None
    if not text:
        return None
    try:
        data = json.loads(text)
    except ValueError:
        return ("other", "an unreadable direct_url.json")
    url = str(data.get("url", ""))
    if (data.get("dir_info") or {}).get("editable") and url.startswith("file://"):
        return ("editable", unquote(urlsplit(url).path))
    return ("other", url or "a direct URL")


# ───────── The command that installs, and the hint that names it ─────────


@dataclass(frozen=True)
class Installer:
    """How to install rote-cli at this version, with some extras, here.

    One derivation, two renderings: :attr:`hint` is what a person types,
    :attr:`argv` is the exact command rote can run itself (with the
    interpreter or uv binary resolved). ``argv`` is None when rote cannot
    install into its own environment; :attr:`why_not` says why, and the
    hint is still the fix."""

    hint: str
    argv: tuple[str, ...] | None
    why_not: str | None = None


def installer(extras: Iterable[str] = (), *, reinstall: bool = False) -> Installer:
    """The command that installs rote-cli ``==`` the running version with
    ``extras``, for the way this rote was installed.

    The version is pinned so installing an extra never moves rote itself.
    A uv tool install (``uv tool install rote-cli``) has no pip, and
    re-running ``uv tool install`` with the extras replaces the tool's
    environment in place, so the command keeps the extras and ``--with``
    packages it already has. Under ``uvx`` there is nothing to install
    into; the next ``uvx`` call names the extras. Otherwise this
    interpreter's pip, or ``uv pip`` for a venv uv made without pip.

    ``reinstall``: restore a missing base dependency. uv needs
    ``--reinstall`` for that, since it treats an installed tool or a cached
    uvx environment as done; pip installs whatever the pinned requirement
    lacks (checked against pip 26.1).
    """
    import shutil
    import sys

    from rote import __version__

    wanted = tuple(extras)
    for extra in wanted:
        _check_extra(extra)
    ctx = _install_context()

    def spec(names: Iterable[str]) -> str:
        names = sorted(set(names))
        bracket = f"[{','.join(names)}]" if names else ""
        return f"{DISTRIBUTION}{bracket}=={__version__}"

    if ctx.kind == "uvx":
        flag = "--reinstall " if reinstall else ""
        return Installer(
            hint=f"uvx {flag}--from {_quote(spec(wanted))} rote ...",
            argv=None,
            why_not="uvx runs rote in a throwaway environment; run the next uvx command "
            "with the extra named",
        )
    if ctx.kind == "uv-tool":
        words = ["tool", "install"]
        if reinstall:
            words.append("--reinstall")
        if ctx.python:
            words += ["--python", ctx.python]
        words.append(spec((*ctx.existing, *wanted)))
        for requirement in ctx.with_requirements:
            words += ["--with", requirement]
        hint = _join(["uv", *words])
        if ctx.source is not None:
            return Installer(hint, None, f"this uv tool was installed from {ctx.source[1]}")
        uv = shutil.which("uv")
        if uv is None:
            return Installer(hint, None, "uv is not on PATH")
        return Installer(hint, (uv, *words))

    # pip-style environment.
    if ctx.source is not None and ctx.source[0] == "editable":
        # A checkout installed with -e: installing from PyPI would replace
        # it with the released wheel.
        names = ",".join(sorted(set(wanted)))
        target = ["-e", f"{ctx.source[1]}[{names}]" if names else ctx.source[1]]
    else:
        target = [spec(wanted)]
    hint = _join(["pip", "install", *target])
    if ctx.source is not None and ctx.source[0] == "other":
        return Installer(hint, None, f"rote-cli was installed from {ctx.source[1]}, not PyPI")
    if importlib.util.find_spec("pip") is not None:
        return Installer(hint, (sys.executable, "-m", "pip", "install", *target))
    uv = shutil.which("uv")
    if uv is not None:
        return Installer(hint, (uv, "pip", "install", "--python", sys.executable, *target))
    return Installer(hint, None, "this environment has no pip, and uv is not on PATH")


def _quote(word: str) -> str:
    """A shell word for a hint. A requirement (``rote-cli[dbos]==1.0``,
    ``rich>=1``) is always single-quoted, whether or not this shell needs
    it: zsh globs an unquoted ``[``, and one spelling is easier to scan."""
    if re.search(r"[\[\]=<>!~]", word) and "'" not in word:
        return f"'{word}'"
    return shlex.quote(word)


def _join(words: Iterable[str]) -> str:
    return " ".join(_quote(w) for w in words)


def _check_extra(extra: str) -> None:
    if extra in EXTRAS:
        return
    if extra in ALIAS_EXTRAS:
        raise ValueError(
            f"the {extra} extra is an empty alias (its packages are base "
            "dependencies); use dependency_hint(<module>) or reinstall_hint()"
        )
    raise ValueError(f"unknown extra {extra!r}; known: {', '.join(EXTRAS)}")


def install_hint(extra: str) -> str:
    """The exact command that installs optional ``extra`` into the running rote.

    Raises ValueError for an alias extra: it installs nothing, so a hint
    naming it would send the user to a command that changes nothing.
    """
    return installer((extra,)).hint


def reinstall_hint() -> str:
    """The command that restores rote-cli's base dependencies, at this version."""
    return installer(reinstall=True).hint


# ───────── What is installed ─────────


def module_is_importable(module: str) -> bool:
    """Whether top-level ``module`` can be imported (no import side effects)."""
    return importlib.util.find_spec(module) is not None


def is_installed(extra: str) -> bool:
    """Whether optional ``extra`` is installed, without importing it."""
    spec = EXTRAS[extra]
    if spec.distribution is None:
        return module_is_importable(spec.module)
    from importlib.metadata import PackageNotFoundError, distribution

    try:
        distribution(spec.distribution)
    except PackageNotFoundError:
        return False
    return True


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
