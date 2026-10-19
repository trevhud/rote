"""Optional package extras and the install hint for each one.

The PyPI distribution is ``rote-cli``. The name ``rote`` on PyPI belongs
to an unrelated library, so a hint that names ``rote`` with an extra
installs the wrong package. Every install hint in rote comes from
:func:`install_hint` so there is one place that spells the name, and
``tests/test_first_run_output.py`` fails on any hint in ``src/`` that
uses the bare name.

This module is stdlib-only and imports nothing from rote, so the CLI can
use it before any optional dependency is importable.
"""

from __future__ import annotations

import importlib.util
import re

__all__ = [
    "DISTRIBUTION",
    "EXTRAS",
    "MissingExtraError",
    "extra_for_missing_module",
    "install_hint",
    "is_installed",
    "require_extra",
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
    "dbos": "dbos",
    "serve": "fastmcp",
    "mcp": "fastmcp",
}

#: Top-level module -> the extra that provides it, for translating a
#: subprocess's ``ModuleNotFoundError`` into an install hint. ``serve``
#: and ``mcp`` both ship fastmcp; ``mcp`` is the smaller one to point at.
_MODULE_TO_EXTRA: dict[str, str] = {
    "temporalio": "temporal",
    "anthropic": "api",
    "openai": "openai-api",
    "dbos": "dbos",
    "fastmcp": "mcp",
    "httpx": "mcp",
}

_MISSING_MODULE_RE = re.compile(r"ModuleNotFoundError: No module named '([A-Za-z0-9_.]+)'")


class MissingExtraError(RuntimeError):
    """An optional extra is not installed; the message carries the install hint."""


def install_hint(extra: str) -> str:
    """The exact command that installs ``extra`` into the running rote.

    A uv tool install (``uv tool install rote-cli``) has no pip, and
    pip-installing into it is not how uv manages it: re-running ``uv tool
    install`` with the extras replaces the tool's environment, so the hint
    keeps the extras it already has. Under ``uvx`` there is nothing to
    install into; the next ``uvx`` call names the extra. Otherwise pip.
    """
    kind, existing = _install_context()
    if kind == "uv-tool":
        extras = ",".join(sorted({*existing, extra}))
        return f"uv tool install '{DISTRIBUTION}[{extras}]'"
    if kind == "uvx":
        return f"uvx --from '{DISTRIBUTION}[{extra}]' rote ..."
    return f"pip install '{DISTRIBUTION}[{extra}]'"


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
                return ("uv-tool", tuple(str(e) for e in extras))
        return ("uv-tool", ())
    if any(part.startswith("archive-v") for part in prefix.parts):
        return ("uvx", ())
    return ("pip", ())


def is_installed(extra: str) -> bool:
    """Whether ``extra``'s marker module is importable (no import side effects)."""
    return importlib.util.find_spec(EXTRAS[extra]) is not None


def require_extra(extra: str, purpose: str) -> None:
    """Raise :class:`MissingExtraError` unless ``extra`` is installed.

    ``purpose`` finishes the sentence "<purpose> requires the <extra>
    extra", e.g. ``"running a dbos app"``.
    """
    if not is_installed(extra):
        raise MissingExtraError(f"{purpose} requires the {extra} extra: {install_hint(extra)}")


def extra_for_missing_module(output: str) -> tuple[str, str] | None:
    """Find a ``ModuleNotFoundError`` for an extra's module in process output.

    Returns ``(module, extra)`` for the last one found, or None when the
    output has none, or the missing module is not something an extra
    provides (a typo in user code, say).
    """
    matches = _MISSING_MODULE_RE.findall(output)
    for name in reversed(matches):
        top = name.split(".", 1)[0]
        if top in _MODULE_TO_EXTRA:
            return top, _MODULE_TO_EXTRA[top]
    return None
