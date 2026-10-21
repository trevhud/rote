"""The XDG base directories rote's per-user files live under.

Every per-user path rote keeps (registries, the token store, config, the
caches, the eval corpus) resolves its own ``ROTE_*`` override first, then
one of these. An empty variable counts as unset, as the XDG Base Directory
spec says, so ``XDG_DATA_HOME=`` means ``~/.local/share``, not the current
directory.

The emitted runtime helpers (``rote.mcp._runtime_helper``,
``rote.inference._runtime_helper`` and their TypeScript twins) cannot
import rote, so they carry their own copy of the same rule. Keep them in
step: an empty variable is unset there too, or a pipeline looks for its
token store in a different place than ``rote mcp login`` wrote it.
"""

from __future__ import annotations

import os
from pathlib import Path


def _base(var: str, *home_default: str) -> Path:
    value = os.environ.get(var)
    return Path(value) if value else Path.home().joinpath(*home_default)


def config_home() -> Path:
    """``$XDG_CONFIG_HOME``, else ``~/.config``."""
    return _base("XDG_CONFIG_HOME", ".config")


def data_home() -> Path:
    """``$XDG_DATA_HOME``, else ``~/.local/share``."""
    return _base("XDG_DATA_HOME", ".local", "share")


def cache_home() -> Path:
    """``$XDG_CACHE_HOME``, else ``~/.cache``."""
    return _base("XDG_CACHE_HOME", ".cache")
