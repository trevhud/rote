"""Layered CLI defaults — ``rote init`` writes them, commands read them.

Two files, one precedence rule applied to every configurable key::

    explicit flag > ROTE_<KEY> env var > project rote.yaml
        > user config.yaml > built-in default

- **User config** — ``~/.config/rote/config.yaml`` (``XDG_CONFIG_HOME``
  honored; ``ROTE_CONFIG_PATH`` overrides outright, which is also how
  the test suite isolates itself from a developer's real file).
- **Project config** — a ``rote.yaml`` discovered by walking up from
  the working directory. Discovery stops after the first directory that
  contains ``.git`` (a project boundary) or at the filesystem root.
  ``ROTE_PROJECT_CONFIG_PATH`` overrides discovery outright.

Config files are deliberately strict: an unknown key or an invalid
value is a loud :class:`ConfigError` at load time, never a silent
skip — a typo'd ``runtime: clouflare`` must not quietly fall back to
the built-in default.

The built-in default layer is *not* stored here — it stays with the
command (e.g. graduate's login-derived runtime default), because it can
depend on runtime state a config file can't see.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml


class ConfigError(RuntimeError):
    """A config file or ROTE_* env var carries an unusable value."""


def _runtime_choices() -> tuple[str, ...]:
    # Lazy: rote.adapters pulls in the IR models; keep `rote --help` cheap.
    from rote.adapters import ADAPTERS

    return tuple(sorted(ADAPTERS))


def _agent_choices() -> tuple[str, ...]:
    from rote.graduator.drivers import DRIVERS

    return tuple(sorted(DRIVERS))


@dataclass(frozen=True)
class ConfigKey:
    """One configurable default: its name, env override, and validity."""

    name: str
    env: str
    description: str
    choices: tuple[str, ...] | None = None
    lazy_choices: str | None = None  # "runtime" | "agent" — resolved on demand

    def valid_choices(self) -> tuple[str, ...] | None:
        if self.lazy_choices == "runtime":
            return _runtime_choices()
        if self.lazy_choices == "agent":
            return _agent_choices()
        return self.choices


#: The v1 key set. Names match the CLI flags they default
#: (``--runtime``, ``--agent``, ``--model``); ``deploy`` maps to the
#: graduate auto-deploy decision (``rote-cloud`` | ``none``).
CONFIG_KEYS: tuple[ConfigKey, ...] = (
    ConfigKey(
        name="runtime",
        env="ROTE_RUNTIME",
        description="default --runtime for graduate/emit",
        lazy_choices="runtime",
    ),
    ConfigKey(
        name="deploy",
        env="ROTE_DEPLOY",
        description="after graduate: upload to rote cloud, or stay local",
        choices=("rote-cloud", "none"),
    ),
    ConfigKey(
        name="agent",
        env="ROTE_AGENT",
        description="graduator driver (unset = auto-detect)",
        lazy_choices="agent",
    ),
    ConfigKey(
        name="model",
        env="ROTE_MODEL",
        description="graduator model override (unset = driver default)",
    ),
)

_KEYS_BY_NAME: dict[str, ConfigKey] = {k.name: k for k in CONFIG_KEYS}


def user_config_path() -> Path:
    override = os.environ.get("ROTE_CONFIG_PATH")
    if override:
        return Path(override)
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "rote" / "config.yaml"


def project_config_path(cwd: Path | None = None) -> Path | None:
    """The project's ``rote.yaml``, or None when no project declares one."""
    override = os.environ.get("ROTE_PROJECT_CONFIG_PATH")
    if override:
        path = Path(override)
        return path if path.is_file() else None
    directory = (cwd or Path.cwd()).resolve()
    while True:
        candidate = directory / "rote.yaml"
        if candidate.is_file():
            return candidate
        if (directory / ".git").exists() or directory.parent == directory:
            return None
        directory = directory.parent


def _validate_value(key: ConfigKey, value: object, origin: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{origin}: `{key.name}` must be a non-empty string, got {value!r}")
    value = value.strip()
    choices = key.valid_choices()
    if choices is not None and value not in choices:
        raise ConfigError(
            f"{origin}: `{key.name}: {value}` is not valid — expected one of: {', '.join(choices)}"
        )
    return value


def load_config_file(path: Path) -> dict[str, str]:
    """Parse and validate one config file. Unknown keys are errors."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ConfigError(f"{path}: not valid YAML: {e}") from e
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: expected a mapping of settings, got {type(raw).__name__}")
    values: dict[str, str] = {}
    for name, value in raw.items():
        key = _KEYS_BY_NAME.get(str(name))
        if key is None:
            known = ", ".join(k.name for k in CONFIG_KEYS)
            raise ConfigError(f"{path}: unknown setting `{name}` — known settings: {known}")
        values[key.name] = _validate_value(key, value, str(path))
    return values


@dataclass(frozen=True)
class ConfigLayers:
    """The two file layers, loaded and validated once per command."""

    user_path: Path
    user: dict[str, str]
    project_path: Path | None
    project: dict[str, str]


def load_layers(cwd: Path | None = None) -> ConfigLayers:
    user_path = user_config_path()
    user = load_config_file(user_path) if user_path.is_file() else {}
    project_path = project_config_path(cwd)
    project = load_config_file(project_path) if project_path is not None else {}
    return ConfigLayers(user_path=user_path, user=user, project_path=project_path, project=project)


@dataclass(frozen=True)
class Resolved:
    """An effective value plus the layer it came from (for `rote config`)."""

    value: str | None
    source: str


def resolve(key_name: str, flag_value: str | None, *, layers: ConfigLayers) -> Resolved:
    """Apply the precedence rule to one key.

    ``flag_value`` is the already-parsed CLI flag (None when the user
    didn't pass it). A ``value=None`` result means "no layer had an
    opinion" — the command applies its built-in default.
    """
    key = _KEYS_BY_NAME[key_name]
    if flag_value is not None:
        return Resolved(value=flag_value, source="flag")
    env_value = os.environ.get(key.env)
    if env_value is not None:
        return Resolved(value=_validate_value(key, env_value, key.env), source=f"env {key.env}")
    if key.name in layers.project:
        assert layers.project_path is not None
        return Resolved(value=layers.project[key.name], source=f"project {layers.project_path}")
    if key.name in layers.user:
        return Resolved(value=layers.user[key.name], source=f"user {layers.user_path}")
    return Resolved(value=None, source="built-in default")


def write_config(path: Path, values: dict[str, str]) -> None:
    """Write a config file the way ``rote init`` presents it: one
    commented line per setting, keys in registry order."""
    for name, value in values.items():
        key = _KEYS_BY_NAME.get(name)
        if key is None:
            raise ConfigError(f"cannot write unknown setting `{name}`")
        _validate_value(key, value, str(path))
    lines = [
        "# rote defaults — written by `rote init`, read by every command.",
        "# Precedence: flag > ROTE_* env > project rote.yaml > this file.",
    ]
    for key in CONFIG_KEYS:
        if key.name in values:
            lines.append(f"{key.name}: {values[key.name]}  # {key.description}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
