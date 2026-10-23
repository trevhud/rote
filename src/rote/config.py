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

Most settings are one string, and the highest layer wins. A **list**
setting (``work_domains``) is different on purpose: every layer adds to
it and none removes (:func:`resolve_list`). It is a safety list, the
domains whose MCP servers ``rote mcp import`` never preselects, so a
committed project ``rote.yaml`` saying ``work_domains: []`` must not
switch off the user's own. Claude Code merges its ``deniedMcpServers``
from every scope for the same reason.

The built-in default layer is *not* stored here — it stays with the
command (e.g. compile's login-derived runtime default), because it can
depend on runtime state a config file can't see.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from rote.xdg import config_home


class ConfigError(RuntimeError):
    """A config file or ROTE_* env var carries an unusable value."""


#: One setting's value: a string, or a tuple for a list setting.
ConfigValue = str | tuple[str, ...]


def _runtime_choices() -> tuple[str, ...]:
    # Lazy: rote.adapters pulls in the IR models; keep `rote --help` cheap.
    from rote.adapters import ADAPTERS

    return tuple(sorted(ADAPTERS))


def _agent_choices() -> tuple[str, ...]:
    from rote.compiler.drivers import DRIVERS

    return tuple(sorted(DRIVERS))


def _inference_choices() -> tuple[str, ...]:
    from rote.inference import PROVIDERS

    return PROVIDERS


def _effort_choices() -> tuple[str, ...]:
    from rote.compiler.drivers.claude import EFFORT_LEVELS

    return EFFORT_LEVELS


def _decider_choices() -> tuple[str, ...]:
    from rote.compiler.system1 import DECIDERS

    return DECIDERS


def _tiebreak_choices() -> tuple[str, ...]:
    from rote.compiler.system1 import TIEBREAKS

    return TIEBREAKS


def _structure_choices() -> tuple[str, ...]:
    from rote.compiler.system1.structure import MODELS

    return ("rules", *MODELS)


@dataclass(frozen=True)
class ConfigKey:
    """One configurable default: its name, env override, and validity."""

    name: str
    env: str
    description: str
    choices: tuple[str, ...] | None = None
    # "runtime" | "agent" | "inference" | "effort" | "decider" | "tiebreak" | "structure",
    # resolved on demand
    lazy_choices: str | None = None
    #: "str": one value, the highest layer wins. "list": every layer adds
    #: (see the module docstring); the env twin is comma-separated.
    kind: Literal["str", "list"] = "str"

    def valid_choices(self) -> tuple[str, ...] | None:
        if self.lazy_choices == "runtime":
            return _runtime_choices()
        if self.lazy_choices == "agent":
            return _agent_choices()
        if self.lazy_choices == "inference":
            return _inference_choices()
        if self.lazy_choices == "effort":
            return _effort_choices()
        if self.lazy_choices == "decider":
            return _decider_choices()
        if self.lazy_choices == "tiebreak":
            return _tiebreak_choices()
        if self.lazy_choices == "structure":
            return _structure_choices()
        return self.choices


#: The v1 key set. Names match the CLI flags they default
#: (``--runtime``, ``--agent``, ``--model``); ``deploy`` maps to the
#: compile auto-deploy decision (``rote-cloud`` | ``none``).
CONFIG_KEYS: tuple[ConfigKey, ...] = (
    ConfigKey(
        name="runtime",
        env="ROTE_RUNTIME",
        description="default --runtime for compile/emit",
        lazy_choices="runtime",
    ),
    ConfigKey(
        name="deploy",
        env="ROTE_DEPLOY",
        description="after compile: upload to rote cloud, or stay local",
        choices=("rote-cloud", "none"),
    ),
    ConfigKey(
        name="agent",
        env="ROTE_AGENT",
        description="compiler driver (unset = auto-detect)",
        lazy_choices="agent",
    ),
    ConfigKey(
        name="model",
        env="ROTE_MODEL",
        description="compiler model override (unset = driver default)",
    ),
    ConfigKey(
        name="effort",
        env="ROTE_EFFORT",
        description="compiler thinking effort, claude + openai-api drivers (unset = low on claude)",
        lazy_choices="effort",
    ),
    ConfigKey(
        name="decider",
        env="ROTE_DECIDER",
        description=(
            "the decision model that checks the structure pass: auto (Clef with "
            "ROTE_CLEF_ACCOUNT_ID and ROTE_CLEF_API_TOKEN, else Jev with TYPESAFE_API_KEY, "
            "else none), clef, jev, or rules (no call); unset = auto"
        ),
        lazy_choices="decider",
    ),
    ConfigKey(
        name="tiebreak",
        env="ROTE_TIEBREAK",
        description=(
            "who decides the steps System-1 is still unsure of: haiku (one small call, "
            "billed like the compile) or none; unset = none"
        ),
        lazy_choices="tiebreak",
    ),
    ConfigKey(
        name="structure",
        env="ROTE_STRUCTURE",
        description=(
            "who decides the pipeline's skeleton: haiku or sonnet (one call on the "
            "compile driver's own client, checked against the skill) or rules (no "
            "call); unset = haiku"
        ),
        lazy_choices="structure",
    ),
    ConfigKey(
        name="inference",
        env="ROTE_INFERENCE",
        description="who serves (and pays for) emitted judges; unset = auto-detect",
        lazy_choices="inference",
    ),
    ConfigKey(
        name="work_domains",
        env="ROTE_WORK_DOMAINS",
        description=(
            "your work domains: rote mcp import never preselects a server on them "
            "(a list; every layer adds to it)"
        ),
        kind="list",
    ),
)

_KEYS_BY_NAME: dict[str, ConfigKey] = {k.name: k for k in CONFIG_KEYS}


def user_config_path() -> Path:
    override = os.environ.get("ROTE_CONFIG_PATH")
    if override:
        return Path(override)
    return config_home() / "rote" / "config.yaml"


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
    if key.kind != "str":
        raise ConfigError(f"{origin}: `{key.name}` is a list setting, not a string")
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{origin}: `{key.name}` must be a non-empty string, got {value!r}")
    value = value.strip()
    choices = key.valid_choices()
    if choices is not None and value not in choices:
        raise ConfigError(
            f"{origin}: `{key.name}: {value}` is not valid — expected one of: {', '.join(choices)}"
        )
    return value


_DOMAIN_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")


def domain_problem(value: str) -> str | None:
    """Why ``value`` is not a domain name (``example.com``), or None.

    A domain alone: no scheme, path, port, wildcard or address, because
    the work check (``rote.clients.normalize.is_work_host``) matches the
    domain and every host under it, and ``*.example.com`` would match
    nothing."""
    if not value:
        return "it is empty"
    if "://" in value or any(c in value for c in "/?#@:"):
        return "give the domain alone, like example.com (no scheme, path or port)"
    if "*" in value:
        return "no wildcard: example.com already covers every host under it"
    labels = value.split(".")
    if len(value) > 253 or not all(_DOMAIN_LABEL.fullmatch(label) for label in labels):
        return "only letters, digits, hyphens and dots, like example.com"
    if all(label.isdigit() for label in labels):
        return "an IP address, not a domain"
    return None


def normalize_domain(value: str) -> str:
    """``value`` lowercased, with surrounding spaces and dots removed."""
    return value.strip().lower().strip(".")


def _validate_list(key: ConfigKey, value: object, origin: str) -> tuple[str, ...]:
    if key.kind != "list":
        raise ConfigError(f"{origin}: `{key.name}` is a string setting, not a list")
    if not isinstance(value, list):
        raise ConfigError(
            f"{origin}: `{key.name}` must be a list (`{key.name}: [example.com]`), got {value!r}"
        )
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ConfigError(f"{origin}: `{key.name}` items must be strings, got {item!r}")
        domain = normalize_domain(item)
        problem = domain_problem(domain)
        if problem is not None:
            raise ConfigError(f"{origin}: `{key.name}` item {item!r} is not a domain: {problem}")
        if domain not in items:
            items.append(domain)
    return tuple(items)


def _validate_any(key: ConfigKey, value: object, origin: str) -> ConfigValue:
    if key.kind == "list":
        return _validate_list(key, value, origin)
    return _validate_value(key, value, origin)


def load_config_file(path: Path) -> dict[str, ConfigValue]:
    """Parse and validate one config file. Unknown keys are errors."""
    return parse_config_text(path.read_text(encoding="utf-8"), path)


def parse_config_text(text: str, path: Path) -> dict[str, ConfigValue]:
    """:func:`load_config_file` on text already read (``path`` names it in
    errors). For a caller that merges into the file and checks at write time
    that it still holds the bytes it parsed."""
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise ConfigError(f"{path}: not valid YAML: {e}") from e
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: expected a mapping of settings, got {type(raw).__name__}")
    values: dict[str, ConfigValue] = {}
    for name, value in raw.items():
        key = _KEYS_BY_NAME.get(str(name))
        if key is None:
            known = ", ".join(k.name for k in CONFIG_KEYS)
            raise ConfigError(f"{path}: unknown setting `{name}` — known settings: {known}")
        values[key.name] = _validate_any(key, value, str(path))
    return values


@dataclass(frozen=True)
class ConfigLayers:
    """The two file layers, loaded and validated once per command."""

    user_path: Path
    user: dict[str, ConfigValue]
    project_path: Path | None
    project: dict[str, ConfigValue]


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
    if key.kind != "str":
        raise ValueError(f"`{key_name}` is a list setting: use resolve_list")
    if flag_value is not None:
        return Resolved(value=flag_value, source="flag")
    env_value = os.environ.get(key.env)
    if env_value is not None:
        return Resolved(value=_validate_value(key, env_value, key.env), source=f"env {key.env}")
    if key.name in layers.project:
        assert layers.project_path is not None
        project_value = layers.project[key.name]
        assert isinstance(project_value, str)
        return Resolved(value=project_value, source=f"project {layers.project_path}")
    if key.name in layers.user:
        user_value = layers.user[key.name]
        assert isinstance(user_value, str)
        return Resolved(value=user_value, source=f"user {layers.user_path}")
    return Resolved(value=None, source="built-in default")


@dataclass(frozen=True)
class ResolvedList:
    """A list setting: every layer's items, in layer order, and who set it."""

    values: tuple[str, ...]
    #: The layers that hold the key. An empty list counts: it was answered.
    sources: tuple[str, ...]

    @property
    def is_set(self) -> bool:
        return bool(self.sources)

    @property
    def source(self) -> str:
        return " + ".join(self.sources) if self.sources else "not set"


def parse_list_env(key: ConfigKey, raw: str) -> tuple[str, ...]:
    """A list setting's env twin: comma-separated, blank items ignored. An
    empty value is refused, as it is for a string setting: it would read as
    "no items", which is an answer nobody gave."""
    if not raw.strip():
        raise ConfigError(f"{key.env}: `{key.name}` must name at least one item when set")
    return _validate_list(key, [v for v in raw.split(",") if v.strip()], key.env)


def resolve_list(
    key_name: str, flag_values: Sequence[str] = (), *, layers: ConfigLayers
) -> ResolvedList:
    """A list setting: the union of the flag, env, project and user layers.

    No layer removes another's items (see the module docstring). Flag
    values are validated here, like a file's."""
    key = _KEYS_BY_NAME[key_name]
    if key.kind != "list":
        raise ValueError(f"`{key_name}` is a string setting: use resolve")
    values: list[str] = []
    sources: list[str] = []

    def add(items: Sequence[str], source: str) -> None:
        sources.append(source)
        values.extend(item for item in items if item not in values)

    if flag_values:
        add(_validate_list(key, list(flag_values), "flag"), "flag")
    env_value = os.environ.get(key.env)
    if env_value is not None:
        add(parse_list_env(key, env_value), f"env {key.env}")
    for layer, label in (
        (layers.project, f"project {layers.project_path}"),
        (layers.user, f"user {layers.user_path}"),
    ):
        if key.name in layer:
            items = layer[key.name]
            assert isinstance(items, tuple)
            add(items, label)
    return ResolvedList(values=tuple(values), sources=tuple(sources))


def _render_value(value: ConfigValue) -> str:
    if isinstance(value, tuple):
        # Each item double-quoted (JSON is YAML): a domain label such as
        # `on` or `no` would otherwise load as a YAML 1.1 boolean.
        return "[" + ", ".join(json.dumps(item) for item in value) + "]"
    return value


def render_config(values: Mapping[str, ConfigValue], *, origin: str = "config") -> str:
    """A config file's text the way ``rote init`` presents it: one
    commented line per setting, keys in registry order. Validates first."""
    for name, value in values.items():
        key = _KEYS_BY_NAME.get(name)
        if key is None:
            raise ConfigError(f"cannot write unknown setting `{name}`")
        _validate_any(key, list(value) if isinstance(value, tuple) else value, origin)
    lines = [
        "# rote defaults: written by `rote init` and `rote mcp import`, read by every command.",
        "# Precedence: flag > ROTE_* env > project rote.yaml > this file (lists: all add).",
    ]
    for key in CONFIG_KEYS:
        if key.name in values:
            lines.append(render_setting_line(key.name, values[key.name]))
    return "\n".join(lines) + "\n"


def render_setting_line(name: str, value: ConfigValue) -> str:
    """One setting's line, as :func:`render_config` writes it."""
    return f"{name}: {_render_value(value)}  # {_KEYS_BY_NAME[name].description}"


def set_in_place(text: str, name: str, value: ConfigValue, path: Path) -> str | None:
    """``text`` with only setting ``name``'s line (and a block list under
    it) replaced by ``value``, or appended; every other line, comments
    included, kept as written. None when the result does not read back as
    the same settings with ``name`` set to ``value`` (an unusual layout the
    line edit cannot handle), so the caller can plan a marked replacement
    instead."""
    lines = text.splitlines(keepends=True)
    line = render_setting_line(name, value) + "\n"
    head = re.compile(rf"""^["']?{re.escape(name)}["']?\s*:""")
    start = next((i for i, row in enumerate(lines) if head.match(row)), None)
    if start is None:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        lines.append(line)
    else:
        end = start + 1
        while end < len(lines) and lines[end].strip() and lines[end][0] in " \t-":
            end += 1
        lines[start:end] = [line]
    edited = "".join(lines)
    try:
        expected = {**parse_config_text(text, path), name: value}
        got = parse_config_text(edited, path)
    except ConfigError:
        return None
    return edited if got == expected else None


def write_config(path: Path, values: Mapping[str, ConfigValue]) -> None:
    """Write :func:`render_config`'s text to ``path`` atomically."""
    from rote.interact.plan import atomic_write_text

    atomic_write_text(path, render_config(values, origin=str(path)))
