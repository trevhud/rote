"""What every client source does the same way: read files, build candidates, rank scopes.

:mod:`rote.clients.normalize` holds the rules for one value (a name, a
header, a URL). This module holds the steps around them that every source
repeats, so a fix lands once:

- :class:`Reader` opens each file once, in the format the client writes
  (JSON, JSON with comments, TOML, YAML), and turns every failure into a
  ``problems`` line instead of an exception.
- :func:`new_candidate`, :func:`blocked` and :func:`remote` build an
  :class:`~rote.clients.base.ImportCandidate`: the name mapped to an
  identifier, a non-importable transport listed with its reason, and a
  remote server's headers and URL put through the normalize rules.
- :func:`rank` applies "the higher scope's entry wins" and marks the losers
  shadowed.
- :func:`xdg_config_home`, :func:`appdata` and :func:`find_upwards` locate
  config dirs from :class:`~rote.clients.base.ClientContext` alone.
"""

from __future__ import annotations

import json
import re
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import json5
import yaml

from rote.clients.base import (
    ClientContext,
    HeaderTemplate,
    ImportCandidate,
    Scope,
    Transport,
)
from rote.clients.normalize import (
    PlaceholderSyntax,
    header_template,
    server_identifier,
    url_problem,
    url_secret_risk,
)

#: ``json``: strict JSON. ``jsonc``: JSON with comments and trailing commas
#: (read with json5, which accepts a little more than JSONC: a file the
#: client would reject can still be listed, never the reverse).
Format = Literal["json", "jsonc", "toml", "yaml"]

STDIO_BLOCKED = "a stdio server (a local command); rote's registry holds HTTP and SSE servers only"
WEBSOCKET_BLOCKED = "a WebSocket server; rote connects over HTTP and SSE only"


class _Yaml12Loader(yaml.SafeLoader):
    """PyYAML's safe loader with YAML 1.2 booleans: only true/false. YAML 1.1
    also reads ``on``, ``off``, ``yes`` and ``no`` as booleans, so a Goose
    extension named ``off`` (serde_yaml, YAML 1.2, keeps it a string) would
    come back as ``False``."""


_BOOL_TAG = "tag:yaml.org,2002:bool"
_YAML12_BOOL = re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$")
_Yaml12Loader.yaml_implicit_resolvers = {
    first: [(tag, regex) for tag, regex in resolvers if tag != _BOOL_TAG]
    + ([(_BOOL_TAG, _YAML12_BOOL)] if first in "tTfF" else [])
    for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}


def _parse(text: str, fmt: Format) -> Any:
    if fmt == "json":
        return json.loads(text)
    if fmt == "jsonc":
        return json5.loads(text)
    if fmt == "toml":
        return tomllib.loads(text)
    return yaml.load(text, Loader=_Yaml12Loader)  # a SafeLoader subclass: safe


class Reader:
    """Reads files once each and records what it read and what failed."""

    def __init__(self) -> None:
        self.files_read: list[Path] = []
        self.problems: list[str] = []
        self._cache: dict[Path, Any] = {}

    def load(self, path: Path, fmt: Format = "json") -> Any:
        """The parsed file, or None when it is absent or unreadable (the
        second is a problem)."""
        if path in self._cache:
            return self._cache[path]
        value: Any = None
        if path.is_file():
            try:
                value = _parse(path.read_text(encoding="utf-8"), fmt)
            except (OSError, UnicodeDecodeError, ValueError, yaml.YAMLError) as e:
                # json.JSONDecodeError, tomllib.TOMLDecodeError and json5's
                # errors are ValueErrors. The message is not shown: a parser
                # quotes the text around the error, which can be a secret.
                self.problems.append(f"{path}: could not read it ({type(e).__name__})")
            else:
                self.files_read.append(path)
        self._cache[path] = value
        return value

    def mapping(self, path: Path, fmt: Format = "json") -> dict[str, Any]:
        value = self.load(path, fmt)
        if value is None:
            return {}
        if not isinstance(value, dict):
            self.problems.append(f"{path}: expected an object, got {type(value).__name__}")
            return {}
        return value

    def section(self, path: Path, data: Mapping[str, Any], key: str) -> dict[str, Any]:
        """``data[key]`` when it is an object; a problem when it is something else."""
        value = data.get(key)
        if value is None:
            return {}
        if not isinstance(value, dict):
            self.problems.append(f"{path}: {key} is a {type(value).__name__}, not an object")
            return {}
        return value


def servers_in(value: Any) -> dict[str, Any] | None:
    """A ``.mcp.json``-style file's servers: wrapped in ``mcpServers`` or a
    bare map of names (Claude Code, VS Code and Copilot CLI accept both)."""
    if not isinstance(value, dict):
        return None
    inner = value.get("mcpServers")
    if isinstance(inner, dict):
        return inner
    return value if "mcpServers" not in value else None


def str_list(value: Any) -> list[str]:
    return [v for v in value if isinstance(v, str)] if isinstance(value, list) else []


def new_candidate(
    client: str,
    original_name: str,
    *,
    scope: Scope,
    origin: Path,
    plugin: str | None = None,
) -> ImportCandidate:
    """The candidate before its entry is read: name mapped, transport unknown.
    Blocked already when the name keeps no letters or digits."""
    original_name = str(original_name)  # a YAML or TOML key can be a number
    try:
        name = server_identifier(original_name)
    except ValueError:
        name = "_"
    base = ImportCandidate(
        client=client,
        name=name,
        original_name=original_name,
        scope=scope,
        origin=origin,
        transport="other",
        plugin=plugin,
    )
    if name == "_":
        return replace(base, blocked="the name has no letters or digits rote can keep")
    return base


def blocked(base: ImportCandidate, transport: Transport, reason: str) -> ImportCandidate:
    """``base`` listed as ``transport`` and not importable, keeping an earlier
    reason (a bad name) when there is one."""
    return replace(base, transport=transport, blocked=base.blocked or reason)


def not_an_object(base: ImportCandidate, entry: Any) -> ImportCandidate:
    return replace(
        base, blocked=base.blocked or f"the entry is a {type(entry).__name__}, not an object"
    )


def remote(
    base: ImportCandidate,
    *,
    url: Any,
    transport: Transport,
    syntax: PlaceholderSyntax,
    headers: Any = None,
    templates: Iterable[HeaderTemplate] = (),
    withheld: Iterable[str] = (),
    scopes: Sequence[str] = (),
) -> ImportCandidate:
    """A remote server entry, put through the normalize rules.

    ``headers`` is the entry's raw header map in the client's own syntax;
    each value is copied as a ``${VAR}`` template or withheld.
    ``templates`` are headers the source already built from the client's
    by-name references (Codex's ``env_http_headers``). ``withheld`` lists
    what the source found and did not copy (an OAuth client id issued to
    that client, a command that makes headers)."""
    base = replace(base, transport=transport)
    if base.blocked is not None:
        return base
    if not isinstance(url, str) or not url:
        return replace(base, blocked="the entry has no url")
    copied: list[HeaderTemplate] = list(templates)
    notes: list[str] = []
    if isinstance(headers, dict):
        for header, value in headers.items():
            if not isinstance(value, str):
                notes.append(f"header {header}: not a string")
                continue
            verdict = header_template(base.name, str(header), value, syntax)
            if verdict.header is not None:
                copied.append(verdict.header)
            if verdict.withheld is not None:
                notes.append(verdict.withheld)
            notes.extend(verdict.notes)
    elif headers is not None:
        notes.append(f"headers: a {type(headers).__name__}, not a map; nothing copied")
    notes.extend(withheld)
    problem = url_problem(url, syntax)
    return replace(
        base,
        url=url,
        headers=tuple(copied),
        scopes=tuple(scopes),
        withheld=tuple(notes),
        blocked=problem,
        risk=url_secret_risk(url) if problem is None else None,
    )


def rank(
    groups: Sequence[tuple[str, Sequence[ImportCandidate]]],
    *,
    fold_case: bool = False,
) -> list[ImportCandidate]:
    """Candidates from highest scope to lowest, the losers marked shadowed.

    ``groups`` is ``(label, candidates)`` in the client's precedence order,
    highest first; ``label`` names the scope in the shadowed reason. An
    entry the client does not use (already ``inactive``) wins nothing, so a
    disabled project entry leaves the user entry active, which is the rule
    in every client read so far where the whole entry wins by name.
    ``fold_case``: the client matches names case-insensitively (VS Code)."""
    ordered: list[ImportCandidate] = []
    winner: dict[str, str] = {}
    for label, candidates in groups:
        for candidate in candidates:
            key = candidate.original_name.casefold() if fold_case else candidate.original_name
            if key in winner:
                candidate = replace(
                    candidate, inactive=f"shadowed by the {winner[key]} entry of the same name"
                )
            elif candidate.inactive is None:
                winner[key] = label
            ordered.append(candidate)
    return ordered


def deep_merge(lower: Mapping[str, Any], higher: Mapping[str, Any]) -> dict[str, Any]:
    """``higher`` over ``lower``, descending into tables; anything else
    (lists included) is replaced. Codex's TOML layer merge and OpenCode's
    ``mergeDeep``."""
    merged = dict(lower)
    for key, value in higher.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@dataclass(frozen=True)
class Merged:
    """One server after a field-by-field merge of every layer that names it."""

    name: str
    entry: Any
    #: The highest layer that names it: where the user would edit it.
    scope: Scope
    origin: Path


def merge_entries(
    layers: Iterable[tuple[Scope, Path, Mapping[str, Any]]], *, deep: bool
) -> list[Merged]:
    """Merge servers by name across ``layers`` (lowest first), for clients
    that merge entries field by field instead of replacing them whole.

    ``deep``: nested tables merge too (Codex, OpenCode); otherwise only the
    entry's top-level fields do (Goose). A layer whose value is not an object
    replaces the entry."""
    merged: dict[str, Any] = {}
    top: dict[str, tuple[Scope, Path]] = {}
    for scope, path, servers in layers:
        for name, entry in servers.items():
            lower = merged.get(name)
            if isinstance(entry, dict) and isinstance(lower, dict):
                merged[name] = deep_merge(lower, entry) if deep else {**lower, **entry}
            else:
                merged[name] = entry
            top[name] = (scope, path)
    return [Merged(name, entry, *top[name]) for name, entry in merged.items()]


def switch_off(
    candidates: Iterable[ImportCandidate], off: Mapping[tuple[Path, str], str]
) -> list[ImportCandidate]:
    """Mark ranked winners the client has switched off, by ``(origin,
    original_name)`` to reason.

    Applied after :func:`rank`, for clients where an entry wins by name
    whether or not it is enabled: a disabled higher entry turns the server
    off rather than uncovering a lower one."""
    return [
        replace(c, inactive=off[(c.origin, c.original_name)])
        if c.inactive is None and (c.origin, c.original_name) in off
        else c
        for c in candidates
    ]


def xdg_config_home(ctx: ClientContext) -> Path:
    """``$XDG_CONFIG_HOME``, else ``~/.config`` (the XDG Base Directory default)."""
    override = ctx.env.get("XDG_CONFIG_HOME", "").strip()
    return Path(override) if override else ctx.home / ".config"


def appdata(ctx: ClientContext) -> Path:
    """Windows' roaming app-data dir: ``%APPDATA%``, else its default under the home."""
    override = ctx.env.get("APPDATA", "").strip()
    return Path(override) if override else ctx.home / "AppData" / "Roaming"


def mac_app_support(ctx: ClientContext) -> Path:
    return ctx.home / "Library" / "Application Support"


def find_upwards(cwd: Path, relative: str | Path) -> Path | None:
    """The nearest ``<dir>/<relative>`` from ``cwd`` up, stopping at the
    repository root (the dir holding ``.git``) so a parent's unrelated
    project is never read as this one's."""
    for directory in (cwd, *cwd.parents):
        candidate = directory / relative
        if candidate.is_file():
            return candidate
        if (directory / ".git").exists():
            return None
    return None


def project_root(cwd: Path) -> Path:
    """The repository root above ``cwd``, else ``cwd`` itself (the workspace
    folder an editor opened)."""
    for directory in (cwd, *cwd.parents):
        if (directory / ".git").exists():
            return directory
    return cwd
