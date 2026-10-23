"""OpenAI Codex CLI's MCP servers, read from its ``config.toml`` layers.

Verified 2026-10-22 against the source on main and the docs:
https://github.com/openai/codex/blob/main/codex-rs/config/src/mcp_types.rs,
https://github.com/openai/codex/blob/main/codex-rs/config/src/loader/mod.rs,
https://github.com/openai/codex/blob/main/codex-rs/config/src/loader/README.md,
https://github.com/openai/codex/blob/main/codex-rs/config/src/merge.rs,
https://github.com/openai/codex/blob/main/codex-rs/utils/home-dir/src/lib.rs,
https://developers.openai.com/codex/mcp.md and
https://developers.openai.com/codex/config-reference.md:

- **user**: ``$CODEX_HOME/config.toml``, ``~/.codex/config.toml`` when
  ``CODEX_HOME`` is unset (every OS; ``XDG_CONFIG_HOME`` is not read).
- **system**: ``/etc/codex/config.toml`` (``%ProgramData%\\OpenAI\\Codex\\``
  on Windows), below the user file.
- **project**: ``.codex/config.toml`` in each directory from the project
  root (the nearest ``.git``) down to the working directory, closest
  winning, and only in a trusted project: ``[projects."<abs path>"]
  trust_level = "trusted"`` in the user config, looked up for the
  directory, then the project root. An untrusted layer is left out.
- Layers merge as TOML tables, the higher layer winning per key, so
  ``mcp_servers.<name>`` merges field by field (a project can set only
  ``enabled = false`` on a user server).
- Entry: ``command`` means stdio, else ``url`` means Streamable HTTP; there
  is no SSE and no ``type`` key. HTTP keys: ``bearer_token_env_var``
  (sent as ``Authorization: Bearer <value>``), ``env_http_headers``
  (header to variable name), ``http_headers`` (literal values),
  ``http_headers_helper`` (a command), ``auth`` (``oauth``, ``chatgpt``,
  ``ema_auth``), ``scopes``, ``[oauth]`` (``client_id``, ``client_secret``,
  ...). Shared: ``enabled`` (default true), ``enabled_tools``,
  ``disabled_tools``, ``required``, timeouts, approval modes.
- No ``${VAR}`` interpolation: a value is sent as written
  (:data:`~rote.clients.normalize.NO_PLACEHOLDERS`); environment indirection
  is only by name, through the two ``*_env_var`` / ``env_*`` keys above.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from rote.clients.base import ClientContext, HeaderTemplate, ImportCandidate, Scope, SourceReading
from rote.clients.common import (
    STDIO_BLOCKED,
    Reader,
    blocked,
    deep_merge,
    merge_entries,
    new_candidate,
    not_an_object,
    remote,
    str_list,
    switch_off,
    trust_failed,
)
from rote.clients.normalize import NO_PLACEHOLDERS

CLIENT_ID = "codex"

LIMITS = (
    "profile files (<name>.config.toml, used with --profile), -c overrides, managed and "
    "cloud-delivered config, and requirements.toml are not read",
    "plugin servers and their [plugins.<name>.mcp_servers] overrides are not listed",
)

_VAR = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def codex_home(ctx: ClientContext) -> Path:
    override = ctx.env.get("CODEX_HOME", "").strip()
    return Path(override) if override else ctx.home / ".codex"


def _project_dirs(cwd: Path) -> list[Path]:
    """From the project root (the nearest ``.git``) down to ``cwd``;
    ``cwd`` alone outside a repository."""
    chain: list[Path] = []
    for directory in (cwd, *cwd.parents):
        chain.append(directory)
        if (directory / ".git").exists():
            return list(reversed(chain))
    return [cwd]


def _trusted(projects: Any, directory: Path, root: Path) -> bool:
    if not isinstance(projects, dict):
        return False
    for key in (directory, root):
        entry = projects.get(str(key))
        if isinstance(entry, dict) and "trust_level" in entry:
            return bool(entry["trust_level"] == "trusted")
    return False


def _candidate(name: str, entry: dict[str, Any], *, scope: Scope, origin: Path) -> ImportCandidate:
    base = new_candidate(CLIENT_ID, name, scope=scope, origin=origin)
    if "command" in entry:
        return blocked(base, "stdio", STDIO_BLOCKED)
    templates: list[HeaderTemplate] = []
    withheld: list[str] = []
    bearer = entry.get("bearer_token_env_var")
    if isinstance(bearer, str) and _VAR.fullmatch(bearer):
        templates.append(HeaderTemplate("Authorization", f"Bearer ${{{bearer}}}", (bearer,)))
    elif bearer is not None:
        withheld.append("bearer_token_env_var: not a variable name, not copied")
    env_headers = entry.get("env_http_headers")
    if isinstance(env_headers, dict):
        for header, var in env_headers.items():
            if isinstance(var, str) and _VAR.fullmatch(var):
                templates.append(HeaderTemplate(str(header), f"${{{var}}}", (var,)))
            else:
                withheld.append(f"header {header}: env_http_headers names no variable, not copied")
    if "bearer_token" in entry:
        withheld.append("bearer_token: an inline token (Codex rejects it), never copied")
    if "http_headers_helper" in entry:
        withheld.append(
            "http_headers_helper: it runs a command to make headers, and rote does not run "
            f"another app's commands; set headers with rote mcp add {base.name} <url> --header"
        )
    auth = entry.get("auth")
    if auth in ("chatgpt", "ema_auth"):
        withheld.append(
            f"auth = {auth}: Codex signs in with its own account; rote mcp login {base.name} "
            "uses the server's OAuth"
        )
    oauth = entry.get("oauth")
    if isinstance(oauth, dict):
        if "client_id" in oauth:
            withheld.append(
                "oauth.client_id: issued to Codex; rote registers its own client at "
                f"rote mcp login {base.name}"
            )
        if "client_secret" in oauth:
            withheld.append("oauth.client_secret: a secret issued to Codex, never copied")
    if entry.get("enabled_tools") is not None or entry.get("disabled_tools") is not None:
        withheld.append(
            "enabled_tools/disabled_tools: rote imports the whole server; the tool filter is "
            "not copied"
        )
    return remote(
        base,
        url=entry.get("url"),
        transport="streamable-http",
        syntax=NO_PLACEHOLDERS,
        headers=entry.get("http_headers"),
        templates=templates,
        withheld=withheld,
        scopes=str_list(entry.get("scopes")),
    )


class CodexSource:
    id = CLIENT_ID
    label = "Codex CLI"

    def _system_file(self, ctx: ClientContext) -> Path | None:
        if ctx.etc_dir is None:
            return None
        if ctx.platform == "win32":
            return ctx.etc_dir / "OpenAI" / "Codex" / "config.toml"
        return ctx.etc_dir / "codex" / "config.toml"

    def config_paths(self, ctx: ClientContext) -> tuple[Path, ...]:
        home = codex_home(ctx)
        project = [d / ".codex" / "config.toml" for d in reversed(_project_dirs(ctx.cwd))]
        system = self._system_file(ctx)
        return (
            *(p for p in project if p.parent != home),
            home / "config.toml",
            *((system,) if system is not None else ()),
        )

    def detect(self, ctx: ClientContext) -> bool:
        return codex_home(ctx).is_dir()

    def read(self, ctx: ClientContext) -> SourceReading:
        reader = Reader()
        home = codex_home(ctx)
        user_path = home / "config.toml"
        user = reader.mapping(user_path, "toml")
        system_path = self._system_file(ctx)
        system = reader.mapping(system_path, "toml") if system_path is not None else {}
        projects = deep_merge(system, user).get("projects")

        dirs = _project_dirs(ctx.cwd)
        layers: list[tuple[Scope, Path, dict[str, Any]]] = []
        if system_path is not None:
            layers.append(("system", system_path, system))
        layers.append(("user", user_path, user))
        untrusted: list[ImportCandidate] = []
        for directory in dirs:
            path = directory / ".codex" / "config.toml"
            if path.parent == home or not path.is_file():
                continue
            data = reader.mapping(path, "toml")
            if _trusted(projects, directory, dirs[0]):
                layers.append(("project", path, data))
                continue
            for name, entry in reader.section(path, data, "mcp_servers").items():
                candidate = (
                    _candidate(name, entry, scope="project", origin=path)
                    if isinstance(entry, dict)
                    else not_an_object(
                        new_candidate(CLIENT_ID, name, scope="project", origin=path), entry
                    )
                )
                untrusted += trust_failed(
                    [candidate],
                    "in a project Codex does not trust; it skips .codex/config.toml there",
                )

        candidates: list[ImportCandidate] = []
        off: dict[tuple[Path, str], str] = {}
        sections = [
            (scope, path, reader.section(path, data, "mcp_servers")) for scope, path, data in layers
        ]
        for m in merge_entries(sections, deep=True):
            if not isinstance(m.entry, dict):
                base = new_candidate(CLIENT_ID, m.name, scope=m.scope, origin=m.origin)
                candidates.append(not_an_object(base, m.entry))
                continue
            candidates.append(_candidate(m.name, m.entry, scope=m.scope, origin=m.origin))
            if m.entry.get("enabled") is False:
                off[(m.origin, m.name)] = "disabled in Codex (enabled = false)"
        return SourceReading(
            client=CLIENT_ID,
            candidates=(*switch_off(candidates, off), *untrusted),
            files_read=tuple(reader.files_read),
            problems=tuple(reader.problems),
            limits=LIMITS,
        )
