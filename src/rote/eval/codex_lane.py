"""Run a skill under ``codex exec``, inside the same gate as a ``claude -p`` run (#334).

Every skill run rote makes (the baseline, ``rote eval --run``, the paired A/B)
was built on ``claude -p``. This is the Codex lane. It runs on the user's Codex
login (their ChatGPT plan), as the Claude lane runs on their Claude
subscription, and it takes the same gate the callers already build for the
Claude lane, translated to Codex's mechanisms. Each guarantee was measured on
codex-cli 0.165.0:

* **Nothing of the user's setup reaches the run.** ``--ignore-user-config``
  still loads the global ``AGENTS.md``, ``~/.agents/skills`` and plugin
  skills (a probe listed 80 of the user's skills, work ones included). So the
  child gets a throwaway ``CODEX_HOME`` and ``HOME``. Codex's own bundled
  skills (imagegen, openai-docs, skill-creator, skill-installer) are all that
  is left, and image generation is off.
* **The login is the user's own file, never a copy.** The throwaway home's
  ``auth.json`` links to the real one. A copied ChatGPT login that refreshes
  can leave the original holding a rotated-out token, and Codex's CI guide
  says never to share one file across concurrent jobs, so a refresh must land
  in the one real file: if Codex replaced the link with a refreshed file, it
  is written back (:func:`_write_back_login`). ``CODEX_API_KEY`` and
  ``OPENAI_API_KEY`` are scrubbed so the plan, not a key, pays.
* **The binary is launched by its real path.** Codex loads ``AGENTS.md``
  through a sandboxed helper that re-executes the path it was started from;
  started through ``~/.local/bin/codex`` with a throwaway home, the sandbox
  refused that path ("execvp ... Operation not permitted").
* **Reads are scoped to the workspace and the paths the gate names.** A
  ``permissions`` profile reads ``:minimal`` (the system), the binary's dir
  and each ``Read(/<dir>/**)`` rule, denies the temp roots (``:minimal``
  allows them, so a file another process left in ``/tmp`` was readable), and
  writes only the workspace. Checked with ``codex sandbox``: the skill read,
  a canary beside it and ``~/.ssh`` refused, writes outside the workspace
  refused.
* **MCP is exactly the gate.** Each wired server is a ``mcp_servers`` entry;
  explicit ``mcp__<server>__<tool>`` ids become its ``enabled_tools`` (a tool
  left out is not in the model's context at all), a ``mcp__<server>__*``
  lifts the gate, and a server the gate gives nothing is not wired.
  ``default_tools_approval_mode="approve"``, because under
  ``approval_policy="never"`` an MCP call that needs approval fails.
  Credentials go through ``http_headers_helper`` (a command printing a JSON
  object of headers, Claude's ``headersHelper`` contract), so no token lands
  in argv or a file.
* **One command may leave the sandbox, as a tool.** ``Bash(rote skill run
  <name> *)`` cannot be a prefix rule: under a custom permissions profile
  Codex disables every escalation, so an ``allow`` rule lets nothing out
  (measured: the rule escaped under ``sandbox_mode`` and not under the
  profile, with any approval policy). It becomes
  :mod:`rote.eval.skill_run_server` instead, a stdio MCP server whose one tool
  runs the driver with the skill's name fixed, and the prompt says to call it.
  Every shell command stays inside the sandbox: no network, no writes outside
  the workspace.

Any allowlist entry this module cannot translate is refused, never dropped or
widened.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rote.eval import skill_run_server
from rote.eval.empirical import EmpiricalError

#: What the Codex lane is called in flags, reports and the corpus.
AGENT = "codex"

#: Variables that would make Codex bill something other than the user's login.
_CREDENTIAL_ENV = ("CODEX_API_KEY", "OPENAI_API_KEY", "CODEX_ACCESS_TOKEN")

#: Features a skill run must not have: each would reach past the gate (the
#: user's plugins and apps, a browser, the screen) or into the user's state
#: (memories, hooks, goals). Names from ``codex features list`` (0.165.0).
FEATURES_OFF = (
    "apps",
    "browser_use",
    "browser_use_external",
    "computer_use",
    "goals",
    "hooks",
    "image_generation",
    "in_app_browser",
    "memories",
    "plugins",
    "remote_plugin",
)

#: The permissions profile a skill run uses.
_PROFILE = "rote_skill_run"

#: ``mcp__<server>__<tool>`` and ``mcp__<server>__*``, as the callers write them.
_MCP_ID = re.compile(r"mcp__(?P<rest>.+)")
_READ_RULE = re.compile(r"Read\(/(?P<path>/.+)/\*\*\)")
_BASH_RULE = re.compile(r"Bash\((?P<command>[^()*]+?) \*\)")
#: The workspace rules every caller passes: the cwd is the workspace.
_WORKSPACE_RULES = frozenset({"Read(./**)", "Edit(./**)"})


def codex_binary() -> Path:
    """The real path of the ``codex`` binary on PATH (see the module note)."""
    found = shutil.which("codex")
    if found is None:
        raise EmpiricalError("`codex` CLI not found: install Codex, then run `codex login`")
    return Path(os.path.realpath(found))


def login_file(environ: Mapping[str, str] | None = None) -> Path:
    """The user's Codex login: ``$CODEX_HOME/auth.json``, else ``~/.codex/auth.json``."""
    env = os.environ if environ is None else environ
    home = env.get("CODEX_HOME") or str(Path.home() / ".codex")
    path = Path(home).expanduser() / "auth.json"
    if not path.is_file():
        raise EmpiricalError(
            f"no Codex login at {path}: run `codex login`. rote runs skills on your "
            "Codex login and needs it in auth.json (a login kept only in the OS "
            "credential store cannot be linked into the run's isolated home)"
        )
    return path


@dataclass(frozen=True)
class Gate:
    """The Claude lane's allowlist, translated: what the run may read, call and run."""

    #: Directories it may read besides the workspace.
    readable: tuple[Path, ...] = ()
    #: ``{server: tools}``; None lifts the read-only gate on that server.
    tools: dict[str, tuple[str, ...] | None] = field(default_factory=dict)
    #: Commands that run outside the sandbox (``rote skill run <name>``), as tools.
    commands: tuple[tuple[str, ...], ...] = ()


def translate_gate(allowed: Sequence[str], servers: Sequence[str]) -> Gate:
    """``allowed`` (the Claude lane's ``--allowedTools`` entries) as a :class:`Gate`.

    Server names are matched as Claude Code spells them inside a tool id, so a
    name with a character it replaces still resolves. An entry with no
    translation raises: a gate entry silently dropped is a gate silently
    changed.
    """
    from rote.eval.baseline import _wire_name

    by_wire = {_wire_name(s): s for s in servers}
    readable: list[Path] = []
    tools: dict[str, list[str] | None] = {}
    commands: list[tuple[str, ...]] = []
    for entry in allowed:
        if entry in _WORKSPACE_RULES:
            continue
        if read := _READ_RULE.fullmatch(entry):
            readable.append(Path(read["path"]))
            continue
        if bash := _BASH_RULE.fullmatch(entry):
            commands.append(tuple(shlex.split(bash["command"])))
            continue
        if mcp := _MCP_ID.fullmatch(entry):
            rest = mcp["rest"]
            wire = next(
                (w for w in sorted(by_wire, key=len, reverse=True) if rest.startswith(f"{w}__")),
                None,
            )
            if wire is None:
                raise EmpiricalError(f"the Codex lane cannot place {entry!r}: no wired server")
            server, tool = by_wire[wire], rest[len(wire) + 2 :]
            if tool == "*":
                tools[server] = None
            elif tools.get(server, []) is not None:
                tools.setdefault(server, [])
                current = tools[server]
                assert current is not None
                current.append(tool)
            continue
        raise EmpiricalError(f"the Codex lane has no translation for the gate entry {entry!r}")
    return Gate(
        readable=tuple(readable),
        tools={s: (None if t is None else tuple(t)) for s, t in tools.items()},
        commands=tuple(commands),
    )


def _toml(value: Any) -> str:
    """A TOML value for ``-c``: JSON's string, number and array forms are TOML's too."""
    return json.dumps(value)


#: A server name ``-c`` can carry. Codex splits a ``-c`` key on dots and keeps
#: TOML quotes as part of the name (measured: ``mcp_servers."x".command``
#: defines a server named ``"x"`` that never loads), so a name is written bare,
#: and one that would need quoting is refused.
_SERVER_NAME = re.compile(r"[A-Za-z0-9_-]+")


def _key(name: str) -> str:
    """A server name as a bare ``-c`` key segment."""
    if not _SERVER_NAME.fullmatch(name):
        raise EmpiricalError(
            f"the Codex lane cannot wire MCP server {name!r}: Codex reads a server name "
            "from a -c key only when it is letters, digits, '_' or '-'"
        )
    return name


def _deny_roots() -> list[str]:
    """The temp roots ``:minimal`` reads: the system temp dir and ``/tmp``."""
    roots = {os.path.realpath(tempfile.gettempdir()), os.path.realpath("/tmp")}
    return sorted(roots)


def server_config(name: str, entry: Mapping[str, Any], real_home: str) -> list[str]:
    """The ``-c`` overrides for one MCP server, from its Claude-lane entry."""
    prefix = f"mcp_servers.{_key(name)}"
    kind = entry.get("type")
    args: list[str] = []
    if kind == "stdio":
        args += ["-c", f"{prefix}.command={_toml(entry['command'])}"]
        args += ["-c", f"{prefix}.args={_toml(list(entry.get('args') or []))}"]
        if entry.get("env"):
            args += ["-c", f"{prefix}.env={_inline_table(entry['env'])}"]
        return args
    if kind != "http":
        raise EmpiricalError(
            f"the Codex lane reaches MCP servers over streamable HTTP or stdio; "
            f"{name!r} is {kind!r}"
        )
    args += ["-c", f"{prefix}.url={_toml(entry['url'])}"]
    if entry.get("headers"):
        args += ["-c", f"{prefix}.http_headers={_inline_table(entry['headers'])}"]
    if entry.get("headersHelper"):
        # The run's HOME is a throwaway; rote's helper reads the user's state.
        helper = f"/usr/bin/env HOME={shlex.quote(real_home)} {entry['headersHelper']}"
        args += ["-c", f"{prefix}.http_headers_helper={_toml(helper)}"]
    return args


def _inline_table(values: Mapping[str, Any]) -> str:
    """A TOML inline table: inside the value Codex parses TOML, so keys are quoted."""
    return "{" + ",".join(f"{json.dumps(k)}={_toml(v)}" for k, v in values.items()) + "}"


@dataclass(frozen=True)
class Prepared:
    """A ready ``codex exec``: its argv (the prompt goes on stdin) and env."""

    argv: list[str]
    env: dict[str, str]
    #: What the prompt must add for this lane (how to run the skill's driver), or "".
    note: str = ""


def _driver_server(
    command: tuple[str, ...], source_env: Mapping[str, str]
) -> tuple[str, dict[str, Any], str]:
    """``(server name, its stdio entry, the prompt note)`` for ``rote skill run <name>``."""
    import sys

    if len(command) != 4 or command[:3] != ("rote", "skill", "run"):
        raise EmpiricalError(
            f"the Codex lane can let only `rote skill run <name>` out of the sandbox, "
            f"not {' '.join(command)!r}"
        )
    name = command[3]
    rote = shutil.which("rote", path=source_env.get("PATH"))
    if rote is None:
        raise EmpiricalError("`rote` is not on the run's PATH, so its driver cannot be run")
    entry = {
        "type": "stdio",
        "command": sys.executable,
        "args": ["-I", "-m", "rote.eval.skill_run_server", name, rote],
        # The driver's own state paths, as the run sees them: no credential.
        "env": {
            k: v for k, v in source_env.items() if k.startswith("ROTE_") or k in ("PATH", "HOME")
        },
    }
    note = (
        f"In this environment the shell cannot run `rote skill run {name} ...`. Run it "
        f"through the `{skill_run_server.TOOL}` tool of the `{skill_run_server.SERVER}` "
        f"MCP server instead: pass every argument after `rote skill run {name}` as one "
        f"list item each (no shell quoting), and read its stdout as the command's output."
    )
    return skill_run_server.SERVER, entry, note


@contextmanager
def prepared(
    root: Path,
    *,
    workdir: Path,
    mcp_servers: Mapping[str, Mapping[str, Any]],
    allowed: Sequence[str],
    model: str | None,
    base_env: Mapping[str, str] | None = None,
    binary: Path | None = None,
) -> Iterator[Prepared]:
    """A ``codex exec`` inside the gate ``allowed`` describes, run from ``workdir``.

    ``root`` holds the throwaway homes; on exit a login Codex refreshed is
    written back to the user's own file. ``binary`` defaults to
    :func:`codex_binary`.
    """
    binary = binary if binary is not None else codex_binary()
    source_env = dict(os.environ if base_env is None else base_env)
    real_login = login_file(source_env)
    real_home = source_env.get("HOME") or str(Path.home())
    gate = translate_gate(allowed, list(mcp_servers))
    codex_home = root / "codex-home"
    user_home = root / "home"
    for d in (codex_home, user_home):
        d.mkdir(parents=True, exist_ok=True)
        d.chmod(0o700)
    link = codex_home / "auth.json"
    link.symlink_to(real_login)
    servers = dict(mcp_servers)
    tools = dict(gate.tools)
    notes: list[str] = []
    for command in gate.commands:
        server, driver_entry, note = _driver_server(command, source_env)
        servers[server] = driver_entry
        tools[server] = (skill_run_server.TOOL,)
        notes.append(note)
    filesystem = {":minimal": "read", str(binary.parent): "read"}
    filesystem |= {root: "deny" for root in _deny_roots()}
    filesystem |= {os.path.realpath(p): "read" for p in gate.readable}
    filesystem[":workspace_roots"] = "write"
    argv = [
        str(binary),
        "exec",
        "--json",
        "--ephemeral",
        "--skip-git-repo-check",
        "-C",
        str(workdir),
        "-c",
        'approval_policy="never"',
        "-c",
        'web_search="disabled"',
        *[arg for feature in FEATURES_OFF for arg in ("--disable", feature)],
        "-c",
        f"default_permissions={_toml(_PROFILE)}",
        "-c",
        f"permissions.{_PROFILE}.filesystem={_inline_table(filesystem)}",
    ]
    for name, entry in sorted(servers.items()):
        if name not in tools:
            continue  # the gate gives it nothing: not wired, as on the Claude lane
        argv += server_config(name, entry, real_home)
        prefix = f"mcp_servers.{_key(name)}"
        argv += ["-c", f'{prefix}.default_tools_approval_mode="approve"']
        allowed_tools = tools[name]
        if allowed_tools is not None:
            argv += ["-c", f"{prefix}.enabled_tools={_toml(list(allowed_tools))}"]
    if model:
        argv += ["--model", model]
    argv.append("-")
    env = {k: v for k, v in source_env.items() if k not in _CREDENTIAL_ENV}
    env |= {"CODEX_HOME": str(codex_home), "HOME": str(user_home)}
    try:
        yield Prepared(argv=argv, env=env, note="\n\n".join(notes))
    finally:
        _write_back_login(link, real_login)


def _write_back_login(link: Path, real: Path) -> None:
    """Keep a refreshed login: Codex replaced the link with a newer file.

    A refresh writes new tokens (and ``last_refresh``) back; written through
    the link it already landed in the real file. Written by replacing the
    link, it would otherwise be lost with the throwaway home, leaving the
    user's file holding a token the refresh may have rotated out.
    """
    if link.is_symlink() or not link.is_file():
        return
    try:
        fresh = json.loads(link.read_text(encoding="utf-8"))
        current = json.loads(real.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if str(fresh.get("last_refresh") or "") <= str(current.get("last_refresh") or ""):
        return
    staged = real.with_name(f".{real.name}.rote-{os.getpid()}")
    staged.write_text(json.dumps(fresh, indent=2), encoding="utf-8")
    staged.chmod(0o600)
    os.replace(staged, real)


@dataclass(frozen=True)
class Outcome:
    """What a ``codex exec --json`` run reported."""

    #: rote's four buckets (:func:`rote.compiler.drivers.codex.split_usage`), or None.
    usage: dict[str, int] | None
    #: ``{server, tool, arguments, result, error}`` per MCP call, in order.
    calls: tuple[dict[str, Any], ...]
    #: The final agent message.
    message: str | None
    #: A ``turn.failed`` or ``error`` event's message.
    error: str | None
    #: Shell commands the agent ran, in order.
    commands: tuple[str, ...]


def parse_events(lines: Sequence[str]) -> Outcome:
    """A run's usage, MCP calls and failure from its JSONL events.

    ``turn.completed`` carries the thread's running total (see
    :func:`rote.compiler.drivers.codex.session_usage`), so the last one is the
    run. A run that reported none has no figure, never zero.
    """
    from rote.compiler.drivers.codex import split_usage

    usage: dict[str, int] | None = None
    calls: list[dict[str, Any]] = []
    commands: list[str] = []
    message: str | None = None
    error: str | None = None
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        kind = event.get("type")
        if kind == "turn.completed" and isinstance(event.get("usage"), dict):
            usage = split_usage(event["usage"])
        elif kind == "turn.failed":
            failure = event.get("error")
            error = str(failure.get("message") if isinstance(failure, dict) else failure)
        elif kind == "error":
            error = str(event.get("message"))
        elif kind == "item.completed" and isinstance(event.get("item"), dict):
            item = event["item"]
            if item.get("type") == "mcp_tool_call":
                calls.append(
                    {
                        "server": item.get("server"),
                        "tool": item.get("tool"),
                        "arguments": item.get("arguments"),
                        "result": item.get("result"),
                        "error": item.get("error"),
                    }
                )
            elif item.get("type") == "command_execution":
                commands.append(str(item.get("command")))
            elif item.get("type") == "agent_message":
                message = str(item.get("text") or "")
    return Outcome(
        usage=usage,
        calls=tuple(calls),
        message=message,
        error=error,
        commands=tuple(commands),
    )
