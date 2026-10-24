"""``rote skill run <name> ...``: find the skill, check it, and run its driver isolated.

The order is the point:

1. resolve ``<name>`` to its install (:func:`rote.skill_target.trust.installed`);
2. check every file against the install record, and the state paths against
   the ones it was made with (:func:`rote.skill_target.trust.check`);
3. start the skill's own ``_rote_skill.py`` with rote's interpreter, ``-I``
   (no ``PYTHON*`` variable, no user site, no cwd on the path) and ``-B`` (no
   ``__pycache__`` written into a dir the next check would then refuse), the
   scripts dir first on the path and rote's site-packages after it, and an
   environment built from an allowlist (:func:`child_env`).

The driver is started through :data:`BOOTSTRAP`, not ``scripts/run.py``,
because the write consent is a parameter of ``_rote_skill.main`` and nothing
else: an argv flag or a variable would be one an agent could pass to ``uv
run scripts/run.py`` as well. ``run.py`` stays the read-only way in without
rote.

The environment is an allowlist, because the agent's shell sets it and a
scrub list only stops the variables someone thought of. What reaches the
driver:

- the process basics (``PATH``, ``HOME``, the user and locale variables,
  ``TMPDIR``);
- the three state paths, set by rote to the values the check just held to
  the install record (``ROTE_MCP_CONFIG``, ``ROTE_MCP_TOKEN_DIR``,
  ``ROTE_SKILL_RUN_DIR``);
- the variables the MCP registry's own header templates name (``${NAME}``)
  for the servers this plan calls, since only the registry, which is bound,
  can name them;
- for a ``--judges helper`` skill, who pays: the vendor keys and the
  ``ROTE_INFERENCE`` / ``ROTE_MODEL_`` / ``ROTE_EFFORT_`` /
  ``ROTE_TEMPERATURE_`` knobs for its own judges.

Everything else is dropped, notably what redirects a run or widens it:
``ROTE_MCP_<SERVER>_URL`` (another endpoint, outside the read-only
declaration), ``ROTE_MCP_READ_ONLY*`` (the driver sets those from the
consent), ``ROTE_SKILL_COMMAND`` (the next command every block prints),
``ROTE_BASE_URL_<ID>``, ``ANTHROPIC_BASE_URL``, ``OPENAI_BASE_URL``,
``ROTE_CLOUD_URL`` / ``ROTE_CLOUD_TOKEN`` (where a judge's payload and key
go), ``XDG_*`` (rote passes the paths themselves), and the proxy and CA
variables (``HTTPS_PROXY``, ``SSL_CERT_FILE``...), which together route a
bearer token through someone else's TLS. A dropped ``ROTE_MCP_*_URL`` is
named on stderr, because a user who set one expects it to work.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping
from typing import Any, TextIO

from rote.skill_target.trust import SkillDir, SkillTrustError

#: Starts the driver with the consent as a parameter. argv: the spec (JSON),
#: then the driver's own argv. Imports only the stdlib and the skill's
#: ``_rote_skill``, from the scripts dir it puts first.
BOOTSTRAP = (
    "import json, sys\n"
    "from pathlib import Path\n"
    "spec = json.loads(sys.argv[1])\n"
    "scripts = Path(spec['scripts'])\n"
    "sys.path.insert(0, str(scripts))\n"
    "from _rote_skill import main\n"
    "sys.exit(main(sys.argv[2:], scripts, spec['command'], "
    "allow_writes=spec['allow_writes'], skill=spec['skill']))\n"
)

#: Passed through when set: what any process needs to start and find its tools.
BASE_VARS = frozenset(
    {"PATH", "HOME", "USER", "LOGNAME", "LANG", "LANGUAGE", "TZ", "TMPDIR", "SYSTEMROOT"}
)
#: The vendor keys a ``--judges helper`` skill's judges pay with.
HELPER_KEYS = frozenset(
    {
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "OPENAI_API_KEY",
        "ROTE_INFERENCE",
        "ROTE_INFERENCE_TIMEOUT",
        "ROTE_JUDGE_EFFORT",
    }
)
#: Per-judge knobs a helper skill honors, ``<prefix><JUDGE ID>``.
HELPER_KNOBS = ("ROTE_INFERENCE_", "ROTE_MODEL_", "ROTE_EFFORT_", "ROTE_TEMPERATURE_")
_ENDPOINT_OVERRIDE = re.compile(r"ROTE_MCP_[A-Z0-9_]+_URL")


def _header_vars(plan: Mapping[str, Any]) -> set[str]:
    """The variables the registry's header templates name for this plan's servers."""
    from rote.mcp._runtime_helper import HEADER_REFERENCE
    from rote.mcp.registry import load_registry

    used: set[str] = set()
    for step in plan.get("steps") or []:
        if step.get("kind") == "mcp":
            used.add(str(step["server"]))
        for body in step.get("body") or []:
            if body.get("kind") == "mcp":
                used.add(str(body["server"]))
        used.update(str(s) for s in (step.get("tool_servers") or {}).values())
    names: set[str] = set()
    for server, config in load_registry().servers.items():
        if server in used:
            for value in (config.headers or {}).values():
                names.update(HEADER_REFERENCE.findall(value))
    return names


def child_env(
    environ: Mapping[str, str], plan: Mapping[str, Any], bound: Mapping[str, str]
) -> tuple[dict[str, str], list[str]]:
    """``(the driver's environment, the endpoint overrides that were dropped)``.

    ``bound`` is the state paths the check held to the install record."""
    allowed = set(BASE_VARS) | _header_vars(plan)
    knobs: tuple[str, ...] = ()
    if plan.get("judges") == "helper":
        allowed |= HELPER_KEYS
        judges = [s["id"].upper() for s in plan.get("steps") or [] if s.get("kind") == "judge"]
        knobs = tuple(f"{prefix}{judge}" for prefix in HELPER_KNOBS for judge in judges)
        allowed |= set(knobs)
    env = {k: v for k, v in environ.items() if k in allowed or k.startswith("LC_")}
    env["ROTE_MCP_CONFIG"] = bound["mcp_config"]
    env["ROTE_MCP_TOKEN_DIR"] = bound["token_dir"]
    env["ROTE_SKILL_RUN_DIR"] = bound["run_dir"]
    if plan.get("judges") == "helper":
        from rote.cloud_auth import credential_path

        env["ROTE_CLOUD_CRED_PATH"] = str(credential_path())
    dropped = sorted(k for k in environ if _ENDPOINT_OVERRIDE.fullmatch(k) and k not in env)
    return env, dropped


def driver_argv(skill: SkillDir, argv: list[str], *, allow_writes: bool) -> list[str]:
    spec = {
        "scripts": str(skill.scripts),
        "command": f"rote skill run {skill.name}",
        "allow_writes": allow_writes,
        "skill": skill.name,
    }
    return [sys.executable, "-I", "-B", "-c", BOOTSTRAP, json.dumps(spec), *argv]


def error_block(name: str, error: SkillTrustError) -> tuple[str, int]:
    """The driver's ERROR block for a refusal rote makes before the driver runs."""
    from rote.skill_target._driver import Block

    lines = [
        error.cause,
        f"Fix: {error.fix}",
        "Fix the command and run it again."
        if error.usage
        else "Tell the user what this says. Do not do the skill's steps yourself.",
    ]
    block = Block("ERROR", name, lines, 2 if error.usage else 1)
    return block.text(), block.status


def run(
    name: str,
    argv: list[str],
    *,
    environ: Mapping[str, str] | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Resolve, check and run one driver command; the driver's exit status."""
    from rote.app_registry import registered_apps
    from rote.skill_target import trust

    environ = os.environ if environ is None else environ
    out = stdout if stdout is not None else sys.stdout
    err = stderr if stderr is not None else sys.stderr
    try:
        try:
            apps = registered_apps()
        except (OSError, ValueError) as e:
            from rote.app_registry import apps_path

            raise SkillTrustError(
                f"the app registry {apps_path()} cannot be read: {e}.",
                "fix or remove that file, then install the skill again",
            ) from None
        app = trust.installed(name, apps)
        skill = trust.check(app)
        allow_writes = trust.write_consent(app, skill)
        bound = trust.bindings()
        env, dropped = child_env(environ, skill.plan(), bound)
    except SkillTrustError as e:
        text, status = error_block(name, e)
        out.write(text)
        out.flush()
        return status
    for var in dropped:
        print(
            f"rote: ignored {var}: a skill run takes endpoints from the MCP registry only "
            "(rote mcp add <server> <url>)",
            file=err,
            flush=True,
        )
    out.flush()
    proc = subprocess.run(
        driver_argv(skill, argv, allow_writes=allow_writes),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=out if _has_fileno(out) else subprocess.PIPE,
        stderr=err if _has_fileno(err) else subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.stdout is not None:
        out.write(proc.stdout)
        out.flush()
    if proc.stderr is not None:
        err.write(proc.stderr)
        err.flush()
    return proc.returncode


def _has_fileno(stream: TextIO) -> bool:
    try:
        stream.fileno()
    except (AttributeError, OSError, ValueError):
        return False
    return True
