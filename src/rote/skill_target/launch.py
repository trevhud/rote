"""``rote skill run <name> ...``: find the skill, check it, and run its driver isolated.

The order is the point:

1. resolve ``<name>`` to its install (:func:`rote.skill_target.trust.installed`);
2. check every file against the install record, reading each once and keeping
   the bytes, and check what the run depends on (state paths, proxy and CA
   variables, servers trusted read-only) against the record
   (:func:`rote.skill_target.trust.check`);
3. write those checked bytes of ``scripts/`` into a private 0700 temp dir
   (:func:`snapshot_scripts`) and run the driver from there, never from the
   skill dir, so a file swapped in after the check is never read;
4. start the copy's ``_rote_skill.py`` with rote's interpreter, ``-I`` (no
   ``PYTHON*`` variable, no user site, no cwd on the path) and ``-B``, the
   scripts dir first on the path and rote's site-packages after it, and an
   environment built from an allowlist (:func:`child_env`). ``-B`` is
   defense in depth now that the copy is deleted after the run.

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
- the state paths, set by rote to the values the check just held to the
  install record (``ROTE_MCP_CONFIG``, ``ROTE_MCP_TOKEN_DIR``,
  ``ROTE_SKILL_RUN_DIR``, and ``ROTE_CLOUD_CRED_PATH`` for a helper skill);
- the proxy variables as the install recorded them, and each recorded CA
  bundle as rote's private copy of the verified bytes;
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
go), and ``XDG_*`` (rote passes the paths themselves). A dropped
``ROTE_MCP_*_URL`` is named on stderr, because a user who set one expects it
to work.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TextIO

from rote.skill_target.trust import SkillTrustError, Verified

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
    from rote.skill_target.trust import plan_servers

    used = plan_servers(plan)
    names: set[str] = set()
    for server, config in load_registry().servers.items():
        if server in used:
            for value in (config.headers or {}).values():
                names.update(HEADER_REFERENCE.findall(value))
    return names


def child_env(
    environ: Mapping[str, str],
    plan: Mapping[str, Any],
    bound: Mapping[str, Any],
    *,
    ca_files: Mapping[str, str] | None = None,
) -> tuple[dict[str, str], list[str]]:
    """``(the driver's environment, the endpoint overrides that were dropped)``.

    ``bound`` is what the check held to the install record
    (:func:`rote.skill_target.trust.bindings`): the state paths, and the proxy
    variables, passed as recorded. ``ca_files`` maps each recorded CA
    variable to the private copy of the bundle the check verified."""
    allowed = set(BASE_VARS) | _header_vars(plan)
    if plan.get("judges") == "helper":
        allowed |= HELPER_KEYS
        judges = [s["id"].upper() for s in plan.get("steps") or [] if s.get("kind") == "judge"]
        allowed |= {f"{prefix}{judge}" for prefix in HELPER_KNOBS for judge in judges}
    env = {k: v for k, v in environ.items() if k in allowed or k.startswith("LC_")}
    env["ROTE_MCP_CONFIG"] = bound["mcp_config"]
    env["ROTE_MCP_TOKEN_DIR"] = bound["token_dir"]
    env["ROTE_SKILL_RUN_DIR"] = bound["run_dir"]
    if plan.get("judges") == "helper":
        env["ROTE_CLOUD_CRED_PATH"] = bound["cloud_cred"]
    env.update(bound.get("proxy") or {})
    env.update(ca_files or {})
    dropped = sorted(k for k in environ if _ENDPOINT_OVERRIDE.fullmatch(k) and k not in env)
    return env, dropped


def driver_argv(scripts: Path, name: str, argv: list[str], *, allow_writes: bool) -> list[str]:
    spec = {
        "scripts": str(scripts),
        "command": f"rote skill run {name}",
        "allow_writes": allow_writes,
        "skill": name,
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


def snapshot_scripts(verified: Verified, into: Path) -> tuple[Path, dict[str, str]]:
    """Write the checked bytes of ``scripts/`` (and the recorded CA bundles) into
    ``into``, a private dir rote just made; return the scripts dir and the CA copies.

    The run starts from these, never from the skill dir: a file swapped there
    after the check (while the first MCP call is still importing fastmcp, say)
    is never read."""
    scripts = into / "scripts"
    for rel, content in verified.scan.data.items():
        if not rel.startswith("scripts/"):
            continue
        target = into / rel
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(content)
    ca_files: dict[str, str] = {}
    for var, recorded in (verified.bound.get("ca") or {}).items():
        path, _, digest = recorded.rpartition(" sha256:")
        content = Path(path).read_bytes()
        if hashlib.sha256(content).hexdigest() != digest:
            raise SkillTrustError(
                f"the CA bundle {path} ({var}) changed while the run was being checked.",
                f"ask the user to review it and run: rote skill trust {verified.skill.name}",
            )
        copy = into / f"ca-{var}.pem"
        copy.write_bytes(content)
        ca_files[var] = str(copy)
    return scripts, ca_files


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
    with tempfile.TemporaryDirectory(prefix="rote-skill-") as private:
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
            verified = trust.check(app, environ)
            allow_writes = trust.write_consent(app, verified.scan)
            scripts, ca_files = snapshot_scripts(verified, Path(private))
            env, dropped = child_env(environ, verified.plan, verified.bound, ca_files=ca_files)
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
            driver_argv(scripts, verified.skill.name, argv, allow_writes=allow_writes),
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
