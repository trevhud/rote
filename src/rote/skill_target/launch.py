"""``rote skill run <name> ...``: find the skill, check it, and run its driver isolated.

The order is the point:

1. resolve ``<name>`` to its install (:func:`rote.skill_target.trust.installed`);
2. check every file against the install record, reading each once and keeping
   the bytes, and check what the run depends on (state paths, proxy and CA
   variables, servers trusted read-only) against the record
   (:func:`rote.skill_target.trust.check`);
3. start rote's interpreter with ``-I`` (no ``PYTHON*`` variable, no user
   site, no cwd on the path), ``-B`` and :data:`BOOTSTRAP` as ``-c``, and hand
   it those checked bytes of ``scripts/`` (``plan.json`` included) over an
   inherited pipe (:func:`frame`). The driver and the extracted modules are
   imported from those bytes, so no copy of the running code exists on disk
   for a process of the same user to rewrite mid-run, and the skill dir is
   never read again. rote's site-packages stay on the path for fastmcp. The
   environment is built from an allowlist (:func:`child_env`).

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

import base64
import contextlib
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

#: The child's whole program, passed with ``-c``: never a file. argv: the read
#: end of a pipe rote holds the other end of, then the driver's own argv. One
#: JSON frame on the pipe carries the spec and every checked file of
#: ``scripts/`` (base64), and a meta-path finder serves the driver and the
#: extracted modules from those bytes. The skill's scripts dir is the modules'
#: ``__file__`` (so errors name file:line), and its ``sys.path`` entry is
#: disabled, so no import ever reads the dir. Stdlib only.
BOOTSTRAP = """\
import base64, importlib.abc, importlib.util, json, os, sys
from pathlib import Path
with os.fdopen(int(sys.argv[1]), "rb") as pipe:
    frame = json.loads(pipe.read())
spec, root = frame["spec"], frame["spec"]["scripts"]
files = {rel: base64.b64decode(data) for rel, data in frame["files"].items()}
dirs = {"/".join(rel.split("/")[:i]) for rel in files for i in range(1, rel.count("/") + 1)}
class Verified(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path=None, target=None):
        rel = name.replace(".", "/")
        if rel + "/__init__.py" in files:
            return importlib.util.spec_from_file_location(
                name, root + "/" + rel + "/__init__.py", loader=self,
                submodule_search_locations=[])
        if rel + ".py" in files:
            return importlib.util.spec_from_file_location(name, root + "/" + rel + ".py",
                                                          loader=self)
        if rel in dirs:
            return importlib.util.spec_from_loader(name, self, is_package=True)
        return None
    def create_module(self, spec):
        return None
    def exec_module(self, module):
        origin = module.__spec__.origin
        if origin and origin.startswith(root + "/"):
            source = files[origin[len(root) + 1:]]
            exec(compile(source, origin, "exec", dont_inherit=True), module.__dict__)
sys.meta_path.insert(0, Verified())
sys.path_importer_cache[root] = None
from _rote_skill import main
sys.exit(main(sys.argv[2:], Path(root), spec["command"], allow_writes=spec["allow_writes"],
              skill=spec["skill"], plan_text=files["plan.json"].decode("utf-8")))
"""


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


def frame(verified: Verified, *, allow_writes: bool) -> bytes:
    """The one JSON frame the bootstrap reads: the spec, and every checked file
    of ``scripts/`` as the bytes the check hashed."""
    skill = verified.skill
    files = {
        rel.removeprefix("scripts/"): base64.b64encode(content).decode("ascii")
        for rel, content in verified.scan.data.items()
        if rel.startswith("scripts/")
    }
    spec = {
        "scripts": str(skill.scripts),
        "command": f"rote skill run {skill.name}",
        "allow_writes": allow_writes,
        "skill": skill.name,
    }
    return json.dumps({"spec": spec, "files": files}).encode("utf-8")


def driver_argv(read_fd: int, argv: list[str]) -> list[str]:
    return [sys.executable, "-I", "-B", "-c", BOOTSTRAP, str(read_fd), *argv]


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


def private_ca_files(verified: Verified, into: Path) -> dict[str, str]:
    """A private copy of each recorded CA bundle, held to the recorded hash.

    A CA bundle has to be a file for the TLS stack to read, so unlike the
    code it lands on disk (a 0700 dir, a 0600 file). A process of the same
    user could still rewrite it mid-run: that residual is the CA's, and the
    code is never on disk at all."""
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
        fd = os.open(copy, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(content)
        ca_files[var] = str(copy)
    return ca_files


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
            ca_files = private_ca_files(verified, Path(private))
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
        read_fd, write_fd = os.pipe()
        try:
            proc = subprocess.Popen(
                driver_argv(read_fd, argv),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=out if _has_fileno(out) else subprocess.PIPE,
                stderr=err if _has_fileno(err) else subprocess.PIPE,
                text=True,
                pass_fds=(read_fd,),
            )
        finally:
            os.close(read_fd)
        with os.fdopen(write_fd, "wb") as pipe, contextlib.suppress(BrokenPipeError):
            # The child reads the whole frame before it prints anything, so
            # writing it all before reading its output cannot deadlock.
            pipe.write(frame(verified, allow_writes=allow_writes))
        captured_out, captured_err = proc.communicate()
    if captured_out:
        out.write(captured_out)
        out.flush()
    if captured_err:
        err.write(captured_err)
        err.flush()
    return proc.returncode


def _has_fileno(stream: TextIO) -> bool:
    try:
        stream.fileno()
    except (AttributeError, OSError, ValueError):
        return False
    return True
