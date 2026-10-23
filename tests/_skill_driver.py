"""Build a compiled skill's scripts dir and run its driver, for the driver tests.

The scripts dir holds what the skill target ships: ``plan.json`` from
:func:`rote.plan.build_plan` (live), the verbatim helpers
(:func:`rote.skill_target.write_scripts`), and ``extracted/`` modules.
:func:`drive` runs the driver in this process, the way ``run.py`` would,
with the driver and everything it imports loaded fresh from that dir, so
one test's ``extracted`` package never answers for another's.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import re
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from rote.ir import Pipeline
from rote.plan import build_plan
from rote.skill_target import write_scripts

_FRESH = ("_rote_skill", "_rote_schema", "_rote_mcp")
SERVER_SCRIPT = Path(__file__).resolve().parent / "_skill_mcp_server.py"


def make_skill(
    root: Path, pipeline: dict[str, Any] | Pipeline, modules: dict[str, str] | None = None
) -> Path:
    """A scripts dir for ``pipeline``, with ``modules`` (``{"math.py": source}``) in extracted/."""
    ir = pipeline if isinstance(pipeline, Pipeline) else Pipeline.model_validate(pipeline)
    scripts = root / "scripts"
    write_scripts(scripts, build_plan(ir, mcp="live"))
    extracted = scripts / "extracted"
    extracted.mkdir(exist_ok=True)
    (extracted / "__init__.py").write_text("", encoding="utf-8")
    for name, source in (modules or {}).items():
        (extracted / name).write_text(source, encoding="utf-8")
    return scripts


def load_driver(scripts: Path, patch: dict[str, Any] | None = None) -> Any:
    for name in list(sys.modules):
        if name in _FRESH or name == "extracted" or name.startswith("extracted."):
            del sys.modules[name]
    spec = importlib.util.spec_from_file_location("_rote_skill", scripts / "_rote_skill.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_rote_skill"] = module
    spec.loader.exec_module(module)
    for key, value in (patch or {}).items():
        setattr(module, key, value)
    return module


def drive(scripts: Path, *argv: str, patch: dict[str, Any] | None = None) -> tuple[int, str]:
    """Run one driver command in-process: ``(exit status, stdout)``."""
    saved_path, saved_env = list(sys.path), dict(os.environ)
    try:
        module = load_driver(scripts, patch)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            status = module.main(list(argv), scripts)
        return status, out.getvalue()
    finally:
        sys.path[:] = saved_path
        os.environ.clear()
        os.environ.update(saved_env)


def header(text: str) -> dict[str, str]:
    """The first line's fields: ``{"kind": ..., "run": ..., "step": ...}``.

    The block's own ``kind=gate`` / ``kind=login`` lands under ``"of"``."""
    first = text.splitlines()[0].split()
    assert first[0] == "ROTE/1", text
    fields = {"kind": first[1], "skill": first[2]}
    for part in first[3:]:
        key, _, value = part.partition("=")
        fields["of" if key == "kind" else key] = value
    return fields


def run_dir(text: str) -> Path:
    from rote.skill_target._driver import run_root

    fields = header(text)
    return run_root() / fields["skill"] / fields["run"]


def journal(text: str) -> dict[str, Any]:
    state: dict[str, Any] = json.loads((run_dir(text) / "run.json").read_text(encoding="utf-8"))
    return state


def result_of(text: str) -> Any:
    match = re.search(r"^result: (\S+) \(", text, re.M)
    assert match, text
    return json.loads(Path(match.group(1)).read_text(encoding="utf-8"))


class FixtureServer:
    """``tests/_skill_mcp_server.py`` on a free localhost port, with its hit log."""

    def __init__(self, hits: Path) -> None:
        self.hits = hits
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}/mcp"
        self.proc = subprocess.Popen(
            [sys.executable, str(SERVER_SCRIPT), str(port), str(hits)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            with contextlib.suppress(OSError), socket.create_connection(("127.0.0.1", port), 0.2):
                return
            if self.proc.poll() is not None:
                raise RuntimeError(self.proc.stderr.read().decode() if self.proc.stderr else "")
            time.sleep(0.05)
        raise RuntimeError("the fixture MCP server did not start")

    def reset(self) -> None:
        self.hits.write_text("", encoding="utf-8")

    def calls(self, tool: str | None = None) -> list[dict[str, Any]]:
        if not self.hits.exists():
            return []
        rows = [json.loads(line) for line in self.hits.read_text().splitlines() if line]
        return [r for r in rows if tool is None or r["tool"] == tool]

    def close(self) -> None:
        self.proc.terminate()
        self.proc.wait(timeout=10)


@contextlib.contextmanager
def fixture_server(hits: Path) -> Iterator[FixtureServer]:
    server = FixtureServer(hits)
    try:
        yield server
    finally:
        server.close()
