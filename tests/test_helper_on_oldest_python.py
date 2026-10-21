"""The emitted judge helper imports and runs on the oldest Python emitted apps support.

``rote.inference._runtime_helper`` is copied verbatim into every emitted
Python app as ``signatures/_rote_inference.py``. rote needs 3.11, but those
apps declare no floor and their runtimes (DBOS, Temporal, a plain script)
run on 3.10, so a 3.11-only name in the helper loses every judge at import.
That happened once (``from datetime import UTC``), and no syntax check can
see it: an import name is not syntax. So this runs the file under 3.10.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from rote.inference import _runtime_helper

#: The oldest Python an emitted app is expected to run on.
EMITTED_PYTHON_FLOOR = "3.10"

_PROBE = """
import importlib.util, sys
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
spec = importlib.util.spec_from_file_location("_rote_inference", sys.argv[1])
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
when = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=20), usegmt=True)
print(sys.version_info[:2], helper._retry_after_s({"retry-after": when}))
"""


def _oldest_python() -> str | None:
    uv = shutil.which("uv")
    if uv is None:
        return None
    found = subprocess.run(
        [uv, "python", "find", "--no-project", EMITTED_PYTHON_FLOOR],
        capture_output=True,
        text=True,
    )
    if found.returncode != 0:
        return None
    return found.stdout.strip() or None


def test_the_emitted_helper_runs_on_the_oldest_supported_python(tmp_path: Path) -> None:
    python = _oldest_python()
    if python is None:
        if os.environ.get("CI"):
            pytest.fail(
                f"CI must install Python {EMITTED_PYTHON_FLOOR}: "
                "add the .github/actions/oldest-python step"
            )
        pytest.skip(
            f"no Python {EMITTED_PYTHON_FLOOR} here (uv python install {EMITTED_PYTHON_FLOOR})"
        )
    helper = tmp_path / "_rote_inference.py"
    helper.write_text(Path(_runtime_helper.__file__).read_text("utf-8"), "utf-8")
    # -I: the copy alone, as an emitted app has it, with no rote on the path.
    run = subprocess.run(
        [python, "-I", "-c", _PROBE, str(helper)], capture_output=True, text=True, cwd=tmp_path
    )
    assert run.returncode == 0, run.stderr[-2000:]
    version, waited = run.stdout.split(") ")
    assert version == f"({EMITTED_PYTHON_FLOOR.replace('.', ', ')}"
    assert 18 <= float(waited) <= 20
