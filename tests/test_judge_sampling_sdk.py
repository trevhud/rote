"""A judge's ``temperature`` reaches the API on every anthropic SDK rote allows.

anthropic 1.0 removed ``temperature`` / ``top_p`` / ``top_k`` from
``messages.create``, so the helper's ``**sampling`` raised ``TypeError``
before any request, on every judge whose IR sets a temperature. anthropic
2.0 then moved to API version 2026-10-21, which answers those parameters
with a 400 even in the request body. Nothing caught either: the judge
tests used no sampling, and the stubs accepted any body under any version.

These run the real SDK against a localhost stub that refuses what the API
refuses (``tests/_messages_api.py``), so no inference is spent.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

from rote.inference import _runtime_helper as helper
from tests._messages_api import reject_like_the_api, send_json, structured_message

#: A model that still honours a non-default temperature (the migration
#: guide's example). The current Sonnet rejects one on any API version.
SAMPLING_MODEL = "claude-sonnet-4-6"

_GRADE = {
    "type": "object",
    "properties": {"grade": {"type": "integer"}},
    "required": ["grade"],
    "additionalProperties": False,
}


class _Stub(BaseHTTPRequestHandler):
    seen: list[tuple[str | None, dict[str, Any]]] = []

    def do_POST(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        payload = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
        type(self).seen.append((self.headers.get("anthropic-version"), payload))
        if reject_like_the_api(self, payload):
            return
        send_json(self, structured_message(payload, {"grade": 9}))

    def log_message(self, *args: object) -> None:
        return


@pytest.fixture()
def stub(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    monkeypatch.setenv("ROTE_INFERENCE", "api")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    _Stub.seen = []
    server = HTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


def _judge(stub: str, sampling: dict[str, Any] | None) -> dict[str, Any]:
    return helper.call_judge(
        node_id="grade_essay",
        client="anthropic",
        model=SAMPLING_MODEL,
        base_url=stub,
        prompt="Grade it.",
        output_schema=_GRADE,
        sampling=sampling,
    )


def test_a_judge_temperature_reaches_the_wire_on_the_installed_sdk(stub: str) -> None:
    assert _judge(stub, {"temperature": 0.2}) == {"grade": 9}
    [(version, request)] = _Stub.seen
    assert request["temperature"] == 0.2
    assert version == helper._SAMPLING_API_VERSION == "2023-06-01"


def test_a_judge_without_sampling_sends_none_and_keeps_the_sdks_version(stub: str) -> None:
    import anthropic

    assert _judge(stub, None) == {"grade": 9}
    [(version, request)] = _Stub.seen
    assert not {"temperature", "top_p", "top_k"} & request.keys()
    assert "extra_body" not in request
    # The SDK's own default header, whatever this SDK sends.
    client = anthropic.Anthropic(api_key="x", base_url=stub)
    assert version == client.default_headers["anthropic-version"]


# ───────── every SDK rote allows: the floor and the newest release ─────────

_PROBE = """
import importlib.util, json, sys
sys.path.insert(0, sys.argv[2])
import anthropic
spec = importlib.util.spec_from_file_location("_rote_inference", sys.argv[1])
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
answer = helper.call_judge(
    node_id="grade_essay", client="anthropic", model=sys.argv[4], base_url=sys.argv[3],
    prompt="Grade it.", output_schema=json.loads(sys.argv[5]),
    sampling={"temperature": 0.2},
)
print(json.dumps({"sdk": anthropic.__version__, "answer": answer}))
"""

#: (pip spec, label). The floor is pyproject's ``anthropic>=1.15``; the
#: unpinned spec is what a fresh install of an emitted app gets today.
_SDKS = [("anthropic==1.15.0", "floor"), ("anthropic", "newest")]


@pytest.mark.slow
@pytest.mark.parametrize(("spec", "label"), _SDKS, ids=[label for _, label in _SDKS])
def test_a_judge_temperature_reaches_the_wire_on_every_allowed_sdk(
    stub: str, tmp_path: Path, spec: str, label: str
) -> None:
    uv = shutil.which("uv")
    if uv is None:
        if os.environ.get("CI"):
            pytest.fail("CI must have uv on PATH")
        pytest.skip("uv not found")
    sdk = tmp_path / "sdk"
    install = subprocess.run(
        [uv, "pip", "install", "--quiet", "--python", sys.executable, "--target", str(sdk), spec],
        capture_output=True,
        text=True,
    )
    assert install.returncode == 0, install.stderr[-2000:]
    copy = tmp_path / "_rote_inference.py"
    copy.write_text(Path(helper.__file__).read_text("utf-8"), "utf-8")
    env = {
        "PATH": os.environ.get("PATH", ""),
        "ROTE_INFERENCE": "api",
        "ANTHROPIC_API_KEY": "test-key-not-real",
    }
    run = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            _PROBE,
            str(copy),
            str(sdk),
            stub,
            SAMPLING_MODEL,
            json.dumps(_GRADE),
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
    )
    assert run.returncode == 0, run.stderr[-2000:]
    result = json.loads(run.stdout.strip().splitlines()[-1])
    assert result["answer"] == {"grade": 9}
    if label == "floor":
        assert result["sdk"] == "1.15.0"
    else:
        # 2.0 is the first release on API version 2026-10-21.
        assert int(result["sdk"].split(".")[0]) >= 2
    [(version, request)] = _Stub.seen
    assert request["temperature"] == 0.2
    assert version == "2023-06-01"
