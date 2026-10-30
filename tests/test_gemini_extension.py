"""Guard the Gemini CLI extension manifest against version drift.

`gemini-extension.json` is crawled by the Gemini CLI extension gallery and
carries its own ``version``, like the Claude Code plugin manifests
(`tests/test_plugin_manifest.py`). Pinning it to ``rote.__version__`` makes a
release bump that forgets it fail the fast suite instead of the tag build.
"""

from __future__ import annotations

import json
from pathlib import Path

from rote import __version__

REPO_ROOT = Path(__file__).resolve().parent.parent
GEMINI_MANIFEST = REPO_ROOT / "gemini-extension.json"


def test_gemini_extension_version_matches_package() -> None:
    data = json.loads(GEMINI_MANIFEST.read_text(encoding="utf-8"))
    assert data["version"] == __version__, (
        f"gemini-extension.json version {data['version']!r} != rote.__version__ "
        f"{__version__!r}; bump gemini-extension.json on release."
    )


def test_gemini_extension_launches_the_serve_extra() -> None:
    data = json.loads(GEMINI_MANIFEST.read_text(encoding="utf-8"))
    server = data["mcpServers"]["rote"]
    assert server["command"] == "uvx"
    assert server["args"] == ["--from", "rote-cli[serve]", "rote", "serve"]
