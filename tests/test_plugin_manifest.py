"""Guard the Claude Code plugin manifests against version drift.

The plugin (`plugin/.claude-plugin/plugin.json`) and the marketplace
entry (`.claude-plugin/marketplace.json`) back the README's
"install from Claude Code" path. They carry their own ``version`` and
had silently drifted three minors behind the package (0.3.0 vs 0.6.0)
because nothing checked. These tests pin them to ``rote.__version__``
so a release bump that forgets the manifests fails CI instead of
shipping a stale-looking front door.
"""

from __future__ import annotations

import json
from pathlib import Path

from rote import __version__

REPO_ROOT = Path(__file__).resolve().parent.parent
PLUGIN_MANIFEST = REPO_ROOT / "plugin" / ".claude-plugin" / "plugin.json"
MARKETPLACE_MANIFEST = REPO_ROOT / ".claude-plugin" / "marketplace.json"


def test_plugin_manifest_version_matches_package() -> None:
    data = json.loads(PLUGIN_MANIFEST.read_text(encoding="utf-8"))
    assert data["version"] == __version__, (
        f"plugin.json version {data['version']!r} != rote.__version__ "
        f"{__version__!r} — bump plugin/.claude-plugin/plugin.json on release."
    )


def test_marketplace_manifest_version_matches_package() -> None:
    data = json.loads(MARKETPLACE_MANIFEST.read_text(encoding="utf-8"))
    plugins = data["plugins"]
    rote_entry = next(p for p in plugins if p["name"] == "rote")
    assert rote_entry["version"] == __version__, (
        f"marketplace.json rote plugin version {rote_entry['version']!r} != "
        f"rote.__version__ {__version__!r} — bump .claude-plugin/marketplace.json on release."
    )
