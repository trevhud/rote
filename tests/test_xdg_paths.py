"""One rule for the XDG dirs, in the CLI, the Python helpers and the TS helper.

rote mcp login writes the token store and the emitted pipeline reads it,
so the two must resolve the same directory under every environment. An
empty XDG variable is unset (the XDG Base Directory spec): ``??`` in the TS
helper read ``XDG_DATA_HOME=`` as the base, so a pipeline looked for its
tokens in ``./rote/mcp-tokens`` while the CLI wrote them under the home.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from rote import xdg
from rote.adapters._ts_common import ROTE_MCP_HELPER_TS
from rote.app_registry import apps_path
from rote.cloud_auth import credential_path
from rote.config import user_config_path
from rote.mcp import _runtime_helper as mcp_helper
from rote.mcp.registry import registry_path
from rote.mcp.tokens import token_dir

_XDG = ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME")


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for key in (
        "ROTE_APPS_PATH",
        "ROTE_CLOUD_CRED_PATH",
        "ROTE_CONFIG_PATH",
        "ROTE_MCP_CONFIG",
        "ROTE_MCP_TOKEN_DIR",
    ):
        monkeypatch.delenv(key, raising=False)
    return tmp_path / "home"


@pytest.mark.parametrize("value", ["", None])
def test_an_empty_xdg_variable_is_unset(
    value: str | None, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for key in _XDG:
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    assert xdg.config_home() == home / ".config"
    assert xdg.data_home() == home / ".local" / "share"
    assert xdg.cache_home() == home / ".cache"


@pytest.mark.parametrize(
    ("resolve", "base", "rel"),
    [
        (apps_path, "XDG_DATA_HOME", "rote/apps.json"),
        (credential_path, "XDG_DATA_HOME", "rote/cloud.json"),
        (user_config_path, "XDG_CONFIG_HOME", "rote/config.yaml"),
        (registry_path, "XDG_CONFIG_HOME", "rote/mcp.json"),
        (token_dir, "XDG_DATA_HOME", "rote/mcp-tokens"),
        (mcp_helper._token_dir, "XDG_DATA_HOME", "rote/mcp-tokens"),
    ],
)
def test_every_per_user_path_follows_the_one_rule(
    resolve: Callable[[], Path],
    base: str,
    rel: str,
    home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Set, the variable is the base; empty, the home default is. The CLI's
    paths and the emitted Python helper's copy agree in both cases."""
    monkeypatch.setenv(base, str(tmp_path / "xdg"))
    assert resolve() == tmp_path / "xdg" / rel
    monkeypatch.setenv(base, "")
    default = {"XDG_DATA_HOME": home / ".local" / "share", "XDG_CONFIG_HOME": home / ".config"}
    assert resolve() == default[base] / rel


def _ts_function(name: str) -> str:
    match = re.search(rf"^function {name}\(.*?^\}}$", ROTE_MCP_HELPER_TS, re.S | re.M)
    assert match, name
    return match.group(0)


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("node") is None, reason="needs node to run the TS helper")
def test_the_ts_helper_reads_the_store_the_cli_writes_when_xdg_is_empty(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The emitted Node helper's tokenDir and registry lookup, run for real
    with XDG_* and ROTE_MCP_CONFIG set but empty, land where the CLI does."""
    for key in ("XDG_CONFIG_HOME", "XDG_DATA_HOME"):
        monkeypatch.setenv(key, "")
    monkeypatch.setenv("ROTE_MCP_CONFIG", "")
    registry = registry_path()
    registry.parent.mkdir(parents=True)
    registry.write_text(json.dumps({"servers": {"docs": {"url": "https://example.test/mcp"}}}))
    script = tmp_path / "paths.ts"
    script.write_text(
        'import fs from "node:fs";\nimport os from "node:os";\nimport path from "node:path";\n'
        "type RegistryEntry = { url?: string };\n"
        f"{_ts_function('registryServers')}\n{_ts_function('tokenDir')}\n"
        "const servers = Object.keys(registryServers());\n"
        "console.log(JSON.stringify({ tokens: tokenDir(), servers }));\n"
    )
    run = subprocess.run(
        ["node", str(script)], capture_output=True, text=True, timeout=60, cwd=tmp_path
    )
    assert run.returncode == 0, run.stderr
    seen = json.loads(run.stdout)
    assert seen == {"tokens": str(token_dir()), "servers": ["docs"]}
