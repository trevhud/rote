"""Shared introspection helpers for emitted DBOS apps.

Used anywhere rote needs to talk to an emitted app's system database
from outside the app process: the eval harness (gate signaling) and the
MCP auth-release path (``rote mcp login`` waking parked workflows).
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml


def dbos_system_database_url(app_dir: Path) -> str:
    """The system database URL the emitted DBOS app will actually use.

    Must mirror the emitted main.py's resolution order exactly —
    ``DBOS_SYSTEM_DATABASE_URL`` env override first, then the default
    SQLite file in the app dir — or messages get delivered to a
    database the app isn't watching.
    """
    override = os.environ.get("DBOS_SYSTEM_DATABASE_URL")
    if override:
        return override
    config = yaml.safe_load((app_dir / "dbos-config.yaml").read_text(encoding="utf-8"))
    name = config["name"] if isinstance(config, dict) and "name" in config else "pipeline"
    return f"sqlite:///{(app_dir / f'{name}.dbos.sqlite').resolve()}"
