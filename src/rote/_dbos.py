"""Shared introspection helpers for emitted DBOS apps.

Used anywhere rote needs to talk to an emitted app's system database
from outside the app process: the eval harness (gate signaling) and the
MCP auth-release path (``rote mcp login`` waking parked workflows).
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml


def _load_dbos_config(app_dir: Path) -> dict[str, object]:
    config = yaml.safe_load((app_dir / "dbos-config.yaml").read_text(encoding="utf-8"))
    return config if isinstance(config, dict) else {}


def dbos_system_database_url(app_dir: Path) -> str:
    """The system database URL an emitted DBOS *Python* app will use.

    Must mirror the emitted main.py's resolution order exactly —
    ``DBOS_SYSTEM_DATABASE_URL`` env override first, then the default
    SQLite file in the app dir — or messages get delivered to a
    database the app isn't watching.
    """
    override = os.environ.get("DBOS_SYSTEM_DATABASE_URL")
    if override:
        return override
    config = _load_dbos_config(app_dir)
    name = str(config.get("name", "pipeline"))
    return f"sqlite:///{(app_dir / f'{name}.dbos.sqlite').resolve()}"


def dbos_ts_system_database_url(app_dir: Path) -> str:
    """The system database URL an emitted DBOS *TypeScript* app will use.

    Mirrors the TS SDK's resolution (verified against the shipped
    ``dist/src/config.js`` of SDK 5.3.0): the emitted main.ts passes
    ``process.env.DBOS_SYSTEM_DATABASE_URL`` to ``DBOS.setConfig``;
    otherwise the SDK default: Postgres at ``PGHOST``/``PGPORT`` with
    ``PGUSER``/``PGPASSWORD`` (falling back to
    ``postgres:dbos@localhost:5432``) and database ``<name>_dbos_sys``.

    dbos-config.yaml contributes only the app ``name``. Since v5 the SDK
    no longer reads that file at launch, so a ``system_database_url``
    key there feeds the ``dbos`` CLI and never the running app. Reading
    it here would deliver release messages to a database the app is not
    watching. The TS SDK is Postgres-only; there is no SQLite branch.
    """
    override = os.environ.get("DBOS_SYSTEM_DATABASE_URL")
    if override:
        return override
    config = _load_dbos_config(app_dir)
    # The SDK's toDbName(): lowercase, '-' and ' ' to '_', and a leading
    # '_' when the name starts with a digit.
    name = str(config.get("name", "pipeline")).lower().replace("-", "_").replace(" ", "_")
    if name[:1].isdigit():
        name = f"_{name}"
    host = os.environ.get("PGHOST", "localhost")
    port = os.environ.get("PGPORT", "5432")
    user = os.environ.get("PGUSER", "postgres")
    password = os.environ.get("PGPASSWORD", "dbos")
    return f"postgresql://{user}:{password}@{host}:{port}/{name}_dbos_sys"
