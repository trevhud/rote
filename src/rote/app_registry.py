"""Registry of apps rote has emitted (``~/.local/share/rote/apps.json``).

``rote emit`` and ``rote compile`` record every output directory here so
later commands can find the user's apps without being told where they
live. Today's consumer is ``rote mcp login``, which scans registered DBOS
apps for workflows parked on a ``rote:auth:<server>`` topic and releases
them after a successful authentication.

The file is plain JSON (``{"version": 1, "apps": [{path, runtime,
pipeline[, name][, trust]}]}``), deduplicated by resolved path — re-emitting to the same
directory updates its entry rather than appending. Entries can go stale
(directories get moved or deleted); consumers must treat a missing
directory as skippable, not fatal. Override the location with
``ROTE_APPS_PATH`` (tests rely on this for isolation).

``name`` is what a person calls the app, when that is not the pipeline's
name: a compiled skill (``--runtime skill``) is recorded under its SKILL.md
``name``, which ``rote skill run <name>`` resolves (bdr-outreach's pipeline
is ``bdr-campaign``). Other runtimes record none.

``trust`` is a compiled skill's install record (``rote skill install``,
``rote skill trust``; :mod:`rote.skill_target.trust`): the hash of every
file the user vouched for, the state paths the install was made against,
and any write consent. Emit never writes it. A re-emit to the same path
keeps it, so the next ``rote skill run`` sees what changed and refuses
until the user trusts it again. Every write keeps every other entry's
record, which is why the file is parsed and rendered by one pair
(:func:`parse_apps`, :func:`render_apps`).
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from rote.xdg import data_home


@dataclass(frozen=True)
class RegisteredApp:
    """One emitted app directory, as recorded at emit time."""

    path: Path
    runtime: str
    pipeline: str
    #: A compiled skill's name (see the module docstring); None otherwise.
    name: str | None = None
    #: A compiled skill's install record (see the module docstring).
    trust: dict[str, Any] | None = None


def apps_path() -> Path:
    override = os.environ.get("ROTE_APPS_PATH")
    if override:
        return Path(override)
    return data_home() / "rote" / "apps.json"


def parse_apps(text: str | None) -> list[RegisteredApp]:
    """The entries in a registry file's ``text`` (None: no file yet)."""
    if text is None:
        return []
    data = json.loads(text)
    entries = data.get("apps", []) if isinstance(data, dict) else []
    apps: list[RegisteredApp] = []
    if isinstance(entries, list):
        for entry in entries:
            if isinstance(entry, dict) and entry.get("path"):
                trust = entry.get("trust")
                apps.append(
                    RegisteredApp(
                        path=Path(str(entry["path"])),
                        runtime=str(entry.get("runtime", "")),
                        pipeline=str(entry.get("pipeline", "")),
                        name=str(entry["name"]) if entry.get("name") else None,
                        trust=trust if isinstance(trust, dict) else None,
                    )
                )
    return apps


def registered_apps() -> list[RegisteredApp]:
    path = apps_path()
    if not path.is_file():
        return []
    return parse_apps(path.read_text(encoding="utf-8"))


def _entry(app: RegisteredApp) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "path": str(app.path),
        "runtime": app.runtime,
        "pipeline": app.pipeline,
    }
    if app.name:
        entry["name"] = app.name
    if app.trust is not None:
        entry["trust"] = app.trust
    return entry


def render_apps(apps: list[RegisteredApp]) -> str:
    """The registry file's text for ``apps``, exactly as :func:`record_app` writes it."""
    return json.dumps({"version": 1, "apps": [_entry(a) for a in apps]}, indent=2) + "\n"


def record_app(
    app_dir: Path | str, runtime: str, pipeline: str, *, name: str | None = None
) -> None:
    """Record (or refresh) an emitted app directory in the registry.

    A compiled skill's install record at the same path is kept (see the
    module docstring)."""
    resolved = Path(app_dir).resolve()
    existing = registered_apps()
    kept = next((a.trust for a in existing if a.path == resolved and a.runtime == runtime), None)
    apps = [a for a in existing if a.path != resolved]
    fresh = RegisteredApp(path=resolved, runtime=runtime, pipeline=pipeline, name=name)
    apps.append(replace(fresh, trust=kept))

    path = apps_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".apps-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(render_apps(apps))
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
