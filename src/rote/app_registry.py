"""Registry of apps rote has emitted (``~/.local/share/rote/apps.json``).

``rote emit`` and ``rote compile`` record every output directory here so
later commands can find the user's apps without being told where they
live. Today's consumer is ``rote mcp login``, which scans registered DBOS
apps for workflows parked on a ``rote:auth:<server>`` topic and releases
them after a successful authentication.

The file is plain JSON (``{"version": 1, "apps": [{path, runtime,
pipeline[, name]}]}``), deduplicated by resolved path — re-emitting to the same
directory updates its entry rather than appending. Entries can go stale
(directories get moved or deleted); consumers must treat a missing
directory as skippable, not fatal. Override the location with
``ROTE_APPS_PATH`` (tests rely on this for isolation).

``name`` is what a person calls the app, when that is not the pipeline's
name: a compiled skill (``--runtime skill``) is recorded under its SKILL.md
``name``, which ``rote skill run <name>`` resolves (bdr-outreach's pipeline
is ``bdr-campaign``). Other runtimes record none.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from rote.xdg import data_home


@dataclass(frozen=True)
class RegisteredApp:
    """One emitted app directory, as recorded at emit time."""

    path: Path
    runtime: str
    pipeline: str
    #: A compiled skill's name (see the module docstring); None otherwise.
    name: str | None = None


def apps_path() -> Path:
    override = os.environ.get("ROTE_APPS_PATH")
    if override:
        return Path(override)
    return data_home() / "rote" / "apps.json"


def registered_apps() -> list[RegisteredApp]:
    path = apps_path()
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("apps", [])
    apps: list[RegisteredApp] = []
    if isinstance(entries, list):
        for entry in entries:
            if isinstance(entry, dict) and entry.get("path"):
                apps.append(
                    RegisteredApp(
                        path=Path(str(entry["path"])),
                        runtime=str(entry.get("runtime", "")),
                        pipeline=str(entry.get("pipeline", "")),
                        name=str(entry["name"]) if entry.get("name") else None,
                    )
                )
    return apps


def _entry(app: RegisteredApp) -> dict[str, str]:
    entry = {"path": str(app.path), "runtime": app.runtime, "pipeline": app.pipeline}
    if app.name:
        entry["name"] = app.name
    return entry


def record_app(
    app_dir: Path | str, runtime: str, pipeline: str, *, name: str | None = None
) -> None:
    """Record (or refresh) an emitted app directory in the registry."""
    resolved = Path(app_dir).resolve()
    apps = [a for a in registered_apps() if a.path != resolved]
    apps.append(RegisteredApp(path=resolved, runtime=runtime, pipeline=pipeline, name=name))

    path = apps_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"version": 1, "apps": [_entry(a) for a in apps]}
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".apps-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
