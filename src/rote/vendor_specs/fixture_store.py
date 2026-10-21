"""A local store of HTTP fixture sets, keyed the way a shared store will key them.

Layout (one JSON file per fixture set)::

    <root>/<vendor>/<api>/<version>/<name>.json

``<root>`` is ``ROTE_FIXTURE_DIR``, else ``~/.local/share/rote/fixtures``.
The file is a :class:`~rote.vendor_specs.fixtures.FixtureSet` as JSON:
the spec key, the spec ``revision`` it was validated against, its
``trust`` (``spec`` / ``docs``) and sources, its ``origin`` (which example
case or compile produced it), and the per-operation responses. A shared
store (rote-cloud, later) holds the same files under the same keys, so
publishing one is a copy. Nothing in the store is ever a real response:
fixtures are authored data, and a live check (``rote.vendor_specs.live``)
compares shapes without keeping what it read.

A set is validated against its spec on the way in and stamped with the
revision it fit. Reading one back re-validates when a spec is given, so a
set that no longer fits a newer revision is an error, not a stale answer.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from rote.vendor_specs.fixtures import FixtureError, FixtureSet, validate
from rote.vendor_specs.model import ApiSpec
from rote.xdg import data_home

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_KEY_PART_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def default_root() -> Path:
    override = os.environ.get("ROTE_FIXTURE_DIR")
    if override:
        return Path(override)
    return data_home() / "rote" / "fixtures"


class FixtureStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or default_root()

    def path(self, spec_key: str, name: str) -> Path:
        parts = spec_key.split("/")
        if len(parts) != 3 or not all(_KEY_PART_RE.match(p) for p in parts):
            raise FixtureError(f"not a spec key (<vendor>/<api>/<version>): {spec_key!r}")
        if not _NAME_RE.match(name):
            raise FixtureError(f"not a fixture name (letters, digits, . _ -): {name!r}")
        return self.root.joinpath(*parts, f"{name}.json")

    def save(self, fixtures: FixtureSet, spec: ApiSpec, name: str, origin: str) -> Path:
        """Validate ``fixtures`` against ``spec``, stamp it, and write it under ``name``."""
        validate(fixtures, spec)
        stamped = fixtures.model_copy(
            update={"name": name, "spec_revision": spec.revision, "origin": origin}
        )
        path = self.path(fixtures.spec, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(stamped.model_dump_json(indent=1) + "\n", encoding="utf-8")
        os.replace(tmp, path)
        return path

    def load(self, spec_key: str, name: str, spec: ApiSpec | None = None) -> FixtureSet:
        path = self.path(spec_key, name)
        if not path.is_file():
            raise FixtureError(f"no fixture set {name!r} for {spec_key} in {self.root}")
        fixtures = FixtureSet.model_validate_json(path.read_text(encoding="utf-8"))
        if fixtures.spec != spec_key:
            raise FixtureError(f"{path} holds fixtures for {fixtures.spec}, not {spec_key}")
        if spec is not None:
            validate(fixtures, spec)
        return fixtures

    def list(self, spec_key: str | None = None) -> list[tuple[str, str]]:
        """``(spec key, name)`` for every stored set, optionally for one spec."""
        out: list[tuple[str, str]] = []
        if not self.root.is_dir():
            return out
        for path in sorted(self.root.glob("*/*/*/*.json")):
            key = "/".join(path.relative_to(self.root).parts[:3])
            if spec_key is None or key == spec_key:
                out.append((key, path.stem))
        return out


__all__ = ["FixtureStore", "default_root"]
