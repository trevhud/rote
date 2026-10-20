"""Where vendor specs come from: a local, shareable cache in front of the vendor.

Layout, chosen so a shared store (rote-cloud, later) can hold the same
files under the same keys::

    <root>/<vendor>/<api>/<version>.json   the vendor's own spec document, verbatim
    <root>/google/_directory.json          the Discovery directory

The vendor's bytes are what is cached, not rote's conversion of them: the
conversion (:func:`rote.vendor_specs.google.from_discovery`) runs on every
read, so a change to it can never be hidden behind a cache written by an
older rote.

``<root>`` is ``ROTE_SPEC_CACHE_DIR``, else ``~/.cache/rote/specs``. A cached
spec is used while younger than the TTL; after that it is refetched, and a
failed refetch is an error (:class:`SpecUnavailable`), never a silent stale
read. Fetching needs no credential: every source here is a public spec.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rote.vendor_specs import google
from rote.vendor_specs.model import ApiSpec

DEFAULT_TTL_SECONDS = 24 * 3600
_FETCH_TIMEOUT_SECONDS = 30


class SpecUnavailable(RuntimeError):
    """A spec that should exist could not be fetched or parsed."""


def default_root() -> Path:
    override = os.environ.get("ROTE_SPEC_CACHE_DIR")
    return Path(override) if override else Path.home() / ".cache" / "rote" / "specs"


def http_get_json(url: str) -> Any:
    """GET ``url`` and parse JSON. The one network call in this package."""
    request = urllib.request.Request(url, headers={"User-Agent": "rote-vendor-specs"})
    try:
        with urllib.request.urlopen(request, timeout=_FETCH_TIMEOUT_SECONDS) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise SpecUnavailable(f"could not fetch {url}: {e}") from e


class SpecStore:
    """Fetches and caches specs; one per compile, memoized in memory too."""

    def __init__(
        self,
        root: Path | None = None,
        fetch: Callable[[str], Any] | None = None,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ) -> None:
        self.root = root or default_root()
        # Resolved at call time so a test's patch of http_get_json lands.
        self._fetch = fetch
        self.ttl_seconds = ttl_seconds
        self._memo: dict[str, ApiSpec | SpecUnavailable] = {}

    def _get(self, url: str) -> Any:
        return (self._fetch or http_get_json)(url)

    def google(self, api: str, version: str) -> ApiSpec | None:
        """The spec for Google ``api`` ``version``; ``None`` when the directory has no such API.

        Raises :class:`SpecUnavailable` when the directory or the document
        cannot be fetched.
        """
        key = f"google/{api}/{version}"
        memo = self._memo.get(key)
        if isinstance(memo, SpecUnavailable):
            raise memo
        if memo is not None:
            return memo
        try:
            spec = self._cached(key) or self._fetch_google(api, version)
        except SpecUnavailable as e:
            self._memo[key] = e
            raise
        if spec is not None:
            self._memo[key] = spec
        return spec

    def _fetch_google(self, api: str, version: str) -> ApiSpec | None:
        directory = self._directory()
        url = google.discovery_url(directory, api, version)
        if url is None:
            return None
        doc = self._get(url)
        spec = _convert(doc, url)
        self._write(f"google/{api}/{version}", {"source_url": url, "document": doc})
        return spec

    def _directory(self) -> dict[str, Any]:
        path = self.root / "google" / "_directory.json"
        cached = self._read_fresh(path)
        if cached is not None:
            return dict(cached)
        directory = self._get(google.DIRECTORY_URL)
        if not isinstance(directory, dict) or "items" not in directory:
            raise SpecUnavailable(f"{google.DIRECTORY_URL} returned no items")
        self._write_path(path, directory)
        return directory

    def _cached(self, key: str) -> ApiSpec | None:
        raw = self._read_fresh(self.root / f"{key}.json")
        if not isinstance(raw, dict) or "document" not in raw:
            return None
        return _convert(raw["document"], str(raw.get("source_url") or ""))

    def _read_fresh(self, path: Path) -> Any:
        try:
            wrapper = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if time.time() - float(wrapper.get("fetched_at", 0)) > self.ttl_seconds:
            return None
        return wrapper.get("body")

    def _write(self, key: str, body: Any) -> None:
        self._write_path(self.root / f"{key}.json", body)

    def _write_path(self, path: Path, body: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"fetched_at": time.time(), "body": body}), encoding="utf-8")
        os.replace(tmp, path)


def _convert(doc: Any, url: str) -> ApiSpec:
    try:
        return google.from_discovery(doc, url)
    except (KeyError, TypeError, ValueError) as e:
        raise SpecUnavailable(f"{url} is not a Discovery document: {e}") from e


__all__ = ["DEFAULT_TTL_SECONDS", "SpecStore", "SpecUnavailable", "default_root", "http_get_json"]
