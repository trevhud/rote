"""Route a pipeline's vendor HTTP requests to the fidelity bench's fake API.

Loaded by Python at startup (``sitecustomize``) when the bench puts this
directory on the pipeline process's ``PYTHONPATH``. Standard library only
at module level, and nothing from rote: it runs inside the emitted app's
own environment.

``ROTE_HTTP_FIXTURE_URL``    the fake's base URL (``http://127.0.0.1:<port>``)
``ROTE_HTTP_FIXTURE_HOSTS``  comma-separated vendor hosts the fake answers for
``ROTE_HTTP_FIXTURE_BLOCK``  comma-separated host suffixes of the vendors in play
                           (``googleapis.com``)

A request to one of the faked hosts goes to ``<fake>/<host><path>?<query>``.
A request to any other host of a vendor in play raises: a fidelity run must
never reach a real vendor. Every other host (a judge's inference endpoint)
is left alone. Patched, when installed: httpx, requests, httplib2 (which
Google's client library uses), urllib, and ``google.auth.default`` (the
fake accepts any bearer token, so a fixture credential stands in for the
user's).

Under those patches, name resolution is guarded too: ``socket.getaddrinfo``
refuses every faked host and every host of a vendor in play. A client this
module does not patch (``http.client``, ``httpcore`` or ``urllib3`` used
directly, aiohttp) still resolves the vendor's name before it connects, so
it fails here instead of reaching the real vendor. A faked host that gets as
far as resolution has bypassed its rewrite, which is the same failure.
"""

from __future__ import annotations

import importlib
import os
import socket
from typing import Any
from urllib.parse import urlsplit

_BASE = os.environ.get("ROTE_HTTP_FIXTURE_URL", "").rstrip("/")
_HOSTS = {h.strip().lower() for h in os.environ.get("ROTE_HTTP_FIXTURE_HOSTS", "").split(",") if h}
_BLOCK = tuple(
    s.strip().lower().lstrip(".")
    for s in os.environ.get("ROTE_HTTP_FIXTURE_BLOCK", "").split(",")
    if s.strip()
)


class NetworkBlocked(ConnectionError):
    """A request to a host the fidelity bench does not fake."""


def _in_play(host: str) -> bool:
    return host in _HOSTS or any(host == s or host.endswith("." + s) for s in _BLOCK)


def rewrite(url: str) -> str:
    parts = urlsplit(str(url))
    host = (parts.hostname or "").lower()
    if host in _HOSTS:
        query = f"?{parts.query}" if parts.query else ""
        return f"{_BASE}/{host}{parts.path}{query}"
    if _in_play(host):
        raise NetworkBlocked(
            f"the fidelity bench blocks {host}: only {', '.join(sorted(_HOSTS)) or 'no host'} "
            f"is faked for this run"
        )
    return str(url)


def _module(name: str) -> Any:
    """``name`` imported, or ``None`` when the app's environment lacks it."""
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def _patch_httpx() -> None:
    httpx = _module("httpx")
    if httpx is None:
        return
    sync = httpx.HTTPTransport.handle_request
    async_ = httpx.AsyncHTTPTransport.handle_async_request

    def handle_request(self: Any, request: Any) -> Any:
        request.url = httpx.URL(rewrite(str(request.url)))
        return sync(self, request)

    async def handle_async_request(self: Any, request: Any) -> Any:
        request.url = httpx.URL(rewrite(str(request.url)))
        return await async_(self, request)

    httpx.HTTPTransport.handle_request = handle_request
    httpx.AsyncHTTPTransport.handle_async_request = handle_async_request


def _patch_requests() -> None:
    adapters = _module("requests.adapters")
    if adapters is None:
        return
    send = adapters.HTTPAdapter.send

    def patched(self: Any, request: Any, **kwargs: Any) -> Any:
        request.url = rewrite(request.url)
        return send(self, request, **kwargs)

    adapters.HTTPAdapter.send = patched


def _patch_httplib2() -> None:
    httplib2 = _module("httplib2")
    if httplib2 is None:
        return
    original = httplib2.Http.request

    def patched(self: Any, uri: str, *args: Any, **kwargs: Any) -> Any:
        return original(self, rewrite(uri), *args, **kwargs)

    httplib2.Http.request = patched


def _patch_urllib() -> None:
    request_module: Any = _module("urllib.request")
    original = request_module.OpenerDirector.open

    def patched(self: Any, fullurl: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(fullurl, str):
            fullurl = rewrite(fullurl)
        else:
            fullurl.full_url = rewrite(fullurl.full_url)
        return original(self, fullurl, *args, **kwargs)

    request_module.OpenerDirector.open = patched


def _patch_google_auth() -> None:
    auth = _module("google.auth")
    credentials = _module("google.auth.credentials")
    if auth is None or credentials is None:
        return

    class FixtureCredentials(credentials.Credentials):  # type: ignore[misc, name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.token = "rote-fidelity-fixture"

        def refresh(self, request: Any) -> None:
            self.token = "rote-fidelity-fixture"

        @property
        def valid(self) -> bool:
            return True

    def default(*args: Any, **kwargs: Any) -> tuple[Any, str]:
        return FixtureCredentials(), "rote-fidelity"

    auth.default = default


def _guard_name_resolution() -> None:
    original = socket.getaddrinfo

    def getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        raw = host.decode("ascii", "replace") if isinstance(host, bytes) else str(host or "")
        name = raw.lower().rstrip(".")
        if _in_play(name):
            raise NetworkBlocked(
                f"the fidelity bench blocks {name}: the request reached name resolution "
                f"without going through the fake, so it came from an HTTP client this "
                f"bench does not patch"
            )
        return original(host, *args, **kwargs)

    socket.getaddrinfo = getaddrinfo


if _BASE:
    _guard_name_resolution()
    _patch_httpx()
    _patch_requests()
    _patch_httplib2()
    _patch_urllib()
    _patch_google_auth()
