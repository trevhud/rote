"""Helpers for the OAuth e2e suites: the switchable AS process and a fake browser.

The server itself is ``tests/_oauth_as_server.py``; see its docstring for
the modes and the control API.
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

REPO = Path(__file__).resolve().parent.parent
SERVER_SCRIPT = Path(__file__).resolve().parent / "_oauth_as_server.py"
CIMD_DOCUMENT = REPO / "docs" / "oauth" / "rote-cli.json"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def wait_port(port: int, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(0.2)
    raise RuntimeError(f"server on :{port} never came up")


@dataclass
class OAuthFixture:
    """A running ``_oauth_as_server.py`` process."""

    base: str
    cimd_url: str
    #: The local CA the document host uses. Point SSL_CERT_FILE at it so
    #: rote's own pre-flight fetch of the document trusts the host.
    ca_pem: Path

    @property
    def mcp_url(self) -> str:
        return f"{self.base}/mcp"

    def issuer(self, name: str) -> str:
        return f"{self.base}/as-{name}"

    def configure(self, **state: Any) -> None:
        response = httpx.post(f"{self.base}/_control", json=state, timeout=10)
        response.raise_for_status()

    def events(self, event: str | None = None) -> list[dict[str, Any]]:
        response = httpx.get(f"{self.base}/_events", timeout=10)
        response.raise_for_status()
        rows: list[dict[str, Any]] = response.json()
        return [r for r in rows if event is None or r["event"] == event]


@contextmanager
def oauth_fixture(workdir: Path) -> Iterator[OAuthFixture]:
    """Start the fixture with rote's real CIMD document, re-homed locally.

    The served document is the repo's ``docs/oauth/rote-cli.json`` with only
    ``client_id`` changed to the local URL, so its ``redirect_uris`` (the
    published callback port set) are exactly what the AS validates.
    """
    http_port, tls_port = free_port(), free_port()
    cimd_url = f"https://127.0.0.1:{tls_port}/oauth/rote-cli.json"
    document = json.loads(CIMD_DOCUMENT.read_text(encoding="utf-8"))
    document["client_id"] = cimd_url
    served = workdir / "rote-cli.json"
    served.write_text(json.dumps(document), encoding="utf-8")
    ca_pem = workdir / "ca.pem"
    proc = subprocess.Popen(
        [
            sys.executable,
            str(SERVER_SCRIPT),
            str(http_port),
            str(tls_port),
            str(served),
            str(ca_pem),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_port(http_port)
        wait_port(tls_port)
        yield OAuthFixture(base=f"http://127.0.0.1:{http_port}", cimd_url=cimd_url, ca_pem=ca_pem)
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def follow_in_background(url: str) -> None:
    """GET ``url`` and follow its redirect chain on a thread, like a browser.

    A real browser takes human-scale time to follow the chain; this thread
    can outrun fastmcp's local callback server binding its port. Retry the
    whole chain (each attempt mints a fresh auth code; the AS honors the
    same state) until the callback hop stops refusing connections.
    """

    def follow() -> None:
        deadline = time.monotonic() + 30
        while True:
            try:
                with httpx.Client(follow_redirects=True, timeout=30) as client:
                    client.get(url)
                return
            except httpx.ConnectError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.5)

    threading.Thread(target=follow, daemon=True).start()


def fake_browser_open() -> Callable[..., bool]:
    """A ``webbrowser.open`` replacement that auto-approves via the redirect chain."""

    def fake_open(url: str, *args: object, **kwargs: object) -> bool:
        follow_in_background(url)
        return True

    return fake_open
