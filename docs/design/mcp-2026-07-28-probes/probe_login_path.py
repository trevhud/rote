"""Trace which client-registration path rote's OAuth login takes.

It never registers a client and never authorizes.

Runs rote's real login code (rote.mcp.auth.build_oauth -> fastmcp OAuth -> MCP
SDK provider) against real servers. Two independent guards keep it free of side
effects, so nothing is created on the third party and no account is touched:

1. Allowlist. Before the client starts, the probe discovers the server's
   authorization-server metadata itself (stdlib GETs) to learn the exact
   ``registration_endpoint``. During the run, an httpx request hook lets
   through only GET requests and a POST to the MCP endpoint itself (the
   unauthenticated call that elicits the 401). Every other request, whatever
   its URL, is aborted before it is sent. That covers DCR at the advertised
   endpoint, DCR at a guessed fallback URL, a token request, and anything
   unforeseen.
2. The authorization redirect is intercepted and never opened, so the flow
   cannot get past it to a token request.

Each blocked request is classified against the discovered metadata:
"DCR at advertised endpoint", "DCR at guessed endpoint" (no
``registration_endpoint`` advertised, but the SDK POSTed to ``/register``
anyway), or "unexpected POST".

Mode "today": build_oauth exactly as rote calls it (no client_metadata_url).
Mode "cimd":  the same OAuth plus client_metadata_url=<hypothetical rote CIMD URL>,
              to show which servers would take the CIMD branch.

Usage: python -I probe_login_path.py name=<mcp-url> [name=<mcp-url> ...]
"""

import asyncio
import json
import os
import sys
import tempfile
import urllib.error
import urllib.request
from urllib.parse import parse_qs, urlparse

tmp = tempfile.mkdtemp(prefix="rote-login-probe-")
os.environ["ROTE_MCP_CONFIG"] = os.path.join(tmp, "mcp.json")
os.environ["ROTE_MCP_TOKEN_DIR"] = os.path.join(tmp, "tokens")

import httpx  # noqa: E402

CIMD_URL = "https://roteskills.com/oauth/rote-cli.json"


class Blocked(Exception):
    pass


class Redirected(Exception):
    pass


def _norm(url: str) -> str:
    p = urlparse(url)
    return f"{p.scheme}://{p.netloc.lower()}{p.path.rstrip('/')}"


def _get_json(url: str) -> dict | None:
    req = urllib.request.Request(url, headers={"User-Agent": "rote-login-probe/0.2"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read(200000))
    except (urllib.error.URLError, ValueError, TimeoutError):
        return None


def discover_registration_endpoint(mcp_url: str) -> str | None:
    """The AS's advertised registration_endpoint, found the way the SDK finds it."""
    p = urlparse(mcp_url)
    base, path = f"{p.scheme}://{p.netloc}", p.path.rstrip("/")
    prm = _get_json(f"{base}/.well-known/oauth-protected-resource{path}") or _get_json(
        f"{base}/.well-known/oauth-protected-resource"
    )
    as_url = (prm or {}).get("authorization_servers", [base])[0]
    a = urlparse(as_url)
    as_base, as_path = f"{a.scheme}://{a.netloc}", a.path.rstrip("/")
    for cand in (
        f"{as_base}/.well-known/oauth-authorization-server{as_path}",
        f"{as_base}/.well-known/openid-configuration{as_path}",
        f"{as_base}/.well-known/oauth-authorization-server",
    ):
        meta = _get_json(cand)
        if meta:
            return meta.get("registration_endpoint")
    return None


# Per-probe state the request hook reads. Set by probe() before each run.
STATE: dict = {"mcp_url": "", "registration_endpoint": None, "blocked": []}


def _classify(request: httpx.Request) -> str:
    target = _norm(str(request.url))
    reg = STATE["registration_endpoint"]
    if reg and target == _norm(reg):
        return "DCR at advertised endpoint"
    if request.method == "POST" and request.url.path.rstrip("/").endswith("/register"):
        return "DCR at guessed endpoint"
    return f"unexpected {request.method}"


async def _on_request(request: httpx.Request) -> None:
    allowed = request.method == "GET" or (
        request.method == "POST" and _norm(str(request.url)) == _norm(STATE["mcp_url"])
    )
    if not allowed:
        kind = _classify(request)
        STATE["blocked"].append((kind, str(request.url)))
        print(f"    BLOCKED ({kind}) {request.method} {request.url}")
        raise Blocked(kind)
    print(f"    -> {request.method} {request.url}")


_orig_init = httpx.AsyncClient.__init__


def _patched_init(self, *a, **kw):
    kw.setdefault("event_hooks", {}).setdefault("request", []).append(_on_request)
    _orig_init(self, *a, **kw)


httpx.AsyncClient.__init__ = _patched_init

import fastmcp  # noqa: E402
from fastmcp.client.auth import OAuth  # noqa: E402

from rote.mcp.auth import build_oauth  # noqa: E402
from rote.mcp.registry import McpServerConfig  # noqa: E402


async def _redirect(self, authorization_url: str) -> None:
    q = parse_qs(urlparse(authorization_url).query)
    print(
        f"    AUTHORIZE (intercepted) client_id={q.get('client_id')} "
        f"resource={q.get('resource')} code_challenge_method={q.get('code_challenge_method')}"
    )
    raise Redirected()


OAuth.redirect_handler = _redirect


async def probe(name: str, url: str, mode: str, registration_endpoint: str | None) -> None:
    STATE.update(mcp_url=url, registration_endpoint=registration_endpoint, blocked=[])
    cfg = McpServerConfig(url=url)
    oauth = build_oauth(name, cfg)
    if mode == "cimd":
        oauth._client_metadata_url = CIMD_URL
        if hasattr(oauth, "context") and hasattr(oauth.context, "client_metadata_url"):
            oauth.context.client_metadata_url = CIMD_URL
    print(f"=== {name} [{mode}] {url}  (advertised registration_endpoint: {registration_endpoint})")
    reached_authorize = False
    try:
        async with fastmcp.Client(url, auth=oauth) as c:
            await c.ping()
        print("    unexpected: connected")
    except BaseException as e:  # noqa: BLE001
        cur, depth = e, 0
        while cur is not None and depth < 8:
            if isinstance(cur, Redirected):
                reached_authorize = True
            subs = getattr(cur, "exceptions", None)
            cur = subs[0] if subs else (cur.__cause__ or cur.__context__)
            depth += 1
    if STATE["blocked"]:
        outcome = STATE["blocked"][0][0]
    elif reached_authorize:
        outcome = "reached authorization (no registration request)"
    else:
        outcome = "other (see output above)"
    print(f"    outcome: {outcome}")


async def main() -> None:
    targets = [a.split("=", 1) for a in sys.argv[1:]]
    endpoints = {name: discover_registration_endpoint(url) for name, url in targets}
    for mode in ("today", "cimd"):
        for name, url in targets:
            await probe(name, url, mode, endpoints[name])


asyncio.run(main())
