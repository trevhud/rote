"""Trace which client-registration path rote's OAuth login takes.

It never registers a client and never authorizes.

Runs rote's real login code (rote.mcp.auth.build_oauth -> fastmcp OAuth -> MCP SDK provider)
against real servers, with an httpx hook that ABORTS before any side-effecting request:
  - a POST to the AS registration endpoint (DCR) is blocked, never sent;
  - the authorization redirect is intercepted, printed, never opened.
So nothing is created on the third party and no account is touched.

Mode "today": build_oauth exactly as rote calls it (no client_metadata_url).
Mode "cimd":  the same OAuth plus client_metadata_url=<hypothetical rote CIMD URL>,
              to show which servers would take the CIMD branch.
"""

import asyncio
import os
import sys
import tempfile
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


_orig_init = httpx.AsyncClient.__init__


async def _on_request(request: httpx.Request) -> None:
    path = request.url.path
    if request.method == "POST" and "regist" in path:
        print(f"    BLOCKED DCR POST {request.url}")
        raise Blocked("dcr")
    print(f"    -> {request.method} {request.url}")


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
        f"    AUTHORIZE (intercepted) client_id={q.get('client_id')} resource={q.get('resource')} "
        f"code_challenge_method={q.get('code_challenge_method')}"
    )
    raise Redirected()


OAuth.redirect_handler = _redirect


async def probe(name: str, url: str, mode: str) -> None:
    cfg = McpServerConfig(url=url)
    oauth = build_oauth(name, cfg)
    if mode == "cimd":
        oauth._client_metadata_url = CIMD_URL
        if hasattr(oauth, "context") and hasattr(oauth.context, "client_metadata_url"):
            oauth.context.client_metadata_url = CIMD_URL
    print(f"=== {name} [{mode}] {url}")
    try:
        async with fastmcp.Client(url, auth=oauth) as c:
            await c.ping()
        print("    unexpected: connected")
    except BaseException as e:  # noqa: BLE001
        chain, cur = [], e
        while cur is not None and len(chain) < 6:
            chain.append(type(cur).__name__)
            subs = getattr(cur, "exceptions", None)
            cur = subs[0] if subs else (cur.__cause__ or cur.__context__)
        outcome = (
            "DCR attempted"
            if "Blocked" in chain
            else ("reached authorization" if "Redirected" in chain else "other")
        )
        print(f"    outcome: {outcome}  (exception chain: {' <- '.join(chain)})")
        if outcome == "other":
            print(f"    detail: {e!r}"[:400])


async def main() -> None:
    targets = [a.split("=", 1) for a in sys.argv[1:]]
    for mode in ("today", "cimd"):
        for name, url in targets:
            await probe(name, url, mode)


asyncio.run(main())
