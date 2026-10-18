"""Run rote's emitted MCP runtime helper (rote.mcp._runtime_helper) against public servers.

Usage: python probe_rote_client.py
Isolated config: ROTE_MCP_CONFIG / ROTE_MCP_TOKEN_DIR point at an empty temp dir,
so no stored login or registry entry is ever read. No credentials are sent.
Wire capture: an httpx event hook prints every request's method, protocol headers,
JSON-RPC method, and the response status + session header.
"""

import asyncio
import json
import os
import sys
import tempfile

tmp = tempfile.mkdtemp(prefix="rote-probe-")
os.environ["ROTE_MCP_CONFIG"] = os.path.join(tmp, "mcp.json")
os.environ["ROTE_MCP_TOKEN_DIR"] = os.path.join(tmp, "tokens")

import httpx  # noqa: E402

_orig_init = httpx.AsyncClient.__init__


async def _on_request(request: httpx.Request) -> None:
    rpc = ""
    try:
        rpc = json.loads(request.content or b"{}").get("method", "")
    except Exception:  # noqa: BLE001
        rpc = "?"
    h = request.headers
    print(
        f"    -> {request.method} rpc={rpc!r} "
        f"MCP-Protocol-Version={h.get('mcp-protocol-version')!r} "
        f"Mcp-Session-Id={'set' if h.get('mcp-session-id') else None} "
        f"Mcp-Method={h.get('mcp-method')!r}"
    )


async def _on_response(response: httpx.Response) -> None:
    issued = "issued" if response.headers.get("mcp-session-id") else None
    print(f"    <- {response.status_code} session={issued}")


def _patched_init(self, *a, **kw):
    hooks = kw.setdefault("event_hooks", {})
    hooks.setdefault("request", []).append(_on_request)
    hooks.setdefault("response", []).append(_on_response)
    _orig_init(self, *a, **kw)


httpx.AsyncClient.__init__ = _patched_init

from rote.mcp import _runtime_helper as h  # noqa: E402

TARGETS = [
    (
        "cfdocs",
        "https://docs.mcp.cloudflare.com/mcp",
        "search_cloudflare_documentation",
        {"query": "Workflows waitForEvent timeout"},
    ),
    (
        "context7",
        "https://mcp.context7.com/mcp",
        "resolve-library-id",
        {"libraryName": "fastmcp", "query": "fastmcp oauth client"},
    ),
    (
        "deepwiki",
        "https://mcp.deepwiki.com/mcp",
        "read_wiki_structure",
        {"repoName": "modelcontextprotocol/python-sdk"},
    ),
]


async def main() -> None:
    for server, url, tool, args in TARGETS:
        print(f"=== {server} {url}")
        try:
            client = h.mcp_client(server, url)
            async with client:
                init = client.initialize_result
                print(f"  negotiated protocolVersion={getattr(init, 'protocolVersion', None)!r}")
        except Exception as e:  # noqa: BLE001
            print(f"  connect FAILED: {type(e).__name__}: {e}")
        try:
            result = await h.call_mcp_tool(server, url, tool, args)
            text = json.dumps(result, default=str) if not isinstance(result, str) else result
            print(f"  call_mcp_tool({tool}) OK: {text[:160]!r}")
        except Exception as e:  # noqa: BLE001
            print(f"  call_mcp_tool({tool}) FAILED: {type(e).__name__}: {e}")


asyncio.run(main())
sys.stdout.flush()
