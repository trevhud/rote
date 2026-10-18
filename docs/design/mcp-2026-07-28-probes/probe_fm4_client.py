"""fastmcp 4 Client (mode="auto", the default) against public no-auth servers.

Shows which era gets negotiated, what goes on the wire, and whether a tool
call's result survives pickle/JSON (the durable-step boundary).
"""

import asyncio
import json
import pickle

import httpx2 as httpx

_orig_init = httpx.AsyncClient.__init__


async def _on_request(request: httpx.Request) -> None:
    try:
        body = json.loads(request.content or b"{}")
    except Exception:  # noqa: BLE001
        body = {}
    h = request.headers
    print(
        f"    -> rpc={body.get('method')!r} MCP-Protocol-Version={h.get('mcp-protocol-version')!r} "
        f"Mcp-Method={h.get('mcp-method')!r} Mcp-Name={h.get('mcp-name')!r} "
        f"session={'set' if h.get('mcp-session-id') else None}"
    )


async def _on_response(response: httpx.Response) -> None:
    print(f"    <- {response.status_code}")


def _patched_init(self, *a, **kw):
    hooks = kw.setdefault("event_hooks", {})
    hooks.setdefault("request", []).append(_on_request)
    hooks.setdefault("response", []).append(_on_response)
    _orig_init(self, *a, **kw)


httpx.AsyncClient.__init__ = _patched_init

from fastmcp import Client  # noqa: E402

TARGETS = [
    (
        "https://docs.mcp.cloudflare.com/mcp",
        "search_cloudflare_documentation",
        {"query": "Workflows"},
    ),
    (
        "https://mcp.context7.com/mcp",
        "resolve-library-id",
        {"libraryName": "fastmcp", "query": "oauth"},
    ),
    (
        "https://mcp.deepwiki.com/mcp",
        "read_wiki_structure",
        {"repoName": "modelcontextprotocol/python-sdk"},
    ),
    ("https://gitmcp.io/docs", None, None),
]


async def main() -> None:
    for url, tool, args in TARGETS:
        print(f"=== {url}")
        try:
            async with Client(url) as c:
                attrs = {
                    k: getattr(c, k, None)
                    for k in ("protocol_version", "negotiated_version", "era")
                }
                print(f"  client attrs: {attrs}")
                tools = await c.list_tools()
                print(f"  list_tools OK ({len(tools)} tools)")
                if tool:
                    r = await c.call_tool(tool, args)
                    d = r.data
                    print(f"  call_tool OK; data type={type(d).__module__}.{type(d).__name__}")
                    for name, fn in (("pickle", pickle.dumps), ("json", json.dumps)):
                        try:
                            fn(d)
                            print(f"    {name}(data) OK")
                        except Exception as e:  # noqa: BLE001
                            print(f"    {name}(data) FAILED: {type(e).__name__}")
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED: {type(e).__name__}: {str(e)[:300]}")


asyncio.run(main())
