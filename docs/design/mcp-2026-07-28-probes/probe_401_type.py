"""What exception type does a 401 surface as through fastmcp's Client, and does rote's
_find_http_401 (which checks httpx.HTTPStatusError) recognize it?

Starts a stdlib HTTP server on 127.0.0.1 that answers every POST with 401 (no OAuth metadata),
then calls it with a plain fastmcp Client (no auth), as the helper does for an unregistered server.
"""

import asyncio
import http.server
import importlib.util
import threading


class H(http.server.BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Bearer realm="probe"')
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):  # noqa: N802
        self.send_response(404)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *a):
        pass


srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
url = f"http://127.0.0.1:{srv.server_address[1]}/mcp"


def find_401_like_rote(exc: BaseException) -> bool:
    """Same walk as rote.mcp._runtime_helper._find_http_401 (httpx.HTTPStatusError only)."""
    if importlib.util.find_spec("httpx") is None:
        return False
    import httpx

    seen, stack = set(), [exc]
    while stack:
        e = stack.pop()
        if id(e) in seen:
            continue
        seen.add(id(e))
        if isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 401:
            return True
        stack.extend(getattr(e, "exceptions", ()) or ())
        for nxt in (e.__cause__, e.__context__):
            if nxt is not None:
                stack.append(nxt)
    return False


async def main() -> None:
    from fastmcp import Client

    try:
        async with Client(url) as c:
            await c.list_tools()
        print("unexpected success")
    except BaseException as e:  # noqa: BLE001
        types, cur = [], e
        while cur is not None and len(types) < 6:
            types.append(f"{type(cur).__module__}.{type(cur).__name__}")
            subs = getattr(cur, "exceptions", None)
            cur = subs[0] if subs else (cur.__cause__ or cur.__context__)
        print("exception chain:", " <- ".join(types))
        print("rote _find_http_401 would detect it:", find_401_like_rote(e))


asyncio.run(main())
srv.shutdown()
