"""Send tools/list WITHOUT initialize to public unauthenticated MCP servers.

Records whether each server answers a sessionless request, and which protocol
version / session headers it returns. No credentials are sent.
"""

import json
import sys
import urllib.error
import urllib.request


def post(url, payload, extra):
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "User-Agent": "rote-protocol-probe/0.1",
    }
    headers.update(extra)
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), method="POST", headers=headers
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, dict(r.headers), r.read(4000).decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read(4000).decode("utf-8", "replace")
    except Exception as e:  # noqa: BLE001
        return None, {}, str(e)


for url in sys.argv[1:]:
    print("===", url)
    for label, extra, params in [
        (
            "no-init tools/list v2026-07-28 + Mcp-Method",
            {"MCP-Protocol-Version": "2026-07-28", "Mcp-Method": "tools/list"},
            {},
        ),
        ("no-init tools/list v2025-11-25", {"MCP-Protocol-Version": "2025-11-25"}, {}),
        ("initialize v2026-07-28", {}, None),
    ]:
        if params is None:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2026-07-28",
                    "capabilities": {},
                    "clientInfo": {"name": "rote-probe", "version": "0.1"},
                },
            }
        else:
            payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": params}
        s, h, b = post(url, payload, extra)
        lh = {k.lower(): v for k, v in h.items()}
        session, ctype = lh.get("mcp-session-id"), lh.get("content-type")
        print(f"  [{label}] status={s} session={session!r} ctype={ctype!r}")
        snippet = b.replace("\n", " ")[:260]
        print("    body:", snippet)
