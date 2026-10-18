"""Send spec-shaped 2026-07-28 requests to public no-auth servers.

No initialize. Every request carries the per-request _meta envelope.
"""

import json
import sys
import urllib.error
import urllib.request

V = "2026-07-28"
META = {
    "io.modelcontextprotocol/protocolVersion": V,
    "io.modelcontextprotocol/clientCapabilities": {},
    "io.modelcontextprotocol/clientInfo": {"name": "rote-probe", "version": "0.1"},
}


def post(url, method, params, name=None):
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "MCP-Protocol-Version": V,
        "Mcp-Method": method,
        "User-Agent": "rote-protocol-probe/0.1",
    }
    if name:
        headers["Mcp-Name"] = name
    p = dict(params)
    p["_meta"] = META
    body = json.dumps({"jsonrpc": "2.0", "id": 7, "method": method, "params": p}).encode()
    req = urllib.request.Request(url, data=body, method="POST", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, dict(r.headers), r.read(1500).decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read(1500).decode("utf-8", "replace")
    except Exception as e:  # noqa: BLE001
        return None, {}, str(e)


for url in sys.argv[1:]:
    print("===", url)
    for method in ("server/discover", "tools/list"):
        s, h, b = post(url, method, {})
        lh = {k.lower(): v for k, v in h.items()}
        print(f"  [{method}] status={s} session={lh.get('mcp-session-id')!r}")
        print("    ", b.replace("\n", " ")[:700])
