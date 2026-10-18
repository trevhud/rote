"""rote's emitted Python helper (main: fastmcp 3) against a modern-only server.

With --fm4, a plain fastmcp 4 Client runs instead.

Usage:
  $MAIN/bin/python -I probe_modern_only.py http://127.0.0.1:8799/mcp
  $FM4/bin/python   -I probe_modern_only.py http://127.0.0.1:8799/mcp --fm4
"""

import asyncio
import json
import os
import pickle
import sys
import tempfile

url = sys.argv[1]
fm4 = "--fm4" in sys.argv
t = tempfile.mkdtemp()
os.environ["ROTE_MCP_CONFIG"] = t + "/m.json"
os.environ["ROTE_MCP_TOKEN_DIR"] = t + "/tok"


async def main() -> None:
    if fm4:
        from fastmcp import Client

        async with Client(url) as c:
            r = await c.call_tool("echo", {"text": "hi"})
            value = r.data
    else:
        from rote.mcp import _runtime_helper as h

        value = await h.call_mcp_tool("modernonly", url, "echo", {"text": "hi"})
    print(f"OK value={value!r} type={type(value).__module__}.{type(value).__name__}")
    for name, fn in (("pickle", pickle.dumps), ("json", json.dumps)):
        try:
            fn(value)
            print(f"  {name} OK")
        except Exception as e:  # noqa: BLE001
            print(f"  {name} FAILED: {type(e).__name__}")


try:
    asyncio.run(main())
except BaseException as e:  # noqa: BLE001
    chain, cur = [], e
    while cur is not None and len(chain) < 6:
        chain.append(f"{type(cur).__name__}: {str(cur)[:220]}")
        subs = getattr(cur, "exceptions", None)
        cur = subs[0] if subs else (cur.__cause__ or cur.__context__)
    print("FAILED\n  " + "\n  <- ".join(chain))
