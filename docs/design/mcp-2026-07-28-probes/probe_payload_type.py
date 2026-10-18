"""Check what call_mcp_tool returns for a server that declares an outputSchema,
and whether that value pickles (DBOS step results) and JSON-serializes."""

import asyncio
import json
import os
import pickle
import tempfile

t = tempfile.mkdtemp()
os.environ["ROTE_MCP_CONFIG"] = t + "/m.json"
os.environ["ROTE_MCP_TOKEN_DIR"] = t + "/tok"

from rote.mcp import _runtime_helper as h  # noqa: E402

r = asyncio.run(
    h.call_mcp_tool(
        "cfdocs",
        "https://docs.mcp.cloudflare.com/mcp",
        "search_cloudflare_documentation",
        {"query": "Workflows"},
    )
)
print("type:", type(r), type(r).__module__)
for name, fn in (("pickle", pickle.dumps), ("json", json.dumps)):
    try:
        fn(r)
        print(name, "OK")
    except Exception as e:  # noqa: BLE001
        print(name, "FAILED:", type(e).__name__, str(e)[:200])
