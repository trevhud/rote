"""fastmcp 4 Client (auto) against `rote serve --http` (FastMCP 3.4.9, sessionful legacy):
which era is negotiated and whether tools/list + a trigger call work."""

import asyncio
import sys

from fastmcp import Client


async def main() -> None:
    async with Client(sys.argv[1]) as c:
        print("negotiated:", c.protocol_version)
        tools = await c.list_tools()
        print("tools:", [t.name for t in tools])
        r = await c.call_tool("bdr-campaign", {"company": "Example"}, raise_on_error=False)
        print(
            "trigger isError:", r.is_error, "text:", (r.content[0].text if r.content else "")[:200]
        )


asyncio.run(main())
