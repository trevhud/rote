"""Does a DBOS step that returns call_mcp_tool's value survive the durable boundary?

The step is shaped like an emitted mcp-bound DBOS step: it calls
rote.mcp._runtime_helper.call_mcp_tool and returns the payload. Target:
Cloudflare's public docs MCP server, whose tool declares an outputSchema, so
fastmcp hydrates result.data into a dynamically generated `types.Root` class.
SQLite system DB in a temp dir.
"""

import asyncio
import os
import tempfile

t = tempfile.mkdtemp(prefix="rote-dbos-root-")
os.environ["ROTE_MCP_CONFIG"] = t + "/m.json"
os.environ["ROTE_MCP_TOKEN_DIR"] = t + "/tok"

from dbos import DBOS, DBOSConfig  # noqa: E402

from rote.mcp import _runtime_helper as h  # noqa: E402

config: DBOSConfig = {"name": "rote-root-probe", "system_database_url": f"sqlite:///{t}/sys.sqlite"}
DBOS(config=config)


@DBOS.step()
def search_docs() -> object:
    return asyncio.run(
        h.call_mcp_tool(
            "cfdocs",
            "https://docs.mcp.cloudflare.com/mcp",
            "search_cloudflare_documentation",
            {"query": "Workflows waitForEvent"},
        )
    )


@DBOS.workflow()
def wf() -> str:
    value = search_docs()
    return f"{type(value).__module__}.{type(value).__name__}"


DBOS.launch()
try:
    print("workflow returned:", wf())
except Exception as e:  # noqa: BLE001
    print(f"workflow FAILED: {type(e).__name__}: {str(e)[:400]}")
finally:
    DBOS.destroy()
