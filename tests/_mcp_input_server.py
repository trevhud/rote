"""An MCP server whose tool answers with a 2026-07-28 ``input_required``.

Run as a script: ``python tests/_mcp_input_server.py <port> <calls_log>``.

``enrich_contact`` asks for an elicitation on the first round and appends
each call's ``contact_id`` to ``calls_log``, so a test can count how many
times a workflow called it. It does not check client capabilities first
(TS SDK servers do, and answer ``-32021`` instead), which makes it the
fixture for the client-side ``input_required`` path.
"""

import sys
from pathlib import Path

import mcp_types
from fastmcp import Context, FastMCP

mcp = FastMCP("asks-a-human")
calls = Path(sys.argv[2])


@mcp.tool
async def enrich_contact(contact_id: str, ctx: Context) -> dict:
    with calls.open("a") as f:
        f.write(contact_id + "\n")
    if ctx.input_responses is None:
        return mcp_types.InputRequiredResult(
            input_requests={
                "approval": mcp_types.ElicitRequest(
                    params=mcp_types.ElicitRequestFormParams(
                        message="Approve this lookup?",
                        requested_schema={
                            "type": "object",
                            "properties": {"ok": {"type": "boolean"}},
                        },
                    )
                )
            }
        )
    return {"contact_id": contact_id}


if __name__ == "__main__":
    mcp.run(transport="http", host="127.0.0.1", port=int(sys.argv[1]))
