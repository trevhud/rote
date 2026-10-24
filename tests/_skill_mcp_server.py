"""The MCP server the compiled-skill driver tests call. Run as a script.

``python tests/_skill_mcp_server.py <port> <hits.jsonl> [<tool>,<tool>...]``

Every tool call appends ``{"tool": ..., "arguments": ...}`` to the hit log
first, so a test counts what reached the server. ``search``, ``lookup``
and ``flaky`` are marked read-only; ``write_note`` is not, so a read-only
run must refuse it before calling. ``microsoft_docs_search`` and
``read_wiki_structure`` answer with docs-digest's recorded results
(``examples/docs-digest/reference.json``), whatever they are asked.
"""

import json
import sys
from pathlib import Path

from fastmcp import FastMCP
from mcp_types import ToolAnnotations

REPO = Path(__file__).resolve().parent.parent
HITS = Path(sys.argv[2])
READ_ONLY = ToolAnnotations(readOnlyHint=True)
RECORDED = {
    c["tool"]: c["result"]
    for c in json.loads((REPO / "examples/docs-digest/reference.json").read_text())["pipeline_run"][
        "tool_calls"
    ]
}

mcp = FastMCP("skill-fixtures")
#: ``argv[3]``, when given: the only tools this server offers (comma-separated),
#: so a test can stand up a second server that lacks some of them.
ONLY = set(sys.argv[3].split(",")) if len(sys.argv) > 3 else None


def _tool(**kwargs: object) -> object:
    def register(fn: object) -> object:
        if ONLY is None or getattr(fn, "__name__", None) in ONLY:
            mcp.tool(**kwargs)(fn)  # type: ignore[call-overload]
        return fn

    return register


def _hit(tool: str, **arguments: object) -> None:
    with HITS.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"tool": tool, "arguments": arguments}) + "\n")


@_tool(annotations=READ_ONLY)
def search(query: str) -> list[dict]:
    _hit("search", query=query)
    return [{"title": f"{query} {i}", "rank": i} for i in range(3)]


@_tool(annotations=READ_ONLY)
def lookup(id: str) -> dict:
    _hit("lookup", id=id)
    return {"id": id, "score": len(id)}


@_tool(annotations=READ_ONLY)
def flaky(key: str) -> dict:
    """Fails on the first call for each key, then answers."""
    seen = [json.loads(line) for line in HITS.read_text().splitlines()] if HITS.exists() else []
    _hit("flaky", key=key)
    if not any(h["tool"] == "flaky" and h["arguments"]["key"] == key for h in seen):
        raise RuntimeError(f"transient failure for {key}")
    return {"key": key, "ok": True}


@_tool()
def write_note(text: str) -> dict:
    _hit("write_note", text=text)
    return {"written": text}


@_tool(annotations=READ_ONLY)
def microsoft_docs_search(query: str) -> dict:
    _hit("microsoft_docs_search", query=query)
    return RECORDED["microsoft_docs_search"]


@_tool(annotations=READ_ONLY)
def read_wiki_structure(repoName: str) -> str:  # noqa: N803 (the real tool's argument name)
    _hit("read_wiki_structure", repoName=repoName)
    return RECORDED["read_wiki_structure"]


#: deal-monitor's intake channel: three deal posts, so a fan or a batch
#: over them is never a no-op.
DEAL_POSTS = [
    {"user": "U01", "ts": "1760000001.0001", "text": "New deal: Acme Corp, 40 SKUs, NA, rep Dana"},
    {"user": "U02", "ts": "1760000002.0001", "text": "New deal: Brightline, 12 SKUs, UK, rep Sam"},
    {"user": "U03", "ts": "1760000003.0001", "text": "New deal: Corvid, 300 SKUs, EU, rep Lee"},
]


@_tool(annotations=READ_ONLY)
def slack_read_channel(user_email: str) -> dict:
    _hit("slack_read_channel", user_email=user_email)
    return {"messages": DEAL_POSTS}


@_tool(annotations=READ_ONLY)
def search_threads(user_email: str, warehouse_domains: list[str] | None = None) -> list[dict]:
    _hit("search_threads", user_email=user_email, warehouse_domains=warehouse_domains)
    return [
        {"thread_id": "t1", "subject": "Quote: Acme Corp", "snippet": "rates attached"},
        {"thread_id": "t2", "subject": "Brightline onboarding", "snippet": "next steps"},
    ]


if __name__ == "__main__":
    mcp.run(transport="http", host="127.0.0.1", port=int(sys.argv[1]), show_banner=False)
