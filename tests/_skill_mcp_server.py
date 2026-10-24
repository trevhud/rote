"""The MCP server the compiled-skill driver tests call. Run as a script.

``python tests/_skill_mcp_server.py <port> <hits.jsonl> [<tool>,<tool>...]``

Every tool call appends ``{"tool": ..., "arguments": ...}`` to the hit log
first, so a test counts what reached the server. ``search``, ``lookup``
and ``flaky`` are marked read-only; ``write_note`` is not, so a read-only
run must refuse it before calling. ``microsoft_docs_search`` and
``read_wiki_structure`` answer with docs-digest's recorded results
(``examples/docs-digest/reference.json``), whatever they are asked.
``slack_read_channel``, ``search_threads`` and ``get_thread`` are
deal-monitor's. ``hold`` waits while a file named ``hold`` sits beside the
hit log, so a test can kill its caller mid-call. With
``ROTE_FIXTURE_BEARER`` set, a request without that bearer token gets 401.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

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


@_tool(annotations=READ_ONLY)
async def hold(key: str) -> dict:
    """Answers only once the file ``hold`` beside the hit log is gone, so a
    test can stop the caller while this call is in flight."""
    _hit("hold", key=key)
    while HITS.with_name("hold").exists():
        await asyncio.sleep(0.05)
    return {"key": key, "held": True}


@_tool(annotations=READ_ONLY)
def echo(text: str) -> str:
    """Return ``text`` as given: a tool result whose text the caller chose."""
    _hit("echo", text=text)
    return text


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


#: deal-monitor's intake channel: six deal posts, so a fan or a batch over
#: them is never a no-op. One is excluded (NA, 2000+ SKUs), one is EU/UK only
#: because its submitter is on the always-UK list, and two match no thread.
DEAL_POSTS = [
    {"user": "U01", "ts": "1760000001.0001", "text": "New deal: Acme Corp, 40 SKUs, NA, rep Dana"},
    {"user": "U02", "ts": "1760000002.0001", "text": "New deal: Brightline, 12 SKUs, UK, rep Sam"},
    {"user": "U03", "ts": "1760000003.0001", "text": "New deal: Corvid, 300 SKUs, EU, rep Lee"},
    {"user": "U04", "ts": "1760000004.0001", "text": "New deal: Dynamo, 5000 SKUs, NA, rep Kai"},
    {
        "user": "U05",
        "ts": "1760000005.0001",
        "text": "New deal: Echo Labs, SKUs unknown, Nashville, rep Sam Patel",
    },
    {
        "user": "U06",
        "ts": "1760000006.0001",
        "text": "New deal: Fjord Supply, 80 SKUs, EU, rep Lee",
    },
]

#: deal-monitor's quoting threads. The fixed searches find the first three;
#: the per-account search finds Acme's again and Echo Labs', so the dedup has
#: a duplicate to drop and one thread matches no opportunity.
THREADS = {
    "t1": {"subject": "Quote Request: Acme Corp", "snippet": "rates attached"},
    "t2": {"subject": "RFP: Brightline", "snippet": "two questions on volume"},
    "t9": {"subject": "New Business: Unrelated Co", "snippet": "intro"},
    "t3": {"subject": "Echo Labs pricing", "snippet": "we will pass"},
}
STANDARD_THREADS = ["t1", "t2", "t9"]
ACCOUNT_THREADS = {"Acme Corp": ["t1"], "Echo Labs": ["t3"]}


@_tool(annotations=READ_ONLY)
def slack_read_channel(user_email: str) -> dict:
    _hit("slack_read_channel", user_email=user_email)
    return {"messages": DEAL_POSTS}


@_tool(annotations=READ_ONLY)
def search_threads(
    user_email: str,
    warehouse_domains: list[str] | None = None,
    opportunities: list[dict] | None = None,
) -> dict:
    """deal-monitor's two searches: the fixed ones, or one per account."""
    _hit(
        "search_threads",
        user_email=user_email,
        warehouse_domains=warehouse_domains,
        opportunities=opportunities,
    )
    if opportunities is None:
        ids = STANDARD_THREADS
    else:
        ids = [t for o in opportunities for t in ACCOUNT_THREADS.get(o["account_name"], [])]
    return {"threads": [{"thread_id": t, **THREADS[t]} for t in ids]}


@_tool(annotations=READ_ONLY)
def get_thread(threads: list[dict], user_email: str) -> dict:
    _hit("get_thread", threads=threads, user_email=user_email)
    return {
        "email_threads": [
            {
                "thread_id": t["thread_id"],
                "subject": THREADS[t["thread_id"]]["subject"],
                "thread_url": f"https://mail.example.com/{t['thread_id']}",
                "messages": [
                    {
                        "from": "ops@example.com",
                        "to": "quotes@wh.example.com",
                        "date": "2026-10-01",
                        "body": THREADS[t["thread_id"]]["snippet"],
                    }
                ],
            }
            for t in threads
        ]
    }


#: With ``ROTE_FIXTURE_BEARER`` set, every request without that bearer token
#: gets a plain 401, the way a server answers a client with no login.
BEARER = os.environ.get("ROTE_FIXTURE_BEARER")


def _require_bearer(app: Any) -> Any:
    async def guarded(scope: dict, receive: Any, send: Any) -> None:
        headers = dict(scope.get("headers") or [])
        if scope["type"] == "http" and headers.get(b"authorization") != f"Bearer {BEARER}".encode():
            await send({"type": "http.response.start", "status": 401, "headers": []})
            await send({"type": "http.response.body", "body": b""})
            return
        await app(scope, receive, send)

    return guarded


if __name__ == "__main__":
    port = int(sys.argv[1])
    if BEARER is None:
        mcp.run(transport="http", host="127.0.0.1", port=port, show_banner=False)
    else:
        import uvicorn

        uvicorn.run(_require_bearer(mcp.http_app()), host="127.0.0.1", port=port, log_level="error")
