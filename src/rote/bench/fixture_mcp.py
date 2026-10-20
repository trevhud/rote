"""A fixture MCP server for the compile bench's fidelity runs.

``python -m rote.bench.fixture_mcp <spec.json> <port> <calls.jsonl>``

Hosted MCP servers cannot be called from a benchmark: they need a login,
their data changes, and some calls write. Both sides of a fidelity
comparison (the skill run as an agent, and the compiled pipeline) talk
to this server instead, so they see byte-identical tool results, and
every call either side makes lands in the same log with its arguments.
That log, not a transcript, is the tool-call trace both sides are scored
on: one recording mechanism, so a difference in the trace is a
difference in behavior.

The spec is one server's slice of a case file::

    {"server": "google_drive",
     "tools": {"read_file_content": {
         "description": "...", "side_effect": false,
         "parameters": {"type": "object", "properties": {...}},
         "responses": [{"when": {"file_id": "1Ab"}, "result": "..."},
                       {"when_contains": {"query": "dock"}, "result": [...]},
                       {"result": "the default"}]}}}

A call takes the first response whose ``when`` arguments are all equal
(``when_contains``: case-insensitive substring) to the call's. Results
come back as JSON text content, the way most real servers answer (see
the "fastmcp result.data is None" gotcha in CLAUDE.md), so the emitted
client's text fallback is exercised exactly as in production. A call no
response matches is answered with an error and still logged: a pipeline
asking for data the skill never asked for is a fidelity finding. The
log records which response answered (``response``): two reads that got
the same response read the same data, however their arguments differ
(an optional ``tab``, a differently worded search), which is what the
read-side comparison keys on.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

from fastmcp import FastMCP
from fastmcp.tools import Tool, ToolResult
from mcp.types import TextContent


def _matches(response: dict[str, Any], arguments: dict[str, Any]) -> bool:
    for key, want in (response.get("when") or {}).items():
        if arguments.get(key) != want:
            return False
    for key, want in (response.get("when_contains") or {}).items():
        got = arguments.get(key)
        if not isinstance(got, str) or str(want).casefold() not in got.casefold():
            return False
    return True


def _schema_error(schema: dict[str, Any], arguments: dict[str, Any]) -> str | None:
    """Why ``arguments`` break the tool's input schema, or ``None``.

    The checks a server's argument validation makes on a flat object
    schema, which is all a fixture declares: required keys present, no key
    the schema does not list (a fixture lists the server's real parameters,
    and an argument the server does not have is one it would reject or
    ignore, never act on), and each value of its declared JSON type.
    """
    properties: dict[str, Any] = schema.get("properties") or {}
    missing = [k for k in schema.get("required") or [] if k not in arguments]
    if missing:
        return f"missing required argument(s): {', '.join(missing)}"
    if schema.get("additionalProperties", False) is False:
        unknown = sorted(set(arguments) - set(properties))
        if unknown:
            return (
                f"unknown argument(s): {', '.join(unknown)} "
                f"(the tool takes: {', '.join(sorted(properties)) or 'none'})"
            )
    for key, value in arguments.items():
        want = (properties.get(key) or {}).get("type")
        if want is not None and not _is_json_type(value, want):
            return f"argument {key!r} must be {want}, got {type(value).__name__}"
    return None


_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
    "null": (type(None),),
}


def _is_json_type(value: Any, want: str | list[str]) -> bool:
    for name in [want] if isinstance(want, str) else want:
        if name in ("integer", "number") and isinstance(value, bool):
            continue
        if isinstance(value, _JSON_TYPES.get(name, (object,))):
            return True
    return False


class FixtureTool(Tool):
    """One tool answering from fixture responses and logging every call."""

    responses: list[dict[str, Any]]
    log_path: str
    server_name: str
    side_effect: bool

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        # A real server rejects arguments its input schema does not allow (a
        # missing required `query`, a `search_query` it never declared). The
        # fixture must too: answering such a call from a catch-all response
        # hands the pipeline data a real server would never have returned.
        invalid = _schema_error(self.parameters, arguments)
        if invalid is None:
            index = next((i for i, r in enumerate(self.responses) if _matches(r, arguments)), None)
        else:
            index = None
        chosen = None if index is None else self.responses[index]
        with Path(self.log_path).open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "ts": time.time(),
                        "server": self.server_name,
                        "tool": self.name,
                        "arguments": arguments,
                        "side_effect": self.side_effect,
                        "matched": chosen is not None,
                        "response": index,
                        "invalid_arguments": invalid,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        if invalid is not None:
            raise ValueError(f"invalid arguments for {self.name}: {invalid}")
        if chosen is None:
            raise ValueError(f"no fixture response for {self.name} with arguments {arguments}")
        result = chosen.get("result")
        text = result if isinstance(result, str) else json.dumps(result)
        return ToolResult(content=[TextContent(type="text", text=text)])


def build_server(spec: dict[str, Any], log_path: Path) -> FastMCP:
    server = FastMCP(spec["server"])
    for name, tool in spec["tools"].items():
        server.add_tool(
            FixtureTool(
                name=name,
                description=tool.get("description", ""),
                parameters=tool.get("parameters") or {"type": "object", "properties": {}},
                responses=list(tool.get("responses") or []),
                log_path=str(log_path),
                server_name=spec["server"],
                side_effect=bool(tool.get("side_effect", False)),
            )
        )
    return server


def main(argv: list[str]) -> None:
    spec = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    build_server(spec, Path(argv[2])).run(
        transport="http", host="127.0.0.1", port=int(argv[1]), show_banner=False
    )


if __name__ == "__main__":
    main(sys.argv[1:])
