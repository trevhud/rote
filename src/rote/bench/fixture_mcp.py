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

A tool that takes SQL is answered from a database instead of a table of
responses (``"sqlite"`` in place of ``"responses"``)::

    {"sqlite": {"seed": "CREATE TABLE ...; INSERT ...;",
                "sql": "sql", "params": "params",
                "answer": [{"results": "$rows", "success": true,
                            "meta": {"changes": "$changes",
                                     "last_row_id": "$last_row_id"}}]}}

A skill that sends its own queries words them differently on every run, and
a canned response per query can only answer the wordings someone thought of.
Each server process seeds its own in-memory SQLite database from ``seed``
and runs the statement the ``sql`` argument holds (bound to the ``params``
argument's values, when the spec names one). ``answer`` is the tool's result
as the real server sends it, with ``$rows``, ``$changes`` and
``$last_row_id`` filled in: the shape is the case's to state, so this module
knows no vendor. A read logs ``response`` as a digest of its rows, so two
wordings that read the same rows read the same data. A statement that
writes is logged with ``side_effect`` true and its ``effect``: the rows it
added and removed, per table. The write lands in that process's throwaway
database only, so a later read in the same run sees it, as on the real
server, and nothing outside the process changes.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

from fastmcp import FastMCP
from fastmcp.tools import Tool, ToolResult
from mcp.types import TextContent

from rote.tool_schema import argument_problem


def _matches(response: dict[str, Any], arguments: dict[str, Any]) -> bool:
    if response.get("exact"):
        # A replay's recorded call answers that call only (rote.replay).
        return arguments == (response.get("when") or {})
    for key, want in (response.get("when") or {}).items():
        if arguments.get(key) != want:
            return False
    for key, want in (response.get("when_contains") or {}).items():
        got = arguments.get(key)
        if not isinstance(got, str) or str(want).casefold() not in got.casefold():
            return False
    return True


def structured_answer(result: Any, output_schema: dict[str, Any] | None) -> dict[str, Any] | None:
    """The structured content a tool with ``output_schema`` answers ``result``
    with, or None when it has no schema or ``result`` cannot meet it.

    Its client validates structured content against the declared schema and,
    under fastmcp's wrap flag, unwraps ``{"result": ...}``, so the fixture
    answers the way the real tool did. A wrapped string result is the raw
    text, never its parse: a ``-> str`` tool returning ``"5"`` or JSON text
    sent that text as a string.
    """
    if not output_schema:
        return None
    if output_schema.get("x-fastmcp-wrap-result"):
        props = output_schema.get("properties") or {}
        inner = props.get("result") if isinstance(props, dict) else None
        if isinstance(result, str):
            try:
                parsed = json.loads(result)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict) and set(parsed) == {"result"}:
                return parsed  # recorded as the wrapped object itself
            if isinstance(inner, dict) and inner.get("type") == "string":
                return {"result": result}
            return {"result": parsed if parsed is not None else result}
        if isinstance(result, dict) and set(result) == {"result"}:
            return result  # already the wrapped object: never wrapped twice
        return {"result": result}
    value = result
    if isinstance(result, str):
        try:
            value = json.loads(result)
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


def _digest(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


#: What a statement may do. A fixture database is in memory and holds only
#: the case's rows, so the statement is the one thing that could reach
#: outside it: ATTACH opens (or creates) any file by path.
_REFUSED_SQL = {
    sqlite3.SQLITE_ATTACH: "ATTACH",
    sqlite3.SQLITE_DETACH: "DETACH",
}

#: Authorizer actions that change the database: rows, or its schema.
_WRITE_SQL = frozenset(
    getattr(sqlite3, name)
    for name in (
        "SQLITE_INSERT",
        "SQLITE_UPDATE",
        "SQLITE_DELETE",
        "SQLITE_CREATE_TABLE",
        "SQLITE_CREATE_INDEX",
        "SQLITE_CREATE_VIEW",
        "SQLITE_CREATE_TRIGGER",
        "SQLITE_DROP_TABLE",
        "SQLITE_DROP_INDEX",
        "SQLITE_DROP_VIEW",
        "SQLITE_DROP_TRIGGER",
        "SQLITE_ALTER_TABLE",
    )
)


class SqlDatabase:
    """One seeded in-memory database answering a SQL tool's calls."""

    def __init__(self, spec: dict[str, Any]) -> None:
        for key in ("seed", "sql", "answer"):
            if key not in spec:
                raise ValueError(f"a sqlite fixture needs {key!r}")
        self.sql_argument = str(spec["sql"])
        self.params_argument = spec.get("params")
        self.answer = spec["answer"]
        # Autocommit: each statement stands alone, as each tool call does.
        self.conn = sqlite3.connect(":memory:", isolation_level=None, check_same_thread=False)
        self.conn.executescript(str(spec["seed"]))
        self._seen: list[int] = []
        self.conn.set_authorizer(self._authorize)

    def _authorize(self, action: int, *_: Any) -> int:
        self._seen.append(action)
        return sqlite3.SQLITE_DENY if action in _REFUSED_SQL else sqlite3.SQLITE_OK

    def _tables(self) -> dict[str, list[str]]:
        """Every table's rows, each as canonical JSON, sorted."""
        names = [
            r[0]
            for r in self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        ]
        out: dict[str, list[str]] = {}
        for name in sorted(names):
            cursor = self.conn.execute(f'SELECT * FROM "{name}"')  # noqa: S608 (a schema name)
            columns = [d[0] for d in cursor.description]
            out[name] = sorted(
                json.dumps(dict(zip(columns, row, strict=True)), sort_keys=True, default=str)
                for row in cursor
            )
        return out

    def run(self, arguments: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        """``(the tool's result, what the call log records about it)``.

        Raises ``ValueError`` for a statement SQLite refuses, with SQLite's
        own message: the agent reads it as it would the real server's."""
        sql = arguments.get(self.sql_argument)
        if not isinstance(sql, str):
            raise ValueError(f"{self.sql_argument} must be a SQL string")
        params = arguments.get(self.params_argument) if self.params_argument else None
        before = self._tables()
        self._seen = []
        changes = self.conn.total_changes
        try:
            cursor = self.conn.execute(sql, list(params or []))
            columns = [d[0] for d in cursor.description or []]
            rows = [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]
        except (sqlite3.Error, sqlite3.Warning) as exc:
            refused = next((_REFUSED_SQL[a] for a in self._seen if a in _REFUSED_SQL), None)
            why = f"{refused} is not allowed here" if refused else str(exc)
            raise ValueError(f"SQL error: {why}") from exc
        wrote = any(a in _WRITE_SQL for a in self._seen)
        fill = {
            "$rows": rows,
            "$changes": self.conn.total_changes - changes,
            # The connection's last insert, which a read did not make.
            "$last_row_id": cursor.lastrowid if wrote else 0,
        }
        result = _filled(self.answer, fill)
        if not wrote:
            return result, {"side_effect": False, "response": f"rows:{_digest(rows)}"}
        after = self._tables()
        effect: dict[str, dict[str, Any]] = {
            table: {
                "added": [json.loads(r) for r in sorted(set(now) - set(before.get(table, [])))],
                "removed": [json.loads(r) for r in sorted(set(before.get(table, [])) - set(now))],
            }
            for table, now in after.items()
            if now != before.get(table)
        }
        effect |= {t: {"dropped": True} for t in before if t not in after}
        return result, {"side_effect": True, "response": None, "effect": effect}


def _filled(template: Any, values: dict[str, Any]) -> Any:
    """``template`` with each ``$name`` string replaced by its value."""
    if isinstance(template, str) and template in values:
        return values[template]
    if isinstance(template, list):
        return [_filled(item, values) for item in template]
    if isinstance(template, dict):
        return {key: _filled(item, values) for key, item in template.items()}
    return template


#: The one reading of a tool's input schema (:mod:`rote.tool_schema`), so
#: the fixtures refuse exactly what verification refuses in a compile.
_schema_error = argument_problem


class FixtureTool(Tool):
    """One tool answering from fixture responses and logging every call."""

    responses: list[dict[str, Any]]
    log_path: str
    server_name: str
    side_effect: bool
    #: Set for a SQL tool answered from a database (:class:`SqlDatabase`).
    database: Any = None

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        if self.database is not None:
            return self._run_sql(arguments)
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
        content = [TextContent(type="text", text=text)]
        structured = structured_answer(result, self.output_schema)
        if structured is None:
            return ToolResult(content=content)
        return ToolResult(content=content, structured_content=structured)

    def _log(self, arguments: dict[str, Any], record: dict[str, Any]) -> None:
        entry = {
            "ts": time.time(),
            "server": self.server_name,
            "tool": self.name,
            "arguments": arguments,
            **record,
        }
        with Path(self.log_path).open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, sort_keys=True, default=str) + "\n")

    def _run_sql(self, arguments: dict[str, Any]) -> ToolResult:
        """Answer from the database, logging what the statement read or changed."""
        refused = {"side_effect": self.side_effect, "matched": False, "response": None}
        invalid = _schema_error(self.parameters, arguments)
        if invalid is not None:
            self._log(arguments, refused | {"invalid_arguments": invalid})
            raise ValueError(f"invalid arguments for {self.name}: {invalid}")
        try:
            result, record = self.database.run(arguments)
        except ValueError as exc:
            self._log(arguments, refused | {"invalid_arguments": None, "error": str(exc)})
            raise
        self._log(arguments, {"matched": True, "invalid_arguments": None} | record)
        return ToolResult(content=[TextContent(type="text", text=json.dumps(result, default=str))])


def build_server(spec: dict[str, Any], log_path: Path) -> FastMCP:
    server = FastMCP(spec["server"])
    for name, tool in spec["tools"].items():
        responses = list(tool.get("responses") or [])
        database = None
        if tool.get("sqlite") is not None:
            if responses:
                raise ValueError(f"{name}: a tool has responses or a sqlite fixture, not both")
            try:
                database = SqlDatabase(tool["sqlite"])
            except (sqlite3.Error, sqlite3.Warning) as exc:
                raise ValueError(f"{name}: the sqlite fixture's seed failed: {exc}") from exc
            argument = database.sql_argument
            if argument not in ((tool.get("parameters") or {}).get("properties") or {}):
                raise ValueError(f"{name}: the tool has no {argument!r} argument to read SQL from")
        schema = tool.get("output_schema")
        # Declare the real tool's output schema only when every response can
        # meet it: the client refuses a declared schema answered without
        # structured content, which would fail a call the server answered.
        if schema and any(structured_answer(r.get("result"), schema) is None for r in responses):
            schema = None
        server.add_tool(
            FixtureTool(
                name=name,
                description=tool.get("description", ""),
                parameters=tool.get("parameters") or {"type": "object", "properties": {}},
                output_schema=schema,
                responses=responses,
                log_path=str(log_path),
                server_name=spec["server"],
                side_effect=bool(tool.get("side_effect", False)),
                database=database,
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
