"""A fixture tool that takes SQL is answered from a seeded SQLite database.

A skill that sends its own queries words them differently on every run, so a
canned response per query answers only the wordings someone thought of. The
fixture server runs the statement against an in-memory database seeded from
the case, answers in the shape the case states, and logs what each statement
read or changed. Real fastmcp, in process; no inference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from rote.bench.fidelity_score import call_key, effect_key
from rote.bench.fixture_mcp import build_server

SEED = """
CREATE TABLE people (id INTEGER PRIMARY KEY, email TEXT NOT NULL, cohort INTEGER NOT NULL);
CREATE TABLE reminders (email TEXT NOT NULL, stage TEXT NOT NULL, at TEXT NOT NULL);
INSERT INTO people (email, cohort) VALUES ('ana@x.test', 7), ('bo@x.test', 7), ('cy@x.test', 8);
INSERT INTO reminders VALUES ('ana@x.test', 'cohort-7-nudge-1', '2026-11-20T10:00:00Z');
"""

#: The result shape is the case's to state: this one is a hosted SQL tool
#: that wraps each statement's rows with its own bookkeeping.
ANSWER = [
    {"results": "$rows", "success": True, "meta": {"changes": "$changes", "id": "$last_row_id"}}
]


def _spec(**sqlite: Any) -> dict[str, Any]:
    return {
        "server": "db",
        "tools": {
            "query": {
                "description": "Run SQL.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sql": {"type": "string"},
                        "params": {"type": ["array", "null"], "items": {"type": "string"}},
                    },
                    "required": ["sql"],
                },
                "sqlite": {
                    "seed": SEED,
                    "sql": "sql",
                    "params": "params",
                    "answer": ANSWER,
                    **sqlite,
                },
            }
        },
    }


def _calls(log: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]


async def _ask(client: Client[Any], sql: str, params: list[str] | None = None) -> Any:
    result = await client.call_tool("query", {"sql": sql, "params": params})
    return json.loads(result.content[0].text)


@pytest.mark.anyio
async def test_a_query_is_run_and_answered_in_the_shape_the_case_states(tmp_path: Path) -> None:
    log = tmp_path / "calls.jsonl"
    async with Client(build_server(_spec(), log)) as client:
        answer = await _ask(
            client, "SELECT email FROM people WHERE cohort = ? ORDER BY email", ["7"]
        )
    assert answer == [
        {
            "results": [{"email": "ana@x.test"}, {"email": "bo@x.test"}],
            "success": True,
            "meta": {"changes": 0, "id": 0},
        }
    ]
    (call,) = _calls(log)
    assert (call["side_effect"], call["matched"]) == (False, True)
    assert "effect" not in call


@pytest.mark.anyio
async def test_two_wordings_that_read_the_same_rows_are_the_same_read(tmp_path: Path) -> None:
    """The read comparison keys on the data, never on how the query was
    spelled. The negative half: a query that reads other rows is another read."""
    log = tmp_path / "calls.jsonl"
    async with Client(build_server(_spec(), log)) as client:
        await _ask(client, "SELECT email FROM people WHERE cohort = 7 ORDER BY email")
        await _ask(client, "select  p.email\nfrom people p where p.cohort=?1 order by 1", ["7"])
        await _ask(client, "SELECT email FROM people WHERE cohort = 8")
    same, reworded, other = (call_key(c) for c in _calls(log))
    assert same == reworded
    assert other != same


@pytest.mark.anyio
async def test_a_write_changes_the_run_s_own_database_and_is_logged_by_its_rows(
    tmp_path: Path,
) -> None:
    log = tmp_path / "calls.jsonl"
    insert = "INSERT INTO reminders (email, stage, at) VALUES (?, ?, ?)"
    row = ["bo@x.test", "cohort-7-nudge-1", "2026-11-26T09:00:00Z"]
    async with Client(build_server(_spec(), log)) as client:
        wrote = await _ask(client, insert, row)
        after = await _ask(client, "SELECT COUNT(*) AS n FROM reminders")
    assert wrote[0]["meta"] == {"changes": 1, "id": 2}
    # A later read in the same run sees the write, as on the real server.
    assert after[0]["results"] == [{"n": 2}]
    write, read = _calls(log)
    assert write["side_effect"] is True and write["response"] is None
    assert write["effect"] == {
        "reminders": {
            "added": [{"email": row[0], "stage": row[1], "at": row[2]}],
            "removed": [],
        }
    }
    assert read["side_effect"] is False

    # A second server starts from the seed: one run's write never reaches another.
    async with Client(build_server(_spec(), tmp_path / "other.jsonl")) as client:
        fresh = await _ask(client, "SELECT COUNT(*) AS n FROM reminders")
    assert fresh[0]["results"] == [{"n": 1}]


@pytest.mark.anyio
async def test_the_same_change_worded_two_ways_is_the_same_write(tmp_path: Path) -> None:
    """A write is identified by the rows it changed, and one that changes
    other rows is a different write."""
    keys = []
    statements = [
        ("UPDATE people SET cohort = 9 WHERE email = 'cy@x.test'", None),
        ("update people set cohort=?1 where id = ?2", ["9", "3"]),
        ("UPDATE people SET cohort = 9 WHERE email = 'bo@x.test'", None),
    ]
    for i, (sql, params) in enumerate(statements):
        log = tmp_path / f"calls-{i}.jsonl"
        async with Client(build_server(_spec(), log)) as client:
            await _ask(client, sql, params)
        (call,) = _calls(log)
        keys.append(effect_key(call, {"db.query": []}))
    assert keys[0] == keys[1]
    assert keys[2] != keys[0]


@pytest.mark.anyio
async def test_a_write_that_matches_no_row_is_still_a_write(tmp_path: Path) -> None:
    """Whether a statement writes is read from the statement, not from how
    many rows it happened to touch."""
    log = tmp_path / "calls.jsonl"
    async with Client(build_server(_spec(), log)) as client:
        await _ask(client, "DELETE FROM reminders WHERE email = 'nobody@x.test'")
    (call,) = _calls(log)
    assert call["side_effect"] is True
    # The table it wrote is named even with no row changed, so the same miss
    # against another table is another write.
    assert call["effect"] == {"reminders": {"added": [], "removed": []}}


@pytest.mark.anyio
async def test_a_statement_sqlite_refuses_is_an_error_the_agent_reads(tmp_path: Path) -> None:
    log = tmp_path / "calls.jsonl"
    async with Client(build_server(_spec(), log)) as client:
        with pytest.raises(ToolError, match="no such table: bookings"):
            await _ask(client, "SELECT * FROM bookings")
    (call,) = _calls(log)
    assert call["matched"] is False and "no such table" in call["error"]


@pytest.mark.anyio
@pytest.mark.parametrize(
    "sql",
    ["ATTACH DATABASE '{path}' AS other", "VACUUM INTO '{path}'"],
)
async def test_a_statement_cannot_reach_a_file(sql: str, tmp_path: Path) -> None:
    """The database is in memory; the statement is the one thing that could
    open or create a file by path."""
    target = tmp_path / "leak.db"
    async with Client(build_server(_spec(), tmp_path / "calls.jsonl")) as client:
        with pytest.raises(ToolError):
            await _ask(client, sql.format(path=target))
    assert not target.exists()


async def _effects(tmp_path: Path, *statements: str, seed: str = SEED) -> list[Any]:
    log = tmp_path / "calls.jsonl"
    async with Client(build_server(_spec(seed=seed), log)) as client:
        for sql in statements:
            await _ask(client, sql)
    return [c.get("effect") for c in _calls(log)]


@pytest.mark.anyio
async def test_writes_that_change_no_row_are_told_apart_by_what_they_wrote(
    tmp_path: Path,
) -> None:
    """Found by review: every write that left the rows as they were logged
    the same empty effect, so a miss on one table matched a miss on another."""
    miss_reminders, miss_people, trigger = await _effects(
        tmp_path,
        "DELETE FROM reminders WHERE email = 'nobody@x.test'",
        "UPDATE people SET cohort = 9 WHERE email = 'nobody@x.test'",
        "CREATE TRIGGER t AFTER INSERT ON people BEGIN SELECT 1; END",
    )
    assert miss_reminders != miss_people
    assert trigger == {"$schema": ["create_trigger"]}


@pytest.mark.anyio
async def test_a_second_copy_of_a_row_is_a_change(tmp_path: Path) -> None:
    """Found by review: the diff was a set difference, so inserting or
    deleting one of two equal rows changed nothing it could see."""
    seed = "CREATE TABLE d (x); INSERT INTO d VALUES (1);"
    added, removed = await _effects(
        tmp_path,
        "INSERT INTO d VALUES (1)",
        "DELETE FROM d WHERE rowid = 1",
        seed=seed,
    )
    assert added == {"d": {"added": [{"x": 1}], "removed": []}}
    assert removed == {"d": {"added": [], "removed": [{"x": 1}]}}


@pytest.mark.anyio
async def test_two_empty_results_are_not_the_same_read(tmp_path: Path) -> None:
    """Found by review: every statement that returned no rows shared one
    digest, so an empty read of the wrong table matched the right one. The
    same columns with no rows is still the same read."""
    log = tmp_path / "calls.jsonl"
    async with Client(build_server(_spec(), log)) as client:
        await _ask(client, "SELECT email FROM people WHERE cohort = 99")
        await _ask(client, "SELECT stage FROM reminders WHERE stage = 'none'")
        await _ask(client, "select p.email from people p where 1 = 0")
    people, reminders, reworded = (call_key(c) for c in _calls(log))
    assert people != reminders
    assert people == reworded


@pytest.mark.anyio
async def test_a_table_whose_name_needs_quoting_does_not_break_the_server(tmp_path: Path) -> None:
    """Found by review: a quote in a table name broke the row dump, and every
    later call failed without a line in the log."""
    log = tmp_path / "calls.jsonl"
    async with Client(build_server(_spec(), log)) as client:
        await _ask(client, 'CREATE TABLE "q""x" (a)')
        await _ask(client, 'INSERT INTO "q""x" VALUES (1)')
        answer = await _ask(client, "SELECT COUNT(*) AS n FROM people")
    assert answer[0]["results"] == [{"n": 3}]
    assert _calls(log)[1]["effect"] == {'q"x': {"added": [{"a": 1}], "removed": []}}
    assert len(_calls(log)) == 3


@pytest.mark.anyio
async def test_a_user_table_named_like_sqlites_own_is_still_seen(tmp_path: Path) -> None:
    """Found by review: `NOT LIKE 'sqlite_%'` hid `sqlitelog`, since `_` is
    LIKE's wildcard."""
    (effect,) = await _effects(
        tmp_path, "INSERT INTO sqlitelog VALUES (1)", seed="CREATE TABLE sqlitelog (x);"
    )
    assert effect == {"sqlitelog": {"added": [{"x": 1}], "removed": []}}


@pytest.mark.anyio
async def test_a_statement_that_never_ends_is_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Found by review: an unbounded recursive query held the server, and
    every other tool on it, until the trial's own timeout."""
    from rote.bench import fixture_mcp

    monkeypatch.setattr(fixture_mcp, "SQL_SECONDS", 0.2)
    log = tmp_path / "calls.jsonl"
    forever = (
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT count(*) FROM c"
    )
    async with Client(build_server(_spec(), log)) as client:
        with pytest.raises(ToolError, match="ran longer than"):
            await _ask(client, forever)
        # The server still answers.
        after = await _ask(client, "SELECT COUNT(*) AS n FROM people")
    assert after[0]["results"] == [{"n": 3}]
    assert [c["matched"] for c in _calls(log)] == [False, True]


@pytest.mark.anyio
async def test_a_result_past_the_row_cap_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rote.bench import fixture_mcp

    monkeypatch.setattr(fixture_mcp, "SQL_MAX_ROWS", 2)
    async with Client(build_server(_spec(), tmp_path / "calls.jsonl")) as client:
        assert len((await _ask(client, "SELECT * FROM people LIMIT 2"))[0]["results"]) == 2
        with pytest.raises(ToolError, match="more than 2 rows"):
            await _ask(client, "SELECT * FROM people")


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("sql", "why"),
    [
        ("BEGIN", "a transaction"),
        ("SAVEPOINT s", "a savepoint"),
        ("PRAGMA user_version = 5", "PRAGMA user_version"),
        ("PRAGMA writable_schema = ON", "PRAGMA writable_schema"),
    ],
)
async def test_a_statement_that_changes_state_without_a_row_is_refused(
    sql: str, why: str, tmp_path: Path
) -> None:
    """Found by review: BEGIN opened a real transaction, so a logged write
    could be rolled back, and a PRAGMA that sets something logged as a read."""
    async with Client(build_server(_spec(), tmp_path / "calls.jsonl")) as client:
        with pytest.raises(ToolError, match=f"{why}.* is not allowed here"):
            await _ask(client, sql)
        # A PRAGMA that only describes the schema is a read, as a skill uses it.
        info = await _ask(client, "PRAGMA table_info(reminders)")
    assert [c["name"] for c in info[0]["results"]] == ["email", "stage", "at"]


@pytest.mark.anyio
async def test_params_that_are_not_a_list_are_refused(tmp_path: Path) -> None:
    """Found by review: sqlite3 binds a string's characters, so a mistyped
    params answered with the wrong value instead of an error."""
    spec = _spec()
    spec["tools"]["query"]["parameters"]["properties"]["params"] = {}
    async with Client(build_server(spec, tmp_path / "calls.jsonl")) as client:
        with pytest.raises(ToolError, match="params must be a list"):
            await client.call_tool("query", {"sql": "SELECT ?", "params": "z"})


def test_a_seed_that_does_not_run_fails_the_server_not_the_first_call(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="the sqlite fixture's seed failed"):
        build_server(_spec(seed="CREATE TABLE (;"), tmp_path / "calls.jsonl")


def test_the_sql_argument_must_be_one_the_tool_has(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no 'statement' argument"):
        build_server(_spec(sql="statement"), tmp_path / "calls.jsonl")


CASES = """\
output_fields:
  unbooked: the people with no reminder yet
servers:
  db:
    tools:
      query:
        side_effect: false
cases:
- id: seeded
  input: {}
  fixtures:
    db:
      query:
        sqlite:
          seed: |
            CREATE TABLE reminders (email TEXT NOT NULL, stage TEXT NOT NULL);
            INSERT INTO reminders VALUES ('ana@x.test', 'nudge-1');
          sql: sql
          params: params
          answer:
          - results: $rows
            success: true
            meta: {changes: $changes}
"""


def test_a_case_file_starts_a_server_whose_tool_runs_the_statement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Through the real path: cases.yaml, a server process, the emitted
    pipelines' own MCP client, and the result as a compiled step receives it."""
    import asyncio

    from rote.bench.fidelity import fixture_servers, load_cases, read_calls
    from rote.mcp._runtime_helper import call_mcp_tool

    example = tmp_path / "example"
    (example / "skill").mkdir(parents=True)
    (example / "fidelity").mkdir()
    tools = _spec()["tools"]["query"]
    (example / "skill" / "mcp-tools.json").write_text(
        json.dumps(
            {
                "servers": {
                    "db": {
                        "tools": [
                            {
                                "name": "query",
                                "description": tools["description"],
                                "inputSchema": tools["parameters"],
                            }
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (example / "fidelity" / "cases.yaml").write_text(CASES, encoding="utf-8")
    casefile = load_cases(example)
    with fixture_servers(casefile, casefile.cases[0], tmp_path / "work") as (urls, log):
        monkeypatch.setenv("ROTE_MCP_DB_URL", urls["db"])
        insert = {"sql": "INSERT INTO reminders VALUES (?, ?)", "params": ["bo@x.test", "nudge-1"]}
        wrote = asyncio.run(call_mcp_tool("db", None, "query", insert))
        read = asyncio.run(
            call_mcp_tool("db", None, "query", {"sql": "SELECT email FROM reminders ORDER BY 1"})
        )
        calls = read_calls(log)
    assert wrote == [{"results": [], "success": True, "meta": {"changes": 1}}]
    assert read[0]["results"] == [{"email": "ana@x.test"}, {"email": "bo@x.test"}]
    assert [c["side_effect"] for c in calls] == [True, False]
    assert calls[0]["effect"] == {
        "reminders": {"added": [{"email": "bo@x.test", "stage": "nudge-1"}], "removed": []}
    }
