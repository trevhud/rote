"""A compiled skill's agent loops: the LOOP block, its tool proxy and its answer.

No inference: the test is the agent. It reads the LOOP block, calls the
loop's tools through ``tool`` and answers through ``answer``, as an agent
running the skill would. The tools are real: ``tests/_skill_mcp_server.py``
(fastmcp over HTTP on localhost), whose hit log is the evidence of what
reached a server.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

from rote.ir import Pipeline, load_pipeline
from rote.plan import build_plan
from rote.skill_target import write_scripts
from tests._messages_api import reject_like_the_api, send_json, structured_message
from tests._skill_driver import (
    FixtureServer,
    drive,
    fixture_server,
    header,
    journal,
    load_driver,
    make_skill,
    result_of,
    run_dir,
)

pytest.importorskip("fastmcp")

REPO = Path(__file__).resolve().parent.parent

MODULES = {
    "loopmods.py": """\
def prep(topic):
    return {"topic": topic, "noise": "UPSTREAM-NOT-DECLARED"}


def scale(values, factor):
    return [v * factor for v in values]


def report(best, count):
    return {"line": f"{best} ({count})"}
"""
}

OUTPUT = {"best": "str", "count": "int"}


def _loop_pipeline(
    *,
    tools: list[str] | None = None,
    tool_servers: dict[str, str] | None = None,
    output: Any = None,
    cap: int = 3,
) -> dict[str, Any]:
    loop: dict[str, Any] = {
        "id": "explore",
        "kind": "agent_loop",
        "description": "Find the best match for the topic.",
        "inputs": {"topic": "prepped.output.topic", "quota": "pipeline.input.quota"},
        "output": OUTPUT if output is None else output,
        "tools": ["search", "lookup", "write_note"] if tools is None else tools,
        "tool_servers": (
            {"search": "fx", "lookup": "fx", "write_note": "fx"}
            if tool_servers is None
            else tool_servers
        ),
        "loop_body": ["scaled", "vetted"],
        "termination": {"condition": "a best match is found", "max_iterations": cap},
    }
    if not loop["output"]:
        del loop["output"]
    reported: dict[str, Any] = {
        "id": "reported",
        "kind": "pure_function",
        "description": "Report the best match.",
        "impl": "extracted/loopmods.py:report",
        "inputs": {"best": "explore.output.best", "count": "explore.output.count"},
    }
    nodes = [
        {
            "id": "prepped",
            "kind": "pure_function",
            "description": "Prepare the topic.",
            "impl": "extracted/loopmods.py:prep",
            "inputs": {"topic": "pipeline.input.topic"},
        },
        loop,
        {
            "id": "scaled",
            "kind": "pure_function",
            "description": "Scale the values.\nA second line nobody needs to read.",
            "impl": "extracted/loopmods.py:scale",
            "input": {"values": "list[int]"},
            "constants": {"factor": 10},
        },
        {
            "id": "vetted",
            "kind": "llm_judge",
            "description": "Vet one title.",
            "signature_spec": {
                "prompt": "Keep {{ title }} only if it names the topic.",
                "input_schema": {
                    "type": "object",
                    "properties": {"title": {"type": "string"}},
                    "required": ["title"],
                },
                "output_schema": {
                    "type": "object",
                    "properties": {"keep": {"type": "boolean"}},
                    "required": ["keep"],
                    "additionalProperties": False,
                },
            },
        },
    ]
    edges = [{"from": "prepped", "to": "explore"}]
    exits = ["explore"]
    if output is None:
        nodes.append(reported)
        edges.append({"from": "explore", "to": "reported"})
        exits = ["reported"]
    return {
        "name": "looper",
        "version": "0.1.0",
        "input": {"type": "In", "required": ["topic", "quota"]},
        "nodes": nodes,
        "edges": edges,
        "entry_nodes": ["prepped"],
        "exit_nodes": exits,
    }


@pytest.fixture(scope="module")
def server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[FixtureServer]:
    with fixture_server(tmp_path_factory.mktemp("mcp") / "hits.jsonl") as s:
        yield s


@pytest.fixture(scope="module")
def search_only(tmp_path_factory: pytest.TempPathFactory) -> Iterator[FixtureServer]:
    """A second server, offering only ``search``."""
    hits = tmp_path_factory.mktemp("other") / "hits.jsonl"
    with fixture_server(hits, only=["search"]) as s:
        yield s


@pytest.fixture
def fx(server: FixtureServer, monkeypatch: pytest.MonkeyPatch) -> FixtureServer:
    server.reset()
    monkeypatch.setenv("ROTE_MCP_FX_URL", server.url)
    return server


@pytest.fixture
def other(search_only: FixtureServer, monkeypatch: pytest.MonkeyPatch) -> FixtureServer:
    search_only.reset()
    monkeypatch.setenv("ROTE_MCP_OTHER_URL", search_only.url)
    return search_only


def _skill(tmp_path: Path, **kw: Any) -> Path:
    return make_skill(tmp_path, _loop_pipeline(**kw), MODULES)


def _start(scripts: Path, **kw: Any) -> tuple[str, str]:
    status, out = drive(scripts, "start", '{"topic": "owl", "quota": 2}', **kw)
    assert status == 0, out
    assert header(out)["kind"] == "LOOP", out
    return header(out)["run"], out


def _tool(scripts: Path, run: str, tool: str, arguments: Any, **kw: Any) -> tuple[int, str]:
    raw = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return drive(scripts, "tool", run, "explore", tool, raw, **kw)


def _calls(out: str) -> int:
    return int(journal(out)["tool_calls"].get("explore", 0))


# ───────── the LOOP block, and a loop run through to DONE ─────────


def test_the_loop_block_carries_the_task_its_inputs_tools_rules_and_answer_schema(
    tmp_path: Path, fx: FixtureServer
) -> None:
    run, out = _start(_skill(tmp_path))
    lines = out.splitlines()
    assert lines[0] == f"ROTE/1 LOOP looper run={run} step=explore calls=0/3"
    # The task is the loop lanes' own prompt, with the loop's bound inputs
    # where the emitted loop puts its payload, and nothing else upstream.
    assert "Find the best match for the topic.\n\n" + '{"topic":"owl","quota":2}' in out
    assert "UPSTREAM-NOT-DECLARED" not in out
    assert "Pipeline steps available to you as tools: scaled" in out
    assert "Stop when: a best match is found" in out
    # Tools: the declared MCP tools with their live input schemas, and the
    # code body sub-node with its payload's shape (its constant left out).
    tools = lines[lines.index("--- TOOLS (at most 3 calls in all; 3 left) ---") + 2 :]
    assert tools[:3] == [
        "scaled {values: [integer]}: Scale the values.",
        "search {query: string}",
        "lookup {id: string}",
    ]
    assert "factor" not in tools[0]
    # A write tool in a read-only run is not offered, and the block says so.
    assert not any(line.startswith("write_note") for line in tools)
    assert "they may write: write_note." in out
    # The judge body is a rule the agent applies, not a tool.
    assert "vetted: Keep {{ title }} only if it names the topic." in out
    assert "vetted gives: {keep: boolean}" in out
    schema_line = lines[lines.index("--- ANSWER SCHEMA (json) ---") + 1]
    assert json.loads(schema_line)["required"] == ["best", "count"]
    assert lines[-1] == f"rote skill run looper answer {run} explore '<your JSON>'"
    assert fx.calls() == []  # listing tools calls none of them


def test_a_loop_calls_its_tools_and_its_answer_feeds_the_next_step(
    tmp_path: Path, fx: FixtureServer
) -> None:
    scripts = _skill(tmp_path)
    run, out = _start(scripts)
    status, found = _tool(scripts, run, "search", {"query": "owl"})
    assert status == 0, found
    assert found.splitlines()[0] == (
        f"ROTE/1 LOOP looper run={run} step=explore tool=search call=1/3"
    )
    # The result as the step would receive it, compact.
    assert found.splitlines()[1] == json.dumps(
        [{"title": f"owl {i}", "rank": i} for i in range(3)], separators=(",", ":")
    )
    assert [c["arguments"] for c in fx.calls("search")] == [{"query": "owl"}]
    # A body sub-node runs in code, its constant merged last: the agent's
    # own `factor` cannot override the pinned one.
    status, scaled = _tool(scripts, run, "scaled", {"values": [1, 2], "factor": 99})
    assert status == 0 and scaled.splitlines()[1] == "[10,20]", scaled
    assert _calls(out) == 2
    # An agent answer: --said is refused, as on a DECIDE.
    status, said = drive(scripts, "answer", run, "explore", "{}", "--said", "ok")
    assert status == 2 and "--said is only for a person's words" in said
    status, done = drive(scripts, "answer", run, "explore", '{"best": "owl 0", "count": 3}')
    assert status == 0 and header(done)["kind"] == "DONE", done
    assert result_of(done) == {"reported": {"line": "owl 0 (3)"}}
    node = json.loads((run_dir(done) / "nodes" / "explore.json").read_text())
    assert node["result"] == {"best": "owl 0", "count": 3}
    assert node["answered_by"] == "agent"
    assert journal(done)["answers"]["explore"]["answered_by"] == "agent"


def test_a_single_type_output_is_unwrapped_and_no_output_is_the_final_text(
    tmp_path: Path, fx: FixtureServer
) -> None:
    listed = _skill(tmp_path / "a", output="list[str]")
    run, out = _start(listed)
    schema = json.loads(
        out.splitlines()[out.splitlines().index("--- ANSWER SCHEMA (json) ---") + 1]
    )
    assert schema["required"] == ["result"]
    _, done = drive(listed, "answer", run, "explore", '{"result": ["owl 0", "owl 2"]}')
    assert result_of(done) == {"explore": ["owl 0", "owl 2"]}
    text = _skill(tmp_path / "b", output={})
    run, out = _start(text)
    _, done = drive(text, "answer", run, "explore", '{"result": "owl 0 is best"}')
    assert result_of(done) == {"explore": {"result": "owl 0 is best"}}


# ───────── what the proxy refuses ─────────


@pytest.mark.parametrize(
    ("tool", "why"),
    [
        ("delete_everything", "delete_everything is not one of this loop's tools"),
        # On the server, but not declared: the loop never reaches it.
        ("microsoft_docs_search", "microsoft_docs_search is not one of this loop's tools"),
        # A judge body is a rule for the agent, never a tool call.
        ("vetted", "vetted is not one of this loop's tools (lookup, scaled, search)"),
    ],
)
def test_an_undeclared_tool_is_refused_and_not_counted(
    tmp_path: Path, fx: FixtureServer, tool: str, why: str
) -> None:
    scripts = _skill(tmp_path)
    run, out = _start(scripts)
    status, refused = _tool(scripts, run, tool, {"query": "owl"})
    assert status == 2, refused
    assert refused.splitlines()[0].endswith(f"tool={tool} refused")
    assert why in refused and "Nothing was called." in refused
    assert fx.calls() == [] and _calls(out) == 0
    status, ok = _tool(scripts, run, "search", {"query": "owl"})  # negative half
    assert status == 0 and len(fx.calls("search")) == 1 and _calls(out) == 1


@pytest.mark.parametrize(
    ("arguments", "why"),
    [
        ({"query": 3}, "arguments.query: expected string, got integer"),
        ({}, "arguments: missing key 'query'"),
        ("[1]", "the arguments must be one JSON object"),
        ("{nope", "the arguments are not JSON (line 1 column 2"),
    ],
)
def test_arguments_are_checked_against_the_tools_schema_before_the_call(
    tmp_path: Path, fx: FixtureServer, arguments: Any, why: str
) -> None:
    scripts = _skill(tmp_path)
    run, out = _start(scripts)
    status, refused = _tool(scripts, run, "search", arguments)
    assert status == 2 and why in refused, refused
    assert fx.calls("search") == [] and _calls(out) == 0


def test_a_write_tool_is_withheld_without_consent(tmp_path: Path, fx: FixtureServer) -> None:
    scripts = _skill(tmp_path)
    run, out = _start(scripts)
    status, refused = _tool(scripts, run, "write_note", {"text": "hi"})
    assert status == 2, refused
    assert "fx does not mark it readOnlyHint" in refused
    assert "rote skill trust looper --allow-writes" in refused
    assert fx.calls("write_note") == [] and _calls(out) == 0
    # Consent arrives only as main()'s parameter: then it is offered and runs.
    run, out = _start(scripts, allow_writes=True)
    assert "write_note {text: string}" in out and "may write" not in out
    status, wrote = _tool(scripts, run, "write_note", {"text": "hi"}, allow_writes=True)
    assert status == 0 and wrote.splitlines()[1] == '{"written":"hi"}', wrote
    assert [c["arguments"] for c in fx.calls("write_note")] == [{"text": "hi"}]


def test_a_loop_whose_only_tools_may_write_does_not_start(
    tmp_path: Path, fx: FixtureServer
) -> None:
    scripts = _skill(tmp_path, tools=["write_note"], tool_servers={"write_note": "fx"})
    status, out = drive(scripts, "start", '{"topic": "owl", "quota": 2}')
    assert status == 1 and header(out)["kind"] == "ERROR"
    assert "every MCP tool the loop declares may write" in out
    assert "Fix: rote skill trust looper --allow-writes" in out


# ───────── which server a tool is called on ─────────


def test_a_tool_is_called_only_on_the_server_tool_servers_names(
    tmp_path: Path, fx: FixtureServer, other: FixtureServer
) -> None:
    """``other`` offers only search. Pinned there, lookup is refused, never
    found on fx instead; search, pinned there, goes there and not to fx."""
    pins = {"search": "other", "lookup": "other"}
    scripts = _skill(tmp_path, tools=["search", "lookup"], tool_servers=pins)
    status, out = drive(scripts, "start", '{"topic": "owl", "quota": 2}')
    assert status == 1 and header(out)["kind"] == "ERROR", out
    assert f"its tool lookup is served by other (tool_servers), and other at {other.url}" in out
    assert "does not offer it" in out
    run = header(out)["run"]
    assert journal(out)["pending"] is None  # no LOOP was handed out
    pins = {"search": "other", "lookup": "fx"}
    scripts = _skill(tmp_path / "ok", tools=["search", "lookup"], tool_servers=pins)
    run, out = _start(scripts)
    status, _ = _tool(scripts, run, "search", {"query": "owl"})
    assert status == 0
    assert [c["arguments"] for c in other.calls("search")] == [{"query": "owl"}]
    assert fx.calls("search") == []


def _registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **servers: FixtureServer) -> None:
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({"servers": {n: {"url": s.url} for n, s in servers.items()}}))
    monkeypatch.setenv("ROTE_MCP_CONFIG", str(path))


def test_an_unpinned_tool_binds_to_the_one_registered_server_that_offers_it(
    tmp_path: Path, fx: FixtureServer, other: FixtureServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ROTE_MCP_FX_URL")
    monkeypatch.delenv("ROTE_MCP_OTHER_URL")
    _registry(tmp_path, monkeypatch, fx=fx, other=other)
    # lookup: only fx offers it. search: both do, and nothing says which.
    scripts = _skill(tmp_path / "a", tools=["lookup"], tool_servers={})
    run, out = _start(scripts)
    status, _ = _tool(scripts, run, "lookup", {"id": "x"})
    assert status == 0 and len(fx.calls("lookup")) == 1
    scripts = _skill(tmp_path / "b", tools=["search"], tool_servers={})
    status, out = drive(scripts, "start", '{"topic": "owl", "quota": 2}')
    assert status == 1 and "its tool search is offered by fx and other" in out
    assert "pin it in the node's tool_servers" in out


# ───────── the cap ─────────


def test_the_call_cap_holds_across_a_login_and_a_resume(
    tmp_path: Path, fx: FixtureServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.test_mcp_client import _StatusServer

    scripts = _skill(tmp_path, cap=3)
    run, out = _start(scripts)
    assert _tool(scripts, run, "search", {"query": "a"})[0] == 0
    denied = _StatusServer(401)
    try:
        monkeypatch.setenv("ROTE_MCP_FX_URL", denied.url)
        status, login = _tool(scripts, run, "lookup", {"id": "b"})
        assert status == 0 and login.splitlines()[0].endswith(
            "step=explore kind=login server=fx"
        ), login
        assert _calls(out) == 2  # the call that met the 401 counts
        status, still = drive(scripts, "resume", run)  # not logged in yet
        assert "kind=login server=fx" in still.splitlines()[0]
    finally:
        denied.close()
    monkeypatch.setenv("ROTE_MCP_FX_URL", fx.url)  # the login landed
    status, again = drive(scripts, "resume", run)
    assert status == 0 and again.splitlines()[0].endswith("step=explore calls=2/3"), again
    assert "(at most 3 calls in all; 1 left)" in again
    status, last = _tool(scripts, run, "search", {"query": "c"})
    assert status == 0 and "That was the loop's last tool call. Answer next:" in last
    status, refused = _tool(scripts, run, "search", {"query": "d"})
    assert status == 2 and "this loop has made all 3 of its tool calls" in refused
    assert f"rote skill run looper answer {run} explore" in refused
    assert [c["arguments"]["query"] for c in fx.calls("search")] == ["a", "c"]
    assert _calls(out) == 3
    status, done = drive(scripts, "answer", run, "explore", '{"best": "a", "count": 1}')
    assert status == 0 and result_of(done) == {"reported": {"line": "a (1)"}}


def test_a_tool_that_fails_is_reported_to_the_agent_and_counted(
    tmp_path: Path, fx: FixtureServer
) -> None:
    scripts = _skill(tmp_path, tools=["flaky"], tool_servers={"flaky": "fx"})
    run, out = _start(scripts)
    status, failed = _tool(scripts, run, "flaky", {"key": "k-loop"})
    assert status == 0 and failed.splitlines()[0].endswith("tool=flaky call=1/3 failed")
    assert "transient failure for k-loop" in failed
    status, ok = _tool(scripts, run, "flaky", {"key": "k-loop"})
    assert status == 0 and ok.splitlines()[1] == '{"key":"k-loop","ok":true}'
    assert len(fx.calls("flaky")) == 2 and _calls(out) == 2  # never retried by the proxy


def test_a_tool_call_for_a_step_that_is_not_a_waiting_loop_calls_nothing(
    tmp_path: Path, fx: FixtureServer
) -> None:
    scripts = _skill(tmp_path)
    run, out = _start(scripts)
    status, note = drive(scripts, "tool", run, "prepped", "search", '{"query": "x"}')
    assert status == 0 and "Note: prepped is not an agent loop waiting for its tools" in note
    assert header(note)["kind"] == "LOOP"  # the pending block, again
    assert fx.calls() == [] and _calls(out) == 0


# ───────── the answer ─────────


def test_a_bad_answer_is_re_asked_with_only_its_errors_then_stops(
    tmp_path: Path, fx: FixtureServer
) -> None:
    scripts = _skill(tmp_path)
    run, out = _start(scripts)
    status, again = drive(scripts, "answer", run, "explore", '{"best": 1}')
    assert status == 0
    assert again.splitlines()[0] == f"ROTE/1 LOOP looper run={run} step=explore attempt=2"
    assert "- answer: missing key 'count'" in again
    assert "- answer.best: expected string, got integer" in again
    assert "--- TASK ---" not in again  # the re-ask repeats nothing
    assert not (run_dir(out) / "nodes" / "explore.json").exists()
    _, again = drive(scripts, "answer", run, "explore", '{"best": "a", "count": 1, "x": 0}')
    assert "- answer: unexpected key 'x'" in again
    status, stopped = drive(scripts, "answer", run, "explore", "[]")
    assert status == 1 and "3 answers did not validate" in stopped
    # A stopped loop's tools stay shut until the run is resumed.
    status, shut = _tool(scripts, run, "search", {"query": "owl"})
    assert status == 1 and header(shut)["kind"] == "ERROR" and fx.calls("search") == []
    status, back = drive(scripts, "resume", run)
    assert status == 0 and header(back)["kind"] == "LOOP"
    status, done = drive(scripts, "answer", run, "explore", '{"best": "a", "count": 1}')
    assert status == 0 and result_of(done) == {"reported": {"line": "a (1)"}}


def test_re_entry_refuses_a_loop_result_the_driver_did_not_write(
    tmp_path: Path, fx: FixtureServer
) -> None:
    scripts = _skill(tmp_path)
    run, out = _start(scripts)
    nodes = run_dir(out) / "nodes"
    (nodes / "explore.json").write_text(json.dumps({"result": {"best": "x", "count": 0}}))
    status, refused = drive(scripts, "status", run)
    assert status == 0 and header(refused)["kind"] == "LOOP"  # status reads no results
    state = json.loads((run_dir(out) / "run.json").read_text())
    state["pending"], state["state"] = None, "running"
    (run_dir(out) / "run.json").write_text(json.dumps(state))
    status, refused = drive(scripts, "resume", run)
    assert status == 1 and "nodes/explore.json was not written by this driver" in refused


# ───────── --judges helper: a judge body is a tool the helper answers ─────────


class _Messages(BaseHTTPRequestHandler):
    """A local Messages stub: every judge it is asked keeps the title."""

    requests: list[dict[str, Any]] = []

    def do_POST(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        payload = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
        type(self).requests.append(payload)
        if reject_like_the_api(self, payload):
            return
        send_json(self, structured_message(payload, {"keep": True}))

    def log_message(self, *args: object) -> None:
        return


def test_under_helper_judges_a_judge_body_is_a_tool_the_helper_answers(
    tmp_path: Path, fx: FixtureServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ROTE_INFERENCE", "api")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    _Messages.requests = []
    stub = HTTPServer(("127.0.0.1", 0), _Messages)
    threading.Thread(target=stub.serve_forever, daemon=True).start()
    monkeypatch.setenv("ROTE_BASE_URL_VETTED", f"http://127.0.0.1:{stub.server_port}")
    try:
        ir = Pipeline.model_validate(_loop_pipeline())
        scripts = tmp_path / "scripts"
        write_scripts(scripts, build_plan(ir, mcp="live", judges="helper"))
        (scripts / "extracted").mkdir()
        (scripts / "extracted" / "__init__.py").write_text("", encoding="utf-8")
        (scripts / "extracted" / "loopmods.py").write_text(MODULES["loopmods.py"])
        run, out = _start(scripts)
        assert "vetted {title: string}: Vet one title." in out
        assert "--- RULES" not in out  # nothing left for the agent to apply itself
        status, bad = _tool(scripts, run, "vetted", {"title": 3})
        assert status == 2 and "arguments.title: expected string" in bad
        assert _Messages.requests == []
        status, kept = _tool(scripts, run, "vetted", {"title": "owl 0"})
        assert status == 0 and kept.splitlines()[1] == '{"keep":true}', kept
        (sent,) = _Messages.requests
        assert sent["messages"][0]["content"] == "Keep owl 0 only if it names the topic."
    finally:
        stub.shutdown()
        stub.server_close()


# ───────── size ─────────


def test_bdr_outreachs_lead_generation_loop_block_stays_small(tmp_path: Path) -> None:
    """Every byte of a LOOP block stays in the agent's context. bdr's
    lead_generation_loop, with its real task, rule and body tool, an empty
    input and its two ZoomInfo tools at a one-field schema."""
    pipeline = load_pipeline(REPO / "examples/bdr-outreach/expected/pipeline.yaml")
    plan = build_plan(pipeline, mcp="live")
    scripts = tmp_path / "scripts"
    write_scripts(scripts, plan)
    driver = load_driver(scripts).Driver(scripts, skill="bdr-outreach")
    step = driver.by_id["lead_generation_loop"]
    tools, withheld = driver._loop_tools({**step, "tools": []}, "0" * 12)
    assert list(tools) == ["enrich_contact_batch"] and withheld == {}
    for name in step["tools"]:
        tools[name] = {"server": "zoominfo", "schema": _ONE_FIELD, "description": "Search."}
    head, tail = step["task"]
    pending = {"task": head + "{}" + tail, "tools": tools, "withheld": {}}
    text = driver._loop_block("0" * 12, step, pending, 0).text()
    assert "vet_contact: You are vetting a contact" in text  # the judge, as a rule
    assert "enrich_contact_batch {contacts: [any]}: Batch enrich contacts" in text
    assert "batch_size" not in text  # its constant never reaches the agent
    assert len(text.encode()) <= 2600, len(text.encode())


_ONE_FIELD = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}


def test_every_committed_loop_builds_a_block(tmp_path: Path) -> None:
    """Each example's loops reach the block builder with nothing missing."""
    for ir in sorted(REPO.glob("examples/*/expected/pipeline.yaml")):
        pipeline = load_pipeline(ir)
        if not any(n.kind == "agent_loop" for n in pipeline.nodes):
            continue
        plan = build_plan(Pipeline.model_validate(pipeline.model_dump()), mcp="live")
        scripts = tmp_path / ir.parent.parent.name / "scripts"
        write_scripts(scripts, plan)
        driver = load_driver(scripts).Driver(scripts)
        for step in (s for s in plan["steps"] if s["kind"] == "loop"):
            tools, _ = driver._loop_tools({**step, "tools": []}, "0" * 12)
            pending = {"task": "".join(step["task"]), "tools": tools, "withheld": {}}
            text = driver._loop_block("0" * 12, step, pending, 0).text()
            assert text.startswith(f"ROTE/1 LOOP {plan['name']} "), ir
