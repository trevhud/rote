"""The compile bench's behavioral fidelity scorer and fixture MCP server.

No inference. The scorer is pure; the fixture server test starts the real
server process and calls it with the same fastmcp client the emitted
pipelines use, because a fixture that answers differently from what a
pipeline sees in production would make every fidelity number wrong.
Expected numbers are worked out by hand from the fixtures.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from rote.bench.fidelity_score import (
    call_key,
    field_reproduced,
    flatten,
    index_trace,
    layout_path,
    normalize,
    score_case,
    stable_reference,
)


def test_normalize_folds_amount_text_case_and_space() -> None:
    assert normalize("$1,234.50") == normalize(1234.5) == 1234.5
    assert normalize(3304.150000001) == 3304.15
    assert normalize("75%") == 75.0
    assert normalize("  Final \n Report ") == "final report"
    assert normalize(True) is True
    # Step outputs carry what the pipeline's code returned, not only JSON.
    from datetime import date
    from decimal import Decimal

    assert normalize(Decimal("3304.150")) == 3304.15
    assert normalize(date(2026, 9, 30)) == "2026-09-30"
    assert field_reproduced("total", 3304.15, index_trace([{"t": Decimal("3304.15")}]))


def test_flatten_turns_record_lists_into_column_sets() -> None:
    flat = flatten(
        {
            "totals": {"batch_usd": 10},
            "ids": ["B", "A", "A"],
            "queue": [
                {"expense_id": "E2", "severity": "review", "detail": {"nested": 1}},
                {"expense_id": "E1", "severity": "block"},
            ],
        }
    )
    assert flat == {
        "totals.batch_usd": 10.0,
        "ids": {"set": ["a", "b"]},
        "queue[].expense_id": {"set": ["e1", "e2"]},
        "queue[].severity": {"set": ["block", "review"]},
    }


def _run(
    output: dict[str, Any] | None, calls: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    return {"output": output, "calls": calls or [], "error": None if output else "no result"}


def _call(tool: str, side_effect: bool = False, **arguments: Any) -> dict[str, Any]:
    return {"server": "drive", "tool": tool, "arguments": arguments, "side_effect": side_effect}


def test_only_fields_every_successful_run_agrees_on_are_stable() -> None:
    runs = [
        _run({"total": 10, "summary": "All good", "ids": ["A", "B"], "outcome": "completed"}),
        _run(
            {"total": "10.00", "summary": "Looks fine", "ids": ["B", "A"], "outcome": "completed"}
        ),
        _run(None),  # a failed run neither confirms nor breaks stability
    ]
    ref = stable_reference(runs)
    assert ref["successful_runs"] == 2
    assert ref["fields"] == {"total": 10.0, "ids": {"set": ["a", "b"]}, "outcome": "completed"}
    assert ref["unstable_fields"] == ["summary"]


def test_one_successful_run_makes_nothing_stable() -> None:
    """One run agrees with itself on everything; that measures no variance."""
    read = _call("read", file_id="A")
    ref = stable_reference([_run({"total": 1}, [read]), _run(None), _run(None)])
    assert ref["insufficient_runs"] is True
    assert ref["fields"] == {}
    assert ref["unstable_fields"] == ["total"]
    assert ref["reads"] == []
    assert ref["coverage"]["covered"] == 0


def test_the_same_number_nested_differently_is_stable() -> None:
    runs = [
        _run({"report": {"kpi": {"active_sites": 3}}, "ids": ["A"]}),
        _run({"report": {"header": {"active_sites": 3}}, "ids": ["A"]}),
    ]
    ref = stable_reference(runs, contract=["report", "ids"])
    assert ref["fields"] == {"report.active_sites": 3.0, "ids": {"set": ["a"]}}
    assert ref["coverage"]["rate"] == 1.0


def test_runs_that_swap_two_same_named_leaves_are_not_stable() -> None:
    """approved and rejected both end in `ids`; collapsing them to
    `decisions.ids` would make a run that swapped them agree with one that
    did not. The same for counts."""
    runs = [
        _run(
            {
                "decisions": {
                    "approved": {"ids": ["E1", "E2"], "n": 2},
                    "rejected": {"ids": ["E3"], "n": 1},
                }
            }
        ),
        _run(
            {
                "decisions": {
                    "approved": {"ids": ["E3"], "n": 1},
                    "rejected": {"ids": ["E1", "E2"], "n": 2},
                }
            }
        ),
    ]
    ref = stable_reference(runs, contract=["decisions"])
    assert ref["fields"] == {}
    assert ref["coverage"]["covered"] == 0
    assert ref["coverage"]["rate"] == 0.0
    assert ref["coverage"]["stable_leaves"] == 0
    # The same runs without the swap are stable, under the full paths.
    same = stable_reference([runs[0], runs[0]], contract=["decisions"])
    assert same["fields"]["decisions.approved.ids"] == {"set": ["e1", "e2"]}
    assert same["fields"]["decisions.rejected.ids"] == {"set": ["e3"]}


def test_a_record_list_marker_survives_the_layout_collapse() -> None:
    """A column of a list of records is not a single value of the same name."""
    assert layout_path("report.queue[].expense_id") == "report[].expense_id"
    assert layout_path("report.section.summary.expense_id") == "report.expense_id"
    assert layout_path("queue[].expense_id") == "queue[].expense_id"
    runs = [
        _run({"r": {"queue": [{"id": "A"}, {"id": "B"}]}, "s": {"x": {"id": "A"}}}),
        _run({"r": {"rows": [{"id": "B"}, {"id": "A"}]}, "s": {"y": {"id": "A"}}}),
    ]
    ref = stable_reference(runs, contract=["r", "s"])
    assert ref["fields"] == {"r[].id": {"set": ["a", "b"]}, "s.id": "a"}


def test_coverage_reports_the_leaf_ratio_beside_fields_covered() -> None:
    runs = [_run({"f": {"a": 1, "b": 2, "c": 3}}), _run({"f": {"a": 1, "b": 9, "c": 8}})]
    cov = stable_reference(runs, contract=["f"])["coverage"]
    assert (cov["covered"], cov["fields"]) == (1, 1)  # "covered" by one leaf...
    assert (cov["stable_leaves"], cov["leaves"], cov["leaf_rate"]) == (1, 3, 0.3333)


def test_coverage_separates_values_layout_and_absence() -> None:
    runs = [
        _run({"a": 1, "b": {"x": 5}, "c": 7, "d": 2, "e": 9}),
        _run({"a": 1, "b": {"y": 5}, "c": 8, "e": 9}),
    ]
    cov = stable_reference(runs, contract=["a", "b", "c", "d", "e"])["coverage"]
    assert (cov["fields"], cov["covered"], cov["rate"]) == (5, 2, 0.4)
    assert cov["low"] is True
    assert cov["uncovered_layout_disagreements"] == ["b"]  # same 5, other key
    assert cov["uncovered_value_disagreements"] == ["c"]  # 7 vs 8
    assert cov["uncovered_missing_in_some_runs"] == ["d"]


def test_fields_outside_the_contract_do_not_count() -> None:
    runs = [_run({"total": 1, "extra": {"a": 1}}), _run({"total": 1, "extra": {"a": 2}})]
    assert stable_reference(runs)["unstable_fields"] == ["extra.a"]
    limited = stable_reference(runs, contract=["total"])
    assert limited["fields"] == {"total": 1.0}
    assert limited["unstable_fields"] == []


def test_a_field_missing_from_one_run_is_unstable() -> None:
    ref = stable_reference([_run({"a": 1, "b": 2}), _run({"a": 1})])
    assert ref["fields"] == {"a": 1.0}
    assert ref["unstable_fields"] == ["b"]


def test_stable_calls_reads_intersect_and_side_effects_keep_order() -> None:
    read_a, read_b = _call("read", file_id="A"), _call("read", file_id="B")
    write_1, write_2 = _call("write", True, name="1"), _call("write", True, name="2")
    ref = stable_reference(
        [
            _run({"x": 1}, [read_a, read_b, write_1, write_2]),
            _run({"x": 1}, [read_b, write_1, write_2]),
        ]
    )
    assert ref["reads"] == [call_key(read_b)]
    assert ref["side_effects"] == [call_key(write_1), call_key(write_2)]
    assert call_key(read_a) in ref["seen_calls"]

    reordered = stable_reference(
        [_run({"x": 1}, [write_1, write_2]), _run({"x": 1}, [write_2, write_1])]
    )
    assert reordered["side_effects"] is None
    assert reordered["side_effects_unordered"] == sorted([call_key(write_1), call_key(write_2)])


def test_a_read_is_identified_by_the_data_it_got_not_its_spelling() -> None:
    """Two searches worded differently that the fixture answered with the
    same response read the same data; a write is identified by its args."""
    agent = {**_call("search", query="dock activity"), "response": 0}
    pipe = {**_call("search", query="label:dock-activity newer_than:1d"), "response": 0}
    other = {**_call("search", query="dock"), "response": 1}
    assert call_key(agent) == call_key(pipe)
    assert call_key(agent) != call_key(other)
    w1 = {**_call("write", True, name="a"), "response": 0}
    w2 = {**_call("write", True, name="b"), "response": 0}
    assert call_key(w1) != call_key(w2)


def test_values_match_by_value_wherever_the_pipeline_put_them() -> None:
    index = index_trace(
        [
            {"summary": {"batch": 3304.15, "queue": [{"id": "EXP-1"}, {"id": "EXP-2"}]}},
            {"auto": ["Dev Patel", "Emi Tanaka"], "review_items": 4, "lines": 3},
        ]
    )
    assert field_reproduced("batch_total_usd", 3304.15, index)
    assert field_reproduced("review_queue[].expense_id", {"set": ["exp-1", "exp-2"]}, index)
    assert field_reproduced("auto_approved_employees", {"set": ["dev patel", "emi tanaka"]}, index)
    assert not field_reproduced("batch_total_usd", 3304.16, index)
    # A small integer must sit under a key sharing a word with the field:
    # "lines": 3 must not reproduce a contract field "blocked_items": 3.
    assert field_reproduced("review_items", 4.0, index)
    assert not field_reproduced("blocked_items", 3.0, index)


def test_a_long_string_counts_inside_a_longer_one_and_a_short_one_does_not() -> None:
    index = index_trace(
        [{"report_path": "/srv/reports/2026-09-expense-audit.md", "note": "not final yet"}]
    )
    assert field_reproduced("report_filename", "2026-09-expense-audit.md", index)
    assert field_reproduced("period", "2026-09", index)
    assert not field_reproduced("report_status", "final", index)


def _reference() -> dict[str, Any]:
    read = _call("read", file_id="A")
    write = _call("write", True, name="r.md")
    return stable_reference(
        [
            _run(
                {"total": 12.5, "ids": ["A", "B"], "status": "final", "outcome": "completed"},
                [read, write],
            ),
            _run(
                {"total": 12.5, "ids": ["B", "A"], "status": "final", "outcome": "completed"},
                [read, write],
            ),
        ]
    )


def test_a_faithful_trace_scores_one() -> None:
    trace = {
        "completed": True,
        "objects": [{"sum": 12.5, "flagged": ["A", "B"], "report_status": "final"}],
        "calls": [_call("read", file_id="A"), _call("write", True, name="r.md")],
        "gate_fit": [{"fit": 1.0}],
    }
    s = score_case(_reference(), trace)
    assert (s["outputs"], s["outcome"], s["reads"], s["side_effects"], s["gates"]) == (
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    )
    assert s["score"] == 1.0


def test_each_kind_of_divergence_costs_its_component() -> None:
    trace = {
        "completed": False,  # died: outcome 0
        "objects": [{"sum": 12.5, "flagged": ["A"]}],  # ids wrong, status missing: 1 of 3
        "calls": [_call("read", file_id="Z")],  # read missed and unexpected; no write
        "gate_fit": [{"fit": 0.5}],
    }
    s = score_case(_reference(), trace)
    assert s["outputs"] == pytest.approx(0.3333)
    assert s["fields_missing"] == ["ids", "status"]
    assert s["outcome"] == 0.0
    assert s["reads"] == 0.0  # recall 0, precision 0
    assert s["side_effects"] == 0.0
    assert s["gates"] == 0.5
    assert s["score"] == pytest.approx((0.3333 + 0 + 0 + 0 + 0.5) / 5, abs=1e-4)


def test_reads_charge_extra_calls_and_side_effects_score_partial_order() -> None:
    """Recall alone would reward a pipeline that reads everything; a write
    sequence that is half right is half right, not wrong."""
    w = [_call("write", True, name=str(i)) for i in range(4)]
    read = _call("read", file_id="A")
    ref = stable_reference([_run({"x": 1}, [read, *w]), _run({"x": 1}, [read, *w])])
    trace = {
        "completed": True,
        "objects": [],
        # Every reference read plus one no agent made; writes 0 and 2 kept,
        # in order, of 4.
        "calls": [read, _call("read", file_id="EXTRA"), w[0], w[2]],
    }
    s = score_case(ref, trace)
    assert s["reads"] == 0.75  # recall 1.0, precision 0.5
    assert s["reads_unexpected"] == [call_key(_call("read", file_id="EXTRA"))]
    assert s["side_effects"] == 0.5


def test_a_field_every_run_left_uncomputed_is_not_an_output() -> None:
    """The stop case: agents agree a field is null because the skill stopped
    before computing it. A pipeline that also stopped has nothing to show."""
    ref = stable_reference(
        [
            _run({"late": ["E1"], "total": None, "outcome": "stopped"}),
            _run({"late": ["E1"], "total": None, "outcome": "stopped"}),
        ]
    )
    s = score_case(ref, {"completed": False, "objects": [{"late_ids": ["E1"]}], "calls": []})
    assert s["fields_checked"] == 1
    assert (s["outputs"], s["outcome"]) == (1.0, 1.0)


def test_components_with_nothing_to_compare_are_left_out() -> None:
    ref = stable_reference([_run({"total": 99.5}), _run({"total": 99.5})])
    s = score_case(ref, {"completed": True, "objects": [{"t": 99.5}], "calls": []})
    assert (s["reads"], s["side_effects"], s["gates"], s["outcome"]) == (None, None, None, None)
    assert s["score"] == 1.0


# ───────── the fixture server, end to end ─────────


def test_fixture_server_answers_from_fixtures_and_logs_every_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Started as the bench starts it, called through rote's own MCP client."""
    pytest.importorskip("fastmcp")
    from rote.bench.fidelity import Case, CaseFile, fixture_servers, read_calls
    from rote.mcp._runtime_helper import call_mcp_tool

    casefile = CaseFile(
        example_dir=tmp_path,
        output_fields={},
        servers={
            "drive": {
                "tools": {
                    "read_file_content": {
                        "description": "read a file",
                        "parameters": {
                            "type": "object",
                            "properties": {"file_id": {"type": "string"}},
                            "required": ["file_id"],
                        },
                    },
                    "search": {
                        "side_effect": False,
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                        },
                    },
                }
            }
        },
    )
    case = Case(
        id="c",
        description="",
        input={},
        gates=[],
        gate_instructions=None,
        fixtures={
            "drive": {
                "read_file_content": [
                    {"when": {"file_id": "A"}, "result": "site,pct\nEAST1,80"},
                    {"when": {"file_id": "B"}, "result": {"rows": [1, 2]}},
                ],
                "search": [
                    {"when_contains": {"query": "DOCK"}, "result": [{"id": "t1"}, {"id": "t2"}]},
                    # A catch-all, so only argument validation can refuse
                    # the misnamed call below.
                    {"result": []},
                ],
            }
        },
    )
    with fixture_servers(casefile, case, tmp_path / "work") as (urls, log):
        monkeypatch.setenv("ROTE_MCP_DRIVE_URL", urls["drive"])
        a = asyncio.run(call_mcp_tool("drive", None, "read_file_content", {"file_id": "A"}))
        b = asyncio.run(call_mcp_tool("drive", None, "read_file_content", {"file_id": "B"}))
        found = asyncio.run(
            call_mcp_tool("drive", None, "search", {"query": "label:dock-activity"})
        )
        with pytest.raises(Exception, match="no fixture response"):
            asyncio.run(call_mcp_tool("drive", None, "read_file_content", {"file_id": "C"}))
        # The argument the server does not have, as a compiled pipeline sent
        # it: a real server refuses it, so the fixture must not answer it.
        with pytest.raises(Exception, match="invalid arguments for search"):
            asyncio.run(
                call_mcp_tool("drive", None, "search", {"search_query": "label:dock-activity"})
            )
        calls = read_calls(log)
    assert a == "site,pct\nEAST1,80"
    assert b == {"rows": [1, 2]}
    assert found == [{"id": "t1"}, {"id": "t2"}]
    assert [(c["tool"], c["arguments"], c["matched"], c["response"]) for c in calls] == [
        ("read_file_content", {"file_id": "A"}, True, 0),
        ("read_file_content", {"file_id": "B"}, True, 1),
        ("search", {"query": "label:dock-activity"}, True, 0),
        ("read_file_content", {"file_id": "C"}, False, None),
        ("search", {"search_query": "label:dock-activity"}, False, None),
    ]
    assert "search_query" in calls[-1]["invalid_arguments"]
    assert calls[0]["invalid_arguments"] is None
    assert json.loads(json.dumps(calls))  # the log is plain JSON


def test_fixture_argument_checks_match_a_real_servers() -> None:
    from rote.bench.fixture_mcp import _schema_error

    schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}, "max_results": {"type": "integer"}},
        "required": ["query"],
    }
    assert _schema_error(schema, {"query": "x", "max_results": 5}) is None
    assert "missing required argument(s): query" in str(_schema_error(schema, {}))
    assert "unknown argument(s): search_query" in str(
        _schema_error(schema, {"query": "x", "search_query": "x"})
    )
    assert "must be integer" in str(_schema_error(schema, {"query": "x", "max_results": "5"}))
    assert "must be integer" in str(_schema_error(schema, {"query": "x", "max_results": True}))
    assert _schema_error({**schema, "additionalProperties": True}, {"query": "x", "z": 1}) is None


def test_every_recorded_reference_call_passes_its_fixture_schema() -> None:
    """The references were recorded before the fixtures checked arguments;
    a reference call the check would now refuse would make the reference
    itself unreachable."""
    from rote.bench.fidelity import load_cases
    from rote.bench.fixture_mcp import _schema_error

    from .conftest import EXAMPLES_DIR

    for example in ("expense-audit", "ops-report"):
        casefile = load_cases(EXAMPLES_DIR / example)
        reference = json.loads(casefile.reference_path.read_text(encoding="utf-8"))
        for case in reference["cases"].values():
            for run in case["runs"]:
                for call in run["calls"]:
                    tool = casefile.servers[call["server"]]["tools"][call["tool"]]
                    assert _schema_error(tool["parameters"], call["arguments"]) is None, call


def test_a_missing_fixture_runtime_stops_the_bench_instead_of_scoring_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wheel without the mcp extra once scored ops-report 0.0 with only
    "fixture MCP server exited with code 1" to show for it."""
    import importlib.util

    from rote.bench import fidelity

    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name, *a: None if name == "fastmcp" else real(name, *a)
    )
    casefile = fidelity.CaseFile(
        example_dir=tmp_path, output_fields={}, servers={"drive": {"tools": {}}}
    )
    case = fidelity.Case(
        id="c", description="", input={}, gates=[], gate_instructions=None, fixtures={}
    )
    with (
        pytest.raises(fidelity.FixtureError, match="need fastmcp"),
        fidelity.fixture_servers(casefile, case, tmp_path / "w"),
    ):
        pass


@pytest.mark.slow
def test_a_misnamed_tool_argument_is_refused_and_blamed_on_the_binding(
    tmp_path: Path,
) -> None:
    """The post-merge bug, end to end: a compile sends `search_query` to a
    tool that takes `query`. The fixture refuses it, the case lists the
    refused call with the reason, and the failure is the mcp binding's,
    not a replay of the node's impl (here a stub that never runs)."""
    pytest.importorskip("dbos")
    pytest.importorskip("fastmcp")
    import yaml

    from rote.bench.fidelity import pipeline_fidelity

    example = tmp_path / "example"
    (example / "fidelity").mkdir(parents=True)
    (example / "skill").mkdir()
    schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }
    # The tool's schema is the skill's bundled tools/list; cases.yaml adds
    # only what a fixture needs beyond it.
    (example / "skill" / "mcp-tools.json").write_text(
        json.dumps(
            {"servers": {"gmail": {"tools": [{"name": "search_threads", "inputSchema": schema}]}}}
        ),
        encoding="utf-8",
    )
    tool = {"side_effect": False}
    (example / "fidelity" / "cases.yaml").write_text(
        yaml.safe_dump(
            {
                "output_fields": {"thread_ids": "ids of the dock threads"},
                "servers": {"gmail": {"tools": {"search_threads": tool}}},
                "cases": [
                    {
                        "id": "c",
                        "input": {"q": "label:dock-activity"},
                        "fixtures": {
                            "gmail": {
                                "search_threads": [
                                    {"when_contains": {"query": "dock"}, "result": [{"id": "t1"}]},
                                    {"result": []},
                                ]
                            }
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    good = {
        "server": "gmail",
        "tool": "search_threads",
        "arguments": {"query": "label:dock-activity"},
        "side_effect": False,
        "matched": True,
        "response": 0,
    }
    run = {"output": {"thread_ids": ["T1", "T2"], "outcome": "completed"}, "calls": [good]}
    (example / "fidelity" / "reference.json").write_text(
        json.dumps({"cases": {"c": {"runs": [run, run]}}}), encoding="utf-8"
    )

    compiled = tmp_path / "compiled"
    (compiled / "extracted").mkdir(parents=True)
    (compiled / "extracted" / "gmail.py").write_text(
        "def search(q):\n    raise NotImplementedError('a stub the mcp backend never runs')\n",
        encoding="utf-8",
    )
    (compiled / "pipeline.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "dock",
                "input": {"type": "In", "required": ["q"]},
                "nodes": [
                    {
                        "id": "fetch_dock_emails",
                        "kind": "external_call",
                        "description": "search the dock threads",
                        "impl": "extracted/gmail.py:search",
                        "inputs": {"q": "pipeline.input.q"},
                        "mcp": {
                            "server": "gmail",
                            "tool": "search_threads",
                            "args": {"search_query": "q"},
                        },
                        "retry": {"max": 0},
                    }
                ],
                "edges": [],
            }
        ),
        encoding="utf-8",
    )

    report = pipeline_fidelity(compiled, example, tmp_path / "work")
    case = report["cases"]["c"]
    assert case["refused_calls"] == [
        {
            "call": "gmail.search_threads",
            "arguments": {"search_query": "label:dock-activity"},
            "why": "missing required argument(s): query",
        }
    ]
    assert case["failure"]["node"] == "fetch_dock_emails"
    assert case["failure"]["at"] == "mcp gmail.search_threads"
    assert "NotImplementedError" not in (case["failure"]["exception"] or "")
    assert case["outcome"] == 0.0
    # The case says how its writes are matched; this one makes none.
    assert (case["reference"]["write_fields"], case["reference"]["write_fields_dropped"]) == (
        {},
        {},
    )


@pytest.mark.slow
def test_a_failed_run_names_the_node_and_line_that_raised(tmp_path: Path) -> None:
    """DBOS re-raises a step's error without its traceback; the replay
    rebuilds the failing node's inputs from what DBOS recorded and finds
    the line. Breaks the gold expense-audit at a known line to check it."""
    pytest.importorskip("dbos")
    import shutil

    from rote.adapters import get_adapter
    from rote.bench.fidelity import failure_site
    from rote.eval.empirical import run_pipeline_trial
    from rote.ir import load_pipeline

    from .conftest import EXAMPLES_DIR

    compiled = tmp_path / "compiled"
    shutil.copytree(
        EXAMPLES_DIR / "expense-audit" / "expected",
        compiled,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    checks = compiled / "extracted" / "checks.py"
    text = checks.read_text(encoding="utf-8")
    anchor = "    _, period_end = period_bounds(period)\n"
    assert text.count(anchor) == 1
    broken = text.replace(anchor, '    submissions[0]["no_such_field"]\n' + anchor)
    checks.write_text(broken, encoding="utf-8")
    line = broken[: broken.index("no_such_field")].count("\n") + 1

    pipeline = load_pipeline(compiled / "pipeline.yaml")
    app = tmp_path / "app"
    get_adapter("dbos", extracted_source_dir=compiled).emit(pipeline, app)
    payload = json.loads((EXAMPLES_DIR / "expense-audit" / "input.json").read_text())
    review = {"reviewer": "x", "approved_ids": [], "rejected_ids": []}
    run = run_pipeline_trial(
        app, payload, signals={"controller_review": review}, timeout_seconds=300
    )
    assert run.error is not None

    site = failure_site(app, pipeline, payload)
    assert site is not None
    assert site["node"] == "find_late_submissions"
    assert site["exception"] == "KeyError: 'no_such_field'"
    assert site["at"] == f"extracted/checks.py:{line}"


def test_a_run_error_keeps_only_the_line_that_says_what_failed() -> None:
    from rote.bench.fidelity import _last_line

    tb = "\n".join(
        [
            "Traceback (most recent call last):",
            '  File "/x/y.py", line 1',
            "",
            "KeyError: 'shipper_brand'",
        ]
    )
    assert _last_line(tb) == "KeyError: 'shipper_brand'"
    assert _last_line(None) is None


# ───────── a write is identified by what the reference agrees on ─────────


def _publish(**arguments: Any) -> dict[str, Any]:
    return {
        "server": "briefs",
        "tool": "publish_brief",
        "arguments": arguments,
        "side_effect": True,
    }


def _prose_reference() -> dict[str, Any]:
    """cf-limits-brief: every agent run published, each with its own markdown."""
    return stable_reference(
        [
            _run(
                {"x": 1},
                [_publish(title="Brief 2026-10-21", markdown=f"draft {i}", reviewer="Dana Lee")],
            )
            for i in range(3)
        ]
    )


def test_a_write_with_prose_arguments_is_keyed_on_the_ones_every_run_agreed_on() -> None:
    ref = _prose_reference()
    assert ref["write_fields"] == {"briefs.publish_brief": ["reviewer", "title"]}
    assert ref["side_effects"] == [
        'briefs.publish_brief({"reviewer": "dana lee", "title": "brief 2026-10-21"})'
    ]
    trace = {
        "completed": True,
        "objects": [],
        "calls": [
            _publish(title="Brief 2026-10-21", markdown="the pipeline's", reviewer="Dana Lee")
        ],
    }
    assert score_case(ref, trace)["side_effects"] == 1.0


def test_an_agreed_argument_still_decides_the_write() -> None:
    trace = {
        "completed": True,
        "objects": [],
        "calls": [_publish(title="Brief 2026-10-21", markdown="m", reviewer="Someone Else")],
    }
    assert score_case(_prose_reference(), trace)["side_effects"] == 0.0


def test_a_write_no_argument_of_which_agrees_is_identified_by_its_tool() -> None:
    ref = stable_reference([_run({"x": 1}, [_publish(markdown=f"draft {i}")]) for i in range(2)])
    assert ref["write_fields"] == {"briefs.publish_brief": []}
    same_tool = {"completed": True, "objects": [], "calls": [_publish(markdown="anything")]}
    assert score_case(ref, same_tool)["side_effects"] == 1.0
    other_tool = {
        "completed": True,
        "objects": [],
        "calls": [{**_publish(markdown="x"), "tool": "delete_brief"}],
    }
    assert score_case(ref, other_tool)["side_effects"] == 0.0


def test_a_run_that_wrote_nothing_makes_no_argument_agree() -> None:
    """One run never published: the write is not stable, and neither are its arguments."""
    ref = stable_reference([_run({"x": 1}, [_publish(title="t")]), _run({"x": 1}, [])])
    assert ref["write_fields"] == {"briefs.publish_brief": []}
    assert ref["side_effects"] is None and ref["side_effects_unordered"] == []


def test_a_reference_without_write_fields_keys_on_every_argument() -> None:
    """A reference recorded before write_fields existed scores as it did."""
    ref = {k: v for k, v in _prose_reference().items() if k != "write_fields"}
    write = _publish(title="Brief 2026-10-21", markdown="m", reviewer="Dana Lee")
    trace = {"completed": True, "objects": [], "calls": [write]}
    assert score_case(ref, trace)["side_effects_pipeline"] == [call_key(write)]


def test_the_scorecard_shows_how_writes_are_matched_and_what_was_not_compared() -> None:
    from rote.bench.runner import _coverage_line

    ref = _prose_reference()
    assert ref["write_fields_dropped"] == {"briefs.publish_brief": ["markdown"]}
    line = _coverage_line("c", {**ref["coverage"], **ref})
    assert (
        "writes briefs.publish_brief: matched on reviewer, title; not compared: markdown "
        "(the reference runs disagreed)"
    ) in line
    tool_only = stable_reference([_run({"x": 1}, [_publish(markdown=f"d{i}")]) for i in range(2)])
    assert "writes briefs.publish_brief: matched on the tool alone; not compared: markdown" in (
        _coverage_line("c", {**tool_only["coverage"], **tool_only})
    )
    agreed = stable_reference([_run({"x": 1}, [_publish(title="t")]) for _ in range(2)])
    assert agreed["write_fields_dropped"] == {}
    assert _coverage_line("c", {**agreed["coverage"], **agreed}).endswith(
        "writes briefs.publish_brief: matched on title"
    )


def _reviewer_reference() -> dict[str, Any]:
    return stable_reference(
        [
            _run({"reviewer": "Dana Lee"}, [_publish(title="t", reviewer="Dana Lee")])
            for _ in range(2)
        ]
    )


def test_a_value_the_pipeline_sends_in_a_write_is_an_output_it_produced() -> None:
    """cf-limits-brief: the reviewer exists only in the gate's answer and the
    publish arguments; the pipeline carried it through, so it counts."""
    trace = {
        "completed": True,
        "objects": [{"brief_id": "b-1"}],
        "calls": [_publish(title="t", reviewer="Dana Lee")],
    }
    s = score_case(_reviewer_reference(), trace)
    assert s["fields_missing"] == [] and s["outputs"] == 1.0


@pytest.mark.parametrize(
    "calls",
    [
        [],  # never sent on: the gate's answer alone is not output
        [_call("lookup", reviewer="Dana Lee")],  # a read's arguments are not output
        [_publish(title="t", reviewer="Someone Else")],
    ],
)
def test_a_value_not_in_the_pipelines_output_or_writes_is_missing(
    calls: list[dict[str, Any]],
) -> None:
    trace = {"completed": True, "objects": [{"brief_id": "b-1"}], "calls": calls}
    assert score_case(_reviewer_reference(), trace)["fields_missing"] == ["reviewer"]


@pytest.mark.parametrize(
    "refusal",
    [
        {"matched": False},
        {"matched": True, "invalid_arguments": "missing required argument(s): markdown"},
    ],
)
def test_a_refused_write_earns_no_output(refusal: dict[str, Any]) -> None:
    """A write a real server would reject produced nothing."""
    trace = {
        "completed": True,
        "objects": [{"brief_id": "b-1"}],
        "calls": [{**_publish(title="t", reviewer="Dana Lee"), **refusal}],
    }
    assert score_case(_reviewer_reference(), trace)["fields_missing"] == ["reviewer"]
    accepted = {**trace, "calls": [{**_publish(title="t", reviewer="Dana Lee"), "matched": True}]}
    assert score_case(_reviewer_reference(), accepted)["fields_missing"] == []


def _answered(tool: str, response: Any, **arguments: Any) -> dict[str, Any]:
    return {**_call(tool, **arguments), "response": response}


def _ran(*calls: dict[str, Any]) -> dict[str, Any]:
    return {"completed": True, "objects": [], "calls": list(calls)}


def test_a_live_answer_that_changed_is_drift_not_a_lost_read() -> None:
    """A live server reindexed between the reference and the run: the
    pipeline sent the reference's exact request and got a new answer. It
    read faithfully; the change is reported as drift, not scored."""
    ref = stable_reference(
        [
            _run({"x": 1}, [_answered("search", "aaa", query="Workers KV limits")]),
            _run({"x": 1}, [_answered("search", "aaa", query="Workers KV limits")]),
        ]
    )
    s = score_case(ref, _ran(_answered("search", "bbb", query="workers kv limits")))
    assert s["reads"] == 1.0
    assert s["reads_missing"] == []
    assert s["reads_unexpected"] == []
    request = call_key(_call("search", query="Workers KV limits"))
    assert s["response_drift"] == [
        {"read": request, "reference_responses": ["aaa"], "pipeline_response": "bbb"}
    ]


def test_a_different_request_is_still_a_lost_read_and_no_drift() -> None:
    """The negative half: drift excuses a changed answer, never a changed
    request. Asking for something else is infidelity whatever comes back."""
    ref = stable_reference(
        [
            _run({"x": 1}, [_answered("search", "aaa", query="Workers KV limits")]),
            _run({"x": 1}, [_answered("search", "aaa", query="Workers KV limits")]),
        ]
    )
    s = score_case(ref, _ran(_answered("search", "bbb", query="KV")))
    assert s["reads"] == 0.0
    assert s["response_drift"] == []
    same = score_case(ref, _ran(_answered("search", "aaa", query="Workers KV limits")))
    assert (same["reads"], same["response_drift"]) == (1.0, [])


def test_a_request_every_reference_run_made_is_a_read_even_when_its_answer_moved() -> None:
    """The reference itself can straddle a reindex: the same request got two
    answers across its runs. The agent still made that request every time."""
    ref = stable_reference(
        [
            _run({"x": 1}, [_answered("search", "aaa", query="D1 limits")]),
            _run({"x": 1}, [_answered("search", "ccc", query="D1 limits")]),
        ]
    )
    s = score_case(ref, _ran(_answered("search", "ccc", query="D1 limits")))
    assert s["reads"] == 1.0
    assert s["response_drift"] == []
    drifted = score_case(ref, _ran(_answered("search", "ddd", query="D1 limits")))
    assert drifted["reads"] == 1.0
    assert drifted["response_drift"] == [
        {
            "read": call_key(_call("search", query="D1 limits")),
            "reference_responses": ["aaa", "ccc"],
            "pipeline_response": "ddd",
        }
    ]


def test_the_bench_prints_each_drifted_read_with_both_answers() -> None:
    from rote.bench.runner import coverage_lines

    request = call_key(_call("search", query="D1 limits"))
    coverage = {"covered": 1, "fields": 1, "stable_leaves": 1, "leaves": 1}
    drift = {"read": request, "reference_responses": ["aaa"], "pipeline_response": "bbb"}
    lines = coverage_lines(
        {"cases": {"base": {"reference": {"coverage": coverage}, "response_drift": [drift]}}}
    )
    assert any(
        f"drift {request}: the reference got aaa, this run got bbb" in line for line in lines
    ), lines
