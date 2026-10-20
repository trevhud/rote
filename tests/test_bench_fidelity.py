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
                    {"when_contains": {"query": "DOCK"}, "result": [{"id": "t1"}, {"id": "t2"}]}
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
        calls = read_calls(log)
    assert a == "site,pct\nEAST1,80"
    assert b == {"rows": [1, 2]}
    assert found == [{"id": "t1"}, {"id": "t2"}]
    assert [(c["tool"], c["arguments"], c["matched"], c["response"]) for c in calls] == [
        ("read_file_content", {"file_id": "A"}, True, 0),
        ("read_file_content", {"file_id": "B"}, True, 1),
        ("search", {"query": "label:dock-activity"}, True, 0),
        ("read_file_content", {"file_id": "C"}, False, None),
    ]
    assert json.loads(json.dumps(calls))  # the log is plain JSON


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
