"""The compile benchmark's scorer, on fixture IR pairs with known scores.

No compile, no inference: the benchmark itself is an eval run by hand
(docs/compile-bench.md). These pin the arithmetic a later rubric change
will be judged by, so every expected number below is worked out by hand
from the fixture, never recomputed from the code under test.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest

from rote.bench.collect import phase_durations
from rote.bench.report import aggregate, compare, headline
from rote.bench.runner import isolated_env
from rote.bench.score import (
    _Context,
    _max_weight_assignment,
    dependency_edges,
    match_nodes,
    score_pipeline,
)
from rote.ir import Pipeline, load_pipeline
from rote.isolation import STATE_PATHS

from .conftest import BDR_PIPELINE_YAML


def _pipeline(
    nodes: list[dict[str, Any]],
    edges: list[tuple[str, str]],
    required: list[str],
    optional: list[str] | None = None,
) -> Pipeline:
    return Pipeline.model_validate(
        {
            "name": "fixture",
            "input": {"type": "In", "required": required, "optional": optional or []},
            "nodes": nodes,
            "edges": [{"from": a, "to": b} for a, b in edges],
        }
    )


def _pure(id_: str, description: str, **extra: Any) -> dict[str, Any]:
    return {
        "id": id_,
        "kind": "pure_function",
        "description": description,
        "impl": f"extracted/m.py:{id_}",
        **extra,
    }


def _expected() -> Pipeline:
    return _pipeline(
        [
            {
                "id": "fetch",
                "kind": "external_call",
                "description": "fetch the ledger export",
                "impl": "extracted/m.py:fetch",
                "mandatory": True,
            },
            _pure("dedupe_rows", "remove duplicate rows by id", mandatory=True),
            {
                "id": "score_item",
                "kind": "llm_judge",
                "description": "score each item for risk",
                "signature": "signatures/score_item.py:ScoreItem",
                "fan_out": True,
            },
            {
                "id": "review",
                "kind": "hitl_gate",
                "description": "controller approves the flagged items",
                "signal": "approve",
                "output": {"approved_ids": "list[str]", "reviewer": "str"},
            },
            {
                "id": "publish",
                "kind": "external_call",
                "description": "publish the report to the drive folder",
                "impl": "extracted/m.py:publish",
            },
        ],
        [
            ("fetch", "dedupe_rows"),
            ("dedupe_rows", "score_item"),
            ("score_item", "review"),
            ("review", "publish"),
        ],
        required=["period"],
        optional=["region"],
    )


def _compiled() -> Pipeline:
    """Against _expected: one rename, one kind error, one missing node, one
    extra node, a lost fan_out, a spurious mandatory, a renamed gate signal
    with half its payload, a gate moved one wave early, one wrong input."""
    return _pipeline(
        [
            {
                "id": "fetch",
                "kind": "external_call",
                "description": "fetch the ledger export",
                "impl": "extracted/m.py:fetch",
                "mandatory": True,
            },
            _pure("remove_duplicates", "remove duplicate rows by id"),
            _pure("score_item", "score each item for risk"),
            {
                "id": "review",
                "kind": "hitl_gate",
                "description": "controller approves the flagged items",
                "signal": "approve_done",
                "output": {"approved_ids": "list[str]"},
            },
            _pure("extra_log", "write audit log line", fan_out=True, mandatory=True),
        ],
        [
            ("fetch", "remove_duplicates"),
            ("remove_duplicates", "score_item"),
            ("remove_duplicates", "review"),
            ("fetch", "extra_log"),
        ],
        required=["period", "user"],
    )


def test_matching_pairs_the_fixture_steps() -> None:
    pairs = {(m.expected_id, m.compiled_id) for m in match_nodes(_expected(), _compiled())}
    assert pairs == {
        ("fetch", "fetch"),
        ("dedupe_rows", "remove_duplicates"),
        ("score_item", "score_item"),
        ("review", "review"),
    }


def test_matching_does_not_depend_on_node_order() -> None:
    compiled = _compiled()
    shuffled = compiled.model_copy(update={"nodes": list(reversed(compiled.nodes))})
    random.Random(7).shuffle(shuffled.nodes)
    assert match_nodes(_expected(), shuffled) == match_nodes(_expected(), compiled)


def test_matching_takes_the_most_similar_pair_first() -> None:
    """One compiled node resembles two expected ones; the closer one wins,
    whichever expected id sorts first."""
    expected = _pipeline(
        [
            _pure("late_invoices", "flag late invoices"),
            _pure("late_submissions", "flag late submissions past the deadline"),
        ],
        [],
        required=[],
    )
    compiled = _pipeline(
        [_pure("late_submissions_check", "flag late submissions past the deadline")],
        [],
        required=[],
    )
    [m] = match_nodes(expected, compiled)
    assert (m.expected_id, m.compiled_id) == ("late_submissions", "late_submissions_check")


def _ext(id_: str, description: str, **extra: Any) -> dict[str, Any]:
    return {
        "id": id_,
        "kind": "external_call",
        "description": description,
        "impl": f"extracted/m.py:{id_}",
        **extra,
    }


def test_a_fully_renamed_compile_still_matches_by_structure() -> None:
    """Every id differs and only one description survives, so the old
    lexical matcher found one pair here. Hand-worked, pass 2 (weights:
    position 2, degree 1, neighbor_kinds 1.5, matched_neighbors 2,
    inputs 1.5, outputs 1, description 2, name 1.5; outputs are the
    fields downstream nodes read, never the declared output):

    fetch_rows ~ pull: position 1, degree 1, neighbor_kinds 1,
    matched_neighbors 1, outputs 1 (the field total_rows reads, rows ->
    row), description 1, name 0. Inputs (neither side has any) and MCP
    (only the expected node is bound: a missing binding is an MCP error,
    not a different step) do not apply.
    (2 + 1 + 1.5 + 2 + 1 + 2 + 0) / 11 = 0.8636.

    total_rows ~ sum_it: everything 1 except description, {sum, amount}
    vs {add, up, amount} = 1/4, and name 0 ("it" is a stopword). Nothing
    reads either node's output, so outputs does not apply.
    (2 + 1 + 1.5 + 2 + 1.5 + 0.5 + 0) / 11.5 = 0.7391.
    """
    expected = _pipeline(
        [
            _ext(
                "fetch_rows",
                "pull the ledger",
                output={"rows": "list"},
                mcp={"server": "ledger", "tool": "export_rows"},
            ),
            _pure(
                "total_rows",
                "sum amounts",
                inputs={"rows": "fetch_rows.output.rows"},
                output={"total": "float"},
            ),
        ],
        [("fetch_rows", "total_rows")],
        required=[],
    )
    compiled = _pipeline(
        [
            _ext("pull", "pull the ledger", output={"rows": "list"}),
            _pure(
                "sum_it",
                "add up amounts",
                inputs={"rows": "pull.output.rows"},
                output={"total": "float"},
            ),
        ],
        [("pull", "sum_it")],
        required=[],
    )
    got = {(m.expected_id, m.compiled_id): m.similarity for m in match_nodes(expected, compiled)}
    assert got == {("fetch_rows", "pull"): 0.8636, ("total_rows", "sum_it"): 0.7391}
    s = score_pipeline(expected, compiled)
    assert s["structure"]["kind_accuracy"] == 1.0
    assert s["graph"]["edge_f1"] == 1.0


def test_a_split_step_matches_the_half_that_sits_where_it_sat() -> None:
    """Expected file_report (render and save) became render_report ->
    save_report. Hand-worked; degree and neighbor_kinds weigh the
    downstream side 3 to 1, as (up + 3 down) / 4:

    pass 1: file_report ~ save_report: position 1, degree 1,
    neighbor_kinds (0 + 3) / 4 = 0.75, inputs 0, description 3/4, name
    1/3; outputs does not apply (nothing reads either). 6.125 / 9.5 =
    0.6447. file_report ~ render_report: position 0.5, degree
    (1 + 1.5) / 4 = 0.625, neighbor_kinds 1/4, inputs 1, outputs 0 (only
    render_report's markdown is read), description 2/4, name 1/3.
    5.0 / 10.5 = 0.4762.

    pass 2 adds matched_neighbors. file_report's predecessor fetch is not
    save_report's predecessor: 0, so 6.125 / 11.5 = 0.5326. Against
    render_report it is 1/2: (5.0 + 1) / 12.5 = 0.48, still lower. fetch's
    successor maps to save_report, but the compiled fetch feeds
    render_report: 0. So fetch ~ fetch is position 1, degree 1,
    neighbor_kinds (1 + 0) / 4, outputs 1, description 1, name 1:
    7.875 / 11 = 0.7159. save_report is the match, render_report the extra.
    """
    expected = _pipeline(
        [
            _ext("fetch", "pull the ledger", output={"rows": "list"}),
            _ext(
                "file_report",
                "render the report and save it to the drive",
                inputs={"rows": "fetch.output.rows"},
                output={"path": "str"},
            ),
        ],
        [("fetch", "file_report")],
        required=[],
    )
    compiled = _pipeline(
        [
            _ext("fetch", "pull the ledger", output={"rows": "list"}),
            _pure(
                "render_report",
                "render the report",
                inputs={"rows": "fetch.output.rows"},
                output={"markdown": "str"},
            ),
            _ext(
                "save_report",
                "save the report to the drive",
                inputs={"markdown": "render_report.output.markdown"},
                output={"path": "str"},
            ),
        ],
        [("fetch", "render_report"), ("render_report", "save_report")],
        required=[],
    )
    got = {(m.expected_id, m.compiled_id): m.similarity for m in match_nodes(expected, compiled)}
    assert got == {("fetch", "fetch"): 0.7159, ("file_report", "save_report"): 0.5326}
    s = score_pipeline(expected, compiled)["structure"]
    assert (s["node_recall"], s["node_precision"]) == (1.0, 0.6667)
    assert s["extra"] == ["render_report"]
    assert s["kind_accuracy"] == 1.0


def test_the_assignment_is_optimal_not_greedy() -> None:
    """Greedy takes the 0.9 and strands row 1 (total 0.9); the optimum
    gives up the 0.9 for two 0.8s (total 1.6)."""
    assert sorted(_max_weight_assignment([[0.9, 0.8], [0.8, 0.0]])) == [(0, 1), (1, 0)]
    # Rectangular, both ways round: only real rows and columns come back.
    assert sorted(_max_weight_assignment([[0.1, 0.7, 0.2]])) == [(0, 1)]
    assert sorted(_max_weight_assignment([[0.3], [0.6], [0.1]])) == [(1, 0)]
    assert _max_weight_assignment([]) == []


def test_gate_names_score_exact_and_normalized() -> None:
    """cs_lead_approved vs cs_lead_approval is one signal spelled twice.
    Exact: no signal match, payload {approver, region} of 6 keys = 0.3333.
    Normalized: both signals are {cs, lead, approv}; the pooled payload
    tokens are {approv, plan, seat, region} on both sides = 1.0."""

    def gate(signal: str, output: dict[str, str]) -> Pipeline:
        node = {
            "id": "sign_off",
            "kind": "hitl_gate",
            "description": "CS lead approves",
            "signal": signal,
            "output": output,
        }
        return _pipeline([node], [], required=[])

    expected = gate(
        "cs_lead_approval", {"approver": "str", "plan": "str", "seats": "int", "region": "str"}
    )
    compiled = gate(
        "cs_lead_approved",
        {"approver": "str", "approved_plan": "str", "approved_seats": "int", "region": "str"},
    )
    g = score_pipeline(expected, compiled)["gates"]
    assert (g["signal_matches"], g["signal_matches_normalized"]) == (0, 1)
    assert g["payload_jaccard_mean"] == pytest.approx(0.3333)
    assert g["payload_jaccard_normalized_mean"] == 1.0
    # A different signal stays different after normalizing.
    other = score_pipeline(expected, gate("controller_review", {"approver": "str"}))["gates"]
    assert other["signal_matches_normalized"] == 0


def test_gate_payload_reads_output_schema_properties_too() -> None:
    def gate(**shape: Any) -> Pipeline:
        node = {"id": "g", "kind": "hitl_gate", "description": "approve", "signal": "s", **shape}
        return _pipeline([node], [], required=[])

    as_dict = gate(output={"approved": "bool", "note": "str"})
    as_schema = gate(output_schema={"type": "object", "properties": {"approved": {}, "note": {}}})
    assert score_pipeline(as_dict, as_schema)["gates"]["payload_jaccard_mean"] == 1.0


def test_structure_and_kind_confusion() -> None:
    s = score_pipeline(_expected(), _compiled())["structure"]
    assert (s["expected_nodes"], s["compiled_nodes"], s["matched"]) == (5, 5, 4)
    assert s["matched_by_id"] == 3
    assert s["node_recall"] == 0.8
    assert s["node_precision"] == 0.8
    # 3 of 5 expected nodes kept their kind: the missing one counts as a miss.
    assert s["kind_accuracy"] == 0.6
    assert s["kind_accuracy_matched"] == 0.75
    # Matching-free: expected {ext 2, pure 1, judge 1, gate 1}, compiled
    # {ext 1, pure 3, gate 1}; shared minimums 1 + 1 + 1 over 5 nodes.
    assert s["kind_mix_overlap"] == 0.6
    assert s["missing"] == ["publish"]
    assert s["extra"] == ["extra_log"]
    assert s["confusion"] == {
        "(extra)->pure_function": 1,
        "external_call->(missing)": 1,
        "external_call->external_call": 1,
        "hitl_gate->hitl_gate": 1,
        "llm_judge->pure_function": 1,
        "pure_function->pure_function": 1,
    }


def test_edges_and_wave_ordering() -> None:
    g = score_pipeline(_expected(), _compiled())["graph"]
    # Among matched nodes, expected f->d, d->s, s->r; compiled f->d, d->s, d->r.
    assert (g["expected_edges"], g["compiled_edges"]) == (3, 3)
    assert g["edge_precision"] == pytest.approx(0.6667)
    assert g["edge_recall"] == pytest.approx(0.6667)
    assert g["edge_f1"] == pytest.approx(0.6667)
    assert (g["expected_waves"], g["compiled_waves"]) == (5, 3)
    # 6 pairs of matched nodes; only (score_item, review) flips from
    # "before" to "together".
    assert g["wave_order_agreement"] == pytest.approx(0.8333)


def test_fan_out_and_mandatory_precision_recall() -> None:
    s = score_pipeline(_expected(), _compiled())
    fan = s["fan_out"]
    assert (fan["true_positive"], fan["precision"], fan["recall"]) == (0, 0.0, 0.0)
    assert fan["missed"] == ["score_item"]
    assert fan["spurious"] == ["extra_log"]
    mand = s["mandatory"]
    assert (mand["expected"], mand["compiled"], mand["true_positive"]) == (2, 2, 1)
    assert (mand["precision"], mand["recall"], mand["f1"]) == (0.5, 0.5, 0.5)
    assert mand["missed"] == ["dedupe_rows"]
    assert mand["spurious"] == ["extra_log"]


def test_gates_compare_signal_and_payload() -> None:
    gates = score_pipeline(_expected(), _compiled())["gates"]
    assert (gates["expected"], gates["compiled"], gates["found"]) == (1, 1, 1)
    assert gates["signal_matches"] == 0
    [row] = gates["gates"]
    assert (row["signal_expected"], row["signal_compiled"]) == ("approve", "approve_done")
    assert row["payload_jaccard"] == 0.5
    assert gates["payload_jaccard_mean"] == 0.5


def test_inputs_overlap() -> None:
    inputs = score_pipeline(_expected(), _compiled())["inputs"]
    assert (inputs["precision"], inputs["recall"]) == (0.5, 0.5)
    assert inputs["jaccard"] == pytest.approx(0.3333)


def test_rates_with_nothing_to_find_are_none_not_perfect() -> None:
    """A flag nobody sets must not read as 1.0 (or 0.0) and skew the mean."""
    p = _pipeline([_pure("a", "do a")], [], required=["x"])
    s = score_pipeline(p, p)
    assert s["fan_out"]["precision"] is None
    assert s["fan_out"]["recall"] is None
    assert s["gates"]["payload_jaccard_mean"] is None
    assert s["graph"]["wave_order_agreement"] is None


def test_a_real_pipeline_scores_perfectly_against_itself() -> None:
    """Non-degenerate sanity: BDR has every node kind, gates, fan_out and
    mandatory nodes, so every rate here has a real denominator."""
    from rote.ir import load_pipeline

    bdr = load_pipeline(BDR_PIPELINE_YAML)
    s = score_pipeline(bdr, bdr)
    assert s["structure"]["kind_accuracy"] == 1.0
    assert s["structure"]["matched_by_id"] == len(bdr.nodes) == 15
    assert s["graph"]["edge_f1"] == 1.0
    assert s["graph"]["wave_order_agreement"] == 1.0
    assert (s["fan_out"]["recall"], s["mandatory"]["recall"]) == (1.0, 1.0)
    assert s["gates"]["signal_matches"] == s["gates"]["expected"] == 2
    assert s["inputs"]["jaccard"] == 1.0


def test_dependency_edges_include_input_references() -> None:
    p = _pipeline(
        [
            _pure("a", "make a"),
            _pure("b", "use a", inputs={"x": "a.output.value", "y": "pipeline.input.z"}),
        ],
        [("a", "b")],
        required=["z"],
    )
    stripped = p.model_copy(update={"edges": []})
    # The validator wants the edge; dependency_edges must not need it.
    assert dependency_edges(stripped) == {("a", "b")}


# ───────── report: flattening, aggregation, comparison ─────────


def _job(cost: float | None, kind: float, fan_recall: float | None) -> dict[str, Any]:
    return {
        "run": {"exit_code": 0, "cost_usd": cost, "wall_seconds": 100.0, "turns": 40},
        "structure": {"kind_accuracy": kind, "compiled_nodes": 9, "expected_nodes": 8},
        "fan_out": {"recall": fan_recall},
    }


def test_aggregate_averages_skips_none_and_sums_cost() -> None:
    rows = [headline(_job(2.0, 0.5, None)), headline(_job(3.0, 1.0, 1.0))]
    agg = aggregate(rows)
    assert agg["jobs"] == 2
    assert agg["kind_accuracy"] == 0.75
    assert agg["fan_out_recall"] == 1.0  # the None row does not drag it to 0.5
    assert agg["cost_usd"] == 5.0
    assert agg["wall_seconds"] == 200.0
    assert agg["turns"] == 40
    assert rows[0]["nodes"] == "9/8"


def test_compare_marks_direction_per_metric() -> None:
    old = {"aggregate": {"cost_usd": 10.0, "kind_accuracy": 0.8, "turns": 40, "jobs": 5}}
    new = {"aggregate": {"cost_usd": 12.5, "kind_accuracy": 0.9, "turns": 40, "jobs": 5}}
    d = compare(old, new)["aggregate"]
    assert d["cost_usd"] == {"old": 10.0, "new": 12.5, "delta": 2.5, "better": False}
    assert d["kind_accuracy"]["better"] is True
    assert d["turns"]["better"] is None
    assert "jobs" not in d


# ───────── collect: phase timing ─────────


def test_phase_durations_from_progress_jsonl(tmp_path: Path) -> None:
    events = [
        {"type": "log", "ts": 100.0, "message": "driver selected"},
        {"type": "phase", "ts": 110.0, "phase": 1, "phase_name": "read"},
        {"type": "turn", "ts": 120.0, "turn": 4},
        {"type": "phase", "ts": 150.0, "phase": 2, "phase_name": "classify"},
        {"type": "warning", "ts": 160.0, "message": "asking the compiler to repair (attempt 1)"},
        {"type": "turn", "ts": 170.0, "turn": 9},
        {"type": "complete", "ts": 200.0, "message": "done"},
        {"type": "summary", "ts": 999.0},
    ]
    path = tmp_path / "progress.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in events) + "\nnot json\n", encoding="utf-8")
    d = phase_durations(path)
    assert d["phases"] == [
        {"phase": 1, "name": "read", "seconds": 40.0},
        {"phase": 2, "name": "classify", "seconds": 50.0},
    ]
    assert d["before_first_phase_seconds"] == 10.0
    # Ends at `complete`, not at the later summary line.
    assert d["agent_seconds"] == 100.0
    assert d["last_turn"] == 9
    assert d["repair_warnings"] == 1


# ───────── runner: isolation ─────────


def test_isolated_env_moves_every_state_path_the_suite_isolates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every rote.isolation state path under the run's state dir (that list
    is checked against the source in test_isolated_rote.py), a cloud login
    in the environment dropped, and the per-run ROTE_* defaults dropped."""
    monkeypatch.setenv("ROTE_RUNTIME", "temporal")
    monkeypatch.setenv("ROTE_MODEL", "claude-opus-4-6")
    monkeypatch.setenv("ROTE_CLOUD_TOKEN", "rote_live_key")
    env = isolated_env(tmp_path / "state")
    assert "ROTE_CLOUD_TOKEN" not in env
    for key in STATE_PATHS:
        assert Path(env[key]).is_relative_to(tmp_path / "state"), key
    assert "ROTE_RUNTIME" not in env
    assert "ROTE_MODEL" not in env


def test_results_are_portable(tmp_path: Path) -> None:
    """A committed results.json must carry no local path: not the run dir,
    not a checkout inside it, not the home directory."""
    from rote.bench.runner import _replacements, portable

    run = tmp_path / "run"
    examples = run / "checkout" / "examples"
    data = {
        "argv": [f"{run}/expense-audit/r1/skill-src", "--out", str(run)],
        "nested": [{"error": f"emit failed in {examples}/x and {Path.home()}/y"}],
        "n": 3,
    }
    assert portable(data, _replacements(run, examples)) == {
        "argv": ["<run>/expense-audit/r1/skill-src", "--out", "<run>"],
        "nested": [{"error": "emit failed in <examples>/x and ~/y"}],
        "n": 3,
    }


def test_resume_keeps_scored_jobs_and_never_forgets_spend() -> None:
    """--resume re-runs a job that produced no pipeline, but the dollars it
    burned stay in spent_usd; a job that produced a (bad) pipeline is a
    result and is kept."""
    from rote.bench.runner import produced_pipeline, total_spend

    scored = {"headline": {}, "run": {"cost_usd": 2.5}}
    bad_ir = {"headline": {}, "run": {"cost_usd": 3.0}, "error": "invalid pipeline.yaml: x"}
    died = {"headline": {}, "run": {"cost_usd": 2.25}, "error": "no pipeline.yaml produced: limit"}
    assert produced_pipeline(scored)
    assert produced_pipeline(bad_ir)
    assert not produced_pipeline(died)
    assert not produced_pipeline({"skipped": "budget"})
    replaced = [{"skill": "x", "repeat": 1, "cost_usd": 2.25}]
    assert total_spend([scored, bad_ir], replaced) == 7.75


def test_a_failed_compile_still_reports_what_it_spent(tmp_path: Path) -> None:
    """No compile-metrics.json on the failure path, so the cost comes from
    the orchestrator's failure log; without it a paid job reads as free and
    --budget undercounts."""
    from rote.bench.collect import collect_job

    line = {
        "type": "log",
        "ts": 1.0,
        "message": (
            "usage before the failure: tokens in=538266 (538253 cached) out=108304; "
            "turns=30; cost=$2.2556864999999995"
        ),
    }
    (tmp_path / "progress.jsonl").write_text(json.dumps(line) + "\n", encoding="utf-8")
    (tmp_path / "bench-compile.stdout.log").write_text(
        json.dumps({"ok": False, "exit_code": 1, "error": "weekly limit"}), encoding="utf-8"
    )
    job = collect_job(
        tmp_path, tmp_path, BDR_PIPELINE_YAML, exit_code=1, wall_seconds=10.0, tsc=False
    )
    run = job["run"]
    assert run["cost_usd"] == pytest.approx(2.2556865)
    assert (run["turns"], run["output_tokens"]) == (30, 108304)
    assert run["input_tokens"] + run["cache_read_tokens"] == 538266
    assert run["usage_source"] == "failure_log"
    assert job["error"] == "no pipeline.yaml produced: weekly limit"
    assert headline(job)["cost_usd"] == pytest.approx(2.2556865)
    # And it counts against --budget the way _run_job feeds it.
    from rote.bench.runner import _Budget

    budget = _Budget(2.0)
    budget.add(run["cost_usd"])
    assert budget.exhausted()


_FIXTURES = Path(__file__).parent / "fixtures" / "bench"
_EXAMPLES = Path(__file__).parent.parent / "examples"


def test_a_moved_join_does_not_swap_the_steps_around_it() -> None:
    """A real compile moved the duplicate and late joins from the fanned
    audit onto the grouping node. With upstream and downstream weighed
    equally, that join placement outweighed position and name: the swap
    won 1.084 to 1.050 and scored a correct fan_out as one miss plus one
    spurious fan. What a step feeds is the steadier signal, and edgeF1
    already charges for where the join landed."""
    expected = load_pipeline(_EXAMPLES / "expense-audit" / "expected" / "pipeline.yaml")
    compiled = load_pipeline(_FIXTURES / "expense-audit-joins-moved.yaml")
    pairs = {m.expected_id: m.compiled_id for m in match_nodes(expected, compiled)}
    assert pairs["group_by_employee"] == "group_audit_employee_s_report"
    assert pairs["audit_report"] == "audit_employee_s_report_each"
    fan = score_pipeline(expected, compiled)["fan_out"]
    assert (fan["precision"], fan["recall"]) == (1.0, 1.0)


def test_ties_resolve_the_same_whatever_the_node_order() -> None:
    """Two compiled nodes identical in everything but id tie exactly, so
    only the id sort decides which one matches. Every input order, of
    either pipeline, must give the same answer: the lower id."""
    expected = _pipeline(
        [_pure("start", "load"), _pure("check", "check rows")],
        [("start", "check")],
        required=[],
    )
    compiled = _pipeline(
        [_pure("start", "load"), _pure("zz_check", "check rows"), _pure("aa_check", "check rows")],
        [("start", "zz_check"), ("start", "aa_check")],
        required=[],
    )
    want = {("check", "aa_check"), ("start", "start")}
    for e_step, c_step in [(1, 1), (-1, 1), (1, -1), (-1, -1)]:
        e = expected.model_copy(update={"nodes": expected.nodes[::e_step]})
        c = compiled.model_copy(update={"nodes": compiled.nodes[::c_step]})
        assert {(m.expected_id, m.compiled_id) for m in match_nodes(e, c)} == want


def test_an_edgeless_loop_body_node_takes_its_loops_rank() -> None:
    """start -> loop -> end has depths 0, 1, 2, so ranks 0, 0.5, 1. The
    body sub-node declares no edges, so plain depth would put it at the
    start (rank 0). It runs inside the loop, so it gets the loop's 0.5."""
    p = _pipeline(
        [
            _pure("start", "load"),
            {
                "id": "loop",
                "kind": "agent_loop",
                "description": "work the queue",
                "tools": ["search"],
                "loop_body": ["body"],
            },
            _pure("body", "one item"),
            _pure("end", "write"),
        ],
        [("start", "loop"), ("loop", "end")],
        required=[],
    )
    rank = _Context(p).rank
    assert (rank["start"], rank["loop"], rank["body"], rank["end"]) == (0.0, 0.5, 0.5, 1.0)


def test_a_gates_declared_payload_does_not_steer_its_match() -> None:
    """gateJ grades the payload, so the payload must not pick the gate.
    Two compiled gates sit in identical positions; the one whose payload
    copies the gold's must not win on that alone. Only fields a
    downstream node reads count as outputs, and none do here, so the two
    tie and the id sort picks aa_gate."""
    gate = {"kind": "hitl_gate", "description": "approve", "signal": "approve"}
    expected = _pipeline(
        [_pure("start", "load"), {"id": "review", **gate, "output": {"approved_ids": "list"}}],
        [("start", "review")],
        required=[],
    )
    compiled = _pipeline(
        [
            _pure("start", "load"),
            {"id": "aa_gate", **gate, "output": {"note": "str"}},
            {"id": "zz_gate", **gate, "output": {"approved_ids": "list"}},
        ],
        [("start", "aa_gate"), ("start", "zz_gate")],
        required=[],
    )
    pairs = {m.expected_id: m.compiled_id for m in match_nodes(expected, compiled)}
    assert pairs["review"] == "aa_gate"


@pytest.mark.parametrize(
    ("metadata", "turns"),
    [
        # claude -p reports num_turns.
        ({"num_turns": 37, "iterations": 99}, 37),
        # The in-process drivers (api, openai-api) report the same count as
        # iterations; the bench showed "-" for every openai-api compile.
        ({"iterations": 23, "repair_rounds": 2}, 23),
    ],
)
def test_turns_come_from_either_driver_shape(
    tmp_path: Path, metadata: dict[str, int], turns: int
) -> None:
    from rote.bench.collect import collect_job

    compiled = tmp_path / "compiled"
    compiled.mkdir()
    (compiled / "pipeline.yaml").write_text("name: [unclosed\n", encoding="utf-8")
    (compiled / "compile-metrics.json").write_text(
        json.dumps({"schema_version": 1, "driver": "x", "metadata": metadata}), encoding="utf-8"
    )
    job = collect_job(
        tmp_path, tmp_path, BDR_PIPELINE_YAML, exit_code=1, wall_seconds=1.0, tsc=False
    )
    assert job["run"]["turns"] == turns
    assert job["run"]["repair_rounds"] == metadata.get("repair_rounds")
