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
from rote.bench.runner import STATE_ENV, isolated_env
from rote.bench.score import dependency_edges, match_nodes, score_pipeline
from rote.ir import Pipeline

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


def test_matching_pairs_by_id_then_description() -> None:
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
    """Same six overrides as conftest's _isolated_mcp_state, all under the
    run's state dir, and the per-run ROTE_* defaults dropped."""
    monkeypatch.setenv("ROTE_RUNTIME", "temporal")
    monkeypatch.setenv("ROTE_MODEL", "claude-opus-4-6")
    env = isolated_env(tmp_path / "state")
    assert set(STATE_ENV) == {
        "ROTE_APPS_PATH",
        "ROTE_CONFIG_PATH",
        "ROTE_PROJECT_CONFIG_PATH",
        "ROTE_CLOUD_CRED_PATH",
        "ROTE_MCP_CONFIG",
        "ROTE_MCP_TOKEN_DIR",
    }
    for key in STATE_ENV:
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
