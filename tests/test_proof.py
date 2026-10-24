"""``rote.proof``: the field compare between a skill and its compiled form,
and the pinned compile argv a measurement uses. No inference."""

from __future__ import annotations

from pathlib import Path

from rote import demo, proof
from rote.ir import load_pipeline

REPO = Path(__file__).resolve().parent.parent
DOCS_DIGEST = REPO / "examples" / "docs-digest" / "compiled" / "pipeline.yaml"
DEAL_MONITOR = REPO / "examples" / "deal-monitor" / "expected" / "pipeline.yaml"


def _spec(name: str, expect: proof.Expect) -> proof.FieldSpec:
    return proof.FieldSpec(name, f"/{name}", f"/{name}", None, None, expect, "why")


def test_each_expectation_reads_its_own_status_and_keeps_both_values() -> None:
    fields = (
        _spec("total", "equal"),
        _spec("headline", "drifts"),
        _spec("summary", "wording"),
        _spec("same", "equal"),
    )
    original = {"total": 5, "headline": "a", "summary": "x", "same": " Mixed Case "}
    compiled = {"total": 4, "headline": "b", "summary": "y", "same": "mixed case"}
    results = {r.spec.name: r for r in proof.compare_outputs(fields, original, compiled)}
    assert {n: r.status for n, r in results.items()} == {
        "total": "differs",
        "headline": "drifted",
        "summary": "differs-judge",
        "same": "matched",
    }
    assert (results["total"].original_value, results["total"].compiled_value) == (5, 4)


def test_a_missing_value_differs_even_on_a_field_that_may_drift() -> None:
    (result,) = proof.compare_outputs([_spec("headline", "drifts")], {"headline": "a"}, {})
    assert result.status == "differs"
    assert result.problem == "compiled output: no value at '/headline' (missing 'headline')"


def test_no_same_result_claim_from_zero_exact_fields() -> None:
    fields = (_spec("headline", "drifts"), _spec("summary", "wording"))
    cmp = proof.Comparison(
        proof.compare_outputs(
            fields, {"headline": "a", "summary": "s"}, {"headline": "a", "summary": "s"}
        )
    )
    # Every value matched, and still nothing is claimed.
    assert all(f.status == "matched" for f in cmp.fields)
    assert cmp.compared == 0
    assert cmp.same_result is False and cmp.ok is False
    assert cmp.headline == "no fields to compare exactly"
    assert proof.verdict(cmp).startswith("No field is expected to match exactly")


def test_same_result_needs_every_exact_field() -> None:
    fields = (_spec("a", "equal"), _spec("b", "equal"), _spec("c", "drifts"))
    one_off = proof.Comparison(
        proof.compare_outputs(fields, {"a": 1, "b": 2, "c": 3}, {"a": 1, "b": 9, "c": 4})
    )
    assert (one_off.reproduced, one_off.compared, one_off.same_result) == (1, 2, False)
    assert one_off.headline == "reproduced 1 of 2 fields"
    all_in = proof.Comparison(
        proof.compare_outputs(fields, {"a": 1, "b": 2, "c": 3}, {"a": 1, "b": 2, "c": 4})
    )
    assert (all_in.reproduced, all_in.same_result) == (2, True)


def test_ir_fields_are_wording_only_below_a_model_call() -> None:
    plain = proof.fields_from_ir(load_pipeline(DOCS_DIGEST))
    assert [(f.name, f.original, f.compiled, f.expect) for f in plain] == [
        ("digest", "/digest", "/write_digest/digest", "equal")
    ]
    judged = load_pipeline(DEAL_MONITOR)
    assert proof.model_nodes(judged), "deal-monitor's example calls a model"
    assert {f.expect for f in proof.fields_from_ir(judged)} == {"wording"}
    contract = proof.contract_fields(judged, ["a"])
    assert contract[0].expect == "wording" and "calls a model" in contract[0].why
    assert proof.contract_fields(load_pipeline(DOCS_DIGEST), ["a"])[0].expect == "equal"


def test_reference_fields_read_both_sides_at_the_skill_pointer_when_asked() -> None:
    ref = demo.load_reference()
    as_pipeline = [proof.FieldSpec.from_reference(f) for f in ref.fields]
    same_shape = [proof.FieldSpec.from_reference(f, same_shape=True) for f in ref.fields]
    assert {f.compiled for f in as_pipeline} == {"/write_digest/digest"}
    assert {f.compiled for f in same_shape} == {"/digest"}


def test_the_compile_argv_pins_what_the_environment_would_otherwise_pick(tmp_path: Path) -> None:
    argv = proof.compile_command(
        tmp_path / "skill", tmp_path / "in.json", tmp_path / "out", runtime="skill"
    )
    assert argv[:2] == ["compile", str(tmp_path / "skill")]

    def value(flag: str) -> str:
        return argv[argv.index(flag) + 1]

    assert value("--runtime") == "skill"
    assert value("--agent") == "claude"
    assert value("--decider") == "rules"
    assert value("--structure") == "haiku"
    assert value("--tiebreak") == "none"
    assert value("--sample-input") == str(tmp_path / "in.json")
    assert "--local" in argv and "--no-eval" in argv
    # The negative half: auto (credential-dependent) and --cloud never appear.
    assert "auto" not in argv and "--cloud" not in argv
    assert "--model" not in argv
    with_model = proof.compile_command(
        tmp_path / "s", tmp_path / "i", tmp_path / "o", runtime="dbos", model="m-1"
    )
    assert with_model[with_model.index("--model") + 1] == "m-1"
