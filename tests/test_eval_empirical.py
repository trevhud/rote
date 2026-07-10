"""Empirical eval mode tests.

Fast and hermetic: the pipeline trial runs a real emitted
python-adapter app (subprocess, no LLM — impls are overlaid with
working functions, same technique as the runtime e2e suites), and the
skill trial runs against a fake ``claude`` executable that mimics the
verified ``--output-format json`` shape.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from rote.adapters import get_adapter
from rote.eval.empirical import (
    EmpiricalError,
    EmpiricalResult,
    MeasuredRun,
    append_corpus,
    compute_agreement,
    mcp_servers_for_pipeline,
    measured_pipeline_cost_usd,
    min_expected_turns_for,
    render_measured_markdown,
    run_pipeline_trial,
    run_skill_trial,
    suggested_priors,
)
from rote.eval.pricing import ModelPrice, ModelTier, PricingCatalog
from rote.ir import Pipeline

# ───────── Fixtures ─────────

TOY_PIPELINE = Pipeline.model_validate(
    {
        "name": "toy-sum",
        "description": "Sum numbers and label the result",
        "input": {
            "type": "Numbers",
            "required": ["numbers"],
            "input_schema": {
                "type": "object",
                "properties": {"numbers": {"type": "array", "items": {"type": "number"}}},
                "required": ["numbers"],
            },
        },
        "nodes": [
            {
                "id": "total",
                "kind": "pure_function",
                "description": "Sum the numbers",
                "impl": "extracted/mathx.py:total",
                "inputs": {"numbers": "pipeline.input.numbers"},
                "output": {"total": "int"},
            },
            {
                "id": "label",
                "kind": "pure_function",
                "description": "Render a label",
                "impl": "extracted/mathx.py:label",
                "inputs": {"total": "total.output.total"},
                "output": {"label": "str"},
            },
        ],
        "edges": [{"from": "total", "to": "label"}],
        "entry_nodes": ["total"],
        "exit_nodes": ["label"],
    }
)

WORKING_IMPLS = """\
def total(**payload):
    return {"total": sum(payload["numbers"])}


def label(**payload):
    return {"label": f"sum={payload['total']}"}
"""


@pytest.fixture()
def toy_app(tmp_path: Path) -> Path:
    """A real, runnable emitted python-adapter app (no LLM anywhere)."""
    app_dir = tmp_path / "app"
    get_adapter("python").emit(TOY_PIPELINE, app_dir)
    (app_dir / "extracted" / "mathx.py").write_text(WORKING_IMPLS, encoding="utf-8")
    return app_dir


CATALOG = PricingCatalog(
    prices=(
        ModelPrice(
            model_id="claude-mid-9",
            provider="anthropic",
            display_name="Mid",
            tier=ModelTier.MID,
            input_per_mtok=2.0,
            output_per_mtok=10.0,
            cache_read_per_mtok=0.2,
            cache_write_per_mtok=2.5,
            source="https://example.test",
            fetched_at="t",
        ),
    ),
    reference_prices={"claude-mid-9": (2.0, 10.0)},
)


# ───────── Pipeline trials (real subprocess) ─────────


def test_pipeline_trial_runs_emitted_app(toy_app: Path) -> None:
    run = run_pipeline_trial(toy_app, {"numbers": [1, 2, 3]})
    assert run.error is None
    assert run.succeeded
    assert run.wall_seconds > 0
    assert "sum=6" in json.dumps(run.output)


def test_pipeline_trials_agree_perfectly(toy_app: Path) -> None:
    runs = [run_pipeline_trial(toy_app, {"numbers": [4, 5]}) for _ in range(2)]
    agreement = compute_agreement([r.output for r in runs])
    assert agreement.successful == 2
    assert agreement.identical_fraction == 1.0


def test_pipeline_trial_reports_app_failure(toy_app: Path) -> None:
    # Missing required input → the app raises → nonzero exit, captured.
    run = run_pipeline_trial(toy_app, {})
    assert not run.succeeded
    assert run.error is not None and "exited" in run.error


def test_pipeline_trial_rejects_signals_for_plain_python(toy_app: Path) -> None:
    with pytest.raises(EmpiricalError, match="not a DBOS app"):
        run_pipeline_trial(toy_app, {"numbers": [1]}, signals={"gate": {}})


def test_pipeline_trial_requires_main_py(tmp_path: Path) -> None:
    with pytest.raises(EmpiricalError, match="no main.py"):
        run_pipeline_trial(tmp_path, {})


# ───────── Skill trials (fake claude executable) ─────────

FAKE_CLAUDE = """\
#!/bin/sh
# Mimics `claude -p ... --output-format json`: writes the agent's
# result.json into the cwd, prints the verified result envelope.
printf '%s' '{"answer": 42, "detail": "computed"}' > result.json
cat <<'EOF'
{"type": "result", "num_turns": 7, "total_cost_usd": 0.31, "duration_ms": 4200,
 "usage": {"input_tokens": 1000, "cache_read_input_tokens": 900,
           "cache_creation_input_tokens": 300, "output_tokens": 210}}
EOF
"""


@pytest.fixture()
def fake_skill(tmp_path: Path) -> tuple[Path, str]:
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("# Toy skill\n\nAnswer the question.\n", encoding="utf-8")
    fake = tmp_path / "claude"
    fake.write_text(FAKE_CLAUDE, encoding="utf-8")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    return skill_dir, str(fake)


def test_skill_trial_parses_usage_and_result(fake_skill: tuple[Path, str]) -> None:
    skill_dir, executable = fake_skill
    run = run_skill_trial(
        skill_dir, {"question": "6*7?"}, model="claude-mid-9", executable=executable
    )
    assert run.succeeded
    assert run.output == {"answer": 42, "detail": "computed"}
    assert run.turns == 7
    assert run.cost_usd == 0.31
    assert run.cache_read_tokens == 900
    assert run.output_tokens == 210
    assert run.model == "claude-mid-9"


def test_skill_trial_missing_skill_md(tmp_path: Path) -> None:
    with pytest.raises(EmpiricalError, match="SKILL.md"):
        run_skill_trial(tmp_path, {}, model="m", executable="claude")


def test_skill_trial_missing_result_is_an_error(tmp_path: Path) -> None:
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("# s\n", encoding="utf-8")
    fake = tmp_path / "claude"
    fake.write_text("#!/bin/sh\necho '{\"num_turns\": 1}'\n", encoding="utf-8")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    run = run_skill_trial(skill_dir, {}, model="m", executable=str(fake))
    assert not run.succeeded
    assert run.error is not None and "result.json" in run.error


# ───────── MCP auto-wiring ─────────


def _mcp_pipeline(**binding_overrides: object) -> Pipeline:
    """A minimal pipeline with MCP-bound external_calls: one explicit-url
    server, one env-resolved server, plus a pure function (never wired)."""
    return Pipeline.model_validate(
        {
            "name": "mcp-toy",
            "input": {"type": "In"},
            "nodes": [
                {
                    "id": "pull_slack",
                    "kind": "external_call",
                    "description": "read the intake channel",
                    "mcp": {
                        "server": "slack",
                        "tool": "slack_get_messages",
                        "url": "https://mcp.example.com/slack",
                        **binding_overrides,
                    },
                },
                {
                    "id": "pull_gmail",
                    "kind": "external_call",
                    "description": "read labeled threads",
                    "mcp": {"server": "gmail", "tool": "gmail_search_threads"},
                },
                {
                    "id": "summarize",
                    "kind": "pure_function",
                    "description": "fold results",
                    "impl": "extracted/fold.py:fold",
                },
            ],
            "edges": [
                {"from": "pull_slack", "to": "summarize"},
                {"from": "pull_gmail", "to": "summarize"},
            ],
        }
    )


def test_mcp_servers_resolve_explicit_url_and_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ROTE_MCP_GMAIL_URL", "https://mcp.example.com/gmail")
    servers, missing = mcp_servers_for_pipeline(_mcp_pipeline())
    assert missing == []
    assert servers == {
        "slack": {"type": "http", "url": "https://mcp.example.com/slack"},
        "gmail": {"type": "http", "url": "https://mcp.example.com/gmail"},
    }


def test_mcp_servers_report_unresolvable_as_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ROTE_MCP_GMAIL_URL", raising=False)
    servers, missing = mcp_servers_for_pipeline(_mcp_pipeline())
    assert list(servers) == ["slack"]
    assert missing == ["gmail"]


def test_mcp_sse_transport_maps_to_sse_type(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROTE_MCP_GMAIL_URL", "https://mcp.example.com/gmail")
    servers, _ = mcp_servers_for_pipeline(_mcp_pipeline(transport="sse"))
    assert servers["slack"]["type"] == "sse"


def test_min_expected_turns_counts_data_pulls() -> None:
    # 2 external_calls → floor stays at the minimum of 3; the toy
    # summation pipeline has none → still 3.
    assert min_expected_turns_for(_mcp_pipeline()) == 3
    assert min_expected_turns_for(TOY_PIPELINE) == 3


FAKE_CLAUDE_ARGS_DUMP = """\
#!/bin/sh
# Records its argv and the mcp-config content (the temp workdir is
# deleted after the trial, so the config must be copied out now).
printf '%s\\n' "$@" > "$ROTE_TEST_DUMP/args.txt"
prev=""
for a in "$@"; do
  if [ "$prev" = "--mcp-config" ]; then cp "$a" "$ROTE_TEST_DUMP/mcp-config.json"; fi
  prev="$a"
done
printf '%s' '{"ok": true}' > result.json
echo '{"type": "result", "subtype": "success", "num_turns": 9, "usage": {}}'
"""


def test_skill_trial_wires_mcp_config_into_claude(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("# s\n", encoding="utf-8")
    dump = tmp_path / "dump"
    dump.mkdir()
    monkeypatch.setenv("ROTE_TEST_DUMP", str(dump))
    fake = tmp_path / "claude"
    fake.write_text(FAKE_CLAUDE_ARGS_DUMP, encoding="utf-8")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)

    servers = {"slack": {"type": "http", "url": "https://mcp.example.com/slack"}}
    run = run_skill_trial(skill_dir, {}, model="m", executable=str(fake), mcp_servers=servers)

    assert run.succeeded and run.representative
    args = (dump / "args.txt").read_text(encoding="utf-8").splitlines()
    assert "--strict-mcp-config" in args
    assert "--mcp-config" in args
    written = json.loads((dump / "mcp-config.json").read_text(encoding="utf-8"))
    assert written == {"mcpServers": servers}
    allowed = args[args.index("--allowedTools") + 1]
    assert "mcp__slack__*" in allowed.split(",")


def test_skill_trial_without_mcp_passes_no_mcp_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("# s\n", encoding="utf-8")
    dump = tmp_path / "dump"
    dump.mkdir()
    monkeypatch.setenv("ROTE_TEST_DUMP", str(dump))
    fake = tmp_path / "claude"
    fake.write_text(FAKE_CLAUDE_ARGS_DUMP, encoding="utf-8")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)

    run_skill_trial(skill_dir, {}, model="m", executable=str(fake))
    args = (dump / "args.txt").read_text(encoding="utf-8").splitlines()
    assert "--mcp-config" not in args
    assert "--strict-mcp-config" not in args


# ───────── Reliability flags ─────────


def _fake_claude_emitting(tmp_path: Path, envelope: str, *, write_result: bool = True) -> str:
    fake = tmp_path / "claude"
    result_line = "printf '%s' '{\"ok\": true}' > result.json\n" if write_result else ""
    fake.write_text(f"#!/bin/sh\n{result_line}cat <<'EOF'\n{envelope}\nEOF\n", encoding="utf-8")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    return str(fake)


@pytest.fixture()
def flag_skill(tmp_path: Path) -> Path:
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("# s\n", encoding="utf-8")
    return skill_dir


def test_hit_max_turns_is_flagged(flag_skill: Path, tmp_path: Path) -> None:
    exe = _fake_claude_emitting(
        tmp_path, '{"type": "result", "subtype": "error_max_turns", "num_turns": 60, "usage": {}}'
    )
    run = run_skill_trial(flag_skill, {}, model="m", executable=exe, min_expected_turns=5)
    assert run.succeeded  # it wrote a result...
    assert "hit_max_turns" in run.flags  # ...but the cost is a floor
    assert not run.representative


def test_too_few_turns_is_flagged(flag_skill: Path, tmp_path: Path) -> None:
    exe = _fake_claude_emitting(
        tmp_path, '{"type": "result", "subtype": "success", "num_turns": 2, "usage": {}}'
    )
    run = run_skill_trial(flag_skill, {}, model="m", executable=exe, min_expected_turns=6)
    assert run.flags == ("suspiciously_few_turns",)
    assert not run.representative


def test_enough_turns_is_unflagged(flag_skill: Path, tmp_path: Path) -> None:
    exe = _fake_claude_emitting(
        tmp_path, '{"type": "result", "subtype": "success", "num_turns": 6, "usage": {}}'
    )
    run = run_skill_trial(flag_skill, {}, model="m", executable=exe, min_expected_turns=6)
    assert run.flags == ()
    assert run.representative


def test_missing_mcp_servers_flagged(flag_skill: Path, tmp_path: Path) -> None:
    exe = _fake_claude_emitting(
        tmp_path, '{"type": "result", "subtype": "success", "num_turns": 9, "usage": {}}'
    )
    run = run_skill_trial(flag_skill, {}, model="m", executable=exe, missing_mcp_servers=["gmail"])
    assert run.flags == ("missing_mcp_servers",)


def test_errored_run_is_flagged(flag_skill: Path, tmp_path: Path) -> None:
    exe = _fake_claude_emitting(
        tmp_path,
        '{"type": "result", "subtype": "success", "num_turns": 9, "usage": {}}',
        write_result=False,
    )
    run = run_skill_trial(flag_skill, {}, model="m", executable=exe)
    assert not run.succeeded
    assert "errored" in run.flags


def test_suggested_priors_exclude_flagged_runs() -> None:
    good = MeasuredRun(wall_seconds=130.0, output={}, turns=10, output_tokens=2500)
    poisoned = MeasuredRun(
        wall_seconds=5.0, output={}, turns=1, output_tokens=50, flags=("suspiciously_few_turns",)
    )
    fitted = suggested_priors((good, poisoned))
    assert fitted["seconds_per_turn"] == 13.0  # the flagged run's 5.0 never averaged in
    assert fitted["output_tokens_per_turn"] == 250.0


def test_corpus_and_json_carry_flags(tmp_path: Path) -> None:
    result = EmpiricalResult(
        trials=1,
        skill_runs=(MeasuredRun(wall_seconds=5.0, output=None, error="boom", flags=("errored",)),),
        pipeline_runs=(),
        skill_model="m",
    )
    corpus = tmp_path / "corpus.jsonl"
    append_corpus(result, generated_at="2026-07-09T00:00:00Z", path=corpus)
    row = json.loads(corpus.read_text(encoding="utf-8"))
    assert row["skill_runs"][0]["flags"] == ["errored"]
    md = render_measured_markdown(result, CATALOG)
    assert "Unrepresentative agent runs" in md
    assert "errored" in md


# ───────── Agreement ─────────


def test_agreement_identical() -> None:
    a = compute_agreement([{"x": 1}, {"x": 1}, {"x": 1}])
    assert a.identical_fraction == 1.0
    assert a.field_agreement == {"x": 1.0}


def test_agreement_divergent_field() -> None:
    a = compute_agreement([{"x": 1, "y": "a"}, {"x": 1, "y": "b"}, {"x": 1, "y": "a"}])
    assert a.modal_count == 2
    assert a.field_agreement["x"] == 1.0
    assert a.field_agreement["y"] == pytest.approx(2 / 3)


def test_agreement_missing_field_counts_as_disagreement() -> None:
    a = compute_agreement([{"x": 1}, {}])
    assert a.field_agreement["x"] == 0.5


def test_agreement_failures_excluded_but_counted() -> None:
    a = compute_agreement([{"x": 1}, None, None])
    assert a.total == 3
    assert a.successful == 1
    assert a.identical_fraction == 1.0


def test_agreement_all_failed() -> None:
    a = compute_agreement([None, None])
    assert a.successful == 0
    assert a.identical_fraction == 0.0


# ───────── Pricing measured usage + prior refits ─────────


def _pipeline_run(usage: list[dict[str, object]]) -> MeasuredRun:
    return MeasuredRun(wall_seconds=1.0, output={}, judge_usage=tuple(usage))


def test_measured_cost_prices_judge_usage() -> None:
    runs = (_pipeline_run([{"model": "claude-mid-9", "input_tokens": 1000, "output_tokens": 100}]),)
    cost, unpriced = measured_pipeline_cost_usd(runs, CATALOG)
    assert cost == pytest.approx((1000 * 2.0 + 100 * 10.0) / 1_000_000)
    assert unpriced == []


def test_measured_cost_flags_unpriced_models() -> None:
    runs = (_pipeline_run([{"model": "mystery-lm", "input_tokens": 10, "output_tokens": 5}]),)
    cost, unpriced = measured_pipeline_cost_usd(runs, CATALOG)
    assert cost == 0.0
    assert unpriced == ["mystery-lm"]


def test_suggested_priors_from_measured_runs() -> None:
    runs = (
        MeasuredRun(wall_seconds=100.0, output={}, turns=10, output_tokens=2000),
        MeasuredRun(wall_seconds=140.0, output={}, turns=10, output_tokens=3000),
    )
    fitted = suggested_priors(runs)
    assert fitted["seconds_per_turn"] == pytest.approx(12.0)
    assert fitted["output_tokens_per_turn"] == pytest.approx(250.0)


def test_suggested_priors_fits_transcript_growth_from_cache_read() -> None:
    """A data-heavy run's huge cache-read inverts to an effective Δ far above
    the text-only default — the signal to raise the payload prior. Uses the
    real opportunity-monitor shape (~22 turns, ~9.5M cache-read)."""
    from rote.eval.priors import Priors

    c0 = Priors().system_overhead_tokens  # 16_000
    runs = (
        MeasuredRun(
            wall_seconds=300.0, output={}, turns=22, cache_read_tokens=9_530_000, output_tokens=6000
        ),
    )
    fitted = suggested_priors(runs)
    expected = 2.0 * (9_530_000 - 21 * c0) / (20 * 21)  # ≈ 43_781
    assert fitted["transcript_growth_per_turn"] == pytest.approx(round(expected, 1))
    assert fitted["transcript_growth_per_turn"] > 40 * Priors().transcript_growth_per_turn


def test_suggested_priors_skips_growth_fit_when_uncomputable() -> None:
    """No cache-read, or fewer than 3 turns, ⇒ no Δ fit (undefined)."""
    runs = (
        MeasuredRun(wall_seconds=10.0, output={}, turns=2, cache_read_tokens=5000),
        MeasuredRun(wall_seconds=10.0, output={}, turns=10, cache_read_tokens=None),
    )
    assert "transcript_growth_per_turn" not in suggested_priors(runs)


# ───────── Rendering + corpus ─────────


def _result() -> EmpiricalResult:
    return EmpiricalResult(
        trials=2,
        skill_runs=(
            MeasuredRun(
                wall_seconds=100.0,
                output={"answer": 42},
                turns=10,
                cost_usd=0.5,
                output_tokens=2000,
            ),
            MeasuredRun(
                wall_seconds=120.0,
                output={"answer": 43},
                turns=12,
                cost_usd=0.6,
                output_tokens=2400,
            ),
        ),
        pipeline_runs=(
            _pipeline_run([{"model": "claude-mid-9", "input_tokens": 500, "output_tokens": 50}]),
            _pipeline_run([{"model": "claude-mid-9", "input_tokens": 500, "output_tokens": 50}]),
        ),
        skill_model="claude-mid-9",
    )


def test_render_measured_markdown_sections() -> None:
    md = render_measured_markdown(_result(), CATALOG)
    assert "## Measured (2 trials per side)" in md
    assert "list-price notional" in md
    assert "measured judge usage" in md
    assert "`answer`: 50% of runs agree" in md
    assert "seconds_per_turn" in md  # suggested re-fit


def test_append_corpus_writes_jsonl(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    path = append_corpus(_result(), generated_at="2026-07-03T00:00:00Z", path=corpus)
    append_corpus(_result(), generated_at="2026-07-03T01:00:00Z", path=corpus)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    record = json.loads(lines[0])
    assert record["trials"] == 2
    assert record["skill_runs"][0]["turns"] == 10


# ───────── Review regressions ─────────


def test_pipeline_trial_preserves_api_keys(toy_app: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Emitted judges construct anthropic.Anthropic() directly — the
    pipeline subprocess must keep ANTHROPIC_API_KEY (unlike the skill
    side's claude -p spawn, which deliberately scrubs it)."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-preserved")
    (toy_app / "extracted" / "mathx.py").write_text(
        WORKING_IMPLS.replace(
            'return {"label": f"sum={payload[\'total\']}"}',
            'import os\n    return {"label": os.environ.get("ANTHROPIC_API_KEY", "MISSING")}',
        ),
        encoding="utf-8",
    )
    run = run_pipeline_trial(toy_app, {"numbers": [1]})
    assert run.succeeded
    assert "sk-test-preserved" in json.dumps(run.output)


def test_pipeline_trial_kills_child_when_signaling_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A signal-delivery failure must reap the parked child, not orphan
    it until its IR gate timeout."""
    import subprocess as sp
    import time as time_mod

    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "dbos-config.yaml").write_text("name: fake\n", encoding="utf-8")
    # Fake DBOS app: reports a workflow id, then parks for minutes —
    # exactly the state a real gated app is in when signaling fails.
    (app_dir / "main.py").write_text(
        "import sys, time\n"
        'print("workflow started: wf-fake", file=sys.stderr, flush=True)\n'
        "time.sleep(300)\n",
        encoding="utf-8",
    )

    def _boom(url: str, workflow_id: str, signals: dict[str, object]) -> None:
        raise EmpiricalError("no dbos client available")

    monkeypatch.setattr("rote.eval.empirical._send_dbos_signals", _boom)

    with pytest.raises(EmpiricalError, match="no dbos client"):
        run_pipeline_trial(app_dir, {}, signals={"gate": {}}, timeout_seconds=30)

    time_mod.sleep(0.2)
    survivors = sp.run(["pgrep", "-f", str(app_dir / "main.py")], capture_output=True, text=True)
    assert survivors.stdout.strip() == "", f"orphaned pids: {survivors.stdout}"
