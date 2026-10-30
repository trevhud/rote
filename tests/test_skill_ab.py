"""``rote.eval.skill_ab``: the paired eval's harness, with no inference.

The parsing tests read a recorded ``claude -p`` transcript (a real
docs-digest compiled-skill run, trimmed: usage kept exactly). The harness
tests run the whole thing with ``tests/fixtures/skill_ab/fake_claude.py`` as
the executable: it acts like an agent, including running ``rote skill run``
for real against the emitted expense-audit skill, and reports fixed usage,
so every number below is known in advance.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from rote.eval import skill_ab
from rote.eval.empirical import skill_run_isolation_args

REPO = Path(__file__).resolve().parent.parent
FIXTURES = REPO / "tests" / "fixtures" / "skill_ab"
FAKE_CLAUDE = FIXTURES / "fake_claude.py"
EXPENSE = REPO / "examples" / "expense-audit"


# ───────── the stream ─────────


def test_a_recorded_run_counts_every_bucket_once_per_message() -> None:
    lines = (FIXTURES / "docs-digest-compiled.stream.jsonl").read_text().splitlines()
    usage, envelope = skill_ab.parse_stream(lines)
    # Four requests, though the stream repeats them over eight assistant events.
    assert usage.requests == 4
    assert usage.per_request_input == (10958, 11359, 12094, 12914)
    # Totals from modelUsage, all three input buckets, matching the per-request sum.
    assert (usage.input, usage.cache_write, usage.cache_read) == (6, 3537, 43782)
    assert usage.total_input == 47325 == sum(usage.per_request_input)
    # Output is the final count, never the stream's per-block placeholders (8+3+1+3).
    assert usage.output == 1512
    assert usage.cost_usd == pytest.approx(0.0570546)
    assert (usage.num_turns, usage.source, usage.warm_start) == (4, "modelUsage", True)
    assert usage.tools_used == ("Bash", "Read", "Write")
    assert envelope["is_error"] is False


def _assistant(message_id: str, **usage: int) -> str:
    return json.dumps(
        {"type": "assistant", "message": {"id": message_id, "usage": usage, "content": []}}
    )


def test_without_model_usage_the_envelope_usage_is_the_total() -> None:
    lines = [
        _assistant("m1", input_tokens=2, cache_creation_input_tokens=0, cache_read_input_tokens=0),
        json.dumps(
            {
                "type": "result",
                "total_cost_usd": 0.5,
                "usage": {
                    "input_tokens": 3,
                    "cache_creation_input_tokens": 40,
                    "cache_read_input_tokens": 500,
                    "output_tokens": 6,
                },
            }
        ),
    ]
    usage, _ = skill_ab.parse_stream(lines)
    assert (usage.total_input, usage.output, usage.source) == (543, 6, "usage")
    assert usage.warm_start is False


def test_a_run_with_no_result_has_no_output_count() -> None:
    usage, envelope = skill_ab.parse_stream(
        [_assistant("m1", input_tokens=1, cache_read_input_tokens=9, output_tokens=4)]
    )
    assert envelope == {}
    assert (usage.output, usage.source, usage.cost_usd, usage.total_input) == (
        None,
        "stream",
        None,
        10,
    )


# ───────── the ledger ─────────


def test_the_ledger_stops_before_a_run_that_would_cross_the_cap(tmp_path: Path) -> None:
    ledger = skill_ab.Ledger(tmp_path / "ledger.json", 1.0)
    ledger.check("x:original:trial", 0.4)
    ledger.record("x:original:trial", "x #1", 0.55, 0.4)
    ledger.record("x:original:trial", "x #2", 0.30, 0.4)
    # The next estimate is the dearest of its kind (0.55): 0.85 + 0.55 > 1.0.
    with pytest.raises(skill_ab.BudgetExhausted, match=r"spent \$0\.850 of \$1\.00"):
        ledger.check("x:original:trial", 0.4)
    # Another kind falls back to its default: 0.85 + 0.10 fits.
    ledger.check("x:compiled:control", 0.10)
    # An unknown cost is recorded at the estimate, never as free, and persists.
    ledger.record("x:compiled:control", "x control", None, 0.10)
    again = skill_ab.Ledger(tmp_path / "ledger.json", 1.0)
    assert again.spent == pytest.approx(0.95)
    assert again.runs[-1]["estimated"] is True


# ───────── driver evidence ─────────

PLAN = {"steps": [{"id": "read", "kind": "mcp"}, {"id": "approve", "kind": "gate"}]}


def _run_folder(
    root: Path, *, state: str = "done", gate: dict[str, Any] | None = None, run_input: Any = None
) -> Path:
    folder = root / "toy" / "abc123"
    (folder / "nodes").mkdir(parents=True)
    (folder / "run.json").write_text(
        json.dumps({"state": state, "input": {"q": 1} if run_input is None else run_input})
    )
    (folder / "result.json").write_text(json.dumps({"report": {"total": 7, "status": "final"}}))
    if gate is not None:
        (folder / "nodes" / "approve.json").write_text(json.dumps(gate))
    return folder


PERSON = {"approved": True}
HUMAN_GATE = {"result": {"approved": True}, "answered_by": "human", "said": "yes, ship it"}


def _evidence(root: Path, **overrides: Any) -> skill_ab.DriverEvidence:
    kwargs: dict[str, Any] = {
        "skill": "toy",
        "plan": PLAN,
        "expected_input": {"q": 1},
        "agent_output": {"total": 7},
        "asks": 1,
        "gate_answers": [PERSON],
    }
    kwargs.update(overrides)
    return skill_ab.driver_evidence(root, **kwargs)


def test_an_honest_run_has_no_problems(tmp_path: Path) -> None:
    _run_folder(tmp_path, gate=HUMAN_GATE)
    found = _evidence(tmp_path)
    assert found.problems == ()
    assert found.gates == (
        {"step": "approve", "answered_by": "human", "said": "yes, ship it", "matches_person": True},
    )
    assert (found.state, found.reported_checked, found.result["report"]["total"]) == ("done", 1, 7)


def test_the_agents_steps_are_counted_by_block_and_held_to_the_agent_tier(
    tmp_path: Path,
) -> None:
    plan = {
        "steps": [
            *PLAN["steps"],
            {"id": "post", "kind": "do"},
            {"id": "push", "kind": "do"},
            {"id": "grade", "kind": "judge"},
        ]
    }
    folder = _run_folder(tmp_path, gate=HUMAN_GATE)
    for step in ("post", "push", "grade"):
        record = {"result": {}, "answered_by": "agent"}
        (folder / "nodes" / f"{step}.json").write_text(json.dumps(record))
    found = _evidence(tmp_path, plan=plan)
    assert found.problems == ()
    assert found.agent_steps == {"DO": 2, "DECIDE": 1}
    assert found.to_dict()["agent_steps"] == {"DO": 2, "DECIDE": 1}
    # A DO result nobody answered as the agent is a tier violation.
    (folder / "nodes" / "push.json").write_text(json.dumps({"result": {}}))
    found = _evidence(tmp_path, plan=plan)
    assert found.problems == ("DO step push was not answered by the agent (None)",)
    assert found.agent_steps == {"DO": 1, "DECIDE": 1}
    # Under --judges helper the helper answers judges, so they are not the agent's.
    found = _evidence(tmp_path, plan={**plan, "judges": "helper"})
    assert found.agent_steps == {"DO": 1}


def test_no_run_folder_is_a_bypass(tmp_path: Path) -> None:
    assert _evidence(tmp_path).problems == ("bypassed the driver: no run folder",)


def test_a_gate_the_agent_answered_fails(tmp_path: Path) -> None:
    # It even carries words: answered_by alone decides who answered.
    _run_folder(
        tmp_path, gate={"result": {"approved": True}, "answered_by": "agent", "said": "yes"}
    )
    assert _evidence(tmp_path).problems == (
        "gate approve was not answered by a person (answered_by='agent')",
    )


def test_a_gate_answered_without_asking_fails(tmp_path: Path) -> None:
    _run_folder(tmp_path, gate=HUMAN_GATE)
    (problem,) = _evidence(tmp_path, asks=0).problems
    assert "answered a gate without asking" in problem


def test_a_gate_answer_that_is_not_what_the_person_said_fails(tmp_path: Path) -> None:
    _run_folder(tmp_path, gate={**HUMAN_GATE, "result": {"approved": False}})
    assert _evidence(tmp_path).problems == (
        "gate approve's recorded answer is not what the person said",
    )


def test_a_misreported_value_and_a_changed_input_are_named(tmp_path: Path) -> None:
    _run_folder(tmp_path, gate=HUMAN_GATE, run_input={"q": 2}, state="waiting")
    problems = _evidence(tmp_path, agent_output={"total": 8, "other": 1}).problems
    assert problems == (
        "the run ended waiting, not done",
        "the agent started the run on a different input than it was given",
        "misreported total: the agent wrote 8, the driver's result has 7",
    )


# ───────── the harness, end to end, against the fake agent ─────────


def test_both_sides_load_bash_and_only_the_compiled_side_may_run_it(tmp_path: Path) -> None:
    args = skill_run_isolation_args(
        tmp_path, tmp_path, {}, ["Bash(rote skill run toy *)"], extra_tools=("Bash",)
    )
    assert args[args.index("--tools") + 1] == "Read,Write,Edit,Glob,Grep,Bash"
    default = skill_run_isolation_args(tmp_path, tmp_path, {}, [])
    assert default[default.index("--tools") + 1] == "Read,Write,Edit,Glob,Grep"


@pytest.fixture
def expense(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> skill_ab.Subject:
    from rote.bench.fidelity import load_cases, load_reference

    monkeypatch.setenv("EXPENSE_AUDIT_REPORT_DIR", str(tmp_path / "reports"))
    casefile = load_cases(EXPENSE)
    recorded = load_reference(casefile)["cases"]["base"]["runs"][0]["output"]
    original = tmp_path / "original-output.json"
    original.write_text(json.dumps(recorded))
    monkeypatch.setenv("FAKE_ORIGINAL_OUTPUT", str(original))
    # The fake copies the driver's result; these are the contract fields the
    # driver's result does not carry, taken from the same recorded output.
    driver_keys = {"reimbursable_usd", "pending_usd", "returned_usd", "batch_total_usd"}
    monkeypatch.setenv(
        "FAKE_EXTRA", json.dumps({k: v for k, v in recorded.items() if k not in driver_keys})
    )
    return skill_ab.subject_for(
        EXPENSE / "expected" / "pipeline.yaml", skill_dir=EXPENSE / "skill", case_id="base"
    )


def _harness(tmp_path: Path, cap: float = 1.0) -> skill_ab.Harness:
    return skill_ab.Harness(
        work=tmp_path / "work",
        model="fake-model",
        ledger=skill_ab.Ledger(tmp_path / "ledger.json", cap),
        executable=str(FAKE_CLAUDE),
        log=lambda line: print(line, file=sys.stderr),
    )


def test_a_paired_run_measures_both_sides_and_checks_the_compiled_one(
    tmp_path: Path, expense: skill_ab.Subject
) -> None:
    (tmp_path / "work").mkdir()
    harness = _harness(tmp_path)
    report = skill_ab.run_subject(harness, expense, trials=1)
    assert report.stopped is None
    assert (report.subject.shape, report.subject.name) == ("compute-heavy", "expense-audit")
    original, compiled = report.sides["original"], report.sides["compiled"]
    (o,), (c,) = original.trials, compiled.trials
    assert o.ok and c.ok, (o.problems, c.problems)
    # The fake reports request i as 1110*i input tokens; a control is one request.
    assert original.base_per_request == compiled.base_per_request == 1110
    # Original: ask, write: 3 requests (1110+2220+3330); K = 6660 - 3*1110.
    assert (o.usage.requests, o.usage.total_input, original.context(o)) == (3, 6660, 3330)
    # Compiled: start, ask, answer, write: 4 requests; K = 11100 - 4*1110.
    assert (c.usage.requests, c.usage.total_input, compiled.context(c)) == (4, 11100, 6660)
    # The person was asked once on each side, and the journal says so.
    assert o.calls == ("rote_gates.ask_person",) and c.calls == ("rote_gates.ask_person",)
    assert c.driver is not None and c.driver.state == "done"
    assert [g["answered_by"] for g in c.driver.gates] == ["human"]
    assert c.driver.reported_checked > 0
    # Two controls and two trials, at the fake's $0.01 a request.
    assert harness.ledger.spent == pytest.approx(0.01 + 0.01 + 0.03 + 0.04)
    # Bash is loaded on both sides; only the compiled side may run it.
    for trial in (o, c):
        argv = json.loads((trial.work_dir / "argv.json").read_text())
        tools = argv[argv.index("--tools") + 1].split(",")
        assert tools == ["Read", "Write", "Edit", "Glob", "Grep", "Bash"], trial.side
    original_rules = json.loads((o.work_dir / "argv.json").read_text())
    allowed = original_rules[original_rules.index("--allowedTools") + 1]
    assert "Bash(" not in allowed
    compiled_rules = json.loads((c.work_dir / "argv.json").read_text())
    assert (
        "Bash(rote skill run expense-audit *)"
        in compiled_rules[compiled_rules.index("--allowedTools") + 1]
    )
    (pair,) = report.pairs
    assert pair.compared == len(expense.contract) == 13
    # The fake's compiled result takes the four totals from the driver, and
    # every one matches the recorded original: the IR grades cap breaches
    # and a missing pre-approval as blocks, as the original does.
    assert pair.reproduced == 13
    assert [f.spec.name for f in pair.fields if f.status != "matched"] == []
    rendered = skill_ab.render_markdown([report], model="fake-model", spent=0.09)
    assert "### compute-heavy" in rendered and "| expense-audit/base | original | 1/1 |" in rendered


def test_a_compiled_agent_that_skips_the_driver_fails_as_a_bypass(
    tmp_path: Path, expense: skill_ab.Subject, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_MODE", "bypass")
    (tmp_path / "work").mkdir()
    harness = _harness(tmp_path)
    compiled_dir = skill_ab.emit_compiled(expense, tmp_path / "work" / "compiled")
    trial = skill_ab.run_trial(harness, expense, "compiled", 1, compiled_dir=compiled_dir)
    assert trial.problems == ("bypassed the driver: no run folder",)
    assert not trial.ok


def test_a_compiled_agent_that_answers_the_gate_itself_fails(
    tmp_path: Path, expense: skill_ab.Subject, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_MODE", "self_answer")
    monkeypatch.setenv("FAKE_SELF_ANSWER", json.dumps(expense.gate_answers[0]))
    (tmp_path / "work").mkdir()
    harness = _harness(tmp_path)
    compiled_dir = skill_ab.emit_compiled(expense, tmp_path / "work" / "compiled")
    trial = skill_ab.run_trial(harness, expense, "compiled", 1, compiled_dir=compiled_dir)
    assert trial.calls == ()
    (problem,) = trial.problems
    assert "1 gate(s) answered after 0 ask_person call(s)" in problem


def test_the_budget_stops_the_eval_and_keeps_what_ran(
    tmp_path: Path, expense: skill_ab.Subject
) -> None:
    (tmp_path / "work").mkdir()
    # Room for the two controls ($0.10 estimated each), not for a trial ($0.40).
    harness = _harness(tmp_path, cap=0.35)
    report = skill_ab.run_subject(harness, expense, trials=3)
    assert report.stopped is not None and "would cross the cap" in report.stopped
    assert report.sides["original"].trials == () and report.sides["compiled"].trials == ()
    assert report.sides["original"].base_per_request == 1110
    assert harness.ledger.spent == pytest.approx(0.02)


def test_an_unknown_case_is_refused(tmp_path: Path) -> None:
    with pytest.raises(skill_ab.SkillAbError, match="no such case"):
        skill_ab.subject_for(
            EXPENSE / "expected" / "pipeline.yaml", skill_dir=EXPENSE / "skill", case_id="nope"
        )


def test_shape_comes_from_the_ir() -> None:
    from rote.ir import load_pipeline

    docs = load_pipeline(REPO / "examples" / "docs-digest" / "compiled" / "pipeline.yaml")
    assert skill_ab.shape_of(docs)[0] == "tool-and-copy"
    expense = load_pipeline(EXPENSE / "expected" / "pipeline.yaml")
    assert skill_ab.shape_of(expense) == (
        "compute-heavy",
        "no MCP reads: every value is computed from the input",
    )


# ───────── the CLI ─────────


def test_the_paired_flags_are_refused_without_run_runtime_skill(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from rote.cli import main

    target = str(EXPENSE / "expected" / "pipeline.yaml")
    assert main(["eval", target, "--case", "base", "--budget", "1"]) == 2
    assert "--case, --budget only apply to --run --runtime skill" in capsys.readouterr().err


def test_the_paired_eval_needs_a_budget_and_one_input(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from rote.cli import main

    target = str(EXPENSE / "expected" / "pipeline.yaml")
    assert main(["eval", target, "--run", "--runtime", "skill", "--case", "base"]) == 2
    assert "needs --budget USD" in capsys.readouterr().err
    both = ["--case", "base", "--input", "x.json", "--budget", "1"]
    assert main(["eval", target, "--run", "--runtime", "skill", *both]) == 2
    assert "exactly one of --input" in capsys.readouterr().err


@pytest.mark.parametrize(
    "extra",
    [["--inference", "api"], ["--python", "python3"], ["--runtime-dir", "x"], ["--prior", "a=1"]],
)
def test_pipeline_trial_flags_are_refused_in_the_paired_eval(
    extra: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    from rote.cli import main

    target = str(EXPENSE / "expected" / "pipeline.yaml")
    argv = ["eval", target, "--run", "--runtime", "skill", "--case", "base", "--budget", "1"]
    assert main([*argv, *extra]) == 2
    assert f"{extra[0]} do not apply to --run --runtime skill" in capsys.readouterr().err


def test_a_given_pipeline_records_its_compile_settings_or_says_it_had_none() -> None:
    docs = skill_ab.given_provenance(
        REPO / "examples" / "docs-digest" / "compiled" / "pipeline.yaml"
    )
    assert (docs["pipeline"], docs["decider"], docs["structure"], docs["tiebreak"]) == (
        "given",
        "rules",
        "claude-haiku-5-5",
        "none",
    )
    assert docs["compile_cost_usd"] == pytest.approx(0.072854)
    expected = skill_ab.given_provenance(EXPENSE / "expected" / "pipeline.yaml")
    assert expected == {"pipeline": "given, with no compile record (hand-written)"}


def test_the_compile_estimate_covers_a_measured_compile() -> None:
    # CLAUDE.md: Sonnet 4.6 compiles ran up to $5.14; an estimate below that
    # lets a first compile overshoot --budget by a multiple.
    assert skill_ab.DEFAULT_COMPILE_ESTIMATE_USD >= 5.14


def test_an_interrupted_compile_is_recorded_at_its_estimate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupted(*args: Any, **kwargs: Any) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr(skill_ab.subprocess, "run", interrupted)
    ledger = skill_ab.Ledger(tmp_path / "ledger.json", 1.0)
    with pytest.raises(KeyboardInterrupt):
        skill_ab.compile_for_eval(
            EXPENSE / "skill",
            {},
            tmp_path,
            ledger,
            model=None,
            log=lambda _: None,
            estimate_usd=0.3,
        )
    (run,) = skill_ab.Ledger(tmp_path / "ledger.json", 1.0).runs
    assert (run["kind"], run["cost_usd"], run["estimated"]) == ("compile", 0.3, True)


# ───────── a server gated per tool, by its own readOnlyHint ─────────


def _registered(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, read_only: bool) -> None:
    from rote.mcp.registry import McpRegistry, McpServerConfig, save_registry
    from rote.mcp.tokens import write_token_file

    url = "https://mcp.atlassian.example/v1/mcp"
    save_registry(McpRegistry(servers={"atlassian": McpServerConfig(url=url, read_only=read_only)}))
    write_token_file("atlassian", {"server_url": url, "tokens": {"access_token": "SECRET-TOKEN"}})

    async def _listed(names: list[str]) -> tuple[list[str], list[str], dict[str, str]]:
        return (
            ["mcp__atlassian__getJiraIssue", "mcp__atlassian__search"],
            ["mcp__atlassian__createJiraIssue"],
            {},
        )

    monkeypatch.setattr("rote.eval.baseline.read_only_allowlist", _listed)


def test_a_server_not_registered_read_only_is_gated_to_the_tools_it_marks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rote.eval.skill_ab import live_servers
    from rote.mcp.tokens import token_file

    _registered(tmp_path, monkeypatch, read_only=False)
    found, gated = live_servers(["atlassian"], {"atlassian": {"getJiraIssue", "search"}})
    assert found == {"atlassian": "https://mcp.atlassian.example/v1/mcp"}
    assert gated["atlassian"].allowed == (
        "mcp__atlassian__getJiraIssue",
        "mcp__atlassian__search",
    )
    assert gated["atlassian"].denied == ("mcp__atlassian__createJiraIssue",)
    assert gated["atlassian"].token == token_file("atlassian")


def test_a_pipeline_that_calls_a_write_tool_is_refused_naming_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rote.eval.skill_ab import SkillAbError, live_servers

    _registered(tmp_path, monkeypatch, read_only=False)
    with pytest.raises(SkillAbError, match="calls createJiraIssue on atlassian"):
        live_servers(["atlassian"], {"atlassian": {"search", "createJiraIssue"}})


def test_a_server_registered_read_only_is_not_gated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rote.eval.skill_ab import live_servers

    _registered(tmp_path, monkeypatch, read_only=True)
    found, gated = live_servers(["atlassian"], {"atlassian": {"createJiraIssue"}})
    assert found == {"atlassian": "https://mcp.atlassian.example/v1/mcp"} and gated == {}


def test_a_gated_login_reaches_the_trial_owner_only_and_through_rote_mcp_headers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rote.eval.skill_ab import _copy_logins, _mcp_entry, live_servers

    _registered(tmp_path, monkeypatch, read_only=False)
    _, gated = live_servers(["atlassian"], {"atlassian": {"search"}})
    tokens = tmp_path / "trial" / "tokens"
    _copy_logins(gated, {"ROTE_MCP_TOKEN_DIR": str(tokens)})
    copied = tokens / "atlassian.json"
    assert copied.read_bytes() == gated["atlassian"].token.read_bytes()  # type: ignore[union-attr]
    assert copied.stat().st_mode & 0o777 == 0o600
    entry = _mcp_entry("atlassian", "https://mcp.atlassian.example/v1/mcp", gated)
    assert entry["headersHelper"].endswith("-m rote mcp headers atlassian")
    assert "SECRET-TOKEN" not in json.dumps(entry)
    # A server that is not gated gets the plain entry and no login.
    assert _mcp_entry("deepwiki", "https://x.example/mcp", gated) == {
        "type": "http",
        "url": "https://x.example/mcp",
    }


# ───────── the codex agent (#334) ─────────


def _codex_harness(tmp_path: Path, runs: list[dict[str, Any]]) -> skill_ab.Harness:
    """A codex harness whose runner records each call and plays a codex run:
    it writes result.json and prints codex's JSONL events."""
    binary = tmp_path / "pkg" / "codex"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\n")

    def runner(
        argv: list[str], cwd: Path, env: dict[str, str], timeout: float, stdin: str | None = None
    ) -> skill_ab.ProcessResult:
        runs.append({"argv": argv, "env": env, "stdin": stdin})
        (cwd / "result.json").write_text(json.dumps({"total": 1}))
        events = [
            {"type": "thread.started", "thread_id": "t"},
            {
                "type": "item.completed",
                "item": {"type": "command_execution", "command": "rote skill run x"},
            },
            {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": 5000,
                    "cached_input_tokens": 3000,
                    "cache_write_input_tokens": 0,
                    "output_tokens": 200,
                    "reasoning_output_tokens": 0,
                },
            },
        ]
        return skill_ab.ProcessResult(tuple(json.dumps(e) for e in events), "", 0, 1.5, False)

    return skill_ab.Harness(
        work=tmp_path / "work",
        model=None,
        ledger=skill_ab.Ledger(tmp_path / "ledger.json", 0.0),
        log=lambda line: print(line, file=sys.stderr),
        agent="codex",
        codex_binary=binary,
        runner=runner,
    )


def test_a_codex_trial_runs_in_the_translated_gate_and_spends_no_dollars(
    tmp_path: Path, expense: skill_ab.Subject, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.test_codex_lane import CHATGPT_LOGIN

    login = tmp_path / "codex-login"
    login.mkdir()
    (login / "auth.json").write_text(json.dumps(CHATGPT_LOGIN))
    monkeypatch.setenv("CODEX_HOME", str(login))
    (tmp_path / "work").mkdir()
    runs: list[dict[str, Any]] = []
    # A zero-dollar cap: a codex trial must neither check nor charge it.
    harness = _codex_harness(tmp_path, runs)
    original = skill_ab.run_trial(harness, expense, "original", 1, compiled_dir=None)
    compiled_dir = skill_ab.emit_compiled(expense, tmp_path / "work" / "compiled")
    compiled = skill_ab.run_trial(harness, expense, "compiled", 1, compiled_dir=compiled_dir)
    assert harness.ledger.spent == 0 and harness.ledger.runs == []
    first, second = runs
    # The prompt goes on stdin (codex exec -), never argv.
    assert first["argv"][-1] == "-" and first["stdin"] and "expense" in first["stdin"].lower()
    assert first["env"]["CODEX_HOME"] != str(login) and "CODEX_API_KEY" not in first["env"]
    # Only the compiled side may run its driver outside the sandbox, as a tool.
    driver = 'mcp_servers.rote_skill.enabled_tools=["run"]'
    assert driver not in first["argv"] and driver in second["argv"]
    assert "rote skill run expense-audit" in second["stdin"]
    assert "`run` tool of the `rote_skill` MCP server" in second["stdin"]
    assert "rote_skill" not in first["stdin"]
    fs = next(a for a in second["argv"] if a.startswith("permissions.rote_skill_run.filesystem="))
    assert str(compiled_dir.resolve()) in fs
    # Usage: what codex reports, and None for what it does not.
    for trial in (original, compiled):
        assert trial.usage.source == "codex" and trial.usage.cost_usd is None
        assert (trial.usage.input, trial.usage.cache_read, trial.usage.output) == (2000, 3000, 200)
        assert trial.usage.requests is None and trial.usage.warm_start is None
        assert trial.usage.to_dict()["cache"] == "unknown"
    assert original.ok, original.problems
    rendered = skill_ab.render_markdown([], model=None, spent=0.0, agent="codex")
    assert rendered.startswith("Agent: codex. Model: the CLI default.")
    assert "no dollar figure" in rendered


def test_the_codex_usage_report_is_a_total_with_unknowns_left_unknown() -> None:
    usage, envelope = skill_ab.codex_usage(
        [json.dumps({"type": "turn.failed", "error": {"message": "usage limit"}})]
    )
    assert envelope == {"is_error": True, "result": "usage limit"}
    assert usage.output is None and usage.requests is None and usage.cost_usd is None
    # No usage at all fails the trial instead of entering the medians as zero.
    _, silent = skill_ab.codex_usage([json.dumps({"type": "thread.started"})])
    assert silent == {"is_error": True, "result": "codex reported no usage"}
    # A recoverable error event with a usage report is not a failed trial.
    report = {"input_tokens": 9, "cached_input_tokens": 0, "output_tokens": 1}
    _, fine = skill_ab.codex_usage(
        [
            json.dumps({"type": "error", "message": "transient"}),
            json.dumps({"type": "turn.completed", "usage": report}),
        ]
    )
    assert fine == {}


def test_a_codex_eval_runs_no_control_runs(
    tmp_path: Path, expense: skill_ab.Subject, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A control measures B from a request count Codex does not report."""
    from tests.test_codex_lane import CHATGPT_LOGIN

    login = tmp_path / "codex-login"
    login.mkdir()
    (login / "auth.json").write_text(json.dumps(CHATGPT_LOGIN))
    monkeypatch.setenv("CODEX_HOME", str(login))
    (tmp_path / "work").mkdir()
    runs: list[dict[str, Any]] = []
    report = skill_ab.run_subject(_codex_harness(tmp_path, runs), expense, trials=1)
    assert len(runs) == 2  # one original trial, one compiled trial, no controls
    assert all(summary.base_per_request is None for summary in report.sides.values())


def test_budget_is_refused_with_the_codex_agent(capsys: pytest.CaptureFixture[str]) -> None:
    from rote.cli import main

    status = main(
        [
            "eval",
            "x",
            "--run",
            "--runtime",
            "skill",
            "--agent",
            "codex",
            "--budget",
            "5",
            "--input",
            "i.json",
        ]
    )
    assert status == 2
    assert "a Codex run is on your plan" in capsys.readouterr().err


def test_a_codex_compile_is_pinned_to_codex() -> None:
    from rote import proof

    argv = proof.compile_command(
        Path("s"), Path("i.json"), Path("o"), runtime="skill", agent="codex"
    )
    assert argv[argv.index("--agent") + 1] == "codex"
    default = proof.compile_command(Path("s"), Path("i.json"), Path("o"), runtime="skill")
    assert default[default.index("--agent") + 1] == "claude"
