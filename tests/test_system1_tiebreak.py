"""`--tiebreak haiku`: a small LLM decides only the steps System-1 is unsure of.

No inference: the structured call is stubbed (tests/conftest.py also blocks
every real lane). What is pinned: which steps are asked about, the exact
schema-locked request, that the answer replaces only those steps, that the
call bills through the compile's own lane, that every failure is loud, and
that the cost reported is the call's own.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rote import cli
from rote.compiler import Compiler, CompilerError, system1
from rote.compiler.system1 import tiebreak as tb
from rote.compiler.system1.decide import REVIEW_BELOW, unit_segments, unsure_units
from rote.compiler.system1.parse import parse_skill
from rote.compiler.system1.tiebreak import Lane, TieBreakUnavailable

REPO = Path(__file__).resolve().parents[1]
RUBRIC = REPO / "skills" / "rote-compile" / "references" / "node-kinds.md"
STEPS = (
    "## Step 1: Total the batch\n\nTotal every line in USD.\n\n"
    "## Step 2: Tidy the notes\n\nTidy the notes for the file.\n\n"
    "## Step 3: Score the leads\n\nScore the leads against the rubric.\n"
)
#: Step 2 has no signal: the rules default it to pure_function at 0.3.
UNSURE = 1


def _skill(tmp_path: Path) -> Path:
    skill = tmp_path / "skill"
    skill.mkdir(exist_ok=True)
    (skill / "SKILL.md").write_text(f"---\nname: tiny\ndescription: x\n---\n\n# Tiny\n\n{STEPS}")
    return skill


CLI_ROW = (
    "claude-haiku-5-5",
    {
        "input_tokens": 2,
        "output_tokens": 300,
        "cache_write_tokens": 7000,
        "cache_read_tokens": 0,
        "cost_usd": 0.0017,
    },
)


class _Stub:
    """Stands in for ``structured_call``: records each request, answers in turn."""

    def __init__(self, *answers: dict[str, Any], rows: Any = (CLI_ROW,)) -> None:
        self.answers = list(answers)
        self.rows = list(rows)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> tuple[dict[str, Any], list[Any]]:
        self.calls.append(kwargs)
        return self.answers.pop(0), self.rows


def _answer(*kinds: str, step: int = UNSURE) -> dict[str, Any]:
    return {"steps": [{"step": step, "nodes": [{"kind": k, "reason": "why"} for k in kinds]}]}


def _install(monkeypatch: pytest.MonkeyPatch, stub: _Stub) -> None:
    monkeypatch.setattr(tb, "_structured_call", lambda: stub)


def test_only_steps_below_review_confidence_are_unsure(tmp_path: Path) -> None:
    skill = parse_skill(_skill(tmp_path))
    segments = unit_segments(skill)
    assert [u.index for u in unsure_units(skill, segments)] == [UNSURE]
    assert min(c for _, _, c, _ in segments[UNSURE]) < REVIEW_BELOW
    assert all(min(c for _, _, c, _ in segments[i]) >= REVIEW_BELOW for i in (0, 2))


def test_the_request_is_one_schema_locked_haiku_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = _Stub(_answer("external_call", "pure_function"))
    _install(monkeypatch, stub)
    work = tmp_path / "work"
    work.mkdir()
    system1.prepare(_skill(tmp_path), work, tiebreak_lane=Lane("claude-cli"), rubric=RUBRIC)
    assert len(stub.calls) == 1
    call = stub.calls[0]
    assert call["model"] == "claude-haiku-5-5"
    assert call["provider"] == "claude-cli" and call["client"] == "anthropic"
    assert call["output_schema"] == tb.SCHEMA
    # Determinism comes from the forced schema, never a sampler setting.
    assert "sampling" not in call and "temperature" not in json.dumps(call["output_schema"])
    prompt = call["prompt"]
    assert RUBRIC.read_text() in prompt
    body = json.loads(prompt.split("=== SKILL ===\n", 1)[1])
    assert [s["step"] for s in body["unsure_steps"]] == [UNSURE]
    assert body["unsure_steps"][0]["classifier_guess"] == ["pure_function"]
    assert body["all_steps"] == [
        "Step 1: Total the batch",
        "Step 2: Tidy the notes",
        "Step 3: Score the leads",
    ]


def test_a_decision_models_guess_rides_along(tmp_path: Path) -> None:
    """With --decider clef|jev, the tie-break sees the model's own answer too."""
    from rote.compiler.system1.decision_model import JEV, Verdict, Verdicts

    skill = parse_skill(_skill(tmp_path))
    verdicts = Verdicts(JEV, {UNSURE: Verdict("llm_judge", 0.61, 0.07, 0.0)})
    segments = unit_segments(skill, verdicts)
    unsure = unsure_units(skill, segments)
    prompt = tb.prompt_for(skill, unsure, segments, verdicts, "rubric")
    step = json.loads(prompt.split("=== SKILL ===\n", 1)[1])["unsure_steps"][0]
    assert step["decision_model_guess"] == {
        "model": "Jev",
        "kind": "llm_judge",
        "kind_confidence": 0.61,
        "person_must_act": 0.07,
    }


def test_the_answer_replaces_only_the_unsure_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, _Stub(_answer("llm_judge", "pure_function")))
    work = tmp_path / "work"
    work.mkdir()
    plan = system1.prepare(_skill(tmp_path), work, tiebreak_lane=Lane("claude-cli"), rubric=RUBRIC)
    kinds = [(n.unit.index, n.kind) for n in plan.nodes]
    assert kinds == [
        (0, "pure_function"),
        (UNSURE, "llm_judge"),
        (UNSURE, "pure_function"),
        (2, "llm_judge"),
    ]
    decided = [d for n in plan.nodes if n.unit.index == UNSURE for d in n.decisions]
    kind_rules = [d.rule for d in decided if d.field == "kind"]
    assert kind_rules == ["tie-break by claude-haiku-5-5: why"] * 2
    # Settled: the decided step's kinds are not listed for review any more.
    unsure_ids = {n.id for n in plan.nodes if n.unit.index == UNSURE}
    assert not [
        line for line in plan.review if line.split("`")[1] in unsure_ids and "` kind = " in line
    ]
    doc = json.loads((work / "system1" / "decisions.json").read_text())["tiebreak"]
    assert doc["model"] == "claude-haiku-5-5" and doc["steps_asked"] == [str(UNSURE)]
    assert doc["cost_usd"] == 0.0017 and doc["lane"] == "claude -p (subscription)"
    preamble = system1.report_extras(work, plan).preamble
    assert "claude-haiku-5-5 for the 1 step(s)" in preamble
    assert "Its model calls cost $0.0017 (claude-haiku-5-5), on top of the agent's run." in preamble


def test_a_rule_constraint_overrules_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A step that saves to a target it never names can never be an
    external_call, whatever the model says: a real expense-audit compile
    died on exactly that stub ("save it to the finance shared folder")."""
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        "---\nname: tiny\ndescription: x\n---\n\n# Tiny\n\n"
        "## Step 1: Total the batch\n\nTotal every line in USD.\n\n"
        "## Step 2: File the summary\n\nSave it to the shared folder as `<period>.md`.\n"
    )
    work = tmp_path / "work"
    work.mkdir()
    parsed = parse_skill(skill)
    assert [u.index for u in unsure_units(parsed, unit_segments(parsed))] == [1]
    _install(monkeypatch, _Stub(_answer("pure_function", "external_call")))
    plan = system1.prepare(skill, work, tiebreak_lane=Lane("claude-cli"), rubric=RUBRIC)
    assert [n.kind for n in plan.nodes if n.unit.index == 1] == ["pure_function"]
    doc = json.loads((work / "system1" / "decisions.json").read_text())["tiebreak"]
    assert doc["overruled"] == {
        "1": [
            "external_call: it writes to no declared or named system (no MCP server or "
            "tool, no named product, API, interface or path), so it keeps the rules' "
            "decision: what it would write is returned in the pipeline's output for the "
            "caller to deliver, and the step is listed in Open questions"
        ]
    }

    questions = system1.report_extras(work, plan).questions
    assert (
        "claude-haiku-5-5 proposed a kind a rule forbids",
        "step `Step 2: File the summary`: external_call: it writes to no declared or named "
        "system (no MCP server or tool, no named product, API, interface or path), so it "
        "keeps the rules' decision: what it would write is returned in the pipeline's "
        "output for the caller to deliver, and the step is listed in Open questions",
    ) in questions

    # Nothing allowed left: the rules' decision stands, node for node.
    rules = unit_segments(parsed)[1]
    _install(monkeypatch, _Stub(_answer("external_call")))
    plan = system1.prepare(skill, work, tiebreak_lane=Lane("claude-cli"), rubric=RUBRIC)
    kept = [n for n in plan.nodes if n.unit.index == 1]
    assert [n.kind for n in kept] == [kind for _, kind, _, _ in rules]
    assert [next(d.rule for d in n.decisions if d.field == "kind") for n in kept] == [
        rule for _, _, _, rule in rules
    ]
    assert plan.tiebreak.segments == {}

    # A registered server the step names declares the system: the call stands.
    _install(monkeypatch, _Stub(_answer("pure_function", "external_call")))
    plan = system1.prepare(
        skill,
        work,
        tiebreak_lane=Lane("claude-cli"),
        rubric=RUBRIC,
        tools={("shared_folder", "upload_file"): object()},
    )
    assert [n.kind for n in plan.nodes if n.unit.index == 1] == [
        "pure_function",
        "external_call",
    ]


_EMAIL = {"send_email": "gmail"}


@pytest.mark.parametrize(
    ("text", "registry", "tools", "forbidden"),
    [
        # Same intent, two phrasings: decided by what is declared, not by wording.
        ("Notify the account owner by email.", {}, [], True),
        ("Email the summary to the customer.", {}, [], True),
        ("Notify the account owner by email.", _EMAIL, [], False),
        ("Email the summary to the customer.", _EMAIL, [], False),
        ("Save it to the finance shared folder as `<period>.md`.", {}, [], True),
        # A named product or service is a system: its API can be bound or researched.
        ("Save it to Google Drive.", {}, [], False),
        ("Email the summary via Gmail.", {}, [], False),
        ("Post it in #ops-alerts.", {}, [], False),
        ("Post the summary to the team's incident channel.", {}, [], True),
        # A sentence's first word is not a name.
        ("Save it to the shared folder. Payroll pays from it.", {}, [], True),
        # A user interface the step operates is the application it drives.
        ("Click the filters panel, then clear all.", {}, [], False),
        ("Save it to the shared drive.", {}, [("shared_drive", "upload_file")], False),
        ("Set the priority with `update_incident`.", {}, [], False),
        ("Append one row with `append_sheet_values`.", {}, [], False),
        ("Invite each admin: `POST /v1/workspaces/{id}/invites`.", {}, [], False),
        ("Fetch https://api.example.com/v1/rates first.", {}, [], False),
        ("Search the receipts with the Gmail MCP server.", {}, [], False),
        # Reads are never constrained, and a local path is a system.
        ("Read the scoring rules at ~/.claude/skills/x/SKILL.md.", {}, [], False),
        ("Read the runbook at ~/runbooks/db.md.", {}, [], False),
        ("Load the last 7 days of tickets.", {}, [], False),
        ("Read the config file and the error log.", {}, [], False),
        # A step that also reads is not constrained: the read is a call.
        ("Fetch the latest rates and save them to the shared folder.", {}, [], False),
        ("Save it to ~/reports/ as a file.", {}, [], False),
        # No write at all: nothing to constrain.
        ("Total the lines in USD.", {}, [], False),
        # Tool names match whole words only.
        ("Do the research and write it up.", {"search": "web", "fetch": "web"}, [], True),
        ("Write it up and save it with search.", {"search": "web"}, [], False),
        # A declared tool the step uses by name.
        ("Run upload_report on the file.", {"upload_report": "drive"}, [], False),
        ("Run upload_report on the file.", {}, [], True),
    ],
)
def test_an_external_call_needs_a_declared_system(
    text: str,
    registry: dict[str, str],
    tools: list[tuple[str, str]],
    forbidden: bool,
    tmp_path: Path,
) -> None:
    from rote.compiler.system1.decide import infeasible_kinds

    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        f"---\nname: tiny\ndescription: x\n---\n\n# Tiny\n\n## Step 1: Do it\n\n{text}\n"
    )
    unit = parse_skill(skill).units[0]
    assert ("external_call" in infeasible_kinds(unit, registry, tools)) is forbidden


def test_without_a_lane_nothing_is_asked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _Stub()
    _install(monkeypatch, stub)
    work = tmp_path / "work"
    work.mkdir()
    plan = system1.prepare(_skill(tmp_path), work)
    assert stub.calls == [] and plan.tiebreak is None
    assert any("no signal" in line for line in plan.review)
    assert "tiebreak" not in json.loads((work / "system1" / "decisions.json").read_text())


def test_a_partial_answer_is_retried_then_fails_loudly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    skill = _skill(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    wrong_step = _answer("pure_function", step=2)
    good = _answer("llm_judge")
    _install(monkeypatch, _Stub(wrong_step, good))
    plan = system1.prepare(skill, work, tiebreak_lane=Lane("claude-cli"), rubric=RUBRIC)
    assert [n.kind for n in plan.nodes if n.unit.index == UNSURE] == ["llm_judge"]
    # Both attempts were paid for, so both count.
    assert plan.tiebreak.cost_usd == pytest.approx(0.0034)

    empty_nodes = {"steps": [{"step": UNSURE, "nodes": []}]}
    _install(monkeypatch, _Stub(wrong_step, empty_nodes))
    with pytest.raises(TieBreakUnavailable, match="did not answer exactly the 1 steps"):
        system1.prepare(skill, work, tiebreak_lane=Lane("claude-cli"), rubric=RUBRIC)


def test_a_failed_call_names_the_opt_out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(**kwargs: Any) -> Any:
        raise RuntimeError("claude -p failed for system1_tiebreak: overloaded")

    monkeypatch.setattr(tb, "_structured_call", lambda: boom)
    work = tmp_path / "work"
    work.mkdir()
    with pytest.raises(TieBreakUnavailable, match="overloaded.*--tiebreak none"):
        system1.prepare(_skill(tmp_path), work, tiebreak_lane=Lane("claude-cli"), rubric=RUBRIC)


class _Named:
    def __init__(self, name: str) -> None:
        self.name = name


def test_the_lane_follows_the_compile_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    from rote.inference import _runtime_helper as helper

    monkeypatch.setattr(helper.shutil, "which", lambda name: "/usr/local/bin/claude")
    assert tb.lane_for(_Named("claude")) == Lane("claude-cli")

    # Any other driver: a key first, then the CLI.
    monkeypatch.setattr(tb.shutil, "which", lambda name: "/usr/local/bin/claude")
    assert tb.lane_for(_Named("codex")) == Lane("claude-cli")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert tb.lane_for(_Named("codex")) == Lane("api")


def test_the_api_driver_lends_its_own_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """A gateway base URL and headers on the compile apply to the tie-break too."""
    pytest.importorskip("anthropic")
    from rote.compiler.drivers.anthropic_api import AnthropicApiDriver

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    driver = AnthropicApiDriver(
        base_url="https://gateway.example/anthropic",
        default_headers={"cf-aig-metadata": "m"},
    )
    lane = tb.lane_for(driver)
    assert lane.provider == "api"
    assert str(lane.sdk_client.base_url).rstrip("/") == "https://gateway.example/anthropic"
    assert lane.sdk_client.default_headers["cf-aig-metadata"] == "m"
    assert lane.model == "claude-haiku-5-5"
    # A gateway that wants provider-prefixed ids gets the same prefix.
    gateway = AnthropicApiDriver(model="anthropic/claude-sonnet-4.6")
    assert tb.lane_for(gateway).model == "anthropic/claude-haiku-5.5"
    assert tb.gateway_model_id("anthropic", "claude-sonnet-4-6") == "anthropic/claude-sonnet-4.6"


def test_no_lane_at_all_is_loud(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tb.shutil, "which", lambda name: None)
    with pytest.raises(TieBreakUnavailable, match="--tiebreak none"):
        tb.lane_for(_Named("codex"))
    with pytest.raises(TieBreakUnavailable, match="cannot use claude -p"):
        tb.lane_for(_Named("claude"))


def test_the_sdk_lane_sends_a_schema_locked_call_with_the_lent_client() -> None:
    """The tie-break goes through the same structured-output code as emitted
    judges (output_config.format, #182), on the compile driver's own client."""
    from rote.inference._runtime_helper import structured_call, structured_output_schema

    sent: dict[str, Any] = {}

    class _Block:
        type = "text"
        text = '{"steps": []}'

    class _Response:
        stop_reason = "end_turn"
        content = [_Block()]
        usage = {
            "input_tokens": 900,
            "output_tokens": 80,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }

    class _Messages:
        def create(self, **kwargs: Any) -> Any:
            sent.update(kwargs)
            return _Response()

    class _Client:
        messages = _Messages()

    pytest.importorskip("anthropic")
    payload, rows = structured_call(
        node_id="system1_tiebreak",
        provider="api",
        client="anthropic",
        model="claude-haiku-5-5",
        prompt="p",
        output_schema=tb.SCHEMA,
        sdk_client=_Client(),
    )
    assert payload == {"steps": []}
    expected = structured_output_schema(tb.SCHEMA, node_id="system1_tiebreak")
    assert sent["output_config"] == {"format": {"type": "json_schema", "schema": expected}}
    assert "tools" not in sent and "tool_choice" not in sent and "temperature" not in sent
    assert rows == [
        (
            "claude-haiku-5-5",
            {
                "input_tokens": 900,
                "output_tokens": 80,
                "cache_write_tokens": 0,
                "cache_read_tokens": 0,
            },
        )
    ]


def test_the_tiebreak_schema_is_expressible_as_structured_output() -> None:
    from rote.inference._runtime_helper import structured_output_schema

    structured_output_schema(tb.SCHEMA, node_id="system1_tiebreak")  # raises if not


class _Catalog:
    def rates_for(self, model: str) -> tuple[float, float, float | None, float | None] | None:
        return (0.10, 0.50, 0.01, 0.125) if model == "claude-haiku-5-5" else None


def test_the_cost_prices_all_four_buckets(monkeypatch: pytest.MonkeyPatch) -> None:
    import rote.eval.pricing as pricing

    monkeypatch.setattr(pricing, "fetch_catalog", lambda **kw: _Catalog())
    row = {
        "model": "claude-haiku-5-5",
        "input_tokens": 1_000_000,
        "output_tokens": 1_000_000,
        "cache_write_tokens": 1_000_000,
        "cache_read_tokens": 1_000_000,
    }
    assert tb._cost([row]) == pytest.approx(0.10 + 0.50 + 0.125 + 0.01)
    # The CLI's own estimate wins where it exists.
    assert tb._cost([{**row, "cost_usd": 0.5}]) == 0.5
    # An unpriceable row makes the cost unknown, not understated.
    assert tb._cost([row, {**row, "model": "haiku"}]) is None

    def offline(**kw: Any) -> Any:
        raise pricing.PricingError("offline")

    monkeypatch.setattr(pricing, "fetch_catalog", offline)
    assert tb._cost([row]) is None


class _NeverRuns:
    name = "never-runs"

    async def run(self, **kwargs: object) -> object:
        raise AssertionError("the agent ran although the tie-break could not")


@pytest.mark.asyncio
async def test_the_compile_fails_loudly_without_a_lane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Compiler, "select_driver", lambda self: _NeverRuns())
    with pytest.raises(CompilerError, match="--tiebreak none"):
        await Compiler(system1_tiebreak="haiku").compile(_skill(tmp_path), tmp_path / "out")


@pytest.mark.asyncio
async def test_the_compile_logs_the_tiebreak_and_its_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Stop(Exception):
        pass

    class _Spy:
        name = "claude"

        async def run(self, **kwargs: object) -> object:
            raise _Stop

    monkeypatch.setattr(Compiler, "select_driver", lambda self: _Spy())
    monkeypatch.setattr(tb, "lane_for", lambda driver: Lane("claude-cli"))
    _install(monkeypatch, _Stub(_answer("llm_judge")))
    logs: list[str] = []
    with pytest.raises(_Stop):
        await Compiler(on_event=lambda e: logs.append(e.message), system1_tiebreak="haiku").compile(
            _skill(tmp_path), tmp_path / "out"
        )
    assert (
        "system1: tie-break asked claude-haiku-5-5 about 1 unsure step(s) via "
        "claude -p (subscription) ($0.0017)"
    ) in logs


def test_an_unknown_tiebreak_is_refused() -> None:
    with pytest.raises(CompilerError, match="unknown --tiebreak 'sonnet'"):
        Compiler(system1_tiebreak="sonnet")


class _Built(Exception):
    """Raised by the patched Compiler constructor: the command got that far."""


def _compiler_kwargs(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    seen: list[dict[str, Any]] = []

    def init(self: Compiler, *a: Any, **kw: Any) -> None:
        seen.append(kw)
        raise _Built

    monkeypatch.setattr(Compiler, "__init__", init)
    return seen


def _main(argv: list[str]) -> int | None:
    """The exit code, or None when the command reached the Compiler."""
    try:
        return cli.main(argv)
    except _Built:
        return None


@pytest.mark.parametrize("command", ["compile", "analyze"])
def test_the_tiebreak_is_a_layered_setting(
    command: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _compiler_kwargs(monkeypatch)
    user = tmp_path / "user.yaml"
    monkeypatch.setenv("ROTE_CONFIG_PATH", str(user))
    argv = [command, str(_skill(tmp_path))]
    if command == "compile":
        argv += ["--local", "--runtime", "dbos", "--no-eval", "--out", str(tmp_path / "o")]

    # Off by default: the cached-API fidelity bench did not clear the bar.
    assert _main(argv) is None and seen[-1]["system1_tiebreak"] == "none"
    user.write_text("tiebreak: haiku\n")
    assert _main(argv) is None and seen[-1]["system1_tiebreak"] == "haiku"
    monkeypatch.setenv("ROTE_TIEBREAK", "none")
    assert _main(argv) is None and seen[-1]["system1_tiebreak"] == "none"
    assert _main([*argv, "--tiebreak", "haiku"]) is None
    assert seen[-1]["system1_tiebreak"] == "haiku"


def test_a_tiebreak_flag_without_system1_is_a_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _compiler_kwargs(monkeypatch)
    argv = [
        "compile",
        str(_skill(tmp_path)),
        "--local",
        "--runtime",
        "dbos",
        "--no-eval",
        "--out",
        str(tmp_path / "o"),
    ]
    assert _main([*argv, "--no-system1", "--tiebreak", "none"]) == 2
    assert _main([*argv, "--update", "--tiebreak", "haiku"]) == 2
    assert seen == []
    # A configured value is a standing preference: it does not block --update.
    monkeypatch.setenv("ROTE_TIEBREAK", "none")
    assert _main([*argv, "--update"]) is None and len(seen) == 1


def test_cloud_mode_rejects_the_tiebreak_flag() -> None:
    args = cli._build_parser().parse_args(["compile", "s", "--out", "o", "--tiebreak", "none"])
    message = cli._reject_local_only_cloud_flags(args)
    assert message is not None and message.startswith("rote compile: --tiebreak runs locally")


def test_rote_config_reports_the_tiebreak(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["config", "--json"]) == 0
    setting = json.loads(capsys.readouterr().out)["settings"]["tiebreak"]
    assert setting == {"value": "none", "source": "built-in default"}
    monkeypatch.setenv("ROTE_TIEBREAK", "haiku")
    assert cli.main(["config", "--json"]) == 0
    setting = json.loads(capsys.readouterr().out)["settings"]["tiebreak"]
    assert setting == {"value": "haiku", "source": "env ROTE_TIEBREAK"}


def test_the_bench_counts_the_decision_calls(tmp_path: Path) -> None:
    from rote.bench.collect import decisions_cost_usd

    compiled = tmp_path / "compiled"
    assert decisions_cost_usd(compiled) == 0.0  # the full-rubric path writes none
    (compiled / "system1").mkdir(parents=True)
    path = compiled / "system1" / "decisions.json"
    path.write_text(json.dumps({"nodes": []}))
    assert decisions_cost_usd(compiled) == 0.0
    path.write_text(json.dumps({"decider": {"cost_usd": 0.004}, "tiebreak": {"cost_usd": 0.0017}}))
    assert decisions_cost_usd(compiled) == pytest.approx(0.0057)
    path.write_text(json.dumps({"tiebreak": {"cost_usd": None}}))
    assert decisions_cost_usd(compiled) is None


def test_a_bench_job_costs_the_agent_run_plus_the_decision_calls(tmp_path: Path) -> None:
    from rote.bench.collect import collect_job

    compiled = tmp_path / "job" / "compiled"
    (compiled / "system1").mkdir(parents=True)
    (compiled / "pipeline.yaml").write_text("not: a pipeline\n")
    (compiled / "compile-metrics.json").write_text(
        json.dumps({"schema_version": 1, "driver": "claude", "metadata": {"cost_usd": 0.5}})
    )
    (compiled / "system1" / "decisions.json").write_text(
        json.dumps({"decider": {"cost_usd": 0.004}, "tiebreak": {"cost_usd": 0.002}})
    )
    run = collect_job(
        tmp_path / "job",
        tmp_path / "skill",
        tmp_path / "expected.yaml",
        exit_code=0,
        wall_seconds=1.0,
        tsc=False,
    )["run"]
    assert run["cost_usd"] == pytest.approx(0.506)
    assert run["driver_cost_usd"] == 0.5 and run["decisions_cost_usd"] == pytest.approx(0.006)


@pytest.mark.parametrize(
    ("text", "fetches"),
    [
        # A read from a named API or product is a call (apibackend's finding:
        # the AniList step became a judge and the compile made no vendor call).
        ("Query the AniList GraphQL API for the user's watch list.", True),
        ("Fetch the rates from https://api.example.com/v1/rates.", True),
        ("Pull this week's tickets from the Zendesk API.", True),
        ("Look up the customer in Salesforce.", True),
        ("Pull the open deals from HubSpot.", True),
        # Only the named-source rule catches this one: the parser has no fetch signal.
        ("Download the invoices through Stripe.", True),
        ("Load this week's invoices via Stripe.", True),
        ("Read the contact list via the Clearbit enrichment service.", True),
        # No named source: the rules' default, not a call.
        ("Load the last 7 days of tickets.", False),
        ("Classify each deal by stage.", False),
    ],
)
def test_a_read_from_a_named_api_is_a_call(text: str, fetches: bool, tmp_path: Path) -> None:
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        f"---\nname: tiny\ndescription: x\n---\n\n# Tiny\n\n## Step 1: Get data\n\n{text}\n"
    )
    parsed = parse_skill(skill)
    kinds = [kind for _, kind, _, _ in unit_segments(parsed)[0]]
    assert ("external_call" in kinds) is fetches


@pytest.mark.parametrize(
    "text",
    [
        # A capital that names a part of the skill, a role, or a label in the data.
        "Read the summary from Step 3.",
        "Look up each line's rate in the FX table.",
        "Retrieve the batch the Controller approved.",
        "Load the lines and group them by Employee.",
        "Use the rows you read in Phase 2.",
        "Pull the totals in the Q3 column.",
        # The source must follow the read verb in the same sentence.
        "Read the error log lines. The API latency doubled overnight.",
        "Read `references/enrich.md` first. Then enrich via ZoomInfo in batches.",
    ],
)
def test_a_capital_that_is_not_a_source_is_not_a_named_read(text: str) -> None:
    from rote.compiler.system1.decide import _reads_a_named_api

    assert not _reads_a_named_api(text)


def test_a_named_read_never_adds_a_second_call(tmp_path: Path) -> None:
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        "---\nname: tiny\ndescription: x\n---\n\n# Tiny\n\n## Step 1: Sync\n\n"
        "Query the AniList GraphQL API and post the summary to #anime.\n"
    )
    kinds = [kind for _, kind, _, _ in unit_segments(parse_skill(skill))[0]]
    assert kinds.count("external_call") == 1


def test_the_call_uses_the_lanes_own_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The api lane lends the compile driver's client (gateway base URL,
    headers, billing). The tie-break must call through it, not build its own."""
    lent = object()
    stub = _Stub(_answer("llm_judge"))
    _install(monkeypatch, stub)
    work = tmp_path / "work"
    work.mkdir()
    system1.prepare(_skill(tmp_path), work, tiebreak_lane=Lane("api", lent), rubric=RUBRIC)
    assert stub.calls[0]["sdk_client"] is lent and stub.calls[0]["provider"] == "api"
