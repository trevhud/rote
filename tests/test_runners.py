"""Tests for the runners layer behind ``rote run``.

Detection and input/signal resolution are pure filesystem/string logic
and are tested directly. Execution wrappers are tested with the
underlying trial primitives monkeypatched — the primitives themselves
are covered by test_baseline.py / the empirical eval tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rote.eval.empirical import MeasuredRun
from rote.runners import (
    RUNNABLE_RUNTIMES,
    RunTarget,
    TargetError,
    detect_target,
    parse_signal_args,
    resolve_gate_signals,
    resolve_input,
    run_pipeline,
    run_skill,
)

# ───────── Target detection ─────────


def _pop_park_watch(kwargs: dict[str, Any], *, interactive: bool = False) -> None:
    """Every parking runtime gets a park watcher; non-TTY by default."""
    from rote.mcp.parked import LogMarkerChannel

    factory = kwargs.pop("park_watch")
    watcher = factory(LogMarkerChannel(Path("unused.log"), lambda park: None))
    assert watcher.interactive is interactive


def _mk(d: Path, *names: str) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    for name in names:
        p = d / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("", encoding="utf-8")
    return d


def test_detect_skill_dir(tmp_path: Path) -> None:
    _mk(tmp_path / "skill", "SKILL.md")
    target = detect_target(tmp_path / "skill")
    assert target.kind == "skill"
    assert target.runtime is None


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        (["main.py"], "python"),
        (["main.py", "dbos-config.yaml"], "dbos"),
        (["dbos-config.yaml", "package.json", "src/main.ts"], "dbos-ts"),
        (["wrangler.jsonc", "package.json", "src/workflow.ts"], "cloudflare"),
        (["package.json", "src/inngest/pipeline.ts"], "inngest"),
        (["workflow.py", "activities.py"], "temporal"),
    ],
)
def test_detect_emitted_runtime_dirs(tmp_path: Path, files: list[str], expected: str) -> None:
    _mk(tmp_path / "app", *files)
    target = detect_target(tmp_path / "app")
    assert target.kind == "pipeline"
    assert target.runtime == expected


def test_detect_compile_out_layout_single_runtime(tmp_path: Path) -> None:
    out = tmp_path / "out"
    _mk(out / "runtime" / "dbos", "main.py", "dbos-config.yaml")
    _mk(out / "compiled", "pipeline.yaml")
    target = detect_target(out)
    assert target.kind == "pipeline"
    assert target.runtime == "dbos"
    assert target.path == (out / "runtime" / "dbos").resolve()
    assert target.pipeline_yaml == (out / "compiled" / "pipeline.yaml").resolve()


def test_detect_compile_out_layout_needs_runtime_flag_when_ambiguous(tmp_path: Path) -> None:
    out = tmp_path / "out"
    _mk(out / "runtime" / "dbos", "main.py", "dbos-config.yaml")
    _mk(out / "runtime" / "python", "main.py")
    with pytest.raises(TargetError, match="--runtime"):
        detect_target(out)
    target = detect_target(out, runtime="python")
    assert target.runtime == "python"


def test_detect_compile_out_layout_unknown_runtime_flag(tmp_path: Path) -> None:
    out = tmp_path / "out"
    _mk(out / "runtime" / "dbos", "main.py", "dbos-config.yaml")
    with pytest.raises(TargetError, match="no emitted `temporal` runtime"):
        detect_target(out, runtime="temporal")


def test_detect_runtime_flag_mismatch_on_direct_dir(tmp_path: Path) -> None:
    _mk(tmp_path / "app", "main.py")
    with pytest.raises(TargetError, match="looks like an emitted `python` runtime"):
        detect_target(tmp_path / "app", runtime="dbos")


def test_detect_rejects_unrecognizable_dir(tmp_path: Path) -> None:
    _mk(tmp_path / "junk", "notes.txt")
    with pytest.raises(TargetError, match="neither a skill"):
        detect_target(tmp_path / "junk")


def test_detect_rejects_missing_path(tmp_path: Path) -> None:
    with pytest.raises(TargetError, match="not a directory"):
        detect_target(tmp_path / "nope")


def test_detect_finds_sibling_pipeline_yaml_for_bare_emit(tmp_path: Path) -> None:
    _mk(tmp_path, "pipeline.yaml")
    _mk(tmp_path / "app", "main.py")
    target = detect_target(tmp_path / "app")
    assert target.pipeline_yaml == (tmp_path / "pipeline.yaml").resolve()


# ───────── Input resolution ─────────


def test_resolve_input_inline_json() -> None:
    assert resolve_input('{"a": 1}', skill_dir=None, assume_yes=False) == {"a": 1}


def test_resolve_input_inline_invalid_json() -> None:
    with pytest.raises(TargetError, match="not valid JSON"):
        resolve_input("{broken", skill_dir=None, assume_yes=False)


def test_resolve_input_file(tmp_path: Path) -> None:
    p = tmp_path / "input.json"
    p.write_text('{"b": 2}', encoding="utf-8")
    assert resolve_input(str(p), skill_dir=None, assume_yes=False) == {"b": 2}


def test_resolve_input_missing_file(tmp_path: Path) -> None:
    with pytest.raises(TargetError, match="neither an inline JSON object nor a file"):
        resolve_input(str(tmp_path / "nope.json"), skill_dir=None, assume_yes=False)


def test_resolve_input_rejects_non_object() -> None:
    with pytest.raises(TargetError, match="JSON object"):
        resolve_input("[1, 2]", skill_dir=None, assume_yes=False)


def test_resolve_input_reads_an_at_file_like_signal(tmp_path: Path) -> None:
    payload = tmp_path / "in.json"
    payload.write_text('{"report_date": "2026-09-14"}', encoding="utf-8")
    assert resolve_input(f"@{payload}", skill_dir=None, assume_yes=False) == {
        "report_date": "2026-09-14"
    }
    assert resolve_input(str(payload), skill_dir=None, assume_yes=False) == {
        "report_date": "2026-09-14"
    }


def test_resolve_input_no_input_no_skill_errors() -> None:
    with pytest.raises(TargetError, match="no source skill to derive"):
        resolve_input(None, skill_dir=None, assume_yes=False)


def test_resolve_input_derives_with_assume_yes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("rote.runners.derive_input_payload", lambda skill_dir: {"derived": True})
    payload = resolve_input(None, skill_dir=tmp_path, assume_yes=True)
    assert payload == {"derived": True}


# ───────── Signal parsing + gate resolution ─────────


def test_parse_signal_args() -> None:
    signals = parse_signal_args(['approve={"ok": true}', 'review={"tier": 1}'])
    assert signals == {"approve": {"ok": True}, "review": {"tier": 1}}


def test_parse_signal_args_requires_name_and_json() -> None:
    with pytest.raises(TargetError, match="NAME=JSON"):
        parse_signal_args(["no-equals-sign"])
    with pytest.raises(TargetError, match="not valid JSON"):
        parse_signal_args(["approve={broken"])


def test_parse_signal_args_reads_an_at_file(tmp_path: Path) -> None:
    """``NAME=@FILE`` is the file's JSON, the help text's documented form."""
    payload = tmp_path / "manual.json"
    payload.write_text('{"sorter": [{"site": "MTN6", "volume": 22800}]}', encoding="utf-8")
    signals = parse_signal_args([f"manual_data=@{payload}", 'approve={"ok": true}'])
    assert signals == {
        "manual_data": {"sorter": [{"site": "MTN6", "volume": 22800}]},
        "approve": {"ok": True},
    }
    with pytest.raises(TargetError, match=r"manual_data: cannot read .*missing\.json"):
        parse_signal_args([f"manual_data=@{tmp_path / 'missing.json'}"])
    (tmp_path / "bad.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(TargetError, match=r"bad\.json is not valid JSON"):
        parse_signal_args([f"manual_data=@{tmp_path / 'bad.json'}"])


def test_resolve_gate_signals_all_provided() -> None:
    provided = {"g1": {"ok": True}}
    assert resolve_gate_signals(["g1"], provided, interactive=False) == provided


def test_resolve_gate_signals_unknown_name() -> None:
    with pytest.raises(TargetError, match="not in the pipeline's gates"):
        resolve_gate_signals(["g1"], {"nope": {}}, interactive=False)


def test_resolve_gate_signals_missing_non_interactive_errors() -> None:
    with pytest.raises(TargetError, match="--signal g2"):
        resolve_gate_signals(["g1", "g2"], {"g1": {}}, interactive=False)


def test_resolve_gate_signals_prompts_interactively(monkeypatch: pytest.MonkeyPatch) -> None:
    answers = iter(['{"approved": true}', ""])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    resolved = resolve_gate_signals(["g1", "g2"], {}, interactive=True)
    assert resolved == {"g1": {"approved": True}, "g2": {}}


def test_resolve_gate_signals_interactive_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("builtins.input", lambda prompt: "{broken")
    with pytest.raises(TargetError, match="not valid JSON"):
        resolve_gate_signals(["g1"], {}, interactive=True)


# ───────── Execution wrappers ─────────


def test_run_skill_applies_wiring(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_wiring(*, allow_writes: bool) -> tuple[dict[str, Any], list[str], dict[str, str]]:
        captured["allow_writes"] = allow_writes
        return (
            {"srv": {"type": "http", "url": "http://x"}},
            ["mcp__srv__lookup"],
            {"other": "unreachable"},
        )

    def fake_trial(skill_dir: Any, payload: Any, **kwargs: Any) -> tuple[Any, list[Any], list[str]]:
        captured["kwargs"] = kwargs
        return MeasuredRun(wall_seconds=1.0, output={"ok": True}), [], []

    monkeypatch.setattr("rote.runners.resolve_mcp_wiring", fake_wiring)
    monkeypatch.setattr("rote.runners.run_baseline_trial", fake_trial)

    outcome = run_skill(tmp_path, {"q": 1}, model="m")
    assert captured["allow_writes"] is False
    assert captured["kwargs"]["mcp_tool_ids"] == ["mcp__srv__lookup"]
    assert captured["kwargs"]["mcp_servers"] == {"srv": {"type": "http", "url": "http://x"}}
    assert outcome.read_only is True
    assert outcome.servers_wired == ("srv",)
    assert outcome.servers_skipped == {"other": "unreachable"}
    assert outcome.run.output == {"ok": True}


def test_rote_run_hands_a_skill_its_gate_answers_from_signal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`rote run <skill> --signal` serves the answers through ask_person, in
    the order given, @FILE included. The prompt says how to ask and carries
    no answer: the agent gets each one when it reaches the gate."""
    import json

    from rote.cli import main as cli_main
    from rote.eval.gate_server import GATE_SERVER, GATE_TOOL

    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: s\n---\nAsk the lead to approve.\n")
    answer = tmp_path / "approval.json"
    answer.write_text('{"approved": true, "reviewer": "Priya"}', encoding="utf-8")
    captured: dict[str, Any] = {}

    def fake_trial(skill_dir: Any, payload: Any, **kwargs: Any) -> tuple[Any, list[Any], list[str]]:
        captured.update(kwargs)
        entry = kwargs["mcp_servers"][GATE_SERVER]
        captured["answers"] = json.loads(Path(entry["args"][2]).read_text(encoding="utf-8"))
        return MeasuredRun(wall_seconds=1.0, output={"ok": True}), [], []

    monkeypatch.setattr("rote.runners.resolve_mcp_wiring", lambda **_: ({}, [], {}))
    monkeypatch.setattr("rote.runners.run_baseline_trial", fake_trial)
    argv = ["run", str(skill), "--input", "{}", "--signal", f"approval=@{answer}"]
    assert cli_main([*argv, "--signal", 'second={"n": 2}']) == 0
    assert captured["answers"] == [{"approved": True, "reviewer": "Priya"}, {"n": 2}]
    assert f"mcp__{GATE_SERVER}__{GATE_TOOL}" in captured["mcp_tool_ids"]
    text = captured["extra_instructions"]
    assert GATE_TOOL in text
    assert "Priya" not in text  # no answer up front
    # Byte-identical to what the bench's fidelity references are told.
    from rote.bench.fidelity import Case as FidelityCase
    from rote.bench.fidelity import CaseFile, agent_instructions

    case = FidelityCase(
        id="c",
        description="",
        input={},
        gates=[{"approved": True, "reviewer": "Priya"}, {"n": 2}],
        fixtures={},
    )
    bench = agent_instructions(CaseFile(tmp_path, {"x": "y"}, {}, [case]), case)
    assert text in bench
    # Without --signal there is no gate server and nothing about gates.
    captured.clear()

    def plain_trial(
        skill_dir: Any, payload: Any, **kwargs: Any
    ) -> tuple[Any, list[Any], list[str]]:
        captured.update(kwargs)
        return MeasuredRun(wall_seconds=1.0, output={"ok": True}), [], []

    monkeypatch.setattr("rote.runners.run_baseline_trial", plain_trial)
    assert cli_main(["run", str(skill), "--input", "{}"]) == 0
    assert captured["extra_instructions"] is None
    assert GATE_SERVER not in (captured["mcp_servers"] or {})


def test_run_pipeline_rejects_unorchestrated_runtimes(tmp_path: Path) -> None:
    # Every current adapter is orchestrated; the guard now protects
    # against future adapters landing before their runner does.
    target = RunTarget(kind="pipeline", path=tmp_path, runtime="somefuture-runtime")
    with pytest.raises(TargetError, match="cannot orchestrate"):
        run_pipeline(target, {})


def test_run_pipeline_dispatches_to_trial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_trial(app_dir: Any, payload: Any, **kwargs: Any) -> MeasuredRun:
        captured["app_dir"] = app_dir
        captured["payload"] = payload
        captured["kwargs"] = kwargs
        return MeasuredRun(wall_seconds=0.5, output={"done": 1})

    monkeypatch.setattr("rote.runners.run_pipeline_trial", fake_trial)
    target = RunTarget(kind="pipeline", path=tmp_path, runtime="dbos")
    outcome = run_pipeline(target, {"x": 1}, signals={"g": {"ok": True}}, timeout_seconds=30.0)
    assert captured["app_dir"] == tmp_path
    assert captured["payload"] == {"x": 1}
    _pop_park_watch(captured["kwargs"])
    assert captured["kwargs"] == {"signals": {"g": {"ok": True}}, "timeout_seconds": 30.0}
    assert outcome.runtime == "dbos"
    assert outcome.run.output == {"done": 1}


@pytest.mark.parametrize(
    ("allow_writes", "seen"),
    [
        (
            False,
            {
                "ROTE_MCP_READ_ONLY": "1",
                "ROTE_MCP_READ_ONLY_SERVERS": '{"docs": "https://docs.example/mcp"}',
            },
        ),
        (True, {"ROTE_MCP_READ_ONLY": None, "ROTE_MCP_READ_ONLY_SERVERS": None}),
    ],
)
def test_a_pipeline_run_is_read_only_unless_writes_are_allowed(
    allow_writes: bool,
    seen: dict[str, str | None],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The app sees the gate for the length of the run, with the servers
    the user registered read-only, and the environment is restored after."""
    import os

    from rote.mcp.registry import McpRegistry, McpServerConfig, save_registry

    save_registry(
        McpRegistry(
            servers={
                "docs": McpServerConfig(url="https://docs.example/mcp", read_only=True),
                "briefs": McpServerConfig(url="https://briefs.example/mcp"),
            }
        )
    )
    observed: dict[str, str | None] = {}

    def fake_trial(app_dir: Any, payload: Any, **kwargs: Any) -> MeasuredRun:
        observed.update({k: os.environ.get(k) for k in seen})
        return MeasuredRun(wall_seconds=0.5, output={"done": 1})

    monkeypatch.setattr("rote.runners.run_pipeline_trial", fake_trial)
    target = RunTarget(kind="pipeline", path=tmp_path, runtime="dbos")
    outcome = run_pipeline(target, {}, allow_writes=allow_writes)
    assert observed == seen
    assert outcome.read_only is not allow_writes
    assert "ROTE_MCP_READ_ONLY" not in os.environ


def test_mcp_add_read_only_is_stored_and_trusted_by_both_sides(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """One declaration, read by the skill side's allowlist predicate and the
    pipeline side's environment alike."""
    from types import SimpleNamespace

    from rote.cli import main as cli_main
    from rote.mcp.live_tools import read_only_tools
    from rote.mcp.registry import declared_read_only, load_registry, read_only_servers

    assert cli_main(["mcp", "add", "deepwiki", "https://mcp.deepwiki.com/mcp", "--read-only"]) == 0
    assert cli_main(["mcp", "add", "briefs", "https://briefs.example/mcp"]) == 0
    assert "read-only: every tool on it" in capsys.readouterr().out
    assert load_registry().servers["deepwiki"].read_only is True
    assert load_registry().servers["briefs"].read_only is None
    assert read_only_servers() == {"deepwiki": "https://mcp.deepwiki.com/mcp"}
    unannotated = [SimpleNamespace(name="ask_question", annotations=None)]
    assert read_only_tools(unannotated, declared_read_only("deepwiki")) == unannotated
    assert read_only_tools(unannotated, declared_read_only("briefs")) == []


def test_a_read_only_declaration_covers_the_registered_endpoint_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`ROTE_MCP_<SERVER>_URL` wins over the registry, so a name registered
    read-only could be pointed at an endpoint whose tools write. The
    declaration is the user's word about the registered URL, nothing else."""
    from rote.mcp import _runtime_helper as helper
    from rote.mcp.registry import McpRegistry, McpServerConfig, declared_read_only, save_registry

    save_registry(
        McpRegistry(
            servers={
                "deepwiki": McpServerConfig(url="https://mcp.deepwiki.com/mcp", read_only=True)
            }
        )
    )
    monkeypatch.delenv("ROTE_MCP_DEEPWIKI_URL", raising=False)
    assert declared_read_only("deepwiki")
    monkeypatch.setenv("ROTE_MCP_DEEPWIKI_URL", "https://MCP.deepwiki.com:443/mcp/")
    assert declared_read_only("deepwiki")  # the same endpoint, spelled differently
    monkeypatch.setenv("ROTE_MCP_DEEPWIKI_URL", "https://writes.example/mcp")
    assert not declared_read_only("deepwiki")
    # The pipeline side applies the same rule to the URL the call resolves to.
    monkeypatch.setenv("ROTE_MCP_READ_ONLY", "1")
    monkeypatch.setenv("ROTE_MCP_READ_ONLY_SERVERS", '{"deepwiki": "https://mcp.deepwiki.com/mcp"}')
    tool = type("T", (), {"name": "ask_question", "annotations": None})()
    assert (
        helper.write_refusal("deepwiki", "https://mcp.deepwiki.com/mcp", "ask_question", [tool])
        is None
    )
    refusal = helper.write_refusal("deepwiki", "https://writes.example/mcp", "ask_question", [tool])
    assert refusal is not None
    assert "registered read-only at https://mcp.deepwiki.com/mcp" in refusal
    assert "https://writes.example/mcp" in refusal


def test_runnable_runtimes_is_the_current_set() -> None:
    assert set(RUNNABLE_RUNTIMES) == {
        "python",
        "dbos",
        "cloudflare",
        "inngest",
        "dbos-ts",
        "temporal",
    }


def test_resolve_input_declined_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("rote.runners.derive_input_payload", lambda skill_dir: {"d": 1})
    monkeypatch.setattr("rote.runners.sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    assert resolve_input(None, skill_dir=tmp_path, assume_yes=False) is None


def test_resolve_input_derived_proposal_is_persisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("rote.runners.derive_input_payload", lambda skill_dir: {"d": 1})
    payload = resolve_input(None, skill_dir=tmp_path, assume_yes=True)
    assert payload == {"d": 1}
    err = capsys.readouterr().err
    assert "derived input proposal" in err
    assert json.dumps({"d": 1}, indent=2) in err


# ───────── Cloudflare runner (unit level — live path in test_cloudflare_e2e) ─────────


def test_run_pipeline_dispatches_cloudflare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import rote.runners.cloudflare as cf_runner

    captured: dict[str, Any] = {}

    def fake_run_cloudflare(app_dir: Any, payload: Any, **kwargs: Any) -> MeasuredRun:
        captured["app_dir"] = app_dir
        captured["kwargs"] = kwargs
        return MeasuredRun(wall_seconds=1.0, output={"cf": True})

    monkeypatch.setattr(cf_runner, "run_cloudflare", fake_run_cloudflare)
    target = RunTarget(kind="pipeline", path=tmp_path, runtime="cloudflare")
    outcome = run_pipeline(
        target,
        {"x": 1},
        signals={"g1": {"ok": True}},
        gate_order=["g1"],
        timeout_seconds=42.0,
    )
    assert outcome.run.output == {"cf": True}
    _pop_park_watch(captured["kwargs"])
    assert captured["kwargs"] == {
        "signals": {"g1": {"ok": True}},
        "gate_order": ["g1"],
        "timeout_seconds": 42.0,
        # Read-only by default, passed as Worker vars (workerd sees no env).
        "worker_vars": {"ROTE_MCP_READ_ONLY": "1", "ROTE_MCP_READ_ONLY_SERVERS": "{}"},
    }


def test_cf_wait_returns_on_terminal_status(monkeypatch: pytest.MonkeyPatch) -> None:
    from rote.runners import cloudflare as cf

    states = iter(
        [
            {"status": "running", "__LOCAL_DEV_STEP_OUTPUTS": [1]},
            {"status": "complete", "output": {"done": 1}},
        ]
    )
    monkeypatch.setattr(cf, "http_get_json", lambda url, timeout=10.0: next(states))
    monkeypatch.setattr(cf.time, "sleep", lambda s: None)
    state = cf._wait_until_parked_or_complete(1, "id", deadline=cf.time.monotonic() + 30)
    assert state["status"] == "complete"


def test_cf_wait_infers_parking_from_step_stability(monkeypatch: pytest.MonkeyPatch) -> None:
    from rote.runners import cloudflare as cf

    stable_state = {"status": "running", "__LOCAL_DEV_STEP_OUTPUTS": [1, 2, 3]}
    monkeypatch.setattr(cf, "http_get_json", lambda url, timeout=10.0: dict(stable_state))
    monkeypatch.setattr(cf.time, "sleep", lambda s: None)
    state = cf._wait_until_parked_or_complete(1, "id", deadline=cf.time.monotonic() + 30)
    assert state["status"] == "running"  # parked, inferred


def test_cf_wait_times_out_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    from rote.eval.empirical import EmpiricalError
    from rote.runners import cloudflare as cf

    growing: list[int] = []

    def _grow(url: str, timeout: float = 10.0) -> dict[str, Any]:
        growing.append(1)
        return {"status": "running", "__LOCAL_DEV_STEP_OUTPUTS": list(growing)}

    monkeypatch.setattr(cf, "http_get_json", _grow)
    monkeypatch.setattr(cf.time, "sleep", lambda s: None)
    with pytest.raises(EmpiricalError, match="timed out"):
        cf._wait_until_parked_or_complete(1, "id", deadline=cf.time.monotonic() + 0.2)


#: What workerd actually prints for an uncaught step error (captured from a
#: live `wrangler dev --local` run), ANSI colors included.
_WRANGLER_STEP_ERROR_LOG = (
    "[wrangler:info] Ready on http://localhost:55878\n"
    "\x1b[31m✘ \x1b[41;31m[\x1b[41;97mERROR\x1b[41;31m]\x1b[0m \x1b[1mUncaught "
    "NonRetryableError: no endpoint for MCP server 'vendor': set the "
    "ROTE_MCP_VENDOR_URL variable (wrangler.jsonc vars, .dev.vars, or a secret). "
    "This is configuration, so the step is not retried. Error\x1b[0m\n"
    "      at null.<anonymous> (file:///app/src/workflow.ts:91:35)\n"
)

_GENERIC_FATAL = {
    "name": "WorkflowFatalError",
    "message": (
        "The execution of the Workflow instance was terminated, as a step threw an "
        "NonRetryableError and it was not handled"
    ),
}


def _scripted_instance(monkeypatch: pytest.MonkeyPatch, states: list[dict[str, Any]]) -> list[str]:
    """Serve ``states`` in order (the last one repeats); record posted events."""
    from rote.runners import cloudflare as cf

    remaining = list(states)
    posted: list[str] = []

    def _get(url: str, timeout: float = 10.0) -> dict[str, Any]:
        return dict(remaining.pop(0) if len(remaining) > 1 else remaining[0])

    def _post(url: str, payload: Any, timeout: float = 10.0) -> dict[str, Any]:
        posted.append(url.rsplit("/", 1)[-1])
        return {}

    monkeypatch.setattr(cf, "http_get_json", _get)
    monkeypatch.setattr(cf, "http_post_json", _post)
    monkeypatch.setattr(cf.time, "sleep", lambda s: None)
    return posted


def test_cf_run_reports_the_step_error_from_the_wrangler_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Local dev's status only says WorkflowFatalError; the message a user
    needs (which variable to set) is in workerd's console output."""
    from rote.runners import cloudflare as cf

    log = tmp_path / "wrangler-dev.log"
    log.write_text(_WRANGLER_STEP_ERROR_LOG, encoding="utf-8")
    _scripted_instance(monkeypatch, [{"status": "errored", "error": _GENERIC_FATAL}])
    run = cf._drive_instance(
        1, "id", [], started=cf.time.monotonic(), deadline=cf.time.monotonic() + 30, log_path=log
    )
    assert run.error is not None
    assert run.error.startswith("workflow errored: WorkflowFatalError: The execution")
    assert run.error.endswith(
        "\nstep error: NonRetryableError: no endpoint for MCP server 'vendor': set the "
        "ROTE_MCP_VENDOR_URL variable (wrangler.jsonc vars, .dev.vars, or a secret). "
        "This is configuration, so the step is not retried."
    )


def test_cf_run_waits_out_a_retrying_step_after_the_last_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A step sleeping between retries looks parked. Once every gate payload
    is out, that is not an error: wait for the real outcome and report it."""
    from rote.runners import cloudflare as cf

    log = tmp_path / "wrangler-dev.log"
    log.write_text(_WRANGLER_STEP_ERROR_LOG, encoding="utf-8")
    quiet = {"status": "running", "__LOCAL_DEV_STEP_OUTPUTS": []}
    # Parked-looking long enough for the one gate, then again (retry
    # backoff), and only then the step's retries run out.
    states = [quiet] * 12 + [{"status": "errored", "error": _GENERIC_FATAL}]
    posted = _scripted_instance(monkeypatch, states)
    run = cf._drive_instance(
        1,
        "id",
        [("approve", {"ok": True})],
        started=cf.time.monotonic(),
        deadline=cf.time.monotonic() + 30,
        log_path=log,
    )
    assert posted == ["approve"]
    assert run.error is not None
    assert "every gate payload" not in run.error
    assert "set the ROTE_MCP_VENDOR_URL variable" in run.error


def test_cf_run_reports_the_newest_step_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An earlier step that failed once and recovered must not be reported
    as the cause of a later fatal failure."""
    from rote.runners import cloudflare as cf

    log = tmp_path / "wrangler-dev.log"
    log.write_text(
        "\x1b[31m✘ [ERROR] Uncaught Error: upstream 503 from the search API\x1b[0m\n"
        "[wrangler:info] GET /status/id 200 OK\n" + _WRANGLER_STEP_ERROR_LOG,
        encoding="utf-8",
    )
    _scripted_instance(monkeypatch, [{"status": "errored", "error": _GENERIC_FATAL}])
    run = cf._drive_instance(
        1, "id", [], started=cf.time.monotonic(), deadline=cf.time.monotonic() + 30, log_path=log
    )
    assert run.error is not None
    assert "\nstep error: NonRetryableError: no endpoint" in run.error
    assert "upstream 503" not in run.error


def test_cf_run_fails_fast_on_a_gate_that_never_consumes_its_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No step output after the last payload and nothing retrying: the gate
    is stuck (a signal name that matches no waitForEvent), so waiting out
    the whole timeout would only delay the same answer."""
    from rote.runners import cloudflare as cf

    log = tmp_path / "wrangler-dev.log"
    log.write_text("[wrangler:info] Ready on http://localhost:1\n", encoding="utf-8")
    parked = {"status": "running", "__LOCAL_DEV_STEP_OUTPUTS": [{"a": 1}]}
    posted = _scripted_instance(monkeypatch, [parked])
    run = cf._drive_instance(
        1,
        "id",
        [("approve", {"ok": True})],
        started=cf.time.monotonic(),
        deadline=cf.time.monotonic() + 5,
        log_path=log,
    )
    assert posted == ["approve"]
    assert run.error is not None
    assert "never consumed, and no step is retrying" in run.error


def test_cf_run_timeout_after_the_last_gate_names_the_step_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rote.eval.empirical import EmpiricalError
    from rote.runners import cloudflare as cf

    log = tmp_path / "wrangler-dev.log"
    log.write_text(_WRANGLER_STEP_ERROR_LOG, encoding="utf-8")
    _scripted_instance(monkeypatch, [{"status": "running", "__LOCAL_DEV_STEP_OUTPUTS": []}])
    with pytest.raises(EmpiricalError) as info:
        cf._drive_instance(
            1,
            "id",
            [],
            started=cf.time.monotonic(),
            deadline=cf.time.monotonic() + 0.2,
            log_path=log,
        )
    assert "did not finish" in str(info.value)
    assert "step error: NonRetryableError: no endpoint" in str(info.value)


# ───────── Inngest runner (unit level — live path in test_inngest_e2e) ─────────


def test_inngest_failure_reports_the_innermost_error(tmp_path: Path) -> None:
    """The shape the dev server's /v1/runs record carries for a failed run
    (captured live): the NonRetriableError wrapper outside, the real error
    as its cause."""
    from rote.runners.inngest import _failure_message

    output = {
        "__serialized": True,
        "name": "NonRetriableError",
        "message": "no endpoint for MCP server 'vendor': set ROTE_MCP_VENDOR_URL",
        "cause": {
            "__serialized": True,
            "name": "RoteMcpConfigError",
            "message": "no endpoint for MCP server 'vendor': set ROTE_MCP_VENDOR_URL",
            "stack": "RoteMcpConfigError: ...\n    at resolveUrl (...)",
        },
    }
    log = tmp_path / "rote-run-app.log"
    log.write_text("      at step.fn (engine.cjs:1424:35)\n", encoding="utf-8")
    assert _failure_message("Failed", output, log) == (
        "run failed: RoteMcpConfigError: no endpoint for MCP server 'vendor': "
        f"set ROTE_MCP_VENDOR_URL (app log: {log})"
    )
    # No error object at all (a cancelled run): the log tail is all there is.
    assert _failure_message("Cancelled", None, log).startswith("run cancelled; app log tail:\n")


def test_run_pipeline_inngest_requires_pipeline(tmp_path: Path) -> None:
    target = RunTarget(kind="pipeline", path=tmp_path, runtime="inngest")
    with pytest.raises(TargetError, match="pipeline.yaml"):
        run_pipeline(target, {}, pipeline=None)


def test_run_pipeline_dispatches_inngest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import rote.runners.inngest as inngest_runner

    captured: dict[str, Any] = {}

    def fake_run_inngest(app_dir: Any, payload: Any, **kwargs: Any) -> MeasuredRun:
        captured["kwargs"] = kwargs
        return MeasuredRun(wall_seconds=2.0, output={"inngest": True})

    monkeypatch.setattr(inngest_runner, "run_inngest", fake_run_inngest)
    target = RunTarget(kind="pipeline", path=tmp_path, runtime="inngest")
    sentinel = object()
    outcome = run_pipeline(
        target,
        {"x": 1},
        signals={"g": {}},
        gate_order=["g"],
        pipeline=sentinel,
        timeout_seconds=9.0,
    )
    assert outcome.run.output == {"inngest": True}
    assert captured["kwargs"]["pipeline"] is sentinel
    assert captured["kwargs"]["gate_order"] == ["g"]


def test_inngest_run_output_parses_run_complete(monkeypatch: pytest.MonkeyPatch) -> None:
    from rote.runners import inngest as ig

    ops = [
        {"op": "Step", "id": "a"},
        {"op": "RunComplete", "data": {"final": {"ok": 1}}},
    ]
    response = {"data": {"run": {"status": "Completed", "output": json.dumps(ops)}}}
    monkeypatch.setattr(ig, "http_json", lambda *a, **k: response)
    assert ig._run_output("http://x", "r1") == {"final": {"ok": 1}}


def test_inngest_run_output_errors_without_run_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rote.eval.empirical import EmpiricalError
    from rote.runners import inngest as ig

    response = {"data": {"run": {"status": "Completed", "output": json.dumps([])}}}
    monkeypatch.setattr(ig, "http_json", lambda *a, **k: response)
    with pytest.raises(EmpiricalError, match="RunComplete"):
        ig._run_output("http://x", "r1")


# ───────── DBOS-TS runner (unit level — live path in test_dbos_ts_e2e) ─────────


def test_run_pipeline_dispatches_dbos_ts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import rote.runners.dbos_ts as ts_runner

    captured: dict[str, Any] = {}

    def fake_run(app_dir: Any, payload: Any, **kwargs: Any) -> MeasuredRun:
        captured["kwargs"] = kwargs
        return MeasuredRun(wall_seconds=3.0, output={"ts": True})

    monkeypatch.setattr(ts_runner, "run_dbos_ts", fake_run)
    target = RunTarget(kind="pipeline", path=tmp_path, runtime="dbos-ts")
    outcome = run_pipeline(target, {"x": 1}, signals={"g": {}}, timeout_seconds=7.0)
    assert outcome.run.output == {"ts": True}
    _pop_park_watch(captured["kwargs"])
    assert captured["kwargs"] == {"signals": {"g": {}}, "timeout_seconds": 7.0}


def test_dbos_ts_unreachable_env_url_is_loud(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rote.eval.empirical import EmpiricalError
    from rote.runners import dbos_ts as ts

    (tmp_path / "dist").mkdir()
    (tmp_path / "dist" / "main.js").write_text("", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "dbos-config.yaml").write_text("name: p\n", encoding="utf-8")
    monkeypatch.setenv("DBOS_SYSTEM_DATABASE_URL", "postgresql://u:p@127.0.0.1:1/x")
    monkeypatch.setattr(ts, "_tcp_reachable", lambda url: False)
    with pytest.raises(EmpiricalError, match="unreachable Postgres"):
        ts.run_dbos_ts(tmp_path, {}, signals={})


def test_dbos_ts_parses_trailing_json_after_sdk_banner() -> None:
    from rote.runners.dbos_ts import _parse_trailing_json

    stdout = (
        "Running DBOS system database migrations...\n"
        "Initializing DBOS (v5.3.0)\n"
        "System Database URL: postgresql://postgres:***@localhost:5432/x\n"
        '{\n  "gate": {\n    "ok": true\n  }\n}\n'
    )
    assert _parse_trailing_json(stdout) == {"gate": {"ok": True}}
    assert _parse_trailing_json("no json here\n") is None
    assert _parse_trailing_json("[1, 2]\n") == {"result": [1, 2]}


# ───────── Temporal runner (unit level) ─────────


def test_run_pipeline_dispatches_temporal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import rote.runners.temporal as t_runner

    captured: dict[str, Any] = {}

    def fake_run(app_dir: Any, payload: Any, **kwargs: Any) -> MeasuredRun:
        captured["kwargs"] = kwargs
        return MeasuredRun(wall_seconds=1.5, output={"temporal": True})

    monkeypatch.setattr(t_runner, "run_temporal", fake_run)
    target = RunTarget(kind="pipeline", path=tmp_path, runtime="temporal")
    outcome = run_pipeline(target, {"x": 1}, signals={"g": {}}, timeout_seconds=5.0)
    assert outcome.run.output == {"temporal": True}
    assert "park_watch" not in captured["kwargs"]  # temporal never parks on auth
    assert captured["kwargs"] == {"signals": {"g": {}}, "timeout_seconds": 5.0}


def test_temporal_load_modules_requires_emitted_files(tmp_path: Path) -> None:
    from rote.eval.empirical import EmpiricalError
    from rote.runners.temporal import _load_app_modules

    with pytest.raises(EmpiricalError, match="workflow.py"):
        _load_app_modules(tmp_path)


def test_temporal_runner_enables_the_dev_ui(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bdr_pipeline: Any
) -> None:
    """`rote run` on a temporal app hosts the bundled Web UI and prints
    its URL — the dev server has one, so surface it (see also the
    inngest runner, whose dashboard rides the managed dev port)."""
    from rote.adapters import get_adapter
    from rote.runners.temporal import run_temporal

    get_adapter("temporal").emit(bdr_pipeline, tmp_path)
    captured: dict[str, Any] = {}

    async def fake_start_local(**kwargs: Any) -> Any:
        captured.update(kwargs)
        raise RuntimeError("stop-after-capture")

    from temporalio.testing import WorkflowEnvironment

    monkeypatch.setattr(WorkflowEnvironment, "start_local", fake_start_local)
    with pytest.raises(RuntimeError, match="stop-after-capture"):
        run_temporal(tmp_path, {}, signals={})
    assert captured["ui"] is True
    assert isinstance(captured["ui_port"], int)
